"""In-memory fakes for testing.

These implementations are deliberately simple, deterministic, multi-tenant aware,
and respect ``TenantScope`` so the conformance suites can exercise contracts
including cross-tenant leakage detection. They are intended for use in unit and
integration tests; they are not durable or thread-safe across processes.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar
from uuid import uuid4

from dreamer.api.audit import AuditSink
from dreamer.api.compat import implements
from dreamer.api.contexts import (
    AcquireLeaseContext,
    AuditContext,
    ClaimContext,
    ClearContextPendingContext,
    CommitWorkspaceContext,
    ContextPhaseContext,
    ContextPhaseServices,
    CountContext,
    DiscardWorkspaceContext,
    GetContextPendingContext,
    ListUnconsumedContext,
    LTMPhaseContext,
    LTMPhaseServices,
    MarkConsumedContext,
    OpenWorkspaceContext,
    PublishContext,
    PurgeConsumedContext,
    RateLimitContext,
    ReclaimContext,
    ReclaimLeasesContext,
    ReleaseContext,
    ReleaseLeaseContext,
    RenewLeaseContext,
    SerializeContext,
    SerializeServices,
    SetContextPendingContext,
    SubmitContext,
    SubscribeContext,
    UsageContext,
)
from dreamer.api.dream import ContextPhaseRunner, LTMPhaseRunner
from dreamer.api.jobs import JobQueue
from dreamer.api.rate_limit import RateLimiter
from dreamer.api.stores import (
    ContextPendingStore,
    ContextStore,
    DreamLeaseStore,
    LTMStore,
    STMSerializer,
    STMStore,
)
from dreamer.api.tenants import TenantScope
from dreamer.api.types import (
    AuditEvent,
    Diff,
    DreamJob,
    DreamLease,
    FileViewable,
    Memory,
    MemoryBatch,
    Principal,
    RateLimitDecision,
    TenantId,
    UsageEvent,
    Workspace,
)
from dreamer.api.usage import UsageSink


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass
class InMemoryWorkspace:
    """A trivial in-memory workspace materialized as a real temp directory.

    Implements ``FileViewable`` so it round-trips through ``LTMStore`` /
    ``ContextStore`` engines that require file access. ``id`` is opaque and
    stable for the workspace's lifetime.
    """

    id: str
    tenant_id: TenantId
    path: Path
    metadata: Mapping[str, Any] = field(default_factory=dict)

    async def file_view(self) -> Path:
        return self.path


@implements(STMStore, version=1)
class InMemorySTMStore:
    """An in-memory STM store.

    Multi-tenant aware: enforces ``TenantScope`` on every method, supports
    idempotency via ``idempotency_key`` scoped to ``(tenant_id, key)``, and
    exposes the same lease-aware semantics SQL-backed stores must satisfy.
    """

    multi_tenant: ClassVar[bool] = True

    def __init__(self) -> None:
        self._memories: dict[str, Memory] = {}
        self._idempotency: dict[tuple[TenantId, str], str] = {}
        self._lock = asyncio.Lock()

    async def submit(self, memory: Memory, *, ctx: SubmitContext) -> Memory:
        TenantScope.assert_matches(ctx.tenant_id)
        async with self._lock:
            if memory.idempotency_key is not None:
                key = (ctx.tenant_id, memory.idempotency_key)
                existing_id = self._idempotency.get(key)
                if existing_id is not None:
                    return self._memories[existing_id]
            persisted = memory.model_copy(
                update={
                    "id": memory.id or str(uuid4()),
                    "tenant_id": ctx.tenant_id,
                }
            )
            assert persisted.id is not None
            self._memories[persisted.id] = persisted
            if persisted.idempotency_key is not None:
                self._idempotency[(ctx.tenant_id, persisted.idempotency_key)] = persisted.id
            return persisted

    async def list_unconsumed(self, *, ctx: ListUnconsumedContext) -> list[Memory]:
        TenantScope.assert_matches(ctx.tenant_id)
        out = [
            m
            for m in self._memories.values()
            if m.tenant_id == ctx.tenant_id
            and m.consumed_at is None
            and m.consumed_by_lease is None
        ]
        out.sort(key=lambda m: m.submitted_at)
        if ctx.limit is not None:
            out = out[: ctx.limit]
        return out

    async def claim_batch(self, *, ctx: ClaimContext) -> MemoryBatch:
        TenantScope.assert_matches(ctx.tenant_id)
        async with self._lock:
            available = [
                m
                for m in self._memories.values()
                if m.tenant_id == ctx.tenant_id
                and m.consumed_at is None
                and m.consumed_by_lease is None
            ]
            available.sort(key=lambda m: m.submitted_at)
            if ctx.max_batch_size is not None:
                available = available[: ctx.max_batch_size]
            claimed: list[Memory] = []
            for m in available:
                assert m.id is not None
                updated = m.model_copy(update={"consumed_by_lease": ctx.lease_id})
                self._memories[m.id] = updated
                claimed.append(updated)
            return MemoryBatch(
                lease_id=ctx.lease_id,
                tenant_id=ctx.tenant_id,
                memories=claimed,
                snapshot_at=ctx.snapshot_at or _utcnow(),
            )

    async def mark_consumed(self, *, ctx: MarkConsumedContext) -> None:
        TenantScope.assert_matches(ctx.tenant_id)
        consumed_at = ctx.consumed_at or _utcnow()
        async with self._lock:
            ids = set(ctx.memory_ids)
            for mem_id, m in list(self._memories.items()):
                if m.tenant_id != ctx.tenant_id:
                    continue
                if m.consumed_by_lease != ctx.lease_id:
                    continue
                if ids and mem_id not in ids:
                    continue
                self._memories[mem_id] = m.model_copy(update={"consumed_at": consumed_at})

    async def release_unconsumed(self, *, ctx: ReleaseContext) -> None:
        TenantScope.assert_matches(ctx.tenant_id)
        async with self._lock:
            for mem_id, m in list(self._memories.items()):
                if m.tenant_id != ctx.tenant_id:
                    continue
                if m.consumed_by_lease != ctx.lease_id:
                    continue
                if m.consumed_at is not None:
                    continue
                self._memories[mem_id] = m.model_copy(update={"consumed_by_lease": None})

    async def count_unconsumed(self, *, ctx: CountContext) -> int:
        TenantScope.assert_matches(ctx.tenant_id)
        return sum(
            1
            for m in self._memories.values()
            if m.tenant_id == ctx.tenant_id
            and m.consumed_at is None
            and m.consumed_by_lease is None
        )

    async def release_for_expired_leases(self, *, ctx: ReclaimContext) -> int:
        TenantScope.assert_matches(ctx.tenant_id)
        if not ctx.expired_lease_ids:
            return 0
        released = 0
        async with self._lock:
            for mem_id, m in list(self._memories.items()):
                if m.tenant_id != ctx.tenant_id:
                    continue
                if m.consumed_at is not None:
                    continue
                if m.consumed_by_lease in ctx.expired_lease_ids:
                    self._memories[mem_id] = m.model_copy(update={"consumed_by_lease": None})
                    released += 1
        return released

    async def purge_consumed(self, *, ctx: PurgeConsumedContext) -> int:
        TenantScope.assert_matches(ctx.tenant_id)
        purged = 0
        async with self._lock:
            for mem_id, m in list(self._memories.items()):
                if m.tenant_id != ctx.tenant_id:
                    continue
                if m.consumed_at is None:
                    continue
                if m.consumed_at < ctx.before:
                    del self._memories[mem_id]
                    if m.idempotency_key is not None:
                        self._idempotency.pop((ctx.tenant_id, m.idempotency_key), None)
                    purged += 1
        return purged


class _BaseInMemoryWorkspaceStore:
    """Shared workspace machinery for the in-memory LTM and context stores.

    Each store keeps a tenant-scoped "committed" tree of file paths → bytes.
    Opening a workspace materializes that tree to a temp directory; committing
    rsyncs the directory back into the in-memory tree and returns a ``Diff``.
    """

    multi_tenant: ClassVar[bool] = True
    workspace_capabilities: ClassVar[frozenset[type]] = frozenset({FileViewable})

    def __init__(self, root: Path | None = None) -> None:
        self._root: Path | None = root
        self._committed: dict[TenantId, dict[str, bytes]] = {}
        self._open_workspaces: dict[str, InMemoryWorkspace] = {}
        self._lock = asyncio.Lock()

    def _ensure_root(self) -> Path:
        if self._root is None:
            import tempfile

            self._root = Path(tempfile.mkdtemp(prefix="dreamer-fake-"))
        self._root.mkdir(parents=True, exist_ok=True)
        return self._root

    def _committed_for(self, tenant_id: TenantId) -> dict[str, bytes]:
        return self._committed.setdefault(tenant_id, {})

    async def _open(self, *, ctx: OpenWorkspaceContext) -> Workspace:
        TenantScope.assert_matches(ctx.tenant_id)
        async with self._lock:
            ws_id = str(uuid4())
            base = self._ensure_root() / ws_id
            base.mkdir(parents=True, exist_ok=True)
            for relpath, content in self._committed_for(ctx.tenant_id).items():
                target = base / relpath
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
            ws = InMemoryWorkspace(id=ws_id, tenant_id=ctx.tenant_id, path=base)
            self._open_workspaces[ws_id] = ws
            return ws

    async def _commit(self, ws: Workspace, *, ctx: CommitWorkspaceContext) -> Diff:
        TenantScope.assert_matches(ctx.tenant_id)
        async with self._lock:
            internal = self._open_workspaces.get(ws.id)
            if internal is None:
                raise RuntimeError(f"unknown workspace id: {ws.id}")
            new_tree: dict[str, bytes] = {}
            for path in internal.path.rglob("*"):
                if path.is_file():
                    rel = path.relative_to(internal.path).as_posix()
                    new_tree[rel] = path.read_bytes()
            old_tree = self._committed_for(ctx.tenant_id)
            added = sorted(set(new_tree) - set(old_tree))
            deleted = sorted(set(old_tree) - set(new_tree))
            modified = sorted(
                k for k in set(new_tree) & set(old_tree) if new_tree[k] != old_tree[k]
            )
            self._committed[ctx.tenant_id] = new_tree
            self._open_workspaces.pop(ws.id, None)
            self._cleanup_dir(internal.path)
            return Diff(added=added, modified=modified, deleted=deleted)

    async def _discard(self, ws: Workspace, *, ctx: DiscardWorkspaceContext) -> None:
        TenantScope.assert_matches(ctx.tenant_id)
        async with self._lock:
            internal = self._open_workspaces.pop(ws.id, None)
            if internal is not None:
                self._cleanup_dir(internal.path)

    @staticmethod
    def _cleanup_dir(path: Path) -> None:
        if not path.exists():
            return
        for child in sorted(path.rglob("*"), reverse=True):
            if child.is_file() or child.is_symlink():
                child.unlink(missing_ok=True)
            elif child.is_dir():
                child.rmdir()
        path.rmdir()


@implements(LTMStore, version=1)
@implements(ContextPendingStore, version=1)
class InMemoryLTMStore(_BaseInMemoryWorkspaceStore):
    """In-memory LTM store with the optional ``ContextPendingStore`` capability."""

    def __init__(self, root: Path | None = None) -> None:
        super().__init__(root=root)
        self._pending: dict[TenantId, Diff] = {}

    async def open_workspace(self, *, ctx: OpenWorkspaceContext) -> Workspace:
        return await self._open(ctx=ctx)

    async def commit_workspace(self, ws: Workspace, *, ctx: CommitWorkspaceContext) -> Diff:
        return await self._commit(ws, ctx=ctx)

    async def discard_workspace(self, ws: Workspace, *, ctx: DiscardWorkspaceContext) -> None:
        await self._discard(ws, ctx=ctx)

    async def set_context_pending(self, diff: Diff, *, ctx: SetContextPendingContext) -> None:
        TenantScope.assert_matches(ctx.tenant_id)
        self._pending[ctx.tenant_id] = diff

    async def get_context_pending(self, *, ctx: GetContextPendingContext) -> Diff | None:
        TenantScope.assert_matches(ctx.tenant_id)
        return self._pending.get(ctx.tenant_id)

    async def clear_context_pending(self, *, ctx: ClearContextPendingContext) -> None:
        TenantScope.assert_matches(ctx.tenant_id)
        self._pending.pop(ctx.tenant_id, None)


@implements(ContextStore, version=1)
class InMemoryContextStore(_BaseInMemoryWorkspaceStore):
    async def open_workspace(self, *, ctx: OpenWorkspaceContext) -> Workspace:
        return await self._open(ctx=ctx)

    async def commit_workspace(self, ws: Workspace, *, ctx: CommitWorkspaceContext) -> Diff:
        return await self._commit(ws, ctx=ctx)

    async def discard_workspace(self, ws: Workspace, *, ctx: DiscardWorkspaceContext) -> None:
        await self._discard(ws, ctx=ctx)


@implements(DreamLeaseStore, version=1)
class InMemoryDreamLeaseStore:
    """In-memory dream lease store with TTL-based expiry."""

    multi_tenant: ClassVar[bool] = True

    def __init__(self, default_ttl_seconds: float = 1800.0) -> None:
        self._default_ttl = default_ttl_seconds
        self._leases: dict[str, DreamLease] = {}
        self._lock = asyncio.Lock()

    async def acquire(self, *, ctx: AcquireLeaseContext) -> DreamLease | None:
        TenantScope.assert_matches(ctx.tenant_id)
        async with self._lock:
            now = _utcnow()
            for lease in self._leases.values():
                if lease.tenant_id != ctx.tenant_id:
                    continue
                if lease.expires_at > now:
                    return None
            ttl = ctx.ttl_seconds or self._default_ttl
            lease = DreamLease(
                id=str(uuid4()),
                tenant_id=ctx.tenant_id,
                acquired_at=now,
                expires_at=now + timedelta(seconds=ttl),
            )
            self._leases[lease.id] = lease
            return lease

    async def renew(self, *, ctx: RenewLeaseContext) -> bool:
        TenantScope.assert_matches(ctx.tenant_id)
        async with self._lock:
            lease = self._leases.get(ctx.lease_id)
            if lease is None or lease.tenant_id != ctx.tenant_id:
                return False
            if lease.expires_at <= _utcnow():
                return False
            ttl = ctx.ttl_seconds or self._default_ttl
            self._leases[ctx.lease_id] = lease.model_copy(
                update={"expires_at": _utcnow() + timedelta(seconds=ttl)}
            )
            return True

    async def release(self, *, ctx: ReleaseLeaseContext) -> None:
        TenantScope.assert_matches(ctx.tenant_id)
        async with self._lock:
            lease = self._leases.get(ctx.lease_id)
            if lease is not None and lease.tenant_id == ctx.tenant_id:
                self._leases.pop(ctx.lease_id, None)

    async def reclaim_expired(self, *, ctx: ReclaimLeasesContext) -> frozenset[str]:
        TenantScope.assert_matches(ctx.tenant_id)
        now = _utcnow()
        reclaimed: set[str] = set()
        async with self._lock:
            for lease_id, lease in list(self._leases.items()):
                if lease.tenant_id != ctx.tenant_id:
                    continue
                if lease.expires_at <= now:
                    del self._leases[lease_id]
                    reclaimed.add(lease_id)
        return frozenset(reclaimed)

    @asynccontextmanager
    async def fast_forward(self, *, by: timedelta) -> AsyncIterator[None]:
        """Test helper: shift all known leases backward in time so that
        ``expires_at`` appears to have already passed by the supplied delta.
        """
        snapshot = dict(self._leases)
        try:
            for lease_id, lease in snapshot.items():
                self._leases[lease_id] = lease.model_copy(
                    update={"expires_at": lease.expires_at - by}
                )
            yield
        finally:
            for lease_id, lease in snapshot.items():
                if lease_id in self._leases:
                    self._leases[lease_id] = lease


@implements(STMSerializer, version=1)
class InMemorySTMSerializer:
    """A trivial deterministic serializer used by the conformance suite."""

    multi_tenant: ClassVar[bool] = True
    kind: ClassVar[str] = "fake-jsonl"

    async def write(
        self,
        batch: MemoryBatch,
        *,
        target: Path,
        ctx: SerializeContext,
        services: SerializeServices,
    ) -> None:
        TenantScope.assert_matches(ctx.tenant_id)
        target.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240 — test fake, blocking I/O acceptable
        out = target / "batch.jsonl"
        out.write_text(  # noqa: ASYNC240 — test fake, blocking I/O acceptable
            "\n".join(m.model_dump_json() for m in batch.memories),
            encoding="utf-8",
        )

    def prompt_fragment(self, batch: MemoryBatch) -> str:
        return f"{len(batch.memories)} memories serialized as JSONL at inbox/batch.jsonl."


@dataclass
class DeterministicDreamCall:
    phase: str
    tenant_id: TenantId
    lease_id: str
    extras: Mapping[str, Any]


@implements(LTMPhaseRunner, version=1)
@implements(ContextPhaseRunner, version=1)
class DeterministicDreamEngine:
    """Deterministic dream engine for tests.

    Records every phase invocation and writes a fixed file inside each
    workspace so that store ``commit_workspace`` calls produce a non-empty
    ``Diff``. Useful for orchestrator integration tests that need a stable
    payload without exercising any LLM.
    """

    multi_tenant: ClassVar[bool] = True
    workspace_requirements: ClassVar[Mapping[str, frozenset[type]]] = {
        "ltm": frozenset({FileViewable}),
        "context": frozenset({FileViewable}),
    }
    accepted_serializer_kinds: ClassVar[frozenset[str]] = frozenset({"*"})

    def __init__(
        self,
        ltm_filename: str = "ltm-note.md",
        context_filename: str = "AGENTS.md",
    ) -> None:
        self._ltm_filename = ltm_filename
        self._context_filename = context_filename
        self.calls: list[DeterministicDreamCall] = []

    async def run_ltm_phase(
        self, *, ctx: LTMPhaseContext, services: LTMPhaseServices
    ) -> None:
        TenantScope.assert_matches(ctx.tenant_id)
        ws_path = await services.ltm_workspace.file_view()  # type: ignore[attr-defined]
        ws_path.mkdir(parents=True, exist_ok=True)
        target = ws_path / self._ltm_filename
        existing = target.read_text(encoding="utf-8") if target.exists() else ""
        target.write_text(
            existing + f"\n# batch {ctx.batch.lease_id} ({len(ctx.batch.memories)} memories)\n",
            encoding="utf-8",
        )
        self.calls.append(
            DeterministicDreamCall(
                phase="ltm",
                tenant_id=ctx.tenant_id,
                lease_id=ctx.lease_id,
                extras={"batch_size": len(ctx.batch.memories)},
            )
        )

    async def run_context_phase(
        self, *, ctx: ContextPhaseContext, services: ContextPhaseServices
    ) -> None:
        TenantScope.assert_matches(ctx.tenant_id)
        ws_path = await services.context_workspace.file_view()  # type: ignore[attr-defined]
        ws_path.mkdir(parents=True, exist_ok=True)
        target = ws_path / self._context_filename
        target.write_text(
            f"# AGENTS\n\nLTM diff added: {len(ctx.ltm_diff.added)}; "
            f"modified: {len(ctx.ltm_diff.modified)}; "
            f"deleted: {len(ctx.ltm_diff.deleted)}.\n",
            encoding="utf-8",
        )
        self.calls.append(
            DeterministicDreamCall(
                phase="context",
                tenant_id=ctx.tenant_id,
                lease_id=ctx.lease_id,
                extras={
                    "added": len(ctx.ltm_diff.added),
                    "modified": len(ctx.ltm_diff.modified),
                    "deleted": len(ctx.ltm_diff.deleted),
                },
            )
        )


@implements(UsageSink, version=1)
class CollectingUsageSink:
    """Records every usage event for inspection."""

    multi_tenant: ClassVar[bool] = True

    def __init__(self) -> None:
        self.events: list[UsageEvent] = []

    async def record(self, event: UsageEvent, *, ctx: UsageContext) -> None:
        self.events.append(event)


@implements(AuditSink, version=1)
class CollectingAuditSink:
    """Records every audit event for inspection."""

    multi_tenant: ClassVar[bool] = True

    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    async def record(self, event: AuditEvent, *, ctx: AuditContext) -> None:
        self.events.append(event)


@implements(RateLimiter, version=1)
class NoOpRateLimiter:
    """Always allows; multi-tenant safe."""

    multi_tenant: ClassVar[bool] = True

    async def check(
        self,
        *,
        principal: Principal,
        tenant_id: TenantId,
        action: str,
        ctx: RateLimitContext,
    ) -> RateLimitDecision:
        return RateLimitDecision(allowed=True)


@implements(JobQueue, version=1)
class NoOpJobQueue:
    """In-memory job queue. ``publish`` invokes the subscribed handler in a
    detached task so triggers never block; failure to subscribe before publish
    raises ``RuntimeError``.
    """

    multi_tenant: ClassVar[bool] = True

    def __init__(self) -> None:
        self._handler: Callable[[DreamJob], Awaitable[None]] | None = None
        self.published: list[DreamJob] = []

    async def publish(self, job: DreamJob, *, ctx: PublishContext) -> None:
        self.published.append(job)
        handler = self._handler
        if handler is None:
            return

        async def _run() -> None:
            await handler(job)

        asyncio.create_task(_run())

    async def subscribe(
        self,
        *,
        handler: Callable[[DreamJob], Awaitable[None]],
        ctx: SubscribeContext,
    ) -> None:
        self._handler = handler


__all__ = [
    "CollectingAuditSink",
    "CollectingUsageSink",
    "DeterministicDreamCall",
    "DeterministicDreamEngine",
    "InMemoryContextStore",
    "InMemoryDreamLeaseStore",
    "InMemoryLTMStore",
    "InMemorySTMSerializer",
    "InMemorySTMStore",
    "InMemoryWorkspace",
    "NoOpJobQueue",
    "NoOpRateLimiter",
]
