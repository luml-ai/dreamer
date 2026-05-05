from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar
from uuid import uuid4

import pytest
import pytest_asyncio

from dreamer.api.capabilities import Transactional
from dreamer.api.compat import implements
from dreamer.api.contexts import (
    AcquireLeaseContext,
    CountContext,
    DreamGateContext,
    DreamGateServices,
    GetContextPendingContext,
    LifecycleContext,
    ListUnconsumedContext,
    PreDreamContext,
    PreDreamServices,
    SecretContext,
    SubmitContext,
    TenantConfigLookupContext,
    TenantRegistryContext,
    TxBeginContext,
    TxCommitContext,
    TxPrepareContext,
    TxRollbackContext,
)
from dreamer.api.dream import (
    ContextPhaseRunner,
    DreamGate,
    LTMPhaseRunner,
)
from dreamer.api.hooks import (
    DreamFailedHook,
    DreamProgressHook,
    PostDreamHook,
    PreDreamHook,
)
from dreamer.api.secrets import SecretResolver
from dreamer.api.tenants import (
    TenantConfigProvider,
    TenantRegistry,
    TenantScope,
)
from dreamer.api.types import (
    Diff,
    DreamJob,
    GateDecision,
    Memory,
    SecretValue,
    TenantConfig,
    TenantId,
)
from dreamer.contrib.jobs.inproc import InProcessJobQueue
from dreamer.server.orchestrator import Orchestrator, StmRetentionConfig
from dreamer.server.runtime import HookRegistry
from dreamer.testing.fakes import (
    CollectingAuditSink,
    CollectingUsageSink,
    DeterministicDreamEngine,
    InMemoryContextStore,
    InMemoryDreamLeaseStore,
    InMemoryLTMStore,
    InMemorySTMStore,
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


@implements(TenantRegistry, version=1)
class FakeTenantRegistry:
    multi_tenant: ClassVar[bool] = True

    def __init__(self, tenants: list[TenantId] | None = None) -> None:
        self.tenants = list(tenants or ["default"])

    async def list_tenants(self, *, ctx: TenantRegistryContext) -> list[TenantId]:
        return list(self.tenants)

    async def exists(self, tenant_id: TenantId, *, ctx: TenantRegistryContext) -> bool:
        return tenant_id in self.tenants


@implements(TenantConfigProvider, version=1)
class FakeTenantConfigProvider:
    multi_tenant: ClassVar[bool] = True

    def __init__(
        self, overrides: dict[TenantId, TenantConfig] | None = None
    ) -> None:
        self.overrides = overrides or {}

    async def get(
        self, tenant_id: TenantId, *, ctx: TenantConfigLookupContext
    ) -> TenantConfig:
        return self.overrides.get(tenant_id, TenantConfig())


@implements(SecretResolver, version=1)
class FakeSecretResolver:
    multi_tenant: ClassVar[bool] = True

    async def get(
        self, name: str, *, tenant_id: TenantId | None, ctx: SecretContext
    ) -> SecretValue:
        return SecretValue(value=f"<{name}>")


@implements(PreDreamHook, version=1)
class CountingPreDream:
    multi_tenant: ClassVar[bool] = True

    def __init__(self) -> None:
        self.calls: list[int] = []

    async def on_pre_dream(
        self, *, ctx: PreDreamContext, services: PreDreamServices
    ) -> None:
        self.calls.append(ctx.unconsumed_count)


@implements(PreDreamHook, version=1)
class RaisingPreDream:
    multi_tenant: ClassVar[bool] = True

    async def on_pre_dream(
        self, *, ctx: PreDreamContext, services: PreDreamServices
    ) -> None:
        raise RuntimeError("pre_dream blew up")


@implements(PostDreamHook, version=1)
class CountingPostDream:
    multi_tenant: ClassVar[bool] = True

    def __init__(self) -> None:
        self.calls: list[Mapping[str, Any]] = []

    async def on_post_dream(self, *, ctx: Any, services: Any) -> None:
        self.calls.append(
            {
                "success": ctx.success,
                "batch_size": ctx.batch_size,
                "ltm_diff": ctx.ltm_diff,
                "context_diff": ctx.context_diff,
                "resumed": ctx.resumed,
                "error": ctx.error,
            }
        )


@implements(PostDreamHook, version=1)
class RaisingPostDream:
    multi_tenant: ClassVar[bool] = True

    def __init__(self) -> None:
        self.called = 0

    async def on_post_dream(self, *, ctx: Any, services: Any) -> None:
        self.called += 1
        raise RuntimeError("post_dream blew up")


@implements(PostDreamHook, version=1)
class _FooHook:
    """Reads ``ctx.params['foo']`` so tests can verify per-tenant override merging."""

    multi_tenant: ClassVar[bool] = True

    def __init__(self, foo: str = "default-foo") -> None:
        self.foo = foo
        self.observed: list[str] = []

    async def on_post_dream(self, *, ctx: Any, services: Any) -> None:
        self.observed.append(ctx.params.get("foo", self.foo))


class _Calls:
    def __init__(self) -> None:
        self.ltm: int = 0
        self.context: int = 0


@implements(LTMPhaseRunner, version=1)
@implements(ContextPhaseRunner, version=1)
class FailingLTMEngine:
    multi_tenant: ClassVar[bool] = True
    workspace_requirements: ClassVar[Mapping[str, frozenset[type]]] = {
        "ltm": frozenset(),
        "context": frozenset(),
    }
    accepted_serializer_kinds: ClassVar[frozenset[str]] = frozenset({"*"})

    def __init__(self) -> None:
        self.calls = _Calls()

    async def run_ltm_phase(self, *, ctx: Any, services: Any) -> None:
        self.calls.ltm += 1
        raise ConnectionError("LTM engine fails")

    async def run_context_phase(self, *, ctx: Any, services: Any) -> None:
        self.calls.context += 1


@implements(DreamProgressHook, version=1)
class _ProgressHook:
    multi_tenant: ClassVar[bool] = True

    def __init__(self) -> None:
        self.captured: list[dict[str, Any]] = []

    async def on_dream_progress(self, *, ctx: Any, services: Any) -> None:
        self.captured.append(
            {"phase": ctx.phase, "message": ctx.message, "payload": dict(ctx.payload)}
        )


@implements(LTMPhaseRunner, version=1)
@implements(ContextPhaseRunner, version=1)
class _EmittingEngine:
    multi_tenant: ClassVar[bool] = True
    workspace_requirements: ClassVar[Mapping[str, frozenset[type]]] = {
        "ltm": frozenset(),
        "context": frozenset(),
    }
    accepted_serializer_kinds: ClassVar[frozenset[str]] = frozenset({"*"})

    async def run_ltm_phase(self, *, ctx: Any, services: Any) -> None:
        ws_path = await services.ltm_workspace.file_view()
        ws_path.mkdir(parents=True, exist_ok=True)
        (ws_path / "L").write_text("ltm", encoding="utf-8")
        await services.emit_progress("ltm working", {"step": "ltm"})

    async def run_context_phase(self, *, ctx: Any, services: Any) -> None:
        ws_path = await services.context_workspace.file_view()
        ws_path.mkdir(parents=True, exist_ok=True)
        (ws_path / "AGENTS.md").write_text("ctx", encoding="utf-8")
        await services.emit_progress("ctx working", {"step": "ctx"})


@implements(DreamFailedHook, version=1)
class _CapturingDreamFailed:
    multi_tenant: ClassVar[bool] = True

    def __init__(self) -> None:
        self.records: list[Any] = []

    async def on_dream_failed(self, *, ctx: Any, services: Any) -> None:
        self.records.append(ctx)


@implements(LTMPhaseRunner, version=1)
@implements(ContextPhaseRunner, version=1)
class FailingContextEngine:
    multi_tenant: ClassVar[bool] = True
    workspace_requirements: ClassVar[Mapping[str, frozenset[type]]] = {
        "ltm": frozenset(),
        "context": frozenset(),
    }
    accepted_serializer_kinds: ClassVar[frozenset[str]] = frozenset({"*"})

    def __init__(self, fail_count: int = 1) -> None:
        self.calls = _Calls()
        self._remaining_failures = fail_count

    async def run_ltm_phase(self, *, ctx: Any, services: Any) -> None:
        self.calls.ltm += 1
        ws_path = await services.ltm_workspace.file_view()
        ws_path.mkdir(parents=True, exist_ok=True)
        (ws_path / "topics" / "x.md").parent.mkdir(parents=True, exist_ok=True)
        (ws_path / "topics" / "x.md").write_text(
            f"# from batch {len(ctx.batch.memories)}\n", encoding="utf-8"
        )

    async def run_context_phase(self, *, ctx: Any, services: Any) -> None:
        self.calls.context += 1
        if self._remaining_failures > 0:
            self._remaining_failures -= 1
            raise ConnectionError("context engine fails")
        ws_path = await services.context_workspace.file_view()
        ws_path.mkdir(parents=True, exist_ok=True)
        (ws_path / "AGENTS.md").write_text("# context\n", encoding="utf-8")


@dataclass
class _TxRecord:
    tx_id: str
    ops: list[str] = field(default_factory=list)


@implements(Transactional, version=1)
class _TransactionalLTM(InMemoryLTMStore):
    def __init__(self) -> None:
        super().__init__()
        self.tx_log: list[_TxRecord] = []
        self._prepare_returns: bool = True

    async def begin(self, *, ctx: TxBeginContext) -> _TxRecord:
        tx = _TxRecord(tx_id=str(uuid4()))
        self.tx_log.append(tx)
        tx.ops.append("begin")
        return tx

    async def prepare(self, tx: _TxRecord, *, ctx: TxPrepareContext) -> bool:
        tx.ops.append("prepare")
        return self._prepare_returns

    async def commit(self, tx: _TxRecord, *, ctx: TxCommitContext) -> None:
        tx.ops.append("commit")

    async def rollback(self, tx: _TxRecord, *, ctx: TxRollbackContext) -> None:
        tx.ops.append("rollback")


@implements(Transactional, version=1)
class _TransactionalContext(InMemoryContextStore):
    def __init__(self) -> None:
        super().__init__()
        self.tx_log: list[_TxRecord] = []
        self._prepare_returns: bool = True

    async def begin(self, *, ctx: TxBeginContext) -> _TxRecord:
        tx = _TxRecord(tx_id=str(uuid4()))
        self.tx_log.append(tx)
        tx.ops.append("begin")
        return tx

    async def prepare(self, tx: _TxRecord, *, ctx: TxPrepareContext) -> bool:
        tx.ops.append("prepare")
        return self._prepare_returns

    async def commit(self, tx: _TxRecord, *, ctx: TxCommitContext) -> None:
        tx.ops.append("commit")

    async def rollback(self, tx: _TxRecord, *, ctx: TxRollbackContext) -> None:
        tx.ops.append("rollback")


@implements(DreamGate, version=1)
class AlwaysSkipGate:
    multi_tenant: ClassVar[bool] = True

    def __init__(self, reason: str = "skip") -> None:
        self.reason = reason
        self.calls = 0

    async def check(
        self, *, ctx: DreamGateContext, services: DreamGateServices
    ) -> GateDecision:
        self.calls += 1
        return GateDecision(proceed=False, reason=self.reason)


@implements(DreamGate, version=1)
class AlwaysProceedGate:
    multi_tenant: ClassVar[bool] = True

    def __init__(self) -> None:
        self.calls = 0

    async def check(
        self, *, ctx: DreamGateContext, services: DreamGateServices
    ) -> GateDecision:
        self.calls += 1
        return GateDecision(proceed=True)


@dataclass
class World:
    stm: InMemorySTMStore
    ltm: InMemoryLTMStore
    context_store: InMemoryContextStore
    leases: InMemoryDreamLeaseStore
    job_queue: InProcessJobQueue
    engine: Any
    audit: CollectingAuditSink
    usage: CollectingUsageSink
    secrets: FakeSecretResolver
    registry: FakeTenantRegistry
    config_provider: FakeTenantConfigProvider
    orchestrator: Orchestrator


async def make_world(
    *,
    ltm: InMemoryLTMStore | None = None,
    context_store: InMemoryContextStore | None = None,
    leases: InMemoryDreamLeaseStore | None = None,
    engine: Any | None = None,
    pre_dream: list[Any] | None = None,
    post_dream: list[Any] | None = None,
    post_ltm_update: list[Any] | None = None,
    post_context_update: list[Any] | None = None,
    on_dream_failed: list[Any] | None = None,
    dream_gates: list[Any] | None = None,
    overrides: dict[TenantId, TenantConfig] | None = None,
    tenants: list[TenantId] | None = None,
) -> World:
    stm = InMemorySTMStore()
    ltm = ltm or InMemoryLTMStore()
    cs = context_store or InMemoryContextStore()
    leases = leases or InMemoryDreamLeaseStore(default_ttl_seconds=60.0)
    queue = InProcessJobQueue()
    engine_ = engine or DeterministicDreamEngine()
    audit = CollectingAuditSink()
    usage = CollectingUsageSink()
    secrets = FakeSecretResolver()
    registry = FakeTenantRegistry(tenants=tenants)
    config_provider = FakeTenantConfigProvider(overrides=overrides)

    hooks_lists: dict[str, list[Any]] = {}
    if pre_dream:
        hooks_lists["pre_dream"] = pre_dream
    if post_dream:
        hooks_lists["post_dream"] = post_dream
    if post_ltm_update:
        hooks_lists["post_ltm_update"] = post_ltm_update
    if post_context_update:
        hooks_lists["post_context_update"] = post_context_update
    if on_dream_failed:
        hooks_lists["on_dream_failed"] = on_dream_failed
    hook_registry = HookRegistry()
    for slot, hooks in hooks_lists.items():
        for h in hooks:
            hook_registry.add(slot, h)

    orch = Orchestrator(
        stm_store=stm,
        ltm_store=ltm,
        context_store=cs,
        dream_lease_store=leases,
        ltm_phase_runner=engine_,
        context_phase_runner=engine_,
        tenant_registry=registry,
        tenant_config_provider=config_provider,
        job_queue=queue,
        hook_registry=hook_registry,
        audit_sinks=[audit],
        usage_sinks=[usage],
        secret_resolver=secrets,
        dream_gates=list(dream_gates or []),
        stm_retention=StmRetentionConfig(keep_days=None, cadence_seconds=86400),
        default_lease_ttl_seconds=60.0,
        heartbeat_interval_seconds=10.0,
    )
    await orch.start(ctx=LifecycleContext(request_id="test.start"))
    return World(
        stm=stm,
        ltm=ltm,
        context_store=cs,
        leases=leases,
        job_queue=queue,
        engine=engine_,
        audit=audit,
        usage=usage,
        secrets=secrets,
        registry=registry,
        config_provider=config_provider,
        orchestrator=orch,
    )


@pytest_asyncio.fixture
async def world() -> AsyncIterator[World]:
    w = await make_world()
    try:
        yield w
    finally:
        await w.orchestrator.stop(ctx=LifecycleContext(request_id="test.stop"))


async def _submit(stm: InMemorySTMStore, tenant_id: TenantId, n: int) -> list[Memory]:
    out: list[Memory] = []
    with TenantScope.set(tenant_id):
        for i in range(n):
            m = await stm.submit(
                Memory(
                    tenant_id=tenant_id,
                    agent_id="agent-x",
                    type="observation",
                    title=f"M{i}",
                    content=f"body {i}",
                    submitted_at=_utcnow(),
                ),
                ctx=SubmitContext(request_id=f"r{i}", tenant_id=tenant_id),
            )
            out.append(m)
    return out


async def _publish_and_wait(orch: Orchestrator, tenant_id: TenantId) -> None:
    await orch._handle_job(DreamJob(tenant_id=tenant_id, trigger_name="external"))


@pytest.mark.asyncio
async def test_happy_path_consume_and_diff(world: World) -> None:
    await _submit(world.stm, "default", 5)
    await _publish_and_wait(world.orchestrator, "default")

    with TenantScope.set("default"):
        unconsumed = await world.stm.list_unconsumed(
            ctx=ListUnconsumedContext(request_id="r", tenant_id="default")
        )
    assert unconsumed == []

    types = [e.event_type for e in world.audit.events]
    assert "dream.lease_acquired" in types
    assert "dream.batch_claimed" in types
    assert "dream.ltm_committed" in types
    assert "dream.context_committed" in types

    phases = sorted(c.phase for c in world.engine.calls)
    assert phases == ["context", "ltm"]

    state = await world.orchestrator.read_state()
    assert state["tenants"]["default"]["last_dream_success"] is True

    with TenantScope.set("default"):
        wm = await world.ltm.get_context_pending(
            ctx=GetContextPendingContext(request_id="r", tenant_id="default")
        )
    assert wm is None


@pytest.mark.asyncio
async def test_empty_batch_skips_phases_but_runs_post_dream() -> None:
    post = CountingPostDream()
    w = await make_world(post_dream=[post])
    try:
        await _publish_and_wait(w.orchestrator, "default")

        assert w.engine.calls == []

        # post_dream still runs even on empty batch.
        assert len(post.calls) == 1
        assert post.calls[0]["batch_size"] == 0
        assert post.calls[0]["success"] is True
        assert post.calls[0]["resumed"] is False
        assert post.calls[0]["ltm_diff"] is None
        assert post.calls[0]["context_diff"] is None
    finally:
        await w.orchestrator.stop(ctx=LifecycleContext(request_id="test.stop"))


@pytest.mark.asyncio
async def test_pre_dream_hook_sees_unconsumed_count() -> None:
    pre = CountingPreDream()
    w = await make_world(pre_dream=[pre])
    try:
        await _submit(w.stm, "default", 3)
        await _publish_and_wait(w.orchestrator, "default")
        assert pre.calls == [3]
    finally:
        await w.orchestrator.stop(ctx=LifecycleContext(request_id="test.stop"))


@pytest.mark.asyncio
async def test_pre_dream_raise_full_unwind() -> None:
    pre = RaisingPreDream()
    failed_hook = _CapturingDreamFailed()
    post = CountingPostDream()
    w = await make_world(
        pre_dream=[pre], post_dream=[post], on_dream_failed=[failed_hook]
    )
    try:
        await _submit(w.stm, "default", 4)
        await _publish_and_wait(w.orchestrator, "default")

        assert w.engine.calls == []

        # pre_dream raises BEFORE claim_batch, so memories stay unconsumed.
        from dreamer.api.contexts import CountContext

        with TenantScope.set("default"):
            count = await w.stm.count_unconsumed(
                ctx=CountContext(request_id="r", tenant_id="default")
            )
        assert count == 4

        assert len(failed_hook.records) == 1
        assert post.calls[-1]["success"] is False
        assert post.calls[-1]["error"] is not None
    finally:
        await w.orchestrator.stop(ctx=LifecycleContext(request_id="test.stop"))


@pytest.mark.asyncio
async def test_concurrent_submission_during_dream_goes_to_next_batch() -> None:
    # The engine pauses mid-LTM-phase so we can submit a memory mid-flight.
    sync = asyncio.Event()
    proceed = asyncio.Event()

    class PausingEngine:
        multi_tenant: ClassVar[bool] = True
        workspace_requirements: ClassVar[Mapping[str, frozenset[type]]] = {
            "ltm": frozenset(),
            "context": frozenset(),
        }
        accepted_serializer_kinds: ClassVar[frozenset[str]] = frozenset({"*"})

        def __init__(self) -> None:
            self.calls = _Calls()

        async def run_ltm_phase(self, *, ctx: Any, services: Any) -> None:
            self.calls.ltm += 1
            sync.set()
            await proceed.wait()
            ws_path = await services.ltm_workspace.file_view()
            ws_path.mkdir(parents=True, exist_ok=True)
            (ws_path / "L").write_text("ltm", encoding="utf-8")

        async def run_context_phase(self, *, ctx: Any, services: Any) -> None:
            self.calls.context += 1
            ws_path = await services.context_workspace.file_view()
            ws_path.mkdir(parents=True, exist_ok=True)
            (ws_path / "AGENTS.md").write_text("ctx", encoding="utf-8")

    PausingEngine_decorated = implements(LTMPhaseRunner, version=1)(PausingEngine)
    PausingEngine_decorated = implements(ContextPhaseRunner, version=1)(
        PausingEngine_decorated
    )

    w = await make_world(engine=PausingEngine_decorated())
    try:
        await _submit(w.stm, "default", 3)
        task = asyncio.create_task(_publish_and_wait(w.orchestrator, "default"))
        await sync.wait()

        new_m, = await _submit(w.stm, "default", 1)
        with TenantScope.set("default"):
            count = await w.stm.count_unconsumed(
                ctx=CountContext(request_id="r", tenant_id="default")
            )
        # Original 3 are claimed; only the new one is unconsumed.
        assert count == 1

        proceed.set()
        await task

        await _publish_and_wait(w.orchestrator, "default")
        with TenantScope.set("default"):
            count2 = await w.stm.count_unconsumed(
                ctx=CountContext(request_id="r", tenant_id="default")
            )
        assert count2 == 0
        assert new_m.id is not None
    finally:
        await w.orchestrator.stop(ctx=LifecycleContext(request_id="test.stop"))


@pytest.mark.asyncio
async def test_lease_conflict_skips_second_dream(world: World) -> None:
    # Manually grab a lease so the orchestrator's acquire returns None.
    with TenantScope.set("default"):

        lease = await world.leases.acquire(
            ctx=AcquireLeaseContext(
                request_id="r", tenant_id="default", ttl_seconds=60
            )
        )
    assert lease is not None
    await _submit(world.stm, "default", 1)
    await _publish_and_wait(world.orchestrator, "default")

    with TenantScope.set("default"):
        count = await world.stm.count_unconsumed(
            ctx=CountContext(request_id="r", tenant_id="default")
        )
    assert count == 1

    skip_events = [
        e
        for e in world.audit.events
        if e.event_type == "dream.skipped"
        and e.payload.get("reason") == "lease_held"
    ]
    assert len(skip_events) >= 1


@pytest.mark.asyncio
async def test_lease_expiration_releases_orphaned_batch() -> None:
    # Simulate a crash: hand-craft a lease + claim against it, then expire.
    leases = InMemoryDreamLeaseStore(default_ttl_seconds=0.05)
    w = await make_world(leases=leases)
    try:
        await _submit(w.stm, "default", 4)
        from dreamer.api.contexts import AcquireLeaseContext, ClaimContext

        with TenantScope.set("default"):
            lease = await leases.acquire(
                ctx=AcquireLeaseContext(
                    request_id="r", tenant_id="default", ttl_seconds=0.05
                )
            )
            assert lease is not None
            batch = await w.stm.claim_batch(
                ctx=ClaimContext(
                    request_id="r", tenant_id="default", lease_id=lease.id
                )
            )
            assert len(batch.memories) == 4

        await asyncio.sleep(0.08)

        # Orchestrator should reclaim the expired lease, release the orphaned
        # batch, and re-claim under a fresh lease.
        await _publish_and_wait(w.orchestrator, "default")

        with TenantScope.set("default"):
            count = await w.stm.count_unconsumed(
                ctx=CountContext(request_id="r", tenant_id="default")
            )
        assert count == 0
    finally:
        await w.orchestrator.stop(ctx=LifecycleContext(request_id="test.stop"))


@pytest.mark.asyncio
async def test_ltm_phase_failure_rolls_back_batch() -> None:
    engine = FailingLTMEngine()
    post = CountingPostDream()
    w = await make_world(engine=engine, post_dream=[post])
    try:
        await _submit(w.stm, "default", 3)
        await _publish_and_wait(w.orchestrator, "default")

        with TenantScope.set("default"):
            count = await w.stm.count_unconsumed(
                ctx=CountContext(request_id="r", tenant_id="default")
            )
        assert count == 3

        with TenantScope.set("default"):
            wm = await w.ltm.get_context_pending(
                ctx=GetContextPendingContext(
                    request_id="r", tenant_id="default"
                )
            )
        assert wm is None

        assert post.calls[-1]["success"] is False
        assert post.calls[-1]["ltm_diff"] is None
        assert post.calls[-1]["context_diff"] is None
    finally:
        await w.orchestrator.stop(ctx=LifecycleContext(request_id="test.stop"))


@pytest.mark.asyncio
async def test_context_phase_failure_persists_watermark_then_resumes() -> None:
    engine = FailingContextEngine(fail_count=1)
    post = CountingPostDream()
    w = await make_world(engine=engine, post_dream=[post])
    try:
        await _submit(w.stm, "default", 5)
        await _publish_and_wait(w.orchestrator, "default")

        with TenantScope.set("default"):
            wm = await w.ltm.get_context_pending(
                ctx=GetContextPendingContext(
                    request_id="r", tenant_id="default"
                )
            )
        assert wm is not None
        assert isinstance(wm, Diff)

        with TenantScope.set("default"):
            count = await w.stm.count_unconsumed(
                ctx=CountContext(request_id="r", tenant_id="default")
            )
        assert count == 5

        ltm_before = engine.calls.ltm
        ctx_before = engine.calls.context
        await _publish_and_wait(w.orchestrator, "default")

        # Resume mode skips the LTM phase.
        assert engine.calls.ltm == ltm_before
        assert engine.calls.context == ctx_before + 1

        with TenantScope.set("default"):
            wm2 = await w.ltm.get_context_pending(
                ctx=GetContextPendingContext(
                    request_id="r", tenant_id="default"
                )
            )
        assert wm2 is None

        # Resume mode releases the batch, leaving STMs unconsumed.
        with TenantScope.set("default"):
            count2 = await w.stm.count_unconsumed(
                ctx=CountContext(request_id="r", tenant_id="default")
            )
        assert count2 == 5

        last = post.calls[-1]
        assert last["success"] is True
        assert last["resumed"] is True
    finally:
        await w.orchestrator.stop(ctx=LifecycleContext(request_id="test.stop"))


@pytest.mark.asyncio
async def test_resume_mode_with_new_stm_keeps_them_unconsumed() -> None:
    engine = FailingContextEngine(fail_count=1)
    w = await make_world(engine=engine)
    try:
        await _submit(w.stm, "default", 2)
        await _publish_and_wait(w.orchestrator, "default")  # fails, watermark set

        await _submit(w.stm, "default", 7)
        with TenantScope.set("default"):
            count = await w.stm.count_unconsumed(
                ctx=CountContext(request_id="r", tenant_id="default")
            )
        assert count == 9

        # Resume mode releases its 9-memory batch back to unconsumed.
        await _publish_and_wait(w.orchestrator, "default")

        with TenantScope.set("default"):
            count2 = await w.stm.count_unconsumed(
                ctx=CountContext(request_id="r", tenant_id="default")
            )
        assert count2 == 9
    finally:
        await w.orchestrator.stop(ctx=LifecycleContext(request_id="test.stop"))


@pytest.mark.asyncio
async def test_repeated_context_failure_keeps_watermark() -> None:
    engine = FailingContextEngine(fail_count=2)
    w = await make_world(engine=engine)
    try:
        await _submit(w.stm, "default", 1)
        await _publish_and_wait(w.orchestrator, "default")
        with TenantScope.set("default"):
            wm = await w.ltm.get_context_pending(
                ctx=GetContextPendingContext(
                    request_id="r", tenant_id="default"
                )
            )
        assert wm is not None

        await _publish_and_wait(w.orchestrator, "default")
        with TenantScope.set("default"):
            wm2 = await w.ltm.get_context_pending(
                ctx=GetContextPendingContext(
                    request_id="r", tenant_id="default"
                )
            )
        assert wm2 is not None

        # Third fire succeeds and clears the watermark.
        await _publish_and_wait(w.orchestrator, "default")
        with TenantScope.set("default"):
            wm3 = await w.ltm.get_context_pending(
                ctx=GetContextPendingContext(
                    request_id="r", tenant_id="default"
                )
            )
        assert wm3 is None
    finally:
        await w.orchestrator.stop(ctx=LifecycleContext(request_id="test.stop"))


@pytest.mark.asyncio
async def test_resume_context_cli_path() -> None:
    engine = FailingContextEngine(fail_count=1)
    w = await make_world(engine=engine)
    try:
        await _submit(w.stm, "default", 4)
        await _publish_and_wait(w.orchestrator, "default")  # fails, watermark set

        result = await w.orchestrator.resume_context("default")
        assert result["status"] == "ok"

        with TenantScope.set("default"):
            wm = await w.ltm.get_context_pending(
                ctx=GetContextPendingContext(
                    request_id="r", tenant_id="default"
                )
            )
        assert wm is None

        result2 = await w.orchestrator.resume_context("default")
        assert result2["status"] == "no_watermark"
    finally:
        await w.orchestrator.stop(ctx=LifecycleContext(request_id="test.stop"))


@pytest.mark.asyncio
async def test_transactional_two_phase_commit_happy_path() -> None:
    ltm = _TransactionalLTM()
    cs = _TransactionalContext()
    post = CountingPostDream()
    w = await make_world(ltm=ltm, context_store=cs, post_dream=[post])
    try:
        await _submit(w.stm, "default", 2)
        await _publish_and_wait(w.orchestrator, "default")

        assert ltm.tx_log and ltm.tx_log[-1].ops == ["begin", "prepare", "commit"]
        assert cs.tx_log and cs.tx_log[-1].ops == ["begin", "prepare", "commit"]

        with TenantScope.set("default"):
            wm = await ltm.get_context_pending(
                ctx=GetContextPendingContext(
                    request_id="r", tenant_id="default"
                )
            )
        assert wm is None

        assert post.calls[-1]["success"] is True
        assert post.calls[-1]["batch_size"] == 2
    finally:
        await w.orchestrator.stop(ctx=LifecycleContext(request_id="test.stop"))


@pytest.mark.asyncio
async def test_transactional_rollback_on_prepare_false() -> None:
    ltm = _TransactionalLTM()
    cs = _TransactionalContext()
    cs._prepare_returns = False
    w = await make_world(ltm=ltm, context_store=cs)
    try:
        await _submit(w.stm, "default", 3)
        await _publish_and_wait(w.orchestrator, "default")

        assert "rollback" in ltm.tx_log[-1].ops
        assert "rollback" in cs.tx_log[-1].ops

        with TenantScope.set("default"):
            count = await w.stm.count_unconsumed(
                ctx=CountContext(request_id="r", tenant_id="default")
            )
        assert count == 3
    finally:
        await w.orchestrator.stop(ctx=LifecycleContext(request_id="test.stop"))


@pytest.mark.asyncio
async def test_gate_skip_does_not_acquire_lease() -> None:
    gate = AlwaysSkipGate(reason="empty")
    w = await make_world(dream_gates=[gate])
    try:
        await _submit(w.stm, "default", 2)
        await _publish_and_wait(w.orchestrator, "default")

        with TenantScope.set("default"):
            count = await w.stm.count_unconsumed(
                ctx=CountContext(request_id="r", tenant_id="default")
            )
        assert count == 2

        skip_events = [
            e
            for e in w.audit.events
            if e.event_type == "dream.skipped"
            and e.payload.get("reason") == "empty"
        ]
        assert len(skip_events) == 1

        assert not any(
            e.event_type == "dream.lease_acquired" for e in w.audit.events
        )
    finally:
        await w.orchestrator.stop(ctx=LifecycleContext(request_id="test.stop"))


@pytest.mark.asyncio
async def test_gate_ordering_first_non_proceed_wins() -> None:
    skip_a = AlwaysSkipGate(reason="budget")
    proceed_b = AlwaysProceedGate()
    w = await make_world(dream_gates=[skip_a, proceed_b])
    try:
        await _submit(w.stm, "default", 1)
        await _publish_and_wait(w.orchestrator, "default")

        assert skip_a.calls == 1
        assert proceed_b.calls == 0  # first non-proceed gate short-circuits
    finally:
        await w.orchestrator.stop(ctx=LifecycleContext(request_id="test.stop"))


@pytest.mark.asyncio
async def test_post_dream_hook_failure_does_not_unwind() -> None:
    raising = RaisingPostDream()
    counting = CountingPostDream()
    w = await make_world(post_dream=[raising, counting])
    try:
        await _submit(w.stm, "default", 2)
        await _publish_and_wait(w.orchestrator, "default")

        # Raising hook does not prevent later hooks running.
        assert raising.called == 1
        assert len(counting.calls) == 1
        assert counting.calls[-1]["success"] is True

        # Hook failure must NOT unwind already-committed memories.
        with TenantScope.set("default"):
            count = await w.stm.count_unconsumed(
                ctx=CountContext(request_id="r", tenant_id="default")
            )
        assert count == 0

        hook_failures = [
            e for e in w.audit.events if e.event_type == "hook.failed"
        ]
        assert len(hook_failures) == 1
        assert hook_failures[0].payload["slot"] == "post_dream"
    finally:
        await w.orchestrator.stop(ctx=LifecycleContext(request_id="test.stop"))


@pytest.mark.asyncio
async def test_per_tenant_hook_params_override() -> None:
    # Without override, hook sees its construction-time `foo`.
    hook_a = _FooHook(foo="global-foo")
    w = await make_world(post_dream=[hook_a])
    try:
        await _submit(w.stm, "default", 1)
        await _publish_and_wait(w.orchestrator, "default")
        assert hook_a.observed == ["global-foo"]
    finally:
        await w.orchestrator.stop(ctx=LifecycleContext(request_id="test.stop"))

    # With a tenant override on the hook FQN, params are merged.
    hook_b = _FooHook(foo="global-foo")
    fqn = f"{_FooHook.__module__}.{_FooHook.__qualname__}"
    overrides = {
        "default": TenantConfig(hook_params={fqn: {"foo": "tenant-foo"}})
    }
    w2 = await make_world(post_dream=[hook_b], overrides=overrides)
    try:
        await _submit(w2.stm, "default", 1)
        await _publish_and_wait(w2.orchestrator, "default")
        assert hook_b.observed == ["tenant-foo"]
    finally:
        await w2.orchestrator.stop(ctx=LifecycleContext(request_id="test.stop"))


@pytest.mark.asyncio
async def test_progress_dispatch_to_progress_hook() -> None:
    progress_hook = _ProgressHook()
    stm = InMemorySTMStore()
    ltm = InMemoryLTMStore()
    cs = InMemoryContextStore()
    leases = InMemoryDreamLeaseStore(default_ttl_seconds=60.0)
    queue = InProcessJobQueue()
    secrets = FakeSecretResolver()
    audit = CollectingAuditSink()
    usage = CollectingUsageSink()
    registry = FakeTenantRegistry()
    config_provider = FakeTenantConfigProvider()
    hook_registry = HookRegistry()
    hook_registry.add("on_dream_progress", progress_hook)
    engine = _EmittingEngine()
    orch = Orchestrator(
        stm_store=stm,
        ltm_store=ltm,
        context_store=cs,
        dream_lease_store=leases,
        ltm_phase_runner=engine,
        context_phase_runner=engine,
        tenant_registry=registry,
        tenant_config_provider=config_provider,
        job_queue=queue,
        hook_registry=hook_registry,
        audit_sinks=[audit],
        usage_sinks=[usage],
        secret_resolver=secrets,
        dream_gates=[],
        stm_retention=StmRetentionConfig(keep_days=None, cadence_seconds=86400),
        default_lease_ttl_seconds=60.0,
        heartbeat_interval_seconds=10.0,
    )
    await orch.start(ctx=LifecycleContext(request_id="test.start"))
    try:
        await _submit(stm, "default", 1)
        await _publish_and_wait(orch, "default")
        # Progress hook is fire-and-forget; poll briefly for delivery.
        for _ in range(5):
            if len(progress_hook.captured) >= 2:
                break
            await asyncio.sleep(0.01)
        phases = sorted(c["phase"] for c in progress_hook.captured)
        assert "ltm" in phases
        assert "context" in phases
    finally:
        await orch.stop(ctx=LifecycleContext(request_id="test.stop"))


@pytest.mark.asyncio
async def test_stm_retention_sweep_purges_consumed() -> None:
    stm = InMemorySTMStore()
    ltm = InMemoryLTMStore()
    cs = InMemoryContextStore()
    leases = InMemoryDreamLeaseStore(default_ttl_seconds=60.0)
    queue = InProcessJobQueue()
    secrets = FakeSecretResolver()
    audit = CollectingAuditSink()
    usage = CollectingUsageSink()
    registry = FakeTenantRegistry()
    config_provider = FakeTenantConfigProvider()
    engine = DeterministicDreamEngine()
    hook_registry = HookRegistry()
    orch = Orchestrator(
        stm_store=stm,
        ltm_store=ltm,
        context_store=cs,
        dream_lease_store=leases,
        ltm_phase_runner=engine,
        context_phase_runner=engine,
        tenant_registry=registry,
        tenant_config_provider=config_provider,
        job_queue=queue,
        hook_registry=hook_registry,
        audit_sinks=[audit],
        usage_sinks=[usage],
        secret_resolver=secrets,
        dream_gates=[],
        stm_retention=StmRetentionConfig(keep_days=0, cadence_seconds=86400),
        default_lease_ttl_seconds=60.0,
        heartbeat_interval_seconds=10.0,
    )
    await orch.start(ctx=LifecycleContext(request_id="test.start"))
    try:
        await _submit(stm, "default", 3)
        await _publish_and_wait(orch, "default")

        await orch._run_retention_sweep()

        # `keep_days=0` purges anything consumed before "now".
        assert len(stm._memories) == 0
    finally:
        await orch.stop(ctx=LifecycleContext(request_id="test.stop"))


@pytest.mark.asyncio
async def test_purge_tenant_cli_path() -> None:
    stm = InMemorySTMStore()
    ltm = InMemoryLTMStore()
    cs = InMemoryContextStore()
    leases = InMemoryDreamLeaseStore(default_ttl_seconds=60.0)
    queue = InProcessJobQueue()
    secrets = FakeSecretResolver()
    audit = CollectingAuditSink()
    usage = CollectingUsageSink()
    registry = FakeTenantRegistry()
    config_provider = FakeTenantConfigProvider()
    engine = DeterministicDreamEngine()
    hook_registry = HookRegistry()
    orch = Orchestrator(
        stm_store=stm,
        ltm_store=ltm,
        context_store=cs,
        dream_lease_store=leases,
        ltm_phase_runner=engine,
        context_phase_runner=engine,
        tenant_registry=registry,
        tenant_config_provider=config_provider,
        job_queue=queue,
        hook_registry=hook_registry,
        audit_sinks=[audit],
        usage_sinks=[usage],
        secret_resolver=secrets,
        dream_gates=[],
        stm_retention=StmRetentionConfig(keep_days=None, cadence_seconds=86400),
        default_lease_ttl_seconds=60.0,
        heartbeat_interval_seconds=10.0,
    )
    await orch.start(ctx=LifecycleContext(request_id="test.start"))
    try:
        await _submit(stm, "default", 2)
        await _publish_and_wait(orch, "default")

        purged = await orch.purge_tenant(
            "default", before=_utcnow() + timedelta(days=1)
        )
        assert purged == 2
    finally:
        await orch.stop(ctx=LifecycleContext(request_id="test.stop"))


@pytest.mark.asyncio
async def test_trigger_dream_publishes_via_queue() -> None:
    stm = InMemorySTMStore()
    ltm = InMemoryLTMStore()
    cs = InMemoryContextStore()
    leases = InMemoryDreamLeaseStore(default_ttl_seconds=60.0)
    queue = InProcessJobQueue()
    secrets = FakeSecretResolver()
    audit = CollectingAuditSink()
    usage = CollectingUsageSink()
    registry = FakeTenantRegistry()
    config_provider = FakeTenantConfigProvider()
    engine = DeterministicDreamEngine()
    hook_registry = HookRegistry()
    orch = Orchestrator(
        stm_store=stm,
        ltm_store=ltm,
        context_store=cs,
        dream_lease_store=leases,
        ltm_phase_runner=engine,
        context_phase_runner=engine,
        tenant_registry=registry,
        tenant_config_provider=config_provider,
        job_queue=queue,
        hook_registry=hook_registry,
        audit_sinks=[audit],
        usage_sinks=[usage],
        secret_resolver=secrets,
        dream_gates=[],
        stm_retention=StmRetentionConfig(keep_days=None, cadence_seconds=86400),
        default_lease_ttl_seconds=60.0,
        heartbeat_interval_seconds=10.0,
    )
    await orch.start(ctx=LifecycleContext(request_id="test.start"))
    try:
        await _submit(stm, "default", 2)
        info = await orch.trigger_dream("default", "external")
        assert info["published"] is True
        for _ in range(20):
            with TenantScope.set("default"):
                count = await stm.count_unconsumed(
                    ctx=CountContext(request_id="r", tenant_id="default")
                )
            if count == 0:
                break
            await asyncio.sleep(0.01)
        with TenantScope.set("default"):
            count = await stm.count_unconsumed(
                ctx=CountContext(request_id="r", tenant_id="default")
            )
        assert count == 0
    finally:
        await orch.stop(ctx=LifecycleContext(request_id="test.stop"))


@pytest.mark.asyncio
async def test_unknown_tenant_skipped() -> None:
    w = await make_world(tenants=["default"])
    try:
        await _publish_and_wait(w.orchestrator, "missing")
        assert not any(
            e.event_type == "dream.lease_acquired" for e in w.audit.events
        )
    finally:
        await w.orchestrator.stop(ctx=LifecycleContext(request_id="test.stop"))
