"""Dream lifecycle orchestrator.

Implements the per-tenant, per-fire dream lifecycle described in
``spec/orchestrator.md``: gates → lease → batch claim → LTM phase → Context
phase → hooks → cleanup, with the cross-store consistency contract
(``ContextPendingStore`` watermark or ``Transactional`` two-phase commit).

The orchestrator subscribes to a ``JobQueue`` for ``DreamJob`` events. It also
exposes ``trigger_dream`` and ``read_state`` coroutines used by
``dreamer.server.control``.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar

from dreamer.api.audit import AuditSink
from dreamer.api.capabilities import Transactional
from dreamer.api.contexts import (
    AcquireLeaseContext,
    AuditContext,
    ClaimContext,
    ClearContextPendingContext,
    Clock,
    CommitWorkspaceContext,
    ContextPhaseContext,
    ContextPhaseServices,
    CountContext,
    DiscardWorkspaceContext,
    DreamFailedContext,
    DreamFailedServices,
    DreamGateContext,
    DreamGateServices,
    DreamProgressContext,
    DreamProgressServices,
    GetContextPendingContext,
    LifecycleContext,
    LTMPhaseContext,
    LTMPhaseServices,
    MarkConsumedContext,
    OpenWorkspaceContext,
    PostContextUpdateContext,
    PostContextUpdateServices,
    PostDreamContext,
    PostDreamServices,
    PostLTMUpdateContext,
    PostLTMUpdateServices,
    PreContextUpdateContext,
    PreContextUpdateServices,
    PreDreamContext,
    PreDreamServices,
    PreLTMUpdateContext,
    PreLTMUpdateServices,
    PurgeConsumedContext,
    ReclaimContext,
    ReclaimLeasesContext,
    ReleaseContext,
    ReleaseLeaseContext,
    RenewLeaseContext,
    SetContextPendingContext,
    SubscribeContext,
    TenantConfigLookupContext,
    TenantRegistryContext,
    TxBeginContext,
    TxCommitContext,
    TxPrepareContext,
    TxRollbackContext,
    UsageContext,
)
from dreamer.api.dream import ContextPhaseRunner, DreamGate, LTMPhaseRunner
from dreamer.api.errors import DreamFailedError
from dreamer.api.hooks import (
    DreamFailedHook,
    DreamProgressHook,
    PostContextUpdateHook,
    PostDreamHook,
    PostLTMUpdateHook,
    PreContextUpdateHook,
    PreDreamHook,
    PreLTMUpdateHook,
)
from dreamer.api.jobs import JobQueue
from dreamer.api.secrets import SecretResolver
from dreamer.api.stores import (
    ContextPendingStore,
    ContextStore,
    DreamLeaseStore,
    LTMStore,
    STMStore,
)
from dreamer.api.tenants import (
    TenantConfigProvider,
    TenantRegistry,
    TenantScope,
)
from dreamer.api.types import (
    AuditEvent,
    Diff,
    DreamJob,
    MemoryBatch,
    TenantConfig,
    TenantId,
    UsageEvent,
    Workspace,
)
from dreamer.api.usage import UsageSink
from dreamer.server.runtime import HookRegistry

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _request_id(prefix: str = "dream") -> str:
    return f"{prefix}.{uuid.uuid4()}"


@dataclass(slots=True)
class StmRetentionConfig:
    """STM retention sweep schedule.

    ``keep_days`` of ``None`` disables purging (consumed STMs retained forever).
    """

    keep_days: int | None = 30
    cadence_seconds: int = 86400


@dataclass(slots=True)
class _TenantState:
    last_dream_at: datetime | None = None
    last_dream_success: bool | None = None
    last_dream_error: str | None = None


@dataclass(slots=True)
class Orchestrator:
    """Per-tenant dream lifecycle owner.

    Instances are stored on the Starlette app so the control surface and tests
    can call into them. Implements ``Lifecycle`` so the framework starts the
    JobQueue subscription on ``start`` and drains in-flight dreams on ``stop``.
    """

    multi_tenant: ClassVar[bool] = True

    stm_store: STMStore
    ltm_store: LTMStore
    context_store: ContextStore
    dream_lease_store: DreamLeaseStore

    ltm_phase_runner: LTMPhaseRunner
    context_phase_runner: ContextPhaseRunner

    tenant_registry: TenantRegistry
    tenant_config_provider: TenantConfigProvider

    job_queue: JobQueue

    hook_registry: HookRegistry

    audit_sinks: list[AuditSink] = field(default_factory=list)
    usage_sinks: list[UsageSink] = field(default_factory=list)
    secret_resolver: SecretResolver | None = None

    dream_gates: list[DreamGate] = field(default_factory=list)
    stm_retention: StmRetentionConfig = field(default_factory=StmRetentionConfig)
    default_lease_ttl_seconds: float = 1800.0
    heartbeat_interval_seconds: float = 60.0
    clock: Clock = field(default=_utcnow)

    _tenant_states: dict[TenantId, _TenantState] = field(default_factory=dict)
    _active_leases: dict[TenantId, str] = field(default_factory=dict)
    _purge_task: asyncio.Task[None] | None = None
    _started: bool = False
    _shutting_down: bool = False

    async def start(self, *, ctx: LifecycleContext) -> None:
        if self._started:
            return
        self._started = True
        self._shutting_down = False
        await self.job_queue.subscribe(
            handler=self._handle_job,
            ctx=SubscribeContext(request_id=_request_id("subscribe")),
        )
        if (
            self.stm_retention.keep_days is not None
            and self.stm_retention.cadence_seconds > 0
        ):
            self._purge_task = asyncio.create_task(self._retention_loop())

    async def stop(self, *, ctx: LifecycleContext) -> None:
        self._shutting_down = True
        if self._purge_task is not None:
            self._purge_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._purge_task
            self._purge_task = None
        self._started = False

    async def trigger_dream(
        self,
        tenant_id: TenantId,
        trigger_name: str = "external",
    ) -> Mapping[str, Any]:
        """Publish a fire-and-forget DreamJob and return queueing info."""
        await self.job_queue.publish(
            DreamJob(tenant_id=tenant_id, trigger_name=trigger_name),
            ctx=_publish_context(tenant_id),
        )
        return {
            "tenant_id": tenant_id,
            "trigger_name": trigger_name,
            "published": True,
        }

    async def read_state(self) -> Mapping[str, Any]:
        """Return a snapshot of orchestrator state for the control surface."""
        tenants: dict[str, Any] = {}
        for tenant_id, state in self._tenant_states.items():
            tenants[tenant_id] = {
                "last_dream_at": state.last_dream_at.isoformat()
                if state.last_dream_at
                else None,
                "last_dream_success": state.last_dream_success,
                "last_dream_error": state.last_dream_error,
                "active_lease_id": self._active_leases.get(tenant_id),
            }
        return {"tenants": tenants}

    async def wait_for_active_lease_release(
        self,
        tenant_id: TenantId,
        timeout_seconds: float,
    ) -> None:
        """Wait until any in-flight dream lease for ``tenant_id`` releases.

        Polls every 100ms; returns silently once the slot is empty or the
        timeout elapses. Used by ``StaticTenantLifecycle.deprovision`` to drain
        in-flight dreams before sweeping ``TenantData`` impls.
        """
        deadline = self.clock() + timedelta(seconds=max(timeout_seconds, 0.0))
        while self._active_leases.get(tenant_id) is not None:
            if self.clock() >= deadline:
                return
            await asyncio.sleep(0.1)

    async def resume_context(self, tenant_id: TenantId) -> Mapping[str, Any]:
        """Run the context phase against the current watermark and exit.

        Used by ``dreamer dream --resume-context``. Returns ``{"status":
        "no_watermark"}`` if there is nothing to resume.
        """
        with TenantScope.set(tenant_id):
            tenant_config = await self.tenant_config_provider.get(
                tenant_id,
                ctx=TenantConfigLookupContext(
                    request_id=_request_id("resume"), tenant_id=tenant_id
                ),
            )
            watermark = await self._get_watermark(tenant_id)
            if watermark is None:
                return {"status": "no_watermark"}
            lease = await self.dream_lease_store.acquire(
                ctx=AcquireLeaseContext(
                    request_id=_request_id("resume"),
                    tenant_id=tenant_id,
                    ttl_seconds=self.default_lease_ttl_seconds,
                )
            )
            if lease is None:
                return {"status": "lease_held"}
            try:
                await self._run_context_only(
                    tenant_id=tenant_id,
                    lease_id=lease.id,
                    watermark=watermark,
                    tenant_config=tenant_config,
                    trigger_name="external",
                )
                return {"status": "ok"}
            finally:
                await self.dream_lease_store.release(
                    ctx=ReleaseLeaseContext(
                        request_id=_request_id("resume"),
                        tenant_id=tenant_id,
                        lease_id=lease.id,
                    )
                )

    async def purge_tenant(
        self,
        tenant_id: TenantId,
        *,
        before: datetime,
    ) -> int:
        """Run STM purge for a single tenant. Returns rows removed."""
        with TenantScope.set(tenant_id):
            return await self.stm_store.purge_consumed(
                ctx=PurgeConsumedContext(
                    request_id=_request_id("purge"),
                    tenant_id=tenant_id,
                    before=before,
                )
            )

    async def _handle_job(self, job: DreamJob) -> None:
        if self._shutting_down:
            return
        try:
            await self._run_dream(job.tenant_id, job.trigger_name)
        except Exception:  # noqa: BLE001 — top-level safety net
            logger.exception(
                "Unhandled error during dream tenant=%s trigger=%s",
                job.tenant_id,
                job.trigger_name,
            )

    async def _run_dream(self, tenant_id: TenantId, trigger_name: str) -> None:
        with TenantScope.set(tenant_id):
            ctx_reg = TenantRegistryContext(request_id=_request_id("dream"))
            try:
                exists = await self.tenant_registry.exists(tenant_id, ctx=ctx_reg)
            except Exception:  # noqa: BLE001 — registry transient errors should not crash
                logger.exception(
                    "tenant_registry.exists raised for tenant=%s; skipping", tenant_id
                )
                return
            if not exists:
                logger.info("dream skipped: tenant %s no longer exists", tenant_id)
                return

            tenant_config = await self.tenant_config_provider.get(
                tenant_id,
                ctx=TenantConfigLookupContext(
                    request_id=_request_id("dream"), tenant_id=tenant_id
                ),
            )

            # Gates run before lease acquisition.
            unconsumed_count_for_gates = await self._count_unconsumed(tenant_id)
            for gate in self.dream_gates:
                gate_ctx = DreamGateContext(
                    request_id=_request_id("gate"),
                    tenant_id=tenant_id,
                    trigger_name=trigger_name,
                    unconsumed_count=unconsumed_count_for_gates,
                    params=self._params_for(gate, tenant_config),
                )
                gate_services = self._build_dream_gate_services()
                decision = await gate.check(ctx=gate_ctx, services=gate_services)
                if not decision.proceed:
                    await self._record_audit(
                        AuditEvent(
                            event_type="dream.skipped",
                            principal_id=None,
                            tenant_id=tenant_id,
                            payload={
                                "reason": decision.reason,
                                "gate": _component_fqn(gate),
                                "trigger_name": trigger_name,
                            },
                            at=self.clock(),
                        ),
                        tenant_id,
                    )
                    return

            expired_lease_ids = await self.dream_lease_store.reclaim_expired(
                ctx=ReclaimLeasesContext(
                    request_id=_request_id("reclaim"), tenant_id=tenant_id
                )
            )
            if expired_lease_ids:
                await self.stm_store.release_for_expired_leases(
                    ctx=ReclaimContext(
                        request_id=_request_id("reclaim"),
                        tenant_id=tenant_id,
                        expired_lease_ids=expired_lease_ids,
                    )
                )

            lease = await self.dream_lease_store.acquire(
                ctx=AcquireLeaseContext(
                    request_id=_request_id("acquire"),
                    tenant_id=tenant_id,
                    ttl_seconds=self.default_lease_ttl_seconds,
                )
            )
            if lease is None:
                await self._record_audit(
                    AuditEvent(
                        event_type="dream.skipped",
                        principal_id=None,
                        tenant_id=tenant_id,
                        payload={"reason": "lease_held", "trigger_name": trigger_name},
                        at=self.clock(),
                    ),
                    tenant_id,
                )
                return

            self._active_leases[tenant_id] = lease.id
            heartbeat = asyncio.create_task(
                self._heartbeat(tenant_id=tenant_id, lease_id=lease.id)
            )
            wall_start = self.clock()
            try:
                await self._record_audit(
                    AuditEvent(
                        event_type="dream.lease_acquired",
                        principal_id=None,
                        tenant_id=tenant_id,
                        payload={"lease_id": lease.id, "trigger_name": trigger_name},
                        at=self.clock(),
                    ),
                    tenant_id,
                )
                await self._run_lifecycle_with_lease(
                    tenant_id=tenant_id,
                    lease_id=lease.id,
                    trigger_name=trigger_name,
                    tenant_config=tenant_config,
                )
            finally:
                heartbeat.cancel()
                with suppress(asyncio.CancelledError):
                    await heartbeat
                await self.dream_lease_store.release(
                    ctx=ReleaseLeaseContext(
                        request_id=_request_id("release"),
                        tenant_id=tenant_id,
                        lease_id=lease.id,
                    )
                )
                self._active_leases.pop(tenant_id, None)
                wall_seconds = (self.clock() - wall_start).total_seconds()
                await self._record_usage(
                    UsageEvent(
                        tenant_id=tenant_id,
                        component=f"{__name__}.Orchestrator",
                        kind="wall_seconds",
                        amount=wall_seconds,
                        unit="s",
                        at=self.clock(),
                        metadata={"trigger_name": trigger_name},
                    ),
                    tenant_id,
                )

    async def _run_lifecycle_with_lease(
        self,
        *,
        tenant_id: TenantId,
        lease_id: str,
        trigger_name: str,
        tenant_config: TenantConfig,
    ) -> None:
        # Fail-safe error handling: each phase gates on success of the
        # previous step's commit/state.
        state = self._tenant_states.setdefault(tenant_id, _TenantState())
        ltm_workspace: Workspace | None = None
        context_workspace: Workspace | None = None
        ltm_committed = False
        ltm_diff: Diff | None = None
        context_diff: Diff | None = None
        batch: MemoryBatch | None = None
        resumed = False
        # Resume-mode: batch must be released back to the pool, not consumed.
        held_batch_for_release = False

        unconsumed_count = await self._count_unconsumed(tenant_id)
        try:
            await self._run_pre_dream(
                tenant_id=tenant_id,
                lease_id=lease_id,
                trigger_name=trigger_name,
                unconsumed_count=unconsumed_count,
                tenant_config=tenant_config,
            )
        except Exception as exc:  # noqa: BLE001 — pre-dream raise → unwind
            await self._handle_failure(
                tenant_id=tenant_id,
                lease_id=lease_id,
                trigger_name=trigger_name,
                phase="pre_dream",
                error=exc,
                ltm_workspace=None,
                context_workspace=None,
                batch=None,
                tenant_config=tenant_config,
            )
            return

        batch = await self.stm_store.claim_batch(
            ctx=ClaimContext(
                request_id=_request_id("claim"),
                tenant_id=tenant_id,
                lease_id=lease_id,
            )
        )
        await self._record_audit(
            AuditEvent(
                event_type="dream.batch_claimed",
                principal_id=None,
                tenant_id=tenant_id,
                payload={
                    "lease_id": lease_id,
                    "batch_size": len(batch.memories),
                },
                at=self.clock(),
            ),
            tenant_id,
        )

        # Resume check must happen before the empty-batch fast path.
        watermark = await self._get_watermark(tenant_id)
        if watermark is not None:
            resumed = True
            held_batch_for_release = True

        # Transactional 2PC path: if both stores implement Transactional and
        # there is no watermark to drain, run the 2PC happy path.
        if (
            not resumed
            and isinstance(self.ltm_store, Transactional)
            and isinstance(self.context_store, Transactional)
        ):
            await self._run_transactional(
                tenant_id=tenant_id,
                lease_id=lease_id,
                trigger_name=trigger_name,
                batch=batch,
                tenant_config=tenant_config,
                state=state,
            )
            return

        # Empty-batch fast path (only when no watermark).
        if not resumed and not batch.memories:
            await self.stm_store.mark_consumed(
                ctx=MarkConsumedContext(
                    request_id=_request_id("mark"),
                    tenant_id=tenant_id,
                    lease_id=lease_id,
                    memory_ids=(),
                )
            )
            await self._run_post_dream(
                tenant_id=tenant_id,
                lease_id=lease_id,
                trigger_name=trigger_name,
                success=True,
                batch_size=0,
                ltm_diff=None,
                context_diff=None,
                resumed=False,
                error=None,
                tenant_config=tenant_config,
            )
            state.last_dream_at = self.clock()
            state.last_dream_success = True
            state.last_dream_error = None
            return

        try:
            if not resumed:
                ltm_workspace = await self.ltm_store.open_workspace(
                    ctx=OpenWorkspaceContext(
                        request_id=_request_id("ltm.open"),
                        tenant_id=tenant_id,
                    )
                )
                try:
                    await self._run_pre_ltm_update(
                        tenant_id=tenant_id,
                        lease_id=lease_id,
                        ltm_workspace_id=ltm_workspace.id,
                        batch_size=len(batch.memories),
                        tenant_config=tenant_config,
                    )
                    await self._run_ltm_phase(
                        tenant_id=tenant_id,
                        lease_id=lease_id,
                        batch=batch,
                        ltm_workspace=ltm_workspace,
                        tenant_config=tenant_config,
                    )
                except Exception as exc:  # noqa: BLE001 — LTM-side failure
                    await self._handle_failure(
                        tenant_id=tenant_id,
                        lease_id=lease_id,
                        trigger_name=trigger_name,
                        phase="ltm",
                        error=exc,
                        ltm_workspace=ltm_workspace,
                        context_workspace=None,
                        batch=batch,
                        tenant_config=tenant_config,
                    )
                    state.last_dream_at = self.clock()
                    state.last_dream_success = False
                    state.last_dream_error = str(exc)
                    return

                try:
                    ltm_diff = await self.ltm_store.commit_workspace(
                        ltm_workspace,
                        ctx=CommitWorkspaceContext(
                            request_id=_request_id("ltm.commit"),
                            tenant_id=tenant_id,
                        ),
                    )
                    ltm_committed = True
                except Exception as exc:  # noqa: BLE001 — commit failed
                    await self._handle_failure(
                        tenant_id=tenant_id,
                        lease_id=lease_id,
                        trigger_name=trigger_name,
                        phase="ltm",
                        error=exc,
                        ltm_workspace=ltm_workspace,
                        context_workspace=None,
                        batch=batch,
                        tenant_config=tenant_config,
                    )
                    state.last_dream_at = self.clock()
                    state.last_dream_success = False
                    state.last_dream_error = str(exc)
                    return
                # Workspace is owned by the store after commit.
                ltm_workspace = None
                await self._record_audit(
                    AuditEvent(
                        event_type="dream.ltm_committed",
                        principal_id=None,
                        tenant_id=tenant_id,
                        payload={
                            "lease_id": lease_id,
                            "added": len(ltm_diff.added),
                            "modified": len(ltm_diff.modified),
                            "deleted": len(ltm_diff.deleted),
                        },
                        at=self.clock(),
                    ),
                    tenant_id,
                )
                # Persist watermark before running context phase.
                if isinstance(self.ltm_store, ContextPendingStore):
                    await self.ltm_store.set_context_pending(
                        ltm_diff,
                        ctx=SetContextPendingContext(
                            request_id=_request_id("watermark.set"),
                            tenant_id=tenant_id,
                        ),
                    )

                # post_ltm_update hooks (advisory).
                await self._run_post_ltm_update(
                    tenant_id=tenant_id,
                    lease_id=lease_id,
                    ltm_workspace_id="<committed>",
                    ltm_diff=ltm_diff,
                    tenant_config=tenant_config,
                )
            else:
                ltm_diff = watermark

            assert ltm_diff is not None  # mypy: tracked above

            try:
                context_workspace = await self.context_store.open_workspace(
                    ctx=OpenWorkspaceContext(
                        request_id=_request_id("ctx.open"),
                        tenant_id=tenant_id,
                    )
                )
                # ContextPhaseServices needs an ltm_workspace; open in read-only
                # mode (post-commit state).
                ltm_ro_workspace = await self.ltm_store.open_workspace(
                    ctx=OpenWorkspaceContext(
                        request_id=_request_id("ltm.read"),
                        tenant_id=tenant_id,
                    )
                )
                try:
                    await self._run_pre_context_update(
                        tenant_id=tenant_id,
                        lease_id=lease_id,
                        ltm_workspace_id=ltm_ro_workspace.id,
                        ltm_diff=ltm_diff,
                        context_workspace_id=context_workspace.id,
                        tenant_config=tenant_config,
                    )
                    await self._run_context_phase(
                        tenant_id=tenant_id,
                        lease_id=lease_id,
                        ltm_workspace=ltm_ro_workspace,
                        ltm_diff=ltm_diff,
                        context_workspace=context_workspace,
                        tenant_config=tenant_config,
                    )
                    context_diff = await self.context_store.commit_workspace(
                        context_workspace,
                        ctx=CommitWorkspaceContext(
                            request_id=_request_id("ctx.commit"),
                            tenant_id=tenant_id,
                        ),
                    )
                    context_workspace = None
                    await self._record_audit(
                        AuditEvent(
                            event_type="dream.context_committed",
                            principal_id=None,
                            tenant_id=tenant_id,
                            payload={
                                "lease_id": lease_id,
                                "added": len(context_diff.added),
                                "modified": len(context_diff.modified),
                                "deleted": len(context_diff.deleted),
                            },
                            at=self.clock(),
                        ),
                        tenant_id,
                    )
                    # Clear the watermark on success.
                    if isinstance(self.ltm_store, ContextPendingStore):
                        await self.ltm_store.clear_context_pending(
                            ctx=ClearContextPendingContext(
                                request_id=_request_id("watermark.clear"),
                                tenant_id=tenant_id,
                            )
                        )
                    await self._run_post_context_update(
                        tenant_id=tenant_id,
                        lease_id=lease_id,
                        context_workspace_id="<committed>",
                        context_diff=context_diff,
                        tenant_config=tenant_config,
                    )
                finally:
                    # Always close the read-only LTM workspace; we don't commit it.
                    with suppress(Exception):
                        await self.ltm_store.discard_workspace(
                            ltm_ro_workspace,
                            ctx=DiscardWorkspaceContext(
                                request_id=_request_id("ltm.read.close"),
                                tenant_id=tenant_id,
                            ),
                        )
            except Exception as exc:  # noqa: BLE001 — context-side failure
                await self._handle_failure(
                    tenant_id=tenant_id,
                    lease_id=lease_id,
                    trigger_name=trigger_name,
                    phase="context",
                    error=exc,
                    ltm_workspace=None,  # already committed if we got here
                    context_workspace=context_workspace,
                    batch=batch,
                    tenant_config=tenant_config,
                    ltm_already_committed=ltm_committed or resumed,
                    ltm_diff=ltm_diff,
                )
                state.last_dream_at = self.clock()
                state.last_dream_success = False
                state.last_dream_error = str(exc)
                return

            # Mark batch consumed, or release back to the pool in resume-mode.
            if held_batch_for_release:
                await self.stm_store.release_unconsumed(
                    ctx=ReleaseContext(
                        request_id=_request_id("release.resume"),
                        tenant_id=tenant_id,
                        lease_id=lease_id,
                    )
                )
            else:
                memory_ids = tuple(m.id for m in batch.memories if m.id is not None)
                await self.stm_store.mark_consumed(
                    ctx=MarkConsumedContext(
                        request_id=_request_id("mark"),
                        tenant_id=tenant_id,
                        lease_id=lease_id,
                        memory_ids=memory_ids,
                    )
                )

            await self._run_post_dream(
                tenant_id=tenant_id,
                lease_id=lease_id,
                trigger_name=trigger_name,
                success=True,
                batch_size=len(batch.memories),
                ltm_diff=ltm_diff,
                context_diff=context_diff,
                resumed=resumed,
                error=None,
                tenant_config=tenant_config,
            )
            state.last_dream_at = self.clock()
            state.last_dream_success = True
            state.last_dream_error = None
        finally:
            if ltm_workspace is not None:
                with suppress(Exception):
                    await self.ltm_store.discard_workspace(
                        ltm_workspace,
                        ctx=DiscardWorkspaceContext(
                            request_id=_request_id("ltm.discard"),
                            tenant_id=tenant_id,
                        ),
                    )
            if context_workspace is not None:
                with suppress(Exception):
                    await self.context_store.discard_workspace(
                        context_workspace,
                        ctx=DiscardWorkspaceContext(
                            request_id=_request_id("ctx.discard"),
                            tenant_id=tenant_id,
                        ),
                    )

    async def _run_context_only(
        self,
        *,
        tenant_id: TenantId,
        lease_id: str,
        watermark: Diff,
        tenant_config: TenantConfig,
        trigger_name: str,
    ) -> None:
        """Run only the context phase against a watermark — used by ``dreamer
        dream --resume-context``. Skips trigger logic, gates, and STM claim.
        """
        context_workspace = await self.context_store.open_workspace(
            ctx=OpenWorkspaceContext(
                request_id=_request_id("ctx.open"),
                tenant_id=tenant_id,
            )
        )
        ltm_ro_workspace = await self.ltm_store.open_workspace(
            ctx=OpenWorkspaceContext(
                request_id=_request_id("ltm.read"),
                tenant_id=tenant_id,
            )
        )
        try:
            await self._run_pre_context_update(
                tenant_id=tenant_id,
                lease_id=lease_id,
                ltm_workspace_id=ltm_ro_workspace.id,
                ltm_diff=watermark,
                context_workspace_id=context_workspace.id,
                tenant_config=tenant_config,
            )
            await self._run_context_phase(
                tenant_id=tenant_id,
                lease_id=lease_id,
                ltm_workspace=ltm_ro_workspace,
                ltm_diff=watermark,
                context_workspace=context_workspace,
                tenant_config=tenant_config,
            )
            context_diff = await self.context_store.commit_workspace(
                context_workspace,
                ctx=CommitWorkspaceContext(
                    request_id=_request_id("ctx.commit"),
                    tenant_id=tenant_id,
                ),
            )
            context_workspace = None  # type: ignore[assignment]
            if isinstance(self.ltm_store, ContextPendingStore):
                await self.ltm_store.clear_context_pending(
                    ctx=ClearContextPendingContext(
                        request_id=_request_id("watermark.clear"),
                        tenant_id=tenant_id,
                    )
                )
            await self._run_post_context_update(
                tenant_id=tenant_id,
                lease_id=lease_id,
                context_workspace_id="<committed>",
                context_diff=context_diff,
                tenant_config=tenant_config,
            )
            await self._run_post_dream(
                tenant_id=tenant_id,
                lease_id=lease_id,
                trigger_name=trigger_name,
                success=True,
                batch_size=0,
                ltm_diff=watermark,
                context_diff=context_diff,
                resumed=True,
                error=None,
                tenant_config=tenant_config,
            )
        finally:
            with suppress(Exception):
                await self.ltm_store.discard_workspace(
                    ltm_ro_workspace,
                    ctx=DiscardWorkspaceContext(
                        request_id=_request_id("ltm.read.close"),
                        tenant_id=tenant_id,
                    ),
                )
            if context_workspace is not None:
                with suppress(Exception):
                    await self.context_store.discard_workspace(
                        context_workspace,
                        ctx=DiscardWorkspaceContext(
                            request_id=_request_id("ctx.discard"),
                            tenant_id=tenant_id,
                        ),
                    )

    async def _run_transactional(
        self,
        *,
        tenant_id: TenantId,
        lease_id: str,
        trigger_name: str,
        batch: MemoryBatch,
        tenant_config: TenantConfig,
        state: _TenantState,
    ) -> None:
        """Two-phase commit happy path. begin → run LTM → run Context → prepare both → commit both.

        Rollback on any failure.
        """
        ltm_tx = await self.ltm_store.begin(  # type: ignore[attr-defined]
            ctx=TxBeginContext(request_id=_request_id("tx.begin.ltm"), tenant_id=tenant_id)
        )
        ctx_tx = await self.context_store.begin(  # type: ignore[attr-defined]
            ctx=TxBeginContext(request_id=_request_id("tx.begin.ctx"), tenant_id=tenant_id)
        )

        ltm_workspace: Workspace | None = None
        context_workspace: Workspace | None = None
        ltm_diff: Diff | None = None
        context_diff: Diff | None = None
        try:
            ltm_workspace = await self.ltm_store.open_workspace(
                ctx=OpenWorkspaceContext(
                    request_id=_request_id("ltm.open"),
                    tenant_id=tenant_id,
                )
            )
            await self._run_pre_ltm_update(
                tenant_id=tenant_id,
                lease_id=lease_id,
                ltm_workspace_id=ltm_workspace.id,
                batch_size=len(batch.memories),
                tenant_config=tenant_config,
            )
            await self._run_ltm_phase(
                tenant_id=tenant_id,
                lease_id=lease_id,
                batch=batch,
                ltm_workspace=ltm_workspace,
                tenant_config=tenant_config,
            )
            ltm_diff = await self.ltm_store.commit_workspace(
                ltm_workspace,
                ctx=CommitWorkspaceContext(
                    request_id=_request_id("ltm.commit"), tenant_id=tenant_id
                ),
            )
            ltm_workspace = None
            await self._run_post_ltm_update(
                tenant_id=tenant_id,
                lease_id=lease_id,
                ltm_workspace_id="<committed>",
                ltm_diff=ltm_diff,
                tenant_config=tenant_config,
            )

            context_workspace = await self.context_store.open_workspace(
                ctx=OpenWorkspaceContext(
                    request_id=_request_id("ctx.open"),
                    tenant_id=tenant_id,
                )
            )
            ltm_ro_workspace = await self.ltm_store.open_workspace(
                ctx=OpenWorkspaceContext(
                    request_id=_request_id("ltm.read"),
                    tenant_id=tenant_id,
                )
            )
            try:
                await self._run_pre_context_update(
                    tenant_id=tenant_id,
                    lease_id=lease_id,
                    ltm_workspace_id=ltm_ro_workspace.id,
                    ltm_diff=ltm_diff,
                    context_workspace_id=context_workspace.id,
                    tenant_config=tenant_config,
                )
                await self._run_context_phase(
                    tenant_id=tenant_id,
                    lease_id=lease_id,
                    ltm_workspace=ltm_ro_workspace,
                    ltm_diff=ltm_diff,
                    context_workspace=context_workspace,
                    tenant_config=tenant_config,
                )
                context_diff = await self.context_store.commit_workspace(
                    context_workspace,
                    ctx=CommitWorkspaceContext(
                        request_id=_request_id("ctx.commit"),
                        tenant_id=tenant_id,
                    ),
                )
                context_workspace = None
            finally:
                with suppress(Exception):
                    await self.ltm_store.discard_workspace(
                        ltm_ro_workspace,
                        ctx=DiscardWorkspaceContext(
                            request_id=_request_id("ltm.read.close"),
                            tenant_id=tenant_id,
                        ),
                    )

            ltm_ready = await self.ltm_store.prepare(  # type: ignore[attr-defined]
                ltm_tx,
                ctx=TxPrepareContext(
                    request_id=_request_id("tx.prep.ltm"), tenant_id=tenant_id
                ),
            )
            ctx_ready = await self.context_store.prepare(  # type: ignore[attr-defined]
                ctx_tx,
                ctx=TxPrepareContext(
                    request_id=_request_id("tx.prep.ctx"), tenant_id=tenant_id
                ),
            )
            if not ltm_ready or not ctx_ready:
                raise DreamFailedError(
                    f"transactional prepare returned False (ltm={ltm_ready}, ctx={ctx_ready})"
                )
            await self.ltm_store.commit(  # type: ignore[attr-defined]
                ltm_tx,
                ctx=TxCommitContext(
                    request_id=_request_id("tx.commit.ltm"), tenant_id=tenant_id
                ),
            )
            await self.context_store.commit(  # type: ignore[attr-defined]
                ctx_tx,
                ctx=TxCommitContext(
                    request_id=_request_id("tx.commit.ctx"), tenant_id=tenant_id
                ),
            )

            memory_ids = tuple(m.id for m in batch.memories if m.id is not None)
            await self.stm_store.mark_consumed(
                ctx=MarkConsumedContext(
                    request_id=_request_id("mark"),
                    tenant_id=tenant_id,
                    lease_id=lease_id,
                    memory_ids=memory_ids,
                )
            )
            await self._run_post_context_update(
                tenant_id=tenant_id,
                lease_id=lease_id,
                context_workspace_id="<committed>",
                context_diff=context_diff,
                tenant_config=tenant_config,
            )
            await self._run_post_dream(
                tenant_id=tenant_id,
                lease_id=lease_id,
                trigger_name=trigger_name,
                success=True,
                batch_size=len(batch.memories),
                ltm_diff=ltm_diff,
                context_diff=context_diff,
                resumed=False,
                error=None,
                tenant_config=tenant_config,
            )
            state.last_dream_at = self.clock()
            state.last_dream_success = True
            state.last_dream_error = None
        except Exception as exc:  # noqa: BLE001 — rollback everything
            with suppress(Exception):
                await self.ltm_store.rollback(  # type: ignore[attr-defined]
                    ltm_tx,
                    ctx=TxRollbackContext(
                        request_id=_request_id("tx.rollback.ltm"),
                        tenant_id=tenant_id,
                    ),
                )
            with suppress(Exception):
                await self.context_store.rollback(  # type: ignore[attr-defined]
                    ctx_tx,
                    ctx=TxRollbackContext(
                        request_id=_request_id("tx.rollback.ctx"),
                        tenant_id=tenant_id,
                    ),
                )
            await self.stm_store.release_unconsumed(
                ctx=ReleaseContext(
                    request_id=_request_id("release.tx"),
                    tenant_id=tenant_id,
                    lease_id=lease_id,
                )
            )
            await self._run_dream_failed(
                tenant_id=tenant_id,
                lease_id=lease_id,
                trigger_name=trigger_name,
                phase="context",
                error=exc,
                tenant_config=tenant_config,
            )
            await self._run_post_dream(
                tenant_id=tenant_id,
                lease_id=lease_id,
                trigger_name=trigger_name,
                success=False,
                batch_size=len(batch.memories),
                ltm_diff=None,
                context_diff=None,
                resumed=False,
                error=str(exc),
                tenant_config=tenant_config,
            )
            state.last_dream_at = self.clock()
            state.last_dream_success = False
            state.last_dream_error = str(exc)
        finally:
            if ltm_workspace is not None:
                with suppress(Exception):
                    await self.ltm_store.discard_workspace(
                        ltm_workspace,
                        ctx=DiscardWorkspaceContext(
                            request_id=_request_id("ltm.discard"),
                            tenant_id=tenant_id,
                        ),
                    )
            if context_workspace is not None:
                with suppress(Exception):
                    await self.context_store.discard_workspace(
                        context_workspace,
                        ctx=DiscardWorkspaceContext(
                            request_id=_request_id("ctx.discard"),
                            tenant_id=tenant_id,
                        ),
                    )

    async def _run_ltm_phase(
        self,
        *,
        tenant_id: TenantId,
        lease_id: str,
        batch: MemoryBatch,
        ltm_workspace: Workspace,
        tenant_config: TenantConfig,
    ) -> None:
        instructions = _instr(tenant_config, "ltm_update")
        ltm_ctx = LTMPhaseContext(
            request_id=_request_id("ltm.phase"),
            tenant_id=tenant_id,
            lease_id=lease_id,
            batch=batch,
            ltm_workspace_id=ltm_workspace.id,
            instructions=instructions,
        )
        ltm_services = LTMPhaseServices(
            emit_progress=self._make_emit_progress(
                tenant_id=tenant_id, lease_id=lease_id, phase="ltm", tenant_config=tenant_config
            ),
            secrets=self._require_secrets(),
            usage=_FanoutUsageSink(self.usage_sinks),
            audit=_FanoutAuditSink(self.audit_sinks),
            clock=self.clock,
            ltm_workspace=ltm_workspace,
        )
        await self.ltm_phase_runner.run_ltm_phase(ctx=ltm_ctx, services=ltm_services)

    async def _run_context_phase(
        self,
        *,
        tenant_id: TenantId,
        lease_id: str,
        ltm_workspace: Workspace,
        ltm_diff: Diff,
        context_workspace: Workspace,
        tenant_config: TenantConfig,
    ) -> None:
        instructions = _instr(tenant_config, "context_update")
        ctx_ctx = ContextPhaseContext(
            request_id=_request_id("ctx.phase"),
            tenant_id=tenant_id,
            lease_id=lease_id,
            ltm_workspace_id=ltm_workspace.id,
            ltm_diff=ltm_diff,
            context_workspace_id=context_workspace.id,
            instructions=instructions,
        )
        ctx_services = ContextPhaseServices(
            emit_progress=self._make_emit_progress(
                tenant_id=tenant_id,
                lease_id=lease_id,
                phase="context",
                tenant_config=tenant_config,
            ),
            secrets=self._require_secrets(),
            usage=_FanoutUsageSink(self.usage_sinks),
            audit=_FanoutAuditSink(self.audit_sinks),
            clock=self.clock,
            ltm_workspace=ltm_workspace,
            context_workspace=context_workspace,
        )
        await self.context_phase_runner.run_context_phase(
            ctx=ctx_ctx, services=ctx_services
        )

    async def _run_pre_dream(
        self,
        *,
        tenant_id: TenantId,
        lease_id: str,
        trigger_name: str,
        unconsumed_count: int,
        tenant_config: TenantConfig,
    ) -> None:
        for hook in self.hook_registry.get("pre_dream"):
            ctx = PreDreamContext(
                request_id=_request_id("pre_dream"),
                tenant_id=tenant_id,
                lease_id=lease_id,
                trigger_name=trigger_name,
                unconsumed_count=unconsumed_count,
                params=self._params_for(hook, tenant_config),
            )
            services = PreDreamServices(
                emit_progress=self._make_emit_progress(
                    tenant_id=tenant_id,
                    lease_id=lease_id,
                    phase="pre_dream",
                    tenant_config=tenant_config,
                ),
                secrets=self._require_secrets(),
                usage=_FanoutUsageSink(self.usage_sinks),
                audit=_FanoutAuditSink(self.audit_sinks),
                clock=self.clock,
            )
            assert isinstance(hook, PreDreamHook)
            await hook.on_pre_dream(ctx=ctx, services=services)

    async def _run_pre_ltm_update(
        self,
        *,
        tenant_id: TenantId,
        lease_id: str,
        ltm_workspace_id: str,
        batch_size: int,
        tenant_config: TenantConfig,
    ) -> None:
        for hook in self.hook_registry.get("pre_ltm_update"):
            ctx = PreLTMUpdateContext(
                request_id=_request_id("pre_ltm"),
                tenant_id=tenant_id,
                lease_id=lease_id,
                ltm_workspace_id=ltm_workspace_id,
                batch_size=batch_size,
                params=self._params_for(hook, tenant_config),
            )
            services = PreLTMUpdateServices(
                emit_progress=self._make_emit_progress(
                    tenant_id=tenant_id,
                    lease_id=lease_id,
                    phase="pre_ltm_update",
                    tenant_config=tenant_config,
                ),
                secrets=self._require_secrets(),
                usage=_FanoutUsageSink(self.usage_sinks),
                audit=_FanoutAuditSink(self.audit_sinks),
                clock=self.clock,
            )
            assert isinstance(hook, PreLTMUpdateHook)
            await hook.on_pre_ltm_update(ctx=ctx, services=services)

    async def _run_post_ltm_update(
        self,
        *,
        tenant_id: TenantId,
        lease_id: str,
        ltm_workspace_id: str,
        ltm_diff: Diff,
        tenant_config: TenantConfig,
    ) -> None:
        failures: list[tuple[Any, BaseException]] = []
        for hook in self.hook_registry.get("post_ltm_update"):
            ctx = PostLTMUpdateContext(
                request_id=_request_id("post_ltm"),
                tenant_id=tenant_id,
                lease_id=lease_id,
                ltm_workspace_id=ltm_workspace_id,
                ltm_diff=ltm_diff,
                params=self._params_for(hook, tenant_config),
            )
            services = PostLTMUpdateServices(
                emit_progress=self._make_emit_progress(
                    tenant_id=tenant_id,
                    lease_id=lease_id,
                    phase="post_ltm_update",
                    tenant_config=tenant_config,
                ),
                secrets=self._require_secrets(),
                usage=_FanoutUsageSink(self.usage_sinks),
                audit=_FanoutAuditSink(self.audit_sinks),
                clock=self.clock,
            )
            assert isinstance(hook, PostLTMUpdateHook)
            try:
                await hook.on_post_ltm_update(ctx=ctx, services=services)
            except Exception as exc:  # noqa: BLE001 — advisory hook
                logger.exception(
                    "post_ltm_update hook %r raised; continuing", hook
                )
                failures.append((hook, exc))
                await self._record_audit(
                    AuditEvent(
                        event_type="hook.failed",
                        principal_id=None,
                        tenant_id=tenant_id,
                        payload={
                            "slot": "post_ltm_update",
                            "hook": _component_fqn(hook),
                            "error": str(exc),
                        },
                        at=self.clock(),
                    ),
                    tenant_id,
                )

    async def _run_pre_context_update(
        self,
        *,
        tenant_id: TenantId,
        lease_id: str,
        ltm_workspace_id: str,
        ltm_diff: Diff,
        context_workspace_id: str,
        tenant_config: TenantConfig,
    ) -> None:
        for hook in self.hook_registry.get("pre_context_update"):
            ctx = PreContextUpdateContext(
                request_id=_request_id("pre_ctx"),
                tenant_id=tenant_id,
                lease_id=lease_id,
                ltm_workspace_id=ltm_workspace_id,
                ltm_diff=ltm_diff,
                context_workspace_id=context_workspace_id,
                params=self._params_for(hook, tenant_config),
            )
            services = PreContextUpdateServices(
                emit_progress=self._make_emit_progress(
                    tenant_id=tenant_id,
                    lease_id=lease_id,
                    phase="pre_context_update",
                    tenant_config=tenant_config,
                ),
                secrets=self._require_secrets(),
                usage=_FanoutUsageSink(self.usage_sinks),
                audit=_FanoutAuditSink(self.audit_sinks),
                clock=self.clock,
            )
            assert isinstance(hook, PreContextUpdateHook)
            await hook.on_pre_context_update(ctx=ctx, services=services)

    async def _run_post_context_update(
        self,
        *,
        tenant_id: TenantId,
        lease_id: str,
        context_workspace_id: str,
        context_diff: Diff,
        tenant_config: TenantConfig,
    ) -> None:
        for hook in self.hook_registry.get("post_context_update"):
            ctx = PostContextUpdateContext(
                request_id=_request_id("post_ctx"),
                tenant_id=tenant_id,
                lease_id=lease_id,
                context_workspace_id=context_workspace_id,
                context_diff=context_diff,
                params=self._params_for(hook, tenant_config),
            )
            services = PostContextUpdateServices(
                emit_progress=self._make_emit_progress(
                    tenant_id=tenant_id,
                    lease_id=lease_id,
                    phase="post_context_update",
                    tenant_config=tenant_config,
                ),
                secrets=self._require_secrets(),
                usage=_FanoutUsageSink(self.usage_sinks),
                audit=_FanoutAuditSink(self.audit_sinks),
                clock=self.clock,
            )
            assert isinstance(hook, PostContextUpdateHook)
            try:
                await hook.on_post_context_update(ctx=ctx, services=services)
            except Exception as exc:  # noqa: BLE001 — advisory hook
                logger.exception(
                    "post_context_update hook %r raised; continuing", hook
                )
                await self._record_audit(
                    AuditEvent(
                        event_type="hook.failed",
                        principal_id=None,
                        tenant_id=tenant_id,
                        payload={
                            "slot": "post_context_update",
                            "hook": _component_fqn(hook),
                            "error": str(exc),
                        },
                        at=self.clock(),
                    ),
                    tenant_id,
                )

    async def _run_post_dream(
        self,
        *,
        tenant_id: TenantId,
        lease_id: str,
        trigger_name: str,
        success: bool,
        batch_size: int,
        ltm_diff: Diff | None,
        context_diff: Diff | None,
        resumed: bool,
        error: str | None,
        tenant_config: TenantConfig,
    ) -> None:
        for hook in self.hook_registry.get("post_dream"):
            ctx = PostDreamContext(
                request_id=_request_id("post_dream"),
                tenant_id=tenant_id,
                lease_id=lease_id,
                trigger_name=trigger_name,
                success=success,
                batch_size=batch_size,
                ltm_diff=ltm_diff,
                context_diff=context_diff,
                resumed=resumed,
                error=error,
                params=self._params_for(hook, tenant_config),
            )
            services = PostDreamServices(
                emit_progress=self._make_emit_progress(
                    tenant_id=tenant_id,
                    lease_id=lease_id,
                    phase="post_dream",
                    tenant_config=tenant_config,
                ),
                secrets=self._require_secrets(),
                usage=_FanoutUsageSink(self.usage_sinks),
                audit=_FanoutAuditSink(self.audit_sinks),
                clock=self.clock,
            )
            assert isinstance(hook, PostDreamHook)
            try:
                await hook.on_post_dream(ctx=ctx, services=services)
            except Exception as exc:  # noqa: BLE001 — advisory hook
                logger.exception("post_dream hook %r raised; continuing", hook)
                await self._record_audit(
                    AuditEvent(
                        event_type="hook.failed",
                        principal_id=None,
                        tenant_id=tenant_id,
                        payload={
                            "slot": "post_dream",
                            "hook": _component_fqn(hook),
                            "error": str(exc),
                        },
                        at=self.clock(),
                    ),
                    tenant_id,
                )

    async def _run_dream_failed(
        self,
        *,
        tenant_id: TenantId,
        lease_id: str | None,
        trigger_name: str,
        phase: str,
        error: BaseException,
        tenant_config: TenantConfig,
    ) -> None:
        for hook in self.hook_registry.get("on_dream_failed"):
            ctx = DreamFailedContext(
                request_id=_request_id("dream_failed"),
                tenant_id=tenant_id,
                lease_id=lease_id,
                trigger_name=trigger_name,
                phase=phase,  # type: ignore[arg-type]
                error=str(error),
                params=self._params_for(hook, tenant_config),
            )
            services = DreamFailedServices(
                emit_progress=self._make_emit_progress(
                    tenant_id=tenant_id,
                    lease_id=lease_id or "",
                    phase="dream_failed",
                    tenant_config=tenant_config,
                ),
                secrets=self._require_secrets(),
                usage=_FanoutUsageSink(self.usage_sinks),
                audit=_FanoutAuditSink(self.audit_sinks),
                clock=self.clock,
            )
            assert isinstance(hook, DreamFailedHook)
            try:
                await hook.on_dream_failed(ctx=ctx, services=services)
            except Exception:  # noqa: BLE001 — never abort cleanup
                logger.exception("on_dream_failed hook %r raised; continuing", hook)

    async def _handle_failure(
        self,
        *,
        tenant_id: TenantId,
        lease_id: str,
        trigger_name: str,
        phase: str,
        error: BaseException,
        ltm_workspace: Workspace | None,
        context_workspace: Workspace | None,
        batch: MemoryBatch | None,
        tenant_config: TenantConfig,
        ltm_already_committed: bool = False,
        ltm_diff: Diff | None = None,
    ) -> None:
        """Dispatch failure cleanup based on where the failure occurred."""
        if ltm_workspace is not None:
            with suppress(Exception):
                await self.ltm_store.discard_workspace(
                    ltm_workspace,
                    ctx=DiscardWorkspaceContext(
                        request_id=_request_id("ltm.discard"),
                        tenant_id=tenant_id,
                    ),
                )
        if context_workspace is not None:
            with suppress(Exception):
                await self.context_store.discard_workspace(
                    context_workspace,
                    ctx=DiscardWorkspaceContext(
                        request_id=_request_id("ctx.discard"),
                        tenant_id=tenant_id,
                    ),
                )

        if batch is not None:
            await self.stm_store.release_unconsumed(
                ctx=ReleaseContext(
                    request_id=_request_id("release.fail"),
                    tenant_id=tenant_id,
                    lease_id=lease_id,
                )
            )

        await self._run_dream_failed(
            tenant_id=tenant_id,
            lease_id=lease_id,
            trigger_name=trigger_name,
            phase=phase,
            error=error,
            tenant_config=tenant_config,
        )
        # Pre-LTM-commit failures: ltm_diff is None. Post-LTM-commit failures:
        # ltm_diff present, watermark already set by the orchestrator before
        # the context phase started.
        await self._run_post_dream(
            tenant_id=tenant_id,
            lease_id=lease_id,
            trigger_name=trigger_name,
            success=False,
            batch_size=len(batch.memories) if batch is not None else 0,
            ltm_diff=ltm_diff if ltm_already_committed else None,
            context_diff=None,
            resumed=False,
            error=str(error),
            tenant_config=tenant_config,
        )

    async def _count_unconsumed(self, tenant_id: TenantId) -> int:
        return await self.stm_store.count_unconsumed(
            ctx=CountContext(
                request_id=_request_id("count"), tenant_id=tenant_id
            )
        )

    async def _get_watermark(self, tenant_id: TenantId) -> Diff | None:
        if not isinstance(self.ltm_store, ContextPendingStore):
            return None
        return await self.ltm_store.get_context_pending(
            ctx=GetContextPendingContext(
                request_id=_request_id("watermark.get"),
                tenant_id=tenant_id,
            )
        )

    def _params_for(
        self, component: object, tenant_config: TenantConfig
    ) -> Mapping[str, Any]:
        """Merge construction-time params with tenant overrides.

        Base params come from the component's ``__init__`` signature: every
        parameter whose name appears as a public instance attribute is
        captured at call time. Tenant overrides come from
        ``TenantConfig.hook_params[fqn]``.
        """
        import inspect

        fqn = _component_fqn(component)
        base: dict[str, Any] = {}
        try:
            sig = inspect.signature(component.__class__.__init__)
        except (TypeError, ValueError):
            sig = None
        if sig is not None:
            for name in sig.parameters:
                if name == "self" or name.startswith("*"):
                    continue
                if hasattr(component, name):
                    try:
                        val = getattr(component, name)
                    except Exception:  # noqa: BLE001
                        continue
                    if not callable(val):
                        base[name] = val
        if tenant_config.hook_params:
            override = tenant_config.hook_params.get(fqn)
            if override:
                base.update(override)
        return base

    def _require_secrets(self) -> SecretResolver:
        if self.secret_resolver is None:
            raise DreamFailedError("secret_resolver is not configured on the orchestrator")
        return self.secret_resolver

    def _make_emit_progress(
        self,
        *,
        tenant_id: TenantId,
        lease_id: str,
        phase: str,
        tenant_config: TenantConfig,
    ) -> Callable[[str, Mapping[str, Any]], Awaitable[None]]:
        async def _emit(message: str, payload: Mapping[str, Any]) -> None:
            for hook in self.hook_registry.get("on_dream_progress"):
                ctx = DreamProgressContext(
                    request_id=_request_id("progress"),
                    tenant_id=tenant_id,
                    lease_id=lease_id,
                    phase=phase,
                    message=message,
                    payload=payload,
                    params=self._params_for(hook, tenant_config),
                )
                services = DreamProgressServices(
                    emit_progress=_emit,
                    secrets=self._require_secrets(),
                    usage=_FanoutUsageSink(self.usage_sinks),
                    audit=_FanoutAuditSink(self.audit_sinks),
                    clock=self.clock,
                )

                async def _dispatch(
                    h: DreamProgressHook = hook,
                    c: DreamProgressContext = ctx,
                    s: DreamProgressServices = services,
                ) -> None:
                    try:
                        await h.on_dream_progress(ctx=c, services=s)
                    except Exception:  # noqa: BLE001 — fire-and-forget
                        logger.exception("on_dream_progress hook raised; ignoring")

                asyncio.create_task(_dispatch())

        return _emit

    def _build_dream_gate_services(self) -> DreamGateServices:
        return DreamGateServices(
            emit_progress=_noop_emit_progress,
            secrets=self._require_secrets(),
            usage=_FanoutUsageSink(self.usage_sinks),
            audit=_FanoutAuditSink(self.audit_sinks),
            clock=self.clock,
        )

    async def _record_audit(self, event: AuditEvent, tenant_id: TenantId) -> None:
        if not self.audit_sinks:
            return
        ctx = AuditContext(request_id=_request_id("audit"), tenant_id=tenant_id)
        await asyncio.gather(
            *(self._safe_audit(sink, event, ctx) for sink in self.audit_sinks),
            return_exceptions=False,
        )

    @staticmethod
    async def _safe_audit(sink: AuditSink, event: AuditEvent, ctx: AuditContext) -> None:
        try:
            await sink.record(event, ctx=ctx)
        except Exception:  # noqa: BLE001 — sinks are fire-and-forget
            logger.exception("AuditSink %r raised; continuing", sink)

    async def _record_usage(self, event: UsageEvent, tenant_id: TenantId) -> None:
        if not self.usage_sinks:
            return
        ctx = UsageContext(request_id=_request_id("usage"), tenant_id=tenant_id)
        await asyncio.gather(
            *(self._safe_usage(sink, event, ctx) for sink in self.usage_sinks),
            return_exceptions=False,
        )

    @staticmethod
    async def _safe_usage(sink: UsageSink, event: UsageEvent, ctx: UsageContext) -> None:
        try:
            await sink.record(event, ctx=ctx)
        except Exception:  # noqa: BLE001 — sinks are fire-and-forget
            logger.exception("UsageSink %r raised; continuing", sink)

    async def _heartbeat(self, *, tenant_id: TenantId, lease_id: str) -> None:
        try:
            while True:
                await asyncio.sleep(self.heartbeat_interval_seconds)
                with TenantScope.set(tenant_id):
                    ok = await self.dream_lease_store.renew(
                        ctx=RenewLeaseContext(
                            request_id=_request_id("renew"),
                            tenant_id=tenant_id,
                            lease_id=lease_id,
                            ttl_seconds=self.default_lease_ttl_seconds,
                        )
                    )
                if not ok:
                    logger.warning(
                        "lease %s for tenant %s could not be renewed",
                        lease_id,
                        tenant_id,
                    )
                    return
        except asyncio.CancelledError:
            return

    async def _retention_loop(self) -> None:
        try:
            while not self._shutting_down:
                await asyncio.sleep(self.stm_retention.cadence_seconds)
                if self._shutting_down:
                    return
                await self._run_retention_sweep()
        except asyncio.CancelledError:
            return

    async def _run_retention_sweep(self) -> None:
        if self.stm_retention.keep_days is None:
            return
        before = self.clock() - timedelta(days=self.stm_retention.keep_days)
        try:
            tenants = await self.tenant_registry.list_tenants(
                ctx=TenantRegistryContext(request_id=_request_id("retention.list"))
            )
        except Exception:  # noqa: BLE001
            logger.exception("retention sweep: list_tenants failed; skipping")
            return
        for tenant_id in tenants:
            try:
                with TenantScope.set(tenant_id):
                    await self.stm_store.purge_consumed(
                        ctx=PurgeConsumedContext(
                            request_id=_request_id("retention.purge"),
                            tenant_id=tenant_id,
                            before=before,
                        )
                    )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "retention sweep: purge failed for tenant=%s; continuing",
                    tenant_id,
                )


def _component_fqn(obj: object) -> str:
    cls = obj.__class__
    return f"{cls.__module__}.{cls.__qualname__}"


def _instr(tenant_config: TenantConfig, phase: str) -> str | None:
    if tenant_config.dream_instructions is None:
        return None
    return tenant_config.dream_instructions.get(phase)


def _publish_context(tenant_id: TenantId) -> Any:
    from dreamer.api.contexts import PublishContext

    return PublishContext(request_id=_request_id("publish"), tenant_id=tenant_id)


async def _noop_emit_progress(message: str, payload: Mapping[str, Any]) -> None:
    return None


class _FanoutUsageSink:
    """Adapt a list of UsageSinks into a single sink (parallel fire-and-forget)."""

    multi_tenant: ClassVar[bool] = True

    def __init__(self, sinks: list[UsageSink]) -> None:
        self._sinks = sinks

    async def record(self, event: UsageEvent, *, ctx: UsageContext) -> None:
        if not self._sinks:
            return
        await asyncio.gather(
            *(_safe_record_usage(s, event, ctx) for s in self._sinks),
            return_exceptions=False,
        )


class _FanoutAuditSink:
    """Adapt a list of AuditSinks into a single sink (parallel fire-and-forget)."""

    multi_tenant: ClassVar[bool] = True

    def __init__(self, sinks: list[AuditSink]) -> None:
        self._sinks = sinks

    async def record(self, event: AuditEvent, *, ctx: AuditContext) -> None:
        if not self._sinks:
            return
        await asyncio.gather(
            *(_safe_record_audit(s, event, ctx) for s in self._sinks),
            return_exceptions=False,
        )


async def _safe_record_usage(sink: UsageSink, event: UsageEvent, ctx: UsageContext) -> None:
    try:
        await sink.record(event, ctx=ctx)
    except Exception:  # noqa: BLE001
        logger.exception("UsageSink %r raised; continuing", sink)


async def _safe_record_audit(
    sink: AuditSink, event: AuditEvent, ctx: AuditContext
) -> None:
    try:
        await sink.record(event, ctx=ctx)
    except Exception:  # noqa: BLE001
        logger.exception("AuditSink %r raised; continuing", sink)


__all__ = [
    "Orchestrator",
    "StmRetentionConfig",
]
