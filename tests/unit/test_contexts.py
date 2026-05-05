from __future__ import annotations

from dataclasses import asdict, fields, is_dataclass
from datetime import UTC, datetime

import pytest

from dreamer.api.contexts import (
    AcquireLeaseContext,
    AuditContext,
    AuthContext,
    ClaimContext,
    ClearContextPendingContext,
    CommitWorkspaceContext,
    ContextPhaseContext,
    ContextPhaseServices,
    ContextReadContext,
    CountContext,
    DeprovisionContext,
    DiscardWorkspaceContext,
    DreamFailedContext,
    DreamFailedServices,
    DreamGateContext,
    DreamGateServices,
    DreamProgressContext,
    DreamProgressServices,
    GetContextPendingContext,
    LifecycleContext,
    ListUnconsumedContext,
    LTMPhaseContext,
    LTMPhaseServices,
    MarkConsumedContext,
    MCPToolContext,
    MiddlewaresContext,
    OpenWorkspaceContext,
    PostContextUpdateContext,
    PostContextUpdateServices,
    PostDreamContext,
    PostDreamServices,
    PostLTMUpdateContext,
    PostLTMUpdateServices,
    PostMemorySubmitContext,
    PreContextUpdateContext,
    PreContextUpdateServices,
    PreDreamContext,
    PreDreamServices,
    PreLTMUpdateContext,
    PreLTMUpdateServices,
    PreMemorySubmitContext,
    ProvisionContext,
    PublishContext,
    PurgeConsumedContext,
    RateLimitContext,
    ReclaimContext,
    ReclaimLeasesContext,
    ReleaseContext,
    ReleaseLeaseContext,
    RenewLeaseContext,
    ResetContext,
    RoutesContext,
    SecretContext,
    SecretRotationContext,
    SerializeContext,
    SerializeServices,
    SetContextPendingContext,
    SubmitContext,
    SubscribeContext,
    TenancyContext,
    TenantConfigLookupContext,
    TenantDataContext,
    TenantRegistryContext,
    TriggerStartContext,
    TriggerStartServices,
    TriggerStopContext,
    TxBeginContext,
    TxCommitContext,
    TxPrepareContext,
    TxRollbackContext,
    UsageContext,
)
from dreamer.api.types import (
    Diff,
    Memory,
    MemoryBatch,
    Principal,
)

ALL_IN_PROCESS_CONTEXTS = [
    AuthContext,
    TenancyContext,
    SubmitContext,
    ListUnconsumedContext,
    ClaimContext,
    MarkConsumedContext,
    ReleaseContext,
    CountContext,
    ReclaimContext,
    PurgeConsumedContext,
    OpenWorkspaceContext,
    CommitWorkspaceContext,
    DiscardWorkspaceContext,
    SetContextPendingContext,
    GetContextPendingContext,
    ClearContextPendingContext,
    AcquireLeaseContext,
    RenewLeaseContext,
    ReleaseLeaseContext,
    ReclaimLeasesContext,
    TenantRegistryContext,
    TenantConfigLookupContext,
    ProvisionContext,
    DeprovisionContext,
    ResetContext,
    PublishContext,
    SubscribeContext,
    SecretContext,
    SecretRotationContext,
    UsageContext,
    AuditContext,
    RateLimitContext,
    MCPToolContext,
    ContextReadContext,
    LifecycleContext,
    RoutesContext,
    MiddlewaresContext,
    TenantDataContext,
    TxBeginContext,
    TxPrepareContext,
    TxCommitContext,
    TxRollbackContext,
    PreMemorySubmitContext,
    PostMemorySubmitContext,
    TriggerStopContext,
]

ALL_PHASE_CONTEXTS = [
    LTMPhaseContext,
    ContextPhaseContext,
    SerializeContext,
    DreamGateContext,
    TriggerStartContext,
    PreDreamContext,
    PostDreamContext,
    PreLTMUpdateContext,
    PostLTMUpdateContext,
    PreContextUpdateContext,
    PostContextUpdateContext,
    DreamFailedContext,
    DreamProgressContext,
]

ALL_PHASE_SERVICES = [
    LTMPhaseServices,
    ContextPhaseServices,
    SerializeServices,
    DreamGateServices,
    TriggerStartServices,
    PreDreamServices,
    PostDreamServices,
    PreLTMUpdateServices,
    PostLTMUpdateServices,
    PreContextUpdateServices,
    PostContextUpdateServices,
    DreamFailedServices,
    DreamProgressServices,
]


@pytest.mark.parametrize(
    "cls",
    ALL_IN_PROCESS_CONTEXTS + ALL_PHASE_CONTEXTS + ALL_PHASE_SERVICES,
)
def test_dataclass_is_frozen_and_slotted(cls: type) -> None:
    assert is_dataclass(cls)
    assert "__slots__" in cls.__dict__, f"{cls.__name__} should have __slots__"
    params = getattr(cls, "__dataclass_params__", None)
    assert params is not None
    assert params.frozen is True, f"{cls.__name__} should be frozen"


def _now() -> datetime:
    return datetime(2026, 5, 2, tzinfo=UTC)


def test_pre_memory_submit_context_has_mutable_memories_list() -> None:
    """`PreMemorySubmitContext.memories` is intentionally a mutable list so
    hooks can drop/expand/replace items in place."""
    p = Principal(id="agent")
    m = Memory(
        tenant_id="default",
        agent_id="agent",
        type="observation",
        title="t",
        content="c",
        submitted_at=_now(),
    )
    ctx = PreMemorySubmitContext(
        request_id="r1",
        tenant_id="default",
        principal=p,
        memories=[m],
    )
    ctx.memories.append(m)
    assert len(ctx.memories) == 2
    ctx.memories.clear()
    assert ctx.memories == []


def test_post_dream_context_carries_diffs_and_resumed_flag() -> None:
    diff = Diff(added=["x.md"])
    ctx = PostDreamContext(
        request_id="r1",
        tenant_id="default",
        lease_id="L1",
        trigger_name="external",
        success=True,
        batch_size=5,
        ltm_diff=diff,
        context_diff=diff,
        resumed=False,
    )
    assert ctx.ltm_diff == diff
    assert ctx.resumed is False


def test_phase_contexts_are_dataclass_asdict_compatible() -> None:
    """`asdict` works on the dataclass shell; Pydantic models embedded inside
    go through their own `model_dump` separately — `asdict` does not recurse
    into them."""
    diff = Diff(added=["a"])
    ctx = PostLTMUpdateContext(
        request_id="r1",
        tenant_id="default",
        lease_id="L1",
        ltm_workspace_id="W1",
        ltm_diff=diff,
    )
    out = asdict(ctx)
    assert out["request_id"] == "r1"
    assert out["ltm_diff"].added == ["a"]


def test_every_phase_context_has_request_id_and_tenant_id() -> None:
    for cls in ALL_PHASE_CONTEXTS:
        names = {f.name for f in fields(cls)}
        assert "request_id" in names, f"{cls.__name__} missing request_id"
        assert "tenant_id" in names, f"{cls.__name__} missing tenant_id"


def test_every_phase_services_has_emit_progress() -> None:
    for cls in ALL_PHASE_SERVICES:
        names = {f.name for f in fields(cls)}
        assert "emit_progress" in names, f"{cls.__name__} missing emit_progress"
        assert "secrets" in names, f"{cls.__name__} missing secrets"
        assert "usage" in names, f"{cls.__name__} missing usage"
        assert "audit" in names, f"{cls.__name__} missing audit"
        assert "clock" in names, f"{cls.__name__} missing clock"


def test_ltm_phase_context_round_trip() -> None:
    m = Memory(
        id="m1",
        tenant_id="default",
        agent_id="a",
        type="observation",
        title="t",
        content="c",
        submitted_at=_now(),
    )
    batch = MemoryBatch(
        lease_id="L1", tenant_id="default", memories=[m], snapshot_at=_now()
    )
    ctx = LTMPhaseContext(
        request_id="r1",
        tenant_id="default",
        lease_id="L1",
        batch=batch,
        ltm_workspace_id="W1",
    )
    assert ctx.batch is batch
    assert ctx.instructions is None
