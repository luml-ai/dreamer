"""`*Context` and `*Services` dataclasses for every Protocol method.

Common fields on every `*Context`: `request_id: str`, and (when applicable)
`tenant_id: TenantId`.

Phase-shaped contexts (those listed in SPEC.md § "Phase-shaped *Context +
*Services pairs") MUST be JSON-serializable in anticipation of a future
`PhaseDispatcher`: they round-trip through Pydantic v2 / `dataclasses.asdict`
cleanly. They MUST NOT carry callables, open resources, or live component
references. The `DreamJob` payload itself follows the same rule, since it
actually does cross `JobQueue` today.

In-process contexts carry whatever the method needs — including process-local
references — and have no serializability requirement.

`*Services` are process-local; they may carry callables, secret resolvers,
sinks, clocks, and live workspace handles. They never cross JobQueue.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from dreamer.api.types import (
    AuditEvent,
    Diff,
    Memory,
    MemoryBatch,
    MemorySubmission,
    Principal,
    TenantId,
    UsageEvent,
    Workspace,
)

if TYPE_CHECKING:
    from starlette.requests import Request as StarletteRequest

    from dreamer.api.audit import AuditSink
    from dreamer.api.secrets import SecretResolver
    from dreamer.api.usage import UsageSink


EmitProgress = Callable[[str, Mapping[str, Any]], Awaitable[None]]
Clock = Callable[[], datetime]

# The shared memory-submit pipeline, bound to the active request. Takes
# `submit_memory`-shaped args; returns one MemorySubmission per persisted
# memory (empty when pre-submit hooks filtered everything).
SubmitMemory = Callable[[Mapping[str, Any]], Awaitable[list[MemorySubmission]]]


# Opaque marker for transactional handles; concrete impls supply their own.
TxHandle = Any


@dataclass(frozen=True, slots=True)
class AuthContext:
    request_id: str
    request: StarletteRequest | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TenancyContext:
    request_id: str
    principal: Principal
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SubmitContext:
    request_id: str
    tenant_id: TenantId
    principal_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ListUnconsumedContext:
    request_id: str
    tenant_id: TenantId
    limit: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ClaimContext:
    request_id: str
    tenant_id: TenantId
    lease_id: str
    max_batch_size: int | None = None
    snapshot_at: datetime | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MarkConsumedContext:
    request_id: str
    tenant_id: TenantId
    lease_id: str
    memory_ids: tuple[str, ...] = ()
    consumed_at: datetime | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ReleaseContext:
    request_id: str
    tenant_id: TenantId
    lease_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CountContext:
    request_id: str
    tenant_id: TenantId
    exclude_types: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ReclaimContext:
    request_id: str
    tenant_id: TenantId
    expired_lease_ids: frozenset[str] = field(default_factory=frozenset)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PurgeConsumedContext:
    request_id: str
    tenant_id: TenantId
    before: datetime
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OpenWorkspaceContext:
    request_id: str
    tenant_id: TenantId
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CommitWorkspaceContext:
    request_id: str
    tenant_id: TenantId
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DiscardWorkspaceContext:
    request_id: str
    tenant_id: TenantId
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SetContextPendingContext:
    request_id: str
    tenant_id: TenantId
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GetContextPendingContext:
    request_id: str
    tenant_id: TenantId
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ClearContextPendingContext:
    request_id: str
    tenant_id: TenantId
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AcquireLeaseContext:
    request_id: str
    tenant_id: TenantId
    ttl_seconds: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RenewLeaseContext:
    request_id: str
    tenant_id: TenantId
    lease_id: str
    ttl_seconds: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ReleaseLeaseContext:
    request_id: str
    tenant_id: TenantId
    lease_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ReclaimLeasesContext:
    request_id: str
    tenant_id: TenantId
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TenantRegistryContext:
    request_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TenantConfigLookupContext:
    request_id: str
    tenant_id: TenantId
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ProvisionContext:
    request_id: str
    tenant_id: TenantId
    init_config: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DeprovisionContext:
    request_id: str
    tenant_id: TenantId
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ResetContext:
    request_id: str
    tenant_id: TenantId
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TenantDataContext:
    request_id: str
    tenant_id: TenantId
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PublishContext:
    request_id: str
    tenant_id: TenantId
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SubscribeContext:
    request_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SecretContext:
    request_id: str
    tenant_id: TenantId | None = None
    if_changed_since: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SecretRotationContext:
    request_id: str
    tenant_id: TenantId | None
    secret_name: str
    new_version: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class UsageContext:
    request_id: str
    tenant_id: TenantId
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AuditContext:
    request_id: str
    tenant_id: TenantId
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RateLimitContext:
    request_id: str
    tenant_id: TenantId
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MCPToolContext:
    request_id: str
    tenant_id: TenantId
    principal: Principal
    tool_name: str
    submit_memory: SubmitMemory | None = None
    """Shared memory-submit pipeline bound to this request. Runs the same
    validation, hooks, and idempotency semantics as the built-in
    `submit_memory` tool; raises `MemorySubmitError` on rejection."""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ContextReadContext:
    request_id: str
    tenant_id: TenantId
    principal_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LifecycleContext:
    request_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RoutesContext:
    request_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MiddlewaresContext:
    request_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TxBeginContext:
    request_id: str
    tenant_id: TenantId
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TxPrepareContext:
    request_id: str
    tenant_id: TenantId
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TxCommitContext:
    request_id: str
    tenant_id: TenantId
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TxRollbackContext:
    request_id: str
    tenant_id: TenantId
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PreMemorySubmitContext:
    request_id: str
    tenant_id: TenantId
    principal: Principal
    memories: list[Memory]
    """Mutable list. Hooks may mutate items in place, clear, append, or replace."""
    params: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PostMemorySubmitContext:
    request_id: str
    tenant_id: TenantId
    principal: Principal
    persisted: tuple[Memory, ...]
    params: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TriggerStopContext:
    request_id: str
    tenant_id: TenantId
    trigger_name: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


# Phase-shaped contexts and services below: contexts are JSON-serializable;
# services may carry callables and live component references.


@dataclass(frozen=True, slots=True)
class LTMPhaseContext:
    request_id: str
    tenant_id: TenantId
    lease_id: str
    batch: MemoryBatch
    ltm_workspace_id: str
    instructions: str | None = None
    params: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LTMPhaseServices:
    emit_progress: EmitProgress
    secrets: SecretResolver
    usage: UsageSink
    audit: AuditSink
    clock: Clock
    ltm_workspace: Workspace


@dataclass(frozen=True, slots=True)
class ContextPhaseContext:
    request_id: str
    tenant_id: TenantId
    lease_id: str
    ltm_workspace_id: str
    ltm_diff: Diff
    context_workspace_id: str
    instructions: str | None = None
    params: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ContextPhaseServices:
    emit_progress: EmitProgress
    secrets: SecretResolver
    usage: UsageSink
    audit: AuditSink
    clock: Clock
    ltm_workspace: Workspace
    context_workspace: Workspace


@dataclass(frozen=True, slots=True)
class SerializeContext:
    request_id: str
    tenant_id: TenantId
    lease_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SerializeServices:
    emit_progress: EmitProgress
    secrets: SecretResolver
    usage: UsageSink
    audit: AuditSink
    clock: Clock


@dataclass(frozen=True, slots=True)
class DreamGateContext:
    request_id: str
    tenant_id: TenantId
    trigger_name: str
    unconsumed_count: int | None = None
    params: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DreamGateServices:
    emit_progress: EmitProgress
    secrets: SecretResolver
    usage: UsageSink
    audit: AuditSink
    clock: Clock


@dataclass(frozen=True, slots=True)
class TriggerStartContext:
    request_id: str
    tenant_id: TenantId
    trigger_name: str
    params: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TriggerStartServices:
    emit_progress: EmitProgress
    secrets: SecretResolver
    usage: UsageSink
    audit: AuditSink
    clock: Clock
    fire: Callable[[], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class PreDreamContext:
    request_id: str
    tenant_id: TenantId
    lease_id: str
    trigger_name: str
    unconsumed_count: int
    params: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PreDreamServices:
    emit_progress: EmitProgress
    secrets: SecretResolver
    usage: UsageSink
    audit: AuditSink
    clock: Clock


@dataclass(frozen=True, slots=True)
class PostDreamContext:
    request_id: str
    tenant_id: TenantId
    lease_id: str
    trigger_name: str
    success: bool
    batch_size: int
    ltm_diff: Diff | None = None
    context_diff: Diff | None = None
    resumed: bool = False
    error: str | None = None
    params: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PostDreamServices:
    emit_progress: EmitProgress
    secrets: SecretResolver
    usage: UsageSink
    audit: AuditSink
    clock: Clock


@dataclass(frozen=True, slots=True)
class PreLTMUpdateContext:
    request_id: str
    tenant_id: TenantId
    lease_id: str
    ltm_workspace_id: str
    batch_size: int
    params: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PreLTMUpdateServices:
    emit_progress: EmitProgress
    secrets: SecretResolver
    usage: UsageSink
    audit: AuditSink
    clock: Clock


@dataclass(frozen=True, slots=True)
class PostLTMUpdateContext:
    request_id: str
    tenant_id: TenantId
    lease_id: str
    ltm_workspace_id: str
    ltm_diff: Diff
    params: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PostLTMUpdateServices:
    emit_progress: EmitProgress
    secrets: SecretResolver
    usage: UsageSink
    audit: AuditSink
    clock: Clock


@dataclass(frozen=True, slots=True)
class PreContextUpdateContext:
    request_id: str
    tenant_id: TenantId
    lease_id: str
    ltm_workspace_id: str
    ltm_diff: Diff
    context_workspace_id: str
    params: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PreContextUpdateServices:
    emit_progress: EmitProgress
    secrets: SecretResolver
    usage: UsageSink
    audit: AuditSink
    clock: Clock


@dataclass(frozen=True, slots=True)
class PostContextUpdateContext:
    request_id: str
    tenant_id: TenantId
    lease_id: str
    context_workspace_id: str
    context_diff: Diff
    params: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PostContextUpdateServices:
    emit_progress: EmitProgress
    secrets: SecretResolver
    usage: UsageSink
    audit: AuditSink
    clock: Clock


@dataclass(frozen=True, slots=True)
class DreamFailedContext:
    request_id: str
    tenant_id: TenantId
    lease_id: str | None
    trigger_name: str
    phase: Literal["pre_dream", "ltm", "context", "post_dream", "post_ltm", "post_context"]
    error: str
    params: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DreamFailedServices:
    emit_progress: EmitProgress
    secrets: SecretResolver
    usage: UsageSink
    audit: AuditSink
    clock: Clock


@dataclass(frozen=True, slots=True)
class DreamProgressContext:
    request_id: str
    tenant_id: TenantId
    lease_id: str
    phase: str
    message: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    params: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DreamProgressServices:
    emit_progress: EmitProgress
    secrets: SecretResolver
    usage: UsageSink
    audit: AuditSink
    clock: Clock


__all__ = [
    "AcquireLeaseContext",
    "AuditContext",
    "AuditEvent",
    "AuthContext",
    "ClaimContext",
    "ClearContextPendingContext",
    "Clock",
    "CommitWorkspaceContext",
    "ContextPhaseContext",
    "ContextPhaseServices",
    "ContextReadContext",
    "CountContext",
    "DeprovisionContext",
    "DiscardWorkspaceContext",
    "DreamFailedContext",
    "DreamFailedServices",
    "DreamGateContext",
    "DreamGateServices",
    "DreamProgressContext",
    "DreamProgressServices",
    "EmitProgress",
    "GetContextPendingContext",
    "LTMPhaseContext",
    "LTMPhaseServices",
    "LifecycleContext",
    "ListUnconsumedContext",
    "MCPToolContext",
    "MarkConsumedContext",
    "MiddlewaresContext",
    "OpenWorkspaceContext",
    "PostContextUpdateContext",
    "PostContextUpdateServices",
    "PostDreamContext",
    "PostDreamServices",
    "PostLTMUpdateContext",
    "PostLTMUpdateServices",
    "PostMemorySubmitContext",
    "PreContextUpdateContext",
    "PreContextUpdateServices",
    "PreDreamContext",
    "PreDreamServices",
    "PreLTMUpdateContext",
    "PreLTMUpdateServices",
    "PreMemorySubmitContext",
    "ProvisionContext",
    "PublishContext",
    "PurgeConsumedContext",
    "RateLimitContext",
    "ReclaimContext",
    "ReclaimLeasesContext",
    "ReleaseContext",
    "ReleaseLeaseContext",
    "RenewLeaseContext",
    "ResetContext",
    "RoutesContext",
    "SecretContext",
    "SecretRotationContext",
    "SerializeContext",
    "SerializeServices",
    "SetContextPendingContext",
    "SubmitContext",
    "SubmitMemory",
    "SubscribeContext",
    "TenancyContext",
    "TenantConfigLookupContext",
    "TenantDataContext",
    "TenantRegistryContext",
    "TriggerStartContext",
    "TriggerStartServices",
    "TriggerStopContext",
    "TxBeginContext",
    "TxCommitContext",
    "TxPrepareContext",
    "TxRollbackContext",
    "TxHandle",
    "UsageContext",
    "UsageEvent",
]
