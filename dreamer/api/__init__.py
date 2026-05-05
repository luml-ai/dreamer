"""Public framework interfaces.

Stable surface for component authors: Protocols, capability Protocols,
data types, errors, and the `@implements` decorator. Everything in
`dreamer.api.*` is the externally consumable contract.
"""

from dreamer.api.audit import AuditSink
from dreamer.api.auth import AuthBackend, Tenancy
from dreamer.api.capabilities import (
    Lifecycle,
    Middlewares,
    Routes,
    Transactional,
)
from dreamer.api.compat import SUPPORTED_PROTOCOL_VERSIONS, implements
from dreamer.api.dream import ContextPhaseRunner, DreamGate, LTMPhaseRunner
from dreamer.api.errors import (
    AuthError,
    ConfigError,
    DreamerError,
    DreamFailedError,
    LeaseHeldError,
    ProtocolComplianceError,
    TenantDataError,
    ValidationError,
    WorkspaceError,
)
from dreamer.api.hooks import (
    DreamFailedHook,
    DreamProgressHook,
    PostContextUpdateHook,
    PostDreamHook,
    PostLTMUpdateHook,
    PostMemorySubmitHook,
    PreContextUpdateHook,
    PreDreamHook,
    PreLTMUpdateHook,
    PreMemorySubmitHook,
)
from dreamer.api.jobs import DreamJob, JobQueue
from dreamer.api.rate_limit import RateLimitDecision, RateLimiter
from dreamer.api.secrets import SecretResolver, SecretRotationHook, SecretValue
from dreamer.api.stores import (
    ContextPendingStore,
    ContextReader,
    ContextStore,
    DreamLeaseStore,
    LTMStore,
    MCPTool,
    STMSerializer,
    STMStore,
)
from dreamer.api.tenants import (
    TenantConfigProvider,
    TenantData,
    TenantLifecycle,
    TenantRegistry,
    TenantScope,
)
from dreamer.api.triggers import Trigger
from dreamer.api.types import (
    DEFAULT_TENANT_ID,
    AuditEvent,
    Diff,
    DreamLease,
    FileViewable,
    GateDecision,
    GraphViewable,
    Memory,
    MemoryBatch,
    MemoryType,
    Principal,
    RecordViewable,
    TenantConfig,
    TenantId,
    UsageEvent,
    Workspace,
)
from dreamer.api.usage import UsageSink

__all__ = [
    "DEFAULT_TENANT_ID",
    "SUPPORTED_PROTOCOL_VERSIONS",
    "AuditEvent",
    "AuditSink",
    "AuthBackend",
    "AuthError",
    "ConfigError",
    "ContextPendingStore",
    "ContextPhaseRunner",
    "ContextReader",
    "ContextStore",
    "Diff",
    "DreamFailedError",
    "DreamFailedHook",
    "DreamGate",
    "DreamJob",
    "DreamLease",
    "DreamLeaseStore",
    "DreamProgressHook",
    "DreamerError",
    "FileViewable",
    "GateDecision",
    "GraphViewable",
    "JobQueue",
    "LTMPhaseRunner",
    "LTMStore",
    "LeaseHeldError",
    "Lifecycle",
    "MCPTool",
    "Memory",
    "MemoryBatch",
    "MemoryType",
    "Middlewares",
    "PostContextUpdateHook",
    "PostDreamHook",
    "PostLTMUpdateHook",
    "PostMemorySubmitHook",
    "PreContextUpdateHook",
    "PreDreamHook",
    "PreLTMUpdateHook",
    "PreMemorySubmitHook",
    "Principal",
    "ProtocolComplianceError",
    "RateLimitDecision",
    "RateLimiter",
    "RecordViewable",
    "Routes",
    "STMSerializer",
    "STMStore",
    "SecretResolver",
    "SecretRotationHook",
    "SecretValue",
    "Tenancy",
    "TenantConfig",
    "TenantConfigProvider",
    "TenantData",
    "TenantDataError",
    "TenantId",
    "TenantLifecycle",
    "TenantRegistry",
    "TenantScope",
    "Transactional",
    "Trigger",
    "UsageEvent",
    "UsageSink",
    "ValidationError",
    "Workspace",
    "WorkspaceError",
    "implements",
]
