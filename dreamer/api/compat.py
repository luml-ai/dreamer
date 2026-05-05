"""`@implements(protocol, version)` decorator + `SUPPORTED_PROTOCOL_VERSIONS`.

A class decorated with `@implements(P, version=N)` records the declaration on
the class itself in the `__dreamer_protocols__: dict[type, int]` attribute.
Multiple `@implements` decorators accumulate (one class can implement many
Protocols at once).

The framework's compatibility window per Protocol is read from
`SUPPORTED_PROTOCOL_VERSIONS` during startup compliance checks.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from dreamer.api.audit import AuditSink
from dreamer.api.auth import AuthBackend, Tenancy
from dreamer.api.capabilities import (
    Lifecycle,
    Middlewares,
    Routes,
    Transactional,
)
from dreamer.api.dream import (
    ContextPhaseRunner,
    DreamGate,
    LTMPhaseRunner,
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
from dreamer.api.jobs import JobQueue
from dreamer.api.rate_limit import RateLimiter
from dreamer.api.secrets import SecretResolver, SecretRotationHook
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
)
from dreamer.api.triggers import Trigger
from dreamer.api.types import (
    FileViewable,
    GraphViewable,
    RecordViewable,
    Workspace,
)
from dreamer.api.usage import UsageSink

T = TypeVar("T", bound=type)


def implements(protocol: type, *, version: int) -> Callable[[T], T]:
    """Decorator: declare that a class implements `protocol` at `version`.

    Stores the declaration on the class itself in
    `__dreamer_protocols__: dict[type, int]`. Multiple `@implements` decorators
    on the same class accumulate (one class may implement many Protocols).
    """

    def deco(cls: T) -> T:
        existing: dict[type, int] = dict(getattr(cls, "__dreamer_protocols__", {}))
        existing[protocol] = version
        cls.__dreamer_protocols__ = existing  # type: ignore[attr-defined]
        return cls

    return deco


SUPPORTED_PROTOCOL_VERSIONS: dict[type, frozenset[int]] = {
    AuthBackend: frozenset({1}),
    Tenancy: frozenset({1}),
    STMStore: frozenset({1}),
    LTMStore: frozenset({1}),
    ContextPendingStore: frozenset({1}),
    ContextStore: frozenset({1}),
    DreamLeaseStore: frozenset({1}),
    LTMPhaseRunner: frozenset({1}),
    ContextPhaseRunner: frozenset({1}),
    Trigger: frozenset({1}),
    PreMemorySubmitHook: frozenset({1}),
    PostMemorySubmitHook: frozenset({1}),
    PreDreamHook: frozenset({1}),
    PostDreamHook: frozenset({1}),
    PreLTMUpdateHook: frozenset({1}),
    PostLTMUpdateHook: frozenset({1}),
    PreContextUpdateHook: frozenset({1}),
    PostContextUpdateHook: frozenset({1}),
    DreamFailedHook: frozenset({1}),
    Lifecycle: frozenset({1}),
    Routes: frozenset({1}),
    Middlewares: frozenset({1}),
    Transactional: frozenset({1}),
    Workspace: frozenset({1}),
    FileViewable: frozenset({1}),
    RecordViewable: frozenset({1}),
    GraphViewable: frozenset({1}),
    TenantRegistry: frozenset({1}),
    TenantConfigProvider: frozenset({1}),
    TenantLifecycle: frozenset({1}),
    TenantData: frozenset({1}),
    JobQueue: frozenset({1}),
    SecretResolver: frozenset({1}),
    SecretRotationHook: frozenset({1}),
    UsageSink: frozenset({1}),
    AuditSink: frozenset({1}),
    RateLimiter: frozenset({1}),
    MCPTool: frozenset({1}),
    STMSerializer: frozenset({1}),
    ContextReader: frozenset({1}),
    DreamGate: frozenset({1}),
    DreamProgressHook: frozenset({1}),
}


__all__ = [
    "SUPPORTED_PROTOCOL_VERSIONS",
    "implements",
]
