"""Tenant Protocols + the `TenantScope` runtime guard.

`TenantScope` is a process-local contextvar set on every authenticated request
and every dream task. Stores MUST call `TenantScope.assert_matches(tenant_id)`
before any I/O. The conformance suite includes a leakage test that fails any
store reading another tenant's data with the wrong scope set.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from contextvars import ContextVar, Token
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Literal,
    Protocol,
    runtime_checkable,
)

from dreamer.api.types import TenantConfig, TenantId

if TYPE_CHECKING:
    from dreamer.api.contexts import (
        DeprovisionContext,
        ProvisionContext,
        ResetContext,
        TenantConfigLookupContext,
        TenantDataContext,
        TenantRegistryContext,
    )


_current_tenant: ContextVar[TenantId | None] = ContextVar("dreamer_tenant_scope", default=None)


class _TenantScopeBinding(AbstractContextManager["_TenantScopeBinding"]):
    """Returned by `TenantScope.set` so callers can `with TenantScope.set(...):`."""

    __slots__ = ("_token",)

    def __init__(self, token: Token[TenantId | None]) -> None:
        self._token = token

    def __enter__(self) -> _TenantScopeBinding:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        _current_tenant.reset(self._token)


class TenantScope:
    """Static-method facade over a process-local `ContextVar[TenantId]`.

    Usage:

        with TenantScope.set(tenant_id):
            await store.do_something()  # store calls TenantScope.assert_matches
    """

    @staticmethod
    def set(tenant_id: TenantId) -> _TenantScopeBinding:
        token = _current_tenant.set(tenant_id)
        return _TenantScopeBinding(token)

    @staticmethod
    def get() -> TenantId | None:
        return _current_tenant.get()

    @staticmethod
    def clear() -> None:
        _current_tenant.set(None)

    @staticmethod
    def assert_matches(tenant_id: TenantId) -> None:
        """Raise `RuntimeError` if the active scope is unset or does not match.

        Stores call this from every I/O method to detect cross-tenant leakage
        before it touches a backend.
        """
        active = _current_tenant.get()
        if active is None:
            raise RuntimeError(
                f"TenantScope is unset; expected tenant_id={tenant_id!r}. "
                "Stores must be invoked under an active TenantScope."
            )
        if active != tenant_id:
            raise RuntimeError(
                f"TenantScope mismatch: active={active!r} but operation is for "
                f"tenant_id={tenant_id!r}. Possible cross-tenant leakage."
            )


@runtime_checkable
class TenantRegistry(Protocol):
    """Source of truth for which tenants exist."""

    multi_tenant: ClassVar[bool] = False

    async def list_tenants(self, *, ctx: TenantRegistryContext) -> list[TenantId]: ...
    async def exists(self, tenant_id: TenantId, *, ctx: TenantRegistryContext) -> bool: ...


@runtime_checkable
class TenantConfigProvider(Protocol):
    """Resolves per-tenant configuration overrides."""

    multi_tenant: ClassVar[bool] = False

    async def get(
        self, tenant_id: TenantId, *, ctx: TenantConfigLookupContext
    ) -> TenantConfig: ...


@runtime_checkable
class TenantLifecycle(Protocol):
    """Provisioning, deprovisioning, and reset of tenants."""

    multi_tenant: ClassVar[bool] = False

    async def provision(self, tenant_id: TenantId, *, ctx: ProvisionContext) -> None: ...
    async def deprovision(
        self,
        tenant_id: TenantId,
        *,
        mode: Literal["soft", "hard"],
        ctx: DeprovisionContext,
    ) -> None: ...
    async def reset(self, tenant_id: TenantId, *, ctx: ResetContext) -> None: ...


@runtime_checkable
class TenantData(Protocol):
    """Optional capability. Components implementing this opt into the
    `TenantLifecycle` dispatch — `provision`, `deprovision` (soft|hard), `reset`
    events flow to every store/component holding per-tenant state."""

    multi_tenant: ClassVar[bool] = False

    async def on_tenant_provisioned(
        self, tenant_id: TenantId, *, ctx: TenantDataContext
    ) -> None: ...
    async def on_tenant_deprovisioned(
        self,
        tenant_id: TenantId,
        *,
        mode: Literal["soft", "hard"],
        ctx: TenantDataContext,
    ) -> None: ...
    async def on_tenant_reset(self, tenant_id: TenantId, *, ctx: TenantDataContext) -> None: ...


__all__: list[str] = [
    "TenantConfig",
    "TenantConfigProvider",
    "TenantData",
    "TenantLifecycle",
    "TenantRegistry",
    "TenantScope",
]
