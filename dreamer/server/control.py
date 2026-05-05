"""In-process control surface (`dreamer.server.control`).

Plain coroutines on a :class:`Control` object held by the running app: the CLI,
tests, and any custom admin-route component call them directly. No HTTP, no
auth — callers enforce auth themselves.

In Task 3 the orchestrator does not yet exist; this module exposes the
control-plane *shape* expected by the CLI and tests, and delegates to a small
set of pluggable hooks so the orchestrator/job queue can be wired up later
without changing call sites.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from dreamer.api.contexts import (
    DeprovisionContext,
    ProvisionContext,
    ResetContext,
    TenantConfigLookupContext,
    TenantRegistryContext,
)
from dreamer.api.errors import ConfigError
from dreamer.api.tenants import (
    TenantConfigProvider,
    TenantLifecycle,
    TenantRegistry,
)
from dreamer.api.types import TenantConfig, TenantId
from dreamer.contrib.tenants.static import (
    ActiveLeaseWaiter,
    StaticTenantLifecycle,
)

DreamTriggerFn = Callable[[TenantId, str], Awaitable[Mapping[str, Any]]]
StateReaderFn = Callable[[], Awaitable[Mapping[str, Any]]]


def _no_orchestrator(*_args: Any, **_kwargs: Any) -> Any:
    raise ConfigError(
        "control surface: no orchestrator is wired; call Control.bind_orchestrator(...) first"
    )


@dataclass(slots=True)
class Control:
    """Control surface used by the CLI, tests, and admin-route components.

    The orchestrator is bound after construction so :class:`Control` can be
    constructed before the runtime is fully assembled (the CLI uses it during
    ``config check`` to expose the resolved component graph without booting a
    full server).
    """

    tenant_registry: TenantRegistry
    tenant_config_provider: TenantConfigProvider
    tenant_lifecycle: TenantLifecycle
    effective_multi_tenant: bool = False
    trigger_dream_fn: DreamTriggerFn = field(default=_no_orchestrator)
    state_reader_fn: StateReaderFn = field(default=_no_orchestrator)

    def bind_orchestrator(
        self,
        *,
        trigger_dream_fn: DreamTriggerFn,
        state_reader_fn: StateReaderFn,
        active_lease_waiter: ActiveLeaseWaiter | None = None,
    ) -> None:
        self.trigger_dream_fn = trigger_dream_fn
        self.state_reader_fn = state_reader_fn
        if active_lease_waiter is not None and isinstance(
            self.tenant_lifecycle, StaticTenantLifecycle
        ):
            self.tenant_lifecycle.bind_active_lease_waiter(active_lease_waiter)

    def _assert_tenant_allowed(self, tenant_id: TenantId) -> None:
        if not self.effective_multi_tenant and tenant_id != "default":
            raise ConfigError(
                "deployment is single-tenant; only tenant_id='default' is allowed"
            )

    async def trigger_dream(
        self,
        tenant_id: TenantId,
        trigger_name: str = "external",
    ) -> Mapping[str, Any]:
        """Fire a dream against the orchestrator. Returns whatever the
        orchestrator returns (typically a small status dict)."""
        self._assert_tenant_allowed(tenant_id)
        return await self.trigger_dream_fn(tenant_id, trigger_name)

    async def read_state(self) -> Mapping[str, Any]:
        """Return the orchestrator's current view: active leases, last dream
        timestamps, unconsumed counts."""
        return await self.state_reader_fn()

    async def list_tenants(self) -> list[TenantId]:
        ctx = TenantRegistryContext(request_id=_request_id())
        return await self.tenant_registry.list_tenants(ctx=ctx)

    async def get_tenant_config(self, tenant_id: TenantId) -> TenantConfig:
        self._assert_tenant_allowed(tenant_id)
        ctx = TenantConfigLookupContext(request_id=_request_id(), tenant_id=tenant_id)
        return await self.tenant_config_provider.get(tenant_id, ctx=ctx)

    async def provision_tenant(
        self,
        tenant_id: TenantId,
        *,
        init_config: Mapping[str, Any] | None = None,
    ) -> None:
        self._assert_tenant_allowed(tenant_id)
        ctx = ProvisionContext(
            request_id=_request_id(),
            tenant_id=tenant_id,
            init_config=dict(init_config or {}),
        )
        await self.tenant_lifecycle.provision(tenant_id, ctx=ctx)

    async def deprovision_tenant(
        self,
        tenant_id: TenantId,
        *,
        mode: Literal["soft", "hard"] = "soft",
    ) -> None:
        self._assert_tenant_allowed(tenant_id)
        ctx = DeprovisionContext(request_id=_request_id(), tenant_id=tenant_id)
        await self.tenant_lifecycle.deprovision(tenant_id, mode=mode, ctx=ctx)

    async def reset_tenant(self, tenant_id: TenantId) -> None:
        self._assert_tenant_allowed(tenant_id)
        ctx = ResetContext(request_id=_request_id(), tenant_id=tenant_id)
        await self.tenant_lifecycle.reset(tenant_id, ctx=ctx)


def _request_id() -> str:
    return f"control.{uuid.uuid4()}"


__all__ = ["Control", "DreamTriggerFn", "StateReaderFn"]
