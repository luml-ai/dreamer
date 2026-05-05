"""Static tenant lifecycle / registry / config provider.

Default contrib components for the three tenancy slots:

- ``StaticTenantRegistry`` — a fixed list of tenants, mutable in-memory so the
  control surface's ``provision`` / ``deprovision`` add and remove tenants
  atomically.
- ``StaticTenantConfigProvider`` — keyed lookups against a static map of
  per-tenant overrides; enforces "tenant ``memory_types`` ⊆ global" both at
  construction and when a tenant config is resolved.
- ``StaticTenantLifecycle`` — iterates every component implementing the
  ``TenantData`` capability and fans out the corresponding lifecycle event,
  aggregating failures into ``TenantDataError`` so the operator can resolve
  cleanly before considering the lifecycle event complete. Waits for any
  in-flight dream lease to release before sweeping ``TenantData`` impls.

All three components are single-tenant by declaration: ``multi_tenant = False``
in the framework's effective-mode computation. Custom MT registries / config
providers / lifecycles can override this without changing the call sites.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable, Mapping
from typing import Any, ClassVar, Literal

from dreamer.api.compat import implements
from dreamer.api.contexts import (
    DeprovisionContext,
    ProvisionContext,
    ResetContext,
    TenantConfigLookupContext,
    TenantDataContext,
    TenantRegistryContext,
)
from dreamer.api.errors import ConfigError, TenantDataError
from dreamer.api.tenants import (
    TenantConfigProvider,
    TenantData,
    TenantLifecycle,
    TenantRegistry,
)
from dreamer.api.types import (
    DEFAULT_TENANT_ID,
    MemoryType,
    TenantConfig,
    TenantId,
)

ActiveLeaseWaiter = Callable[[TenantId, float], Awaitable[None]]

__all__ = [
    "ActiveLeaseWaiter",
    "StaticTenantConfigProvider",
    "StaticTenantLifecycle",
    "StaticTenantRegistry",
]


@implements(TenantRegistry, version=1)
class StaticTenantRegistry:
    """A fixed list of tenants kept in memory.

    The list is initialised from config but mutates in response to provision /
    deprovision calls so the control surface can grow and shrink the population
    without requiring a config reload.
    """

    multi_tenant: ClassVar[bool] = False

    def __init__(self, tenants: Iterable[TenantId] | None = None) -> None:
        seed = list(tenants) if tenants is not None else [DEFAULT_TENANT_ID]
        self._tenants: list[TenantId] = []
        for tid in seed:
            if tid not in self._tenants:
                self._tenants.append(tid)
        self._lock = asyncio.Lock()

    async def list_tenants(self, *, ctx: TenantRegistryContext) -> list[TenantId]:
        async with self._lock:
            return list(self._tenants)

    async def exists(self, tenant_id: TenantId, *, ctx: TenantRegistryContext) -> bool:
        async with self._lock:
            return tenant_id in self._tenants

    async def add(self, tenant_id: TenantId) -> bool:
        """Insert a tenant if not already present. Returns True if added."""
        async with self._lock:
            if tenant_id in self._tenants:
                return False
            self._tenants.append(tenant_id)
            return True

    async def remove(self, tenant_id: TenantId) -> bool:
        """Remove a tenant if present. Returns True if removed."""
        async with self._lock:
            if tenant_id not in self._tenants:
                return False
            self._tenants.remove(tenant_id)
            return True


@implements(TenantConfigProvider, version=1)
class StaticTenantConfigProvider:
    """Resolves per-tenant overrides from a static mapping.

    ``overrides`` maps ``tenant_id`` to a ``TenantConfig`` (already-built dataclass)
    or to a dict matching ``TenantConfig`` fields. ``memory_types`` overrides MUST
    be a subset of the global ``memory_types`` (matched by ``name``); a violation
    raises ``ConfigError`` either at construction time or — if the global types
    are passed in later — on the first ``get`` call.
    """

    multi_tenant: ClassVar[bool] = False

    def __init__(
        self,
        overrides: Mapping[TenantId, TenantConfig | Mapping[str, Any]] | None = None,
        *,
        global_memory_types: Iterable[MemoryType] | None = None,
    ) -> None:
        self._global_memory_types: tuple[MemoryType, ...] = (
            tuple(global_memory_types) if global_memory_types is not None else ()
        )
        self._overrides: dict[TenantId, TenantConfig] = {}
        for tenant_id, value in (overrides or {}).items():
            self._overrides[tenant_id] = _coerce_tenant_config(value)
        self._validate_all()

    def set_global_memory_types(self, memory_types: Iterable[MemoryType]) -> None:
        """Late-binding setter for the global type set, used when the loader
        wires up the provider before STM types are resolved."""
        self._global_memory_types = tuple(memory_types)
        self._validate_all()

    async def get(
        self, tenant_id: TenantId, *, ctx: TenantConfigLookupContext
    ) -> TenantConfig:
        cfg = self._overrides.get(tenant_id)
        if cfg is None:
            return TenantConfig()
        # Re-validate on resolve so any late-set globals are honoured.
        self._validate_one(tenant_id, cfg)
        return cfg

    def _validate_all(self) -> None:
        for tenant_id, cfg in self._overrides.items():
            self._validate_one(tenant_id, cfg)

    def _validate_one(self, tenant_id: TenantId, cfg: TenantConfig) -> None:
        if cfg.memory_types is None:
            return
        if not self._global_memory_types:
            return
        global_names = {mt.name for mt in self._global_memory_types}
        offenders = [mt.name for mt in cfg.memory_types if mt.name not in global_names]
        if offenders:
            raise ConfigError(
                f"tenant {tenant_id!r} memory_types must be a subset of global "
                f"memory_types; unknown: {sorted(offenders)}"
            )


def _coerce_tenant_config(value: TenantConfig | Mapping[str, Any]) -> TenantConfig:
    if isinstance(value, TenantConfig):
        return value
    if not isinstance(value, Mapping):
        raise ConfigError(
            f"tenant override must be a mapping or TenantConfig, got {type(value).__name__}"
        )
    memory_types = value.get("memory_types")
    coerced_types: tuple[MemoryType, ...] | None
    if memory_types is None:
        coerced_types = None
    else:
        coerced_types = tuple(_coerce_memory_type(mt) for mt in memory_types)
    return TenantConfig(
        memory_types=coerced_types,
        dream_instructions=value.get("dream_instructions"),
        hook_params=value.get("hook_params"),
        metadata=value.get("metadata") or {},
    )


def _coerce_memory_type(value: MemoryType | Mapping[str, Any]) -> MemoryType:
    if isinstance(value, MemoryType):
        return value
    if isinstance(value, Mapping):
        return MemoryType.model_validate(dict(value))
    raise ConfigError(
        f"memory_types entry must be a mapping or MemoryType, got {type(value).__name__}"
    )


@implements(TenantLifecycle, version=1)
class StaticTenantLifecycle:
    """Iterates ``TenantData``-capable components on lifecycle events.

    Components are passed in as a list at construction; the framework wires the
    full component graph through. ``StaticTenantRegistry`` (when provided) is
    updated on provision and deprovision so the registry stays in sync.

    On ``deprovision``, the lifecycle waits for any in-flight dream lease to
    release before sweeping ``TenantData`` impls. The orchestrator exposes the
    wait-for-lease behaviour via a callable handed in through
    :meth:`bind_active_lease_waiter`; absent that, deprovision proceeds
    immediately (single-process default + no orchestrator wired).
    """

    multi_tenant: ClassVar[bool] = False

    def __init__(
        self,
        tenant_data_components: Iterable[Any] | None = None,
        *,
        registry: StaticTenantRegistry | None = None,
        deprovision_lease_timeout_seconds: float = 60.0,
    ) -> None:
        self._tenant_data: list[Any] = [
            c for c in (tenant_data_components or []) if isinstance(c, TenantData)
        ]
        self._registry = registry
        self._deprovision_lease_timeout = float(deprovision_lease_timeout_seconds)
        self._wait_for_lease_release: ActiveLeaseWaiter | None = None

    def set_tenant_data_components(self, components: Iterable[Any]) -> None:
        """Replace the registered TenantData fan-out list. Called by the app
        factory after the full component graph has been built."""
        self._tenant_data = [c for c in components if isinstance(c, TenantData)]

    def set_registry(self, registry: StaticTenantRegistry) -> None:
        self._registry = registry

    def bind_active_lease_waiter(self, waiter: ActiveLeaseWaiter) -> None:
        """Bind a coroutine ``waiter(tenant_id, timeout)`` that resolves once
        any in-flight dream lease for the tenant has released or the timeout
        elapses. Called by the app factory after the orchestrator is wired."""
        self._wait_for_lease_release = waiter

    async def provision(
        self, tenant_id: TenantId, *, ctx: ProvisionContext
    ) -> None:
        if self._registry is not None:
            await self._registry.add(tenant_id)
        await self._dispatch(
            "on_tenant_provisioned",
            tenant_id,
            ctx=TenantDataContext(
                request_id=ctx.request_id,
                tenant_id=tenant_id,
                metadata=dict(ctx.init_config),
            ),
        )

    async def deprovision(
        self,
        tenant_id: TenantId,
        *,
        mode: Literal["soft", "hard"],
        ctx: DeprovisionContext,
    ) -> None:
        if self._wait_for_lease_release is not None:
            try:
                await self._wait_for_lease_release(
                    tenant_id, self._deprovision_lease_timeout
                )
            except Exception as exc:  # noqa: BLE001 — lease wait failure should not silently drop
                raise TenantDataError(
                    f"deprovision waited on in-flight lease for tenant {tenant_id!r} "
                    f"and the wait raised: {type(exc).__name__}: {exc}",
                    failures=[exc],
                ) from exc
        await self._dispatch(
            "on_tenant_deprovisioned",
            tenant_id,
            ctx=TenantDataContext(request_id=ctx.request_id, tenant_id=tenant_id),
            mode=mode,
        )
        if self._registry is not None:
            await self._registry.remove(tenant_id)

    async def reset(self, tenant_id: TenantId, *, ctx: ResetContext) -> None:
        await self._dispatch(
            "on_tenant_reset",
            tenant_id,
            ctx=TenantDataContext(request_id=ctx.request_id, tenant_id=tenant_id),
        )

    async def _dispatch(
        self,
        method_name: str,
        tenant_id: TenantId,
        *,
        ctx: TenantDataContext,
        mode: Literal["soft", "hard"] | None = None,
    ) -> None:
        failures: list[BaseException] = []
        offenders: list[str] = []
        for component in self._tenant_data:
            handler = getattr(component, method_name, None)
            if handler is None:
                continue
            try:
                if mode is not None:
                    await handler(tenant_id, mode=mode, ctx=ctx)
                else:
                    await handler(tenant_id, ctx=ctx)
            except Exception as exc:  # noqa: BLE001 — aggregate per spec
                failures.append(exc)
                offenders.append(_component_label(component))
        if failures:
            joined = ", ".join(offenders)
            raise TenantDataError(
                f"{method_name} failed for components: {joined}",
                failures=failures,
            )


def _component_label(component: object) -> str:
    cls = type(component)
    return f"{cls.__module__}.{cls.__qualname__}"
