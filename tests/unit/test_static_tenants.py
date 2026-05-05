from __future__ import annotations

import asyncio
from typing import ClassVar, Literal

import pytest

from dreamer.api.contexts import (
    DeprovisionContext,
    ProvisionContext,
    ResetContext,
    TenantConfigLookupContext,
    TenantDataContext,
    TenantRegistryContext,
)
from dreamer.api.errors import ConfigError, TenantDataError
from dreamer.api.types import MemoryType, TenantConfig, TenantId
from dreamer.contrib.tenants.static import (
    StaticTenantConfigProvider,
    StaticTenantLifecycle,
    StaticTenantRegistry,
)


def _reg_ctx() -> TenantRegistryContext:
    return TenantRegistryContext(request_id="test")


def _cfg_ctx(tenant_id: TenantId) -> TenantConfigLookupContext:
    return TenantConfigLookupContext(request_id="test", tenant_id=tenant_id)


def _provision_ctx(tenant_id: TenantId) -> ProvisionContext:
    return ProvisionContext(request_id="test", tenant_id=tenant_id)


def _deprovision_ctx(tenant_id: TenantId) -> DeprovisionContext:
    return DeprovisionContext(request_id="test", tenant_id=tenant_id)


def _reset_ctx(tenant_id: TenantId) -> ResetContext:
    return ResetContext(request_id="test", tenant_id=tenant_id)


@pytest.mark.asyncio
async def test_registry_defaults_to_default_tenant() -> None:
    reg = StaticTenantRegistry()
    assert await reg.list_tenants(ctx=_reg_ctx()) == ["default"]
    assert await reg.exists("default", ctx=_reg_ctx())
    assert not await reg.exists("missing", ctx=_reg_ctx())


@pytest.mark.asyncio
async def test_registry_initial_list_dedupes() -> None:
    reg = StaticTenantRegistry(["a", "b", "a"])
    assert await reg.list_tenants(ctx=_reg_ctx()) == ["a", "b"]


@pytest.mark.asyncio
async def test_registry_add_and_remove() -> None:
    reg = StaticTenantRegistry(["default"])
    assert await reg.add("acme") is True
    assert await reg.add("acme") is False
    assert await reg.list_tenants(ctx=_reg_ctx()) == ["default", "acme"]
    assert await reg.remove("acme") is True
    assert await reg.remove("acme") is False
    assert await reg.list_tenants(ctx=_reg_ctx()) == ["default"]


@pytest.mark.asyncio
async def test_config_provider_returns_default_when_unknown() -> None:
    provider = StaticTenantConfigProvider()
    cfg = await provider.get("missing", ctx=_cfg_ctx("missing"))
    assert isinstance(cfg, TenantConfig)
    assert cfg.memory_types is None


@pytest.mark.asyncio
async def test_config_provider_uses_dict_overrides() -> None:
    provider = StaticTenantConfigProvider(
        overrides={
            "acme": {
                "dream_instructions": {"ltm_update": "be brief"},
                "metadata": {"owner": "team-a"},
            }
        }
    )
    cfg = await provider.get("acme", ctx=_cfg_ctx("acme"))
    assert cfg.dream_instructions == {"ltm_update": "be brief"}
    assert cfg.metadata == {"owner": "team-a"}


@pytest.mark.asyncio
async def test_config_provider_uses_tenantconfig_overrides() -> None:
    cfg_in = TenantConfig(metadata={"owner": "team-b"})
    provider = StaticTenantConfigProvider(overrides={"acme": cfg_in})
    cfg_out = await provider.get("acme", ctx=_cfg_ctx("acme"))
    assert cfg_out is cfg_in


def test_config_provider_rejects_invalid_value_type() -> None:
    with pytest.raises(ConfigError, match="must be a mapping"):
        StaticTenantConfigProvider(overrides={"acme": "not-a-mapping"})  # type: ignore[dict-item]


def test_config_provider_subset_enforced_at_construction() -> None:
    global_types = (
        MemoryType(name="failure", description="x"),
        MemoryType(name="observation", description="y"),
    )
    StaticTenantConfigProvider(
        overrides={
            "acme": {
                "memory_types": [
                    {"name": "failure", "description": "tenant-only"},
                ]
            }
        },
        global_memory_types=global_types,
    )
    with pytest.raises(ConfigError, match="subset of global"):
        StaticTenantConfigProvider(
            overrides={
                "acme": {
                    "memory_types": [
                        {"name": "rare-type", "description": "not in global"}
                    ]
                }
            },
            global_memory_types=global_types,
        )


@pytest.mark.asyncio
async def test_config_provider_subset_enforced_after_late_global_set() -> None:
    provider = StaticTenantConfigProvider(
        overrides={
            "acme": {
                "memory_types": [
                    {"name": "rare-type", "description": "x"}
                ]
            }
        }
    )
    cfg = await provider.get("acme", ctx=_cfg_ctx("acme"))
    assert cfg.memory_types is not None and cfg.memory_types[0].name == "rare-type"
    with pytest.raises(ConfigError, match="subset of global"):
        provider.set_global_memory_types(
            (MemoryType(name="failure", description="x"),)
        )


class _RecordingTenantData:
    multi_tenant: ClassVar[bool] = True

    def __init__(self) -> None:
        self.events: list[str] = []

    async def on_tenant_provisioned(
        self, tenant_id: TenantId, *, ctx: TenantDataContext
    ) -> None:
        self.events.append(f"provision:{tenant_id}")

    async def on_tenant_deprovisioned(
        self,
        tenant_id: TenantId,
        *,
        mode: Literal["soft", "hard"],
        ctx: TenantDataContext,
    ) -> None:
        self.events.append(f"deprovision:{tenant_id}:{mode}")

    async def on_tenant_reset(
        self, tenant_id: TenantId, *, ctx: TenantDataContext
    ) -> None:
        self.events.append(f"reset:{tenant_id}")


class _RaisingTenantData:
    multi_tenant: ClassVar[bool] = True

    async def on_tenant_provisioned(
        self, tenant_id: TenantId, *, ctx: TenantDataContext
    ) -> None:
        raise RuntimeError("boom-provision")

    async def on_tenant_deprovisioned(
        self,
        tenant_id: TenantId,
        *,
        mode: Literal["soft", "hard"],
        ctx: TenantDataContext,
    ) -> None:
        raise RuntimeError(f"boom-{mode}")

    async def on_tenant_reset(
        self, tenant_id: TenantId, *, ctx: TenantDataContext
    ) -> None:
        raise RuntimeError("boom-reset")


@pytest.mark.asyncio
async def test_lifecycle_provision_dispatches_and_updates_registry() -> None:
    sink = _RecordingTenantData()
    registry = StaticTenantRegistry(["default"])
    lifecycle = StaticTenantLifecycle([sink], registry=registry)

    await lifecycle.provision("acme", ctx=_provision_ctx("acme"))

    assert sink.events == ["provision:acme"]
    assert "acme" in await registry.list_tenants(ctx=_reg_ctx())


@pytest.mark.asyncio
async def test_lifecycle_deprovision_dispatches_and_removes() -> None:
    sink = _RecordingTenantData()
    registry = StaticTenantRegistry(["default", "acme"])
    lifecycle = StaticTenantLifecycle([sink], registry=registry)

    await lifecycle.deprovision("acme", mode="hard", ctx=_deprovision_ctx("acme"))

    assert sink.events == ["deprovision:acme:hard"]
    assert "acme" not in await registry.list_tenants(ctx=_reg_ctx())


@pytest.mark.asyncio
async def test_lifecycle_reset_dispatches_only_reset() -> None:
    sink = _RecordingTenantData()
    lifecycle = StaticTenantLifecycle([sink])

    await lifecycle.reset("acme", ctx=_reset_ctx("acme"))

    assert sink.events == ["reset:acme"]


@pytest.mark.asyncio
async def test_lifecycle_aggregates_failures() -> None:
    sink_ok = _RecordingTenantData()
    sink_bad = _RaisingTenantData()
    lifecycle = StaticTenantLifecycle([sink_ok, sink_bad])

    with pytest.raises(TenantDataError) as excinfo:
        await lifecycle.deprovision("acme", mode="soft", ctx=_deprovision_ctx("acme"))
    err = excinfo.value
    assert "_RaisingTenantData" in str(err)
    assert len(err.failures) == 1
    assert "boom-soft" in str(err.failures[0])
    assert sink_ok.events == ["deprovision:acme:soft"]


@pytest.mark.asyncio
async def test_lifecycle_filters_non_tenant_data_components() -> None:
    not_a_tenant_data_impl = object()
    sink = _RecordingTenantData()
    lifecycle = StaticTenantLifecycle([not_a_tenant_data_impl, sink])
    await lifecycle.reset("acme", ctx=_reset_ctx("acme"))
    assert sink.events == ["reset:acme"]


@pytest.mark.asyncio
async def test_lifecycle_set_tenant_data_components_replaces_list() -> None:
    sink_a = _RecordingTenantData()
    sink_b = _RecordingTenantData()
    lifecycle = StaticTenantLifecycle([sink_a])
    lifecycle.set_tenant_data_components([sink_b])
    await lifecycle.reset("acme", ctx=_reset_ctx("acme"))
    assert sink_a.events == []
    assert sink_b.events == ["reset:acme"]


@pytest.mark.asyncio
async def test_lifecycle_deprovision_waits_for_lease() -> None:
    sink = _RecordingTenantData()
    lifecycle = StaticTenantLifecycle([sink])

    waited_with: list[tuple[str, float]] = []

    async def waiter(tenant_id: str, deadline_seconds: float) -> None:
        waited_with.append((tenant_id, deadline_seconds))
        await asyncio.sleep(0)

    lifecycle.bind_active_lease_waiter(waiter)
    await lifecycle.deprovision("acme", mode="hard", ctx=_deprovision_ctx("acme"))

    assert waited_with == [("acme", 60.0)]
    assert sink.events == ["deprovision:acme:hard"]


@pytest.mark.asyncio
async def test_lifecycle_lease_wait_failure_surfaces_as_tenant_data_error() -> None:
    sink = _RecordingTenantData()
    lifecycle = StaticTenantLifecycle([sink])

    async def waiter(tenant_id: str, deadline_seconds: float) -> None:
        raise RuntimeError("orchestrator gone")

    lifecycle.bind_active_lease_waiter(waiter)
    with pytest.raises(TenantDataError, match="in-flight lease"):
        await lifecycle.deprovision(
            "acme", mode="hard", ctx=_deprovision_ctx("acme")
        )
    assert sink.events == []
