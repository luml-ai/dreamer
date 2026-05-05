from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, Literal

import pytest

from dreamer.api.contexts import (
    DeprovisionContext,
    ProvisionContext,
    ResetContext,
    TenantConfigLookupContext,
    TenantRegistryContext,
)
from dreamer.api.errors import ConfigError
from dreamer.api.types import TenantConfig, TenantId
from dreamer.server.control import Control


class StubRegistry:
    multi_tenant: ClassVar[bool] = False

    def __init__(self, tenants: list[TenantId]) -> None:
        self.tenants = list(tenants)

    async def list_tenants(self, *, ctx: TenantRegistryContext) -> list[TenantId]:
        return list(self.tenants)

    async def exists(self, tenant_id: TenantId, *, ctx: TenantRegistryContext) -> bool:
        return tenant_id in self.tenants


class StubConfigProvider:
    multi_tenant: ClassVar[bool] = False

    async def get(
        self, tenant_id: TenantId, *, ctx: TenantConfigLookupContext
    ) -> TenantConfig:
        return TenantConfig()


class StubLifecycle:
    multi_tenant: ClassVar[bool] = False

    def __init__(self) -> None:
        self.events: list[str] = []

    async def provision(self, tenant_id: TenantId, *, ctx: ProvisionContext) -> None:
        self.events.append(f"provision:{tenant_id}")

    async def deprovision(
        self,
        tenant_id: TenantId,
        *,
        mode: Literal["soft", "hard"],
        ctx: DeprovisionContext,
    ) -> None:
        self.events.append(f"deprovision:{tenant_id}:{mode}")

    async def reset(self, tenant_id: TenantId, *, ctx: ResetContext) -> None:
        self.events.append(f"reset:{tenant_id}")


def _build(*, mt: bool = False) -> tuple[Control, StubLifecycle, StubRegistry]:
    registry = StubRegistry(["default"])
    provider = StubConfigProvider()
    lifecycle = StubLifecycle()
    control = Control(
        tenant_registry=registry,
        tenant_config_provider=provider,
        tenant_lifecycle=lifecycle,
        effective_multi_tenant=mt,
    )
    return control, lifecycle, registry


@pytest.mark.asyncio
async def test_list_tenants_delegates_to_registry() -> None:
    control, _, _ = _build()
    assert await control.list_tenants() == ["default"]


@pytest.mark.asyncio
async def test_default_tenant_passes_in_single_tenant_mode() -> None:
    control, lifecycle, _ = _build(mt=False)
    await control.provision_tenant("default")
    assert lifecycle.events == ["provision:default"]


@pytest.mark.asyncio
async def test_non_default_tenant_blocked_in_single_tenant_mode() -> None:
    control, _, _ = _build(mt=False)
    with pytest.raises(ConfigError, match="single-tenant"):
        await control.provision_tenant("acme")
    with pytest.raises(ConfigError, match="single-tenant"):
        await control.deprovision_tenant("acme")
    with pytest.raises(ConfigError, match="single-tenant"):
        await control.reset_tenant("acme")
    with pytest.raises(ConfigError, match="single-tenant"):
        await control.trigger_dream("acme")
    with pytest.raises(ConfigError, match="single-tenant"):
        await control.get_tenant_config("acme")


@pytest.mark.asyncio
async def test_multi_tenant_mode_allows_all_tenants() -> None:
    control, lifecycle, _ = _build(mt=True)
    await control.provision_tenant("acme")
    await control.deprovision_tenant("acme", mode="soft")
    await control.reset_tenant("acme")
    assert lifecycle.events == [
        "provision:acme",
        "deprovision:acme:soft",
        "reset:acme",
    ]


@pytest.mark.asyncio
async def test_trigger_dream_requires_orchestrator() -> None:
    control, _, _ = _build(mt=False)
    with pytest.raises(ConfigError, match="no orchestrator"):
        await control.trigger_dream("default")


@pytest.mark.asyncio
async def test_trigger_dream_after_bind() -> None:
    control, _, _ = _build(mt=False)

    captured: list[tuple[str, str]] = []

    async def fake_trigger(t: TenantId, name: str) -> Mapping[str, Any]:
        captured.append((t, name))
        return {"ok": True}

    async def fake_state() -> Mapping[str, Any]:
        return {"active_leases": []}

    control.bind_orchestrator(trigger_dream_fn=fake_trigger, state_reader_fn=fake_state)
    result = await control.trigger_dream("default", "external")
    assert result == {"ok": True}
    assert captured == [("default", "external")]
    state = await control.read_state()
    assert state == {"active_leases": []}


@pytest.mark.asyncio
async def test_get_tenant_config_default_tenant() -> None:
    control, _, _ = _build(mt=False)
    cfg = await control.get_tenant_config("default")
    assert isinstance(cfg, TenantConfig)
