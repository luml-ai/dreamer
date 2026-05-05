from __future__ import annotations

import asyncio
from typing import Any, ClassVar, Literal

import pytest

from dreamer.api.contexts import (
    DeprovisionContext,
    LifecycleContext,
    ProvisionContext,
    ResetContext,
    TenantConfigLookupContext,
    TenantDataContext,
)
from dreamer.api.errors import ConfigError, TenantDataError
from dreamer.api.tenants import TenantScope
from dreamer.api.types import TenantId
from dreamer.contrib.jobs.inproc import InProcessJobQueue
from dreamer.contrib.tenants.static import (
    StaticTenantConfigProvider,
    StaticTenantLifecycle,
    StaticTenantRegistry,
)
from dreamer.server.control import Control
from dreamer.server.orchestrator import Orchestrator, StmRetentionConfig
from dreamer.server.runtime import HookRegistry
from dreamer.testing.fakes import (
    CollectingAuditSink,
    CollectingUsageSink,
    DeterministicDreamEngine,
    InMemoryContextStore,
    InMemoryDreamLeaseStore,
    InMemoryLTMStore,
    InMemorySTMStore,
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


def _build_control(
    *,
    effective_multi_tenant: bool = False,
    tenants: list[TenantId] | None = None,
) -> tuple[Control, StaticTenantLifecycle, StaticTenantRegistry, _RecordingTenantData]:
    sink = _RecordingTenantData()
    registry = StaticTenantRegistry(tenants or ["default"])
    provider = StaticTenantConfigProvider()
    lifecycle = StaticTenantLifecycle([sink], registry=registry)
    control = Control(
        tenant_registry=registry,
        tenant_config_provider=provider,
        tenant_lifecycle=lifecycle,
        effective_multi_tenant=effective_multi_tenant,
    )
    return control, lifecycle, registry, sink


@pytest.mark.asyncio
async def test_control_provision_dispatches_through_lifecycle() -> None:
    control, _, registry, sink = _build_control(effective_multi_tenant=True)

    await control.provision_tenant("acme", init_config={"plan": "pro"})

    assert "acme" in await registry.list_tenants(
        ctx=type("_", (), {"request_id": "test"})()
    )
    assert sink.events == ["provision:acme"]


@pytest.mark.asyncio
async def test_control_deprovision_removes_from_registry() -> None:
    control, _, registry, sink = _build_control(
        effective_multi_tenant=True, tenants=["default", "acme"]
    )

    await control.deprovision_tenant("acme", mode="hard")

    assert sink.events == ["deprovision:acme:hard"]
    tenants = await registry.list_tenants(
        ctx=type("_", (), {"request_id": "test"})()
    )
    assert "acme" not in tenants


@pytest.mark.asyncio
async def test_reset_preserves_tenant_identity() -> None:
    control, _, registry, sink = _build_control(
        effective_multi_tenant=True, tenants=["default", "acme"]
    )

    await control.reset_tenant("acme")

    tenants = await registry.list_tenants(
        ctx=type("_", (), {"request_id": "test"})()
    )
    assert "acme" in tenants
    assert sink.events == ["reset:acme"]


@pytest.mark.asyncio
async def test_control_blocks_non_default_in_single_tenant_mode() -> None:
    control, _, _, _ = _build_control(effective_multi_tenant=False)

    with pytest.raises(ConfigError, match="single-tenant"):
        await control.provision_tenant("acme")
    with pytest.raises(ConfigError, match="single-tenant"):
        await control.deprovision_tenant("acme")
    with pytest.raises(ConfigError, match="single-tenant"):
        await control.reset_tenant("acme")


@pytest.mark.asyncio
async def test_control_aggregates_tenant_data_failure() -> None:
    class _Boomer:
        multi_tenant: ClassVar[bool] = True

        async def on_tenant_provisioned(
            self, tenant_id: TenantId, *, ctx: TenantDataContext
        ) -> None:
            raise RuntimeError("boom")

        async def on_tenant_deprovisioned(
            self,
            tenant_id: TenantId,
            *,
            mode: Literal["soft", "hard"],
            ctx: TenantDataContext,
        ) -> None:
            raise RuntimeError("boom")

        async def on_tenant_reset(
            self, tenant_id: TenantId, *, ctx: TenantDataContext
        ) -> None:
            raise RuntimeError("boom")

    sink = _RecordingTenantData()
    registry = StaticTenantRegistry(["default"])
    lifecycle = StaticTenantLifecycle([sink, _Boomer()], registry=registry)
    control = Control(
        tenant_registry=registry,
        tenant_config_provider=StaticTenantConfigProvider(),
        tenant_lifecycle=lifecycle,
        effective_multi_tenant=True,
    )

    with pytest.raises(TenantDataError, match="_Boomer"):
        await control.provision_tenant("acme")
    # The well-behaved sink saw provision before the failure ran.
    assert sink.events == ["provision:acme"]


async def _build_orchestrator(
    *,
    tenants: list[TenantId],
) -> tuple[Orchestrator, InMemoryDreamLeaseStore]:
    stm = InMemorySTMStore()
    ltm = InMemoryLTMStore()
    cs = InMemoryContextStore()
    leases = InMemoryDreamLeaseStore(default_ttl_seconds=60.0)
    queue = InProcessJobQueue()
    engine = DeterministicDreamEngine()

    registry = StaticTenantRegistry(tenants)
    provider = StaticTenantConfigProvider()

    orch = Orchestrator(
        stm_store=stm,
        ltm_store=ltm,
        context_store=cs,
        dream_lease_store=leases,
        ltm_phase_runner=engine,
        context_phase_runner=engine,
        tenant_registry=registry,
        tenant_config_provider=provider,
        job_queue=queue,
        hook_registry=HookRegistry(),
        audit_sinks=[CollectingAuditSink()],
        usage_sinks=[CollectingUsageSink()],
        dream_gates=[],
        stm_retention=StmRetentionConfig(keep_days=None, cadence_seconds=86400),
        default_lease_ttl_seconds=60.0,
        heartbeat_interval_seconds=10.0,
    )
    await orch.start(ctx=LifecycleContext(request_id="lifecycle.test.start"))
    return orch, leases


@pytest.mark.asyncio
async def test_deprovision_waits_for_active_lease() -> None:
    orch, leases = await _build_orchestrator(tenants=["acme"])
    try:
        sink = _RecordingTenantData()
        registry = StaticTenantRegistry(["acme"])
        lifecycle = StaticTenantLifecycle([sink], registry=registry)
        lifecycle.bind_active_lease_waiter(orch.wait_for_active_lease_release)

        # Simulate an in-flight lease by manually populating the orchestrator's
        # active leases dict, then schedule a release after a short delay.
        orch._active_leases["acme"] = "lease-1"

        async def release_after_delay() -> None:
            await asyncio.sleep(0.2)
            orch._active_leases.pop("acme", None)

        release_task = asyncio.create_task(release_after_delay())

        start = asyncio.get_event_loop().time()
        await lifecycle.deprovision(
            "acme", mode="hard", ctx=DeprovisionContext(request_id="t", tenant_id="acme")
        )
        elapsed = asyncio.get_event_loop().time() - start
        assert elapsed >= 0.15, "expected a real wait before sweep"
        assert sink.events == ["deprovision:acme:hard"]
        await release_task
    finally:
        await orch.stop(ctx=LifecycleContext(request_id="lifecycle.test.stop"))


@pytest.mark.asyncio
async def test_deprovision_lease_wait_times_out_and_proceeds() -> None:
    orch, _ = await _build_orchestrator(tenants=["acme"])
    try:
        sink = _RecordingTenantData()
        lifecycle = StaticTenantLifecycle(
            [sink],
            registry=StaticTenantRegistry(["acme"]),
            deprovision_lease_timeout_seconds=0.2,
        )
        lifecycle.bind_active_lease_waiter(orch.wait_for_active_lease_release)
        orch._active_leases["acme"] = "lease-stale"

        await lifecycle.deprovision(
            "acme",
            mode="soft",
            ctx=DeprovisionContext(request_id="t", tenant_id="acme"),
        )
        assert sink.events == ["deprovision:acme:soft"]
    finally:
        orch._active_leases.pop("acme", None)
        await orch.stop(ctx=LifecycleContext(request_id="lifecycle.test.stop"))


@pytest.mark.asyncio
async def test_config_provider_resolves_per_tenant_overrides() -> None:
    from dreamer.api.types import MemoryType, TenantConfig

    provider = StaticTenantConfigProvider(
        overrides={
            "acme": TenantConfig(
                memory_types=(MemoryType(name="failure", description="x"),),
                dream_instructions={"ltm_update": "tenant-specific"},
            )
        },
        global_memory_types=(
            MemoryType(name="failure", description="x"),
            MemoryType(name="observation", description="y"),
        ),
    )

    cfg = await provider.get(
        "acme", ctx=TenantConfigLookupContext(request_id="t", tenant_id="acme")
    )
    assert cfg.memory_types is not None
    assert cfg.memory_types[0].name == "failure"
    assert cfg.dream_instructions == {"ltm_update": "tenant-specific"}

    default_cfg = await provider.get(
        "default", ctx=TenantConfigLookupContext(request_id="t", tenant_id="default")
    )
    assert default_cfg.memory_types is None


@pytest.mark.asyncio
async def test_lifecycle_does_not_leak_tenant_scope() -> None:
    sink = _RecordingTenantData()
    lifecycle = StaticTenantLifecycle([sink])
    TenantScope.clear()
    await lifecycle.reset("acme", ctx=ResetContext(request_id="t", tenant_id="acme"))
    assert TenantScope.get() is None


@pytest.mark.asyncio
async def test_provision_init_config_is_passed_through_metadata() -> None:
    class _Capturer:
        multi_tenant: ClassVar[bool] = True

        def __init__(self) -> None:
            self.captured: list[dict[str, Any]] = []

        async def on_tenant_provisioned(
            self, tenant_id: TenantId, *, ctx: TenantDataContext
        ) -> None:
            self.captured.append(dict(ctx.metadata))

        async def on_tenant_deprovisioned(
            self,
            tenant_id: TenantId,
            *,
            mode: Literal["soft", "hard"],
            ctx: TenantDataContext,
        ) -> None:
            return None

        async def on_tenant_reset(
            self, tenant_id: TenantId, *, ctx: TenantDataContext
        ) -> None:
            return None

    capturer = _Capturer()
    lifecycle = StaticTenantLifecycle([capturer], registry=StaticTenantRegistry())
    await lifecycle.provision(
        "acme",
        ctx=ProvisionContext(
            request_id="t", tenant_id="acme", init_config={"plan": "pro"}
        ),
    )
    assert capturer.captured == [{"plan": "pro"}]
