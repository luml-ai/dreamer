from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any, ClassVar

import pytest
import pytest_asyncio

from dreamer.api.compat import implements
from dreamer.api.contexts import (
    PublishContext,
    SubmitContext,
    SubscribeContext,
    TenantRegistryContext,
)
from dreamer.api.errors import ConfigError
from dreamer.api.jobs import JobQueue
from dreamer.api.tenants import TenantRegistry, TenantScope
from dreamer.api.triggers import Trigger
from dreamer.api.types import DreamJob, Memory, TenantId
from dreamer.contrib.triggers.cron import CronTrigger
from dreamer.contrib.triggers.external import ExternalTrigger
from dreamer.contrib.triggers.multitenant import MultiTenantTrigger
from dreamer.contrib.triggers.threshold import STMCountThresholdTrigger
from dreamer.server.runtime import (
    TriggerHost,
    build_trigger_host,
    build_trigger_registry,
    subscribe_orchestrator,
)
from dreamer.testing.fakes import (
    CollectingAuditSink,
    CollectingUsageSink,
    InMemorySTMStore,
)


@implements(TenantRegistry, version=1)
class FakeTenantRegistry:
    multi_tenant: ClassVar[bool] = True

    def __init__(self, tenants: list[TenantId] | None = None) -> None:
        self.tenants: list[TenantId] = list(tenants or ["default"])

    def set_tenants(self, tenants: list[TenantId]) -> None:
        self.tenants = list(tenants)

    async def list_tenants(self, *, ctx: TenantRegistryContext) -> list[TenantId]:
        return list(self.tenants)

    async def exists(self, tenant_id: TenantId, *, ctx: TenantRegistryContext) -> bool:
        return tenant_id in self.tenants


class FakeSecretResolver:
    multi_tenant: ClassVar[bool] = True

    async def get(self, name: str, *, tenant_id: TenantId | None, ctx: Any) -> Any:
        from dreamer.api.types import SecretValue

        return SecretValue(value=f"<{name}>")


@implements(JobQueue, version=1)
class CapturingJobQueue:
    multi_tenant: ClassVar[bool] = True

    def __init__(self) -> None:
        self.published: list[DreamJob] = []
        self._handler: Any = None
        self._event = asyncio.Event()

    async def publish(self, job: DreamJob, *, ctx: PublishContext) -> None:
        self.published.append(job)
        self._event.set()
        if self._handler is not None:
            await self._handler(job)

    async def subscribe(self, *, handler: Any, ctx: SubscribeContext) -> None:
        self._handler = handler

    async def wait_published(self, *, count: int, deadline_seconds: float = 2.0) -> None:
        async with asyncio.timeout(deadline_seconds):
            while len(self.published) < count:
                self._event.clear()
                if len(self.published) >= count:
                    return
                await self._event.wait()


@pytest_asyncio.fixture
async def queue() -> AsyncIterator[CapturingJobQueue]:
    yield CapturingJobQueue()


def _make_host(
    *,
    triggers: list[Trigger],
    queue: CapturingJobQueue,
) -> TriggerHost:
    return build_trigger_host(
        triggers=triggers,
        job_queue=queue,
        secret_resolver=FakeSecretResolver(),
        usage_sinks=[CollectingUsageSink()],
        audit_sinks=[CollectingAuditSink()],
        progress_hooks=[],
    )


@pytest.mark.asyncio
async def test_cron_trigger_fires_on_schedule(queue: CapturingJobQueue) -> None:
    """Verifies the wired ``services.fire`` produces a correctly-shaped
    ``DreamJob`` without waiting for a real cron boundary (APScheduler owns
    the schedule)."""
    trigger = CronTrigger(name="every_6h", expression="0 */6 * * *")
    host = _make_host(triggers=[trigger], queue=queue)

    services = host._build_services(trigger)
    await services.fire()

    assert queue.published == [
        DreamJob(tenant_id="default", trigger_name="every_6h", payload={}),
    ]


@pytest.mark.asyncio
async def test_cron_trigger_starts_and_stops_cleanly(queue: CapturingJobQueue) -> None:
    trigger = CronTrigger(name="every_minute", expression="* * * * *")
    host = _make_host(triggers=[trigger], queue=queue)
    await host.start_all()
    try:
        assert trigger._scheduler is not None
    finally:
        await host.stop_all()
    assert trigger._scheduler is None


def test_cron_trigger_invalid_expression_rejected_at_construction() -> None:
    with pytest.raises(ConfigError, match="invalid cron expression"):
        CronTrigger(name="bad", expression="not-a-cron-expression")


def test_cron_trigger_empty_expression_rejected() -> None:
    with pytest.raises(ConfigError):
        CronTrigger(name="bad", expression="")


def test_cron_trigger_empty_name_rejected() -> None:
    with pytest.raises(ConfigError):
        CronTrigger(name="", expression="* * * * *")


async def _submit_n(stm: InMemorySTMStore, tenant_id: TenantId, n: int) -> None:
    from datetime import UTC, datetime

    with TenantScope.set(tenant_id):
        for i in range(n):
            await stm.submit(
                Memory(
                    tenant_id=tenant_id,
                    agent_id="a",
                    type="observation",
                    title=f"m{i}",
                    content="x",
                    submitted_at=datetime.now(UTC),
                ),
                ctx=SubmitContext(request_id=f"r{i}", tenant_id=tenant_id),
            )


@pytest.mark.asyncio
async def test_threshold_trigger_fires_once_at_threshold(
    queue: CapturingJobQueue,
) -> None:
    stm = InMemorySTMStore()
    trigger = STMCountThresholdTrigger(
        name="bigbatch",
        threshold=3,
        interval_seconds=0.05,
        stm_store=stm,
    )
    host = _make_host(triggers=[trigger], queue=queue)
    await host.start_all()
    try:
        await _submit_n(stm, "default", 2)
        await asyncio.sleep(0.15)
        assert queue.published == []

        await _submit_n(stm, "default", 1)
        await queue.wait_published(count=1, deadline_seconds=2.0)
        assert len(queue.published) == 1
        assert queue.published[0].trigger_name == "bigbatch"

        # Adding more without dropping below threshold must not re-fire.
        await _submit_n(stm, "default", 5)
        await asyncio.sleep(0.2)
        assert len(queue.published) == 1
    finally:
        await host.stop_all()


@pytest.mark.asyncio
async def test_threshold_trigger_re_fires_after_drop_and_recross(
    queue: CapturingJobQueue,
) -> None:
    from datetime import UTC, datetime

    from dreamer.api.contexts import (
        ClaimContext,
        MarkConsumedContext,
    )

    stm = InMemorySTMStore()
    trigger = STMCountThresholdTrigger(
        name="bigbatch",
        threshold=2,
        interval_seconds=0.05,
        stm_store=stm,
    )
    host = _make_host(triggers=[trigger], queue=queue)
    await host.start_all()
    try:
        await _submit_n(stm, "default", 2)
        await queue.wait_published(count=1, deadline_seconds=2.0)
        assert len(queue.published) == 1

        with TenantScope.set("default"):
            batch = await stm.claim_batch(
                ctx=ClaimContext(
                    request_id="claim", tenant_id="default", lease_id="lease-1"
                )
            )
            await stm.mark_consumed(
                ctx=MarkConsumedContext(
                    request_id="mc",
                    tenant_id="default",
                    lease_id="lease-1",
                    memory_ids=tuple(m.id for m in batch.memories if m.id),
                    consumed_at=datetime.now(UTC),
                )
            )
        await asyncio.sleep(0.2)  # allow trigger to observe drop

        await _submit_n(stm, "default", 2)
        await queue.wait_published(count=2, deadline_seconds=2.0)
        assert len(queue.published) == 2
    finally:
        await host.stop_all()


def test_threshold_trigger_invalid_args() -> None:
    stm = InMemorySTMStore()
    with pytest.raises(ConfigError):
        STMCountThresholdTrigger(
            name="", threshold=1, interval_seconds=1.0, stm_store=stm
        )
    with pytest.raises(ConfigError):
        STMCountThresholdTrigger(
            name="ok", threshold=0, interval_seconds=1.0, stm_store=stm
        )
    with pytest.raises(ConfigError):
        STMCountThresholdTrigger(
            name="ok", threshold=1, interval_seconds=0, stm_store=stm
        )


@pytest.mark.asyncio
async def test_external_trigger_does_not_self_fire(queue: CapturingJobQueue) -> None:
    trigger = ExternalTrigger(name="external")
    host = _make_host(triggers=[trigger], queue=queue)
    await host.start_all()
    try:
        await asyncio.sleep(0.1)
        assert queue.published == []
    finally:
        await host.stop_all()


@pytest.mark.asyncio
async def test_external_trigger_fires_via_host_services(
    queue: CapturingJobQueue,
) -> None:
    trigger = ExternalTrigger(name="external", tenant_id="acme")
    host = _make_host(triggers=[trigger], queue=queue)
    services = host._build_services(trigger)
    await services.fire()
    assert queue.published == [
        DreamJob(tenant_id="acme", trigger_name="external", payload={}),
    ]


def test_duplicate_top_level_trigger_identity_rejected() -> None:
    a = ExternalTrigger(name="external", tenant_id="default")
    b = ExternalTrigger(name="external", tenant_id="default")
    with pytest.raises(ConfigError, match="duplicate trigger identity"):
        build_trigger_registry([a, b])


def test_same_name_different_tenant_allowed() -> None:
    a = ExternalTrigger(name="ext", tenant_id="acme")
    b = ExternalTrigger(name="ext", tenant_id="beta")
    registry = build_trigger_registry([a, b])
    assert len(registry.triggers) == 2


def test_multitenant_trigger_rejected_in_single_tenant_mode(
    queue: CapturingJobQueue,
) -> None:
    registry = FakeTenantRegistry(tenants=["acme", "beta"])
    trigger = MultiTenantTrigger(
        name="fanout",
        tenant_registry=registry,
        job_queue=queue,
        inner={
            "template_class": "dreamer.contrib.triggers.external.ExternalTrigger",
            "template_params": {"name": "ext"},
        },
        refresh_interval_seconds=0.05,
    )
    with pytest.raises(ConfigError, match="single-tenant"):
        build_trigger_registry([trigger], effective_multi_tenant=False)


@pytest.mark.asyncio
async def test_multitenant_trigger_fans_out_per_tenant(
    queue: CapturingJobQueue,
) -> None:
    registry = FakeTenantRegistry(tenants=["acme", "beta"])
    trigger = MultiTenantTrigger(
        name="fanout",
        tenant_registry=registry,
        job_queue=queue,
        inner={
            "template_class": "dreamer.contrib.triggers.external.ExternalTrigger",
            "template_params": {"name": "inner"},
        },
        refresh_interval_seconds=0.05,
    )
    host = _make_host(triggers=[trigger], queue=queue)
    await host.start_all()
    try:
        await asyncio.sleep(0.2)
        assert set(trigger._materialized.keys()) == {"acme", "beta"}
        # ExternalTrigger has no internal scheduler — fire each inner manually.
        for tenant_id, inner in trigger._materialized.items():
            fire = trigger._inner_fire(tenant_id, inner.name)
            await fire()
        await queue.wait_published(count=2, deadline_seconds=2.0)
        observed = {(j.tenant_id, j.trigger_name) for j in queue.published}
        assert observed == {("acme", "inner"), ("beta", "inner")}
    finally:
        await host.stop_all()


@pytest.mark.asyncio
async def test_multitenant_trigger_picks_up_new_tenant_between_ticks(
    queue: CapturingJobQueue,
) -> None:
    registry = FakeTenantRegistry(tenants=["acme"])
    trigger = MultiTenantTrigger(
        name="fanout",
        tenant_registry=registry,
        job_queue=queue,
        inner={
            "template_class": "dreamer.contrib.triggers.external.ExternalTrigger",
            "template_params": {"name": "inner"},
        },
        refresh_interval_seconds=0.05,
    )
    host = _make_host(triggers=[trigger], queue=queue)
    await host.start_all()
    try:
        await asyncio.sleep(0.2)
        assert set(trigger._materialized.keys()) == {"acme"}
        registry.set_tenants(["acme", "beta"])
        for _ in range(40):
            if "beta" in trigger._materialized:
                break
            await asyncio.sleep(0.05)
        assert set(trigger._materialized.keys()) == {"acme", "beta"}
    finally:
        await host.stop_all()


@pytest.mark.asyncio
async def test_multitenant_trigger_drops_inner_for_disappeared_tenant(
    queue: CapturingJobQueue,
) -> None:
    registry = FakeTenantRegistry(tenants=["acme", "beta"])
    trigger = MultiTenantTrigger(
        name="fanout",
        tenant_registry=registry,
        job_queue=queue,
        inner={
            "template_class": "dreamer.contrib.triggers.external.ExternalTrigger",
            "template_params": {"name": "inner"},
        },
        refresh_interval_seconds=0.05,
    )
    host = _make_host(triggers=[trigger], queue=queue)
    await host.start_all()
    try:
        await asyncio.sleep(0.2)
        assert set(trigger._materialized.keys()) == {"acme", "beta"}
        registry.set_tenants(["acme"])
        for _ in range(40):
            if "beta" not in trigger._materialized:
                break
            await asyncio.sleep(0.05)
        assert set(trigger._materialized.keys()) == {"acme"}
    finally:
        await host.stop_all()


@pytest.mark.asyncio
async def test_deprovisioned_tenant_orchestrator_skips() -> None:
    """Trigger machinery doesn't enforce the recheck — that's the
    orchestrator's job. The stub handler exercises the same recheck path."""
    registry = FakeTenantRegistry(tenants=["acme"])
    queue = CapturingJobQueue()
    handled: list[DreamJob] = []
    skipped: list[DreamJob] = []

    async def handler(job: DreamJob) -> None:
        if not await registry.exists(
            job.tenant_id, ctx=TenantRegistryContext(request_id="r")
        ):
            skipped.append(job)
            return
        handled.append(job)

    await subscribe_orchestrator(job_queue=queue, handler=handler)

    await queue.publish(
        DreamJob(tenant_id="acme", trigger_name="t"),
        ctx=PublishContext(request_id="r1", tenant_id="acme"),
    )
    registry.set_tenants([])
    await queue.publish(
        DreamJob(tenant_id="acme", trigger_name="t"),
        ctx=PublishContext(request_id="r2", tenant_id="acme"),
    )

    assert len(handled) == 1
    assert len(skipped) == 1


def test_multitenant_trigger_rejects_invalid_inner_class(
    queue: CapturingJobQueue,
) -> None:
    registry = FakeTenantRegistry()
    with pytest.raises(ConfigError, match="not declare @implements"):
        MultiTenantTrigger(
            name="fanout",
            tenant_registry=registry,
            job_queue=queue,
            inner={
                "template_class": "dreamer.testing.fakes.InMemorySTMStore",
                "template_params": {},
            },
        )


def test_multitenant_trigger_rejects_unknown_inner_module(
    queue: CapturingJobQueue,
) -> None:
    registry = FakeTenantRegistry()
    with pytest.raises(ConfigError, match="could not import"):
        MultiTenantTrigger(
            name="fanout",
            tenant_registry=registry,
            job_queue=queue,
            inner={
                "template_class": "no.such.module.Trigger",
                "template_params": {},
            },
        )


def test_multitenant_trigger_rejects_inner_with_unknown_params(
    queue: CapturingJobQueue,
) -> None:
    registry = FakeTenantRegistry()
    with pytest.raises(ConfigError, match="does not accept params"):
        MultiTenantTrigger(
            name="fanout",
            tenant_registry=registry,
            job_queue=queue,
            inner={
                "template_class": "dreamer.contrib.triggers.external.ExternalTrigger",
                "template_params": {"unknown": "value"},
            },
        )


def test_multitenant_trigger_rejects_inner_declaring_tenant_id(
    queue: CapturingJobQueue,
) -> None:
    registry = FakeTenantRegistry()
    with pytest.raises(ConfigError, match="must not declare 'tenant_id'"):
        MultiTenantTrigger(
            name="fanout",
            tenant_registry=registry,
            job_queue=queue,
            inner={
                "template_class": "dreamer.contrib.triggers.external.ExternalTrigger",
                "template_params": {"tenant_id": "acme"},
            },
        )


def test_multitenant_trigger_rejects_invalid_inner_cron_expression(
    queue: CapturingJobQueue,
) -> None:
    """Inner CronTrigger validates the expression at materialization, which
    happens on first refresh tick — ConfigError must surface from the wrapper."""
    registry = FakeTenantRegistry(tenants=["acme"])
    queue_ = CapturingJobQueue()
    trigger = MultiTenantTrigger(
        name="fanout",
        tenant_registry=registry,
        job_queue=queue_,
        inner={
            "template_class": "dreamer.contrib.triggers.cron.CronTrigger",
            "template_params": {"name": "every", "expression": "***INVALID***"},
        },
        refresh_interval_seconds=0.05,
    )
    with pytest.raises(ConfigError, match="invalid cron expression"):
        trigger._materialize("acme")


def test_multitenant_trigger_rejects_inner_missing_tenant_id_param(
    queue: CapturingJobQueue,
) -> None:
    registry = FakeTenantRegistry()

    @implements(Trigger, version=1)
    class _NoTenantIdTrigger:
        multi_tenant: ClassVar[bool] = True
        name = "x"
        tenant_id = "default"

        def __init__(self, *, name: str) -> None:  # noqa: ARG002 — purposely ignore
            self.name = name

        async def start(self, *, ctx: Any, services: Any) -> None: ...
        async def stop(self, *, ctx: Any) -> None: ...

    # Make the locally-defined class importable by fqn.
    import sys

    module = sys.modules[__name__]
    module._NoTenantIdTrigger = _NoTenantIdTrigger  # type: ignore[attr-defined]
    fqn = f"{__name__}._NoTenantIdTrigger"
    with pytest.raises(ConfigError, match="must accept a 'tenant_id'"):
        MultiTenantTrigger(
            name="fanout",
            tenant_registry=registry,
            job_queue=queue,
            inner={"template_class": fqn, "template_params": {"name": "ok"}},
        )


def test_multitenant_trigger_rejects_unknown_inner_keys(
    queue: CapturingJobQueue,
) -> None:
    registry = FakeTenantRegistry()
    with pytest.raises(ConfigError, match="unexpected keys"):
        MultiTenantTrigger(
            name="fanout",
            tenant_registry=registry,
            job_queue=queue,
            inner={
                "template_class": "dreamer.contrib.triggers.external.ExternalTrigger",
                "template_params": {"name": "inner"},
                "extra": "junk",
            },
        )


def test_multitenant_trigger_rejects_non_mapping_inner(
    queue: CapturingJobQueue,
) -> None:
    registry = FakeTenantRegistry()
    with pytest.raises(ConfigError, match="must be a mapping"):
        MultiTenantTrigger(
            name="fanout",
            tenant_registry=registry,
            job_queue=queue,
            inner="not a mapping",  # type: ignore[arg-type]
        )


def test_multitenant_trigger_rejects_empty_template_class(
    queue: CapturingJobQueue,
) -> None:
    registry = FakeTenantRegistry()
    with pytest.raises(ConfigError, match="template_class"):
        MultiTenantTrigger(
            name="fanout",
            tenant_registry=registry,
            job_queue=queue,
            inner={"template_class": "", "template_params": {}},
        )
