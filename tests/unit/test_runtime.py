from __future__ import annotations

from typing import ClassVar

import pytest

from dreamer.api.contexts import (
    LifecycleContext,
    PostDreamContext,
    PostDreamServices,
    PreDreamContext,
    PreDreamServices,
    TriggerStartContext,
    TriggerStartServices,
    TriggerStopContext,
)
from dreamer.api.errors import ConfigError
from dreamer.server.runtime import (
    HookRegistry,
    LifecycleDispatcher,
    build_hook_registry,
    build_lifecycle_dispatcher,
    build_trigger_registry,
)


class FakeLifecycleComponent:
    multi_tenant: ClassVar[bool] = False

    def __init__(self) -> None:
        self.started = 0
        self.stopped = 0

    async def start(self, *, ctx: LifecycleContext) -> None:
        self.started += 1

    async def stop(self, *, ctx: LifecycleContext) -> None:
        self.stopped += 1


class FakeNonLifecycleComponent:
    pass

class FakeStoppingError:
    multi_tenant: ClassVar[bool] = False

    async def start(self, *, ctx: LifecycleContext) -> None:
        return None

    async def stop(self, *, ctx: LifecycleContext) -> None:
        raise RuntimeError("stop blew up")


class FakePreDreamHook:
    multi_tenant: ClassVar[bool] = False

    async def on_pre_dream(
        self, *, ctx: PreDreamContext, services: PreDreamServices
    ) -> None:
        return None


class FakePostDreamHook:
    multi_tenant: ClassVar[bool] = False

    async def on_post_dream(
        self, *, ctx: PostDreamContext, services: PostDreamServices
    ) -> None:
        return None


class FakeTrigger:
    multi_tenant: ClassVar[bool] = False

    def __init__(self, name: str, tenant_id: str = "default") -> None:
        self.name = name
        self.tenant_id = tenant_id

    async def start(
        self, *, ctx: TriggerStartContext, services: TriggerStartServices
    ) -> None:
        return None

    async def stop(self, *, ctx: TriggerStopContext) -> None:
        return None


def test_hook_registry_groups_by_slot() -> None:
    pre = FakePreDreamHook()
    post = FakePostDreamHook()
    registry = build_hook_registry({
        "pre_dream": [pre],
        "post_dream": [post],
    })
    assert registry.get("pre_dream") == [pre]
    assert registry.get("post_dream") == [post]
    assert registry.get("pre_ltm_update") == []


def test_hook_registry_accepts_namespaced_keys() -> None:
    pre = FakePreDreamHook()
    registry = build_hook_registry({"hooks.pre_dream": [pre]})
    assert registry.get("pre_dream") == [pre]


def test_hook_registry_rejects_unknown_slot() -> None:
    registry = HookRegistry()
    with pytest.raises(ConfigError, match="unknown hook slot"):
        registry.add("not_a_hook", FakePreDreamHook())


def test_trigger_registry_adds_unique_triggers() -> None:
    a = FakeTrigger("a")
    b = FakeTrigger("b")
    registry = build_trigger_registry([a, b])
    assert registry.triggers == [a, b]


def test_trigger_registry_rejects_duplicate_identity() -> None:
    a = FakeTrigger("alpha", "default")
    b = FakeTrigger("alpha", "default")
    with pytest.raises(ConfigError, match="duplicate trigger identity"):
        build_trigger_registry([a, b])


def test_trigger_registry_allows_same_name_different_tenants() -> None:
    a = FakeTrigger("alpha", "tenant_a")
    b = FakeTrigger("alpha", "tenant_b")
    registry = build_trigger_registry([a, b])
    assert len(registry.triggers) == 2


@pytest.mark.asyncio
async def test_lifecycle_dispatcher_starts_in_order_stops_in_reverse() -> None:
    a = FakeLifecycleComponent()
    b = FakeLifecycleComponent()
    c = FakeNonLifecycleComponent()
    dispatcher = build_lifecycle_dispatcher(components=[a, b, c, None])
    await dispatcher.start_all()
    assert a.started == 1 and b.started == 1
    await dispatcher.stop_all()
    assert a.stopped == 1 and b.stopped == 1


@pytest.mark.asyncio
async def test_lifecycle_stop_swallows_exceptions() -> None:
    a = FakeStoppingError()
    b = FakeLifecycleComponent()
    dispatcher = LifecycleDispatcher()
    dispatcher.register(a)
    dispatcher.register(b)
    await dispatcher.start_all()
    await dispatcher.stop_all()
    assert b.stopped == 1


def test_lifecycle_dispatcher_no_double_register() -> None:
    a = FakeLifecycleComponent()
    dispatcher = LifecycleDispatcher()
    dispatcher.register(a)
    dispatcher.register(a)
    assert dispatcher.components == [a]


def test_capability_probes_match_isinstance_protocol() -> None:
    from dreamer.server.runtime import has_routes
    assert has_routes(FakeLifecycleComponent()) is False
