"""Runtime registries and capability probing.

The runtime module is the home for in-process bookkeeping that the server uses
across the lifecycle: a Lifecycle dispatcher (start/stop), the trigger
registry + host, the hook registry, and small ``isinstance`` probes for
optional capabilities.

It does not import any contrib component and contains no business logic; the
server orchestrator and control surface use it as a structured handle on the
component graph produced by the config loader.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, ClassVar

from dreamer.api.audit import AuditSink
from dreamer.api.capabilities import Lifecycle, Middlewares, Routes, Transactional
from dreamer.api.contexts import (
    LifecycleContext,
    PublishContext,
    SubscribeContext,
    TriggerStartContext,
    TriggerStartServices,
    TriggerStopContext,
)
from dreamer.api.errors import ConfigError
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
from dreamer.api.secrets import SecretResolver
from dreamer.api.stores import ContextPendingStore, ContextReader
from dreamer.api.tenants import TenantData
from dreamer.api.triggers import Trigger
from dreamer.api.types import DreamJob
from dreamer.api.usage import UsageSink

logger = logging.getLogger(__name__)


# Maps hook config-slot name → expected Protocol.
HOOK_PROTOCOLS: dict[str, type] = {
    "pre_dream": PreDreamHook,
    "post_dream": PostDreamHook,
    "pre_ltm_update": PreLTMUpdateHook,
    "post_ltm_update": PostLTMUpdateHook,
    "pre_context_update": PreContextUpdateHook,
    "post_context_update": PostContextUpdateHook,
    "pre_memory_submit": PreMemorySubmitHook,
    "post_memory_submit": PostMemorySubmitHook,
    "on_dream_failed": DreamFailedHook,
    "on_dream_progress": DreamProgressHook,
}


@dataclass(slots=True)
class HookRegistry:
    by_slot: dict[str, list[Any]] = field(default_factory=dict)

    def add(self, slot: str, hook: Any) -> None:
        if slot not in HOOK_PROTOCOLS:
            raise ConfigError(f"unknown hook slot: {slot}")
        self.by_slot.setdefault(slot, []).append(hook)

    def get(self, slot: str) -> list[Any]:
        return list(self.by_slot.get(slot, ()))


def build_hook_registry(hooks_lists: dict[str, list[Any]]) -> HookRegistry:
    """Construct a :class:`HookRegistry` from the loader's per-slot lists.

    ``hooks_lists`` keys may be either bare slot names (``"post_dream"``) or
    namespaced (``"hooks.post_dream"``). Both are accepted to support the two
    layouts the config loader exposes.
    """
    registry = HookRegistry()
    for slot in HOOK_PROTOCOLS:
        items = hooks_lists.get(slot) or hooks_lists.get(f"hooks.{slot}") or []
        for hook in items:
            registry.add(slot, hook)
    return registry


@dataclass(slots=True)
class TriggerRegistry:
    triggers: list[Trigger] = field(default_factory=list)

    def add(self, trigger: Trigger) -> None:
        identity = (trigger.tenant_id, trigger.name)
        for existing in self.triggers:
            if (existing.tenant_id, existing.name) == identity:
                raise ConfigError(
                    f"duplicate trigger identity: ({identity[0]}, {identity[1]})"
                )
        self.triggers.append(trigger)


def build_trigger_registry(
    triggers: Iterable[Trigger],
    *,
    effective_multi_tenant: bool = True,
) -> TriggerRegistry:
    """Build the trigger registry, enforcing identity uniqueness.

    ``effective_multi_tenant`` is the deployment mode. When False, any trigger
    whose class is ``MultiTenantTrigger`` (identified by module path + name) is
    rejected with ``ConfigError`` — fanout triggers are meaningless in a
    single-tenant deployment and would suggest a configuration mistake.
    """
    registry = TriggerRegistry()
    for trigger in triggers:
        if not effective_multi_tenant and _is_multitenant_trigger(trigger):
            raise ConfigError(
                f"MultiTenantTrigger configured in a single-tenant deployment: "
                f"trigger {trigger.name!r}"
            )
        registry.add(trigger)
    return registry


def _is_multitenant_trigger(trigger: object) -> bool:
    cls = type(trigger)
    return (
        cls.__module__ == "dreamer.contrib.triggers.multitenant"
        and cls.__name__ == "MultiTenantTrigger"
    )


@dataclass(slots=True)
class TriggerHost:
    """Drives the lifecycle of every configured ``Trigger``.

    On ``start_all``, calls each trigger's ``Trigger.start(ctx, services)`` with
    a per-trigger ``services.fire`` bound to the trigger's identity. The
    callback publishes a ``DreamJob`` to the configured ``JobQueue`` so the
    orchestrator picks the job up and runs the dream lifecycle.

    Triggers are NOT registered with :class:`LifecycleDispatcher` — their
    ``start``/``stop`` signature requires a ``services`` argument that the
    lifecycle dispatcher does not supply.
    """

    triggers: list[Trigger] = field(default_factory=list)
    job_queue: JobQueue | None = None
    secret_resolver: SecretResolver | None = None
    usage_sinks: list[UsageSink] = field(default_factory=list)
    audit_sinks: list[AuditSink] = field(default_factory=list)
    progress_hooks: list[DreamProgressHook] = field(default_factory=list)
    _started: list[Trigger] = field(default_factory=list)

    async def start_all(self) -> None:
        if self.job_queue is None:
            raise ConfigError("TriggerHost.start_all: job_queue is unset")
        if self.secret_resolver is None:
            raise ConfigError("TriggerHost.start_all: secret_resolver is unset")
        for trigger in self.triggers:
            services = self._build_services(trigger)
            ctx = TriggerStartContext(
                request_id=f"trigger.start.{trigger.tenant_id}.{trigger.name}",
                tenant_id=trigger.tenant_id,
                trigger_name=trigger.name,
            )
            try:
                await trigger.start(ctx=ctx, services=services)
            except Exception:
                logger.exception(
                    "TriggerHost: failed to start trigger tenant=%s name=%s",
                    trigger.tenant_id,
                    trigger.name,
                )
                continue
            self._started.append(trigger)

    async def stop_all(self) -> None:
        for trigger in reversed(self._started):
            ctx = TriggerStopContext(
                request_id=f"trigger.stop.{trigger.tenant_id}.{trigger.name}",
                tenant_id=trigger.tenant_id,
                trigger_name=trigger.name,
            )
            try:
                await trigger.stop(ctx=ctx)
            except Exception:
                logger.exception(
                    "TriggerHost: stop raised for trigger tenant=%s name=%s",
                    trigger.tenant_id,
                    trigger.name,
                )
        self._started.clear()

    def _build_services(self, trigger: Trigger) -> TriggerStartServices:
        assert self.job_queue is not None
        assert self.secret_resolver is not None
        job_queue = self.job_queue
        tenant_id = trigger.tenant_id
        trigger_name = trigger.name

        async def _fire() -> None:
            await job_queue.publish(
                DreamJob(tenant_id=tenant_id, trigger_name=trigger_name),
                ctx=PublishContext(
                    request_id=f"trigger.fire.{tenant_id}.{trigger_name}",
                    tenant_id=tenant_id,
                ),
            )

        progress_hooks = list(self.progress_hooks)

        async def _emit_progress(message: str, payload: Mapping[str, Any]) -> None:
            if not progress_hooks:
                return
            from dreamer.api.contexts import (
                DreamProgressContext,
                DreamProgressServices,
            )

            ctx = DreamProgressContext(
                request_id=f"trigger.progress.{tenant_id}.{trigger_name}",
                tenant_id=tenant_id,
                lease_id="",
                phase="trigger",
                message=message,
                payload=dict(payload),
            )
            services = DreamProgressServices(
                emit_progress=_emit_progress,
                secrets=self.secret_resolver,  # type: ignore[arg-type]
                usage=_NoOpUsageSink(),
                audit=_NoOpAuditSink(),
                clock=_default_clock,
            )
            for hook in progress_hooks:
                try:
                    await hook.on_dream_progress(ctx=ctx, services=services)
                except Exception:
                    logger.exception(
                        "TriggerHost: progress hook raised for tenant=%s name=%s",
                        tenant_id,
                        trigger_name,
                    )

        return TriggerStartServices(
            emit_progress=_emit_progress,
            secrets=self.secret_resolver,
            usage=_FanoutUsage(self.usage_sinks),
            audit=_FanoutAudit(self.audit_sinks),
            clock=_default_clock,
            fire=_fire,
        )


def _default_clock() -> datetime:
    return datetime.now(UTC)


class _NoOpUsageSink:
    multi_tenant: ClassVar[bool] = True

    async def record(self, event: Any, *, ctx: Any) -> None:
        return None


class _NoOpAuditSink:
    multi_tenant: ClassVar[bool] = True

    async def record(self, event: Any, *, ctx: Any) -> None:
        return None


class _FanoutUsage:
    multi_tenant: ClassVar[bool] = True

    def __init__(self, sinks: list[UsageSink]) -> None:
        self.sinks = list(sinks)

    async def record(self, event: Any, *, ctx: Any) -> None:
        for sink in self.sinks:
            try:
                await sink.record(event, ctx=ctx)
            except Exception:
                logger.exception("UsageSink.record raised; continuing")


class _FanoutAudit:
    multi_tenant: ClassVar[bool] = True

    def __init__(self, sinks: list[AuditSink]) -> None:
        self.sinks = list(sinks)

    async def record(self, event: Any, *, ctx: Any) -> None:
        for sink in self.sinks:
            try:
                await sink.record(event, ctx=ctx)
            except Exception:
                logger.exception("AuditSink.record raised; continuing")


def build_trigger_host(
    *,
    triggers: Iterable[Trigger],
    job_queue: JobQueue,
    secret_resolver: SecretResolver,
    usage_sinks: Iterable[UsageSink] | None = None,
    audit_sinks: Iterable[AuditSink] | None = None,
    progress_hooks: Iterable[DreamProgressHook] | None = None,
) -> TriggerHost:
    return TriggerHost(
        triggers=list(triggers),
        job_queue=job_queue,
        secret_resolver=secret_resolver,
        usage_sinks=list(usage_sinks or []),
        audit_sinks=list(audit_sinks or []),
        progress_hooks=list(progress_hooks or []),
    )


async def subscribe_orchestrator(
    *,
    job_queue: JobQueue,
    handler: Callable[[DreamJob], Awaitable[None]],
    request_id: str = "orchestrator.subscribe",
) -> None:
    await job_queue.subscribe(
        handler=handler,
        ctx=SubscribeContext(request_id=request_id),
    )


@dataclass(slots=True)
class LifecycleDispatcher:
    """Tracks every component implementing :class:`Lifecycle` and dispatches
    ``start``/``stop`` in registration order. ``stop`` runs in reverse.

    ``Trigger`` components are intentionally **not** registered here. They
    share the ``start``/``stop`` method names with ``Lifecycle`` but
    ``Trigger.start`` takes an additional required ``services`` argument that
    the lifecycle dispatcher does not supply. Triggers are driven by the
    dedicated trigger host (see :func:`run_triggers`)."""

    components: list[Lifecycle] = field(default_factory=list)

    def register(self, candidate: Any) -> None:
        # Triggers carry their own start/stop with a required `services`
        # parameter; they are driven by the trigger host, not this dispatcher.
        if isinstance(candidate, Trigger):
            return
        if isinstance(candidate, Lifecycle):
            if candidate not in self.components:
                self.components.append(candidate)

    async def start_all(self, *, request_id: str = "lifecycle.start") -> None:
        ctx = LifecycleContext(request_id=request_id)
        for c in self.components:
            await c.start(ctx=ctx)

    async def stop_all(self, *, request_id: str = "lifecycle.stop") -> None:
        ctx = LifecycleContext(request_id=request_id)
        for c in reversed(self.components):
            try:
                await c.stop(ctx=ctx)
            except Exception:  # noqa: BLE001 — stop must keep going
                logger.exception("Lifecycle.stop raised for %r; continuing", c)


def build_lifecycle_dispatcher(*, components: Iterable[Any]) -> LifecycleDispatcher:
    dispatcher = LifecycleDispatcher()
    for c in components:
        if c is None:
            continue
        dispatcher.register(c)
    return dispatcher


def has_context_pending(component: object) -> bool:
    return isinstance(component, ContextPendingStore)


def has_context_reader(component: object) -> bool:
    return isinstance(component, ContextReader)


def has_transactional(component: object) -> bool:
    return isinstance(component, Transactional)


def has_routes(component: object) -> bool:
    return isinstance(component, Routes)


def has_middlewares(component: object) -> bool:
    return isinstance(component, Middlewares)


def has_tenant_data(component: object) -> bool:
    return isinstance(component, TenantData)


__all__ = [
    "HOOK_PROTOCOLS",
    "HookRegistry",
    "LifecycleDispatcher",
    "TriggerHost",
    "TriggerRegistry",
    "build_hook_registry",
    "build_lifecycle_dispatcher",
    "build_trigger_host",
    "build_trigger_registry",
    "has_context_pending",
    "has_context_reader",
    "has_middlewares",
    "has_routes",
    "has_tenant_data",
    "has_transactional",
    "subscribe_orchestrator",
]
