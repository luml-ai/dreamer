"""MultiTenantTrigger: per-tenant fanout wrapper.

Wraps an inner trigger template and materializes one inner trigger per tenant
each time the registry's tenant set is observed. Inner trigger configurations
omit ``tenant_id`` — the wrapper injects it at materialization time.

Inner-trigger templates are validated at construction time *structurally*:
the wrapper imports the declared class, asserts it implements ``Trigger@1``,
and asserts the param map is well-typed. Instantiation is deferred to
materialization (each fire), when the wrapper injects ``tenant_id``.

YAML form::

    triggers:
      - class: dreamer.contrib.triggers.multitenant.MultiTenantTrigger
        params:
          name: every_6h_fanout
          tenant_registry: { ref: tenant_registry }
          job_queue: { ref: job_queue }
          inner:
            template_class: dreamer.contrib.triggers.cron.CronTrigger
            template_params:
              name: every_6h
              expression: "0 */6 * * *"

The two-tier ``template_class`` / ``template_params`` shape is required so
that the YAML loader (which auto-instantiates ``{class, params}`` blocks) does
not eagerly construct the inner.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import logging
from collections.abc import Mapping
from typing import Any, ClassVar

from dreamer.api.compat import implements
from dreamer.api.contexts import (
    PublishContext,
    TenantRegistryContext,
    TriggerStartContext,
    TriggerStartServices,
    TriggerStopContext,
)
from dreamer.api.errors import ConfigError
from dreamer.api.jobs import JobQueue
from dreamer.api.tenants import TenantRegistry
from dreamer.api.triggers import Trigger
from dreamer.api.types import DreamJob, TenantId

logger = logging.getLogger(__name__)


@implements(Trigger, version=1)
class MultiTenantTrigger:
    """Wraps an inner trigger template; fans out per tenant on each tick.

    The wrapper itself is a single ``Trigger`` instance with its own composite
    identity ``(tenant_id="*", name)``. On ``start``, it begins a polling task
    that observes ``tenant_registry.list_tenants()`` and materializes inner
    triggers per tenant. Inner trigger uniqueness is enforced only within a
    single tick's materialization (one inner per tenant).

    Inner triggers fire by publishing ``DreamJob`` directly to the configured
    ``JobQueue`` with the tenant's identity — so each per-tenant fire produces
    a job with ``(tenant_id=<tenant>, trigger_name=<inner.name>)``.
    """

    multi_tenant: ClassVar[bool] = True

    # The fanout wrapper covers all tenants; its own `tenant_id` is the wildcard
    # marker so the top-level trigger registry sees a single, distinct identity
    # per wrapper.
    WILDCARD_TENANT: ClassVar[TenantId] = "*"

    def __init__(
        self,
        *,
        name: str,
        tenant_registry: TenantRegistry,
        job_queue: JobQueue,
        inner: Mapping[str, Any],
        refresh_interval_seconds: float = 5.0,
    ) -> None:
        if not name:
            raise ConfigError("MultiTenantTrigger: name must be a non-empty string")
        if refresh_interval_seconds <= 0:
            raise ConfigError(
                "MultiTenantTrigger: refresh_interval_seconds must be > 0"
            )
        self.name = name
        self.tenant_id = self.WILDCARD_TENANT
        self.refresh_interval_seconds = refresh_interval_seconds
        self._tenant_registry = tenant_registry
        self._job_queue = job_queue
        self._inner_class, self._inner_params = self._parse_inner_template(inner)
        self._validate_inner_template()
        self._task: asyncio.Task[None] | None = None
        self._stop_event: asyncio.Event | None = None
        self._materialized: dict[TenantId, Trigger] = {}
        self._outer_services: TriggerStartServices | None = None

    @staticmethod
    def _parse_inner_template(
        inner: Mapping[str, Any],
    ) -> tuple[type, dict[str, Any]]:
        if not isinstance(inner, Mapping):
            raise ConfigError(
                "MultiTenantTrigger: 'inner' must be a mapping with "
                "'template_class' and optional 'template_params'"
            )
        template_class = inner.get("template_class")
        if not isinstance(template_class, str) or not template_class:
            raise ConfigError(
                "MultiTenantTrigger: inner.template_class must be a non-empty "
                "fully-qualified class name (got "
                f"{template_class!r})"
            )
        if "." not in template_class:
            raise ConfigError(
                "MultiTenantTrigger: inner.template_class must be a "
                f"fully-qualified class name (got {template_class!r})"
            )
        raw_params = inner.get("template_params") or {}
        if not isinstance(raw_params, Mapping):
            raise ConfigError(
                "MultiTenantTrigger: inner.template_params must be a mapping"
            )
        if "tenant_id" in raw_params:
            raise ConfigError(
                "MultiTenantTrigger: inner.template_params must not declare "
                "'tenant_id' — the wrapper injects it per-fire"
            )
        extras = set(inner.keys()) - {"template_class", "template_params"}
        if extras:
            raise ConfigError(
                f"MultiTenantTrigger: inner has unexpected keys: {sorted(extras)}; "
                "allowed keys are 'template_class' and 'template_params'"
            )
        module_path, _, class_name = template_class.rpartition(".")
        try:
            module = importlib.import_module(module_path)
        except Exception as exc:
            raise ConfigError(
                f"MultiTenantTrigger: could not import inner template "
                f"{template_class}: {type(exc).__name__}: {exc}"
            ) from exc
        cls = getattr(module, class_name, None)
        if cls is None or not isinstance(cls, type):
            raise ConfigError(
                f"MultiTenantTrigger: {template_class} does not resolve to a class"
            )
        return cls, dict(raw_params)

    def _validate_inner_template(self) -> None:
        cls = self._inner_class
        declared = getattr(cls, "__dreamer_protocols__", {})
        if Trigger not in declared:
            raise ConfigError(
                f"MultiTenantTrigger: inner template {cls.__module__}.{cls.__qualname__}"
                " does not declare @implements(Trigger)"
            )
        try:
            init_sig = inspect.signature(cls)
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"MultiTenantTrigger: could not introspect "
                f"{cls.__qualname__}.__init__: {exc}"
            ) from exc
        accepted_params: set[str] = set()
        accepts_var_keyword = False
        for param_name, param in init_sig.parameters.items():
            if param_name == "self":
                continue
            if param.kind is inspect.Parameter.VAR_KEYWORD:
                accepts_var_keyword = True
                continue
            if param.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                accepted_params.add(param_name)
        if not accepts_var_keyword:
            unknown = set(self._inner_params.keys()) - accepted_params
            if unknown:
                raise ConfigError(
                    f"MultiTenantTrigger: inner template {cls.__qualname__} does not "
                    f"accept params: {sorted(unknown)}"
                )
            if "tenant_id" not in accepted_params:
                raise ConfigError(
                    f"MultiTenantTrigger: inner template {cls.__qualname__} must accept "
                    "a 'tenant_id' parameter"
                )

    def _materialize(self, tenant_id: TenantId) -> Trigger:
        params = dict(self._inner_params)
        params["tenant_id"] = tenant_id
        try:
            instance = self._inner_class(**params)
        except TypeError as exc:
            raise ConfigError(
                f"MultiTenantTrigger: failed to materialize inner for tenant "
                f"{tenant_id!r}: {exc}"
            ) from exc
        if not isinstance(instance, Trigger):
            raise ConfigError(
                f"MultiTenantTrigger: materialized inner is not a Trigger: {instance!r}"
            )
        return instance

    def _inner_fire(self, tenant_id: TenantId, trigger_name: str) -> Any:
        async def _fire() -> None:
            await self._job_queue.publish(
                DreamJob(tenant_id=tenant_id, trigger_name=trigger_name),
                ctx=PublishContext(
                    request_id=f"trigger.fire.{tenant_id}.{trigger_name}",
                    tenant_id=tenant_id,
                ),
            )

        return _fire

    async def start(
        self, *, ctx: TriggerStartContext, services: TriggerStartServices
    ) -> None:
        if self._task is not None:
            return
        self._outer_services = services
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(self._refresh_loop())
        logger.info(
            "MultiTenantTrigger started: name=%s refresh=%.2fs",
            self.name,
            self.refresh_interval_seconds,
        )

    async def stop(self, *, ctx: TriggerStopContext) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        task = self._task
        self._task = None
        if task is not None:
            try:
                await asyncio.wait_for(task, timeout=self.refresh_interval_seconds + 1.0)
            except TimeoutError:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        for tenant_id, inner in list(self._materialized.items()):
            try:
                await inner.stop(
                    ctx=TriggerStopContext(
                        request_id=f"trigger.stop.{tenant_id}.{self.name}",
                        tenant_id=tenant_id,
                        trigger_name=inner.name,
                    )
                )
            except Exception:
                logger.exception(
                    "MultiTenantTrigger %s: inner stop raised for tenant=%s",
                    self.name,
                    tenant_id,
                )
        self._materialized.clear()
        self._outer_services = None

    async def _refresh_loop(self) -> None:
        assert self._stop_event is not None
        while not self._stop_event.is_set():
            try:
                await self._refresh_once()
            except Exception:
                logger.exception(
                    "MultiTenantTrigger %s: refresh raised", self.name
                )
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=self.refresh_interval_seconds
                )
                return
            except TimeoutError:
                continue

    async def _refresh_once(self) -> None:
        request_id = f"trigger.refresh.{self.name}"
        tenants = await self._tenant_registry.list_tenants(
            ctx=TenantRegistryContext(request_id=request_id)
        )
        seen: set[TenantId] = set()
        for tenant_id in tenants:
            if tenant_id in seen:
                continue
            seen.add(tenant_id)
            if tenant_id in self._materialized:
                continue
            await self._start_inner_for(tenant_id)
        gone = set(self._materialized.keys()) - seen
        for tenant_id in gone:
            inner = self._materialized.pop(tenant_id)
            try:
                await inner.stop(
                    ctx=TriggerStopContext(
                        request_id=f"trigger.stop.{tenant_id}.{inner.name}",
                        tenant_id=tenant_id,
                        trigger_name=inner.name,
                    )
                )
            except Exception:
                logger.exception(
                    "MultiTenantTrigger %s: inner stop raised for tenant=%s",
                    self.name,
                    tenant_id,
                )

    async def _start_inner_for(self, tenant_id: TenantId) -> None:
        assert self._outer_services is not None
        inner = self._materialize(tenant_id)
        inner_services = TriggerStartServices(
            emit_progress=self._outer_services.emit_progress,
            secrets=self._outer_services.secrets,
            usage=self._outer_services.usage,
            audit=self._outer_services.audit,
            clock=self._outer_services.clock,
            fire=self._inner_fire(tenant_id, inner.name),
        )
        try:
            await inner.start(
                ctx=TriggerStartContext(
                    request_id=f"trigger.start.{tenant_id}.{inner.name}",
                    tenant_id=tenant_id,
                    trigger_name=inner.name,
                ),
                services=inner_services,
            )
        except Exception:
            logger.exception(
                "MultiTenantTrigger %s: failed to start inner for tenant=%s",
                self.name,
                tenant_id,
            )
            return
        self._materialized[tenant_id] = inner


__all__ = ["MultiTenantTrigger"]
