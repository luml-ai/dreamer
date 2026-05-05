"""Server bootstrap helpers.

Builds the runtime objects that the framework needs from a
:class:`ResolvedConfig`: the :class:`Orchestrator`, the :class:`TriggerHost`,
and a :class:`Runtime` aggregate that owns the lifecycle of both alongside
the :class:`AppHandle` returned by :func:`create_app`.

``Runtime`` exposes two flow-shapes used by the CLI:

- ``async with runtime.serving(): ...`` — ``dreamer serve``. The Starlette
  app's lifespan drives the lifecycle dispatcher (and the secret watcher);
  this wrapper additionally starts / stops the configured trigger host.
- ``async with runtime.session(): ...`` — ``dreamer dream``. Bypasses
  Starlette and uvicorn entirely; manually drives the lifecycle dispatcher
  so the orchestrator and stores boot for a single one-shot operation.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, cast

from dreamer.api.audit import AuditSink
from dreamer.api.config import ResolvedConfig
from dreamer.api.dream import ContextPhaseRunner, DreamGate, LTMPhaseRunner
from dreamer.api.errors import ConfigError
from dreamer.api.jobs import JobQueue
from dreamer.api.secrets import SecretResolver
from dreamer.api.stores import (
    ContextStore,
    DreamLeaseStore,
    LTMStore,
    STMStore,
)
from dreamer.api.tenants import TenantConfigProvider, TenantRegistry
from dreamer.api.triggers import Trigger
from dreamer.api.usage import UsageSink
from dreamer.contrib.context.fanout import FanoutContextStore
from dreamer.server.app import AppHandle, create_app
from dreamer.server.orchestrator import Orchestrator, StmRetentionConfig
from dreamer.server.runtime import (
    TriggerHost,
    build_hook_registry,
    build_trigger_host,
    build_trigger_registry,
)


@dataclass(slots=True)
class Runtime:
    """Aggregate of the running server and its in-process orchestrator.

    Constructed by :func:`build_runtime`. The CLI (``serve`` / ``dream``)
    drives the lifecycle through one of the two ``async with`` helpers
    defined on this class.
    """

    handle: AppHandle
    orchestrator: Orchestrator
    triggers: list[Trigger] = field(default_factory=list)
    trigger_host: TriggerHost | None = None

    @contextlib.asynccontextmanager
    async def serving(self) -> AsyncIterator[None]:
        """Start the trigger host alongside the Starlette app.

        Used by ``dreamer serve``: the Starlette app's own ``_lifespan``
        already drives ``LifecycleDispatcher`` and the secret watcher when
        uvicorn boots, so this wrapper only spins up the configured trigger
        host (which is **not** a ``Lifecycle`` component because triggers
        require ``services`` to start).
        """
        if self.trigger_host is not None and self.triggers:
            await self.trigger_host.start_all()
        try:
            yield
        finally:
            if self.trigger_host is not None and self.triggers:
                with contextlib.suppress(Exception):
                    await self.trigger_host.stop_all()

    @contextlib.asynccontextmanager
    async def session(self) -> AsyncIterator[None]:
        """Drive the lifecycle dispatcher manually for a one-shot operation.

        Used by ``dreamer dream``: the Starlette/uvicorn loop is not running,
        so the framework must explicitly start / stop every ``Lifecycle``
        component before / after the work happens. Triggers are intentionally
        **not** started — ``dream`` runs a single cycle on demand.
        """
        await self.handle.lifecycle.start_all()
        try:
            yield
        finally:
            with contextlib.suppress(Exception):
                await self.handle.lifecycle.stop_all()


def build_runtime(resolved: ResolvedConfig) -> Runtime:
    """Build the full server runtime from a resolved config.

    Wires:

    - :class:`AppHandle` (Starlette app + Control + lifecycle dispatcher).
    - :class:`Orchestrator` from the resolved component graph.
    - :class:`TriggerHost` for the configured triggers.
    - Binds the orchestrator into ``handle.control`` so the CLI / admin
      components can ``await control.trigger_dream(...)``.

    The returned :class:`Runtime` is *not yet started*; call
    ``async with runtime.serving():`` (HTTP) or ``async with
    runtime.session():`` (CLI dream) depending on the entry point.
    """
    handle = create_app(resolved)

    orchestrator = _build_orchestrator(resolved)

    # Orchestrator owns the JobQueue subscription + STM retention loop, so it
    # must participate in the lifecycle dispatcher.
    handle.lifecycle.register(orchestrator)

    triggers = list(resolved.component_lists.get("triggers", []))
    build_trigger_registry(
        triggers,
        effective_multi_tenant=handle.control.effective_multi_tenant,
    )

    trigger_host = build_trigger_host(
        triggers=triggers,
        job_queue=_require(resolved, "job_queue"),
        secret_resolver=_require(resolved, "secret_resolver"),
        usage_sinks=resolved.component_lists.get("usage_sinks") or [],
        audit_sinks=resolved.component_lists.get("audit_sinks") or [],
        progress_hooks=resolved.component_lists.get("hooks.on_dream_progress") or [],
    )

    handle.control.bind_orchestrator(
        trigger_dream_fn=orchestrator.trigger_dream,
        state_reader_fn=orchestrator.read_state,
        active_lease_waiter=orchestrator.wait_for_active_lease_release,
    )

    return Runtime(
        handle=handle,
        orchestrator=orchestrator,
        triggers=triggers,
        trigger_host=trigger_host,
    )


def _build_orchestrator(resolved: ResolvedConfig) -> Orchestrator:
    components = resolved.components
    lists = resolved.component_lists

    stm_store: STMStore = _require(resolved, "stm_store")
    ltm_store: LTMStore = _require(resolved, "ltm_store")
    context_store = _resolve_context_store(resolved)
    dream_lease_store: DreamLeaseStore = _require(resolved, "dream_lease_store")

    ltm_runner, context_runner = _resolve_phase_runners(resolved)

    tenant_registry: TenantRegistry = _require(resolved, "tenant_registry")
    tenant_config_provider: TenantConfigProvider = _require(
        resolved, "tenant_config_provider"
    )

    job_queue: JobQueue = _require(resolved, "job_queue")
    secret_resolver: SecretResolver | None = components.get("secret_resolver")

    audit_sinks: list[AuditSink] = list(lists.get("audit_sinks") or [])
    usage_sinks: list[UsageSink] = list(lists.get("usage_sinks") or [])
    dream_gates: list[DreamGate] = list(lists.get("dream_gates") or [])

    hook_registry = build_hook_registry(lists)

    retention = StmRetentionConfig(
        keep_days=resolved.raw.stm_retention.keep_days,
        cadence_seconds=resolved.raw.stm_retention.cadence_seconds,
    )

    default_lease_ttl = float(getattr(dream_lease_store, "default_ttl_seconds", 1800.0))

    return Orchestrator(
        stm_store=stm_store,
        ltm_store=ltm_store,
        context_store=context_store,
        dream_lease_store=dream_lease_store,
        ltm_phase_runner=ltm_runner,
        context_phase_runner=context_runner,
        tenant_registry=tenant_registry,
        tenant_config_provider=tenant_config_provider,
        job_queue=job_queue,
        hook_registry=hook_registry,
        audit_sinks=audit_sinks,
        usage_sinks=usage_sinks,
        secret_resolver=secret_resolver,
        dream_gates=dream_gates,
        stm_retention=retention,
        default_lease_ttl_seconds=default_lease_ttl,
    )


def _resolve_context_store(resolved: ResolvedConfig) -> ContextStore:
    """Return the configured context store, wrapping a list in a fanout.

    The :class:`FanoutContextStore` reports its ``multi_tenant`` and
    ``workspace_capabilities`` as instance attributes (computed from its
    backings), so mypy's strict ``ClassVar`` Protocol check rejects it; the
    ``cast`` here is intentional and the orchestrator's runtime ``isinstance``
    probes (e.g. ``ContextReader``) handle the dynamic capability surface.
    """
    explicit = resolved.components.get("context_store")
    if explicit is not None:
        return cast(ContextStore, explicit)
    listed = resolved.component_lists.get("context_store") or []
    if not listed:
        raise ConfigError("required slot 'context_store' is unset")
    if len(listed) == 1:
        return cast(ContextStore, listed[0])
    return cast(ContextStore, FanoutContextStore(backings=listed))


def _resolve_phase_runners(
    resolved: ResolvedConfig,
) -> tuple[LTMPhaseRunner, ContextPhaseRunner]:
    """Return the LTM and Context phase runners.

    ``dream_engine_overrides`` is reserved on :class:`RootConfig` but not yet
    wired by the runtime — the v1 default ships a single component
    implementing both phase Protocols. If overrides appear in config, fail
    fast so a misconfigured deployment is not silently ignored.
    """
    raw = resolved.raw
    overrides = raw.dream_engine_overrides or {}
    has_phase_override = isinstance(overrides, dict) and (
        overrides.get("ltm_phase") or overrides.get("context_phase")
    )
    if has_phase_override:
        raise ConfigError(
            "dream_engine_overrides is reserved but not yet wired by the runtime; "
            "configure a single component implementing both phase runners under "
            "'dream_engine' instead"
        )

    engine = resolved.components.get("dream_engine")
    if engine is None:
        raise ConfigError("required slot 'dream_engine' is unset")
    if not isinstance(engine, LTMPhaseRunner):
        raise ConfigError(
            "dream_engine must implement LTMPhaseRunner@1 "
            f"(got {type(engine).__module__}.{type(engine).__qualname__})"
        )
    if not isinstance(engine, ContextPhaseRunner):
        raise ConfigError(
            "dream_engine must implement ContextPhaseRunner@1 "
            f"(got {type(engine).__module__}.{type(engine).__qualname__})"
        )
    return engine, engine


def _require(resolved: ResolvedConfig, slot: str) -> Any:
    value = resolved.components.get(slot)
    if value is None:
        raise ConfigError(f"required slot {slot!r} is unset")
    return value


__all__ = ["Runtime", "build_runtime"]
