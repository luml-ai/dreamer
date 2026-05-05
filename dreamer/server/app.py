"""Starlette app factory.

Builds the framework's `Starlette` application from a `ResolvedConfig`:

- Installs middleware from any component implementing `Middlewares`.
- Calls `register_routes(app, ctx=...)` for any component implementing `Routes`.
- Mounts the MCP sub-app at `/mcp` (auth-gated).
- Mounts `GET /healthz` (unauthenticated, returns per-component readiness).
- Mounts `GET /context/{path...}` and `GET /context/?prefix=...` only when the
  configured `ContextStore` implements `ContextReader@1` (auth-gated through
  the same `auth` slot).
- Owns the `MCPMount.lifespan` context so the streamable-http session manager
  has a live task group for the duration of the server.

There are **no** built-in `/admin/*` routes. Admin operators install or write
a `Routes`-capable component that resolves the in-process `dreamer.server.control`
surface.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.routing import Mount, Route
from starlette.types import Receive, Scope, Send

from dreamer.api.audit import AuditSink
from dreamer.api.auth import AuthBackend, Tenancy
from dreamer.api.capabilities import Middlewares, Routes
from dreamer.api.config import ResolvedConfig
from dreamer.api.contexts import (
    AuthContext,
    ContextReadContext,
    MiddlewaresContext,
    RoutesContext,
)
from dreamer.api.errors import AuthError, ConfigError
from dreamer.api.rate_limit import RateLimiter
from dreamer.api.runtime_state import RequestState
from dreamer.api.secrets import SecretResolver, SecretRotationHook
from dreamer.api.stores import (
    ContextReader,
    ContextStore,
    DreamLeaseStore,
    LTMStore,
    MCPTool,
    STMStore,
)
from dreamer.api.tenants import (
    TenantConfigProvider,
    TenantLifecycle,
    TenantRegistry,
    TenantScope,
)
from dreamer.api.types import MemoryType, TenantId
from dreamer.api.usage import UsageSink
from dreamer.contrib.tenants.static import (
    StaticTenantConfigProvider,
    StaticTenantLifecycle,
    StaticTenantRegistry,
)
from dreamer.server.compliance import check_components
from dreamer.server.control import Control
from dreamer.server.mcp_app import (
    MCPMount,
    MCPPipeline,
    build_mcp_mount,
    new_request_id,
)
from dreamer.server.runtime import (
    LifecycleDispatcher,
    build_hook_registry,
    build_lifecycle_dispatcher,
)
from dreamer.server.secret_watcher import SecretWatcher

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AppHandle:
    """The result of :func:`create_app`.

    Holds the live ASGI app plus the runtime objects callers may need (e.g.
    the in-process control surface for tests, CLI, or admin components).
    """

    app: Starlette
    control: Control
    lifecycle: LifecycleDispatcher
    mcp: MCPMount
    secret_watcher: SecretWatcher | None = None


@dataclass(slots=True)
class _Components:
    """Typed view over `ResolvedConfig.components` for the app factory."""

    auth: AuthBackend
    tenancy: Tenancy
    tenant_registry: TenantRegistry
    tenant_config_provider: TenantConfigProvider
    tenant_lifecycle: TenantLifecycle
    stm_store: STMStore
    ltm_store: LTMStore
    context_store: ContextStore | None
    dream_lease_store: DreamLeaseStore
    secret_resolver: SecretResolver
    rate_limiter: RateLimiter
    usage_sinks: list[UsageSink] = field(default_factory=list)
    audit_sinks: list[AuditSink] = field(default_factory=list)
    mcp_tools: list[MCPTool] = field(default_factory=list)
    memory_types: tuple[MemoryType, ...] = ()


def _extract_components(resolved: ResolvedConfig) -> _Components:
    c = resolved.components
    lists = resolved.component_lists
    return _Components(
        auth=_require(c, "auth"),
        tenancy=_require(c, "tenancy"),
        tenant_registry=_require(c, "tenant_registry"),
        tenant_config_provider=_require(c, "tenant_config_provider"),
        tenant_lifecycle=_require(c, "tenant_lifecycle"),
        stm_store=_require(c, "stm_store"),
        ltm_store=_require(c, "ltm_store"),
        context_store=c.get("context_store"),
        dream_lease_store=_require(c, "dream_lease_store"),
        secret_resolver=_require(c, "secret_resolver"),
        rate_limiter=_require(c, "rate_limiter"),
        usage_sinks=list(lists.get("usage_sinks") or []),
        audit_sinks=list(lists.get("audit_sinks") or []),
        mcp_tools=list(lists.get("mcp_tools") or []),
        memory_types=_extract_memory_types(resolved),
    )


def _require(components: dict[str, Any], slot: str) -> Any:
    value = components.get(slot)
    if value is None:
        raise ConfigError(f"required slot {slot!r} is unset")
    return value


def _extract_memory_types(resolved: ResolvedConfig) -> tuple[MemoryType, ...]:
    """Read the global `memory_types`.

    `memory_types` may live in two places: a top-level `memory_types:` block on
    the raw RootConfig (preferred when several components need to share the
    list) or — per the v1 default scaffold — as `stm_store.params.memory_types`
    on the configured STMStore. The factory looks at the top-level field
    first, then falls back to the configured STMStore's `memory_types`
    attribute. Either form ends as `tuple[MemoryType, ...]`.
    """
    raw = resolved.raw
    declared = getattr(raw, "memory_types", None)
    if not declared:
        stm_store = resolved.components.get("stm_store")
        declared = getattr(stm_store, "memory_types", None) if stm_store is not None else None
    if not declared:
        return ()
    out: list[MemoryType] = []
    for item in declared:
        if isinstance(item, MemoryType):
            out.append(item)
        elif isinstance(item, dict):
            out.append(MemoryType.model_validate(item))
        else:
            raise ConfigError(
                f"memory_types entry must be a mapping or MemoryType, got {type(item).__name__}"
            )
    return tuple(out)


def _extract_max_content_bytes(resolved: ResolvedConfig) -> int:
    stm_store = resolved.components.get("stm_store")
    value = getattr(stm_store, "max_content_bytes", None) if stm_store is not None else None
    if isinstance(value, int) and value > 0:
        return value
    return 8192


def _healthz_route(components: _Components) -> Route:
    async def endpoint(request: Request) -> Response:
        slots: dict[str, dict[str, Any]] = {}
        for slot, component in (
            ("auth", components.auth),
            ("tenancy", components.tenancy),
            ("tenant_registry", components.tenant_registry),
            ("tenant_config_provider", components.tenant_config_provider),
            ("tenant_lifecycle", components.tenant_lifecycle),
            ("stm_store", components.stm_store),
            ("ltm_store", components.ltm_store),
            ("context_store", components.context_store),
            ("dream_lease_store", components.dream_lease_store),
            ("secret_resolver", components.secret_resolver),
            ("rate_limiter", components.rate_limiter),
        ):
            slots[slot] = await _component_health(component)
        for i, usage_sink in enumerate(components.usage_sinks):
            slots[f"usage_sinks[{i}]"] = await _component_health(usage_sink)
        for j, audit_sink in enumerate(components.audit_sinks):
            slots[f"audit_sinks[{j}]"] = await _component_health(audit_sink)
        for k, tool in enumerate(components.mcp_tools):
            slots[f"mcp_tools[{k}]"] = await _component_health(tool)
        return JSONResponse({"status": "ok", "components": slots})

    return Route("/healthz", endpoint, methods=["GET"])


async def _component_health(component: Any) -> dict[str, Any]:
    if component is None:
        return {"fqn": None, "ready": False}
    fqn = f"{type(component).__module__}.{type(component).__qualname__}"
    ready = True
    detail: str | None = None
    health_fn = getattr(component, "health", None)
    if callable(health_fn):
        try:
            result = health_fn()
            if hasattr(result, "__await__"):
                result = await result
            if isinstance(result, dict):
                ready = bool(result.get("ready", True))
                detail = result.get("detail")
            elif isinstance(result, bool):
                ready = result
        except Exception as exc:  # noqa: BLE001 — health probe must never crash /healthz
            ready = False
            detail = f"{type(exc).__name__}: {exc}"
    out: dict[str, Any] = {"fqn": fqn, "ready": ready}
    if detail is not None:
        out["detail"] = detail
    return out


def _context_routes(components: _Components, *, reader: ContextReader) -> list[Route]:
    async def list_endpoint(request: Request) -> Response:
        state = RequestState.current()
        if state is None:
            return _unauthorized()
        prefix = request.query_params.get("prefix", "") or ""
        tenant_id = await _ensure_tenant(state, components)
        with TenantScope.set(tenant_id):
            ctx = ContextReadContext(
                request_id=state.request_id,
                tenant_id=tenant_id,
                principal_id=state.principal.id,
            )
            try:
                files = await reader.list(prefix=prefix, ctx=ctx)
            except FileNotFoundError:
                return JSONResponse({"files": []})
        return JSONResponse({"files": files})

    async def read_endpoint(request: Request) -> Response:
        state = RequestState.current()
        if state is None:
            return _unauthorized()
        path = request.path_params.get("path", "")
        tenant_id = await _ensure_tenant(state, components)
        with TenantScope.set(tenant_id):
            ctx = ContextReadContext(
                request_id=state.request_id,
                tenant_id=tenant_id,
                principal_id=state.principal.id,
            )
            try:
                body = await reader.read(path, ctx=ctx)
            except FileNotFoundError:
                return PlainTextResponse("not found", status_code=404)
        media = _guess_media_type(path)
        return Response(content=body, media_type=media)

    return [
        Route("/context/", list_endpoint, methods=["GET"]),
        Route("/context", list_endpoint, methods=["GET"]),
        Route("/context/{path:path}", read_endpoint, methods=["GET"]),
    ]


def _guess_media_type(path: str) -> str:
    lower = path.lower()
    if lower.endswith((".md", ".markdown")):
        return "text/markdown; charset=utf-8"
    if lower.endswith(".json"):
        return "application/json"
    if lower.endswith(".txt"):
        return "text/plain; charset=utf-8"
    if lower.endswith(".yaml") or lower.endswith(".yml"):
        return "application/yaml"
    return "application/octet-stream"


async def _ensure_tenant(state: RequestState, components: _Components) -> TenantId:
    if state.tenant_id is not None:
        return state.tenant_id
    from dreamer.api.contexts import TenancyContext  # noqa: PLC0415

    ctx = TenancyContext(request_id=state.request_id, principal=state.principal)
    tenant_id = await components.tenancy.tenant_for(state.principal, ctx=ctx)
    state.tenant_id = tenant_id
    return tenant_id


def _unauthorized() -> Response:
    return JSONResponse({"error": "auth_failed"}, status_code=401)


@dataclass(slots=True)
class AuthMiddleware:
    """ASGI middleware that authenticates HTTP requests against the configured
    `AuthBackend` and binds `RequestState` for the duration of the request.

    Routes registered under `unauthenticated_paths` are passed through
    unchanged. The default Starlette dispatch model means this middleware sees
    every request including the MCP sub-app and `/context/...`.
    """

    app: Any
    auth: AuthBackend
    audit_sinks: list[AuditSink]
    unauthenticated_paths: tuple[str, ...] = ("/healthz",)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        # CORS preflight requests do not (and cannot) carry the Authorization
        # header. Let them through so a downstream CORSMiddleware (or the
        # browser default) can answer; otherwise every cross-origin client
        # would see a 401 before ever issuing the real request.
        if scope.get("method") == "OPTIONS":
            await self.app(scope, receive, send)
            return
        path: str = scope.get("path", "")
        if any(path == p or path.startswith(p + "/") for p in self.unauthenticated_paths):
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive=receive, send=send)
        request_id = new_request_id()
        ctx = AuthContext(request_id=request_id, request=request)
        try:
            principal = await self.auth.authenticate(request, ctx=ctx)
        except AuthError as exc:
            await _emit_auth_failed(self.audit_sinks, str(exc), request_id=request_id)
            response = JSONResponse({"error": "auth_failed"}, status_code=401)
            await response(scope, receive, send)
            return
        except Exception as exc:  # noqa: BLE001 — surface as 401 + log
            logger.exception("AuthBackend raised unexpectedly")
            await _emit_auth_failed(
                self.audit_sinks,
                f"{type(exc).__name__}: {exc}",
                request_id=request_id,
            )
            response = JSONResponse({"error": "auth_failed"}, status_code=401)
            await response(scope, receive, send)
            return
        state = RequestState(principal=principal, request_id=request_id)
        with RequestState.bind(state):
            await self.app(scope, receive, send)


async def _emit_auth_failed(
    audit_sinks: list[AuditSink],
    detail: str,
    *,
    request_id: str,
) -> None:
    if not audit_sinks:
        return
    from datetime import UTC, datetime  # noqa: PLC0415

    from dreamer.api.contexts import AuditContext  # noqa: PLC0415
    from dreamer.api.types import DEFAULT_TENANT_ID, AuditEvent  # noqa: PLC0415
    from dreamer.server.sinks import emit_audit  # noqa: PLC0415

    await emit_audit(
        audit_sinks,
        AuditEvent(
            event_type="auth.failed",
            principal_id=None,
            tenant_id=DEFAULT_TENANT_ID,
            payload={"detail": detail},
            at=datetime.now(UTC),
        ),
        ctx=AuditContext(request_id=request_id, tenant_id=DEFAULT_TENANT_ID),
    )


def create_app(resolved: ResolvedConfig) -> AppHandle:
    """Build the Starlette ASGI app + Control surface for the framework."""
    components = _extract_components(resolved)

    pipeline = MCPPipeline(
        tenancy=components.tenancy,
        stm_store=components.stm_store,
        tenant_config_provider=components.tenant_config_provider,
        rate_limiter=components.rate_limiter,
        hook_registry=build_hook_registry(resolved.component_lists),
        audit_sinks=components.audit_sinks,
        usage_sinks=components.usage_sinks,
        mcp_tools=components.mcp_tools,
        memory_types=components.memory_types,
        max_content_bytes=_extract_max_content_bytes(resolved),
    )
    mcp = build_mcp_mount(pipeline)

    routes: list[Any] = [_healthz_route(components)]
    context_store = components.context_store
    if isinstance(context_store, ContextReader):
        routes.extend(_context_routes(components, reader=context_store))
    routes.append(Mount("/mcp", app=mcp.asgi))

    middlewares: list[Middleware] = []
    # Outermost: CORS. Browser-based clients (e.g. MCP Inspector) issue a
    # preflight OPTIONS without the Authorization header — CORSMiddleware
    # short-circuits it with the right Access-Control-* headers before auth
    # ever runs. Permissive defaults are appropriate for a self-hosted
    # framework; operators can layer a stricter CORSMiddleware via the
    # `Middlewares` capability.
    middlewares.append(
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["mcp-session-id"],
        )
    )
    middlewares.append(
        Middleware(
            AuthMiddleware,
            auth=components.auth,
            audit_sinks=components.audit_sinks,
        )
    )
    for component in _all_components(resolved):
        if isinstance(component, Middlewares):
            mw_ctx = MiddlewaresContext(request_id="middlewares.boot")
            for extra in component.middlewares(ctx=mw_ctx):
                middlewares.append(extra)

    lifecycle = build_lifecycle_dispatcher(components=_all_components(resolved))

    secret_hooks = [c for c in _all_components(resolved) if isinstance(c, SecretRotationHook)]
    watcher: SecretWatcher | None = None
    if secret_hooks:
        watcher = SecretWatcher(
            resolver=components.secret_resolver,
            hooks=secret_hooks,
        )

    @contextlib.asynccontextmanager
    async def _lifespan(app: Starlette) -> AsyncIterator[None]:
        async with mcp.lifespan():
            await lifecycle.start_all()
            if watcher is not None:
                await watcher.start()
            try:
                yield
            finally:
                if watcher is not None:
                    await watcher.stop()
                await lifecycle.stop_all()

    app = Starlette(
        routes=routes,
        middleware=middlewares,
        lifespan=_lifespan,
    )

    for component in _all_components(resolved):
        if isinstance(component, Routes):
            ctx = RoutesContext(request_id="routes.register")
            component.register_routes(app, ctx=ctx)

    effective_multi_tenant = _compute_effective_multi_tenant(resolved)
    _wire_tenant_lifecycle(components, all_components=_all_components(resolved))

    control = Control(
        tenant_registry=components.tenant_registry,
        tenant_config_provider=components.tenant_config_provider,
        tenant_lifecycle=components.tenant_lifecycle,
        effective_multi_tenant=effective_multi_tenant,
    )

    return AppHandle(
        app=app,
        control=control,
        lifecycle=lifecycle,
        mcp=mcp,
        secret_watcher=watcher,
    )


def _compute_effective_multi_tenant(resolved: ResolvedConfig) -> bool:
    """Compute the effective multi-tenancy mode for the running deployment.

    Reuses the compliance checker's per-slot table so the answer is consistent
    with ``dreamer config check``.
    """
    from dreamer.cli.main import _iter_slot_bindings  # noqa: PLC0415

    bindings = list(_iter_slot_bindings(resolved))
    report = check_components(
        bindings,
        declared_mode=resolved.declared_multi_tenancy,  # type: ignore[arg-type]
    )
    return report.effective_multi_tenant


def _wire_tenant_lifecycle(
    components: _Components,
    *,
    all_components: list[Any],
) -> None:
    """Hand the full component graph to a default ``StaticTenantLifecycle``.

    Custom lifecycle implementations can do their own discovery — we only wire
    the contrib default. Likewise the default ``StaticTenantConfigProvider``
    gets the global ``memory_types`` so per-tenant overrides can be validated as
    a subset.
    """
    if isinstance(components.tenant_lifecycle, StaticTenantLifecycle):
        components.tenant_lifecycle.set_tenant_data_components(all_components)
        if isinstance(components.tenant_registry, StaticTenantRegistry):
            components.tenant_lifecycle.set_registry(components.tenant_registry)
    if isinstance(components.tenant_config_provider, StaticTenantConfigProvider):
        components.tenant_config_provider.set_global_memory_types(
            components.memory_types
        )


def _all_components(resolved: ResolvedConfig) -> list[Any]:
    """Yield every configured component, including hook lists, mcp_tools, etc."""
    seen: list[Any] = []
    seen_ids: set[int] = set()

    def add(c: Any) -> None:
        if c is None:
            return
        if id(c) in seen_ids:
            return
        seen_ids.add(id(c))
        seen.append(c)

    for c in resolved.components.values():
        add(c)
    for items in resolved.component_lists.values():
        for c in items:
            add(c)
    return seen


__all__ = ["AppHandle", "AuthMiddleware", "create_app"]
