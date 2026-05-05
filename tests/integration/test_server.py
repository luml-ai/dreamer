from __future__ import annotations

import asyncio
import contextlib
import json
import uuid
from collections.abc import AsyncIterator
from typing import Any, ClassVar

import httpx
import pytest
import pytest_asyncio

from dreamer.api.auth import AuthBackend
from dreamer.api.compat import implements
from dreamer.api.config import ResolvedConfig, RootConfig
from dreamer.api.contexts import (
    AuthContext,
    PostMemorySubmitContext,
    PreMemorySubmitContext,
    RateLimitContext,
    SecretContext,
    TenantConfigLookupContext,
)
from dreamer.api.errors import AuthError
from dreamer.api.hooks import (
    PostMemorySubmitHook,
    PreMemorySubmitHook,
)
from dreamer.api.rate_limit import RateLimiter
from dreamer.api.runtime_state import RequestState
from dreamer.api.types import (
    DEFAULT_TENANT_ID,
    MemoryType,
    Principal,
    RateLimitDecision,
    SecretValue,
    TenantConfig,
    TenantId,
)
from dreamer.contrib.context.markdown import MarkdownContextStore
from dreamer.contrib.jobs.inproc import InProcessJobQueue
from dreamer.contrib.tenancy.single import SingleTenant
from dreamer.server.app import AppHandle, create_app
from dreamer.server.mcp_app import (
    MCPPipeline,
    _dispatch,
    _submit_memory_tool,
)
from dreamer.server.runtime import HookRegistry
from dreamer.server.secret_watcher import SecretWatcher
from dreamer.testing.fakes import (
    CollectingAuditSink,
    CollectingUsageSink,
    InMemoryContextStore,
    InMemoryDreamLeaseStore,
    InMemoryLTMStore,
    InMemorySTMStore,
    NoOpRateLimiter,
)


class StaticTokenAuth:
    multi_tenant: ClassVar[bool] = False

    def __init__(self, *, token: str, tenant_id: TenantId = DEFAULT_TENANT_ID) -> None:
        self.token = token
        self.tenant_id = tenant_id

    async def authenticate(self, request: Any, *, ctx: AuthContext) -> Principal:
        header = request.headers.get("authorization") or ""
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or token.strip() != self.token:
            raise AuthError("auth_failed: bad token")
        return Principal(id="agent-1", tenant_id=self.tenant_id)


implements(AuthBackend, version=1)(StaticTokenAuth)


class StaticTenantRegistry:
    multi_tenant: ClassVar[bool] = False

    def __init__(self, tenants: list[TenantId] | None = None) -> None:
        self.tenants = list(tenants or [DEFAULT_TENANT_ID])

    async def list_tenants(self, *, ctx: Any) -> list[TenantId]:
        return list(self.tenants)

    async def exists(self, tenant_id: TenantId, *, ctx: Any) -> bool:
        return tenant_id in self.tenants


class StaticTenantConfigProvider:
    multi_tenant: ClassVar[bool] = False

    def __init__(self, overrides: dict[TenantId, TenantConfig] | None = None) -> None:
        self.overrides = dict(overrides or {})

    async def get(
        self, tenant_id: TenantId, *, ctx: TenantConfigLookupContext
    ) -> TenantConfig:
        return self.overrides.get(tenant_id, TenantConfig())


class StaticTenantLifecycle:
    multi_tenant: ClassVar[bool] = False

    async def provision(self, tenant_id: TenantId, *, ctx: Any) -> None:
        return None

    async def deprovision(
        self, tenant_id: TenantId, *, mode: str, ctx: Any
    ) -> None:
        return None

    async def reset(self, tenant_id: TenantId, *, ctx: Any) -> None:
        return None


class EnvSecretResolver:
    multi_tenant: ClassVar[bool] = False

    def __init__(self) -> None:
        self.values: dict[str, SecretValue] = {}

    async def get(
        self, name: str, *, tenant_id: TenantId | None, ctx: SecretContext
    ) -> SecretValue:
        return self.values.get(name, SecretValue(value="", ttl_seconds=None))


class TightRateLimiter:
    multi_tenant: ClassVar[bool] = False

    def __init__(self, *, allow: bool = True, retry_after: float = 0.9) -> None:
        self.allow = allow
        self.retry_after = retry_after
        self.calls = 0

    async def check(
        self,
        *,
        principal: Principal,
        tenant_id: TenantId,
        action: str,
        ctx: RateLimitContext,
    ) -> RateLimitDecision:
        self.calls += 1
        if self.allow:
            return RateLimitDecision(allowed=True)
        return RateLimitDecision(
            allowed=False, retry_after_seconds=self.retry_after, reason="too fast"
        )


implements(RateLimiter, version=1)(TightRateLimiter)


class DropAllPreSubmit:
    multi_tenant: ClassVar[bool] = False

    async def on_pre_memory_submit(self, *, ctx: PreMemorySubmitContext) -> None:
        ctx.memories.clear()


implements(PreMemorySubmitHook, version=1)(DropAllPreSubmit)


class ExpandPreSubmit:
    multi_tenant: ClassVar[bool] = False

    async def on_pre_memory_submit(self, *, ctx: PreMemorySubmitContext) -> None:
        if not ctx.memories:
            return
        original = ctx.memories[0]
        for i in range(2):
            ctx.memories.append(
                original.model_copy(
                    update={
                        "id": None,
                        "title": f"{original.title} (clone {i})",
                        "idempotency_key": None,
                    }
                )
            )


implements(PreMemorySubmitHook, version=1)(ExpandPreSubmit)


class RaisingPreSubmit:
    multi_tenant: ClassVar[bool] = False

    async def on_pre_memory_submit(self, *, ctx: PreMemorySubmitContext) -> None:
        raise ValueError("forbidden")


implements(PreMemorySubmitHook, version=1)(RaisingPreSubmit)


class CountingPostSubmit:
    multi_tenant: ClassVar[bool] = False

    def __init__(self) -> None:
        self.count = 0

    async def on_post_memory_submit(self, *, ctx: PostMemorySubmitContext) -> None:
        self.count += len(ctx.persisted)


implements(PostMemorySubmitHook, version=1)(CountingPostSubmit)


class RotatingResolver:
    multi_tenant: ClassVar[bool] = False

    def __init__(self, name: str, versions: list[str]) -> None:
        self.name = name
        self.versions = versions
        self.index = 0
        self.calls: list[str] = []

    async def get(
        self, name: str, *, tenant_id: TenantId | None, ctx: SecretContext
    ) -> SecretValue:
        self.calls.append(name)
        if name != self.name:
            return SecretValue(value="", version=None)
        version = self.versions[min(self.index, len(self.versions) - 1)]
        self.index += 1
        return SecretValue(value="value", version=version, ttl_seconds=2.0)


class CapturingRotationHook:
    multi_tenant: ClassVar[bool] = False
    secret_dependencies: ClassVar[frozenset[str]] = frozenset({"GH_TOKEN"})

    def __init__(self) -> None:
        self.events: list[str] = []

    async def on_secret_rotated(self, name: str, *, ctx: Any) -> None:
        self.events.append(name)


def _memory_types() -> tuple[MemoryType, ...]:
    return (
        MemoryType(name="failure", description="An unexpected failure."),
        MemoryType(name="observation", description="A general observation."),
        MemoryType(
            name="code_snippet",
            description="A useful code pattern.",
        ),
    )


def _failure_with_schema() -> tuple[MemoryType, ...]:
    return (
        MemoryType(
            name="failure",
            description="A failure (with metadata schema).",
            metadata_schema={
                "type": "object",
                "required": ["severity"],
                "properties": {"severity": {"enum": ["low", "high"]}},
            },
        ),
    )


def _make_pipeline(
    *,
    stm_store: InMemorySTMStore,
    audit_sink: CollectingAuditSink,
    usage_sink: CollectingUsageSink,
    rate_limiter: RateLimiter | None = None,
    pre_hooks: list[PreMemorySubmitHook] | None = None,
    post_hooks: list[PostMemorySubmitHook] | None = None,
    overrides: dict[TenantId, TenantConfig] | None = None,
    memory_types: tuple[MemoryType, ...] | None = None,
    max_content_bytes: int = 8192,
) -> MCPPipeline:
    registry = HookRegistry()
    for h in pre_hooks or []:
        registry.add("pre_memory_submit", h)
    for h in post_hooks or []:
        registry.add("post_memory_submit", h)
    return MCPPipeline(
        tenancy=SingleTenant(),
        stm_store=stm_store,
        tenant_config_provider=StaticTenantConfigProvider(overrides=overrides or {}),
        rate_limiter=rate_limiter or NoOpRateLimiter(),
        hook_registry=registry,
        audit_sinks=[audit_sink],
        usage_sinks=[usage_sink],
        mcp_tools=[],
        memory_types=memory_types or _memory_types(),
        max_content_bytes=max_content_bytes,
    )


@contextlib.contextmanager
def _bind_request(principal: Principal | None = None) -> Any:
    if principal is None:
        principal = Principal(id="agent-1", tenant_id=DEFAULT_TENANT_ID)
    state = RequestState(principal=principal, request_id=uuid.uuid4().hex)
    with RequestState.bind(state):
        yield state


def _decode_block(blocks: Any) -> dict[str, Any]:
    return json.loads(blocks[0].text)


def test_submit_memory_tool_description_carries_guidance() -> None:
    # The tool description is the agent's only in-band guidance on when to
    # submit and what to keep out (secrets). Lock the load-bearing sections
    # so future edits don't silently drop them.
    tool = _submit_memory_tool(_memory_types())
    desc = tool.description or ""
    for needle in (
        "WHEN TO SUBMIT",
        "ERRORS — resolve first",
        "OBSERVATIONS — capture what you could not have known",
        "SECURITY — never leak secrets",
        "Available memory types",
        "failure",
        "observation",
        "code_snippet",
    ):
        assert needle in desc, f"missing guidance section: {needle!r}"
    # `agent_id` is no longer a public field — guard against doc drift.
    assert "agent_id" not in desc
    assert "agent_id" not in (tool.inputSchema.get("properties") or {})
    assert "agent_id" not in (tool.inputSchema.get("required") or [])


def test_submit_memory_tool_description_omits_types_section_when_none() -> None:
    tool = _submit_memory_tool(())
    assert "Available memory types" not in (tool.description or "")


@pytest.mark.asyncio
async def test_submit_memory_happy_path() -> None:
    stm = InMemorySTMStore()
    audit = CollectingAuditSink()
    usage = CollectingUsageSink()
    pipeline = _make_pipeline(stm_store=stm, audit_sink=audit, usage_sink=usage)
    args = {
        "type": "observation",
        "title": "cache miss surprise",
        "content": "rebuilt the cache and the second hit went to the wrong shard",
    }
    with _bind_request() as state:
        result = await _dispatch(
            pipeline=pipeline, tool_name="submit_memory", args=args, state=state
        )
    body = _decode_block(result)
    assert body["ok"] is True
    submitted = body["result"]["submitted"]
    assert len(submitted) == 1
    assert submitted[0]["title"] == "cache miss surprise"
    assert submitted[0]["tenant_id"] == DEFAULT_TENANT_ID
    assert submitted[0]["consumed_at"] is None
    assert any(e.event_type == "mcp.submit_memory" for e in audit.events)


@pytest.mark.asyncio
async def test_submit_memory_idempotency() -> None:
    stm = InMemorySTMStore()
    audit = CollectingAuditSink()
    usage = CollectingUsageSink()
    pipeline = _make_pipeline(stm_store=stm, audit_sink=audit, usage_sink=usage)
    args = {
        "type": "observation",
        "title": "a thing",
        "content": "x",
        "idempotency_key": "abc",
    }
    with _bind_request() as state:
        first = _decode_block(
            await _dispatch(pipeline=pipeline, tool_name="submit_memory", args=args, state=state)
        )
        second = _decode_block(
            await _dispatch(pipeline=pipeline, tool_name="submit_memory", args=args, state=state)
        )
    assert first["result"]["submitted"][0]["id"] == second["result"]["submitted"][0]["id"]


@pytest.mark.asyncio
async def test_pre_memory_submit_drop() -> None:
    stm = InMemorySTMStore()
    audit = CollectingAuditSink()
    usage = CollectingUsageSink()
    post = CountingPostSubmit()
    pipeline = _make_pipeline(
        stm_store=stm,
        audit_sink=audit,
        usage_sink=usage,
        pre_hooks=[DropAllPreSubmit()],
        post_hooks=[post],
    )
    args = {
        "type": "observation",
        "title": "x",
        "content": "y",
    }
    with _bind_request() as state:
        body = _decode_block(
            await _dispatch(pipeline=pipeline, tool_name="submit_memory", args=args, state=state)
        )
    assert body["ok"] is True
    assert body["result"]["submitted"] == []
    assert body["result"]["filtered"] is True
    assert post.count == 0
    from dreamer.api.contexts import CountContext
    from dreamer.api.tenants import TenantScope

    with TenantScope.set(DEFAULT_TENANT_ID):
        count = await stm.count_unconsumed(
            ctx=CountContext(request_id="t", tenant_id=DEFAULT_TENANT_ID)
        )
    assert count == 0


@pytest.mark.asyncio
async def test_pre_memory_submit_expand() -> None:
    stm = InMemorySTMStore()
    audit = CollectingAuditSink()
    usage = CollectingUsageSink()
    post = CountingPostSubmit()
    pipeline = _make_pipeline(
        stm_store=stm,
        audit_sink=audit,
        usage_sink=usage,
        pre_hooks=[ExpandPreSubmit()],
        post_hooks=[post],
    )
    args = {
        "type": "observation",
        "title": "base",
        "content": "x",
    }
    with _bind_request() as state:
        body = _decode_block(
            await _dispatch(pipeline=pipeline, tool_name="submit_memory", args=args, state=state)
        )
    assert body["ok"] is True
    assert len(body["result"]["submitted"]) == 3
    titles = {m["title"] for m in body["result"]["submitted"]}
    assert titles == {"base", "base (clone 0)", "base (clone 1)"}
    assert post.count == 3


@pytest.mark.asyncio
async def test_pre_memory_submit_raise() -> None:
    stm = InMemorySTMStore()
    audit = CollectingAuditSink()
    usage = CollectingUsageSink()
    pipeline = _make_pipeline(
        stm_store=stm,
        audit_sink=audit,
        usage_sink=usage,
        pre_hooks=[RaisingPreSubmit()],
    )
    args = {
        "type": "observation",
        "title": "x",
        "content": "y",
    }
    with _bind_request() as state:
        body = _decode_block(
            await _dispatch(pipeline=pipeline, tool_name="submit_memory", args=args, state=state)
        )
    assert body["ok"] is False
    assert body["error"] == "hook_failed"
    assert "forbidden" in body["message"]
    assert any(e.event_type == "hook.failed" for e in audit.events)


@pytest.mark.asyncio
async def test_rejected_memory_type() -> None:
    stm = InMemorySTMStore()
    audit = CollectingAuditSink()
    usage = CollectingUsageSink()
    pipeline = _make_pipeline(stm_store=stm, audit_sink=audit, usage_sink=usage)
    args = {
        "type": "random_thought",
        "title": "x",
        "content": "y",
    }
    with _bind_request() as state:
        body = _decode_block(
            await _dispatch(pipeline=pipeline, tool_name="submit_memory", args=args, state=state)
        )
    assert body["ok"] is False
    assert body["error"] == "type_not_allowed"
    assert "failure" in body["message"]


@pytest.mark.asyncio
async def test_tenant_override_restricts_types() -> None:
    stm = InMemorySTMStore()
    audit = CollectingAuditSink()
    usage = CollectingUsageSink()
    overrides = {
        DEFAULT_TENANT_ID: TenantConfig(
            memory_types=(MemoryType(name="observation", description="only obs"),),
        )
    }
    pipeline = _make_pipeline(
        stm_store=stm,
        audit_sink=audit,
        usage_sink=usage,
        overrides=overrides,
    )
    args = {
        "type": "failure",
        "title": "x",
        "content": "y",
    }
    with _bind_request() as state:
        body = _decode_block(
            await _dispatch(pipeline=pipeline, tool_name="submit_memory", args=args, state=state)
        )
    assert body["ok"] is False
    assert body["error"] == "type_not_allowed"
    assert "observation" in body["message"]


@pytest.mark.asyncio
async def test_metadata_schema_validation() -> None:
    stm = InMemorySTMStore()
    audit = CollectingAuditSink()
    usage = CollectingUsageSink()
    pipeline = _make_pipeline(
        stm_store=stm,
        audit_sink=audit,
        usage_sink=usage,
        memory_types=_failure_with_schema(),
    )
    args = {
        "type": "failure",
        "title": "x",
        "content": "y",
        "metadata": {},
    }
    with _bind_request() as state:
        body = _decode_block(
            await _dispatch(pipeline=pipeline, tool_name="submit_memory", args=args, state=state)
        )
    assert body["ok"] is False
    assert body["error"] == "invalid_metadata"
    assert "severity" in body["message"]


@pytest.mark.asyncio
async def test_oversize_content_rejected() -> None:
    stm = InMemorySTMStore()
    audit = CollectingAuditSink()
    usage = CollectingUsageSink()
    pipeline = _make_pipeline(
        stm_store=stm,
        audit_sink=audit,
        usage_sink=usage,
        max_content_bytes=64,
    )
    args = {
        "type": "observation",
        "title": "x",
        "content": "x" * 200,
    }
    with _bind_request() as state:
        body = _decode_block(
            await _dispatch(pipeline=pipeline, tool_name="submit_memory", args=args, state=state)
        )
    assert body["ok"] is False
    assert body["error"] == "invalid_args"
    assert "max_content_bytes" in body["message"]


@pytest.mark.asyncio
async def test_rate_limit_denied() -> None:
    stm = InMemorySTMStore()
    audit = CollectingAuditSink()
    usage = CollectingUsageSink()
    rate = TightRateLimiter(allow=False, retry_after=0.42)
    pipeline = _make_pipeline(
        stm_store=stm,
        audit_sink=audit,
        usage_sink=usage,
        rate_limiter=rate,
    )
    args = {
        "type": "observation",
        "title": "x",
        "content": "y",
    }
    with _bind_request() as state:
        body = _decode_block(
            await _dispatch(pipeline=pipeline, tool_name="submit_memory", args=args, state=state)
        )
    assert body["ok"] is False
    assert body["error"] == "rate_limited"
    assert body["retry_after_seconds"] == pytest.approx(0.42)
    rate_limited_events = [
        e for e in audit.events if e.payload.get("rate_limited") is True
    ]
    assert rate_limited_events, "rate-limited audit event missing"


def _build_app(
    *,
    context_store: Any | None = None,
    audit_sink: CollectingAuditSink | None = None,
    auth: Any | None = None,
    memory_types: tuple[MemoryType, ...] | None = None,
) -> AppHandle:
    sink = audit_sink or CollectingAuditSink()
    raw = RootConfig.model_validate({})
    resolved = ResolvedConfig(
        raw=raw,
        components={
            "auth": auth or StaticTokenAuth(token="test-token"),
            "admin_auth": None,
            "tenancy": SingleTenant(),
            "tenant_registry": StaticTenantRegistry(),
            "tenant_config_provider": StaticTenantConfigProvider(),
            "tenant_lifecycle": StaticTenantLifecycle(),
            "job_queue": InProcessJobQueue(),
            "secret_resolver": EnvSecretResolver(),
            "rate_limiter": NoOpRateLimiter(),
            "stm_store": InMemorySTMStore(),
            "ltm_store": InMemoryLTMStore(),
            "context_store": context_store,
            "dream_lease_store": InMemoryDreamLeaseStore(),
            "stm_serializer": None,
            "dream_engine": None,
        },
        component_lists={
            "usage_sinks": [CollectingUsageSink()],
            "audit_sinks": [sink],
            "mcp_tools": [],
            "triggers": [],
            "dream_gates": [],
            "hooks": [],
            **{f"hooks.{name}": [] for name in (
                "pre_dream",
                "post_dream",
                "pre_ltm_update",
                "post_ltm_update",
                "pre_context_update",
                "post_context_update",
                "pre_memory_submit",
                "post_memory_submit",
                "on_dream_failed",
                "on_dream_progress",
            )},
        },
        declared_multi_tenancy="auto",
    )
    if memory_types is not None:
        resolved.raw.__pydantic_extra__["memory_types"] = list(memory_types)
    return create_app(resolved)


@pytest.fixture
def context_store_root(tmp_path) -> Any:
    return tmp_path / "context"


@pytest_asyncio.fixture
async def app_handle_with_context(
    context_store_root,
) -> AsyncIterator[tuple[AppHandle, httpx.AsyncClient]]:
    from asgi_lifespan import LifespanManager  # noqa: PLC0415

    store = MarkdownContextStore(root=context_store_root)
    handle = _build_app(context_store=store)
    async with LifespanManager(handle.app) as manager:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=manager.app),
            base_url="http://test",
        ) as client:
            yield handle, client


@pytest_asyncio.fixture
async def app_handle_without_context() -> AsyncIterator[tuple[AppHandle, httpx.AsyncClient]]:
    from asgi_lifespan import LifespanManager  # noqa: PLC0415

    # InMemoryContextStore does NOT implement ContextReader.
    store = InMemoryContextStore()
    handle = _build_app(context_store=store)
    async with LifespanManager(handle.app) as manager:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=manager.app),
            base_url="http://test",
        ) as client:
            yield handle, client


@pytest.mark.asyncio
async def test_healthz_unauthenticated(
    app_handle_with_context: tuple[AppHandle, httpx.AsyncClient],
) -> None:
    _, client = app_handle_with_context
    response = await client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "auth" in body["components"]
    assert body["components"]["auth"]["ready"] is True


@pytest.mark.asyncio
async def test_context_route_serves_with_auth(
    app_handle_with_context: tuple[AppHandle, httpx.AsyncClient],
    context_store_root,
) -> None:
    _, client = app_handle_with_context
    (context_store_root / "AGENTS.md").write_text("# AGENTS\n\nhi", encoding="utf-8")
    response = await client.get(
        "/context/AGENTS.md",
        headers={"Authorization": "Bearer test-token"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/markdown")
    assert response.text == "# AGENTS\n\nhi"


@pytest.mark.asyncio
async def test_context_route_missing_returns_404(
    app_handle_with_context: tuple[AppHandle, httpx.AsyncClient],
) -> None:
    _, client = app_handle_with_context
    response = await client.get(
        "/context/missing.md",
        headers={"Authorization": "Bearer test-token"},
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_context_listing_with_prefix(
    app_handle_with_context: tuple[AppHandle, httpx.AsyncClient],
    context_store_root,
) -> None:
    _, client = app_handle_with_context
    (context_store_root / "AGENTS.md").write_text("hello", encoding="utf-8")
    response = await client.get(
        "/context/",
        headers={"Authorization": "Bearer test-token"},
    )
    assert response.status_code == 200
    body = response.json()
    assert "AGENTS.md" in body["files"]


@pytest.mark.asyncio
async def test_context_route_unauthenticated_returns_401(
    app_handle_with_context: tuple[AppHandle, httpx.AsyncClient],
) -> None:
    _, client = app_handle_with_context
    response = await client.get("/context/AGENTS.md")
    assert response.status_code == 401
    assert response.json() == {"error": "auth_failed"}


@pytest.mark.asyncio
async def test_options_preflight_skips_auth(
    app_handle_with_context: tuple[AppHandle, httpx.AsyncClient],
) -> None:
    # CORS preflight cannot include the Authorization header; CORSMiddleware
    # must answer it directly (200 + Access-Control-* headers) without ever
    # hitting auth or the /mcp mount's slash-redirect. Browser-based clients
    # like the MCP Inspector silently fail otherwise — they don't follow 3xx
    # on preflight, and they reject responses without Allow-Origin.
    _, client = app_handle_with_context
    response = await client.options(
        "/mcp",
        headers={
            "Origin": "http://localhost:6274",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") in {"*", "http://localhost:6274"}
    assert "POST" in response.headers.get("access-control-allow-methods", "")


@pytest.mark.asyncio
async def test_context_route_not_mounted_when_reader_absent(
    app_handle_without_context: tuple[AppHandle, httpx.AsyncClient],
) -> None:
    _, client = app_handle_without_context
    response = await client.get(
        "/context/AGENTS.md",
        headers={"Authorization": "Bearer test-token"},
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_no_admin_routes_in_v1(
    app_handle_with_context: tuple[AppHandle, httpx.AsyncClient],
) -> None:
    handle, _ = app_handle_with_context
    paths: list[str] = []
    for route in handle.app.router.routes:
        path = getattr(route, "path", None)
        if path:
            paths.append(path)
    for p in paths:
        assert not p.startswith("/admin"), f"unexpected admin route: {p}"


@pytest.mark.asyncio
async def test_secret_watcher_dispatches_on_version_change() -> None:
    resolver = RotatingResolver("GH_TOKEN", versions=["v1", "v1", "v2", "v2"])
    hook = CapturingRotationHook()
    watcher = SecretWatcher(
        resolver=resolver,
        hooks=[hook],
        poll_interval_seconds=0.05,
    )
    await watcher.start()
    try:
        for _ in range(40):
            if hook.events:
                break
            await asyncio.sleep(0.05)
    finally:
        await watcher.stop()
    assert hook.events == ["GH_TOKEN"]


@pytest.mark.asyncio
async def test_secret_watcher_no_hooks_is_noop() -> None:
    resolver = RotatingResolver("X", versions=["v1"])
    watcher = SecretWatcher(resolver=resolver, hooks=[])
    await watcher.start()
    await watcher.stop()
    assert resolver.calls == []


@pytest.mark.asyncio
async def test_mcp_route_rejects_bad_token(
    app_handle_with_context: tuple[AppHandle, httpx.AsyncClient],
) -> None:
    _, client = app_handle_with_context
    response = await client.post(
        "/mcp/",
        headers={"Authorization": "Bearer wrong"},
        json={
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "0"},
            },
            "id": 1,
        },
    )
    assert response.status_code == 401
    assert response.json() == {"error": "auth_failed"}


@pytest.mark.asyncio
async def test_mcp_initialize_and_list_tools_through_transport(
    app_handle_with_context: tuple[AppHandle, httpx.AsyncClient],
) -> None:
    _, client = app_handle_with_context
    init_response = await client.post(
        "/mcp/",
        headers={
            "Authorization": "Bearer test-token",
            "Accept": "application/json, text/event-stream",
        },
        json={
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "0"},
            },
            "id": 1,
        },
    )
    assert init_response.status_code == 200, init_response.text
    body = init_response.json()
    assert body["jsonrpc"] == "2.0"
    assert "result" in body
    assert body["result"]["protocolVersion"]


@pytest.mark.asyncio
async def test_auth_failed_audit_event_emitted(
    app_handle_with_context: tuple[AppHandle, httpx.AsyncClient],
) -> None:
    handle, client = app_handle_with_context
    response = await client.post(
        "/mcp/",
        headers={"Authorization": "Bearer wrong"},
        json={
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "0"},
            },
            "id": 1,
        },
    )
    assert response.status_code == 401
    audit_events = handle.mcp.pipeline.audit_sinks[0].events  # type: ignore[attr-defined]
    assert any(e.event_type == "auth.failed" for e in audit_events)
