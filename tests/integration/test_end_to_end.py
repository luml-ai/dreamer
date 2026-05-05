from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar

import git as gitpy
import httpx
import pytest
import pytest_asyncio
from asgi_lifespan import LifespanManager

from dreamer.api.auth import AuthBackend
from dreamer.api.compat import implements
from dreamer.api.config import ResolvedConfig, RootConfig
from dreamer.api.contexts import (
    AuthContext,
    CountContext,
    GetContextPendingContext,
    LifecycleContext,
    ListUnconsumedContext,
    SecretContext,
    SetContextPendingContext,
    SubmitContext,
)
from dreamer.api.errors import AuthError
from dreamer.api.secrets import SecretResolver
from dreamer.api.tenants import TenantScope
from dreamer.api.types import (
    DEFAULT_TENANT_ID,
    Diff,
    DreamJob,
    Memory,
    MemoryType,
    Principal,
    SecretValue,
    TenantId,
)
from dreamer.contrib.context.markdown import MarkdownContextStore
from dreamer.contrib.hooks.git import GitCommit
from dreamer.contrib.jobs.inproc import InProcessJobQueue
from dreamer.contrib.tenancy.single import SingleTenant
from dreamer.contrib.tenants.static import (
    StaticTenantConfigProvider,
    StaticTenantLifecycle,
    StaticTenantRegistry,
)
from dreamer.server.app import AppHandle, create_app
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
    NoOpRateLimiter,
)


@implements(AuthBackend, version=1)
class _StaticTokenAuth:
    multi_tenant: ClassVar[bool] = False

    def __init__(self, *, token: str, tenant_id: TenantId = DEFAULT_TENANT_ID) -> None:
        self.token = token
        self.tenant_id = tenant_id

    async def authenticate(self, request: Any, *, ctx: AuthContext) -> Principal:
        header = request.headers.get("authorization") or ""
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or token.strip() != self.token:
            raise AuthError("auth_failed: bad token")
        return Principal(id="agent-e2e", tenant_id=self.tenant_id)


@implements(SecretResolver, version=1)
class _NullSecretResolver:
    multi_tenant: ClassVar[bool] = True

    async def get(
        self, name: str, *, tenant_id: TenantId | None, ctx: SecretContext
    ) -> SecretValue:
        return SecretValue(value="")


def _make_resolved(
    *,
    auth: _StaticTokenAuth,
    stm_store: InMemorySTMStore,
    ltm_store: InMemoryLTMStore,
    context_store: MarkdownContextStore,
    audit_sink: CollectingAuditSink,
    usage_sink: CollectingUsageSink,
    memory_types: tuple[MemoryType, ...],
    post_dream_hooks: list[Any] | None = None,
) -> ResolvedConfig:
    raw = RootConfig.model_validate({})
    extra = raw.__pydantic_extra__
    assert extra is not None
    extra["memory_types"] = list(memory_types)
    return ResolvedConfig(
        raw=raw,
        components={
            "auth": auth,
            "admin_auth": None,
            "tenancy": SingleTenant(),
            "tenant_registry": StaticTenantRegistry([DEFAULT_TENANT_ID]),
            "tenant_config_provider": StaticTenantConfigProvider(),
            "tenant_lifecycle": StaticTenantLifecycle(),
            "job_queue": InProcessJobQueue(),
            "secret_resolver": _NullSecretResolver(),
            "rate_limiter": NoOpRateLimiter(),
            "stm_store": stm_store,
            "ltm_store": ltm_store,
            "context_store": context_store,
            "dream_lease_store": InMemoryDreamLeaseStore(default_ttl_seconds=60),
            "stm_serializer": None,
            "dream_engine": None,
        },
        component_lists={
            "usage_sinks": [usage_sink],
            "audit_sinks": [audit_sink],
            "mcp_tools": [],
            "triggers": [],
            "dream_gates": [],
            "hooks": [],
            "hooks.pre_dream": [],
            "hooks.post_dream": list(post_dream_hooks or []),
            "hooks.pre_ltm_update": [],
            "hooks.post_ltm_update": [],
            "hooks.pre_context_update": [],
            "hooks.post_context_update": [],
            "hooks.pre_memory_submit": [],
            "hooks.post_memory_submit": [],
            "hooks.on_dream_failed": [],
            "hooks.on_dream_progress": [],
        },
        declared_multi_tenancy="auto",
    )


def _build_orchestrator(
    *,
    stm_store: InMemorySTMStore,
    ltm_store: InMemoryLTMStore,
    context_store: MarkdownContextStore,
    leases: InMemoryDreamLeaseStore,
    job_queue: InProcessJobQueue,
    audit_sink: CollectingAuditSink,
    usage_sink: CollectingUsageSink,
    secret_resolver: _NullSecretResolver,
    engine: DeterministicDreamEngine,
    post_dream_hooks: list[Any] | None = None,
) -> Orchestrator:
    hook_registry = HookRegistry()
    for h in post_dream_hooks or []:
        hook_registry.add("post_dream", h)
    return Orchestrator(
        stm_store=stm_store,
        ltm_store=ltm_store,
        context_store=context_store,
        dream_lease_store=leases,
        ltm_phase_runner=engine,
        context_phase_runner=engine,
        tenant_registry=StaticTenantRegistry([DEFAULT_TENANT_ID]),
        tenant_config_provider=StaticTenantConfigProvider(),
        job_queue=job_queue,
        hook_registry=hook_registry,
        audit_sinks=[audit_sink],
        usage_sinks=[usage_sink],
        secret_resolver=secret_resolver,
        dream_gates=[],
        stm_retention=StmRetentionConfig(keep_days=None, cadence_seconds=86400),
        default_lease_ttl_seconds=60.0,
        heartbeat_interval_seconds=10.0,
    )


@pytest.fixture
def workspace_dir(tmp_path: Path) -> Path:
    """Tmp git repo with `memory/` and `context/` subdirs ready for GitCommit."""
    repo_path = tmp_path / "workspace"
    repo_path.mkdir()
    repo = gitpy.Repo.init(str(repo_path), initial_branch="main")
    repo.config_writer().set_value("user", "email", "tests@dreamer.test").release()
    repo.config_writer().set_value("user", "name", "Tests").release()
    # Seed an initial commit so `main` exists as a real branch for GitCommit.
    (repo_path / ".gitkeep").write_text("", encoding="utf-8")
    repo.index.add([".gitkeep"])
    repo.index.commit("seed")
    (repo_path / "memory").mkdir()
    (repo_path / "context").mkdir()
    return repo_path


@pytest_asyncio.fixture
async def e2e_app(
    workspace_dir: Path,
) -> AsyncIterator[
    tuple[
        AppHandle,
        httpx.AsyncClient,
        Orchestrator,
        InMemorySTMStore,
        InMemoryLTMStore,
        MarkdownContextStore,
        Path,
        CollectingAuditSink,
    ]
]:
    """The app and orchestrator share store instances so MCP-submitted memories
    surface in batch claims, and orchestrator commits surface in /context reads."""
    auth = _StaticTokenAuth(token="e2e-token")
    audit_sink = CollectingAuditSink()
    usage_sink = CollectingUsageSink()
    memory_types = (
        MemoryType(name="failure", description="An unexpected failure."),
        MemoryType(name="observation", description="A general observation."),
    )
    stm_store = InMemorySTMStore()
    ltm_store = InMemoryLTMStore(root=workspace_dir / "memory")
    context_store = MarkdownContextStore(root=workspace_dir / "context")

    git_hook = GitCommit(
        repo=workspace_dir,
        branch="dreamer",
        base_branch="main",
        push=False,
        expect_clean_branch=False,
    )

    resolved = _make_resolved(
        auth=auth,
        stm_store=stm_store,
        ltm_store=ltm_store,
        context_store=context_store,
        audit_sink=audit_sink,
        usage_sink=usage_sink,
        memory_types=memory_types,
        post_dream_hooks=[git_hook],
    )
    handle = create_app(resolved)

    leases = resolved.components["dream_lease_store"]
    job_queue = resolved.components["job_queue"]
    secrets = resolved.components["secret_resolver"]
    engine = DeterministicDreamEngine()
    orchestrator = _build_orchestrator(
        stm_store=stm_store,
        ltm_store=ltm_store,
        context_store=context_store,
        leases=leases,
        job_queue=job_queue,
        audit_sink=audit_sink,
        usage_sink=usage_sink,
        secret_resolver=secrets,
        engine=engine,
        post_dream_hooks=[git_hook],
    )

    async with LifespanManager(handle.app) as manager:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=manager.app),
            base_url="http://test",
        ) as client:
            await orchestrator.start(
                ctx=LifecycleContext(request_id="test.start")
            )
            try:
                yield (
                    handle,
                    client,
                    orchestrator,
                    stm_store,
                    ltm_store,
                    context_store,
                    workspace_dir,
                    audit_sink,
                )
            finally:
                await orchestrator.stop(
                    ctx=LifecycleContext(request_id="test.stop")
                )


async def _initialize_mcp(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/mcp/",
        headers={
            "Authorization": "Bearer e2e-token",
            "Accept": "application/json, text/event-stream",
        },
        json={
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "e2e-test", "version": "0"},
            },
            "id": 1,
        },
    )
    assert response.status_code == 200, response.text


async def _submit_via_mcp(
    client: httpx.AsyncClient,
    *,
    title: str,
    content: str,
    type_name: str = "observation",
) -> dict[str, Any]:
    response = await client.post(
        "/mcp/",
        headers={
            "Authorization": "Bearer e2e-token",
            "Accept": "application/json, text/event-stream",
        },
        json={
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "submit_memory",
                "arguments": {
                    "type": type_name,
                    "title": title,
                    "content": content,
                },
            },
            "id": 42,
        },
    )
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


@pytest.mark.asyncio
async def test_full_dream_lifecycle_end_to_end(
    e2e_app: tuple[
        AppHandle,
        httpx.AsyncClient,
        Orchestrator,
        InMemorySTMStore,
        InMemoryLTMStore,
        MarkdownContextStore,
        Path,
        CollectingAuditSink,
    ],
) -> None:
    handle, client, orch, stm_store, ltm_store, context_store, workspace_dir, audit_sink = (
        e2e_app
    )

    await _initialize_mcp(client)

    for i in range(3):
        body = await _submit_via_mcp(
            client,
            title=f"observation #{i}",
            content=f"E2E memory {i} — captured at {datetime.now(UTC).isoformat()}",
        )
        # Streamable-http returns SSE-wrapped JSON-RPC; parse the result.
        result = body["result"]
        text = result["content"][0]["text"]
        parsed = json.loads(text)
        assert parsed["ok"] is True, parsed
        assert len(parsed["result"]["submitted"]) == 1

    with TenantScope.set(DEFAULT_TENANT_ID):
        count = await stm_store.count_unconsumed(
            ctx=CountContext(request_id="check", tenant_id=DEFAULT_TENANT_ID)
        )
    assert count == 3

    # Run handler synchronously so completion is deterministic.
    await orch._handle_job(  # noqa: SLF001 — same path control.trigger_dream uses
        DreamJob(tenant_id=DEFAULT_TENANT_ID, trigger_name="external")
    )

    with TenantScope.set(DEFAULT_TENANT_ID):
        unconsumed = await stm_store.list_unconsumed(
            ctx=ListUnconsumedContext(
                request_id="check", tenant_id=DEFAULT_TENANT_ID
            )
        )
    assert unconsumed == []

    audit_types = [e.event_type for e in audit_sink.events]
    assert "dream.lease_acquired" in audit_types
    assert "dream.batch_claimed" in audit_types
    assert "dream.ltm_committed" in audit_types
    assert "dream.context_committed" in audit_types

    response = await client.get(
        "/context/AGENTS.md",
        headers={"Authorization": "Bearer e2e-token"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/markdown")
    assert "AGENTS" in response.text
    assert "LTM diff added" in response.text

    repo = gitpy.Repo(str(workspace_dir))
    assert repo.active_branch.name == "dreamer"
    commits = list(repo.iter_commits("dreamer"))
    assert len(commits) >= 2  # seed commit + at least one dreamer commit
    head_message = str(commits[0].message)
    assert "dreamer" in head_message
    committed_paths = [str(p) for p in commits[0].stats.files.keys()]
    assert any("context/" in p for p in committed_paths)


@pytest.mark.asyncio
async def test_resume_context_against_injected_watermark(
    e2e_app: tuple[
        AppHandle,
        httpx.AsyncClient,
        Orchestrator,
        InMemorySTMStore,
        InMemoryLTMStore,
        MarkdownContextStore,
        Path,
        CollectingAuditSink,
    ],
) -> None:
    _, _, orch, _stm, ltm_store, context_store, _, _ = e2e_app

    diff = Diff(
        added=["topics/example.md"],
        modified=[],
        deleted=[],
    )
    with TenantScope.set(DEFAULT_TENANT_ID):
        await ltm_store.set_context_pending(
            diff,
            ctx=SetContextPendingContext(
                request_id="planted", tenant_id=DEFAULT_TENANT_ID
            ),
        )

    result = await orch.resume_context(DEFAULT_TENANT_ID)
    assert result["status"] == "ok"

    with TenantScope.set(DEFAULT_TENANT_ID):
        wm = await ltm_store.get_context_pending(
            ctx=GetContextPendingContext(
                request_id="check", tenant_id=DEFAULT_TENANT_ID
            )
        )
    assert wm is None

    second = await orch.resume_context(DEFAULT_TENANT_ID)
    assert second["status"] == "no_watermark"


@pytest.mark.asyncio
async def test_purge_consumed_via_orchestrator(
    e2e_app: tuple[
        AppHandle,
        httpx.AsyncClient,
        Orchestrator,
        InMemorySTMStore,
        InMemoryLTMStore,
        MarkdownContextStore,
        Path,
        CollectingAuditSink,
    ],
) -> None:
    _, client, orch, stm_store, _ltm, _ctx, _, _ = e2e_app

    await _initialize_mcp(client)
    for i in range(2):
        await _submit_via_mcp(client, title=f"to-purge {i}", content="x")

    await orch._handle_job(  # noqa: SLF001
        DreamJob(tenant_id=DEFAULT_TENANT_ID, trigger_name="external")
    )

    assert len(stm_store._memories) == 2  # noqa: SLF001
    for mem in stm_store._memories.values():  # noqa: SLF001
        assert mem.consumed_at is not None

    # Future `before` so every consumed row falls inside it.
    removed = await orch.purge_tenant(
        DEFAULT_TENANT_ID, before=datetime.now(UTC) + timedelta(days=1)
    )
    assert removed == 2
    assert stm_store._memories == {}  # noqa: SLF001


@pytest.mark.asyncio
async def test_tenancy_leakage_blocked_by_scope() -> None:
    """Multi-tenant in-memory fakes call ``TenantScope.assert_matches`` on
    every public method. Submitting under one scope then reading under
    another must raise — this asserts that contract end-to-end."""
    stm_store = InMemorySTMStore()

    async def _submit_for(tenant: TenantId, title: str) -> Memory:
        with TenantScope.set(tenant):
            return await stm_store.submit(
                Memory(
                    tenant_id=tenant,
                    agent_id="agent",
                    type="observation",
                    title=title,
                    content="x",
                    submitted_at=datetime.now(UTC),
                ),
                ctx=SubmitContext(request_id="sub", tenant_id=tenant),
            )

    a_mem = await _submit_for("tenant-a", "A's secret")
    b_mem = await _submit_for("tenant-b", "B's secret")
    assert a_mem.tenant_id == "tenant-a"
    assert b_mem.tenant_id == "tenant-b"

    with TenantScope.set("tenant-a"):
        own = await stm_store.list_unconsumed(
            ctx=ListUnconsumedContext(request_id="r", tenant_id="tenant-a")
        )
        assert {m.title for m in own} == {"A's secret"}

        # Cross-tenant ctx (asking for tenant-b while scope is tenant-a) raises.
        with pytest.raises(Exception):  # noqa: BLE001 — TenantScopeError or similar
            await stm_store.list_unconsumed(
                ctx=ListUnconsumedContext(request_id="r", tenant_id="tenant-b")
            )

    # Without any scope, every store call must raise.
    TenantScope.clear()
    with pytest.raises(Exception):  # noqa: BLE001
        await stm_store.count_unconsumed(
            ctx=CountContext(request_id="r", tenant_id="tenant-a")
        )


@pytest.mark.asyncio
async def test_orchestrator_does_not_leak_across_tenants() -> None:
    stm_store = InMemorySTMStore()
    ltm_store = InMemoryLTMStore()
    context_store = InMemoryContextStore()
    leases = InMemoryDreamLeaseStore(default_ttl_seconds=60.0)
    job_queue = InProcessJobQueue()
    audit_sink = CollectingAuditSink()
    usage_sink = CollectingUsageSink()
    engine = DeterministicDreamEngine()

    orch = Orchestrator(
        stm_store=stm_store,
        ltm_store=ltm_store,
        context_store=context_store,
        dream_lease_store=leases,
        ltm_phase_runner=engine,
        context_phase_runner=engine,
        tenant_registry=StaticTenantRegistry(["tenant-a", "tenant-b"]),
        tenant_config_provider=StaticTenantConfigProvider(),
        job_queue=job_queue,
        hook_registry=HookRegistry(),
        audit_sinks=[audit_sink],
        usage_sinks=[usage_sink],
        secret_resolver=_NullSecretResolver(),
        dream_gates=[],
        stm_retention=StmRetentionConfig(keep_days=None, cadence_seconds=86400),
        default_lease_ttl_seconds=60.0,
        heartbeat_interval_seconds=10.0,
    )
    await orch.start(ctx=LifecycleContext(request_id="t.start"))
    try:
        for i in range(2):
            with TenantScope.set("tenant-a"):
                await stm_store.submit(
                    Memory(
                        tenant_id="tenant-a",
                        agent_id="a",
                        type="observation",
                        title=f"A-{i}",
                        content="x",
                        submitted_at=datetime.now(UTC),
                    ),
                    ctx=SubmitContext(request_id="s", tenant_id="tenant-a"),
                )
        for i in range(3):
            with TenantScope.set("tenant-b"):
                await stm_store.submit(
                    Memory(
                        tenant_id="tenant-b",
                        agent_id="b",
                        type="observation",
                        title=f"B-{i}",
                        content="x",
                        submitted_at=datetime.now(UTC),
                    ),
                    ctx=SubmitContext(request_id="s", tenant_id="tenant-b"),
                )

        await orch._handle_job(  # noqa: SLF001
            DreamJob(tenant_id="tenant-a", trigger_name="external")
        )

        with TenantScope.set("tenant-a"):
            a_remaining = await stm_store.count_unconsumed(
                ctx=CountContext(request_id="r", tenant_id="tenant-a")
            )
        with TenantScope.set("tenant-b"):
            b_remaining = await stm_store.count_unconsumed(
                ctx=CountContext(request_id="r", tenant_id="tenant-b")
            )
        assert a_remaining == 0, "A's memories should be consumed"
        assert b_remaining == 3, "B's memories must NOT be consumed by A's dream"

        await orch._handle_job(  # noqa: SLF001
            DreamJob(tenant_id="tenant-b", trigger_name="external")
        )
        with TenantScope.set("tenant-a"):
            assert (
                await stm_store.count_unconsumed(
                    ctx=CountContext(request_id="r", tenant_id="tenant-a")
                )
            ) == 0
        with TenantScope.set("tenant-b"):
            assert (
                await stm_store.count_unconsumed(
                    ctx=CountContext(request_id="r", tenant_id="tenant-b")
                )
            ) == 0

        tenants_seen = {c.tenant_id for c in engine.calls}
        assert tenants_seen == {"tenant-a", "tenant-b"}
    finally:
        await orch.stop(ctx=LifecycleContext(request_id="t.stop"))
