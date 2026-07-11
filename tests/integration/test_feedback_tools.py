from __future__ import annotations

import contextlib
import json
import uuid
from typing import Any, ClassVar

import pytest

from dreamer.api.compat import implements
from dreamer.api.contexts import (
    ListUnconsumedContext,
    PreMemorySubmitContext,
    TenantConfigLookupContext,
)
from dreamer.api.hooks import PreMemorySubmitHook
from dreamer.api.runtime_state import RequestState
from dreamer.api.tenants import TenantScope
from dreamer.api.types import (
    DEFAULT_TENANT_ID,
    Memory,
    MemoryType,
    Principal,
    TenantConfig,
    TenantId,
)
from dreamer.contrib.mcp_tools.feedback import (
    CONTEXT_CONFIRMED_MEMORY_TYPE,
    CONTEXT_FLAGGED_MEMORY_TYPE,
    ConfirmContextTool,
    FlagContextTool,
)
from dreamer.contrib.tenancy.single import SingleTenant
from dreamer.server.mcp_app import MCPPipeline, _dispatch
from dreamer.server.runtime import HookRegistry
from dreamer.testing.fakes import (
    CollectingAuditSink,
    CollectingUsageSink,
    InMemorySTMStore,
    NoOpRateLimiter,
)


class StaticTenantConfigProvider:
    multi_tenant: ClassVar[bool] = False

    async def get(self, tenant_id: TenantId, *, ctx: TenantConfigLookupContext) -> TenantConfig:
        return TenantConfig()


class RecordingPreSubmit:
    multi_tenant: ClassVar[bool] = False

    def __init__(self) -> None:
        self.seen: list[Memory] = []

    async def on_pre_memory_submit(self, *, ctx: PreMemorySubmitContext) -> None:
        self.seen.extend(ctx.memories)


implements(PreMemorySubmitHook, version=1)(RecordingPreSubmit)


def _memory_types() -> tuple[MemoryType, ...]:
    return (
        MemoryType(name="observation", description="A general observation."),
        CONTEXT_CONFIRMED_MEMORY_TYPE,
        CONTEXT_FLAGGED_MEMORY_TYPE,
    )


def _make_pipeline(
    *,
    stm_store: InMemorySTMStore,
    audit_sink: CollectingAuditSink | None = None,
    pre_hooks: list[PreMemorySubmitHook] | None = None,
    memory_types: tuple[MemoryType, ...] | None = None,
) -> MCPPipeline:
    registry = HookRegistry()
    for h in pre_hooks or []:
        registry.add("pre_memory_submit", h)
    return MCPPipeline(
        tenancy=SingleTenant(),
        stm_store=stm_store,
        tenant_config_provider=StaticTenantConfigProvider(),
        rate_limiter=NoOpRateLimiter(),
        hook_registry=registry,
        audit_sinks=[audit_sink] if audit_sink else [],
        usage_sinks=[CollectingUsageSink()],
        mcp_tools=[ConfirmContextTool(), FlagContextTool()],
        memory_types=memory_types if memory_types is not None else _memory_types(),
    )


@contextlib.contextmanager
def _bind_request(principal_id: str = "agent-1") -> Any:
    state = RequestState(
        principal=Principal(id=principal_id, tenant_id=DEFAULT_TENANT_ID),
        request_id=uuid.uuid4().hex,
    )
    with RequestState.bind(state):
        yield state


async def _call(
    pipeline: MCPPipeline,
    tool: str,
    args: dict[str, Any],
    *,
    principal_id: str = "agent-1",
) -> dict[str, Any]:
    with _bind_request(principal_id) as state:
        blocks = await _dispatch(pipeline=pipeline, tool_name=tool, args=args, state=state)
    return json.loads(blocks[0].text)


async def _unconsumed(stm: InMemorySTMStore) -> list[Memory]:
    with TenantScope.set(DEFAULT_TENANT_ID):
        return await stm.list_unconsumed(
            ctx=ListUnconsumedContext(request_id="t", tenant_id=DEFAULT_TENANT_ID)
        )


@pytest.mark.asyncio
async def test_confirm_dedupes_within_same_day_and_principal() -> None:
    stm = InMemorySTMStore()
    pipeline = _make_pipeline(stm_store=stm)

    first = await _call(pipeline, "confirm_context", {"target": "test-db-reset"})
    second = await _call(pipeline, "confirm_context", {"target": "test-db-reset"})

    assert first["ok"] is True
    assert first["result"]["deduplicated"] is False
    assert second["ok"] is True
    assert second["result"]["deduplicated"] is True
    assert second["result"]["memory_id"] == first["result"]["memory_id"]

    memories = await _unconsumed(stm)
    assert len(memories) == 1
    assert memories[0].type == "context_confirmed"
    assert memories[0].metadata == {"target": "test-db-reset"}


@pytest.mark.asyncio
async def test_distinct_principals_confirm_independently() -> None:
    stm = InMemorySTMStore()
    pipeline = _make_pipeline(stm_store=stm)

    a = await _call(pipeline, "confirm_context", {"target": "test-db-reset"}, principal_id="a")
    b = await _call(pipeline, "confirm_context", {"target": "test-db-reset"}, principal_id="b")

    assert a["result"]["deduplicated"] is False
    assert b["result"]["deduplicated"] is False
    assert len(await _unconsumed(stm)) == 2


@pytest.mark.asyncio
async def test_malformed_target_rejected_unknown_target_accepted() -> None:
    stm = InMemorySTMStore()
    pipeline = _make_pipeline(stm_store=stm)

    malformed = await _call(pipeline, "confirm_context", {"target": "Not A Slug!"})
    assert malformed["ok"] is False
    assert malformed["error"] == "invalid_args"
    assert await _unconsumed(stm) == []

    unknown = await _call(pipeline, "confirm_context", {"target": "no-such-topic"})
    assert unknown["ok"] is True
    assert len(await _unconsumed(stm)) == 1


@pytest.mark.asyncio
async def test_anchored_flag_carries_evidence() -> None:
    stm = InMemorySTMStore()
    pipeline = _make_pipeline(stm_store=stm)

    body = await _call(
        pipeline,
        "flag_context",
        {
            "observation": "Context said reset with make db-reset; the target was removed.",
            "targets": ["test-db-reset"],
            "quote": "run `make db-reset` before integration tests",
        },
    )
    assert body["ok"] is True
    assert body["result"]["anchored"] is True

    memories = await _unconsumed(stm)
    assert len(memories) == 1
    flag = memories[0]
    assert flag.type == "context_flagged"
    assert flag.content.startswith("Context said reset with make db-reset")
    assert flag.metadata["targets"] == ["test-db-reset"]
    assert flag.metadata["quote"] == "run `make db-reset` before integration tests"


@pytest.mark.asyncio
async def test_unanchored_flag_accepted() -> None:
    stm = InMemorySTMStore()
    pipeline = _make_pipeline(stm_store=stm)

    body = await _call(
        pipeline,
        "flag_context",
        {"observation": "Two sections of the bundle contradict each other about retries."},
    )
    assert body["ok"] is True
    assert body["result"]["anchored"] is False

    memories = await _unconsumed(stm)
    assert len(memories) == 1
    assert memories[0].metadata == {}
    assert memories[0].title == "context flagged: unanchored"


@pytest.mark.asyncio
async def test_feedback_tools_honor_submit_pipeline() -> None:
    stm = InMemorySTMStore()
    audit = CollectingAuditSink()
    hook = RecordingPreSubmit()
    pipeline = _make_pipeline(stm_store=stm, audit_sink=audit, pre_hooks=[hook])

    await _call(pipeline, "confirm_context", {"target": "topic-a"})
    assert [m.type for m in hook.seen] == ["context_confirmed"]
    assert any(e.event_type == "mcp.confirm_context" for e in audit.events)

    # An equivalent direct submit_memory call goes through the same hook.
    await _call(
        pipeline,
        "submit_memory",
        {
            "type": "context_confirmed",
            "title": "context confirmed: topic-b",
            "content": "",
            "metadata": {"target": "topic-b"},
        },
    )
    assert [m.type for m in hook.seen] == ["context_confirmed", "context_confirmed"]
    assert any(e.event_type == "mcp.submit_memory" for e in audit.events)


@pytest.mark.asyncio
async def test_direct_submit_memory_validates_feedback_metadata_schema() -> None:
    stm = InMemorySTMStore()
    pipeline = _make_pipeline(stm_store=stm)

    body = await _call(
        pipeline,
        "submit_memory",
        {
            "type": "context_confirmed",
            "title": "missing target",
            "content": "",
            "metadata": {},
        },
    )
    assert body["ok"] is False
    assert body["error"] == "invalid_metadata"


@pytest.mark.asyncio
async def test_undeclared_feedback_type_errors_clearly() -> None:
    stm = InMemorySTMStore()
    pipeline = _make_pipeline(
        stm_store=stm,
        memory_types=(MemoryType(name="observation", description="only obs"),),
    )

    body = await _call(pipeline, "confirm_context", {"target": "test-db-reset"})
    assert body["ok"] is False
    assert body["error"] == "type_not_allowed"
    assert "context_confirmed" in body["message"]
    assert await _unconsumed(stm) == []
