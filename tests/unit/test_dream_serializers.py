from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
import pytest_asyncio

from dreamer.api.contexts import SerializeContext, SerializeServices
from dreamer.api.stores import STMSerializer
from dreamer.api.tenants import TenantScope
from dreamer.api.types import Memory, MemoryBatch
from dreamer.contrib.dream.serializers import (
    JsonlSerializer,
    MarkdownPerMemorySerializer,
)
from dreamer.testing.conformance import STMSerializerConformance


@pytest_asyncio.fixture(autouse=True)
async def _clear_tenant_scope() -> AsyncIterator[None]:
    TenantScope.clear()
    yield
    TenantScope.clear()


class TestMarkdownPerMemoryConformance(STMSerializerConformance):
    async def make_stm_serializer(self) -> AsyncIterator[STMSerializer]:
        yield MarkdownPerMemorySerializer()


class TestJsonlConformance(STMSerializerConformance):
    async def make_stm_serializer(self) -> AsyncIterator[STMSerializer]:
        yield JsonlSerializer()


def _make_memory(
    *,
    mid: str,
    title: str,
    content: str = "body",
    tags: list[str] | None = None,
    metadata: dict[str, object] | None = None,
) -> Memory:
    return Memory(
        id=mid,
        tenant_id="default",
        agent_id="agent-1",
        type="observation",
        title=title,
        content=content,
        tags=tags or [],
        metadata=metadata or {},
        submitted_at=datetime(2026, 5, 2, 10, 0, 0, tzinfo=UTC),
    )


def _batch(memories: list[Memory]) -> MemoryBatch:
    return MemoryBatch(
        lease_id="L1",
        tenant_id="default",
        memories=memories,
        snapshot_at=datetime(2026, 5, 2, 10, 5, 0, tzinfo=UTC),
    )


def _services() -> SerializeServices:
    async def emit(_msg: str, _payload: object) -> None:
        return None

    class _Resolver:
        multi_tenant = True

        async def get(self, *_a: object, **_kw: object) -> object:  # pragma: no cover
            raise NotImplementedError

    class _Sink:
        multi_tenant = True

        async def record(self, *_a: object, **_kw: object) -> None:
            return None

    return SerializeServices(
        emit_progress=emit,
        secrets=_Resolver(),  # type: ignore[arg-type]
        usage=_Sink(),  # type: ignore[arg-type]
        audit=_Sink(),  # type: ignore[arg-type]
        clock=lambda: datetime.now(UTC),
    )


def _ctx() -> SerializeContext:
    return SerializeContext(request_id="r1", tenant_id="default", lease_id="L1")


@pytest.mark.asyncio
async def test_markdown_per_memory_writes_one_file_per_memory(tmp_path: Path) -> None:
    s = MarkdownPerMemorySerializer()
    batch = _batch(
        [
            _make_memory(mid="m1", title="First memory"),
            _make_memory(mid="m2", title="Second memory"),
        ]
    )
    target = tmp_path / "inbox"
    with TenantScope.set("default"):
        await s.write(batch, target=target, ctx=_ctx(), services=_services())

    files = sorted(p.name for p in target.iterdir())
    assert files == ["m1-first-memory.md", "m2-second-memory.md"]


@pytest.mark.asyncio
async def test_markdown_per_memory_frontmatter_contains_canonical_fields(
    tmp_path: Path,
) -> None:
    s = MarkdownPerMemorySerializer()
    memory = _make_memory(
        mid="abc",
        title="Title with: colon",
        content="line1\nline2",
        tags=["alpha", "beta"],
        metadata={"source": "claude-code", "score": 0.9},
    )
    target = tmp_path / "inbox"
    with TenantScope.set("default"):
        await s.write(_batch([memory]), target=target, ctx=_ctx(), services=_services())

    text = (target / "abc-title-with-colon.md").read_text(encoding="utf-8")
    assert text.startswith("---\n")
    assert "id: abc" in text
    assert "tenant_id: default" in text
    assert "agent_id: agent-1" in text
    assert "type: observation" in text
    assert "tags: [alpha, beta]" in text
    assert "submitted_at: 2026-05-02T10:00:00Z" in text
    assert "metadata:\n  score: 0.9\n  source: claude-code" in text
    assert "\n---\n\nline1\nline2\n" in text


@pytest.mark.asyncio
async def test_markdown_per_memory_handles_missing_id(tmp_path: Path) -> None:
    s = MarkdownPerMemorySerializer()
    memory = Memory(
        tenant_id="default",
        agent_id="agent-1",
        type="observation",
        title="hello",
        content="body",
        submitted_at=datetime(2026, 5, 2, tzinfo=UTC),
    )
    target = tmp_path / "inbox"
    with TenantScope.set("default"):
        await s.write(_batch([memory]), target=target, ctx=_ctx(), services=_services())
    assert (target / "no-id-hello.md").is_file()


@pytest.mark.asyncio
async def test_markdown_per_memory_titleless_falls_back(tmp_path: Path) -> None:
    s = MarkdownPerMemorySerializer()
    target = tmp_path / "inbox"
    with TenantScope.set("default"):
        await s.write(
            _batch([_make_memory(mid="m1", title="    ")]),
            target=target,
            ctx=_ctx(),
            services=_services(),
        )
    assert (target / "m1.md").is_file()


@pytest.mark.asyncio
async def test_markdown_per_memory_prompt_fragment_mentions_inbox() -> None:
    s = MarkdownPerMemorySerializer()
    fragment = s.prompt_fragment(_batch([_make_memory(mid="m1", title="t")]))
    assert "inbox/" in fragment
    assert "<memory-id>" in fragment


@pytest.mark.asyncio
async def test_markdown_per_memory_rejects_wrong_tenant_scope(tmp_path: Path) -> None:
    s = MarkdownPerMemorySerializer()
    target = tmp_path / "inbox"
    with TenantScope.set("other"):
        with pytest.raises(Exception):
            await s.write(
                _batch([_make_memory(mid="m1", title="t")]),
                target=target,
                ctx=_ctx(),
                services=_services(),
            )


@pytest.mark.asyncio
async def test_jsonl_writes_one_line_per_memory(tmp_path: Path) -> None:
    s = JsonlSerializer()
    batch = _batch(
        [
            _make_memory(mid="m1", title="first"),
            _make_memory(mid="m2", title="second"),
        ]
    )
    target = tmp_path / "inbox"
    with TenantScope.set("default"):
        await s.write(batch, target=target, ctx=_ctx(), services=_services())

    text = (target / "batch.jsonl").read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if line]
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert [p["id"] for p in parsed] == ["m1", "m2"]
    assert all(p["tenant_id"] == "default" for p in parsed)


@pytest.mark.asyncio
async def test_jsonl_empty_batch_writes_empty_file(tmp_path: Path) -> None:
    s = JsonlSerializer()
    target = tmp_path / "inbox"
    with TenantScope.set("default"):
        await s.write(_batch([]), target=target, ctx=_ctx(), services=_services())
    out = target / "batch.jsonl"
    assert out.is_file()
    assert out.read_text(encoding="utf-8") == ""


@pytest.mark.asyncio
async def test_jsonl_prompt_fragment_describes_layout() -> None:
    s = JsonlSerializer()
    fragment = s.prompt_fragment(_batch([_make_memory(mid="m1", title="t")]))
    assert "inbox/batch.jsonl" in fragment


@pytest.mark.asyncio
async def test_jsonl_rejects_wrong_tenant_scope(tmp_path: Path) -> None:
    s = JsonlSerializer()
    target = tmp_path / "inbox"
    with TenantScope.set("other"):
        with pytest.raises(Exception):
            await s.write(
                _batch([_make_memory(mid="m1", title="t")]),
                target=target,
                ctx=_ctx(),
                services=_services(),
            )


def test_kinds_are_distinct() -> None:
    assert MarkdownPerMemorySerializer.kind == "markdown-per-memory"
    assert JsonlSerializer.kind == "jsonl"
    assert MarkdownPerMemorySerializer.kind != JsonlSerializer.kind
