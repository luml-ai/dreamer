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
    type: str = "observation",
    content: str = "body",
    tags: list[str] | None = None,
    metadata: dict[str, object] | None = None,
    submitted_at: datetime | None = None,
) -> Memory:
    return Memory(
        id=mid,
        tenant_id="default",
        agent_id="agent-1",
        type=type,
        title=title,
        content=content,
        tags=tags or [],
        metadata=metadata or {},
        submitted_at=submitted_at or datetime(2026, 5, 2, 10, 0, 0, tzinfo=UTC),
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


def _mixed_batch() -> MemoryBatch:
    def confirmation(mid: str, target: object, *, hour: int) -> Memory:
        return _make_memory(
            mid=mid,
            title=f"context confirmed: {mid}",
            type="context_confirmed",
            content="",
            metadata={"target": target},
            submitted_at=datetime(2026, 5, 2, hour, 0, 0, tzinfo=UTC),
        )

    return _batch(
        [
            confirmation("c1", "topic-a", hour=8),
            confirmation("c2", "topic-a", hour=9),
            confirmation("c3", "Not A Slug!", hour=10),
            _make_memory(
                mid="f1",
                title="context flagged: topic-b",
                type="context_flagged",
                content="Context said X; observed Y.",
                metadata={"targets": ["topic-b"], "quote": "do X"},
            ),
            _make_memory(mid="m1", title="first obs"),
            _make_memory(mid="m2", title="second obs"),
        ]
    )


@pytest.mark.asyncio
async def test_markdown_mixed_batch_aggregates_confirmations_and_keeps_flags(
    tmp_path: Path,
) -> None:
    s = MarkdownPerMemorySerializer()
    target = tmp_path / "inbox"
    with TenantScope.set("default"):
        await s.write(_mixed_batch(), target=target, ctx=_ctx(), services=_services())

    # Observations serialize exactly as today, at the inbox root.
    root_files = sorted(p.name for p in target.iterdir() if p.is_file())
    assert root_files == ["m1-first-obs.md", "m2-second-obs.md"]

    # No individual confirmation files anywhere.
    all_files = [p.name for p in target.rglob("*") if p.is_file()]
    assert not any(name.startswith(("c1-", "c2-", "c3-")) for name in all_files)

    report = (target / "feedback" / "confirmations.md").read_text(encoding="utf-8")
    assert "`topic-a`: 2 confirmation(s), latest 2026-05-02T09:00:00Z" in report
    assert "Not A Slug!" in report

    flag_files = sorted(p.name for p in (target / "feedback" / "flags").iterdir())
    assert flag_files == ["f1-context-flagged-topic-b.md"]
    flag_text = (target / "feedback" / "flags" / flag_files[0]).read_text(encoding="utf-8")
    assert "Context said X; observed Y." in flag_text
    assert "targets: [topic-b]" in flag_text
    assert "quote: " in flag_text

    fragment = s.prompt_fragment(_mixed_batch())
    assert "2 markdown file(s)" in fragment
    assert "inbox/feedback/confirmations.md" in fragment
    assert "inbox/feedback/flags/" in fragment
    assert "unanchored" in fragment


@pytest.mark.asyncio
async def test_jsonl_mixed_batch_aggregates_confirmations_and_keeps_flags(
    tmp_path: Path,
) -> None:
    s = JsonlSerializer()
    target = tmp_path / "inbox"
    with TenantScope.set("default"):
        await s.write(_mixed_batch(), target=target, ctx=_ctx(), services=_services())

    lines = [
        json.loads(line)
        for line in (target / "batch.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert [p["id"] for p in lines] == ["m1", "m2", "f1"]
    flag = next(p for p in lines if p["type"] == "context_flagged")
    assert flag["metadata"] == {"targets": ["topic-b"], "quote": "do X"}

    report = (target / "feedback" / "confirmations.md").read_text(encoding="utf-8")
    assert "`topic-a`: 2 confirmation(s), latest 2026-05-02T09:00:00Z" in report
    assert "Not A Slug!" in report

    fragment = s.prompt_fragment(_mixed_batch())
    assert "3 JSON object(s)" in fragment
    assert "context_flagged" in fragment
    assert "inbox/feedback/confirmations.md" in fragment


@pytest.mark.asyncio
async def test_feedback_free_batch_writes_no_feedback_dir(tmp_path: Path) -> None:
    for serializer in (MarkdownPerMemorySerializer(), JsonlSerializer()):
        target = tmp_path / f"inbox-{serializer.kind}"
        with TenantScope.set("default"):
            await serializer.write(
                _batch([_make_memory(mid="m1", title="obs")]),
                target=target,
                ctx=_ctx(),
                services=_services(),
            )
        assert not (target / "feedback").exists()


def test_kinds_are_distinct() -> None:
    assert MarkdownPerMemorySerializer.kind == "markdown-per-memory"
    assert JsonlSerializer.kind == "jsonl"
    assert MarkdownPerMemorySerializer.kind != JsonlSerializer.kind
