from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pytest
import pytest_asyncio

from dreamer.api.contexts import (
    ContextPhaseContext,
    ContextPhaseServices,
    LTMPhaseContext,
    LTMPhaseServices,
)
from dreamer.api.errors import DreamFailedError
from dreamer.api.tenants import TenantScope
from dreamer.api.types import Diff, Memory, MemoryBatch, UsageEvent
from dreamer.contrib.dream._docker import DockerClaudeAgentRunner
from dreamer.contrib.dream._local import (
    AgentRunResult,
    ClaudeAgentRunner,
    LocalClaudeAgentRunner,
)
from dreamer.contrib.dream.claude_agent import ClaudeAgentDreamEngine
from dreamer.contrib.dream.serializers import (
    JsonlSerializer,
    MarkdownPerMemorySerializer,
)


@pytest_asyncio.fixture(autouse=True)
async def _clear_tenant_scope() -> AsyncIterator[None]:
    TenantScope.clear()
    yield
    TenantScope.clear()


@dataclass
class _RecordingRunner(ClaudeAgentRunner):
    name: str = "stub"
    side_effect: object = None
    invocations: list[Mapping[str, object]] = field(default_factory=list)
    mutate_path: tuple[str, str] | None = None  # (relpath, content)
    delete_paths: tuple[str, ...] = ()
    tokens_in: int | None = 100
    tokens_out: int | None = 50

    async def run(
        self,
        *,
        prompt: str,
        sandbox: Path,
        timeout_seconds: float,
        env: Mapping[str, str] | None = None,
    ) -> AgentRunResult:
        self.invocations.append(
            {
                "prompt": prompt,
                "sandbox": sandbox,
                "timeout_seconds": timeout_seconds,
                "env": env,
            }
        )
        if isinstance(self.side_effect, BaseException):
            raise self.side_effect
        if self.mutate_path is not None:
            rel, content = self.mutate_path
            target = sandbox / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        for rel in self.delete_paths:
            target = sandbox / rel
            if target.is_file():
                target.unlink()
        return AgentRunResult(tokens_in=self.tokens_in, tokens_out=self.tokens_out, raw=None)


@dataclass
class _StubWorkspace:
    id: str
    tenant_id: str
    path: Path
    metadata: Mapping[str, object] = field(default_factory=dict)

    async def file_view(self) -> Path:
        self.path.mkdir(parents=True, exist_ok=True)
        return self.path


class _RecordingUsageSink:
    multi_tenant = True

    def __init__(self) -> None:
        self.events: list[UsageEvent] = []

    async def record(self, event: UsageEvent, *, ctx: object) -> None:
        self.events.append(event)


class _NoopAuditSink:
    multi_tenant = True

    async def record(self, *_a: object, **_kw: object) -> None:
        return None


class _NoopSecretResolver:
    multi_tenant = True

    async def get(self, *_a: object, **_kw: object) -> object:  # pragma: no cover
        raise NotImplementedError


_ProgressLog = list[tuple[str, dict[str, object]]]


def _ltm_services(
    workspace: _StubWorkspace,
) -> tuple[LTMPhaseServices, _RecordingUsageSink, _ProgressLog]:
    progress: _ProgressLog = []

    async def emit(message: str, payload: Mapping[str, object]) -> None:
        progress.append((message, dict(payload)))

    usage = _RecordingUsageSink()
    services = LTMPhaseServices(
        emit_progress=emit,
        secrets=_NoopSecretResolver(),  # type: ignore[arg-type]
        usage=usage,  # type: ignore[arg-type]
        audit=_NoopAuditSink(),  # type: ignore[arg-type]
        clock=lambda: datetime.now(UTC),
        ltm_workspace=workspace,  # type: ignore[arg-type]
    )
    return services, usage, progress


def _context_services(
    ltm_ws: _StubWorkspace, ctx_ws: _StubWorkspace
) -> tuple[ContextPhaseServices, _RecordingUsageSink, _ProgressLog]:
    progress: _ProgressLog = []

    async def emit(message: str, payload: Mapping[str, object]) -> None:
        progress.append((message, dict(payload)))

    usage = _RecordingUsageSink()
    services = ContextPhaseServices(
        emit_progress=emit,
        secrets=_NoopSecretResolver(),  # type: ignore[arg-type]
        usage=usage,  # type: ignore[arg-type]
        audit=_NoopAuditSink(),  # type: ignore[arg-type]
        clock=lambda: datetime.now(UTC),
        ltm_workspace=ltm_ws,  # type: ignore[arg-type]
        context_workspace=ctx_ws,  # type: ignore[arg-type]
    )
    return services, usage, progress


def _make_memory(mid: str, title: str = "title") -> Memory:
    return Memory(
        id=mid,
        tenant_id="default",
        agent_id="agent-1",
        type="observation",
        title=title,
        content="body",
        submitted_at=datetime(2026, 5, 2, 10, 0, 0, tzinfo=UTC),
    )


def _batch(memories: list[Memory]) -> MemoryBatch:
    return MemoryBatch(
        lease_id="L1",
        tenant_id="default",
        memories=memories,
        snapshot_at=datetime(2026, 5, 2, 10, 5, 0, tzinfo=UTC),
    )


def _ltm_ctx(batch: MemoryBatch, *, instructions: str | None = None) -> LTMPhaseContext:
    return LTMPhaseContext(
        request_id="r1",
        tenant_id="default",
        lease_id="L1",
        batch=batch,
        ltm_workspace_id="ws-ltm",
        instructions=instructions,
    )


def _context_ctx(*, diff: Diff, instructions: str | None = None) -> ContextPhaseContext:
    return ContextPhaseContext(
        request_id="r1",
        tenant_id="default",
        lease_id="L1",
        ltm_workspace_id="ws-ltm",
        ltm_diff=diff,
        context_workspace_id="ws-ctx",
        instructions=instructions,
    )


def test_accepted_serializer_kinds_includes_defaults() -> None:
    assert "markdown-per-memory" in ClaudeAgentDreamEngine.accepted_serializer_kinds
    assert "jsonl" in ClaudeAgentDreamEngine.accepted_serializer_kinds


def test_workspace_requirements_declared() -> None:
    reqs = ClaudeAgentDreamEngine.workspace_requirements
    assert reqs["ltm"]
    assert reqs["context"]


def test_engine_defaults_to_unisolated_local_runner_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, "dreamer.contrib.dream.claude_agent"):
        engine = ClaudeAgentDreamEngine(serializer=JsonlSerializer())
    assert isinstance(engine.runner, LocalClaudeAgentRunner)
    assert any("unisolated" in record.message for record in caplog.records)


def test_engine_accepts_explicit_runner_without_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    runner = _RecordingRunner()
    with caplog.at_level(logging.WARNING, "dreamer.contrib.dream.claude_agent"):
        engine = ClaudeAgentDreamEngine(serializer=JsonlSerializer(), runner=runner)
    assert engine.runner is runner
    assert not any("unisolated" in record.message for record in caplog.records)


def test_docker_runner_scaffold_is_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        DockerClaudeAgentRunner()


def test_log_message_renders_text_tool_use_and_result(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from dreamer.contrib.dream._local import _log_message  # noqa: PLC0415

    @dataclass
    class _Block:
        text: str = ""
        name: str = ""
        input: dict[str, object] = field(default_factory=dict)  # noqa: A003
        thinking: str = ""
        is_error: bool = False
        content: object = None

    _Block.__name__ = "TextBlock"
    text_block = _Block(text="hello world")

    tool_block = type("ToolUseBlock", (), {})()
    tool_block.name = "Edit"  # type: ignore[attr-defined]
    tool_block.input = {"file_path": "/tmp/x.md", "old_string": "a"}  # type: ignore[attr-defined]

    asst = type("AssistantMessage", (), {})()
    asst.content = [text_block, tool_block]  # type: ignore[attr-defined]

    result_block = type("ToolResultBlock", (), {})()
    result_block.is_error = False  # type: ignore[attr-defined]
    result_block.content = "wrote 12 bytes"  # type: ignore[attr-defined]

    user = type("UserMessage", (), {})()
    user.content = [result_block]  # type: ignore[attr-defined]

    with caplog.at_level(logging.INFO, "dreamer.contrib.dream.claude_agent"):
        _log_message(asst)
        _log_message(user)

    messages = [r.getMessage() for r in caplog.records]
    assert any("agent: hello world" in m for m in messages)
    assert any("agent tool: Edit(file_path=/tmp/x.md)" in m for m in messages)
    assert any("agent tool result [ok]: wrote 12 bytes" in m for m in messages)


def test_log_message_renders_system_message(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from dreamer.contrib.dream._local import _log_message  # noqa: PLC0415

    sys_msg = type("SystemMessage", (), {})()
    sys_msg.subtype = "init"  # type: ignore[attr-defined]

    with caplog.at_level(logging.INFO, "dreamer.contrib.dream.claude_agent"):
        _log_message(sys_msg)

    assert any("agent system: init" in r.getMessage() for r in caplog.records)


def test_log_message_truncates_long_text(caplog: pytest.LogCaptureFixture) -> None:
    from dreamer.contrib.dream._local import _log_message  # noqa: PLC0415

    block = type("TextBlock", (), {})()
    block.text = "x" * 1000  # type: ignore[attr-defined]
    msg = type("AssistantMessage", (), {})()
    msg.content = [block]  # type: ignore[attr-defined]

    with caplog.at_level(logging.INFO, "dreamer.contrib.dream.claude_agent"):
        _log_message(msg)

    rendered = caplog.records[0].getMessage()
    assert rendered.endswith("…")
    assert len(rendered) < 600


@pytest.mark.asyncio
async def test_ltm_phase_writes_inbox_and_mirrors_mutations(tmp_path: Path) -> None:
    runner = _RecordingRunner(
        mutate_path=("memory/topics/new-topic.md", "---\nslug: new\n---\nbody\n"),
    )
    engine = ClaudeAgentDreamEngine(
        serializer=MarkdownPerMemorySerializer(),
        runner=runner,
        timeout_seconds=5,
    )
    ws = _StubWorkspace(id="ws-ltm", tenant_id="default", path=tmp_path / "ltm")
    services, usage, progress = _ltm_services(ws)
    batch = _batch([_make_memory("m1", "First"), _make_memory("m2", "Second")])

    with TenantScope.set("default"):
        await engine.run_ltm_phase(ctx=_ltm_ctx(batch), services=services)

    assert len(runner.invocations) == 1
    inv = runner.invocations[0]
    sandbox = inv["sandbox"]
    assert isinstance(sandbox, Path)

    landed = ws.path / "topics/new-topic.md"
    assert landed.is_file()
    assert landed.read_text() == "---\nslug: new\n---\nbody\n"

    assert not sandbox.exists()

    kinds = [e.kind for e in usage.events]
    assert "llm_tokens_in" in kinds
    assert "llm_tokens_out" in kinds

    assert any(name == "ltm.phase.starting" for name, _ in progress)
    assert any(name == "ltm.phase.completed" for name, _ in progress)


@pytest.mark.asyncio
async def test_ltm_phase_prompt_includes_serializer_fragment(tmp_path: Path) -> None:
    md = MarkdownPerMemorySerializer()
    jsonl = JsonlSerializer()

    md_runner = _RecordingRunner()
    md_engine = ClaudeAgentDreamEngine(serializer=md, runner=md_runner)
    jsonl_runner = _RecordingRunner()
    jsonl_engine = ClaudeAgentDreamEngine(serializer=jsonl, runner=jsonl_runner)

    batch = _batch([_make_memory("m1", "t")])

    ws_md = _StubWorkspace(id="ws-md", tenant_id="default", path=tmp_path / "md")
    ws_jsonl = _StubWorkspace(id="ws-jsonl", tenant_id="default", path=tmp_path / "jsonl")
    md_services, _, _ = _ltm_services(ws_md)
    jsonl_services, _, _ = _ltm_services(ws_jsonl)

    with TenantScope.set("default"):
        await md_engine.run_ltm_phase(ctx=_ltm_ctx(batch), services=md_services)
        await jsonl_engine.run_ltm_phase(ctx=_ltm_ctx(batch), services=jsonl_services)

    md_prompt = md_runner.invocations[0]["prompt"]
    jsonl_prompt = jsonl_runner.invocations[0]["prompt"]
    assert isinstance(md_prompt, str) and isinstance(jsonl_prompt, str)
    assert "<memory-id>" in md_prompt
    assert "inbox/batch.jsonl" in jsonl_prompt
    assert md_prompt != jsonl_prompt


@pytest.mark.asyncio
async def test_ltm_phase_appends_operator_instructions(tmp_path: Path) -> None:
    runner = _RecordingRunner()
    engine = ClaudeAgentDreamEngine(serializer=JsonlSerializer(), runner=runner)
    ws = _StubWorkspace(id="ws", tenant_id="default", path=tmp_path / "ltm")
    services, _, _ = _ltm_services(ws)

    with TenantScope.set("default"):
        await engine.run_ltm_phase(
            ctx=_ltm_ctx(_batch([_make_memory("m1", "t")]), instructions="be terse"),
            services=services,
        )

    prompt = runner.invocations[0]["prompt"]
    assert isinstance(prompt, str)
    assert "Operator instructions" in prompt
    assert "be terse" in prompt


@pytest.mark.asyncio
async def test_ltm_phase_writes_prompt_md_to_sandbox(tmp_path: Path) -> None:
    captured: dict[str, Path | str] = {}

    class _CapturingRunner(_RecordingRunner):
        async def run(
            self,
            *,
            prompt: str,
            sandbox: Path,
            timeout_seconds: float,
            env: Mapping[str, str] | None = None,
        ) -> AgentRunResult:
            captured["sandbox"] = sandbox
            captured["prompt_md"] = sandbox / "PROMPT.md"
            captured["prompt_text"] = sandbox / "PROMPT.md"  # placeholder
            captured["actual"] = (sandbox / "PROMPT.md").read_text(encoding="utf-8")
            return await super().run(
                prompt=prompt,
                sandbox=sandbox,
                timeout_seconds=timeout_seconds,
                env=env,
            )

    runner = _CapturingRunner()
    engine = ClaudeAgentDreamEngine(serializer=JsonlSerializer(), runner=runner)
    ws = _StubWorkspace(id="ws", tenant_id="default", path=tmp_path / "ltm")
    services, _, _ = _ltm_services(ws)

    with TenantScope.set("default"):
        await engine.run_ltm_phase(
            ctx=_ltm_ctx(_batch([_make_memory("m1", "t")])), services=services
        )
    assert captured["actual"]


@pytest.mark.asyncio
async def test_ltm_phase_failure_cleans_up_sandbox(tmp_path: Path) -> None:
    runner = _RecordingRunner(side_effect=DreamFailedError("phase timed out"))
    engine = ClaudeAgentDreamEngine(serializer=JsonlSerializer(), runner=runner)
    ws = _StubWorkspace(id="ws", tenant_id="default", path=tmp_path / "ltm")
    services, _, _ = _ltm_services(ws)

    with TenantScope.set("default"):
        with pytest.raises(DreamFailedError):
            await engine.run_ltm_phase(
                ctx=_ltm_ctx(_batch([_make_memory("m1", "t")])),
                services=services,
            )
    sandbox = runner.invocations[0]["sandbox"]
    assert isinstance(sandbox, Path)
    assert not sandbox.exists()


@pytest.mark.asyncio
async def test_ltm_phase_rejects_wrong_tenant_scope(tmp_path: Path) -> None:

    runner = _RecordingRunner()
    engine = ClaudeAgentDreamEngine(serializer=JsonlSerializer(), runner=runner)
    ws = _StubWorkspace(id="ws", tenant_id="default", path=tmp_path / "ltm")
    services, _, _ = _ltm_services(ws)

    with TenantScope.set("other"):
        with pytest.raises(Exception):
            await engine.run_ltm_phase(
                ctx=_ltm_ctx(_batch([_make_memory("m1", "t")])),
                services=services,
            )


@pytest.mark.asyncio
async def test_context_phase_includes_ltm_diff_in_prompt(tmp_path: Path) -> None:
    runner = _RecordingRunner()
    engine = ClaudeAgentDreamEngine(serializer=JsonlSerializer(), runner=runner)
    ltm_ws = _StubWorkspace(id="ws-ltm", tenant_id="default", path=tmp_path / "ltm")
    ctx_ws = _StubWorkspace(id="ws-ctx", tenant_id="default", path=tmp_path / "ctx")
    services, _, _ = _context_services(ltm_ws, ctx_ws)

    diff = Diff(
        added=["topics/new.md"],
        modified=["topics/existing.md"],
        deleted=["incidents/2026-04/x.md"],
    )

    with TenantScope.set("default"):
        await engine.run_context_phase(ctx=_context_ctx(diff=diff), services=services)

    prompt = runner.invocations[0]["prompt"]
    assert isinstance(prompt, str)
    assert "topics/new.md" in prompt
    assert "topics/existing.md" in prompt
    assert "incidents/2026-04/x.md" in prompt
    assert "Added" in prompt and "Modified" in prompt and "Deleted" in prompt


@pytest.mark.asyncio
async def test_context_phase_mirrors_mutations_back(tmp_path: Path) -> None:
    runner = _RecordingRunner(
        mutate_path=("context/AGENTS.md", "# AGENTS\n\nfresh\n"),
    )
    engine = ClaudeAgentDreamEngine(serializer=JsonlSerializer(), runner=runner)
    ltm_ws = _StubWorkspace(id="ws-ltm", tenant_id="default", path=tmp_path / "ltm")
    ctx_ws = _StubWorkspace(id="ws-ctx", tenant_id="default", path=tmp_path / "ctx")
    services, usage, progress = _context_services(ltm_ws, ctx_ws)

    with TenantScope.set("default"):
        await engine.run_context_phase(ctx=_context_ctx(diff=Diff()), services=services)

    landed = ctx_ws.path / "AGENTS.md"
    assert landed.read_text(encoding="utf-8") == "# AGENTS\n\nfresh\n"

    assert any(e.kind == "llm_tokens_in" for e in usage.events)
    assert any(name == "context.phase.starting" for name, _ in progress)
    assert any(name == "context.phase.completed" for name, _ in progress)


@pytest.mark.asyncio
async def test_context_phase_deletes_files_removed_by_agent(tmp_path: Path) -> None:
    runner = _RecordingRunner(
        delete_paths=("context/skills/old/SKILL.md",),
    )
    engine = ClaudeAgentDreamEngine(serializer=JsonlSerializer(), runner=runner)
    ltm_ws = _StubWorkspace(id="ws-ltm", tenant_id="default", path=tmp_path / "ltm")
    ctx_ws = _StubWorkspace(id="ws-ctx", tenant_id="default", path=tmp_path / "ctx")
    skill = ctx_ws.path / "skills/old/SKILL.md"
    skill.parent.mkdir(parents=True, exist_ok=True)
    skill.write_text("---\nname: old\n---\n", encoding="utf-8")

    services, _, _ = _context_services(ltm_ws, ctx_ws)
    with TenantScope.set("default"):
        await engine.run_context_phase(ctx=_context_ctx(diff=Diff()), services=services)
    assert not skill.exists()


@pytest.mark.asyncio
async def test_context_phase_failure_cleans_up_sandbox(tmp_path: Path) -> None:
    runner = _RecordingRunner(side_effect=RuntimeError("boom"))
    engine = ClaudeAgentDreamEngine(serializer=JsonlSerializer(), runner=runner)
    ltm_ws = _StubWorkspace(id="ws-ltm", tenant_id="default", path=tmp_path / "ltm")
    ctx_ws = _StubWorkspace(id="ws-ctx", tenant_id="default", path=tmp_path / "ctx")
    services, _, _ = _context_services(ltm_ws, ctx_ws)

    with TenantScope.set("default"):
        with pytest.raises(RuntimeError):
            await engine.run_context_phase(ctx=_context_ctx(diff=Diff()), services=services)
    sandbox = runner.invocations[0]["sandbox"]
    assert isinstance(sandbox, Path)
    assert not sandbox.exists()


@pytest.mark.asyncio
async def test_local_runner_translates_timeout_to_dream_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = LocalClaudeAgentRunner()

    async def _hang(*_a: object, **_kw: object) -> None:
        import asyncio  # noqa: PLC0415

        await asyncio.sleep(60)

    monkeypatch.setattr(runner, "_invoke", _hang)  # type: ignore[method-assign]
    with pytest.raises(DreamFailedError, match="phase timed out"):
        await runner.run(
            prompt="hi",
            sandbox=tmp_path,
            timeout_seconds=0.05,
        )
