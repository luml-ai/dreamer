"""Default dream engine: ``ClaudeAgentDreamEngine``.

Implements both :class:`LTMPhaseRunner@1` and :class:`ContextPhaseRunner@1`.
Builds a per-phase sandbox containing ``memory/`` (LTM workspace), optionally
``context/`` (context phase only), ``inbox/`` (LTM phase only, written by the
configured :class:`STMSerializer`), and ``PROMPT.md``. Delegates execution to
a :class:`ClaudeAgentRunner` (local in-process by default), then mirrors
workspace mutations back and tears the sandbox down. Emits ``UsageEvent``s
for token counts and periodic ``services.emit_progress`` updates.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar
from uuid import uuid4

from dreamer.api.compat import implements
from dreamer.api.contexts import (
    ContextPhaseContext,
    ContextPhaseServices,
    LTMPhaseContext,
    LTMPhaseServices,
    SerializeContext,
    SerializeServices,
    UsageContext,
)
from dreamer.api.dream import ContextPhaseRunner, LTMPhaseRunner
from dreamer.api.errors import DreamFailedError
from dreamer.api.stores import STMSerializer
from dreamer.api.tenants import TenantScope
from dreamer.api.types import FileViewable, UsageEvent, Workspace
from dreamer.contrib.dream._local import (
    AgentRunResult,
    ClaudeAgentRunner,
    LocalClaudeAgentRunner,
)

LOGGER = logging.getLogger("dreamer.contrib.dream.claude_agent")

DEFAULT_LTM_PROMPT_TEMPLATE = """\
You are folding new short-term memories (STM) into long-term memory (LTM).

# Workspace layout

- `memory/` — the LTM tree. Edit files here. Layout rules are documented in
  `memory/_meta/schema.md` (read it first). Topics live under `topics/`,
  incidents under `incidents/<YYYY-MM>/`. Every topic and incident must have
  YAML frontmatter (see schema). The store regenerates `INDEX.md` on commit
  — do **not** edit it.
- `inbox/` — the new STM batch the dream agent must process this run. The
  exact layout depends on the configured serializer (see "STM input" below).

# Goal

Fold the STM batch into LTM. Evolve LTM, do not replace it. Preserve human
edits. When information already lives in a topic, update that topic; when an
event is genuinely new, create an incident under `incidents/<YYYY-MM>/`.
Keep frontmatter accurate (slugs, tags, sources, related links).

# Reinforce and prune

The batch may carry agent feedback about existing LTM entries: aggregated
confirmations ("this guidance proved useful") and flags ("this guidance
proved wrong — here is what I observed"). Apply this policy:

- **Signal hierarchy** (strongest first): evidenced flags > absence of
  confirmations > age. Age alone is only a tiebreaker, never a reason to
  prune on its own.
- **Reinforce**: for each confirmed target, increment its `confirmations`
  frontmatter count and set `last_confirmed` to the report's latest
  timestamp.
- **Weaken, then flip**: a single flag against a well-confirmed entry
  weakens it — annotate the entry with the contradiction and record the
  decision in the operations log — rather than rewriting it. Rewrite or
  supersede only when flags corroborate (multiple independent flags, or
  evidence you can verify from the batch). A flag against a
  `importance: pinned` entry is NEVER acted on autonomously — record it
  prominently in the operations log for human attention.
- **Dispose of every flag**: each flag must end this run either attributed
  to an entry and acted on, identified as a synthesis problem to fix in the
  context bundle, or explicitly discarded with a reason in the operations
  log. Nothing lingers unresolved.
- **Retirement is archival**: retire an entry by MOVING it to
  `archive/<original relative path>` with `retired_at`, `retired_reason`,
  and (when superseded) `superseded_by` frontmatter. Hard deletion is
  reserved for `importance: ephemeral` entries and exact duplicates.
- **Prune candidates**: `ephemeral` entries past their usefulness and old
  incidents with zero confirmations. Be conservative — consolidation and
  contradiction-resolution carry the value; aggressive time-decay does not.
- **Log every decision**: append one line per reinforce/archive/discard
  decision to `archive/LOG.md` (what changed, why).

# STM input
"""

DEFAULT_CONTEXT_PROMPT_TEMPLATE = """\
You are updating the agent-facing context store from changes that just landed
in long-term memory (LTM).

# Workspace layout

- `memory/` — read-only LTM. Layout in `memory/_meta/schema.md`.
- `context/` — agent-facing context (this is what other agents read). Layout
  in `context/_meta/schema.md`. `AGENTS.md` is the entry point; skills live
  under `skills/<skill-name>/SKILL.md` with frontmatter (`name`,
  `description`, `version`).

# Anchors and feedback instructions

Generated context must close the feedback loop:

1. Guidance derived from an LTM topic carries a VISIBLE `[mem: <slug>]`
   marker in the text (plain visible text, not an HTML comment — some
   clients strip comments), where `<slug>` is the source topic's slug.
2. The bundle includes a short standing instruction telling agents to call
   the `confirm_context` tool when guidance marked `[mem: <slug>]` proves
   useful, and the `flag_context` tool (with what they observed) when it
   proves wrong or misleading, passing the slug from the marker.

Markers are hints, not a gate: missing markers degrade feedback quality but
never break anything.

# What changed

The most recent LTM commit produced the following diff. Use it to scope
your edits to context — do not re-derive everything from scratch.

"""


def _format_diff(label: str, paths: Sequence[str]) -> str:
    if not paths:
        return f"- {label}: (none)\n"
    lines = "\n".join(f"  - {p}" for p in paths)
    return f"- {label}:\n{lines}\n"


@implements(LTMPhaseRunner, version=1)
@implements(ContextPhaseRunner, version=1)
class ClaudeAgentDreamEngine:
    multi_tenant: ClassVar[bool] = False
    workspace_requirements: ClassVar[Mapping[str, frozenset[type]]] = {
        "ltm": frozenset({FileViewable}),
        "context": frozenset({FileViewable}),
    }
    accepted_serializer_kinds: ClassVar[frozenset[str]] = frozenset(
        {"markdown-per-memory", "jsonl"}
    )

    def __init__(
        self,
        *,
        serializer: STMSerializer,
        runner: ClaudeAgentRunner | None = None,
        sandbox_root: Path | str | None = None,
        timeout_seconds: float = 1200.0,
        ltm_prompt_template: str | None = None,
        context_prompt_template: str | None = None,
    ) -> None:
        self.serializer = serializer
        self.timeout_seconds = timeout_seconds
        self.sandbox_root = Path(sandbox_root) if sandbox_root else None
        self.ltm_prompt_template = ltm_prompt_template or DEFAULT_LTM_PROMPT_TEMPLATE
        self.context_prompt_template = context_prompt_template or DEFAULT_CONTEXT_PROMPT_TEMPLATE
        self.runner = runner if runner is not None else LocalClaudeAgentRunner()
        self._warn_unisolated()

    def _warn_unisolated(self) -> None:
        if isinstance(self.runner, LocalClaudeAgentRunner):
            LOGGER.warning(
                "Claude Agent dream engine running unisolated (in-process); "
                "container isolation is not implemented yet."
            )

    async def run_ltm_phase(self, *, ctx: LTMPhaseContext, services: LTMPhaseServices) -> None:
        TenantScope.assert_matches(ctx.tenant_id)

        ltm_workspace_path = await _file_view(services.ltm_workspace)
        sandbox = self._make_sandbox(prefix=f"ltm-{ctx.lease_id}")
        try:
            inbox = sandbox / "inbox"
            inbox.mkdir(parents=True, exist_ok=True)
            await self.serializer.write(
                ctx.batch,
                target=inbox,
                ctx=SerializeContext(
                    request_id=ctx.request_id,
                    tenant_id=ctx.tenant_id,
                    lease_id=ctx.lease_id,
                ),
                services=_downcast_serialize_services(services),
            )

            _mirror_in(sandbox / "memory", ltm_workspace_path)
            prompt = self._compose_ltm_prompt(ctx=ctx)
            (sandbox / "PROMPT.md").write_text(prompt, encoding="utf-8")

            await services.emit_progress(
                "ltm.phase.starting",
                {
                    "lease_id": ctx.lease_id,
                    "batch_size": len(ctx.batch.memories),
                    "sandbox": str(sandbox),
                    "runner": self.runner.name,
                },
            )

            result = await self.runner.run(
                prompt=prompt,
                sandbox=sandbox,
                timeout_seconds=self.timeout_seconds,
            )

            _mirror_out(sandbox / "memory", ltm_workspace_path)
            await self._emit_token_usage(
                ctx_request_id=ctx.request_id,
                tenant_id=ctx.tenant_id,
                services=services,
                phase="ltm",
                result=result,
            )
            await services.emit_progress(
                "ltm.phase.completed",
                {
                    "lease_id": ctx.lease_id,
                    "tokens_in": result.tokens_in,
                    "tokens_out": result.tokens_out,
                },
            )
        finally:
            shutil.rmtree(sandbox, ignore_errors=True)

    async def run_context_phase(
        self, *, ctx: ContextPhaseContext, services: ContextPhaseServices
    ) -> None:
        TenantScope.assert_matches(ctx.tenant_id)

        ltm_path = await _file_view(services.ltm_workspace)
        ctx_path = await _file_view(services.context_workspace)
        sandbox = self._make_sandbox(prefix=f"context-{ctx.lease_id}")
        try:
            _mirror_in(sandbox / "memory", ltm_path)
            _mirror_in(sandbox / "context", ctx_path)
            prompt = self._compose_context_prompt(ctx=ctx)
            (sandbox / "PROMPT.md").write_text(prompt, encoding="utf-8")

            await services.emit_progress(
                "context.phase.starting",
                {
                    "lease_id": ctx.lease_id,
                    "ltm_added": len(ctx.ltm_diff.added),
                    "ltm_modified": len(ctx.ltm_diff.modified),
                    "ltm_deleted": len(ctx.ltm_diff.deleted),
                    "sandbox": str(sandbox),
                    "runner": self.runner.name,
                },
            )

            result = await self.runner.run(
                prompt=prompt,
                sandbox=sandbox,
                timeout_seconds=self.timeout_seconds,
            )

            _mirror_out(sandbox / "context", ctx_path)
            await self._emit_token_usage(
                ctx_request_id=ctx.request_id,
                tenant_id=ctx.tenant_id,
                services=services,
                phase="context",
                result=result,
            )
            await services.emit_progress(
                "context.phase.completed",
                {
                    "lease_id": ctx.lease_id,
                    "tokens_in": result.tokens_in,
                    "tokens_out": result.tokens_out,
                },
            )
        finally:
            shutil.rmtree(sandbox, ignore_errors=True)

    def _compose_ltm_prompt(self, *, ctx: LTMPhaseContext) -> str:
        body = self.ltm_prompt_template
        body += self.serializer.prompt_fragment(ctx.batch)
        if ctx.instructions:
            body += "\n\n# Operator instructions\n\n" + ctx.instructions.strip() + "\n"
        return body.rstrip() + "\n"

    def _compose_context_prompt(self, *, ctx: ContextPhaseContext) -> str:
        body = self.context_prompt_template
        body += _format_diff("Added", ctx.ltm_diff.added)
        body += _format_diff("Modified", ctx.ltm_diff.modified)
        body += _format_diff("Deleted", ctx.ltm_diff.deleted)
        if ctx.instructions:
            body += "\n# Operator instructions\n\n" + ctx.instructions.strip() + "\n"
        return body.rstrip() + "\n"

    def _make_sandbox(self, *, prefix: str) -> Path:
        if self.sandbox_root is None:
            import tempfile  # noqa: PLC0415

            base = Path(tempfile.mkdtemp(prefix=f"dreamer-{prefix}-"))
        else:
            self.sandbox_root.mkdir(parents=True, exist_ok=True)
            base = self.sandbox_root / f"{prefix}-{uuid4().hex[:8]}"
            base.mkdir(parents=True, exist_ok=True)
        return base

    @staticmethod
    async def _emit_token_usage(
        *,
        ctx_request_id: str,
        tenant_id: str,
        services: LTMPhaseServices | ContextPhaseServices,
        phase: str,
        result: AgentRunResult,
    ) -> None:
        now = datetime.now(UTC)
        component = f"dreamer.contrib.dream.claude_agent.{phase}"
        if result.tokens_in is not None:
            await services.usage.record(
                UsageEvent(
                    tenant_id=tenant_id,
                    component=component,
                    kind="llm_tokens_in",
                    amount=float(result.tokens_in),
                    unit="tokens",
                    at=now,
                ),
                ctx=_usage_ctx(ctx_request_id, tenant_id),
            )
        if result.tokens_out is not None:
            await services.usage.record(
                UsageEvent(
                    tenant_id=tenant_id,
                    component=component,
                    kind="llm_tokens_out",
                    amount=float(result.tokens_out),
                    unit="tokens",
                    at=now,
                ),
                ctx=_usage_ctx(ctx_request_id, tenant_id),
            )


async def _file_view(workspace: Workspace) -> Path:
    if not isinstance(workspace, FileViewable):
        raise DreamFailedError("ClaudeAgentDreamEngine requires FileViewable workspaces")
    return await workspace.file_view()


def _downcast_serialize_services(
    services: LTMPhaseServices,
) -> SerializeServices:
    return SerializeServices(
        emit_progress=services.emit_progress,
        secrets=services.secrets,
        usage=services.usage,
        audit=services.audit,
        clock=services.clock,
    )


def _mirror_in(target: Path, source: Path) -> None:
    """Copy ``source`` into ``target`` (recursive). Both may be empty.

    Sync helper so async callers don't trip ASYNC240; the underlying work is
    already blocking I/O.
    """
    target.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        return
    for item in source.rglob("*"):
        rel = item.relative_to(source)
        dst = target / rel
        if item.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
        elif item.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dst)


def _mirror_out(source: Path, target: Path) -> None:
    """Mirror ``source`` back to ``target`` (delete-replace).

    Removes files under ``target`` that no longer exist under ``source`` so
    the agent's deletions stick, then drops empty directories.
    """
    target.mkdir(parents=True, exist_ok=True)
    desired: dict[str, Path] = {}
    if source.exists():
        for item in source.rglob("*"):
            if item.is_file():
                desired[item.relative_to(source).as_posix()] = item
    existing: dict[str, Path] = {
        path.relative_to(target).as_posix(): path for path in target.rglob("*") if path.is_file()
    }
    for rel, path in existing.items():
        if rel not in desired:
            path.unlink()
    for rel, src_path in desired.items():
        dst = target / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst)
    for path in sorted(target.rglob("*"), reverse=True):
        if path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass


def _usage_ctx(request_id: str, tenant_id: str) -> UsageContext:
    return UsageContext(request_id=request_id, tenant_id=tenant_id)


__all__ = (
    "DEFAULT_CONTEXT_PROMPT_TEMPLATE",
    "DEFAULT_LTM_PROMPT_TEMPLATE",
    "ClaudeAgentDreamEngine",
)
