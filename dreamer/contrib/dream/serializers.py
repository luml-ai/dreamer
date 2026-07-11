"""STM serializers for the dream sandbox.

Each serializer materializes a :class:`MemoryBatch` under a sandbox ``inbox/``
directory in a layout the dream engine's LLM tools can navigate. The
``prompt_fragment`` is appended to the LTM-phase prompt so the agent knows
where to look.

Feedback memories are separated from substantive ones: ``context_confirmed``
memories collapse into one deterministic aggregate report (never individual
files), ``context_flagged`` memories materialize in full but distinctly
labeled, and everything else serializes exactly as before.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

from dreamer.api.compat import implements
from dreamer.api.contexts import SerializeContext, SerializeServices
from dreamer.api.stores import STMSerializer
from dreamer.api.tenants import TenantScope
from dreamer.api.types import Memory, MemoryBatch
from dreamer.contrib.mcp_tools.feedback import (
    CONTEXT_CONFIRMED_TYPE_NAME,
    CONTEXT_FLAGGED_TYPE_NAME,
    SLUG_PATTERN,
)

FEEDBACK_DIRNAME = "feedback"
CONFIRMATIONS_FILENAME = "confirmations.md"
FLAGS_DIRNAME = "flags"

_SLUG_RE = re.compile(SLUG_PATTERN)


def _kebab(value: str) -> str:
    out: list[str] = []
    last_dash = True
    for ch in value.strip().lower():
        if ch.isalnum():
            out.append(ch)
            last_dash = False
        elif not last_dash:
            out.append("-")
            last_dash = True
    while out and out[-1] == "-":
        out.pop()
    return "".join(out) or "untitled"


def _format_iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _yaml_scalar(value: Any) -> str:
    """Render a Python scalar/list as a YAML-safe inline value.

    Handles strings (with simple quoting when needed), ints, floats, bools,
    datetimes, ``None``, and flat lists of those. Anything more exotic is
    coerced via ``repr`` as a safety net.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, datetime):
        return _format_iso(value)
    if isinstance(value, list):
        rendered = ", ".join(_yaml_scalar(item) for item in value)
        return f"[{rendered}]"
    if isinstance(value, str):
        if value == "":
            return "''"
        # Quote when the string contains characters that have YAML meaning in
        # a flow scalar context, leads with a structural sigil, or has
        # leading/trailing whitespace (which YAML strips).
        risky_anywhere = any(ch in value for ch in ":#[]{},\n\"'")
        leading_sigil = value[0] in "-?*&!%@>|"
        if risky_anywhere or leading_sigil or value[0].isspace() or value[-1].isspace():
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'
        return value
    return repr(value)


def _frontmatter(memory: Memory) -> str:
    # Deterministic field order so per-memory diffs stay stable.
    lines: list[str] = []
    lines.append(f"id: {_yaml_scalar(memory.id)}")
    lines.append(f"tenant_id: {_yaml_scalar(memory.tenant_id)}")
    lines.append(f"agent_id: {_yaml_scalar(memory.agent_id)}")
    lines.append(f"type: {_yaml_scalar(memory.type)}")
    lines.append(f"title: {_yaml_scalar(memory.title)}")
    lines.append(f"tags: {_yaml_scalar(list(memory.tags))}")
    lines.append(f"submitted_at: {_yaml_scalar(memory.submitted_at)}")
    if memory.idempotency_key is not None:
        lines.append(f"idempotency_key: {_yaml_scalar(memory.idempotency_key)}")
    if memory.metadata:
        lines.append("metadata:")
        for key in sorted(memory.metadata):
            lines.append(f"  {key}: {_yaml_scalar(memory.metadata[key])}")
    return "\n".join(lines)


def _per_memory_filename(memory: Memory) -> str:
    # Falls back to ``no-id`` when the id is missing — STMStore.submit always
    # assigns one, but the serializer must not crash on a malformed batch.
    base = memory.id or "no-id"
    slug = _kebab(memory.title or "")
    if slug and slug != "untitled":
        return f"{base}-{slug}.md"
    return f"{base}.md"


def _write_per_memory(target: Path, memories: list[Memory]) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for memory in memories:
        path = target / _per_memory_filename(memory)
        body = memory.content or ""
        text = f"---\n{_frontmatter(memory)}\n---\n\n{body}\n"
        path.write_text(text, encoding="utf-8")


def _write_jsonl(target: Path, filename: str, memories: list[Memory]) -> None:
    target.mkdir(parents=True, exist_ok=True)
    path = target / filename
    lines = [m.model_dump_json() for m in memories]
    text = "\n".join(lines)
    if lines:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def _split_feedback(
    memories: list[Memory],
    *,
    confirmed_type: str,
    flagged_type: str,
) -> tuple[list[Memory], list[Memory], list[Memory]]:
    """Return ``(substantive, confirmations, flags)`` preserving batch order."""
    substantive: list[Memory] = []
    confirmations: list[Memory] = []
    flags: list[Memory] = []
    for memory in memories:
        if memory.type == confirmed_type:
            confirmations.append(memory)
        elif memory.type == flagged_type:
            flags.append(memory)
        else:
            substantive.append(memory)
    return substantive, confirmations, flags


def _confirmation_report(confirmations: list[Memory], *, confirmed_type: str) -> str:
    """One deterministic aggregate: per target, count + latest timestamp.

    Malformed targets (missing or not slug-shaped) are listed verbatim —
    existence against LTM is resolved at dream time, not here.
    """
    per_target: dict[str, tuple[int, datetime]] = {}
    malformed: list[str] = []
    for memory in confirmations:
        target = memory.metadata.get("target")
        if not isinstance(target, str) or not _SLUG_RE.fullmatch(target):
            malformed.append(repr(target) if not isinstance(target, str) else target)
            continue
        count, latest = per_target.get(target, (0, memory.submitted_at))
        per_target[target] = (count + 1, max(latest, memory.submitted_at))

    lines = [
        "# Context confirmations",
        "",
        f"Aggregated `{confirmed_type}` feedback from this batch: agents "
        "reported that the LTM entries below proved useful. For each target, "
        "increment its `confirmations` count and set `last_confirmed` to the "
        "latest timestamp.",
        "",
    ]
    for target in sorted(per_target):
        count, latest = per_target[target]
        lines.append(f"- `{target}`: {count} confirmation(s), latest {_format_iso(latest)}")
    if not per_target:
        lines.append("(no well-formed targets in this batch)")
    if malformed:
        lines.append("")
        lines.append("Malformed targets (listed verbatim, could not be aggregated):")
        for value in sorted(malformed):
            lines.append(f"- {value}")
    lines.append("")
    return "\n".join(lines)


@implements(STMSerializer, version=1)
class MarkdownPerMemorySerializer:
    """Default serializer: one markdown file per memory.

    Produces ``<target>/<memory-id>-<title-slug>.md`` with YAML frontmatter
    and the memory ``content`` as the body. Feedback memories go under
    ``<target>/feedback/`` instead: confirmations as one aggregate report,
    flags as full per-memory files.
    """

    multi_tenant: ClassVar[bool] = True
    kind: ClassVar[str] = "markdown-per-memory"

    def __init__(
        self,
        *,
        confirmed_type: str = CONTEXT_CONFIRMED_TYPE_NAME,
        flagged_type: str = CONTEXT_FLAGGED_TYPE_NAME,
    ) -> None:
        self.confirmed_type = confirmed_type
        self.flagged_type = flagged_type

    def _split(self, batch: MemoryBatch) -> tuple[list[Memory], list[Memory], list[Memory]]:
        return _split_feedback(
            batch.memories,
            confirmed_type=self.confirmed_type,
            flagged_type=self.flagged_type,
        )

    async def write(
        self,
        batch: MemoryBatch,
        *,
        target: Path,
        ctx: SerializeContext,
        services: SerializeServices,
    ) -> None:
        TenantScope.assert_matches(ctx.tenant_id)
        substantive, confirmations, flags = self._split(batch)
        _write_per_memory(target, substantive)
        if flags:
            _write_per_memory(target / FEEDBACK_DIRNAME / FLAGS_DIRNAME, flags)
        if confirmations:
            feedback_dir = target / FEEDBACK_DIRNAME
            feedback_dir.mkdir(parents=True, exist_ok=True)
            (feedback_dir / CONFIRMATIONS_FILENAME).write_text(
                _confirmation_report(confirmations, confirmed_type=self.confirmed_type),
                encoding="utf-8",
            )

    def prompt_fragment(self, batch: MemoryBatch) -> str:
        substantive, confirmations, flags = self._split(batch)
        parts = [
            f"STM is at `inbox/`: {len(substantive)} markdown file(s), one per memory, "
            f"named `<memory-id>-<title-slug>.md`. Each starts with a YAML frontmatter "
            f"block (id, tenant_id, agent_id, type, title, tags, submitted_at, "
            f"optional metadata), followed by the memory body."
        ]
        if confirmations:
            parts.append(
                f"Agent feedback: `inbox/{FEEDBACK_DIRNAME}/{CONFIRMATIONS_FILENAME}` "
                f"aggregates {len(confirmations)} `{self.confirmed_type}` "
                f"confirmation(s) per target."
            )
        if flags:
            parts.append(
                f"`inbox/{FEEDBACK_DIRNAME}/{FLAGS_DIRNAME}/` holds "
                f"{len(flags)} `{self.flagged_type}` report(s) in full — context "
                f"guidance that agents observed to be wrong or misleading; flags "
                f"without target metadata are unanchored and need attribution."
            )
        return " ".join(parts)


@implements(STMSerializer, version=1)
class JsonlSerializer:
    """Single ``batch.jsonl`` with one ``Memory`` per line.

    Substantive memories come first, then flags, each group preserving
    ``batch.memories`` order. Confirmations collapse into
    ``feedback/confirmations.md`` instead of appearing as lines; flags stay
    in the JSONL, distinguishable by their ``type`` field.
    """

    multi_tenant: ClassVar[bool] = True
    kind: ClassVar[str] = "jsonl"

    BATCH_FILENAME: ClassVar[str] = "batch.jsonl"

    def __init__(
        self,
        *,
        confirmed_type: str = CONTEXT_CONFIRMED_TYPE_NAME,
        flagged_type: str = CONTEXT_FLAGGED_TYPE_NAME,
    ) -> None:
        self.confirmed_type = confirmed_type
        self.flagged_type = flagged_type

    def _split(self, batch: MemoryBatch) -> tuple[list[Memory], list[Memory], list[Memory]]:
        return _split_feedback(
            batch.memories,
            confirmed_type=self.confirmed_type,
            flagged_type=self.flagged_type,
        )

    async def write(
        self,
        batch: MemoryBatch,
        *,
        target: Path,
        ctx: SerializeContext,
        services: SerializeServices,
    ) -> None:
        TenantScope.assert_matches(ctx.tenant_id)
        substantive, confirmations, flags = self._split(batch)
        _write_jsonl(target, self.BATCH_FILENAME, substantive + flags)
        if confirmations:
            feedback_dir = target / FEEDBACK_DIRNAME
            feedback_dir.mkdir(parents=True, exist_ok=True)
            (feedback_dir / CONFIRMATIONS_FILENAME).write_text(
                _confirmation_report(confirmations, confirmed_type=self.confirmed_type),
                encoding="utf-8",
            )

    def prompt_fragment(self, batch: MemoryBatch) -> str:
        substantive, confirmations, flags = self._split(batch)
        parts = [
            f"STM is at `inbox/{self.BATCH_FILENAME}`: {len(substantive) + len(flags)} "
            f"JSON object(s), one per line. Each line is a `Memory` payload with "
            f"fields id, tenant_id, agent_id, type, title, content, tags, metadata, "
            f"submitted_at."
        ]
        if flags:
            parts.append(
                f"{len(flags)} line(s) have type `{self.flagged_type}` — context "
                f"guidance that agents observed to be wrong or misleading; flags "
                f"without target metadata are unanchored and need attribution."
            )
        if confirmations:
            parts.append(
                f"Agent feedback: `inbox/{FEEDBACK_DIRNAME}/{CONFIRMATIONS_FILENAME}` "
                f"aggregates {len(confirmations)} `{self.confirmed_type}` "
                f"confirmation(s) per target."
            )
        return " ".join(parts)


__all__: tuple[str, ...] = (
    "CONFIRMATIONS_FILENAME",
    "FEEDBACK_DIRNAME",
    "FLAGS_DIRNAME",
    "JsonlSerializer",
    "MarkdownPerMemorySerializer",
)
