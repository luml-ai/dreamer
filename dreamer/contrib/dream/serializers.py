"""STM serializers for the dream sandbox.

Each serializer materializes a :class:`MemoryBatch` under a sandbox ``inbox/``
directory in a layout the dream engine's LLM tools can navigate. The
``prompt_fragment`` is appended to the LTM-phase prompt so the agent knows
where to look.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

from dreamer.api.compat import implements
from dreamer.api.contexts import SerializeContext, SerializeServices
from dreamer.api.stores import STMSerializer
from dreamer.api.tenants import TenantScope
from dreamer.api.types import Memory, MemoryBatch


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


def _write_per_memory(target: Path, batch: MemoryBatch) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for memory in batch.memories:
        path = target / _per_memory_filename(memory)
        body = memory.content or ""
        text = f"---\n{_frontmatter(memory)}\n---\n\n{body}\n"
        path.write_text(text, encoding="utf-8")


def _write_jsonl(target: Path, filename: str, batch: MemoryBatch) -> None:
    target.mkdir(parents=True, exist_ok=True)
    path = target / filename
    lines = [m.model_dump_json() for m in batch.memories]
    text = "\n".join(lines)
    if lines:
        text += "\n"
    path.write_text(text, encoding="utf-8")


@implements(STMSerializer, version=1)
class MarkdownPerMemorySerializer:
    """Default serializer: one markdown file per memory.

    Produces ``<target>/<memory-id>-<title-slug>.md`` with YAML frontmatter
    and the memory ``content`` as the body.
    """

    multi_tenant: ClassVar[bool] = True
    kind: ClassVar[str] = "markdown-per-memory"

    async def write(
        self,
        batch: MemoryBatch,
        *,
        target: Path,
        ctx: SerializeContext,
        services: SerializeServices,
    ) -> None:
        TenantScope.assert_matches(ctx.tenant_id)
        _write_per_memory(target, batch)

    def prompt_fragment(self, batch: MemoryBatch) -> str:
        n = len(batch.memories)
        return (
            f"STM is at `inbox/`: {n} markdown file(s), one per memory, named "
            f"`<memory-id>-<title-slug>.md`. Each starts with a YAML frontmatter "
            f"block (id, tenant_id, agent_id, type, title, tags, submitted_at, "
            f"optional metadata), followed by the memory body."
        )


@implements(STMSerializer, version=1)
class JsonlSerializer:
    """Single ``batch.jsonl`` with one ``Memory`` per line.

    Lines are emitted in the order memories appear in ``batch.memories`` so
    consumers can rely on a stable ordering.
    """

    multi_tenant: ClassVar[bool] = True
    kind: ClassVar[str] = "jsonl"

    BATCH_FILENAME: ClassVar[str] = "batch.jsonl"

    async def write(
        self,
        batch: MemoryBatch,
        *,
        target: Path,
        ctx: SerializeContext,
        services: SerializeServices,
    ) -> None:
        TenantScope.assert_matches(ctx.tenant_id)
        _write_jsonl(target, self.BATCH_FILENAME, batch)

    def prompt_fragment(self, batch: MemoryBatch) -> str:
        n = len(batch.memories)
        return (
            f"STM is at `inbox/{self.BATCH_FILENAME}`: {n} JSON object(s), one "
            f"per line. Each line is a `Memory` payload with fields id, "
            f"tenant_id, agent_id, type, title, content, tags, metadata, "
            f"submitted_at."
        )


__all__: tuple[str, ...] = (
    "JsonlSerializer",
    "MarkdownPerMemorySerializer",
)
