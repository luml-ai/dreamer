"""Feedback MCP tools: `confirm_context` and `flag_context`.

They close the LTM feedback loop: agents report whether guidance in a
generated context bundle proved useful (`confirm_context`) or wrong
(`flag_context`). Both submit ordinary STM memories of the declared
feedback types through the server's shared submit pipeline
(`MCPToolContext.submit_memory`), so validation, hooks, audit, and
idempotency are identical to a direct `submit_memory` call.

The feedback memory types are conventions declared in config; the
`MemoryType` constants here are the canonical definitions the `dreamer init`
scaffold mirrors.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, ClassVar

from dreamer.api.compat import implements
from dreamer.api.contexts import MCPToolContext
from dreamer.api.stores import MCPTool
from dreamer.api.types import MemorySubmission, MemoryType

CONTEXT_CONFIRMED_TYPE_NAME = "context_confirmed"
CONTEXT_FLAGGED_TYPE_NAME = "context_flagged"

# LTM entries are kebab-case slugs (`topics/<slug>.md`). Format-only check:
# existence is never validated — feedback about a since-retired slug is
# still information for the dream.
SLUG_PATTERN = r"^[a-z0-9]+(?:-[a-z0-9]+)*$"

CONTEXT_CONFIRMED_MEMORY_TYPE = MemoryType(
    name=CONTEXT_CONFIRMED_TYPE_NAME,
    description=(
        "Keep-alive feedback: context guidance identified by an LTM slug "
        "proved useful in practice. Consumed in aggregate at dream time."
    ),
    metadata_schema={
        "type": "object",
        "required": ["target"],
        "properties": {"target": {"type": "string", "pattern": SLUG_PATTERN}},
    },
)

CONTEXT_FLAGGED_MEMORY_TYPE = MemoryType(
    name=CONTEXT_FLAGGED_TYPE_NAME,
    description=(
        "Correction feedback: context guidance proved wrong or misleading. "
        "Content is the observation narrative; metadata may anchor it to "
        "LTM slugs and quote the misleading bundle text."
    ),
    metadata_schema={
        "type": "object",
        "properties": {
            "targets": {
                "type": "array",
                "items": {"type": "string", "pattern": SLUG_PATTERN},
            },
            "quote": {"type": "string"},
        },
    },
)


def _require_submit(ctx: MCPToolContext) -> Any:
    if ctx.submit_memory is None:
        raise RuntimeError(
            f"{ctx.tool_name}: MCPToolContext.submit_memory is not bound; "
            "this tool must run inside the server MCP pipeline"
        )
    return ctx.submit_memory


@implements(MCPTool, version=1)
class ConfirmContextTool:
    """Submits a `context_confirmed` memory for a `[mem: <slug>]` target."""

    multi_tenant: ClassVar[bool] = True

    name = "confirm_context"
    description = (
        "Report that a piece of context guidance PROVED USEFUL. Call this "
        "when guidance marked with a [mem: <slug>] anchor in your context "
        "bundle turned out to be correct and actually helped — e.g. you "
        "followed it and it worked, or it saved you from a known pitfall. "
        "Pass the slug from the marker as `target`. Confirmations keep "
        "useful long-term memories alive; without them, entries decay and "
        "may be retired. Repeat confirmations of the same target on the "
        "same day are deduplicated automatically — calling twice is "
        "harmless. If guidance proved WRONG or misleading, call "
        "flag_context instead."
    )

    def __init__(self, *, memory_type: str = CONTEXT_CONFIRMED_TYPE_NAME) -> None:
        self.memory_type = memory_type

    def input_schema(self) -> Mapping[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["target"],
            "properties": {
                "target": {
                    "type": "string",
                    "pattern": SLUG_PATTERN,
                    "maxLength": 200,
                    "description": "The LTM slug from the [mem: <slug>] marker.",
                },
                "note": {
                    "type": "string",
                    "description": "Optional short note on how the guidance helped.",
                },
            },
        }

    async def call(self, args: Mapping[str, Any], *, ctx: MCPToolContext) -> Any:
        submit = _require_submit(ctx)
        target = str(args["target"])
        day = datetime.now(UTC).strftime("%Y-%m-%d")
        submissions: list[MemorySubmission] = await submit(
            {
                "type": self.memory_type,
                "title": f"context confirmed: {target}"[:120],
                "content": str(args.get("note") or ""),
                "metadata": {"target": target},
                "idempotency_key": f"{self.memory_type}:{ctx.principal.id}:{target}:{day}",
            }
        )
        if not submissions:
            return {"stored": False, "filtered": True}
        submission = submissions[0]
        return {
            "stored": True,
            "target": target,
            "deduplicated": submission.deduplicated,
            "memory_id": submission.memory.id,
        }


@implements(MCPTool, version=1)
class FlagContextTool:
    """Submits a `context_flagged` memory carrying a contradiction narrative."""

    multi_tenant: ClassVar[bool] = True

    name = "flag_context"
    description = (
        "Report that context guidance PROVED WRONG or misleading. Call this "
        "whenever what your context bundle said (or implied) contradicted "
        "what you actually observed — a stale fact, a command that no "
        "longer works, advice that caused a failure, or two sections that "
        "contradict each other. In `observation`, state what the context "
        "said or implied AND what turned out to be true, with enough "
        "evidence for a reviewer to verify. If the guidance carried "
        "[mem: <slug>] markers, pass those slugs as `targets`; otherwise "
        "quote the misleading bundle text in `quote`. Abstract flags with "
        "neither are also valuable — attribution happens later. Flags are "
        "the primary signal for correcting long-term memory; do not stay "
        "silent when context misleads you."
    )

    def __init__(self, *, memory_type: str = CONTEXT_FLAGGED_TYPE_NAME) -> None:
        self.memory_type = memory_type

    def input_schema(self) -> Mapping[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["observation"],
            "properties": {
                "observation": {
                    "type": "string",
                    "minLength": 1,
                    "description": (
                        "What the context said or implied, and what turned out "
                        "to be true. Include the evidence."
                    ),
                },
                "targets": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "pattern": SLUG_PATTERN,
                        "maxLength": 200,
                    },
                    "description": "LTM slugs from [mem: <slug>] markers, if known.",
                },
                "quote": {
                    "type": "string",
                    "description": (
                        "The bundle text that misled, verbatim — for flags "
                        "without target slugs."
                    ),
                },
            },
        }

    async def call(self, args: Mapping[str, Any], *, ctx: MCPToolContext) -> Any:
        submit = _require_submit(ctx)
        observation = str(args["observation"])
        targets = [str(t) for t in args.get("targets") or []]
        quote = args.get("quote")

        metadata: dict[str, Any] = {}
        if targets:
            metadata["targets"] = targets
        if quote is not None:
            metadata["quote"] = str(quote)

        title = (
            f"context flagged: {', '.join(targets)}" if targets else "context flagged: unanchored"
        )
        submissions: list[MemorySubmission] = await submit(
            {
                "type": self.memory_type,
                "title": title[:120],
                "content": observation,
                "metadata": metadata,
            }
        )
        if not submissions:
            return {"stored": False, "filtered": True}
        submission = submissions[0]
        return {
            "stored": True,
            "targets": targets,
            "anchored": bool(targets or quote),
            "memory_id": submission.memory.id,
        }


__all__ = [
    "CONTEXT_CONFIRMED_MEMORY_TYPE",
    "CONTEXT_CONFIRMED_TYPE_NAME",
    "CONTEXT_FLAGGED_MEMORY_TYPE",
    "CONTEXT_FLAGGED_TYPE_NAME",
    "SLUG_PATTERN",
    "ConfirmContextTool",
    "FlagContextTool",
]
