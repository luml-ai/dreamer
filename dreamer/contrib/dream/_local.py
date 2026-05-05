"""Local-mode Claude Agent runner.

Runs a Claude Agent SDK invocation in-process with ``cwd`` set to the sandbox
directory. The SDK is imported lazily inside ``run`` so the ``dreamer-server``
core can be installed without the ``claude-agent`` extra and so tests can stub
the runner without pulling in the SDK.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dreamer.api.errors import DreamFailedError

LOGGER = logging.getLogger("dreamer.contrib.dream.claude_agent")


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    """Result of a Claude Agent SDK invocation.

    Token counts are best-effort and may be ``None`` when the SDK doesn't
    expose them. ``raw`` carries the original SDK response for debugging.
    """

    tokens_in: int | None
    tokens_out: int | None
    raw: Any | None = None


class ClaudeAgentRunner:
    """Pluggable runner; subclasses dispatch the prompt in their environment."""

    name: str = "base"

    async def run(
        self,
        *,
        prompt: str,
        sandbox: Path,
        timeout_seconds: float,
        env: Mapping[str, str] | None = None,
    ) -> AgentRunResult:  # pragma: no cover — abstract
        raise NotImplementedError


class LocalClaudeAgentRunner(ClaudeAgentRunner):
    """In-process runner with no isolation.

    The agent has access to the host filesystem outside the sandbox via path
    traversal — operators who need isolation should switch to
    :class:`DockerClaudeAgentRunner`.
    """

    name = "local"

    def __init__(self, *, model: str | None = None) -> None:
        self.model = model

    async def run(
        self,
        *,
        prompt: str,
        sandbox: Path,
        timeout_seconds: float,
        env: Mapping[str, str] | None = None,
    ) -> AgentRunResult:
        try:
            return await asyncio.wait_for(
                self._invoke(prompt=prompt, sandbox=sandbox, env=env),
                timeout=timeout_seconds,
            )
        except TimeoutError as e:
            raise DreamFailedError("phase timed out") from e

    async def _invoke(
        self,
        *,
        prompt: str,
        sandbox: Path,
        env: Mapping[str, str] | None,
    ) -> AgentRunResult:
        try:
            from claude_agent_sdk import (  # type: ignore[import-not-found]  # noqa: PLC0415
                ClaudeAgentOptions,
                ClaudeSDKClient,
                ResultMessage,
            )
        except ImportError as e:  # pragma: no cover — depends on extras
            raise DreamFailedError(
                "Claude Agent SDK is not installed; install dreamer-server[claude-agent]"
            ) from e

        options = ClaudeAgentOptions(
            cwd=sandbox,
            env=dict(env) if env else {},
            permission_mode="bypassPermissions",
            model=self.model,
        )

        last_result: Any = None
        async with ClaudeSDKClient(options=options) as client:
            await client.query(prompt)
            async for msg in client.receive_response():
                _log_message(msg)
                if isinstance(msg, ResultMessage):
                    last_result = msg
        return _summarize(last_result)


def _log_message(msg: Any) -> None:
    """Emit a single human-readable INFO line per agent message."""
    cls_name = type(msg).__name__
    if cls_name == "AssistantMessage":
        for block in getattr(msg, "content", ()) or ():
            block_name = type(block).__name__
            if block_name == "TextBlock":
                text = (getattr(block, "text", "") or "").strip()
                if text:
                    LOGGER.info("agent: %s", _truncate(text, 500))
            elif block_name == "ToolUseBlock":
                name = getattr(block, "name", "?")
                summary = _summarize_tool_input(getattr(block, "input", {}) or {})
                LOGGER.info("agent tool: %s(%s)", name, summary)
            elif block_name == "ThinkingBlock":
                thinking = (getattr(block, "thinking", "") or "").strip()
                if thinking:
                    LOGGER.info("agent (thinking): %s", _truncate(thinking, 200))
            else:
                LOGGER.info("agent block: %s", block_name)
        return
    if cls_name == "UserMessage":
        for block in _user_blocks(msg):
            block_name = type(block).__name__
            if block_name == "ToolResultBlock":
                is_error = bool(getattr(block, "is_error", False))
                tag = "error" if is_error else "ok"
                content = _stringify_tool_result(getattr(block, "content", None))
                LOGGER.info("agent tool result [%s]: %s", tag, _truncate(content, 200))
            else:
                LOGGER.info("agent user block: %s", block_name)
        return
    if cls_name == "RateLimitEvent":
        info = getattr(msg, "rate_limit_info", None)
        status = getattr(info, "status", None)
        if status and status != "allowed":
            LOGGER.warning(
                "agent rate limit: status=%s type=%s resets_at=%s",
                status,
                getattr(info, "rate_limit_type", "?"),
                getattr(info, "resets_at", "?"),
            )
        return
    if cls_name == "ResultMessage":
        stop = getattr(msg, "stop_reason", None)
        cost = getattr(msg, "total_cost_usd", None)
        usage = getattr(msg, "usage", None) or {}
        ti = usage.get("input_tokens") if isinstance(usage, Mapping) else None
        to = usage.get("output_tokens") if isinstance(usage, Mapping) else None
        LOGGER.info(
            "agent done: stop=%s tokens=%s/%s cost=%s",
            stop,
            ti,
            to,
            f"${cost:.4f}" if isinstance(cost, (int, float)) else "?",
        )
        return
    if cls_name == "SystemMessage":
        subtype = getattr(msg, "subtype", "?")
        LOGGER.info("agent system: %s", subtype)
        return
    if cls_name == "StreamEvent":
        return  # partial-token frames; only emitted with include_partial_messages=True
    LOGGER.info("agent message: %s", cls_name)


def _user_blocks(msg: Any) -> list[Any]:
    content = getattr(msg, "content", None)
    if isinstance(content, list):
        return content
    return []


def _summarize_tool_input(inp: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key in ("file_path", "path", "command", "pattern", "query"):
        value = inp.get(key)
        if value:
            parts.append(f"{key}={_truncate(str(value), 120)}")
            break
    if not parts and inp:
        first_key = next(iter(inp))
        parts.append(f"{first_key}={_truncate(str(inp[first_key]), 120)}")
    return ", ".join(parts) if parts else ""


def _stringify_tool_result(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, Mapping):
                text = item.get("text") or item.get("content") or ""
                if text:
                    chunks.append(str(text))
        return " ".join(chunks)
    return str(content)


def _truncate(text: str, limit: int) -> str:
    text = text.replace("\n", " ⏎ ")
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _summarize(response: Any) -> AgentRunResult:
    # ResultMessage.usage may be a dict or an attribute object depending on
    # SDK version, so duck-type both shapes.
    if response is None:
        return AgentRunResult(tokens_in=None, tokens_out=None, raw=None)
    usage = getattr(response, "usage", None)
    if isinstance(usage, Mapping):
        tokens_in = _coerce_int(usage.get("input_tokens"))
        tokens_out = _coerce_int(usage.get("output_tokens"))
    elif usage is not None:
        tokens_in = _coerce_int(getattr(usage, "input_tokens", None))
        tokens_out = _coerce_int(getattr(usage, "output_tokens", None))
    else:
        tokens_in = tokens_out = None
    return AgentRunResult(tokens_in=tokens_in, tokens_out=tokens_out, raw=response)


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


__all__ = ("AgentRunResult", "ClaudeAgentRunner", "LocalClaudeAgentRunner")
