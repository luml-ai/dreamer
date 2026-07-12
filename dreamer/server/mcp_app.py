"""MCP HTTP entry path: server creation, request pipeline, and tool dispatch.

Every MCP request — built-in `submit_memory` plus every configured `MCPTool` —
passes through the tenant-facing `auth` slot:

1. `auth.authenticate(request)` → `Principal` (raised in the Starlette layer).
2. `Tenancy.tenant_for(principal)` → `TenantId`. `TenantScope` is set.
3. `RateLimiter.check(principal, tenant_id, action="mcp.<tool>")` →
   structured rate-limit error on deny.
4. `AuditSink.record(AuditEvent("mcp.<tool>", ...))` (parallel fan-out).
5. The tool runs.

`submit_memory` additionally consults the per-tenant `memory_types` from
`TenantConfigProvider`, runs `pre_memory_submit` hooks, persists each surviving
memory via `STMStore.submit`, and runs `post_memory_submit` hooks.
"""

from __future__ import annotations

import contextlib
import json
import logging
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import jsonschema
import mcp.types as mcp_types
from mcp.server.lowlevel import Server as MCPServer
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.types import Receive, Scope, Send

from dreamer.api.audit import AuditSink
from dreamer.api.auth import Tenancy
from dreamer.api.contexts import (
    AuditContext,
    MCPToolContext,
    PostMemorySubmitContext,
    PreMemorySubmitContext,
    RateLimitContext,
    SubmitContext,
    TenancyContext,
)
from dreamer.api.errors import MemorySubmitError, ValidationError
from dreamer.api.rate_limit import RateLimiter
from dreamer.api.runtime_state import RequestState
from dreamer.api.stores import MCPTool, STMStore
from dreamer.api.tenants import TenantConfigProvider, TenantScope
from dreamer.api.types import (
    AuditEvent,
    Memory,
    MemorySubmission,
    MemoryType,
    Principal,
    TenantId,
)
from dreamer.api.usage import UsageSink
from dreamer.server.runtime import HookRegistry
from dreamer.server.sinks import emit_audit

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


SUBMIT_MEMORY_TOOL_NAME = "submit_memory"


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(slots=True)
class MCPPipeline:
    """Runtime dependencies for the MCP request pipeline.

    Held by `create_mcp_server` so each tool dispatch can resolve the active
    request's principal/tenant, enforce rate limits, run hooks, and write to
    sinks. This struct is shared across requests; per-request state (principal,
    tenant_id, request_id) flows through `RequestState`.
    """

    tenancy: Tenancy
    stm_store: STMStore
    tenant_config_provider: TenantConfigProvider
    rate_limiter: RateLimiter
    hook_registry: HookRegistry
    audit_sinks: list[AuditSink] = field(default_factory=list)
    usage_sinks: list[UsageSink] = field(default_factory=list)
    mcp_tools: list[MCPTool] = field(default_factory=list)
    memory_types: tuple[MemoryType, ...] = ()
    max_content_bytes: int = 8192
    max_title_chars: int = 120


def _submit_memory_tool(memory_types: Sequence[MemoryType]) -> mcp_types.Tool:
    """Build the static `submit_memory` tool schema.

    The `type` enum advertises the **global** memory types declared in config.
    Per-tenant overrides are restrictive subsets enforced at call time, but
    the static MCP schema must reflect the union (MCP tool schemas are static
    per registration, so it can't depend on the caller's tenant).
    """
    type_names = sorted({mt.name for mt in memory_types})
    type_descriptions = "\n".join(f"- {mt.name}: {mt.description}" for mt in memory_types)
    type_property: dict[str, Any] = {"type": "string"}
    if type_names:
        type_property["enum"] = type_names
    schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["type", "title", "content"],
        "properties": {
            "type": type_property,
            "title": {"type": "string", "maxLength": 120},
            "content": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}, "default": []},
            "metadata": {"type": "object", "default": {}},
            "idempotency_key": {"type": "string"},
        },
    }
    types_header = (
        "\n\nAvailable memory types (operator-declared; tenant-restricted at call time):\n"
    )
    types_section = f"{types_header}{type_descriptions}" if type_descriptions else ""
    description = (
        "Submit a single short-term memory for later consolidation into "
        "long-term memory. Memories are durable and shape future sessions, "
        "so submit with intent — not as a log of routine activity.\n\n"
        "WHEN TO SUBMIT — only if a future agent could NOT derive the same "
        "conclusion in seconds from the code, docs, task description, or "
        "common background knowledge. Good candidates: explicit user "
        "preferences or constraints, domain rules invisible in the code, "
        "behavior that contradicts what the naming or docs suggest, resolved "
        "failure modes. Do NOT submit progress updates, restatements of code "
        "or the user's request, generic best practices, speculation, or "
        "multiple memories about the same insight.\n\n"
        "ERRORS — resolve first, then log. The value is the *resolution*, "
        "not the symptom. Workflow: reproduce → fix → verify → submit ONE "
        "memory with what failed, root cause, and the fix (specific enough "
        "that a future agent hitting the same symptom can apply it). If you "
        "cannot resolve, escalate to the user — do not submit a dead-end "
        "memory; it anchors future runs to the wrong answer.\n\n"
        "OBSERVATIONS — capture what you could not have known: insights the "
        "user stated explicitly, undocumented invariants, surprising "
        "behavior, preferred tooling/workflow you would not have picked by "
        "default. Skip anything already covered by your background knowledge "
        "or by reading the repo.\n\n"
        "SECURITY — never leak secrets. Treat `title`, `content`, `tags`, "
        "and `metadata` as if they will be checked into a public repo. Never "
        "include API keys, tokens, passwords, private keys, session cookies, "
        "credentialed connection strings, PII, or raw contents of `.env` / "
        "`*secret*` / `id_rsa*` files. If the insight involves a secret, "
        "describe the shape of the problem without the value. When in "
        "doubt, redact.\n\n"
        "FIELDS\n"
        "- `title`: one sentence (≤120 chars) naming the lesson, not the "
        "  event ('Migration 0042 needs CONCURRENTLY to avoid table lock', "
        "  not 'fixed migration').\n"
        "- `content`: smallest self-contained explanation: situation, what "
        "  was learned, how to act on it next time.\n"
        "- `type`: pick the declared type that best matches; if none fits, "
        "  the memory probably isn't worth submitting.\n"
        "- `idempotency_key`: set when the same insight could be submitted "
        "  twice (retry after a transient error)."
        f"{types_section}"
    )
    return mcp_types.Tool(
        name=SUBMIT_MEMORY_TOOL_NAME,
        description=description,
        inputSchema=schema,
    )


def create_mcp_server(pipeline: MCPPipeline, *, name: str = "dreamer") -> MCPServer:
    """Build the MCP `Server` with `submit_memory` + every configured `MCPTool`."""
    server: MCPServer = MCPServer(name=name)

    submit_tool = _submit_memory_tool(pipeline.memory_types)

    @server.list_tools()  # type: ignore[no-untyped-call,untyped-decorator]
    async def _list_tools() -> list[mcp_types.Tool]:
        tools: list[mcp_types.Tool] = [submit_tool]
        for tool in pipeline.mcp_tools:
            tools.append(
                mcp_types.Tool(
                    name=tool.name,
                    description=tool.description,
                    inputSchema=dict(tool.input_schema()),
                )
            )
        return tools

    @server.call_tool(validate_input=False)  # type: ignore[untyped-decorator]
    async def _call_tool(name: str, args: Mapping[str, Any]) -> list[mcp_types.TextContent]:
        request_state = RequestState.current()
        if request_state is None:
            return _error_block("internal", "request state not bound; auth middleware missing")
        return await _dispatch(
            pipeline=pipeline,
            tool_name=name,
            args=args,
            state=request_state,
        )

    return server


async def _dispatch(
    *,
    pipeline: MCPPipeline,
    tool_name: str,
    args: Mapping[str, Any],
    state: RequestState,
) -> list[mcp_types.TextContent]:
    principal = state.principal
    request_id = state.request_id
    action = f"mcp.{tool_name}"

    tenancy_ctx = TenancyContext(request_id=request_id, principal=principal)
    tenant_id: TenantId = await pipeline.tenancy.tenant_for(principal, ctx=tenancy_ctx)
    state.tenant_id = tenant_id

    with TenantScope.set(tenant_id):
        rl_ctx = RateLimitContext(request_id=request_id, tenant_id=tenant_id)
        decision = await pipeline.rate_limiter.check(
            principal=principal,
            tenant_id=tenant_id,
            action=action,
            ctx=rl_ctx,
        )
        if not decision.allowed:
            await emit_audit(
                pipeline.audit_sinks,
                AuditEvent(
                    event_type=action,
                    principal_id=principal.id,
                    tenant_id=tenant_id,
                    payload={
                        "rate_limited": True,
                        "retry_after_seconds": decision.retry_after_seconds,
                        "reason": decision.reason,
                    },
                    at=_utcnow(),
                ),
                ctx=AuditContext(request_id=request_id, tenant_id=tenant_id),
            )
            return _error_block(
                "rate_limited",
                decision.reason or "rate limit exceeded",
                retry_after_seconds=decision.retry_after_seconds,
            )

        await emit_audit(
            pipeline.audit_sinks,
            AuditEvent(
                event_type=action,
                principal_id=principal.id,
                tenant_id=tenant_id,
                payload={"args_keys": sorted(args.keys())},
                at=_utcnow(),
            ),
            ctx=AuditContext(request_id=request_id, tenant_id=tenant_id),
        )

        if tool_name == SUBMIT_MEMORY_TOOL_NAME:
            return await _handle_submit_memory(
                pipeline=pipeline,
                args=args,
                tenant_id=tenant_id,
                principal=principal,
                request_id=request_id,
            )

        for tool in pipeline.mcp_tools:
            if tool.name == tool_name:
                return await _handle_custom_tool(
                    pipeline=pipeline,
                    tool=tool,
                    args=args,
                    tenant_id=tenant_id,
                    principal=principal,
                    request_id=request_id,
                )

        return _error_block("unknown_tool", f"tool {tool_name!r} is not registered")


async def _handle_custom_tool(
    *,
    pipeline: MCPPipeline,
    tool: MCPTool,
    args: Mapping[str, Any],
    tenant_id: TenantId,
    principal: Principal,
    request_id: str,
) -> list[mcp_types.TextContent]:
    schema = tool.input_schema()
    try:
        jsonschema.validate(instance=dict(args), schema=dict(schema))
    except jsonschema.ValidationError as exc:
        return _error_block("invalid_args", exc.message)

    async def _submit(memory_args: Mapping[str, Any]) -> list[MemorySubmission]:
        return await submit_memories(
            pipeline=pipeline,
            args=memory_args,
            tenant_id=tenant_id,
            principal=principal,
            request_id=request_id,
        )

    ctx = MCPToolContext(
        request_id=request_id,
        tenant_id=tenant_id,
        principal=principal,
        tool_name=tool.name,
        submit_memory=_submit,
    )
    try:
        result = await tool.call(args, ctx=ctx)
    except MemorySubmitError as exc:
        return _error_block(exc.code, str(exc))
    except Exception as exc:  # noqa: BLE001 — surface tool error
        logger.exception("MCPTool %r raised", tool.name)
        return _error_block("tool_error", str(exc))
    return _ok_block(result)


async def _handle_submit_memory(
    *,
    pipeline: MCPPipeline,
    args: Mapping[str, Any],
    tenant_id: TenantId,
    principal: Principal,
    request_id: str,
) -> list[mcp_types.TextContent]:
    try:
        submissions = await submit_memories(
            pipeline=pipeline,
            args=args,
            tenant_id=tenant_id,
            principal=principal,
            request_id=request_id,
        )
    except MemorySubmitError as exc:
        return _error_block(exc.code, str(exc))
    if not submissions:
        return _ok_block({"submitted": [], "filtered": True})
    return _ok_block({"submitted": [_memory_to_dict(s.memory) for s in submissions]})


async def submit_memories(
    *,
    pipeline: MCPPipeline,
    args: Mapping[str, Any],
    tenant_id: TenantId,
    principal: Principal,
    request_id: str,
) -> list[MemorySubmission]:
    """The shared memory-submit pipeline behind `submit_memory` and any
    `MCPTool` that stores memories (via `MCPToolContext.submit_memory`).

    Every entry point gets identical semantics: type governance, metadata
    schema validation, pre/post hooks, and store idempotency. Raises
    `MemorySubmitError` on rejection; returns `[]` when pre-submit hooks
    filtered every candidate.
    """
    tenant_config = await pipeline.tenant_config_provider.get(
        tenant_id,
        ctx=_tenant_lookup_ctx(request_id, tenant_id),
    )
    effective_types = _effective_types(tenant_config.memory_types, pipeline.memory_types)
    type_names = {mt.name for mt in effective_types}
    type_by_name = {mt.name: mt for mt in effective_types}

    # Defensive: TenantConfigProvider should have already enforced subset
    # semantics at config load and resolve time. If a tenant somehow declares a
    # type missing from the global set, the global set wins.
    global_names = {mt.name for mt in pipeline.memory_types}
    if not type_names.issubset(global_names):
        offending = sorted(type_names - global_names)
        raise ValidationError(
            "tenant_config.memory_types must be a subset of global memory_types; "
            f"offending: {offending}"
        )

    try:
        candidate = _build_memory(
            args,
            tenant_id=tenant_id,
            principal=principal,
            max_content_bytes=pipeline.max_content_bytes,
        )
    except ValidationError as exc:
        raise MemorySubmitError("invalid_args", str(exc)) from exc
    if candidate.type not in type_names:
        raise MemorySubmitError(
            "type_not_allowed",
            f"memory type {candidate.type!r} is not allowed; "
            f"effective types: {sorted(type_names)}",
        )
    schema_check = _validate_metadata(candidate, type_by_name.get(candidate.type))
    if schema_check is not None:
        raise MemorySubmitError("invalid_metadata", schema_check)

    memories: list[Memory] = [candidate]
    pre_ctx = PreMemorySubmitContext(
        request_id=request_id,
        tenant_id=tenant_id,
        principal=principal,
        memories=memories,
    )
    try:
        for hook in pipeline.hook_registry.get("pre_memory_submit"):
            await hook.on_pre_memory_submit(ctx=pre_ctx)
    except Exception as exc:  # noqa: BLE001 — return structured error to the caller
        await emit_audit(
            pipeline.audit_sinks,
            AuditEvent(
                event_type="hook.failed",
                principal_id=principal.id,
                tenant_id=tenant_id,
                payload={"slot": "pre_memory_submit", "error": str(exc)},
                at=_utcnow(),
            ),
            ctx=AuditContext(request_id=request_id, tenant_id=tenant_id),
        )
        raise MemorySubmitError("hook_failed", f"pre_memory_submit: {exc}") from exc

    if not memories:
        return []

    submissions: list[MemorySubmission] = []
    for mem in memories:
        # Hook may have appended a different type; re-check.
        if mem.type not in type_names:
            raise MemorySubmitError(
                "type_not_allowed",
                f"hook-emitted memory type {mem.type!r} is not allowed; "
                f"effective types: {sorted(type_names)}",
            )
        # Pre-assigning the id lets us detect idempotency dedup: the store
        # returns the previously persisted memory (different id) on a hit.
        bound = mem.model_copy(
            update={"tenant_id": tenant_id, "id": mem.id or str(uuid.uuid4())}
        )
        submit_ctx = SubmitContext(
            request_id=request_id,
            tenant_id=tenant_id,
            principal_id=principal.id,
        )
        stored = await pipeline.stm_store.submit(bound, ctx=submit_ctx)
        submissions.append(
            MemorySubmission(memory=stored, deduplicated=stored.id != bound.id)
        )

    post_ctx = PostMemorySubmitContext(
        request_id=request_id,
        tenant_id=tenant_id,
        principal=principal,
        persisted=tuple(s.memory for s in submissions),
    )
    for hook in pipeline.hook_registry.get("post_memory_submit"):
        try:
            await hook.on_post_memory_submit(ctx=post_ctx)
        except Exception:  # noqa: BLE001 — post hooks are advisory
            logger.exception(
                "post_memory_submit hook %r raised; continuing", type(hook).__qualname__
            )

    return submissions


def _build_memory(
    args: Mapping[str, Any],
    *,
    tenant_id: TenantId,
    principal: Principal,
    max_content_bytes: int,
) -> Memory:
    if not isinstance(args, Mapping):
        raise ValidationError("args must be a mapping")
    missing = [k for k in ("type", "title", "content") if k not in args]
    if missing:
        raise ValidationError(f"missing required fields: {sorted(missing)}")
    title = args.get("title")
    if not isinstance(title, str) or len(title) > 120:
        raise ValidationError("title must be a string of length ≤120")
    content = args.get("content")
    if not isinstance(content, str):
        raise ValidationError("content must be a string")
    encoded = content.encode("utf-8")
    if len(encoded) > max_content_bytes:
        raise ValidationError(
            f"content exceeds max_content_bytes ({len(encoded)} > {max_content_bytes})"
        )
    tags = args.get("tags") or []
    if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
        raise ValidationError("tags must be a list of strings")
    metadata = args.get("metadata") or {}
    if not isinstance(metadata, Mapping):
        raise ValidationError("metadata must be a mapping")
    idempotency_key = args.get("idempotency_key")
    if idempotency_key is not None and not isinstance(idempotency_key, str):
        raise ValidationError("idempotency_key must be a string when provided")
    # `agent_id` is no longer part of the public schema. Default to the auth
    # principal so memories remain attributable for audit/grouping without
    # asking the agent to invent a label it has no canonical source for.
    agent_id = principal.id
    type_value = args.get("type")
    if not isinstance(type_value, str):
        raise ValidationError("type must be a string")

    return Memory(
        tenant_id=tenant_id,
        agent_id=agent_id,
        type=type_value,
        title=title,
        content=content,
        tags=list(tags),
        metadata=dict(metadata),
        submitted_at=_utcnow(),
        idempotency_key=idempotency_key,
    )


def _validate_metadata(memory: Memory, declared: MemoryType | None) -> str | None:
    if declared is None or declared.metadata_schema is None:
        return None
    try:
        jsonschema.validate(instance=dict(memory.metadata), schema=dict(declared.metadata_schema))
    except jsonschema.ValidationError as exc:
        return str(exc.message)
    return None


def _effective_types(
    overrides: tuple[MemoryType, ...] | None,
    global_types: tuple[MemoryType, ...],
) -> tuple[MemoryType, ...]:
    if overrides is None:
        return global_types
    return tuple(overrides)


def _tenant_lookup_ctx(request_id: str, tenant_id: TenantId) -> Any:
    """Lazy import to avoid circular dependencies."""
    from dreamer.api.contexts import TenantConfigLookupContext

    return TenantConfigLookupContext(request_id=request_id, tenant_id=tenant_id)


def _memory_to_dict(memory: Memory) -> dict[str, Any]:
    payload = memory.model_dump(mode="json")
    return payload


def _ok_block(payload: Any) -> list[mcp_types.TextContent]:
    text = json.dumps({"ok": True, "result": payload}, default=str)
    return [mcp_types.TextContent(type="text", text=text)]


def _error_block(
    error: str, message: str, *, retry_after_seconds: float | None = None
) -> list[mcp_types.TextContent]:
    body: dict[str, Any] = {"ok": False, "error": error, "message": message}
    if retry_after_seconds is not None:
        body["retry_after_seconds"] = retry_after_seconds
    return [mcp_types.TextContent(type="text", text=json.dumps(body))]


@dataclass(slots=True)
class MCPMount:
    """Bundle of the MCP server, the session manager, and the ASGI handler.

    `lifespan` must be entered when the parent Starlette app boots — the
    streamable-http session manager holds an internal task group that needs a
    live event loop.
    """

    server: MCPServer
    manager: StreamableHTTPSessionManager
    pipeline: MCPPipeline

    async def asgi(self, scope: Scope, receive: Receive, send: Send) -> None:
        await self.manager.handle_request(scope, receive, send)

    @contextlib.asynccontextmanager
    async def lifespan(self) -> AsyncIterator[None]:
        async with self.manager.run():
            yield


def build_mcp_mount(
    pipeline: MCPPipeline,
    *,
    name: str = "dreamer",
    json_response: bool = True,
    stateless: bool = True,
) -> MCPMount:
    """Build the MCP server + ASGI mount.

    `stateless=True` defaults match the framework's "MCP HTTP entry path"
    contract: every request creates a fresh transport, no session state, and
    `json_response` returns a single JSON body so simple clients can call
    `submit_memory` without negotiating SSE. Operators can override via config.
    """
    server = create_mcp_server(pipeline, name=name)
    manager = StreamableHTTPSessionManager(
        app=server,
        json_response=json_response,
        stateless=stateless,
    )
    return MCPMount(server=server, manager=manager, pipeline=pipeline)


def new_request_id() -> str:
    return uuid.uuid4().hex


__all__ = [
    "MCPMount",
    "MCPPipeline",
    "SUBMIT_MEMORY_TOOL_NAME",
    "build_mcp_mount",
    "create_mcp_server",
    "new_request_id",
    "submit_memories",
]
