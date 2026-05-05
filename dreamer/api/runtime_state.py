"""Per-request state contextvar.

Carries the resolved `Principal`, `request_id`, and (once tenancy resolves)
`tenant_id` for the duration of a single in-flight request. Set by the
auth middleware, read by the MCP tool dispatcher and by the `/context/...`
HTTP handler.

This is distinct from `TenantScope`: `TenantScope` is the *scope guard* that
stores assert against; `RequestState` is request metadata used by the request
pipeline itself (audit emission, logging, etc.).
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from contextvars import ContextVar, Token
from dataclasses import dataclass

from dreamer.api.types import Principal, TenantId


@dataclass(slots=True)
class RequestState:
    """Per-request principal + tenant + request id."""

    principal: Principal
    request_id: str
    tenant_id: TenantId | None = None

    @staticmethod
    def current() -> RequestState | None:
        return _current_request_state.get()

    @staticmethod
    def bind(state: RequestState) -> _RequestStateBinding:
        token = _current_request_state.set(state)
        return _RequestStateBinding(token)


_current_request_state: ContextVar[RequestState | None] = ContextVar(
    "dreamer_request_state", default=None
)


class _RequestStateBinding(AbstractContextManager["_RequestStateBinding"]):
    """Returned by :meth:`RequestState.bind` so callers can `with` it."""

    __slots__ = ("_token",)

    def __init__(self, token: Token[RequestState | None]) -> None:
        self._token = token

    def __enter__(self) -> _RequestStateBinding:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        _current_request_state.reset(self._token)


__all__ = ["RequestState"]
