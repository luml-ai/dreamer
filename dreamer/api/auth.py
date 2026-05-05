"""AuthBackend and Tenancy Protocols."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

from dreamer.api.types import Principal, TenantId

if TYPE_CHECKING:
    from starlette.requests import Request

    from dreamer.api.contexts import AuthContext, TenancyContext


@runtime_checkable
class AuthBackend(Protocol):
    """Authenticates a request and produces a Principal.

    Backends MAY also expose Routes and/or Middlewares capabilities (login flows,
    session middleware). Simple bearer-token impls only implement `authenticate`.
    """

    multi_tenant: ClassVar[bool] = False

    async def authenticate(self, request: Request, *, ctx: AuthContext) -> Principal: ...


@runtime_checkable
class Tenancy(Protocol):
    """Override point for deriving tenant id from principal."""

    multi_tenant: ClassVar[bool] = False

    async def tenant_for(self, principal: Principal, *, ctx: TenancyContext) -> TenantId: ...


__all__ = ["AuthBackend", "Tenancy"]
