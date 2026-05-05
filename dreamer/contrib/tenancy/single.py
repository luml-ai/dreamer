"""``SingleTenant``: default ``Tenancy`` impl that returns ``principal.tenant_id``.

Multi-tenant deployments swap this out for a richer ``Tenancy`` that derives
``tenant_id`` from claims, headers, or other principal attributes.
"""

from __future__ import annotations

from typing import ClassVar

from dreamer.api.auth import Tenancy
from dreamer.api.compat import implements
from dreamer.api.contexts import TenancyContext
from dreamer.api.types import Principal, TenantId


@implements(Tenancy, version=1)
class SingleTenant:
    """Passthrough tenancy. Returns ``principal.tenant_id`` (defaults to ``"default"``)."""

    multi_tenant: ClassVar[bool] = False

    async def tenant_for(
        self, principal: Principal, *, ctx: TenancyContext
    ) -> TenantId:
        return principal.tenant_id


__all__ = ["SingleTenant"]
