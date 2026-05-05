from __future__ import annotations

import pytest

from dreamer.api.contexts import TenancyContext
from dreamer.api.types import DEFAULT_TENANT_ID, Principal
from dreamer.contrib.tenancy.single import SingleTenant


def _ctx(principal: Principal) -> TenancyContext:
    return TenancyContext(request_id="r1", principal=principal)


@pytest.mark.asyncio
async def test_returns_default_for_default_principal() -> None:
    tenancy = SingleTenant()
    p = Principal(id="agent-1")
    assert await tenancy.tenant_for(p, ctx=_ctx(p)) == DEFAULT_TENANT_ID


@pytest.mark.asyncio
async def test_passes_through_explicit_tenant_id() -> None:
    tenancy = SingleTenant()
    p = Principal(id="agent-1", tenant_id="alt")
    assert await tenancy.tenant_for(p, ctx=_ctx(p)) == "alt"
