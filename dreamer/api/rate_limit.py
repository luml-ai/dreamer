"""RateLimiter Protocol + re-export of RateLimitDecision."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

from dreamer.api.types import Principal, RateLimitDecision, TenantId

if TYPE_CHECKING:
    from dreamer.api.contexts import RateLimitContext


@runtime_checkable
class RateLimiter(Protocol):
    """Checked at MCP entry (`submit_memory` + every `MCPTool.call`)."""

    multi_tenant: ClassVar[bool] = False

    async def check(
        self,
        *,
        principal: Principal,
        tenant_id: TenantId,
        action: str,
        ctx: RateLimitContext,
    ) -> RateLimitDecision: ...


__all__ = ["RateLimitDecision", "RateLimiter"]
