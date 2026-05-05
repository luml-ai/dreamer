"""``NoOpRateLimiter`` — default rate limiter that always allows.

Multi-tenant safe (no shared state). Real deployments install a backend
that enforces per-action / per-tenant limits.
"""

from __future__ import annotations

from typing import ClassVar

from dreamer.api.compat import implements
from dreamer.api.contexts import RateLimitContext
from dreamer.api.rate_limit import RateLimiter
from dreamer.api.types import Principal, RateLimitDecision, TenantId


@implements(RateLimiter, version=1)
class NoOpRateLimiter:
    """Always returns ``allowed=True``.

    Used as the default so out-of-the-box deployments never block on rate
    limits; production setups should swap in a real limiter.
    """

    multi_tenant: ClassVar[bool] = True

    async def check(
        self,
        *,
        principal: Principal,
        tenant_id: TenantId,
        action: str,
        ctx: RateLimitContext,
    ) -> RateLimitDecision:
        return RateLimitDecision(allowed=True)


__all__ = ["NoOpRateLimiter"]
