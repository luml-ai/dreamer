from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import ClassVar

from dreamer.api.compat import implements
from dreamer.api.contexts import DreamGateContext, DreamGateServices
from dreamer.api.dream import DreamGate
from dreamer.api.errors import ConfigError
from dreamer.api.types import GateDecision, TenantId

CostQuery = Callable[[TenantId], Awaitable[float]]


@implements(DreamGate, version=1)
class DailyCostBudget:
    """Skip dreams when today's tenant cost exceeds ``daily_limit_dollars``.

    Parameters:
        daily_limit_dollars: hard cap; ``proceed=False`` once exceeded.
        query_today_cost: ``async (tenant_id) -> float`` returning today's spend.
    """

    multi_tenant: ClassVar[bool] = True

    def __init__(
        self,
        *,
        daily_limit_dollars: float,
        query_today_cost: CostQuery,
    ) -> None:
        if daily_limit_dollars < 0:
            raise ConfigError(
                "DailyCostBudget: daily_limit_dollars must be >= 0"
            )
        self.daily_limit_dollars = daily_limit_dollars
        self.query_today_cost = query_today_cost

    async def check(
        self, *, ctx: DreamGateContext, services: DreamGateServices
    ) -> GateDecision:
        try:
            spent = await self.query_today_cost(ctx.tenant_id)
        except Exception as exc:  # noqa: BLE001 — failure is fail-open by default
            return GateDecision(
                proceed=True,
                reason=f"budget_query_failed: {exc!s}",
                metadata={"error": str(exc)},
            )
        if spent >= self.daily_limit_dollars:
            return GateDecision(
                proceed=False,
                reason="budget_exceeded",
                metadata={
                    "spent_today": spent,
                    "limit": self.daily_limit_dollars,
                },
            )
        return GateDecision(
            proceed=True,
            metadata={
                "spent_today": spent,
                "limit": self.daily_limit_dollars,
            },
        )


__all__ = ["CostQuery", "DailyCostBudget"]
