from __future__ import annotations

from typing import ClassVar

from dreamer.api.compat import implements
from dreamer.api.contexts import DreamGateContext, DreamGateServices
from dreamer.api.dream import DreamGate
from dreamer.api.types import GateDecision


@implements(DreamGate, version=1)
class SkipIfEmptyBatch:
    """Skip dreams when ``ctx.unconsumed_count == 0``.

    Treats ``unconsumed_count is None`` as "unknown" and proceeds — a
    conservative default for callers that don't supply the snapshot.
    """

    multi_tenant: ClassVar[bool] = True

    async def check(
        self, *, ctx: DreamGateContext, services: DreamGateServices
    ) -> GateDecision:
        if ctx.unconsumed_count is None:
            return GateDecision(proceed=True, reason="unknown_count")
        if ctx.unconsumed_count == 0:
            return GateDecision(
                proceed=False,
                reason="empty",
                metadata={"unconsumed_count": 0},
            )
        return GateDecision(
            proceed=True,
            metadata={"unconsumed_count": ctx.unconsumed_count},
        )


__all__ = ["SkipIfEmptyBatch"]
