"""UsageSink Protocol + re-export of UsageEvent."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

from dreamer.api.types import UsageEvent

if TYPE_CHECKING:
    from dreamer.api.contexts import UsageContext


@runtime_checkable
class UsageSink(Protocol):
    """Per-phase, per-tenant usage emission for billing and quota enforcement."""

    multi_tenant: ClassVar[bool] = False

    async def record(self, event: UsageEvent, *, ctx: UsageContext) -> None: ...


__all__ = ["UsageEvent", "UsageSink"]
