"""Trigger Protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

from dreamer.api.types import TenantId

if TYPE_CHECKING:
    from dreamer.api.contexts import (
        TriggerStartContext,
        TriggerStartServices,
        TriggerStopContext,
    )


@runtime_checkable
class Trigger(Protocol):
    """Independent component that fires the orchestrator on its own schedule.

    Identity is the composite (tenant_id, name).
    """

    multi_tenant: ClassVar[bool] = False
    name: str
    tenant_id: TenantId

    async def start(
        self, *, ctx: TriggerStartContext, services: TriggerStartServices
    ) -> None: ...
    async def stop(self, *, ctx: TriggerStopContext) -> None: ...


__all__ = ["Trigger"]
