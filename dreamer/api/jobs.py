"""JobQueue Protocol + DreamJob payload."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

from dreamer.api.types import DreamJob

if TYPE_CHECKING:
    from dreamer.api.contexts import PublishContext, SubscribeContext


@runtime_checkable
class JobQueue(Protocol):
    """Decouples trigger fire from in-process orchestrator execution.

    Default `InProcessJobQueue` schedules the subscribed handler as an
    `asyncio.create_task` and returns immediately — `publish` is fire-and-forget,
    so a trigger never blocks waiting for a dream to complete.
    """

    multi_tenant: ClassVar[bool] = False

    async def publish(self, job: DreamJob, *, ctx: PublishContext) -> None: ...
    async def subscribe(
        self,
        *,
        handler: Callable[[DreamJob], Awaitable[None]],
        ctx: SubscribeContext,
    ) -> None: ...


__all__ = ["DreamJob", "JobQueue"]
