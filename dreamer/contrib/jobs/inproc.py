"""In-process job queue.

`publish` schedules `asyncio.create_task` of the subscribed handler and returns
immediately — fire-and-forget so a trigger never blocks waiting for a dream to
complete. `subscribe` registers exactly one handler; calling it twice replaces
the previous handler. Tasks are tracked so `aclose()` can await any in-flight
handler invocations during graceful shutdown.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import ClassVar

from dreamer.api.compat import implements
from dreamer.api.contexts import LifecycleContext, PublishContext, SubscribeContext
from dreamer.api.jobs import JobQueue
from dreamer.api.types import DreamJob

logger = logging.getLogger(__name__)


@implements(JobQueue, version=1)
class InProcessJobQueue:
    """Default in-process job queue.

    Implements `Lifecycle@1` so `stop` can drain in-flight handlers when the
    server shuts down.
    """

    multi_tenant: ClassVar[bool] = True

    def __init__(self) -> None:
        self._handler: Callable[[DreamJob], Awaitable[None]] | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        self._closed = False

    async def publish(self, job: DreamJob, *, ctx: PublishContext) -> None:
        if self._closed:
            logger.warning("InProcessJobQueue closed; dropping job %r", job)
            return
        handler = self._handler
        if handler is None:
            logger.warning(
                "InProcessJobQueue has no subscriber; dropping job tenant=%s trigger=%s",
                job.tenant_id,
                job.trigger_name,
            )
            return

        async def _run() -> None:
            try:
                await handler(job)
            except Exception:  # noqa: BLE001 — handler errors must not crash the queue
                logger.exception(
                    "InProcessJobQueue handler raised for tenant=%s trigger=%s",
                    job.tenant_id,
                    job.trigger_name,
                )

        task = asyncio.create_task(_run())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def subscribe(
        self,
        *,
        handler: Callable[[DreamJob], Awaitable[None]],
        ctx: SubscribeContext,
    ) -> None:
        self._handler = handler

    async def start(self, *, ctx: LifecycleContext) -> None:
        self._closed = False

    async def stop(self, *, ctx: LifecycleContext) -> None:
        self._closed = True
        if not self._tasks:
            return
        await asyncio.gather(*self._tasks, return_exceptions=True)


__all__ = ["InProcessJobQueue"]
