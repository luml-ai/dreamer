"""Cron-schedule trigger.

Uses APScheduler internally as a process-local helper. The orchestrator-supplied
``services.fire`` callback is invoked at each cron boundary; the callback
publishes a ``DreamJob`` to the configured ``JobQueue`` for the orchestrator
to pick up. APScheduler is *not* exposed as a Protocol — it is purely an
internal scheduling helper.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from dreamer.api.compat import implements
from dreamer.api.contexts import (
    TriggerStartContext,
    TriggerStartServices,
    TriggerStopContext,
)
from dreamer.api.errors import ConfigError
from dreamer.api.triggers import Trigger
from dreamer.api.types import TenantId

if TYPE_CHECKING:
    from apscheduler.schedulers.asyncio import (  # type: ignore[import-untyped]
        AsyncIOScheduler,
    )

logger = logging.getLogger(__name__)


@implements(Trigger, version=1)
class CronTrigger:
    """Fires on a cron schedule via APScheduler.

    Identity is the composite ``(tenant_id, name)``. The trigger fires by
    calling ``services.fire`` (the orchestrator-supplied callback) at each cron
    boundary.
    """

    multi_tenant: ClassVar[bool] = False

    def __init__(
        self,
        *,
        name: str,
        expression: str,
        tenant_id: TenantId = "default",
    ) -> None:
        if not name:
            raise ConfigError("CronTrigger: name must be a non-empty string")
        if not expression or not expression.strip():
            raise ConfigError("CronTrigger: expression must be a non-empty cron string")
        self.name = name
        self.tenant_id = tenant_id
        self.expression = expression
        self._scheduler: AsyncIOScheduler | None = None
        self._validate_expression(expression)

    @staticmethod
    def _validate_expression(expression: str) -> None:
        """Validate the cron expression at construction time.

        Raises ``ConfigError`` so a malformed expression fails fast at config
        load — not silently swallowed at first tick.
        """
        from apscheduler.triggers.cron import (  # type: ignore[import-untyped]
            CronTrigger as APSchedulerCronTrigger,
        )

        try:
            APSchedulerCronTrigger.from_crontab(expression)
        except Exception as exc:
            raise ConfigError(
                f"CronTrigger: invalid cron expression {expression!r}: {exc}"
            ) from exc

    async def start(
        self, *, ctx: TriggerStartContext, services: TriggerStartServices
    ) -> None:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.triggers.cron import (
            CronTrigger as APSchedulerCronTrigger,
        )

        if self._scheduler is not None:
            return

        scheduler = AsyncIOScheduler()
        fire = services.fire

        async def _fire_wrapper() -> None:
            try:
                await fire()
            except Exception:
                logger.exception(
                    "CronTrigger %s/%s: fire callback raised",
                    self.tenant_id,
                    self.name,
                )

        scheduler.add_job(
            _fire_wrapper,
            trigger=APSchedulerCronTrigger.from_crontab(self.expression),
            id=f"{self.tenant_id}:{self.name}",
            name=f"dreamer.cron[{self.tenant_id}/{self.name}]",
            replace_existing=True,
            misfire_grace_time=None,
            coalesce=True,
            max_instances=1,
        )
        scheduler.start()
        self._scheduler = scheduler
        logger.info(
            "CronTrigger started: tenant=%s name=%s expression=%r",
            self.tenant_id,
            self.name,
            self.expression,
        )

    async def stop(self, *, ctx: TriggerStopContext) -> None:
        if self._scheduler is None:
            return
        try:
            self._scheduler.shutdown(wait=False)
        except Exception:
            logger.exception(
                "CronTrigger %s/%s: scheduler shutdown raised",
                self.tenant_id,
                self.name,
            )
        self._scheduler = None


__all__ = ["CronTrigger"]
