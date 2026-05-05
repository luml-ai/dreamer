"""``LogUsageSink`` — default usage sink that emits a structured log line.

Used for billing/observability. Operators expecting durable accounting
should configure a buffering sink alongside or in place of this one.
"""

from __future__ import annotations

import logging
from typing import ClassVar

from dreamer.api.compat import implements
from dreamer.api.contexts import UsageContext
from dreamer.api.types import UsageEvent
from dreamer.api.usage import UsageSink

logger = logging.getLogger("dreamer.usage")


@implements(UsageSink, version=1)
class LogUsageSink:
    multi_tenant: ClassVar[bool] = True

    def __init__(self, *, level: str = "INFO") -> None:
        self.level = getattr(logging, level.upper(), logging.INFO)

    async def record(self, event: UsageEvent, *, ctx: UsageContext) -> None:
        logger.log(
            self.level,
            "usage tenant=%s component=%s kind=%s amount=%s unit=%s at=%s",
            event.tenant_id,
            event.component,
            event.kind,
            event.amount,
            event.unit,
            event.at.isoformat(),
        )


__all__ = ["LogUsageSink"]
