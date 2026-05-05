from __future__ import annotations

import logging
from typing import ClassVar

from dreamer.api.audit import AuditSink
from dreamer.api.compat import implements
from dreamer.api.contexts import AuditContext
from dreamer.api.types import AuditEvent

logger = logging.getLogger("dreamer.audit")


@implements(AuditSink, version=1)
class LogAuditSink:
    multi_tenant: ClassVar[bool] = True

    def __init__(self, *, level: str = "INFO") -> None:
        self.level = getattr(logging, level.upper(), logging.INFO)

    async def record(self, event: AuditEvent, *, ctx: AuditContext) -> None:
        logger.log(
            self.level,
            "audit event_type=%s tenant=%s principal=%s payload=%s at=%s",
            event.event_type,
            event.tenant_id,
            event.principal_id,
            event.payload,
            event.at.isoformat(),
        )


__all__ = ["LogAuditSink"]
