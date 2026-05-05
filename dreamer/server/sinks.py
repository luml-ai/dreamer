"""Fan-out helpers for usage and audit sink emission.

`UsageSink` and `AuditSink` are fire-and-forget. The framework dispatches to
all configured sinks in parallel; a failed sink is logged but does not abort
the dream or the request. Operators wanting strong delivery guarantees
configure a buffering sink that persists locally and ships out-of-band.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence

from dreamer.api.audit import AuditSink
from dreamer.api.contexts import AuditContext, UsageContext
from dreamer.api.types import AuditEvent, UsageEvent
from dreamer.api.usage import UsageSink

logger = logging.getLogger(__name__)


async def emit_audit(
    sinks: Sequence[AuditSink], event: AuditEvent, *, ctx: AuditContext
) -> None:
    """Fan out an audit event to every configured sink in parallel.

    Failures are logged and swallowed so a broken sink cannot abort the
    request.
    """
    if not sinks:
        return
    results = await asyncio.gather(
        *(sink.record(event, ctx=ctx) for sink in sinks),
        return_exceptions=True,
    )
    for sink, result in zip(sinks, results, strict=True):
        if isinstance(result, BaseException):
            logger.warning(
                "AuditSink %s.%s raised %s for event %s",
                type(sink).__module__,
                type(sink).__qualname__,
                type(result).__name__,
                event.event_type,
            )


async def emit_usage(
    sinks: Sequence[UsageSink], event: UsageEvent, *, ctx: UsageContext
) -> None:
    """Fan out a usage event to every configured sink in parallel."""
    if not sinks:
        return
    results = await asyncio.gather(
        *(sink.record(event, ctx=ctx) for sink in sinks),
        return_exceptions=True,
    )
    for sink, result in zip(sinks, results, strict=True):
        if isinstance(result, BaseException):
            logger.warning(
                "UsageSink %s.%s raised %s for event %s/%s",
                type(sink).__module__,
                type(sink).__qualname__,
                type(result).__name__,
                event.component,
                event.kind,
            )


__all__ = ["emit_audit", "emit_usage"]
