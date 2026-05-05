"""AuditSink Protocol + re-export of AuditEvent."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

from dreamer.api.types import AuditEvent

if TYPE_CHECKING:
    from dreamer.api.contexts import AuditContext


@runtime_checkable
class AuditSink(Protocol):
    """Append-only event trail. Distinct from `UsageSink` (compliance, not billing)."""

    multi_tenant: ClassVar[bool] = False

    async def record(self, event: AuditEvent, *, ctx: AuditContext) -> None: ...


__all__ = ["AuditEvent", "AuditSink"]
