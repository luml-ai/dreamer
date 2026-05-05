"""External trigger.

No internal scheduling. Fires only when the CLI or an admin-route component
invokes ``control.trigger_dream(tenant_id, trigger_name)``. ``start`` and
``stop`` are no-ops.
"""

from __future__ import annotations

from typing import ClassVar

from dreamer.api.compat import implements
from dreamer.api.contexts import (
    TriggerStartContext,
    TriggerStartServices,
    TriggerStopContext,
)
from dreamer.api.errors import ConfigError
from dreamer.api.triggers import Trigger
from dreamer.api.types import TenantId


@implements(Trigger, version=1)
class ExternalTrigger:
    """Trigger fired only via :func:`dreamer.server.control.trigger_dream`.

    Identity is the composite ``(tenant_id, name)``. Useful for jobs scheduled
    by Kubernetes CronJobs, Celery, AWS EventBridge, or any other out-of-process
    scheduler that hits the framework via the CLI or a custom admin route.
    """

    multi_tenant: ClassVar[bool] = False

    def __init__(
        self, *, name: str = "external", tenant_id: TenantId = "default"
    ) -> None:
        if not name:
            raise ConfigError("ExternalTrigger: name must be a non-empty string")
        self.name = name
        self.tenant_id = tenant_id

    async def start(
        self, *, ctx: TriggerStartContext, services: TriggerStartServices
    ) -> None:
        return None

    async def stop(self, *, ctx: TriggerStopContext) -> None:
        return None


__all__ = ["ExternalTrigger"]
