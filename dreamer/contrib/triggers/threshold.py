"""STM-count threshold trigger.

Plain interval poller — no APScheduler dependency. Polls
``stm_store.count_unconsumed`` every ``interval_seconds``. Fires (calls
``services.fire``) on the transition from below-threshold to at-or-above
threshold; does **not** re-fire while the count remains ≥ threshold (only
fires again after the count drops below threshold and re-crosses).
"""

from __future__ import annotations

import asyncio
import logging
from typing import ClassVar

from dreamer.api.compat import implements
from dreamer.api.contexts import (
    CountContext,
    TriggerStartContext,
    TriggerStartServices,
    TriggerStopContext,
)
from dreamer.api.errors import ConfigError
from dreamer.api.stores import STMStore
from dreamer.api.tenants import TenantScope
from dreamer.api.triggers import Trigger
from dreamer.api.types import TenantId

logger = logging.getLogger(__name__)


@implements(Trigger, version=1)
class STMCountThresholdTrigger:
    """Fires when STM unconsumed count crosses up through ``threshold``.

    Identity is the composite ``(tenant_id, name)``. Fires only on the
    transition crossing the threshold; remaining at or above threshold does
    not produce repeat fires until the count first drops below threshold.
    """

    multi_tenant: ClassVar[bool] = False

    def __init__(
        self,
        *,
        name: str,
        threshold: int,
        interval_seconds: float,
        stm_store: STMStore,
        tenant_id: TenantId = "default",
    ) -> None:
        if not name:
            raise ConfigError("STMCountThresholdTrigger: name must be a non-empty string")
        if threshold < 1:
            raise ConfigError("STMCountThresholdTrigger: threshold must be >= 1")
        if interval_seconds <= 0:
            raise ConfigError("STMCountThresholdTrigger: interval_seconds must be > 0")
        self.name = name
        self.tenant_id = tenant_id
        self.threshold = threshold
        self.interval_seconds = interval_seconds
        self._stm_store = stm_store
        self._task: asyncio.Task[None] | None = None
        self._stop_event: asyncio.Event | None = None
        self._above_threshold = False

    async def start(
        self, *, ctx: TriggerStartContext, services: TriggerStartServices
    ) -> None:
        if self._task is not None:
            return
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(self._run(services))
        logger.info(
            "STMCountThresholdTrigger started: tenant=%s name=%s threshold=%d "
            "interval=%.2fs",
            self.tenant_id,
            self.name,
            self.threshold,
            self.interval_seconds,
        )

    async def stop(self, *, ctx: TriggerStopContext) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        task = self._task
        self._task = None
        if task is None:
            return
        try:
            await asyncio.wait_for(task, timeout=self.interval_seconds + 1.0)
        except TimeoutError:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    async def _run(self, services: TriggerStartServices) -> None:
        assert self._stop_event is not None
        request_id = f"trigger.{self.tenant_id}.{self.name}"
        ctx = CountContext(request_id=request_id, tenant_id=self.tenant_id)
        while not self._stop_event.is_set():
            try:
                await self._poll_once(ctx=ctx, services=services)
            except Exception:
                logger.exception(
                    "STMCountThresholdTrigger %s/%s: poll raised",
                    self.tenant_id,
                    self.name,
                )
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=self.interval_seconds
                )
                return
            except TimeoutError:
                continue

    async def _poll_once(
        self, *, ctx: CountContext, services: TriggerStartServices
    ) -> None:
        with TenantScope.set(self.tenant_id):
            count = await self._stm_store.count_unconsumed(ctx=ctx)
        previously_above = self._above_threshold
        if count >= self.threshold:
            if not previously_above:
                self._above_threshold = True
                try:
                    await services.fire()
                except Exception:
                    logger.exception(
                        "STMCountThresholdTrigger %s/%s: fire callback raised",
                        self.tenant_id,
                        self.name,
                    )
        else:
            self._above_threshold = False


__all__ = ["STMCountThresholdTrigger"]
