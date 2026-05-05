from __future__ import annotations

import logging
from typing import ClassVar

from dreamer.api.compat import implements
from dreamer.api.contexts import (
    DreamFailedContext,
    DreamFailedServices,
    PostContextUpdateContext,
    PostContextUpdateServices,
    PostDreamContext,
    PostDreamServices,
    PostLTMUpdateContext,
    PostLTMUpdateServices,
)
from dreamer.api.hooks import (
    DreamFailedHook,
    PostContextUpdateHook,
    PostDreamHook,
    PostLTMUpdateHook,
)

logger = logging.getLogger(__name__)


@implements(PostDreamHook, version=1)
@implements(PostLTMUpdateHook, version=1)
@implements(PostContextUpdateHook, version=1)
@implements(DreamFailedHook, version=1)
class LogHook:
    """Logs the canonical post-* and on_dream_failed events."""

    multi_tenant: ClassVar[bool] = True

    def __init__(self, *, level: str = "INFO") -> None:
        self.level = getattr(logging, level.upper(), logging.INFO)

    async def on_post_dream(
        self, *, ctx: PostDreamContext, services: PostDreamServices
    ) -> None:
        logger.log(
            self.level,
            "post_dream tenant=%s lease=%s trigger=%s success=%s batch_size=%d "
            "ltm_diff=%s context_diff=%s resumed=%s error=%s",
            ctx.tenant_id,
            ctx.lease_id,
            ctx.trigger_name,
            ctx.success,
            ctx.batch_size,
            _diff_summary(ctx.ltm_diff),
            _diff_summary(ctx.context_diff),
            ctx.resumed,
            ctx.error,
        )

    async def on_post_ltm_update(
        self, *, ctx: PostLTMUpdateContext, services: PostLTMUpdateServices
    ) -> None:
        logger.log(
            self.level,
            "post_ltm_update tenant=%s lease=%s ltm_diff=%s",
            ctx.tenant_id,
            ctx.lease_id,
            _diff_summary(ctx.ltm_diff),
        )

    async def on_post_context_update(
        self, *, ctx: PostContextUpdateContext, services: PostContextUpdateServices
    ) -> None:
        logger.log(
            self.level,
            "post_context_update tenant=%s lease=%s context_diff=%s",
            ctx.tenant_id,
            ctx.lease_id,
            _diff_summary(ctx.context_diff),
        )

    async def on_dream_failed(
        self, *, ctx: DreamFailedContext, services: DreamFailedServices
    ) -> None:
        logger.log(
            max(self.level, logging.WARNING),
            "dream_failed tenant=%s lease=%s phase=%s trigger=%s error=%s",
            ctx.tenant_id,
            ctx.lease_id,
            ctx.phase,
            ctx.trigger_name,
            ctx.error,
        )


def _diff_summary(diff: object) -> str:
    if diff is None:
        return "None"
    added = getattr(diff, "added", None) or []
    modified = getattr(diff, "modified", None) or []
    deleted = getattr(diff, "deleted", None) or []
    return f"+{len(added)}~{len(modified)}-{len(deleted)}"


__all__ = ["LogHook"]
