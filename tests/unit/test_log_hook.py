from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from dreamer.api.contexts import (
    DreamFailedContext,
    DreamFailedServices,
    PostContextUpdateContext,
    PostContextUpdateServices,
    PostDreamContext,
    PostDreamServices,
    PostLTMUpdateContext,
    PostLTMUpdateServices,
    SecretContext,
)
from dreamer.api.types import Diff, SecretValue
from dreamer.contrib.hooks.log import LogHook


class _NullSecrets:
    async def get(
        self,
        name: str,
        *,
        tenant_id: str | None,
        ctx: SecretContext,
    ) -> SecretValue:
        return SecretValue(value="")


class _NullSink:
    async def record(self, event: Any, *, ctx: Any) -> None:
        return None


async def _noop_emit(message: str, payload: Mapping[str, Any]) -> None:
    return None


def _common_kwargs() -> dict[str, Any]:
    return dict(
        emit_progress=_noop_emit,
        secrets=_NullSecrets(),
        usage=_NullSink(),
        audit=_NullSink(),
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )


@pytest.mark.asyncio
async def test_log_hook_post_dream(caplog: pytest.LogCaptureFixture) -> None:
    hook = LogHook()
    ctx = PostDreamContext(
        request_id="r",
        tenant_id="default",
        lease_id="L",
        trigger_name="t",
        success=True,
        batch_size=3,
        ltm_diff=Diff(added=["a"]),
        context_diff=Diff(modified=["b"]),
        resumed=False,
        error=None,
    )
    services = PostDreamServices(**_common_kwargs())
    with caplog.at_level("INFO"):
        await hook.on_post_dream(ctx=ctx, services=services)
    assert any("post_dream" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_log_hook_post_ltm_update(caplog: pytest.LogCaptureFixture) -> None:
    hook = LogHook()
    ctx = PostLTMUpdateContext(
        request_id="r",
        tenant_id="default",
        lease_id="L",
        ltm_workspace_id="ws",
        ltm_diff=Diff(added=["a", "b"]),
    )
    services = PostLTMUpdateServices(**_common_kwargs())
    with caplog.at_level("INFO"):
        await hook.on_post_ltm_update(ctx=ctx, services=services)
    assert any("post_ltm_update" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_log_hook_post_context_update(caplog: pytest.LogCaptureFixture) -> None:
    hook = LogHook()
    ctx = PostContextUpdateContext(
        request_id="r",
        tenant_id="default",
        lease_id="L",
        context_workspace_id="ws",
        context_diff=Diff(modified=["AGENTS.md"]),
    )
    services = PostContextUpdateServices(**_common_kwargs())
    with caplog.at_level("INFO"):
        await hook.on_post_context_update(ctx=ctx, services=services)
    assert any("post_context_update" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_log_hook_dream_failed(caplog: pytest.LogCaptureFixture) -> None:
    hook = LogHook()
    ctx = DreamFailedContext(
        request_id="r",
        tenant_id="default",
        lease_id="L",
        trigger_name="t",
        phase="ltm",
        error="boom",
    )
    services = DreamFailedServices(**_common_kwargs())
    with caplog.at_level("WARNING"):
        await hook.on_dream_failed(ctx=ctx, services=services)
    assert any("dream_failed" in rec.message for rec in caplog.records)
