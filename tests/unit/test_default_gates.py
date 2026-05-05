from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from dreamer.api.contexts import (
    DreamGateContext,
    DreamGateServices,
    SecretContext,
)
from dreamer.api.errors import ConfigError
from dreamer.api.types import (
    SecretValue,
)
from dreamer.contrib.gates import DailyCostBudget, SkipIfEmptyBatch


class _NullSecrets:
    async def get(
        self,
        name: str,
        *,
        tenant_id: str | None,
        ctx: SecretContext,
    ) -> SecretValue:
        return SecretValue(value="", ttl_seconds=None, version=None)


class _NullSink:
    async def record(self, event: Any, *, ctx: Any) -> None:
        return None


async def _noop_emit(message: str, payload: Mapping[str, Any]) -> None:
    return None


def _services() -> DreamGateServices:
    return DreamGateServices(
        emit_progress=_noop_emit,
        secrets=_NullSecrets(),
        usage=_NullSink(),
        audit=_NullSink(),
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )


def _ctx(*, unconsumed_count: int | None) -> DreamGateContext:
    return DreamGateContext(
        request_id="rq",
        tenant_id="default",
        trigger_name="every_6h",
        unconsumed_count=unconsumed_count,
    )


@pytest.mark.asyncio
async def test_skip_if_empty_batch_returns_proceed_false_for_zero() -> None:
    decision = await SkipIfEmptyBatch().check(
        ctx=_ctx(unconsumed_count=0), services=_services()
    )
    assert decision.proceed is False
    assert decision.reason == "empty"


@pytest.mark.asyncio
async def test_skip_if_empty_batch_proceeds_when_nonzero() -> None:
    decision = await SkipIfEmptyBatch().check(
        ctx=_ctx(unconsumed_count=5), services=_services()
    )
    assert decision.proceed is True


@pytest.mark.asyncio
async def test_skip_if_empty_batch_proceeds_when_unknown() -> None:
    decision = await SkipIfEmptyBatch().check(
        ctx=_ctx(unconsumed_count=None), services=_services()
    )
    assert decision.proceed is True
    assert "unknown" in decision.reason


@pytest.mark.asyncio
async def test_daily_cost_budget_blocks_when_exceeded() -> None:
    async def query(tenant_id: str) -> float:
        return 10.0

    gate = DailyCostBudget(daily_limit_dollars=5.0, query_today_cost=query)
    decision = await gate.check(
        ctx=_ctx(unconsumed_count=3), services=_services()
    )
    assert decision.proceed is False
    assert decision.reason == "budget_exceeded"
    assert decision.metadata["spent_today"] == 10.0


@pytest.mark.asyncio
async def test_daily_cost_budget_proceeds_when_under_limit() -> None:
    async def query(tenant_id: str) -> float:
        return 1.0

    gate = DailyCostBudget(daily_limit_dollars=5.0, query_today_cost=query)
    decision = await gate.check(
        ctx=_ctx(unconsumed_count=3), services=_services()
    )
    assert decision.proceed is True
    assert decision.metadata["spent_today"] == 1.0


@pytest.mark.asyncio
async def test_daily_cost_budget_fails_open_on_query_error() -> None:
    async def query(tenant_id: str) -> float:
        raise RuntimeError("billing system down")

    gate = DailyCostBudget(daily_limit_dollars=5.0, query_today_cost=query)
    decision = await gate.check(
        ctx=_ctx(unconsumed_count=3), services=_services()
    )
    assert decision.proceed is True
    assert "budget_query_failed" in decision.reason


def test_daily_cost_budget_validates_limit() -> None:
    async def query(tenant_id: str) -> float:
        return 0.0

    with pytest.raises(ConfigError):
        DailyCostBudget(daily_limit_dollars=-1.0, query_today_cost=query)
