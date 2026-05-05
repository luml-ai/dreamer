from __future__ import annotations

from datetime import UTC, datetime

import pytest

from dreamer.api.contexts import (
    AuditContext,
    RateLimitContext,
    SecretContext,
    UsageContext,
)
from dreamer.api.types import (
    AuditEvent,
    Principal,
    UsageEvent,
)
from dreamer.contrib.audit.log import LogAuditSink
from dreamer.contrib.rate_limit.noop import NoOpRateLimiter
from dreamer.contrib.secrets.env import EnvSecretResolver
from dreamer.contrib.usage.log import LogUsageSink


@pytest.mark.asyncio
async def test_log_audit_sink_emits_log(caplog: pytest.LogCaptureFixture) -> None:
    sink = LogAuditSink()
    event = AuditEvent(
        event_type="dream.lease_acquired",
        principal_id=None,
        tenant_id="default",
        payload={"k": "v"},
        at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    with caplog.at_level("INFO", logger="dreamer.audit"):
        await sink.record(event, ctx=AuditContext(request_id="r", tenant_id="default"))
    assert any("dream.lease_acquired" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_log_usage_sink_emits_log(caplog: pytest.LogCaptureFixture) -> None:
    sink = LogUsageSink()
    event = UsageEvent(
        tenant_id="default",
        component="dreamer.test",
        kind="wall_seconds",
        amount=1.5,
        unit="s",
        at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    with caplog.at_level("INFO", logger="dreamer.usage"):
        await sink.record(event, ctx=UsageContext(request_id="r", tenant_id="default"))
    assert any("wall_seconds" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_env_secret_resolver_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_SECRET", "super")
    resolver = EnvSecretResolver()
    value = await resolver.get(
        "MY_SECRET",
        tenant_id="default",
        ctx=SecretContext(request_id="r", tenant_id="default"),
    )
    assert value.value == "super"
    assert value.ttl_seconds is None
    assert value.version is None


@pytest.mark.asyncio
async def test_env_secret_resolver_returns_empty_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("UNDEFINED_SECRET_XYZ", raising=False)
    resolver = EnvSecretResolver()
    value = await resolver.get(
        "UNDEFINED_SECRET_XYZ",
        tenant_id="default",
        ctx=SecretContext(request_id="r", tenant_id="default"),
    )
    assert value.value == ""


@pytest.mark.asyncio
async def test_env_secret_resolver_per_tenant_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DREAMER_acme_GH_TOKEN", "scoped")
    monkeypatch.setenv("GH_TOKEN", "global")
    resolver = EnvSecretResolver(prefix="DREAMER_${tenant_id}_")
    scoped = await resolver.get(
        "GH_TOKEN",
        tenant_id="acme",
        ctx=SecretContext(request_id="r", tenant_id="acme"),
    )
    assert scoped.value == "scoped"
    fallback = await resolver.get(
        "GH_TOKEN",
        tenant_id="other",
        ctx=SecretContext(request_id="r", tenant_id="other"),
    )
    assert fallback.value == "global"


@pytest.mark.asyncio
async def test_noop_rate_limiter_always_allows() -> None:
    rl = NoOpRateLimiter()
    decision = await rl.check(
        principal=Principal(id="p"),
        tenant_id="default",
        action="mcp.submit_memory",
        ctx=RateLimitContext(request_id="r", tenant_id="default"),
    )
    assert decision.allowed is True
    assert decision.retry_after_seconds is None
