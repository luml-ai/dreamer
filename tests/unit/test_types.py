from __future__ import annotations

from datetime import UTC, datetime

import pytest

from dreamer.api.types import (
    DEFAULT_TENANT_ID,
    AuditEvent,
    Diff,
    DreamJob,
    DreamLease,
    GateDecision,
    Memory,
    MemoryBatch,
    MemoryType,
    Principal,
    RateLimitDecision,
    SecretValue,
    TenantConfig,
    UsageEvent,
)


def _now() -> datetime:
    return datetime(2026, 5, 2, 10, 15, 0, tzinfo=UTC)


def test_principal_round_trip() -> None:
    p = Principal(id="agent-1", tenant_id="acme", metadata={"role": "admin"})
    p2 = Principal.model_validate(p.model_dump())
    assert p2 == p
    assert p2.tenant_id == "acme"


def test_principal_default_tenant_id() -> None:
    p = Principal(id="agent-1")
    assert p.tenant_id == DEFAULT_TENANT_ID
    assert DEFAULT_TENANT_ID == "default"


def test_principal_extra_fields_allowed() -> None:
    p = Principal.model_validate({"id": "x", "tenant_id": "default", "extra": 5})
    assert p.id == "x"


def test_memory_type_round_trip_with_schema() -> None:
    mt = MemoryType(
        name="failure",
        description="An unexpected failure",
        metadata_schema={"type": "object"},
    )
    mt2 = MemoryType.model_validate(mt.model_dump())
    assert mt2 == mt
    assert mt2.metadata_schema_kind == "json-schema-2020-12"


def test_memory_round_trip_with_id() -> None:
    m = Memory(
        id="01H...",
        tenant_id="default",
        agent_id="agent-1",
        type="observation",
        title="something happened",
        content="body",
        tags=["a", "b"],
        metadata={"k": "v"},
        submitted_at=_now(),
        idempotency_key="abc",
    )
    m2 = Memory.model_validate(m.model_dump())
    assert m2 == m
    assert m2.consumed_at is None
    assert m2.consumed_by_lease is None


def test_memory_pre_persistence_id_none() -> None:
    m = Memory(
        tenant_id="default",
        agent_id="agent-1",
        type="observation",
        title="t",
        content="c",
        submitted_at=_now(),
    )
    assert m.id is None


def test_memory_batch_round_trip() -> None:
    m = Memory(
        id="m1",
        tenant_id="default",
        agent_id="agent-1",
        type="observation",
        title="t",
        content="c",
        submitted_at=_now(),
    )
    b = MemoryBatch(
        lease_id="lease-1",
        tenant_id="default",
        memories=[m],
        snapshot_at=_now(),
    )
    b2 = MemoryBatch.model_validate(b.model_dump())
    assert b2 == b


def test_diff_round_trip() -> None:
    d = Diff(
        added=["a.md"],
        modified=["b.md"],
        deleted=["c.md"],
        metadata={"bytes": 42},
    )
    d2 = Diff.model_validate(d.model_dump())
    assert d2 == d


def test_diff_defaults() -> None:
    d = Diff()
    assert d.added == []
    assert d.modified == []
    assert d.deleted == []


def test_dream_lease_round_trip() -> None:
    lease = DreamLease(
        id="lease-1",
        tenant_id="default",
        acquired_at=_now(),
        expires_at=_now(),
    )
    lease2 = DreamLease.model_validate(lease.model_dump())
    assert lease2 == lease


def test_dream_job_dataclass_default_payload() -> None:
    job = DreamJob(tenant_id="default", trigger_name="external")
    assert job.payload == {}
    assert job.tenant_id == "default"


def test_dream_job_is_frozen() -> None:
    job = DreamJob(tenant_id="default", trigger_name="external")
    with pytest.raises(Exception):
        job.tenant_id = "other"  # type: ignore[misc]


def test_tenant_config_defaults() -> None:
    tc = TenantConfig()
    assert tc.memory_types is None
    assert tc.dream_instructions is None
    assert tc.hook_params is None
    assert tc.metadata == {}


def test_tenant_config_is_frozen() -> None:
    tc = TenantConfig()
    with pytest.raises(Exception):
        tc.memory_types = ()  # type: ignore[misc]


def test_gate_decision_defaults() -> None:
    gd = GateDecision(proceed=True)
    assert gd.proceed is True
    assert gd.reason == ""
    assert gd.metadata == {}


def test_rate_limit_decision_defaults() -> None:
    d = RateLimitDecision(allowed=True)
    assert d.retry_after_seconds is None
    assert d.reason == ""


def test_secret_value_defaults() -> None:
    s = SecretValue(value="sek")
    assert s.value == "sek"
    assert s.ttl_seconds is None
    assert s.version is None


def test_usage_event_round_trip() -> None:
    u = UsageEvent(
        tenant_id="default",
        component="dreamer.contrib.dream.claude_agent.ClaudeAgentDreamEngine",
        kind="llm_tokens_in",
        amount=42.0,
        unit="tokens",
        at=_now(),
    )
    u2 = UsageEvent.model_validate(u.model_dump())
    assert u2 == u


def test_audit_event_round_trip() -> None:
    e = AuditEvent(
        event_type="memory.submit",
        principal_id="agent-1",
        tenant_id="default",
        payload={"id": "m1"},
        at=_now(),
    )
    e2 = AuditEvent.model_validate(e.model_dump())
    assert e2 == e
