"""Public framework types.

Core data model: the Pydantic models and frozen dataclasses every component
deals in. Also defines the abstract `Workspace` Protocol plus its built-in
capability Protocols (`FileViewable`, `RecordViewable`, `GraphViewable`).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

TenantId = str

DEFAULT_TENANT_ID: TenantId = "default"


class Principal(BaseModel):
    id: str
    tenant_id: TenantId = DEFAULT_TENANT_ID
    metadata: dict[str, Any] = {}
    model_config = ConfigDict(extra="allow")


class MemoryType(BaseModel):
    """Declared in config; agents may only submit memories of declared types."""

    name: str
    description: str
    metadata_schema: dict[str, Any] | None = None
    metadata_schema_kind: Literal["json-schema-2020-12"] = "json-schema-2020-12"


class Memory(BaseModel):
    id: str | None = None
    tenant_id: TenantId
    agent_id: str
    type: str
    title: str
    content: str
    tags: list[str] = []
    metadata: dict[str, Any] = {}
    submitted_at: datetime
    consumed_at: datetime | None = None
    consumed_by_lease: str | None = None
    idempotency_key: str | None = None
    model_config = ConfigDict(extra="allow")


class MemorySubmission(BaseModel):
    """Outcome of persisting one memory through the shared submit pipeline.

    ``deduplicated`` is true when the store returned a previously persisted
    memory for the same ``(tenant_id, idempotency_key)`` instead of storing a
    new one.
    """

    memory: Memory
    deduplicated: bool = False


class MemoryBatch(BaseModel):
    lease_id: str
    tenant_id: TenantId
    memories: list[Memory]
    snapshot_at: datetime
    metadata: dict[str, Any] = {}


class Diff(BaseModel):
    """File-level diff returned by stores after a dream sub-phase."""

    added: list[str] = []
    modified: list[str] = []
    deleted: list[str] = []
    metadata: dict[str, Any] = {}


class DreamLease(BaseModel):
    id: str
    tenant_id: TenantId
    acquired_at: datetime
    expires_at: datetime


@runtime_checkable
class Workspace(Protocol):
    """Identity + metadata. No I/O.

    `id` is opaque, stable for the workspace's lifetime; surfaced in *Context
    fields like `ltm_workspace_id` and used for tracing, audit events, and
    cross-process correlation.
    """

    id: str
    tenant_id: TenantId
    metadata: Mapping[str, Any]


@runtime_checkable
class FileViewable(Protocol):
    """The workspace can be viewed as a local filesystem directory."""

    async def file_view(self) -> Path: ...


@runtime_checkable
class RecordViewable(Protocol):
    """The workspace can be read/written as records (key → value)."""

    async def list_keys(self, *, prefix: str = "") -> AsyncIterator[str]: ...
    async def read_record(self, key: str) -> Any: ...
    async def write_record(self, key: str, value: Any) -> None: ...
    async def delete_record(self, key: str) -> None: ...


@runtime_checkable
class GraphViewable(Protocol):
    """The workspace can be queried/mutated as a knowledge graph."""

    async def query(self, sparql: str) -> AsyncIterator[Mapping[str, Any]]: ...
    async def update(self, sparql_update: str) -> None: ...
    async def export(self, *, format: str = "turtle") -> bytes: ...


@dataclass(frozen=True, slots=True)
class DreamJob:
    """Single payload type that crosses `JobQueue`."""

    tenant_id: TenantId
    trigger_name: str
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TenantConfig:
    """Per-tenant overrides. Each field is optional; None = inherit global."""

    memory_types: tuple[MemoryType, ...] | None = None
    dream_instructions: Mapping[str, str] | None = None
    hook_params: Mapping[str, Mapping[str, Any]] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GateDecision:
    proceed: bool
    reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RateLimitDecision:
    allowed: bool
    retry_after_seconds: float | None = None
    reason: str = ""


@dataclass(frozen=True, slots=True)
class SecretValue:
    value: str
    ttl_seconds: float | None = None
    version: str | None = None


class UsageEvent(BaseModel):
    tenant_id: TenantId
    component: str
    kind: str
    amount: float
    unit: str
    metadata: dict[str, Any] = {}
    at: datetime
    model_config = ConfigDict(extra="allow")


class AuditEvent(BaseModel):
    event_type: str
    principal_id: str | None
    tenant_id: TenantId
    payload: dict[str, Any]
    at: datetime
    model_config = ConfigDict(extra="allow")


__all__ = [
    "DEFAULT_TENANT_ID",
    "AuditEvent",
    "Diff",
    "DreamJob",
    "DreamLease",
    "FileViewable",
    "GateDecision",
    "GraphViewable",
    "Memory",
    "MemoryBatch",
    "MemorySubmission",
    "MemoryType",
    "Principal",
    "RateLimitDecision",
    "RecordViewable",
    "SecretValue",
    "TenantConfig",
    "TenantId",
    "UsageEvent",
    "Workspace",
]
