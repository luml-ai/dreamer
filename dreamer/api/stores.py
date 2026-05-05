"""Store Protocols: STMStore, LTMStore, ContextPendingStore, ContextStore,
DreamLeaseStore, ContextReader, STMSerializer."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

from dreamer.api.types import (
    Diff,
    DreamLease,
    Memory,
    MemoryBatch,
    Workspace,
)

if TYPE_CHECKING:
    from dreamer.api.contexts import (
        AcquireLeaseContext,
        ClaimContext,
        ClearContextPendingContext,
        CommitWorkspaceContext,
        ContextReadContext,
        CountContext,
        DiscardWorkspaceContext,
        GetContextPendingContext,
        ListUnconsumedContext,
        MarkConsumedContext,
        MCPToolContext,
        OpenWorkspaceContext,
        PurgeConsumedContext,
        ReclaimContext,
        ReclaimLeasesContext,
        ReleaseContext,
        ReleaseLeaseContext,
        RenewLeaseContext,
        SerializeContext,
        SerializeServices,
        SetContextPendingContext,
        SubmitContext,
    )


@runtime_checkable
class STMStore(Protocol):
    """Short-term memory store."""

    multi_tenant: ClassVar[bool] = False

    async def submit(self, memory: Memory, *, ctx: SubmitContext) -> Memory: ...
    async def list_unconsumed(self, *, ctx: ListUnconsumedContext) -> list[Memory]: ...
    async def claim_batch(self, *, ctx: ClaimContext) -> MemoryBatch: ...
    async def mark_consumed(self, *, ctx: MarkConsumedContext) -> None: ...
    async def release_unconsumed(self, *, ctx: ReleaseContext) -> None: ...
    async def count_unconsumed(self, *, ctx: CountContext) -> int: ...
    async def release_for_expired_leases(self, *, ctx: ReclaimContext) -> int: ...
    async def purge_consumed(self, *, ctx: PurgeConsumedContext) -> int: ...


@runtime_checkable
class LTMStore(Protocol):
    """Long-term memory store. Workspace-shaped."""

    multi_tenant: ClassVar[bool] = False
    workspace_capabilities: ClassVar[frozenset[type]]

    async def open_workspace(self, *, ctx: OpenWorkspaceContext) -> Workspace: ...
    async def commit_workspace(
        self, ws: Workspace, *, ctx: CommitWorkspaceContext
    ) -> Diff: ...
    async def discard_workspace(
        self, ws: Workspace, *, ctx: DiscardWorkspaceContext
    ) -> None: ...


@runtime_checkable
class ContextPendingStore(Protocol):
    """Optional capability on `LTMStore`. Persists the `context_pending`
    watermark used by the cross-store consistency contract."""

    multi_tenant: ClassVar[bool] = False

    async def set_context_pending(
        self, diff: Diff, *, ctx: SetContextPendingContext
    ) -> None: ...
    async def get_context_pending(self, *, ctx: GetContextPendingContext) -> Diff | None: ...
    async def clear_context_pending(self, *, ctx: ClearContextPendingContext) -> None: ...


@runtime_checkable
class ContextStore(Protocol):
    """Same shape as LTMStore (workspace_capabilities, open/commit/discard)."""

    multi_tenant: ClassVar[bool] = False
    workspace_capabilities: ClassVar[frozenset[type]]

    async def open_workspace(self, *, ctx: OpenWorkspaceContext) -> Workspace: ...
    async def commit_workspace(
        self, ws: Workspace, *, ctx: CommitWorkspaceContext
    ) -> Diff: ...
    async def discard_workspace(
        self, ws: Workspace, *, ctx: DiscardWorkspaceContext
    ) -> None: ...


@runtime_checkable
class DreamLeaseStore(Protocol):
    """Per-tenant dream lease."""

    multi_tenant: ClassVar[bool] = False

    async def acquire(self, *, ctx: AcquireLeaseContext) -> DreamLease | None: ...
    async def renew(self, *, ctx: RenewLeaseContext) -> bool: ...
    async def release(self, *, ctx: ReleaseLeaseContext) -> None: ...
    async def reclaim_expired(self, *, ctx: ReclaimLeasesContext) -> frozenset[str]: ...


@runtime_checkable
class ContextReader(Protocol):
    """Optional capability on ContextStore: serve context content over HTTP."""

    multi_tenant: ClassVar[bool] = False

    async def read(self, path: str, *, ctx: ContextReadContext) -> bytes: ...
    async def list(self, *, prefix: str = "", ctx: ContextReadContext) -> list[str]: ...


@runtime_checkable
class MCPTool(Protocol):
    """Custom MCP tool exposed alongside the built-in submit_memory."""

    multi_tenant: ClassVar[bool] = False
    name: str
    description: str

    def input_schema(self) -> Mapping[str, Any]: ...
    async def call(self, args: Mapping[str, Any], *, ctx: MCPToolContext) -> Any: ...


@runtime_checkable
class STMSerializer(Protocol):
    """Materializes a MemoryBatch into a sandbox directory for the dream engine."""

    multi_tenant: ClassVar[bool] = False
    kind: ClassVar[str]

    async def write(
        self,
        batch: MemoryBatch,
        *,
        target: Path,
        ctx: SerializeContext,
        services: SerializeServices,
    ) -> None: ...
    def prompt_fragment(self, batch: MemoryBatch) -> str: ...


__all__ = [
    "ContextPendingStore",
    "ContextReader",
    "ContextStore",
    "DreamLeaseStore",
    "LTMStore",
    "MCPTool",
    "STMSerializer",
    "STMStore",
]
