"""Multi-target context publishing via :class:`FanoutContextStore`.

Wraps a list of inner :class:`ContextStore` backings; intersects their
``workspace_capabilities`` (rejecting non-uniform sets), routes a single
staging directory through ``FileViewable``, replicates on commit, and on
any backing failure rolls back remaining opens before raising
:class:`WorkspaceError` describing which backings already succeeded.

LTM is intentionally not fanned out — replicating LTM is better expressed as
a ``post_ltm_update`` backup hook.
"""

from __future__ import annotations

import asyncio
import shutil
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from dreamer.api.contexts import (
    CommitWorkspaceContext,
    DiscardWorkspaceContext,
    OpenWorkspaceContext,
)
from dreamer.api.errors import ConfigError, WorkspaceError
from dreamer.api.stores import ContextStore
from dreamer.api.tenants import TenantScope
from dreamer.api.types import Diff, FileViewable, TenantId, Workspace


@dataclass
class FanoutWorkspace:
    """Composite workspace returned by :class:`FanoutContextStore`.

    ``staging_path`` is the single local directory the engine edits when
    ``FileViewable`` is in the intersected capability set. ``inner`` lists the
    per-backing workspaces in the same order as the backings themselves.
    """

    id: str
    tenant_id: TenantId
    staging_path: Path
    inner: tuple[Workspace, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    async def file_view(self) -> Path:
        return self.staging_path


class FanoutContextStore:
    """Aggregating context store; commits write through to every backing.

    Not decorated with ``@implements(ContextStore)`` because it composes
    backings and doesn't have its own version contract. The framework picks it
    up by checking ``ContextStore`` Protocol membership; :class:`runtime_checkable`
    handles structural compliance.
    """

    def __init__(self, backings: Iterable[ContextStore]) -> None:
        self._backings: tuple[ContextStore, ...] = tuple(backings)
        if len(self._backings) < 1:
            raise ConfigError("FanoutContextStore requires at least one backing")
        first = self._backings[0].workspace_capabilities
        for backing in self._backings[1:]:
            if backing.workspace_capabilities != first:
                raise ConfigError(
                    "FanoutContextStore backings have non-uniform capabilities; "
                    "align them or run independent stores via separate hooks"
                )
        self.workspace_capabilities: frozenset[type] = frozenset(first)
        self.multi_tenant: bool = all(
            getattr(b, "multi_tenant", False) for b in self._backings
        )
        self._lock = asyncio.Lock()
        self._open: dict[str, FanoutWorkspace] = {}

    @property
    def backings(self) -> tuple[ContextStore, ...]:
        return self._backings

    async def open_workspace(self, *, ctx: OpenWorkspaceContext) -> Workspace:
        TenantScope.assert_matches(ctx.tenant_id)
        ws_id = str(uuid4())
        import tempfile

        staging = Path(tempfile.mkdtemp(prefix=f"dreamer-fanout-{ws_id}-"))
        inner: list[Workspace] = []
        try:
            for backing in self._backings:
                ws = await backing.open_workspace(ctx=ctx)
                inner.append(ws)
                if FileViewable in self.workspace_capabilities:
                    # Seed staging from the first successfully opened workspace
                    # so the engine sees the canonical pre-edit tree.
                    if len(inner) == 1:
                        path = await ws.file_view()  # type: ignore[attr-defined]
                        for p in path.rglob("*"):
                            if p.is_file():
                                rel = p.relative_to(path)
                                dest = staging / rel
                                dest.parent.mkdir(parents=True, exist_ok=True)
                                dest.write_bytes(p.read_bytes())
        except Exception:
            for backing, ws in zip(self._backings, inner, strict=False):
                try:
                    await backing.discard_workspace(
                        ws,
                        ctx=DiscardWorkspaceContext(
                            request_id=ctx.request_id, tenant_id=ctx.tenant_id
                        ),
                    )
                except Exception:
                    pass
            shutil.rmtree(staging, ignore_errors=True)
            raise
        async with self._lock:
            fw = FanoutWorkspace(
                id=ws_id,
                tenant_id=ctx.tenant_id,
                staging_path=staging,
                inner=tuple(inner),
                metadata={"backings": [type(b).__name__ for b in self._backings]},
            )
            self._open[ws_id] = fw
            return fw

    async def commit_workspace(
        self, ws: Workspace, *, ctx: CommitWorkspaceContext
    ) -> Diff:
        TenantScope.assert_matches(ctx.tenant_id)
        async with self._lock:
            fw = self._open.get(ws.id)
            if fw is None:
                raise RuntimeError(f"unknown workspace id: {ws.id}")

        per_store: list[dict[str, Any]] = []
        committed_indices: list[int] = []
        added: set[str] = set()
        modified: set[str] = set()
        deleted: set[str] = set()

        # Sequential commit so a mid-stream failure leaves a clear boundary
        # between committed and not-yet-committed backings.
        for idx, (backing, inner_ws) in enumerate(
            zip(self._backings, fw.inner, strict=True)
        ):
            try:
                if FileViewable in self.workspace_capabilities:
                    inner_path = await inner_ws.file_view()  # type: ignore[attr-defined]
                    self._mirror(fw.staging_path, inner_path)
                diff = await backing.commit_workspace(
                    inner_ws,
                    ctx=CommitWorkspaceContext(
                        request_id=ctx.request_id,
                        tenant_id=ctx.tenant_id,
                        metadata=ctx.metadata,
                    ),
                )
            except Exception as e:
                await self._rollback(
                    fw, ctx=ctx, failed_index=idx, committed_indices=committed_indices
                )
                async with self._lock:
                    self._open.pop(fw.id, None)
                shutil.rmtree(fw.staging_path, ignore_errors=True)
                raise WorkspaceError(
                    f"FanoutContextStore partial commit: backing index {idx} "
                    f"({type(backing).__name__}) failed after "
                    f"{len(committed_indices)} successful commits: {e}"
                ) from e
            committed_indices.append(idx)
            per_store.append(
                {
                    "backing": type(backing).__name__,
                    "added": list(diff.added),
                    "modified": list(diff.modified),
                    "deleted": list(diff.deleted),
                    "metadata": dict(diff.metadata),
                }
            )
            added.update(diff.added)
            modified.update(diff.modified)
            deleted.update(diff.deleted)

        async with self._lock:
            self._open.pop(fw.id, None)
        shutil.rmtree(fw.staging_path, ignore_errors=True)
        return Diff(
            added=sorted(added),
            modified=sorted(modified),
            deleted=sorted(deleted),
            metadata={"per_store": per_store},
        )

    async def discard_workspace(
        self, ws: Workspace, *, ctx: DiscardWorkspaceContext
    ) -> None:
        TenantScope.assert_matches(ctx.tenant_id)
        async with self._lock:
            fw = self._open.pop(ws.id, None)
        if fw is None:
            return
        for backing, inner_ws in zip(self._backings, fw.inner, strict=True):
            try:
                await backing.discard_workspace(inner_ws, ctx=ctx)
            except Exception:
                pass
        shutil.rmtree(fw.staging_path, ignore_errors=True)

    @staticmethod
    def _mirror(source: Path, target: Path) -> None:
        """Copy ``source`` over ``target``, deleting any extras under target."""
        target.mkdir(parents=True, exist_ok=True)
        desired: set[str] = set()
        for path in source.rglob("*"):
            if path.is_file():
                rel = path.relative_to(source).as_posix()
                desired.add(rel)
                dest = target / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(path.read_bytes())
        for path in list(target.rglob("*")):
            if path.is_file():
                rel = path.relative_to(target).as_posix()
                if rel not in desired:
                    path.unlink()
        for path in sorted(target.rglob("*"), reverse=True):
            if path.is_dir():
                try:
                    path.rmdir()
                except OSError:
                    pass

    async def _rollback(
        self,
        fw: FanoutWorkspace,
        *,
        ctx: CommitWorkspaceContext,
        failed_index: int,
        committed_indices: list[int],
    ) -> None:
        """Best-effort rollback for the current fanout commit.

        The not-yet-attempted inner workspaces are simply discarded. Already-
        committed backings are signalled via ``discard_workspace``; default
        impls treat post-commit discard as a no-op, which means non-undoable
        backings naturally surface as a partial commit.
        """
        for idx, (backing, inner_ws) in enumerate(
            zip(self._backings, fw.inner, strict=True)
        ):
            if idx == failed_index:
                try:
                    await backing.discard_workspace(
                        inner_ws,
                        ctx=DiscardWorkspaceContext(
                            request_id=ctx.request_id,
                            tenant_id=ctx.tenant_id,
                            metadata=ctx.metadata,
                        ),
                    )
                except Exception:
                    pass
            elif idx > failed_index:
                try:
                    await backing.discard_workspace(
                        inner_ws,
                        ctx=DiscardWorkspaceContext(
                            request_id=ctx.request_id,
                            tenant_id=ctx.tenant_id,
                            metadata=ctx.metadata,
                        ),
                    )
                except Exception:
                    pass
            else:
                try:
                    await backing.discard_workspace(
                        inner_ws,
                        ctx=DiscardWorkspaceContext(
                            request_id=ctx.request_id,
                            tenant_id=ctx.tenant_id,
                            metadata=ctx.metadata,
                        ),
                    )
                except Exception:
                    pass


__all__ = [
    "FanoutContextStore",
    "FanoutWorkspace",
]
