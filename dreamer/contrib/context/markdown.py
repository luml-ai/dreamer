"""Default markdown-backed Context store with optional ``ContextReader@1``.

Layout under ``MarkdownContextStore.root``:

    context/
      AGENTS.md
      skills/<skill-name>/
        SKILL.md
        [resources/, ...]
      _meta/
        schema.md
"""

from __future__ import annotations

import asyncio
import shutil
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar
from uuid import uuid4

from dreamer.api.compat import implements
from dreamer.api.contexts import (
    CommitWorkspaceContext,
    ContextReadContext,
    DiscardWorkspaceContext,
    OpenWorkspaceContext,
)
from dreamer.api.stores import ContextReader, ContextStore
from dreamer.api.tenants import TenantScope
from dreamer.api.types import Diff, FileViewable, TenantId, Workspace

SCHEMA_FILENAME = "_meta/schema.md"


DEFAULT_SCHEMA_TEXT = """\
# Context schema

This directory is the agent-facing context store. Files here are the only
artifact other agents read. Keep `AGENTS.md` short (under 200 lines); promote
detail into skills.

## Files

- `AGENTS.md` — entry point.
- `skills/<skill-name>/SKILL.md` — one skill per directory; frontmatter mirrors
  Claude Code's skills format (`name`, `description`, `version`).
- `skills/<skill-name>/resources/` — optional ancillary files.
- `_meta/` — store-owned metadata. Do not edit.

Bump the skill `version` in frontmatter when materially changing a skill so
agents know to re-read it.
"""


@dataclass
class ContextWorkspace:
    id: str
    tenant_id: TenantId
    path: Path
    metadata: Mapping[str, Any] = field(default_factory=dict)

    async def file_view(self) -> Path:
        return self.path


def _scan_tree(root: Path) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    if not root.exists():
        return out
    for path in root.rglob("*"):
        if path.is_file():
            rel = path.relative_to(root).as_posix()
            out[rel] = path.read_bytes()
    return out


def _materialize_tree(target: Path, tree: Mapping[str, bytes]) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for rel, content in tree.items():
        dest = target / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)


def _replace_tree(target: Path, tree: Mapping[str, bytes]) -> None:
    target.mkdir(parents=True, exist_ok=True)
    desired = set(tree.keys())
    existing = {
        path.relative_to(target).as_posix(): path
        for path in target.rglob("*")
        if path.is_file()
    }
    for rel, path in existing.items():
        if rel not in desired:
            path.unlink()
    for rel, content in tree.items():
        dest = target / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
    for path in sorted(target.rglob("*"), reverse=True):
        if path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass


def _diff_trees(old: Mapping[str, bytes], new: Mapping[str, bytes]) -> Diff:
    added = sorted(set(new) - set(old))
    deleted = sorted(set(old) - set(new))
    modified = sorted(k for k in set(new) & set(old) if new[k] != old[k])
    return Diff(added=added, modified=modified, deleted=deleted)


def _safe_join(root: Path, relative: str) -> Path:
    """Resolve ``relative`` under ``root`` and reject path traversal."""
    candidate = (root / relative).resolve(strict=False)
    root_resolved = root.resolve(strict=False)
    try:
        candidate.relative_to(root_resolved)
    except ValueError as e:
        raise FileNotFoundError(relative) from e
    return candidate


@implements(ContextStore, version=1)
@implements(ContextReader, version=1)
class MarkdownContextStore:
    multi_tenant: ClassVar[bool] = False
    workspace_capabilities: ClassVar[frozenset[type]] = frozenset({FileViewable})

    def __init__(
        self,
        *,
        root: str | Path,
        schema_text: str | None = None,
    ) -> None:
        self.root = Path(root)
        self.schema_text = schema_text or DEFAULT_SCHEMA_TEXT
        self._lock = asyncio.Lock()
        self._open_workspaces: dict[str, ContextWorkspace] = {}
        self._ensure_root()

    def _ensure_root(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "skills").mkdir(parents=True, exist_ok=True)
        meta = self.root / "_meta"
        meta.mkdir(parents=True, exist_ok=True)
        schema_path = self.root / SCHEMA_FILENAME
        if not schema_path.exists():
            schema_path.write_text(self.schema_text, encoding="utf-8")
        # AGENTS.md is owned by the dream agent — leaving it unseeded means the
        # first commit's diff surfaces it as ``added``.

    @asynccontextmanager
    async def _scoped(self, tenant_id: TenantId) -> AsyncIterator[None]:
        TenantScope.assert_matches(tenant_id)
        yield

    async def open_workspace(self, *, ctx: OpenWorkspaceContext) -> Workspace:
        TenantScope.assert_matches(ctx.tenant_id)
        async with self._lock:
            self._ensure_root()
            ws_id = str(uuid4())
            ws_path = self.root.parent / f".{self.root.name}-ws-{ws_id}"
            tree = _scan_tree(self.root)
            _materialize_tree(ws_path, tree)
            (ws_path / "skills").mkdir(parents=True, exist_ok=True)
            (ws_path / "_meta").mkdir(parents=True, exist_ok=True)
            ws = ContextWorkspace(
                id=ws_id,
                tenant_id=ctx.tenant_id,
                path=ws_path,
                metadata={"root": str(self.root)},
            )
            self._open_workspaces[ws_id] = ws
            return ws

    async def commit_workspace(
        self, ws: Workspace, *, ctx: CommitWorkspaceContext
    ) -> Diff:
        TenantScope.assert_matches(ctx.tenant_id)
        async with self._lock:
            internal = self._open_workspaces.get(ws.id)
            if internal is None:
                raise RuntimeError(f"unknown workspace id: {ws.id}")
            new_tree = _scan_tree(internal.path)
            old_tree = _scan_tree(self.root)
            diff = _diff_trees(old_tree, new_tree)
            _replace_tree(self.root, new_tree)
            self._open_workspaces.pop(ws.id, None)
            shutil.rmtree(internal.path, ignore_errors=True)
            (self.root / "skills").mkdir(parents=True, exist_ok=True)
            (self.root / "_meta").mkdir(parents=True, exist_ok=True)
            return diff

    async def discard_workspace(
        self, ws: Workspace, *, ctx: DiscardWorkspaceContext
    ) -> None:
        TenantScope.assert_matches(ctx.tenant_id)
        async with self._lock:
            internal = self._open_workspaces.pop(ws.id, None)
            if internal is None:
                return
            shutil.rmtree(internal.path, ignore_errors=True)

    async def read(self, path: str, *, ctx: ContextReadContext) -> bytes:
        TenantScope.assert_matches(ctx.tenant_id)
        target = _safe_join(self.root, path)
        if not target.is_file():
            raise FileNotFoundError(path)
        return target.read_bytes()

    async def list(
        self, *, prefix: str = "", ctx: ContextReadContext
    ) -> list[str]:
        TenantScope.assert_matches(ctx.tenant_id)
        if prefix:
            base = _safe_join(self.root, prefix)
        else:
            base = self.root
        if not base.exists():
            return []
        out: list[str] = []
        for path in sorted(base.rglob("*")):
            if path.is_file():
                rel = path.relative_to(self.root).as_posix()
                out.append(rel)
        return out


__all__ = [
    "DEFAULT_SCHEMA_TEXT",
    "ContextWorkspace",
    "MarkdownContextStore",
]
