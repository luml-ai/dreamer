"""Default markdown-backed LTM store.

Layout under ``MarkdownLTMStore.root``:

    memory/
      INDEX.md                    # auto-maintained TOC of topics + recent incidents
      topics/<slug>.md            # long-lived topic pages
      incidents/<YYYY-MM>/<YYYY-MM-DD>-<slug>.md  # point-in-time observations
      _meta/
        schema.md                 # human-readable description of the structure
        context_pending.json      # watermark file (this store's ContextPendingStore impl)

``commit_workspace`` mirrors the workspace's filesystem state back into the
canonical root, regenerates ``INDEX.md`` from the post-commit frontmatter, and
returns a path-level :class:`Diff`. The dream agent never edits ``INDEX.md``
manually — the store owns it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from collections import defaultdict
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal
from uuid import uuid4

import yaml

from dreamer.api.compat import implements
from dreamer.api.contexts import (
    ClearContextPendingContext,
    CommitWorkspaceContext,
    DiscardWorkspaceContext,
    GetContextPendingContext,
    OpenWorkspaceContext,
    SetContextPendingContext,
)
from dreamer.api.errors import ConfigError, WorkspaceError
from dreamer.api.stores import ContextPendingStore, LTMStore
from dreamer.api.tenants import TenantScope
from dreamer.api.types import Diff, FileViewable, TenantId, Workspace

logger = logging.getLogger(__name__)

INDEX_FILENAME = "INDEX.md"
SCHEMA_FILENAME = "_meta/schema.md"
WATERMARK_FILENAME = "_meta/context_pending.json"


DEFAULT_SCHEMA_TEXT = """\
# LTM schema

This directory is the long-term memory store for `dreamer`. The dream agent
reads this file every run to recall the rules of the layout.

## Files

- `INDEX.md` — auto-generated table of contents (do not edit manually).
- `topics/<slug>.md` — evergreen synthesis pages.
- `incidents/<YYYY-MM>/<YYYY-MM-DD>-<slug>.md` — point-in-time observations.
- `archive/<original relative path>` — retired entries (see "Archive" below).
- `archive/LOG.md` — operations log, one line per reinforce/archive/discard
  decision.
- `_meta/` — store-owned metadata. Do not edit.

## Frontmatter

Every topic and incident file MUST start with YAML frontmatter:

    ---
    title: <human title>
    slug: <kebab-case>
    type: topic | incident
    tags: [tag1, tag2]
    created_at: 2026-01-01T00:00:00Z
    updated_at: 2026-01-01T00:00:00Z
    related: [topic-slug-1]
    sources: [stm:<memory-id>]
    ---

Optional reinforcement fields, maintained by the dream agent from agent
feedback:

    confirmations: <int>            # count of distinct confirmation events
    last_confirmed: <timestamp>     # most recent confirmation or re-validation
    importance: pinned | normal | ephemeral   # absent means normal

`pinned` entries must never be removed, archived, or downgraded by a dream.
`ephemeral` entries are expected to be short-lived and may be hard-deleted
once past their usefulness.

The store regenerates `INDEX.md` deterministically by reading frontmatter, so
keep the frontmatter accurate.

## Archive

Retiring an entry means MOVING it to `archive/<original relative path>`
(e.g. `topics/x.md` → `archive/topics/x.md`), never silently deleting it.
Add retirement frontmatter to the archived file:

    retired_at: <timestamp>
    retired_reason: <why this entry was retired>
    superseded_by: <slug of the replacing entry, if any>

Only `topics/` and `incidents/` are indexed — archived entries drop out of
`INDEX.md` automatically. Every dream that reinforces, archives, or discards
appends one line per decision to `archive/LOG.md`: what changed, why, and —
for discarded unanchored flags — the stated reason.
"""


@dataclass
class MarkdownWorkspace:
    id: str
    tenant_id: TenantId
    path: Path
    metadata: Mapping[str, Any] = field(default_factory=dict)

    async def file_view(self) -> Path:
        return self.path


def _is_guarded(rel: str) -> bool:
    """True for files the commit guards watch: `topics/` and `incidents/`."""
    return rel.startswith(("topics/", "incidents/"))


def _kebab(value: str) -> str:
    out = []
    last_dash = True
    for ch in value.strip().lower():
        if ch.isalnum():
            out.append(ch)
            last_dash = False
        elif not last_dash:
            out.append("-")
            last_dash = True
    while out and out[-1] == "-":
        out.pop()
    return "".join(out) or "untitled"


def _parse_frontmatter(text: str) -> dict[str, Any] | None:
    """Parse YAML frontmatter from markdown text, or None.

    Text without a leading ``---`` block, with malformed YAML, or with a
    non-mapping document is treated as having no frontmatter.
    """
    if not text.startswith("---"):
        return None
    rest = text[3:]
    end = rest.find("\n---")
    if end < 0:
        return None
    block = rest[:end]
    try:
        loaded = yaml.safe_load(block)
    except yaml.YAMLError:
        return None
    if not isinstance(loaded, dict):
        return None
    return loaded


def _read_frontmatter(path: Path) -> dict[str, Any] | None:
    """Return parsed YAML frontmatter from a markdown file, or None."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    return _parse_frontmatter(text)


def _walk_markdown(root: Path, subdir: str) -> list[tuple[Path, dict[str, Any]]]:
    base = root / subdir
    if not base.exists():
        return []
    out: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(base.rglob("*.md")):
        if not path.is_file():
            continue
        fm = _read_frontmatter(path)
        if fm is None:
            continue
        out.append((path, fm))
    return out


def _format_index(
    *, topics: list[tuple[Path, dict[str, Any]]], incidents: list[tuple[Path, dict[str, Any]]],
    root: Path,
) -> str:
    """Build the deterministic INDEX.md body.

    Topics are grouped by tag (first-tag wins; entries without tags fall under
    "untagged"). Incidents are grouped by their ``YYYY-MM`` parent directory.
    """
    lines: list[str] = ["# LTM Index", ""]

    by_tag: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for path, fm in topics:
        tags = fm.get("tags") or []
        primary = str(tags[0]) if tags else "untagged"
        title = str(fm.get("title") or path.stem)
        rel = path.relative_to(root).as_posix()
        by_tag[primary].append((title, rel))

    if by_tag:
        lines.append("## Topics")
        lines.append("")
        for tag in sorted(by_tag):
            lines.append(f"### {tag}")
            lines.append("")
            for title, rel in sorted(by_tag[tag]):
                lines.append(f"- [{title}]({rel})")
            lines.append("")

    by_month: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for path, fm in incidents:
        rel = path.relative_to(root).as_posix()
        parts = rel.split("/")
        month = parts[1] if len(parts) >= 3 and parts[0] == "incidents" else "unknown"
        title = str(fm.get("title") or path.stem)
        date = str(fm.get("created_at") or path.stem)
        by_month[month].append((date, title, rel))

    if by_month:
        lines.append("## Incidents")
        lines.append("")
        for month in sorted(by_month, reverse=True):
            lines.append(f"### {month}")
            lines.append("")
            for _, title, rel in sorted(by_month[month], reverse=True):
                lines.append(f"- [{title}]({rel})")
            lines.append("")

    if not by_tag and not by_month:
        lines.append("_(empty)_")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _scan_tree(root: Path) -> dict[str, bytes]:
    """Snapshot the canonical tree as ``relpath → content``.

    Only regular files are recorded. ``INDEX.md`` is included so commit-time
    diffs surface it; the watermark file lives under ``_meta`` and is also
    captured here.
    """
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
    """Mirror ``tree`` into ``target`` exactly.

    Deletes any file or empty directory not present in ``tree``. Used by
    ``commit_workspace`` to atomically replace the canonical root with the
    workspace contents.
    """
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


@implements(LTMStore, version=1)
@implements(ContextPendingStore, version=1)
class MarkdownLTMStore:
    """Default file-backed LTM store with ``ContextPendingStore@1`` watermark.

    ``root`` is the directory whose layout is described in :data:`DEFAULT_SCHEMA_TEXT`.
    Single-tenant by default; the workspace and watermark are scoped to the
    active :class:`TenantScope` regardless.
    """

    multi_tenant: ClassVar[bool] = False
    workspace_capabilities: ClassVar[frozenset[type]] = frozenset({FileViewable})

    def __init__(
        self,
        *,
        root: str | Path,
        regenerate_index: bool = True,
        schema_text: str | None = None,
        max_autonomous_removals: int | None = None,
        enforce_pinned: bool = True,
        on_guard_violation: Literal["fail", "warn"] = "fail",
    ) -> None:
        if on_guard_violation not in ("fail", "warn"):
            raise ConfigError(
                "MarkdownLTMStore: on_guard_violation must be 'fail' or 'warn', "
                f"got {on_guard_violation!r}"
            )
        if max_autonomous_removals is not None and max_autonomous_removals < 0:
            raise ConfigError(
                "MarkdownLTMStore: max_autonomous_removals must be >= 0 or null"
            )
        self.root = Path(root)
        self.regenerate_index = regenerate_index
        self.schema_text = schema_text or DEFAULT_SCHEMA_TEXT
        self.max_autonomous_removals = max_autonomous_removals
        self.enforce_pinned = enforce_pinned
        self.on_guard_violation = on_guard_violation
        self._lock = asyncio.Lock()
        self._open_workspaces: dict[str, MarkdownWorkspace] = {}
        self._ensure_root()

    def _ensure_root(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "topics").mkdir(parents=True, exist_ok=True)
        (self.root / "incidents").mkdir(parents=True, exist_ok=True)
        meta = self.root / "_meta"
        meta.mkdir(parents=True, exist_ok=True)
        schema_path = self.root / SCHEMA_FILENAME
        if not schema_path.exists():
            schema_path.write_text(self.schema_text, encoding="utf-8")
        index_path = self.root / INDEX_FILENAME
        if not index_path.exists():
            index_path.write_text("# LTM Index\n\n_(empty)_\n", encoding="utf-8")

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
            # Mirror the canonical layout's empty directories so the agent can
            # write into `topics/` and `incidents/` without first creating them.
            (ws_path / "topics").mkdir(parents=True, exist_ok=True)
            (ws_path / "incidents").mkdir(parents=True, exist_ok=True)
            (ws_path / "_meta").mkdir(parents=True, exist_ok=True)
            ws = MarkdownWorkspace(
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

            if self.regenerate_index:
                # Walk frontmatter from the workspace itself so the regenerated
                # index reflects the post-commit state.
                topics = _walk_markdown(internal.path, "topics")
                incidents = _walk_markdown(internal.path, "incidents")
                index_text = _format_index(
                    topics=topics, incidents=incidents, root=internal.path
                )
                new_tree[INDEX_FILENAME] = index_text.encode("utf-8")

            old_tree = _scan_tree(self.root)
            diff = _diff_trees(old_tree, new_tree)

            violations = self._guard_violations(old_tree, new_tree)
            if violations:
                summary = "; ".join(violations)
                if self.on_guard_violation == "fail":
                    # Raise before any mutation: the canonical tree stays
                    # untouched and the orchestrator's failure path takes over.
                    raise WorkspaceError(f"LTM commit guard violation(s): {summary}")
                logger.warning("LTM commit guard violation(s) (warn mode): %s", summary)

            _replace_tree(self.root, new_tree)
            self._open_workspaces.pop(ws.id, None)
            shutil.rmtree(internal.path, ignore_errors=True)
            # Only re-create the directory shell (without seeding default
            # files) so an explicit deletion of INDEX.md / schema.md sticks
            # until the next ``open_workspace`` call.
            (self.root / "topics").mkdir(parents=True, exist_ok=True)
            (self.root / "incidents").mkdir(parents=True, exist_ok=True)
            (self.root / "_meta").mkdir(parents=True, exist_ok=True)
            return diff

    def _guard_violations(
        self, old_tree: Mapping[str, bytes], new_tree: Mapping[str, bytes]
    ) -> list[str]:
        """Deterministic commit-time safety rails.

        A *removal* is a ``topics/`` or ``incidents/`` file present in the
        canonical tree but absent from the committed tree without a
        counterpart at ``archive/<original relative path>`` — archival moves
        are not removals. ``pinned`` entries may be edited in place but never
        removed, archived, or downgraded.
        """
        violations: list[str] = []

        removed = [rel for rel in old_tree if _is_guarded(rel) and rel not in new_tree]
        true_removals = sorted(rel for rel in removed if f"archive/{rel}" not in new_tree)
        if (
            self.max_autonomous_removals is not None
            and len(true_removals) > self.max_autonomous_removals
        ):
            violations.append(
                f"removal budget exceeded: {len(true_removals)} removal(s) > "
                f"max_autonomous_removals={self.max_autonomous_removals}; "
                f"offending paths: {true_removals}"
            )

        if self.enforce_pinned:
            for rel in sorted(old_tree):
                if not _is_guarded(rel) or not rel.endswith(".md"):
                    continue
                fm = _parse_frontmatter(old_tree[rel].decode("utf-8", errors="replace"))
                if fm is None or fm.get("importance") != "pinned":
                    continue
                if rel not in new_tree:
                    action = "archived" if f"archive/{rel}" in new_tree else "removed"
                    violations.append(f"pinned entry {rel!r} may not be {action}")
                    continue
                new_fm = _parse_frontmatter(
                    new_tree[rel].decode("utf-8", errors="replace")
                )
                if new_fm is None or new_fm.get("importance") != "pinned":
                    violations.append(
                        f"pinned entry {rel!r} may not have its importance downgraded"
                    )
        return violations

    async def discard_workspace(
        self, ws: Workspace, *, ctx: DiscardWorkspaceContext
    ) -> None:
        TenantScope.assert_matches(ctx.tenant_id)
        async with self._lock:
            internal = self._open_workspaces.pop(ws.id, None)
            if internal is None:
                return
            shutil.rmtree(internal.path, ignore_errors=True)

    def _watermark_path(self) -> Path:
        return self.root / WATERMARK_FILENAME

    async def set_context_pending(
        self, diff: Diff, *, ctx: SetContextPendingContext
    ) -> None:
        TenantScope.assert_matches(ctx.tenant_id)
        path = self._watermark_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "tenant_id": ctx.tenant_id,
            "diff": diff.model_dump(mode="json"),
        }
        path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")

    async def get_context_pending(
        self, *, ctx: GetContextPendingContext
    ) -> Diff | None:
        TenantScope.assert_matches(ctx.tenant_id)
        path = self._watermark_path()
        if not path.exists():
            return None
        raw = path.read_text(encoding="utf-8")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        diff_payload = data.get("diff")
        if not isinstance(diff_payload, dict):
            return None
        return Diff.model_validate(diff_payload)

    async def clear_context_pending(
        self, *, ctx: ClearContextPendingContext
    ) -> None:
        TenantScope.assert_matches(ctx.tenant_id)
        path = self._watermark_path()
        if path.exists():
            path.unlink()


__all__ = [
    "DEFAULT_SCHEMA_TEXT",
    "INDEX_FILENAME",
    "MarkdownLTMStore",
    "MarkdownWorkspace",
    "WATERMARK_FILENAME",
    "_format_index",
    "_kebab",
    "_read_frontmatter",
]
