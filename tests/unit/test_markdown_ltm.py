from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

from dreamer.api.contexts import (
    ClearContextPendingContext,
    CommitWorkspaceContext,
    DiscardWorkspaceContext,
    GetContextPendingContext,
    OpenWorkspaceContext,
    SetContextPendingContext,
)
from dreamer.api.errors import ConfigError, WorkspaceError
from dreamer.api.stores import ContextPendingStore
from dreamer.api.tenants import TenantScope
from dreamer.api.types import Diff, FileViewable
from dreamer.contrib.ltm.markdown import (
    INDEX_FILENAME,
    SCHEMA_FILENAME,
    WATERMARK_FILENAME,
    MarkdownLTMStore,
)


@pytest_asyncio.fixture(autouse=True)
async def _clear_tenant_scope() -> AsyncIterator[None]:
    TenantScope.clear()
    yield
    TenantScope.clear()


@pytest.fixture()
def store(tmp_path: Path) -> MarkdownLTMStore:
    return MarkdownLTMStore(root=tmp_path / "memory")


@pytest.fixture()
def store_no_index(tmp_path: Path) -> MarkdownLTMStore:
    return MarkdownLTMStore(root=tmp_path / "memory", regenerate_index=False)


def _topic(
    slug: str,
    *,
    title: str,
    tags: list[str],
    body: str = "body",
    extra_fm: str = "",
) -> str:
    return (
        f"---\n"
        f"title: {title}\n"
        f"slug: {slug}\n"
        f"type: topic\n"
        f"tags: {tags!r}\n"
        f"created_at: 2026-05-02T00:00:00Z\n"
        f"updated_at: 2026-05-02T00:00:00Z\n"
        f"{extra_fm}"
        f"---\n\n# {title}\n\n{body}\n"
    )


def _incident(
    *, slug: str, title: str, created_at: str, body: str = "body"
) -> str:
    return (
        f"---\n"
        f"title: {title}\n"
        f"slug: {slug}\n"
        f"type: incident\n"
        f"tags: []\n"
        f"created_at: {created_at}\n"
        f"updated_at: {created_at}\n"
        f"---\n\n# {title}\n\n{body}\n"
    )


def test_root_scaffolded_on_init(store: MarkdownLTMStore) -> None:
    assert (store.root / "topics").is_dir()
    assert (store.root / "incidents").is_dir()
    assert (store.root / "_meta").is_dir()
    assert (store.root / SCHEMA_FILENAME).is_file()
    assert (store.root / INDEX_FILENAME).is_file()


def test_implements_protocols(store: MarkdownLTMStore) -> None:
    assert isinstance(store, ContextPendingStore)
    assert FileViewable in store.workspace_capabilities


@pytest.mark.asyncio
async def test_index_regenerated_with_topic_and_incident(
    store: MarkdownLTMStore,
) -> None:
    with TenantScope.set("default"):
        ws = await store.open_workspace(
            ctx=OpenWorkspaceContext(request_id="r1", tenant_id="default")
        )
        view = await ws.file_view()  # type: ignore[attr-defined]
        topic = view / "topics" / "auth-rotation.md"
        topic.parent.mkdir(parents=True, exist_ok=True)
        topic.write_text(
            _topic(
                "auth-rotation", title="Auth rotation", tags=["security", "auth"]
            ),
            encoding="utf-8",
        )
        incident_dir = view / "incidents" / "2026-05"
        incident_dir.mkdir(parents=True, exist_ok=True)
        (incident_dir / "2026-05-02-flaky-build.md").write_text(
            _incident(
                slug="flaky-build",
                title="Flaky build",
                created_at="2026-05-02T10:00:00Z",
            ),
            encoding="utf-8",
        )
        diff = await store.commit_workspace(
            ws, ctx=CommitWorkspaceContext(request_id="r1", tenant_id="default")
        )
        assert "topics/auth-rotation.md" in diff.added
        assert "incidents/2026-05/2026-05-02-flaky-build.md" in diff.added
        assert INDEX_FILENAME in diff.modified or INDEX_FILENAME in diff.added

        index_text = (store.root / INDEX_FILENAME).read_text(encoding="utf-8")
        assert "## Topics" in index_text
        assert "Auth rotation" in index_text
        assert "## Incidents" in index_text
        assert "Flaky build" in index_text
        assert "### 2026-05" in index_text
        assert "### security" in index_text


@pytest.mark.asyncio
async def test_index_regeneration_is_deterministic(store: MarkdownLTMStore) -> None:
    with TenantScope.set("default"):
        for _run in range(2):
            ws = await store.open_workspace(
                ctx=OpenWorkspaceContext(request_id="r1", tenant_id="default")
            )
            view = await ws.file_view()  # type: ignore[attr-defined]
            (view / "topics" / "alpha.md").write_text(
                _topic("alpha", title="Alpha", tags=["x"]), encoding="utf-8"
            )
            (view / "topics" / "beta.md").write_text(
                _topic("beta", title="Beta", tags=["x"]), encoding="utf-8"
            )
            await store.commit_workspace(
                ws,
                ctx=CommitWorkspaceContext(request_id="r1", tenant_id="default"),
            )
        first = (store.root / INDEX_FILENAME).read_text(encoding="utf-8")

        ws = await store.open_workspace(
            ctx=OpenWorkspaceContext(request_id="r1", tenant_id="default")
        )
        diff = await store.commit_workspace(
            ws, ctx=CommitWorkspaceContext(request_id="r1", tenant_id="default")
        )
        second = (store.root / INDEX_FILENAME).read_text(encoding="utf-8")
        assert first == second
        assert INDEX_FILENAME not in diff.modified


@pytest.mark.asyncio
async def test_index_skipped_when_disabled(store_no_index: MarkdownLTMStore) -> None:
    with TenantScope.set("default"):
        ws = await store_no_index.open_workspace(
            ctx=OpenWorkspaceContext(request_id="r1", tenant_id="default")
        )
        view = await ws.file_view()  # type: ignore[attr-defined]
        (view / "topics" / "x.md").write_text(
            _topic("x", title="X", tags=[]), encoding="utf-8"
        )
        (view / INDEX_FILENAME).unlink()
        diff = await store_no_index.commit_workspace(
            ws, ctx=CommitWorkspaceContext(request_id="r1", tenant_id="default")
        )
        assert INDEX_FILENAME in diff.deleted
        assert not (store_no_index.root / INDEX_FILENAME).exists()


@pytest.mark.asyncio
async def test_watermark_round_trip(store: MarkdownLTMStore) -> None:
    sample = Diff(added=["a.md"], modified=["b.md"], metadata={"phase": "test"})
    with TenantScope.set("default"):
        assert (
            await store.get_context_pending(
                ctx=GetContextPendingContext(request_id="r1", tenant_id="default")
            )
            is None
        )
        await store.set_context_pending(
            sample,
            ctx=SetContextPendingContext(request_id="r1", tenant_id="default"),
        )
        wm_path = store.root / WATERMARK_FILENAME
        assert wm_path.is_file()
        payload = json.loads(wm_path.read_text(encoding="utf-8"))
        assert payload["tenant_id"] == "default"
        assert payload["diff"]["added"] == ["a.md"]
        loaded = await store.get_context_pending(
            ctx=GetContextPendingContext(request_id="r1", tenant_id="default")
        )
        assert loaded is not None
        assert loaded.added == sample.added
        assert loaded.modified == sample.modified
        assert loaded.metadata == {"phase": "test"}
        await store.clear_context_pending(
            ctx=ClearContextPendingContext(request_id="r1", tenant_id="default")
        )
        assert not wm_path.exists()


@pytest.mark.asyncio
async def test_discard_workspace_cleans_up_dir(store: MarkdownLTMStore) -> None:
    with TenantScope.set("default"):
        ws = await store.open_workspace(
            ctx=OpenWorkspaceContext(request_id="r1", tenant_id="default")
        )
        view = await ws.file_view()  # type: ignore[attr-defined]
        assert view.exists()
        await store.discard_workspace(
            ws, ctx=DiscardWorkspaceContext(request_id="r1", tenant_id="default")
        )
        assert not view.exists()


@pytest.mark.asyncio
async def test_file_without_frontmatter_skipped(store: MarkdownLTMStore) -> None:
    with TenantScope.set("default"):
        ws = await store.open_workspace(
            ctx=OpenWorkspaceContext(request_id="r1", tenant_id="default")
        )
        view = await ws.file_view()  # type: ignore[attr-defined]
        (view / "topics" / "no-frontmatter.md").write_text(
            "# Just a heading\n", encoding="utf-8"
        )
        await store.commit_workspace(
            ws, ctx=CommitWorkspaceContext(request_id="r1", tenant_id="default")
        )
        index_text = (store.root / INDEX_FILENAME).read_text(encoding="utf-8")
        assert "no-frontmatter" not in index_text


@pytest.mark.asyncio
async def test_archived_entries_excluded_from_index(store: MarkdownLTMStore) -> None:
    with TenantScope.set("default"):
        ws = await store.open_workspace(
            ctx=OpenWorkspaceContext(request_id="r1", tenant_id="default")
        )
        view = await ws.file_view()  # type: ignore[attr-defined]
        (view / "topics" / "alive.md").write_text(
            _topic("alive", title="Alive topic", tags=["x"]), encoding="utf-8"
        )
        archived = view / "archive" / "topics" / "retired.md"
        archived.parent.mkdir(parents=True, exist_ok=True)
        archived.write_text(
            _topic("retired", title="Retired topic", tags=["x"]).replace(
                "---\n\n#",
                "retired_at: 2026-07-01T00:00:00Z\n"
                "retired_reason: superseded\n"
                "superseded_by: alive\n"
                "---\n\n#",
            ),
            encoding="utf-8",
        )
        (view / "archive" / "LOG.md").write_text(
            "- 2026-07-01: archived topics/retired.md (superseded by alive)\n",
            encoding="utf-8",
        )
        diff = await store.commit_workspace(
            ws, ctx=CommitWorkspaceContext(request_id="r1", tenant_id="default")
        )
        assert "archive/topics/retired.md" in diff.added
        assert "archive/LOG.md" in diff.added
        assert (store.root / "archive" / "topics" / "retired.md").is_file()

        index_text = (store.root / INDEX_FILENAME).read_text(encoding="utf-8")
        assert "Alive topic" in index_text
        assert "Retired topic" not in index_text
        assert "archive/" not in index_text


def test_schema_text_documents_reinforcement_and_archive() -> None:
    from dreamer.contrib.ltm.markdown import DEFAULT_SCHEMA_TEXT

    for needle in (
        "confirmations:",
        "last_confirmed:",
        "importance: pinned | normal | ephemeral",
        "archive/<original relative path>",
        "archive/LOG.md",
        "retired_at:",
        "retired_reason:",
        "superseded_by:",
    ):
        assert needle in DEFAULT_SCHEMA_TEXT, f"schema text missing: {needle!r}"


async def _commit_files(
    store: MarkdownLTMStore, files: dict[str, str], *, delete: list[str] | None = None
) -> None:
    with TenantScope.set("default"):
        ws = await store.open_workspace(
            ctx=OpenWorkspaceContext(request_id="r", tenant_id="default")
        )
        view = await ws.file_view()  # type: ignore[attr-defined]
        for rel, text in files.items():
            dest = view / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(text, encoding="utf-8")
        for rel in delete or []:
            (view / rel).unlink()
        await store.commit_workspace(
            ws, ctx=CommitWorkspaceContext(request_id="r", tenant_id="default")
        )


def _five_topics() -> dict[str, str]:
    return {
        f"topics/t{i}.md": _topic(f"t{i}", title=f"T{i}", tags=["x"]) for i in range(5)
    }


@pytest.mark.asyncio
async def test_removal_budget_enforced(tmp_path: Path) -> None:
    store = MarkdownLTMStore(root=tmp_path / "memory", max_autonomous_removals=2)
    await _commit_files(store, _five_topics())

    with TenantScope.set("default"):
        ws = await store.open_workspace(
            ctx=OpenWorkspaceContext(request_id="r", tenant_id="default")
        )
        view = await ws.file_view()  # type: ignore[attr-defined]
        for i in range(3):
            (view / "topics" / f"t{i}.md").unlink()
        with pytest.raises(WorkspaceError) as exc:
            await store.commit_workspace(
                ws, ctx=CommitWorkspaceContext(request_id="r", tenant_id="default")
            )
    message = str(exc.value)
    assert "max_autonomous_removals=2" in message
    for i in range(3):
        assert f"topics/t{i}.md" in message
        assert (store.root / "topics" / f"t{i}.md").is_file()


@pytest.mark.asyncio
async def test_archival_does_not_count_against_budget(tmp_path: Path) -> None:
    store = MarkdownLTMStore(root=tmp_path / "memory", max_autonomous_removals=2)
    await _commit_files(store, _five_topics())

    with TenantScope.set("default"):
        ws = await store.open_workspace(
            ctx=OpenWorkspaceContext(request_id="r", tenant_id="default")
        )
        view = await ws.file_view()  # type: ignore[attr-defined]
        for i in range(3):
            src = view / "topics" / f"t{i}.md"
            dest = view / "archive" / "topics" / f"t{i}.md"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(
                src.read_text(encoding="utf-8").replace(
                    "---\n\n#",
                    "retired_at: 2026-07-11T00:00:00Z\n"
                    "retired_reason: stale\n"
                    "---\n\n#",
                ),
                encoding="utf-8",
            )
            src.unlink()
        await store.commit_workspace(
            ws, ctx=CommitWorkspaceContext(request_id="r", tenant_id="default")
        )
    index_text = (store.root / INDEX_FILENAME).read_text(encoding="utf-8")
    for i in range(3):
        assert (store.root / "archive" / "topics" / f"t{i}.md").is_file()
        assert f"topics/t{i}.md)" not in index_text


@pytest.mark.asyncio
async def test_pinned_entry_protected(tmp_path: Path) -> None:
    pinned = _topic("keep", title="Keep", tags=["x"], extra_fm="importance: pinned\n")

    async def fresh_store(idx: int) -> MarkdownLTMStore:
        store = MarkdownLTMStore(root=tmp_path / f"memory{idx}")
        await _commit_files(store, {"topics/keep.md": pinned})
        return store

    async def commit_mutation(
        store: MarkdownLTMStore, mutate: Any
    ) -> None:
        with TenantScope.set("default"):
            ws = await store.open_workspace(
                ctx=OpenWorkspaceContext(request_id="r", tenant_id="default")
            )
            view = await ws.file_view()  # type: ignore[attr-defined]
            mutate(view)
            await store.commit_workspace(
                ws, ctx=CommitWorkspaceContext(request_id="r", tenant_id="default")
            )

    def delete(view: Path) -> None:
        (view / "topics" / "keep.md").unlink()

    def archive(view: Path) -> None:
        dest = view / "archive" / "topics" / "keep.md"
        dest.parent.mkdir(parents=True, exist_ok=True)
        (view / "topics" / "keep.md").rename(dest)

    def downgrade(view: Path) -> None:
        path = view / "topics" / "keep.md"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "importance: pinned", "importance: normal"
            ),
            encoding="utf-8",
        )

    for idx, mutate in enumerate((delete, archive, downgrade)):
        store = await fresh_store(idx)
        with pytest.raises(WorkspaceError, match="topics/keep.md"):
            await commit_mutation(store, mutate)
        assert (store.root / "topics" / "keep.md").is_file()

    # In-place content edit that keeps `importance: pinned` commits fine.
    def edit(view: Path) -> None:
        path = view / "topics" / "keep.md"
        path.write_text(
            path.read_text(encoding="utf-8") + "\nnew insight\n", encoding="utf-8"
        )

    store = await fresh_store(99)
    await commit_mutation(store, edit)
    text = (store.root / "topics" / "keep.md").read_text(encoding="utf-8")
    assert "new insight" in text


@pytest.mark.asyncio
async def test_warn_mode_lets_violating_commit_through(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    store = MarkdownLTMStore(
        root=tmp_path / "memory",
        max_autonomous_removals=2,
        on_guard_violation="warn",
    )
    await _commit_files(store, _five_topics())

    with caplog.at_level("WARNING", logger="dreamer.contrib.ltm.markdown"):
        await _commit_files(
            store, {}, delete=[f"topics/t{i}.md" for i in range(3)]
        )
    for i in range(3):
        assert not (store.root / "topics" / f"t{i}.md").exists()
    assert any("guard violation" in r.message for r in caplog.records)
    assert any("max_autonomous_removals=2" in r.message for r in caplog.records)


def test_invalid_guard_params_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        MarkdownLTMStore(root=tmp_path / "m1", on_guard_violation="explode")  # type: ignore[arg-type]
    with pytest.raises(ConfigError):
        MarkdownLTMStore(root=tmp_path / "m2", max_autonomous_removals=-1)


@pytest.mark.asyncio
async def test_tenant_scope_mismatch_blocks_set_pending(
    store: MarkdownLTMStore,
) -> None:
    with TenantScope.set("a"):
        with pytest.raises(RuntimeError):
            await store.set_context_pending(
                Diff(),
                ctx=SetContextPendingContext(request_id="r1", tenant_id="b"),
            )
