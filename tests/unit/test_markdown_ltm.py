from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path

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


def _topic(slug: str, *, title: str, tags: list[str], body: str = "body") -> str:
    return (
        f"---\n"
        f"title: {title}\n"
        f"slug: {slug}\n"
        f"type: topic\n"
        f"tags: {tags!r}\n"
        f"created_at: 2026-05-02T00:00:00Z\n"
        f"updated_at: 2026-05-02T00:00:00Z\n"
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
async def test_tenant_scope_mismatch_blocks_set_pending(
    store: MarkdownLTMStore,
) -> None:
    with TenantScope.set("a"):
        with pytest.raises(RuntimeError):
            await store.set_context_pending(
                Diff(),
                ctx=SetContextPendingContext(request_id="r1", tenant_id="b"),
            )
