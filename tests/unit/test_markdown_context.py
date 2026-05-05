from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio

from dreamer.api.contexts import (
    CommitWorkspaceContext,
    ContextReadContext,
    OpenWorkspaceContext,
)
from dreamer.api.stores import ContextReader
from dreamer.api.tenants import TenantScope
from dreamer.api.types import FileViewable
from dreamer.contrib.context.markdown import MarkdownContextStore


@pytest_asyncio.fixture(autouse=True)
async def _clear_tenant_scope() -> AsyncIterator[None]:
    TenantScope.clear()
    yield
    TenantScope.clear()


@pytest.fixture()
def store(tmp_path: Path) -> MarkdownContextStore:
    return MarkdownContextStore(root=tmp_path / "context")


def test_root_scaffolded_with_skills_and_schema(
    store: MarkdownContextStore,
) -> None:
    """AGENTS.md is owned by the dream agent and is created on first commit;
    the store only scaffolds the directory shell + schema."""
    assert not (store.root / "AGENTS.md").exists()
    assert (store.root / "skills").is_dir()
    assert (store.root / "_meta" / "schema.md").is_file()


def test_implements_protocols(store: MarkdownContextStore) -> None:
    assert isinstance(store, ContextReader)
    assert FileViewable in store.workspace_capabilities


@pytest.mark.asyncio
async def test_read_returns_canonical_content(store: MarkdownContextStore) -> None:
    (store.root / "AGENTS.md").write_text("hello\n", encoding="utf-8")
    with TenantScope.set("default"):
        body = await store.read(
            "AGENTS.md",
            ctx=ContextReadContext(request_id="r1", tenant_id="default"),
        )
    assert body == b"hello\n"


@pytest.mark.asyncio
async def test_read_missing_path_raises(store: MarkdownContextStore) -> None:
    with TenantScope.set("default"):
        with pytest.raises(FileNotFoundError):
            await store.read(
                "missing.md",
                ctx=ContextReadContext(request_id="r1", tenant_id="default"),
            )


@pytest.mark.asyncio
async def test_read_rejects_path_traversal(store: MarkdownContextStore) -> None:
    with TenantScope.set("default"):
        with pytest.raises(FileNotFoundError):
            await store.read(
                "../../etc/passwd",
                ctx=ContextReadContext(request_id="r1", tenant_id="default"),
            )


@pytest.mark.asyncio
async def test_list_returns_relative_paths(store: MarkdownContextStore) -> None:
    (store.root / "AGENTS.md").write_text("# agents\n", encoding="utf-8")
    skill_dir = store.root / "skills" / "git-cleanup"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: git-cleanup\ndescription: x\nversion: 0\n---\n", encoding="utf-8"
    )
    with TenantScope.set("default"):
        listing = await store.list(
            ctx=ContextReadContext(request_id="r1", tenant_id="default")
        )
    assert "AGENTS.md" in listing
    assert "skills/git-cleanup/SKILL.md" in listing


@pytest.mark.asyncio
async def test_list_with_prefix(store: MarkdownContextStore) -> None:
    skill_dir = store.root / "skills" / "git-cleanup"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: git-cleanup\ndescription: x\nversion: 0\n---\n", encoding="utf-8"
    )
    with TenantScope.set("default"):
        listing = await store.list(
            prefix="skills",
            ctx=ContextReadContext(request_id="r1", tenant_id="default"),
        )
    assert listing == ["skills/git-cleanup/SKILL.md"]


@pytest.mark.asyncio
async def test_workspace_commit_round_trip(store: MarkdownContextStore) -> None:
    with TenantScope.set("default"):
        ws = await store.open_workspace(
            ctx=OpenWorkspaceContext(request_id="r1", tenant_id="default")
        )
        view = await ws.file_view()  # type: ignore[attr-defined]
        (view / "AGENTS.md").write_text("v2\n", encoding="utf-8")
        (view / "skills" / "new-skill").mkdir(parents=True, exist_ok=True)
        (view / "skills" / "new-skill" / "SKILL.md").write_text(
            "---\nname: new-skill\ndescription: hello\nversion: 0\n---\n",
            encoding="utf-8",
        )
        diff = await store.commit_workspace(
            ws, ctx=CommitWorkspaceContext(request_id="r1", tenant_id="default")
        )
        assert "skills/new-skill/SKILL.md" in diff.added
        assert (store.root / "skills" / "new-skill" / "SKILL.md").read_text() == (
            "---\nname: new-skill\ndescription: hello\nversion: 0\n---\n"
        )
