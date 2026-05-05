from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, ClassVar

import pytest
import pytest_asyncio

from dreamer.api.contexts import (
    CommitWorkspaceContext,
    DiscardWorkspaceContext,
    OpenWorkspaceContext,
)
from dreamer.api.errors import ConfigError, WorkspaceError
from dreamer.api.tenants import TenantScope
from dreamer.api.types import Diff, FileViewable, RecordViewable, Workspace
from dreamer.contrib.context.fanout import FanoutContextStore
from dreamer.contrib.context.markdown import MarkdownContextStore


@pytest_asyncio.fixture(autouse=True)
async def _clear_tenant_scope() -> AsyncIterator[None]:
    TenantScope.clear()
    yield
    TenantScope.clear()


class _RecordOnlyContextStore:
    multi_tenant: ClassVar[bool] = False
    workspace_capabilities: ClassVar[frozenset[type]] = frozenset({RecordViewable})

    async def open_workspace(self, *, ctx: OpenWorkspaceContext) -> Workspace:
        raise NotImplementedError

    async def commit_workspace(
        self, ws: Workspace, *, ctx: CommitWorkspaceContext
    ) -> Diff:
        raise NotImplementedError

    async def discard_workspace(
        self, ws: Workspace, *, ctx: DiscardWorkspaceContext
    ) -> None:
        return None


class _FailingMarkdownContextStore(MarkdownContextStore):
    multi_tenant: ClassVar[bool] = False
    fail_on_commit: bool = False

    def __init__(self, *, root: Path | str) -> None:
        super().__init__(root=root)
        self.commit_calls = 0
        self.discard_calls = 0

    async def commit_workspace(
        self, ws: Workspace, *, ctx: CommitWorkspaceContext
    ) -> Diff:
        self.commit_calls += 1
        if self.fail_on_commit:
            raise RuntimeError("simulated commit failure")
        return await super().commit_workspace(ws, ctx=ctx)

    async def discard_workspace(
        self, ws: Workspace, *, ctx: DiscardWorkspaceContext
    ) -> None:
        self.discard_calls += 1
        await super().discard_workspace(ws, ctx=ctx)


def test_uniform_capabilities_accepted(tmp_path: Path) -> None:
    a = MarkdownContextStore(root=tmp_path / "a")
    b = MarkdownContextStore(root=tmp_path / "b")
    fan = FanoutContextStore([a, b])
    assert fan.workspace_capabilities == frozenset({FileViewable})
    assert fan.backings == (a, b)


def test_non_uniform_capabilities_rejected(tmp_path: Path) -> None:
    a = MarkdownContextStore(root=tmp_path / "a")
    b = _RecordOnlyContextStore()
    with pytest.raises(ConfigError, match="non-uniform capabilities"):
        FanoutContextStore([a, b])


def test_empty_backings_rejected() -> None:
    with pytest.raises(ConfigError, match="at least one backing"):
        FanoutContextStore([])


@pytest.mark.asyncio
async def test_open_seeds_staging_from_first_backing(tmp_path: Path) -> None:
    a = MarkdownContextStore(root=tmp_path / "a")
    (a.root / "skills" / "demo").mkdir(parents=True, exist_ok=True)
    (a.root / "skills" / "demo" / "SKILL.md").write_text(
        "---\nname: demo\ndescription: x\nversion: 0\n---\n", encoding="utf-8"
    )
    b = MarkdownContextStore(root=tmp_path / "b")
    fan = FanoutContextStore([a, b])
    with TenantScope.set("default"):
        ws = await fan.open_workspace(
            ctx=OpenWorkspaceContext(request_id="r1", tenant_id="default")
        )
        view = await ws.file_view()  # type: ignore[attr-defined]
        assert (view / "skills" / "demo" / "SKILL.md").is_file()
        await fan.discard_workspace(
            ws, ctx=DiscardWorkspaceContext(request_id="r1", tenant_id="default")
        )


@pytest.mark.asyncio
async def test_commit_fan_outs_to_every_backing(tmp_path: Path) -> None:
    a = MarkdownContextStore(root=tmp_path / "a")
    b = MarkdownContextStore(root=tmp_path / "b")
    fan = FanoutContextStore([a, b])
    with TenantScope.set("default"):
        ws = await fan.open_workspace(
            ctx=OpenWorkspaceContext(request_id="r1", tenant_id="default")
        )
        view = await ws.file_view()  # type: ignore[attr-defined]
        (view / "AGENTS.md").write_text("# fanned\n", encoding="utf-8")
        diff = await fan.commit_workspace(
            ws, ctx=CommitWorkspaceContext(request_id="r1", tenant_id="default")
        )
    assert (a.root / "AGENTS.md").read_text() == "# fanned\n"
    assert (b.root / "AGENTS.md").read_text() == "# fanned\n"
    assert "AGENTS.md" in diff.modified or "AGENTS.md" in diff.added
    per_store: Any = diff.metadata.get("per_store")
    assert isinstance(per_store, list) and len(per_store) == 2
    assert {p["backing"] for p in per_store} == {"MarkdownContextStore"}


@pytest.mark.asyncio
async def test_partial_commit_failure_rolls_back_remaining(tmp_path: Path) -> None:
    a = MarkdownContextStore(root=tmp_path / "a")
    b = _FailingMarkdownContextStore(root=tmp_path / "b")
    c = MarkdownContextStore(root=tmp_path / "c")
    b.fail_on_commit = True
    fan = FanoutContextStore([a, b, c])

    with TenantScope.set("default"):
        ws = await fan.open_workspace(
            ctx=OpenWorkspaceContext(request_id="r1", tenant_id="default")
        )
        view = await ws.file_view()  # type: ignore[attr-defined]
        (view / "AGENTS.md").write_text("# new\n", encoding="utf-8")
        with pytest.raises(WorkspaceError, match="partial commit"):
            await fan.commit_workspace(
                ws, ctx=CommitWorkspaceContext(request_id="r1", tenant_id="default")
            )
    assert b.discard_calls >= 1
    assert not (c.root / "AGENTS.md").exists()


@pytest.mark.asyncio
async def test_discard_workspace_propagates_to_all_backings(
    tmp_path: Path,
) -> None:
    a = _FailingMarkdownContextStore(root=tmp_path / "a")
    b = _FailingMarkdownContextStore(root=tmp_path / "b")
    fan = FanoutContextStore([a, b])
    with TenantScope.set("default"):
        ws = await fan.open_workspace(
            ctx=OpenWorkspaceContext(request_id="r1", tenant_id="default")
        )
        await fan.discard_workspace(
            ws, ctx=DiscardWorkspaceContext(request_id="r1", tenant_id="default")
        )
    assert a.discard_calls == 1
    assert b.discard_calls == 1


def test_multi_tenant_intersection(tmp_path: Path) -> None:
    a = MarkdownContextStore(root=tmp_path / "a")
    b = MarkdownContextStore(root=tmp_path / "b")
    fan = FanoutContextStore([a, b])
    assert fan.multi_tenant is False


def test_multi_tenant_when_all_backings_are_mt(tmp_path: Path) -> None:
    class MTBacking(MarkdownContextStore):
        multi_tenant: ClassVar[bool] = True

    a = MTBacking(root=tmp_path / "a")
    b = MTBacking(root=tmp_path / "b")
    fan = FanoutContextStore([a, b])
    assert fan.multi_tenant is True
