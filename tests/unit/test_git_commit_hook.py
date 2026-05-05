from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import git as gitpy
import pytest

from dreamer.api.contexts import (
    AuditContext,
    PostDreamContext,
    PostDreamServices,
    SecretContext,
    UsageContext,
)
from dreamer.api.errors import WorkspaceError
from dreamer.api.types import (
    AuditEvent,
    Diff,
    SecretValue,
    UsageEvent,
)
from dreamer.contrib.hooks.git import GitCommit


class _NullSecrets:
    async def get(
        self,
        name: str,
        *,
        tenant_id: str | None,
        ctx: SecretContext,
    ) -> SecretValue:
        return SecretValue(value="", ttl_seconds=None, version=None)


class _NullUsageSink:
    async def record(self, event: UsageEvent, *, ctx: UsageContext) -> None:
        return None


class _NullAuditSink:
    async def record(self, event: AuditEvent, *, ctx: AuditContext) -> None:
        return None


async def _noop_emit(message: str, payload: Mapping[str, Any]) -> None:
    return None


def _clock() -> datetime:
    return datetime(2026, 1, 1, tzinfo=UTC)


def _services() -> PostDreamServices:
    return PostDreamServices(
        emit_progress=_noop_emit,
        secrets=_NullSecrets(),
        usage=_NullUsageSink(),
        audit=_NullAuditSink(),
        clock=_clock,
    )


def _post_dream_ctx(
    *,
    success: bool = True,
    ltm_diff: Diff | None = None,
    context_diff: Diff | None = None,
    batch_size: int = 5,
    trigger_name: str = "every_6h",
    params: Mapping[str, Any] | None = None,
) -> PostDreamContext:
    return PostDreamContext(
        request_id="req-1",
        tenant_id="default",
        lease_id="lease-1",
        trigger_name=trigger_name,
        success=success,
        batch_size=batch_size,
        ltm_diff=ltm_diff,
        context_diff=context_diff,
        resumed=False,
        error=None,
        params=params or {},
    )


def _init_repo(path: Path, *, with_main: bool = True) -> gitpy.Repo:
    repo = gitpy.Repo.init(str(path))
    repo.config_writer().set_value("user", "email", "tests@dreamer.test").release()
    repo.config_writer().set_value("user", "name", "Tests").release()
    (path / ".gitignore").write_text(".venv/\n")
    repo.index.add([".gitignore"])
    repo.index.commit("init")
    if with_main:
        repo.git.branch("-M", "main")
    return repo


def _make_dirty_paths(repo_path: Path) -> None:
    (repo_path / "memory").mkdir(exist_ok=True)
    (repo_path / "memory" / "note.md").write_text("note\n")
    (repo_path / "context").mkdir(exist_ok=True)
    (repo_path / "context" / "AGENTS.md").write_text("agents\n")


@pytest.mark.asyncio
async def test_happy_path_commits_memory_and_context(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path, with_main=True)
    repo.git.checkout("-b", "dreamer")
    _make_dirty_paths(tmp_path)
    hook = GitCommit(repo=tmp_path, push=False, expect_clean_branch=True)
    ctx = _post_dream_ctx(
        ltm_diff=Diff(added=["memory/note.md"]),
        context_diff=Diff(added=["context/AGENTS.md"]),
        batch_size=5,
    )

    head_before = repo.head.commit.hexsha
    await hook.on_post_dream(ctx=ctx, services=_services())

    repo = gitpy.Repo(str(tmp_path))
    assert repo.head.commit.hexsha != head_before
    assert repo.active_branch.name == "dreamer"
    msg = repo.head.commit.message.strip()
    assert "5 memories from every_6h" in msg


@pytest.mark.asyncio
async def test_no_op_on_empty_diffs(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    repo.git.checkout("-b", "dreamer")
    hook = GitCommit(repo=tmp_path, push=False)
    ctx = _post_dream_ctx(
        ltm_diff=Diff(),
        context_diff=Diff(),
    )
    head_before = repo.head.commit.hexsha
    await hook.on_post_dream(ctx=ctx, services=_services())
    assert repo.head.commit.hexsha == head_before


@pytest.mark.asyncio
async def test_failure_path_does_not_commit(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    repo.git.checkout("-b", "dreamer")
    _make_dirty_paths(tmp_path)
    hook = GitCommit(repo=tmp_path, push=False)
    ctx = _post_dream_ctx(
        success=False,
        ltm_diff=Diff(added=["memory/note.md"]),
    )
    head_before = repo.head.commit.hexsha
    await hook.on_post_dream(ctx=ctx, services=_services())
    assert repo.head.commit.hexsha == head_before


@pytest.mark.asyncio
async def test_first_run_creates_branch_from_base(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path, with_main=True)
    assert repo.active_branch.name == "main"
    _make_dirty_paths(tmp_path)
    hook = GitCommit(
        repo=tmp_path,
        push=False,
        branch="dreamer",
        base_branch="main",
    )
    ctx = _post_dream_ctx(
        ltm_diff=Diff(added=["memory/note.md"]),
        context_diff=Diff(added=["context/AGENTS.md"]),
    )
    await hook.on_post_dream(ctx=ctx, services=_services())
    repo = gitpy.Repo(str(tmp_path))
    assert repo.active_branch.name == "dreamer"
    assert any(h.name == "dreamer" for h in repo.heads)


@pytest.mark.asyncio
async def test_aborts_on_unrelated_dirty_state(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    repo.git.checkout("-b", "dreamer")
    _make_dirty_paths(tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "code.py").write_text("# user code\n")
    hook = GitCommit(repo=tmp_path, push=False, expect_clean_branch=True)
    ctx = _post_dream_ctx(ltm_diff=Diff(added=["memory/note.md"]))
    with pytest.raises(WorkspaceError):
        await hook.on_post_dream(ctx=ctx, services=_services())


@pytest.mark.asyncio
async def test_aborts_when_on_unrelated_branch(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    repo.create_head("dreamer", repo.head.commit)
    repo.git.checkout("-b", "feature/x")
    _make_dirty_paths(tmp_path)
    hook = GitCommit(repo=tmp_path, push=False, branch="dreamer")
    ctx = _post_dream_ctx(ltm_diff=Diff(added=["memory/note.md"]))
    with pytest.raises(WorkspaceError):
        await hook.on_post_dream(ctx=ctx, services=_services())


@pytest.mark.asyncio
async def test_skips_when_repo_path_missing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    missing = tmp_path / "no-such-repo"
    hook = GitCommit(repo=missing, push=False)
    ctx = _post_dream_ctx(ltm_diff=Diff(added=["memory/note.md"]))
    with caplog.at_level("WARNING"):
        await hook.on_post_dream(ctx=ctx, services=_services())
    assert any("does not exist" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_skips_when_not_a_git_tree(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    not_a_repo = tmp_path / "plain"
    not_a_repo.mkdir()
    hook = GitCommit(repo=not_a_repo, push=False)
    ctx = _post_dream_ctx(ltm_diff=Diff(added=["memory/note.md"]))
    with caplog.at_level("WARNING"):
        await hook.on_post_dream(ctx=ctx, services=_services())
    assert any("not a git working tree" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_per_tenant_param_override_changes_branch(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path, with_main=True)
    _make_dirty_paths(tmp_path)
    hook = GitCommit(
        repo=tmp_path, push=False, branch="dreamer", base_branch="main"
    )
    ctx = _post_dream_ctx(
        ltm_diff=Diff(added=["memory/note.md"]),
        params={"branch": "dreamer-acme"},
    )
    await hook.on_post_dream(ctx=ctx, services=_services())
    repo = gitpy.Repo(str(tmp_path))
    assert repo.active_branch.name == "dreamer-acme"


@pytest.mark.asyncio
async def test_custom_commit_message_template(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    repo.git.checkout("-b", "dreamer")
    _make_dirty_paths(tmp_path)
    hook = GitCommit(
        repo=tmp_path,
        push=False,
        commit_message_template="custom {batch_size} from {tenant_id}",
    )
    ctx = _post_dream_ctx(
        ltm_diff=Diff(added=["memory/note.md"]), batch_size=3
    )
    await hook.on_post_dream(ctx=ctx, services=_services())
    repo = gitpy.Repo(str(tmp_path))
    assert repo.head.commit.message.strip() == "custom 3 from default"


def test_init_validation() -> None:
    with pytest.raises(Exception):
        GitCommit(branch="")
    with pytest.raises(Exception):
        GitCommit(base_branch="")
