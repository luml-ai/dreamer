"""Default git/PR post-dream hooks.

``GitCommit`` commits the LTM/context workspace to a dreamer-owned branch.
``GithubOpenPR`` opens or comments on a single rolling PR for that branch.

Both hooks are advisory ``PostDreamHook@1`` impls — failures are logged by
the orchestrator's hook runner; they never roll back already-committed state.
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Any, ClassVar

from dreamer.api.compat import implements
from dreamer.api.contexts import PostDreamContext, PostDreamServices, SecretContext
from dreamer.api.errors import ConfigError, WorkspaceError
from dreamer.api.hooks import PostDreamHook

logger = logging.getLogger(__name__)


def _is_diff_empty(diff: Any) -> bool:
    if diff is None:
        return True
    return not (diff.added or diff.modified or diff.deleted)


def _git_available() -> bool:
    try:
        import git as _git  # noqa: F401

        return True
    except ImportError:
        return False


@implements(PostDreamHook, version=1)
class GitCommit:
    """Commit ``memory/`` and ``context/`` changes onto a dreamer-owned branch.

    Conservative defaults: ``expect_clean_branch=True`` refuses to commit if
    the working tree has unrelated dirty paths. Only fires when
    ``ctx.success`` is ``True`` and at least one diff is non-empty.
    """

    multi_tenant: ClassVar[bool] = False

    def __init__(
        self,
        *,
        repo: Path | str = Path("./workspace"),
        branch: str = "dreamer",
        base_branch: str = "main",
        push: bool = True,
        remote: str = "origin",
        commit_message_template: str | None = None,
        expect_clean_branch: bool = True,
    ) -> None:
        if not branch:
            raise ConfigError("GitCommit: branch must be non-empty")
        if not base_branch:
            raise ConfigError("GitCommit: base_branch must be non-empty")
        self.repo = Path(repo)
        self.branch = branch
        self.base_branch = base_branch
        self.push = push
        self.remote = remote
        self.commit_message_template = commit_message_template
        self.expect_clean_branch = expect_clean_branch

    async def on_post_dream(
        self, *, ctx: PostDreamContext, services: PostDreamServices
    ) -> None:
        params = dict(ctx.params) if ctx.params else {}
        repo_path = Path(params.get("repo", self.repo))
        branch = str(params.get("branch", self.branch))
        base_branch = str(params.get("base_branch", self.base_branch))
        push = bool(params.get("push", self.push))
        remote = str(params.get("remote", self.remote))
        template = params.get("commit_message_template", self.commit_message_template)
        expect_clean = bool(params.get("expect_clean_branch", self.expect_clean_branch))

        if not ctx.success:
            return
        if _is_diff_empty(ctx.ltm_diff) and _is_diff_empty(ctx.context_diff):
            logger.info(
                "GitCommit: no diffs for tenant=%s lease=%s; skipping commit",
                ctx.tenant_id,
                ctx.lease_id,
            )
            return
        if not _git_available():
            logger.warning(
                "GitCommit: gitpython is not installed; "
                "skipping commit (install dreamer-server[git])"
            )
            return
        repo_exists = await asyncio.to_thread(repo_path.exists)
        if not repo_exists:
            logger.warning("GitCommit: repo path %s does not exist; skipping", repo_path)
            return

        await asyncio.to_thread(
            _run_git_commit,
            repo_path=repo_path,
            branch=branch,
            base_branch=base_branch,
            push=push,
            remote=remote,
            template=template,
            expect_clean=expect_clean,
            ctx=ctx,
        )


def _run_git_commit(
    *,
    repo_path: Path,
    branch: str,
    base_branch: str,
    push: bool,
    remote: str,
    template: str | None,
    expect_clean: bool,
    ctx: PostDreamContext,
) -> None:
    import git
    from git.exc import InvalidGitRepositoryError

    try:
        repo = git.Repo(str(repo_path))
    except InvalidGitRepositoryError:
        logger.warning(
            "GitCommit: %s is not a git working tree; skipping", repo_path
        )
        return
    except Exception:  # noqa: BLE001 — gitpython has many low-level failure modes
        logger.exception("GitCommit: failed to open repo at %s; skipping", repo_path)
        return

    _ensure_branch_checked_out(repo=repo, branch=branch, base_branch=base_branch)

    if expect_clean:
        unrelated = _unrelated_dirty_paths(repo=repo)
        if unrelated:
            raise WorkspaceError(
                "GitCommit: working tree has uncommitted changes outside "
                f"memory/ and context/: {sorted(unrelated)[:5]}"
            )

    repo.git.add("memory", "context")

    if not repo.is_dirty(index=True, working_tree=False, untracked_files=False):
        diff_paths = repo.index.diff("HEAD") if _has_head(repo) else None
        if not diff_paths:
            logger.info(
                "GitCommit: no changes to commit on %s; skipping",
                branch,
            )
            return

    message = _render_commit_message(template=template, ctx=ctx)
    repo.index.commit(message)

    if push:
        try:
            origin = repo.remote(name=remote)
        except ValueError:
            logger.warning(
                "GitCommit: remote %s is not configured; skipping push", remote
            )
            return
        try:
            origin.push(refspec=f"{branch}:{branch}", set_upstream=True)
        except Exception:  # noqa: BLE001 — push errors are advisory
            logger.exception("GitCommit: push to %s failed", remote)


def _ensure_branch_checked_out(*, repo: Any, branch: str, base_branch: str) -> None:
    """Ensure ``branch`` is the current checkout.

    On first run, create ``branch`` from ``base_branch`` if missing.
    Otherwise raise if the working tree is on something else.
    """
    try:
        active = repo.active_branch.name
    except TypeError:
        # Detached HEAD.
        raise WorkspaceError(
            "GitCommit: repository is in detached HEAD state; expected branch "
            f"{branch!r}"
        ) from None

    if active == branch:
        return

    branch_exists = any(h.name == branch for h in repo.heads)
    if branch_exists:
        raise WorkspaceError(
            f"GitCommit: working tree is on {active!r}; expected {branch!r}. "
            "Operators must keep dreamer pinned to its branch."
        )

    base_exists = any(h.name == base_branch for h in repo.heads)
    if not base_exists:
        raise WorkspaceError(
            f"GitCommit: base branch {base_branch!r} does not exist; cannot create {branch!r}"
        )

    if active != base_branch:
        raise WorkspaceError(
            f"GitCommit: working tree is on {active!r}; expected {branch!r} or {base_branch!r}"
        )

    new_branch = repo.create_head(branch, repo.heads[base_branch])
    new_branch.checkout()


def _unrelated_dirty_paths(*, repo: Any) -> set[str]:
    out: set[str] = set()
    if _has_head(repo):
        for diff in repo.index.diff(None):
            for path in (diff.a_path, diff.b_path):
                if path and not _path_under_dreamer(path):
                    out.add(path)
        for diff in repo.index.diff("HEAD"):
            for path in (diff.a_path, diff.b_path):
                if path and not _path_under_dreamer(path):
                    out.add(path)
    for path in repo.untracked_files:
        if path and not _path_under_dreamer(path):
            out.add(path)
    return out


def _path_under_dreamer(path: str) -> bool:
    return path.startswith("memory/") or path.startswith("context/")


def _has_head(repo: Any) -> bool:
    try:
        repo.head.commit  # noqa: B018 — accessing the property triggers the lookup
        return True
    except Exception:  # noqa: BLE001
        return False


def _render_commit_message(*, template: str | None, ctx: PostDreamContext) -> str:
    if template is None:
        template = (
            "dreamer: update memory and context "
            "({batch_size} memories from {trigger_name})"
        )
    return template.format(
        batch_size=ctx.batch_size,
        trigger_name=ctx.trigger_name,
        tenant_id=ctx.tenant_id,
        request_id=ctx.request_id,
    )


_REPO_SLUG_RE = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")


@implements(PostDreamHook, version=1)
class GithubOpenPR:
    """Open (or comment on) a single rolling PR from the dreamer branch."""

    multi_tenant: ClassVar[bool] = False

    def __init__(
        self,
        *,
        token: str | None = None,
        token_secret_name: str | None = "GITHUB_TOKEN",
        head: str = "dreamer",
        base: str = "main",
        repo_slug: str | None = None,
        repo_path: Path | str | None = None,
        title_template: str = "dreamer: rolling memory + context updates",
        api_base: str = "https://api.github.com",
        http_client_factory: Any = None,
    ) -> None:
        if not head:
            raise ConfigError("GithubOpenPR: head must be non-empty")
        if not base:
            raise ConfigError("GithubOpenPR: base must be non-empty")
        if repo_slug is not None and not _REPO_SLUG_RE.match(repo_slug):
            raise ConfigError(
                f"GithubOpenPR: repo_slug {repo_slug!r} must match owner/repo"
            )
        self.token = token
        self.token_secret_name = token_secret_name
        self.head = head
        self.base = base
        self.repo_slug = repo_slug
        self.repo_path = Path(repo_path) if repo_path is not None else None
        self.title_template = title_template
        self.api_base = api_base.rstrip("/")
        self._http_client_factory = http_client_factory
        self.secret_dependencies: frozenset[str] = (
            frozenset({token_secret_name}) if token_secret_name else frozenset()
        )

    async def on_post_dream(
        self, *, ctx: PostDreamContext, services: PostDreamServices
    ) -> None:
        params = dict(ctx.params) if ctx.params else {}
        head = str(params.get("head", self.head))
        base = str(params.get("base", self.base))
        repo_slug = params.get("repo_slug", self.repo_slug)
        repo_path = params.get("repo_path", self.repo_path)
        if repo_path is not None:
            repo_path = Path(repo_path)
        title_template = str(params.get("title_template", self.title_template))
        token_param = params.get("token", self.token)
        token_secret_name = params.get("token_secret_name", self.token_secret_name)

        if not ctx.success:
            return
        if _is_diff_empty(ctx.ltm_diff) and _is_diff_empty(ctx.context_diff):
            return

        token = await self._resolve_token(
            services=services,
            tenant_id=ctx.tenant_id,
            token_param=token_param,
            token_secret_name=token_secret_name,
            request_id=ctx.request_id,
        )
        if not token:
            logger.warning(
                "GithubOpenPR: no GitHub token available; skipping PR for tenant=%s",
                ctx.tenant_id,
            )
            return

        if repo_slug is None:
            repo_slug = _autodetect_repo_slug(repo_path)
            if repo_slug is None:
                logger.warning(
                    "GithubOpenPR: cannot determine repo_slug; "
                    "configure repo_slug or repo_path"
                )
                return

        client_factory = self._http_client_factory
        if client_factory is None:
            try:
                import httpx
            except ImportError:
                logger.warning(
                    "GithubOpenPR: httpx is not installed; install dreamer-server[git]"
                )
                return
            client_factory = lambda: httpx.AsyncClient(timeout=30.0)  # noqa: E731

        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        body = _render_pr_body(ctx)
        title = title_template.format(
            tenant_id=ctx.tenant_id,
            trigger_name=ctx.trigger_name,
            batch_size=ctx.batch_size,
        )

        async with client_factory() as client:
            existing = await _list_open_prs(
                client=client,
                api_base=self.api_base,
                headers=headers,
                repo_slug=repo_slug,
                head=head,
                base=base,
            )
            if existing.get("open"):
                pr_number = existing["pr_number"]
                await _comment_on_pr(
                    client=client,
                    api_base=self.api_base,
                    headers=headers,
                    repo_slug=repo_slug,
                    pr_number=pr_number,
                    body=body,
                )
                return
            if existing.get("closed"):
                logger.info(
                    "GithubOpenPR: existing PR %s is closed; not reopening",
                    existing.get("pr_number"),
                )
                return
            await _open_pr(
                client=client,
                api_base=self.api_base,
                headers=headers,
                repo_slug=repo_slug,
                head=head,
                base=base,
                title=title,
                body=body,
            )

    async def _resolve_token(
        self,
        *,
        services: PostDreamServices,
        tenant_id: str,
        token_param: Any,
        token_secret_name: Any,
        request_id: str,
    ) -> str | None:
        if isinstance(token_param, str) and token_param:
            return token_param
        if not token_secret_name:
            return None
        if services.secrets is None:
            return None
        try:
            value = await services.secrets.get(
                str(token_secret_name),
                tenant_id=tenant_id,
                ctx=SecretContext(request_id=request_id, tenant_id=tenant_id),
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "GithubOpenPR: secret resolution for %s failed",
                token_secret_name,
            )
            return None
        return value.value or None


def _autodetect_repo_slug(repo_path: Path | None) -> str | None:
    if repo_path is None:
        return None
    if not _git_available():
        return None
    try:
        import git

        repo = git.Repo(str(repo_path))
        url = next(iter(repo.remote(name="origin").urls), None)
    except Exception:  # noqa: BLE001
        return None
    if not url:
        return None
    return _slug_from_url(url)


def _slug_from_url(url: str) -> str | None:
    """Extract ``owner/repo`` from common GitHub URL forms.

    Handles ``git@github.com:owner/repo.git``, ``https://github.com/owner/repo.git``,
    ``ssh://git@github.com/owner/repo``, and trailing ``.git`` removal.
    """
    candidate = url.strip()
    for prefix in ("ssh://git@github.com/", "git@github.com:"):
        if candidate.startswith(prefix):
            candidate = candidate[len(prefix):]
            break
    else:
        for prefix in ("https://github.com/", "http://github.com/"):
            if candidate.startswith(prefix):
                candidate = candidate[len(prefix):]
                break
    if candidate.endswith(".git"):
        candidate = candidate[:-4]
    candidate = candidate.strip("/")
    if _REPO_SLUG_RE.match(candidate):
        return candidate
    return None


async def _list_open_prs(
    *,
    client: Any,
    api_base: str,
    headers: dict[str, str],
    repo_slug: str,
    head: str,
    base: str,
) -> dict[str, Any]:
    """Look for any PR matching head→base.

    Returns ``{"open": True, "pr_number": N}`` if open, ``{"closed": True,
    "pr_number": N}`` if closed/merged, otherwise ``{}``.
    """
    owner, _, _ = repo_slug.partition("/")
    head_param = f"{owner}:{head}"
    open_resp = await client.get(
        f"{api_base}/repos/{repo_slug}/pulls",
        headers=headers,
        params={"state": "open", "head": head_param, "base": base},
    )
    open_resp.raise_for_status()
    items = open_resp.json()
    if items:
        return {"open": True, "pr_number": items[0]["number"]}
    closed_resp = await client.get(
        f"{api_base}/repos/{repo_slug}/pulls",
        headers=headers,
        params={"state": "closed", "head": head_param, "base": base},
    )
    closed_resp.raise_for_status()
    closed_items = closed_resp.json()
    if closed_items:
        return {"closed": True, "pr_number": closed_items[0]["number"]}
    return {}


async def _comment_on_pr(
    *,
    client: Any,
    api_base: str,
    headers: dict[str, str],
    repo_slug: str,
    pr_number: int,
    body: str,
) -> None:
    resp = await client.post(
        f"{api_base}/repos/{repo_slug}/issues/{pr_number}/comments",
        headers=headers,
        json={"body": body},
    )
    resp.raise_for_status()


async def _open_pr(
    *,
    client: Any,
    api_base: str,
    headers: dict[str, str],
    repo_slug: str,
    head: str,
    base: str,
    title: str,
    body: str,
) -> None:
    resp = await client.post(
        f"{api_base}/repos/{repo_slug}/pulls",
        headers=headers,
        json={"title": title, "head": head, "base": base, "body": body},
    )
    resp.raise_for_status()


def _render_pr_body(ctx: PostDreamContext) -> str:
    parts = [
        f"Dreamer update from trigger `{ctx.trigger_name}`.",
        f"- batch size: {ctx.batch_size}",
        f"- tenant: `{ctx.tenant_id}`",
    ]
    if ctx.ltm_diff is not None:
        parts.append(
            f"- LTM: +{len(ctx.ltm_diff.added)} ~{len(ctx.ltm_diff.modified)} "
            f"-{len(ctx.ltm_diff.deleted)}"
        )
    if ctx.context_diff is not None:
        parts.append(
            f"- Context: +{len(ctx.context_diff.added)} "
            f"~{len(ctx.context_diff.modified)} -{len(ctx.context_diff.deleted)}"
        )
    return "\n".join(parts)


__all__ = ["GitCommit", "GithubOpenPR"]
