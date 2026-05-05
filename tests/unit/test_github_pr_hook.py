"""The GitHub REST surface is exercised via a stub ``httpx``-shaped client that
records every request so we can assert URL/payload exactness."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from dreamer.api.contexts import (
    AuditContext,
    PostDreamContext,
    PostDreamServices,
    SecretContext,
    UsageContext,
)
from dreamer.api.types import (
    AuditEvent,
    Diff,
    SecretValue,
    UsageEvent,
)
from dreamer.contrib.hooks.git import GithubOpenPR, _slug_from_url


class _FakeResponse:
    def __init__(self, status_code: int = 200, body: Any = None) -> None:
        self.status_code = status_code
        self._body = body if body is not None else []

    def json(self) -> Any:
        return self._body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeClient:
    def __init__(
        self,
        *,
        get_handler: Any = None,
        post_handler: Any = None,
    ) -> None:
        self.requests: list[dict[str, Any]] = []
        self._get_handler = get_handler
        self._post_handler = post_handler

    async def __aenter__(self) -> _FakeClient:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def get(
        self, url: str, *, headers: dict[str, str], params: dict[str, str]
    ) -> _FakeResponse:
        self.requests.append(
            {"method": "GET", "url": url, "headers": headers, "params": params}
        )
        return self._get_handler(url=url, params=params)

    async def post(
        self, url: str, *, headers: dict[str, str], json: dict[str, Any]
    ) -> _FakeResponse:
        self.requests.append(
            {"method": "POST", "url": url, "headers": headers, "json": json}
        )
        return self._post_handler(url=url, json=json)


class _StaticSecrets:
    def __init__(self, name: str, value: str) -> None:
        self.name = name
        self.value = value
        self.calls: list[tuple[str, str | None]] = []

    async def get(
        self,
        name: str,
        *,
        tenant_id: str | None,
        ctx: SecretContext,
    ) -> SecretValue:
        self.calls.append((name, tenant_id))
        if name == self.name:
            return SecretValue(value=self.value)
        return SecretValue(value="")


class _NullUsageSink:
    async def record(self, event: UsageEvent, *, ctx: UsageContext) -> None:
        return None


class _NullAuditSink:
    async def record(self, event: AuditEvent, *, ctx: AuditContext) -> None:
        return None


async def _noop_emit(message: str, payload: Mapping[str, Any]) -> None:
    return None


def _services(*, secrets: Any) -> PostDreamServices:
    return PostDreamServices(
        emit_progress=_noop_emit,
        secrets=secrets,
        usage=_NullUsageSink(),
        audit=_NullAuditSink(),
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )


def _post_dream_ctx(
    *,
    success: bool = True,
    ltm_diff: Diff | None = None,
    context_diff: Diff | None = None,
    batch_size: int = 4,
    params: Mapping[str, Any] | None = None,
) -> PostDreamContext:
    return PostDreamContext(
        request_id="req-pr",
        tenant_id="default",
        lease_id="lease-pr",
        trigger_name="every_6h",
        success=success,
        batch_size=batch_size,
        ltm_diff=ltm_diff,
        context_diff=context_diff,
        resumed=False,
        error=None,
        params=params or {},
    )


def test_slug_from_url_handles_common_forms() -> None:
    assert _slug_from_url("git@github.com:owner/repo.git") == "owner/repo"
    assert _slug_from_url("https://github.com/owner/repo.git") == "owner/repo"
    assert _slug_from_url("https://github.com/owner/repo") == "owner/repo"
    assert _slug_from_url("ssh://git@github.com/owner/repo") == "owner/repo"
    assert _slug_from_url("not-a-url") is None


@pytest.mark.asyncio
async def test_opens_pr_when_no_existing_open_or_closed_match() -> None:
    secrets = _StaticSecrets("GITHUB_TOKEN", "ghp_abc")

    def get_handler(*, url: str, params: dict[str, str]) -> _FakeResponse:
        return _FakeResponse(200, [])

    posted: list[dict[str, Any]] = []

    def post_handler(*, url: str, json: dict[str, Any]) -> _FakeResponse:
        posted.append({"url": url, "json": json})
        return _FakeResponse(201, {"number": 42})

    client = _FakeClient(get_handler=get_handler, post_handler=post_handler)

    hook = GithubOpenPR(
        head="dreamer",
        base="main",
        repo_slug="owner/repo",
        http_client_factory=lambda: client,
    )
    ctx = _post_dream_ctx(
        ltm_diff=Diff(added=["memory/a.md"]),
        context_diff=Diff(modified=["context/AGENTS.md"]),
    )
    await hook.on_post_dream(ctx=ctx, services=_services(secrets=secrets))

    methods = [r["method"] for r in client.requests]
    assert methods == ["GET", "GET", "POST"]
    assert posted[0]["url"].endswith("/repos/owner/repo/pulls")
    payload = posted[0]["json"]
    assert payload["head"] == "dreamer"
    assert payload["base"] == "main"
    assert "dreamer:" in payload["title"]
    assert "batch size: 4" in payload["body"]


@pytest.mark.asyncio
async def test_comments_when_open_pr_already_exists() -> None:
    secrets = _StaticSecrets("GITHUB_TOKEN", "ghp_abc")

    def get_handler(*, url: str, params: dict[str, str]) -> _FakeResponse:
        if params.get("state") == "open":
            return _FakeResponse(200, [{"number": 17}])
        return _FakeResponse(200, [])

    posted: list[dict[str, Any]] = []

    def post_handler(*, url: str, json: dict[str, Any]) -> _FakeResponse:
        posted.append({"url": url, "json": json})
        return _FakeResponse(201, {"id": 999})

    client = _FakeClient(get_handler=get_handler, post_handler=post_handler)

    hook = GithubOpenPR(
        head="dreamer",
        base="main",
        repo_slug="owner/repo",
        http_client_factory=lambda: client,
    )
    ctx = _post_dream_ctx(ltm_diff=Diff(added=["memory/a.md"]))
    await hook.on_post_dream(ctx=ctx, services=_services(secrets=secrets))

    assert len(posted) == 1
    assert posted[0]["url"].endswith("/repos/owner/repo/issues/17/comments")
    assert "Dreamer update" in posted[0]["json"]["body"]


@pytest.mark.asyncio
async def test_does_not_reopen_closed_pr(caplog: pytest.LogCaptureFixture) -> None:
    secrets = _StaticSecrets("GITHUB_TOKEN", "ghp_abc")

    def get_handler(*, url: str, params: dict[str, str]) -> _FakeResponse:
        if params.get("state") == "open":
            return _FakeResponse(200, [])
        return _FakeResponse(200, [{"number": 9, "state": "closed"}])

    def post_handler(*, url: str, json: dict[str, Any]) -> _FakeResponse:
        raise AssertionError("must not POST when closed PR exists")

    client = _FakeClient(get_handler=get_handler, post_handler=post_handler)

    hook = GithubOpenPR(
        head="dreamer",
        base="main",
        repo_slug="owner/repo",
        http_client_factory=lambda: client,
    )
    ctx = _post_dream_ctx(ltm_diff=Diff(added=["memory/a.md"]))
    with caplog.at_level("INFO"):
        await hook.on_post_dream(ctx=ctx, services=_services(secrets=secrets))
    assert any("closed" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_skipped_when_no_diff() -> None:
    secrets = _StaticSecrets("GITHUB_TOKEN", "ghp_abc")

    def get_handler(*, url: str, params: dict[str, str]) -> _FakeResponse:
        raise AssertionError("must not call GitHub when both diffs empty")

    def post_handler(*, url: str, json: dict[str, Any]) -> _FakeResponse:
        raise AssertionError("must not POST when both diffs empty")

    client = _FakeClient(get_handler=get_handler, post_handler=post_handler)

    hook = GithubOpenPR(
        head="dreamer",
        base="main",
        repo_slug="owner/repo",
        http_client_factory=lambda: client,
    )
    ctx = _post_dream_ctx(ltm_diff=Diff(), context_diff=Diff())
    await hook.on_post_dream(ctx=ctx, services=_services(secrets=secrets))


@pytest.mark.asyncio
async def test_skipped_when_failure() -> None:
    secrets = _StaticSecrets("GITHUB_TOKEN", "ghp_abc")

    def get_handler(*, url: str, params: dict[str, str]) -> _FakeResponse:
        raise AssertionError("must not call GitHub on failure")

    def post_handler(*, url: str, json: dict[str, Any]) -> _FakeResponse:
        raise AssertionError("must not POST on failure")

    client = _FakeClient(get_handler=get_handler, post_handler=post_handler)
    hook = GithubOpenPR(
        head="dreamer",
        base="main",
        repo_slug="owner/repo",
        http_client_factory=lambda: client,
    )
    ctx = _post_dream_ctx(success=False, ltm_diff=Diff(added=["memory/a.md"]))
    await hook.on_post_dream(ctx=ctx, services=_services(secrets=secrets))


@pytest.mark.asyncio
async def test_skipped_when_token_missing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class _EmptySecrets:
        async def get(
            self,
            name: str,
            *,
            tenant_id: str | None,
            ctx: SecretContext,
        ) -> SecretValue:
            return SecretValue(value="")

    def get_handler(*, url: str, params: dict[str, str]) -> _FakeResponse:
        raise AssertionError("must not call GitHub without a token")

    def post_handler(*, url: str, json: dict[str, Any]) -> _FakeResponse:
        raise AssertionError("must not POST without a token")

    client = _FakeClient(get_handler=get_handler, post_handler=post_handler)
    hook = GithubOpenPR(
        head="dreamer",
        base="main",
        repo_slug="owner/repo",
        http_client_factory=lambda: client,
    )
    ctx = _post_dream_ctx(ltm_diff=Diff(added=["memory/a.md"]))
    with caplog.at_level("WARNING"):
        await hook.on_post_dream(ctx=ctx, services=_services(secrets=_EmptySecrets()))
    assert any("no GitHub token" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_per_tenant_repo_slug_override() -> None:
    secrets = _StaticSecrets("GITHUB_TOKEN", "ghp_abc")

    def get_handler(*, url: str, params: dict[str, str]) -> _FakeResponse:
        return _FakeResponse(200, [])

    posted: list[dict[str, Any]] = []

    def post_handler(*, url: str, json: dict[str, Any]) -> _FakeResponse:
        posted.append({"url": url, "json": json})
        return _FakeResponse(201, {"number": 1})

    client = _FakeClient(get_handler=get_handler, post_handler=post_handler)
    hook = GithubOpenPR(
        head="dreamer",
        base="main",
        repo_slug="owner/default",
        http_client_factory=lambda: client,
    )
    ctx = _post_dream_ctx(
        ltm_diff=Diff(added=["memory/a.md"]),
        params={"repo_slug": "owner/override"},
    )
    await hook.on_post_dream(ctx=ctx, services=_services(secrets=secrets))
    assert posted[0]["url"].endswith("/repos/owner/override/pulls")
