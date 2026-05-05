from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from click.testing import CliRunner

from dreamer.api.contexts import AuthContext
from dreamer.api.errors import AuthError
from dreamer.api.types import DEFAULT_TENANT_ID
from dreamer.contrib.auth.simple_token import SimpleTokenAuth
from dreamer.contrib.auth.simple_token.backend import _hash_token
from dreamer.contrib.auth.simple_token.cli import main as auth_cli
from dreamer.contrib.stm.sqlite import (
    _engines,
    auth_tokens_table,
)


@pytest_asyncio.fixture(autouse=True)
async def _isolate_engines() -> AsyncIterator[None]:
    yield
    for engine in list(_engines.values()):
        await engine.dispose()
    _engines.clear()


class _FakeRequest:
    def __init__(self, headers: dict[str, str]) -> None:
        self.headers = headers


def _ctx(req_id: str = "r1") -> AuthContext:
    return AuthContext(request_id=req_id)


@pytest.mark.asyncio
async def test_create_then_authenticate(tmp_path: Path) -> None:
    backend = SimpleTokenAuth(db_path=str(tmp_path / "auth.db"))
    token_id, plaintext = await backend.create_token("agent-A")
    request = _FakeRequest({"authorization": f"Bearer {plaintext}"})
    principal = await backend.authenticate(request, ctx=_ctx())
    assert principal.id == token_id
    assert principal.tenant_id == DEFAULT_TENANT_ID
    assert principal.metadata.get("token_name") == "agent-A"


@pytest.mark.asyncio
async def test_unknown_token_raises(tmp_path: Path) -> None:
    backend = SimpleTokenAuth(db_path=str(tmp_path / "auth.db"))
    await backend.create_token("agent-A")
    request = _FakeRequest({"authorization": "Bearer not-a-real-token"})
    with pytest.raises(AuthError):
        await backend.authenticate(request, ctx=_ctx())


@pytest.mark.asyncio
async def test_missing_authorization_header_raises(tmp_path: Path) -> None:
    backend = SimpleTokenAuth(db_path=str(tmp_path / "auth.db"))
    request = _FakeRequest({})
    with pytest.raises(AuthError):
        await backend.authenticate(request, ctx=_ctx())


@pytest.mark.asyncio
async def test_non_bearer_scheme_rejected(tmp_path: Path) -> None:
    backend = SimpleTokenAuth(db_path=str(tmp_path / "auth.db"))
    request = _FakeRequest({"authorization": "Basic abc"})
    with pytest.raises(AuthError):
        await backend.authenticate(request, ctx=_ctx())


@pytest.mark.asyncio
async def test_revoked_token_rejected_but_last_used_updates(tmp_path: Path) -> None:
    backend = SimpleTokenAuth(db_path=str(tmp_path / "auth.db"))
    token_id, plaintext = await backend.create_token("agent-A")
    assert await backend.revoke_token(token_id) is True
    request = _FakeRequest({"authorization": f"Bearer {plaintext}"})
    with pytest.raises(AuthError):
        await backend.authenticate(request, ctx=_ctx())
    [record] = await backend.list_tokens()
    assert record.revoked_at is not None
    # `last_used_at` bumps on each use, even when revoked.
    assert record.last_used_at is not None


@pytest.mark.asyncio
async def test_list_does_not_reveal_plaintext(tmp_path: Path) -> None:
    backend = SimpleTokenAuth(db_path=str(tmp_path / "auth.db"))
    _, plaintext = await backend.create_token("agent-A")
    records = await backend.list_tokens()
    assert len(records) == 1
    rec = records[0]
    digest = _hash_token(plaintext)
    for slot in rec.__slots__:
        value = getattr(rec, slot)
        assert value != plaintext
        assert value != digest


@pytest.mark.asyncio
async def test_multiple_active_tokens_supported(tmp_path: Path) -> None:
    backend = SimpleTokenAuth(db_path=str(tmp_path / "auth.db"))
    _id_a, plain_a = await backend.create_token("agent-A")
    _id_b, plain_b = await backend.create_token("agent-B")
    pa = await backend.authenticate(
        _FakeRequest({"authorization": f"Bearer {plain_a}"}), ctx=_ctx()
    )
    pb = await backend.authenticate(
        _FakeRequest({"authorization": f"Bearer {plain_b}"}), ctx=_ctx()
    )
    assert pa.id != pb.id
    assert pa.metadata.get("token_name") == "agent-A"
    assert pb.metadata.get("token_name") == "agent-B"


@pytest.mark.asyncio
async def test_last_used_updates_on_success(tmp_path: Path) -> None:
    backend = SimpleTokenAuth(db_path=str(tmp_path / "auth.db"))
    _, plaintext = await backend.create_token("agent-A")
    [pre] = await backend.list_tokens()
    assert pre.last_used_at is None
    await backend.authenticate(
        _FakeRequest({"authorization": f"Bearer {plaintext}"}), ctx=_ctx()
    )
    [post] = await backend.list_tokens()
    assert post.last_used_at is not None


@pytest.mark.asyncio
async def test_token_hash_matches_storage(tmp_path: Path) -> None:
    backend = SimpleTokenAuth(db_path=str(tmp_path / "auth.db"))
    _, plaintext = await backend.create_token("agent-A")
    digest = _hash_token(plaintext)
    async with backend._session() as session:
        from sqlalchemy import select  # noqa: PLC0415

        rows = (await session.execute(select(auth_tokens_table))).mappings().all()
        assert len(rows) == 1
        row: dict[str, Any] = dict(rows[0])
        assert row["hash"] == digest
        assert plaintext not in row["hash"]


def test_cli_token_create_list_revoke(tmp_path: Path) -> None:
    db = tmp_path / "auth.db"
    runner = CliRunner()

    create = runner.invoke(
        auth_cli, ["token", "create", "--name", "agent-A", "--db", str(db)]
    )
    assert create.exit_code == 0, create.output
    assert "name:  agent-A" in create.output
    assert "token: " in create.output

    listed = runner.invoke(auth_cli, ["token", "list", "--db", str(db)])
    assert listed.exit_code == 0, listed.output
    assert "agent-A" in listed.output
    plaintext_line = [
        line for line in create.output.splitlines() if line.startswith("token: ")
    ][0]
    plaintext = plaintext_line.split(": ", 1)[1].strip()
    assert plaintext not in listed.output

    id_line = [
        line for line in create.output.splitlines() if line.startswith("id:")
    ][0]
    token_id = id_line.split(maxsplit=1)[1].strip()

    revoke = runner.invoke(auth_cli, ["token", "revoke", token_id, "--db", str(db)])
    assert revoke.exit_code == 0
    assert f"revoked: {token_id}" in revoke.output

    again = runner.invoke(auth_cli, ["token", "revoke", token_id, "--db", str(db)])
    assert again.exit_code == 0
    assert "no-op" in again.output


def test_cli_list_empty(tmp_path: Path) -> None:
    db = tmp_path / "auth.db"
    runner = CliRunner()
    listed = runner.invoke(auth_cli, ["token", "list", "--db", str(db)])
    assert listed.exit_code == 0
    assert "(no tokens)" in listed.output
