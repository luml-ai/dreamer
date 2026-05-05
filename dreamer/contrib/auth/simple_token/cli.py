from __future__ import annotations

import asyncio

import click

from dreamer.contrib.auth.simple_token.backend import SimpleTokenAuth


@click.group()
def main() -> None:
    """Manage tokens for the default SimpleTokenAuth backend."""


@main.group()
def token() -> None:
    """Token operations."""


@token.command("create")
@click.option(
    "--name",
    required=True,
    help="Human-readable label for the token (e.g. agent name).",
)
@click.option(
    "--db",
    "db_path",
    default="./dreamer.db",
    show_default=True,
    help="Path to the SQLite database holding tokens.",
)
def token_create(name: str, db_path: str) -> None:
    """Mint a new token. The plaintext value is printed exactly once."""
    backend = SimpleTokenAuth(db_path=db_path)
    token_id, plaintext = asyncio.run(backend.create_token(name))
    click.echo(f"id:    {token_id}")
    click.echo(f"name:  {name}")
    click.echo(f"token: {plaintext}")
    click.echo("(record the token now — it cannot be recovered later)")


@token.command("list")
@click.option(
    "--db",
    "db_path",
    default="./dreamer.db",
    show_default=True,
    help="Path to the SQLite database holding tokens.",
)
def token_list(db_path: str) -> None:
    """List known tokens (id, name, lifecycle timestamps; never the plaintext)."""
    backend = SimpleTokenAuth(db_path=db_path)
    records = asyncio.run(backend.list_tokens())
    if not records:
        click.echo("(no tokens)")
        return
    for r in records:
        revoked = r.revoked_at.isoformat() if r.revoked_at else "-"
        last_used = r.last_used_at.isoformat() if r.last_used_at else "-"
        click.echo(
            f"{r.id}\t{r.name}\tcreated={r.created_at.isoformat()}\t"
            f"revoked={revoked}\tlast_used={last_used}"
        )


@token.command("revoke")
@click.argument("token_id")
@click.option(
    "--db",
    "db_path",
    default="./dreamer.db",
    show_default=True,
    help="Path to the SQLite database holding tokens.",
)
def token_revoke(token_id: str, db_path: str) -> None:
    """Revoke a token by id. Idempotent — already-revoked tokens stay revoked."""
    backend = SimpleTokenAuth(db_path=db_path)
    revoked = asyncio.run(backend.revoke_token(token_id))
    if revoked:
        click.echo(f"revoked: {token_id}")
    else:
        click.echo(f"no-op: {token_id} (unknown id or already revoked)")


if __name__ == "__main__":  # pragma: no cover
    main()
