"""Bearer-token AuthBackend.

Tokens are random 32-byte URL-safe values displayed once on creation; the
SQLite backing stores only SHA-256 hashes. ``last_used_at`` is updated on
every authentication attempt (including failures for revoked tokens) so the
operator's audit trail is honest.
"""

from __future__ import annotations

import hashlib
import secrets as secrets_module
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, ClassVar
from uuid import uuid4

from sqlalchemy import (
    and_,
    delete,
    insert,
    select,
    update,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from dreamer.api.auth import AuthBackend
from dreamer.api.compat import implements
from dreamer.api.contexts import AuthContext
from dreamer.api.errors import AuthError
from dreamer.api.types import DEFAULT_TENANT_ID, Principal
from dreamer.contrib.stm.sqlite import (
    _engine_for,
    _ensure_aware,
    _rowcount,
    auth_tokens_table,
    init_schema,
)

if TYPE_CHECKING:
    from starlette.requests import Request


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class TokenRecord:
    id: str
    name: str
    created_at: datetime
    revoked_at: datetime | None
    last_used_at: datetime | None


@implements(AuthBackend, version=1)
class SimpleTokenAuth:
    """Bearer-token AuthBackend backed by SQLite.

    Reads ``Authorization: Bearer <token>`` from the incoming request,
    SHA-256 hashes the value, and looks it up in the ``auth_tokens`` table.
    """

    multi_tenant: ClassVar[bool] = False

    def __init__(self, *, db_path: str = "./dreamer.db") -> None:
        import asyncio  # noqa: PLC0415

        self.db_path = db_path
        self._engine = _engine_for(db_path)
        self._sessionmaker = async_sessionmaker(self._engine, expire_on_commit=False)
        self._init_lock = asyncio.Lock()
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            await init_schema(self._engine)
            self._initialized = True

    @asynccontextmanager
    async def _session(self) -> AsyncIterator[AsyncSession]:
        await self._ensure_initialized()
        async with self._sessionmaker() as session:
            yield session

    async def create_token(self, name: str) -> tuple[str, str]:
        """Mint a new token. Returns ``(id, plaintext)``; the plaintext is
        only ever returned here — the table holds the hash.
        """
        plaintext = secrets_module.token_urlsafe(32)
        token_id = str(uuid4())
        async with self._session() as session:
            await session.execute(
                insert(auth_tokens_table).values(
                    id=token_id,
                    name=name,
                    hash=_hash_token(plaintext),
                    created_at=_utcnow(),
                    revoked_at=None,
                    last_used_at=None,
                )
            )
            await session.commit()
        return token_id, plaintext

    async def list_tokens(self) -> list[TokenRecord]:
        async with self._session() as session:
            rows = (
                await session.execute(
                    select(auth_tokens_table).order_by(auth_tokens_table.c.created_at)
                )
            ).mappings().all()
            return [
                TokenRecord(
                    id=r["id"],
                    name=r["name"],
                    created_at=_ensure_aware(r["created_at"]),
                    revoked_at=_ensure_aware(r["revoked_at"]) if r["revoked_at"] else None,
                    last_used_at=_ensure_aware(r["last_used_at"])
                    if r["last_used_at"]
                    else None,
                )
                for r in rows
            ]

    async def revoke_token(self, token_id: str) -> bool:
        async with self._session() as session:
            result = await session.execute(
                update(auth_tokens_table)
                .where(
                    and_(
                        auth_tokens_table.c.id == token_id,
                        auth_tokens_table.c.revoked_at.is_(None),
                    )
                )
                .values(revoked_at=_utcnow())
            )
            await session.commit()
            return _rowcount(result) > 0

    async def delete_token(self, token_id: str) -> bool:
        async with self._session() as session:
            result = await session.execute(
                delete(auth_tokens_table).where(auth_tokens_table.c.id == token_id)
            )
            await session.commit()
            return _rowcount(result) > 0

    async def authenticate(self, request: Request, *, ctx: AuthContext) -> Principal:
        token = self._extract_bearer(request)
        if not token:
            raise AuthError("auth_failed: missing or malformed Authorization header")

        digest = _hash_token(token)
        async with self._session() as session:
            row = (
                await session.execute(
                    select(auth_tokens_table).where(auth_tokens_table.c.hash == digest)
                )
            ).mappings().first()

            if row is None:
                raise AuthError("auth_failed: unknown token")

            now = _utcnow()
            await session.execute(
                update(auth_tokens_table)
                .where(auth_tokens_table.c.id == row["id"])
                .values(last_used_at=now)
            )
            await session.commit()

            if row["revoked_at"] is not None:
                raise AuthError("auth_failed: token revoked")

            return Principal(
                id=row["id"],
                tenant_id=DEFAULT_TENANT_ID,
                metadata={"token_name": row["name"]},
            )

    @staticmethod
    def _extract_bearer(request: Request) -> str | None:
        try:
            header = request.headers.get("authorization")
        except AttributeError:
            return None
        if not header:
            return None
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer":
            return None
        return token.strip() or None


__all__ = ["SimpleTokenAuth", "TokenRecord"]
