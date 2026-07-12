"""SQLite-backed default STM store and DreamLease store.

Both implementations use SQLAlchemy 2.x async with aiosqlite. They MAY share
the same database file (the typical default config does); each defines its
own table so a single ``db_path`` works fine.

Tenant scope: ``TenantScope.assert_matches`` runs on every method that touches
storage. The default ``multi_tenant = False`` is preserved per spec — the
store still keys data on ``tenant_id`` so MT-capable forks can flip the flag,
but in single-tenant deployments it's the operator's responsibility to ensure
only ``"default"`` is in use.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar
from uuid import uuid4

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Index,
    MetaData,
    String,
    Table,
    Text,
    and_,
    delete,
    func,
    insert,
    select,
    update,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from dreamer.api.compat import implements
from dreamer.api.contexts import (
    AcquireLeaseContext,
    ClaimContext,
    CountContext,
    ListUnconsumedContext,
    MarkConsumedContext,
    PurgeConsumedContext,
    ReclaimContext,
    ReclaimLeasesContext,
    ReleaseContext,
    ReleaseLeaseContext,
    RenewLeaseContext,
    SubmitContext,
)
from dreamer.api.stores import DreamLeaseStore, STMStore
from dreamer.api.tenants import TenantScope
from dreamer.api.types import DreamLease, Memory, MemoryBatch

_metadata = MetaData()


# Locks live on each store instance, NOT in this dict — sharing locks across
# pytest event loops makes them unusable.
_engines: dict[str, AsyncEngine] = {}


def _resolve_db_path(db_path: str) -> str:
    """Return an absolute path string for the configured ``db_path``.

    ``:memory:`` and any URL-style value (``sqlite://...``) are passed through
    unchanged so callers can opt into ephemeral or pooled-shared databases.
    """
    if db_path == ":memory:" or db_path.startswith("sqlite"):
        return db_path
    return os.path.abspath(db_path)


def _engine_for(db_path: str) -> AsyncEngine:
    resolved = _resolve_db_path(db_path)
    engine = _engines.get(resolved)
    if engine is not None:
        return engine
    if resolved.startswith("sqlite"):
        url = resolved if "+aiosqlite" in resolved else resolved.replace(
            "sqlite://", "sqlite+aiosqlite://", 1
        )
    elif resolved == ":memory:":
        url = "sqlite+aiosqlite:///:memory:"
    else:
        url = f"sqlite+aiosqlite:///{resolved}"
    engine = create_async_engine(url, future=True)
    _engines[resolved] = engine
    return engine


memories_table = Table(
    "stm_memories",
    _metadata,
    Column("id", String, primary_key=True),
    Column("tenant_id", String, nullable=False, index=True),
    Column("agent_id", String, nullable=False),
    Column("type", String, nullable=False),
    Column("title", String, nullable=False),
    Column("content", Text, nullable=False),
    Column("tags", JSON, nullable=False, default=list),
    Column("metadata", JSON, nullable=False, default=dict),
    Column("submitted_at", DateTime(timezone=True), nullable=False),
    Column("consumed_at", DateTime(timezone=True), nullable=True),
    Column("consumed_by_lease", String, nullable=True, index=True),
    Column("idempotency_key", String, nullable=True),
    Column("extra", JSON, nullable=False, default=dict),
)


# Partial unique index lets multiple rows have idempotency_key = NULL while
# enforcing uniqueness over (tenant_id, idempotency_key) when set.
Index(
    "ix_stm_memories_tenant_idempotency",
    memories_table.c.tenant_id,
    memories_table.c.idempotency_key,
    unique=True,
    sqlite_where=memories_table.c.idempotency_key.is_not(None),
)


leases_table = Table(
    "dream_leases",
    _metadata,
    Column("id", String, primary_key=True),
    Column("tenant_id", String, nullable=False, index=True),
    Column("acquired_at", DateTime(timezone=True), nullable=False),
    Column("expires_at", DateTime(timezone=True), nullable=False),
)


auth_tokens_table = Table(
    "auth_tokens",
    _metadata,
    Column("id", String, primary_key=True),
    Column("name", String, nullable=False),
    Column("hash", String, nullable=False, unique=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("revoked_at", DateTime(timezone=True), nullable=True),
    Column("last_used_at", DateTime(timezone=True), nullable=True),
)


async def init_schema(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(_metadata.create_all)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _rowcount(result: Any) -> int:
    """Return ``result.rowcount`` as a non-negative int.

    SQLAlchemy's typed Result base has no ``rowcount``; only CursorResult
    (returned by INSERT/UPDATE/DELETE execution) does. We use getattr for a
    forgiving type contract.
    """
    return int(getattr(result, "rowcount", 0) or 0)


def _ensure_aware(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt


def _row_to_memory(row: Any) -> Memory:
    extra = dict(row["extra"] or {})
    return Memory(
        id=row["id"],
        tenant_id=row["tenant_id"],
        agent_id=row["agent_id"],
        type=row["type"],
        title=row["title"],
        content=row["content"],
        tags=list(row["tags"] or []),
        metadata=dict(row["metadata"] or {}),
        submitted_at=_ensure_aware(row["submitted_at"]),
        consumed_at=_ensure_aware(row["consumed_at"]) if row["consumed_at"] else None,
        consumed_by_lease=row["consumed_by_lease"],
        idempotency_key=row["idempotency_key"],
        **extra,
    )


def _memory_to_row(memory: Memory) -> dict[str, Any]:
    base_field_names = {
        "id",
        "tenant_id",
        "agent_id",
        "type",
        "title",
        "content",
        "tags",
        "metadata",
        "submitted_at",
        "consumed_at",
        "consumed_by_lease",
        "idempotency_key",
    }
    dumped = memory.model_dump()
    extras = {k: v for k, v in dumped.items() if k not in base_field_names}
    return {
        "id": memory.id,
        "tenant_id": memory.tenant_id,
        "agent_id": memory.agent_id,
        "type": memory.type,
        "title": memory.title,
        "content": memory.content,
        "tags": list(memory.tags),
        "metadata": dict(memory.metadata),
        "submitted_at": memory.submitted_at,
        "consumed_at": memory.consumed_at,
        "consumed_by_lease": memory.consumed_by_lease,
        "idempotency_key": memory.idempotency_key,
        "extra": extras,
    }


@implements(STMStore, version=1)
class SQLiteSTMStore:
    """Default SQLite STM store. Single-tenant by default."""

    multi_tenant: ClassVar[bool] = False

    def __init__(
        self,
        *,
        db_path: str = "./dreamer.db",
        max_batch_size: int = 200,
        max_content_bytes: int = 8192,
        memory_types: list[Mapping[str, Any]] | None = None,
    ) -> None:
        self.db_path = db_path
        self.max_batch_size = max_batch_size
        self.max_content_bytes = max_content_bytes
        # ``memory_types`` is informational here: governance is done at the
        # MCP entry point and config layer. We accept it so the same config
        # block describes both the entry-point validation and the store's
        # provenance. We do not use it in storage decisions.
        self.memory_types = list(memory_types or [])
        self._engine = _engine_for(db_path)
        self._sessionmaker = async_sessionmaker(self._engine, expire_on_commit=False)
        # ``_init_lock`` guards schema init only. ``_write_lock`` serializes
        # transactional sections (claim_batch, submit) to avoid races on
        # idempotency_key uniqueness or on the claim → update sequence.
        self._init_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()
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

    async def submit(self, memory: Memory, *, ctx: SubmitContext) -> Memory:
        TenantScope.assert_matches(ctx.tenant_id)
        if len(memory.content.encode("utf-8")) > self.max_content_bytes:
            raise ValueError(
                f"memory content exceeds max_content_bytes={self.max_content_bytes}"
            )

        await self._ensure_initialized()
        async with self._write_lock, self._sessionmaker() as session:
            if memory.idempotency_key is not None:
                existing = (
                    await session.execute(
                        select(memories_table).where(
                            and_(
                                memories_table.c.tenant_id == ctx.tenant_id,
                                memories_table.c.idempotency_key
                                == memory.idempotency_key,
                            )
                        )
                    )
                ).mappings().first()
                if existing is not None:
                    return _row_to_memory(existing)

            persisted = memory.model_copy(
                update={
                    "id": memory.id or str(uuid4()),
                    "tenant_id": ctx.tenant_id,
                }
            )
            row = _memory_to_row(persisted)
            try:
                await session.execute(insert(memories_table).values(**row))
                await session.commit()
            except IntegrityError:
                # Lost a race on the unique idempotency_key: fetch the winner.
                await session.rollback()
                if persisted.idempotency_key is not None:
                    existing = (
                        await session.execute(
                            select(memories_table).where(
                                and_(
                                    memories_table.c.tenant_id == ctx.tenant_id,
                                    memories_table.c.idempotency_key
                                    == persisted.idempotency_key,
                                )
                            )
                        )
                    ).mappings().first()
                    if existing is not None:
                        return _row_to_memory(existing)
                raise
            return persisted

    async def list_unconsumed(self, *, ctx: ListUnconsumedContext) -> list[Memory]:
        TenantScope.assert_matches(ctx.tenant_id)
        async with self._session() as session:
            stmt = (
                select(memories_table)
                .where(
                    and_(
                        memories_table.c.tenant_id == ctx.tenant_id,
                        memories_table.c.consumed_at.is_(None),
                        memories_table.c.consumed_by_lease.is_(None),
                    )
                )
                .order_by(memories_table.c.submitted_at, memories_table.c.id)
            )
            if ctx.limit is not None:
                stmt = stmt.limit(ctx.limit)
            rows = (await session.execute(stmt)).mappings().all()
            return [_row_to_memory(r) for r in rows]

    async def claim_batch(self, *, ctx: ClaimContext) -> MemoryBatch:
        TenantScope.assert_matches(ctx.tenant_id)
        max_size = ctx.max_batch_size if ctx.max_batch_size is not None else self.max_batch_size
        await self._ensure_initialized()
        async with self._write_lock, self._sessionmaker() as session:
            # SELECT candidate ids then UPDATE — one logical transaction so
            # concurrent claim_batch calls receive disjoint id sets.
            select_stmt = (
                select(memories_table.c.id)
                .where(
                    and_(
                        memories_table.c.tenant_id == ctx.tenant_id,
                        memories_table.c.consumed_at.is_(None),
                        memories_table.c.consumed_by_lease.is_(None),
                    )
                )
                .order_by(memories_table.c.submitted_at, memories_table.c.id)
                .limit(max_size)
            )
            ids = [r[0] for r in (await session.execute(select_stmt)).all()]
            if ids:
                await session.execute(
                    update(memories_table)
                    .where(memories_table.c.id.in_(ids))
                    .values(consumed_by_lease=ctx.lease_id)
                )
            await session.commit()

            if not ids:
                return MemoryBatch(
                    lease_id=ctx.lease_id,
                    tenant_id=ctx.tenant_id,
                    memories=[],
                    snapshot_at=ctx.snapshot_at or _utcnow(),
                )
            rows = (
                await session.execute(
                    select(memories_table)
                    .where(memories_table.c.id.in_(ids))
                    .order_by(memories_table.c.submitted_at, memories_table.c.id)
                )
            ).mappings().all()
            memories = [_row_to_memory(r) for r in rows]
            return MemoryBatch(
                lease_id=ctx.lease_id,
                tenant_id=ctx.tenant_id,
                memories=memories,
                snapshot_at=ctx.snapshot_at or _utcnow(),
            )

    async def mark_consumed(self, *, ctx: MarkConsumedContext) -> None:
        TenantScope.assert_matches(ctx.tenant_id)
        consumed_at = ctx.consumed_at or _utcnow()
        async with self._session() as session:
            conditions = [
                memories_table.c.tenant_id == ctx.tenant_id,
                memories_table.c.consumed_by_lease == ctx.lease_id,
            ]
            if ctx.memory_ids:
                conditions.append(memories_table.c.id.in_(ctx.memory_ids))
            await session.execute(
                update(memories_table)
                .where(and_(*conditions))
                .values(consumed_at=consumed_at)
            )
            await session.commit()

    async def release_unconsumed(self, *, ctx: ReleaseContext) -> None:
        TenantScope.assert_matches(ctx.tenant_id)
        async with self._session() as session:
            await session.execute(
                update(memories_table)
                .where(
                    and_(
                        memories_table.c.tenant_id == ctx.tenant_id,
                        memories_table.c.consumed_by_lease == ctx.lease_id,
                        memories_table.c.consumed_at.is_(None),
                    )
                )
                .values(consumed_by_lease=None)
            )
            await session.commit()

    async def count_unconsumed(self, *, ctx: CountContext) -> int:
        TenantScope.assert_matches(ctx.tenant_id)
        async with self._session() as session:
            stmt = select(func.count()).select_from(memories_table).where(
                and_(
                    memories_table.c.tenant_id == ctx.tenant_id,
                    memories_table.c.consumed_at.is_(None),
                    memories_table.c.consumed_by_lease.is_(None),
                )
            )
            if ctx.exclude_types:
                stmt = stmt.where(memories_table.c.type.not_in(list(ctx.exclude_types)))
            return int((await session.execute(stmt)).scalar_one())

    async def release_for_expired_leases(self, *, ctx: ReclaimContext) -> int:
        TenantScope.assert_matches(ctx.tenant_id)
        if not ctx.expired_lease_ids:
            return 0
        async with self._session() as session:
            result = await session.execute(
                update(memories_table)
                .where(
                    and_(
                        memories_table.c.tenant_id == ctx.tenant_id,
                        memories_table.c.consumed_at.is_(None),
                        memories_table.c.consumed_by_lease.in_(
                            list(ctx.expired_lease_ids)
                        ),
                    )
                )
                .values(consumed_by_lease=None)
            )
            await session.commit()
            return _rowcount(result)

    async def purge_consumed(self, *, ctx: PurgeConsumedContext) -> int:
        TenantScope.assert_matches(ctx.tenant_id)
        async with self._session() as session:
            result = await session.execute(
                delete(memories_table).where(
                    and_(
                        memories_table.c.tenant_id == ctx.tenant_id,
                        memories_table.c.consumed_at.is_not(None),
                        memories_table.c.consumed_at < ctx.before,
                    )
                )
            )
            await session.commit()
            return _rowcount(result)


@implements(DreamLeaseStore, version=1)
class SQLiteDreamLeaseStore:
    """TTL-based SQLite dream lease store."""

    multi_tenant: ClassVar[bool] = False

    def __init__(
        self,
        *,
        db_path: str = "./dreamer.db",
        default_ttl_seconds: float = 1800.0,
    ) -> None:
        self.db_path = db_path
        self.default_ttl_seconds = float(default_ttl_seconds)
        self._engine = _engine_for(db_path)
        self._sessionmaker = async_sessionmaker(self._engine, expire_on_commit=False)
        self._init_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()
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

    async def acquire(self, *, ctx: AcquireLeaseContext) -> DreamLease | None:
        TenantScope.assert_matches(ctx.tenant_id)
        await self._ensure_initialized()
        async with self._write_lock, self._sessionmaker() as session:
            now = _utcnow()
            existing = (
                await session.execute(
                    select(leases_table).where(
                        and_(
                            leases_table.c.tenant_id == ctx.tenant_id,
                            leases_table.c.expires_at > now,
                        )
                    )
                )
            ).mappings().first()
            if existing is not None:
                return None
            ttl = ctx.ttl_seconds if ctx.ttl_seconds is not None else self.default_ttl_seconds
            lease = DreamLease(
                id=str(uuid4()),
                tenant_id=ctx.tenant_id,
                acquired_at=now,
                expires_at=now + timedelta(seconds=ttl),
            )
            await session.execute(
                insert(leases_table).values(
                    id=lease.id,
                    tenant_id=lease.tenant_id,
                    acquired_at=lease.acquired_at,
                    expires_at=lease.expires_at,
                )
            )
            await session.commit()
            return lease

    async def renew(self, *, ctx: RenewLeaseContext) -> bool:
        TenantScope.assert_matches(ctx.tenant_id)
        async with self._session() as session:
            now = _utcnow()
            lease = (
                await session.execute(
                    select(leases_table).where(
                        and_(
                            leases_table.c.id == ctx.lease_id,
                            leases_table.c.tenant_id == ctx.tenant_id,
                        )
                    )
                )
            ).mappings().first()
            if lease is None:
                return False
            if _ensure_aware(lease["expires_at"]) <= now:
                return False
            ttl = ctx.ttl_seconds if ctx.ttl_seconds is not None else self.default_ttl_seconds
            new_expires = now + timedelta(seconds=ttl)
            await session.execute(
                update(leases_table)
                .where(leases_table.c.id == ctx.lease_id)
                .values(expires_at=new_expires)
            )
            await session.commit()
            return True

    async def release(self, *, ctx: ReleaseLeaseContext) -> None:
        TenantScope.assert_matches(ctx.tenant_id)
        async with self._session() as session:
            await session.execute(
                delete(leases_table).where(
                    and_(
                        leases_table.c.id == ctx.lease_id,
                        leases_table.c.tenant_id == ctx.tenant_id,
                    )
                )
            )
            await session.commit()

    async def reclaim_expired(self, *, ctx: ReclaimLeasesContext) -> frozenset[str]:
        TenantScope.assert_matches(ctx.tenant_id)
        async with self._session() as session:
            now = _utcnow()
            ids = [
                row[0]
                for row in (
                    await session.execute(
                        select(leases_table.c.id).where(
                            and_(
                                leases_table.c.tenant_id == ctx.tenant_id,
                                leases_table.c.expires_at <= now,
                            )
                        )
                    )
                ).all()
            ]
            if ids:
                await session.execute(
                    delete(leases_table).where(leases_table.c.id.in_(ids))
                )
                await session.commit()
            return frozenset(ids)

    async def fast_forward(self, *, by: timedelta) -> None:
        """Test helper: shift every known lease backward in time."""
        async with self._session() as session:
            rows = (await session.execute(select(leases_table))).mappings().all()
            for row in rows:
                new_expires = _ensure_aware(row["expires_at"]) - by
                new_acquired = _ensure_aware(row["acquired_at"]) - by
                await session.execute(
                    update(leases_table)
                    .where(leases_table.c.id == row["id"])
                    .values(expires_at=new_expires, acquired_at=new_acquired)
                )
            await session.commit()


