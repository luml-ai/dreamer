from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import timedelta
from pathlib import Path

import pytest_asyncio

from dreamer.api.stores import DreamLeaseStore, STMStore
from dreamer.api.tenants import TenantScope
from dreamer.contrib.stm.sqlite import (
    SQLiteDreamLeaseStore,
    SQLiteSTMStore,
    _engines,
)
from dreamer.testing.conformance import (
    DreamLeaseStoreConformance,
    STMStoreConformance,
)


@pytest_asyncio.fixture(autouse=True)
async def _clear_tenant_scope() -> AsyncIterator[None]:
    TenantScope.clear()
    yield
    TenantScope.clear()


async def _dispose_engines() -> None:
    for engine in list(_engines.values()):
        await engine.dispose()
    _engines.clear()


class TestSQLiteSTMStoreConformance(STMStoreConformance):
    @pytest_asyncio.fixture(autouse=True)
    async def _seed_tmp_path(self, tmp_path: Path) -> AsyncIterator[None]:
        self.tmp_path = tmp_path
        yield
        await _dispose_engines()

    async def make_stm_store(self) -> AsyncIterator[STMStore]:
        db = self.tmp_path / "stm.db"
        yield SQLiteSTMStore(db_path=str(db))


class TestSQLiteDreamLeaseStoreConformance(DreamLeaseStoreConformance):
    @pytest_asyncio.fixture(autouse=True)
    async def _seed_tmp_path(self, tmp_path: Path) -> AsyncIterator[None]:
        self.tmp_path = tmp_path
        yield
        await _dispose_engines()

    async def make_lease_store(self) -> AsyncIterator[DreamLeaseStore]:
        db = self.tmp_path / "leases.db"
        yield SQLiteDreamLeaseStore(db_path=str(db), default_ttl_seconds=60.0)

    async def fast_forward(
        self, store: DreamLeaseStore, *, by: timedelta
    ) -> None:
        assert isinstance(store, SQLiteDreamLeaseStore)
        await store.fast_forward(by=by)
