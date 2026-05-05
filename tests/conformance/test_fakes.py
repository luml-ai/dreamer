from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import timedelta

import pytest_asyncio

from dreamer.api.stores import (
    ContextStore,
    DreamLeaseStore,
    LTMStore,
    STMSerializer,
    STMStore,
)
from dreamer.api.tenants import TenantScope
from dreamer.testing.conformance import (
    ContextStoreConformance,
    DreamLeaseStoreConformance,
    LTMStoreConformance,
    STMSerializerConformance,
    STMStoreConformance,
)
from dreamer.testing.fakes import (
    InMemoryContextStore,
    InMemoryDreamLeaseStore,
    InMemoryLTMStore,
    InMemorySTMSerializer,
    InMemorySTMStore,
)


@pytest_asyncio.fixture(autouse=True)
async def _clear_tenant_scope() -> AsyncIterator[None]:
    TenantScope.clear()
    yield
    TenantScope.clear()


class TestInMemorySTMStoreConformance(STMStoreConformance):
    async def make_stm_store(self) -> AsyncIterator[STMStore]:
        yield InMemorySTMStore()


class TestInMemoryDreamLeaseStoreConformance(DreamLeaseStoreConformance):
    async def make_lease_store(self) -> AsyncIterator[DreamLeaseStore]:
        yield InMemoryDreamLeaseStore(default_ttl_seconds=60)

    async def fast_forward(
        self, store: DreamLeaseStore, *, by: timedelta
    ) -> None:
        assert isinstance(store, InMemoryDreamLeaseStore)
        for lease_id, lease in list(store._leases.items()):
            store._leases[lease_id] = lease.model_copy(
                update={"expires_at": lease.expires_at - by}
            )


class TestInMemoryLTMStoreConformance(LTMStoreConformance):
    async def make_ltm_store(self) -> AsyncIterator[LTMStore]:
        yield InMemoryLTMStore()


class TestInMemoryContextStoreConformance(ContextStoreConformance):
    async def make_context_store(self) -> AsyncIterator[ContextStore]:
        yield InMemoryContextStore()


class TestInMemorySTMSerializerConformance(STMSerializerConformance):
    async def make_stm_serializer(self) -> AsyncIterator[STMSerializer]:
        yield InMemorySTMSerializer()
