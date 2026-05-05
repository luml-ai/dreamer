"""DreamLeaseStore conformance suite."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import timedelta

import pytest
import pytest_asyncio

from dreamer.api.contexts import (
    AcquireLeaseContext,
    ReclaimLeasesContext,
    ReleaseLeaseContext,
    RenewLeaseContext,
)
from dreamer.api.stores import DreamLeaseStore
from dreamer.api.tenants import TenantScope


class DreamLeaseStoreConformance:
    @pytest_asyncio.fixture
    async def lease_store(self) -> AsyncIterator[DreamLeaseStore]:
        async for store in self.make_lease_store():
            yield store

    async def make_lease_store(self) -> AsyncIterator[DreamLeaseStore]:
        raise NotImplementedError("override in subclass")
        yield  # pragma: no cover

    async def fast_forward(self, store: DreamLeaseStore, *, by: timedelta) -> None:
        """Hook for impls that can simulate elapsed time without sleeping.

        Default: no-op (the test uses a tiny ttl + ``await asyncio.sleep``).
        Override to make ``by`` instantaneous.
        """

    @pytest.mark.asyncio
    async def test_acquire_then_release(self, lease_store: DreamLeaseStore) -> None:
        with TenantScope.set("default"):
            lease = await lease_store.acquire(
                ctx=AcquireLeaseContext(
                    request_id="r1", tenant_id="default", ttl_seconds=60
                )
            )
            assert lease is not None
            await lease_store.release(
                ctx=ReleaseLeaseContext(
                    request_id="r1", tenant_id="default", lease_id=lease.id
                )
            )
            again = await lease_store.acquire(
                ctx=AcquireLeaseContext(
                    request_id="r1", tenant_id="default", ttl_seconds=60
                )
            )
            assert again is not None

    @pytest.mark.asyncio
    async def test_concurrent_acquire_only_one_wins(
        self, lease_store: DreamLeaseStore
    ) -> None:
        with TenantScope.set("default"):
            results = await asyncio.gather(
                lease_store.acquire(
                    ctx=AcquireLeaseContext(
                        request_id="r1", tenant_id="default", ttl_seconds=60
                    )
                ),
                lease_store.acquire(
                    ctx=AcquireLeaseContext(
                        request_id="r1", tenant_id="default", ttl_seconds=60
                    )
                ),
            )
            success = [r for r in results if r is not None]
            assert len(success) == 1

    @pytest.mark.asyncio
    async def test_renew_extends_lease(self, lease_store: DreamLeaseStore) -> None:
        with TenantScope.set("default"):
            lease = await lease_store.acquire(
                ctx=AcquireLeaseContext(
                    request_id="r1", tenant_id="default", ttl_seconds=60
                )
            )
            assert lease is not None
            ok = await lease_store.renew(
                ctx=RenewLeaseContext(
                    request_id="r1",
                    tenant_id="default",
                    lease_id=lease.id,
                    ttl_seconds=60,
                )
            )
            assert ok is True

    @pytest.mark.asyncio
    async def test_reclaim_expired(self, lease_store: DreamLeaseStore) -> None:
        with TenantScope.set("default"):
            lease = await lease_store.acquire(
                ctx=AcquireLeaseContext(
                    request_id="r1", tenant_id="default", ttl_seconds=0.05
                )
            )
            assert lease is not None
            await asyncio.sleep(0.08)
            await self.fast_forward(lease_store, by=timedelta(seconds=1))
            reclaimed = await lease_store.reclaim_expired(
                ctx=ReclaimLeasesContext(request_id="r1", tenant_id="default")
            )
            assert isinstance(reclaimed, frozenset)
            assert len(reclaimed) >= 1
            assert lease.id in reclaimed

    @pytest.mark.asyncio
    async def test_renew_unknown_lease_returns_false(
        self, lease_store: DreamLeaseStore
    ) -> None:
        with TenantScope.set("default"):
            ok = await lease_store.renew(
                ctx=RenewLeaseContext(
                    request_id="r1",
                    tenant_id="default",
                    lease_id="bogus",
                    ttl_seconds=60,
                )
            )
            assert ok is False

    @pytest.mark.asyncio
    async def test_tenant_scope_mismatch_rejected(
        self, lease_store: DreamLeaseStore
    ) -> None:
        with TenantScope.set("a"):
            await lease_store.acquire(
                ctx=AcquireLeaseContext(
                    request_id="r1", tenant_id="a", ttl_seconds=60
                )
            )
        with TenantScope.set("b"):
            with pytest.raises(RuntimeError):
                await lease_store.acquire(
                    ctx=AcquireLeaseContext(
                        request_id="r1", tenant_id="a", ttl_seconds=60
                    )
                )
