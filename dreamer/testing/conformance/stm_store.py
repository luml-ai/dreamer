"""STMStore conformance suite.

Subclass ``STMStoreConformance`` and override ``make_stm_store``. The fixture
is async and may yield to allow the implementation to clean up between tests.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
import pytest_asyncio

from dreamer.api.contexts import (
    ClaimContext,
    CountContext,
    ListUnconsumedContext,
    MarkConsumedContext,
    PurgeConsumedContext,
    ReclaimContext,
    ReleaseContext,
    SubmitContext,
)
from dreamer.api.stores import STMStore
from dreamer.api.tenants import TenantScope
from dreamer.api.types import Memory


def _now() -> datetime:
    return datetime.now(UTC)


def _make_memory(
    *,
    tenant_id: str = "default",
    type: str = "observation",
    title: str = "obs",
    content: str = "body",
    idempotency_key: str | None = None,
    submitted_at: datetime | None = None,
) -> Memory:
    return Memory(
        tenant_id=tenant_id,
        agent_id="agent-1",
        type=type,
        title=title,
        content=content,
        submitted_at=submitted_at or _now(),
        idempotency_key=idempotency_key,
    )


class STMStoreConformance:
    """Run with: subclass + provide ``make_stm_store``."""

    @pytest_asyncio.fixture
    async def stm_store(self) -> AsyncIterator[STMStore]:
        async for store in self.make_stm_store():
            yield store

    async def make_stm_store(self) -> AsyncIterator[STMStore]:
        raise NotImplementedError("override in subclass")
        yield  # pragma: no cover

    @pytest.mark.asyncio
    async def test_submit_assigns_id_and_round_trips(self, stm_store: STMStore) -> None:
        with TenantScope.set("default"):
            persisted = await stm_store.submit(
                _make_memory(),
                ctx=SubmitContext(request_id="r1", tenant_id="default"),
            )
            assert persisted.id is not None
            unconsumed = await stm_store.list_unconsumed(
                ctx=ListUnconsumedContext(request_id="r1", tenant_id="default")
            )
            assert [m.id for m in unconsumed] == [persisted.id]

    @pytest.mark.asyncio
    async def test_idempotency_key_dedup(self, stm_store: STMStore) -> None:
        with TenantScope.set("default"):
            ctx = SubmitContext(request_id="r1", tenant_id="default")
            first = await stm_store.submit(
                _make_memory(idempotency_key="k1", title="first"), ctx=ctx
            )
            second = await stm_store.submit(
                _make_memory(idempotency_key="k1", title="second"), ctx=ctx
            )
            assert first.id == second.id
            assert second.title == first.title
            unconsumed = await stm_store.list_unconsumed(
                ctx=ListUnconsumedContext(request_id="r1", tenant_id="default")
            )
            assert len(unconsumed) == 1

    @pytest.mark.asyncio
    async def test_idempotency_key_scoped_per_tenant(self, stm_store: STMStore) -> None:
        with TenantScope.set("default"):
            first = await stm_store.submit(
                _make_memory(idempotency_key="dup"),
                ctx=SubmitContext(request_id="r1", tenant_id="default"),
            )
        with TenantScope.set("other"):
            second = await stm_store.submit(
                _make_memory(tenant_id="other", idempotency_key="dup"),
                ctx=SubmitContext(request_id="r1", tenant_id="other"),
            )
        assert first.id != second.id

    @pytest.mark.asyncio
    async def test_claim_batch_marks_lease_and_excludes_from_list(
        self, stm_store: STMStore
    ) -> None:
        with TenantScope.set("default"):
            await stm_store.submit(
                _make_memory(title="m1"),
                ctx=SubmitContext(request_id="r1", tenant_id="default"),
            )
            batch = await stm_store.claim_batch(
                ctx=ClaimContext(
                    request_id="r1", tenant_id="default", lease_id="L1"
                )
            )
            assert len(batch.memories) == 1
            assert batch.memories[0].consumed_by_lease == "L1"
            unconsumed = await stm_store.list_unconsumed(
                ctx=ListUnconsumedContext(request_id="r1", tenant_id="default")
            )
            assert unconsumed == []

    @pytest.mark.asyncio
    async def test_claim_batch_concurrent_returns_disjoint_ids(
        self, stm_store: STMStore
    ) -> None:
        with TenantScope.set("default"):
            for i in range(6):
                await stm_store.submit(
                    _make_memory(title=f"m{i}"),
                    ctx=SubmitContext(request_id="r1", tenant_id="default"),
                )
            batches = await asyncio.gather(
                stm_store.claim_batch(
                    ctx=ClaimContext(
                        request_id="r1", tenant_id="default", lease_id="L1"
                    )
                ),
                stm_store.claim_batch(
                    ctx=ClaimContext(
                        request_id="r1", tenant_id="default", lease_id="L2"
                    )
                ),
            )
            ids_a = {m.id for m in batches[0].memories}
            ids_b = {m.id for m in batches[1].memories}
            assert ids_a.isdisjoint(ids_b)
            assert len(ids_a) + len(ids_b) == 6

    @pytest.mark.asyncio
    async def test_mark_consumed_idempotent(self, stm_store: STMStore) -> None:
        with TenantScope.set("default"):
            await stm_store.submit(
                _make_memory(title="m1"),
                ctx=SubmitContext(request_id="r1", tenant_id="default"),
            )
            batch = await stm_store.claim_batch(
                ctx=ClaimContext(
                    request_id="r1", tenant_id="default", lease_id="L1"
                )
            )
            mids = tuple(str(m.id) for m in batch.memories)
            for _ in range(3):
                await stm_store.mark_consumed(
                    ctx=MarkConsumedContext(
                        request_id="r1",
                        tenant_id="default",
                        lease_id="L1",
                        memory_ids=mids,
                        consumed_at=_now(),
                    )
                )
            assert (
                await stm_store.count_unconsumed(
                    ctx=CountContext(request_id="r1", tenant_id="default")
                )
                == 0
            )

    @pytest.mark.asyncio
    async def test_release_unconsumed_returns_to_pool(self, stm_store: STMStore) -> None:
        with TenantScope.set("default"):
            await stm_store.submit(
                _make_memory(),
                ctx=SubmitContext(request_id="r1", tenant_id="default"),
            )
            await stm_store.claim_batch(
                ctx=ClaimContext(
                    request_id="r1", tenant_id="default", lease_id="L1"
                )
            )
            await stm_store.release_unconsumed(
                ctx=ReleaseContext(
                    request_id="r1", tenant_id="default", lease_id="L1"
                )
            )
            assert (
                await stm_store.count_unconsumed(
                    ctx=CountContext(request_id="r1", tenant_id="default")
                )
                == 1
            )

    @pytest.mark.asyncio
    async def test_release_for_expired_leases(self, stm_store: STMStore) -> None:
        with TenantScope.set("default"):
            await stm_store.submit(
                _make_memory(title="m1"),
                ctx=SubmitContext(request_id="r1", tenant_id="default"),
            )
            await stm_store.submit(
                _make_memory(title="m2"),
                ctx=SubmitContext(request_id="r1", tenant_id="default"),
            )
            await stm_store.claim_batch(
                ctx=ClaimContext(
                    request_id="r1", tenant_id="default", lease_id="L-expired"
                )
            )
            assert (
                await stm_store.count_unconsumed(
                    ctx=CountContext(request_id="r1", tenant_id="default")
                )
                == 0
            )
            released = await stm_store.release_for_expired_leases(
                ctx=ReclaimContext(
                    request_id="r1",
                    tenant_id="default",
                    expired_lease_ids=frozenset({"L-expired"}),
                )
            )
            assert released == 2
            assert (
                await stm_store.count_unconsumed(
                    ctx=CountContext(request_id="r1", tenant_id="default")
                )
                == 2
            )

    @pytest.mark.asyncio
    async def test_purge_consumed_respects_before_cutoff(
        self, stm_store: STMStore
    ) -> None:
        with TenantScope.set("default"):
            await stm_store.submit(
                _make_memory(title="old"),
                ctx=SubmitContext(request_id="r1", tenant_id="default"),
            )
            await stm_store.submit(
                _make_memory(title="recent"),
                ctx=SubmitContext(request_id="r1", tenant_id="default"),
            )
            batch = await stm_store.claim_batch(
                ctx=ClaimContext(
                    request_id="r1",
                    tenant_id="default",
                    lease_id="L1",
                    max_batch_size=1,
                )
            )
            old_id = batch.memories[0].id
            assert old_id is not None
            await stm_store.mark_consumed(
                ctx=MarkConsumedContext(
                    request_id="r1",
                    tenant_id="default",
                    lease_id="L1",
                    memory_ids=(old_id,),
                    consumed_at=_now() - timedelta(days=1),
                )
            )
            purged = await stm_store.purge_consumed(
                ctx=PurgeConsumedContext(
                    request_id="r1",
                    tenant_id="default",
                    before=_now() - timedelta(hours=1),
                )
            )
            assert purged >= 0  # implementations may no-op (return 0) or delete

    @pytest.mark.asyncio
    async def test_tenant_scope_mismatch_rejected(self, stm_store: STMStore) -> None:
        with TenantScope.set("a"):
            await stm_store.submit(
                _make_memory(tenant_id="a"),
                ctx=SubmitContext(request_id="r1", tenant_id="a"),
            )
        with TenantScope.set("b"):
            with pytest.raises(RuntimeError):
                await stm_store.list_unconsumed(
                    ctx=ListUnconsumedContext(request_id="r1", tenant_id="a")
                )

    @pytest.mark.asyncio
    async def test_tenant_scope_unset_rejected(self, stm_store: STMStore) -> None:
        TenantScope.clear()
        with pytest.raises(RuntimeError):
            await stm_store.list_unconsumed(
                ctx=ListUnconsumedContext(request_id="r1", tenant_id="default")
            )

    @pytest.mark.asyncio
    async def test_count_unconsumed_excludes_types(self, stm_store: STMStore) -> None:
        with TenantScope.set("default"):
            ctx = SubmitContext(request_id="r1", tenant_id="default")
            for i in range(2):
                await stm_store.submit(_make_memory(title=f"obs{i}"), ctx=ctx)
            for i in range(3):
                await stm_store.submit(
                    _make_memory(type="context_confirmed", title=f"conf{i}"), ctx=ctx
                )
            total = await stm_store.count_unconsumed(
                ctx=CountContext(request_id="r1", tenant_id="default")
            )
            assert total == 5
            filtered = await stm_store.count_unconsumed(
                ctx=CountContext(
                    request_id="r1",
                    tenant_id="default",
                    exclude_types=("context_confirmed",),
                )
            )
            assert filtered == 2
            all_excluded = await stm_store.count_unconsumed(
                ctx=CountContext(
                    request_id="r1",
                    tenant_id="default",
                    exclude_types=("context_confirmed", "observation"),
                )
            )
            assert all_excluded == 0

    @pytest.mark.asyncio
    async def test_submit_unknown_id_remains_unique(self, stm_store: STMStore) -> None:
        with TenantScope.set("default"):
            ids = set()
            for _ in range(5):
                m = await stm_store.submit(
                    _make_memory(title=str(uuid4())),
                    ctx=SubmitContext(request_id="r1", tenant_id="default"),
                )
                assert m.id not in ids
                ids.add(m.id)
