"""STMSerializer conformance suite."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
import pytest_asyncio

from dreamer.api.contexts import SerializeContext, SerializeServices
from dreamer.api.stores import STMSerializer
from dreamer.api.tenants import TenantScope
from dreamer.api.types import Memory, MemoryBatch


class STMSerializerConformance:
    @pytest_asyncio.fixture
    async def stm_serializer(self) -> AsyncIterator[STMSerializer]:
        async for s in self.make_stm_serializer():
            yield s

    async def make_stm_serializer(self) -> AsyncIterator[STMSerializer]:
        raise NotImplementedError("override in subclass")
        yield  # pragma: no cover

    def make_serialize_services(self) -> SerializeServices:
        """Override if the serializer requires non-default services."""
        from dreamer.testing.fakes import (
            CollectingAuditSink,
            CollectingUsageSink,
        )

        async def emit(_msg: str, _payload: object) -> None:
            return None

        async def secret_get(*args: object, **kwargs: object) -> object:
            raise NotImplementedError

        class _Resolver:
            multi_tenant = True

            async def get(self, name: str, *, tenant_id: object, ctx: object) -> object:
                return await secret_get(name, tenant_id=tenant_id, ctx=ctx)

        return SerializeServices(
            emit_progress=emit,
            secrets=_Resolver(),  # type: ignore[arg-type]
            usage=CollectingUsageSink(),
            audit=CollectingAuditSink(),
            clock=lambda: datetime.now(UTC),
        )

    @pytest.mark.asyncio
    async def test_kind_declared(self, stm_serializer: STMSerializer) -> None:
        assert isinstance(stm_serializer.kind, str)
        assert stm_serializer.kind

    @pytest.mark.asyncio
    async def test_write_creates_files_under_target(
        self, stm_serializer: STMSerializer, tmp_path: Path
    ) -> None:
        target = tmp_path / "inbox"
        with TenantScope.set("default"):
            batch = _make_batch()
            await stm_serializer.write(
                batch,
                target=target,
                ctx=SerializeContext(
                    request_id="r1", tenant_id="default", lease_id="L1"
                ),
                services=self.make_serialize_services(),
            )
            assert any(target.iterdir())

    @pytest.mark.asyncio
    async def test_prompt_fragment_returns_string(
        self, stm_serializer: STMSerializer
    ) -> None:
        fragment = stm_serializer.prompt_fragment(_make_batch())
        assert isinstance(fragment, str)
        assert fragment.strip()


def _make_batch() -> MemoryBatch:
    now = datetime.now(UTC)
    memory = Memory(
        id="m1",
        tenant_id="default",
        agent_id="agent-1",
        type="observation",
        title="example",
        content="body",
        submitted_at=now,
    )
    return MemoryBatch(
        lease_id="L1",
        tenant_id="default",
        memories=[memory],
        snapshot_at=now,
    )
