from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest_asyncio

from dreamer.api.stores import ContextStore, LTMStore
from dreamer.api.tenants import TenantScope
from dreamer.contrib.context.markdown import MarkdownContextStore
from dreamer.contrib.ltm.markdown import MarkdownLTMStore
from dreamer.testing.conformance import (
    ContextStoreConformance,
    LTMStoreConformance,
)


@pytest_asyncio.fixture(autouse=True)
async def _clear_tenant_scope() -> AsyncIterator[None]:
    TenantScope.clear()
    yield
    TenantScope.clear()


class TestMarkdownLTMStoreConformance(LTMStoreConformance):
    @pytest_asyncio.fixture(autouse=True)
    async def _seed_tmp_path(self, tmp_path: Path) -> AsyncIterator[None]:
        self.tmp_path = tmp_path
        yield

    async def make_ltm_store(self) -> AsyncIterator[LTMStore]:
        yield MarkdownLTMStore(root=self.tmp_path / "memory")


class TestMarkdownContextStoreConformance(ContextStoreConformance):
    @pytest_asyncio.fixture(autouse=True)
    async def _seed_tmp_path(self, tmp_path: Path) -> AsyncIterator[None]:
        self.tmp_path = tmp_path
        yield

    async def make_context_store(self) -> AsyncIterator[ContextStore]:
        yield MarkdownContextStore(root=self.tmp_path / "context")
