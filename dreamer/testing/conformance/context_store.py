"""ContextStore conformance suite.

Mirrors the LTMStore workspace contract minus the ``ContextPendingStore``
capability (watermarks live on the LTM side).
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio

from dreamer.api.contexts import (
    CommitWorkspaceContext,
    DiscardWorkspaceContext,
    OpenWorkspaceContext,
)
from dreamer.api.stores import ContextStore
from dreamer.api.tenants import TenantScope
from dreamer.api.types import Diff, FileViewable


class ContextStoreConformance:
    @pytest_asyncio.fixture
    async def context_store(self) -> AsyncIterator[ContextStore]:
        async for store in self.make_context_store():
            yield store

    async def make_context_store(self) -> AsyncIterator[ContextStore]:
        raise NotImplementedError("override in subclass")
        yield  # pragma: no cover

    @pytest.mark.asyncio
    async def test_workspace_capabilities_declared(
        self, context_store: ContextStore
    ) -> None:
        caps = context_store.workspace_capabilities
        assert isinstance(caps, frozenset)
        assert len(caps) >= 1

    @pytest.mark.asyncio
    async def test_open_commit_discard_round_trip(
        self, context_store: ContextStore
    ) -> None:
        if FileViewable not in context_store.workspace_capabilities:
            pytest.skip("FileViewable not declared")
        with TenantScope.set("default"):
            ws = await context_store.open_workspace(
                ctx=OpenWorkspaceContext(request_id="r1", tenant_id="default")
            )
            assert isinstance(ws, FileViewable)
            view = await ws.file_view()
            (view / "AGENTS.md").write_text("hello\n", encoding="utf-8")
            diff = await context_store.commit_workspace(
                ws,
                ctx=CommitWorkspaceContext(request_id="r1", tenant_id="default"),
            )
            assert isinstance(diff, Diff)
            assert "AGENTS.md" in diff.added
            ws2 = await context_store.open_workspace(
                ctx=OpenWorkspaceContext(request_id="r1", tenant_id="default")
            )
            await context_store.discard_workspace(
                ws2,
                ctx=DiscardWorkspaceContext(request_id="r1", tenant_id="default"),
            )

    @pytest.mark.asyncio
    async def test_modified_file_appears_in_diff(
        self, context_store: ContextStore
    ) -> None:
        if FileViewable not in context_store.workspace_capabilities:
            pytest.skip("FileViewable not declared")
        with TenantScope.set("default"):
            first = await context_store.open_workspace(
                ctx=OpenWorkspaceContext(request_id="r1", tenant_id="default")
            )
            assert isinstance(first, FileViewable)
            view = await first.file_view()
            (view / "AGENTS.md").write_text("v1\n", encoding="utf-8")
            await context_store.commit_workspace(
                first,
                ctx=CommitWorkspaceContext(request_id="r1", tenant_id="default"),
            )
            second = await context_store.open_workspace(
                ctx=OpenWorkspaceContext(request_id="r1", tenant_id="default")
            )
            assert isinstance(second, FileViewable)
            view2 = await second.file_view()
            (view2 / "AGENTS.md").write_text("v2\n", encoding="utf-8")
            diff = await context_store.commit_workspace(
                second,
                ctx=CommitWorkspaceContext(request_id="r1", tenant_id="default"),
            )
            assert "AGENTS.md" in diff.modified

    @pytest.mark.asyncio
    async def test_tenant_scope_mismatch_rejected(
        self, context_store: ContextStore
    ) -> None:
        with TenantScope.set("a"):
            with pytest.raises(RuntimeError):
                await context_store.open_workspace(
                    ctx=OpenWorkspaceContext(request_id="r1", tenant_id="b")
                )
