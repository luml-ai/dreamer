"""LTMStore conformance suite.

Verifies workspace open/commit/discard semantics, the ``Diff`` shape, the
optional ``ContextPendingStore`` capability, and tenant scope enforcement.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio

from dreamer.api.contexts import (
    ClearContextPendingContext,
    CommitWorkspaceContext,
    DiscardWorkspaceContext,
    GetContextPendingContext,
    OpenWorkspaceContext,
    SetContextPendingContext,
)
from dreamer.api.stores import ContextPendingStore, LTMStore
from dreamer.api.tenants import TenantScope
from dreamer.api.types import Diff, FileViewable


class LTMStoreConformance:
    @pytest_asyncio.fixture
    async def ltm_store(self) -> AsyncIterator[LTMStore]:
        async for store in self.make_ltm_store():
            yield store

    async def make_ltm_store(self) -> AsyncIterator[LTMStore]:
        raise NotImplementedError("override in subclass")
        yield  # pragma: no cover

    @pytest.mark.asyncio
    async def test_workspace_capabilities_declared(self, ltm_store: LTMStore) -> None:
        caps = ltm_store.workspace_capabilities
        assert isinstance(caps, frozenset)
        assert len(caps) >= 1

    @pytest.mark.asyncio
    async def test_open_returns_file_view_when_advertised(
        self, ltm_store: LTMStore
    ) -> None:
        if FileViewable not in ltm_store.workspace_capabilities:
            pytest.skip("FileViewable not declared")
        with TenantScope.set("default"):
            ws = await ltm_store.open_workspace(
                ctx=OpenWorkspaceContext(request_id="r1", tenant_id="default")
            )
            assert isinstance(ws, FileViewable)
            view = await ws.file_view()
            assert view.exists()
            await ltm_store.discard_workspace(
                ws,
                ctx=DiscardWorkspaceContext(request_id="r1", tenant_id="default"),
            )

    @pytest.mark.asyncio
    async def test_commit_returns_diff(self, ltm_store: LTMStore) -> None:
        if FileViewable not in ltm_store.workspace_capabilities:
            pytest.skip("FileViewable not declared")
        with TenantScope.set("default"):
            ws = await ltm_store.open_workspace(
                ctx=OpenWorkspaceContext(request_id="r1", tenant_id="default")
            )
            assert isinstance(ws, FileViewable)
            view = await ws.file_view()
            (view / "note.md").write_text("hello\n", encoding="utf-8")
            diff = await ltm_store.commit_workspace(
                ws, ctx=CommitWorkspaceContext(request_id="r1", tenant_id="default")
            )
            assert isinstance(diff, Diff)
            assert "note.md" in diff.added

    @pytest.mark.asyncio
    async def test_discard_after_commit_is_safe(self, ltm_store: LTMStore) -> None:
        if FileViewable not in ltm_store.workspace_capabilities:
            pytest.skip("FileViewable not declared")
        with TenantScope.set("default"):
            ws = await ltm_store.open_workspace(
                ctx=OpenWorkspaceContext(request_id="r1", tenant_id="default")
            )
            await ltm_store.commit_workspace(
                ws, ctx=CommitWorkspaceContext(request_id="r1", tenant_id="default")
            )
            # Discard after commit should not raise; impls that already cleaned up
            # the workspace MUST tolerate the second call.
            await ltm_store.discard_workspace(
                ws,
                ctx=DiscardWorkspaceContext(request_id="r1", tenant_id="default"),
            )

    @pytest.mark.asyncio
    async def test_context_pending_round_trip_when_supported(
        self, ltm_store: LTMStore
    ) -> None:
        if not isinstance(ltm_store, ContextPendingStore):
            pytest.skip("LTMStore does not implement ContextPendingStore")
        with TenantScope.set("default"):
            assert (
                await ltm_store.get_context_pending(
                    ctx=GetContextPendingContext(request_id="r1", tenant_id="default")
                )
                is None
            )
            sample = Diff(added=["x.md"])
            await ltm_store.set_context_pending(
                sample,
                ctx=SetContextPendingContext(request_id="r1", tenant_id="default"),
            )
            current = await ltm_store.get_context_pending(
                ctx=GetContextPendingContext(request_id="r1", tenant_id="default")
            )
            assert current is not None
            assert current.added == sample.added
            await ltm_store.clear_context_pending(
                ctx=ClearContextPendingContext(
                    request_id="r1", tenant_id="default"
                )
            )
            assert (
                await ltm_store.get_context_pending(
                    ctx=GetContextPendingContext(request_id="r1", tenant_id="default")
                )
                is None
            )

    @pytest.mark.asyncio
    async def test_tenant_scope_mismatch_rejected(self, ltm_store: LTMStore) -> None:
        with TenantScope.set("a"):
            with pytest.raises(RuntimeError):
                await ltm_store.open_workspace(
                    ctx=OpenWorkspaceContext(request_id="r1", tenant_id="b")
                )
