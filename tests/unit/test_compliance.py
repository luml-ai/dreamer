from __future__ import annotations

from datetime import datetime
from typing import Any, ClassVar

import pytest

from dreamer.api.compat import implements
from dreamer.api.contexts import (
    ClaimContext,
    CountContext,
    DiscardWorkspaceContext,
    ListUnconsumedContext,
    MarkConsumedContext,
    OpenWorkspaceContext,
    PostDreamContext,
    PostDreamServices,
    PurgeConsumedContext,
    ReclaimContext,
    ReleaseContext,
    SubmitContext,
)
from dreamer.api.errors import ProtocolComplianceError
from dreamer.api.hooks import PostDreamHook
from dreamer.api.stores import LTMStore, STMStore
from dreamer.api.types import (
    Diff,
    FileViewable,
    Memory,
    MemoryBatch,
    Workspace,
)
from dreamer.server.compliance import (
    SlotBinding,
    check_components,
)


@implements(STMStore, version=1)
class GoodSTMStore:
    multi_tenant: ClassVar[bool] = False

    async def submit(self, memory: Memory, *, ctx: SubmitContext) -> Memory:
        return memory

    async def list_unconsumed(self, *, ctx: ListUnconsumedContext) -> list[Memory]:
        return []

    async def claim_batch(self, *, ctx: ClaimContext) -> MemoryBatch:
        return MemoryBatch(
            lease_id="L",
            tenant_id=ctx.tenant_id,
            memories=[],
            snapshot_at=datetime.now(),
        )

    async def mark_consumed(self, *, ctx: MarkConsumedContext) -> None:
        return None

    async def release_unconsumed(self, *, ctx: ReleaseContext) -> None:
        return None

    async def count_unconsumed(self, *, ctx: CountContext) -> int:
        return 0

    async def release_for_expired_leases(self, *, ctx: ReclaimContext) -> int:
        return 0

    async def purge_consumed(self, *, ctx: PurgeConsumedContext) -> int:
        return 0


@implements(LTMStore, version=1)
class GoodLTMStore:
    multi_tenant: ClassVar[bool] = True
    workspace_capabilities: ClassVar[frozenset[type]] = frozenset({FileViewable})

    async def open_workspace(self, *, ctx: OpenWorkspaceContext) -> Workspace:
        raise NotImplementedError

    async def commit_workspace(self, ws: Workspace, *, ctx: Any) -> Diff:
        return Diff()

    async def discard_workspace(self, ws: Workspace, *, ctx: DiscardWorkspaceContext) -> None:
        return None


@implements(PostDreamHook, version=1)
class GoodPostDreamHook:
    multi_tenant: ClassVar[bool] = False

    async def on_post_dream(
        self, *, ctx: PostDreamContext, services: PostDreamServices
    ) -> None:
        return None


@implements(PostDreamHook, version=1)
class HookMissingServices:
    multi_tenant: ClassVar[bool] = False

    async def on_post_dream(self, *, ctx: PostDreamContext) -> None:
        return None


@implements(PostDreamHook, version=99)
class HookBadVersion:
    multi_tenant: ClassVar[bool] = False

    async def on_post_dream(
        self, *, ctx: PostDreamContext, services: PostDreamServices
    ) -> None:
        return None


class HookNoImplements:
    multi_tenant: ClassVar[bool] = False

    async def on_post_dream(
        self, *, ctx: PostDreamContext, services: PostDreamServices
    ) -> None:
        return None


def test_compliant_component_passes() -> None:
    bindings = [
        SlotBinding(slot="stm_store", component=GoodSTMStore(), expected_protocols=(STMStore,)),
    ]
    report = check_components(bindings, declared_mode="auto")
    assert report.ok
    assert report.errors == ()
    assert report.effective_multi_tenant is False


def test_missing_implements_declaration() -> None:
    hook = HookNoImplements()
    bindings = [
        SlotBinding(
            slot="hooks.post_dream[0]",
            component=hook,
            expected_protocols=(PostDreamHook,),
        )
    ]
    report = check_components(bindings)
    assert not report.ok
    assert any("does not declare" in e for e in report.errors)


def test_signature_mismatch_caught() -> None:
    hook = HookMissingServices()
    bindings = [
        SlotBinding(
            slot="hooks.post_dream[0]",
            component=hook,
            expected_protocols=(PostDreamHook,),
        )
    ]
    report = check_components(bindings)
    assert not report.ok
    assert any("signature mismatch" in e or "parameter names differ" in e
               for e in report.errors)


def test_unsupported_version_rejected() -> None:
    hook = HookBadVersion()
    bindings = [
        SlotBinding(
            slot="hooks.post_dream[0]",
            component=hook,
            expected_protocols=(PostDreamHook,),
        )
    ]
    report = check_components(bindings)
    assert not report.ok
    assert any("version 99" in e for e in report.errors)


def test_mt_required_with_non_mt_component_fails() -> None:
    bindings = [
        SlotBinding(slot="stm_store", component=GoodSTMStore(), expected_protocols=(STMStore,)),
    ]
    report = check_components(bindings, declared_mode="required")
    assert not report.ok
    assert any("multi_tenancy: required" in e for e in report.errors)


def test_mt_required_with_all_mt_passes() -> None:
    bindings = [
        SlotBinding(slot="ltm_store", component=GoodLTMStore(), expected_protocols=(LTMStore,)),
    ]
    report = check_components(bindings, declared_mode="required")
    assert report.ok
    assert report.effective_multi_tenant is True


def test_mt_forbidden_with_all_mt_fails() -> None:
    bindings = [
        SlotBinding(slot="ltm_store", component=GoodLTMStore(), expected_protocols=(LTMStore,)),
    ]
    report = check_components(bindings, declared_mode="forbidden")
    assert not report.ok
    assert any("multi_tenancy: forbidden" in e for e in report.errors)


def test_mt_table_marks_offenders() -> None:
    bindings = [
        SlotBinding(slot="stm_store", component=GoodSTMStore(), expected_protocols=(STMStore,)),
        SlotBinding(slot="ltm_store", component=GoodLTMStore(), expected_protocols=(LTMStore,)),
    ]
    report = check_components(bindings)
    by_slot = {e.slot: e for e in report.mt_table}
    assert by_slot["stm_store"].multi_tenant is False
    assert by_slot["ltm_store"].multi_tenant is True
    assert report.effective_multi_tenant is False


def test_compliance_report_raise_if_failed() -> None:
    bindings = [
        SlotBinding(slot="hooks.post_dream[0]",
                    component=HookNoImplements(),
                    expected_protocols=(PostDreamHook,)),
    ]
    report = check_components(bindings)
    with pytest.raises(ProtocolComplianceError):
        report.raise_if_failed()


def test_empty_bindings_returns_passing_report() -> None:
    report = check_components([])
    assert report.ok
    assert report.mt_table == ()
    assert report.effective_multi_tenant is False
