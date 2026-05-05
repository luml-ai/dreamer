from __future__ import annotations

from dreamer.api.audit import AuditSink
from dreamer.api.auth import AuthBackend, Tenancy
from dreamer.api.capabilities import Lifecycle, Middlewares, Routes, Transactional
from dreamer.api.compat import SUPPORTED_PROTOCOL_VERSIONS, implements
from dreamer.api.dream import ContextPhaseRunner, DreamGate, LTMPhaseRunner
from dreamer.api.hooks import (
    DreamFailedHook,
    DreamProgressHook,
    PostContextUpdateHook,
    PostDreamHook,
    PostLTMUpdateHook,
    PostMemorySubmitHook,
    PreContextUpdateHook,
    PreDreamHook,
    PreLTMUpdateHook,
    PreMemorySubmitHook,
)
from dreamer.api.jobs import JobQueue
from dreamer.api.rate_limit import RateLimiter
from dreamer.api.secrets import SecretResolver, SecretRotationHook
from dreamer.api.stores import (
    ContextPendingStore,
    ContextReader,
    ContextStore,
    DreamLeaseStore,
    LTMStore,
    MCPTool,
    STMSerializer,
    STMStore,
)
from dreamer.api.tenants import TenantConfigProvider, TenantData, TenantLifecycle, TenantRegistry
from dreamer.api.triggers import Trigger
from dreamer.api.types import FileViewable, GraphViewable, RecordViewable, Workspace
from dreamer.api.usage import UsageSink


def test_implements_records_protocol_and_version() -> None:
    @implements(STMStore, version=1)
    class FakeStore:
        pass

    assert STMStore in FakeStore.__dreamer_protocols__
    assert FakeStore.__dreamer_protocols__[STMStore] == 1


def test_implements_accumulates_across_decorators() -> None:
    @implements(LTMStore, version=1)
    @implements(ContextPendingStore, version=1)
    class FakeStore:
        pass

    assert LTMStore in FakeStore.__dreamer_protocols__
    assert ContextPendingStore in FakeStore.__dreamer_protocols__
    assert FakeStore.__dreamer_protocols__[LTMStore] == 1
    assert FakeStore.__dreamer_protocols__[ContextPendingStore] == 1


def test_implements_does_not_pollute_unrelated_classes() -> None:
    @implements(STMStore, version=1)
    class A:
        pass

    class B:
        pass

    assert hasattr(A, "__dreamer_protocols__")
    assert not hasattr(B, "__dreamer_protocols__")


def test_implements_subclass_does_not_share_dict_object() -> None:
    """Decorating a parent and then a child should not retroactively mutate the
    parent's `__dreamer_protocols__` (otherwise sibling classes would all see
    each other's declarations)."""

    @implements(STMStore, version=1)
    class Parent:
        pass

    @implements(LTMStore, version=1)
    class Child(Parent):
        pass

    assert STMStore in Parent.__dreamer_protocols__
    assert LTMStore not in Parent.__dreamer_protocols__
    assert STMStore in Child.__dreamer_protocols__
    assert LTMStore in Child.__dreamer_protocols__


def test_supported_protocol_versions_covers_every_protocol() -> None:
    expected = {
        AuthBackend,
        Tenancy,
        STMStore,
        LTMStore,
        ContextPendingStore,
        ContextStore,
        DreamLeaseStore,
        LTMPhaseRunner,
        ContextPhaseRunner,
        Trigger,
        PreMemorySubmitHook,
        PostMemorySubmitHook,
        PreDreamHook,
        PostDreamHook,
        PreLTMUpdateHook,
        PostLTMUpdateHook,
        PreContextUpdateHook,
        PostContextUpdateHook,
        DreamFailedHook,
        Lifecycle,
        Routes,
        Middlewares,
        Transactional,
        Workspace,
        FileViewable,
        RecordViewable,
        GraphViewable,
        TenantRegistry,
        TenantConfigProvider,
        TenantLifecycle,
        TenantData,
        JobQueue,
        SecretResolver,
        SecretRotationHook,
        UsageSink,
        AuditSink,
        RateLimiter,
        MCPTool,
        STMSerializer,
        ContextReader,
        DreamGate,
        DreamProgressHook,
    }
    assert set(SUPPORTED_PROTOCOL_VERSIONS.keys()) == expected
    for proto, versions in SUPPORTED_PROTOCOL_VERSIONS.items():
        assert versions == frozenset({1}), f"{proto} should be at v1"
