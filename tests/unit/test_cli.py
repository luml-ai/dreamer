from __future__ import annotations

import sys
import textwrap
import types
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Literal

import pytest
from click.testing import CliRunner

from dreamer.api.auth import AuthBackend, Tenancy
from dreamer.api.compat import implements
from dreamer.api.contexts import (
    AcquireLeaseContext,
    AuthContext,
    ClaimContext,
    CommitWorkspaceContext,
    ContextPhaseContext,
    ContextPhaseServices,
    CountContext,
    DeprovisionContext,
    DiscardWorkspaceContext,
    ListUnconsumedContext,
    LTMPhaseContext,
    LTMPhaseServices,
    MarkConsumedContext,
    OpenWorkspaceContext,
    ProvisionContext,
    PublishContext,
    PurgeConsumedContext,
    RateLimitContext,
    ReclaimContext,
    ReclaimLeasesContext,
    ReleaseContext,
    ReleaseLeaseContext,
    RenewLeaseContext,
    ResetContext,
    SecretContext,
    SerializeContext,
    SerializeServices,
    SubmitContext,
    SubscribeContext,
    TenancyContext,
    TenantConfigLookupContext,
    TenantRegistryContext,
)
from dreamer.api.dream import ContextPhaseRunner, LTMPhaseRunner
from dreamer.api.jobs import JobQueue
from dreamer.api.rate_limit import RateLimiter
from dreamer.api.secrets import SecretResolver
from dreamer.api.stores import (
    ContextStore,
    DreamLeaseStore,
    LTMStore,
    STMSerializer,
    STMStore,
)
from dreamer.api.tenants import (
    TenantConfigProvider,
    TenantLifecycle,
    TenantRegistry,
)
from dreamer.api.types import (
    Diff,
    DreamLease,
    FileViewable,
    Memory,
    MemoryBatch,
    Principal,
    RateLimitDecision,
    SecretValue,
    TenantConfig,
    TenantId,
    Workspace,
)
from dreamer.cli.main import main


@implements(AuthBackend, version=1)
class FakeAuth:
    multi_tenant: ClassVar[bool] = False

    async def authenticate(self, request: Any, *, ctx: AuthContext) -> Principal:
        return Principal(id="anon")


@implements(Tenancy, version=1)
class FakeTenancy:
    multi_tenant: ClassVar[bool] = False

    async def tenant_for(self, principal: Principal, *, ctx: TenancyContext) -> TenantId:
        return "default"


@implements(TenantRegistry, version=1)
class FakeTenantRegistry:
    multi_tenant: ClassVar[bool] = False

    async def list_tenants(self, *, ctx: TenantRegistryContext) -> list[TenantId]:
        return ["default"]

    async def exists(self, tenant_id: TenantId, *, ctx: TenantRegistryContext) -> bool:
        return tenant_id == "default"


@implements(TenantConfigProvider, version=1)
class FakeConfigProvider:
    multi_tenant: ClassVar[bool] = False

    async def get(
        self, tenant_id: TenantId, *, ctx: TenantConfigLookupContext
    ) -> TenantConfig:
        return TenantConfig()


@implements(TenantLifecycle, version=1)
class FakeLifecycle:
    multi_tenant: ClassVar[bool] = False

    async def provision(self, tenant_id: TenantId, *, ctx: ProvisionContext) -> None:
        return None

    async def deprovision(
        self,
        tenant_id: TenantId,
        *,
        mode: Literal["soft", "hard"],
        ctx: DeprovisionContext,
    ) -> None:
        return None

    async def reset(self, tenant_id: TenantId, *, ctx: ResetContext) -> None:
        return None


@implements(JobQueue, version=1)
class FakeJobQueue:
    multi_tenant: ClassVar[bool] = False

    async def publish(self, job: Any, *, ctx: PublishContext) -> None:
        return None

    async def subscribe(self, *, handler: Any, ctx: SubscribeContext) -> None:
        return None


@implements(SecretResolver, version=1)
class FakeSecretResolver:
    multi_tenant: ClassVar[bool] = False

    async def get(
        self, name: str, *, tenant_id: TenantId | None, ctx: SecretContext
    ) -> SecretValue:
        return SecretValue(value="x")


@implements(RateLimiter, version=1)
class FakeRateLimiter:
    multi_tenant: ClassVar[bool] = False

    async def check(
        self,
        *,
        principal: Principal,
        tenant_id: TenantId,
        action: str,
        ctx: RateLimitContext,
    ) -> RateLimitDecision:
        return RateLimitDecision(allowed=True)


@implements(STMStore, version=1)
class FakeSTMStore:
    multi_tenant: ClassVar[bool] = False

    async def submit(self, memory: Memory, *, ctx: SubmitContext) -> Memory:
        return memory

    async def list_unconsumed(self, *, ctx: ListUnconsumedContext) -> list[Memory]:
        return []

    async def claim_batch(self, *, ctx: ClaimContext) -> MemoryBatch:
        return MemoryBatch(
            lease_id=ctx.lease_id,
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
class FakeLTMStore:
    multi_tenant: ClassVar[bool] = False
    workspace_capabilities: ClassVar[frozenset[type]] = frozenset({FileViewable})

    async def open_workspace(self, *, ctx: OpenWorkspaceContext) -> Workspace:
        raise NotImplementedError

    async def commit_workspace(self, ws: Workspace, *, ctx: CommitWorkspaceContext) -> Diff:
        return Diff()

    async def discard_workspace(self, ws: Workspace, *, ctx: DiscardWorkspaceContext) -> None:
        return None


@implements(ContextStore, version=1)
class FakeContextStore:
    multi_tenant: ClassVar[bool] = False
    workspace_capabilities: ClassVar[frozenset[type]] = frozenset({FileViewable})

    async def open_workspace(self, *, ctx: OpenWorkspaceContext) -> Workspace:
        raise NotImplementedError

    async def commit_workspace(self, ws: Workspace, *, ctx: CommitWorkspaceContext) -> Diff:
        return Diff()

    async def discard_workspace(self, ws: Workspace, *, ctx: DiscardWorkspaceContext) -> None:
        return None


@implements(DreamLeaseStore, version=1)
class FakeLeaseStore:
    multi_tenant: ClassVar[bool] = False

    async def acquire(self, *, ctx: AcquireLeaseContext) -> DreamLease | None:
        return None

    async def renew(self, *, ctx: RenewLeaseContext) -> bool:
        return False

    async def release(self, *, ctx: ReleaseLeaseContext) -> None:
        return None

    async def reclaim_expired(self, *, ctx: ReclaimLeasesContext) -> frozenset[str]:
        return frozenset()


@implements(STMSerializer, version=1)
class FakeSerializer:
    multi_tenant: ClassVar[bool] = False
    kind: ClassVar[str] = "fake"

    async def write(
        self,
        batch: MemoryBatch,
        *,
        target: Path,
        ctx: SerializeContext,
        services: SerializeServices,
    ) -> None:
        return None

    def prompt_fragment(self, batch: MemoryBatch) -> str:
        return ""


@implements(LTMPhaseRunner, version=1)
@implements(ContextPhaseRunner, version=1)
class FakeDreamEngine:
    multi_tenant: ClassVar[bool] = False
    workspace_requirements: ClassVar[dict[str, frozenset[type]]] = {
        "ltm": frozenset({FileViewable}),
        "context": frozenset({FileViewable}),
    }
    accepted_serializer_kinds: ClassVar[frozenset[str]] = frozenset({"fake"})

    async def run_ltm_phase(
        self, *, ctx: LTMPhaseContext, services: LTMPhaseServices
    ) -> None:
        return None

    async def run_context_phase(
        self, *, ctx: ContextPhaseContext, services: ContextPhaseServices
    ) -> None:
        return None


class BrokenPostDreamHook:
    """No @implements declaration → compliance must reject."""

    multi_tenant: ClassVar[bool] = False

    async def on_post_dream(self, *, ctx: Any, services: Any) -> None:
        return None


@pytest.fixture(autouse=True)
def _register_module() -> Any:
    module_name = "_dreamer_cli_test_components"
    module = types.ModuleType(module_name)
    g = globals()
    for name in (
        "FakeAuth",
        "FakeTenancy",
        "FakeTenantRegistry",
        "FakeConfigProvider",
        "FakeLifecycle",
        "FakeJobQueue",
        "FakeSecretResolver",
        "FakeRateLimiter",
        "FakeSTMStore",
        "FakeLTMStore",
        "FakeContextStore",
        "FakeLeaseStore",
        "FakeSerializer",
        "FakeDreamEngine",
        "BrokenPostDreamHook",
    ):
        setattr(module, name, g[name])
    sys.modules[module_name] = module
    yield
    sys.modules.pop(module_name, None)


def _good_config() -> str:
    return textwrap.dedent("""\
        auth: {class: _dreamer_cli_test_components.FakeAuth}
        tenancy: {class: _dreamer_cli_test_components.FakeTenancy}
        tenant_registry: {class: _dreamer_cli_test_components.FakeTenantRegistry}
        tenant_config_provider: {class: _dreamer_cli_test_components.FakeConfigProvider}
        tenant_lifecycle: {class: _dreamer_cli_test_components.FakeLifecycle}
        job_queue: {class: _dreamer_cli_test_components.FakeJobQueue}
        secret_resolver: {class: _dreamer_cli_test_components.FakeSecretResolver}
        rate_limiter: {class: _dreamer_cli_test_components.FakeRateLimiter}
        stm_store: {class: _dreamer_cli_test_components.FakeSTMStore}
        ltm_store: {class: _dreamer_cli_test_components.FakeLTMStore}
        context_store: {class: _dreamer_cli_test_components.FakeContextStore}
        dream_lease_store: {class: _dreamer_cli_test_components.FakeLeaseStore}
        stm_serializer: {class: _dreamer_cli_test_components.FakeSerializer}
        dream_engine: {class: _dreamer_cli_test_components.FakeDreamEngine}
    """)


def test_config_check_succeeds_on_valid_config(tmp_path: Path) -> None:
    p = tmp_path / "dreamer.yaml"
    p.write_text(_good_config())
    runner = CliRunner()
    result = runner.invoke(main, ["config", "check", str(p)])
    assert result.exit_code == 0, result.output
    assert "Resolved component graph" in result.output
    assert "Multi-tenancy table" in result.output
    assert "effective multi_tenancy = False" in result.output
    assert "config check: OK" in result.output


def test_config_check_fails_when_class_missing(tmp_path: Path) -> None:
    p = tmp_path / "dreamer.yaml"
    p.write_text(_good_config().replace(
        "stm_store: {class: _dreamer_cli_test_components.FakeSTMStore}",
        "stm_store: {class: nonexistent.module.X}",
    ))
    runner = CliRunner()
    result = runner.invoke(main, ["config", "check", str(p)])
    assert result.exit_code != 0
    assert "could not import" in result.output


def test_config_check_fails_when_required_slot_missing(tmp_path: Path) -> None:
    p = tmp_path / "dreamer.yaml"
    body = _good_config().replace(
        "stm_store: {class: _dreamer_cli_test_components.FakeSTMStore}\n", ""
    )
    p.write_text(body)
    runner = CliRunner()
    result = runner.invoke(main, ["config", "check", str(p)])
    assert result.exit_code != 0
    assert "required slot 'stm_store'" in result.output


def test_config_check_flags_compliance_failure(tmp_path: Path) -> None:
    body = _good_config() + textwrap.dedent("""\
        hooks:
          post_dream:
            - {class: _dreamer_cli_test_components.BrokenPostDreamHook}
    """)
    p = tmp_path / "dreamer.yaml"
    p.write_text(body)
    runner = CliRunner()
    result = runner.invoke(main, ["config", "check", str(p)])
    assert result.exit_code != 0
    assert "Compliance errors" in result.output
    assert "does not declare" in result.output


def test_config_check_with_multi_tenancy_required_fails(tmp_path: Path) -> None:
    body = _good_config() + "\nmulti_tenancy: required\n"
    p = tmp_path / "dreamer.yaml"
    p.write_text(body)
    runner = CliRunner()
    result = runner.invoke(main, ["config", "check", str(p)])
    assert result.exit_code != 0
    assert "multi_tenancy: required" in result.output


def test_init_serve_dream_commands_registered() -> None:
    assert "init" in main.commands
    assert "serve" in main.commands
    assert "dream" in main.commands


