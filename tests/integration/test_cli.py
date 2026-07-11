from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any, ClassVar

import pytest
from click.testing import CliRunner

from dreamer.api.auth import AuthBackend
from dreamer.api.compat import implements
from dreamer.api.contexts import AuthContext, SecretContext
from dreamer.api.secrets import SecretResolver
from dreamer.api.types import Principal, SecretValue, TenantId
from dreamer.cli.main import main


@implements(AuthBackend, version=1)
class _FakeAuth:
    multi_tenant: ClassVar[bool] = False

    async def authenticate(self, request: Any, *, ctx: AuthContext) -> Principal:
        return Principal(id="anon")


@implements(SecretResolver, version=1)
class _FakeSecretResolver:
    multi_tenant: ClassVar[bool] = False

    async def get(
        self, name: str, *, tenant_id: TenantId | None, ctx: SecretContext
    ) -> SecretValue:
        return SecretValue(value="x")


def _fake_components_yaml() -> str:
    return textwrap.dedent(
        """\
        auth: {class: tests.integration.test_cli._FakeAuth}
        tenancy: {class: dreamer.contrib.tenancy.single.SingleTenant}
        tenant_registry:
          class: dreamer.contrib.tenants.static.StaticTenantRegistry
          params: {tenants: [default]}
        tenant_config_provider:
          class: dreamer.contrib.tenants.static.StaticTenantConfigProvider
          params: {overrides: {}}
        tenant_lifecycle:
          class: dreamer.contrib.tenants.static.StaticTenantLifecycle
        job_queue: {class: dreamer.contrib.jobs.inproc.InProcessJobQueue}
        secret_resolver:
          class: tests.integration.test_cli._FakeSecretResolver
        rate_limiter:
          class: dreamer.testing.fakes.NoOpRateLimiter
        stm_store:
          class: dreamer.testing.fakes.InMemorySTMStore
        ltm_store:
          class: dreamer.testing.fakes.InMemoryLTMStore
        context_store:
          class: dreamer.testing.fakes.InMemoryContextStore
        dream_lease_store:
          class: dreamer.testing.fakes.InMemoryDreamLeaseStore
          params: {default_ttl_seconds: 60}
        stm_serializer:
          class: dreamer.testing.fakes.InMemorySTMSerializer
        dream_engine:
          class: dreamer.testing.fakes.DeterministicDreamEngine
        triggers:
          - class: dreamer.contrib.triggers.external.ExternalTrigger
            params: {name: external, tenant_id: default}
        stm_retention:
          keep_days: 30
          cadence_seconds: 86400
        """
    )


def test_init_scaffolds_files(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["init", "--path", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "dreamer.yaml").is_file()
    assert (tmp_path / ".gitignore").is_file()
    assert (tmp_path / "workspace" / "memory").is_dir()
    assert (tmp_path / "workspace" / "context").is_dir()
    contents = (tmp_path / "dreamer.yaml").read_text(encoding="utf-8")
    assert "dreamer.contrib.stm.sqlite.SQLiteSTMStore" in contents
    assert "dreamer.contrib.ltm.markdown.MarkdownLTMStore" in contents
    assert "dreamer.contrib.context.markdown.MarkdownContextStore" in contents
    gi = (tmp_path / ".gitignore").read_text(encoding="utf-8")
    assert "*.db" in gi


def test_init_refuses_to_overwrite_without_force(tmp_path: Path) -> None:
    (tmp_path / "dreamer.yaml").write_text("# hi\n", encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(main, ["init", "--path", str(tmp_path)])
    assert result.exit_code != 0
    assert "refusing to overwrite" in result.output.lower()
    assert (tmp_path / "dreamer.yaml").read_text(encoding="utf-8") == "# hi\n"


def test_init_force_overwrites(tmp_path: Path) -> None:
    (tmp_path / "dreamer.yaml").write_text("# hi\n", encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(main, ["init", "--path", str(tmp_path), "--force"])
    assert result.exit_code == 0, result.output
    contents = (tmp_path / "dreamer.yaml").read_text(encoding="utf-8")
    assert "dreamer.contrib.stm.sqlite.SQLiteSTMStore" in contents


def test_init_scaffold_wires_feedback_and_passes_config_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["init", "--path", str(tmp_path)])
    assert result.exit_code == 0, result.output

    contents = (tmp_path / "dreamer.yaml").read_text(encoding="utf-8")
    for needle in (
        "name: context_confirmed",
        "name: context_flagged",
        "dreamer.contrib.mcp_tools.feedback.ConfirmContextTool",
        "dreamer.contrib.mcp_tools.feedback.FlagContextTool",
        "exclude_types: [context_confirmed, context_flagged]",
        "max_autonomous_removals: 5",
        "enforce_pinned: true",
        "on_guard_violation: fail",
    ):
        assert needle in contents, f"scaffold missing: {needle!r}"

    # Relative paths in the scaffold (./dreamer.db, ./workspace) resolve
    # against the project directory.
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(main, ["config", "check", str(tmp_path / "dreamer.yaml")])
    assert result.exit_code == 0, result.output
    assert "config check: OK" in result.output


@pytest.mark.asyncio
async def test_serve_runtime_lifecycle_drains(tmp_path: Path) -> None:
    """Drive the lifecycle directly via the runtime helper the CLI uses,
    exercising the same start/stop wiring that ``dreamer serve`` triggers
    through uvicorn's lifespan integration."""
    from asgi_lifespan import LifespanManager

    from dreamer.api.config import load
    from dreamer.server.bootstrap import build_runtime

    p = tmp_path / "dreamer.yaml"
    p.write_text(_fake_components_yaml(), encoding="utf-8")
    resolved = load(p)
    runtime = build_runtime(resolved)

    # LifespanManager mirrors what uvicorn does for `serve`; serving() adds
    # the trigger host on top.
    async with LifespanManager(runtime.handle.app):
        async with runtime.serving():
            assert runtime.orchestrator._started is True  # noqa: SLF001
    assert runtime.orchestrator._started is False  # noqa: SLF001


def test_serve_fails_on_missing_config(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        main, ["serve", "--config", str(tmp_path / "missing.yaml")]
    )
    assert result.exit_code != 0
    assert "ConfigError" in result.output


def test_dream_one_shot_runs_one_cycle(tmp_path: Path) -> None:
    p = tmp_path / "dreamer.yaml"
    p.write_text(_fake_components_yaml(), encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "dream",
            "--config",
            str(p),
            "--tenant",
            "default",
            "--trigger",
            "external",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "dream completed" in result.output


def test_dream_resume_context_no_watermark_exits_cleanly(tmp_path: Path) -> None:
    p = tmp_path / "dreamer.yaml"
    p.write_text(_fake_components_yaml(), encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "dream",
            "--config",
            str(p),
            "--tenant",
            "default",
            "--resume-context",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "no context_pending watermark" in result.output


@pytest.mark.asyncio
async def test_dream_resume_context_with_watermark_runs_context_phase(
    tmp_path: Path,
) -> None:
    """Assemble the runtime in-process so we can plant a watermark on the
    same LTMStore instance the resume run reads (a CLI invocation would
    rebuild a fresh in-memory store with no watermark)."""
    from dreamer.api.config import load
    from dreamer.api.contexts import (
        GetContextPendingContext,
        SetContextPendingContext,
    )
    from dreamer.api.tenants import TenantScope
    from dreamer.api.types import Diff
    from dreamer.server.bootstrap import build_runtime

    p = tmp_path / "dreamer.yaml"
    p.write_text(_fake_components_yaml(), encoding="utf-8")
    resolved = load(p)
    runtime = build_runtime(resolved)
    ltm_store = resolved.components["ltm_store"]

    async with runtime.session():
        with TenantScope.set("default"):
            await ltm_store.set_context_pending(
                Diff(added=["topics/x.md"], modified=[], deleted=[]),
                ctx=SetContextPendingContext(
                    request_id="planted", tenant_id="default"
                ),
            )

        result = await runtime.orchestrator.resume_context("default")
        assert result["status"] == "ok"

        with TenantScope.set("default"):
            wm = await ltm_store.get_context_pending(
                ctx=GetContextPendingContext(request_id="check", tenant_id="default")
            )
        assert wm is None


def test_dream_purge_invokes_purge_consumed(tmp_path: Path) -> None:
    p = tmp_path / "dreamer.yaml"
    p.write_text(_fake_components_yaml(), encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "dream",
            "--config",
            str(p),
            "--tenant",
            "default",
            "--purge",
            "--before",
            "2026-04-01T00:00:00",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "purge_consumed removed" in result.output
    assert "before=2026-04-01T00:00:00" in result.output


def test_dream_purge_default_before_uses_keep_days(tmp_path: Path) -> None:
    p = tmp_path / "dreamer.yaml"
    p.write_text(_fake_components_yaml(), encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(
        main, ["dream", "--config", str(p), "--tenant", "default", "--purge"]
    )
    assert result.exit_code == 0, result.output
    assert "purge_consumed removed" in result.output


def test_dream_purge_requires_before_when_retention_disabled(
    tmp_path: Path,
) -> None:
    body = _fake_components_yaml().replace(
        "stm_retention:\n  keep_days: 30",
        "stm_retention:\n  keep_days: null",
    )
    assert "keep_days: null" in body, "test setup: keep_days replacement failed"
    p = tmp_path / "dreamer.yaml"
    p.write_text(body, encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(
        main, ["dream", "--config", str(p), "--tenant", "default", "--purge"]
    )
    assert result.exit_code != 0
    assert "--before is required" in result.output


def test_dream_resume_and_purge_are_mutually_exclusive(tmp_path: Path) -> None:
    p = tmp_path / "dreamer.yaml"
    p.write_text(_fake_components_yaml(), encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "dream",
            "--config",
            str(p),
            "--tenant",
            "default",
            "--resume-context",
            "--purge",
        ],
    )
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output


def test_dream_invalid_before_rejected(tmp_path: Path) -> None:
    p = tmp_path / "dreamer.yaml"
    p.write_text(_fake_components_yaml(), encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "dream",
            "--config",
            str(p),
            "--tenant",
            "default",
            "--purge",
            "--before",
            "not-a-date",
        ],
    )
    assert result.exit_code != 0
    assert "--before is not a valid ISO-8601" in result.output


def test_dream_fails_on_missing_config(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        main, ["dream", "--config", str(tmp_path / "missing.yaml")]
    )
    assert result.exit_code != 0
    assert "ConfigError" in result.output
