"""``dreamer`` CLI entry point.

Implements the four top-level commands documented in ``spec/cli.md``:

- ``dreamer init`` — scaffold a new project
- ``dreamer serve`` — run the ASGI app via uvicorn
- ``dreamer dream`` — run a one-shot dream cycle (``--resume-context`` or
  ``--purge`` for the alternative one-shot flows)
- ``dreamer config check`` — load + compliance-check a config file
"""

from __future__ import annotations

import asyncio
import logging
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import click

from dreamer.api.audit import AuditSink
from dreamer.api.auth import AuthBackend, Tenancy
from dreamer.api.config import ResolvedConfig, load
from dreamer.api.dream import (
    ContextPhaseRunner,
    DreamGate,
    LTMPhaseRunner,
)
from dreamer.api.errors import ConfigError, ProtocolComplianceError
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
from dreamer.api.secrets import SecretResolver
from dreamer.api.stores import (
    ContextStore,
    DreamLeaseStore,
    LTMStore,
    MCPTool,
    STMSerializer,
    STMStore,
)
from dreamer.api.tenants import (
    TenantConfigProvider,
    TenantLifecycle,
    TenantRegistry,
)
from dreamer.api.triggers import Trigger
from dreamer.api.usage import UsageSink
from dreamer.server.bootstrap import Runtime, build_runtime
from dreamer.server.compliance import (
    SlotBinding,
    check_components,
)

LOG_LEVEL_CHOICES = ("debug", "info", "warning", "error", "critical")


def _configure_logging(level: str) -> None:
    """Bootstrap stdlib logging so dreamer + agent loggers actually emit.

    Without this, the root logger sits at WARNING and per-frame INFO lines
    from the dream engine never reach the operator's terminal.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


@click.group()
def main() -> None:
    """`dreamer` command-line interface."""


@main.group()
def config() -> None:
    """Config-related commands."""


@config.command("check")
@click.argument("path", type=click.Path(path_type=Path), default=Path("dreamer.yaml"))
def config_check(path: Path) -> None:
    """Run the loader + compliance checker against a config file.

    Prints the resolved component graph and the per-slot multi_tenancy table.
    Exits non-zero on error.
    """
    try:
        resolved = load(path)
    except ConfigError as exc:
        click.echo(f"ConfigError: {exc}", err=True)
        sys.exit(2)

    _print_resolved_graph(resolved)

    bindings = list(_iter_slot_bindings(resolved))
    report = check_components(
        bindings,
        declared_mode=resolved.declared_multi_tenancy,  # type: ignore[arg-type]
    )

    click.echo("\nMulti-tenancy table (per slot):")
    for entry in report.mt_table:
        marker = "  " if entry.multi_tenant else "* "
        click.echo(
            f"  {marker}{entry.slot:30s} {entry.declaring_class:30s} "
            f"multi_tenant={entry.multi_tenant}"
        )
    click.echo(f"\neffective multi_tenancy = {report.effective_multi_tenant}")
    click.echo(f"declared multi_tenancy  = {report.declared_mode}")

    if report.errors:
        click.echo("\nCompliance errors:")
        for err in report.errors:
            click.echo(f"  - {err}", err=True)
        sys.exit(2)

    click.echo("\nconfig check: OK")


@main.command()
@click.option(
    "--path",
    "path",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=Path("."),
    show_default=True,
    help="Target directory to scaffold into.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Overwrite existing dreamer.yaml / .gitignore if present.",
)
def init(path: Path, *, force: bool) -> None:
    """Scaffold a new dreamer project at ``path``.

    Creates ``dreamer.yaml`` (with the default config skeleton),
    ``workspace/memory/``, ``workspace/context/``, and a ``.gitignore`` that
    keeps the SQLite database out of source control.
    """
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)

    config_path = target / "dreamer.yaml"
    gitignore_path = target / ".gitignore"
    workspace_memory = target / "workspace" / "memory"
    workspace_context = target / "workspace" / "context"

    if not force:
        if config_path.exists():
            click.echo(
                f"refusing to overwrite existing {config_path} (use --force)",
                err=True,
            )
            sys.exit(2)
        if gitignore_path.exists():
            click.echo(
                f"refusing to overwrite existing {gitignore_path} (use --force)",
                err=True,
            )
            sys.exit(2)

    config_path.write_text(_DEFAULT_CONFIG_YAML, encoding="utf-8")
    gitignore_path.write_text(_DEFAULT_GITIGNORE, encoding="utf-8")
    workspace_memory.mkdir(parents=True, exist_ok=True)
    workspace_context.mkdir(parents=True, exist_ok=True)

    click.echo(f"scaffolded {config_path}")
    click.echo(f"scaffolded {gitignore_path}")
    click.echo(f"scaffolded {workspace_memory}/")
    click.echo(f"scaffolded {workspace_context}/")
    click.echo("\nNext steps:")
    click.echo("  - edit dreamer.yaml to wire your auth + dream engine")
    click.echo("  - run `dreamer config check` to validate")
    click.echo("  - run `dreamer serve` to start the server")


@main.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path),
    default=Path("dreamer.yaml"),
    show_default=True,
    help="Path to the dreamer config file.",
)
@click.option("--host", "host", type=str, default=None, help="Host to bind.")
@click.option("--port", "port", type=int, default=None, help="Port to bind.")
@click.option(
    "--log-level",
    "log_level_override",
    type=click.Choice(LOG_LEVEL_CHOICES, case_sensitive=False),
    default=None,
    help="Override the log level (otherwise uses server.log_level from config).",
)
def serve(
    config_path: Path,
    *,
    host: str | None,
    port: int | None,
    log_level_override: str | None,
) -> None:
    """Run the dreamer ASGI server via uvicorn.

    Starts the Starlette app from the configured component graph; SIGTERM
    triggers a graceful shutdown that runs ``Lifecycle.stop`` on every
    component and stops the trigger host.
    """
    try:
        resolved = load(config_path)
    except ConfigError as exc:
        click.echo(f"ConfigError: {exc}", err=True)
        sys.exit(2)

    try:
        runtime = build_runtime(resolved)
    except (ConfigError, ProtocolComplianceError) as exc:
        click.echo(f"{type(exc).__name__}: {exc}", err=True)
        sys.exit(2)

    bind_host = host or resolved.raw.server.host
    bind_port = port if port is not None else resolved.raw.server.port
    log_level = (log_level_override or resolved.raw.server.log_level or "info").lower()
    _configure_logging(log_level)

    _print_startup_summary(resolved, bind_host=bind_host, bind_port=bind_port)

    asyncio.run(_run_uvicorn(runtime, host=bind_host, port=bind_port, log_level=log_level))


async def _run_uvicorn(
    runtime: Runtime,
    *,
    host: str,
    port: int,
    log_level: str,
) -> None:
    """Drive uvicorn against ``runtime.handle.app`` while triggers run.

    The Starlette app's lifespan starts the lifecycle dispatcher (and the
    secret watcher) automatically. We additionally start the trigger host
    here because triggers carry ``services`` and are not registered with the
    lifecycle dispatcher. SIGTERM / SIGINT is handled by uvicorn's own
    graceful shutdown path.
    """
    import uvicorn  # noqa: PLC0415

    config = uvicorn.Config(
        app=runtime.handle.app,
        host=host,
        port=port,
        log_level=log_level,
        lifespan="on",
    )
    server = uvicorn.Server(config)

    # uvicorn handles SIGTERM/SIGINT itself; we only need to make sure the
    # trigger host is started/stopped around the server's run.
    async with runtime.serving():
        try:
            await server.serve()
        except asyncio.CancelledError:
            # Cancelled mid-serve — let `serving()` drain the trigger host.
            raise


@main.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path),
    default=Path("dreamer.yaml"),
    show_default=True,
    help="Path to the dreamer config file.",
)
@click.option(
    "--tenant",
    "tenant_id",
    type=str,
    default="default",
    show_default=True,
    help="Tenant id to dream against.",
)
@click.option(
    "--trigger",
    "trigger_name",
    type=str,
    default="external",
    show_default=True,
    help="Trigger name recorded on the dream's audit events.",
)
@click.option(
    "--resume-context",
    "resume_context",
    is_flag=True,
    default=False,
    help=(
        "Skip trigger logic, gates, and STM claim; run only the context phase "
        "against the current ``context_pending`` watermark."
    ),
)
@click.option(
    "--purge",
    "purge",
    is_flag=True,
    default=False,
    help=(
        "Run STMStore.purge_consumed for the tenant outside the normal "
        "stm_retention cadence. No lease, no phase runs."
    ),
)
@click.option(
    "--before",
    "before",
    type=str,
    default=None,
    help=(
        "ISO-8601 timestamp; consumed STMs older than this are purged. "
        "Defaults to now - stm_retention.keep_days."
    ),
)
@click.option(
    "--log-level",
    "log_level_override",
    type=click.Choice(LOG_LEVEL_CHOICES, case_sensitive=False),
    default=None,
    help="Override the log level (otherwise uses server.log_level from config).",
)
def dream(
    config_path: Path,
    *,
    tenant_id: str,
    trigger_name: str,
    resume_context: bool,
    purge: bool,
    before: str | None,
    log_level_override: str | None,
) -> None:
    """Run a single dream cycle (or one-shot maintenance op) and exit.

    See ``spec/cli.md`` for the full set of variants. ``--resume-context`` and
    ``--purge`` are mutually exclusive.
    """
    if resume_context and purge:
        click.echo("--resume-context and --purge are mutually exclusive", err=True)
        sys.exit(2)

    try:
        resolved = load(config_path)
    except ConfigError as exc:
        click.echo(f"ConfigError: {exc}", err=True)
        sys.exit(2)

    log_level = (log_level_override or resolved.raw.server.log_level or "info").lower()
    _configure_logging(log_level)

    try:
        runtime = build_runtime(resolved)
    except (ConfigError, ProtocolComplianceError) as exc:
        click.echo(f"{type(exc).__name__}: {exc}", err=True)
        sys.exit(2)

    if purge:
        before_dt = _parse_before(before, resolved=resolved)
        asyncio.run(_run_purge(runtime, tenant_id=tenant_id, before=before_dt))
        return

    if resume_context:
        asyncio.run(_run_resume_context(runtime, tenant_id=tenant_id))
        return

    asyncio.run(_run_one_shot(runtime, tenant_id=tenant_id, trigger_name=trigger_name))


async def _run_one_shot(
    runtime: Runtime, *, tenant_id: str, trigger_name: str
) -> None:
    async with runtime.session():
        # Run the orchestrator handler synchronously rather than via the
        # job queue so the CLI can wait for completion and surface failures.
        from dreamer.api.types import DreamJob  # noqa: PLC0415

        await runtime.orchestrator._handle_job(  # noqa: SLF001 — CLI entry
            DreamJob(tenant_id=tenant_id, trigger_name=trigger_name)
        )
        state = await runtime.orchestrator.read_state()
        tenant_state = state.get("tenants", {}).get(tenant_id)
        if tenant_state is None:
            click.echo(f"dream completed (no state recorded) for tenant {tenant_id}")
            return
        success = tenant_state.get("last_dream_success")
        if success is False:
            click.echo(
                f"dream failed for tenant {tenant_id}: "
                f"{tenant_state.get('last_dream_error')}",
                err=True,
            )
            sys.exit(1)
        click.echo(f"dream completed for tenant {tenant_id}")


async def _run_resume_context(runtime: Runtime, *, tenant_id: str) -> None:
    async with runtime.session():
        result = await runtime.orchestrator.resume_context(tenant_id)
        status = result.get("status")
        if status == "no_watermark":
            click.echo("no context_pending watermark; nothing to do")
            return
        if status == "lease_held":
            click.echo(f"dream already running for {tenant_id}, exiting", err=True)
            sys.exit(1)
        click.echo(f"resume-context completed for tenant {tenant_id}")


async def _run_purge(
    runtime: Runtime, *, tenant_id: str, before: datetime
) -> None:
    async with runtime.session():
        removed = await runtime.orchestrator.purge_tenant(tenant_id, before=before)
        click.echo(
            f"purge_consumed removed {removed} row(s) for tenant {tenant_id} "
            f"(before={before.isoformat()})"
        )


def _parse_before(before: str | None, *, resolved: ResolvedConfig) -> datetime:
    """Resolve ``--before`` to an aware datetime.

    When ``--before`` is omitted, default to ``now - stm_retention.keep_days``
    (UTC). If retention is disabled (``keep_days=None``) and no ``--before``
    is given, refuse rather than guess.
    """
    if before is not None:
        try:
            value = datetime.fromisoformat(before)
        except ValueError as exc:
            click.echo(f"--before is not a valid ISO-8601 timestamp: {exc}", err=True)
            sys.exit(2)
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value
    keep_days = resolved.raw.stm_retention.keep_days
    if keep_days is None:
        click.echo(
            "--before is required when stm_retention.keep_days is null", err=True
        )
        sys.exit(2)
    return datetime.now(UTC) - timedelta(days=int(keep_days))


def _print_resolved_graph(resolved: ResolvedConfig) -> None:
    click.echo("Resolved component graph:")
    for slot, comp in resolved.components.items():
        click.echo(f"  {slot:30s} = {_describe(comp)}")
    for list_slot, items in resolved.component_lists.items():
        if not items or list_slot.startswith("hooks") and list_slot != "hooks":
            continue
        click.echo(f"  {list_slot:30s} = [{len(items)}]")
        for i, item in enumerate(items):
            click.echo(f"    [{i}] {_describe(item)}")
    for slot, items in resolved.component_lists.items():
        if not slot.startswith("hooks.") or not items:
            continue
        click.echo(f"  {slot:30s} = [{len(items)}]")
        for i, item in enumerate(items):
            click.echo(f"    [{i}] {_describe(item)}")


def _describe(component: Any) -> str:
    if component is None:
        return "<unset>"
    cls = type(component)
    return f"{cls.__module__}.{cls.__qualname__}"


def _print_startup_summary(
    resolved: ResolvedConfig, *, bind_host: str, bind_port: int
) -> None:
    click.echo(f"dreamer serve: binding {bind_host}:{bind_port}")
    _print_resolved_graph(resolved)
    bindings = list(_iter_slot_bindings(resolved))
    report = check_components(
        bindings,
        declared_mode=resolved.declared_multi_tenancy,  # type: ignore[arg-type]
    )
    click.echo("\nMulti-tenancy table (per slot):")
    for entry in report.mt_table:
        marker = "  " if entry.multi_tenant else "* "
        click.echo(
            f"  {marker}{entry.slot:30s} {entry.declaring_class:30s} "
            f"multi_tenant={entry.multi_tenant}"
        )
    click.echo(f"\neffective multi_tenancy = {report.effective_multi_tenant}")


_SINGLETON_PROTOCOL_MAP: dict[str, type] = {
    "auth": AuthBackend,
    "admin_auth": AuthBackend,
    "tenancy": Tenancy,
    "tenant_registry": TenantRegistry,
    "tenant_config_provider": TenantConfigProvider,
    "tenant_lifecycle": TenantLifecycle,
    "job_queue": JobQueue,
    "secret_resolver": SecretResolver,
    "rate_limiter": RateLimiter,
    "stm_store": STMStore,
    "ltm_store": LTMStore,
    "context_store": ContextStore,
    "dream_lease_store": DreamLeaseStore,
    "stm_serializer": STMSerializer,
    "dream_engine": LTMPhaseRunner,  # default engine implements both phases
}

_LIST_PROTOCOL_MAP: dict[str, type] = {
    "usage_sinks": UsageSink,
    "audit_sinks": AuditSink,
    "mcp_tools": MCPTool,
    "triggers": Trigger,
    "dream_gates": DreamGate,
}

_HOOK_PROTOCOL_MAP: dict[str, type] = {
    "hooks.pre_dream": PreDreamHook,
    "hooks.post_dream": PostDreamHook,
    "hooks.pre_ltm_update": PreLTMUpdateHook,
    "hooks.post_ltm_update": PostLTMUpdateHook,
    "hooks.pre_context_update": PreContextUpdateHook,
    "hooks.post_context_update": PostContextUpdateHook,
    "hooks.pre_memory_submit": PreMemorySubmitHook,
    "hooks.post_memory_submit": PostMemorySubmitHook,
    "hooks.on_dream_failed": DreamFailedHook,
    "hooks.on_dream_progress": DreamProgressHook,
}


def _iter_slot_bindings(resolved: ResolvedConfig) -> list[SlotBinding]:
    bindings: list[SlotBinding] = []

    for slot, protocol in _SINGLETON_PROTOCOL_MAP.items():
        comp = resolved.components.get(slot)
        if comp is None:
            continue
        expected = _expected_protocols(slot, protocol, comp)
        bindings.append(SlotBinding(slot=slot, component=comp, expected_protocols=expected))

    for slot, protocol in _LIST_PROTOCOL_MAP.items():
        items = resolved.component_lists.get(slot, [])
        for i, comp in enumerate(items):
            bindings.append(
                SlotBinding(
                    slot=f"{slot}[{i}]",
                    component=comp,
                    expected_protocols=(protocol,),
                )
            )

    for slot, protocol in _HOOK_PROTOCOL_MAP.items():
        items = resolved.component_lists.get(slot, [])
        for i, comp in enumerate(items):
            bindings.append(
                SlotBinding(
                    slot=f"{slot}[{i}]",
                    component=comp,
                    expected_protocols=(protocol,),
                )
            )

    return bindings


def _expected_protocols(slot: str, primary: type, comp: object) -> tuple[type, ...]:
    """Return the tuple of Protocols that ``comp`` is expected to implement
    in slot ``slot``. The default ``dream_engine`` slot expects *both*
    LTMPhaseRunner and ContextPhaseRunner."""
    if slot == "dream_engine":
        return (LTMPhaseRunner, ContextPhaseRunner)
    return (primary,)


_DEFAULT_GITIGNORE = """# dreamer scaffold
*.db
*.db-journal
*.db-shm
*.db-wal
"""


_DEFAULT_CONFIG_YAML = """\
server:
  host: 0.0.0.0
  port: 8080
  log_level: info

auth:
  class: dreamer.contrib.auth.simple_token.backend.SimpleTokenAuth
  params:
    db_path: ./dreamer.db

tenancy:
  class: dreamer.contrib.tenancy.single.SingleTenant

tenant_registry:
  class: dreamer.contrib.tenants.static.StaticTenantRegistry
  params:
    tenants: [default]

tenant_config_provider:
  class: dreamer.contrib.tenants.static.StaticTenantConfigProvider
  params:
    overrides: {}

tenant_lifecycle:
  class: dreamer.contrib.tenants.static.StaticTenantLifecycle

job_queue:
  class: dreamer.contrib.jobs.inproc.InProcessJobQueue

secret_resolver:
  class: dreamer.contrib.secrets.env.EnvSecretResolver

usage_sinks:
  - class: dreamer.contrib.usage.log.LogUsageSink

audit_sinks:
  - class: dreamer.contrib.audit.log.LogAuditSink

rate_limiter:
  class: dreamer.contrib.rate_limit.noop.NoOpRateLimiter

stm_retention:
  keep_days: 30
  cadence_seconds: 86400

stm_store:
  class: dreamer.contrib.stm.sqlite.SQLiteSTMStore
  params:
    db_path: ./dreamer.db
    max_batch_size: 200
    max_content_bytes: 8192
    memory_types:
      - name: failure
        description: An unexpected failure or error worth remembering.
      - name: observation
        description: A general observation that may not be obvious from the code.
      - name: code_snippet
        description: A useful code pattern worth preserving.

ltm_store:
  class: dreamer.contrib.ltm.markdown.MarkdownLTMStore
  params:
    root: ./workspace/memory

context_store:
  class: dreamer.contrib.context.markdown.MarkdownContextStore
  params:
    root: ./workspace/context

mcp_tools: []

dream_lease_store:
  class: dreamer.contrib.stm.sqlite.SQLiteDreamLeaseStore
  params:
    db_path: ./dreamer.db
    default_ttl_seconds: 1800

stm_serializer:
  class: dreamer.contrib.dream.serializers.MarkdownPerMemorySerializer

dream_engine:
  class: dreamer.contrib.dream.claude_agent.ClaudeAgentDreamEngine
  params:
    serializer: { ref: stm_serializer }
    mode: auto
    timeout_seconds: 1200

triggers:
  - class: dreamer.contrib.triggers.external.ExternalTrigger
    params:
      name: external
      tenant_id: default

dream_gates: []

hooks:
  pre_dream: []
  post_dream: []
  pre_ltm_update: []
  post_ltm_update: []
  pre_context_update: []
  post_context_update: []
  pre_memory_submit: []
  post_memory_submit: []
  on_dream_failed: []
  on_dream_progress: []
"""


__all__ = ["main"]


if __name__ == "__main__":  # pragma: no cover
    try:
        main()
    except ProtocolComplianceError as exc:
        click.echo(f"ProtocolComplianceError: {exc}", err=True)
        sys.exit(2)
