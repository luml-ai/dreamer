"""YAML config loader.

Recursively resolves ``{class, params}`` blocks bottom-up via lazy import,
then resolves ``ref:`` references after the top-level component graph is
built. Applies namespaced ``${env|secret|tenant|file|vault|component:...}``
interpolation per Design rules. Unknown namespaces and reserved-but-unwired
namespaces both fail with :class:`ConfigError`.
"""

from __future__ import annotations

import importlib
import logging
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

from dreamer.api.errors import ConfigError

logger = logging.getLogger(__name__)


class ServerSection(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080
    log_level: str = "info"
    model_config = ConfigDict(extra="allow")


class StmRetentionSection(BaseModel):
    keep_days: int | None = 30
    cadence_seconds: int = 86400
    model_config = ConfigDict(extra="allow")


class HooksSection(BaseModel):
    pre_dream: list[Any] = []
    post_dream: list[Any] = []
    pre_ltm_update: list[Any] = []
    post_ltm_update: list[Any] = []
    pre_context_update: list[Any] = []
    post_context_update: list[Any] = []
    pre_memory_submit: list[Any] = []
    post_memory_submit: list[Any] = []
    on_dream_failed: list[Any] = []
    on_dream_progress: list[Any] = []
    model_config = ConfigDict(extra="allow")


class RootConfig(BaseModel):
    """Top-level configuration model.

    Component blocks (``auth``, ``stm_store``, ``triggers``, …) are stored as
    raw ``Any`` so the loader can perform recursive ``{class, params}``
    resolution. The loader populates :class:`ResolvedConfig` from this raw
    model.
    """

    server: ServerSection = ServerSection()
    auth: Any | None = None
    admin_auth: Any | None = None
    tenancy: Any | None = None
    tenant_registry: Any | None = None
    tenant_config_provider: Any | None = None
    tenant_lifecycle: Any | None = None
    job_queue: Any | None = None
    secret_resolver: Any | None = None
    usage_sinks: list[Any] = []
    audit_sinks: list[Any] = []
    rate_limiter: Any | None = None
    stm_retention: StmRetentionSection = StmRetentionSection()
    stm_store: Any | None = None
    ltm_store: Any | None = None
    context_store: Any | None = None
    mcp_tools: list[Any] = []
    dream_lease_store: Any | None = None
    stm_serializer: Any | None = None
    dream_engine: Any | None = None
    dream_engine_overrides: Mapping[str, Any] | None = None
    triggers: list[Any] = []
    dream_gates: list[Any] = []
    hooks: HooksSection = HooksSection()
    multi_tenancy: str = "auto"
    model_config = ConfigDict(extra="allow")


@dataclass(slots=True)
class ResolvedConfig:
    """Output of :func:`load`: the raw RootConfig + the resolved component graph.

    The loader builds the graph by walking ``RootConfig``, recursively turning
    every ``{class, params}`` block into an instance, and resolving ``ref:``
    references by name once the top-level graph is built.
    """

    raw: RootConfig
    components: dict[str, Any] = field(default_factory=dict)
    component_lists: dict[str, list[Any]] = field(default_factory=dict)
    declared_multi_tenancy: str = "auto"
    interpolation_warnings: list[str] = field(default_factory=list)


_ALIAS_LOGGED: dict[str, bool] = {}


def load(path: str | Path) -> ResolvedConfig:
    """Parse YAML at ``path`` and resolve the component graph."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    raw_text = path.read_text(encoding="utf-8")
    return load_text(raw_text, source=str(path))


def load_text(text: str, *, source: str = "<string>") -> ResolvedConfig:
    """Parse YAML from a string and resolve the component graph."""
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML parse error in {source}: {exc}") from exc

    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"top-level config must be a mapping, got {type(data).__name__}")

    try:
        raw = RootConfig.model_validate(data)
    except Exception as exc:
        raise ConfigError(f"config validation failed: {exc}") from exc

    return _resolve(raw)


_SINGLETON_SLOTS = (
    "auth",
    "admin_auth",
    "tenancy",
    "tenant_registry",
    "tenant_config_provider",
    "tenant_lifecycle",
    "job_queue",
    "secret_resolver",
    "rate_limiter",
    "stm_store",
    "ltm_store",
    "context_store",
    "dream_lease_store",
    "stm_serializer",
    "dream_engine",
)

_LIST_SLOTS = (
    "usage_sinks",
    "audit_sinks",
    "mcp_tools",
    "triggers",
    "dream_gates",
)

_REQUIRED_SLOTS = (
    "auth",
    "tenancy",
    "tenant_registry",
    "tenant_config_provider",
    "tenant_lifecycle",
    "stm_store",
    "ltm_store",
    "context_store",
    "dream_lease_store",
    "stm_serializer",
    "dream_engine",
    "secret_resolver",
    "rate_limiter",
    "job_queue",
)

_HOOK_SLOT_NAMES = (
    "pre_dream",
    "post_dream",
    "pre_ltm_update",
    "post_ltm_update",
    "pre_context_update",
    "post_context_update",
    "pre_memory_submit",
    "post_memory_submit",
    "on_dream_failed",
    "on_dream_progress",
)


def _resolve(raw: RootConfig) -> ResolvedConfig:
    components: dict[str, Any] = {}
    component_lists: dict[str, list[Any]] = {}

    for slot in _SINGLETON_SLOTS:
        block = getattr(raw, slot)
        components[slot] = _resolve_node(block, slot=slot, ref_targets=components)

    for slot in _LIST_SLOTS:
        items = getattr(raw, slot)
        component_lists[slot] = [
            _resolve_node(item, slot=f"{slot}[{i}]", ref_targets=components)
            for i, item in enumerate(items or [])
        ]

    # context_store may be configured as a list; the server wraps a list in
    # FanoutContextStore. Build the candidate list here regardless.
    if isinstance(raw.context_store, list):
        components["context_store"] = None
        component_lists["context_store"] = [
            _resolve_node(item, slot=f"context_store[{i}]", ref_targets=components)
            for i, item in enumerate(raw.context_store)
        ]

    hooks_lists: dict[str, list[Any]] = {}
    for hook_name in _HOOK_SLOT_NAMES:
        items = getattr(raw.hooks, hook_name)
        hooks_lists[hook_name] = [
            _resolve_node(item, slot=f"hooks.{hook_name}[{i}]", ref_targets=components)
            for i, item in enumerate(items or [])
        ]
    component_lists["hooks"] = []
    component_lists.update({f"hooks.{k}": v for k, v in hooks_lists.items()})

    # context_store may be configured as either a singleton or a list of stores
    # (the server wraps a list in FanoutContextStore); both satisfy the contract.
    for slot in _REQUIRED_SLOTS:
        if components.get(slot) is not None:
            continue
        if slot == "context_store" and component_lists.get("context_store"):
            continue
        raise ConfigError(f"required slot {slot!r} is unset")

    declared_mode = (raw.multi_tenancy or "auto").lower()
    if declared_mode not in ("auto", "required", "forbidden"):
        raise ConfigError(
            f"multi_tenancy must be one of auto|required|forbidden, got {declared_mode!r}"
        )

    return ResolvedConfig(
        raw=raw,
        components=components,
        component_lists=component_lists,
        declared_multi_tenancy=declared_mode,
        interpolation_warnings=list(_alias_warnings()),
    )


def _resolve_node(node: Any, *, slot: str, ref_targets: Mapping[str, Any]) -> Any:
    """Recursively resolve a single config node.

    Returns:
      - For a ``{class, params}`` mapping: an instance of the imported class.
      - For a ``{ref: name}`` mapping: a `_RefMarker` (resolved in pass 2).
      - For other mappings: a dict with values resolved recursively.
      - For lists: a list with each item resolved.
      - For strings: the string with interpolations applied.
      - Otherwise: the value unchanged.
    """
    if node is None:
        return None
    if isinstance(node, str):
        return _interpolate(node)
    if isinstance(node, list):
        return [_resolve_node(v, slot=f"{slot}[{i}]", ref_targets=ref_targets)
                for i, v in enumerate(node)]
    if isinstance(node, Mapping):
        if set(node.keys()) == {"ref"}:
            target = node["ref"]
            if not isinstance(target, str):
                raise ConfigError(
                    f"slot {slot}: 'ref' must be a string, got {type(target).__name__}"
                )
            # Refs may only point at top-level singleton slots, all of which are
            # populated before list/hook slots are resolved.
            if target not in _SINGLETON_SLOTS:
                raise ConfigError(
                    f"ref:{target}: target is not a top-level singleton slot. "
                    f"Refs may only point to: {sorted(_SINGLETON_SLOTS)}"
                )
            target_value = ref_targets.get(target)
            if target_value is None:
                raise ConfigError(
                    f"ref:{target}: target slot is not configured (or unresolved)"
                )
            return target_value
        if "class" in node:
            return _instantiate(node, slot=slot, ref_targets=ref_targets)
        return {
            k: _resolve_node(v, slot=f"{slot}.{k}", ref_targets=ref_targets)
            for k, v in node.items()
        }
    return node


def _instantiate(node: Mapping[str, Any], *, slot: str, ref_targets: Mapping[str, Any]) -> Any:
    fqn = node.get("class")
    if not isinstance(fqn, str) or not fqn:
        raise ConfigError(f"slot {slot}: 'class' must be a non-empty string")
    if "extra" in node:
        raise ConfigError(
            f"slot {slot}: only 'class' and 'params' are allowed in component blocks"
        )
    extra_keys = set(node.keys()) - {"class", "params"}
    if extra_keys:
        raise ConfigError(
            f"slot {slot}: unexpected keys in component block: {sorted(extra_keys)}"
        )
    raw_params = node.get("params") or {}
    if not isinstance(raw_params, Mapping):
        raise ConfigError(f"slot {slot}: 'params' must be a mapping")
    cls = _import_class(fqn, slot=slot)
    resolved_params = {
        k: _resolve_node(v, slot=f"{slot}.params.{k}", ref_targets=ref_targets)
        for k, v in raw_params.items()
    }
    try:
        return cls(**resolved_params)
    except TypeError as exc:
        keys = sorted(resolved_params.keys())
        raise ConfigError(
            f"slot {slot}: could not construct {fqn} with params={keys}: {exc}"
        ) from exc


def _import_class(fqn: str, *, slot: str) -> type:
    if "." not in fqn:
        raise ConfigError(
            f"slot {slot}: 'class' must be fully-qualified (module.path.ClassName), got {fqn!r}"
        )
    module_path, _, class_name = fqn.rpartition(".")
    try:
        module = importlib.import_module(module_path)
    except Exception as exc:
        raise ConfigError(
            f"slot {slot}: could not import {fqn}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    cls = getattr(module, class_name, None)
    if cls is None:
        raise ConfigError(
            f"slot {slot}: module {module_path} has no attribute {class_name}"
        )
    if not isinstance(cls, type):
        raise ConfigError(
            f"slot {slot}: {fqn} is not a class (got {type(cls).__name__})"
        )
    return cls


# Reserved namespaces. Some are not yet implemented in v1.
_RESERVED_NAMESPACES = frozenset({"env", "secret", "tenant", "file", "vault", "component"})
_UNWIRED_NAMESPACES = frozenset({"tenant", "file", "vault", "component"})

_INTERP_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _alias_warnings() -> list[str]:
    """Return collected alias-form interpolation warnings."""
    return list(_ALIAS_LOGGED.keys())


def _interpolate(value: str) -> str:
    """Interpolate every ``${...}`` occurrence in ``value``."""
    if "${" not in value:
        return value

    def replace(match: re.Match[str]) -> str:
        body = match.group(1)
        return _resolve_interpolation(body)

    return _INTERP_PATTERN.sub(replace, value)


def _resolve_interpolation(body: str) -> str:
    """Resolve a single ``${...}`` body, returning the substitution string."""
    name_part, _, default_part = body.partition(":-")
    has_default = ":-" in body
    default = default_part if has_default else None

    if ":" in name_part:
        namespace, _, name = name_part.partition(":")
    else:
        # Alias form: implicit env namespace (${VAR} == ${env:VAR}).
        if name_part.startswith("$") or not name_part.strip():
            raise ConfigError(f"invalid interpolation: ${{{body}}}")
        namespace = "env"
        name = name_part
        if name not in _ALIAS_LOGGED:
            _ALIAS_LOGGED[name] = True
            logger.info(
                "config interpolation: alias form ${%s} used; canonical form is ${env:%s}",
                name,
                name,
            )

    if not name:
        raise ConfigError(f"invalid interpolation: ${{{body}}}: empty name")

    if namespace not in _RESERVED_NAMESPACES:
        raise ConfigError(
            f"unknown interpolation namespace: {namespace}; "
            "reserved: env, secret, tenant, file, vault, component"
        )

    if namespace == "env":
        result = os.environ.get(name)
        if result is None:
            if default is not None:
                return default
            raise ConfigError(f"environment variable {name!r} is unset and has no default")
        return result

    if namespace == "secret":
        # Lazy resolution: leave a sentinel string; resolved at use time by the
        # SecretResolver. Components that consume secret values must call
        # `secret_resolver.get(...)` themselves; the sentinel is a literal that
        # surfaces clearly if a component tries to use it without resolving.
        return f"{{{{secret:{name}}}}}"

    if namespace in _UNWIRED_NAMESPACES:
        raise ConfigError(
            f"interpolation namespace {namespace!r} is reserved but not implemented in v1"
        )

    # Should be unreachable given the reserved/unwired check above.
    raise ConfigError(f"interpolation namespace {namespace!r} is not handled")  # pragma: no cover


def reset_alias_log() -> None:
    """Reset the alias-form interpolation log (used by tests)."""
    _ALIAS_LOGGED.clear()


__all__ = [
    "HooksSection",
    "ResolvedConfig",
    "RootConfig",
    "ServerSection",
    "StmRetentionSection",
    "load",
    "load_text",
    "reset_alias_log",
]
