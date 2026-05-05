from __future__ import annotations

import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from dreamer.api.config import (
    ResolvedConfig,
    load,
    load_text,
    reset_alias_log,
)
from dreamer.api.errors import ConfigError


class FakeComponentA:
    def __init__(self, value: str = "default", port: int | None = None) -> None:
        self.value = value
        self.port = port


class FakeComponentB:
    def __init__(self, backends: list[Any] | None = None, name: str = "b") -> None:
        self.backends = list(backends or [])
        self.name = name


class FakeStmStore:
    def __init__(self, db_path: str = ":memory:", token: str | None = None) -> None:
        self.db_path = db_path
        self.token = token


class FakeTenancy:
    pass


class FakeTrigger:
    def __init__(self, *, name: str, tenant_id: str = "default", stm_store: Any = None) -> None:
        self.name = name
        self.tenant_id = tenant_id
        self.stm_store = stm_store


@pytest.fixture(autouse=True)
def _register_test_module() -> None:
    import types

    module_name = "_dreamer_test_components"
    module = types.ModuleType(module_name)
    module.FakeComponentA = FakeComponentA  # type: ignore[attr-defined]
    module.FakeComponentB = FakeComponentB  # type: ignore[attr-defined]
    module.FakeStmStore = FakeStmStore  # type: ignore[attr-defined]
    module.FakeTenancy = FakeTenancy  # type: ignore[attr-defined]
    module.FakeTrigger = FakeTrigger  # type: ignore[attr-defined]
    sys.modules[module_name] = module
    yield
    sys.modules.pop(module_name, None)


@pytest.fixture(autouse=True)
def _reset_alias_log() -> None:
    reset_alias_log()


_MINIMAL_BLOCK = "{class: _dreamer_test_components.FakeComponentA}"


def _minimal_required_config(**overrides: str) -> str:
    defaults = {slot: _MINIMAL_BLOCK for slot in (
        "auth", "tenancy", "tenant_registry", "tenant_config_provider",
        "tenant_lifecycle", "stm_store", "ltm_store", "context_store",
        "dream_lease_store", "stm_serializer", "dream_engine",
        "secret_resolver", "rate_limiter", "job_queue",
    )}
    defaults.update(overrides)
    lines = [f"{k}: {v}" for k, v in defaults.items()]
    return "\n".join(lines) + "\n"


def test_canonical_env_interpolation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DREAMER_TEST_VALUE", "from-env")
    body = _minimal_required_config(
        auth='{class: _dreamer_test_components.FakeComponentA, '
             'params: {value: "${env:DREAMER_TEST_VALUE}"}}',
    )
    result = load_text(body)
    assert isinstance(result, ResolvedConfig)
    assert result.components["auth"].value == "from-env"


def test_alias_env_interpolation_logs_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("DREAMER_ALIAS_VAR", "alias-val")
    body = _minimal_required_config(
        auth='{class: _dreamer_test_components.FakeComponentA, '
             'params: {value: "${DREAMER_ALIAS_VAR}"}}',
    )
    with caplog.at_level("INFO", logger="dreamer.api.config"):
        result = load_text(body)
    assert result.components["auth"].value == "alias-val"
    info_messages = [r for r in caplog.records if r.levelname == "INFO"]
    matching = [r for r in info_messages if "alias form" in r.message]
    assert len(matching) == 1
    caplog.clear()
    with caplog.at_level("INFO", logger="dreamer.api.config"):
        load_text(body)
    matching2 = [r for r in caplog.records
                 if r.levelname == "INFO" and "alias form" in r.message]
    assert matching2 == []


def test_env_default_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DREAMER_NO_SUCH_VAR", raising=False)
    body = _minimal_required_config(
        auth='{class: _dreamer_test_components.FakeComponentA, '
             'params: {value: "${env:DREAMER_NO_SUCH_VAR:-info}"}}',
    )
    result = load_text(body)
    assert result.components["auth"].value == "info"


def test_env_missing_no_default_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DREAMER_MISSING_VAR", raising=False)
    body = _minimal_required_config(
        auth='{class: _dreamer_test_components.FakeComponentA, '
             'params: {value: "${env:DREAMER_MISSING_VAR}"}}',
    )
    with pytest.raises(ConfigError, match="DREAMER_MISSING_VAR"):
        load_text(body)


def test_unknown_namespace_rejected() -> None:
    body = _minimal_required_config(
        auth='{class: _dreamer_test_components.FakeComponentA, '
             'params: {value: "${madeup:VAR}"}}',
    )
    with pytest.raises(ConfigError, match="unknown interpolation namespace: madeup"):
        load_text(body)


def test_reserved_unwired_namespace_rejected() -> None:
    for ns in ("tenant", "file", "vault", "component"):
        body = _minimal_required_config(
            auth=f'{{class: _dreamer_test_components.FakeComponentA, '
                 f'params: {{value: "${{{ns}:slug}}"}}}}',
        )
        with pytest.raises(ConfigError, match=f"namespace '{ns}'"):
            load_text(body)


def test_secret_resolution_lazy() -> None:
    """${secret:...} resolves to a literal sentinel until used; the loader
    does not call SecretResolver."""
    body = _minimal_required_config(
        auth='{class: _dreamer_test_components.FakeStmStore, '
             'params: {token: "${secret:GH_TOKEN}"}}',
    )
    result = load_text(body)
    assert result.components["auth"].token == "{{secret:GH_TOKEN}}"


def test_recursive_class_params_instantiation() -> None:
    body = _minimal_required_config(
        auth=textwrap.dedent("""
            {class: _dreamer_test_components.FakeComponentB, params: {
                backends: [
                    {class: _dreamer_test_components.FakeComponentA, params: {value: alpha}},
                    {class: _dreamer_test_components.FakeComponentA, params: {value: beta}},
                ],
                name: outer
            }}
        """).strip(),
    )
    result = load_text(body)
    auth = result.components["auth"]
    assert isinstance(auth, FakeComponentB)
    assert auth.name == "outer"
    assert len(auth.backends) == 2
    assert all(isinstance(b, FakeComponentA) for b in auth.backends)
    assert [b.value for b in auth.backends] == ["alpha", "beta"]


def test_ref_resolves_to_same_instance() -> None:
    body = _minimal_required_config(
        stm_store='{class: _dreamer_test_components.FakeStmStore, '
                  'params: {db_path: "./live.db"}}',
        triggers='[{class: _dreamer_test_components.FakeTrigger, params: '
                 '{name: t1, stm_store: {ref: stm_store}}}]',
    )
    result = load_text(body)
    stm = result.components["stm_store"]
    triggers = result.component_lists["triggers"]
    assert len(triggers) == 1
    assert triggers[0].stm_store is stm


def test_missing_required_slot_fails() -> None:
    body = _minimal_required_config()
    body = "\n".join(line for line in body.splitlines() if not line.startswith("auth:"))
    with pytest.raises(ConfigError, match="required slot 'auth' is unset"):
        load_text(body + "\n")


def test_class_must_be_fully_qualified() -> None:
    body = _minimal_required_config(auth="{class: NotQualified}")
    with pytest.raises(ConfigError, match="fully-qualified"):
        load_text(body)


def test_class_module_not_importable() -> None:
    body = _minimal_required_config(auth="{class: nonexistent.module.X}")
    with pytest.raises(ConfigError, match="could not import"):
        load_text(body)


def test_class_attr_not_in_module() -> None:
    body = _minimal_required_config(
        auth="{class: _dreamer_test_components.NoSuchClass}"
    )
    with pytest.raises(ConfigError, match="has no attribute"):
        load_text(body)


def test_extra_keys_in_component_block_rejected() -> None:
    body = _minimal_required_config(
        auth="{class: _dreamer_test_components.FakeComponentA, foo: bar}"
    )
    with pytest.raises(ConfigError, match="unexpected keys"):
        load_text(body)


def test_multi_tenancy_default_auto() -> None:
    body = _minimal_required_config()
    result = load_text(body)
    assert result.declared_multi_tenancy == "auto"


def test_multi_tenancy_explicit_required() -> None:
    body = _minimal_required_config() + "\nmulti_tenancy: required\n"
    result = load_text(body)
    assert result.declared_multi_tenancy == "required"


def test_multi_tenancy_invalid_value_rejected() -> None:
    body = _minimal_required_config() + "\nmulti_tenancy: maybe\n"
    with pytest.raises(ConfigError, match="multi_tenancy"):
        load_text(body)


def test_load_path(tmp_path: Path) -> None:
    body = _minimal_required_config()
    p = tmp_path / "dreamer.yaml"
    p.write_text(body)
    result = load(p)
    assert isinstance(result, ResolvedConfig)


def test_load_path_missing(tmp_path: Path) -> None:
    p = tmp_path / "missing.yaml"
    with pytest.raises(ConfigError, match="not found"):
        load(p)


def test_yaml_parse_error() -> None:
    with pytest.raises(ConfigError, match="YAML"):
        load_text("auth: [unclosed")


def test_top_level_must_be_mapping() -> None:
    with pytest.raises(ConfigError, match="mapping"):
        load_text("- 1\n- 2\n")


def test_context_store_list_form() -> None:
    body = _minimal_required_config()
    body = "\n".join(line for line in body.splitlines() if not line.startswith("context_store:"))
    body += "\n" + textwrap.dedent("""\
        context_store:
          - {class: _dreamer_test_components.FakeStmStore, params: {db_path: a}}
          - {class: _dreamer_test_components.FakeStmStore, params: {db_path: b}}
    """)
    result = load_text(body)
    assert len(result.component_lists["context_store"]) == 2


def test_hooks_resolved_into_per_slot_lists() -> None:
    body = _minimal_required_config() + textwrap.dedent("""
        hooks:
          post_dream:
            - {class: _dreamer_test_components.FakeComponentA, params: {value: pd}}
          pre_dream: []
    """)
    result = load_text(body)
    pd = result.component_lists["hooks.post_dream"]
    assert len(pd) == 1
    assert pd[0].value == "pd"
    assert result.component_lists["hooks.pre_dream"] == []


def test_invalid_interpolation_empty_name() -> None:
    body = _minimal_required_config(
        auth='{class: _dreamer_test_components.FakeComponentA, params: {value: "${env:}"}}'
    )
    with pytest.raises(ConfigError, match="empty name"):
        load_text(body)


def test_ref_to_unconfigured_slot() -> None:
    body = _minimal_required_config(
        triggers='[{class: _dreamer_test_components.FakeTrigger, params: '
                 '{name: t1, stm_store: {ref: nonexistent_slot}}}]',
    )
    with pytest.raises(ConfigError, match="ref:nonexistent_slot"):
        load_text(body)


def test_params_must_be_mapping() -> None:
    body = _minimal_required_config(
        auth='{class: _dreamer_test_components.FakeComponentA, params: [a, b]}'
    )
    with pytest.raises(ConfigError, match="'params' must be a mapping"):
        load_text(body)
