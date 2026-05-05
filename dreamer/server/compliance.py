"""Compliance checker.

Walks the `__dreamer_protocols__` declarations on each component, runs the
version-set check against ``SUPPORTED_PROTOCOL_VERSIONS``, and runs a
signature/parameter-kind/annotation check using ``inspect.signature`` plus
``typing.get_type_hints`` against the corresponding Protocol method. Failures
are aggregated into a single :class:`ProtocolComplianceError` report so the
operator sees every problem at once.

Multi-tenancy is computed at the deployment level: a deployment is multi-tenant
only if every relevant component opts in via ``multi_tenant = True``. The
checker computes the per-slot table and the effective mode and asserts the
operator's optional ``multi_tenancy: required|forbidden|auto`` declaration.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal, get_type_hints

from dreamer.api.compat import SUPPORTED_PROTOCOL_VERSIONS
from dreamer.api.errors import ProtocolComplianceError

MultiTenancyMode = Literal["auto", "required", "forbidden"]


@dataclass(frozen=True, slots=True)
class SlotBinding:
    """One configured slot pointing at a single component instance."""

    slot: str
    component: object
    expected_protocols: tuple[type, ...]
    """Protocols the slot expects the component to implement (e.g. ``(STMStore,)``).

    Stores like ``stm_store`` declare exactly one expected Protocol; a hooks
    slot like ``hooks.post_dream`` may declare a single hook Protocol per slot
    entry. The checker enforces that the component declares each expected
    Protocol via ``@implements``.
    """


@dataclass(frozen=True, slots=True)
class MultiTenancyEntry:
    slot: str
    component: object
    multi_tenant: bool
    declaring_class: str


@dataclass(frozen=True, slots=True)
class ComplianceReport:
    """Outcome of a compliance check.

    ``errors`` is empty on success. Even on success, ``mt_table`` and
    ``effective_multi_tenant`` are populated so the operator can audit the
    decision.
    """

    errors: tuple[str, ...] = ()
    mt_table: tuple[MultiTenancyEntry, ...] = ()
    effective_multi_tenant: bool = False
    declared_mode: MultiTenancyMode = "auto"

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_failed(self) -> None:
        if self.errors:
            joined = "\n  - ".join(self.errors)
            raise ProtocolComplianceError(
                f"compliance check failed with {len(self.errors)} error(s):\n  - {joined}"
            )


@dataclass(slots=True)
class _Accumulator:
    errors: list[str] = field(default_factory=list)
    mt_table: list[MultiTenancyEntry] = field(default_factory=list)


def check_components(
    bindings: Iterable[SlotBinding],
    *,
    declared_mode: MultiTenancyMode = "auto",
) -> ComplianceReport:
    """Run the version + signature + multi-tenancy checks across every binding."""
    acc = _Accumulator()

    bindings_list = list(bindings)
    for binding in bindings_list:
        _check_binding(binding, acc)

    effective = _compute_effective_multi_tenant(acc.mt_table)
    _check_declared_mode(declared_mode, effective, acc)

    return ComplianceReport(
        errors=tuple(acc.errors),
        mt_table=tuple(acc.mt_table),
        effective_multi_tenant=effective,
        declared_mode=declared_mode,
    )


def _check_binding(binding: SlotBinding, acc: _Accumulator) -> None:
    component = binding.component
    component_label = _component_label(component)
    declared = _declared_protocols(component)

    for expected in binding.expected_protocols:
        if expected not in declared:
            acc.errors.append(
                f"slot {binding.slot!r}: component {component_label} does not declare "
                f"@implements({expected.__name__}); declared: "
                f"{sorted(p.__name__ for p in declared)}"
            )
            continue

        version = declared[expected]
        supported = SUPPORTED_PROTOCOL_VERSIONS.get(expected)
        if supported is None:
            acc.errors.append(
                f"slot {binding.slot!r}: protocol {expected.__name__} is not in "
                "SUPPORTED_PROTOCOL_VERSIONS (framework bug)"
            )
            continue
        if version not in supported:
            acc.errors.append(
                f"slot {binding.slot!r}: component {component_label} declares version "
                f"{version} of {expected.__name__}, supported: {sorted(supported)}"
            )

        _check_protocol_signature(binding.slot, component, expected, acc)

    for capability in _detected_optional_capabilities(component):
        if capability in binding.expected_protocols:
            continue
        _check_protocol_signature(binding.slot, component, capability, acc)

    mt_value = _resolve_multi_tenant_attr(component)
    declaring_cls = type(component).__name__
    acc.mt_table.append(
        MultiTenancyEntry(
            slot=binding.slot,
            component=component,
            multi_tenant=mt_value,
            declaring_class=declaring_cls,
        )
    )


def _check_protocol_signature(
    slot: str,
    component: object,
    protocol: type,
    acc: _Accumulator,
) -> None:
    """Compare every method on `protocol` against the implementation."""
    proto_methods = _protocol_methods(protocol)
    for method_name, proto_func in proto_methods.items():
        impl = getattr(component, method_name, None)
        if impl is None:
            acc.errors.append(
                f"slot {slot!r}: component {_component_label(component)} is missing "
                f"required method {protocol.__name__}.{method_name}"
            )
            continue

        try:
            proto_sig = inspect.signature(proto_func)
            impl_sig = inspect.signature(impl)
        except (TypeError, ValueError) as exc:
            acc.errors.append(
                f"slot {slot!r}: could not introspect signature of "
                f"{protocol.__name__}.{method_name}: {exc}"
            )
            continue

        diff = _compare_signatures(
            proto_sig=proto_sig,
            impl_sig=impl_sig,
            proto_func=proto_func,
            impl_func=impl,
        )
        if diff:
            acc.errors.append(
                f"slot {slot!r}: component {_component_label(component)} method "
                f"{protocol.__name__}.{method_name} signature mismatch: {diff}; "
                f"expected {proto_sig}; observed {impl_sig}"
            )


def _compare_signatures(
    *,
    proto_sig: inspect.Signature,
    impl_sig: inspect.Signature,
    proto_func: Any,
    impl_func: Any,
) -> str | None:
    proto_params = _normalize_params(proto_sig)
    impl_params = _normalize_params(impl_sig)

    if list(proto_params.keys()) != list(impl_params.keys()):
        return (
            f"parameter names differ: expected {list(proto_params.keys())}, "
            f"got {list(impl_params.keys())}"
        )

    for name, proto_param in proto_params.items():
        impl_param = impl_params[name]
        if proto_param.kind != impl_param.kind:
            return (
                f"parameter {name!r} kind mismatch: expected {proto_param.kind!r}, "
                f"got {impl_param.kind!r}"
            )
        proto_required = proto_param.default is inspect.Parameter.empty
        impl_required = impl_param.default is inspect.Parameter.empty
        if proto_required != impl_required:
            proto_kind = "required" if proto_required else "has-default"
            impl_kind = "required" if impl_required else "has-default"
            return (
                f"parameter {name!r} default presence mismatch: "
                f"expected {proto_kind}, got {impl_kind}"
            )

    proto_hints = _safe_get_type_hints(proto_func)
    impl_hints = _safe_get_type_hints(impl_func)

    if proto_hints is None or impl_hints is None:
        # One side is unannotated; accept either.
        return None

    for name in proto_params:
        proto_t = proto_hints.get(name)
        impl_t = impl_hints.get(name)
        if proto_t is None or impl_t is None:
            continue
        if not _annotations_equal(proto_t, impl_t):
            return (
                f"parameter {name!r} annotation mismatch: expected {proto_t!r}, "
                f"got {impl_t!r}"
            )

    proto_ret = proto_hints.get("return")
    impl_ret = impl_hints.get("return")
    if proto_ret is not None and impl_ret is not None and not _annotations_equal(
        proto_ret, impl_ret
    ):
        return f"return annotation mismatch: expected {proto_ret!r}, got {impl_ret!r}"

    return None


def _normalize_params(sig: inspect.Signature) -> dict[str, inspect.Parameter]:
    params = {}
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        params[name] = param
    return params


def _annotations_equal(a: Any, b: Any) -> bool:
    """Treat async-def vs Awaitable[X], Optional[X] vs X|None as equivalent.

    Pragmatic equality: rely on string repr and direct object identity. Modern
    Python collapses ``Optional[X]`` to ``X | None`` in `get_type_hints`, so
    string equality is reasonable.
    """
    if a is b:
        return True
    return repr(a) == repr(b)


def _safe_get_type_hints(func: Any) -> dict[str, Any] | None:
    try:
        return get_type_hints(func, include_extras=False)
    except Exception:
        return None


def _protocol_methods(protocol: type) -> dict[str, Any]:
    methods: dict[str, Any] = {}
    seen: set[str] = set()
    for klass in protocol.__mro__:
        if klass is object:
            continue
        if klass.__name__ == "Protocol" and klass.__module__ == "typing":
            continue
        for name, value in vars(klass).items():
            if name.startswith("_") or name in seen:
                continue
            if not callable(value):
                continue
            seen.add(name)
            methods[name] = value
    return methods


def _declared_protocols(component: object) -> dict[type, int]:
    cls = component if isinstance(component, type) else type(component)
    declared = getattr(cls, "__dreamer_protocols__", {})
    if not isinstance(declared, dict):
        return {}
    return dict(declared)


def _detected_optional_capabilities(component: object) -> list[type]:
    declared = _declared_protocols(component)
    return [
        proto
        for proto in declared
        if proto.__name__ in {"Lifecycle", "Routes", "Middlewares", "Transactional",
                              "TenantData", "ContextPendingStore", "ContextReader",
                              "SecretRotationHook", "DreamProgressHook"}
    ]


def _resolve_multi_tenant_attr(component: object) -> bool:
    value = getattr(component, "multi_tenant", False)
    if not isinstance(value, bool):
        return False
    return value


def _component_label(component: object) -> str:
    if isinstance(component, type):
        cls = component
    else:
        cls = type(component)
    module = getattr(cls, "__module__", "?")
    name = getattr(cls, "__qualname__", cls.__name__)
    return f"{module}.{name}"


def _compute_effective_multi_tenant(table: list[MultiTenancyEntry]) -> bool:
    if not table:
        return False
    return all(entry.multi_tenant for entry in table)


def _check_declared_mode(
    declared_mode: MultiTenancyMode,
    effective: bool,
    acc: _Accumulator,
) -> None:
    if declared_mode == "required" and not effective:
        offenders = [e for e in acc.mt_table if not e.multi_tenant]
        names = ", ".join(f"{e.slot}={_component_label(e.component)}" for e in offenders)
        acc.errors.append(
            "multi_tenancy: required, but the following components do not declare "
            f"multi_tenant=True: {names}"
        )
    elif declared_mode == "forbidden" and effective:
        acc.errors.append(
            "multi_tenancy: forbidden, but every configured component declares "
            "multi_tenant=True (deployment would be multi-tenant)"
        )


def declared_classvar(cls: type, name: str) -> Any:
    """Best-effort read of a Protocol-declared `ClassVar`. Used by tests."""
    return getattr(cls, name, None)


__all__: list[str] = [
    "ComplianceReport",
    "MultiTenancyEntry",
    "MultiTenancyMode",
    "SlotBinding",
    "check_components",
    "declared_classvar",
]
