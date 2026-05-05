"""Dream phase Protocols + DreamGate."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

from dreamer.api.types import GateDecision

if TYPE_CHECKING:
    from dreamer.api.contexts import (
        ContextPhaseContext,
        ContextPhaseServices,
        DreamGateContext,
        DreamGateServices,
        LTMPhaseContext,
        LTMPhaseServices,
    )


@runtime_checkable
class LTMPhaseRunner(Protocol):
    """Runs the LTM phase of a dream."""

    multi_tenant: ClassVar[bool] = False
    workspace_requirements: ClassVar[Mapping[str, frozenset[type]]]
    accepted_serializer_kinds: ClassVar[frozenset[str]]

    async def run_ltm_phase(
        self, *, ctx: LTMPhaseContext, services: LTMPhaseServices
    ) -> None: ...


@runtime_checkable
class ContextPhaseRunner(Protocol):
    """Runs the context phase of a dream."""

    multi_tenant: ClassVar[bool] = False
    workspace_requirements: ClassVar[Mapping[str, frozenset[type]]]

    async def run_context_phase(
        self, *, ctx: ContextPhaseContext, services: ContextPhaseServices
    ) -> None: ...


@runtime_checkable
class DreamGate(Protocol):
    """Decides whether a dream should proceed."""

    multi_tenant: ClassVar[bool] = False

    async def check(
        self, *, ctx: DreamGateContext, services: DreamGateServices
    ) -> GateDecision: ...


__all__ = [
    "ContextPhaseRunner",
    "DreamGate",
    "LTMPhaseRunner",
]
