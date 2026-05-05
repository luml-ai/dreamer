"""Optional capability Protocols.

These are *additional* Protocols that any component may implement. The runtime
probes via `isinstance(component, Capability)` (works because they're
`@runtime_checkable`) and uses them if present.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

if TYPE_CHECKING:
    from starlette.applications import Starlette
    from starlette.middleware import Middleware

    from dreamer.api.contexts import (
        LifecycleContext,
        MiddlewaresContext,
        RoutesContext,
        TxBeginContext,
        TxCommitContext,
        TxPrepareContext,
        TxRollbackContext,
    )


TxHandle = Any


@runtime_checkable
class Lifecycle(Protocol):
    """Component lifecycle hooks dispatched at server start/stop."""

    multi_tenant: ClassVar[bool] = False

    async def start(self, *, ctx: LifecycleContext) -> None: ...
    async def stop(self, *, ctx: LifecycleContext) -> None: ...


@runtime_checkable
class Routes(Protocol):
    """Mount HTTP routes on the Starlette app at server boot."""

    multi_tenant: ClassVar[bool] = False

    def register_routes(self, app: Starlette, *, ctx: RoutesContext) -> None: ...


@runtime_checkable
class Middlewares(Protocol):
    """Provide ASGI middleware factories to install at server boot."""

    multi_tenant: ClassVar[bool] = False

    def middlewares(self, *, ctx: MiddlewaresContext) -> list[Middleware]: ...


@runtime_checkable
class Transactional(Protocol):
    """Optional. When present on both `LTMStore` and `ContextStore`, the
    orchestrator will use a two-phase commit across the dream's LTM and Context
    updates."""

    multi_tenant: ClassVar[bool] = False

    async def begin(self, *, ctx: TxBeginContext) -> TxHandle: ...
    async def prepare(self, tx: TxHandle, *, ctx: TxPrepareContext) -> bool: ...
    async def commit(self, tx: TxHandle, *, ctx: TxCommitContext) -> None: ...
    async def rollback(self, tx: TxHandle, *, ctx: TxRollbackContext) -> None: ...


__all__ = [
    "Lifecycle",
    "Middlewares",
    "Routes",
    "Transactional",
    "TxHandle",
]
