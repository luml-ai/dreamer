"""SecretResolver and SecretRotationHook Protocols."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

from dreamer.api.types import SecretValue, TenantId

if TYPE_CHECKING:
    from dreamer.api.contexts import SecretContext, SecretRotationContext


@runtime_checkable
class SecretResolver(Protocol):
    """Resolves named secrets at request time."""

    multi_tenant: ClassVar[bool] = False

    async def get(
        self, name: str, *, tenant_id: TenantId | None, ctx: SecretContext
    ) -> SecretValue: ...


@runtime_checkable
class SecretRotationHook(Protocol):
    """Optional capability. Components that hold long-lived secret-derived
    state may implement this to be notified when a secret they depend on
    rotates."""

    multi_tenant: ClassVar[bool] = False
    secret_dependencies: ClassVar[frozenset[str]]

    async def on_secret_rotated(self, name: str, *, ctx: SecretRotationContext) -> None: ...


__all__ = [
    "SecretResolver",
    "SecretRotationHook",
    "SecretValue",
]
