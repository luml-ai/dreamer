"""Hook Protocols.

One Protocol per hook point with a distinct method name. A single class can
implement multiple hook Protocols by defining the corresponding methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

if TYPE_CHECKING:
    from dreamer.api.contexts import (
        DreamFailedContext,
        DreamFailedServices,
        DreamProgressContext,
        DreamProgressServices,
        PostContextUpdateContext,
        PostContextUpdateServices,
        PostDreamContext,
        PostDreamServices,
        PostLTMUpdateContext,
        PostLTMUpdateServices,
        PostMemorySubmitContext,
        PreContextUpdateContext,
        PreContextUpdateServices,
        PreDreamContext,
        PreDreamServices,
        PreLTMUpdateContext,
        PreLTMUpdateServices,
        PreMemorySubmitContext,
    )


@runtime_checkable
class PreMemorySubmitHook(Protocol):
    multi_tenant: ClassVar[bool] = False

    async def on_pre_memory_submit(self, *, ctx: PreMemorySubmitContext) -> None: ...


@runtime_checkable
class PostMemorySubmitHook(Protocol):
    multi_tenant: ClassVar[bool] = False

    async def on_post_memory_submit(self, *, ctx: PostMemorySubmitContext) -> None: ...


@runtime_checkable
class PreDreamHook(Protocol):
    multi_tenant: ClassVar[bool] = False

    async def on_pre_dream(
        self, *, ctx: PreDreamContext, services: PreDreamServices
    ) -> None: ...


@runtime_checkable
class PostDreamHook(Protocol):
    multi_tenant: ClassVar[bool] = False

    async def on_post_dream(
        self, *, ctx: PostDreamContext, services: PostDreamServices
    ) -> None: ...


@runtime_checkable
class PreLTMUpdateHook(Protocol):
    multi_tenant: ClassVar[bool] = False

    async def on_pre_ltm_update(
        self, *, ctx: PreLTMUpdateContext, services: PreLTMUpdateServices
    ) -> None: ...


@runtime_checkable
class PostLTMUpdateHook(Protocol):
    multi_tenant: ClassVar[bool] = False

    async def on_post_ltm_update(
        self, *, ctx: PostLTMUpdateContext, services: PostLTMUpdateServices
    ) -> None: ...


@runtime_checkable
class PreContextUpdateHook(Protocol):
    multi_tenant: ClassVar[bool] = False

    async def on_pre_context_update(
        self, *, ctx: PreContextUpdateContext, services: PreContextUpdateServices
    ) -> None: ...


@runtime_checkable
class PostContextUpdateHook(Protocol):
    multi_tenant: ClassVar[bool] = False

    async def on_post_context_update(
        self, *, ctx: PostContextUpdateContext, services: PostContextUpdateServices
    ) -> None: ...


@runtime_checkable
class DreamFailedHook(Protocol):
    multi_tenant: ClassVar[bool] = False

    async def on_dream_failed(
        self, *, ctx: DreamFailedContext, services: DreamFailedServices
    ) -> None: ...


@runtime_checkable
class DreamProgressHook(Protocol):
    multi_tenant: ClassVar[bool] = False

    async def on_dream_progress(
        self, *, ctx: DreamProgressContext, services: DreamProgressServices
    ) -> None: ...


__all__ = [
    "DreamFailedHook",
    "DreamProgressHook",
    "PostContextUpdateHook",
    "PostDreamHook",
    "PostLTMUpdateHook",
    "PostMemorySubmitHook",
    "PreContextUpdateHook",
    "PreDreamHook",
    "PreLTMUpdateHook",
    "PreMemorySubmitHook",
]
