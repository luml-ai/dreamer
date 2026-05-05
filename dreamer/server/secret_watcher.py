"""Background watcher that polls TTL-bearing `SecretResolver` for rotations.

Components implementing `SecretRotationHook` declare which secrets they
depend on (`secret_dependencies: ClassVar[frozenset[str]]`). The watcher
polls the configured `SecretResolver`, tracks the last-seen `version` token
per `(secret_name, tenant_id)`, and dispatches `on_secret_rotated` to every
component whose `secret_dependencies` includes the rotated secret.

Polling cadence defaults to half the smallest `ttl_seconds` reported by the
resolver, with a floor and ceiling to keep the loop sane. If no secret is
TTL-bearing, the watcher exits its loop after the first poll — there is
nothing to watch.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Iterable

from dreamer.api.contexts import SecretContext, SecretRotationContext
from dreamer.api.secrets import SecretResolver, SecretRotationHook
from dreamer.api.types import DEFAULT_TENANT_ID, TenantId

logger = logging.getLogger(__name__)


_DEFAULT_POLL_INTERVAL = 60.0  # seconds, when no TTL info is available
_MIN_POLL_INTERVAL = 1.0
_MAX_POLL_INTERVAL = 300.0


class SecretWatcher:

    def __init__(
        self,
        resolver: SecretResolver,
        hooks: Iterable[SecretRotationHook],
        *,
        tenants: Iterable[TenantId] = (DEFAULT_TENANT_ID,),
        poll_interval_seconds: float | None = None,
    ) -> None:
        self.resolver = resolver
        self.hooks: list[SecretRotationHook] = list(hooks)
        self.tenants: list[TenantId] = list(tenants)
        self.poll_interval_seconds = poll_interval_seconds
        self._versions: dict[tuple[str, TenantId | None], str | None] = {}
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    def watched_secrets(self) -> set[str]:
        out: set[str] = set()
        for hook in self.hooks:
            out.update(getattr(hook, "secret_dependencies", ()) or ())
        return out

    async def start(self) -> None:
        if not self.hooks:
            return
        if self._task is not None:
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._run(), name="dreamer.secret_watcher")

    async def stop(self) -> None:
        if self._task is None:
            return
        self._stop.set()
        try:
            await asyncio.wait_for(self._task, timeout=5.0)
        except TimeoutError:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    async def _run(self) -> None:
        try:
            while not self._stop.is_set():
                interval = await self._poll_once()
                interval = max(_MIN_POLL_INTERVAL, min(_MAX_POLL_INTERVAL, interval))
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=interval)
                except TimeoutError:
                    continue
        except asyncio.CancelledError:
            pass
        except Exception:  # noqa: BLE001 — never crash the server on a watcher fault
            logger.exception("SecretWatcher loop terminated unexpectedly")

    async def _poll_once(self) -> float:
        secrets = self.watched_secrets()
        if not secrets:
            return _DEFAULT_POLL_INTERVAL
        ttl_observations: list[float] = []
        for name in sorted(secrets):
            for tenant_id in self.tenants:
                key = (name, tenant_id)
                last_version = self._versions.get(key)
                req_id = f"secret_watcher.{uuid.uuid4().hex}"
                ctx = SecretContext(
                    request_id=req_id,
                    tenant_id=tenant_id,
                    if_changed_since=last_version,
                )
                try:
                    secret = await self.resolver.get(name, tenant_id=tenant_id, ctx=ctx)
                except Exception:  # noqa: BLE001 — log but keep polling
                    logger.exception(
                        "SecretResolver.get failed for %s@%s", name, tenant_id
                    )
                    continue
                if secret.ttl_seconds is not None:
                    ttl_observations.append(secret.ttl_seconds)
                if key not in self._versions:
                    # First observation: just record the baseline.
                    self._versions[key] = secret.version
                    continue
                if secret.version != last_version:
                    self._versions[key] = secret.version
                    await self._dispatch(name, tenant_id=tenant_id, new_version=secret.version)
        if self.poll_interval_seconds is not None:
            return self.poll_interval_seconds
        if ttl_observations:
            # Half the smallest TTL keeps within Nyquist for a single rotation event.
            return min(ttl_observations) / 2.0
        return _DEFAULT_POLL_INTERVAL

    async def _dispatch(
        self, name: str, *, tenant_id: TenantId | None, new_version: str | None
    ) -> None:
        ctx = SecretRotationContext(
            request_id=f"secret_rotation.{uuid.uuid4().hex}",
            tenant_id=tenant_id,
            secret_name=name,
            new_version=new_version,
        )
        for hook in self.hooks:
            deps: frozenset[str] = getattr(hook, "secret_dependencies", frozenset()) or frozenset()
            if name not in deps:
                continue
            try:
                await hook.on_secret_rotated(name, ctx=ctx)
            except Exception:  # noqa: BLE001 — log but keep going
                logger.exception(
                    "SecretRotationHook %s.on_secret_rotated raised for %s",
                    type(hook).__qualname__,
                    name,
                )


__all__ = ["SecretWatcher"]
