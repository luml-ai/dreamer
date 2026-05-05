"""``EnvSecretResolver`` — default secret resolver reading from ``os.environ``.

Returns ``ttl_seconds=None, version=None`` so callers re-fetch every time
(env-backed secrets are cheap; cache invalidation is irrelevant).
"""

from __future__ import annotations

import os
from typing import ClassVar

from dreamer.api.compat import implements
from dreamer.api.contexts import SecretContext
from dreamer.api.secrets import SecretResolver
from dreamer.api.types import SecretValue, TenantId


@implements(SecretResolver, version=1)
class EnvSecretResolver:
    """Resolve named secrets from process environment variables.

    Per-tenant scoping is supported via ``${tenant_id}`` interpolation in
    ``prefix`` / ``suffix`` (e.g. ``prefix="DREAMER_${tenant_id}_"`` produces
    ``DREAMER_acme_GITHUB_TOKEN`` when ``name="GITHUB_TOKEN"``). Falls back
    to the bare ``name`` if no scoped variable is present.
    """

    multi_tenant: ClassVar[bool] = True

    def __init__(
        self,
        *,
        prefix: str | None = None,
        suffix: str | None = None,
    ) -> None:
        self.prefix = prefix
        self.suffix = suffix

    async def get(
        self,
        name: str,
        *,
        tenant_id: TenantId | None,
        ctx: SecretContext,
    ) -> SecretValue:
        candidates = self._candidate_names(name=name, tenant_id=tenant_id)
        for candidate in candidates:
            value = os.environ.get(candidate)
            if value is not None:
                return SecretValue(value=value, ttl_seconds=None, version=None)
        return SecretValue(value="", ttl_seconds=None, version=None)

    def _candidate_names(
        self, *, name: str, tenant_id: TenantId | None
    ) -> list[str]:
        out: list[str] = []
        if (self.prefix or self.suffix) and tenant_id is not None:
            prefix = (self.prefix or "").replace("${tenant_id}", tenant_id)
            suffix = (self.suffix or "").replace("${tenant_id}", tenant_id)
            out.append(f"{prefix}{name}{suffix}")
        out.append(name)
        return out


__all__ = ["EnvSecretResolver"]
