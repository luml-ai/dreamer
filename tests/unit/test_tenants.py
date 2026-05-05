from __future__ import annotations

import asyncio

import pytest

from dreamer.api.tenants import TenantScope


def test_assert_matches_raises_when_unset() -> None:
    TenantScope.clear()
    with pytest.raises(RuntimeError, match="TenantScope is unset"):
        TenantScope.assert_matches("default")


def test_assert_matches_raises_on_mismatch() -> None:
    with TenantScope.set("acme"):
        with pytest.raises(RuntimeError, match="TenantScope mismatch"):
            TenantScope.assert_matches("default")


def test_assert_matches_passes_on_match() -> None:
    with TenantScope.set("default"):
        TenantScope.assert_matches("default")
    with pytest.raises(RuntimeError, match="TenantScope is unset"):
        TenantScope.assert_matches("default")


def test_set_returns_context_manager_that_restores_prior() -> None:
    with TenantScope.set("acme"):
        assert TenantScope.get() == "acme"
        with TenantScope.set("widgets"):
            assert TenantScope.get() == "widgets"
        assert TenantScope.get() == "acme"


def test_clear_resets_scope() -> None:
    with TenantScope.set("acme"):
        TenantScope.clear()
        assert TenantScope.get() is None


def test_scope_is_per_task_in_asyncio() -> None:
    """`ContextVar` semantics: separate tasks get isolated scopes."""

    captured: list[str | None] = []

    async def child(name: str) -> None:
        with TenantScope.set(name):
            await asyncio.sleep(0)
            captured.append(TenantScope.get())

    async def parent() -> None:
        await asyncio.gather(child("acme"), child("widgets"))

    asyncio.run(parent())
    assert sorted(captured) == ["acme", "widgets"]
