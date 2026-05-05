"""Docker-mode Claude Agent runner — scaffold.

Container isolation is not implemented yet; this module keeps the symbols the
engine imports so wiring stays stable, but every call raises
``NotImplementedError``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from dreamer.contrib.dream._local import AgentRunResult, ClaudeAgentRunner


def docker_available() -> bool:
    raise NotImplementedError("docker runner not implemented")


def image_available(image: str) -> bool:
    raise NotImplementedError("docker runner not implemented")


class DockerClaudeAgentRunner(ClaudeAgentRunner):
    name = "docker"

    def __init__(
        self,
        *,
        image: str | None = None,
        entrypoint: Sequence[str] | None = None,
        claude_home: Path | None = None,
        extra_args: Sequence[str] = (),
    ) -> None:
        raise NotImplementedError("docker runner not implemented")

    async def run(
        self,
        *,
        prompt: str,
        sandbox: Path,
        timeout_seconds: float,
        env: Mapping[str, str] | None = None,
    ) -> AgentRunResult:
        raise NotImplementedError("docker runner not implemented")


__all__ = (
    "DockerClaudeAgentRunner",
    "docker_available",
    "image_available",
)
