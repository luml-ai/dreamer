"""Public framework error types."""

from __future__ import annotations


class DreamerError(Exception):
    """Base class for all framework errors."""


class AuthError(DreamerError):
    """Authentication or authorization failure."""


class ConfigError(DreamerError):
    """Configuration loading, parsing, or validation failure."""


class ValidationError(DreamerError):
    """Request payload validation failure (e.g. unknown memory type, oversize)."""


class MemorySubmitError(DreamerError):
    """The shared memory-submit pipeline rejected a submission.

    ``code`` is the structured error identifier surfaced to MCP callers
    (e.g. ``type_not_allowed``, ``invalid_metadata``, ``hook_failed``).
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class LeaseHeldError(DreamerError):
    """Dream lease is already held by another worker."""


class WorkspaceError(DreamerError):
    """Store-level workspace failure (e.g. dirty checkout, unexpected branch)."""


class DreamFailedError(DreamerError):
    """A dream phase raised; orchestrator surfaces this to dream-failed hooks."""


class ProtocolComplianceError(DreamerError):
    """A configured component does not satisfy the framework Protocol contract.

    Raised at startup or by `dreamer config check`; carries a human-readable
    summary listing the slot, the component, and the specific violations
    (signature, version, capability mismatch, or multi-tenancy declaration).
    """


class TenantDataError(DreamerError):
    """Aggregated failure across `TenantData` components during a tenant
    lifecycle event (provision/deprovision/reset)."""

    def __init__(self, message: str, *, failures: list[BaseException] | None = None) -> None:
        super().__init__(message)
        self.failures: list[BaseException] = list(failures) if failures else []


__all__ = [
    "AuthError",
    "ConfigError",
    "DreamFailedError",
    "DreamerError",
    "LeaseHeldError",
    "ProtocolComplianceError",
    "TenantDataError",
    "ValidationError",
    "WorkspaceError",
]
