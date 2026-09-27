"""Failure classes shared by frozen environment construction and verification."""


class EnvironmentIncompatible(ValueError):
    """Committed inputs cannot produce the normative frozen environment."""


class EnvironmentUnavailable(RuntimeError):
    """A selected compatible locked wheel cannot currently be acquired."""
