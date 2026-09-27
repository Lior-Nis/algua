"""Stable, bounded failure taxonomy for frozen artifact commands."""
from __future__ import annotations

from algua.registry.deployment import DeploymentError


class FrozenArtifactError(ValueError):
    default_message = "frozen artifact operation failed"

    def __init__(self) -> None:
        super().__init__(self.default_message)


class FrozenSourceInvalid(FrozenArtifactError):
    default_message = "frozen source or build inputs are invalid"


class FrozenSourceDrift(FrozenArtifactError):
    default_message = "qualified source or candidate evidence drifted"


class FrozenAssetsUnsupported(FrozenArtifactError):
    default_message = "frozen planner assets are unsupported"


class FrozenBundleCorrupt(FrozenArtifactError):
    default_message = "frozen bundle is missing, corrupt, or unsafe"


class FrozenEnvironmentUnavailable(RuntimeError):
    def __init__(self) -> None:
        super().__init__("a compatible locked environment is temporarily unavailable")


class FrozenEnvironmentIncompatible(FrozenArtifactError):
    default_message = "locked environment is incompatible with the frozen contract"


class FrozenEnvironmentCorrupt(FrozenArtifactError):
    default_message = "frozen environment is missing, corrupt, or unsafe"


class FrozenDescriptorConflict(DeploymentError):
    def __init__(
        self, message: str = "frozen descriptor conflicts with immutable ledger bytes",
    ) -> None:
        super().__init__(message)


class ArtifactNotFound(LookupError):
    def __init__(self) -> None:
        super().__init__("artifact descriptor was not found")
