"""Registry-side refusals for frozen tenants (Story 1.3c §2 and §8).

Each carries its stable snake_case ``code`` and the ``deployment_id`` it is bound to, so the CLI can
report ``{"error": code, "deployment_id": ...}`` without reading class names or messages. They are
``DeploymentError``s: an active deployment that cannot be run on this path, isolated per tenant
exactly as a working-tree descriptor failure is today. Messages are fixed text; the underlying
1.3b or decoding failure travels only as ``__cause__``.
"""
from __future__ import annotations

from typing import ClassVar

from algua.registry.deployment import DeploymentError


class FrozenTenantError(DeploymentError):
    code: ClassVar[str]
    default_message: ClassVar[str]

    def __init__(self, detail: str | None = None, *, deployment_id: int | None = None) -> None:
        self.deployment_id = deployment_id
        message = self.default_message if detail is None else f"{self.default_message}: {detail}"
        super().__init__(message)


class FrozenContentUnavailable(FrozenTenantError):
    """The descriptor, bundle or environment is missing or fails offline verification."""

    code = "frozen_content_unavailable"
    default_message = "frozen content is missing or failed offline verification"


class FrozenTenantUnsupported(FrozenTenantError):
    """The recorded content is well-formed but this supervisor cannot run it (strict decoder)."""

    code = "frozen_content_unsupported"
    default_message = "frozen content is unsupported"


class FrozenLiveUnsupported(FrozenTenantError):
    """A frozen deployment reached a checkout-verified path (the live lane)."""

    code = "frozen_live_unsupported"
    default_message = "a frozen deployment cannot run on a working-tree or live path"

    def __init__(self, deployment_id: int | None = None) -> None:
        super().__init__(deployment_id=deployment_id)
