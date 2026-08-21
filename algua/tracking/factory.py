"""Name→factory registry for experiment trackers (spec §5 item 3).

Adding a backend is one module plus one ``register_tracker`` entry — the extension point the
``ExperimentTracker`` Protocol has had since #45 without ever being wired to anything.

Selection comes from ``settings.tracking_backend``; ``mlflow`` is the default, so the behaviour of
every existing ``--track`` run is unchanged.
"""

from __future__ import annotations

from collections.abc import Callable

from algua.config.settings import get_settings
from algua.tracking.base import ExperimentTracker, NoopTracker
from algua.tracking.mlflow_tracker import MlflowTracker

_REGISTRY: dict[str, Callable[[], ExperimentTracker]] = {
    "mlflow": MlflowTracker,
    "noop": NoopTracker,
}


def register_tracker(name: str, factory: Callable[[], ExperimentTracker]) -> None:
    """Register a tracker backend under ``name`` (last registration wins)."""
    _REGISTRY[name] = factory


def get_tracker(name: str | None = None) -> ExperimentTracker:
    """The configured tracker. ``name`` overrides ``settings.tracking_backend`` (for tests).

    Unknown names fail closed with the valid set named, rather than silently falling back to a
    default — a typo'd backend must not quietly log somewhere unintended.
    """
    key = name if name is not None else get_settings().tracking_backend
    try:
        return _REGISTRY[key]()
    except KeyError:
        valid = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"unknown tracking backend {key!r}; valid: {valid}") from None
