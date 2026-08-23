"""Backend-neutral experiment-tracking vocabulary.

``ExperimentTracker`` is the structural seam every tracker backend implements; ``TRACKING_SKIPPED``
and ``NoopTracker`` are backend-neutral too, so they live here rather than inside a concrete
adapter module. Mirrors ``algua/execution/errors.py``, which plays the same leaf role for the
broker seam: backend-neutral consumers import from here, never from ``mlflow_tracker``.

Imports nothing from ``algua`` beyond the result/contract types the Protocol's signatures need —
those are data shapes, not backend types, so depending on them keeps this module a true leaf with
respect to any concrete tracker.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from algua.backtest.result import BacktestResult
from algua.backtest.sweep import SweepResult
from algua.backtest.walkforward import WalkForwardResult

# ---------------------------------------------------------------------------
# Protocol (#45)
# ---------------------------------------------------------------------------

@runtime_checkable
class ExperimentTracker(Protocol):
    """Structural protocol for experiment loggers.

    A backend returns a real MLflow-style run id on success, or ``TRACKING_SKIPPED`` to signal it
    logged nothing (e.g. a no-op backend). It must never fabricate a plausible-looking run id on a
    skip or a failure: ``algua.tracking.record.record_tracking`` maps a real id to the payload's
    "succeeded" state, so an invented id would be indistinguishable from an actual logged run. See
    ``record_tracking`` for the full four-state contract (not-requested / succeeded / failed /
    skipped).
    """

    def log_backtest(
        self, result: BacktestResult, params: dict[str, Any], *, tracking_uri: str
    ) -> str: ...

    def log_sweep(self, result: SweepResult, *, tracking_uri: str) -> str: ...

    def log_walk_forward(
        self, result: WalkForwardResult, params: dict[str, Any], *, tracking_uri: str
    ) -> str: ...


# ---------------------------------------------------------------------------
# NoopTracker (stage 5a: wiring the #45 Protocol / PR#110 deferral)
# ---------------------------------------------------------------------------

#: Returned by :class:`NoopTracker` in place of a run id. ``record_tracking`` translates this into
#: an explicit ``mlflow_tracking_skipped`` key rather than letting a null run id masquerade as a
#: failure — the JSON contract distinguishes "backend disabled" from "backend errored".
TRACKING_SKIPPED = "__tracking_skipped__"


class NoopTracker:
    """An :class:`ExperimentTracker` that logs nothing and never raises.

    Selected with ``ALGUA_TRACKING_BACKEND=noop`` — for environments without an MLflow store, and
    as the honest second implementation that proves the seam. It deliberately does NOT invent a run
    id: fabricating one would make the payload claim a run succeeded when nothing was logged.
    """

    def log_backtest(
        self, result: BacktestResult, params: dict[str, Any], *, tracking_uri: str
    ) -> str:
        return TRACKING_SKIPPED

    def log_sweep(self, result: SweepResult, *, tracking_uri: str) -> str:
        return TRACKING_SKIPPED

    def log_walk_forward(
        self, result: WalkForwardResult, params: dict[str, Any], *, tracking_uri: str
    ) -> str:
        return TRACKING_SKIPPED
