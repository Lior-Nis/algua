from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from algua.backtest.walkforward import WalkForwardResult


@dataclass
class GateCriteria:
    """Thresholds for promoting backtested -> shortlisted. Holdout checks are the search-breadth
    defense (the holdout was never used during selection)."""

    min_holdout_sharpe: float = 0.5
    min_holdout_return: float = 0.0       # strict > 0
    min_pct_positive_windows: float = 0.6
    min_window_sharpe: float = 0.0        # the worst window's Sharpe must be >= this


@dataclass
class GateDecision:
    passed: bool
    checks: list[dict[str, Any]]
    n_combos: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "checks": self.checks, "n_combos": self.n_combos}


_OPS: dict[str, Callable[[float, float], bool]] = {
    ">=": lambda v, t: v >= t,
    ">": lambda v, t: v > t,
}


@dataclass(frozen=True)
class GateSpec:
    """Declarative description of one gate check (#40).

    Adding a gate is appending a spec here — no edits to the evaluation loop. `source` and
    `metric_key` locate the value on the WalkForwardResult; `threshold_attr` names the
    GateCriteria field; `op` is the comparison operator.
    """

    name: str
    source: str  # "holdout" -> wf.holdout_metrics, "stability" -> wf.stability
    metric_key: str
    threshold_attr: str
    op: str


GATE_SPECS: tuple[GateSpec, ...] = (
    GateSpec("holdout_sharpe", "holdout", "sharpe", "min_holdout_sharpe", ">="),
    GateSpec("holdout_return", "holdout", "total_return", "min_holdout_return", ">"),
    GateSpec("pct_positive_windows", "stability", "pct_positive_windows",
             "min_pct_positive_windows", ">="),
    GateSpec("min_window_sharpe", "stability", "min_sharpe", "min_window_sharpe", ">="),
)


def evaluate_gate(
    wf: WalkForwardResult, criteria: GateCriteria, *, n_combos: int | None = None
) -> GateDecision:
    """Judge a walk-forward result against the gate criteria. Pure; no side effects.

    Driven by GATE_SPECS so the metric dict and the gate stay in sync without hand editing.
    """
    sources = {"holdout": wf.holdout_metrics, "stability": wf.stability}
    checks: list[dict[str, Any]] = []
    for spec in GATE_SPECS:
        value = float(sources[spec.source][spec.metric_key])
        threshold = float(getattr(criteria, spec.threshold_attr))
        # A non-finite metric (inf trivially clears >=/>, NaN is never a real result) is a
        # gate failure, never a pass — and is never recorded as a NaN/inf in the payload.
        if not math.isfinite(value):
            checks.append({"name": spec.name, "value": None, "threshold": threshold,
                           "op": spec.op, "passed": False})
            continue
        passed = _OPS[spec.op](value, threshold)
        checks.append({"name": spec.name, "value": value, "threshold": threshold,
                       "op": spec.op, "passed": bool(passed)})
    return GateDecision(passed=all(c["passed"] for c in checks), checks=checks, n_combos=n_combos)
