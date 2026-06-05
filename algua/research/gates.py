from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from algua.backtest._constants import ANN
from algua.backtest.walkforward import WalkForwardResult


@dataclass
class GateCriteria:
    """Thresholds for promoting backtested -> shortlisted. Holdout checks are the search-breadth
    defense (the holdout was never used during selection)."""

    min_holdout_sharpe: float = 0.5
    min_holdout_return: float = 0.0       # strict > 0
    min_pct_positive_windows: float = 0.6
    min_window_sharpe: float = 0.0        # the worst window's Sharpe must be >= this


def sharpe_haircut(n_combos: int, n_bars: int) -> float:
    """Deflated-Sharpe haircut: how many Sharpe units to add to the holdout-Sharpe bar after
    searching ``n_combos`` parameter combinations over a holdout of ``n_bars`` observations.

    Rationale (Bailey & López de Prado, "The Deflated Sharpe Ratio", 2014). Selecting the best
    of N independent trials inflates the winner's Sharpe: the expected maximum of N standard
    normals grows like ``sqrt(2 * ln(N))`` standard errors. The standard error of a *per-period*
    Sharpe estimate over T observations is ``≈ 1/sqrt(T)``. So the per-period inflation is
    ``sqrt(2 * ln(N)) / sqrt(T)``.

    UNIT MATCH (critical): the holdout Sharpe in ``algua.backtest.metrics`` is ANNUALIZED —
    ``sharpe = (mean * ANN) / (std * sqrt(ANN)) = (mean/std) * sqrt(ANN)`` — i.e. the per-period
    Sharpe scaled by ``sqrt(ANN)``. The haircut must live in the same units as the threshold it
    raises, so the per-period standard-error term is scaled by ``sqrt(ANN)`` identically:

        haircut = sqrt(2 * ln(max(N, 1))) * sqrt(ANN) / sqrt(T)

    Invariants: 0 at N=1 (``ln(1) == 0`` — no penalty for a single pre-registered hypothesis),
    monotonically non-decreasing in N, and uses the holdout sample size T (not a constant).
    """
    n = max(int(n_combos), 1)
    if n_bars <= 0:
        return 0.0
    return math.sqrt(2.0 * math.log(n)) * math.sqrt(ANN) / math.sqrt(n_bars)


@dataclass
class GateDecision:
    passed: bool
    checks: list[dict[str, Any]]
    n_combos: int | None = None
    breadth_provenance: str | None = None
    base_min_holdout_sharpe: float | None = None
    effective_min_holdout_sharpe: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": self.checks,
            "n_combos": self.n_combos,
            "breadth_provenance": self.breadth_provenance,
            "base_min_holdout_sharpe": self.base_min_holdout_sharpe,
            "effective_min_holdout_sharpe": self.effective_min_holdout_sharpe,
        }


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


_HOLDOUT_SHARPE_SPEC = "holdout_sharpe"


def evaluate_gate(
    wf: WalkForwardResult,
    criteria: GateCriteria,
    *,
    n_combos: int | None = None,
    breadth_provenance: str | None = None,
) -> GateDecision:
    """Judge a walk-forward result against the gate criteria. Pure; no side effects.

    Driven by GATE_SPECS so the metric dict and the gate stay in sync without hand editing.

    The holdout-Sharpe threshold is DEFLATED by ``sharpe_haircut(n_combos, T)`` where T is the
    holdout sample size (``wf.holdout_metrics["n_bars"]``). This is the multiple-testing defense:
    the more combos the agent searched, the higher the bar the selected strategy must clear on the
    untouched holdout. At N=1 (or no breadth) the haircut is 0 and the effective bar equals the
    base. The other gate checks are left untouched. ``breadth_provenance`` ("measured"/"declared")
    is carried into the decision for the audit trail; it does not change the math.
    """
    sources = {"holdout": wf.holdout_metrics, "stability": wf.stability}
    base_holdout_sharpe = float(criteria.min_holdout_sharpe)
    haircut = (
        sharpe_haircut(n_combos, int(wf.holdout_metrics["n_bars"]))
        if n_combos is not None
        else 0.0
    )
    effective_holdout_sharpe = base_holdout_sharpe + haircut

    checks: list[dict[str, Any]] = []
    for spec in GATE_SPECS:
        value = float(sources[spec.source][spec.metric_key])
        threshold = float(getattr(criteria, spec.threshold_attr))
        if spec.name == _HOLDOUT_SHARPE_SPEC:
            threshold = effective_holdout_sharpe
        # A non-finite metric (inf trivially clears >=/>, NaN is never a real result) is a
        # gate failure, never a pass — and is never recorded as a NaN/inf in the payload.
        if not math.isfinite(value):
            checks.append({"name": spec.name, "value": None, "threshold": threshold,
                           "op": spec.op, "passed": False})
            continue
        passed = _OPS[spec.op](value, threshold)
        checks.append({"name": spec.name, "value": value, "threshold": threshold,
                       "op": spec.op, "passed": bool(passed)})
    return GateDecision(
        passed=all(c["passed"] for c in checks),
        checks=checks,
        n_combos=n_combos,
        breadth_provenance=breadth_provenance,
        base_min_holdout_sharpe=base_holdout_sharpe,
        effective_min_holdout_sharpe=effective_holdout_sharpe,
    )
