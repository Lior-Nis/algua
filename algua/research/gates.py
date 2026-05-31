from __future__ import annotations

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


def _check(name: str, value: float, threshold: float, op: str) -> dict[str, Any]:
    if op == ">=":
        ok = value >= threshold
    elif op == ">":
        ok = value > threshold
    else:  # pragma: no cover - guarded by the fixed call sites below
        raise ValueError(f"unknown op {op!r}")
    return {"name": name, "value": float(value), "threshold": float(threshold),
            "op": op, "passed": bool(ok)}


def evaluate_gate(
    wf: WalkForwardResult, criteria: GateCriteria, *, n_combos: int | None = None
) -> GateDecision:
    """Judge a walk-forward result against the gate criteria. Pure; no side effects."""
    h = wf.holdout_metrics
    s = wf.stability
    checks = [
        _check("holdout_sharpe", h["sharpe"], criteria.min_holdout_sharpe, ">="),
        _check("holdout_return", h["total_return"], criteria.min_holdout_return, ">"),
        _check("pct_positive_windows", s["pct_positive_windows"],
               criteria.min_pct_positive_windows, ">="),
        _check("min_window_sharpe", s["min_sharpe"], criteria.min_window_sharpe, ">="),
    ]
    return GateDecision(passed=all(c["passed"] for c in checks), checks=checks, n_combos=n_combos)
