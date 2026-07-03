"""Advisory LIVE performance-decay monitoring (#391): realized-live vs certified-baseline.

The forward-test certificate (``forward_gate_evaluations``) is verified exactly ONCE, at the
``forward_tested -> live`` transition, and never re-checked while a strategy is live. This module
is the ongoing outcomes-analysis the live lane was missing (SR 11-7): it compares a strategy's
REALIZED live return distribution against the design objective its certificate was earned on, and
surfaces a decay / recert-needed signal.

PURE — no I/O, no SQL, no clock. The CLI (``algua monitoring decay``) does all the SQL, applies the
SAME admissibility filters the forward gate uses to the ``lane='live'`` ticks, windows them to
AFTER the certificate instant, and hands this module an already-clean return series plus the
certified baseline. This module only judges.

ADVISORY ONLY: it gates NOTHING, persists NOTHING, transitions NOTHING, and never touches the
registry, promotion/forward gates, or the live/paper order path. A ``decay_warn`` verdict is a
re-audition / recertification prompt for a human, not an enforcement action.

Fail-closed: a missing/non-passing certificate or a missing design objective yields ``unknown``;
too-few observations or sparse ticking (below the forward gate's coverage floor) yields
``insufficient_data``; a non-finite realized Sharpe never clears the bar. There is deliberately no
path from missing/degenerate data to a false ``ok``.

No look-ahead: the inputs are backward-looking realized returns from already-recorded equity ticks,
windowed strictly after the certification instant. The module never reads the future.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from algua.backtest.metrics import metrics_from_returns
from algua.research.forward_gates import (
    CERTIFICATE_FRESH_SESSIONS,
    DEGRADATION_FACTOR,
    MIN_FORWARD_OBSERVATIONS,
    MIN_SESSION_COVERAGE,
    SHARPE_FLOOR,
)

# Verdicts. `ok` is the ONLY healthy state and requires every fail-closed guard below to pass.
VERDICT_OK = "ok"
VERDICT_DECAY_WARN = "decay_warn"
VERDICT_INSUFFICIENT = "insufficient_data"
VERDICT_UNKNOWN = "unknown"


@dataclass(frozen=True)
class CertifiedBaseline:
    """The design objective a live strategy was certified against — the newest forward-gate row
    for the CURRENT identity (pass-or-fail selection is done by the caller; a non-passing newest
    row must be passed as ``baseline=None``, so a newer failed re-eval invalidates an older pass,
    exactly like the live wall).

    ``holdout_sharpe`` is the RAW measured holdout Sharpe recorded on the certificate; ``None`` (or
    non-finite) means there is no design objective to compare against and the verdict fails closed
    to ``unknown``.
    """

    holdout_sharpe: float | None
    certified_realized_sharpe: float | None
    created_at: str
    age_sessions: int | None


@dataclass
class DecayReport:
    verdict: str
    recert_needed: bool
    checks: list[dict[str, Any]] = field(default_factory=list)
    n_live_observations: int = 0
    session_coverage: float | None = None
    n_inadmissible_ticks: int = 0
    realized_sharpe: float | None = None
    realized_vol: float | None = None
    realized_max_drawdown: float | None = None
    decay_bar: float | None = None
    baseline: CertifiedBaseline | None = None

    def to_dict(self) -> dict[str, Any]:
        def _clean(x: float | None) -> float | None:
            return float(x) if x is not None and math.isfinite(x) else None

        baseline = None
        if self.baseline is not None:
            baseline = {
                "holdout_sharpe": _clean(self.baseline.holdout_sharpe),
                "certified_realized_sharpe": _clean(self.baseline.certified_realized_sharpe),
                "created_at": self.baseline.created_at,
                "age_sessions": self.baseline.age_sessions,
            }
        return {
            "verdict": self.verdict,
            "recert_needed": self.recert_needed,
            "checks": self.checks,
            "n_live_observations": self.n_live_observations,
            "session_coverage": _clean(self.session_coverage),
            "n_inadmissible_ticks": self.n_inadmissible_ticks,
            "realized_sharpe": _clean(self.realized_sharpe),
            "realized_vol": _clean(self.realized_vol),
            "realized_max_drawdown": _clean(self.realized_max_drawdown),
            "decay_bar": _clean(self.decay_bar),
            "certified_baseline": baseline,
            "advisory": True,
        }


def _check(name: str, passed: bool, detail: str | None = None) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": None if passed else detail}


def decay_report(
    live_returns: pd.Series,
    session_coverage: float,
    n_inadmissible_ticks: int,
    baseline: CertifiedBaseline | None,
    *,
    min_observations: int = MIN_FORWARD_OBSERVATIONS,
    min_session_coverage: float = MIN_SESSION_COVERAGE,
    recert_stale_sessions: int = CERTIFICATE_FRESH_SESSIONS,
) -> DecayReport:
    """Judge realized live performance against the certified baseline. Pure; no side effects.

    Guards are evaluated in order and the FIRST failing one wins, so ``ok`` requires the whole
    chain to pass: a certificate exists and passed (baseline not None); it carries a finite design
    objective; there are enough live observations; coverage clears the gate's floor; the realized
    Sharpe is finite; and finally realized Sharpe >= max(DEGRADATION_FACTOR * holdout, SHARPE_FLOOR)
    with the gate's own finite-guard on the bar's operands.

    ``recert_needed`` is independent of the performance verdict: it is True whenever the certificate
    is missing, its age is uncomputable, or it is older than ``recert_stale_sessions`` — symmetric
    with the paper lane's ``MAX_STALENESS_SESSIONS`` wall (advisory here, not enforced).
    """
    r = live_returns.dropna()
    n = int(len(r))
    m = metrics_from_returns(r)
    realized_sharpe = float(m["sharpe"])
    realized_vol = float(m["ann_volatility"])
    realized_dd = abs(float(m["max_drawdown"]))

    age = baseline.age_sessions if baseline is not None else None
    recert_needed = baseline is None or age is None or age > recert_stale_sessions

    report = DecayReport(
        verdict=VERDICT_UNKNOWN,
        recert_needed=recert_needed,
        n_live_observations=n,
        session_coverage=float(session_coverage) if math.isfinite(session_coverage) else None,
        n_inadmissible_ticks=int(n_inadmissible_ticks),
        realized_sharpe=realized_sharpe,
        realized_vol=realized_vol,
        realized_max_drawdown=realized_dd,
        baseline=baseline,
    )

    # 1. Certificate must exist and be a pass (caller passes None for a non-passing newest row).
    if baseline is None:
        report.checks.append(_check(
            "certificate_present", False,
            "no passing forward-test certificate for the current identity"))
        report.verdict = VERDICT_UNKNOWN
        return report
    report.checks.append(_check("certificate_present", True))

    # 2. Design objective must be a finite number to compare against.
    holdout = baseline.holdout_sharpe
    if holdout is None or not math.isfinite(holdout):
        report.checks.append(_check(
            "design_objective_finite", False,
            "certificate has no finite holdout Sharpe to compare against"))
        report.verdict = VERDICT_UNKNOWN
        return report
    report.checks.append(_check("design_objective_finite", True))

    # 3. Enough live observations — an underpowered live window is not a healthy verdict.
    obs_ok = n >= int(min_observations)
    report.checks.append(_check(
        "min_live_observations", obs_ok,
        f"{n} live return observation(s) < {int(min_observations)}"))
    if not obs_ok:
        report.verdict = VERDICT_INSUFFICIENT
        return report

    # 4. Session coverage — sparse ticking lumps multi-day returns into "daily" observations and
    # inflates the annualized Sharpe; the same floor the forward gate enforces.
    cov_ok = math.isfinite(session_coverage) and session_coverage >= float(min_session_coverage)
    report.checks.append(_check(
        "session_coverage", cov_ok,
        f"live session coverage {session_coverage} < {float(min_session_coverage)}"))
    if not cov_ok:
        report.verdict = VERDICT_INSUFFICIENT
        return report

    # 5. Realized Sharpe must be a real measurement — a non-finite value never clears the bar.
    if not math.isfinite(realized_sharpe):
        report.checks.append(_check(
            "realized_sharpe_finite", False, "realized live Sharpe is non-finite"))
        report.verdict = VERDICT_INSUFFICIENT
        return report
    report.checks.append(_check("realized_sharpe_finite", True))

    # 6. Performance — realized Sharpe vs the degraded certified bar. Each operand must be finite
    # BEFORE max(): max() silently drops a NaN operand, so a NaN component could otherwise RELAX
    # the bar instead of failing closed (the forward gate's rule).
    factor = float(DEGRADATION_FACTOR)
    floor = float(SHARPE_FLOOR)
    if not (math.isfinite(factor) and math.isfinite(floor)):
        report.checks.append(_check(
            "realized_sharpe_vs_bar", False, "non-finite decay-bar criteria"))
        report.verdict = VERDICT_UNKNOWN
        return report
    bar = max(factor * float(holdout), floor)
    report.decay_bar = bar
    passed = realized_sharpe >= bar
    report.checks.append(_check(
        "realized_sharpe_vs_bar", passed,
        f"realized live Sharpe {realized_sharpe:.4f} < decay bar {bar:.4f} "
        f"(0.5*holdout={factor * float(holdout):.4f}, floor={floor})"))
    report.verdict = VERDICT_OK if passed else VERDICT_DECAY_WARN
    return report
