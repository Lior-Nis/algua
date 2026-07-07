"""Forward-test gate criteria (#124): pure evaluation of wall-clock paper evidence.

The paper-side analog of ``algua.research.gates`` — judges whether a strategy's PAPER-lane
forward-test window earns the ``paper -> forward_tested`` transition. This module is the pure
criteria evaluator ONLY: it receives an already-assembled ``ForwardEvidence`` (built by the
protected orchestrator in ``algua/registry/forward_promotion.py`` from tick rows, the broker's
activities endpoint, and the qualified backtest gate row) and returns a JSON-clean
``ForwardGateDecision``. No I/O, no SQL, no broker calls.

CODEOWNERS-protected: the module-level threshold constants are walls, not knobs. Relaxing any of
them is human-only at the CLI (exactly like ``--allow-non-pit``); an agent may only TIGHTEN a
threshold via ``ForwardGateCriteria``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any

from algua.backtest._constants import ANN

# Standard normal for the one-sided Sharpe lower-confidence-bound (PSR) wall. stdlib only — this
# pure wall must NOT pull scipy in (research.gates owns the scipy-backed DSR path).
_NORM = NormalDist()

# Window floor: minimum daily RETURN observations in the forward window. Symmetric with
# gates.MIN_HOLDOUT_OBSERVATIONS (~one trading quarter); underpowered windows fail closed.
# Protected — NOT an agent-tunable knob (relaxation is human-only; agents may only tighten).
MIN_FORWARD_OBSERVATIONS = 63

# Coverage: decided sessions / trading sessions in [first, last] admissible tick. Sparse ticking
# lumps multi-day returns into "daily" observations and inflates the annualized mean. Protected —
# NOT an agent-tunable knob.
MIN_SESSION_COVERAGE = 0.9

# Performance bar: realized_sharpe >= max(DEGRADATION_FACTOR * holdout_sharpe, SHARPE_FLOOR).
# Direction: RAISING the factor RAISES the realized-Sharpe bar (stricter) — the name reads
# inverted (it scales the bar; it is not an "allowed degradation" budget).
# Both protected — NOT agent-tunable knobs.
DEGRADATION_FACTOR = 0.5
SHARPE_FLOOR = 0.3

# Statistical-significance wall on the realized forward Sharpe: the one-sided lower confidence
# bound on the realized Sharpe (at this confidence) must clear ZERO — equivalently the
# Probabilistic Sharpe Ratio P(true forward Sharpe > 0) must be >= this level. Mirrors the
# strategy-holdout DSR posture (gates.DSR_ALPHA=0.05 => 0.95 confidence). Power tradeoff: at
# MIN_FORWARD_OBSERVATIONS=63 this demands an observed ANNUAL Sharpe of ~3.3; the remedy for a
# marginal strategy is a LONGER forward window (the standard error shrinks with T), NOT a weaker
# bar. Protected — an agent may only RAISE it (stricter); lowering is human-only.
FORWARD_SHARPE_CONFIDENCE = 0.95  # one-sided confidence that true forward Sharpe > 0

# Volatility floor: near-zero vol makes Sharpe undefined/explosive; a do-nothing strategy must
# not pass. Protected — NOT an agent-tunable knob.
MIN_FORWARD_VOL = 0.02

# Drawdown cap over the admissible evidence series (measured by the gate itself; the kill-switch
# breaker is optional and resettable). Protected — NOT an agent-tunable knob.
MAX_FORWARD_DRAWDOWN = 0.25

# Recency: newest admissible tick must be within this many trading sessions of the gate run
# (sessions, not calendar days — long weekends must not false-fail). Protected — NOT an
# agent-tunable knob.
MAX_STALENESS_SESSIONS = 5

# Consumable-token freshness for the paper -> forward_tested edge (consumed by transitions.py).
# Protected — NOT an agent-tunable knob.
FORWARD_TOKEN_TTL_DAYS = 7

# Live-wall certificate freshness: forward_tested -> live demands a passing evaluation at most
# this many sessions old (checked by forward_promotion.py). Protected — NOT an agent-tunable knob.
CERTIFICATE_FRESH_SESSIONS = 10


@dataclass
class ForwardGateCriteria:
    """Thresholds for promoting paper -> forward_tested. Defaults are the protected walls above;
    an agent may pass TIGHTER values, never looser (relaxation flags are human-only at the CLI —
    this module never relaxes)."""

    min_forward_observations: int = MIN_FORWARD_OBSERVATIONS
    min_session_coverage: float = MIN_SESSION_COVERAGE
    degradation_factor: float = DEGRADATION_FACTOR
    sharpe_floor: float = SHARPE_FLOOR
    min_forward_vol: float = MIN_FORWARD_VOL
    max_forward_drawdown: float = MAX_FORWARD_DRAWDOWN
    max_staleness_sessions: int = MAX_STALENESS_SESSIONS
    forward_sharpe_confidence: float = FORWARD_SHARPE_CONFIDENCE


@dataclass(frozen=True)
class ForwardEvidence:
    """Assembled forward-test evidence for one strategy's window. Built by the protected
    orchestrator (``forward_promotion.py``); this module only judges it.

    ``holdout_sharpe`` is the RAW measured holdout Sharpe from the newest qualified
    ``gate_evaluations`` row (passed, PIT, no override, identity-matched) — ``None`` means no
    such row exists and the performance check fails closed. ``staleness_sessions`` is ``None``
    when there are no admissible evidence ticks at all — also fail closed.
    """

    n_return_observations: int
    session_coverage: float
    realized_sharpe: float
    # Return-series moments feeding the non-normality-adjusted Sharpe standard error (the
    # realized_sharpe_lcb / PSR wall). ``realized_kurtosis`` is RAW Pearson kurtosis (~3 for a
    # normal series — matches metrics_from_returns 'kurtosis').
    realized_skew: float
    realized_kurtosis: float
    realized_vol: float
    realized_max_drawdown: float
    holdout_sharpe: float | None
    n_reconcile_failures: int
    n_defective_ticks: int
    kill_switch_tripped: bool
    global_halt_engaged: bool
    n_kill_trips_in_window: int
    single_account_ok: bool
    activities_ok: bool
    n_external_cash_flows: int
    n_unattributable_fills: int
    staleness_sessions: int | None


@dataclass
class ForwardGateDecision:
    passed: bool
    checks: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "checks": self.checks}


_OPS: dict[str, Callable[[float, float], bool]] = {
    ">=": lambda v, t: v >= t,
    "<=": lambda v, t: v <= t,
}


def _metric_check(name: str, value: float, op: str, threshold: float,
                  detail: str | None = None) -> dict[str, Any]:
    """One metric comparison with gates.py's finite-guard: a non-finite value (inf trivially
    clears >=, NaN is never a real measurement) or threshold FAILS CLOSED, never passes — and is
    nulled out of the payload so it stays JSON-clean."""
    if not math.isfinite(threshold):
        return {"name": name, "value": None, "op": op, "threshold": None, "passed": False,
                "detail": detail}
    if not math.isfinite(value):
        return {"name": name, "value": None, "op": op, "threshold": threshold, "passed": False,
                "detail": detail}
    passed = _OPS[op](value, threshold)
    return {"name": name, "value": float(value), "op": op, "threshold": float(threshold),
            "passed": bool(passed), "detail": detail if not passed else None}


def _bool_check(name: str, ok: bool, fail_detail: str) -> dict[str, Any]:
    """One boolean precondition; ``detail`` explains the failure (None on pass)."""
    return {"name": name, "passed": bool(ok), "detail": None if ok else fail_detail}


def _forward_sharpe_psr(
    sharpe_ann: float, n_obs: int, skew: float, raw_kurtosis: float,
) -> float | None:
    """Probabilistic Sharpe Ratio for the realized forward Sharpe: P(true per-period Sharpe > 0).

    Uses the SAME Lo(2002)/Mertens non-normality variance term as
    ``algua.research.gates`` / ``algua.research.dsr.dsr_confidence`` (kept self-contained on
    purpose so this pure wall does not import the scipy-backed DSR path):

        SR   = sharpe_ann / sqrt(ANN)                       # de-annualize to per-period
        var  = 1 - skew*SR + ((rawKurt - 1)/4) * SR^2       # against a zero benchmark (SR* = 0)
        z    = SR * sqrt(T - 1) / sqrt(var)
        PSR  = Phi(z)

    ``PSR >= c`` is exactly equivalent to the one-sided c-confidence lower bound on the realized
    Sharpe clearing zero. Fails closed (returns ``None``) on any degenerate input: ``n_obs <= 1``
    (needs sqrt(T-1) > 0), any non-finite moment, or a non-finite/<= 0 variance term.
    """
    if n_obs <= 1:
        return None
    if not (math.isfinite(sharpe_ann) and math.isfinite(skew) and math.isfinite(raw_kurtosis)):
        return None
    sr = sharpe_ann / math.sqrt(ANN)
    var_term = 1.0 - skew * sr + ((raw_kurtosis - 1.0) / 4.0) * sr * sr
    if not math.isfinite(var_term) or var_term <= 0.0:
        return None
    z = sr * math.sqrt(n_obs - 1) / math.sqrt(var_term)
    conf = float(_NORM.cdf(z))
    return conf if math.isfinite(conf) else None


def evaluate_forward_gate(
    evidence: ForwardEvidence,
    criteria: ForwardGateCriteria,
) -> ForwardGateDecision:
    """Judge assembled forward-test evidence against the gate criteria. Pure; no side effects.

    Checks in spec order (#124 "Gate criteria (v1)"): window floor, session coverage, performance
    vs the degraded holdout bar, volatility floor, drawdown cap, integrity (reconcile / defective
    ticks / kill switch / global halt / kill-trip events), account hygiene (single tenant /
    activities / external cash flows / unattributable fills), staleness. ``passed`` is the
    conjunction of every check. Check names are stable — they land in decision_json audit rows.
    """
    checks: list[dict[str, Any]] = []

    # 1. Window floor — counted in RETURN observations, symmetric with the holdout floor.
    checks.append(_metric_check(
        "min_forward_observations", float(evidence.n_return_observations), ">=",
        float(criteria.min_forward_observations)))

    # 2. Session coverage — forces an essentially daily series so the ANN math is honest.
    checks.append(_metric_check(
        "session_coverage", float(evidence.session_coverage), ">=",
        float(criteria.min_session_coverage)))

    # 3. Performance — realized Sharpe vs max(factor * raw holdout Sharpe, floor). No qualified
    # backtest gate row (holdout None, or non-finite defensively) fails closed: a re-coded
    # strategy needs a fresh `research promote` before its forward test can count. The bar's
    # COMPONENTS must each be finite BEFORE max(): max() silently drops a NaN operand
    # (max(0.25, nan) == 0.25), so a NaN/inf criteria component could otherwise RELAX the bar
    # instead of failing closed.
    holdout = evidence.holdout_sharpe
    factor = float(criteria.degradation_factor)
    floor = float(criteria.sharpe_floor)
    if holdout is None or not math.isfinite(holdout):
        realized = evidence.realized_sharpe
        checks.append({
            "name": "realized_sharpe",
            "value": float(realized) if math.isfinite(realized) else None,
            "op": ">=", "threshold": None, "passed": False,
            "detail": "no qualified backtest gate row for current identity",
        })
    elif not (math.isfinite(factor) and math.isfinite(floor)):
        realized = evidence.realized_sharpe
        checks.append({
            "name": "realized_sharpe",
            "value": float(realized) if math.isfinite(realized) else None,
            "op": ">=", "threshold": None, "passed": False,
            "detail": "non-finite performance criteria (degradation_factor/sharpe_floor)",
        })
    else:
        bar = max(factor * float(holdout), floor)
        checks.append(_metric_check(
            "realized_sharpe", float(evidence.realized_sharpe), ">=", bar))

    # 3b. Statistical-significance wall on the realized Sharpe (PSR / one-sided lower bound).
    # INDEPENDENT of holdout availability: a lucky short window can clear the point performance
    # bar without the realized Sharpe being distinguishable from zero. A degenerate PSR (n<=1 or
    # non-finite moments) fails closed.
    psr = _forward_sharpe_psr(
        evidence.realized_sharpe, evidence.n_return_observations,
        evidence.realized_skew, evidence.realized_kurtosis)
    if psr is None:
        checks.append({
            "name": "realized_sharpe_lcb", "value": None, "op": ">=",
            "threshold": float(criteria.forward_sharpe_confidence), "passed": False,
            "detail": "forward Sharpe confidence undegenerate (n<=1 or non-finite moments)",
        })
    else:
        checks.append(_metric_check(
            "realized_sharpe_lcb", psr, ">=", float(criteria.forward_sharpe_confidence),
            detail="one-sided lower Sharpe bound must clear zero at this confidence (PSR)"))

    # 4. Volatility floor — a do-nothing strategy must not pass on an undefined/explosive Sharpe.
    checks.append(_metric_check(
        "min_forward_vol", float(evidence.realized_vol), ">=",
        float(criteria.min_forward_vol)))

    # 5. Drawdown cap — measured from the evidence series itself.
    checks.append(_metric_check(
        "max_forward_drawdown", float(evidence.realized_max_drawdown), "<=",
        float(criteria.max_forward_drawdown)))

    # 6. Integrity — every tick in the integrity universe reconciled and well-formed; breakers
    # currently clear; a tripped-then-resumed forward test is a FAILED forward test.
    checks.append(_bool_check(
        "reconcile_ok", evidence.n_reconcile_failures == 0,
        f"{evidence.n_reconcile_failures} reconcile failure(s) in window"))
    checks.append(_bool_check(
        "no_defective_ticks", evidence.n_defective_ticks == 0,
        f"{evidence.n_defective_ticks} defective tick(s) in window"))
    checks.append(_bool_check(
        "kill_switch_clear", not evidence.kill_switch_tripped,
        "per-strategy kill switch is tripped"))
    checks.append(_bool_check(
        "global_halt_clear", not evidence.global_halt_engaged,
        "global halt is engaged"))
    checks.append(_bool_check(
        "no_kill_trips_in_window", evidence.n_kill_trips_in_window == 0,
        f"{evidence.n_kill_trips_in_window} kill-switch trip event(s) in window"))

    # 7. Account hygiene — single-account evidence (siblings on the same account are allowed),
    # exhaustively-fetched activity history, no external capital movements, every fill
    # attributable to an order in the paper book.
    checks.append(_bool_check(
        "single_account", evidence.single_account_ok,
        "evidence spans more than one paper account in the window"))
    checks.append(_bool_check(
        "activities_ok", evidence.activities_ok,
        "broker account-activities fetch incomplete or failed (fails closed)"))
    checks.append(_bool_check(
        "no_external_cash_flows", evidence.n_external_cash_flows == 0,
        f"{evidence.n_external_cash_flows} external cash flow(s) in window"))
    checks.append(_bool_check(
        "no_unattributable_fills", evidence.n_unattributable_fills == 0,
        f"{evidence.n_unattributable_fills} unattributable fill(s) on the account"))

    # 8. Staleness — the strategy must still be actively forward testing. No admissible evidence
    # ticks at all (None) fails closed.
    if evidence.staleness_sessions is None:
        checks.append({
            "name": "max_staleness_sessions", "value": None, "op": "<=",
            "threshold": float(criteria.max_staleness_sessions), "passed": False,
            "detail": "no admissible evidence ticks",
        })
    else:
        checks.append(_metric_check(
            "max_staleness_sessions", float(evidence.staleness_sessions), "<=",
            float(criteria.max_staleness_sessions)))

    return ForwardGateDecision(passed=all(c["passed"] for c in checks), checks=checks)
