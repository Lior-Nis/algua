from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from scipy.stats import norm as _norm

from algua.backtest._constants import ANN
from algua.backtest.walkforward import WalkForwardResult

# Funnel-level multiple-testing window (Wall A). Protected constant, not an agent-tunable knob
# (relaxing it would weaken the gate). Rolling window keeps the bar bounded; the wait-out-the-window
# trade-off is accepted and auditable via search_trials.created_at.
FUNNEL_WINDOW_DAYS = 90

# Minimum holdout sample (Wall C). A holdout with fewer observations is underpowered and fails
# closed — complements the 1/sqrt(T) haircut, which is ZERO at N=1. ~one trading quarter. Protected.
MIN_HOLDOUT_OBSERVATIONS = 63

# DSR evidence layer (#211, Phase 1). Protected constants — relaxing them weakens the gate.
DSR_ALPHA = 0.05  # require >= 95% confidence the true Sharpe beats the selection-inflated benchmark
EULER_MASCHERONI = 0.5772156649015329  # gamma_E, the DSR expected-max weight (NOT e^-1)


@dataclass
class GateCriteria:
    """Thresholds for promoting backtested -> candidate. Holdout checks are the search-breadth
    defense (the holdout was never used during selection)."""

    min_holdout_sharpe: float = 0.5
    min_holdout_return: float = 0.0       # strict > 0
    min_pct_positive_windows: float = 0.6
    min_window_sharpe: float = 0.0        # the worst window's Sharpe must be >= this
    min_holdout_observations: int = MIN_HOLDOUT_OBSERVATIONS  # Wall C: power floor, fails closed


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

    DEGENERATE HOLDOUT (T <= 0) FAILS CLOSED: a zero-length holdout carries no out-of-sample
    evidence, so the multiple-testing penalty is UNDEFINED, not zero. Returning ``inf`` lifts the
    effective holdout-Sharpe bar out of reach so the gate cannot pass on an empty holdout — the
    opposite of waiving the penalty (which returning 0.0 would silently do).

    NOTE: N is the RAW combo count with no deduplication. Correlated combos (e.g. neighboring
    parameter values that produce near-duplicate strategies) make the effective number of
    independent trials smaller, so ``sqrt(2*ln N)`` is an upper bound — the haircut errs on the
    strict side, which is intentional.
    """
    n = max(int(n_combos), 1)
    if n_bars <= 0:
        return math.inf
    return math.sqrt(2.0 * math.log(n)) * math.sqrt(ANN) / math.sqrt(n_bars)


def dsr_confidence(
    sr_obs_per_period: float,
    t: int,
    skew: float,
    raw_kurtosis: float,
    n_trials: int,
    trial_sr_var_per_period: float,
) -> float | None:
    """Deflated-Sharpe-Ratio confidence (Bailey & López de Prado): the probability — in [0,1],
    NOT a p-value — that the true (per-period) Sharpe exceeds the expected maximum Sharpe of
    ``n_trials`` selections.

        SR* = sqrt(var) * [ (1-gamma_E)*Z^-1(1-1/N) + gamma_E*Z^-1(1-1/(N*e)) ]   for N > 1
        SR* = 0                                                                    for N <= 1
        DSR = Phi( (SR_obs - SR*) * sqrt(T-1) / sqrt(1 - skew*SR_obs + (kurt-1)/4 * SR_obs^2) )

    ``raw_kurtosis`` is Pearson kurtosis (=3 for Gaussian), so the variance term reduces to the
    Lo/Mertens 1 + SR^2/2 for a normal series. Inputs are PER-PERIOD; the caller converts from the
    system's annualized Sharpes. Returns None (fail closed) on any degenerate input."""
    n = int(n_trials)
    if n < 1:                      # invalid breadth
        return None
    if t <= 1:                     # PSR needs sqrt(T-1) > 0; underpowered holdout
        return None
    if not math.isfinite(sr_obs_per_period) or not math.isfinite(skew) \
            or not math.isfinite(raw_kurtosis):
        return None
    if not math.isfinite(trial_sr_var_per_period) or trial_sr_var_per_period < 0.0:
        return None

    sr = sr_obs_per_period
    if n <= 1:
        sr_star = 0.0
    else:
        # E[max] of n trial Sharpes (Gaussian approximation), scaled by the trial-SR spread.
        sr_star = math.sqrt(trial_sr_var_per_period) * (
            (1.0 - EULER_MASCHERONI) * float(_norm.ppf(1.0 - 1.0 / n))
            + EULER_MASCHERONI * float(_norm.ppf(1.0 - 1.0 / (n * math.e)))
        )
    if not math.isfinite(sr_star):
        return None

    var_term = 1.0 - skew * sr + ((raw_kurtosis - 1.0) / 4.0) * sr * sr
    if not math.isfinite(var_term) or var_term <= 0.0:
        return None
    z = (sr - sr_star) * math.sqrt(t - 1) / math.sqrt(var_term)
    conf = float(_norm.cdf(z))
    return conf if math.isfinite(conf) else None


def effective_funnel_breadth(own_lifetime: int, windowed_total: int) -> int:
    """Effective funnel breadth fed to the haircut (Wall A): ``max`` of this strategy's LIFETIME
    recorded breadth and the funnel-wide breadth recorded in the rolling window (``windowed_total``
    INCLUDES this strategy's own windowed sweeps, so no double-count, no name-exclusion subtlety).
    An *effective funnel-breadth policy*, NOT a literal independent-trial count. A lone hypothesis
    with no siblings has ``windowed_total <= own_lifetime`` ⇒ returns ``own_lifetime`` ⇒ identical
    to the prior per-strategy behavior (no regression)."""
    return max(int(own_lifetime), int(windowed_total))


@dataclass
class GateDecision:
    passed: bool
    checks: list[dict[str, Any]]
    n_combos: int | None = None
    breadth_provenance: str | None = None
    base_min_holdout_sharpe: float | None = None
    effective_min_holdout_sharpe: float | None = None
    own_lifetime_combos: int | None = None
    windowed_total_combos: int | None = None
    funnel_window_days: int | None = None
    pit_ok: bool | None = None
    pit_override: bool = False
    dsr_binding: bool = False
    dsr_confidence: float | None = None
    dsr_skip_reason: str | None = None
    dsr_sr_star: float | None = None
    dsr_n_trials: int | None = None
    dsr_trial_sr_var_ann: float | None = None
    dsr_t: int | None = None
    dsr_skew: float | None = None
    dsr_raw_kurtosis: float | None = None

    def to_dict(self) -> dict[str, Any]:
        # A degenerate holdout drives the effective bar to inf (fail-closed); null it so the
        # payload stays JSON-clean, mirroring how non-finite check values are nulled.
        eff = self.effective_min_holdout_sharpe

        def _f(x: float | None) -> float | None:
            return x if x is None or math.isfinite(x) else None

        return {
            "passed": self.passed,
            "checks": self.checks,
            "n_combos": self.n_combos,
            "breadth_provenance": self.breadth_provenance,
            "base_min_holdout_sharpe": self.base_min_holdout_sharpe,
            "effective_min_holdout_sharpe": (
                eff if eff is None or math.isfinite(eff) else None
            ),
            "own_lifetime_combos": self.own_lifetime_combos,
            "windowed_total_combos": self.windowed_total_combos,
            "funnel_window_days": self.funnel_window_days,
            "pit_ok": self.pit_ok,
            "pit_override": self.pit_override,
            "dsr_binding": self.dsr_binding,
            "dsr_confidence": _f(self.dsr_confidence),
            "dsr_skip_reason": self.dsr_skip_reason,
            "dsr_sr_star": _f(self.dsr_sr_star),
            "dsr_n_trials": self.dsr_n_trials,
            "dsr_trial_sr_var_ann": _f(self.dsr_trial_sr_var_ann),
            "dsr_t": self.dsr_t,
            "dsr_skew": _f(self.dsr_skew),
            "dsr_raw_kurtosis": _f(self.dsr_raw_kurtosis),
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
    GateSpec("min_holdout_observations", "holdout", "n_bars", "min_holdout_observations", ">="),
)


_HOLDOUT_SHARPE_SPEC = "holdout_sharpe"


def evaluate_gate(
    wf: WalkForwardResult,
    criteria: GateCriteria,
    *,
    n_combos: int | None = None,
    breadth_provenance: str | None = None,
    pit_ok: bool,
    allow_non_pit: bool = False,
    own_lifetime_combos: int | None = None,
    windowed_total_combos: int | None = None,
    funnel_window_days: int | None = None,
    dsr_binding: bool = False,
    dsr_trial_var_ann: float | None = None,
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
        # A non-finite EFFECTIVE THRESHOLD means a degenerate (zero-length) holdout drove the
        # haircut to inf: there is no out-of-sample evidence, so the check FAILS CLOSED — never a
        # pass — and the inf threshold is nulled out of the payload to keep it JSON-clean.
        if not math.isfinite(threshold):
            checks.append({"name": spec.name, "value": None, "threshold": None,
                           "op": spec.op, "passed": False})
            continue
        # A non-finite metric (inf trivially clears >=/>, NaN is never a real result) is a
        # gate failure, never a pass — and is never recorded as a NaN/inf in the payload.
        if not math.isfinite(value):
            checks.append({"name": spec.name, "value": None, "threshold": threshold,
                           "op": spec.op, "passed": False})
            continue
        passed = _OPS[spec.op](value, threshold)
        checks.append({"name": spec.name, "value": value, "threshold": threshold,
                       "op": spec.op, "passed": bool(passed)})
    # PIT precondition (Wall B): boolean, not a metric comparison. pit_ok is computed by the
    # protected promotion orchestrator (presence + coverage). Non-PIT fails closed unless a human
    # passed allow_non_pit (recorded as an audited override).
    pit_passed = bool(pit_ok or allow_non_pit)
    pit_override = bool((not pit_ok) and allow_non_pit)
    checks.append({"name": "pit_required", "passed": pit_passed,
                   "pit_ok": bool(pit_ok), "override": "non_pit" if pit_override else None})
    # DSR evidence (#211): a tighten-only AND-check, appended ONLY when binding (measured trial
    # dispersion is available). When not binding it is omitted entirely so `passed` is unchanged.
    # Unit conversion lives here: holdout Sharpe and trial variance are ANNUALIZED; DSR per-period.
    dsr_conf: float | None = None
    dsr_sr_star: float | None = None
    dsr_skip_reason: str | None = None
    n_for_dsr = n_combos if n_combos is not None else 1
    t_hold = int(wf.holdout_metrics["n_bars"])
    skew = float(wf.holdout_metrics.get("skewness", 0.0))
    raw_kurt = float(wf.holdout_metrics.get("kurtosis", 3.0))
    sr_obs_ann = float(wf.holdout_metrics["sharpe"])
    if dsr_binding:
        var_pp = (dsr_trial_var_ann / ANN) if dsr_trial_var_ann is not None else None
        if var_pp is not None and math.isfinite(var_pp):
            dsr_conf = dsr_confidence(
                sr_obs_ann / math.sqrt(ANN), t_hold, skew, raw_kurt, n_for_dsr, var_pp)
        passed_dsr = dsr_conf is not None and dsr_conf >= (1.0 - DSR_ALPHA)
        dsr_value = dsr_conf if (dsr_conf is not None and math.isfinite(dsr_conf)) else None
        checks.append({"name": "dsr_evidence", "value": dsr_value,
                       "threshold": 1.0 - DSR_ALPHA, "op": ">=", "passed": bool(passed_dsr)})
        if dsr_conf is None:
            # measured sweep exists but stats missing -> fail closed
            dsr_skip_reason = "no_dispersion"
    else:
        dsr_skip_reason = "no_measured_dispersion"
    return GateDecision(
        passed=all(c["passed"] for c in checks),
        checks=checks,
        n_combos=n_combos,
        breadth_provenance=breadth_provenance,
        base_min_holdout_sharpe=base_holdout_sharpe,
        effective_min_holdout_sharpe=effective_holdout_sharpe,
        own_lifetime_combos=own_lifetime_combos,
        windowed_total_combos=windowed_total_combos,
        funnel_window_days=funnel_window_days,
        pit_ok=bool(pit_ok),
        pit_override=pit_override,
        dsr_binding=bool(dsr_binding),
        dsr_confidence=dsr_conf,
        dsr_skip_reason=dsr_skip_reason,
        dsr_sr_star=dsr_sr_star,
        dsr_n_trials=(n_for_dsr if dsr_binding else None),
        dsr_trial_sr_var_ann=(dsr_trial_var_ann if dsr_binding else None),
        dsr_t=(t_hold if dsr_binding else None),
        dsr_skew=(skew if dsr_binding else None),
        dsr_raw_kurtosis=(raw_kurt if dsr_binding else None),
    )
