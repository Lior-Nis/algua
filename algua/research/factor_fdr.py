"""Factor-level funnel-FDR correction (issue #219, slice E of #140).

Mirrors the strategy-level multiple-testing machinery from #137 / #211 one level down:

  - breadth_benchmark_t:  sqrt(2*ln(N)) expected-max inflator for the IC t-stat.
  - trial_ir_variance:    sample variance of evaluated factor IRs (the DSR
                          "trial-Sharpe spread" applied to the IC series).
  - correct_factor_ic:    produces the full `fdr` block emitted by `algua factor eval`.

Key equivalence: a factor's IR (mean_ic / ic_std) is a Sharpe ratio of the per-timestamp
IC time series.  Because IC is already per-period, dsr_confidence() applies directly —
no sqrt(ANN) scaling is needed (unlike the strategy haircut in gates.py).

Import-linter boundary: this module MAY import from algua.research.gates (reuses
dsr_confidence / DSR_ALPHA / FUNNEL_WINDOW_DAYS / effective_funnel_breadth) and from
algua.backtest (pure computation).  It MUST NOT import algua.cli or algua.registry —
those cross the seam and are wired at the CLI layer.
"""
from __future__ import annotations

import math
from typing import Any, Iterable

from algua.research.gates import DSR_ALPHA, dsr_confidence, effective_funnel_breadth

# Re-export constants callers need without opening gates.py explicitly.
__all__ = [
    "breadth_benchmark_t",
    "correct_factor_ic",
    "trial_ir_variance",
]


def breadth_benchmark_t(n_hypotheses: int) -> float:
    """Expected-max inflator for a factor IC t-stat: sqrt(2 * ln(max(N, 1))).

    0 at N=1 (a single pre-registered factor incurs no multiple-testing penalty),
    monotonically non-decreasing for N > 1.  Analogous to the strategy haircut in
    gates.sharpe_haircut() but applied directly to the t-stat (no 1/sqrt(T) — the
    t-stat already absorbs sample size via IR * sqrt(n_obs)).
    """
    n = max(int(n_hypotheses), 1)
    return math.sqrt(2.0 * math.log(n))


def trial_ir_variance(irs: Iterable[float | None]) -> float | None:
    """Sample variance (ddof=1) of finite IR values across evaluated factor hypotheses.

    Returns None when fewer than two finite values are present — fail-closed: the DSR
    binding check will not fire, so the verdict falls back to breadth-only.  A variance
    of 0.0 (all identical IRs) is valid: it means SR*=0 in the DSR formula.
    """
    finite = [v for v in irs if v is not None and math.isfinite(v)]
    if len(finite) < 2:
        return None
    n = len(finite)
    mean = sum(finite) / n
    sse = sum((x - mean) ** 2 for x in finite)
    return sse / (n - 1)


def correct_factor_ic(
    *,
    t_stat: float | None,
    ir: float | None,
    n_obs: int,
    ic_skew: float | None,
    ic_kurtosis: float | None,
    n_hypotheses: int,
    trial_ir_var: float | None,
    alpha: float = DSR_ALPHA,
) -> dict[str, Any]:
    """Compute the `fdr` block for a factor evaluation result.

    Two-layer correction (tighten-only AND semantics):

    1. Breadth layer (always-on): t_stat must exceed the expected-max inflator
       breadth_benchmark_t(n_hypotheses) plus the one-sided z-threshold at alpha.

    2. DSR layer (binds only when dispersion is measurable):
       Binding when trial_ir_var is not None AND n_hypotheses >= 2.  Calls
       dsr_confidence() with IR as the per-period Sharpe and the IC series moments
       as the non-normality adjustment.  Returns None (skip) when not binding.

    The `significant` verdict = breadth_significant AND (not dsr_binding OR dsr_significant).
    A failing DSR can only tighten a breadth-pass, never relax a breadth-fail.

    All paths set fdr_corrected: True to signal that the caller has applied the
    correction (contrast with the raw `fdr_corrected: False` from factor_ic()).
    """
    from scipy.stats import norm as _norm

    n = int(n_hypotheses)
    bm_t = breadth_benchmark_t(n)
    z_alpha = float(_norm.ppf(1.0 - alpha))

    # Breadth check.
    if t_stat is None or not math.isfinite(t_stat):
        breadth_significant = False
    else:
        breadth_significant = bool(t_stat >= z_alpha + bm_t)

    # DSR binding check.
    dsr_binding = (trial_ir_var is not None) and (n >= 2)
    dsr_conf: float | None = None
    dsr_sig: bool | None = None
    dsr_skip_reason: str | None = None

    if dsr_binding:
        # IC is per-period; no ANN scaling needed (unlike strategy Sharpe in gates.py).
        safe_ir = ir if (ir is not None and math.isfinite(ir)) else 0.0
        safe_skew = ic_skew if (ic_skew is not None and math.isfinite(ic_skew)) else 0.0
        safe_kurt = ic_kurtosis if (ic_kurtosis is not None and math.isfinite(ic_kurtosis)) else 3.0
        dsr_conf = dsr_confidence(
            sr_obs_per_period=safe_ir,
            t=n_obs,
            skew=safe_skew,
            raw_kurtosis=safe_kurt,
            n_trials=n,
            trial_sr_var_per_period=trial_ir_var,
        )
        if dsr_conf is not None:
            dsr_sig = bool(dsr_conf >= 1.0 - alpha)
        else:
            # dsr_confidence returned None (degenerate input); fail closed.
            dsr_sig = False
            dsr_skip_reason = "dsr_confidence returned None (degenerate input)"
    else:
        if trial_ir_var is None:
            dsr_skip_reason = "no measured IR dispersion (trial_ir_var is None)"
        else:
            dsr_skip_reason = "fewer than 2 distinct factor hypotheses (n_hypotheses < 2)"

    # AND-check: DSR can only tighten (never relax) the breadth verdict.
    if not breadth_significant:
        significant = False
    elif dsr_binding:
        significant = bool(dsr_sig)
    else:
        significant = breadth_significant

    return {
        "n_hypotheses": n,
        "breadth_benchmark_t": bm_t,
        "breadth_significant": breadth_significant,
        "dsr_binding": dsr_binding,
        "dsr_confidence": dsr_conf,
        "dsr_significant": dsr_sig,
        "dsr_skip_reason": dsr_skip_reason,
        "significant": significant,
        "fdr_corrected": True,
    }
