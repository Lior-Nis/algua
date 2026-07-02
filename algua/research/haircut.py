"""Deflated-Sharpe haircut — the multiple-testing Sharpe-bar inflation (from gates.py, #335).

Pure-maths leaf: imports only the backtest annualization constant. No I/O, no research-internal
imports.
"""
from __future__ import annotations

import math

from algua.backtest._constants import ANN


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
