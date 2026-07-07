"""Combinatorially-Symmetric Cross-Validation (CSCV) — Probability of Backtest Overfitting.

Advisory-only overfitting diagnostic (Bailey & Lopez de Prado, 2015). This module is
PURE: it imports ONLY numpy, itertools, math, dataclasses, and typing — no I/O, no algua
imports — so it never crosses a module boundary (in particular it does NOT reach the
registry or the backtest engine, honoring the "research never imports registry/backtest"
contract).

``pbo(matrix, *, rank_by)`` consumes a performance matrix of shape (N, T): N trials (rows,
the sweep-grid combos, in generated-combo order) by T periods (cols). The periods axis is
the **walk-forward windows**; the single-use holdout is excluded upstream and never enters
this computation. The result is an AGGREGATE-ONLY diagnostic — it carries no raw logits and
no per-split internals, it is NOT a promotion gate, and it mints no token.

PBO = P(the IS-optimal trial lands below the OOS median across combinatorially-symmetric
train/test splits). A high PBO means the selection rule "pick the in-sample-best combo" does
NOT generalize out-of-sample.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Sub-period bound: partition the T windows into at most this many CONTIGUOUS groups so the
# split count C(S, S/2) stays bounded (<= C(16, 8) = 12_870) regardless of T. Even by
# construction (C(S, S/2) needs an even S); an odd S is decremented by 1.
CSCV_MAX_SUBPERIODS = 16
# Below this many windows CSCV is too coarse to be meaningful — fail closed.
CSCV_MIN_WINDOWS = 4

# Kept in lockstep with algua.backtest.sweep._RANK_KEYS (duplicated, not imported, to keep this
# module import-pure). The CLI passes the SAME --rank-by it hands sweep_with_matrix().
_RANK_KEYS = {"mean_sharpe", "min_sharpe"}


@dataclass
class PboResult:
    """Aggregate-only CSCV/PBO diagnostic. Carries NO raw logits and NO per-split internals —
    those are computed inside ``pbo`` and discarded, so no per-combo/per-split selection oracle
    ever reaches an agent-readable surface."""

    pbo: float | None
    split_count: int
    trial_count: int
    window_count: int
    subperiod_count: int
    rank_by: str
    warnings: list[str] = field(default_factory=list)


def _contiguous_groups(t: int, s: int) -> list[list[int]]:
    """Partition the T window-column indices [0, T) into S CONTIGUOUS, balanced groups (sizes
    differ by <= 1; earlier groups absorb the remainder). A group is a BUNDLE OF ORIGINAL WINDOWS,
    used purely to bound the split combinatorics — there is NO group-mean cell; the reductions in
    ``pbo`` run over the constituent windows' TRUE per-window Sharpes."""
    base, rem = divmod(t, s)
    groups: list[list[int]] = []
    start = 0
    for g in range(s):
        size = base + (1 if g < rem else 0)
        groups.append(list(range(start, start + size)))
        start += size
    return groups


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """1-based ASCENDING average ranks (ties share the mean of their ordinal positions). Smallest
    value -> rank 1; largest -> rank N. Average ranks (not ordinal ``argsort(argsort())``) so a tie
    cannot bias the relative-rank omega toward an arbitrary member."""
    n = len(values)
    order = np.argsort(values, kind="stable")
    sorted_vals = values[order]
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # mean of ordinal positions i..j, 1-based
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def pbo(matrix: Any, *, rank_by: str = "mean_sharpe") -> PboResult:
    """Probability of Backtest Overfitting via bounded-S CSCV.

    Args:
        matrix: array-like coercible to a float ndarray of shape (N, T) — N trials (rows, in
            generated-combo order) by T periods/walk-forward windows (cols).
        rank_by: the IS-best selection reduction, aligned EXACTLY with sweep's ``--rank-by``:
            ``"mean_sharpe"`` (mean over the split's train windows) or ``"min_sharpe"`` (min over
            the split's train windows — sees the true worst train window, no group-mean masking).

    Returns:
        An aggregate-only :class:`PboResult`. FAILS CLOSED — ``pbo=None`` plus a warning, never a
        raise — on a non-2-D / non-coercible matrix, ``< 2`` trials (a single combo is trivially
        always-selected), ``< CSCV_MIN_WINDOWS`` windows, or any non-finite cell.

    Raises:
        ValueError: only on an invalid ``rank_by`` (an API-contract violation, not data — the CLI
            passes a value already validated against sweep's rank set).

    The IS/OOS asymmetry is intentional (Bailey & LdP): the IS-best trial per split is selected by
    the ``rank_by`` reduction over the split's TRUE per-window TRAIN Sharpes (tie-break: ascending
    train-Sharpe std over those same train windows, then lowest positional trial index — only
    within-split info + the fixed generated-combo-order row index, never cross-split/OOS/global
    values); its OOS performance is then its MEAN Sharpe over the split's TEST windows, ranked among
    all trials. ``omega = avg_rank/(N+1) in (0, 1)`` (finite logit),
    ``lambda = ln(omega/(1-omega))``, ``PBO = fraction of splits with lambda <= 0`` (IS-best
    at/below the OOS median).
    """
    if rank_by not in _RANK_KEYS:
        raise ValueError(f"rank_by must be one of {sorted(_RANK_KEYS)}, got {rank_by!r}")

    try:
        m = np.asarray(matrix, dtype=float)
    except (ValueError, TypeError):
        return PboResult(
            pbo=None, split_count=0, trial_count=0, window_count=0, subperiod_count=0,
            rank_by=rank_by,
            warnings=["matrix not coercible to a float 2-D array; PBO not computed (fail closed)"],
        )

    if m.ndim != 2:
        return PboResult(
            pbo=None, split_count=0, trial_count=0, window_count=0, subperiod_count=0,
            rank_by=rank_by,
            warnings=[f"PBO needs a 2-D (N trials x T windows) matrix; got ndim={m.ndim} "
                      "(fail closed)"],
        )

    n_trials, t_windows = int(m.shape[0]), int(m.shape[1])

    if n_trials < 2:
        return PboResult(
            pbo=None, split_count=0, trial_count=n_trials, window_count=t_windows,
            subperiod_count=0, rank_by=rank_by,
            warnings=[f"PBO needs >= 2 trials (a single combo is trivially always-selected); got "
                      f"{n_trials} (fail closed)"],
        )

    if t_windows < CSCV_MIN_WINDOWS:
        return PboResult(
            pbo=None, split_count=0, trial_count=n_trials, window_count=t_windows,
            subperiod_count=0, rank_by=rank_by,
            warnings=[f"PBO needs >= {CSCV_MIN_WINDOWS} windows; got {t_windows} (fail closed)"],
        )

    if not np.all(np.isfinite(m)):
        return PboResult(
            pbo=None, split_count=0, trial_count=n_trials, window_count=t_windows,
            subperiod_count=0, rank_by=rank_by,
            warnings=["non-finite Sharpe in matrix; PBO not computed (fail closed)"],
        )

    # Bounded, even sub-period count. Partition the T windows into S contiguous balanced groups;
    # each split assigns S/2 groups (hence their constituent windows) to train, the rest to test.
    s = min(t_windows, CSCV_MAX_SUBPERIODS)
    if s % 2 != 0:
        s -= 1  # C(S, S/2) needs an even S
    groups = _contiguous_groups(t_windows, s)
    half = s // 2
    split_count = math.comb(s, half)

    is_below_median = 0
    for train_groups in itertools.combinations(range(s), half):
        train_set = set(train_groups)
        train_cols = [c for g in train_groups for c in groups[g]]
        test_cols = [c for g in range(s) if g not in train_set for c in groups[g]]

        train_block = m[:, train_cols]  # (N, |train windows|) — TRUE per-window Sharpes
        # IS-best per rank_by, over the split's TRUE train windows (never a group mean).
        if rank_by == "mean_sharpe":
            scores = train_block.mean(axis=1)
        else:  # "min_sharpe" — sees the true worst train window (no group-mean masking)
            scores = train_block.min(axis=1)
        stds = train_block.std(axis=1)
        # Select: max score, then min train-Sharpe std, then lowest positional (generated-combo)
        # trial index — all within-split info; the index is a stable identity, never a global rank.
        # lexsort's LAST key is primary: -scores (ascending of the negation == descending score),
        # tie-broken by stds ascending, and stable order breaks a full tie by lowest index.
        n_star = int(np.lexsort((stds, -scores))[0])

        # OOS: mean Sharpe over the split's TEST windows, ranked among all trials.
        oos = m[:, test_cols].mean(axis=1)
        ranks = _average_ranks(oos)
        omega = float(ranks[n_star]) / (n_trials + 1)  # in (0, 1) -> finite logit
        logit = math.log(omega / (1.0 - omega))
        if logit <= 0.0:
            is_below_median += 1

    pbo_value = is_below_median / split_count
    return PboResult(
        pbo=float(pbo_value),
        split_count=int(split_count),
        trial_count=n_trials,
        window_count=t_windows,
        subperiod_count=s,
        rank_by=rank_by,
        warnings=[],
    )
