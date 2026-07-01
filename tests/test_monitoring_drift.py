"""Pure-math tests for the leading-indicator drift layer (issue #343)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algua.monitoring import drift as D
from algua.monitoring.drift import (
    ALARM,
    INSUFFICIENT,
    OK,
    WARN,
    drift_report,
    mean_signal_turnover,
    membership_jaccard,
    population_stability_index,
)

SYMS = ["A", "B", "C", "D", "E", "F"]


def _panel(rows: list[list[float]], start: str = "2023-01-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(rows), freq="D", tz="UTC")
    return pd.DataFrame(rows, index=idx, columns=SYMS)


def _stable_scores(n: int, start: str) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    base = np.arange(len(SYMS), dtype=float)
    rows = [list(base + rng.normal(0, 0.05, len(SYMS))) for _ in range(n)]
    return _panel(rows, start)


def _fwd_from(scores: pd.DataFrame, sign: float, seed: int) -> pd.DataFrame:
    """Forward returns correlated (sign>0) or anti-correlated (sign<0) with scores."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 0.05, scores.shape)
    return pd.DataFrame(sign * scores.to_numpy() * 0.01 + noise * 0.001,
                        index=scores.index, columns=scores.columns)


def test_split_has_no_lookahead():
    panel = _stable_scores(40, "2023-01-01")
    rep = drift_report(panel, None)
    assert rep.reference["end"] < rep.recent["start"]
    assert rep.reference["bars"] == 20 and rep.recent["bars"] == 20


def test_stable_signal_all_ok():
    ref = _stable_scores(20, "2023-01-01")
    rec = _stable_scores(20, "2023-02-01")
    scores = pd.concat([ref, rec])
    fwd = _fwd_from(scores, sign=1.0, seed=1)
    rep = drift_report(scores, fwd)
    assert rep.verdict == OK
    assert rep.leading["signal_distribution_psi"]["status"] == OK
    assert rep.leading["turnover_drift"]["status"] == OK
    assert rep.corroborating["ic_decay"]["status"] in (OK, INSUFFICIENT)
    assert rep.note is None


def test_ic_collapse_is_corroborating_not_headline():
    # Identical score distribution both halves (tier A stable) but recent labels anti-correlate.
    ref = _stable_scores(20, "2023-01-01")
    rec = _stable_scores(20, "2023-02-01")
    scores = pd.concat([ref, rec])
    fwd_ref = _fwd_from(ref, sign=1.0, seed=2)
    fwd_rec = _fwd_from(rec, sign=-1.0, seed=3)
    fwd = pd.concat([fwd_ref, fwd_rec])
    rep = drift_report(scores, fwd, min_obs=5)
    # Leading tier A stays clean; the decay shows up ONLY in the corroborating block + a note.
    assert rep.verdict == OK
    assert rep.corroborating["ic_decay"]["status"] == ALARM
    assert rep.note is not None


def test_distribution_shape_shift_alarms_headline():
    ref = _stable_scores(20, "2023-01-01")
    # Recent: one huge outlier per bar -> post-z-score shape differs sharply from reference.
    rec_rows = [[0.0, 1.0, 2.0, 3.0, 4.0, 50.0] for _ in range(20)]
    rec = _panel(rec_rows, "2023-02-01")
    scores = pd.concat([ref, rec])
    rep = drift_report(scores, None, psi_bins=10)
    assert rep.leading["signal_distribution_psi"]["status"] == ALARM
    assert rep.verdict == ALARM


def test_rising_turnover_flags():
    # A wide CONTINUOUS cross-section (PSI is meant for such, not tiny discrete panels).
    cols = [f"S{i}" for i in range(30)]
    idx = pd.date_range("2023-01-01", periods=40, freq="D", tz="UTC")
    rng = np.random.default_rng(7)
    latent = rng.normal(0, 1, len(cols))  # fixed latent ranking for the reference era
    rows = []
    for i in range(40):
        if i < 20:  # reference: stable ordering (latent + tiny noise) -> low turnover
            rows.append(latent + rng.normal(0, 0.02, len(cols)))
        else:  # recent: fresh independent draw each bar -> scrambled ranks, SAME distribution
            rows.append(rng.normal(0, 1, len(cols)))
    scores = pd.DataFrame(rows, index=idx, columns=cols)
    rep = drift_report(scores, None)
    assert rep.leading["turnover_drift"]["status"] in (WARN, ALARM)
    # Same N(0,1) distribution both eras -> PSI must NOT alarm.
    assert rep.leading["signal_distribution_psi"]["status"] in (OK, WARN, INSUFFICIENT)


def test_near_zero_reference_turnover_does_not_explode():
    base = np.arange(len(SYMS), dtype=float)
    ref = _panel([list(base) for _ in range(20)], "2023-01-01")  # turnover exactly 0
    rec_rows = [list(base) for _ in range(20)]
    rec_rows[0] = list(base[[1, 0, 2, 3, 4, 5]])  # one tiny swap -> tiny absolute turnover
    rec = _panel(rec_rows, "2023-02-01")
    scores = pd.concat([ref, rec])
    rep = drift_report(scores, None)
    # A tiny absolute recent turnover must not be alarmed just because reference was ~0.
    assert rep.leading["turnover_drift"]["status"] != ALARM


def test_insufficient_recent_ic_is_not_false_ok():
    scores = _stable_scores(40, "2023-01-01")
    fwd = _fwd_from(scores, sign=1.0, seed=4)
    rep = drift_report(scores, fwd, min_obs=1000)  # no window can reach 1000 usable bars
    assert rep.corroborating["ic_decay"]["status"] == INSUFFICIENT
    assert rep.corroborating["hit_rate_drift"]["status"] == INSUFFICIENT


def test_no_labels_tier_b_insufficient():
    scores = _stable_scores(40, "2023-01-01")
    rep = drift_report(scores, None)
    assert rep.corroborating["ic_decay"]["status"] == INSUFFICIENT


def test_pinned_split():
    scores = _stable_scores(40, "2023-01-01")
    split = pd.Timestamp("2023-01-10", tz="UTC")  # 10 reference bars, 30 recent
    rep = drift_report(scores, None, split=split)
    assert rep.reference["bars"] == 10 and rep.recent["bars"] == 30


def test_constant_reference_psi_insufficient():
    ref = _panel([[3.0] * len(SYMS) for _ in range(20)], "2023-01-01")  # zero variance per bar
    rec = _stable_scores(20, "2023-02-01")
    assert population_stability_index(
        D._standardize_pooled(ref), D._standardize_pooled(rec), bins=10
    ) is None


def test_turnover_intersection_only():
    # Bars with disjoint scored symbols -> no >=2 overlap -> None, not spurious turnover.
    idx = pd.date_range("2023-01-01", periods=2, freq="D", tz="UTC")
    p = pd.DataFrame(
        [[1.0, 2.0, np.nan, np.nan, np.nan, np.nan],
         [np.nan, np.nan, 3.0, 4.0, np.nan, np.nan]],
        index=idx, columns=SYMS,
    )
    assert mean_signal_turnover(p) is None


def test_constant_signal_is_insufficient_not_false_ok():
    # A fully-constant signal: PSI can't bin, turnover carries no ordering, coverage is full.
    # The headline must NOT be a confident `ok` — no distribution detector actually ran.
    const = _panel([[3.0] * len(SYMS) for _ in range(40)], "2023-01-01")
    rep = drift_report(const, None)
    assert rep.leading["signal_distribution_psi"]["status"] == INSUFFICIENT
    assert rep.leading["turnover_drift"]["status"] == INSUFFICIENT
    assert rep.verdict == INSUFFICIENT


def test_disjoint_symbol_sets_flag_membership():
    # Same COUNT per bar, but the recent era trades a wholly different symbol set (universe churn).
    # Count-coverage and the identity-free pooled PSI both miss it; membership drift must catch it.
    ref = pd.DataFrame(
        np.random.default_rng(0).normal(0, 1, (20, 3)),
        index=pd.date_range("2023-01-01", periods=20, freq="D", tz="UTC"),
        columns=["A", "B", "C"],
    )
    rec = pd.DataFrame(
        np.random.default_rng(1).normal(0, 1, (20, 3)),
        index=pd.date_range("2023-01-21", periods=20, freq="D", tz="UTC"),
        columns=["X", "Y", "Z"],
    )
    scores = pd.concat([ref, rec], axis=1).sort_index()
    rep = drift_report(scores, None)
    assert membership_jaccard(scores.iloc[:20], scores.iloc[20:]) == 0.0
    assert rep.leading["membership_drift"]["status"] == ALARM
    assert rep.verdict == ALARM


def test_forward_embargo_purges_reference_labels_across_split():
    # A reference IC computed WITH the embargo purge must not silently consume recent-era labels.
    # With a large embargo the entire reference window is purged -> reference IC insufficient.
    scores = _stable_scores(40, "2023-01-01")
    fwd = _fwd_from(scores, sign=1.0, seed=9)
    rep = drift_report(scores, fwd, min_obs=1, forward_embargo=100)
    assert rep.corroborating["ic_decay"]["reference_n"] == 0
    assert rep.corroborating["ic_decay"]["status"] == INSUFFICIENT


def test_degenerate_split_raises():
    scores = _stable_scores(40, "2023-01-01")
    with pytest.raises(ValueError):
        drift_report(scores, None, reference_frac=0.0)
    with pytest.raises(ValueError):
        # split before all bars -> empty reference
        drift_report(scores, None, split=pd.Timestamp("2020-01-01", tz="UTC"))
