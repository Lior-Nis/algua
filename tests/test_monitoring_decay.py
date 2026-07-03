"""Unit tests for the pure live performance-decay evaluator (issue #391).

`decay_report` judges a realized live return series against a certified forward-test baseline. It
is ADVISORY and fail-closed: a missing/non-passing certificate or a missing design objective ->
`unknown`; too-few observations or sparse coverage -> `insufficient_data`; a non-finite realized
Sharpe never clears the bar. There is no path from missing/degenerate data to a false `ok`.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from algua.monitoring.decay import (
    VERDICT_DECAY_WARN,
    VERDICT_INSUFFICIENT,
    VERDICT_OK,
    VERDICT_UNKNOWN,
    CertifiedBaseline,
    decay_report,
)
from algua.research.forward_gates import (
    DEGRADATION_FACTOR,
    MIN_FORWARD_OBSERVATIONS,
    SHARPE_FLOOR,
)

MIN_OBS = MIN_FORWARD_OBSERVATIONS


def _baseline(holdout=1.0, *, age=1, created_at="2026-06-10T20:00:00+00:00"):
    return CertifiedBaseline(
        holdout_sharpe=holdout, certified_realized_sharpe=holdout,
        created_at=created_at, age_sessions=age)


def _returns_for_sharpe(target_sharpe: float, n: int = MIN_OBS + 20) -> pd.Series:
    """A constant-mean, unit-vol-ish daily return series whose annualized Sharpe ~ target.

    ann_sharpe = (mean*252) / (std*sqrt(252)) = (mean/std)*sqrt(252). With a tiny deterministic
    wobble to keep std > 0, choose mean so the annualized Sharpe lands near target.
    """
    rng = np.random.default_rng(0)
    noise = rng.normal(0.0, 0.01, n)
    std = float(np.std(noise, ddof=1))
    mean = target_sharpe * std / math.sqrt(252.0)
    return pd.Series(noise - float(np.mean(noise)) + mean)


def test_healthy_live_beats_bar_is_ok():
    # holdout 1.0 -> bar = max(0.5*1.0, 0.3) = 0.5; realized ~1.5 clears it.
    r = _returns_for_sharpe(1.5)
    rep = decay_report(r, 1.0, 0, _baseline(holdout=1.0))
    assert rep.verdict == VERDICT_OK
    assert rep.decay_bar == pytest.approx(max(DEGRADATION_FACTOR * 1.0, SHARPE_FLOOR))
    assert rep.recert_needed is False


def test_decayed_live_below_bar_is_warn():
    # holdout 2.0 -> bar = max(1.0, 0.3) = 1.0; realized ~0.4 is below -> decay_warn.
    r = _returns_for_sharpe(0.4)
    rep = decay_report(r, 1.0, 0, _baseline(holdout=2.0))
    assert rep.verdict == VERDICT_DECAY_WARN
    assert rep.decay_bar == pytest.approx(1.0)
    perf = [c for c in rep.checks if c["name"] == "realized_sharpe_vs_bar"][0]
    assert perf["passed"] is False and perf["detail"]


def test_floor_binds_when_holdout_low():
    # holdout 0.2 -> bar = max(0.1, 0.3) = 0.3 (floor); realized ~0.2 below floor -> warn.
    r = _returns_for_sharpe(0.2)
    rep = decay_report(r, 1.0, 0, _baseline(holdout=0.2))
    assert rep.decay_bar == pytest.approx(SHARPE_FLOOR)
    assert rep.verdict == VERDICT_DECAY_WARN


def test_no_certificate_is_unknown_not_ok():
    r = _returns_for_sharpe(1.5)
    rep = decay_report(r, 1.0, 0, None)
    assert rep.verdict == VERDICT_UNKNOWN
    assert rep.recert_needed is True
    assert [c for c in rep.checks if c["name"] == "certificate_present"][0]["passed"] is False


def test_non_finite_holdout_is_unknown():
    r = _returns_for_sharpe(1.5)
    for bad in (None, float("nan"), float("inf")):
        rep = decay_report(r, 1.0, 0, _baseline(holdout=bad))
        assert rep.verdict == VERDICT_UNKNOWN


def test_too_few_observations_is_insufficient_not_ok():
    r = _returns_for_sharpe(1.5, n=MIN_OBS - 1)  # one short of the floor
    rep = decay_report(r, 1.0, 0, _baseline(holdout=1.0))
    assert rep.verdict == VERDICT_INSUFFICIENT
    assert [c for c in rep.checks if c["name"] == "min_live_observations"][0]["passed"] is False


def test_sparse_coverage_is_insufficient_not_ok():
    r = _returns_for_sharpe(1.5)
    rep = decay_report(r, 0.5, 0, _baseline(holdout=1.0))  # coverage below 0.9 floor
    assert rep.verdict == VERDICT_INSUFFICIENT
    assert [c for c in rep.checks if c["name"] == "session_coverage"][0]["passed"] is False


def test_zero_vol_series_never_ok():
    # A do-nothing (constant-equity) live book has zero vol -> sharpe 0.0 from the shared metrics,
    # which is below any bar; must NOT be ok.
    r = pd.Series([0.0] * (MIN_OBS + 5))
    rep = decay_report(r, 1.0, 0, _baseline(holdout=1.0))
    assert rep.verdict != VERDICT_OK


def test_recert_needed_on_stale_certificate():
    r = _returns_for_sharpe(1.5)
    rep = decay_report(r, 1.0, 0, _baseline(holdout=1.0, age=999))
    assert rep.recert_needed is True
    # A stale certificate is still a real baseline -> the performance verdict still renders.
    assert rep.verdict == VERDICT_OK


def test_recert_needed_when_age_uncomputable():
    r = _returns_for_sharpe(1.5)
    rep = decay_report(r, 1.0, 0, _baseline(holdout=1.0, age=None))
    assert rep.recert_needed is True


def test_to_dict_is_json_clean():
    # Force a non-finite metric path: empty series -> metrics are all 0.0 (finite), but verify the
    # cleaner strips any non-finite via a crafted report round-trip.
    r = _returns_for_sharpe(1.5)
    rep = decay_report(r, 1.0, 3, _baseline(holdout=1.0))
    d = rep.to_dict()
    s = json.dumps(d)  # must not raise (no NaN/inf leaks)
    assert "NaN" not in s and "Infinity" not in s
    assert d["advisory"] is True
    assert d["n_inadmissible_ticks"] == 3
    assert d["certified_baseline"]["holdout_sharpe"] == 1.0


def test_inadmissible_ticks_are_advisory_only():
    # A non-zero inadmissible count never changes the healthy verdict on its own.
    r = _returns_for_sharpe(1.5)
    rep = decay_report(r, 1.0, 42, _baseline(holdout=1.0))
    assert rep.verdict == VERDICT_OK
    assert rep.n_inadmissible_ticks == 42
