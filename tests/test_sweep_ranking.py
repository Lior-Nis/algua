"""Tests for sweep ranking: tie-break is deterministic (lower std_sharpe wins).

Also covers non-finite score/std_sharpe (NaN, ±inf) always sorting last.
"""
import math

from algua.backtest.sweep import _rank_records


def _rec(score: float, std_sharpe: float, params: dict) -> dict:
    return {
        "params": params,
        "score": score,
        "stability": {"mean_sharpe": score, "std_sharpe": std_sharpe, "min_sharpe": 0.0,
                      "pct_positive_windows": 1.0},
        "config_hash": "abc",
        "n_windows": 4,
    }


def test_higher_score_wins():
    records = [
        _rec(0.5, 0.1, {"a": 1}),
        _rec(1.0, 0.2, {"a": 2}),
        _rec(0.3, 0.05, {"a": 3}),
    ]
    ranked = _rank_records(records)
    assert ranked[0]["params"] == {"a": 2}
    assert ranked[-1]["params"] == {"a": 3}


def test_tie_broken_by_lower_std_sharpe():
    # Two records with identical score — lower std_sharpe wins
    records = [
        _rec(1.0, 0.5, {"a": "unstable"}),
        _rec(1.0, 0.1, {"a": "stable"}),
    ]
    ranked = _rank_records(records)
    assert ranked[0]["params"] == {"a": "stable"}
    assert ranked[1]["params"] == {"a": "unstable"}


def test_ranking_is_stable_across_calls():
    records_a = [
        _rec(1.0, 0.3, {"a": 1}),
        _rec(1.0, 0.3, {"a": 2}),
    ]
    records_b = [
        _rec(1.0, 0.3, {"a": 1}),
        _rec(1.0, 0.3, {"a": 2}),
    ]
    assert _rank_records(records_a) == _rank_records(records_b)


# ---------------------------------------------------------------------------
# Non-finite score / std_sharpe — regression tests (PR #106 review comment)
# ---------------------------------------------------------------------------

def test_nan_score_sorts_last():
    """A NaN score must never beat a finite score."""
    records = [
        _rec(math.nan, 0.0, {"a": "nan_score"}),
        _rec(1.0, 0.1, {"a": "finite"}),
    ]
    ranked = _rank_records(records)
    assert ranked[0]["params"] == {"a": "finite"}
    assert ranked[-1]["params"] == {"a": "nan_score"}


def test_inf_score_does_not_rank_first():
    """+inf score is treated as degenerate and sorts last (not as a legitimate best)."""
    records = [
        _rec(math.inf, 0.0, {"a": "inf_score"}),
        _rec(1.0, 0.1, {"a": "finite"}),
    ]
    ranked = _rank_records(records)
    assert ranked[0]["params"] == {"a": "finite"}
    assert ranked[-1]["params"] == {"a": "inf_score"}


def test_nan_std_sharpe_sorts_last():
    """A NaN std_sharpe (tie-break column) must not accidentally float a record to first place."""
    records = [
        _rec(1.0, math.nan, {"a": "nan_std"}),
        _rec(1.0, 0.1, {"a": "finite_std"}),
    ]
    ranked = _rank_records(records)
    assert ranked[0]["params"] == {"a": "finite_std"}
    assert ranked[-1]["params"] == {"a": "nan_std"}


def test_multiple_non_finite_all_trail_finite():
    """All non-finite records (any mix of NaN/inf) rank after all finite records."""
    records = [
        _rec(math.nan, 0.0, {"a": "nan1"}),
        _rec(math.inf, 0.0, {"a": "inf1"}),
        _rec(-math.inf, 0.0, {"a": "neginf"}),
        _rec(0.5, 0.2, {"a": "best"}),
        _rec(0.3, 0.1, {"a": "second"}),
    ]
    ranked = _rank_records(records)
    finite_params = [r["params"]["a"] for r in ranked if math.isfinite(r["score"])]
    non_finite_params = [r["params"]["a"] for r in ranked if not math.isfinite(r["score"])]
    # All finite combos must appear before all non-finite combos
    finite_positions = [ranked.index(r) for r in ranked if math.isfinite(r["score"])]
    non_finite_positions = [ranked.index(r) for r in ranked if not math.isfinite(r["score"])]
    assert max(finite_positions) < min(non_finite_positions), (
        f"finite {finite_params} should all precede non-finite {non_finite_params}"
    )
