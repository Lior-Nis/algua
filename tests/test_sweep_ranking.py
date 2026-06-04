"""Tests for #49: sweep ranking tie-break is deterministic (lower std_sharpe wins)."""
from algua.backtest.sweep import _rank_records


def _rec(score: float, std_sharpe: float, params: dict) -> dict:
    return {
        "params": params,
        "score": score,
        "stability": {"mean_sharpe": score, "std_sharpe": std_sharpe, "min_sharpe": 0.0,
                      "pct_positive_windows": 1.0},
        "config_hash": "abc",
        "n_windows": 4,
        "holdout": {"n_bars": 10, "sharpe": 0.5, "total_return": 0.1, "max_drawdown": -0.05},
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
