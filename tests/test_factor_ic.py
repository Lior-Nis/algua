import numpy as np
import pandas as pd
import pytest

from algua.backtest.factor_eval import factor_ic


def _panels(score_rows, return_rows):
    cols = ["AAA", "BBB", "CCC", "DDD"]
    idx = pd.date_range("2023-01-01", periods=len(score_rows), freq="D")
    return (pd.DataFrame(score_rows, index=idx, columns=cols),
            pd.DataFrame(return_rows, index=idx, columns=cols))


def test_perfectly_monotone_factor_has_ic_one():
    rows = [[1, 2, 3, 4], [4, 3, 2, 1], [2, 4, 1, 3]]
    scores, rets = _panels(rows, rows)  # scores rank == return rank every bar
    ic = factor_ic(scores, rets, min_cross_section=3)
    assert ic["mean_ic"] == pytest.approx(1.0)
    assert ic["hit_rate"] == pytest.approx(1.0)
    assert ic["n_obs"] == 3


def test_sign_flipped_factor_has_ic_minus_one():
    rows = [[1, 2, 3, 4], [4, 3, 2, 1]]
    scores, rets = _panels(rows, [[-v for v in r] for r in rows])
    ic = factor_ic(scores, rets, min_cross_section=3)
    assert ic["mean_ic"] == pytest.approx(-1.0)


def test_noise_factor_has_ic_near_zero():
    rng = np.random.default_rng(0)
    n = 200
    scores = pd.DataFrame(rng.normal(size=(n, 4)), columns=["AAA", "BBB", "CCC", "DDD"])
    rets = pd.DataFrame(rng.normal(size=(n, 4)), columns=["AAA", "BBB", "CCC", "DDD"])
    ic = factor_ic(scores, rets, min_cross_section=3)
    assert abs(ic["mean_ic"]) < 0.1
    assert ic["n_obs"] == n


def test_constant_cross_sections_are_skipped():
    rows = [[5, 5, 5, 5], [5, 5, 5, 5]]
    scores, rets = _panels(rows, [[1, 2, 3, 4], [4, 3, 2, 1]])
    ic = factor_ic(scores, rets, min_cross_section=3)
    assert ic["n_obs"] == 0
    assert ic["mean_ic"] is None


def test_too_few_observations_returns_none():
    rows = [[1, 2, 3, 4]]
    scores, rets = _panels(rows, rows)
    ic = factor_ic(scores, rets, min_cross_section=3)
    assert ic["n_obs"] == 1
    assert ic["ir"] is None and ic["t_stat"] is None
