import math

from scipy.stats import norm

from algua.backtest._constants import ANN
from algua.research.gates import (
    DSR_BOOTSTRAP_LOWER_QUANTILE,
    DSR_BOOTSTRAP_RESAMPLES,
    MAX_BOOTSTRAP_BLOCK_LEN_FRACTION,
    dsr_confidence,
    dsr_sr_star,
    dsr_sr_star_annualized,
    floored_trial_var_per_period,
)


def test_constants():
    assert DSR_BOOTSTRAP_RESAMPLES == 2000
    assert DSR_BOOTSTRAP_LOWER_QUANTILE == 0.05
    assert MAX_BOOTSTRAP_BLOCK_LEN_FRACTION == 0.5


def test_floored_var_own_first_then_max():
    assert floored_trial_var_per_period(0.04, 0.01) == 0.04        # floor below own -> own
    assert floored_trial_var_per_period(0.01, 0.09) == 0.09        # floor above own -> floor
    assert floored_trial_var_per_period(0.04, None) == 0.04        # no floor -> own
    assert floored_trial_var_per_period(-1.0, 0.09) is None        # degenerate own fails closed
    assert floored_trial_var_per_period(float("nan"), 0.09) is None


def test_dsr_sr_star_matches_inline_formula():
    assert dsr_sr_star(1, 0.04) == 0.0           # n<=1 -> 0
    assert dsr_sr_star(0, 0.04) is None          # n<1 -> None
    assert dsr_sr_star(50, -1.0) is None         # negative var -> None
    v = dsr_sr_star(50, 0.04)
    assert v is not None and v > 0.0


def test_dsr_sr_star_consistency_with_dsr_confidence():
    # dsr_confidence must use exactly dsr_sr_star internally — verify the refactor is behavior-true.
    n, var = 40, 0.04
    sr_star = dsr_sr_star(n, var)
    # Reconstruct the closed-form confidence from sr_star and compare to dsr_confidence.
    sr, t, skew, kurt = 0.12, 90, -0.2, 4.0
    var_term = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr
    z = (sr - sr_star) * math.sqrt(t - 1) / math.sqrt(var_term)
    assert dsr_confidence(sr, t, skew, kurt, n, var) == float(norm.cdf(z))


def test_dsr_sr_star_annualized_floors_and_converts():
    # annualized var / ANN, floored, then sr_star
    own_ann, floor_ann, n = 0.04 * ANN, 0.09 * ANN, 40
    assert dsr_sr_star_annualized(n, own_ann, floor_ann) == dsr_sr_star(n, 0.09)
    assert dsr_sr_star_annualized(n, None, floor_ann) is None      # no own var -> None
