import pandas as pd
import pytest

from algua.risk.limits import RiskBreach, check_drawdown, check_gross_exposure


def test_risk_breach_is_value_error_with_kind():
    exc = RiskBreach("gross_exposure", "too big")
    assert isinstance(exc, ValueError)
    assert exc.kind == "gross_exposure"
    assert exc.detail == "too big"


def test_gross_exposure_within_limit_passes():
    check_gross_exposure(pd.Series({"AAA": 0.6, "BBB": 0.4}), 1.0)  # == 1.0, ok
    check_gross_exposure(pd.Series(dtype="float64"), 1.0)            # empty, ok


def test_gross_exposure_over_limit_raises():
    with pytest.raises(RiskBreach) as ei:
        check_gross_exposure(pd.Series({"AAA": 1.0, "BBB": 1.0}), 1.0)
    assert ei.value.kind == "gross_exposure"


def test_drawdown_within_limit_passes():
    check_drawdown(equity=95.0, peak=100.0, max_drawdown=0.1)  # 5% < 10%
    check_drawdown(equity=50.0, peak=100.0, max_drawdown=None)  # disabled (explicit sentinel)


def test_drawdown_over_limit_raises():
    with pytest.raises(RiskBreach) as ei:
        check_drawdown(equity=80.0, peak=100.0, max_drawdown=0.1)  # 20% > 10%
    assert ei.value.kind == "drawdown"


def test_check_long_only_passes_and_raises():
    from algua.risk.limits import check_long_only

    check_long_only(pd.Series({"AAA": 0.6, "BBB": 0.4}), "s")  # ok
    check_long_only(pd.Series(dtype="float64"), "s")           # empty ok
    with pytest.raises(RiskBreach) as ei:
        check_long_only(pd.Series({"AAA": -0.5}), "s")
    assert ei.value.kind == "long_only"


def test_max_weight_per_symbol_passes_at_or_under_cap():
    from algua.risk.limits import check_max_weight_per_symbol
    check_max_weight_per_symbol(pd.Series({"AAA": 0.5, "BBB": 0.5}), 0.5)   # == cap, ok
    check_max_weight_per_symbol(pd.Series({"AAA": -0.5}), 0.5)              # short |w|==cap, ok
    check_max_weight_per_symbol(pd.Series(dtype="float64"), 0.5)           # empty, ok


def test_max_weight_per_symbol_breaches_over_cap_long_and_short():
    from algua.risk.limits import RiskBreach, check_max_weight_per_symbol
    with pytest.raises(RiskBreach) as ei_long:
        check_max_weight_per_symbol(pd.Series({"AAA": 0.6, "BBB": 0.4}), 0.5)
    assert ei_long.value.kind == "max_weight_per_symbol"
    with pytest.raises(RiskBreach) as ei_short:
        check_max_weight_per_symbol(pd.Series({"AAA": -0.6}), 0.5)
    assert ei_short.value.kind == "max_weight_per_symbol"


def test_finite_weights_passes_on_clean_series():
    from algua.risk.limits import check_finite_weights
    check_finite_weights(pd.Series({"AAA": 0.5, "BBB": -0.5}), "s")
    check_finite_weights(pd.Series(dtype="float64"), "s")


def test_finite_weights_breaches_on_nan_inf_dupes():
    import numpy as np

    from algua.risk.limits import RiskBreach, check_finite_weights
    for bad in (
        pd.Series({"AAA": np.nan}),
        pd.Series({"AAA": np.inf}),
        pd.Series({"AAA": -np.inf}),
    ):
        with pytest.raises(RiskBreach) as ei:
            check_finite_weights(bad, "s")
        assert ei.value.kind == "non_finite_weight"
    dupe = pd.Series([0.5, 0.5], index=["AAA", "AAA"])
    with pytest.raises(RiskBreach) as ei_dupe:
        check_finite_weights(dupe, "s")
    assert ei_dupe.value.kind == "non_finite_weight"


def test_short_policy_long_only_rejects_negatives():
    from algua.risk.limits import RiskBreach, check_short_policy
    check_short_policy(pd.Series({"AAA": 0.6, "BBB": 0.4}), allow_short=False, strategy_name="s")
    check_short_policy(pd.Series(dtype="float64"), allow_short=False, strategy_name="s")
    with pytest.raises(RiskBreach) as ei:
        check_short_policy(pd.Series({"AAA": -0.5}), allow_short=False, strategy_name="s")
    assert ei.value.kind == "long_only"


def test_short_policy_allows_negatives_when_allow_short():
    from algua.risk.limits import check_short_policy
    check_short_policy(pd.Series({"AAA": -0.5, "BBB": 0.5}), allow_short=True, strategy_name="s")


def _contract(**kw):
    from algua.contracts.types import ExecutionContract
    return ExecutionContract(rebalance_frequency="1d", **kw)


def test_validate_decision_weights_runs_all_rails_in_order():
    from algua.risk.limits import RiskBreach, validate_decision_weights

    # clean long-only vector passes
    validate_decision_weights(pd.Series({"AAA": 0.6, "BBB": 0.4}), _contract(), "s")

    # finite runs first: a NaN breaches as non_finite even though it also "looks" long-only-clean
    import numpy as np
    with pytest.raises(RiskBreach) as ei_fin:
        validate_decision_weights(pd.Series({"AAA": np.nan}), _contract(), "s")
    assert ei_fin.value.kind == "non_finite_weight"

    # short policy before cap/gross: a short under default long-only breaches long_only
    with pytest.raises(RiskBreach) as ei_short:
        validate_decision_weights(pd.Series({"AAA": -0.3}), _contract(), "s")
    assert ei_short.value.kind == "long_only"

    # per-symbol cap binds (allow_short so it isn't caught by long_only first)
    with pytest.raises(RiskBreach) as ei_cap:
        validate_decision_weights(
            pd.Series({"AAA": 0.9}), _contract(max_weight_per_symbol=0.5), "s"
        )
    assert ei_cap.value.kind == "max_weight_per_symbol"

    # gross still enforced last
    with pytest.raises(RiskBreach) as ei_gross:
        validate_decision_weights(
            pd.Series({"AAA": 0.7, "BBB": 0.7}), _contract(max_gross_exposure=1.0), "s"
        )
    assert ei_gross.value.kind == "gross_exposure"
