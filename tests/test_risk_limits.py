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
    check_drawdown(equity=50.0, peak=100.0, max_drawdown=1.0)  # disabled


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
