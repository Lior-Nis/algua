import math

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


def test_universe_membership_passes_in_universe_and_zeros():
    from algua.risk.limits import check_universe_membership
    # in-universe nonzero weights pass
    check_universe_membership(pd.Series({"AAA": 0.6, "BBB": 0.4}), {"AAA", "BBB"}, "s")
    # a ZERO weight for an out-of-universe symbol passes (mirrors `!= 0.0`)
    check_universe_membership(pd.Series({"AAA": 0.5, "ZZZ": 0.0}), {"AAA"}, "s")
    # empty weights pass
    check_universe_membership(pd.Series(dtype="float64"), set(), "s")


def test_universe_membership_breaches_on_out_of_universe_nonzero():
    from algua.risk.limits import RiskBreach, check_universe_membership
    with pytest.raises(RiskBreach) as ei:
        check_universe_membership(pd.Series({"AAA": 0.5, "ZZZ": 0.5}), {"AAA", "BBB"}, "s")
    assert ei.value.kind == "out_of_universe"
    assert "ZZZ" in ei.value.detail and "s" in ei.value.detail


def test_universe_membership_empty_allowed_breaches_any_nonzero():
    from algua.risk.limits import RiskBreach, check_universe_membership
    with pytest.raises(RiskBreach) as ei:
        check_universe_membership(pd.Series({"AAA": 0.5}), set(), "s")
    assert ei.value.kind == "out_of_universe"


def test_universe_membership_non_string_label_does_not_raise_typeerror():
    from algua.risk.limits import RiskBreach, check_universe_membership
    # mixed/non-string offender labels must render via key=str, not raise a bare TypeError
    with pytest.raises(RiskBreach) as ei:
        check_universe_membership(pd.Series([0.5, 0.5], index=["AAA", 7]), {"AAA"}, "s")
    assert ei.value.kind == "out_of_universe"
    assert "7" in ei.value.detail  # confirms key=str rendered the int label, not just no TypeError


def _contract(**kw):
    from algua.contracts.types import ExecutionContract
    return ExecutionContract(rebalance_frequency="1d", **kw)


def test_validate_decision_weights_runs_all_rails_in_order():
    from algua.risk.limits import RiskBreach, validate_decision_weights

    # clean long-only vector passes
    validate_decision_weights(
        pd.Series({"AAA": 0.6, "BBB": 0.4}), _contract(), "s", allowed_symbols={"AAA", "BBB"}
    )

    # finite runs first: a NaN breaches as non_finite even though it also "looks" long-only-clean
    import numpy as np
    with pytest.raises(RiskBreach) as ei_fin:
        validate_decision_weights(
            pd.Series({"AAA": np.nan}), _contract(), "s", allowed_symbols={"AAA", "BBB"}
        )
    assert ei_fin.value.kind == "non_finite_weight"

    # short policy before cap/gross: a short under default long-only breaches long_only
    with pytest.raises(RiskBreach) as ei_short:
        validate_decision_weights(
            pd.Series({"AAA": -0.3}), _contract(), "s", allowed_symbols={"AAA", "BBB"}
        )
    assert ei_short.value.kind == "long_only"

    # per-symbol cap binds (allow_short so it isn't caught by long_only first)
    with pytest.raises(RiskBreach) as ei_cap:
        validate_decision_weights(
            pd.Series({"AAA": 0.9}), _contract(max_weight_per_symbol=0.5), "s",
            allowed_symbols={"AAA", "BBB"},
        )
    assert ei_cap.value.kind == "max_weight_per_symbol"

    # gross still enforced last
    with pytest.raises(RiskBreach) as ei_gross:
        validate_decision_weights(
            pd.Series({"AAA": 0.7, "BBB": 0.7}), _contract(max_gross_exposure=1.0), "s",
            allowed_symbols={"AAA", "BBB"},
        )
    assert ei_gross.value.kind == "gross_exposure"


def test_validate_decision_weights_universe_after_finite_before_value_checks():
    from algua.risk.limits import RiskBreach, validate_decision_weights

    # clean in-universe vector passes
    validate_decision_weights(
        pd.Series({"AAA": 0.6, "BBB": 0.4}), _contract(), "s", allowed_symbols={"AAA", "BBB"}
    )
    # an out-of-universe nonzero weight breaches out_of_universe
    with pytest.raises(RiskBreach) as ei_u:
        validate_decision_weights(
            pd.Series({"AAA": 0.5, "ZZZ": 0.5}), _contract(), "s", allowed_symbols={"AAA", "BBB"}
        )
    assert ei_u.value.kind == "out_of_universe"
    # finite runs BEFORE universe: a NaN on an out-of-universe symbol surfaces non_finite first
    import numpy as np
    with pytest.raises(RiskBreach) as ei_fin:
        validate_decision_weights(
            pd.Series({"ZZZ": np.nan}), _contract(), "s", allowed_symbols={"AAA", "BBB"}
        )
    assert ei_fin.value.kind == "non_finite_weight"


def test_finite_weights_rejects_bool_dtype_and_null_labels():
    import numpy as np

    from algua.risk.limits import RiskBreach, check_finite_weights
    with pytest.raises(RiskBreach) as ei_bool:
        check_finite_weights(pd.Series({"AAA": True, "BBB": False}), "s")
    assert ei_bool.value.kind == "non_finite_weight"
    with pytest.raises(RiskBreach) as ei_null:
        check_finite_weights(pd.Series([0.5], index=[np.nan]), "s")
    assert ei_null.value.kind == "non_finite_weight"


def test_check_mark_freshness_passes_when_all_fresh():
    from algua.risk.limits import check_mark_freshness
    check_mark_freshness({"AAA": 0.0, "BBB": 1.0}, max_stale=2)


def test_check_mark_freshness_passes_on_empty_mapping():
    from algua.risk.limits import check_mark_freshness
    check_mark_freshness({}, max_stale=2)


def test_check_mark_freshness_raises_on_mixed_fresh_and_stale():
    from algua.risk.limits import RiskBreach, check_mark_freshness
    with pytest.raises(RiskBreach) as ei:
        check_mark_freshness({"AAA": 1.0, "BBB": 3.0}, max_stale=2)
    assert ei.value.kind == "stale_marks"
    assert "BBB" in ei.value.detail
    assert "stale" in ei.value.detail


def test_check_mark_freshness_raises_on_no_mark():
    from algua.risk.limits import RiskBreach, check_mark_freshness
    with pytest.raises(RiskBreach) as ei:
        check_mark_freshness({"BBB": math.inf}, max_stale=2)
    assert ei.value.kind == "stale_marks"
    assert "no_mark" in ei.value.detail


def test_check_mark_freshness_raises_on_future_dated():
    from algua.risk.limits import RiskBreach, check_mark_freshness
    with pytest.raises(RiskBreach) as ei:
        check_mark_freshness({"BBB": -1.0}, max_stale=2)
    assert ei.value.kind == "stale_marks"
    assert "future_dated" in ei.value.detail


def test_check_mark_freshness_lists_offenders_in_sorted_order():
    # Story 1.3c §7: the breach text must not depend on the caller's iteration order, so a frozen
    # child and the in-process planner (different hash seeds) report byte-identical details.
    from algua.risk.limits import RiskBreach, check_mark_freshness

    details = []
    for stale in (
        {"ZZZ": 3.0, "AAA": math.inf, "MMM": -1.0, "BBB": 1.0},
        {"BBB": 1.0, "MMM": -1.0, "AAA": math.inf, "ZZZ": 3.0},
    ):
        with pytest.raises(RiskBreach) as ei:
            check_mark_freshness(stale, max_stale=2)
        details.append(ei.value.detail)
    assert details[0] == details[1]
    assert (
        "{'AAA': 'no_mark', 'MMM': 'future_dated(-1)', 'ZZZ': 'stale(3)'}" in details[0]
    )


def test_unmappable_mark_names_the_same_symbol_whatever_the_iteration_order():
    # The planner's per-symbol session mapping stops at the first unmappable mark; which symbol
    # the breach names must not depend on the order a set happened to iterate in.
    from datetime import UTC, datetime
    from types import SimpleNamespace

    from algua.live.planner_early import assert_marks_usable
    from algua.risk.limits import RiskBreach

    def unmappable(timestamp, now):
        raise ValueError("out of calendar bounds")

    calendar = SimpleNamespace(sessions_stale=unmappable)
    ts = datetime(2023, 1, 4, tzinfo=UTC)
    details = []
    for symbols in (["ZZZ", "AAA"], ["AAA", "ZZZ"]):
        with pytest.raises(RiskBreach) as ei:
            assert_marks_usable(
                symbols, {"AAA": ts, "ZZZ": ts}, {"AAA": 1.0, "ZZZ": 1.0},
                datetime(2023, 1, 5, tzinfo=UTC), calendar,
            )
        assert ei.value.kind == "stale_marks"
        details.append(ei.value.detail)
    assert details[0] == details[1]
    assert details[0].startswith("cannot map AAA mark")


_FRESH_PROCESS_BREACH = """
from datetime import UTC, datetime
from algua.live.planner_early import assert_marks_usable
from algua.risk.limits import RiskBreach
try:
    assert_marks_usable({"AAA", "BBB", "CCC", "DDD", "EEE", "FFF"}, {}, {},
                        datetime(2023, 1, 5, tzinfo=UTC), object())
except RiskBreach as exc:
    print(exc.detail)
"""


def test_mark_breach_text_is_identical_across_fresh_processes():
    # The planner values a SET of held/consumed symbols; set order follows PYTHONHASHSEED, which a
    # fresh frozen child does not share with the supervisor. The detail must not follow it.
    import os
    import subprocess
    import sys

    details = set()
    for seed in ("0", "1", "2", "3"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        out = subprocess.run(
            [sys.executable, "-c", _FRESH_PROCESS_BREACH],
            env=env, capture_output=True, text=True, check=True, timeout=60,
        )
        details.add(out.stdout)
    assert len(details) == 1, details
    assert "{'AAA': 'no_mark', 'BBB': 'no_mark', 'CCC': 'no_mark'," in details.pop()
