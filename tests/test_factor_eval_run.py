from datetime import UTC, datetime

from algua.backtest._sample import SyntheticProvider
from algua.backtest.factor_eval import evaluate_factor
from algua.features.catalogue import get_factor


def test_evaluate_factor_returns_backtest_and_ic_blocks():
    spec = get_factor("xs_trailing_return")
    result = evaluate_factor(
        spec,
        SyntheticProvider(seed=0),
        datetime(2023, 1, 1, tzinfo=UTC),
        datetime(2023, 6, 30, tzinfo=UTC),
        symbols=["AAA", "BBB", "CCC"],
        params={"lookback": 10},
        construction="top_k_equal_weight",
        construction_params={"top_k": 1},
        horizon=1,
    )
    payload = result.to_dict()
    assert payload["factor"] == "xs_trailing_return"
    assert payload["standalone"] is True
    assert "metrics" in payload["backtest"]
    assert payload["backtest"]["strategy"] == "__factor__:xs_trailing_return"
    assert payload["ic"]["method"] == "spearman"
    assert payload["ic"]["fdr_corrected"] is False
    assert payload["horizon"] == 1
