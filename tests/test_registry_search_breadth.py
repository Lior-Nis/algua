from algua.backtest.sweep import SweepResult
from algua.registry.db import connect, migrate
from algua.registry.search_breadth import record_search_breadth
from algua.registry.store import SqliteStrategyRepository


def _conn(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    return conn


def _sweep(n_combos: int = 3) -> SweepResult:
    return SweepResult(
        strategy="momo",
        data_source="SyntheticProvider",
        snapshot_id=None,
        timeframe="1d",
        seed=0,
        period={"start": "2022-01-01", "end": "2023-12-31"},
        windows=4,
        holdout_frac=0.2,
        grid={"lookback": [10, 20, 40]},
        n_combos=n_combos,
        rank_by="mean_sharpe",
        ranked=[],
        best=None,
    )


def test_record_search_breadth_records_and_returns_cumulative(tmp_path):
    conn = _conn(tmp_path)
    repo = SqliteStrategyRepository(conn)
    result = _sweep(n_combos=3)
    out = record_search_breadth(repo, "momo", result)
    assert out["n_combos"] == result.n_combos
    assert out["cumulative"] == result.n_combos
    out2 = record_search_breadth(repo, "momo", result)
    assert out2["cumulative"] == 2 * result.n_combos
