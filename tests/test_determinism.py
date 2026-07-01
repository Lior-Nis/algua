"""Same-process reproducibility guard (#341, item 4).

Proves that one seeded ``SyntheticProvider`` backtest, run twice in the SAME process with the SAME
parquet writer version, yields byte-identical ``series.parquet`` bytes and an identical result
dict. This is a deliberately NARROW same-process / single-writer-version determinism check: it does
NOT prove cross-platform parquet byte stability, snapshot-ingest determinism, MLflow-artifact
determinism, or determinism of arbitrary strategies. It is the CI backstop for a
reproducibility-first system that previously had no "two runs => identical output" assertion.

Bytes are compared in memory (no file writes) so the test cannot dirty the workspace or perturb the
git-based ``code_hash`` stamp between the two runs.
"""

from datetime import UTC, datetime
from typing import Any

import pandas as pd

from algua.backtest._sample import SyntheticProvider
from algua.backtest.engine import run
from algua.backtest.result import series_frame
from algua.contracts.types import ExecutionContract
from algua.data.files import frame_to_parquet_bytes
from algua.strategies.base import LoadedStrategy, StrategyConfig

START = datetime(2024, 1, 1, tzinfo=UTC)
END = datetime(2024, 4, 1, tzinfo=UTC)
SEED = 7


def _equal_weight_strategy() -> LoadedStrategy:
    cfg = StrategyConfig(
        name="ew",
        universe=["AAA", "BBB"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
        params={},
        construction="passthrough",
    )

    def signal(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        syms = view["symbol"].unique()
        return pd.Series(1.0 / len(syms), index=sorted(syms))

    def passthrough(scores: pd.Series, view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        return scores

    return LoadedStrategy(config=cfg, signal_fn=signal, construct_fn=passthrough)


def _run() -> Any:
    return run(_equal_weight_strategy(), SyntheticProvider(seed=SEED), START, END)


def test_two_seeded_runs_produce_byte_identical_series_parquet() -> None:
    def series_bytes() -> bytes:
        frame, metadata = series_frame(_run())
        return frame_to_parquet_bytes(frame, metadata)

    assert series_bytes() == series_bytes()


def test_two_seeded_runs_produce_identical_result_dict() -> None:
    assert _run().to_dict() == _run().to_dict()
