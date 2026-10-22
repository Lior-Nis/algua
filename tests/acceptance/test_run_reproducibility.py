"""Human-reviewable reproducibility and persisted run-provenance evidence."""
from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import allure
import pandas as pd
import pytest

from algua.backtest.engine import run
from algua.data.serve import StoreBackedProvider
from algua.data.store import DataStore
from algua.registry.db import connect, migrate
from algua.registry.run_views import run_detail_payload
from algua.registry.runs import record_backtest_run
from algua.registry.store import SqliteStrategyRepository
from algua.strategies.loader import load_strategy
from tests.acceptance.evidence import attach_comparison, attach_json
from tests.acceptance.reference_market import decision_bars

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "reference_market_v1"
SEED = 1729


class _PinnedSnapshotProvider:
    """A snapshot provider carrying the explicit random seed used by the run."""

    def __init__(self, provider: StoreBackedProvider) -> None:
        self.provider = provider
        self.snapshot_id = provider.snapshot_id
        self.seed = SEED

    def get_bars(
        self,
        symbols: list[str],
        start,
        end,
        timeframe: str = "1d",
    ) -> pd.DataFrame:
        return self.provider.get_bars(symbols, start, end, timeframe)


def _identity(result) -> dict[str, object]:
    return {
        "snapshot_id": result.snapshot_id,
        "seed": result.seed,
        "config_hash": result.config_hash,
        "code_hash_present": result.code_hash is not None,
        "dependency_hash_present": result.dependency_hash is not None,
        "timeframe": result.timeframe,
        "period": result.period,
    }


@pytest.mark.acceptance
@allure.parent_suite("Algua acceptance")
@allure.suite("Run reproducibility")
@allure.title("PROV-01 — a pinned snapshot and seed reproduce and persist the same run")
def test_prov_01_pinned_backtest_reproduces_and_persists_identity(tmp_path: Path) -> None:
    expected = json.loads(
        (FIXTURE_ROOT / "expected" / "run_reproducibility.json").read_text(
            encoding="utf-8"
        )
    )
    bars = decision_bars(periods=62)
    store = DataStore(tmp_path / "data")
    snapshot = store.ingest_bars(
        provider="acceptance-fixture",
        symbols=["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"],
        start=bars.index.min().date().isoformat(),
        end=(bars.index.max() + timedelta(days=1)).date().isoformat(),
        as_of="2026-04-01T00:00:00+00:00",
        source="reference_market_v1",
        frame=bars.reset_index().rename(columns={"timestamp": "ts"}),
        timeframe="1d",
        adjustment="none",
    )
    provider = _PinnedSnapshotProvider(StoreBackedProvider(store, snapshot.snapshot_id))
    strategy = load_strategy("cross_sectional_momentum", reload=True)
    start = bars.index.min().to_pydatetime()
    end = (bars.index.max() + timedelta(days=1)).to_pydatetime()

    first = run(strategy, provider, start, end, seed=SEED)
    replay = run(strategy, provider, start, end, seed=SEED)
    assert first.returns is not None
    assert replay.returns is not None

    conn = connect(tmp_path / "registry.db")
    migrate(conn)
    repo = SqliteStrategyRepository(conn)
    first_id = record_backtest_run(repo, strategy.name, first, params=strategy.config.params)
    replay_id = record_backtest_run(repo, strategy.name, replay, params=strategy.config.params)
    first_ledger = run_detail_payload(repo, first_id)
    replay_ledger = run_detail_payload(repo, replay_id)

    actual = {
        "requested": {"snapshot_id": snapshot.snapshot_id, "seed": SEED},
        "run_identity": _identity(first),
        "selected_metrics": {
            "total_return": first.metrics["total_return"],
            "sharpe": first.metrics["sharpe"],
            "max_drawdown": first.metrics["max_drawdown"],
        },
        "replay_checks": {
            "identity_equal": _identity(first) == _identity(replay),
            "metrics_equal": first.metrics == replay.metrics,
            "returns_equal": bool(first.returns.equals(replay.returns)),
        },
        "ledger_checks": {
            "first_snapshot_id": first_ledger["snapshot_id"],
            "replay_snapshot_id": replay_ledger["snapshot_id"],
            "first_seed": first_ledger["seed"],
            "replay_seed": replay_ledger["seed"],
            "config_hash_equal": (
                first_ledger["config_hash"] == replay_ledger["config_hash"]
                == first.config_hash
            ),
            "code_hash_equal": (
                first_ledger["code_hash"] == replay_ledger["code_hash"]
                == first.code_hash
            ),
            "dependency_hash_equal": (
                first_ledger["dependency_hash"] == replay_ledger["dependency_hash"]
                == first.dependency_hash
            ),
        },
    }
    conn.close()

    attach_json("Actual — complete first run identity", _identity(first))
    attach_json("Actual — persisted first run", first_ledger)
    attach_json("Actual — persisted replay run", replay_ledger)
    attach_comparison("Expected vs actual reproducibility", expected, actual)

    assert actual == expected
