"""Promotion-preflight guard for needs_fundamentals strategies (issue #132, Task 3).

Uses a temp module written into the momentum family dir (same mechanism as
test_promotion_needs_news.py) because this tests a temp needs_fundamentals=True strategy.
After #132 slice 4 the PIT block is removed, so preflight now proceeds past needs_fundamentals
and fails on missing measured breadth.
"""
from __future__ import annotations

import importlib
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

import algua.strategies.momentum as _momentum_fam
from algua.backtest._sample import SyntheticProvider
from algua.contracts.lifecycle import Actor, Stage
from algua.registry.db import connect, migrate
from algua.registry.promotion import promotion_preflight
from algua.registry.store import SqliteStrategyRepository
from algua.registry.transitions import transition_strategy

_FAM_PATH = Path(_momentum_fam.__path__[0])


def _repo(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    return SqliteStrategyRepository(conn)


def _write_module(name: str, body: str) -> Path:
    path = _FAM_PATH / f"{name}.py"
    path.write_text(body)
    return path


def _clear_cache(dotted: str) -> None:
    sys.modules.pop(dotted, None)


def test_promotion_preflight_passes_fundamentals_pit_check(tmp_path):
    name = "tmp_fundamentals_promote"
    dotted = f"algua.strategies.momentum.{name}"
    body = (
        "import pandas as pd\n"
        "from algua.contracts.types import ExecutionContract\n"
        "from algua.strategies.base import StrategyConfig\n"
        f"CONFIG = StrategyConfig(name=\'{name}\', universe=[\'AAPL\'],\n"
        "    execution=ExecutionContract(rebalance_frequency=\'1d\'),\n"
        "    construction=\'equal_weight_positive\', needs_fundamentals=True)\n"
        "def signal(view, params, fundamentals):\n"
        "    return pd.Series(dtype=\'float64\')\n"
    )
    path = _write_module(name, body)
    _clear_cache(dotted)
    try:
        # Import the temp module so load_strategy can find it via the filesystem
        importlib.import_module(dotted)
        repo = _repo(tmp_path)
        repo.add(name)
        # Advance to backtested with HUMAN actor (skips the gate-token requirement)
        transition_strategy(repo, name, Stage.BACKTESTED, Actor.HUMAN, "seed")
        # After #132 slice 4: needs_fundamentals no longer blocks promotion. Preflight proceeds
        # past the (removed) PIT block and fails LATER on missing measured breadth — proving the
        # PIT block is gone.
        with pytest.raises(ValueError, match="no recorded search breadth"):
            promotion_preflight(
                repo,
                name,
                actor=Actor.AGENT,
                declared_combos=None,
                allow_holdout_reuse=False,
                allow_non_pit=False,
                provider=SyntheticProvider(seed=0),
                start=datetime(2024, 1, 1, tzinfo=UTC),
                end=datetime(2024, 6, 1, tzinfo=UTC),
            )
    finally:
        path.unlink(missing_ok=True)
        _clear_cache(dotted)
