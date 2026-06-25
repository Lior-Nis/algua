from datetime import UTC, datetime

import pytest

from algua.backtest._sample import SyntheticProvider
from algua.contracts.lifecycle import Actor, Stage
from algua.registry.db import connect, migrate
from algua.registry.store import SqliteStrategyRepository
from algua.strategies.base import assert_tradable_without_fundamentals
from algua.strategies.loader import load_strategy


def _repo(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    return SqliteStrategyRepository(conn)


def test_helper_blocks_fundamentals_strategy():
    strat = load_strategy("fundamentals_earnings_tilt")
    with pytest.raises(ValueError, match="fundamentals"):
        assert_tradable_without_fundamentals(strat)


def test_helper_allows_plain_strategy():
    assert_tradable_without_fundamentals(load_strategy("cross_sectional_momentum"))  # no raise


def test_promotion_preflight_passes_fundamentals_pit_check(tmp_path):
    # After #132 slice 4: the needs_fundamentals block is removed from promotion_preflight.
    # Preflight now proceeds past the PIT check and fails LATER at the #222 family-governance gate.
    # The paper/live guards (assert_tradable_without_fundamentals) are tested separately above.
    from algua.registry.promotion import promotion_preflight
    from algua.registry.transitions import transition_strategy

    repo = _repo(tmp_path)
    repo.add("fundamentals_earnings_tilt")
    # advance to backtested (HUMAN actor to skip any gate-token requirement)
    transition_strategy(repo, "fundamentals_earnings_tilt", Stage.BACKTESTED, Actor.HUMAN, "seed")
    # Preflight no longer refuses on needs_fundamentals; it fails later at family governance.
    with pytest.raises(ValueError, match="family registry is empty"):
        promotion_preflight(
            repo,
            "fundamentals_earnings_tilt",
            actor=Actor.AGENT,
            declared_combos=None,
            allow_holdout_reuse=False,
            allow_non_pit=False,
            provider=SyntheticProvider(seed=0),
            start=datetime(2024, 1, 1, tzinfo=UTC),
            end=datetime(2024, 6, 1, tzinfo=UTC),
        )
