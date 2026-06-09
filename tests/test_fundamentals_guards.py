import pytest

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


def test_promotion_preflight_blocks_fundamentals(tmp_path):
    # build a registry with a fundamentals strategy at stage backtested, then preflight must refuse
    from algua.registry.promotion import promotion_preflight
    from algua.registry.transitions import transition_strategy

    repo = _repo(tmp_path)
    repo.add("fundamentals_earnings_tilt")
    # advance to backtested (HUMAN actor to skip any gate-token requirement)
    transition_strategy(repo, "fundamentals_earnings_tilt", Stage.BACKTESTED, Actor.HUMAN, "seed")
    with pytest.raises(ValueError, match="fundamentals"):
        promotion_preflight(
            repo,
            "fundamentals_earnings_tilt",
            actor=Actor.AGENT,
            declared_combos=None,
            allow_holdout_reuse=False,
            allow_non_pit=False,
        )
