"""Tests for the funnel-wide trial-Sharpe dispersion-floor accessor (#221 Slice 0)."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from algua.registry.db import connect, migrate
from algua.registry.repository import FunnelFloor
from algua.registry.store import SqliteStrategyRepository


@pytest.fixture()
def make_repo(tmp_path):
    """Factory that creates a fresh in-memory-equivalent SqliteStrategyRepository."""
    def _make():
        conn = connect(tmp_path / "t.db")
        migrate(conn)
        return SqliteStrategyRepository(conn)
    return _make


def _add_trials(repo, name, triples):
    # triples: list of (count, mean, var_ann). Each becomes one search_trials row.
    for count, mean, var in triples:
        repo.record_search_trial(
            name, count, "{}",
            trial_sharpe_count=count, trial_sharpe_mean=mean, trial_sharpe_var_ann=var,
        )


def test_floor_none_below_min_strategies(make_repo):
    repo = make_repo()
    # 4 strategies < MIN_FUNNEL_FLOOR_STRATEGIES (5) -> None var, but counts still reported.
    for i in range(4):
        _add_trials(repo, f"s{i}", [(10, 1.0, 0.25)])
    floor = repo.funnel_trial_sharpe_var(90)
    assert isinstance(floor, FunnelFloor)
    assert floor.var_ann is None
    assert floor.n_strategies == 4
    assert floor.n_total_rows == 4


def test_floor_is_mean_of_per_strategy_pooled_var(make_repo):
    repo = make_repo()
    # 5 strategies, each a SINGLE row whose per-strategy pooled var == its var_ann
    # (Σn==total over one row group; pooled sample var of one (n,mean,var) group == var).
    vars_ann = [0.10, 0.20, 0.30, 0.40, 0.50]
    for i, v in enumerate(vars_ann):
        _add_trials(repo, f"s{i}", [(10, 1.0, v)])
    floor = repo.funnel_trial_sharpe_var(90)
    assert floor.n_strategies == 5
    assert floor.var_ann == pytest.approx(sum(vars_ann) / 5)


def test_anti_gaming_one_vote_per_strategy(make_repo):
    repo = make_repo()
    # One family runs 100 near-duplicate combos (low dispersion); 4 others run 1 each.
    # The big family must contribute ONE vote (its per-strategy var), not 100-count domination.
    _add_trials(repo, "big", [(1, 1.0, 0.0)] * 100)   # 100 rows, per-strategy pooled var ~0
    for i in range(4):
        _add_trials(repo, f"small{i}", [(10, 1.0, 0.40)])
    floor = repo.funnel_trial_sharpe_var(90)
    assert floor.n_strategies == 5
    # mean of [~0, 0.40, 0.40, 0.40, 0.40] == 0.32, NOT pulled toward 0 by the 100 big rows.
    assert floor.var_ann == pytest.approx((0.0 + 0.40 * 4) / 5, abs=1e-9)


def test_all_null_strategy_excluded(make_repo):
    repo = make_repo()
    for i in range(5):
        _add_trials(repo, f"good{i}", [(10, 1.0, 0.30)])
    # A strategy with a NULL stat row -> per-strategy pooled var None -> excluded entirely.
    repo.record_search_trial("bad", 10, "{}")  # no trial_sharpe_* -> NULLs
    floor = repo.funnel_trial_sharpe_var(90)
    assert floor.n_strategies == 5  # 'bad' excluded
    assert floor.var_ann == pytest.approx(0.30)


def test_window_selects_strategy_then_pools_all_its_rows(make_repo):
    """The window SELECTS strategies but does NOT slice rows.

    Positive case: a strategy with ONE recent (in-window) row AND additional OLD
    (out-of-window) rows must pool ALL its rows (in + out), while a strategy whose
    rows are ALL out-of-window is NOT selected at all.

    Timestamp injection: record_search_trial stamps created_at with _now() and
    provides no override, so we UPDATE created_at directly via SQL after inserting,
    using the underlying ``repo._conn`` handle.
    """
    repo = make_repo()
    conn = repo._conn

    now = datetime.now(UTC)
    old_ts = (now - timedelta(days=200)).isoformat()    # well outside a 90-day window
    recent_ts = (now - timedelta(days=1)).isoformat()   # inside a 90-day window

    # Strategy "in_and_old": one recent row (var=0.10, n=10) + one old row (var=0.90, n=10).
    # The recent row qualifies the strategy. Both rows are pooled for its per-strategy var.
    # Pooled sample var: two rows, same mean=1.0, vars 0.10 and 0.90, n=10 each.
    # Grand mean = 1.0. SSE = (10-1)*0.10 + 10*(1.0-1.0)^2 + (10-1)*0.90 + 10*(1.0-1.0)^2
    #            = 9*0.10 + 9*0.90 = 0.90 + 8.10 = 9.0
    # Pooled var = 9.0 / (20 - 1) = 9.0 / 19
    repo.record_search_trial("in_and_old", 10, "{}", trial_sharpe_count=10,
                              trial_sharpe_mean=1.0, trial_sharpe_var_ann=0.10)
    row_a_id = conn.execute(
        "SELECT id FROM search_trials WHERE strategy_name='in_and_old' ORDER BY id DESC LIMIT 1"
    ).fetchone()["id"]
    repo.record_search_trial("in_and_old", 10, "{}", trial_sharpe_count=10,
                              trial_sharpe_mean=1.0, trial_sharpe_var_ann=0.90)
    row_b_id = conn.execute(
        "SELECT id FROM search_trials WHERE strategy_name='in_and_old' ORDER BY id DESC LIMIT 1"
    ).fetchone()["id"]
    # Stamp timestamps: first row recent (qualifies the strategy), second row old.
    with conn:
        conn.execute("UPDATE search_trials SET created_at=? WHERE id=?", (recent_ts, row_a_id))
        conn.execute("UPDATE search_trials SET created_at=? WHERE id=?", (old_ts, row_b_id))

    # Strategy "all_old": two rows both old — must NOT be selected at all.
    repo.record_search_trial("all_old", 10, "{}", trial_sharpe_count=10,
                              trial_sharpe_mean=1.0, trial_sharpe_var_ann=0.50)
    repo.record_search_trial("all_old", 10, "{}", trial_sharpe_count=10,
                              trial_sharpe_mean=1.0, trial_sharpe_var_ann=0.50)
    with conn:
        conn.execute("UPDATE search_trials SET created_at=? WHERE strategy_name='all_old'",
                     (old_ts,))

    # 4 filler strategies each with one recent row (var=0.20) to reach the MIN=5 threshold.
    for i in range(4):
        repo.record_search_trial(f"filler{i}", 10, "{}", trial_sharpe_count=10,
                                  trial_sharpe_mean=1.0, trial_sharpe_var_ann=0.20)
        filler_id = conn.execute(
            "SELECT id FROM search_trials"
            f" WHERE strategy_name='filler{i}' ORDER BY id DESC LIMIT 1"
        ).fetchone()["id"]
        with conn:
            conn.execute("UPDATE search_trials SET created_at=? WHERE id=?",
                         (recent_ts, filler_id))

    floor = repo.funnel_trial_sharpe_var(90)

    # "all_old" was never selected — exactly 5 strategies (in_and_old + 4 fillers).
    assert floor.n_strategies == 5

    # "in_and_old" per-strategy var pools BOTH rows (old row is included despite being
    # outside the window, because the window only selected the strategy, not sliced rows).
    expected_in_and_old_var = 9.0 / 19.0
    # 4 fillers each contribute 0.20 (single-row, pooled var == the row's var_ann).
    expected_var_ann = (expected_in_and_old_var + 0.20 * 4) / 5
    assert floor.var_ann == pytest.approx(expected_var_ann, rel=1e-9)

    # Negative-window boundary: -1 days selects nothing.
    floor_neg = repo.funnel_trial_sharpe_var(-1)
    assert floor_neg.n_strategies == 0
    assert floor_neg.var_ann is None
