from datetime import UTC, date, datetime
from pathlib import Path

import pytest

import algua.strategies.momentum as momentum_pkg
from algua.backtest._sample import SyntheticProvider
from algua.backtest.engine import BacktestError
from algua.contracts.lifecycle import Actor, Stage, TransitionError
from algua.registry.db import connect, migrate
from algua.registry.promotion import (
    guard_agent_relaxations,
    promotion_preflight,
    resolve_pit_ok,
)
from algua.registry.store import SqliteStrategyRepository

_START = datetime(2024, 1, 1, tzinfo=UTC)
_END = datetime(2024, 6, 1, tzinfo=UTC)


def _repo(tmp_path):
    conn = connect(tmp_path / "p.db")
    migrate(conn)
    return SqliteStrategyRepository(conn)


def test_guard_agent_relaxations_refuses_agent():
    for kw in (dict(declared_combos=9, allow_holdout_reuse=False, allow_non_pit=False),
               dict(declared_combos=None, allow_holdout_reuse=True, allow_non_pit=False),
               dict(declared_combos=None, allow_holdout_reuse=False, allow_non_pit=True)):
        with pytest.raises(ValueError, match="human"):
            guard_agent_relaxations(Actor.AGENT, **kw)


def test_guard_allows_clean_agent_and_any_human():
    guard_agent_relaxations(Actor.AGENT, declared_combos=None, allow_holdout_reuse=False,
                            allow_non_pit=False)
    guard_agent_relaxations(Actor.HUMAN, declared_combos=9, allow_holdout_reuse=True,
                            allow_non_pit=True)


def test_resolve_pit_ok_requires_coverage():
    cover = [{"snapshot_id": "u1", "effective_date": "2021-06-01"}]
    late = [{"snapshot_id": "u1", "effective_date": "2022-06-01"}]
    assert resolve_pit_ok("sp", cover, date(2022, 1, 1)) is True
    assert resolve_pit_ok("sp", late, date(2022, 1, 1)) is False
    assert resolve_pit_ok(None, None, date(2022, 1, 1)) is False


def test_resolve_pit_ok_fails_closed_on_malformed_snapshot():
    # A malformed/missing effective_date must NOT raise (the holdout was already recorded) — it
    # fails closed to non-PIT (not promotable).
    missing = [{"snapshot_id": "u1"}]  # KeyError
    bad_format = [{"snapshot_id": "u1", "effective_date": "not-a-date"}]  # ValueError
    wrong_type = [{"snapshot_id": "u1", "effective_date": None}]  # TypeError
    assert resolve_pit_ok("sp", missing, date(2022, 1, 1)) is False
    assert resolve_pit_ok("sp", bad_format, date(2022, 1, 1)) is False
    assert resolve_pit_ok("sp", wrong_type, date(2022, 1, 1)) is False


@pytest.mark.parametrize("stages", [
    (),                                              # idea
    (Stage.BACKTESTED, Stage.CANDIDATE),           # candidate
    (Stage.BACKTESTED, Stage.CANDIDATE, Stage.PAPER),  # paper (PAPER->CANDIDATE is legal!)
])
def test_preflight_refuses_non_backtested_source(tmp_path, stages):
    repo = _repo(tmp_path)
    rec = repo.add("alpha")
    repo.record_search_trial("alpha", 4, "{}")  # measured breadth present (so stage is the refusal)
    for s in stages:
        rec = repo.apply_transition(rec, s, Actor.HUMAN, "setup")
    with pytest.raises(TransitionError, match="backtested"):
        promotion_preflight(repo, "alpha", actor=Actor.AGENT, declared_combos=None,
                            allow_holdout_reuse=False, allow_non_pit=False,
                            provider=SyntheticProvider(seed=0), start=_START, end=_END)
    # Refused in preflight => no gate row, no holdout burn.
    assert repo._conn.execute("SELECT COUNT(*) c FROM gate_evaluations").fetchone()["c"] == 0
    assert repo._conn.execute("SELECT COUNT(*) c FROM holdout_evaluations").fetchone()["c"] == 0


def test_preflight_refuses_system_actor_before_any_holdout_or_gate_row(tmp_path):
    # SYSTEM is not a valid promote actor. The refusal is the FIRST check in preflight (before the
    # relaxation guard and before walk_forward), so no holdout is burned and no gate row is minted.
    repo = _repo(tmp_path)
    rec = repo.add("alpha")
    repo.apply_transition(rec, Stage.BACKTESTED, Actor.AGENT, "bt")
    repo.record_search_trial("alpha", 4, "{}")  # measured breadth present (isolate the actor check)
    with pytest.raises(ValueError, match="agent or human"):
        promotion_preflight(repo, "alpha", actor=Actor.SYSTEM, declared_combos=None,
                            allow_holdout_reuse=False, allow_non_pit=False,
                            provider=SyntheticProvider(seed=0), start=_START, end=_END)
    assert repo._conn.execute("SELECT COUNT(*) c FROM gate_evaluations").fetchone()["c"] == 0
    assert repo._conn.execute("SELECT COUNT(*) c FROM holdout_evaluations").fetchone()["c"] == 0


def test_preflight_refuses_agent_without_measured_breadth(tmp_path):
    repo = _repo(tmp_path)
    rec = repo.add("alpha")
    repo.apply_transition(rec, Stage.BACKTESTED, Actor.AGENT, "bt")
    with pytest.raises(ValueError, match="search breadth"):
        promotion_preflight(repo, "alpha", actor=Actor.AGENT, declared_combos=None,
                            allow_holdout_reuse=False, allow_non_pit=False,
                            provider=SyntheticProvider(seed=0), start=_START, end=_END)


def test_preflight_resolves_measured_funnel_breadth(tmp_path):
    repo = _repo(tmp_path)
    rec = repo.add("alpha")
    repo.apply_transition(rec, Stage.BACKTESTED, Actor.AGENT, "bt")
    repo.record_search_trial("alpha", 4, "{}")
    repo.record_search_trial("beta", 10, "{}")  # sibling effort raises the funnel bar
    ctx = promotion_preflight(repo, "alpha", actor=Actor.AGENT, declared_combos=None,
                              allow_holdout_reuse=False, allow_non_pit=False,
                              provider=SyntheticProvider(seed=0), start=_START, end=_END)
    assert ctx.own == 4 and ctx.windowed_total == 14 and ctx.n_funnel == 14
    assert ctx.provenance == "measured"


def _write_tmp_strategy(filename: str, body: str) -> Path:
    """Write a temp strategy module into the momentum family dir so load_strategy can find it.
    Caller must unlink it in a finally block (mirrors tests/test_fast_path.py loader tests)."""
    p = Path(momentum_pkg.__path__[0]) / filename
    p.write_text(body)
    return p


def test_preflight_refuses_divergent_signal_panel_before_holdout(tmp_path):
    """A strategy whose signal_panel diverges from its per-bar signal is refused in preflight --
    before any holdout or gate row -- by the exhaustive parity gate (#178)."""
    mod = _write_tmp_strategy(
        "tmp_divergent_panel.py",
        "from typing import Any\n"
        "import pandas as pd\n"
        "from algua.contracts.types import ExecutionContract\n"
        "from algua.strategies.base import StrategyConfig\n"
        "CONFIG = StrategyConfig(name='tmp_divergent_panel', universe=['AAA', 'BBB'],\n"
        "    execution=ExecutionContract("
        "rebalance_frequency='1d', decision_lag_bars=1, warmup_bars=0),\n"
        "    construction='equal_weight_positive')\n"
        "def signal(view, params):\n"
        "    syms = sorted(view['symbol'].unique())\n"
        "    return pd.Series(1.0, index=syms)\n"
        "def signal_panel(bars, params):\n"
        "    adj = bars.reset_index().pivot("
        "index='timestamp', columns='symbol', values='adj_close')\n"
        "    out = pd.DataFrame(0.0, index=adj.index, columns=adj.columns)\n"
        "    out['AAA'] = 1.0\n"
        "    return out\n",
    )
    try:
        repo = _repo(tmp_path)
        rec = repo.add("tmp_divergent_panel")
        repo.apply_transition(rec, Stage.BACKTESTED, Actor.AGENT, "bt")
        repo.record_search_trial("tmp_divergent_panel", 4, "{}")
        with pytest.raises(BacktestError, match="parity"):
            promotion_preflight(
                repo, "tmp_divergent_panel", actor=Actor.AGENT, declared_combos=None,
                allow_holdout_reuse=False, allow_non_pit=False,
                provider=SyntheticProvider(seed=2), start=_START, end=_END)
        assert repo._conn.execute("SELECT COUNT(*) c FROM gate_evaluations").fetchone()["c"] == 0
        assert repo._conn.execute("SELECT COUNT(*) c FROM holdout_evaluations").fetchone()["c"] == 0
    finally:
        mod.unlink()


def test_preflight_passes_parity_for_faithful_bundled_strategy(tmp_path):
    """A real bundled strategy with a faithful signal_panel passes the parity gate and preflight
    resolves breadth as usual (no false positive)."""
    repo = _repo(tmp_path)
    rec = repo.add("cross_sectional_momentum")
    repo.apply_transition(rec, Stage.BACKTESTED, Actor.AGENT, "bt")
    repo.record_search_trial("cross_sectional_momentum", 4, "{}")
    ctx = promotion_preflight(
        repo, "cross_sectional_momentum", actor=Actor.AGENT, declared_combos=None,
        allow_holdout_reuse=False, allow_non_pit=False,
        provider=SyntheticProvider(seed=7), start=_START, end=_END)
    assert ctx.own == 4 and ctx.provenance == "measured"
