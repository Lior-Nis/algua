from __future__ import annotations

import pytest

from algua.contracts.lifecycle import Actor, Stage
from algua.registry.db import connect, migrate
from algua.registry.gating import load_gated_strategy
from algua.registry.store import SqliteStrategyRepository
from algua.risk import global_halt, kill_switch


def _conn(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    return conn


def _register_paper(conn, name="cross_sectional_momentum"):
    repo = SqliteStrategyRepository(conn)
    rec = repo.add(name)
    # Walk the lifecycle via apply_transition (no CLI runner needed; store.py's CAS does not
    # re-validate allowed transitions — that guard lives only in transitions.py).
    for stage in (Stage.BACKTESTED, Stage.CANDIDATE, Stage.PAPER):
        rec = repo.apply_transition(rec, to=stage, actor=Actor.AGENT, reason="test")
    return repo


def test_load_gated_strategy_returns_strategy_and_record(tmp_path):
    conn = _conn(tmp_path)
    _register_paper(conn)
    strategy, rec = load_gated_strategy(conn, "cross_sectional_momentum", "paper run")
    assert strategy.config.name == "cross_sectional_momentum"
    assert rec.stage is Stage.PAPER


def test_load_gated_strategy_rejects_wrong_stage(tmp_path):
    conn = _conn(tmp_path)
    repo = SqliteStrategyRepository(conn)
    repo.add("cross_sectional_momentum")  # stage = idea
    with pytest.raises(ValueError, match="requires 'paper'"):
        load_gated_strategy(conn, "cross_sectional_momentum", "paper run")


def test_load_gated_strategy_rejects_tripped_kill_switch(tmp_path):
    conn = _conn(tmp_path)
    _register_paper(conn)
    kill_switch.trip(conn, "cross_sectional_momentum", reason="x", actor="agent")
    with pytest.raises(ValueError, match="kill-switch"):
        load_gated_strategy(conn, "cross_sectional_momentum", "paper run")


def test_load_gated_strategy_rejects_global_halt(tmp_path):
    conn = _conn(tmp_path)
    _register_paper(conn)
    global_halt.engage(conn, reason="x", actor="agent")
    with pytest.raises(ValueError, match="global halt"):
        load_gated_strategy(conn, "cross_sectional_momentum", "paper run")
