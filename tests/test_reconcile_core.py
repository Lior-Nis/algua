from __future__ import annotations

from algua.execution.reconcile_core import next_cycle, reconcile_account
from algua.registry.db import connect, migrate


def _conn(tmp_path):
    c = connect(tmp_path / "r.db")
    migrate(c)
    return c


def _rec(c, broker, expected):
    return reconcile_account(
        c, broker, expected, next_cycle(c, table="paper_cycle"),
        state_table="paper_reconcile_state", tolerance=1e-6, grace_cycles=3)


def test_clean_match(tmp_path):
    c = _conn(tmp_path)
    r = _rec(c, {"AAPL": 10.0}, {"AAPL": 10.0})
    assert r.clean and not r.halt and r.mismatches == []


def test_tolerance_absorbs_subtolerance_diff(tmp_path):
    c = _conn(tmp_path)
    r = _rec(c, {"AAPL": 10.0 + 5e-7}, {"AAPL": 10.0})
    assert r.clean and not r.halt


def test_mismatch_pending_then_unexplained_halts(tmp_path):
    c = _conn(tmp_path)
    # cycle 1: books expect 10, broker flat -> pending, not halt
    r1 = _rec(c, {}, {"AAPL": 10.0})
    assert not r1.clean and not r1.halt
    assert r1.mismatches[0]["status"] == "pending"
    # cycles 2,3,4: still mismatched; at cycle 4 (4 - first_seen 1 >= 3) -> unexplained -> halt
    r = r1
    for _ in range(3):
        r = _rec(c, {}, {"AAPL": 10.0})
    assert r.halt and r.mismatches[0]["status"] == "unexplained"


def test_resolved_mismatch_clears_state(tmp_path):
    c = _conn(tmp_path)
    _rec(c, {}, {"AAPL": 10.0})                 # pending row written
    r = _rec(c, {"AAPL": 10.0}, {"AAPL": 10.0})  # now matches
    assert r.clean
    assert c.execute("SELECT COUNT(*) FROM paper_reconcile_state").fetchone()[0] == 0
