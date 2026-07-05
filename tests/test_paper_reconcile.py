from __future__ import annotations

from algua.execution import paper_reconcile as P
from algua.registry.db import connect, migrate


def _conn(tmp_path):
    c = connect(tmp_path / "r.db")
    migrate(c)
    return c


def _add_strategy(c, name, stage):
    c.execute(
        "INSERT INTO strategies(name, stage, created_at, updated_at) VALUES (?,?,?,?)",
        (name, stage, "2026-01-01", "2026-01-01"))
    c.commit()


def _add_fill(c, activity_id, strategy, symbol, qty):
    c.execute(
        "INSERT INTO paper_venue_fills(activity_id, broker_order_id, strategy, symbol, qty, price,"
        " fill_ts) VALUES (?,?,?,?,?,?,?)",
        (activity_id, None, strategy, symbol, qty, 100.0, "2026-01-02"))
    c.commit()


def test_expected_net_sums_all_fills(tmp_path):
    c = _conn(tmp_path)
    _add_strategy(c, "pap", "paper")
    _add_fill(c, "a1", "pap", "AAPL", 10.0)
    _add_fill(c, "a2", None, "AAPL", 5.0)        # orphan still counts toward the all-fills sum
    assert P.paper_account_expected_net(c) == {"AAPL": 15.0}


def test_attributed_net_excludes_orphan_and_nonpaper(tmp_path):
    c = _conn(tmp_path)
    _add_strategy(c, "pap", "paper")
    _add_strategy(c, "live_one", "live")
    _add_fill(c, "a1", "pap", "AAPL", 10.0)      # counts
    _add_fill(c, "a2", "live_one", "AAPL", 7.0)  # excluded (non-paper)
    _add_fill(c, "a3", None, "AAPL", 5.0)        # excluded (orphan)
    assert P.attributed_paper_net(c) == {"AAPL": 10.0}


def test_reconcile_fails_closed_on_unattributable_holding(tmp_path):
    c = _conn(tmp_path)
    _add_strategy(c, "pap", "paper")
    _add_fill(c, "a1", "pap", "AAPL", 10.0)      # books (attributed) expect 10
    # broker shows 15 (a sibling/manual 5 nobody paper-owns) -> residual -> not clean
    r = P.reconcile(c, {"AAPL": 15.0}, P.next_cycle(c))
    assert not r.clean
    assert r.mismatches[0]["symbol"] == "AAPL"


def test_reconcile_clean_when_attributed_explains_broker(tmp_path):
    c = _conn(tmp_path)
    _add_strategy(c, "pap", "paper")
    _add_fill(c, "a1", "pap", "AAPL", 10.0)
    r = P.reconcile(c, {"AAPL": 10.0}, P.next_cycle(c))
    assert r.clean and not r.halt


def test_attributed_net_includes_forward_tested(tmp_path):
    """A forward_tested strategy is still on the paper LANE (parity with load_gated_strategy /
    `paper run-all`'s stage IN ('paper','forward_tested') admission): its fills must keep
    explaining its broker holdings, or the account-wide reconcile would treat them as an
    unattributable residual and defer/halt the whole multi-tenant cycle."""
    c = _conn(tmp_path)
    _add_strategy(c, "grad", "forward_tested")
    _add_fill(c, "a1", "grad", "AAPL", 10.0)
    assert P.attributed_paper_net(c) == {"AAPL": 10.0}


def test_reconcile_clean_when_forward_tested_explains_broker(tmp_path):
    c = _conn(tmp_path)
    _add_strategy(c, "grad", "forward_tested")
    _add_fill(c, "a1", "grad", "AAPL", 10.0)
    r = P.reconcile(c, {"AAPL": 10.0}, P.next_cycle(c))
    assert r.clean and not r.halt
