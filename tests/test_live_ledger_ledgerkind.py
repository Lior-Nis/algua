import sqlite3

from algua.execution.live_ledger import (
    LedgerKind,
    backfill_paper_venue_broker_order_id,
    believed_positions,
    fill_cursor,
    fills_table,
    ingest_activities,
    owned_open_order_ids,
    paper_believed_positions,
    record_paper_venue_order,
)
from algua.registry.db import migrate


def _conn(tmp_path):
    c = sqlite3.connect(tmp_path / "r.db")
    c.row_factory = sqlite3.Row
    migrate(c)
    return c


def _fill(aid, sym, qty, side, oid="o1"):
    return {
        "id": aid,
        "activity_type": "FILL",
        "side": side,
        "qty": abs(qty),
        "price": 10.0,
        "symbol": sym,
        "order_id": oid,
        "transaction_time": "2026-01-01T00:00:00Z",
    }


def test_paper_ingest_writes_paper_tables_not_live(tmp_path):
    c = _conn(tmp_path)
    # an order mapping so the fill attributes
    c.execute(
        "INSERT INTO paper_venue_orders(strategy,symbol,side,client_order_id,broker_order_id,"
        "strategy_id,status,submitted_ts) VALUES ('s','AAA','buy','c','o1',1,'submitted','t')"
    )
    c.commit()
    ingest_activities(
        c, [_fill("a1", "AAA", 5, "buy")], LedgerKind.PAPER,
        cursor_value="2026-01-02T00:00:00Z",
    )
    assert paper_believed_positions(c, "s") == {"AAA": 5.0}
    assert believed_positions(c, "s", LedgerKind.LIVE) == {}          # live ledger untouched
    assert fill_cursor(c, LedgerKind.PAPER) == "2026-01-02T00:00:00Z"  # explicit cursor stored
    assert c.execute("SELECT COUNT(*) FROM live_fills").fetchone()[0] == 0


def test_paper_ingest_dedups_by_activity_id(tmp_path):
    c = _conn(tmp_path)
    ingest_activities(c, [_fill("a1", "AAA", 5, "buy")], LedgerKind.PAPER, cursor_value="t1")
    ingest_activities(c, [_fill("a1", "AAA", 5, "buy")], LedgerKind.PAPER, cursor_value="t2")
    assert c.execute("SELECT COUNT(*) FROM paper_venue_fills").fetchone()[0] == 1


def test_paper_ingest_quarantines_malformed(tmp_path):
    c = _conn(tmp_path)
    bad = {
        "id": "a9", "activity_type": "FILL", "side": "buy", "qty": "x",  # bad qty
        "price": 1.0, "symbol": "AAA", "order_id": "o", "transaction_time": "t",
    }
    ingest_activities(c, [bad], LedgerKind.PAPER, cursor_value="t1")
    assert c.execute("SELECT COUNT(*) FROM paper_venue_activity_quarantine").fetchone()[0] == 1
    assert fill_cursor(c, LedgerKind.PAPER) == "t1"  # cursor still advanced past poison


def test_record_then_backfill_attributes_early_fill(tmp_path):
    c = sqlite3.connect(tmp_path / "r.db")
    c.row_factory = sqlite3.Row
    migrate(c)
    # intent recorded BEFORE submit (no broker id yet)
    record_paper_venue_order(c, "s", "AAA", "buy", 100.0, "c1", strategy_id=1)
    # a fill arrives under broker id o1 while the mapping is still missing -> strategy NULL
    ingest_activities(
        c,
        [{"id": "a1", "activity_type": "FILL", "side": "buy", "qty": 5, "price": 10.0,
          "symbol": "AAA", "order_id": "o1", "transaction_time": "t"}],
        LedgerKind.PAPER,
        cursor_value="t1",
    )
    assert paper_believed_positions(c, "s") == {}          # not yet attributed
    backfill_paper_venue_broker_order_id(c, "c1", "o1")    # mapping lands
    assert paper_believed_positions(c, "s") == {"AAA": 5.0}  # back-attributed


def test_fills_table_returns_per_kind_table():
    assert fills_table(LedgerKind.LIVE) == "live_fills"
    assert fills_table(LedgerKind.PAPER) == "paper_venue_fills"


class _B:
    def list_open_orders(self):
        return [
            {"id": "o1", "client_order_id": "coid-s1"},
            {"id": "o2", "client_order_id": "coid-s2"},
        ]


def test_owned_open_order_ids_paper_kind_reads_paper_venue_orders(tmp_path):
    c = _conn(tmp_path)
    for sid, nm in ((1, "s1"), (2, "s2")):
        c.execute(
            "INSERT INTO strategies(id,name,stage,created_at,updated_at)"
            " VALUES (?,?,?,?,?)",
            (sid, nm, "paper", "t", "t"),
        )
    c.commit()
    record_paper_venue_order(c, "s1", "AAA", "buy", None, "coid-s1", strategy_id=1)
    record_paper_venue_order(c, "s2", "BBB", "buy", None, "coid-s2", strategy_id=2)
    # scoped to s1 AND to the paper order ledger: only s1's open order, never s2's sibling order.
    assert owned_open_order_ids(c, _B(), "s1", kind=LedgerKind.PAPER) == ["o1"]
