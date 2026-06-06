# Multi-Strategy Live Accounting — Slice A (Books Foundation) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the per-strategy live books (allocations, orders, fills, activities) with crash-safe idempotent ingestion and average-cost P&L/NAV derivations — purely additive, **shadow mode**, with a hard guard allowing at most ONE live strategy. NO change to sizing/flatten/reconcile (those are slices B/C).

**Architecture:** New sqlite tables (schema bump 12→13). A `algua/registry/allocations.py` module for the allocation lifecycle (Σ≤equity enforced). A `algua/execution/live_ledger.py` module for order recording, idempotent activity ingestion, and average-cost P&L/NAV derivations. A `live allocate` CLI command + a go-live guard (requires an allocation AND ≤1 live strategy). A read-only broker `account_activities` method.

**Tech Stack:** Python 3.12, sqlite3, Typer, requests, pytest, ruff, mypy, import-linter. Spec: `docs/superpowers/specs/2026-06-06-multi-strategy-live-accounting-design.md`.

---

## File structure

| File | Responsibility |
|---|---|
| `algua/registry/db.py` (modify) | 5 new tables in `_SCHEMA`; `SCHEMA_VERSION` 12→13. |
| `algua/registry/allocations.py` (new) | allocation lifecycle: `allocate` (Σ≤equity), `active_allocation`, `total_allocated`, `deallocate`. |
| `algua/execution/live_ledger.py` (new) | `record_live_order` + `backfill_broker_order_id`; `ingest_activities` (idempotent); `believed_positions`; `position_pnl` (average-cost); `strategy_nav`. |
| `algua/execution/alpaca_broker.py` (modify) | read-only `account_activities(after)` on `_AlpacaBroker`. |
| `algua/cli/live_cmd.py` (modify) | `live allocate` command + `_live_account_equity()` read helper. |
| `algua/cli/registry_cmd.py` (modify) | go-live guard: require an allocation + refuse a 2nd live strategy. |

---

### Task 1: schema + allocation lifecycle

**Files:** Modify `algua/registry/db.py`; Create `algua/registry/allocations.py`; Test `tests/test_allocations.py`.

Context: `db.py` has `SCHEMA_VERSION = 12` and a `_SCHEMA` string of `CREATE TABLE IF NOT EXISTS ...` blocks ending with `live_authorizations`; `migrate(conn)` runs `conn.executescript(_SCHEMA)` then stamps `user_version`. `connect(db_path)` sets `row_factory = sqlite3.Row`. Tests elsewhere build a conn via `from algua.registry.db import connect, migrate`. `SqliteStrategyRepository(conn).get(name)` returns a record with `.id` and `.stage`; raises `LookupError` if absent.

- [ ] **Step 1: Add the tables.** In `algua/registry/db.py`, bump `SCHEMA_VERSION = 13` and append these blocks to the `_SCHEMA` string (before its closing `"""`):
```sql
CREATE TABLE IF NOT EXISTS strategy_allocations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id   INTEGER NOT NULL REFERENCES strategies(id),
    capital       REAL NOT NULL,
    effective_ts  TEXT NOT NULL,
    actor         TEXT NOT NULL,
    revoked_ts    TEXT
);
CREATE TABLE IF NOT EXISTS live_orders (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy          TEXT NOT NULL,
    symbol            TEXT NOT NULL,
    side              TEXT NOT NULL,
    intended_notional REAL,
    client_order_id   TEXT NOT NULL UNIQUE,
    broker_order_id   TEXT,
    status            TEXT NOT NULL,
    submitted_ts      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS live_fills (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    activity_id     TEXT NOT NULL UNIQUE,
    broker_order_id TEXT,
    strategy        TEXT,
    symbol          TEXT NOT NULL,
    qty             REAL NOT NULL,
    price           REAL NOT NULL,
    fill_ts         TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS live_activities (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    activity_id  TEXT NOT NULL UNIQUE,
    type         TEXT NOT NULL,
    symbol       TEXT,
    amount       REAL,
    ts           TEXT,
    raw          TEXT
);
CREATE TABLE IF NOT EXISTS live_fill_cursor (
    name    TEXT PRIMARY KEY,
    cursor  TEXT
);
```
Note `live_fills.qty` is SIGNED (buy +, sell −) — Task 4/5 depend on that.

- [ ] **Step 2: Write the failing test.** Create `tests/test_allocations.py`:
```python
import pytest

from algua.registry import allocations
from algua.registry.db import connect, migrate
from algua.registry.store import SqliteStrategyRepository


def _conn(tmp_path):
    conn = connect(tmp_path / "a.db")
    migrate(conn)
    return conn


def _strategy(conn, name="s1"):
    repo = SqliteStrategyRepository(conn)
    repo.add(name)
    return repo.get(name).id


def test_allocate_and_active(tmp_path):
    conn = _conn(tmp_path)
    sid = _strategy(conn)
    allocations.allocate(conn, sid, capital=10_000.0, actor="human", account_equity=50_000.0)
    a = allocations.active_allocation(conn, sid)
    assert a is not None and a["capital"] == 10_000.0
    assert allocations.total_allocated(conn) == 10_000.0


def test_reallocation_replaces_not_doublecounts(tmp_path):
    conn = _conn(tmp_path)
    sid = _strategy(conn)
    allocations.allocate(conn, sid, capital=10_000.0, actor="human", account_equity=50_000.0)
    allocations.allocate(conn, sid, capital=20_000.0, actor="human", account_equity=50_000.0)
    # old row revoked; only the new capital counts toward the sum
    assert allocations.total_allocated(conn) == 20_000.0
    assert allocations.active_allocation(conn, sid)["capital"] == 20_000.0


def test_sum_cannot_exceed_equity(tmp_path):
    conn = _conn(tmp_path)
    s1, s2 = _strategy(conn, "s1"), _strategy(conn, "s2")
    allocations.allocate(conn, s1, capital=40_000.0, actor="human", account_equity=50_000.0)
    with pytest.raises(allocations.AllocationError, match="exceeds"):
        allocations.allocate(conn, s2, capital=20_000.0, actor="human", account_equity=50_000.0)


def test_deallocate_requires_flat(tmp_path):
    conn = _conn(tmp_path)
    sid = _strategy(conn)
    allocations.allocate(conn, sid, capital=10_000.0, actor="human", account_equity=50_000.0)
    allocations.deallocate(conn, sid, actor="human", is_flat=True)
    assert allocations.active_allocation(conn, sid) is None
    allocations.allocate(conn, sid, capital=10_000.0, actor="human", account_equity=50_000.0)
    with pytest.raises(allocations.AllocationError, match="flat"):
        allocations.deallocate(conn, sid, actor="human", is_flat=False)
```

- [ ] **Step 3: Run** `cd /home/liornisimov/Projects/algua/.claude/worktrees/sp6-live-oms && uv run pytest tests/test_allocations.py -q` → FAIL (module missing).

- [ ] **Step 4: Implement** `algua/registry/allocations.py`:
```python
"""Per-strategy live capital allocations (the fixed sizing denominator). Append-only lifecycle:
the active allocation is the newest non-revoked row for a strategy. Σ(active capital) is capped at
account equity so the book can never over-commit the shared account."""
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime


class AllocationError(ValueError):
    """An allocation request that would over-commit the account, or a deallocation of a
    non-flat strategy."""


def total_allocated(conn: sqlite3.Connection) -> float:
    row = conn.execute(
        "SELECT COALESCE(SUM(capital), 0.0) AS t FROM strategy_allocations WHERE revoked_ts IS NULL"
    ).fetchone()
    return float(row["t"])


def active_allocation(conn: sqlite3.Connection, strategy_id: int) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM strategy_allocations WHERE strategy_id = ? AND revoked_ts IS NULL "
        "ORDER BY id DESC LIMIT 1",
        (strategy_id,),
    ).fetchone()


def allocate(conn: sqlite3.Connection, strategy_id: int, capital: float, actor: str,
             account_equity: float) -> None:
    """Set a strategy's live capital base. Revokes any prior active allocation (re-allocation),
    then enforces Σ(active capital across all strategies) ≤ account_equity. A re-allocation resets
    the strategy's NAV drawdown peak (the operator's deliberate capital change)."""
    if capital <= 0.0:
        raise AllocationError("capital must be positive")
    now = datetime.now(UTC).isoformat()
    existing = active_allocation(conn, strategy_id)
    prior = float(existing["capital"]) if existing is not None else 0.0
    prospective = total_allocated(conn) - prior + capital
    if prospective > account_equity:
        raise AllocationError(
            f"Σ allocations {prospective:.2f} exceeds account equity {account_equity:.2f}"
        )
    if existing is not None:
        conn.execute("UPDATE strategy_allocations SET revoked_ts = ? WHERE id = ?",
                     (now, existing["id"]))
    conn.execute(
        "INSERT INTO strategy_allocations(strategy_id, capital, effective_ts, actor) "
        "VALUES (?,?,?,?)",
        (strategy_id, capital, now, actor),
    )
    conn.commit()


def deallocate(conn: sqlite3.Connection, strategy_id: int, actor: str, is_flat: bool) -> None:
    """Revoke a strategy's active allocation. Requires the strategy flat with no open orders
    (the caller computes `is_flat` from the ledger + broker)."""
    if not is_flat:
        raise AllocationError("cannot deallocate a strategy that is not flat / has open orders")
    existing = active_allocation(conn, strategy_id)
    if existing is None:
        return
    conn.execute("UPDATE strategy_allocations SET revoked_ts = ? WHERE id = ?",
                 (datetime.now(UTC).isoformat(), existing["id"]))
    conn.commit()
```

- [ ] **Step 5: Run** the test → PASS.

- [ ] **Step 6: Gate + commit.** `uv run ruff check . && uv run mypy algua && uv run pytest tests/test_allocations.py -q && uv run lint-imports`
```bash
git add algua/registry/db.py algua/registry/allocations.py tests/test_allocations.py
git commit -m "feat(live-books): strategy_allocations lifecycle + Σ≤equity (schema v13)"
```

---

### Task 2: average-cost P&L + believed positions

**Files:** Create `algua/execution/live_ledger.py`; Test `tests/test_live_ledger_pnl.py`.

Context: `live_fills.qty` is SIGNED. P&L uses average-cost with signed transitions: same-direction fills update the average; opposite-direction fills realize P&L on the closed quantity; a fill that crosses through zero closes the old side then opens the new at the fill price.

- [ ] **Step 1: Write the failing test.** Create `tests/test_live_ledger_pnl.py`:
```python
from algua.execution.live_ledger import PositionPnl, position_pnl


def test_long_then_partial_close_realizes():
    # buy 10@100, buy 10@110 -> avg 105, qty 20; sell 5@120 -> realized 5*(120-105)=75
    fills = [(10.0, 100.0), (10.0, 110.0), (-5.0, 120.0)]
    r = position_pnl(fills, mark=120.0)
    assert r.qty == 15.0
    assert r.avg_cost == 105.0
    assert r.realized == 75.0
    assert r.unrealized == 15.0 * (120.0 - 105.0)


def test_flip_long_to_short():
    # buy 10@100; sell 15@120 -> close 10 (realize 10*(120-100)=200), open short 5@120
    fills = [(10.0, 100.0), (-15.0, 120.0)]
    r = position_pnl(fills, mark=130.0)
    assert r.qty == -5.0
    assert r.avg_cost == 120.0
    assert r.realized == 200.0
    # short unrealized = (avg-mark)*|qty| = (120-130)*5 = -50; (mark-avg)*qty == (130-120)*-5 == -50
    assert r.unrealized == -50.0


def test_short_then_cover():
    # sell 10@100 (short); buy 4@90 -> realize 4*(100-90)=40 covering short
    fills = [(-10.0, 100.0), (4.0, 90.0)]
    r = position_pnl(fills, mark=95.0)
    assert r.qty == -6.0
    assert r.avg_cost == 100.0
    assert r.realized == 40.0


def test_flat_is_zero():
    r = position_pnl([], mark=100.0)
    assert r.qty == 0.0 and r.realized == 0.0 and r.unrealized == 0.0
```

- [ ] **Step 2: Run** `uv run pytest tests/test_live_ledger_pnl.py -q` → FAIL.

- [ ] **Step 3: Implement** `algua/execution/live_ledger.py` (this task adds the P&L core; later tasks append to the same file):
```python
"""Per-strategy live books: order recording, crash-safe activity ingestion, and average-cost
P&L / NAV derivations. The broker account is the netted custodian; this ledger is the source of
truth for per-strategy attribution. Pure derivations are kept side-effect-free for testing."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class PositionPnl:
    qty: float          # signed net position
    avg_cost: float     # average cost of the open position (0.0 when flat)
    realized: float     # realized P&L across the fill sequence
    unrealized: float   # (mark - avg_cost) * qty  (correct for long AND short)


def position_pnl(fills: list[tuple[float, float]], mark: float) -> PositionPnl:
    """Average-cost P&L from a time-ordered list of (signed_qty, price) fills.

    Same-direction (or from-flat) fills update the average cost; opposite-direction fills realize
    P&L on the closed quantity; a fill crossing through zero closes the old side then opens the new
    side at the fill price. Unrealized uses the signed qty so it is correct for shorts:
    (mark - avg) * qty."""
    qty = 0.0
    avg = 0.0
    realized = 0.0
    for f_qty, price in fills:
        if qty == 0.0 or (qty > 0) == (f_qty > 0):
            # opening or adding in the same direction: weighted-average the cost
            new_qty = qty + f_qty
            avg = (avg * abs(qty) + price * abs(f_qty)) / abs(new_qty) if new_qty != 0.0 else 0.0
            qty = new_qty
        else:
            # reducing / closing the opposite side
            closing = min(abs(f_qty), abs(qty))
            # realized: long close gains (price-avg); short close gains (avg-price)
            realized += (price - avg) * closing if qty > 0 else (avg - price) * closing
            remaining = abs(f_qty) - closing
            qty = qty + f_qty
            if remaining > 0.0:        # crossed through zero -> open the new side at this price
                avg = price
            elif qty == 0.0:
                avg = 0.0
    unrealized = (mark - avg) * qty
    return PositionPnl(qty=qty, avg_cost=avg, realized=realized, unrealized=unrealized)
```

- [ ] **Step 4: Run** the test → PASS.

- [ ] **Step 5: Gate + commit.** `uv run ruff check . && uv run mypy algua && uv run pytest tests/test_live_ledger_pnl.py -q && uv run lint-imports`
```bash
git add algua/execution/live_ledger.py tests/test_live_ledger_pnl.py
git commit -m "feat(live-books): average-cost signed-transition P&L (long/short/flip/partial)"
```

---

### Task 3: order recording + believed positions + NAV

**Files:** Modify `algua/execution/live_ledger.py`; Test `tests/test_live_ledger_orders.py`.

Context: `client_order_id` is the durable primary identity (UNIQUE in `live_orders`); `broker_order_id` is backfilled when the broker accepts. `believed_positions` sums signed `live_fills.qty` per symbol for a strategy. `strategy_nav` = allocation + Σ realized + Σ unrealized (per held symbol, marked).

- [ ] **Step 1: Write the failing test.** Create `tests/test_live_ledger_orders.py`:
```python
from algua.execution import live_ledger as L
from algua.registry.db import connect, migrate


def _conn(tmp_path):
    conn = connect(tmp_path / "l.db")
    migrate(conn)
    return conn


def _fill(conn, activity_id, strategy, symbol, qty, price, boid="b1"):
    conn.execute(
        "INSERT INTO live_fills(activity_id, broker_order_id, strategy, symbol, qty, price, fill_ts)"
        " VALUES (?,?,?,?,?,?,?)",
        (activity_id, boid, strategy, symbol, qty, price, "2026-06-06T00:00:00+00:00"),
    )
    conn.commit()


def test_record_order_is_idempotent_on_client_id(tmp_path):
    conn = _conn(tmp_path)
    L.record_live_order(conn, "s1", "AAA", "buy", 1000.0, "coid-1")
    L.record_live_order(conn, "s1", "AAA", "buy", 1000.0, "coid-1")  # retry, same client id
    n = conn.execute("SELECT COUNT(*) FROM live_orders").fetchone()[0]
    assert n == 1


def test_backfill_broker_order_id(tmp_path):
    conn = _conn(tmp_path)
    L.record_live_order(conn, "s1", "AAA", "buy", 1000.0, "coid-1")
    L.backfill_broker_order_id(conn, "coid-1", "broker-9")
    row = conn.execute("SELECT broker_order_id FROM live_orders WHERE client_order_id='coid-1'"
                       ).fetchone()
    assert row["broker_order_id"] == "broker-9"


def test_believed_positions_sums_signed_fills(tmp_path):
    conn = _conn(tmp_path)
    _fill(conn, "a1", "s1", "AAA", 10.0, 100.0)
    _fill(conn, "a2", "s1", "AAA", -4.0, 110.0)
    _fill(conn, "a3", "s1", "BBB", 5.0, 50.0)
    assert L.believed_positions(conn, "s1") == {"AAA": 6.0, "BBB": 5.0}


def test_strategy_nav(tmp_path):
    conn = _conn(tmp_path)
    _fill(conn, "a1", "s1", "AAA", 10.0, 100.0)  # long 10 @100
    # allocation 10_000; mark 105 -> unrealized 50, realized 0 -> NAV 10_050
    nav = L.strategy_nav(conn, "s1", allocation=10_000.0, marks={"AAA": 105.0})
    assert nav == 10_050.0
```

- [ ] **Step 2: Run** `uv run pytest tests/test_live_ledger_orders.py -q` → FAIL.

- [ ] **Step 3: Append to** `algua/execution/live_ledger.py`:
```python
from datetime import UTC, datetime


def record_live_order(conn: sqlite3.Connection, strategy: str, symbol: str, side: str,
                      intended_notional: float, client_order_id: str) -> None:
    """Record a live order at submit time, keyed by client_order_id (the durable identity). A retry
    that re-submits the same client_order_id is a no-op (INSERT OR IGNORE on the UNIQUE column)."""
    conn.execute(
        "INSERT OR IGNORE INTO live_orders"
        "(strategy, symbol, side, intended_notional, client_order_id, status, submitted_ts)"
        " VALUES (?,?,?,?,?,?,?)",
        (strategy, symbol, side, intended_notional, client_order_id, "submitted",
         datetime.now(UTC).isoformat()),
    )
    conn.commit()


def backfill_broker_order_id(conn: sqlite3.Connection, client_order_id: str,
                             broker_order_id: str) -> None:
    """Attach the broker's order id once the broker accepts (covers a submit that timed out after
    Alpaca accepted it: the client_order_id row exists, the broker id arrives later)."""
    conn.execute("UPDATE live_orders SET broker_order_id = ? WHERE client_order_id = ?",
                 (broker_order_id, client_order_id))
    conn.commit()


def believed_positions(conn: sqlite3.Connection, strategy: str) -> dict[str, float]:
    """Per-symbol signed net position for a strategy = Σ its own live_fills.qty (nonzero only)."""
    rows = conn.execute(
        "SELECT symbol, SUM(qty) AS q FROM live_fills WHERE strategy = ? GROUP BY symbol",
        (strategy,),
    ).fetchall()
    return {r["symbol"]: float(r["q"]) for r in rows if float(r["q"]) != 0.0}


def _fills_for(conn: sqlite3.Connection, strategy: str, symbol: str) -> list[tuple[float, float]]:
    rows = conn.execute(
        "SELECT qty, price FROM live_fills WHERE strategy = ? AND symbol = ? ORDER BY fill_ts, id",
        (strategy, symbol),
    ).fetchall()
    return [(float(r["qty"]), float(r["price"])) for r in rows]


def strategy_nav(conn: sqlite3.Connection, strategy: str, allocation: float,
                 marks: dict[str, float]) -> float:
    """NAV = allocation + Σ realized + Σ unrealized across the strategy's symbols. `marks` supplies
    the current price per symbol (a missing mark falls back to the average cost → 0 unrealized)."""
    symbols = {r["symbol"] for r in conn.execute(
        "SELECT DISTINCT symbol FROM live_fills WHERE strategy = ?", (strategy,))}
    total = allocation
    for sym in symbols:
        fills = _fills_for(conn, strategy, sym)
        pnl = position_pnl(fills, mark=marks.get(sym, fills[-1][1] if fills else 0.0))
        total += pnl.realized + pnl.unrealized
    return total
```

- [ ] **Step 4: Run** the test → PASS.

- [ ] **Step 5: Gate + commit.** `uv run ruff check . && uv run mypy algua && uv run pytest tests/test_live_ledger_orders.py -q && uv run lint-imports`
```bash
git add algua/execution/live_ledger.py tests/test_live_ledger_orders.py
git commit -m "feat(live-books): order recording (client-id primary + boid backfill) + believed positions + NAV"
```

---

### Task 4: crash-safe idempotent activity ingestion + broker endpoint

**Files:** Modify `algua/execution/live_ledger.py`, `algua/execution/alpaca_broker.py`; Test `tests/test_live_ledger_ingest.py`, `tests/test_alpaca_broker.py`.

Context: Alpaca `GET /v2/account/activities` returns a list of activity dicts. A FILL activity has `id`, `activity_type="FILL"`, `order_id`, `symbol`, `side` ("buy"/"sell"), `qty` (positive string), `price`, `transaction_time`. Non-fill cash activities (e.g. `DIV`) have `id`, `activity_type`, `symbol?`, `net_amount`, `date`. Ingestion must be idempotent by `activity_id` and advance the cursor in the SAME transaction as the inserts. `_AlpacaBroker` has `_get(path)` and `_read(resp, path)`.

- [ ] **Step 1: Write the failing tests.** Create `tests/test_live_ledger_ingest.py`:
```python
from algua.execution import live_ledger as L
from algua.registry.db import connect, migrate


def _conn(tmp_path):
    conn = connect(tmp_path / "i.db")
    migrate(conn)
    return conn


def _fill_act(aid, order_id, symbol, side, qty, price):
    return {"id": aid, "activity_type": "FILL", "order_id": order_id, "symbol": symbol,
            "side": side, "qty": str(qty), "price": str(price),
            "transaction_time": "2026-06-06T00:00:00Z"}


def test_ingest_signs_qty_and_attributes_by_order(tmp_path):
    conn = _conn(tmp_path)
    L.record_live_order(conn, "s1", "AAA", "buy", 1000.0, "coid-1")
    L.backfill_broker_order_id(conn, "coid-1", "order-1")
    L.ingest_activities(conn, [_fill_act("act-1", "order-1", "AAA", "buy", 10, 100.0),
                               _fill_act("act-2", "order-1", "AAA", "sell", 4, 110.0)])
    assert L.believed_positions(conn, "s1") == {"AAA": 6.0}  # +10 then -4, attributed to s1


def test_ingest_is_idempotent_on_activity_id(tmp_path):
    conn = _conn(tmp_path)
    L.record_live_order(conn, "s1", "AAA", "buy", 1000.0, "coid-1")
    L.backfill_broker_order_id(conn, "coid-1", "order-1")
    acts = [_fill_act("act-1", "order-1", "AAA", "buy", 10, 100.0)]
    L.ingest_activities(conn, acts)
    L.ingest_activities(conn, acts)  # replay (overlap window) — must not double-count
    assert L.believed_positions(conn, "s1") == {"AAA": 10.0}
    assert conn.execute("SELECT COUNT(*) FROM live_fills").fetchone()[0] == 1


def test_ingest_records_cash_activity_to_live_activities(tmp_path):
    conn = _conn(tmp_path)
    L.ingest_activities(conn, [{"id": "div-1", "activity_type": "DIV", "symbol": "AAA",
                                "net_amount": "12.50", "date": "2026-06-06"}])
    row = conn.execute("SELECT type, amount FROM live_activities WHERE activity_id='div-1'"
                       ).fetchone()
    assert row["type"] == "DIV" and row["amount"] == 12.50
    assert conn.execute("SELECT COUNT(*) FROM live_fills").fetchone()[0] == 0


def test_cursor_advances_to_latest_id(tmp_path):
    conn = _conn(tmp_path)
    L.ingest_activities(conn, [{"id": "z-9", "activity_type": "DIV", "net_amount": "1", "date": "d"}])
    assert L.fill_cursor(conn) == "z-9"
```
Append to `tests/test_alpaca_broker.py` (uses the existing `_FakeRequests`/`_FakeResp` + `_broker()` helpers — read them first; route by the `/v2/account/activities` suffix):
```python
def test_account_activities_reads_list(monkeypatch):
    monkeypatch.setattr(ab, "requests", _FakeRequests(
        {"/v2/account/activities": _FakeResp(200, [{"id": "a1", "activity_type": "FILL"}])}))
    acts = _broker().account_activities()
    assert acts == [{"id": "a1", "activity_type": "FILL"}]
```

- [ ] **Step 2: Run** `uv run pytest tests/test_live_ledger_ingest.py tests/test_alpaca_broker.py::test_account_activities_reads_list -q` → FAIL.

- [ ] **Step 3a: Add the broker endpoint.** In `algua/execution/alpaca_broker.py`, add to `_AlpacaBroker` (read-only; no tollbooth needed — reading activities is not trading):
```python
    def account_activities(self, after: str | None = None) -> list[Any]:
        """Account activities (fills + cash), oldest-first. `after` is an activity-id/time cursor;
        the ledger re-pulls an overlap window and dedupes by activity id, so exact cursor semantics
        are not load-bearing for correctness."""
        path = "/v2/account/activities"
        if after:
            path += f"?after={after}"
        return self._read(self._get(path), path)
```

- [ ] **Step 3b: Add ingestion.** Append to `algua/execution/live_ledger.py`:
```python
import json


def fill_cursor(conn: sqlite3.Connection) -> str | None:
    row = conn.execute("SELECT cursor FROM live_fill_cursor WHERE name = 'activities'").fetchone()
    return row["cursor"] if row else None


def ingest_activities(conn: sqlite3.Connection, activities: list[dict]) -> None:
    """Idempotently record a batch of Alpaca activities and advance the cursor in ONE transaction.

    FILL activities become signed `live_fills` rows (buy +qty, sell -qty), attributed to a strategy
    via order_id -> live_orders.broker_order_id; non-fill activities become `live_activities` rows.
    Dedupe is by `activity_id` (UNIQUE + INSERT OR IGNORE), so re-pulling an overlap window never
    double-counts. The cursor advances to the max activity id seen, in the same transaction as the
    inserts, so a crash leaves the books and the cursor consistent (overlap replay re-dedupes)."""
    try:
        max_id: str | None = None
        for act in activities:
            aid = str(act["id"])
            max_id = aid if max_id is None or aid > max_id else max_id
            if act.get("activity_type") == "FILL":
                signed = float(act["qty"]) * (1.0 if act["side"] == "buy" else -1.0)
                boid = act.get("order_id")
                strat_row = conn.execute(
                    "SELECT strategy FROM live_orders WHERE broker_order_id = ?", (boid,)
                ).fetchone()
                conn.execute(
                    "INSERT OR IGNORE INTO live_fills"
                    "(activity_id, broker_order_id, strategy, symbol, qty, price, fill_ts)"
                    " VALUES (?,?,?,?,?,?,?)",
                    (aid, boid, strat_row["strategy"] if strat_row else None, act["symbol"],
                     signed, float(act["price"]), act.get("transaction_time", "")),
                )
            else:
                conn.execute(
                    "INSERT OR IGNORE INTO live_activities"
                    "(activity_id, type, symbol, amount, ts, raw) VALUES (?,?,?,?,?,?)",
                    (aid, act.get("activity_type", "UNKNOWN"), act.get("symbol"),
                     float(act["net_amount"]) if act.get("net_amount") is not None else None,
                     act.get("date") or act.get("transaction_time"), json.dumps(act)),
                )
        if max_id is not None:
            conn.execute(
                "INSERT INTO live_fill_cursor(name, cursor) VALUES ('activities', ?) "
                "ON CONFLICT(name) DO UPDATE SET cursor = excluded.cursor",
                (max_id,),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
```

- [ ] **Step 4: Run** the tests → PASS.

- [ ] **Step 5: Gate + commit.** `uv run ruff check . && uv run mypy algua && uv run pytest tests/test_live_ledger_ingest.py tests/test_alpaca_broker.py -q && uv run lint-imports`
```bash
git add algua/execution/live_ledger.py algua/execution/alpaca_broker.py tests/test_live_ledger_ingest.py tests/test_alpaca_broker.py
git commit -m "feat(live-books): crash-safe idempotent activity ingestion + broker account_activities"
```

---

### Task 5: `live allocate` CLI + go-live guard (≤1 live strategy + requires allocation)

**Files:** Modify `algua/cli/live_cmd.py`, `algua/cli/registry_cmd.py`; Test `tests/test_cli_live.py`, `tests/test_cli_registry.py`.

Context: `live_cmd.py` has the `live` Typer group (`live_app`) + `registry_conn`/`ok`/`emit`/`json_errors`/`get_settings`/`SqliteStrategyRepository`. `registry_cmd.py`'s `transition` completes a go-live after signature verification (the `if target is Stage.LIVE:` block calling `verify_and_consume` then `record_approval`). `repo.list_strategies(Stage.LIVE)` returns the strategies currently at live stage. The `live allocate` command needs the live account equity for the Σ≤equity check; it reads it with the live keys (read-only, no authorization needed — not trading) via a monkeypatchable helper.

- [ ] **Step 1: Write the failing tests.** Append to `tests/test_cli_live.py` (reuses its `_isolated` fixture setting `ALGUA_ALPACA_LIVE_API_KEY/SECRET`; uses `runner`/`app`):
```python
def test_live_allocate_records_and_enforces_sum(monkeypatch):
    from contextlib import closing

    from algua.config.settings import get_settings
    from algua.registry.db import connect, migrate
    monkeypatch.setattr("algua.cli.live_cmd._live_account_equity", lambda: 50_000.0)
    assert runner.invoke(app, ["registry", "add", "s1"]).exit_code == 0
    r = runner.invoke(app, ["live", "allocate", "s1", "--capital", "10000"])
    assert r.exit_code == 0, r.stdout
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        n = conn.execute("SELECT COUNT(*) FROM strategy_allocations WHERE revoked_ts IS NULL"
                         ).fetchone()[0]
        assert n == 1
    # over-commit refused
    runner.invoke(app, ["registry", "add", "s2"])
    r2 = runner.invoke(app, ["live", "allocate", "s2", "--capital", "45000"])
    assert r2.exit_code == 1 and json.loads(r2.stdout)["ok"] is False
```
Append to `tests/test_cli_registry.py` (read it for its helpers that drive a strategy to `paper`; mirror them — `_to_paper(name)` or equivalent). The go-live guard tests:
```python
def test_go_live_requires_allocation(monkeypatch, tmp_path):
    # a strategy at paper with NO allocation cannot be issued a go-live challenge
    name = _seed_paper(monkeypatch, tmp_path, "s1")  # helper that brings s1 to paper stage
    r = runner.invoke(app, ["registry", "transition", name, "--to", "live", "--actor", "human"])
    assert r.exit_code == 1 and "allocation" in r.stdout.lower()


def test_go_live_refuses_second_live_strategy(monkeypatch, tmp_path):
    # with one strategy already live, a second is refused (slice-A hard guard, ≤1 live)
    _force_live(monkeypatch, tmp_path, "already")   # helper inserting a strategy at stage 'live'
    name = _seed_paper(monkeypatch, tmp_path, "s2")
    _allocate(monkeypatch, tmp_path, name, 1000.0)  # give it an allocation so we hit the ≤1 guard
    r = runner.invoke(app, ["registry", "transition", name, "--to", "live", "--actor", "human"])
    assert r.exit_code == 1 and "one live strategy" in r.stdout.lower()
```
(Write `_seed_paper`/`_force_live`/`_allocate` as small local helpers mirroring the existing test setup in that file — `_force_live` can `UPDATE strategies SET stage='live'`; `_allocate` calls `live allocate` with `_live_account_equity` monkeypatched.)

- [ ] **Step 2: Run** `uv run pytest tests/test_cli_live.py -k allocate tests/test_cli_registry.py -k "go_live" -q` → FAIL.

- [ ] **Step 3a: Add the `live allocate` command** to `algua/cli/live_cmd.py`:
```python
def _live_account_equity() -> float:
    """Read the live account equity (read-only; no go-live authorization needed — not trading)."""
    s = get_settings()
    if not s.alpaca_live_api_key or not s.alpaca_live_api_secret:
        raise ValueError("Alpaca LIVE credentials not configured")
    import requests
    resp = requests.get(f"{s.alpaca_live_url.rstrip('/')}/v2/account",
                        headers={"APCA-API-KEY-ID": s.alpaca_live_api_key,
                                 "APCA-API-SECRET-KEY": s.alpaca_live_api_secret}, timeout=30)
    if resp.status_code != 200:
        raise ValueError(f"alpaca {resp.status_code} reading account equity")
    return float(resp.json()["equity"])


@live_app.command("allocate")
@json_errors(ValueError, LookupError, AllocationError)
def allocate(name: str, capital: float = typer.Option(..., "--capital", help="live capital base $")
             ) -> None:
    """Set a strategy's live capital base (its fixed sizing denominator). Enforces that the sum of
    all live allocations does not exceed account equity."""
    with registry_conn() as conn:
        sid = SqliteStrategyRepository(conn).get(name).id
        allocations.allocate(conn, sid, capital=capital, actor="human",
                             account_equity=_live_account_equity())
    emit(ok({"strategy": name, "capital": capital}))
```
Add imports to `live_cmd.py`: `from algua.registry import allocations` and `from algua.registry.allocations import AllocationError`.

- [ ] **Step 3b: Add the go-live guard** in `algua/cli/registry_cmd.py`. At the START of the `if target is Stage.LIVE and signature is None:` branch (before printing the challenge), and also guard the completion path — simplest: right after resolving `rec`/`repo` for a live target, insert:
```python
        if target is Stage.LIVE:
            from algua.registry import allocations
            if len(repo.list_strategies(Stage.LIVE)) > 0 and rec.stage is not Stage.LIVE:
                raise TransitionError(
                    "refusing: only one live strategy is allowed until multi-strategy controls "
                    "land (slice C). Retire the current live strategy first.")
            if allocations.active_allocation(conn, rec.id) is None:
                raise TransitionError(
                    f"{name} has no live allocation; run `algua live allocate {name} --capital X` "
                    "before going live.")
```
(Place this so it runs for BOTH the challenge-issue step and the `--signature` completion step — i.e. once `rec` and the registry `conn`/`repo` are available and `target is Stage.LIVE`, before either branch acts. `TransitionError` is already imported and rendered as `{ok:false}` by this command's error decorator.)

- [ ] **Step 4: Run** the tests → PASS.

- [ ] **Step 5: FULL gate + commit.** `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports` (all green; contracts 0 broken).
```bash
git add algua/cli/live_cmd.py algua/cli/registry_cmd.py tests/test_cli_live.py tests/test_cli_registry.py
git commit -m "feat(live-books): live allocate CLI + go-live guard (requires allocation, ≤1 live strategy)"
```

---

## Self-review notes

- **Spec coverage (Slice A items in §8):** tables (Task 1); `live allocate` lifecycle + Σ≤equity (Tasks 1, 5); idempotent all-activity ingestion with atomic cursor + activity-id dedupe + overlap replay (Task 4); order recording client-id-primary + broker-id backfill (Task 3); average-cost P&L + believed_positions + NAV (Tasks 2, 3); go-live-requires-allocation + ≤1-live hard guard (Task 5). No sizing/flatten/reconcile change — those are slices B/C (not in this plan). ✓
- **Shadow-mode invariant:** nothing here is wired into a trading loop; `account_activities`/`ingest_activities`/`record_live_order`/the derivations are built + unit-tested but not yet called by `live trade-tick` (that's slice B/C). The only behavior change to an existing command is the go-live guard (additive refusal), which is the intended shadow-mode safety.
- **Type consistency:** `position_pnl(list[tuple[float,float]], mark) -> PositionPnl` (Task 2) is reused by `strategy_nav` (Task 3); `believed_positions`/`record_live_order`/`backfill_broker_order_id`/`ingest_activities`/`fill_cursor` all in `live_ledger.py`; `allocations.allocate/active_allocation/total_allocated/deallocate/AllocationError` (Task 1) reused in Task 5. `live_fills.qty` signed everywhere.
- **No placeholders:** every step has complete code. The `tests/test_cli_registry.py` helpers (`_seed_paper`/`_force_live`/`_allocate`) are described as small local mirrors of that file's existing setup — the implementer must read the file and match its real fixtures; the assertion intent is explicit.
- **Idempotency/crash-safety (Codex CRITICAL):** Task 4 inserts fills + advances the cursor in one transaction with rollback-on-error, dedupes by `activity_id`, and the test replays the same batch to prove no double-count.
