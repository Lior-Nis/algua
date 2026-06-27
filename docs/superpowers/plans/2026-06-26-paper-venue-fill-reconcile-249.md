# Paper-venue fill reconcile (#249) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the wall-clock `paper trade-tick` lane a real per-strategy position belief from a `paper_venue_fills` ledger and reconcile it (single-tenant, tolerance) against the broker, fixing the phantom-flatten bug while keeping the forward-gate `reconcile_ok` integrity honest.

**Architecture:** Generalize the #250-hardened live activity ledger to a second, paper-scoped ledger via a closed `LedgerKind` enum; record wall-clock orders crash-safely into a dedicated `paper_venue_orders` table (intent-before-submit via a new `before_submit` `run_tick` hook); reconcile the venue belief against the single sizing snapshot inside `run_tick` with a `1e-6` tolerance; flatten strategy-scoped via `submit_offset`.

**Tech Stack:** Python 3.12, SQLite (`algua/registry/db.py`), Typer CLI, pytest, Alpaca paper broker adapter.

## Global Constraints

- Quality gate (run after every task): `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
- `SCHEMA_VERSION` bump MUST accompany its table additions in `_SCHEMA` (`algua/registry/db.py`); tables are `CREATE TABLE IF NOT EXISTS` in `_SCHEMA` and created idempotently by `migrate()` via `executescript(_SCHEMA)`. No per-version migration logic for brand-new tables.
- No backwards-compat shims / dual code paths / defaulted compatibility params (project rule). Thread `LedgerKind.LIVE` explicitly at every existing live call site.
- Keep `algua/contracts` and `algua/features` pure; do not introduce cross-module imports beyond contracts. Run `uv run lint-imports`.
- `git add` scoped paths only (never `git add -A`).
- Commit after each task with a `feat(249):` / `test(249):` / `refactor(249):` message; end the body with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- The reconcile path must NOT be tested by monkeypatching `run_tick`/`trade_tick` wholesale: use a fake paper broker whose `account_activities_window` / `snapshot` / `get_positions` / `submit_offset` are scripted.

## File Structure

- `algua/registry/db.py` — add 5 `paper_venue_*` tables to `_SCHEMA`; bump `SCHEMA_VERSION` to 30.
- `algua/execution/live_ledger.py` — add `LedgerKind`, `LedgerTables`, `_TABLES`; generalize `ingest_activities`, `_ingest_one_activity`, `_quarantine_activity`, `fill_cursor`, `believed_positions` on `kind`; add `cursor_value` param; add `paper_believed_positions`, `record_paper_venue_order`, `backfill_paper_venue_broker_order_id`. (Module is the shared ledger; the paper analog lives here next to the live one.)
- `algua/execution/live_sizing.py`, `algua/cli/live_cmd.py`, `algua/registry/store.py` — thread `LedgerKind.LIVE` into existing live ledger calls.
- `algua/live/live_loop.py` — add `before_submit` + `venue_belief` hooks to `TickHooks`; fire `before_submit` before `submit_sized`; replace the `derived_positions` reconcile branch with a tolerance comparison vs `positions_before`; keep `TickResult.reconcile_ok`; promote a shared `_RECONCILE_TOL`.
- `algua/cli/paper_cmd.py` — paper venue ingest helper; wire `trade-tick` (ingest → hooks → reconcile_ok stamp); strategy-scoped offset flatten in the breach handler + `flatten`; `paper show` venue view; thread `LedgerKind.LIVE` into the live-strategy paths.
- `algua/execution/order_state.py` — remove `clear_derived_positions`; retain `derive_positions` (sim view).
- `algua/registry/forward_promotion.py` — `_classify_activities` matches FILLs against `paper_venue_orders`.
- Tests: `tests/test_live_ledger_*.py`, `tests/test_live_loop.py`, `tests/test_cli_paper.py`, `tests/test_forward_promotion.py`, `tests/test_registry_db.py`, plus a new `tests/test_paper_venue_reconcile.py`.

---

## SLICE 1 — Ledger foundation

### Task 1: Schema — five `paper_venue_*` tables, bump to v30

**Files:**
- Modify: `algua/registry/db.py` (`_SCHEMA` after the `live_activity_quarantine` block ~line 325; `SCHEMA_VERSION` line 16; a migrate comment ~line 629)
- Test: `tests/test_registry_db.py`

**Interfaces:**
- Produces: tables `paper_venue_orders`, `paper_venue_fills`, `paper_venue_activities`, `paper_venue_fill_cursor`, `paper_venue_activity_quarantine`; `SCHEMA_VERSION == 30`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_registry_db.py
def test_paper_venue_tables_created_at_v30(tmp_path):
    import sqlite3
    from algua.registry.db import migrate, SCHEMA_VERSION
    assert SCHEMA_VERSION == 30
    conn = sqlite3.connect(tmp_path / "r.db"); conn.row_factory = sqlite3.Row
    migrate(conn)
    names = {r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"paper_venue_orders", "paper_venue_fills", "paper_venue_activities",
            "paper_venue_fill_cursor", "paper_venue_activity_quarantine"} <= names
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 30
    # partial-unique broker_order_id index allows many NULLs, one non-null owner
    conn.execute("INSERT INTO paper_venue_orders(strategy,symbol,side,client_order_id,"
                 "strategy_id,status,submitted_ts) VALUES ('s','AAA','buy','c1',1,'submitted','t')")
    conn.execute("INSERT INTO paper_venue_orders(strategy,symbol,side,client_order_id,"
                 "strategy_id,status,submitted_ts) VALUES ('s','AAA','buy','c2',1,'submitted','t')")
    conn.commit()  # two NULL broker ids OK
```

- [ ] **Step 2: Run test, verify it fails**

Run: `uv run pytest tests/test_registry_db.py::test_paper_venue_tables_created_at_v30 -v`
Expected: FAIL (`SCHEMA_VERSION == 29`, tables missing).

- [ ] **Step 3: Add the tables to `_SCHEMA` and bump the version**

In `algua/registry/db.py`, set `SCHEMA_VERSION = 30`. Immediately after the `live_activity_quarantine` table block inside `_SCHEMA`, insert the five tables exactly as written in the design spec §"Data model" (the full `CREATE TABLE IF NOT EXISTS paper_venue_orders … paper_venue_activity_quarantine` block, including the `ux_paper_venue_orders_broker_order_id` partial unique index and the `ix_paper_venue_fills_strategy_symbol` index).

Add a migrate comment near the other version notes:
```python
    # v30 (#249): paper_venue_* (orders/fills/activities/cursor/quarantine) are brand-new tables;
    # executescript(_SCHEMA) above creates them (CREATE TABLE IF NOT EXISTS).
```

- [ ] **Step 4: Run test, verify it passes**

Run: `uv run pytest tests/test_registry_db.py::test_paper_venue_tables_created_at_v30 -v`
Expected: PASS.

- [ ] **Step 5: Run the full gate + commit**

```bash
uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports
git add algua/registry/db.py tests/test_registry_db.py
git commit -m "feat(249): paper_venue_* schema (v30) — ledger/orders/cursor/quarantine

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

### Task 2: `LedgerKind` seam — generalize the ledger; thread `LedgerKind.LIVE`

**Files:**
- Modify: `algua/execution/live_ledger.py`
- Modify (thread `LedgerKind.LIVE`): `algua/execution/live_sizing.py:43`, `algua/cli/live_cmd.py:141,161,162,276`, `algua/registry/store.py:347-348`, `algua/cli/paper_cmd.py:112,113,117,195,546,548`
- Test: `tests/test_live_ledger_ledgerkind.py` (new), existing `tests/test_live_ledger_*.py` (regression)

**Interfaces:**
- Produces:
  - `class LedgerKind(Enum)` with `LIVE`, `PAPER`.
  - `ingest_activities(conn, activities, kind: LedgerKind, *, cursor_value: str | None = None) -> None` — when `cursor_value` is given (paper), it is stored as the cursor; when `None` (live), `max(activity_id)` is stored as today.
  - `fill_cursor(conn, kind: LedgerKind) -> str | None`
  - `believed_positions(conn, strategy, kind: LedgerKind) -> dict[str, float]`
  - `paper_believed_positions(conn, strategy) -> dict[str, float]`
- Consumes: tables from Task 1; `strategy_live_symbols` stays live-only (unchanged) — used by the live resume path in `paper_cmd`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_live_ledger_ledgerkind.py
import sqlite3, pytest
from algua.registry.db import migrate
from algua.execution.live_ledger import (
    LedgerKind, ingest_activities, fill_cursor, believed_positions, paper_believed_positions)

def _conn(tmp_path):
    c = sqlite3.connect(tmp_path / "r.db"); c.row_factory = sqlite3.Row; migrate(c); return c

def _fill(aid, sym, qty, side, oid="o1"):
    return {"id": aid, "activity_type": "FILL", "side": side, "qty": abs(qty),
            "price": 10.0, "symbol": sym, "order_id": oid, "transaction_time": "2026-01-01T00:00:00Z"}

def test_paper_ingest_writes_paper_tables_not_live(tmp_path):
    c = _conn(tmp_path)
    # an order mapping so the fill attributes
    c.execute("INSERT INTO paper_venue_orders(strategy,symbol,side,client_order_id,broker_order_id,"
              "strategy_id,status,submitted_ts) VALUES ('s','AAA','buy','c','o1',1,'submitted','t')")
    c.commit()
    ingest_activities(c, [_fill("a1", "AAA", 5, "buy")], LedgerKind.PAPER, cursor_value="2026-01-02T00:00:00Z")
    assert paper_believed_positions(c, "s") == {"AAA": 5.0}
    assert believed_positions(c, "s", LedgerKind.LIVE) == {}          # live ledger untouched
    assert fill_cursor(c, LedgerKind.PAPER) == "2026-01-02T00:00:00Z"  # explicit cursor stored
    assert c.execute("SELECT COUNT(*) FROM live_fills").fetchone()[0] == 0

def test_paper_ingest_dedups_by_activity_id(tmp_path):
    c = _conn(tmp_path)
    ingest_activities(c, [_fill("a1", "AAA", 5, "buy")], LedgerKind.PAPER, cursor_value="t1")
    ingest_activities(c, [_fill("a1", "AAA", 5, "buy")], LedgerKind.PAPER, cursor_value="t2")  # replay
    assert c.execute("SELECT COUNT(*) FROM paper_venue_fills").fetchone()[0] == 1

def test_paper_ingest_quarantines_malformed(tmp_path):
    c = _conn(tmp_path)
    bad = {"id": "a9", "activity_type": "FILL", "side": "buy", "qty": "x",  # bad qty
           "price": 1.0, "symbol": "AAA", "order_id": "o", "transaction_time": "t"}
    ingest_activities(c, [bad], LedgerKind.PAPER, cursor_value="t1")
    assert c.execute("SELECT COUNT(*) FROM paper_venue_activity_quarantine").fetchone()[0] == 1
    assert fill_cursor(c, LedgerKind.PAPER) == "t1"  # cursor still advanced past poison
```

- [ ] **Step 2: Run, verify fail**

Run: `uv run pytest tests/test_live_ledger_ledgerkind.py -v`
Expected: FAIL (`LedgerKind` undefined / signature mismatch).

- [ ] **Step 3: Implement the seam**

In `algua/execution/live_ledger.py` add near the top:
```python
from dataclasses import dataclass
from enum import Enum

@dataclass(frozen=True)
class LedgerTables:
    fills: str
    activities: str
    cursor: str
    orders: str
    quarantine: str

class LedgerKind(Enum):
    LIVE = "live"
    PAPER = "paper"

_TABLES = {
    LedgerKind.LIVE: LedgerTables("live_fills", "live_activities", "live_fill_cursor",
                                  "live_orders", "live_activity_quarantine"),
    LedgerKind.PAPER: LedgerTables("paper_venue_fills", "paper_venue_activities",
                                   "paper_venue_fill_cursor", "paper_venue_orders",
                                   "paper_venue_activity_quarantine"),
}
```
Refactor `fill_cursor`, `believed_positions`, `ingest_activities`, `_ingest_one_activity`,
`_quarantine_activity` to take `kind: LedgerKind`, resolve `t = _TABLES[kind]`, and f-string the
table names from `t` (`t.fills`, `t.activities`, `t.cursor`, `t.orders`, `t.quarantine`). In
`ingest_activities`, add the keyword param `cursor_value: str | None = None`; at the cursor-advance
step write `cursor_value if cursor_value is not None else max_id`. Add:
```python
def paper_believed_positions(conn, strategy):
    return believed_positions(conn, strategy, LedgerKind.PAPER)
```

- [ ] **Step 4: Thread `LedgerKind.LIVE` into every existing live call site**

Update: `live_sizing.py:43`, `live_cmd.py:141,161,162,276`, `store.py:348`, and the live-strategy
paths in `paper_cmd.py:112,113,195,546,548` to pass `LedgerKind.LIVE` (import it). `fill_cursor(conn)` → `fill_cursor(conn, LedgerKind.LIVE)`; `ingest_activities(conn, acts)` → `ingest_activities(conn, acts, LedgerKind.LIVE)`; `believed_positions(conn, name)` → `believed_positions(conn, name, LedgerKind.LIVE)`. (`strategy_live_symbols` is unchanged.)

- [ ] **Step 5: Run the new + regression tests**

Run: `uv run pytest tests/test_live_ledger_ledgerkind.py tests/test_live_ledger_ingest.py tests/test_live_ledger_orders.py tests/test_cli_live.py -v`
Expected: PASS (new) and PASS (existing live ledger tests unchanged in behavior).

- [ ] **Step 6: Gate + commit**

```bash
uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports
git add algua/execution/live_ledger.py algua/execution/live_sizing.py algua/cli/live_cmd.py \
        algua/registry/store.py algua/cli/paper_cmd.py tests/test_live_ledger_ledgerkind.py
git commit -m "feat(249): LedgerKind seam — generalize ingest/belief ledger; thread LIVE

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

### Task 3: Paginated, fail-closed paper venue ingest helper (broker-time cursor)

**Files:**
- Modify: `algua/cli/paper_cmd.py` (add `_ingest_paper_venue(conn, name, broker)` helper)
- Test: `tests/test_paper_venue_reconcile.py` (new)

**Interfaces:**
- Produces: `_ingest_paper_venue(conn: sqlite3.Connection, broker) -> None` — reads `fill_cursor(conn, LedgerKind.PAPER)` (default a far-past bound when None), calls `broker.account_activities_window(after=cursor, until=broker.clock())`, and `ingest_activities(conn, acts, LedgerKind.PAPER, cursor_value=until)`. Raises on transport/pagination failure (fail-closed).
- Consumes: `account_activities_window` (exhaustive, raises on partial — `alpaca_broker.py:354`), `broker.clock()`.

- [ ] **Step 1: Write failing tests** (use a scripted fake broker; no monkeypatch of `run_tick`)

```python
# tests/test_paper_venue_reconcile.py
import sqlite3, pytest
from algua.registry.db import migrate
from algua.execution.live_ledger import LedgerKind, fill_cursor, paper_believed_positions
from algua.cli.paper_cmd import _ingest_paper_venue

_FAR_PAST = "1970-01-01T00:00:00Z"

class FakeBroker:
    def __init__(self, windows, clock="2026-01-02T00:00:00Z"):
        # windows: dict cursor -> list[activity] ; or callable to raise
        self._windows = windows; self._clock = clock
    def clock(self): return self._clock
    def account_activities_window(self, after, until):
        out = self._windows.get(after)
        if isinstance(out, Exception): raise out
        return out or []

def _conn(tmp_path):
    c = sqlite3.connect(tmp_path / "r.db"); c.row_factory = sqlite3.Row; migrate(c); return c

def _fill(aid, sym, qty, side, oid):
    return {"id": aid, "activity_type": "FILL", "side": side, "qty": abs(qty), "price": 10.0,
            "symbol": sym, "order_id": oid, "transaction_time": "2026-01-01T12:00:00Z"}

def test_ingest_uses_far_past_first_then_advances_cursor(tmp_path):
    c = _conn(tmp_path)
    c.execute("INSERT INTO paper_venue_orders(strategy,symbol,side,client_order_id,broker_order_id,"
              "strategy_id,status,submitted_ts) VALUES ('s','AAA','buy','c','o1',1,'submitted','t')")
    c.commit()
    broker = FakeBroker({_FAR_PAST: [_fill("a1", "AAA", 5, "buy", "o1")]})
    _ingest_paper_venue(c, broker)
    assert paper_believed_positions(c, "s") == {"AAA": 5.0}
    assert fill_cursor(c, LedgerKind.PAPER) == "2026-01-02T00:00:00Z"  # = until (broker clock)

def test_ingest_fails_closed_on_transport_error(tmp_path):
    c = _conn(tmp_path)
    broker = FakeBroker({_FAR_PAST: RuntimeError("503")})
    with pytest.raises(RuntimeError):
        _ingest_paper_venue(c, broker)
    assert fill_cursor(c, LedgerKind.PAPER) is None  # no cursor advance on failure
```

- [ ] **Step 2: Run, verify fail**

Run: `uv run pytest tests/test_paper_venue_reconcile.py -v`
Expected: FAIL (`_ingest_paper_venue` undefined).

- [ ] **Step 3: Implement the helper**

```python
# algua/cli/paper_cmd.py
_PAPER_CURSOR_FAR_PAST = "1970-01-01T00:00:00Z"

def _ingest_paper_venue(conn: sqlite3.Connection, broker: object) -> None:
    """Exhaustively ingest the paper venue's activities into paper_venue_fills, fail-closed.
    Cursor is a broker-time high-water: fetch (cursor, broker.clock()] via the paginated
    account_activities_window (raises on a partial page), dedup by activity_id, then persist the
    `until` clock as the new cursor in the SAME ingest transaction."""
    after = fill_cursor(conn, LedgerKind.PAPER) or _PAPER_CURSOR_FAR_PAST
    until = broker.clock()  # type: ignore[attr-defined]
    acts = broker.account_activities_window(after, until)  # type: ignore[attr-defined]
    ingest_activities(conn, acts, LedgerKind.PAPER, cursor_value=until)
```

- [ ] **Step 4: Run, verify pass**

Run: `uv run pytest tests/test_paper_venue_reconcile.py -v`
Expected: PASS.

- [ ] **Step 5: Gate + commit**

```bash
uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports
git add algua/cli/paper_cmd.py tests/test_paper_venue_reconcile.py
git commit -m "feat(249): paginated fail-closed paper venue ingest (broker-time cursor)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## SLICE 2 — Crash-safe venue order recording

### Task 4: `record_paper_venue_order` + `backfill_paper_venue_broker_order_id`

**Files:**
- Modify: `algua/execution/live_ledger.py`
- Test: `tests/test_live_ledger_ledgerkind.py`

**Interfaces:**
- Produces:
  - `record_paper_venue_order(conn, strategy, symbol, side, intended_notional, client_order_id, *, strategy_id) -> None` — `INSERT OR IGNORE` into `paper_venue_orders` (status `submitted`), idempotent on the `client_order_id` UNIQUE.
  - `backfill_paper_venue_broker_order_id(conn, client_order_id, broker_order_id) -> None` — set `broker_order_id`, then `UPDATE paper_venue_fills SET strategy=<that order's strategy> WHERE broker_order_id=? AND strategy IS NULL`.

- [ ] **Step 1: Write failing tests**

```python
def test_record_then_backfill_attributes_early_fill(tmp_path):
    from algua.execution.live_ledger import (
        record_paper_venue_order, backfill_paper_venue_broker_order_id,
        ingest_activities, LedgerKind, paper_believed_positions)
    c = sqlite3.connect(tmp_path / "r.db"); c.row_factory = sqlite3.Row
    from algua.registry.db import migrate; migrate(c)
    # intent recorded BEFORE submit (no broker id yet)
    record_paper_venue_order(c, "s", "AAA", "buy", 100.0, "c1", strategy_id=1)
    # a fill arrives under broker id o1 while the mapping is still missing -> strategy NULL
    ingest_activities(c, [{"id":"a1","activity_type":"FILL","side":"buy","qty":5,"price":10.0,
                           "symbol":"AAA","order_id":"o1","transaction_time":"t"}],
                      LedgerKind.PAPER, cursor_value="t1")
    assert paper_believed_positions(c, "s") == {}              # not yet attributed
    backfill_paper_venue_broker_order_id(c, "c1", "o1")        # mapping lands
    assert paper_believed_positions(c, "s") == {"AAA": 5.0}    # back-attributed
```

- [ ] **Step 2–4:** Run (fail) → implement (mirror `record_live_order` / `backfill_broker_order_id` against `paper_venue_orders`/`paper_venue_fills`) → run (pass).

Run: `uv run pytest tests/test_live_ledger_ledgerkind.py::test_record_then_backfill_attributes_early_fill -v`

- [ ] **Step 5: Gate + commit** (`feat(249): crash-safe paper_venue_orders record + backfill`).

### Task 5: `before_submit` hook in `run_tick`

**Files:**
- Modify: `algua/live/live_loop.py` (`TickHooks` + the submit loop ~line 209)
- Test: `tests/test_live_loop.py`

**Interfaces:**
- Produces: `TickHooks.before_submit: Callable[[OrderIntent, str | None], None] | None` fired immediately before `broker.submit_sized` for each intent.

- [ ] **Step 1: Write failing test**

```python
def test_before_submit_fires_before_submit(monkeypatch):
    # Use the existing run_tick test harness/fake broker in this file; assert order of calls.
    calls = []
    # ... build strategy, provider with one decideable bar, fake broker whose submit_sized appends
    #     ("submit", coid); before_submit appends ("before", coid)
    hooks = TickHooks(client_order_id_for=lambda s,t,sym: "cid",
                      before_submit=lambda intent, coid: calls.append(("before", coid)))
    run_tick(strategy, broker, provider, start, end, hooks=hooks)
    assert calls and calls[0] == ("before", "cid")  # before precedes the broker submit
```

- [ ] **Step 2–4:** Run (fail) → add the `before_submit` field to `TickHooks` and call `if hooks.before_submit is not None: hooks.before_submit(intent, coid)` on the line immediately before `order_id = broker.submit_sized(...)` → run (pass).

- [ ] **Step 5: Gate + commit** (`feat(249): before_submit run_tick hook (pre-submit intent recording)`).

### Task 6: Wire `trade-tick` recording onto `paper_venue_orders`; move `_strategy_held_symbols`

**Files:**
- Modify: `algua/cli/paper_cmd.py` (`trade_tick` `_persist`/hooks; `_strategy_held_symbols`)
- Test: `tests/test_cli_paper.py`

**Interfaces:**
- Consumes: `record_paper_venue_order`, `backfill_paper_venue_broker_order_id`, `before_submit`.
- Produces: `_strategy_held_symbols` reads `paper_venue_orders` (∪ universe) instead of `paper_orders`.

- [ ] **Step 1: Write failing test** — after a `trade-tick` that submits one order, assert a `paper_venue_orders` row exists with the `client_order_id` (intent) and a backfilled `broker_order_id`; assert `_strategy_held_symbols` includes a venue-ordered symbol no longer in the universe.

- [ ] **Step 2–4:** Run (fail) → in `trade_tick`, set `hooks.before_submit = lambda intent, coid: record_paper_venue_order(conn, name, intent.symbol, intent.side.value, None, coid, strategy_id=rec.id)` and `hooks.on_submitted = lambda rec_: backfill_paper_venue_broker_order_id(conn, rec_.client_order_id, rec_.order_id)`; repoint `_strategy_held_symbols` SQL to `paper_venue_orders`. Remove the old `record_submitted_order` wall-clock call. → run (pass).

- [ ] **Step 5: Gate + commit** (`feat(249): trade-tick records intent-before-submit into paper_venue_orders`).

### Task 7: Forward gate attributes fills via `paper_venue_orders`

**Files:**
- Modify: `algua/registry/forward_promotion.py` (`_classify_activities` line 153)
- Test: `tests/test_forward_promotion.py`

**Interfaces:**
- Produces: `_classify_activities` matches a FILL via `SELECT 1 FROM paper_venue_orders WHERE strategy_id=? AND broker_order_id=?`.

- [ ] **Step 1: Write failing test** — a window with one FILL whose `order_id` matches a `paper_venue_orders` row for the strategy → `n_unattributable == 0`; an unmatched FILL → `n_unattributable == 1`.

- [ ] **Step 2–4:** Run (fail) → change the table in the `_classify_activities` query from `paper_orders` to `paper_venue_orders`; update any existing forward-promotion test fixtures that pre-seed `paper_orders` for attribution to seed `paper_venue_orders` instead → run (pass).

- [ ] **Step 5: Gate + commit** (`refactor(249): forward gate attributes paper fills via paper_venue_orders`).

---

## SLICE 3 — Single-tenant reconcile

### Task 8: `venue_belief` hook + tolerance reconcile in `run_tick`

**Files:**
- Modify: `algua/live/live_loop.py` (`TickHooks`, reconcile branch ~171-179, keep `TickResult.reconcile_ok`); promote `_RECONCILE_TOL = 1e-6` to a shared location importable by `live_loop` (e.g. define in `live_loop.py` and have `paper_cmd` import it, or a small constant module). Remove `TickHooks.derived_positions`.
- Modify: `algua/execution/order_state.py` (remove `clear_derived_positions`; keep `derive_positions`)
- Test: `tests/test_live_loop.py`

**Interfaces:**
- Produces: `TickHooks.venue_belief: Callable[[], dict[str, float]] | None`. Reconcile: `belief = {s:q for s,q in hooks.venue_belief().items() if q!=0.0}` compared to `positions_before` per symbol over the union with `abs(b - p) > _RECONCILE_TOL` → `RiskBreach("reconcile", …)`; `reconcile_ok` set accordingly on `TickResult`.

- [ ] **Step 1: Write failing tests**

```python
def test_reconcile_tolerates_fractional_residual():
    # venue_belief {AAA: 5.0}; broker snapshot qtys {AAA: 5.0 + 4e-7} -> within 1e-6 -> no breach
    ...
    res = run_tick(strategy, broker, provider, start, end,
                   hooks=TickHooks(venue_belief=lambda: {"AAA": 5.0}))
    assert res.reconcile_ok is True

def test_reconcile_trips_on_unexplained_holding():
    # venue_belief {} ; broker holds {AAA: 5.0} -> drift > tol -> RiskBreach('reconcile')
    with pytest.raises(RiskBreach):
        run_tick(strategy, broker, provider, start, end, hooks=TickHooks(venue_belief=lambda: {}))
```

- [ ] **Step 2–4:** Run (fail) → replace the `derived_positions` branch with the `venue_belief` tolerance comparison vs `positions_before` (union of symbols; `abs` diff > `_RECONCILE_TOL`); delete the `derived_positions` field and `clear_derived_positions`; keep `TickResult.reconcile_ok`. Update any `test_live_loop.py` tests that referenced `derived_positions` to use `venue_belief`. → run (pass).

- [ ] **Step 5: Gate + commit** (`feat(249): single-tenant tolerance reconcile via venue_belief hook`).

### Task 9: Wire `trade-tick` reconcile end-to-end (phantom-flatten regression)

**Files:**
- Modify: `algua/cli/paper_cmd.py` (`trade_tick`: call `_ingest_paper_venue` before `run_tick`; pass `venue_belief=lambda: paper_believed_positions(conn, name)`; stamp `result.reconcile_ok`; `paper show` uses `derive_positions` only for sim)
- Test: `tests/test_paper_venue_reconcile.py`

**Interfaces:**
- Consumes: `_ingest_paper_venue`, `paper_believed_positions`, `venue_belief` hook.

- [ ] **Step 1: Write the regression test** — drive two real `trade_tick` invocations against a scripted fake paper broker: tick 1 (flat → flat) submits an order; simulate the venue fill landing (the broker's `account_activities_window` now returns the fill, and `snapshot`/`get_positions` now hold it); tick 2 must reconcile **clean** (no `RiskBreach`, `reconcile_ok=True`) — the phantom-flatten regression. A second test: with a fill the ledger never saw (orphan in broker book), tick 2 trips `RiskBreach('reconcile')`.

- [ ] **Step 2–4:** Run (fail) → in `trade_tick`, before building hooks call `_ingest_paper_venue(conn, broker)` (inside `try/except` per Task 11), set `hooks.venue_belief = lambda: paper_believed_positions(conn, name)`; keep stamping `result.reconcile_ok` on the snapshot. → run (pass).

- [ ] **Step 5: Gate + commit** (`feat(249): trade-tick reconciles venue belief — fixes #249 phantom flatten`).

---

## SLICE 4 — Offset flatten + fail-closed evidence

### Task 10: Strategy-scoped `submit_offset` flatten (breach handler + `flatten`)

**Files:**
- Modify: `algua/cli/paper_cmd.py` (breach handler in `trade_tick` ~341-363; `flatten` command ~469-501)
- Test: `tests/test_cli_paper.py`

**Interfaces:**
- Consumes: `paper_believed_positions`, `record_paper_venue_order`, `backfill_paper_venue_broker_order_id`, `broker.submit_offset`, `_RECONCILE_TOL`, `_ingest_paper_venue`.

- [ ] **Step 1: Write failing test** — a breach (or `flatten`) on strategy A, with A believed-holding `{AAA: 5}` and a sibling-held `BBB` on the account, submits exactly one `submit_offset("AAA", 5, …)` and never touches `BBB`; the payload reports `liquidation_submitted: True`; a believed qty `<= 1e-6` is skipped (no `submit_offset` call, no `"noop"` backfill).

- [ ] **Step 2–4:** Run (fail) → replace `broker.close_positions(symbols)` in both the breach handler and `flatten` with the offset loop from spec §"Strategy-scoped offset flatten" (ingest → iterate `paper_believed_positions` → skip `abs(qty)<=_RECONCILE_TOL` → `record_paper_venue_order` → `submit_offset` → `backfill_paper_venue_broker_order_id`); report `liquidation_submitted`. Remove the `clear_derived_positions` calls. → run (pass).

- [ ] **Step 5: Gate + commit** (`feat(249): strategy-scoped submit_offset flatten (no sibling liquidation)`).

### Task 11: Fail-closed evidence — ingestion failure aborts the tick with no snapshot

**Files:**
- Modify: `algua/cli/paper_cmd.py` (`trade_tick` ingest wrapping)
- Test: `tests/test_paper_venue_reconcile.py`

**Interfaces:**
- Produces: on `_ingest_paper_venue` raising, `trade_tick` emits a fail-closed error payload and `raise typer.Exit(1)` **before** `run_tick`, recording **no** tick snapshot.

- [ ] **Step 1: Write failing test** — scripted broker whose `account_activities_window` raises on the tick; assert `trade_tick` exits non-zero and `tick_snapshots` has **no** new row for the strategy (so no `reconcile_ok=True` is fabricated).

- [ ] **Step 2–4:** Run (fail) → wrap the `_ingest_paper_venue(conn, broker)` call (Task 9) in `try/except BrokerError/Exception` that emits a breach/error payload and exits 1 before `run_tick`; ensure the snapshot write is only reached on a successful tick. → run (pass).

- [ ] **Step 5: Gate + commit** (`feat(249): trade-tick fails closed on venue ingest failure (no snapshot)`).

### Task 12: `paper show` venue view for wall-clock strategies

**Files:**
- Modify: `algua/cli/paper_cmd.py` (`show` ~193-199), `algua/execution/order_state.py` if a venue count helper is needed
- Test: `tests/test_cli_paper.py`

**Interfaces:**
- Produces: `paper show` reports `positions = paper_believed_positions(conn, name)` and venue order count when the strategy has `paper_venue_orders` rows; otherwise the sim `derive_positions`/`paper_orders` view (unchanged for sim-only strategies).

- [ ] **Step 1: Write failing test** — a strategy with `paper_venue_fills` shows venue positions; a strategy with only sim `paper_fills` still shows the sim view.

- [ ] **Step 2–4:** Run (fail) → branch `show` on the presence of venue orders/fills (a cheap `SELECT 1 FROM paper_venue_orders WHERE strategy=? LIMIT 1`); keep `Stage.LIVE` path (live `believed_positions`) unchanged. → run (pass).

- [ ] **Step 5: Final gate + commit** (`feat(249): paper show reports venue belief for wall-clock strategies`).

---

## Self-review notes (coverage map)

- Spec §"Reconcile inside run_tick" → Tasks 8, 9. §"Cursor" → Task 3. §"Fail-closed evidence" → Task 11.
- §"Data model" → Task 1. §"LedgerKind seam" → Task 2. §"Crash-safe recording" → Tasks 4, 5, 6; reader move → Tasks 6, 7, 12.
- §"Strategy-scoped offset flatten" → Task 10. Sim/venue display split (Codex M11) → Task 12.
- Slice order honored: S2 (crash-safe record, Tasks 4–7) precedes S3 (reconcile, Tasks 8–9).
- After all tasks: full gate green, then GATE 2 (multi-model code review) on the branch diff before merge.
