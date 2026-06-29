# Multi-tenant attributed paper reconcile (#313) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the account-level integrity primitive a multi-tenant paper book needs — `attributed_paper_net` + a paper account `reconcile()` — by extracting the live reconcile algorithm into a shared, lane-agnostic core both lanes delegate to.

**Architecture:** Extract the grace-window + persisted-state reconcile algorithm from `live_reconcile` into a new `reconcile_core` parameterized by `(expected, state_table, cycle table)`. Refactor `live_reconcile` to delegate (behavior-preserving). Add a new `paper_reconcile` whose per-cycle reconcile gates on `attributed_paper_net` (orphans/non-paper excluded → fails closed on unattributable holdings). Purely additive: no `run_tick`/CLI change; nothing is wired yet.

**Tech Stack:** Python 3.12, SQLite (`paper_venue_fills` from #249; two new state tables), pytest.

## Global Constraints

- Run everything via `uv run ...`. Quality gate green before every commit: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`.
- Ruff line length ≤ 100 columns.
- `git add` only the named files — never `git add -A` (untracked WIP exists in the tree).
- Commit trailer on every commit: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- A `SCHEMA_VERSION` bump MUST be accompanied by its migration (a new table in `_SCHEMA`) — never bump the number without the DDL that earns it (`db.py` header rule).
- Keep the reconcile **behavior-preserving** for the live lane: `tests/test_live_reconcile.py` and `tests/test_live_ledger_orders.py` must stay green unchanged.
- `state_table` / cycle-table names passed to the core are controlled string constants (never user input); f-string interpolation of them in SQL is acceptable and intended.
- Branch: `paper-multitenant-reconcile-313` (already created off `main`, which has #249).

---

## File structure

- `algua/registry/db.py` — **Modify.** Add `paper_reconcile_state` + `paper_cycle` to `_SCHEMA`; bump `SCHEMA_VERSION` 30→31.
- `algua/execution/reconcile_core.py` — **Create.** Lane-agnostic `ReconcileResult`, `reconcile_account(...)`, `next_cycle(...)`, default constants.
- `algua/execution/live_reconcile.py` — **Modify.** Delegate `reconcile`/`next_cycle` to the core; keep `account_expected_net`, `attributed_live_net`.
- `algua/execution/paper_reconcile.py` — **Create.** `paper_account_expected_net`, `attributed_paper_net`, `reconcile`, `next_cycle`.
- `tests/test_reconcile_core.py` — **Create.** Core algorithm tests.
- `tests/test_paper_reconcile.py` — **Create.** Paper lane tests.
- `tests/test_registry_db.py` — **Modify.** Assert the two new tables + `SCHEMA_VERSION == 31`.

---

### Task 1: Schema v30→31 — paper reconcile state + cycle tables

**Files:**
- Modify: `algua/registry/db.py` (`_SCHEMA` string; `SCHEMA_VERSION`)
- Test: `tests/test_registry_db.py`

**Interfaces:**
- Consumes: existing `connect`, `migrate` from `algua.registry.db`.
- Produces: tables `paper_reconcile_state(symbol PK, expected_qty, broker_qty, first_seen_cycle, status)` and `paper_cycle(id PK CHECK(id=1), n)`; `SCHEMA_VERSION == 31`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_registry_db.py`:

```python
def test_paper_reconcile_and_cycle_tables_exist(tmp_path):
    from algua.registry.db import SCHEMA_VERSION, connect, migrate
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    tables = {r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert "paper_reconcile_state" in tables
    assert "paper_cycle" in tables
    assert SCHEMA_VERSION == 31
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_registry_db.py::test_paper_reconcile_and_cycle_tables_exist -v`
Expected: FAIL — tables absent / `SCHEMA_VERSION == 30`.

- [ ] **Step 3: Add the tables and bump the version**

In `algua/registry/db.py`, change `SCHEMA_VERSION = 30` to `SCHEMA_VERSION = 31`. Then append these two tables to the `_SCHEMA` string (place them next to the existing `live_reconcile_state` / `live_cycle` definitions for locality):

```sql
CREATE TABLE IF NOT EXISTS paper_reconcile_state (
    symbol           TEXT PRIMARY KEY,
    expected_qty     REAL NOT NULL,
    broker_qty       REAL NOT NULL,
    first_seen_cycle INTEGER NOT NULL,
    status           TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS paper_cycle (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    n  INTEGER NOT NULL
);
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_registry_db.py::test_paper_reconcile_and_cycle_tables_exist -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add algua/registry/db.py tests/test_registry_db.py
git commit -m "feat(db): paper_reconcile_state + paper_cycle tables (schema v31) for #313

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `reconcile_core` — extract the lane-agnostic algorithm

**Files:**
- Create: `algua/execution/reconcile_core.py`
- Test: `tests/test_reconcile_core.py`

**Interfaces:**
- Consumes: a `state_table` and cycle `table` (must already exist — Task 1 provides `paper_*`; `live_*` exist).
- Produces:
  - `@dataclass(frozen=True) ReconcileResult` with `clean: bool`, `halt: bool`, `mismatches: list[dict]`.
  - `DEFAULT_TOLERANCE = 1e-6`, `DEFAULT_GRACE_CYCLES = 3`.
  - `reconcile_account(conn, broker_net: dict[str, float], expected: dict[str, float], cycle: int, *, state_table: str, tolerance: float = DEFAULT_TOLERANCE, grace_cycles: int = DEFAULT_GRACE_CYCLES) -> ReconcileResult`.
  - `next_cycle(conn, *, table: str) -> int`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_reconcile_core.py` (tests use the `paper_*` tables from Task 1 to prove parameterization):

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_reconcile_core.py -v`
Expected: FAIL — `ModuleNotFoundError: algua.execution.reconcile_core`.

- [ ] **Step 3: Write the core module**

Create `algua/execution/reconcile_core.py` (this is the current `live_reconcile.reconcile`/`next_cycle` algorithm with `expected` injected and the table names parameterized):

```python
"""Lane-agnostic account reconcile: compare a books'-expected net position to the broker's net per
symbol, classifying mismatches with a grace window backed by a persisted per-symbol state table.
Both the live and paper lanes delegate here; each supplies its own `expected` net and table names."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

DEFAULT_TOLERANCE = 1e-6      # absolute share tolerance for rounding / fractional shares
DEFAULT_GRACE_CYCLES = 3      # cycles a mismatch may persist before it escalates to unexplained


@dataclass(frozen=True)
class ReconcileResult:
    clean: bool              # True == no mismatch at all -> safe to trade this cycle
    halt: bool               # True == a mismatch persisted past the grace window -> halt
    mismatches: list[dict]   # per-symbol {symbol, expected, broker, status}


def next_cycle(conn: sqlite3.Connection, *, table: str) -> int:
    """Monotonic, persisted cycle counter (survives restarts so the grace window is stable)."""
    conn.execute(
        f"INSERT INTO {table}(id, n) VALUES (1, 1) ON CONFLICT(id) DO UPDATE SET n = n + 1"
    )
    conn.commit()
    return int(conn.execute(f"SELECT n FROM {table} WHERE id = 1").fetchone()["n"])


def reconcile_account(
    conn: sqlite3.Connection,
    broker_net: dict[str, float],
    expected: dict[str, float],
    cycle: int,
    *,
    state_table: str,
    tolerance: float = DEFAULT_TOLERANCE,
    grace_cycles: int = DEFAULT_GRACE_CYCLES,
) -> ReconcileResult:
    """Compare `expected` (the caller's books-net) to `broker_net` per symbol. Within tolerance →
    clear any pending row. Otherwise record/keep a row keyed by first_seen_cycle; once it has
    persisted `grace_cycles`, mark it unexplained and signal halt. `clean` is True only when nothing
    mismatches (the caller trades only on a clean cycle; a pending mismatch defers, not halts)."""
    pending = {r["symbol"] for r in conn.execute(f"SELECT symbol FROM {state_table}")}
    symbols = set(expected) | set(broker_net) | pending
    mismatches: list[dict] = []
    halt = False
    for sym in sorted(symbols):
        diff = broker_net.get(sym, 0.0) - expected.get(sym, 0.0)
        if abs(diff) <= tolerance:
            conn.execute(f"DELETE FROM {state_table} WHERE symbol = ?", (sym,))
            continue
        row = conn.execute(
            f"SELECT first_seen_cycle FROM {state_table} WHERE symbol = ?", (sym,)
        ).fetchone()
        first_seen = int(row["first_seen_cycle"]) if row is not None else cycle
        status = "unexplained" if cycle - first_seen >= grace_cycles else "pending"
        conn.execute(
            f"INSERT INTO {state_table}"
            "(symbol, expected_qty, broker_qty, first_seen_cycle, status) VALUES (?,?,?,?,?)"
            " ON CONFLICT(symbol) DO UPDATE SET expected_qty = excluded.expected_qty,"
            "  broker_qty = excluded.broker_qty, status = excluded.status",
            (sym, expected.get(sym, 0.0), broker_net.get(sym, 0.0), first_seen, status),
        )
        mismatches.append({
            "symbol": sym,
            "expected": expected.get(sym, 0.0),
            "broker": broker_net.get(sym, 0.0),
            "status": status,
        })
        if status == "unexplained":
            halt = True
    conn.commit()
    return ReconcileResult(clean=not mismatches, halt=halt, mismatches=mismatches)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_reconcile_core.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add algua/execution/reconcile_core.py tests/test_reconcile_core.py
git commit -m "feat(execution): lane-agnostic reconcile_core (extracted grace/state algorithm) #313

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Refactor `live_reconcile` to delegate to the core

**Files:**
- Modify: `algua/execution/live_reconcile.py`
- Test: `tests/test_live_reconcile.py` + `tests/test_live_ledger_orders.py` (existing — must stay green unchanged)

**Interfaces:**
- Consumes: `reconcile_core.reconcile_account`, `reconcile_core.next_cycle`, `ReconcileResult`, `DEFAULT_TOLERANCE`, `DEFAULT_GRACE_CYCLES`.
- Produces (unchanged public surface): `account_expected_net(conn)`, `attributed_live_net(conn)`, `next_cycle(conn) -> int`, `reconcile(conn, broker_net, cycle, tolerance=..., grace_cycles=...) -> ReconcileResult`. `live_cmd.py` calls `live_reconcile.next_cycle` / `live_reconcile.reconcile`; `paper_cmd.py` imports `attributed_live_net` — these names must remain.

- [ ] **Step 1: Confirm the existing live tests pass before the change (baseline)**

Run: `uv run pytest tests/test_live_reconcile.py tests/test_live_ledger_orders.py -v`
Expected: PASS (record the count — it must be identical after the refactor).

- [ ] **Step 2: Rewrite `live_reconcile.py` to delegate**

Replace the module body so the duplicated algorithm and constants are gone, the `ReconcileResult`/algorithm come from the core, and `account_expected_net` / `attributed_live_net` stay. Final file:

```python
"""Account-level reconcile for the LIVE lane: the books' expected net (Σ all live_fills) must match
the broker's netted book per symbol. The grace-window + persisted-state algorithm lives in
`reconcile_core`; this module supplies the live `expected` net and the live table names."""
from __future__ import annotations

import sqlite3

from algua.execution.reconcile_core import (
    DEFAULT_GRACE_CYCLES,
    DEFAULT_TOLERANCE,
    ReconcileResult,
    reconcile_account,
)
from algua.execution.reconcile_core import next_cycle as _core_next_cycle

__all__ = ["ReconcileResult", "account_expected_net", "attributed_live_net", "next_cycle", "reconcile"]


def account_expected_net(conn: sqlite3.Connection) -> dict[str, float]:
    """The books' belief of the account net per symbol = Σ all live_fills.qty (signed), across every
    strategy (the account is shared). Zero nets are omitted."""
    rows = conn.execute("SELECT symbol, SUM(qty) AS q FROM live_fills GROUP BY symbol").fetchall()
    return {r["symbol"]: float(r["q"]) for r in rows if float(r["q"]) != 0.0}


def attributed_live_net(conn: sqlite3.Connection) -> dict[str, float]:
    """The books' expected net per symbol counting ONLY fills attributed to a CURRENTLY-LIVE
    strategy. Orphan fills (strategy IS NULL) and non-live fills are EXCLUDED, so they can never
    'explain' a broker position. Zero nets are omitted."""
    rows = conn.execute(
        "SELECT f.symbol AS symbol, SUM(f.qty) AS q FROM live_fills f "
        "JOIN strategies s ON s.name = f.strategy AND s.stage = 'live' "
        "GROUP BY f.symbol"
    ).fetchall()
    return {r["symbol"]: float(r["q"]) for r in rows if float(r["q"]) != 0.0}


def next_cycle(conn: sqlite3.Connection) -> int:
    return _core_next_cycle(conn, table="live_cycle")


def reconcile(
    conn: sqlite3.Connection,
    broker_net: dict[str, float],
    cycle: int,
    tolerance: float = DEFAULT_TOLERANCE,
    grace_cycles: int = DEFAULT_GRACE_CYCLES,
) -> ReconcileResult:
    return reconcile_account(
        conn, broker_net, account_expected_net(conn), cycle,
        state_table="live_reconcile_state", tolerance=tolerance, grace_cycles=grace_cycles,
    )
```

- [ ] **Step 3: Run the existing live tests + core tests (regression)**

Run: `uv run pytest tests/test_live_reconcile.py tests/test_live_ledger_orders.py tests/test_reconcile_core.py -v`
Expected: PASS, with the **same** live test count as Step 1 (proves behavior-preserving).

- [ ] **Step 4: Type + lint the touched files**

Run: `uv run mypy algua/execution/live_reconcile.py algua/execution/reconcile_core.py && uv run ruff check algua/execution/live_reconcile.py`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add algua/execution/live_reconcile.py
git commit -m "refactor(execution): live_reconcile delegates to reconcile_core (behavior-preserving) #313

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: `paper_reconcile` — the paper lane (attributed, fail-closed)

**Files:**
- Create: `algua/execution/paper_reconcile.py`
- Test: `tests/test_paper_reconcile.py`

**Interfaces:**
- Consumes: `reconcile_core.reconcile_account` / `next_cycle` / `ReconcileResult` / defaults; tables `paper_venue_fills` (#249), `paper_reconcile_state`, `paper_cycle` (Task 1).
- Produces:
  - `paper_account_expected_net(conn) -> dict[str, float]` — Σ all `paper_venue_fills` per symbol.
  - `attributed_paper_net(conn) -> dict[str, float]` — Σ fills joined to `strategies.stage = 'paper'`.
  - `next_cycle(conn) -> int` (paper_cycle).
  - `reconcile(conn, broker_net, cycle, tolerance=..., grace_cycles=...) -> ReconcileResult` — gates on `attributed_paper_net`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_paper_reconcile.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_paper_reconcile.py -v`
Expected: FAIL — `ModuleNotFoundError: algua.execution.paper_reconcile`.

- [ ] **Step 3: Write the paper module**

Create `algua/execution/paper_reconcile.py`:

```python
"""Account-level reconcile for the PAPER lane (#313). Mirrors live_reconcile but gates on
`attributed_paper_net`: an account holding NO current-paper strategy owns leaves a residual and
fails closed — the multi-tenant safety semantics. The grace window (in reconcile_core) absorbs a
just-ingested fill whose order is not yet broker-id-backfilled (briefly orphan -> pending, not halt)."""
from __future__ import annotations

import sqlite3

from algua.execution.reconcile_core import (
    DEFAULT_GRACE_CYCLES,
    DEFAULT_TOLERANCE,
    ReconcileResult,
    reconcile_account,
)
from algua.execution.reconcile_core import next_cycle as _core_next_cycle

__all__ = [
    "attributed_paper_net", "next_cycle", "paper_account_expected_net", "reconcile",
]


def paper_account_expected_net(conn: sqlite3.Connection) -> dict[str, float]:
    """Σ ALL paper_venue_fills.qty (signed) per symbol — every recorded paper fill, attributed or
    not. Zero nets omitted. (Diagnostic/account analog of live account_expected_net.)"""
    rows = conn.execute(
        "SELECT symbol, SUM(qty) AS q FROM paper_venue_fills GROUP BY symbol").fetchall()
    return {r["symbol"]: float(r["q"]) for r in rows if float(r["q"]) != 0.0}


def attributed_paper_net(conn: sqlite3.Connection) -> dict[str, float]:
    """Σ paper_venue_fills counting ONLY fills attributed to a CURRENTLY-PAPER strategy. Orphan
    (strategy IS NULL) and non-paper fills are EXCLUDED so they can never 'explain' a broker
    position. Zero nets omitted."""
    rows = conn.execute(
        "SELECT f.symbol AS symbol, SUM(f.qty) AS q FROM paper_venue_fills f "
        "JOIN strategies s ON s.name = f.strategy AND s.stage = 'paper' "
        "GROUP BY f.symbol"
    ).fetchall()
    return {r["symbol"]: float(r["q"]) for r in rows if float(r["q"]) != 0.0}


def next_cycle(conn: sqlite3.Connection) -> int:
    return _core_next_cycle(conn, table="paper_cycle")


def reconcile(
    conn: sqlite3.Connection,
    broker_net: dict[str, float],
    cycle: int,
    tolerance: float = DEFAULT_TOLERANCE,
    grace_cycles: int = DEFAULT_GRACE_CYCLES,
) -> ReconcileResult:
    return reconcile_account(
        conn, broker_net, attributed_paper_net(conn), cycle,
        state_table="paper_reconcile_state", tolerance=tolerance, grace_cycles=grace_cycles,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_paper_reconcile.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Full quality gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add algua/execution/paper_reconcile.py tests/test_paper_reconcile.py
git commit -m "feat(execution): paper_reconcile — attributed_paper_net + fail-closed account reconcile #313

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-review

- **Spec coverage:** §3 `reconcile_core` → Task 2; `live_reconcile` refactor → Task 3; `paper_reconcile` (`paper_account_expected_net`, `attributed_paper_net`, `reconcile` on attributed, `next_cycle`) → Task 4; §4 schema v31 tables → Task 1; §5 testing → tests in Tasks 2/3/4. §6 non-goals respected (no run_tick/CLI/run-all/NAV/forward-gate change). §7 risk (live behavior-preserving) → Task 3 Steps 1+3.
- **Placeholder scan:** none — every code step has full code.
- **Type consistency:** `reconcile_account(conn, broker_net, expected, cycle, *, state_table, tolerance, grace_cycles)` and `next_cycle(conn, *, table)` are defined identically in Task 2 and called with those exact signatures in Tasks 3 and 4; `ReconcileResult` fields (`clean`/`halt`/`mismatches`) consistent throughout; `paper_account_expected_net`/`attributed_paper_net` names match the spec.
