# Multi-Strategy Live Accounting — Slice B (Portfolio Loop + Reconcile) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `algua live run-all` — a single sequenced portfolio loop that, each cycle, re-verifies each live strategy, ingests fills (Slice A), reconciles the account books against the broker (classified, with a grace window), and ticks each live strategy with a **scoped** owned-order cancel. Sizing semantics are carried over from slice 3 unchanged (Slice C flips sizing). Real money only moves when the account reconciles clean.

**Architecture:** A new `algua/execution/live_reconcile.py` (account expected-net + classified reconcile + cycle counter). Scoped-cancel primitives on the broker (`list_open_orders`/`cancel_order`) + a loop helper. A `cancel` hook in `run_tick` so the live path cancels only a strategy's own orders (account-wide stays paper-only / global-halt-only). The `live run-all` command, reusing an extracted `_run_strategy_tick` helper. Schema bump 13→14 (`live_reconcile_state`, `live_cycle`).

**Tech Stack:** Python 3.12, sqlite3, Typer, requests, pytest, ruff, mypy, import-linter. Spec: `docs/superpowers/specs/2026-06-06-multi-strategy-live-accounting-design.md` (§3 loop, §5 reconcile). Builds on Slice A (`live_ledger`, `allocations`, schema v13).

**Defaults:** reconcile tolerance `1e-6` shares; grace window `3` cycles.

---

## File structure

| File | Responsibility |
|---|---|
| `algua/registry/db.py` (modify) | `live_reconcile_state`, `live_cycle` tables; `SCHEMA_VERSION` 13→14. |
| `algua/execution/live_reconcile.py` (new) | `account_expected_net`, `next_cycle`, `reconcile` (classified, grace window), `ReconcileResult`. |
| `algua/execution/alpaca_broker.py` (modify) | `list_open_orders()`, `cancel_order(order_id)`. |
| `algua/execution/live_ledger.py` (modify) | `owned_open_order_ids(conn, broker, strategy)` (attribute open orders to a strategy). |
| `algua/live/live_loop.py` (modify) | `TickHooks.cancel` hook; `run_tick` uses it instead of always account-wide cancel. |
| `algua/cli/live_cmd.py` (modify) | extract `_run_strategy_tick`; add `live run-all` + `_scoped_cancel`. |

---

### Task 1: schema + classified reconcile module

**Files:** Modify `algua/registry/db.py`; Create `algua/execution/live_reconcile.py`; Test `tests/test_live_reconcile.py`.

Context: Slice A's `live_fills.qty` is SIGNED. The account's expected net per symbol = Σ all `live_fills.qty` grouped by symbol (every strategy's fills, since the account is shared). The reconcile compares that to the broker's actual net book and classifies per-symbol mismatches: within tolerance → reconciled (clear any pending row); newly mismatched → `pending` (record `first_seen_cycle`); mismatched for ≥ `grace_cycles` → `unexplained` (→ halt). `next_cycle` is a persisted monotonic counter so the grace window survives restarts.

- [ ] **Step 1: Add the tables.** In `algua/registry/db.py`, bump `SCHEMA_VERSION = 13` → `14` and append to `_SCHEMA`:
```sql
CREATE TABLE IF NOT EXISTS live_reconcile_state (
    symbol           TEXT PRIMARY KEY,
    expected_qty     REAL NOT NULL,
    broker_qty       REAL NOT NULL,
    first_seen_cycle INTEGER NOT NULL,
    status           TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS live_cycle (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    n  INTEGER NOT NULL
);
```

- [ ] **Step 2: Write the failing tests.** Create `tests/test_live_reconcile.py`:
```python
from algua.execution import live_reconcile as R
from algua.registry.db import connect, migrate


def _conn(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    return conn


def _fill(conn, aid, strategy, symbol, qty, price="100"):
    conn.execute(
        "INSERT INTO live_fills(activity_id, broker_order_id, strategy, symbol, qty, price, fill_ts)"
        " VALUES (?,?,?,?,?,?,?)",
        (aid, "b", strategy, symbol, qty, price, "2026-06-06T00:00:00+00:00"),
    )
    conn.commit()


def test_account_expected_net_sums_all_strategies(tmp_path):
    conn = _conn(tmp_path)
    _fill(conn, "a1", "s1", "AAA", 10.0)
    _fill(conn, "a2", "s2", "AAA", 5.0)   # a sibling holds the same symbol
    _fill(conn, "a3", "s1", "BBB", -3.0)
    assert R.account_expected_net(conn) == {"AAA": 15.0, "BBB": -3.0}


def test_next_cycle_is_monotonic_and_persistent(tmp_path):
    conn = _conn(tmp_path)
    assert R.next_cycle(conn) == 1
    assert R.next_cycle(conn) == 2


def test_reconcile_clean_when_books_match_broker(tmp_path):
    conn = _conn(tmp_path)
    _fill(conn, "a1", "s1", "AAA", 10.0)
    res = R.reconcile(conn, broker_net={"AAA": 10.0}, cycle=1)
    assert res.clean and not res.halt and res.mismatches == []


def test_reconcile_tolerates_rounding(tmp_path):
    conn = _conn(tmp_path)
    _fill(conn, "a1", "s1", "AAA", 10.0)
    res = R.reconcile(conn, broker_net={"AAA": 10.0 + 5e-7}, cycle=1)
    assert res.clean and not res.halt


def test_reconcile_pending_then_escalates_to_halt(tmp_path):
    conn = _conn(tmp_path)
    _fill(conn, "a1", "s1", "AAA", 10.0)
    broker = {"AAA": 12.0}  # broker holds 2 more than the books explain
    r1 = R.reconcile(conn, broker, cycle=1)
    assert not r1.clean and not r1.halt and r1.mismatches[0]["status"] == "pending"
    r2 = R.reconcile(conn, broker, cycle=2)
    assert not r2.clean and not r2.halt  # still within grace (default 3)
    r3 = R.reconcile(conn, broker, cycle=4)  # cycle - first_seen (1) >= 3 -> unexplained
    assert r3.halt and r3.mismatches[0]["status"] == "unexplained"


def test_reconcile_clears_pending_when_it_resolves(tmp_path):
    conn = _conn(tmp_path)
    _fill(conn, "a1", "s1", "AAA", 10.0)
    R.reconcile(conn, {"AAA": 12.0}, cycle=1)            # pending recorded
    R.reconcile(conn, {"AAA": 10.0}, cycle=2)            # resolves -> row cleared
    assert conn.execute("SELECT COUNT(*) FROM live_reconcile_state").fetchone()[0] == 0
```

- [ ] **Step 3: Run** `cd /home/liornisimov/Projects/algua/.claude/worktrees/sp6-live-runall && uv run pytest tests/test_live_reconcile.py -q` → FAIL.

- [ ] **Step 4: Implement** `algua/execution/live_reconcile.py`:
```python
"""Account-level reconcile: the books' expected net position (Σ all strategies' signed fills) must
match the broker's netted book per symbol. Mismatches are CLASSIFIED, not binary-halted — a brief
timing skew (a fill in positions but not yet in the activities feed) is tolerated for a grace window
and only a persistent unexplained gap halts the account. The grace window survives restarts via a
persisted cycle counter."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

_TOLERANCE = 1e-6      # absolute share tolerance for rounding / fractional shares
_GRACE_CYCLES = 3      # cycles a mismatch may persist before it escalates to unexplained -> halt


@dataclass(frozen=True)
class ReconcileResult:
    clean: bool              # True == no mismatch at all -> safe to trade this cycle
    halt: bool               # True == a mismatch persisted past the grace window -> global halt
    mismatches: list[dict]   # per-symbol {symbol, expected, broker, status}


def account_expected_net(conn: sqlite3.Connection) -> dict[str, float]:
    """The books' belief of the account net position per symbol = Σ all live_fills.qty (signed),
    across every strategy (the account is shared). Zero nets are omitted."""
    rows = conn.execute("SELECT symbol, SUM(qty) AS q FROM live_fills GROUP BY symbol").fetchall()
    return {r["symbol"]: float(r["q"]) for r in rows if float(r["q"]) != 0.0}


def next_cycle(conn: sqlite3.Connection) -> int:
    """Monotonic, persisted cycle counter (survives restarts so the grace window is stable)."""
    conn.execute(
        "INSERT INTO live_cycle(id, n) VALUES (1, 1) ON CONFLICT(id) DO UPDATE SET n = n + 1"
    )
    conn.commit()
    return int(conn.execute("SELECT n FROM live_cycle WHERE id = 1").fetchone()["n"])


def reconcile(conn: sqlite3.Connection, broker_net: dict[str, float], cycle: int,
              tolerance: float = _TOLERANCE, grace_cycles: int = _GRACE_CYCLES) -> ReconcileResult:
    """Compare the books' expected net to the broker's net per symbol. Within tolerance → clear any
    pending row. Otherwise record/keep a pending row keyed by first_seen_cycle; once it has persisted
    `grace_cycles`, mark it unexplained and signal halt. `clean` is True only when NOTHING mismatches
    (the caller trades only on a clean cycle; a pending mismatch defers trading, not halts)."""
    expected = account_expected_net(conn)
    symbols = set(expected) | set(broker_net)
    mismatches: list[dict] = []
    halt = False
    for sym in sorted(symbols):
        diff = broker_net.get(sym, 0.0) - expected.get(sym, 0.0)
        if abs(diff) <= tolerance:
            conn.execute("DELETE FROM live_reconcile_state WHERE symbol = ?", (sym,))
            continue
        row = conn.execute(
            "SELECT first_seen_cycle FROM live_reconcile_state WHERE symbol = ?", (sym,)
        ).fetchone()
        first_seen = int(row["first_seen_cycle"]) if row is not None else cycle
        status = "unexplained" if cycle - first_seen >= grace_cycles else "pending"
        conn.execute(
            "INSERT INTO live_reconcile_state"
            "(symbol, expected_qty, broker_qty, first_seen_cycle, status) VALUES (?,?,?,?,?)"
            " ON CONFLICT(symbol) DO UPDATE SET expected_qty = excluded.expected_qty,"
            "  broker_qty = excluded.broker_qty, status = excluded.status",
            (sym, expected.get(sym, 0.0), broker_net.get(sym, 0.0), first_seen, status),
        )
        mismatches.append({"symbol": sym, "expected": expected.get(sym, 0.0),
                           "broker": broker_net.get(sym, 0.0), "status": status})
        if status == "unexplained":
            halt = True
    conn.commit()
    return ReconcileResult(clean=not mismatches, halt=halt, mismatches=mismatches)
```

- [ ] **Step 5: Run** the test → PASS.

- [ ] **Step 6: Gate + commit.** `uv run ruff check . && uv run mypy algua && uv run pytest tests/test_live_reconcile.py -q && uv run lint-imports`
```bash
git add algua/registry/db.py algua/execution/live_reconcile.py tests/test_live_reconcile.py
git commit -m "feat(live-loop): account-level classified reconcile + cycle counter (schema v14)"
```

---

### Task 2: scoped-cancel broker primitives + owned-order attribution

**Files:** Modify `algua/execution/alpaca_broker.py`, `algua/execution/live_ledger.py`; Test `tests/test_alpaca_broker.py`, `tests/test_live_ledger_orders.py`.

Context: account-wide `cancel_open_orders()` would cancel siblings' orders (Codex CRITICAL). Scoped cancel = list the account's open orders, keep those whose `client_order_id` maps to THIS strategy (via `live_orders.strategy`), cancel each by id. `_AlpacaBroker` has `_get`/`_read`/`_delete`. `tests/test_alpaca_broker.py` has `_FakeRequests` (routes GET by URL suffix; records POST; add DELETE routing if absent — read it first) + `_FakeResp` + `_broker()`.

- [ ] **Step 1: Write the failing tests.** Append to `tests/test_alpaca_broker.py` (FIRST read `_FakeRequests`; if it has no `delete` method, add one that routes like `get` and records the path):
```python
def test_list_open_orders(monkeypatch):
    monkeypatch.setattr(ab, "requests", _FakeRequests(
        {"/v2/orders?status=open": _FakeResp(200, [{"id": "o1", "client_order_id": "c1"}])}))
    assert _broker().list_open_orders() == [{"id": "o1", "client_order_id": "c1"}]


def test_cancel_order_by_id(monkeypatch):
    fake = _FakeRequests({}, )
    monkeypatch.setattr(ab, "requests", fake)
    # a 204 (or 200) is success; a 404 (already gone) is a no-op, not an error
    monkeypatch.setattr(fake, "delete", lambda url, headers=None, timeout=None: _FakeResp(204))
    _broker().cancel_order("o1")  # must not raise
```
Append to `tests/test_live_ledger_orders.py`:
```python
def test_owned_open_order_ids_filters_to_strategy(tmp_path):
    conn = _conn(tmp_path)
    L.record_live_order(conn, "s1", "AAA", "buy", 1000.0, "c1")
    L.record_live_order(conn, "s2", "BBB", "buy", 1000.0, "c2")

    class _B:
        def list_open_orders(self):
            return [{"id": "o1", "client_order_id": "c1"},   # s1's
                    {"id": "o2", "client_order_id": "c2"},   # s2's
                    {"id": "o3", "client_order_id": "unknown"}]  # not ours

    assert L.owned_open_order_ids(conn, _B(), "s1") == ["o1"]
```

- [ ] **Step 2: Run** `uv run pytest "tests/test_alpaca_broker.py::test_list_open_orders" "tests/test_alpaca_broker.py::test_cancel_order_by_id" "tests/test_live_ledger_orders.py::test_owned_open_order_ids_filters_to_strategy" -q` → FAIL.

- [ ] **Step 3a: Broker primitives.** Add to `_AlpacaBroker` in `algua/execution/alpaca_broker.py`:
```python
    def list_open_orders(self) -> list[Any]:
        """All OPEN orders on the account (GET /v2/orders?status=open). Each carries `id` and
        `client_order_id`; the caller scopes cancellation to a strategy by client_order_id."""
        return self._read(self._get("/v2/orders?status=open"), "/v2/orders")

    def cancel_order(self, order_id: str) -> None:
        """Cancel ONE order by broker id (DELETE /v2/orders/{id}). 404/422 (already gone/terminal)
        is a no-op; any other non-2xx raises BrokerError."""
        path = f"/v2/orders/{order_id}"
        resp = self._delete(path)
        if resp.status_code in (404, 422):
            return
        self._read(resp, path, ok=(200, 204))
```

- [ ] **Step 3b: Owned-order attribution.** Append to `algua/execution/live_ledger.py`:
```python
def owned_open_order_ids(conn: sqlite3.Connection, broker: object, strategy: str) -> list[str]:
    """The broker order ids of THIS strategy's currently-open orders: list the account's open
    orders and keep those whose client_order_id maps (via live_orders) to `strategy`. Used to
    scope cancellation so one strategy never cancels a sibling's orders."""
    open_orders = broker.list_open_orders()  # type: ignore[attr-defined]
    owned = {
        r["client_order_id"]
        for r in conn.execute(
            "SELECT client_order_id FROM live_orders WHERE strategy = ?", (strategy,)
        )
    }
    return [o["id"] for o in open_orders if o.get("client_order_id") in owned]
```

- [ ] **Step 4: Run** the tests → PASS.

- [ ] **Step 5: Gate + commit.** `uv run ruff check . && uv run mypy algua && uv run pytest tests/test_alpaca_broker.py tests/test_live_ledger_orders.py -q && uv run lint-imports`
```bash
git add algua/execution/alpaca_broker.py algua/execution/live_ledger.py tests/test_alpaca_broker.py tests/test_live_ledger_orders.py
git commit -m "feat(live-loop): scoped-cancel primitives (list_open_orders/cancel_order) + owned-order attribution"
```

---

### Task 3: `cancel` hook in run_tick (scoped cancel seam)

**Files:** Modify `algua/live/live_loop.py`; Test `tests/test_live_loop.py`.

Context: `run_tick` calls `broker.cancel_open_orders()` (account-wide) at line ~147 before the submit phase. For multi-strategy live that cancels siblings. Add a `cancel` hook to `TickHooks`; `run_tick` calls `hooks.cancel()` if supplied else the account-wide default — behaviour-preserving for paper (which passes no `cancel`).

- [ ] **Step 1: Write the failing test.** Append to `tests/test_live_loop.py` (read `_FakeBroker`/`_FakeProvider`/`_strategy`/`_bars`/`DATES`/`TickHooks` first):
```python
def test_run_tick_uses_cancel_hook_when_supplied():
    from algua.live.live_loop import TickHooks
    broker = _FakeBroker()
    called = {"scoped": 0, "account_wide": 0}
    broker.cancel_open_orders = lambda: called.__setitem__("account_wide",
                                                           called["account_wide"] + 1)
    hooks = TickHooks(cancel=lambda: called.__setitem__("scoped", called["scoped"] + 1))
    run_tick(_strategy({"AAA": 0.5}), broker, _FakeProvider(_bars({"AAA": [100.0, 100.0, 100.0]})),
             DATES[0], DATES[-1], hooks=hooks)
    assert called == {"scoped": 1, "account_wide": 0}  # the hook replaced the account-wide cancel
```

- [ ] **Step 2: Run** `uv run pytest tests/test_live_loop.py::test_run_tick_uses_cancel_hook_when_supplied -q` → FAIL.

- [ ] **Step 3: Implement.** In `algua/live/live_loop.py`:
  - add the field to `TickHooks` (after `should_halt`):
```python
    cancel: Callable[[], None] | None = None
```
  with a docstring line in the `TickHooks` docstring:
```
    - `cancel() -> None`: how to cancel stale open orders before the submit phase. Defaults to the
      broker's ACCOUNT-WIDE cancel (paper); the live multi-strategy loop supplies a SCOPED cancel so
      a strategy never cancels a sibling's orders.
```
  - replace the call site `broker.cancel_open_orders()` (line ~147) with:
```python
    (hooks.cancel or broker.cancel_open_orders)()
```

- [ ] **Step 4: Run** the test → PASS. Also run `uv run pytest tests/test_live_loop.py -q` to confirm no regression (paper path unchanged when `cancel is None`).

- [ ] **Step 5: Gate + commit.** `uv run ruff check . && uv run mypy algua && uv run pytest tests/test_live_loop.py -q && uv run lint-imports`
```bash
git add algua/live/live_loop.py tests/test_live_loop.py
git commit -m "feat(live-loop): TickHooks.cancel seam (scoped cancel for live; account-wide default for paper)"
```

---

### Task 4: `live run-all` portfolio loop

**Files:** Modify `algua/cli/live_cmd.py`; Test `tests/test_cli_live.py`.

Context: `live_cmd.py` has `trade_tick` (the single-strategy live tick) — extract its per-strategy body into a reusable `_run_strategy_tick(conn, name, authorization, broker, provider, max_drawdown) -> dict` that does the hooks + run_tick + breach handling + snapshot persistence and RETURNS a result dict (instead of `emit`-ing), so both `trade-tick` and `run-all` call it. `run-all` then: list live strategies → global-halt check → re-verify each (skip+flag failures) → build ONE account broker from a verified authorization → ingest activities → reconcile → (halt / skip / trade) → tick each verified strategy with a SCOPED cancel hook. The ≤1-live guard (Slice A) means ≤1 strategy runs until Slice C, but the loop is built for many. Imports needed: `from algua.execution import live_reconcile`, `from algua.execution.live_ledger import ingest_activities, fill_cursor, owned_open_order_ids`, `from algua.contracts.lifecycle import Stage`.

- [ ] **Step 1: Write the failing tests.** Append to `tests/test_cli_live.py` (reuses `_isolated`, `_to_live`, `_auth`, `runner`, `app`; monkeypatches as the happy-path test does):
```python
def test_run_all_no_live_strategies_is_noop(monkeypatch):
    r = runner.invoke(app, ["live", "run-all", "--snapshot", "x"])
    assert r.exit_code == 0
    assert json.loads(r.stdout)["strategies"] == []


def test_run_all_halts_on_unexplained_reconcile_drift(monkeypatch):
    from algua.registry import global_halt
    _to_live()
    monkeypatch.setattr("algua.cli.live_cmd.verify_live_authorization", lambda *a, **k: _auth())
    monkeypatch.setattr("algua.cli.live_cmd._alpaca_live_broker", lambda auth: object())
    monkeypatch.setattr("algua.cli.live_cmd._select_provider", lambda demo, snapshot: object())
    # broker activities -> none; positions -> an unexplained holding the books don't have
    monkeypatch.setattr("algua.cli.live_cmd.ingest_activities", lambda conn, acts: None)
    monkeypatch.setattr("algua.cli.live_cmd.fill_cursor", lambda conn: None)
    monkeypatch.setattr("algua.cli.live_cmd._broker_account_activities", lambda broker, after: [])
    monkeypatch.setattr("algua.cli.live_cmd._broker_net_positions", lambda broker: {"ZZZ": 99.0})
    # --grace-cycles 0 forces the mismatch straight to unexplained -> assert global halt engaged
    r = runner.invoke(app, ["live", "run-all", "--snapshot", "x", "--grace-cycles", "0"])
    assert r.exit_code == 1
    payload = json.loads(r.stdout)
    assert payload["ok"] is False and payload["reconcile"]["halt"] is True
    from contextlib import closing

    from algua.config.settings import get_settings
    from algua.registry.db import connect, migrate
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        assert global_halt.is_engaged(conn)


def test_run_all_ticks_strategy_when_clean(monkeypatch):
    from algua.live.live_loop import TickResult
    _to_live()
    fake = TickResult(decision_ts=None, target_weights={}, positions_before={},
                      submitted=[], equity=10_000.0, peak_equity=10_000.0)
    monkeypatch.setattr("algua.cli.live_cmd.verify_live_authorization", lambda *a, **k: _auth())
    monkeypatch.setattr("algua.cli.live_cmd._alpaca_live_broker", lambda auth: object())
    monkeypatch.setattr("algua.cli.live_cmd._select_provider", lambda demo, snapshot: object())
    monkeypatch.setattr("algua.cli.live_cmd.ingest_activities", lambda conn, acts: None)
    monkeypatch.setattr("algua.cli.live_cmd.fill_cursor", lambda conn: None)
    monkeypatch.setattr("algua.cli.live_cmd._broker_account_activities", lambda broker, after: [])
    monkeypatch.setattr("algua.cli.live_cmd._broker_net_positions", lambda broker: {})
    monkeypatch.setattr("algua.cli.live_cmd._run_strategy_tick",
                        lambda *a, **k: {"strategy": "cross_sectional_momentum", "submitted": []})
    r = runner.invoke(app, ["live", "run-all", "--snapshot", "x"])
    assert r.exit_code == 0
    payload = json.loads(r.stdout)
    assert payload["reconcile"]["clean"] is True
    assert payload["strategies"][0]["strategy"] == "cross_sectional_momentum"
```

- [ ] **Step 2: Run** `uv run pytest tests/test_cli_live.py -k run_all -q` → FAIL.

- [ ] **Step 3a: Extract `_run_strategy_tick`.** Refactor `trade_tick` in `algua/cli/live_cmd.py`: move the body from `strategy = load_strategy(name)` through the snapshot-persistence block into a helper that RETURNS a dict instead of emitting, and takes a `cancel` callable for the tick hook. `trade_tick` calls it and `emit(ok(result))`. Signature:
```python
def _run_strategy_tick(conn, name: str, authorization, broker, provider, max_drawdown,
                       cancel=None) -> dict:
    """Drive ONE strategy's live tick: hooks (incl. the scoped `cancel`), run_tick, breach handling
    (trip + scoped flatten), snapshot persistence. Returns a result dict; raises typer.Exit(1) with
    an emitted breach/halt payload on TickHalted/RiskBreach (same behaviour as the single-strategy
    command)."""
    strategy = load_strategy(name)
    def _persist(record: SubmittedOrder) -> None:
        audit_append(conn, actor="agent", action="live_order",
                     reason=f"{record.side} {record.symbol} {record.order_id}", strategy=name)
    hooks = TickHooks(
        client_order_id_for=client_order_id, on_submitted=_persist, cancel=cancel,
        should_halt=lambda: (kill_switch.is_tripped(conn, name) or global_halt.is_engaged(conn)
                             or not authorization_active(conn, authorization)),
        peak_equity=get_peak_equity(conn, name), derived_positions=None,
    )
    # ... (the existing try/except TickHalted/RiskBreach + snapshot block, but `return {...}`
    #      on success instead of emit) ...
```
Keep `trade_tick` behaviour identical (it now calls `_run_strategy_tick(conn, name, authorization, broker, provider, max_drawdown)` then `emit(ok(result))`). Confirm the existing `trade-tick` tests still pass.

- [ ] **Step 3b: Add broker-call shims + `run-all`.** Add thin wrappers (so tests can monkeypatch them) and the command:
```python
def _broker_account_activities(broker, after):
    return broker.account_activities(after=after)


def _broker_net_positions(broker) -> dict:
    pos = broker.get_positions()  # pandas Series symbol->qty
    return {sym: float(q) for sym, q in pos.items() if float(q) != 0.0}


@live_app.command("run-all")
@json_errors(ValueError, LookupError, BrokerError, LiveAuthorizationError)
def run_all(
    snapshot: str = typer.Option(..., "--snapshot"),
    start: str = typer.Option("2023-01-01", "--start"),
    end: str = typer.Option("2023-12-31", "--end"),
    max_drawdown: float = typer.Option(None, "--max-drawdown"),
    grace_cycles: int = typer.Option(3, "--grace-cycles",
                                     help="cycles a reconcile mismatch may persist before halting"),
    tolerance: float = typer.Option(1e-6, "--tolerance", help="reconcile share tolerance"),
) -> None:
    """One sequenced portfolio cycle over ALL live strategies: re-verify each, ingest fills,
    reconcile the account against the broker, then tick each (scoped cancel). Trades only when the
    account reconciles clean; a persistent unexplained drift engages the global halt."""
    with registry_conn() as conn:
        repo = SqliteStrategyRepository(conn)
        live = repo.list_strategies(Stage.LIVE)
        if not live:
            emit(ok({"strategies": [], "note": "no live strategies"}))
            return
        if global_halt.is_engaged(conn):
            emit(_breach_payload("global halt engaged", halted=True))
            raise typer.Exit(1)
        # re-verify each; skip + flag failures, keep one authorization for the account broker
        verified: list[tuple[str, object]] = []
        skipped: list[dict] = []
        for rec in live:
            try:
                verified.append((rec.name,
                                 verify_live_authorization(conn, repo, rec.name,
                                                           ALLOWED_SIGNERS_PATH)))
            except LiveAuthorizationError as exc:
                skipped.append({"strategy": rec.name, "reason": str(exc)})
        if not verified:
            emit(ok({"strategies": [], "skipped": skipped, "note": "no authorized live strategies"}))
            return
        broker = _alpaca_live_broker(verified[0][1])
        provider = _select_provider(False, snapshot)
        # ingest fills, then reconcile the account before trading
        ingest_activities(conn, _broker_account_activities(broker, fill_cursor(conn)))
        cycle = live_reconcile.next_cycle(conn)
        recon = live_reconcile.reconcile(conn, _broker_net_positions(broker), cycle,
                                         tolerance=tolerance, grace_cycles=grace_cycles)
        recon_payload = {"cycle": cycle, "clean": recon.clean, "halt": recon.halt,
                         "mismatches": recon.mismatches}
        if recon.halt:
            global_halt.engage(conn, reason=f"reconcile drift {recon.mismatches}", actor="system")
            emit({"ok": False, "reconcile": recon_payload, "skipped": skipped})
            raise typer.Exit(1)
        if not recon.clean:
            emit(ok({"reconcile": recon_payload, "skipped": skipped,
                     "note": "reconcile pending; deferring trades this cycle", "strategies": []}))
            return
        results = []
        for name, authorization in verified:
            results.append(_run_strategy_tick(
                conn, name, authorization, broker, provider, max_drawdown,
                cancel=lambda n=name: _scoped_cancel(conn, broker, n)))
    emit(ok({"reconcile": recon_payload, "skipped": skipped, "strategies": results}))


def _scoped_cancel(conn, broker, strategy: str) -> None:
    """Cancel only THIS strategy's open orders (never a sibling's)."""
    for oid in owned_open_order_ids(conn, broker, strategy):
        broker.cancel_order(oid)
```
Add imports: `from algua.contracts.lifecycle import Stage`, `from algua.execution import live_reconcile`, `from algua.execution.live_ledger import fill_cursor, ingest_activities, owned_open_order_ids`.

- [ ] **Step 4: Run** `uv run pytest tests/test_cli_live.py -q` → PASS (the run-all tests + the unchanged trade-tick tests).

- [ ] **Step 5: FULL gate + commit.** `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports` (all green; contracts 0 broken).
```bash
git add algua/cli/live_cmd.py tests/test_cli_live.py
git commit -m "feat(live-loop): live run-all sequenced portfolio loop (verify -> ingest -> reconcile -> scoped-cancel tick)"
```

---

## Self-review notes

- **Spec coverage (Slice B in §8 + §3/§5):** the sequenced `live run-all` loop — re-verify each, ingest, reconcile, sequenced tick (Task 4); classified account reconcile with tolerance + grace window + persistent-unexplained → global halt (Task 1); scoped owned-order cancel, account-wide reserved for halt (Tasks 2-3). Single-strategy sizing carried over (no Slice-C sizing change here). ✓
- **Codex CRITICAL coverage:** scoped cancel (Tasks 2-3) closes "account-wide cancel cancels siblings"; the reconcile accounting (Task 1) is the classified equation that distinguishes pending from unexplained (no constant false-halts); in-flight orders don't move broker POSITIONS so `Σ settled fills == positions` holds once ingested — a transient gap is the grace window's job. (Buying-power reservation + per-strategy sizing + liquidation workflow remain Slice C.)
- **Type consistency:** `ReconcileResult(clean, halt, mismatches)` (Task 1) consumed in `run-all` (Task 4); `owned_open_order_ids(conn, broker, strategy)` (Task 2) used by `_scoped_cancel` (Task 4); `TickHooks.cancel` (Task 3) supplied by `_run_strategy_tick`'s `cancel` param (Task 4); `list_open_orders`/`cancel_order` (Task 2) used by `owned_open_order_ids`/`_scoped_cancel`.
- **Shadow-mode / guard:** the Slice-A ≤1-live guard still holds (go-live refuses a 2nd live strategy), so `run-all` ticks ≤1 strategy until Slice C; the loop + reconcile + scoped cancel are exercised now and ready for many. No sizing/flatten semantics changed.
- **No placeholders:** Task 4 Step 3a references "the existing try/except + snapshot block" — the implementer MOVES that exact block from `trade_tick` (it already exists in the file) into the helper, changing only `emit(...)`→`return {...}` on the success path; this is an extraction, not new code. The breach/halt paths keep emitting + `raise typer.Exit(1)`.
