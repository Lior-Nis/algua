# Paper Status / Health + Per-Tick Snapshot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist a per-tick account/positions snapshot and grow `paper show <name>` into a consolidated operability view (stage, kill-switch, drawdown, last-tick snapshot, recent orders, health).

**Architecture:** A new append-only `tick_snapshots` table (schema v5) + three `order_state` helpers; `TickResult` gains an `equity` field that `trade-tick` persists on the full-tick path; `paper show` becomes a pure read of persisted state assembling the consolidated JSON. No broker call in `show`; no change to the tick's trading logic.

**Tech Stack:** Python 3.12, sqlite3, Typer, pytest, ruff, mypy, import-linter. Spec: `docs/superpowers/specs/2026-06-05-paper-status-health-design.md`.

---

## File structure

| File | Responsibility |
|---|---|
| `algua/registry/db.py` (modify) | `tick_snapshots` table + index; `SCHEMA_VERSION` 4 → 5. |
| `algua/execution/order_state.py` (modify) | `record_tick_snapshot` / `latest_tick_snapshot` / `recent_orders`. |
| `algua/live/live_loop.py` (modify) | `TickResult.equity` field; full-tick return sets it from `snap.equity`. |
| `algua/cli/paper_cmd.py` (modify) | `trade-tick` persists a snapshot on the full-tick path; `show` emits the consolidated view. |

---

### Task 1: `tick_snapshots` table (schema v5) + store helpers

**Files:** Modify `algua/registry/db.py`, `algua/execution/order_state.py`; Test `tests/test_paper_db.py`, `tests/test_order_state.py`.

Context: `tests/test_paper_db.py` imports `SCHEMA_VERSION` from `algua.registry.db` and asserts `user_version == SCHEMA_VERSION` (a constant, so the bump needs no number edit) and has a `_tables(conn)` helper. `tests/test_order_state.py` has a `_conn(tmp_path)` fixture (connect + migrate) and imports helpers from `algua.execution.order_state`. `order_state.py` imports `re`, `sqlite3`, `from datetime import UTC, datetime`, `pandas` — it does NOT import `json` yet.

- [ ] **Step 1: Add failing tests** — append to `tests/test_paper_db.py`:

```python
def test_migrate_creates_tick_snapshots_table(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    assert "tick_snapshots" in _tables(conn)
```

And append to `tests/test_order_state.py` (add `record_tick_snapshot`, `latest_tick_snapshot`, `recent_orders` to the existing `from algua.execution.order_state import (...)` block):

```python
def test_tick_snapshot_roundtrip_latest_wins(tmp_path):
    conn = _conn(tmp_path)
    assert latest_tick_snapshot(conn, "s") is None
    record_tick_snapshot(conn, "s", tick_ts="2023-06-01T00:00:00+00:00",
                         decision_ts="2023-05-31T00:00:00+00:00", equity=100.0, peak_equity=100.0,
                         positions={"AAA": 10.0}, n_submitted=1, reconcile_ok=True)
    record_tick_snapshot(conn, "s", tick_ts="2023-06-02T00:00:00+00:00",
                         decision_ts="2023-06-01T00:00:00+00:00", equity=120.0, peak_equity=120.0,
                         positions={"AAA": 12.0}, n_submitted=0, reconcile_ok=False)
    latest = latest_tick_snapshot(conn, "s")
    assert latest["equity"] == 120.0 and latest["positions"] == {"AAA": 12.0}
    assert latest["reconcile_ok"] is False and latest["n_submitted"] == 0


def test_recent_orders_newest_first_and_limit(tmp_path):
    conn = _conn(tmp_path)
    for i in range(3):
        record_submitted_order(conn, "s", f"SYM{i}", "buy", 1.0, "2023-06-01T00:00:00+00:00",
                               f"o-{i}")
    rows = recent_orders(conn, "s", limit=2)
    assert [r["broker_order_id"] for r in rows] == ["o-2", "o-1"]  # newest first, limited
    assert rows[0]["symbol"] == "SYM2" and rows[0]["side"] == "buy"
```

- [ ] **Step 2: Run** `uv run pytest tests/test_paper_db.py tests/test_order_state.py -q` → FAIL (table missing; helpers undefined).

- [ ] **Step 3: Add the table + version bump** — in `algua/registry/db.py`, change `SCHEMA_VERSION = 4` to `SCHEMA_VERSION = 5`, and append to the `_SCHEMA` string (before the closing `"""`):

```sql
CREATE TABLE IF NOT EXISTS tick_snapshots (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy     TEXT NOT NULL,
    tick_ts      TEXT NOT NULL,
    decision_ts  TEXT,
    equity       REAL NOT NULL,
    peak_equity  REAL,
    positions    TEXT NOT NULL,
    n_submitted  INTEGER NOT NULL,
    reconcile_ok INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_tick_snapshots_strategy_ts ON tick_snapshots(strategy, tick_ts);
```

- [ ] **Step 4: Add the helpers** — in `algua/execution/order_state.py`, add `import json` to the imports, and append these functions:

```python
def record_tick_snapshot(
    conn: sqlite3.Connection, strategy: str, *, tick_ts: str, decision_ts: str | None,
    equity: float, peak_equity: float | None, positions: dict[str, float], n_submitted: int,
    reconcile_ok: bool,
) -> None:
    """Append one completed-tick snapshot (equity + positions) for a strategy — the per-tick
    operability/equity-curve record read by `paper show` (#snapshot)."""
    conn.execute(
        "INSERT INTO tick_snapshots(strategy, tick_ts, decision_ts, equity, peak_equity, "
        "positions, n_submitted, reconcile_ok) VALUES (?,?,?,?,?,?,?,?)",
        (strategy, tick_ts, decision_ts, equity, peak_equity, json.dumps(positions),
         n_submitted, 1 if reconcile_ok else 0),
    )
    conn.commit()


def latest_tick_snapshot(conn: sqlite3.Connection, strategy: str) -> dict | None:
    """The most recent tick snapshot for a strategy (positions parsed back to a dict), or None."""
    row = conn.execute(
        "SELECT tick_ts, decision_ts, equity, peak_equity, positions, n_submitted, reconcile_ok "
        "FROM tick_snapshots WHERE strategy = ? ORDER BY id DESC LIMIT 1", (strategy,)
    ).fetchone()
    if row is None:
        return None
    return {
        "tick_ts": row["tick_ts"], "decision_ts": row["decision_ts"], "equity": row["equity"],
        "peak_equity": row["peak_equity"], "positions": json.loads(row["positions"]),
        "n_submitted": row["n_submitted"], "reconcile_ok": bool(row["reconcile_ok"]),
    }


def recent_orders(conn: sqlite3.Connection, strategy: str, limit: int = 10) -> list[dict]:
    """The most recent paper_orders rows for a strategy, newest first."""
    rows = conn.execute(
        "SELECT symbol, side, status, broker_order_id, submitted_ts FROM paper_orders "
        "WHERE strategy = ? ORDER BY id DESC LIMIT ?", (strategy, limit),
    ).fetchall()
    return [dict(r) for r in rows]
```

- [ ] **Step 5: Run** `uv run pytest tests/test_paper_db.py tests/test_order_state.py tests/test_registry_db.py -q` → PASS.

- [ ] **Step 6: Gate + commit** — `uv run ruff check . && uv run mypy algua && uv run pytest tests/test_paper_db.py tests/test_order_state.py tests/test_registry_db.py -q`

```bash
git add algua/registry/db.py algua/execution/order_state.py tests/test_paper_db.py tests/test_order_state.py
git commit -m "feat(registry): tick_snapshots table + snapshot/recent-orders helpers (schema v5)"
```

---

### Task 2: `TickResult.equity` + `trade-tick` persists a snapshot

**Files:** Modify `algua/live/live_loop.py`, `algua/cli/paper_cmd.py`; Test `tests/test_live_loop.py`, `tests/test_cli_paper.py`.

Context: `TickResult` (in `live_loop.py`) is a `@dataclass` with fields `decision_ts, target_weights, positions_before, submitted, peak_equity=None, reconcile_ok=True, realized_gross=0.0`. `run_tick` takes `snap = broker.snapshot(strategy.universe)` and ends with `return TickResult(decision_ts=t, target_weights=..., positions_before=positions_before, submitted=submitted, peak_equity=peak, reconcile_ok=reconcile_ok, realized_gross=realized_gross)`. The early returns (`bars.empty`, warm-up) use `TickResult(None, {}, _positions(broker), [])` — no snapshot taken, `peak_equity` stays `None`. In `paper_cmd.py`, `trade-tick`'s success path is `if result.peak_equity is not None: update_peak_equity(conn, name, result.peak_equity)` then an `audit_append`. `paper_cmd.py` does NOT import `datetime`.

- [ ] **Step 1: Add failing tests** — append to `tests/test_live_loop.py` (uses `_FakeBroker`, `_FakeProvider`, `_strategy`, `_bars`, `DATES`):

```python
def test_run_tick_result_carries_equity():
    broker = _FakeBroker(equity=123_456.0)  # _FakeBroker.snapshot() returns TickSnapshot(equity=...)
    bars = _bars({"AAA": [100.0, 100.0, 100.0]})
    result = run_tick(_strategy({"AAA": 1.0}), broker, _FakeProvider(bars), DATES[0], DATES[-1])
    assert result.equity == 123_456.0  # the tick's snapshot equity is surfaced on the result
```

And append to `tests/test_cli_paper.py` (uses `runner`, `app`, `_to_paper`, `json`, `_AccountBroker`/`_seed_peak` if present, and monkeypatching of `algua.cli.paper_cmd.run_tick`):

```python
def test_trade_tick_persists_snapshot(monkeypatch):
    from contextlib import closing

    from algua.config.settings import get_settings
    from algua.execution.order_state import latest_tick_snapshot
    from algua.live.live_loop import TickResult
    from algua.registry.db import connect, migrate

    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    _to_paper()
    fake = TickResult(decision_ts=datetime(2023, 6, 1, tzinfo=UTC), target_weights={"AAA": 1.0},
                      positions_before={"AAA": 5.0}, submitted=[{"symbol": "AAA"}],
                      equity=99000.0, peak_equity=99000.0)
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings", lambda: object())
    monkeypatch.setattr("algua.cli.paper_cmd._select_provider", lambda demo, snapshot: object())
    monkeypatch.setattr("algua.cli.paper_cmd.run_tick", lambda *a, **k: fake)
    result = runner.invoke(app, ["paper", "trade-tick", "cross_sectional_momentum",
                                 "--snapshot", "snap1"])
    assert result.exit_code == 0, result.stdout
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        snap = latest_tick_snapshot(conn, "cross_sectional_momentum")
    assert snap is not None and snap["equity"] == 99000.0
    assert snap["positions"] == {"AAA": 5.0} and snap["n_submitted"] == 1
```

(Note: `tests/test_cli_paper.py` already imports `json`, `UTC`/`datetime`, and `TickResult` at the top; the local `from algua.live.live_loop import TickResult` above is belt-and-suspenders — remove it if it duplicates a top-level import and trips ruff.)

- [ ] **Step 2: Run** `uv run pytest tests/test_live_loop.py tests/test_cli_paper.py -q` → FAIL (`TickResult.equity` undefined; no snapshot persisted).

- [ ] **Step 3: Add the `equity` field** — in `algua/live/live_loop.py` `TickResult`, insert `equity` after `submitted`:

```python
@dataclass
class TickResult:
    decision_ts: datetime | None
    target_weights: dict[str, float]
    positions_before: dict[str, float]
    submitted: list[dict[str, Any]]
    equity: float = 0.0
    peak_equity: float | None = None
    reconcile_ok: bool = True
    realized_gross: float = 0.0
```

- [ ] **Step 4: Set it on the full-tick return** — in `run_tick`, add `equity=snap.equity,` to the final `return TickResult(...)` (the one with `peak_equity=peak`):

```python
    return TickResult(
        decision_ts=t,
        target_weights={s: float(w) for s, w in weights.items()},
        positions_before=positions_before,
        submitted=submitted,
        equity=snap.equity,
        peak_equity=peak,
        reconcile_ok=reconcile_ok,
        realized_gross=realized_gross,
    )
```

- [ ] **Step 5: Persist the snapshot in `trade-tick`** — in `algua/cli/paper_cmd.py`: add `from datetime import UTC, datetime` to the imports; add `record_tick_snapshot` to the `from algua.execution.order_state import (...)` block; then change the success block:

```python
        if result.peak_equity is not None:
            update_peak_equity(conn, name, result.peak_equity)
```
to:
```python
        if result.peak_equity is not None:
            update_peak_equity(conn, name, result.peak_equity)
            record_tick_snapshot(
                conn, name, tick_ts=datetime.now(UTC).isoformat(),
                decision_ts=result.decision_ts.isoformat() if result.decision_ts else None,
                equity=result.equity, peak_equity=result.peak_equity,
                positions=result.positions_before, n_submitted=len(result.submitted),
                reconcile_ok=result.reconcile_ok,
            )
```

- [ ] **Step 6: Run** `uv run pytest tests/test_live_loop.py tests/test_cli_paper.py -q` → PASS.

- [ ] **Step 7: Gate + commit** — `uv run ruff check . && uv run mypy algua && uv run pytest tests/test_live_loop.py tests/test_cli_paper.py -q`

```bash
git add algua/live/live_loop.py algua/cli/paper_cmd.py tests/test_live_loop.py tests/test_cli_paper.py
git commit -m "feat(live): TickResult.equity + trade-tick persists a per-tick snapshot"
```

---

### Task 3: enriched `paper show` (consolidated view + health)

**Files:** Modify `algua/cli/paper_cmd.py`; Test `tests/test_cli_paper.py`.

Context: current `show` reads `count_orders`, `derive_positions`, `kill_switch.get` and emits `{strategy, n_orders, positions, kill_switch}`. `paper_cmd.py` already imports `SqliteStrategyRepository` (from `algua.registry.store`), `get_peak_equity` (from `order_state`), `kill_switch`, `ok`, `registry_conn`. Add `latest_tick_snapshot` and `recent_orders` to the `order_state` import block. `SqliteStrategyRepository(conn).get(name)` returns a record with `.stage` (a `Stage` enum, use `.value`) and raises `LookupError` for an unknown name — `show` is decorated `@json_errors(ValueError, LookupError)`, so that already renders `{ok:false}`.

- [ ] **Step 1: Add failing tests** — append to `tests/test_cli_paper.py`:

```python
def _seed_snapshot(name, *, equity, peak, reconcile_ok=True, positions=None):
    from contextlib import closing

    from algua.config.settings import get_settings
    from algua.execution.order_state import record_tick_snapshot, update_peak_equity
    from algua.registry.db import connect, migrate
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        update_peak_equity(conn, name, peak)
        record_tick_snapshot(conn, name, tick_ts="2023-06-01T00:00:00+00:00",
                             decision_ts="2023-05-31T00:00:00+00:00", equity=equity,
                             peak_equity=peak, positions=positions or {}, n_submitted=0,
                             reconcile_ok=reconcile_ok)


def test_show_consolidated_view():
    _to_paper()
    _seed_snapshot("cross_sectional_momentum", equity=90.0, peak=100.0, positions={"AAA": 3.0})
    payload = json.loads(runner.invoke(app, ["paper", "show", "cross_sectional_momentum"]).stdout)
    assert payload["stage"] == "paper"
    assert payload["drawdown"]["peak_equity"] == 100.0 and payload["drawdown"]["last_equity"] == 90.0
    assert abs(payload["drawdown"]["drawdown"] - 0.10) < 1e-9
    assert payload["last_tick"]["positions"] == {"AAA": 3.0}
    assert payload["health"] == "ok"
    assert "recent_orders" in payload


def test_show_health_halted():
    _to_paper()
    runner.invoke(app, ["paper", "kill", "cross_sectional_momentum", "--reason", "x"])
    payload = json.loads(runner.invoke(app, ["paper", "show", "cross_sectional_momentum"]).stdout)
    assert payload["health"] == "halted"


def test_show_health_drift():
    _to_paper()
    _seed_snapshot("cross_sectional_momentum", equity=90.0, peak=100.0, reconcile_ok=False)
    payload = json.loads(runner.invoke(app, ["paper", "show", "cross_sectional_momentum"]).stdout)
    assert payload["health"] == "drift"


def test_show_health_idle_no_ticks():
    _to_paper()
    payload = json.loads(runner.invoke(app, ["paper", "show", "cross_sectional_momentum"]).stdout)
    assert payload["health"] == "idle" and payload["last_tick"] is None


def test_show_unknown_strategy_errors():
    result = runner.invoke(app, ["paper", "show", "no_such_strategy"])
    assert result.exit_code == 1 and json.loads(result.stdout)["ok"] is False
```

- [ ] **Step 2: Run** `uv run pytest tests/test_cli_paper.py -q -k show` → FAIL.

- [ ] **Step 3: Rewrite `show`** — replace the whole `show` command body in `algua/cli/paper_cmd.py` (add `latest_tick_snapshot`, `recent_orders` to the `order_state` import block first):

```python
@paper_app.command("show")
@json_errors(ValueError, LookupError)
def show(name: str) -> None:
    """Consolidated per-strategy operability view — stage, kill-switch, drawdown, last tick,
    recent orders, and a health rollup. A pure read of persisted state (no broker call)."""
    with registry_conn() as conn:
        rec = SqliteStrategyRepository(conn).get(name)  # unknown name -> LookupError -> {ok:false}
        n_orders = count_orders(conn, name)
        positions = derive_positions(conn, name)
        ks = kill_switch.get(conn, name)
        peak = get_peak_equity(conn, name)
        last = latest_tick_snapshot(conn, name)
        orders = recent_orders(conn, name, 10)
    tripped = ks is not None
    last_equity = last["equity"] if last else None
    drawdown = (
        1.0 - last_equity / peak
        if last_equity is not None and peak is not None and peak > 0 else None
    )
    if tripped:
        health = "halted"
    elif last is not None and not last["reconcile_ok"]:
        health = "drift"
    elif last is None:
        health = "idle"
    else:
        health = "ok"
    emit(ok({
        "strategy": name,
        "stage": rec.stage.value,
        "kill_switch": {"tripped": tripped, "reason": ks["reason"] if ks else None},
        "drawdown": {"peak_equity": peak, "last_equity": last_equity, "drawdown": drawdown},
        "last_tick": last,
        "positions": positions,
        "n_orders": n_orders,
        "recent_orders": orders,
        "health": health,
    }))
```

- [ ] **Step 4: Run** `uv run pytest tests/test_cli_paper.py -q` → PASS (existing `show` assertions still hold — `strategy`/`n_orders`/`positions`/`kill_switch` keys are unchanged).

- [ ] **Step 5: Full gate + commit** — `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports` (all pass; `lint-imports` stays `10 kept, 0 broken`).

```bash
git add algua/cli/paper_cmd.py tests/test_cli_paper.py
git commit -m "feat(cli): consolidated paper show (stage/drawdown/last-tick/recent-orders/health)"
```

---

## Self-review notes

- **Spec coverage:** `tick_snapshots` table + helpers (§3 → Task 1); `TickResult.equity` + `trade-tick` persistence on the full-tick path only (§3 → Task 2; gated on `peak_equity is not None`, which is set only when `snap` was taken — so no-bar/warm-up ticks don't snapshot); enriched `show` with stage/drawdown/last_tick/recent_orders (§4 → Task 3); health rollup branches (§5 → Task 3 tests); schema v5 + table-present test (§3 → Task 1). The version-assertion test needs no edit (it asserts against the `SCHEMA_VERSION` constant). Live acceptance is documented in the spec (§6), not code.
- **Type consistency:** `record_tick_snapshot(conn, strategy, *, tick_ts, decision_ts, equity, peak_equity, positions, n_submitted, reconcile_ok)` and `latest_tick_snapshot(conn, strategy) -> dict|None` / `recent_orders(conn, strategy, limit)` are used identically in Task 1 (definition + tests), Task 2 (`trade-tick`), and Task 3 (`show`). `TickResult.equity` (float, default 0.0) added in Task 2 is read in Task 2's `trade-tick` and produced by `run_tick`. `rec.stage.value` matches the `Stage` enum used elsewhere in `paper_cmd.py`.
- **No placeholders:** every code step is complete. The one conditional instruction (inspect `_FakeBroker.snapshot` in Task 2 Step 1) is because the broker double's exact snapshot-equity value lives in `tests/test_live_loop.py`; the assertion just ties `result.equity` to that same snapshot equity.
