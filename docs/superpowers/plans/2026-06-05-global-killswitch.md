# Global ("Halt-All") Kill-Switch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An account-level panic button — `paper halt-all` halts every paper strategy and flattens the whole Alpaca account; `paper resume-all` clears it and re-bases all drawdown peaks.

**Architecture:** A single-row `global_halt` table (schema v6) behind a `global_halt` module (mirrors `kill_switch`); the account-wide `close_all_positions()` broker primitive; `halt-all`/`resume-all` CLI commands (trip-before-close fail-safe); and a global-halt check added to the shared `_load_gated_strategy` gate + the `trade-tick` pre-submit hook, plus a `paper show` reflection.

**Tech Stack:** Python 3.12, sqlite3, requests, Typer, pytest, ruff, mypy, import-linter. Spec: `docs/superpowers/specs/2026-06-05-global-killswitch-design.md`.

**Naming note:** the module's setter is `engage` (not `set` — that shadows the builtin and trips ruff); boolean is `is_engaged`. This supersedes the spec's `set`/`is_set`.

---

## File structure

| File | Responsibility |
|---|---|
| `algua/registry/db.py` (modify) | `global_halt` table; `SCHEMA_VERSION` 5 → 6. |
| `algua/risk/global_halt.py` (new) | `engage` / `is_engaged` / `clear` / `get`. |
| `algua/execution/order_state.py` (modify) | `clear_all_peaks` (wipe every strategy's peak). |
| `algua/execution/alpaca_broker.py` (modify) | `close_all_positions()` (account-wide liquidate + cancel). |
| `algua/cli/paper_cmd.py` (modify) | `paper halt-all` + `paper resume-all`; global-halt gate; `show` reflection. |

---

### Task 1: `global_halt` table (schema v6) + module + `clear_all_peaks`

**Files:** Modify `algua/registry/db.py`, `algua/execution/order_state.py`; Create `algua/risk/global_halt.py`; Test `tests/test_paper_db.py`, `tests/test_global_halt.py` (new), `tests/test_order_state.py`.

Context: `algua/registry/db.py` has `SCHEMA_VERSION = 5` and a `_SCHEMA` string of `CREATE TABLE IF NOT EXISTS ...`. `tests/test_paper_db.py` has `_tables(conn)` + `connect`/`migrate`. `tests/test_order_state.py` has `_conn(tmp_path)` and imports order_state helpers (incl. `clear_peak_equity`, `update_peak_equity`, `get_peak_equity`). `algua/risk/kill_switch.py` is the sibling module to mirror.

- [ ] **Step 1: Add failing tests.**

Create `tests/test_global_halt.py`:
```python
from algua.registry.db import connect, migrate
from algua.risk import global_halt


def _conn(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    return conn


def test_engage_is_engaged_clear(tmp_path):
    conn = _conn(tmp_path)
    assert global_halt.is_engaged(conn) is False
    global_halt.engage(conn, reason="panic", actor="human")
    assert global_halt.is_engaged(conn) is True
    info = global_halt.get(conn)
    assert info["reason"] == "panic" and info["actor"] == "human"
    assert global_halt.clear(conn) is True
    assert global_halt.is_engaged(conn) is False
    assert global_halt.clear(conn) is False  # already clear -> no row removed


def test_engage_is_single_row(tmp_path):
    conn = _conn(tmp_path)
    global_halt.engage(conn, reason="a", actor="agent")
    global_halt.engage(conn, reason="b", actor="human")  # upsert, not a second row
    assert conn.execute("SELECT COUNT(*) FROM global_halt").fetchone()[0] == 1
    assert global_halt.get(conn)["reason"] == "b"
```

Append to `tests/test_paper_db.py`:
```python
def test_migrate_creates_global_halt_table(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    assert "global_halt" in _tables(conn)
```

Append to `tests/test_order_state.py` (add `clear_all_peaks` to the `from algua.execution.order_state import (...)` block, keeping it sorted):
```python
def test_clear_all_peaks_wipes_every_strategy(tmp_path):
    conn = _conn(tmp_path)
    update_peak_equity(conn, "a", 100.0)
    update_peak_equity(conn, "b", 200.0)
    clear_all_peaks(conn)
    assert get_peak_equity(conn, "a") is None and get_peak_equity(conn, "b") is None
```

- [ ] **Step 2: Run** `uv run pytest tests/test_global_halt.py tests/test_paper_db.py tests/test_order_state.py -q` → FAIL.

- [ ] **Step 3: Add the table + version bump** — in `algua/registry/db.py`, change `SCHEMA_VERSION = 5` to `SCHEMA_VERSION = 6`, and append to `_SCHEMA` (before the closing `"""`):
```sql
CREATE TABLE IF NOT EXISTS global_halt (
    id         INTEGER PRIMARY KEY CHECK (id = 1),
    reason     TEXT,
    actor      TEXT NOT NULL,
    created_at TEXT NOT NULL
);
```

- [ ] **Step 4: Create `algua/risk/global_halt.py`:**
```python
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime


def engage(conn: sqlite3.Connection, *, reason: str, actor: str) -> None:
    """Engage the account-wide halt (single row id=1). Idempotent: re-engaging updates reason/actor."""
    conn.execute(
        "INSERT INTO global_halt(id, reason, actor, created_at) VALUES (1,?,?,?) "
        "ON CONFLICT(id) DO UPDATE SET "
        "reason=excluded.reason, actor=excluded.actor, created_at=excluded.created_at",
        (reason, actor, datetime.now(UTC).isoformat()),
    )
    conn.commit()


def is_engaged(conn: sqlite3.Connection) -> bool:
    return conn.execute("SELECT 1 FROM global_halt WHERE id = 1").fetchone() is not None


def clear(conn: sqlite3.Connection) -> bool:
    """Clear the halt. Returns whether a row was actually removed."""
    cur = conn.execute("DELETE FROM global_halt")
    conn.commit()
    return cur.rowcount > 0


def get(conn: sqlite3.Connection) -> dict[str, str] | None:
    row = conn.execute(
        "SELECT reason, actor, created_at FROM global_halt WHERE id = 1"
    ).fetchone()
    if row is None:
        return None
    return {"reason": row["reason"], "actor": row["actor"], "created_at": row["created_at"]}
```

- [ ] **Step 5: Add `clear_all_peaks`** — in `algua/execution/order_state.py`, after `clear_peak_equity`:
```python
def clear_all_peaks(conn: sqlite3.Connection) -> None:
    """Wipe every strategy's persisted peak — used by the global resume-all after the whole account
    is flattened, so each strategy re-bases its drawdown high-water mark on its next tick (#27)."""
    conn.execute("DELETE FROM strategy_peaks")
    conn.commit()
```

- [ ] **Step 6: Run** `uv run pytest tests/test_global_halt.py tests/test_paper_db.py tests/test_order_state.py tests/test_registry_db.py -q` → PASS.

- [ ] **Step 7: Gate + commit** — `uv run ruff check . && uv run mypy algua && uv run pytest tests/test_global_halt.py tests/test_paper_db.py tests/test_order_state.py tests/test_registry_db.py -q`
```bash
git add algua/registry/db.py algua/risk/global_halt.py algua/execution/order_state.py tests/test_global_halt.py tests/test_paper_db.py tests/test_order_state.py
git commit -m "feat(risk): global_halt table + module + clear_all_peaks (schema v6)"
```

---

### Task 2: `close_all_positions()` on the broker

**Files:** Modify `algua/execution/alpaca_broker.py`; Test `tests/test_alpaca_broker.py`.

Context: `alpaca_broker.py` has `close_positions(symbols)` (per-symbol), `_multistatus_failures(results)` (a non-dict item or non-2xx `status` counts as failure), `_read(resp, path, ok=(...))`, `_delete(path)`. `tests/test_alpaca_broker.py` has `_FakeRequestsWithDelete(delete_resp)` (records `.deleted` URLs, returns `delete_resp` for any DELETE), `_FakeResp(status, payload, text)`, `_broker()`, and `import pytest`.

- [ ] **Step 1: Add failing tests** — append to `tests/test_alpaca_broker.py`:
```python
def test_close_all_positions_ok(monkeypatch):
    fake = _FakeRequestsWithDelete(_FakeResp(207, [{"symbol": "AAA", "status": 200}]))
    monkeypatch.setattr(ab, "requests", fake)
    _broker().close_all_positions()
    assert fake.deleted == ["https://paper-api.alpaca.markets/v2/positions?cancel_orders=true"]


def test_close_all_positions_empty_is_noop(monkeypatch):
    monkeypatch.setattr(ab, "requests", _FakeRequestsWithDelete(_FakeResp(207, [])))
    _broker().close_all_positions()  # no positions -> no error


def test_close_all_positions_partial_failure_raises(monkeypatch):
    fake = _FakeRequestsWithDelete(_FakeResp(207, [{"symbol": "AAA", "status": 200},
                                                   {"symbol": "BBB", "status": 500}]))
    monkeypatch.setattr(ab, "requests", fake)
    with pytest.raises(BrokerError):
        _broker().close_all_positions()
```
(Confirm `ab` is the module alias and `BrokerError` is imported at the top of the test file; if not, mirror the existing `cancel_open_orders` tests' imports.)

- [ ] **Step 2: Run** `uv run pytest tests/test_alpaca_broker.py -q` → FAIL.

- [ ] **Step 3: Implement** — in `algua/execution/alpaca_broker.py`, add after `close_positions`:
```python
    def close_all_positions(self) -> None:
        """Liquidate the ENTIRE account: DELETE /v2/positions?cancel_orders=true — Alpaca cancels
        all open orders then market-closes every position, returning a 207 multi-status; raise if
        any per-position close failed. Account-wide — used ONLY by the global halt (per-strategy
        flatten uses close_positions(universe)). Empty account -> empty list -> no-op."""
        results = self._read(
            self._delete("/v2/positions?cancel_orders=true"), "/v2/positions", ok=(200, 207)
        )
        if isinstance(results, list) and _multistatus_failures(results):
            raise BrokerError(f"alpaca failed to close some positions: {results}")
```

- [ ] **Step 4: Run** `uv run pytest tests/test_alpaca_broker.py -q` → PASS.

- [ ] **Step 5: Gate + commit** — `uv run ruff check . && uv run mypy algua && uv run pytest tests/test_alpaca_broker.py -q`
```bash
git add algua/execution/alpaca_broker.py tests/test_alpaca_broker.py
git commit -m "feat(broker): account-wide close_all_positions for the global halt"
```

---

### Task 3: `paper halt-all` + `paper resume-all` commands

**Files:** Modify `algua/cli/paper_cmd.py`; Test `tests/test_cli_paper.py`.

Context: `paper_cmd.py` imports `from algua.risk import kill_switch`, `from algua.cli._common import ok, registry_conn, utc`, `audit_append`, `emit`, `json_errors`, `typer`, `_alpaca_broker_from_settings`, `BrokerError`. The `flatten` command is the trip-before-close pattern to mirror. `tests/test_cli_paper.py` has `runner`, `app`, `json`, `_to_paper`, and a `_FlattenBroker` (exposes `cancel_open_orders`/`close_positions`). The CLI emits via `emit(ok({...}))` (adds `ok:true`) and plain `emit({"ok": False, ...})` for failures.

- [ ] **Step 1: Add failing tests** — append to `tests/test_cli_paper.py`:
```python
class _HaltBroker:
    def __init__(self, fail=False):
        self.fail = fail
        self.closed_all = False

    def close_all_positions(self):
        if self.fail:
            raise BrokerError("alpaca failed to close some positions: [...]")
        self.closed_all = True


def test_halt_all_engages_and_flattens(monkeypatch):
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    broker = _HaltBroker()
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings", lambda: broker)
    result = runner.invoke(app, ["paper", "halt-all", "--reason", "panic"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["global_halt"] == "set" and payload["liquidation_submitted"] is True
    assert broker.closed_all is True


def test_halt_all_close_failure_stays_engaged(monkeypatch):
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings",
                        lambda: _HaltBroker(fail=True))
    result = runner.invoke(app, ["paper", "halt-all", "--reason", "panic"])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False and payload["global_halt"] == "set"
    assert payload["liquidation_submitted"] is False
    # still engaged (fail-safe)
    from contextlib import closing

    from algua.config.settings import get_settings
    from algua.registry.db import connect, migrate
    from algua.risk import global_halt
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        assert global_halt.is_engaged(conn) is True


def test_resume_all_clears_and_wipes_peaks_but_keeps_strategy_switch(monkeypatch):
    from contextlib import closing

    from algua.config.settings import get_settings
    from algua.execution.order_state import get_peak_equity, update_peak_equity
    from algua.registry.db import connect, migrate
    from algua.risk import global_halt, kill_switch

    _to_paper()
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        global_halt.engage(conn, reason="x", actor="human")
        update_peak_equity(conn, "cross_sectional_momentum", 100.0)
        kill_switch.trip(conn, "cross_sectional_momentum", reason="indiv", actor="human")
    result = runner.invoke(app, ["paper", "resume-all"])
    assert result.exit_code == 0, result.stdout
    assert json.loads(result.stdout)["global_halt"] == "reset"
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        assert global_halt.is_engaged(conn) is False
        assert get_peak_equity(conn, "cross_sectional_momentum") is None  # peaks wiped
        assert kill_switch.is_tripped(conn, "cross_sectional_momentum") is True  # left untouched
```

- [ ] **Step 2: Run** `uv run pytest tests/test_cli_paper.py -q -k "halt_all or resume_all"` → FAIL.

- [ ] **Step 3: Implement** — in `algua/cli/paper_cmd.py`: add `global_halt` to the risk import (`from algua.risk import global_halt, kill_switch`) and `clear_all_peaks` to the `order_state` import block (keep sorted); append the two commands:
```python
@paper_app.command("halt-all")
@json_errors(ValueError, LookupError, BrokerError)
def halt_all(
    reason: str = typer.Option(..., "--reason", help="why the whole account is being halted"),
    actor: str = typer.Option("agent", "--actor", help="human | agent"),
) -> None:
    """ACCOUNT-WIDE emergency: engage the global halt and flatten the ENTIRE Alpaca account."""
    with registry_conn() as conn:
        broker = _alpaca_broker_from_settings()
        # Engage first (fail-safe): all trading is stopped even if the close call then fails.
        global_halt.engage(conn, reason=reason, actor=actor)
        audit_append(conn, actor=actor, action="halt_all", reason=reason, strategy=None)
        try:
            broker.close_all_positions()
        except BrokerError as exc:
            audit_append(conn, actor="system", action="flatten_failed", reason=str(exc),
                         strategy=None)
            emit({"ok": False, "global_halt": "set", "liquidation_submitted": False,
                  "error": str(exc)})
            raise typer.Exit(1) from exc
    emit(ok({"global_halt": "set", "liquidation_submitted": True}))


@paper_app.command("resume-all")
@json_errors(ValueError)
def resume_all(
    actor: str = typer.Option("human", "--actor", help="human | agent"),
) -> None:
    """Clear the global halt and re-base every strategy's drawdown peak (the account was flattened
    to cash). Per-strategy kill-switches are left untouched."""
    with registry_conn() as conn:
        was_set = global_halt.is_engaged(conn)
        if was_set:
            audit_append(conn, actor=actor, action="resume_all",
                         reason="clear global halt; re-base all drawdown peaks", strategy=None)
            # Re-base peaks first, clear the halt LAST so the un-halt is the final write (#109).
            clear_all_peaks(conn)
            global_halt.clear(conn)
    emit(ok({"global_halt": "reset" if was_set else "not_set"}))
```
(Confirm `audit_append`'s `strategy` param accepts `None` — the `audit_log.strategy` column is nullable; if the signature requires a non-None strategy, pass `strategy="*"` instead and note it.)

- [ ] **Step 4: Run** `uv run pytest tests/test_cli_paper.py -q` → PASS.

- [ ] **Step 5: Gate + commit** — `uv run ruff check . && uv run mypy algua && uv run pytest tests/test_cli_paper.py -q`
```bash
git add algua/cli/paper_cmd.py tests/test_cli_paper.py
git commit -m "feat(cli): paper halt-all + resume-all (global kill-switch)"
```

---

### Task 4: gating + `paper show` reflection

**Files:** Modify `algua/cli/paper_cmd.py`; Test `tests/test_cli_paper.py`.

Context: `_load_gated_strategy(conn, name, command)` centralises the stage + per-strategy kill-switch gate that both `run` and `trade-tick` use; it raises `ValueError` on a failed gate. `trade-tick` builds `TickHooks(..., should_halt=lambda: kill_switch.is_tripped(conn, name), ...)`. `show` computes `health` (`halted` if `tripped`, else drift/idle/ok) and emits a `kill_switch` block `{"tripped": ..., "reason": ...}`. `global_halt` is now imported (Task 3).

- [ ] **Step 1: Add failing tests** — append to `tests/test_cli_paper.py`:
```python
def _engage_global_halt():
    from contextlib import closing

    from algua.config.settings import get_settings
    from algua.registry.db import connect, migrate
    from algua.risk import global_halt
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        global_halt.engage(conn, reason="halted", actor="human")


def test_trade_tick_refused_when_globally_halted(monkeypatch):
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    _to_paper()
    _engage_global_halt()
    result = runner.invoke(app, ["paper", "trade-tick", "cross_sectional_momentum",
                                 "--snapshot", "snap1"])
    assert result.exit_code == 1 and json.loads(result.stdout)["ok"] is False


def test_paper_run_refused_when_globally_halted():
    _to_paper()
    _engage_global_halt()
    result = runner.invoke(app, ["paper", "run", "cross_sectional_momentum", "--demo",
                                 "--start", "2022-01-01", "--end", "2023-12-31"])
    assert result.exit_code == 1 and json.loads(result.stdout)["ok"] is False


def test_show_reflects_global_halt():
    _to_paper()
    _engage_global_halt()
    payload = json.loads(runner.invoke(app, ["paper", "show", "cross_sectional_momentum"]).stdout)
    assert payload["health"] == "halted"
    assert payload["kill_switch"]["global_halt"] is True
```

- [ ] **Step 2: Run** `uv run pytest tests/test_cli_paper.py -q -k "globally_halted or reflects_global"` → FAIL.

- [ ] **Step 3a: Gate `run` + `trade-tick`** — in `_load_gated_strategy`, add the global-halt check before the per-strategy kill check:
```python
    if global_halt.is_engaged(conn):
        raise ValueError("global halt active; clear with 'algua paper resume-all'")
    if kill_switch.is_tripped(conn, name):
        raise ValueError(f"kill-switch tripped for {name}; reset with 'algua paper resume {name}'")
```

- [ ] **Step 3b: Pre-submit hook** — in `trade-tick`, change the hook to also abort on a mid-tick halt-all:
```python
            should_halt=lambda: kill_switch.is_tripped(conn, name) or global_halt.is_engaged(conn),
```

- [ ] **Step 3c: `show` reflection** — in `show`, read the global halt and fold it into health + the kill_switch block. After `ks = kill_switch.get(conn, name)` add `halted_globally = global_halt.is_engaged(conn)` (inside the `with`); change `tripped = ks is not None` handling so:
```python
    tripped = ks is not None
    if tripped or halted_globally:
        health = "halted"
    elif last is not None and not last["reconcile_ok"]:
        health = "drift"
    elif last is None:
        health = "idle"
    else:
        health = "ok"
```
and change the emitted `kill_switch` block to:
```python
        "kill_switch": {"tripped": tripped, "reason": ks["reason"] if ks else None,
                        "global_halt": halted_globally},
```

- [ ] **Step 4: Run** `uv run pytest tests/test_cli_paper.py -q` → PASS (existing `show`/`trade-tick`/`run` tests still hold — they run with no global halt engaged).

- [ ] **Step 5: Full gate + commit** — `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports` (all pass; `lint-imports` stays kept, 0 broken).
```bash
git add algua/cli/paper_cmd.py tests/test_cli_paper.py
git commit -m "feat(cli): trade-tick/run/show respect the global halt"
```

---

## Self-review notes

- **Spec coverage:** `global_halt` table + module (§4 → Task 1); `close_all_positions` (§5 → Task 2); `halt-all`/`resume-all` with trip-before-close + re-base-all-peaks + fail-safe ordering (§6 → Task 3); gating in the shared gate + pre-submit hook (§7 → Task 4); `show` reflection (§8 → Task 4); tests for every bullet in §9. Schema v6 + table-present test (§4 → Task 1). The version-assertion test needs no edit (asserts the `SCHEMA_VERSION` constant). Live acceptance is documented in the spec (§9), not code.
- **Type consistency:** `global_halt.engage(conn, *, reason, actor)` / `is_engaged(conn) -> bool` / `clear(conn) -> bool` / `get(conn)` are used identically across Tasks 1, 3, 4. `clear_all_peaks(conn)` (Task 1) is called in `resume-all` (Task 3). `close_all_positions()` (Task 2) is called in `halt-all` (Task 3). The `engage`/`is_engaged` names supersede the spec's `set`/`is_set` (avoids shadowing the `set` builtin) — applied uniformly.
- **No placeholders:** every code step is complete. Two confirm-before-adapting notes (audit_append's nullable `strategy`; the test file's `ab`/`BrokerError` imports) are verifications against existing code, not gaps.
