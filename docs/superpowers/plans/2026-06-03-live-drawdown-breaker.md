# Live Drawdown Breaker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An automatic equity-based circuit breaker — `trade-live` trips the kill-switch and auto-flattens when a strategy's drawdown from its persisted peak equity exceeds `--max-drawdown`.

**Architecture:** A new `live_equity_peak` table (schema v4) persists each strategy's lifetime high-water mark across ticks. `trade-live` reads the peak + current account equity, runs the existing pure `check_drawdown` as a pre-flight gate, and reuses the B2b-1 `except RiskBreach` handler (trip → `cancel_open_orders` → `close_positions(universe)`). `paper resume` clears the peak so a resumed strategy re-bases. `run_tick` is untouched.

**Tech Stack:** Python 3.12, sqlite3, Typer, pytest, ruff, mypy, import-linter. Spec: `docs/superpowers/specs/2026-06-03-live-drawdown-breaker-design.md`.

---

## File structure

| File | Responsibility |
|---|---|
| `algua/registry/db.py` (modify) | `live_equity_peak` table; `SCHEMA_VERSION` 3 → 4. |
| `algua/registry/store.py` (modify) | `get_equity_peak` / `set_equity_peak` / `clear_equity_peak`. |
| `algua/cli/paper_cmd.py` (modify) | `trade-live` `--max-drawdown` + pre-flight breaker; `paper resume` clears the peak. |
| `tests/test_paper_db.py`, `tests/test_registry_db.py`, `tests/test_cli_paper.py` (modify) | Version bump, store round-trip, breaker + resume CLI tests. |

---

### Task 1: schema v4 + equity-peak store helpers

**Files:** Modify `algua/registry/db.py`, `algua/registry/store.py`; Test `tests/test_paper_db.py`, `tests/test_registry_db.py`.

- [ ] **Step 1: Bump the existing version assertions** — these will fail after the schema bump; update them first.

In `tests/test_paper_db.py` line ~19 (`test_migrate_is_idempotent`):
```python
    assert conn.execute("PRAGMA user_version;").fetchone()[0] == 4
```
In `tests/test_registry_db.py` (both assertions, lines ~10 and ~17):
```python
    assert conn.execute("PRAGMA user_version;").fetchone()[0] == 4
```

- [ ] **Step 2: Add failing tests** — append to `tests/test_paper_db.py`:

```python
from algua.registry import store


def test_migrate_creates_live_equity_peak_table(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    assert "live_equity_peak" in _tables(conn)


def test_equity_peak_roundtrip_and_upsert(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    assert store.get_equity_peak(conn, "s") is None      # no row yet
    store.set_equity_peak(conn, "s", 100.0)
    assert store.get_equity_peak(conn, "s") == 100.0
    store.set_equity_peak(conn, "s", 125.5)              # UPSERT overwrites
    assert store.get_equity_peak(conn, "s") == 125.5


def test_clear_equity_peak(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    store.set_equity_peak(conn, "s", 100.0)
    store.clear_equity_peak(conn, "s")
    assert store.get_equity_peak(conn, "s") is None
```

- [ ] **Step 3: Run** `uv run pytest tests/test_paper_db.py -q` → FAIL (table missing / `get_equity_peak` undefined).

- [ ] **Step 4: Add the table + version bump** — in `algua/registry/db.py`, change `SCHEMA_VERSION = 3` to `SCHEMA_VERSION = 4`, and append this table to the `_SCHEMA` string (before the closing `"""`):

```sql
CREATE TABLE IF NOT EXISTS live_equity_peak (
    strategy    TEXT PRIMARY KEY,
    peak_equity REAL NOT NULL,
    updated_ts  TEXT NOT NULL
);
```

- [ ] **Step 5: Add the store helpers** — in `algua/registry/store.py`, append (it already imports `sqlite3`, `datetime`/`UTC`, and defines `_now()`):

```python
def get_equity_peak(conn: sqlite3.Connection, strategy: str) -> float | None:
    """Persisted lifetime high-water-mark equity for a live strategy, or None if untracked."""
    row = conn.execute(
        "SELECT peak_equity FROM live_equity_peak WHERE strategy = ?", (strategy,)
    ).fetchone()
    return float(row["peak_equity"]) if row is not None else None


def set_equity_peak(conn: sqlite3.Connection, strategy: str, peak: float) -> None:
    conn.execute(
        "INSERT INTO live_equity_peak(strategy, peak_equity, updated_ts) VALUES (?,?,?) "
        "ON CONFLICT(strategy) DO UPDATE SET peak_equity=excluded.peak_equity, "
        "updated_ts=excluded.updated_ts",
        (strategy, peak, _now()),
    )
    conn.commit()


def clear_equity_peak(conn: sqlite3.Connection, strategy: str) -> None:
    """Drop the stored peak so the next tick re-bases the high-water mark (used on resume)."""
    conn.execute("DELETE FROM live_equity_peak WHERE strategy = ?", (strategy,))
    conn.commit()
```

- [ ] **Step 6: Run** `uv run pytest tests/test_paper_db.py tests/test_registry_db.py -q` → PASS.

- [ ] **Step 7: Gate + commit** — `uv run ruff check . && uv run mypy algua && uv run pytest tests/test_paper_db.py tests/test_registry_db.py -q`

```bash
git add algua/registry/db.py algua/registry/store.py tests/test_paper_db.py tests/test_registry_db.py
git commit -m "feat(registry): live_equity_peak table + store helpers (schema v4)"
```

---

### Task 2: `trade-live` pre-flight drawdown breaker

**Files:** Modify `algua/cli/paper_cmd.py`; Test `tests/test_cli_paper.py`.

- [ ] **Step 1: Add failing tests** — append to `tests/test_cli_paper.py`. These reuse the existing `_FlattenBroker` (exposes `cancel_open_orders` + `close_positions`) from the auto-flatten slice; `AccountState`, `TickResult`, `pytest`, and `json` are already imported at the top of the file. Append:

```python
class _AccountBroker(_FlattenBroker):
    """_FlattenBroker (cancel_open_orders + close_positions) plus an account() stub."""
    def __init__(self, equity, fail=False):
        super().__init__(fail=fail)
        self._equity = equity

    def account(self):
        return AccountState(equity=self._equity, cash=self._equity, buying_power=self._equity)


def _seed_peak(name, value):
    from contextlib import closing
    from algua.config.settings import get_settings
    from algua.registry import store
    from algua.registry.db import connect, migrate
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        store.set_equity_peak(conn, name, value)


def test_trade_live_drawdown_trips_and_flattens(monkeypatch):
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    _to_paper()
    _seed_peak("cross_sectional_momentum", 100.0)
    broker = _AccountBroker(equity=80.0)  # 20% drawdown
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings", lambda: broker)
    # run_tick must NOT be reached when the breaker trips
    monkeypatch.setattr("algua.cli.paper_cmd.run_tick",
                        lambda *a, **k: pytest.fail("run_tick should not run on breach"))
    result = runner.invoke(app, ["paper", "trade-live", "cross_sectional_momentum",
                                 "--snapshot", "x", "--max-drawdown", "0.10"])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["kind"] == "drawdown" and payload["kill_switch"] == "tripped"
    assert payload["liquidation_submitted"] is True
    assert broker.closed_symbols  # flattened to the strategy universe
    show = json.loads(runner.invoke(app, ["paper", "show", "cross_sectional_momentum"]).stdout)
    assert show["kill_switch"]["tripped"] is True


def test_trade_live_new_high_persists_peak(monkeypatch):
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    _to_paper()
    _seed_peak("cross_sectional_momentum", 100.0)
    broker = _AccountBroker(equity=130.0)  # new high
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings", lambda: broker)
    monkeypatch.setattr("algua.cli.paper_cmd.run_tick",
                        lambda *a, **k: TickResult(None, {}, {}, []))
    result = runner.invoke(app, ["paper", "trade-live", "cross_sectional_momentum",
                                 "--snapshot", "x", "--max-drawdown", "0.10"])
    assert result.exit_code == 0, result.stdout
    from contextlib import closing
    from algua.config.settings import get_settings
    from algua.registry import store
    from algua.registry.db import connect, migrate
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        assert store.get_equity_peak(conn, "cross_sectional_momentum") == 130.0


def test_trade_live_drawdown_disabled_default(monkeypatch):
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    _to_paper()
    _seed_peak("cross_sectional_momentum", 100.0)
    broker = _AccountBroker(equity=10.0)  # huge drawdown, but breaker disabled (default 1.0)
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings", lambda: broker)
    monkeypatch.setattr("algua.cli.paper_cmd.run_tick",
                        lambda *a, **k: TickResult(None, {}, {}, []))
    result = runner.invoke(app, ["paper", "trade-live", "cross_sectional_momentum",
                                 "--snapshot", "x"])  # no --max-drawdown -> 1.0
    assert result.exit_code == 0, result.stdout
```

- [ ] **Step 2: Run** `uv run pytest tests/test_cli_paper.py -q` → FAIL (no `--max-drawdown`, breaker not wired).

- [ ] **Step 3: Add the `check_drawdown` import** — in `algua/cli/paper_cmd.py`, change:
```python
from algua.risk.limits import RiskBreach
```
to:
```python
from algua.risk.limits import RiskBreach, check_drawdown
```

- [ ] **Step 4: Add the `--max-drawdown` option** to `trade_live`. Change its signature:
```python
def trade_live(
    name: str,
    snapshot: str = typer.Option(..., "--snapshot", help="ingested bars snapshot id"),
    start: str = typer.Option("2023-01-01", "--start"),
    end: str = typer.Option("2023-12-31", "--end"),
) -> None:
```
to:
```python
def trade_live(
    name: str,
    snapshot: str = typer.Option(..., "--snapshot", help="ingested bars snapshot id"),
    start: str = typer.Option("2023-01-01", "--start"),
    end: str = typer.Option("2023-12-31", "--end"),
    max_drawdown: float = typer.Option(
        1.0, "--max-drawdown", help="trip+flatten if drawdown from peak equity exceeds this "
        "(0..1; 1.0 disables)"),
) -> None:
```

- [ ] **Step 5: Wire the pre-flight breaker.** In `trade_live`, the block currently reads:
```python
        broker = _alpaca_broker_from_settings()
        provider = _select_provider(False, snapshot)
        try:
            result = run_tick(strategy, broker, provider, _utc(start), _utc(end))
        except RiskBreach as exc:
```
Replace it with (compute equity/peak **before** the `try` so `new_peak` is always bound when persisted; the breach handler is unchanged from B2b-1):
```python
        broker = _alpaca_broker_from_settings()
        provider = _select_provider(False, snapshot)
        equity = broker.account().equity
        peak = store.get_equity_peak(conn, name)
        new_peak = max(peak, equity) if peak is not None else equity
        try:
            check_drawdown(equity, new_peak, max_drawdown)  # pre-flight circuit breaker
            result = run_tick(strategy, broker, provider, _utc(start), _utc(end))
        except RiskBreach as exc:
```

- [ ] **Step 6: Persist the peak on the success path.** Immediately after the `except RiskBreach` block ends (the line after `raise typer.Exit(1) from exc`), and before `now = datetime.now(UTC).isoformat()`, insert:
```python
        store.set_equity_peak(conn, name, new_peak)
```
The full success path now reads:
```python
            raise typer.Exit(1) from exc
        store.set_equity_peak(conn, name, new_peak)
        now = datetime.now(UTC).isoformat()
        decision_ts_str = result.decision_ts.isoformat() if result.decision_ts else None
```

- [ ] **Step 7: Run** `uv run pytest tests/test_cli_paper.py -q` → PASS.

- [ ] **Step 8: Gate + commit** — `uv run ruff check . && uv run mypy algua && uv run pytest tests/test_cli_paper.py -q`

```bash
git add algua/cli/paper_cmd.py tests/test_cli_paper.py
git commit -m "feat(live): pre-flight drawdown breaker on trade-live (trip + auto-flatten)"
```

---

### Task 3: `paper resume` clears the peak (reset-on-resume)

**Files:** Modify `algua/cli/paper_cmd.py`; Test `tests/test_cli_paper.py`.

- [ ] **Step 1: Add a failing test** — append to `tests/test_cli_paper.py`:

```python
def test_resume_clears_equity_peak(monkeypatch):
    _to_paper()
    _seed_peak("cross_sectional_momentum", 100.0)
    result = runner.invoke(app, ["paper", "resume", "cross_sectional_momentum"])
    assert result.exit_code == 0, result.stdout
    from contextlib import closing
    from algua.config.settings import get_settings
    from algua.registry import store
    from algua.registry.db import connect, migrate
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        assert store.get_equity_peak(conn, "cross_sectional_momentum") is None
```

- [ ] **Step 2: Run** `uv run pytest tests/test_cli_paper.py::test_resume_clears_equity_peak -q` → FAIL (peak still present).

- [ ] **Step 3: Clear the peak in `resume`.** The `resume` command currently reads:
```python
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        was_tripped = kill_switch.is_tripped(conn, name)
        if was_tripped:
            # Audit BEFORE clearing: if the reset write fails, the switch stays tripped
            # (fail-safe — still halted) rather than cleared with no audit trail.
            audit_append(conn, actor="human", action="kill_switch_reset",
                         reason="manual resume", strategy=name)
            kill_switch.reset(conn, name)
    emit({"strategy": name, "kill_switch": "reset" if was_tripped else "not_tripped"})
```
Add `store.clear_equity_peak(conn, name)` after the `kill_switch.reset` call (inside the `if`, so the peak re-bases only when a tripped strategy is actually resumed):
```python
            kill_switch.reset(conn, name)
            store.clear_equity_peak(conn, name)  # re-base the high-water mark on resume
```

- [ ] **Step 4: Run** `uv run pytest tests/test_cli_paper.py::test_resume_clears_equity_peak -q` → PASS.

- [ ] **Step 5: Full gate + commit** — `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports` (all pass; `lint-imports` stays `10 kept, 0 broken`).

```bash
git add algua/cli/paper_cmd.py tests/test_cli_paper.py
git commit -m "feat(live): paper resume re-bases the drawdown peak (reset-on-resume)"
```

---

## Self-review notes

- **Spec coverage:** `live_equity_peak` table + store helpers (§3,§4 → Task 1); `--max-drawdown` + pre-flight `check_drawdown` gate + HWM persist + reuse of the B2b-1 `except RiskBreach` handler (§5 → Task 2); reset-on-resume (§6 → Task 3); tests incl. trip+flatten, new-high persist, disabled-default, resume-clears (§7 → Tasks 1–3). Schema-version assertions updated (§4 → Task 1 Step 1). Live acceptance is documented in the spec (§7), not code. No new import contract (§3).
- **Type consistency:** `get_equity_peak -> float | None`, `set_equity_peak(conn, strategy, peak)`, `clear_equity_peak(conn, strategy)` are used identically across Tasks 1–3. `check_drawdown(equity, peak, max_drawdown)` matches the existing signature in `algua/risk/limits.py`. `_AccountBroker.account()` returns the real `AccountState` (with `.equity`) that `trade_live` reads. `new_peak` is computed before the `try` so it is always bound at `set_equity_peak`.
- **No placeholders:** every code step is complete.
- **Note:** the breach path raises `typer.Exit(1)`, so `set_equity_peak` is unreachable after a breach — the peak is intentionally not advanced on a trip (spec §5).
