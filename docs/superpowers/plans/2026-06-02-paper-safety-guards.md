# Paper Safety Guards Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add risk limits (gross exposure), a drawdown circuit-breaker, a per-strategy kill-switch (halt-only, human reset), and a warm-up gate to the paper replay loop — fail-closed: a breach trips the kill-switch.

**Architecture:** Pure checks in `algua/risk/limits.py` (`RiskBreach(ValueError)`); DB-backed kill-switch state in `algua/risk/kill_switch.py`; `run_paper` enforces warm-up + gross + drawdown and raises `RiskBreach` (stays pure); the CLI owns the DB side — refuses to run when tripped and trips the switch on a breach (persisting nothing for a breached run).

**Tech Stack:** Python 3.12, pandas, sqlite3, Typer, pytest, ruff, mypy, import-linter. Spec: `docs/superpowers/specs/2026-06-02-paper-safety-guards-design.md`.

---

## File structure

| File | Responsibility |
|---|---|
| `algua/contracts/types.py` (modify) | `ExecutionContract.warmup_bars: int = 0` + validation. |
| `algua/risk/__init__.py` (new) | Package marker. |
| `algua/risk/limits.py` (new) | `RiskBreach`, `check_gross_exposure`, `check_drawdown`. |
| `algua/risk/kill_switch.py` (new) | `trip`, `is_tripped`, `reset`, `get`. |
| `algua/registry/db.py` (modify) | Schema v3: `kill_switches` table. |
| `algua/live/paper_loop.py` (modify) | Warm-up + gross + drawdown enforcement; `max_drawdown` param. |
| `algua/cli/paper_cmd.py` (modify) | Kill-switch gate + trip-on-breach + `paper kill`/`resume`/`show`. |
| `pyproject.toml` (modify) | Import contracts for `algua.risk`. |

---

### Task 1: `warmup_bars` on ExecutionContract

**Files:** Modify `algua/contracts/types.py`; Test `tests/test_contracts.py`.

- [ ] **Step 1: Add the failing test** — append to `tests/test_contracts.py`:

```python
def test_execution_contract_warmup_bars_default_and_validation():
    from algua.contracts.types import ExecutionContract
    import pytest

    assert ExecutionContract(rebalance_frequency="1d").warmup_bars == 0
    assert ExecutionContract(rebalance_frequency="1d", warmup_bars=30).warmup_bars == 30
    with pytest.raises(ValueError, match="warmup_bars"):
        ExecutionContract(rebalance_frequency="1d", warmup_bars=-1)
```

- [ ] **Step 2: Run** `uv run pytest tests/test_contracts.py -q` → FAIL.

- [ ] **Step 3: Implement** — in `algua/contracts/types.py`, add the field after `max_gross_exposure: float = 1.0`:

```python
    warmup_bars: int = 0
```

and add to `__post_init__` (after the existing `decision_lag_bars` check):

```python
        if self.warmup_bars < 0:
            raise ValueError("warmup_bars must be >= 0")
```

- [ ] **Step 4: Run** `uv run pytest tests/test_contracts.py -q` → PASS.

- [ ] **Step 5: Gate + commit** — `uv run ruff check . && uv run mypy algua && uv run pytest tests/test_contracts.py -q`

```bash
git add algua/contracts/types.py tests/test_contracts.py
git commit -m "feat(risk): add warmup_bars to ExecutionContract"
```

---

### Task 2: Risk limit checks (`risk/limits.py`)

**Files:** Create `algua/risk/__init__.py` (empty), `algua/risk/limits.py`; Test `tests/test_risk_limits.py`.

- [ ] **Step 1: Add the failing test** — create `tests/test_risk_limits.py`:

```python
import pandas as pd
import pytest

from algua.risk.limits import RiskBreach, check_drawdown, check_gross_exposure


def test_risk_breach_is_value_error_with_kind():
    exc = RiskBreach("gross_exposure", "too big")
    assert isinstance(exc, ValueError)
    assert exc.kind == "gross_exposure"
    assert exc.detail == "too big"


def test_gross_exposure_within_limit_passes():
    check_gross_exposure(pd.Series({"AAA": 0.6, "BBB": 0.4}), 1.0)  # == 1.0, ok
    check_gross_exposure(pd.Series(dtype="float64"), 1.0)            # empty, ok


def test_gross_exposure_over_limit_raises():
    with pytest.raises(RiskBreach) as ei:
        check_gross_exposure(pd.Series({"AAA": 1.0, "BBB": 1.0}), 1.0)
    assert ei.value.kind == "gross_exposure"


def test_drawdown_within_limit_passes():
    check_drawdown(equity=95.0, peak=100.0, max_drawdown=0.1)  # 5% < 10%
    check_drawdown(equity=50.0, peak=100.0, max_drawdown=1.0)  # disabled


def test_drawdown_over_limit_raises():
    with pytest.raises(RiskBreach) as ei:
        check_drawdown(equity=80.0, peak=100.0, max_drawdown=0.1)  # 20% > 10%
    assert ei.value.kind == "drawdown"
```

- [ ] **Step 2: Run** `uv run pytest tests/test_risk_limits.py -q` → FAIL.

- [ ] **Step 3: Implement** — create `algua/risk/__init__.py` (empty) and `algua/risk/limits.py`:

```python
from __future__ import annotations

import pandas as pd


class RiskBreach(ValueError):
    """A hard risk-limit breach. Subclasses ValueError so existing CLI error handling
    (json_errors) still renders it; the CLI inspects `.kind` to trip the kill-switch."""

    def __init__(self, kind: str, detail: str) -> None:
        super().__init__(detail)
        self.kind = kind
        self.detail = detail


def check_gross_exposure(weights: pd.Series, max_gross: float) -> None:
    if len(weights) == 0:
        return
    gross = float(weights.abs().sum())
    if gross > max_gross + 1e-9:
        raise RiskBreach(
            "gross_exposure",
            f"gross exposure {gross:.4f} exceeds max_gross_exposure {max_gross:.4f}",
        )


def check_drawdown(equity: float, peak: float, max_drawdown: float) -> None:
    if max_drawdown >= 1.0 or peak <= 0:
        return  # disabled, or no peak yet
    if equity < peak * (1.0 - max_drawdown):
        dd = 1.0 - (equity / peak)
        raise RiskBreach(
            "drawdown",
            f"drawdown {dd:.4f} exceeds max_drawdown {max_drawdown:.4f} "
            f"(equity {equity:.2f}, peak {peak:.2f})",
        )
```

- [ ] **Step 4: Run** `uv run pytest tests/test_risk_limits.py -q` → PASS (5 passed).

- [ ] **Step 5: Gate + commit** — `uv run ruff check . && uv run mypy algua && uv run pytest tests/test_risk_limits.py -q`

```bash
git add algua/risk/__init__.py algua/risk/limits.py tests/test_risk_limits.py
git commit -m "feat(risk): RiskBreach + gross-exposure + drawdown checks"
```

---

### Task 3: Schema v3 — `kill_switches` table

**Files:** Modify `algua/registry/db.py`, `tests/test_registry_db.py`, `tests/test_paper_db.py`.

- [ ] **Step 1: Update existing version assertions + add a failing test.**

In `tests/test_registry_db.py`, change both `== 2` assertions of `PRAGMA user_version` to `== 3`.
In `tests/test_paper_db.py`, change the `== 2` assertion in `test_migrate_is_idempotent` to `== 3`, and add:

```python
def test_migrate_creates_kill_switches_table(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    assert "kill_switches" in _tables(conn)
```

- [ ] **Step 2: Run** `uv run pytest tests/test_registry_db.py tests/test_paper_db.py -q` → FAIL (version is 2; no kill_switches).

- [ ] **Step 3: Implement** — in `algua/registry/db.py`: change `SCHEMA_VERSION = 2` to `SCHEMA_VERSION = 3`, and append inside `_SCHEMA` (before the closing triple-quote):

```sql
CREATE TABLE IF NOT EXISTS kill_switches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy TEXT NOT NULL UNIQUE,
    reason TEXT,
    actor TEXT NOT NULL,
    created_at TEXT NOT NULL
);
```

- [ ] **Step 4: Run** `uv run pytest tests/test_registry_db.py tests/test_paper_db.py -q` → PASS.

- [ ] **Step 5: Gate + commit** — `uv run ruff check . && uv run mypy algua && uv run pytest -q` (FULL suite — a schema bump can break other tests).

```bash
git add algua/registry/db.py tests/test_registry_db.py tests/test_paper_db.py
git commit -m "feat(risk): kill_switches table (schema v3)"
```

---

### Task 4: Kill-switch state (`risk/kill_switch.py`)

**Files:** Create `algua/risk/kill_switch.py`; Test `tests/test_kill_switch.py`.

- [ ] **Step 1: Add the failing test** — create `tests/test_kill_switch.py`:

```python
from algua.registry.db import connect, migrate
from algua.risk import kill_switch


def _conn(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    return conn


def test_trip_then_is_tripped(tmp_path):
    conn = _conn(tmp_path)
    assert kill_switch.is_tripped(conn, "s") is False
    kill_switch.trip(conn, "s", reason="boom", actor="system")
    assert kill_switch.is_tripped(conn, "s") is True
    info = kill_switch.get(conn, "s")
    assert info["reason"] == "boom" and info["actor"] == "system"


def test_reset_clears(tmp_path):
    conn = _conn(tmp_path)
    kill_switch.trip(conn, "s", reason="boom", actor="system")
    assert kill_switch.reset(conn, "s") is True
    assert kill_switch.is_tripped(conn, "s") is False
    assert kill_switch.reset(conn, "s") is False  # nothing to reset


def test_retrip_updates_reason(tmp_path):
    conn = _conn(tmp_path)
    kill_switch.trip(conn, "s", reason="first", actor="agent")
    kill_switch.trip(conn, "s", reason="second", actor="system")
    assert kill_switch.get(conn, "s")["reason"] == "second"
```

- [ ] **Step 2: Run** `uv run pytest tests/test_kill_switch.py -q` → FAIL.

- [ ] **Step 3: Implement** — create `algua/risk/kill_switch.py`:

```python
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime


def trip(conn: sqlite3.Connection, strategy: str, *, reason: str, actor: str) -> None:
    conn.execute(
        "INSERT INTO kill_switches(strategy, reason, actor, created_at) VALUES (?,?,?,?) "
        "ON CONFLICT(strategy) DO UPDATE SET "
        "reason=excluded.reason, actor=excluded.actor, created_at=excluded.created_at",
        (strategy, reason, actor, datetime.now(UTC).isoformat()),
    )
    conn.commit()


def is_tripped(conn: sqlite3.Connection, strategy: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM kill_switches WHERE strategy = ?", (strategy,)
    ).fetchone()
    return row is not None


def reset(conn: sqlite3.Connection, strategy: str) -> bool:
    cur = conn.execute("DELETE FROM kill_switches WHERE strategy = ?", (strategy,))
    conn.commit()
    return cur.rowcount > 0


def get(conn: sqlite3.Connection, strategy: str) -> dict[str, str] | None:
    row = conn.execute(
        "SELECT strategy, reason, actor, created_at FROM kill_switches WHERE strategy = ?",
        (strategy,),
    ).fetchone()
    if row is None:
        return None
    return {"strategy": row["strategy"], "reason": row["reason"],
            "actor": row["actor"], "created_at": row["created_at"]}
```

- [ ] **Step 4: Run** `uv run pytest tests/test_kill_switch.py -q` → PASS (3 passed).

- [ ] **Step 5: Gate + commit** — `uv run ruff check . && uv run mypy algua && uv run pytest tests/test_kill_switch.py -q`

```bash
git add algua/risk/kill_switch.py tests/test_kill_switch.py
git commit -m "feat(risk): per-strategy kill-switch state (trip/is_tripped/reset/get)"
```

---

### Task 5: Enforce guards in `run_paper`

**Files:** Modify `algua/live/paper_loop.py`; Test `tests/test_paper_loop.py`.

- [ ] **Step 1: Add failing tests** — append to `tests/test_paper_loop.py`:

```python
from algua.risk.limits import RiskBreach


def _falling(symbol="AAA"):
    return _bars({symbol: [100.0, 90.0, 80.0, 70.0]})


def _strategy(weights: dict, warmup_bars: int = 0):
    cfg = StrategyConfig(
        name="cfg", universe=sorted(weights),
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1,
                                     warmup_bars=warmup_bars),
    )
    return LoadedStrategy(config=cfg, fn=lambda view, params: pd.Series(weights))


def test_gross_exposure_breach_raises():
    bars = _bars({"AAA": [100.0, 100.0, 100.0, 100.0], "BBB": [100.0, 100.0, 100.0, 100.0]})
    strat = _strategy({"AAA": 1.0, "BBB": 1.0})  # gross 2.0 > 1.0
    with pytest.raises(RiskBreach) as ei:
        run_paper(strat, SimBroker(cash=10_000.0), _FakeProvider(bars), DATES[0], DATES[-1])
    assert ei.value.kind == "gross_exposure"


def test_drawdown_breach_raises():
    strat = _strategy({"AAA": 1.0})  # all-in a falling stock
    with pytest.raises(RiskBreach) as ei:
        run_paper(strat, SimBroker(cash=10_000.0), _FakeProvider(_falling()),
                  DATES[0], DATES[-1], max_drawdown=0.05)
    assert ei.value.kind == "drawdown"


def test_warmup_gate_delays_first_order():
    bars = _bars({"AAA": [100.0, 100.0, 100.0, 100.0]})
    strat = _strategy({"AAA": 1.0}, warmup_bars=2)
    result = run_paper(strat, SimBroker(cash=10_000.0), _FakeProvider(bars), DATES[0], DATES[-1])
    # bars_seen reaches 2 on the 2nd bar (DATES[1]); no order is decided on DATES[0]
    assert all(o.decision_ts >= DATES[1] for o in result.orders)
    assert len(result.orders) >= 1
```

(The existing `test_run_paper_rejects_negative_weights_long_only` still passes because `RiskBreach` is a `ValueError` matching "long-only".)

- [ ] **Step 2: Run** `uv run pytest tests/test_paper_loop.py -q` → FAIL (new tests).

- [ ] **Step 3: Implement** — in `algua/live/paper_loop.py`:

Add imports near the top (after the existing `from algua.execution.sim_broker import ...`):

```python
from algua.risk.limits import RiskBreach, check_drawdown, check_gross_exposure
```

Change the `run_paper` signature to add `max_drawdown`:

```python
def run_paper(
    strategy: LoadedStrategy,
    broker: SimBroker,
    provider: Any,
    start: datetime,
    end: datetime,
    timeframe: str = "1d",
    max_drawdown: float = 1.0,
) -> PaperRunResult:
```

Replace the loop body (from `ts = list(opens.index)` down to the `fills.extend(...)` line) with:

```python
    ts = list(opens.index)
    warmup = strategy.execution.warmup_bars
    max_gross = strategy.execution.max_gross_exposure
    peak = broker.equity(closes.loc[ts[0]]) if ts else broker.cash
    bars_seen = 0

    orders: list[OrderIntent] = []
    fills: list[Fill] = []
    for i in range(len(ts) - 1):  # only bars with a successor can fill
        t, t_next = ts[i], ts[i + 1]
        bars_seen += 1
        view = bars.loc[:t]
        weights = strategy.target_weights(view)
        if len(weights) and bool((weights < 0).any()):
            negative = sorted(weights[weights < 0].index)
            raise RiskBreach(
                "long_only",
                f"long-only: strategy '{strategy.name}' returned negative target weight(s) "
                f"for {negative} at {t}",
            )
        check_gross_exposure(weights, max_gross)
        equity = broker.equity(closes.loc[t])
        peak = max(peak, equity)
        check_drawdown(equity, peak, max_drawdown)
        if bars_seen >= warmup:
            for intent in build_intents(weights, broker.get_positions(), closes.loc[t], equity, t):
                broker.submit(intent)
                orders.append(intent)
            fills.extend(broker.fill_pending(opens.loc[t_next], fill_ts=t_next))
```

(The block after the loop — `final_positions`, `reconcile_ok`, `return PaperRunResult(...)` — is unchanged. Remove the old long-only `raise ValueError(...)` block, since it's replaced by the `RiskBreach("long_only", ...)` above.)

- [ ] **Step 4: Run** `uv run pytest tests/test_paper_loop.py -q` → PASS (all, including the unchanged long-only test).

- [ ] **Step 5: Gate + commit** — `uv run ruff check . && uv run mypy algua && uv run pytest tests/test_paper_loop.py -q`

```bash
git add algua/live/paper_loop.py tests/test_paper_loop.py
git commit -m "feat(risk): enforce warm-up + gross-exposure + drawdown in run_paper"
```

---

### Task 6: CLI — kill-switch gate, trip-on-breach, `paper kill`/`resume`/`show`

**Files:** Modify `algua/cli/paper_cmd.py`; Test `tests/test_cli_paper.py`.

- [ ] **Step 1: Add failing tests** — append to `tests/test_cli_paper.py`:

```python
from algua.risk.limits import RiskBreach


def test_manual_kill_blocks_run_then_resume_allows(monkeypatch):
    _to_paper()
    assert runner.invoke(app, ["paper", "kill", "cross_sectional_momentum",
                               "--reason", "manual"]).exit_code == 0
    blocked = runner.invoke(app, ["paper", "run", "cross_sectional_momentum", "--demo",
                                  "--start", "2022-01-01", "--end", "2023-12-31"])
    assert blocked.exit_code == 1
    assert json.loads(blocked.stdout)["ok"] is False
    assert runner.invoke(app, ["paper", "resume", "cross_sectional_momentum"]).exit_code == 0
    ok = runner.invoke(app, ["paper", "run", "cross_sectional_momentum", "--demo",
                             "--start", "2022-01-01", "--end", "2023-12-31"])
    assert ok.exit_code == 0


def test_breach_trips_killswitch_and_persists_nothing(monkeypatch):
    _to_paper()

    def _boom(*a, **k):
        raise RiskBreach("drawdown", "drawdown 0.30 exceeds max_drawdown 0.10")

    monkeypatch.setattr("algua.cli.paper_cmd.run_paper", _boom)
    result = runner.invoke(app, ["paper", "run", "cross_sectional_momentum", "--demo",
                                 "--start", "2022-01-01", "--end", "2023-12-31",
                                 "--max-drawdown", "0.1"])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False and payload["kind"] == "drawdown"
    show = json.loads(runner.invoke(app, ["paper", "show", "cross_sectional_momentum"]).stdout)
    assert show["kill_switch"]["tripped"] is True
    assert show["n_orders"] == 0  # breached run persisted nothing
```

- [ ] **Step 2: Run** `uv run pytest tests/test_cli_paper.py -q` → FAIL.

- [ ] **Step 3: Implement** — edit `algua/cli/paper_cmd.py`.

Add imports (after the existing `from algua.live.paper_loop import run_paper`):

```python
from algua.risk import kill_switch
from algua.risk.limits import RiskBreach
```

Replace the `run` command's body (the part inside `with closing(...) as conn:` plus the final `emit`) with:

```python
    if cash <= 0:
        raise ValueError("--cash must be > 0")
    strategy = load_strategy(name)
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        rec = store.get_strategy(conn, name)
        if rec.stage is not Stage.PAPER:
            raise ValueError(f"{name} is at stage '{rec.stage.value}'; paper run requires 'paper'")
        if kill_switch.is_tripped(conn, name):
            raise ValueError(
                f"kill-switch tripped for {name}; reset with 'algua paper resume {name}'"
            )
        provider = _select_provider(demo, snapshot)
        try:
            result = run_paper(strategy, SimBroker(cash=cash), provider,
                               _utc(start), _utc(end), max_drawdown=max_drawdown)
        except RiskBreach as exc:
            kill_switch.trip(conn, name, reason=exc.detail, actor="system")
            audit_append(conn, actor="system", action="kill_switch_trip",
                         reason=f"{exc.kind}: {exc.detail}", strategy=name)
            emit({"ok": False, "kind": exc.kind, "kill_switch": "tripped", "error": exc.detail})
            raise typer.Exit(1) from exc
        persist_run(conn, result)
        audit_append(
            conn, actor="agent", action="paper_run",
            reason=f"{len(result.orders)} orders, {len(result.fills)} fills",
            strategy=name,
        )

    emit({
        "strategy": result.strategy,
        "orders": len(result.orders),
        "fills": len(result.fills),
        "final_positions": result.final_positions,
        "final_cash": result.final_cash,
        "final_equity": result.final_equity,
        "reconcile_ok": result.reconcile_ok,
    })
```

Add the `--max-drawdown` option to `run`'s signature (after `cash`):

```python
    max_drawdown: float = typer.Option(1.0, "--max-drawdown",
                                       help="trip the kill-switch if equity falls this fraction below peak (1.0 = off)"),
```

Extend `show` to report kill-switch state — replace its `emit(...)` with:

```python
        ks = kill_switch.get(conn, name)
    emit({
        "strategy": name, "n_orders": n_orders, "positions": positions,
        "kill_switch": {"tripped": ks is not None, "reason": ks["reason"] if ks else None},
    })
```

(Move the `ks = kill_switch.get(conn, name)` line inside the `with closing(...)` block, before the `emit`.)

Add two new commands at the end of the file:

```python
@paper_app.command("kill")
@json_errors(ValueError)
def kill(
    name: str,
    reason: str = typer.Option(..., "--reason", help="why the strategy is being halted"),
    actor: str = typer.Option("agent", "--actor", help="human | agent"),
) -> None:
    """Manually trip the kill-switch for a strategy (halts paper runs until reset)."""
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        kill_switch.trip(conn, name, reason=reason, actor=actor)
        audit_append(conn, actor=actor, action="kill_switch_trip", reason=reason, strategy=name)
    emit({"strategy": name, "kill_switch": "tripped", "reason": reason})


@paper_app.command("resume")
@json_errors(ValueError)
def resume(name: str) -> None:
    """Reset (clear) a strategy's kill-switch so paper runs may resume. Human action."""
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        was_tripped = kill_switch.reset(conn, name)
        if was_tripped:
            audit_append(conn, actor="human", action="kill_switch_reset",
                         reason="manual resume", strategy=name)
    emit({"strategy": name, "kill_switch": "reset" if was_tripped else "not_tripped"})
```

- [ ] **Step 4: Run** `uv run pytest tests/test_cli_paper.py -q` → PASS.

- [ ] **Step 5: Gate + commit** — `uv run ruff check . && uv run mypy algua && uv run pytest -q` (FULL suite).

```bash
git add algua/cli/paper_cmd.py tests/test_cli_paper.py
git commit -m "feat(risk): paper kill/resume + kill-switch gate + trip-on-breach"
```

---

### Task 7: Import boundaries + full gate

**Files:** Modify `pyproject.toml`.

- [ ] **Step 1: Add contracts** — append to `pyproject.toml`:

```toml
[[tool.importlinter.contracts]]
name = "risk layer stays off the cli layer"
type = "forbidden"
source_modules = ["algua.risk"]
forbidden_modules = ["algua.cli"]

[[tool.importlinter.contracts]]
name = "backtest engine stays off the risk lane"
type = "forbidden"
source_modules = ["algua.backtest"]
forbidden_modules = ["algua.risk"]
```

- [ ] **Step 2: Verify** — `uv run lint-imports` → expect `Contracts: 10 kept, 0 broken.` (If broken: `algua.risk` must not import `algua.cli`; `algua.live` importing `algua.risk` is allowed.)

- [ ] **Step 3: Full gate** — `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports` → all pass.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "test(risk): enforce risk-layer import boundaries"
```

---

## Self-review notes

- **Spec coverage:** `warmup_bars` (§2,§3 → Task 1); gross + drawdown checks + `RiskBreach` (§3,§4 → Task 2); `kill_switches` schema (§3 → Task 3); kill-switch state (§3 → Task 4); loop enforcement of warm-up/gross/drawdown raising `RiskBreach` (§4 → Task 5); CLI gate + trip-on-breach + `kill`/`resume`/`show` + `--max-drawdown` + no-persist-on-breach (§5 → Task 6); import boundaries (§3 → Task 7). Fail-closed (breach → trip) is realized in Task 6's `except RiskBreach` path.
- **Type consistency:** `RiskBreach(kind, detail)` and `check_gross_exposure(weights, max_gross)` / `check_drawdown(equity, peak, max_drawdown)` are identical across Tasks 2/5/6; `kill_switch.trip(conn, strategy, *, reason, actor)` / `is_tripped` / `reset` / `get` match across Tasks 4/6; `run_paper(..., max_drawdown=1.0)` matches between Task 5 and the CLI in Task 6.
- **Regression guard:** Task 3 bumps the schema to v3, so it explicitly updates the `user_version` assertions in BOTH `test_registry_db.py` and `test_paper_db.py` and runs the FULL suite (the v2 bump previously broke a foundation test that only the full suite caught).
- **No placeholders:** every code step is complete.
