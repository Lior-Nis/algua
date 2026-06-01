# Paper Trading Walking Skeleton — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a `paper`-stage strategy bar-by-bar through a local simulated broker (replay), generating orders that fill at the next bar's open, persisting order/fill/audit state, and reconciling — all offline and deterministic.

**Architecture:** A pure replay loop (`run_paper`) reuses the strategy's `target_weights` unchanged, diffs target vs current weights to emit `OrderIntent`s, and a `SimBroker` sizes + fills them at `t+1` open (sells before buys, no negative cash). Loop and broker are pure/in-memory and injected; persistence (`order_state`, `audit`) and provider/broker wiring happen in the CLI. Paper state lives in the existing SQLite registry DB.

**Tech Stack:** Python 3.12, pandas, pydantic, Typer, sqlite3, pytest, ruff, mypy, import-linter. Spec: `docs/superpowers/specs/2026-06-01-paper-trading-walking-skeleton-design.md`.

---

## File structure

| File | Responsibility |
|---|---|
| `algua/registry/db.py` (modify) | Bump `SCHEMA_VERSION` → 2; add `paper_orders`, `paper_fills`, `audit_log` tables. |
| `algua/execution/__init__.py` | Package marker. |
| `algua/execution/sim_broker.py` | `Fill` dataclass + `SimBroker` (cash/positions, `submit`, `get_positions`, `equity`, `fill_pending`). |
| `algua/execution/order_state.py` | `persist_run`, `derive_positions`, `reconcile`. |
| `algua/live/__init__.py` | Package marker. |
| `algua/live/paper_loop.py` | `PaperRunResult`, `build_intents`, `run_paper` (the replay loop). |
| `algua/audit/__init__.py` | Package marker. |
| `algua/audit/log.py` | `append`, `read`. |
| `algua/cli/paper_cmd.py` | `algua paper run` / `algua paper show`. |
| `algua/cli/main.py` (modify) | Register `paper_cmd`. |
| `pyproject.toml` (modify) | New import-linter contracts. |
| `tests/test_*` | One test file per module + a CLI e2e. |

---

### Task 1: DB tables for paper state

**Files:**
- Modify: `algua/registry/db.py`
- Test: `tests/test_paper_db.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_paper_db.py`:

```python
from algua.registry.db import connect, migrate


def _tables(conn):
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return {r["name"] for r in rows}


def test_migrate_creates_paper_tables(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    tables = _tables(conn)
    assert {"paper_orders", "paper_fills", "audit_log"} <= tables


def test_migrate_is_idempotent(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    migrate(conn)  # second run must not raise
    assert conn.execute("PRAGMA user_version;").fetchone()[0] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_paper_db.py -q`
Expected: FAIL (`paper_orders` not in tables; user_version is 1).

- [ ] **Step 3: Implement the migration**

In `algua/registry/db.py`: change `SCHEMA_VERSION = 1` to `SCHEMA_VERSION = 2`, and append these tables inside the `_SCHEMA` string (before the closing `"""`):

```sql
CREATE TABLE IF NOT EXISTS paper_orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    target_weight REAL NOT NULL,
    decision_ts TEXT NOT NULL,
    submitted_ts TEXT NOT NULL,
    status TEXT NOT NULL,
    broker_order_id TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS paper_fills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id INTEGER NOT NULL REFERENCES paper_orders(id),
    symbol TEXT NOT NULL,
    qty REAL NOT NULL,
    price REAL NOT NULL,
    fill_ts TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    reason TEXT,
    strategy TEXT
);
```

The existing `migrate()` runs `executescript(_SCHEMA)` when `user_version < SCHEMA_VERSION`; all tables use `IF NOT EXISTS`, so bumping to 2 re-runs the script safely on old and new DBs.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_paper_db.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Quality gate + commit**

Run: `uv run ruff check . && uv run mypy algua && uv run pytest tests/test_paper_db.py -q`

```bash
git add algua/registry/db.py tests/test_paper_db.py
git commit -m "feat(paper): add paper_orders/paper_fills/audit_log tables (schema v2)"
```

---

### Task 2: SimBroker

**Files:**
- Create: `algua/execution/__init__.py` (empty)
- Create: `algua/execution/sim_broker.py`
- Test: `tests/test_sim_broker.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_sim_broker.py`:

```python
from datetime import UTC, datetime

import pandas as pd
from algua.contracts.types import OrderIntent, Side
from algua.execution.sim_broker import SimBroker

T0 = datetime(2023, 1, 2, tzinfo=UTC)
T1 = datetime(2023, 1, 3, tzinfo=UTC)


def _intent(symbol, weight, side):
    return OrderIntent(symbol=symbol, side=side, target_weight=weight, decision_ts=T0)


def test_fill_buys_at_next_open_and_spends_cash():
    b = SimBroker(cash=10_000.0)
    b.submit(_intent("AAA", 0.5, Side.BUY))
    fills = b.fill_pending(pd.Series({"AAA": 100.0}), fill_ts=T1)
    # 0.5 * 10_000 / 100 = 50 shares
    assert len(fills) == 1 and fills[0].qty == 50.0 and fills[0].price == 100.0
    assert b.cash == 10_000.0 - 50 * 100.0
    assert b.get_positions()["AAA"] == 50.0


def test_sells_processed_before_buys_to_free_cash():
    b = SimBroker(cash=0.0)
    b.positions["AAA"] = 100.0  # hold 100 AAA
    b.submit(_intent("AAA", 0.0, Side.SELL))   # exit AAA
    b.submit(_intent("BBB", 1.0, Side.BUY))    # rotate into BBB
    opens = pd.Series({"AAA": 100.0, "BBB": 50.0})
    fills = b.fill_pending(opens, fill_ts=T1)
    syms = {f.symbol: f.qty for f in fills}
    assert syms["AAA"] == -100.0          # sold all AAA (frees 10_000 cash)
    assert syms["BBB"] > 0                 # bought BBB with freed cash
    assert b.cash >= 0.0


def test_buy_clamped_to_available_cash():
    b = SimBroker(cash=150.0)
    b.submit(_intent("AAA", 1.0, Side.BUY))
    b.fill_pending(pd.Series({"AAA": 100.0}), fill_ts=T1)  # wants 1 share max affordable
    assert b.get_positions().get("AAA", 0.0) == 1.0
    assert b.cash >= 0.0


def test_pending_cleared_after_fill():
    b = SimBroker(cash=1000.0)
    b.submit(_intent("AAA", 0.5, Side.BUY))
    b.fill_pending(pd.Series({"AAA": 100.0}), fill_ts=T1)
    # a second fill with no new submits produces nothing
    assert b.fill_pending(pd.Series({"AAA": 100.0}), fill_ts=T1) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_sim_broker.py -q`
Expected: FAIL (`algua.execution.sim_broker` missing).

- [ ] **Step 3: Implement SimBroker**

Create `algua/execution/__init__.py` (empty) and `algua/execution/sim_broker.py`:

```python
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from algua.contracts.types import OrderIntent


@dataclass(frozen=True)
class Fill:
    symbol: str
    qty: float  # signed shares: +buy, -sell
    price: float
    decision_ts: datetime
    fill_ts: datetime
    broker_order_id: str


class SimBroker:
    """In-process paper broker: fills submitted orders at the next bar's open, full fill,
    no slippage. Sells are applied before buys so freed cash funds buys; cash never goes
    negative. Implements the contracts Broker surface (submit, get_positions) plus the
    sim-only equity()/fill_pending() the replay loop drives."""

    def __init__(self, cash: float) -> None:
        self.cash = float(cash)
        self.positions: dict[str, float] = {}
        self._pending: list[tuple[str, OrderIntent]] = []
        self._seq = 0

    def submit(self, intent: OrderIntent) -> str:
        self._seq += 1
        order_id = f"sim-{self._seq}"
        self._pending.append((order_id, intent))
        return order_id

    def get_positions(self) -> pd.Series:
        return pd.Series(
            {s: q for s, q in self.positions.items() if q != 0.0}, dtype="float64"
        )

    def equity(self, prices: pd.Series) -> float:
        held = sum(q * float(prices.get(s, 0.0)) for s, q in self.positions.items())
        return self.cash + held

    def fill_pending(self, opens: pd.Series, fill_ts: datetime) -> list[Fill]:
        eq = self.equity(opens)
        planned: list[tuple[str, OrderIntent, float, float]] = []  # id, intent, qty, price
        for order_id, intent in self._pending:
            price = float(opens.get(intent.symbol, float("nan")))
            if not price > 0:
                continue
            target_shares = math.floor(intent.target_weight * eq / price)
            qty = target_shares - self.positions.get(intent.symbol, 0.0)
            if qty != 0.0:
                planned.append((order_id, intent, qty, price))
        planned.sort(key=lambda p: p[2])  # sells (negative qty) first
        fills: list[Fill] = []
        for order_id, intent, qty, price in planned:
            if qty > 0:  # buy: clamp to cash on hand
                qty = min(qty, float(math.floor(self.cash / price)))
                if qty <= 0:
                    continue
            self.cash -= qty * price
            self.positions[intent.symbol] = self.positions.get(intent.symbol, 0.0) + qty
            fills.append(
                Fill(intent.symbol, float(qty), price, intent.decision_ts, fill_ts, order_id)
            )
        self._pending.clear()
        return fills
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_sim_broker.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Quality gate + commit**

Run: `uv run ruff check . && uv run mypy algua && uv run pytest tests/test_sim_broker.py -q`

```bash
git add algua/execution/__init__.py algua/execution/sim_broker.py tests/test_sim_broker.py
git commit -m "feat(paper): SimBroker fills at next-bar open, sells-before-buys, cash-safe"
```

---

### Task 3: The replay loop

**Files:**
- Create: `algua/live/__init__.py` (empty)
- Create: `algua/live/paper_loop.py`
- Test: `tests/test_paper_loop.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_paper_loop.py`:

```python
from datetime import UTC, datetime

import pandas as pd
from algua.contracts.types import ExecutionContract
from algua.execution.sim_broker import SimBroker
from algua.live.paper_loop import build_intents, run_paper
from algua.strategies.base import LoadedStrategy, StrategyConfig

DATES = [datetime(2023, 1, d, tzinfo=UTC) for d in (2, 3, 4, 5)]


def _bars(symbol_prices: dict[str, list[float]]) -> pd.DataFrame:
    rows = []
    for sym, prices in symbol_prices.items():
        for ts, px in zip(DATES, prices, strict=True):
            rows.append({"timestamp": ts, "symbol": sym, "open": px, "high": px,
                         "low": px, "close": px, "adj_close": px, "volume": 1000})
    return pd.DataFrame(rows).set_index("timestamp").sort_index()


class _FakeProvider:
    def __init__(self, bars):
        self._bars = bars

    def get_bars(self, symbols, start, end, timeframe):
        return self._bars


def _all_in(symbol: str) -> LoadedStrategy:
    cfg = StrategyConfig(
        name="all_in", universe=[symbol],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
    )
    return LoadedStrategy(config=cfg, fn=lambda view, params: pd.Series({symbol: 1.0}))


def test_build_intents_emits_on_weight_change():
    weights = pd.Series({"AAA": 1.0})
    positions = pd.Series(dtype="float64")
    closes = pd.Series({"AAA": 100.0})
    intents = build_intents(weights, positions, closes, equity=10_000.0,
                            decision_ts=DATES[0])
    assert len(intents) == 1 and intents[0].symbol == "AAA"


def test_build_intents_noop_when_already_at_target():
    intents = build_intents(pd.Series(dtype="float64"), pd.Series(dtype="float64"),
                            pd.Series(dtype="float64"), equity=0.0, decision_ts=DATES[0])
    assert intents == []


def test_run_paper_buys_and_reconciles():
    bars = _bars({"AAA": [100.0, 100.0, 100.0, 100.0]})
    result = run_paper(_all_in("AAA"), SimBroker(cash=10_000.0), _FakeProvider(bars),
                       DATES[0], DATES[-1])
    assert result.reconcile_ok is True
    assert result.final_positions.get("AAA", 0.0) == 100.0  # 10_000 / 100
    assert len(result.fills) >= 1


def test_fills_never_share_timestamp_with_their_decision_bar():
    bars = _bars({"AAA": [100.0, 100.0, 100.0, 100.0]})
    result = run_paper(_all_in("AAA"), SimBroker(cash=10_000.0), _FakeProvider(bars),
                       DATES[0], DATES[-1])
    for f in result.fills:
        assert f.fill_ts > f.decision_ts  # t -> t+1, never same-bar
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_paper_loop.py -q`
Expected: FAIL (`algua.live.paper_loop` missing).

- [ ] **Step 3: Implement the loop**

Create `algua/live/__init__.py` (empty) and `algua/live/paper_loop.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from algua.contracts.types import OrderIntent, Side
from algua.execution.sim_broker import Fill, SimBroker
from algua.strategies.base import LoadedStrategy

_EPS = 1e-6


@dataclass
class PaperRunResult:
    strategy: str
    orders: list[OrderIntent]
    fills: list[Fill]
    final_positions: dict[str, float]
    final_cash: float
    final_equity: float
    reconcile_ok: bool


def build_intents(
    weights: pd.Series,
    positions: pd.Series,
    closes: pd.Series,
    equity: float,
    decision_ts: datetime,
) -> list[OrderIntent]:
    """Emit one OrderIntent per symbol whose target weight differs from its current weight."""
    intents: list[OrderIntent] = []
    symbols = sorted(set(weights.index) | set(positions.index))
    for sym in symbols:
        target = float(weights.get(sym, 0.0))
        shares = float(positions.get(sym, 0.0))
        current = (shares * float(closes.get(sym, 0.0)) / equity) if equity > 0 else 0.0
        if abs(target - current) > _EPS:
            side = Side.BUY if target > current else Side.SELL
            intents.append(
                OrderIntent(symbol=sym, side=side, target_weight=target, decision_ts=decision_ts)
            )
    return intents


def run_paper(
    strategy: LoadedStrategy,
    broker: SimBroker,
    provider: object,  # contracts.DataProvider; object to keep this module import-light
    start: datetime,
    end: datetime,
    timeframe: str = "1d",
) -> PaperRunResult:
    """Replay the strategy bar-by-bar: decide weights on closed bar t (data <= t), submit
    orders, fill at t+1 open. Pure over the injected broker + provider."""
    bars = provider.get_bars(strategy.universe, start, end, timeframe).sort_index()
    opens = bars.reset_index().pivot(index="timestamp", columns="symbol", values="open").sort_index()
    closes = bars.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close").sort_index()
    ts = list(opens.index)

    orders: list[OrderIntent] = []
    fills: list[Fill] = []
    for i in range(len(ts) - 1):  # only bars with a successor can fill
        t, t_next = ts[i], ts[i + 1]
        view = bars.loc[:t]
        weights = strategy.target_weights(view)
        equity = broker.equity(closes.loc[t])
        for intent in build_intents(weights, broker.get_positions(), closes.loc[t], equity, t):
            broker.submit(intent)
            orders.append(intent)
        fills.extend(broker.fill_pending(opens.loc[t_next], fill_ts=t_next))

    final_positions = {s: float(q) for s, q in broker.get_positions().items()}
    final_equity = broker.equity(closes.loc[ts[-1]]) if ts else broker.cash
    derived: dict[str, float] = {}
    for f in fills:
        derived[f.symbol] = derived.get(f.symbol, 0.0) + f.qty
    reconcile_ok = {s: q for s, q in derived.items() if q != 0.0} == final_positions
    return PaperRunResult(
        strategy=strategy.name, orders=orders, fills=fills,
        final_positions=final_positions, final_cash=broker.cash,
        final_equity=final_equity, reconcile_ok=reconcile_ok,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_paper_loop.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Quality gate + commit**

Run: `uv run ruff check . && uv run mypy algua && uv run pytest tests/test_paper_loop.py -q`

```bash
git add algua/live/__init__.py algua/live/paper_loop.py tests/test_paper_loop.py
git commit -m "feat(paper): replay loop reusing target_weights, fills at t+1 open"
```

---

### Task 4: Order-state persistence + reconcile

**Files:**
- Create: `algua/execution/order_state.py`
- Test: `tests/test_order_state.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_order_state.py`:

```python
from datetime import UTC, datetime

import pandas as pd
from algua.contracts.types import OrderIntent, Side
from algua.execution.order_state import derive_positions, persist_run, reconcile
from algua.execution.sim_broker import Fill
from algua.live.paper_loop import PaperRunResult
from algua.registry.db import connect, migrate

T0 = datetime(2023, 1, 2, tzinfo=UTC)
T1 = datetime(2023, 1, 3, tzinfo=UTC)


def _conn(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    return conn


def test_persist_run_writes_orders_and_fills(tmp_path):
    conn = _conn(tmp_path)
    result = PaperRunResult(
        strategy="s",
        orders=[OrderIntent("AAA", Side.BUY, 1.0, T0)],
        fills=[Fill("AAA", 50.0, 100.0, T0, T1, "sim-1")],
        final_positions={"AAA": 50.0}, final_cash=5000.0,
        final_equity=10000.0, reconcile_ok=True,
    )
    persist_run(conn, result)
    assert conn.execute("SELECT COUNT(*) FROM paper_orders").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM paper_fills").fetchone()[0] == 1
    assert derive_positions(conn, "s") == {"AAA": 50.0}


def test_reconcile_true_on_match_false_on_mismatch():
    assert reconcile({"AAA": 50.0}, pd.Series({"AAA": 50.0})) is True
    assert reconcile({"AAA": 50.0}, pd.Series({"AAA": 49.0})) is False
    assert reconcile({}, pd.Series(dtype="float64")) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_order_state.py -q`
Expected: FAIL (`algua.execution.order_state` missing).

- [ ] **Step 3: Implement order_state**

Create `algua/execution/order_state.py`:

```python
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

import pandas as pd

from algua.live.paper_loop import PaperRunResult


def persist_run(conn: sqlite3.Connection, result: PaperRunResult) -> None:
    """Persist a run's orders and fills. Fills link to their order by broker_order_id."""
    now = datetime.now(UTC).isoformat()
    fills_by_order: dict[str, list] = {}
    for f in result.fills:
        fills_by_order.setdefault(f.broker_order_id, []).append(f)

    # Re-derive each order's broker id from submission order: orders were submitted in list
    # order, so assign sim-ids the same way the broker did (sim-1, sim-2, ...).
    for seq, intent in enumerate(result.orders, start=1):
        broker_order_id = f"sim-{seq}"
        matched = fills_by_order.get(broker_order_id, [])
        status = "filled" if matched else "noop"
        cur = conn.execute(
            "INSERT INTO paper_orders"
            "(strategy, symbol, side, target_weight, decision_ts, submitted_ts, status, broker_order_id)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (result.strategy, intent.symbol, intent.side.value, intent.target_weight,
             intent.decision_ts.isoformat(), now, status, broker_order_id),
        )
        order_row_id = cur.lastrowid
        for f in matched:
            conn.execute(
                "INSERT INTO paper_fills(order_id, symbol, qty, price, fill_ts) VALUES (?,?,?,?,?)",
                (order_row_id, f.symbol, f.qty, f.price, f.fill_ts.isoformat()),
            )
    conn.commit()


def derive_positions(conn: sqlite3.Connection, strategy: str) -> dict[str, float]:
    rows = conn.execute(
        "SELECT f.symbol AS symbol, SUM(f.qty) AS qty FROM paper_fills f "
        "JOIN paper_orders o ON o.id = f.order_id WHERE o.strategy = ? GROUP BY f.symbol",
        (strategy,),
    ).fetchall()
    return {r["symbol"]: float(r["qty"]) for r in rows if float(r["qty"]) != 0.0}


def reconcile(derived: dict[str, float], broker_positions: pd.Series) -> bool:
    broker = {s: float(q) for s, q in broker_positions.items() if float(q) != 0.0}
    return {s: q for s, q in derived.items() if q != 0.0} == broker
```

Note: the broker assigns ids `sim-1, sim-2, …` in submission order, and `run_paper` appends to `orders` in that same order, so re-deriving `sim-{seq}` here matches the fills' `broker_order_id` without threading ids through `PaperRunResult`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_order_state.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Quality gate + commit**

Run: `uv run ruff check . && uv run mypy algua && uv run pytest tests/test_order_state.py -q`

```bash
git add algua/execution/order_state.py tests/test_order_state.py
git commit -m "feat(paper): persist orders/fills + derive positions + reconcile"
```

---

### Task 5: Audit log

**Files:**
- Create: `algua/audit/__init__.py` (empty)
- Create: `algua/audit/log.py`
- Test: `tests/test_audit_log.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_audit_log.py`:

```python
from algua.audit.log import append, read
from algua.registry.db import connect, migrate


def test_append_and_read_roundtrip(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    append(conn, actor="agent", action="paper_run", reason="2 orders", strategy="s")
    rows = read(conn, strategy="s")
    assert len(rows) == 1
    assert rows[0]["actor"] == "agent"
    assert rows[0]["action"] == "paper_run"
    assert rows[0]["strategy"] == "s"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_audit_log.py -q`
Expected: FAIL (`algua.audit.log` missing).

- [ ] **Step 3: Implement the audit log**

Create `algua/audit/__init__.py` (empty) and `algua/audit/log.py`:

```python
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime


def append(
    conn: sqlite3.Connection, *, actor: str, action: str, reason: str | None = None,
    strategy: str | None = None,
) -> None:
    conn.execute(
        "INSERT INTO audit_log(ts, actor, action, reason, strategy) VALUES (?,?,?,?,?)",
        (datetime.now(UTC).isoformat(), actor, action, reason, strategy),
    )
    conn.commit()


def read(conn: sqlite3.Connection, *, strategy: str | None = None) -> list[sqlite3.Row]:
    if strategy is None:
        return conn.execute("SELECT * FROM audit_log ORDER BY id").fetchall()
    return conn.execute(
        "SELECT * FROM audit_log WHERE strategy = ? ORDER BY id", (strategy,)
    ).fetchall()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_audit_log.py -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Quality gate + commit**

Run: `uv run ruff check . && uv run mypy algua && uv run pytest tests/test_audit_log.py -q`

```bash
git add algua/audit/__init__.py algua/audit/log.py tests/test_audit_log.py
git commit -m "feat(paper): lean append-only audit log"
```

---

### Task 6: CLI — `paper run` / `paper show` + end-to-end

**Files:**
- Create: `algua/cli/paper_cmd.py`
- Modify: `algua/cli/main.py` (register `paper_cmd`)
- Test: `tests/test_cli_paper.py`

- [ ] **Step 1: Write the failing e2e test**

Create `tests/test_cli_paper.py`:

```python
import json

import pytest
from typer.testing import CliRunner

from algua.cli.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "p.db"))
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))


def _to_paper(name="cross_sectional_momentum"):
    # idea -> backtested -> shortlisted -> paper
    assert runner.invoke(app, ["backtest", "run", name, "--demo", "--register",
                               "--start", "2022-01-01", "--end", "2023-12-31"]).exit_code == 0
    assert runner.invoke(app, ["registry", "transition", name, "--to", "shortlisted",
                               "--actor", "agent", "--reason", "ok"]).exit_code == 0
    assert runner.invoke(app, ["registry", "transition", name, "--to", "paper",
                               "--actor", "agent", "--reason", "paper"]).exit_code == 0


def test_paper_run_executes_and_reconciles():
    _to_paper()
    result = runner.invoke(app, ["paper", "run", "cross_sectional_momentum", "--demo",
                                 "--start", "2022-01-01", "--end", "2023-12-31"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["strategy"] == "cross_sectional_momentum"
    assert payload["reconcile_ok"] is True
    assert payload["orders"] >= 1
    # state persisted + visible via `paper show`
    show = json.loads(runner.invoke(app, ["paper", "show", "cross_sectional_momentum"]).stdout)
    assert show["n_orders"] >= 1


def test_paper_run_rejects_non_paper_stage():
    runner.invoke(app, ["registry", "add", "cross_sectional_momentum"])  # stage = idea
    result = runner.invoke(app, ["paper", "run", "cross_sectional_momentum", "--demo",
                                 "--start", "2022-01-01", "--end", "2023-12-31"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli_paper.py -q`
Expected: FAIL (no `paper` command).

- [ ] **Step 3: Implement the CLI**

Create `algua/cli/paper_cmd.py`:

```python
from __future__ import annotations

from contextlib import closing

import typer

from algua.audit.log import append as audit_append
from algua.backtest.engine import BacktestError
from algua.cli.app import app, emit
from algua.cli.backtest_cmd import _select_provider, _utc
from algua.cli.errors import json_errors
from algua.config.settings import get_settings
from algua.contracts.lifecycle import Stage
from algua.execution.order_state import derive_positions, persist_run
from algua.execution.sim_broker import SimBroker
from algua.live.paper_loop import run_paper
from algua.registry import store
from algua.registry.db import connect, migrate
from algua.strategies.loader import load_strategy

paper_app = typer.Typer(help="Paper trading: run a paper-stage strategy", no_args_is_help=True)
app.add_typer(paper_app, name="paper")


@paper_app.command("run")
@json_errors(ValueError, LookupError, BacktestError)
def run(
    name: str,
    start: str = typer.Option("2023-01-01", "--start"),
    end: str = typer.Option("2023-12-31", "--end"),
    demo: bool = typer.Option(False, "--demo", help="use the synthetic data provider"),
    snapshot: str = typer.Option(None, "--snapshot", help="paper-run an ingested bars snapshot"),  # noqa: B008
    cash: float = typer.Option(100_000.0, "--cash", help="starting paper cash"),
) -> None:
    """Replay a paper-stage strategy through the sim broker and persist orders/fills."""
    if cash <= 0:
        raise ValueError("--cash must be > 0")
    strategy = load_strategy(name)
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        rec = store.get_strategy(conn, name)
        if rec.stage is not Stage.PAPER:
            raise ValueError(f"{name} is at stage '{rec.stage.value}'; paper run requires 'paper'")
        provider = _select_provider(demo, snapshot)
        result = run_paper(strategy, SimBroker(cash=cash), provider, _utc(start), _utc(end))
        persist_run(conn, result)
        audit_append(conn, actor="agent", action="paper_run",
                     reason=f"{len(result.orders)} orders, {len(result.fills)} fills", strategy=name)

    emit({
        "strategy": result.strategy,
        "orders": len(result.orders),
        "fills": len(result.fills),
        "final_positions": result.final_positions,
        "final_cash": result.final_cash,
        "final_equity": result.final_equity,
        "reconcile_ok": result.reconcile_ok,
    })


@paper_app.command("show")
@json_errors(ValueError, LookupError)
def show(name: str) -> None:
    """Show persisted paper state (orders count + derived positions) for a strategy."""
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        n_orders = conn.execute(
            "SELECT COUNT(*) FROM paper_orders WHERE strategy = ?", (name,)
        ).fetchone()[0]
        positions = derive_positions(conn, name)
    emit({"strategy": name, "n_orders": n_orders, "positions": positions})
```

- [ ] **Step 4: Register the command in main.py**

Modify `algua/cli/main.py` — add `paper_cmd` to the registration import tuple (alphabetical, after `operator`-less list; place after `data_cmd`):

```python
from algua.cli import (  # noqa: F401 - imports register subcommands
    backtest_cmd,
    data_cmd,
    paper_cmd,
    registry_cmd,
    research_cmd,
    strategy_cmd,
)
```

- [ ] **Step 5: Run the e2e tests to verify they pass**

Run: `uv run pytest tests/test_cli_paper.py -q`
Expected: PASS (2 passed).

- [ ] **Step 6: Quality gate + commit**

Run: `uv run ruff check . && uv run mypy algua && uv run pytest -q`

```bash
git add algua/cli/paper_cmd.py algua/cli/main.py tests/test_cli_paper.py
git commit -m "feat(paper): `algua paper run` / `paper show` CLI + end-to-end"
```

---

### Task 7: Import boundaries + full gate

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add import-linter contracts**

Append to `pyproject.toml` after the existing contracts:

```toml
[[tool.importlinter.contracts]]
name = "execution / live / audit stay off the cli layer"
type = "forbidden"
source_modules = ["algua.execution", "algua.live", "algua.audit"]
forbidden_modules = ["algua.cli"]

[[tool.importlinter.contracts]]
name = "backtest engine stays off the execution and live lanes"
type = "forbidden"
source_modules = ["algua.backtest"]
forbidden_modules = ["algua.execution", "algua.live"]
```

- [ ] **Step 2: Verify the contracts hold**

Run: `uv run lint-imports`
Expected: `Contracts: 8 kept, 0 broken.` (If `algua.execution`/`live`/`audit` broke contract 1, a module imports `algua.cli` — invert the dependency. Note `algua.execution.order_state` importing `algua.live.paper_loop` is allowed; only `cli` is forbidden.)

- [ ] **Step 3: Run the full quality gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all pass (pytest green, ruff "All checks passed!", mypy "Success", import-linter "8 kept, 0 broken").

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "test(paper): enforce execution/live/audit import boundaries"
```

---

## Self-review notes

- **Spec coverage:** sim broker fills at t+1 open / sells-before-buys / cash-safe (§2,§4 → Task 2); replay loop reusing `target_weights`, weights decided ≤ t (§3 → Task 3); order-intent state + derive-from-fills + reconcile (§2,§3 → Task 4); audit log (§2 → Task 5); `paper run`/`show` + lifecycle gate + `t→t+1` e2e + negative (§4,§5 → Task 6); DB tables (§2 → Task 1); import boundaries (§2 → Task 7). The `t→t+1` invariant is covered by `test_fills_never_share_timestamp_with_their_decision_bar` (Task 3) and the SimBroker fill-at-next-open tests (Task 2).
- **Type consistency:** `OrderIntent(symbol, side, target_weight, decision_ts)`, `Fill(symbol, qty, price, decision_ts, fill_ts, broker_order_id)`, `PaperRunResult(...)`, and `SimBroker(cash)` signatures are identical across Tasks 2–6. `run_paper(strategy, broker, provider, start, end)` matches between Task 3 and the CLI in Task 6. Broker ids `sim-{seq}` are assigned identically in `SimBroker.submit` (Task 2) and re-derived in `persist_run` (Task 4).
- **No placeholders:** every code step is complete and runnable.
- **Note on reconcile scope:** `reconcile(derived, broker_positions)` is exercised in-run (`run_paper`, against the live `SimBroker`) and unit-tested in Task 4. `paper show` only reports persisted state (no live broker exists between runs); the cross-check against a real broker's positions is the seam the Alpaca adapter will exercise later.
