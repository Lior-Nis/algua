# Live Per-Strategy Sizing — Slice C1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a LIVE strategy size off `min(allocation, NAV)` (its ledger-backed virtual portfolio) and drawdown off its NAV instead of account equity, flatten per-strategy on a breach (single-shot offset of its believed qty), gate resume on ledger-flat, and add a minimal buying-power preflight — all under the existing ≤1-live guard (C2 lifts it).

**Architecture:** A new `algua/execution/live_sizing.py` builds a ledger-backed `SizingSnapshot` + NAV (fail-closed on missing marks). `run_tick` gains a `live_snapshot` hook so the live path injects that snapshot and drawdowns off NAV (paper unchanged). `_run_strategy_tick` wires the strategy's allocation + NAV peak + per-strategy liquidation; `live trade-tick` is removed (run-all is the sole reconcile-gated live path). Schema 14→15 (`live_nav_peaks`).

**Tech Stack:** Python 3.12, pandas, sqlite3, Typer, pytest, ruff, mypy, import-linter. Spec: `docs/superpowers/specs/2026-06-06-live-per-strategy-sizing-design.md`. Builds on Slices A/B (`live_ledger`, `live_reconcile`, `allocations`, `run-all`).

---

## File structure

| File | Responsibility |
|---|---|
| `algua/registry/db.py` (modify) | `live_nav_peaks` table; `SCHEMA_VERSION` 14→15. |
| `algua/execution/order_state.py` (modify) | `get_nav_peak`/`update_nav_peak`/`clear_nav_peak`. |
| `algua/execution/live_sizing.py` (new) | `SizingSnapshot` + `build_live_sizing_snapshot` (equity=min(allocation,NAV), believed×mark, fail-closed). |
| `algua/live/live_loop.py` (modify) | `TickHooks.live_snapshot`/`live_positions`; drawdown off NAV; ledger-aware early returns. |
| `algua/execution/alpaca_broker.py` (modify) | `submit_offset(symbol, signed_qty, coid)` (qty market order). |
| `algua/cli/live_cmd.py` (modify) | wire live sizing into `_run_strategy_tick`; per-strategy liquidation; BP preflight; remove `live trade-tick`. |
| `algua/cli/paper_cmd.py` (modify) | ledger-flat resume gate. |

---

### Task 1: NAV peak series (schema v15 + helpers)

**Files:** Modify `algua/registry/db.py`, `algua/execution/order_state.py`; Test `tests/test_order_state.py`.

Context: `order_state.py` has `get_peak_equity`/`update_peak_equity`/`clear_peak_equity` over the `strategy_peaks` table (account-equity peak). C1 needs a SEPARATE per-strategy NAV peak so the live drawdown breaker is NAV-based without overloading the account-equity series. Mirror the existing helpers exactly.

- [ ] **Step 1: Add the table.** In `algua/registry/db.py`, bump `SCHEMA_VERSION = 14` → `15` and append to `_SCHEMA`:
```sql
CREATE TABLE IF NOT EXISTS live_nav_peaks (
    strategy   TEXT PRIMARY KEY,
    peak       REAL NOT NULL,
    updated_ts TEXT NOT NULL
);
```

- [ ] **Step 2: Write the failing test.** Append to `tests/test_order_state.py` (read its imports/`_conn` helper first; it builds a conn via `connect`+`migrate`):
```python
def test_nav_peak_ratchets_and_clears(tmp_path):
    from algua.execution.order_state import clear_nav_peak, get_nav_peak, update_nav_peak
    conn = _conn(tmp_path)
    assert get_nav_peak(conn, "s1") is None
    assert update_nav_peak(conn, "s1", 10_000.0) == 10_000.0
    assert update_nav_peak(conn, "s1", 9_000.0) == 10_000.0   # only ratchets up
    assert update_nav_peak(conn, "s1", 11_000.0) == 11_000.0
    assert get_nav_peak(conn, "s1") == 11_000.0
    clear_nav_peak(conn, "s1")
    assert get_nav_peak(conn, "s1") is None
```
(If `tests/test_order_state.py` has no `_conn`, mirror whatever the other tests in it use to build a migrated connection.)

- [ ] **Step 3: Run** `cd /home/liornisimov/Projects/algua/.claude/worktrees/sp6-live-sizing && uv run pytest tests/test_order_state.py -k nav_peak -q` → FAIL.

- [ ] **Step 4: Implement** — append to `algua/execution/order_state.py` (mirrors `get/update/clear_peak_equity`):
```python
def get_nav_peak(conn: sqlite3.Connection, strategy: str) -> float | None:
    row = conn.execute(
        "SELECT peak FROM live_nav_peaks WHERE strategy = ?", (strategy,)
    ).fetchone()
    return float(row["peak"]) if row is not None else None


def update_nav_peak(conn: sqlite3.Connection, strategy: str, nav: float) -> float:
    """Persist the running per-strategy NAV peak (the live drawdown denominator) and return it.
    Ratchets up only; a tick's NAV below it is the drawdown the breaker acts on."""
    prior = get_nav_peak(conn, strategy)
    peak = nav if prior is None else max(prior, nav)
    conn.execute(
        "INSERT INTO live_nav_peaks(strategy, peak, updated_ts) VALUES (?,?,?) "
        "ON CONFLICT(strategy) DO UPDATE SET peak=excluded.peak, updated_ts=excluded.updated_ts",
        (strategy, peak, datetime.now(UTC).isoformat()),
    )
    conn.commit()
    return peak


def clear_nav_peak(conn: sqlite3.Connection, strategy: str) -> None:
    conn.execute("DELETE FROM live_nav_peaks WHERE strategy = ?", (strategy,))
    conn.commit()
```
(`sqlite3`, `datetime`/`UTC` are already imported in this file.)

- [ ] **Step 5: Run** the test → PASS.

- [ ] **Step 6: Gate + commit.** `uv run ruff check . && uv run mypy algua && uv run pytest tests/test_order_state.py -q && uv run lint-imports`
```bash
git add algua/registry/db.py algua/execution/order_state.py tests/test_order_state.py
git commit -m "feat(live-sizing): per-strategy NAV peak series (schema v15)"
```

---

### Task 2: `SizingSnapshot` + `build_live_sizing_snapshot`

**Files:** Create `algua/execution/live_sizing.py`; Test `tests/test_live_sizing.py`.

Context: the ledger-backed snapshot for a live strategy. `equity` (the sizing denominator) = `min(allocation, NAV)`; `qtys` = `believed_positions(strategy)` folded over `universe ∪ held`; `market_values` = `qty × latest-bar-close`. NAV = `allocation + Σ(realized+unrealized)` via `position_pnl`. Marks come from the tidy/long bars DataFrame (`symbol`, `close` columns; index `timestamp`): `bars.groupby("symbol")["close"].last()`. FAIL CLOSED: a held symbol (believed qty != 0) with no/zero/negative mark raises `LiveSizingError` (the loop skips the strategy) — never fall back to average cost. `believed_positions` and `position_pnl` are in `algua/execution/live_ledger.py`; `active_allocation` in `algua/registry/allocations.py`.

- [ ] **Step 1: Write the failing tests.** Create `tests/test_live_sizing.py`:
```python
import pandas as pd
import pytest

from algua.execution import live_sizing as S
from algua.registry.db import connect, migrate


def _conn(tmp_path):
    conn = connect(tmp_path / "s.db")
    migrate(conn)
    return conn


def _fill(conn, aid, strategy, symbol, qty, price):
    conn.execute(
        "INSERT INTO live_fills(activity_id, broker_order_id, strategy, symbol, qty, price, fill_ts)"
        " VALUES (?,?,?,?,?,?,?)",
        (aid, "b", strategy, symbol, qty, price, "2026-06-06T00:00:00+00:00"),
    )
    conn.commit()


def _bars(close_by_symbol):
    rows = []
    for sym, closes in close_by_symbol.items():
        for i, c in enumerate(closes):
            rows.append({"timestamp": pd.Timestamp("2026-06-01", tz="UTC") + pd.Timedelta(days=i),
                         "symbol": sym, "open": c, "high": c, "low": c, "close": c, "volume": 1.0})
    return pd.DataFrame(rows).set_index("timestamp").sort_index()


def test_sizing_equity_is_allocation_when_nav_above(tmp_path):
    conn = _conn(tmp_path)
    _fill(conn, "a1", "s1", "AAA", 10.0, 100.0)         # long 10 @100
    bars = _bars({"AAA": [100.0, 110.0]})                # mark 110 -> unrealized +100 -> NAV 10_100
    snap, nav = S.build_live_sizing_snapshot(conn, "s1", allocation=10_000.0, bars=bars,
                                             universe=["AAA"])
    assert nav == 10_100.0
    assert snap.equity == 10_000.0                       # min(allocation, NAV) = allocation
    assert snap.qtys["AAA"] == 10.0
    assert snap.market_values["AAA"] == 10.0 * 110.0


def test_sizing_equity_derisks_when_nav_below_allocation(tmp_path):
    conn = _conn(tmp_path)
    _fill(conn, "a1", "s1", "AAA", 10.0, 100.0)
    bars = _bars({"AAA": [100.0, 50.0]})                 # mark 50 -> unrealized -500 -> NAV 9_500
    snap, nav = S.build_live_sizing_snapshot(conn, "s1", allocation=10_000.0, bars=bars,
                                             universe=["AAA"])
    assert nav == 9_500.0
    assert snap.equity == 9_500.0                        # min(allocation, NAV) = NAV (de-risked)


def test_universe_symbol_with_no_position_is_flat(tmp_path):
    conn = _conn(tmp_path)
    bars = _bars({"AAA": [100.0], "BBB": [50.0]})
    snap, nav = S.build_live_sizing_snapshot(conn, "s1", allocation=10_000.0, bars=bars,
                                             universe=["AAA", "BBB"])
    assert nav == 10_000.0
    assert snap.qtys == {"AAA": 0.0, "BBB": 0.0}
    assert snap.market_values == {"AAA": 0.0, "BBB": 0.0}


def test_held_symbol_missing_mark_fails_closed(tmp_path):
    conn = _conn(tmp_path)
    _fill(conn, "a1", "s1", "ZZZ", 5.0, 10.0)            # held ZZZ, but no bar for ZZZ
    bars = _bars({"AAA": [100.0]})
    with pytest.raises(S.LiveSizingError, match="mark"):
        S.build_live_sizing_snapshot(conn, "s1", allocation=10_000.0, bars=bars, universe=["AAA"])
```

- [ ] **Step 2: Run** `uv run pytest tests/test_live_sizing.py -q` → FAIL.

- [ ] **Step 3: Implement** `algua/execution/live_sizing.py`:
```python
"""The ledger-backed sizing view for a LIVE strategy (its virtual subaccount). Equity is the SIZING
denominator = min(allocation, NAV); NAV (allocation + realized + unrealized) is the drawdown basis.
Marks are the latest closed bar; a held symbol with no usable mark FAILS CLOSED (the loop skips the
strategy) rather than falling back to average cost — which would hide a loss and suppress the
drawdown breaker."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import pandas as pd

from algua.execution.live_ledger import believed_positions, position_pnl


class LiveSizingError(ValueError):
    """A live strategy cannot be sized this tick (e.g. a held symbol has no usable mark)."""


@dataclass(frozen=True)
class SizingSnapshot:
    """Ledger belief (NOT broker truth — that's TickSnapshot). Same fields run_tick reads:
    equity is the sizing denominator, market_values/qtys are this strategy's believed book."""

    equity: float
    market_values: dict[str, float]
    qtys: dict[str, float]


def _latest_marks(bars: pd.DataFrame) -> dict[str, float]:
    if bars.empty:
        return {}
    return {str(sym): float(c) for sym, c in bars.groupby("symbol")["close"].last().items()}


def build_live_sizing_snapshot(
    conn: sqlite3.Connection, strategy: str, allocation: float, bars: pd.DataFrame,
    universe: list[str],
) -> tuple[SizingSnapshot, float]:
    held = believed_positions(conn, strategy)          # {symbol: signed qty}, nonzero only
    marks = _latest_marks(bars)
    symbols = set(universe) | set(held)

    nav = allocation
    market_values: dict[str, float] = {}
    qtys: dict[str, float] = {}
    for sym in symbols:
        qty = held.get(sym, 0.0)
        qtys[sym] = qty
        mark = marks.get(sym)
        if qty != 0.0 and (mark is None or mark <= 0.0):
            raise LiveSizingError(
                f"{strategy}: held symbol {sym!r} has no usable mark (got {mark!r}) — refusing to "
                "size on a fail-closed mark"
            )
        market_values[sym] = qty * (mark or 0.0)
        if qty != 0.0:
            fills = [(float(r["qty"]), float(r["price"])) for r in conn.execute(
                "SELECT qty, price FROM live_fills WHERE strategy = ? AND symbol = ? "
                "ORDER BY fill_ts, id", (strategy, sym))]
            pnl = position_pnl(fills, mark=mark)         # mark is positive here
            nav += pnl.realized + pnl.unrealized

    return SizingSnapshot(equity=min(allocation, nav), market_values=market_values, qtys=qtys), nav
```

- [ ] **Step 4: Run** the test → PASS.

- [ ] **Step 5: Gate + commit.** `uv run ruff check . && uv run mypy algua && uv run pytest tests/test_live_sizing.py -q && uv run lint-imports`
```bash
git add algua/execution/live_sizing.py tests/test_live_sizing.py
git commit -m "feat(live-sizing): ledger-backed SizingSnapshot (min(allocation,NAV), fail-closed marks)"
```

---

### Task 3: the `run_tick` live-snapshot seam

**Files:** Modify `algua/live/live_loop.py`; Test `tests/test_live_loop.py`.

Context: `run_tick` calls `snap = broker.snapshot(universe)` and uses `snap.equity` for BOTH sizing and drawdown. Add `TickHooks.live_snapshot(bars) -> (snapshot, nav)`: when supplied, build the snapshot from the hook and drawdown against the returned NAV (decoupled from the sizing denominator). Add `TickHooks.live_positions() -> dict` for the two early-return paths (empty bars / warmup) so live reports LEDGER positions, not `broker.get_positions()`. Paper passes neither → unchanged. `_positions(broker)` and the early returns are near the top of `run_tick`; the snapshot/drawdown block is the `snap = broker.snapshot(...)` / `check_drawdown(snap.equity, ...)` lines; the final `TickResult(...)` builds `equity=snap.equity, peak_equity=peak`.

- [ ] **Step 1: Write the failing tests.** Append to `tests/test_live_loop.py`:
```python
def test_run_tick_live_snapshot_sizes_off_hook_and_drawdowns_off_nav(monkeypatch):
    from algua.execution.alpaca_broker import TickSnapshot
    from algua.live.live_loop import RiskBreach, TickHooks
    broker = _FakeBroker()
    bars = _bars({"AAA": [100.0, 100.0, 100.0]})
    # ledger snapshot: allocation 10k as equity, flat; NAV 6k well below a 10k peak -> drawdown trips
    snap = TickSnapshot(equity=10_000.0, market_values={"AAA": 0.0}, qtys={"AAA": 0.0})
    hooks = TickHooks(live_snapshot=lambda b: (snap, 6_000.0), peak_equity=10_000.0)
    with pytest.raises(RiskBreach):  # NAV 6k vs peak 10k = 40% drawdown
        run_tick(_strategy({"AAA": 0.5}), broker, _FakeProvider(bars), DATES[0], DATES[-1],
                 hooks=hooks, max_drawdown=0.2)


def test_run_tick_live_positions_on_warmup_early_return():
    from algua.live.live_loop import TickHooks
    broker = _FakeBroker()
    bars = _bars({"AAA": [100.0]})  # 1 bar; warmup default likely holds it flat -> early return
    hooks = TickHooks(live_positions=lambda: {"AAA": 7.0})
    res = run_tick(_strategy({"AAA": 0.5}, warmup_bars=5), broker, _FakeProvider(bars),
                   DATES[0], DATES[-1], hooks=hooks)
    assert res.positions_before == {"AAA": 7.0}  # ledger positions, not broker's
```
(Adjust `_strategy(..., warmup_bars=5)` to however `_strategy` accepts warmup in this file; the intent: a warmup early-return reports `live_positions()` when supplied.)

- [ ] **Step 2: Run** `uv run pytest tests/test_live_loop.py -k "live_snapshot or live_positions" -q` → FAIL.

- [ ] **Step 3: Implement** in `algua/live/live_loop.py`:
  - Add to `TickHooks` (after `cancel`):
```python
    live_snapshot: Callable[[Any], tuple[Any, float]] | None = None
    live_positions: Callable[[], dict[str, float]] | None = None
```
  with docstring lines: `live_snapshot(bars) -> (SizingSnapshot, nav)` supplies the ledger-backed sizing snapshot + NAV (live); when set, sizing is off the snapshot equity and drawdown off NAV. `live_positions()` supplies ledger positions for the no-decision early-return paths.
  - Replace each early-return `_positions(broker)` with `_early_positions(hooks, broker)`, and add the helper:
```python
def _early_positions(hooks: TickHooks, broker: _AlpacaBroker) -> dict[str, float]:
    return hooks.live_positions() if hooks.live_positions is not None else _positions(broker)
```
  - Replace the snapshot + drawdown block:
```python
    if hooks.live_snapshot is not None:
        snap, drawdown_equity = hooks.live_snapshot(bars)
    else:
        snap = broker.snapshot(strategy.universe)
        drawdown_equity = snap.equity

    peak = drawdown_equity if hooks.peak_equity is None else max(hooks.peak_equity, drawdown_equity)
    check_drawdown(drawdown_equity, peak, max_drawdown)
```
  - In the FINAL `TickResult(...)`, set `equity=drawdown_equity` and `peak_equity=peak` (so live persists NAV; paper's `drawdown_equity == snap.equity`, unchanged). (`Any` is already imported.)

- [ ] **Step 4: Run** the tests → PASS, then `uv run pytest tests/test_live_loop.py -q` (paper path unchanged when no hooks).

- [ ] **Step 5: Gate + commit.** `uv run ruff check . && uv run mypy algua && uv run pytest tests/test_live_loop.py -q && uv run lint-imports`
```bash
git add algua/live/live_loop.py tests/test_live_loop.py
git commit -m "feat(live-sizing): run_tick live_snapshot seam (sizing off ledger snapshot, drawdown off NAV)"
```

---

### Task 4: broker `submit_offset` (qty market order for liquidation)

**Files:** Modify `algua/execution/alpaca_broker.py`; Test `tests/test_alpaca_broker.py`.

Context: per-strategy liquidation submits a market order for an ABSOLUTE share qty (the strategy's believed position), not a notional/weight. Add `submit_offset(symbol, signed_qty, client_order_id)` → POST `/v2/orders` with `qty=abs(signed_qty)`, `side="sell"` if `signed_qty>0` (close a long) else `"buy"` (cover a short). `_FakeRequests` records POSTs in `.posted` (read it).

- [ ] **Step 1: Write the failing test.** Append to `tests/test_alpaca_broker.py`:
```python
def test_submit_offset_posts_qty_order(monkeypatch):
    fake = _FakeRequests({}, post_resp=_FakeResp(201, {"id": "off-1"}))
    monkeypatch.setattr(ab, "requests", fake)
    oid = _broker().submit_offset("AAA", 7.0, "coid-flat")      # long 7 -> SELL 7
    assert oid == "off-1"
    assert fake.posted[0] == {"symbol": "AAA", "qty": "7", "side": "sell", "type": "market",
                              "time_in_force": "day", "client_order_id": "coid-flat"}
    _broker().submit_offset("BBB", -3.0, "coid-cover")          # short 3 -> BUY 3
    assert fake.posted[1]["side"] == "buy" and fake.posted[1]["qty"] == "3"
```

- [ ] **Step 2: Run** `uv run pytest tests/test_alpaca_broker.py::test_submit_offset_posts_qty_order -q` → FAIL.

- [ ] **Step 3: Implement** — add to `_AlpacaBroker`:
```python
    def submit_offset(self, symbol: str, signed_qty: float, client_order_id: str) -> str:
        """Submit a market order to OFFSET a believed position: sell `signed_qty` shares if long
        (signed_qty>0), buy them back if short (<0). Used by per-strategy liquidation — sized to the
        strategy's ledger qty so the account net moves by exactly this strategy's contribution."""
        qty = abs(signed_qty)
        if qty == 0.0:
            return "noop"
        body: dict[str, Any] = {
            "symbol": symbol, "qty": format(qty, "g"),
            "side": "sell" if signed_qty > 0 else "buy",
            "type": "market", "time_in_force": "day", "client_order_id": client_order_id,
        }
        data = self._read(self._post("/v2/orders", body), "/v2/orders", ok=(200, 201))
        order_id = data.get("id") if isinstance(data, dict) else None
        if not order_id:
            raise BrokerError(f"alpaca /v2/orders (offset): response missing 'id': {data}")
        return str(order_id)
```

- [ ] **Step 4: Run** the test → PASS.

- [ ] **Step 5: Gate + commit.** `uv run ruff check . && uv run mypy algua && uv run pytest tests/test_alpaca_broker.py -q && uv run lint-imports`
```bash
git add algua/execution/alpaca_broker.py tests/test_alpaca_broker.py
git commit -m "feat(live-sizing): broker submit_offset (qty market order for per-strategy liquidation)"
```

---

### Task 5: wire live sizing into `_run_strategy_tick` + per-strategy liquidation + remove `live trade-tick`

**Files:** Modify `algua/cli/live_cmd.py`; Test `tests/test_cli_live.py`.

Context: `_run_strategy_tick(conn, name, authorization, broker, provider, max_drawdown, ..., cancel=None)` currently sizes via `broker.snapshot` (account equity) and on `RiskBreach` does account-wide `cancel_open_orders` + `close_positions(universe)`. Rewire it for live: look up the allocation, inject `live_snapshot`/`live_positions`, use the NAV peak, add a minimal BP preflight, and replace the breach flatten with the per-strategy liquidation. Then DELETE the `live trade-tick` command (run-all is the sole reconcile-gated live path, Codex #6) and its now-obsolete single-strategy tests. Helpers available: `believed_positions`, `ingest_activities`, `fill_cursor`, `_broker_account_activities`, `_scoped_cancel`, `client_order_id`, `build_live_sizing_snapshot`, `active_allocation`, `get_nav_peak`/`update_nav_peak`, `broker.account()` (has `.buying_power`), `broker.submit_offset`.

- [ ] **Step 1: Write/adjust the tests.** In `tests/test_cli_live.py`: DELETE the five `test_live_trade_tick_*` tests (the command is removed). Add a run-all breach + liquidation test:
```python
def test_run_all_breach_liquidates_per_strategy(monkeypatch):
    from algua.live.live_loop import RiskBreach
    _to_live()
    monkeypatch.setattr("algua.cli.live_cmd.verify_live_authorization", lambda *a, **k: _auth())

    class _LiqBroker:
        def __init__(self):
            self.offsets = []
        def account_activities(self, after=None):
            return []
        def get_positions(self):
            import pandas as pd
            return pd.Series(dtype=float)
        def list_open_orders(self):
            return []
        def cancel_order(self, oid):
            pass
        def submit_offset(self, symbol, qty, coid):
            self.offsets.append((symbol, qty)); return "off"

    broker = _LiqBroker()
    monkeypatch.setattr("algua.cli.live_cmd._alpaca_live_broker", lambda auth: broker)
    monkeypatch.setattr("algua.cli.live_cmd._select_provider", lambda demo, snapshot: object())
    monkeypatch.setattr("algua.cli.live_cmd._broker_account_activities", lambda b, a: [])
    monkeypatch.setattr("algua.cli.live_cmd._broker_net_positions", lambda b: {})
    monkeypatch.setattr("algua.cli.live_cmd.believed_positions",
                        lambda conn, name: {"AAA": 5.0})  # strategy believes it holds 5 AAA
    monkeypatch.setattr("algua.cli.live_cmd.run_tick",
                        lambda *a, **k: (_ for _ in ()).throw(RiskBreach("drawdown", "dd")))
    r = runner.invoke(app, ["live", "run-all", "--snapshot", "x"])
    assert r.exit_code == 1
    assert broker.offsets == [("AAA", 5.0)]  # offset sized to the believed qty
```

- [ ] **Step 2: Run** `uv run pytest tests/test_cli_live.py -q` → FAIL (command gone / liquidation differs).

- [ ] **Step 3: Implement.** In `algua/cli/live_cmd.py`:
  - Add imports: `from algua.execution.live_sizing import LiveSizingError, build_live_sizing_snapshot`, `from algua.execution.live_ledger import believed_positions` (extend the existing live_ledger import), `from algua.execution.order_state import get_nav_peak, update_nav_peak` (extend the existing order_state import), `from algua.registry.allocations import active_allocation`.
  - In `_run_strategy_tick`, after `strategy = load_strategy(name)`, look up the allocation + BP preflight + build the hooks:
```python
    alloc = active_allocation(conn, SqliteStrategyRepository(conn).get(name).id)
    if alloc is None:
        raise ValueError(f"{name} has no live allocation")
    allocation = float(alloc["capital"])
    if allocation > broker.account().buying_power:           # minimal C1 BP preflight (codex #8)
        raise ValueError(f"{name}: allocation {allocation} exceeds account buying power")

    def _live_snap(bars):
        return build_live_sizing_snapshot(conn, name, allocation, bars, strategy.universe)

    hooks = TickHooks(
        client_order_id_for=client_order_id, on_submitted=_persist, cancel=cancel,
        live_snapshot=_live_snap, live_positions=lambda: believed_positions(conn, name),
        should_halt=lambda: (kill_switch.is_tripped(conn, name) or global_halt.is_engaged(conn)
                             or not authorization_active(conn, authorization)),
        peak_equity=get_nav_peak(conn, name),
    )
```
  (remove the old `derived_positions=None` / `peak_equity=get_peak_equity(...)` lines; `_persist` stays as today.)
  - Replace the `except RiskBreach` flatten body with the per-strategy liquidation:
```python
        except RiskBreach as exc:
            _trip(conn, name, exc)
            liquidation_submitted = True
            flatten_error = None
            try:
                _scoped_cancel(conn, broker, name)                       # cancel only our orders
                ingest_activities(conn, _broker_account_activities(broker, fill_cursor(conn)))
                for sym, qty in believed_positions(conn, name).items():  # fresh believed qty
                    broker.submit_offset(sym, qty, client_order_id(name, datetime.now(UTC), sym))
            except BrokerError as fexc:
                liquidation_submitted = False
                flatten_error = str(fexc)
                audit_append(conn, actor="system", action="flatten_failed",
                             reason=str(fexc), strategy=name)
            payload = _breach_payload(exc.detail, kind=exc.kind,
                                      liquidation_submitted=liquidation_submitted)
            if flatten_error is not None:
                payload["flatten_error"] = flatten_error
            emit(payload)
            raise typer.Exit(1) from exc
        except LiveSizingError as exc:        # fail-closed mark -> skip this strategy, don't trade
            audit_append(conn, actor="system", action="live_sizing_skipped",
                         reason=str(exc), strategy=name)
            return {"strategy": name, "skipped": str(exc)}
```
  - On the success path, replace `update_peak_equity(...)` with `update_nav_peak(conn, name, result.peak_equity)` (result.peak_equity is now the NAV peak). Keep `record_tick_snapshot(...)` (its `equity=result.equity` is now the NAV).
  - DELETE the entire `@live_app.command("trade-tick")` function (`trade_tick`). Keep `_run_strategy_tick` (run-all calls it).

- [ ] **Step 4: Run** `uv run pytest tests/test_cli_live.py -q` → PASS.

- [ ] **Step 5: Gate + commit.** `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports` (all green).
```bash
git add algua/cli/live_cmd.py tests/test_cli_live.py
git commit -m "feat(live-sizing): per-strategy sizing+NAV-drawdown+liquidation in run-all; remove live trade-tick"
```

---

### Task 6: ledger-flat resume gate

**Files:** Modify `algua/cli/paper_cmd.py`; Test `tests/test_cli_paper.py`.

Context: a single-shot liquidation can leave a residual on a partial fill (Codex HIGH #5); reconcile won't catch it (ledger == broker on the residual). So `paper resume` / `resume-all` must refuse to resume a LIVE strategy whose `believed_positions` is non-empty (it isn't actually flat). `paper resume` is at `algua/cli/paper_cmd.py` (`@paper_app.command("resume")`); it calls `kill_switch.reset(conn, name)`. `believed_positions` is in `algua/execution/live_ledger.py`; a strategy's stage is `SqliteStrategyRepository(conn).get(name).stage` (`Stage.LIVE`).

- [ ] **Step 1: Write the failing test.** Append to `tests/test_cli_paper.py` (read its helpers for bringing a strategy to a killed live state; mirror them — a strategy at `Stage.LIVE`, kill-switch tripped, with a non-empty `live_fills` belief):
```python
def test_resume_refused_while_live_strategy_not_flat(monkeypatch, tmp_path):
    name = _seed_live_killed_with_position(monkeypatch, tmp_path, "s1")  # local helper
    r = runner.invoke(app, ["paper", "resume", name])
    assert r.exit_code == 1 and "not flat" in r.stdout.lower()
    # once flat (belief cleared), resume succeeds
    _clear_belief(tmp_path, name)
    assert runner.invoke(app, ["paper", "resume", name]).exit_code == 0
```
(Write `_seed_live_killed_with_position` / `_clear_belief` mirroring the file's existing setup: bring a strategy to `live` stage, trip its kill-switch, insert a `live_fills` row; `_clear_belief` deletes it. If `tests/test_cli_paper.py` lacks suitable helpers, build the state directly via `connect`+`migrate` on `get_settings().db_path`.)

- [ ] **Step 2: Run** `uv run pytest tests/test_cli_paper.py -k resume_refused_while_live -q` → FAIL.

- [ ] **Step 3: Implement.** In `algua/cli/paper_cmd.py`'s `resume` command, before `kill_switch.reset(conn, name)`, add the gate:
```python
        from algua.contracts.lifecycle import Stage
        from algua.execution.live_ledger import believed_positions
        rec = SqliteStrategyRepository(conn).get(name)
        if rec.stage is Stage.LIVE and believed_positions(conn, name):
            raise ValueError(
                f"{name} is not flat (believed positions {believed_positions(conn, name)}); "
                "re-flatten before resuming a live strategy")
```
(Use the module's existing `SqliteStrategyRepository`/imports; the command already raises `ValueError` rendered as `{ok:false}` via its error decorator. Apply the same guard in `resume_all` if it resumes individual live strategies — for `resume_all`, skip + flag any non-flat live strategy rather than aborting the whole batch.)

- [ ] **Step 4: Run** the test → PASS.

- [ ] **Step 5: FULL gate + commit.** `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports` (all green; contracts 0 broken).
```bash
git add algua/cli/paper_cmd.py tests/test_cli_paper.py
git commit -m "feat(live-sizing): ledger-flat resume gate (a live strategy must be flat to resume)"
```

---

## Self-review notes

- **Spec coverage (C1 in §3/§4/§6):** SizingSnapshot + build_live_sizing_snapshot with min(allocation,NAV) + fail-closed marks (Task 2); the run_tick live_snapshot seam + drawdown off NAV + ledger-aware early returns (Task 3); per-strategy NAV peak series (Task 1, wired Task 5); _run_strategy_tick sizes off allocation + NAV peak + BP preflight + per-strategy liquidation via submit_offset (Tasks 4-5); live routes only through run-all / `trade-tick` removed (Task 5); ledger-flat resume gate (Task 6). The ≤1-live guard is UNTOUCHED (C2 lifts it). ✓
- **Codex findings folded in:** min(allocation,NAV) (#2 → Task 2); fail-closed marks (#7 → Task 2); SizingSnapshot distinct type + ledger early returns (#10 → Tasks 2-3); single-strategy BP preflight (#8 → Task 5); live only through run-all (#6 → Task 5); ledger-flat resume gate (#5 → Task 6). The virtual-subaccount model makes the believed-qty offset (Task 5) move the account net by exactly the strategy's contribution (#1/#4). Gross BP pool + guard lift are C2 (not this plan).
- **Type consistency:** `build_live_sizing_snapshot(conn, strategy, allocation, bars, universe) -> (SizingSnapshot, nav)` (Task 2) is the `live_snapshot` hook return shape (Task 3) and is called by `_live_snap` (Task 5); `submit_offset(symbol, signed_qty, coid)` (Task 4) called in the liquidation loop (Task 5); `get_nav_peak`/`update_nav_peak` (Task 1) used in Task 5; `believed_positions` reused across Tasks 5-6.
- **No placeholders:** the few "mirror the file's existing helpers" notes (Tasks 1, 6 test setup) are instructions to match real fixtures the implementer must read; the assertion intent + the production code are fully specified. Task 5 deletes a named existing function (`trade_tick`) and its named tests — an explicit removal, not vague.
- **Paper-preserving:** Task 3's seam is inert without `live_snapshot`/`live_positions` (paper passes neither → `broker.snapshot` + equity for both, `drawdown_equity == snap.equity`), so the existing paper/live_loop tests hold.
