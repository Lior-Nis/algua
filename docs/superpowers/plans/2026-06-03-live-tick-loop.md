# Wall-Clock Live Tick Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A single wall-clock tick (`algua paper trade-live <name> --snapshot <id>`) that drives `AlpacaPaperBroker` with Alpaca as the source of truth: compute target weights from the latest closed session, cancel stale orders, submit market-order deltas, record + audit.

**Architecture:** `run_tick` is pure orchestration over an injected `AlpacaPaperBroker` + snapshot bar provider. Reuses `risk/limits` (after extracting a shared `check_long_only`), `kill_switch`, and `_select_provider`'s snapshot path. No local position ledger — re-reads Alpaca each tick. Auto-flatten, global switch, drawdown breaker = B2b.

**Tech Stack:** Python 3.12, requests, pandas, Typer, sqlite3, pytest, ruff, mypy, import-linter. Spec: `docs/superpowers/specs/2026-06-03-live-tick-loop-design.md`.

---

## File structure

| File | Responsibility |
|---|---|
| `algua/risk/limits.py` (modify) | Add `check_long_only(weights, strategy_name)`. |
| `algua/live/paper_loop.py` (modify) | Use `check_long_only` instead of the inline block. |
| `algua/execution/alpaca_broker.py` (modify) | Add `cancel_open_orders()` (`DELETE /v2/orders`). |
| `algua/live/live_loop.py` (new) | `TickResult` + `run_tick`. |
| `algua/cli/paper_cmd.py` (modify) | `paper trade-live` command. |

---

### Task 1: Extract `check_long_only`

**Files:** Modify `algua/risk/limits.py`, `algua/live/paper_loop.py`; Test `tests/test_risk_limits.py`.

- [ ] **Step 1: Add failing test** — append to `tests/test_risk_limits.py`:

```python
def test_check_long_only_passes_and_raises():
    from algua.risk.limits import check_long_only

    check_long_only(pd.Series({"AAA": 0.6, "BBB": 0.4}), "s")  # ok
    check_long_only(pd.Series(dtype="float64"), "s")           # empty ok
    with pytest.raises(RiskBreach) as ei:
        check_long_only(pd.Series({"AAA": -0.5}), "s")
    assert ei.value.kind == "long_only"
```

- [ ] **Step 2: Run** `uv run pytest tests/test_risk_limits.py -q` → FAIL.

- [ ] **Step 3: Implement** — in `algua/risk/limits.py`, add after `check_gross_exposure`:

```python
def check_long_only(weights: pd.Series, strategy_name: str) -> None:
    if len(weights) and bool((weights < 0).any()):
        negative = sorted(weights[weights < 0].index)
        raise RiskBreach(
            "long_only",
            f"long-only: strategy '{strategy_name}' returned negative target weight(s) "
            f"for {negative}",
        )
```

- [ ] **Step 4: Use it in `run_paper`** — in `algua/live/paper_loop.py`, replace the inline long-only block:

```python
        weights = strategy.target_weights(view)
        if len(weights) and bool((weights < 0).any()):
            negative = sorted(weights[weights < 0].index)
            raise RiskBreach(
                "long_only",
                f"long-only: strategy '{strategy.name}' returned negative target weight(s) "
                f"for {negative} at {t}",
            )
        check_gross_exposure(weights, max_gross)
```

with:

```python
        weights = strategy.target_weights(view)
        check_long_only(weights, strategy.name)
        check_gross_exposure(weights, max_gross)
```

and update the import line to add `check_long_only`:

```python
from algua.risk.limits import RiskBreach, check_drawdown, check_gross_exposure, check_long_only
```

(`RiskBreach` may now be unused in `paper_loop.py` — if ruff flags it, drop it from the import.)

- [ ] **Step 5: Run** `uv run pytest tests/test_risk_limits.py tests/test_paper_loop.py -q` → PASS (the existing `test_run_paper_rejects_negative_weights_long_only` still passes; its message still contains "long-only").

- [ ] **Step 6: Gate + commit** — `uv run ruff check . && uv run mypy algua && uv run pytest -q`

```bash
git add algua/risk/limits.py algua/live/paper_loop.py tests/test_risk_limits.py
git commit -m "refactor(risk): extract shared check_long_only used by run_paper + live tick"
```

---

### Task 2: `cancel_open_orders` on the adapter

**Files:** Modify `algua/execution/alpaca_broker.py`; Test `tests/test_alpaca_broker.py`.

- [ ] **Step 1: Add failing test** — append to `tests/test_alpaca_broker.py` (the `_FakeRequests` there only handles get/post; add `delete` support inline via a subclass):

```python
class _FakeRequestsWithDelete(_FakeRequests):
    def __init__(self, delete_resp):
        super().__init__({})
        self._delete_resp = delete_resp
        self.deleted = []

    def delete(self, url, headers=None, timeout=None):
        self.deleted.append(url)
        return self._delete_resp


def test_cancel_open_orders_ok(monkeypatch):
    fake = _FakeRequestsWithDelete(_FakeResp(207, []))
    monkeypatch.setattr(ab, "requests", fake)
    _broker().cancel_open_orders()
    assert fake.deleted == ["https://paper-api.alpaca.markets/v2/orders"]


def test_cancel_open_orders_non_2xx_raises(monkeypatch):
    monkeypatch.setattr(ab, "requests", _FakeRequestsWithDelete(_FakeResp(500, text="boom")))
    with pytest.raises(BrokerError):
        _broker().cancel_open_orders()
```

- [ ] **Step 2: Run** `uv run pytest tests/test_alpaca_broker.py -q` → FAIL.

- [ ] **Step 3: Implement** — in `algua/execution/alpaca_broker.py`, add a `_delete` helper next to `_get`/`_post`:

```python
    def _delete(self, path: str) -> requests.Response:
        try:
            return requests.delete(f"{self.base_url}{path}", headers=self._headers(),
                                   timeout=_TIMEOUT)
        except RequestException as exc:
            raise BrokerError(f"alpaca DELETE {path} failed: {exc}") from exc
```

and add the public method (after `get_positions`):

```python
    def cancel_open_orders(self) -> None:
        """Cancel all open orders (DELETE /v2/orders). Alpaca returns 207 multi-status with
        per-order results; we only require an overall-success status."""
        resp = self._delete("/v2/orders")
        if resp.status_code not in (200, 207):
            raise BrokerError(f"alpaca {resp.status_code} on /v2/orders: {resp.text}")
```

- [ ] **Step 4: Run** `uv run pytest tests/test_alpaca_broker.py -q` → PASS.

- [ ] **Step 5: Gate + commit** — `uv run ruff check . && uv run mypy algua && uv run pytest tests/test_alpaca_broker.py -q`

```bash
git add algua/execution/alpaca_broker.py tests/test_alpaca_broker.py
git commit -m "feat(broker): AlpacaPaperBroker.cancel_open_orders (clear stale orders)"
```

---

### Task 3: `run_tick` (the live loop)

**Files:** Create `algua/live/live_loop.py`; Test `tests/test_live_loop.py`.

- [ ] **Step 1: Add failing test** — create `tests/test_live_loop.py`:

```python
from datetime import UTC, datetime

import pandas as pd
import pytest

from algua.contracts.types import ExecutionContract
from algua.live.live_loop import run_tick
from algua.risk.limits import RiskBreach
from algua.strategies.base import LoadedStrategy, StrategyConfig

DATES = [datetime(2023, 1, d, tzinfo=UTC) for d in (2, 3, 4)]


def _bars(symbol_prices):
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


class _FakeBroker:
    def __init__(self, positions=None):
        self._positions = pd.Series(positions or {}, dtype="float64")
        self.submitted = []
        self.cancels = 0

    def get_positions(self):
        return self._positions

    def cancel_open_orders(self):
        self.cancels += 1

    def submit(self, intent):
        self.submitted.append(intent)
        return f"order-{len(self.submitted)}"


def _strategy(weights, warmup_bars=0):
    cfg = StrategyConfig(
        name="cfg", universe=sorted(weights),
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1,
                                     warmup_bars=warmup_bars),
    )
    return LoadedStrategy(config=cfg, fn=lambda view, params: pd.Series(weights))


def test_run_tick_submits_target_and_cancels_first():
    broker = _FakeBroker()
    bars = _bars({"AAA": [100.0, 100.0, 100.0]})
    result = run_tick(_strategy({"AAA": 1.0}), broker, _FakeProvider(bars), DATES[0], DATES[-1])
    assert broker.cancels == 1
    assert len(result.submitted) == 1 and result.submitted[0]["symbol"] == "AAA"
    assert result.submitted[0]["order_id"] == "order-1"
    assert result.decision_ts == DATES[-1]


def test_run_tick_exits_dropped_symbol():
    broker = _FakeBroker(positions={"BBB": 10.0})  # held but not in target
    bars = _bars({"AAA": [100.0, 100.0, 100.0], "BBB": [50.0, 50.0, 50.0]})
    result = run_tick(_strategy({"AAA": 1.0}), broker, _FakeProvider(bars), DATES[0], DATES[-1])
    syms = {o["symbol"]: o["target_weight"] for o in result.submitted}
    assert syms["BBB"] == 0.0  # exit order for the dropped name


def test_run_tick_warmup_not_met_submits_nothing():
    broker = _FakeBroker()
    bars = _bars({"AAA": [100.0, 100.0, 100.0]})  # 3 sessions
    result = run_tick(_strategy({"AAA": 1.0}, warmup_bars=5), broker, _FakeProvider(bars),
                      DATES[0], DATES[-1])
    assert result.submitted == [] and broker.submitted == []


def test_run_tick_gross_breach_raises():
    broker = _FakeBroker()
    bars = _bars({"AAA": [100.0, 100.0, 100.0], "BBB": [100.0, 100.0, 100.0]})
    with pytest.raises(RiskBreach) as ei:
        run_tick(_strategy({"AAA": 1.0, "BBB": 1.0}), broker, _FakeProvider(bars),
                 DATES[0], DATES[-1])
    assert ei.value.kind == "gross_exposure"
```

- [ ] **Step 2: Run** `uv run pytest tests/test_live_loop.py -q` → FAIL.

- [ ] **Step 3: Implement** — create `algua/live/live_loop.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from algua.contracts.types import OrderIntent, Side
from algua.execution.alpaca_broker import AlpacaPaperBroker
from algua.risk.limits import check_gross_exposure, check_long_only
from algua.strategies.base import LoadedStrategy


@dataclass
class TickResult:
    decision_ts: datetime | None
    target_weights: dict[str, float]
    positions_before: dict[str, float]
    submitted: list[dict[str, Any]]


def run_tick(
    strategy: LoadedStrategy,
    broker: AlpacaPaperBroker,
    provider: Any,
    start: datetime,
    end: datetime,
    timeframe: str = "1d",
) -> TickResult:
    """One wall-clock tick: decide on the latest closed session, submit market-order deltas to
    Alpaca (the source of truth). Pure over the injected broker + provider."""
    bars = provider.get_bars(strategy.universe, start, end, timeframe).sort_index()
    positions_before = {s: float(q) for s, q in broker.get_positions().items()}
    if bars.empty:
        return TickResult(None, {}, positions_before, [])

    t = bars.index.max()
    if bars.index.nunique() < strategy.execution.warmup_bars:
        return TickResult(t, {}, positions_before, [])  # warm-up not met

    weights = strategy.target_weights(bars.loc[:t])
    check_long_only(weights, strategy.name)
    check_gross_exposure(weights, strategy.execution.max_gross_exposure)

    broker.cancel_open_orders()
    submitted: list[dict[str, Any]] = []
    symbols = sorted(set(weights.index) | set(broker.get_positions().index))
    for sym in symbols:
        target = float(weights.get(sym, 0.0))
        side = Side.BUY if target > 0 else Side.SELL
        order_id = broker.submit(OrderIntent(symbol=sym, side=side, target_weight=target,
                                             decision_ts=t))
        if order_id != "noop":
            submitted.append({"symbol": sym, "side": side.value,
                              "target_weight": target, "order_id": order_id})
    return TickResult(t, {s: float(w) for s, w in weights.items()}, positions_before, submitted)
```

- [ ] **Step 4: Run** `uv run pytest tests/test_live_loop.py -q` → PASS (4 passed).

- [ ] **Step 5: Gate + commit** — `uv run ruff check . && uv run mypy algua && uv run pytest tests/test_live_loop.py -q`

```bash
git add algua/live/live_loop.py tests/test_live_loop.py
git commit -m "feat(live): run_tick — one wall-clock tick driving AlpacaPaperBroker"
```

---

### Task 4: `paper trade-live` CLI + full gate

**Files:** Modify `algua/cli/paper_cmd.py`; Test `tests/test_cli_paper.py`.

- [ ] **Step 1: Add failing tests** — append to `tests/test_cli_paper.py`:

```python
from algua.live.live_loop import TickResult


def test_trade_live_rejects_non_paper_stage(monkeypatch):
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    runner.invoke(app, ["registry", "add", "cross_sectional_momentum"])  # stage = idea
    result = runner.invoke(app, ["paper", "trade-live", "cross_sectional_momentum",
                                 "--snapshot", "snap1"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False


def test_trade_live_refused_when_killed(monkeypatch):
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    _to_paper()
    runner.invoke(app, ["paper", "kill", "cross_sectional_momentum", "--reason", "x"])
    result = runner.invoke(app, ["paper", "trade-live", "cross_sectional_momentum",
                                 "--snapshot", "snap1"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False


def test_trade_live_submits_and_persists(monkeypatch):
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    _to_paper()
    ts = datetime(2023, 6, 1, tzinfo=UTC)
    fake_result = TickResult(
        decision_ts=ts, target_weights={"AAA": 1.0}, positions_before={},
        submitted=[{"symbol": "AAA", "side": "buy", "target_weight": 1.0, "order_id": "o-1"}],
    )
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings", lambda: object())
    monkeypatch.setattr("algua.cli.paper_cmd._select_provider", lambda demo, snapshot: object())
    monkeypatch.setattr("algua.cli.paper_cmd.run_tick",
                        lambda strategy, broker, provider, start, end: fake_result)
    result = runner.invoke(app, ["paper", "trade-live", "cross_sectional_momentum",
                                 "--snapshot", "snap1"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["submitted"][0]["order_id"] == "o-1"
    show = json.loads(runner.invoke(app, ["paper", "show", "cross_sectional_momentum"]).stdout)
    assert show["n_orders"] == 1
```

(Add `from datetime import UTC, datetime` to the test imports if not present.)

- [ ] **Step 2: Run** `uv run pytest tests/test_cli_paper.py -q` → FAIL.

- [ ] **Step 3: Implement** — in `algua/cli/paper_cmd.py`:

(a) Add imports (after `from algua.live.paper_loop import run_paper`):

```python
from algua.live.live_loop import run_tick
```

and at the top with the stdlib imports:

```python
from datetime import UTC, datetime
```

(b) Add the command at the end of the file:

```python
@paper_app.command("trade-live")
@json_errors(ValueError, LookupError, BrokerError)
def trade_live(
    name: str,
    snapshot: str = typer.Option(..., "--snapshot", help="ingested bars snapshot id"),
    start: str = typer.Option("2023-01-01", "--start"),
    end: str = typer.Option("2023-12-31", "--end"),
) -> None:
    """Run ONE wall-clock tick: submit Alpaca market-order deltas toward the strategy's target."""
    strategy = load_strategy(name)
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        rec = store.get_strategy(conn, name)
        if rec.stage is not Stage.PAPER:
            raise ValueError(f"{name} is at stage '{rec.stage.value}'; trade-live requires 'paper'")
        if kill_switch.is_tripped(conn, name):
            raise ValueError(
                f"kill-switch tripped for {name}; reset with 'algua paper resume {name}'"
            )
        broker = _alpaca_broker_from_settings()
        provider = _select_provider(False, snapshot)
        try:
            result = run_tick(strategy, broker, provider, _utc(start), _utc(end))
        except RiskBreach as exc:
            kill_switch.trip(conn, name, reason=exc.detail, actor="system")
            audit_append(conn, actor="system", action="kill_switch_trip",
                         reason=f"{exc.kind}: {exc.detail}", strategy=name)
            emit({"ok": False, "kind": exc.kind, "kill_switch": "tripped", "error": exc.detail})
            raise typer.Exit(1) from exc
        now = datetime.now(UTC).isoformat()
        for o in result.submitted:
            conn.execute(
                "INSERT INTO paper_orders"
                "(strategy, symbol, side, target_weight, decision_ts, submitted_ts,"
                " status, broker_order_id) VALUES (?,?,?,?,?,?,?,?)",
                (name, o["symbol"], o["side"], o["target_weight"],
                 result.decision_ts.isoformat(), now, "submitted", o["order_id"]),
            )
        conn.commit()
        audit_append(conn, actor="agent", action="trade_live",
                     reason=f"{len(result.submitted)} orders submitted", strategy=name)

    emit({
        "strategy": name,
        "decision_ts": result.decision_ts.isoformat() if result.decision_ts else None,
        "target_weights": result.target_weights,
        "positions_before": result.positions_before,
        "submitted": result.submitted,
    })
```

- [ ] **Step 4: Run** `uv run pytest tests/test_cli_paper.py -q` → PASS.

- [ ] **Step 5: Verify** — `uv run algua paper trade-live --help` shows the command. (No live creds needed; tests cover the logic.)

- [ ] **Step 6: Full gate + commit** — `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports` (all pass; `lint-imports` stays `10 kept, 0 broken`).

```bash
git add algua/cli/paper_cmd.py tests/test_cli_paper.py
git commit -m "feat(live): `algua paper trade-live` one-tick command + persistence"
```

---

## Self-review notes

- **Spec coverage:** `check_long_only` extraction (§2 → Task 1); `cancel_open_orders` (§2 → Task 2); `run_tick` flow incl. warm-up skip, long-only+gross guards, cancel-then-submit, dropped-symbol exit, `TickResult` (§3 → Task 3); `trade-live` gate + RiskBreach-trips-switch + persist submitted + emit (§4,§5 → Task 4); live smoke documented in spec (§6). No new import contract (§2,§6).
- **Type consistency:** `run_tick(strategy, broker, provider, start, end, timeframe="1d") -> TickResult` matches between Task 3 and the CLI/tests in Task 4; `TickResult(decision_ts, target_weights, positions_before, submitted)` identical across Tasks 3–4; `check_long_only(weights, strategy_name)` matches between Tasks 1 and 3; the fake broker in Task 3 implements `get_positions`/`cancel_open_orders`/`submit` (the subset `run_tick` uses).
- **No placeholders:** every code step is complete.
- **Note:** the CLI patches `run_tick`/`_alpaca_broker_from_settings`/`_select_provider` in the persist test so no network call happens; `paper show` reflects the persisted submitted order (`n_orders == 1`), confirming live submissions land in `paper_orders`.
