# Live Buying-Power Reservation — Slice C2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reserve account buying power across the `run-all` loop so many live strategies can share one account without collectively over-committing — then LIFT the ≤1-live guard so the platform runs N strategies live. Each strategy's gross buys draw down a cycle-start pool in deterministic order; an order the pool can't cover is trimmed/skipped and audited.

**Architecture:** `submit_sized` gains a `reserve` seam (the loop caps a BUY's notional via a hook); `run-all` owns a per-cycle pool (account buying power) and passes each strategy a reserve closure that decrements it + logs trims/skips to a new `live_reservations` table. `paper show` becomes stage-aware (live → believed positions + NAV peak). Finally the go-live "only one live strategy" refusal is removed. Schema 15→16.

**Tech Stack:** Python 3.12, sqlite3, Typer, pytest, ruff, mypy, import-linter. Spec: `docs/superpowers/specs/2026-06-06-live-per-strategy-sizing-design.md` (§5). Builds on C1 (`live_sizing`, NAV peak, per-strategy liquidation) + B (`run-all`).

---

## File structure

| File | Responsibility |
|---|---|
| `algua/registry/db.py` (modify) | `live_reservations` table; `SCHEMA_VERSION` 15→16. |
| `algua/execution/live_reservations.py` (new) | `record_reservation(conn, cycle, strategy, symbol, intended, permitted)`. |
| `algua/execution/alpaca_broker.py` (modify) | `submit_sized` `reserve` seam (cap a BUY's notional). |
| `algua/live/live_loop.py` (modify) | `TickHooks.reserve_buy`; `run_tick` passes it to `submit_sized`. |
| `algua/cli/live_cmd.py` (modify) | per-cycle buying-power pool + reserve closure; thread `reserve_buy` through `_run_strategy_tick`. |
| `algua/cli/paper_cmd.py` (modify) | stage-aware `paper show` (live → believed positions + NAV peak). |
| `algua/cli/registry_cmd.py` (modify) | remove the ≤1-live guard. |

---

### Task 1: `live_reservations` table + recorder

**Files:** Modify `algua/registry/db.py`; Create `algua/execution/live_reservations.py`; Test `tests/test_live_reservations.py`.

Context: an append-only audit of every trim/skip so a starved strategy is visible to the operator (Codex #11). `reason` is `"trimmed"` (0 < permitted < intended) or `"skipped"` (permitted == 0).

- [ ] **Step 1: Add the table.** In `algua/registry/db.py`, bump `SCHEMA_VERSION = 15` → `16` and append to `_SCHEMA`:
```sql
CREATE TABLE IF NOT EXISTS live_reservations (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle              INTEGER NOT NULL,
    strategy           TEXT NOT NULL,
    symbol             TEXT NOT NULL,
    intended_notional  REAL NOT NULL,
    permitted_notional REAL NOT NULL,
    reason             TEXT NOT NULL,
    ts                 TEXT NOT NULL
);
```

- [ ] **Step 2: Write the failing test.** Create `tests/test_live_reservations.py`:
```python
from algua.execution import live_reservations as R
from algua.registry.db import connect, migrate


def _conn(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    return conn


def test_record_reservation_trim_and_skip(tmp_path):
    conn = _conn(tmp_path)
    R.record_reservation(conn, cycle=1, strategy="s1", symbol="AAA",
                         intended=1000.0, permitted=400.0)   # partial -> trimmed
    R.record_reservation(conn, cycle=1, strategy="s2", symbol="BBB",
                         intended=500.0, permitted=0.0)       # none -> skipped
    rows = conn.execute(
        "SELECT strategy, reason, permitted_notional FROM live_reservations ORDER BY id"
    ).fetchall()
    assert (rows[0]["strategy"], rows[0]["reason"]) == ("s1", "trimmed")
    assert (rows[1]["strategy"], rows[1]["reason"], rows[1]["permitted_notional"]) == \
        ("s2", "skipped", 0.0)
```

- [ ] **Step 3: Run** `cd /home/liornisimov/Projects/algua/.claude/worktrees/sp6-live-bp && uv run pytest tests/test_live_reservations.py -q` → FAIL.

- [ ] **Step 4: Implement** `algua/execution/live_reservations.py`:
```python
"""Append-only audit of buying-power trims/skips: when the shared per-cycle pool can't fully fund a
strategy's intended buy, the shortfall is recorded so a starved strategy is visible (not a silent
no-op)."""
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime


def record_reservation(conn: sqlite3.Connection, cycle: int, strategy: str, symbol: str,
                       intended: float, permitted: float) -> None:
    """Record that an intended buy of `intended` was funded to `permitted` (reason 'skipped' when
    permitted == 0, else 'trimmed'). Only call when permitted < intended (a full fund is silent)."""
    reason = "skipped" if permitted <= 0.0 else "trimmed"
    conn.execute(
        "INSERT INTO live_reservations"
        "(cycle, strategy, symbol, intended_notional, permitted_notional, reason, ts)"
        " VALUES (?,?,?,?,?,?,?)",
        (cycle, strategy, symbol, intended, permitted, reason, datetime.now(UTC).isoformat()),
    )
    conn.commit()
```

- [ ] **Step 5: Run** the test → PASS.

- [ ] **Step 6: Gate + commit.** `uv run ruff check . && uv run mypy algua && uv run pytest tests/test_live_reservations.py -q && uv run lint-imports`
```bash
git add algua/registry/db.py algua/execution/live_reservations.py tests/test_live_reservations.py
git commit -m "feat(live-bp): live_reservations audit table + recorder (schema v16)"
```

---

### Task 2: `submit_sized` reserve seam

**Files:** Modify `algua/execution/alpaca_broker.py`, `algua/live/live_loop.py`; Test `tests/test_alpaca_broker.py`, `tests/test_live_loop.py`.

Context: `submit_sized` computes `sized = size_order(...)` then POSTs `notional`. Add an optional `reserve: Callable[[str, float], float] | None` consulted ONLY for a BUY: it returns the permitted notional (≤ requested); `0` → skip the order (return `"skipped"`), otherwise POST `min(requested, permitted)`. Sells are never reserved. `run_tick` passes `hooks.reserve_buy` and treats `"skipped"` like `"noop"`.

- [ ] **Step 1: Write the failing tests.** Append to `tests/test_alpaca_broker.py` (uses `_FakeRequests`/`_FakeResp`/`_broker`/`ab`; a flat snapshot so a 0.5 target on 100k equity = a 50k buy):
```python
def test_submit_sized_reserve_trims_buy(monkeypatch):
    fake = _FakeRequests(
        {"/v2/account": _FakeResp(200, {"equity": "100000", "cash": "0", "buying_power": "0"}),
         "/v2/positions": _FakeResp(200, [])},
        post_resp=_FakeResp(201, {"id": "o1"}))
    monkeypatch.setattr(ab, "requests", fake)
    snap = ab.TickSnapshot(equity=100000.0, market_values={"AAA": 0.0}, qtys={"AAA": 0.0})
    # reserve grants only 20k of the intended 50k -> posted notional is trimmed to 20000.00
    _broker().submit_sized(OrderIntent("AAA", Side.BUY, 0.5, T0), snap,
                           reserve=lambda sym, n: 20_000.0)
    assert fake.posted[0]["notional"] == "20000.00"


def test_submit_sized_reserve_zero_skips_buy(monkeypatch):
    fake = _FakeRequests(
        {"/v2/account": _FakeResp(200, {"equity": "100000", "cash": "0", "buying_power": "0"}),
         "/v2/positions": _FakeResp(200, [])},
        post_resp=_FakeResp(201, {"id": "o1"}))
    monkeypatch.setattr(ab, "requests", fake)
    snap = ab.TickSnapshot(equity=100000.0, market_values={"AAA": 0.0}, qtys={"AAA": 0.0})
    assert _broker().submit_sized(OrderIntent("AAA", Side.BUY, 0.5, T0), snap,
                                  reserve=lambda sym, n: 0.0) == "skipped"
    assert fake.posted == []   # nothing posted


def test_submit_sized_reserve_ignores_sells(monkeypatch):
    fake = _FakeRequests(
        {"/v2/account": _FakeResp(200, {"equity": "100000", "cash": "0", "buying_power": "0"}),
         "/v2/positions": _FakeResp(200, [{"symbol": "AAA", "qty": "600",
                                           "market_value": "60000"}])},
        post_resp=_FakeResp(201, {"id": "o2"}))
    monkeypatch.setattr(ab, "requests", fake)
    # a SELL toward 0.5 must not consult reserve (a reserve returning 0 would wrongly skip a sell)
    _broker().submit_sized(OrderIntent("AAA", Side.SELL, 0.5, T0),
                           _broker().snapshot(["AAA"]), reserve=lambda sym, n: 0.0)
    assert fake.posted[0]["side"] == "sell"
```
Append to `tests/test_live_loop.py`:
```python
def test_run_tick_threads_reserve_buy_to_submit_sized():
    from algua.live.live_loop import TickHooks
    seen = {}
    broker = _FakeBroker()
    orig = broker.submit_sized
    broker.submit_sized = lambda intent, snap, coid=None, reserve=None: (
        seen.__setitem__("reserve", reserve) or orig(intent, snap, coid))
    hooks = TickHooks(reserve_buy=lambda sym, n: n)
    run_tick(_strategy({"AAA": 0.5}), broker, _FakeProvider(_bars({"AAA": [100.0, 100.0, 100.0]})),
             DATES[0], DATES[-1], hooks=hooks)
    assert seen["reserve"] is hooks.reserve_buy   # run_tick forwarded the hook
```
(Adapt the `_FakeBroker.submit_sized` shim to that double's actual signature — the intent is: `run_tick` passes `hooks.reserve_buy` into `submit_sized(..., reserve=...)`.)

- [ ] **Step 2: Run** `uv run pytest tests/test_alpaca_broker.py -k reserve tests/test_live_loop.py -k reserve_buy -q` → FAIL.

- [ ] **Step 3a: Implement the seam** in `algua/execution/alpaca_broker.py` `submit_sized` — change the signature and the buy path:
```python
    def submit_sized(self, intent: OrderIntent, snap: TickSnapshot,
                     client_order_id: str | None = None,
                     reserve: Callable[[str, float], float] | None = None) -> str:
```
(add `from typing import Callable` if not already imported), and after `side = "buy" if sized.delta_notional > 0 else "sell"`:
```python
        amount = abs(sized.delta_notional)
        if side == "buy" and reserve is not None:
            permitted = reserve(intent.symbol, amount)
            if permitted <= 0.0:
                return "skipped"
            amount = min(amount, permitted)
        notional = format(Decimal(str(amount)).quantize(Decimal("0.01")), "f")
```
(replace the old `notional = format(...abs(sized.delta_notional)...)` line; the rest — body/POST — unchanged.)

- [ ] **Step 3b: Thread it through run_tick.** In `algua/live/live_loop.py`:
  - add to `TickHooks` (after `live_positions`): `reserve_buy: Callable[[str, float], float] | None = None` with a docstring line (the loop's buying-power reservation; caps a BUY's notional, `0` skips).
  - in the submit loop, change `order_id = broker.submit_sized(intent, snap, coid)` to `order_id = broker.submit_sized(intent, snap, coid, reserve=hooks.reserve_buy)`, and treat `"skipped"` like `"noop"`: change `if order_id == "noop":` to `if order_id in ("noop", "skipped"):`.

- [ ] **Step 4: Run** the tests → PASS, then `uv run pytest tests/test_alpaca_broker.py tests/test_live_loop.py -q` (no regression — `reserve=None` is the default, paper unaffected).

- [ ] **Step 5: Gate + commit.** `uv run ruff check . && uv run mypy algua && uv run pytest tests/test_alpaca_broker.py tests/test_live_loop.py -q && uv run lint-imports`
```bash
git add algua/execution/alpaca_broker.py algua/live/live_loop.py tests/test_alpaca_broker.py tests/test_live_loop.py
git commit -m "feat(live-bp): submit_sized reserve seam + TickHooks.reserve_buy (cap/skip buys)"
```

---

### Task 3: the per-cycle buying-power pool in `run-all`

**Files:** Modify `algua/cli/live_cmd.py`; Test `tests/test_cli_live.py`.

Context: `run-all` ticks verified strategies in order after a clean reconcile. Snapshot account buying power ONCE per cycle into a mutable pool; give each strategy a reserve closure that draws GROSS buys down, records a trim/skip to `live_reservations`, and returns the permitted notional. Thread it into `_run_strategy_tick` → the hooks. A `_broker_buying_power(broker)` shim (monkeypatchable) reads it. `_run_strategy_tick` gains a `reserve_buy=None` param passed to `TickHooks`. The `cycle` (from `next_cycle`) is in scope for the reservation rows.

- [ ] **Step 1: Write the failing test.** Append to `tests/test_cli_live.py`:
```python
def test_run_all_reserves_buying_power_across_strategies(monkeypatch):
    _to_live()
    captured = {}
    monkeypatch.setattr("algua.cli.live_cmd.verify_live_authorization", lambda *a, **k: _auth())
    monkeypatch.setattr("algua.cli.live_cmd._alpaca_live_broker", lambda auth: object())
    monkeypatch.setattr("algua.cli.live_cmd._select_provider", lambda demo, snapshot: object())
    monkeypatch.setattr("algua.cli.live_cmd._broker_account_activities", lambda b, a: [])
    monkeypatch.setattr("algua.cli.live_cmd._broker_net_positions", lambda b: {})
    monkeypatch.setattr("algua.cli.live_cmd._broker_buying_power", lambda b: 30_000.0)

    def _fake_tick(conn, name, auth, broker, provider, max_drawdown, reserve_buy=None, cancel=None):
        captured["first"] = reserve_buy("AAA", 50_000.0)   # ask for 50k from a 30k pool
        captured["second"] = reserve_buy("BBB", 50_000.0)  # pool now drained
        return {"strategy": name}

    monkeypatch.setattr("algua.cli.live_cmd._run_strategy_tick", _fake_tick)
    r = runner.invoke(app, ["live", "run-all", "--snapshot", "x"])
    assert r.exit_code == 0
    assert captured["first"] == 30_000.0   # trimmed to the pool
    assert captured["second"] == 0.0       # nothing left -> skipped
    from contextlib import closing

    from algua.config.settings import get_settings
    from algua.registry.db import connect, migrate
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        n = conn.execute("SELECT COUNT(*) FROM live_reservations").fetchone()[0]
        assert n == 2   # one trim + one skip recorded
```

- [ ] **Step 2: Run** `uv run pytest tests/test_cli_live.py -k reserves_buying_power -q` → FAIL.

- [ ] **Step 3: Implement** in `algua/cli/live_cmd.py`:
  - add imports: `from algua.execution.live_reservations import record_reservation`.
  - add the shim near `_broker_net_positions`:
```python
def _broker_buying_power(broker) -> float:
    return float(broker.account().buying_power)
```
  - in `run_all`, AFTER the `if not recon.clean:` early return and BEFORE the `results = []` loop, create the pool:
```python
        pool = {"available": _broker_buying_power(broker)}

        def _reserve_for(strategy_name):
            def _reserve(symbol: str, notional: float) -> float:
                permitted = min(notional, max(0.0, pool["available"]))
                pool["available"] -= permitted
                if permitted < notional:   # trimmed or fully skipped -> audit the shortfall
                    record_reservation(conn, cycle, strategy_name, symbol, notional, permitted)
                return permitted
            return _reserve
```
  - change the tick call to pass the per-strategy reserve:
```python
        results = []
        for name, authorization in verified:
            results.append(_run_strategy_tick(
                conn, name, authorization, broker, provider, max_drawdown,
                reserve_buy=_reserve_for(name),
                cancel=lambda n=name: _scoped_cancel(conn, broker, n)))
```
  - in `_run_strategy_tick`'s signature add `reserve_buy=None` (before/after `cancel=None`), and add `reserve_buy=reserve_buy` to the `TickHooks(...)` construction.

- [ ] **Step 4: Run** `uv run pytest tests/test_cli_live.py -q` → PASS.

- [ ] **Step 5: Gate + commit.** `uv run ruff check . && uv run mypy algua && uv run pytest tests/test_cli_live.py -q && uv run lint-imports`
```bash
git add algua/cli/live_cmd.py tests/test_cli_live.py
git commit -m "feat(live-bp): per-cycle shared buying-power pool in run-all (gross-buy reserve + audit)"
```

---

### Task 4: stage-aware `paper show`

**Files:** Modify `algua/cli/paper_cmd.py`; Test `tests/test_cli_paper.py`.

Context: `paper show` reads `derive_positions` (paper ledger) + `get_peak_equity` (paper peak) regardless of stage, so a LIVE strategy shows empty paper positions + the wrong peak (Codex C1 LOW). For a live strategy use `believed_positions` + `get_nav_peak`. The reads are around `positions = derive_positions(conn, name)` / `peak = get_peak_equity(conn, name)` in `show`.

- [ ] **Step 1: Write the failing test.** Append to `tests/test_cli_paper.py` (reuse the live-state helpers added in C1, e.g. `_seed_live_killed_with_position`, or build a live strategy with a `live_fills` belief directly):
```python
def test_show_live_strategy_uses_believed_positions_and_nav_peak(monkeypatch, tmp_path):
    import json
    from contextlib import closing

    from algua.config.settings import get_settings
    from algua.execution.order_state import update_nav_peak
    from algua.registry.db import connect, migrate
    name = _seed_live_killed_with_position(monkeypatch, tmp_path)  # live stage + a live_fills belief
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        update_nav_peak(conn, name, 12_345.0)
    payload = json.loads(runner.invoke(app, ["paper", "show", name]).stdout)
    assert payload["drawdown"]["peak"] == 12_345.0        # NAV peak, not the (absent) paper peak
    assert payload["positions"]                            # believed positions, not empty paper
```
(Match the exact `paper show` payload keys for peak/positions by reading `show`; adjust the asserted paths to the real shape.)

- [ ] **Step 2: Run** `uv run pytest tests/test_cli_paper.py -k show_live_strategy -q` → FAIL.

- [ ] **Step 3: Implement** in `algua/cli/paper_cmd.py` `show`: branch the two reads on stage (import `believed_positions` + `get_nav_peak`):
```python
        if rec.stage is Stage.LIVE:
            from algua.execution.live_ledger import believed_positions
            from algua.execution.order_state import get_nav_peak
            positions = believed_positions(conn, name)
            peak = get_nav_peak(conn, name)
        else:
            positions = derive_positions(conn, name)
            peak = get_peak_equity(conn, name)
```
(replace the existing unconditional `positions = derive_positions(...)` / `peak = get_peak_equity(...)` lines; leave the rest of the payload assembly unchanged.)

- [ ] **Step 4: Run** the test → PASS (and `uv run pytest tests/test_cli_paper.py -q` — paper `show` unchanged).

- [ ] **Step 5: Gate + commit.** `uv run ruff check . && uv run mypy algua && uv run pytest tests/test_cli_paper.py -q && uv run lint-imports`
```bash
git add algua/cli/paper_cmd.py tests/test_cli_paper.py
git commit -m "feat(live-bp): stage-aware paper show (live -> believed positions + NAV peak)"
```

---

### Task 5: lift the ≤1-live guard

**Files:** Modify `algua/cli/registry_cmd.py`; Test `tests/test_cli_registry.py`.

Context: the go-live transition refuses a 2nd live strategy (`if len(repo.list_strategies(Stage.LIVE)) > 0 and rec.stage is not Stage.LIVE: raise TransitionError("only one live strategy...")`). With the per-strategy sizing (C1) + the buying-power pool (C2) + the reconcile (B) all in place, MANY strategies may trade live on one account. Remove the refusal; KEEP the allocation requirement.

- [ ] **Step 1: Update the tests.** In `tests/test_cli_registry.py`, the test that asserts the 2nd-live refusal (e.g. `test_go_live_refuses_second_live_strategy`) is now WRONG — replace it with one asserting a 2nd strategy CAN reach the go-live challenge once it has an allocation:
```python
def test_go_live_allows_second_live_strategy_with_allocation(monkeypatch, tmp_path):
    # one strategy already live; a SECOND with an allocation now reaches the go-live challenge
    _force_live(monkeypatch, tmp_path, "already")        # existing helper: a strategy at stage 'live'
    name = _seed_paper(monkeypatch, tmp_path, "s2")
    _allocate(monkeypatch, tmp_path, name, 1000.0)
    r = runner.invoke(app, ["registry", "transition", name, "--to", "live", "--actor", "human"])
    assert r.exit_code == 0                              # a challenge is issued (no ≤1-live refusal)
    assert json.loads(r.stdout)["action"] == "go_live_challenge"
```
(Reuse the same `_force_live`/`_seed_paper`/`_allocate` helpers the removed test used. The "requires allocation" test stays — only the ≤1-live refusal test changes.)

- [ ] **Step 2: Run** `uv run pytest tests/test_cli_registry.py -k "second_live or requires_allocation" -q` → FAIL (the guard still refuses).

- [ ] **Step 3: Implement.** In `algua/cli/registry_cmd.py`, DELETE the ≤1-live refusal block:
```python
            if len(repo.list_strategies(Stage.LIVE)) > 0 and rec.stage is not Stage.LIVE:
                raise TransitionError(
                    "refusing: only one live strategy is allowed until multi-strategy controls "
                    "land (slice C). Retire the current live strategy first.")
```
Keep the `active_allocation is None` check immediately after it. (`repo.list_strategies` may now be unused in this block — leave other usages intact.)

- [ ] **Step 4: Run** `uv run pytest tests/test_cli_registry.py -q` → PASS.

- [ ] **Step 5: FULL gate + commit.** `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports` (all green; contracts 0 broken).
```bash
git add algua/cli/registry_cmd.py tests/test_cli_registry.py
git commit -m "feat(live-bp): lift the one-live-strategy guard (multi-strategy live enabled)"
```

---

## Self-review notes

- **Spec coverage (C2 in §5):** the shared per-cycle pool drawing down GROSS buys in deterministic registry order with trim/skip + persisted reason (Tasks 1-3); `paper show` stage-aware (Task 4, the deferred C1 LOW); lift the ≤1-live guard (Task 5). Sells don't add back (the pool only decrements on buys; the reserve seam ignores sells, Task 2). ✓
- **Loop-order fairness:** deterministic registry order (the existing `verified` order from `list_strategies(Stage.LIVE)`), audited via `live_reservations`; pro-rata remains a future option (spec §8) — not built here.
- **Type consistency:** `reserve_buy: Callable[[str,float],float]` (Task 2 hook) is produced by `_reserve_for(name)` (Task 3) and consumed by `submit_sized(..., reserve=...)` (Task 2); `record_reservation(conn, cycle, strategy, symbol, intended, permitted)` (Task 1) called by the pool closure (Task 3); `_broker_buying_power` (Task 3) mirrors the existing `_broker_net_positions` shim.
- **No placeholders:** the test-helper "reuse C1/registry helpers" notes (Tasks 4-5) point at named existing fixtures the implementer must read; production code + assertion intent are fully specified. Task 5 deletes a named block and replaces a named test.
- **Safety ordering:** the guard lift (Task 5) is LAST — multi-strategy live only becomes reachable after the pool (Tasks 1-3) that bounds collective buying power exists. `reserve=None` keeps paper + any non-reserved path unchanged.
