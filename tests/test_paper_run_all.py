"""Tests for `paper run-all`: multi-tenant batch tick (#316b Task 3).

Four required behaviours:
  1. Both paper strategies tick; envelope lists both.
  2. Not-clean reconcile defers the whole cycle; no strategy trades.
  3. A breach in one strategy trips + scoped-flattens only it; the envelope still surfaces the
     sibling that was ticked before it; exit non-zero; the sibling's resting order is NOT
     cancelled by the breacher's scoped cancel.
  4. The reservation pool trims a second strategy's BUY when the pool is exhausted.
"""
from __future__ import annotations

import json
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

import algua.strategies.momentum as _momentum_pkg
from algua.cli.main import app
from algua.config.settings import get_settings
from algua.execution.alpaca_broker import AccountState
from algua.live.live_loop import TickResult
from algua.registry.db import connect, migrate
from algua.registry.store import SqliteStrategyRepository
from algua.risk.limits import RiskBreach

runner = CliRunner()

_SNAP = "snap1"
# _S1 is registered first → lower DB id → ticked first in run-all (ORDER BY id)
_S1 = "cross_sectional_momentum"
# _S2 is registered second → higher DB id → ticked second
_S2 = "liquid10_momentum"

_START = "2026-01-01"
_END = "2026-02-01"


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "p.db"))
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")


@pytest.fixture(autouse=True)
def _second_strategy():
    """`_S2` is a second REAL, loadable, demo-backtestable paper strategy (run-all loads + ticks
    EVERY paper strategy, unlike live run-all which skips unauthorized ones before load). Write it
    as a temp module in the momentum family so `_index()` discovers it, then unlink it after the
    test (mirrors tests/test_promotion._write_tmp_strategy)."""
    p = Path(_momentum_pkg.__path__[0]) / f"{_S2}.py"
    p.write_text(
        '"""Second demo strategy for paper run-all tests: trailing-return momentum, top-k."""\n'
        "from __future__ import annotations\n"
        "from typing import Any\n"
        "import pandas as pd\n"
        "from algua.contracts.types import ExecutionContract\n"
        "from algua.features.alphas import xs_trailing_return\n"
        "from algua.strategies.base import StrategyConfig\n"
        f"CONFIG = StrategyConfig(name={_S2!r},\n"
        "    universe=['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL'],\n"
        "    execution=ExecutionContract(rebalance_frequency='1d', decision_lag_bars=1),\n"
        "    params={'lookback': 60}, construction='top_k_equal_weight',\n"
        "    construction_params={'top_k': 3}, feature_lookback=60)\n"
        "def signal(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:\n"
        "    return xs_trailing_return(view, params)\n"
    )
    try:
        yield
    finally:
        p.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_paper(name: str) -> None:
    assert runner.invoke(app, ["backtest", "run", name, "--demo", "--register",
                               "--start", "2022-01-01", "--end", "2023-12-31"]).exit_code == 0
    assert runner.invoke(app, ["registry", "transition", name, "--to", "candidate",
                               "--actor", "human", "--reason", "ok"]).exit_code == 0
    assert runner.invoke(app, ["registry", "transition", name, "--to", "paper",
                               "--actor", "agent", "--reason", "paper"]).exit_code == 0


def _force_stage(name: str, stage_value: str) -> None:
    """Force a strategy's lifecycle stage directly (bypasses the promote gate for test setup)."""
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        rec = SqliteStrategyRepository(conn).get(name)
        conn.execute("UPDATE strategies SET stage = ? WHERE id = ?", (stage_value, rec.id))
        conn.commit()


def _seed_allocation(name: str, capital: float = 10_000.0) -> None:
    """Insert a strategy_allocations row directly (no paper-allocate CLI dependency)."""
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        rec = SqliteStrategyRepository(conn).get(name)
        conn.execute(
            "INSERT INTO strategy_allocations(strategy_id, capital, effective_ts, actor) "
            "VALUES (?,?,?,?)",
            (rec.id, capital, datetime.now(UTC).isoformat(), "agent"),
        )
        conn.commit()


def _seed_paper_venue_order(name: str, symbol: str,
                             client_order_id: str, broker_order_id: str) -> None:
    """Seed a paper_venue_orders row so owned_open_order_ids can attribute it to `name`."""
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        rec = SqliteStrategyRepository(conn).get(name)
        conn.execute(
            "INSERT OR IGNORE INTO paper_venue_orders"
            "(strategy, symbol, side, intended_notional, client_order_id, broker_order_id,"
            " strategy_id, status, submitted_ts) VALUES (?,?,?,?,?,?,?,?,?)",
            (name, symbol, "buy", None, client_order_id, broker_order_id,
             rec.id, "submitted", "2023-01-01T00:00:00Z"),
        )
        conn.commit()


# ---------------------------------------------------------------------------
# Broker fake
# ---------------------------------------------------------------------------

class _RunAllBroker:
    """Multi-strategy paper broker stub for run-all tests.

    Tracks per-broker-call data so tests can assert: both submitted, scoped cancel, pool trim.
    """

    def __init__(
        self,
        equity: float = 100_000.0,
        positions: dict | None = None,
        open_orders: list | None = None,
    ) -> None:
        self._equity = equity
        self._positions: dict = positions or {}
        self._open_orders: list = open_orders or []
        self.submitted: list = []          # (sym, qty, coid) from submit_offset
        self.cancelled_ids: list = []      # order ids passed to cancel_order

    def account(self) -> AccountState:
        return AccountState(equity=self._equity, cash=self._equity,
                            buying_power=self._equity, account_id="test-acct")

    def clock(self) -> str:
        return "2023-06-01T14:00:00+00:00"

    def account_activities_window(self, after: str, until: str) -> list:
        return []

    def get_positions(self):
        return pd.Series(self._positions, dtype="float64")

    def list_open_orders(self) -> list:
        return list(self._open_orders)

    def cancel_order(self, order_id: str) -> None:
        self.cancelled_ids.append(order_id)

    def cancel_open_orders(self) -> None:
        pass

    def submit_offset(self, sym: str, qty: float, coid: str) -> str:
        self.submitted.append((sym, qty, coid))
        return f"o-offset-{sym}"


# ---------------------------------------------------------------------------
# Common fake tick result factory
# ---------------------------------------------------------------------------

def _success_result() -> TickResult:
    return TickResult(
        decision_ts=datetime(2023, 6, 1, tzinfo=UTC),
        target_weights={},
        positions_before={},
        submitted=[],
        peak_equity=10_000.0,
        equity=10_000.0,
    )


# ---------------------------------------------------------------------------
# Test 1: both paper strategies tick
# ---------------------------------------------------------------------------

def test_run_all_ticks_all_paper_strategies(monkeypatch):
    """Two paper strategies, both allocated; clean reconcile. Both tick; envelope lists both."""
    _to_paper(_S1)
    _to_paper(_S2)
    _seed_allocation(_S1)
    _seed_allocation(_S2)

    broker = _RunAllBroker()  # clean account — no broker positions
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings", lambda: broker)
    monkeypatch.setattr("algua.cli.paper_cmd._select_provider", lambda demo, snap: object())
    monkeypatch.setattr(
        "algua.cli.paper_cmd.run_tick",
        lambda strategy, broker, provider, start, end, hooks=None, max_drawdown=None:
            _success_result(),
    )

    result = runner.invoke(
        app, ["paper", "run-all", "--snapshot", _SNAP, "--start", _START, "--end", _END]
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload.get("ok") is True
    names = {s["strategy"] for s in payload["strategies"]}
    assert names == {_S1, _S2}


# ---------------------------------------------------------------------------
# Test 2: not-clean reconcile defers the whole cycle
# ---------------------------------------------------------------------------

def test_run_all_defers_whole_cycle_on_unreconciled_account(monkeypatch):
    """Broker shows an unattributable holding → reconcile not clean → NO strategy trades."""
    _to_paper(_S1)
    _seed_allocation(_S1)

    # Broker holds AAPL with no matching paper_venue_fills (orphan position)
    broker = _RunAllBroker(positions={"AAPL": 50.0})
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings", lambda: broker)
    monkeypatch.setattr("algua.cli.paper_cmd._select_provider", lambda demo, snap: object())

    # run_tick should NOT be called — assert by raising if it is
    def _should_not_be_called(*a, **k):
        raise AssertionError("run_tick called despite not-clean reconcile")

    monkeypatch.setattr("algua.cli.paper_cmd.run_tick", _should_not_be_called)

    result = runner.invoke(
        app, ["paper", "run-all", "--snapshot", _SNAP, "--start", _START, "--end", _END]
    )
    # Exit 0: deferred (not a halt; first cycle is pending)
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["strategies"] == []
    assert payload.get("deferred") is True
    # No orders submitted
    assert broker.submitted == []


# ---------------------------------------------------------------------------
# Test 3: breach in one strategy is scoped; sibling's resting order preserved
# ---------------------------------------------------------------------------

def test_run_all_breach_scoped_flatten_surfaces_siblings(monkeypatch):
    """S1 (ticked first, succeeds) has a resting open order.
    S2 (ticked second) breaches.
    Scoped cancel must NOT cancel S1's resting order.
    Envelope surfaces both results; exit non-zero."""
    # _S1 registered first → lower id → ticked first (succeeds)
    # _S2 registered second → higher id → ticked second (breaches)
    _to_paper(_S1)
    _to_paper(_S2)
    _seed_allocation(_S1)
    _seed_allocation(_S2)

    # Seed resting open orders for both strategies in paper_venue_orders
    # so owned_open_order_ids can resolve ownership
    _seed_paper_venue_order(_S1, "AAA", "coid-s1-AAA", "oid-s1-AAA")
    _seed_paper_venue_order(_S2, "BBB", "coid-s2-BBB", "oid-s2-BBB")

    sibling_open_order_id = "oid-s1-AAA"
    breacher_open_order_id = "oid-s2-BBB"

    broker = _RunAllBroker(
        open_orders=[
            {"id": sibling_open_order_id, "client_order_id": "coid-s1-AAA"},
            {"id": breacher_open_order_id, "client_order_id": "coid-s2-BBB"},
        ]
    )
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings", lambda: broker)
    monkeypatch.setattr("algua.cli.paper_cmd._select_provider", lambda demo, snap: object())

    call_n: list[int] = [0]

    def _fake_run_tick(strategy, broker, provider, start, end, hooks=None, max_drawdown=None):
        call_n[0] += 1
        if call_n[0] == 2:  # S2's tick (second call) → breach
            raise RiskBreach("drawdown", "drawdown 0.30 exceeds max_drawdown 0.10")
        return _success_result()

    monkeypatch.setattr("algua.cli.paper_cmd.run_tick", _fake_run_tick)

    result = runner.invoke(
        app, ["paper", "run-all", "--snapshot", _SNAP, "--start", _START, "--end", _END]
    )

    payload = json.loads(result.stdout)
    assert result.exit_code != 0
    # Envelope contains both strategies' results (breach breaks the loop after appending S2)
    strats_in_envelope = {s["strategy"] for s in payload["strategies"]}
    assert _S1 in strats_in_envelope  # sibling surfaced
    assert _S2 in strats_in_envelope  # breacher surfaced
    # At least one result has ok=False (the breach)
    assert any(s.get("ok") is False for s in payload["strategies"])
    # Scoped cancel touched ONLY S2's order — S1's resting order is NOT cancelled
    assert sibling_open_order_id not in broker.cancelled_ids
    assert breacher_open_order_id in broker.cancelled_ids


# ---------------------------------------------------------------------------
# Test 4: reservation pool caps a second strategy's buy
# ---------------------------------------------------------------------------

def test_run_all_reservation_pool_caps_concurrent_buys(monkeypatch):
    """Pool = buying_power; first strategy consumes most of it; second strategy's buy is trimmed."""
    _to_paper(_S1)
    _to_paper(_S2)
    _seed_allocation(_S1)
    _seed_allocation(_S2)

    equity = 5_000.0
    broker = _RunAllBroker(equity=equity)  # buying_power = equity = 5000
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings", lambda: broker)
    monkeypatch.setattr("algua.cli.paper_cmd._select_provider", lambda demo, snap: object())

    # Track what each strategy's reserve_buy returns (permitted notional)
    s1_permitted: list[float] = []
    s2_permitted: list[float] = []
    call_n: list[int] = [0]

    s1_requested = 4_500.0  # S1 takes most of the pool
    s2_requested = 3_000.0  # S2 asks more than what remains

    def _fake_run_tick(strategy, broker_, provider, start, end, hooks=None, max_drawdown=None):
        call_n[0] += 1
        # Invoke reserve_buy so the pool closure is exercised
        if call_n[0] == 1 and hooks is not None and hooks.reserve_buy is not None:
            perm = hooks.reserve_buy("AAA", s1_requested)
            s1_permitted.append(perm)
        elif call_n[0] == 2 and hooks is not None and hooks.reserve_buy is not None:
            perm = hooks.reserve_buy("BBB", s2_requested)
            s2_permitted.append(perm)
        return _success_result()

    monkeypatch.setattr("algua.cli.paper_cmd.run_tick", _fake_run_tick)

    result = runner.invoke(
        app, ["paper", "run-all", "--snapshot", _SNAP, "--start", _START, "--end", _END]
    )
    assert result.exit_code == 0, result.stdout

    # S1 consumed s1_requested from the pool (pool started at equity)
    assert s1_permitted == [s1_requested]
    remaining_after_s1 = equity - s1_requested  # = 500.0
    # S2's permitted must be capped at what remained after S1
    assert len(s2_permitted) == 1
    assert s2_permitted[0] <= remaining_after_s1
    # And specifically it should equal remaining_after_s1 (not less, pool not over-consumed)
    assert abs(s2_permitted[0] - remaining_after_s1) < 1e-9


# ---------------------------------------------------------------------------
# Test 5: forward_tested strategies are ticked too (stage IN paper, forward_tested)
# ---------------------------------------------------------------------------

def test_run_all_ticks_forward_tested_strategies(monkeypatch):
    """A forward_tested strategy is a paper-lane trading target (parity with load_gated_strategy /
    trade-tick): run-all must tick it — NOT silently skip it because it filtered stage='paper'."""
    _to_paper(_S1)
    _to_paper(_S2)
    _force_stage(_S2, "forward_tested")  # S2 is now forward_tested, still paper-lane
    _seed_allocation(_S1)
    _seed_allocation(_S2)

    broker = _RunAllBroker()
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings", lambda: broker)
    monkeypatch.setattr("algua.cli.paper_cmd._select_provider", lambda demo, snap: object())
    monkeypatch.setattr(
        "algua.cli.paper_cmd.run_tick",
        lambda strategy, broker, provider, start, end, hooks=None, max_drawdown=None:
            _success_result(),
    )

    result = runner.invoke(
        app, ["paper", "run-all", "--snapshot", _SNAP, "--start", _START, "--end", _END]
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    names = {s["strategy"] for s in payload["strategies"]}
    assert names == {_S1, _S2}  # the forward_tested strategy was ticked, not dropped


# ---------------------------------------------------------------------------
# Test 6: pool debits only what actually posts (no phantom debit for a skipped buy)
# ---------------------------------------------------------------------------

def test_run_all_pool_does_not_phantom_debit_skipped_buys(monkeypatch):
    """A sub-MIN_NOTIONAL grant is SKIPPED by submit_sized (posts nothing), so it must NOT decrement
    the shared pool. The pool is debited by posted_notional(grant), not the raw grant — otherwise a
    phantom debit wrongly starves a later sibling of buying power."""
    _to_paper(_S1)
    _to_paper(_S2)
    _seed_allocation(_S1)
    _seed_allocation(_S2)

    equity = 5_000.0
    broker = _RunAllBroker(equity=equity)
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings", lambda: broker)
    monkeypatch.setattr("algua.cli.paper_cmd._select_provider", lambda demo, snap: object())

    s2_permitted: list[float] = []
    call_n: list[int] = [0]
    tiny = 0.50  # below MIN_NOTIONAL ($1): submit_sized returns "skipped", posts nothing

    def _fake_run_tick(strategy, broker_, provider, start, end, hooks=None, max_drawdown=None):
        call_n[0] += 1
        if call_n[0] == 1 and hooks is not None and hooks.reserve_buy is not None:
            hooks.reserve_buy("AAA", tiny)  # sub-min buy: skipped, posts nothing
        elif call_n[0] == 2 and hooks is not None and hooks.reserve_buy is not None:
            s2_permitted.append(hooks.reserve_buy("BBB", equity))  # asks for the whole account
        return _success_result()

    monkeypatch.setattr("algua.cli.paper_cmd.run_tick", _fake_run_tick)

    result = runner.invoke(
        app, ["paper", "run-all", "--snapshot", _SNAP, "--start", _START, "--end", _END]
    )
    assert result.exit_code == 0, result.stdout
    # S1's sub-min buy posted $0, so the pool is still the full account: S2 is granted all of it.
    # (Under the phantom-debit bug the pool would have decremented by $0.50 and S2 would see only
    # $4999.50.)
    assert s2_permitted == [equity]


# ---------------------------------------------------------------------------
# Test 7: an omitted --max-drawdown still trips a breach at the default bound
# ---------------------------------------------------------------------------

def test_run_all_omitted_max_drawdown_uses_default_bound(monkeypatch):
    """Parity with trade-tick / live run-all (#452 GATE-2): an omitted --max-drawdown must resolve
    to the default-ON ``settings.strategy_max_drawdown_default`` bound, NOT leave the breaker OFF.
    The fake run_tick asserts the bound threaded into it equals that default and then breaches, so
    the envelope surfaces a scoped-flattened breach and exits non-zero."""
    _to_paper(_S1)
    _seed_allocation(_S1)
    _seed_paper_venue_order(_S1, "AAA", "coid-s1-AAA", "oid-s1-AAA")

    broker = _RunAllBroker(
        open_orders=[{"id": "oid-s1-AAA", "client_order_id": "coid-s1-AAA"}]
    )
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings", lambda: broker)
    monkeypatch.setattr("algua.cli.paper_cmd._select_provider", lambda demo, snap: object())

    default_bound = get_settings().strategy_max_drawdown_default
    seen_max_dd: list = []

    def _fake_run_tick(strategy, broker_, provider, start, end, hooks=None, max_drawdown=None):
        seen_max_dd.append(max_drawdown)
        # The omitted flag must have resolved to the default-ON bound, not None (breaker OFF).
        raise RiskBreach("drawdown", f"drawdown 0.99 exceeds max_drawdown {max_drawdown}")

    monkeypatch.setattr("algua.cli.paper_cmd.run_tick", _fake_run_tick)

    # NOTE: no --max-drawdown passed
    result = runner.invoke(
        app, ["paper", "run-all", "--snapshot", _SNAP, "--start", _START, "--end", _END]
    )
    assert result.exit_code != 0, result.stdout
    assert seen_max_dd == [default_bound]
    payload = json.loads(result.stdout)
    assert payload.get("ok") is False
    assert any(s.get("ok") is False for s in payload["strategies"])


# ---------------------------------------------------------------------------
# Test 8: omitted --start/--end resolve to a rolling window ending today
# ---------------------------------------------------------------------------

def test_run_all_omitted_window_resolves_to_rolling_today(monkeypatch):
    """Parity with trade-tick / live run-all (#452 GATE-2): omitted --start/--end must resolve to a
    recent rolling wall-clock window ending TODAY (UTC), NOT the literal 2023 defaults that would
    size/risk-check against a frozen stale window. The fake run_tick captures the (start, end)
    threaded into it (parsed to datetimes via ``utc()``) and asserts end == today and start is a
    recent (not-2023) date."""
    _to_paper(_S1)
    _seed_allocation(_S1)

    broker = _RunAllBroker()
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings", lambda: broker)
    monkeypatch.setattr("algua.cli.paper_cmd._select_provider", lambda demo, snap: object())

    seen_window: list = []

    def _fake_run_tick(strategy, broker_, provider, start, end, hooks=None, max_drawdown=None):
        seen_window.append((start, end))
        return _success_result()

    monkeypatch.setattr("algua.cli.paper_cmd.run_tick", _fake_run_tick)

    # NOTE: no --start/--end passed
    result = runner.invoke(app, ["paper", "run-all", "--snapshot", _SNAP])
    assert result.exit_code == 0, result.stdout
    assert len(seen_window) == 1
    start, end = seen_window[0]
    today = datetime.now(UTC).date()
    assert end.date() == today  # end resolves to today, not literal 2023-12-31
    assert start.year != 2023  # start is a recent rolling date, not the 2023 default
    assert start < end  # a real lookback window
