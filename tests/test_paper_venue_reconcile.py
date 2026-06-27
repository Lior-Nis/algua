"""Task-3: paginated, fail-closed paper venue ingest helper (broker-time cursor).

Tests confirm:
  - first call defaults to _FAR_PAST and advances cursor to broker.clock()
  - transport failure propagates and leaves cursor untouched (fail-closed)

Task-9: end-to-end reconcile regression tests (#249 phantom-flatten fix).
"""
import json
import sqlite3

import pytest
from typer.testing import CliRunner

from algua.cli.main import app
from algua.cli.paper_cmd import _ingest_paper_venue
from algua.execution.alpaca_broker import AccountState, TickSnapshot
from algua.execution.live_ledger import LedgerKind, fill_cursor, paper_believed_positions
from algua.registry.db import migrate

_FAR_PAST = "1970-01-01T00:00:00Z"

runner = CliRunner()


class FakeBroker:
    def __init__(self, windows: dict, clock: str = "2026-01-02T00:00:00Z"):
        # windows: dict cursor -> list[activity] OR Exception instance to raise
        self._windows = windows
        self._clock = clock

    def clock(self) -> str:
        return self._clock

    def account_activities_window(self, after: str, until: str) -> list[dict]:
        out = self._windows.get(after)
        if isinstance(out, Exception):
            raise out
        return out or []


def _conn(tmp_path):
    c = sqlite3.connect(tmp_path / "r.db")
    c.row_factory = sqlite3.Row
    migrate(c)
    return c


def _fill(aid: str, sym: str, qty: float, side: str, oid: str) -> dict:
    return {
        "id": aid,
        "activity_type": "FILL",
        "side": side,
        "qty": abs(qty),
        "price": 10.0,
        "symbol": sym,
        "order_id": oid,
        "transaction_time": "2026-01-01T12:00:00Z",
    }


def test_ingest_uses_far_past_first_then_advances_cursor(tmp_path):
    c = _conn(tmp_path)
    c.execute(
        "INSERT INTO paper_venue_orders(strategy,symbol,side,client_order_id,broker_order_id,"
        "strategy_id,status,submitted_ts) VALUES ('s','AAA','buy','c','o1',1,'submitted','t')"
    )
    c.commit()
    broker = FakeBroker({_FAR_PAST: [_fill("a1", "AAA", 5, "buy", "o1")]})
    _ingest_paper_venue(c, broker)
    assert paper_believed_positions(c, "s") == {"AAA": 5.0}
    assert fill_cursor(c, LedgerKind.PAPER) == "2026-01-02T00:00:00Z"  # = until (broker clock)


def test_ingest_fails_closed_on_transport_error(tmp_path):
    c = _conn(tmp_path)
    broker = FakeBroker({_FAR_PAST: RuntimeError("503")})
    with pytest.raises(RuntimeError):
        _ingest_paper_venue(c, broker)
    assert fill_cursor(c, LedgerKind.PAPER) is None  # cursor must NOT advance on failure


# ---------------------------------------------------------------------------
# Task-9: end-to-end reconcile regression (#249 phantom-flatten fix)
# ---------------------------------------------------------------------------

_NAME = "cross_sectional_momentum"
_START = "2022-01-01"
_END = "2023-06-01"


def _to_paper(name: str = _NAME) -> None:
    assert runner.invoke(app, ["backtest", "run", name, "--demo", "--register",
                               "--start", "2022-01-01", "--end", "2023-12-31"]).exit_code == 0
    assert runner.invoke(app, ["registry", "transition", name, "--to", "candidate",
                               "--actor", "human", "--reason", "ok"]).exit_code == 0
    assert runner.invoke(app, ["registry", "transition", name, "--to", "paper",
                               "--actor", "agent", "--reason", "paper"]).exit_code == 0


class _PaperVenueTestBroker:
    """Scripted paper broker for #249 reconcile regression tests.

    Tracks submitted_orders so the test can simulate fills landing between Tick 1 and
    Tick 2 without knowing which symbols the strategy chose (signal is deterministic but
    non-obvious from the test body).
    """

    def __init__(self, equity: float = 100_000.0) -> None:
        self._equity = equity
        self._clock_ts = "2024-01-15T14:00:00Z"
        self._activities: list[dict] = []
        self._order_counter = 0
        self.submitted_orders: list[dict] = []
        self._positions: dict[str, float] = {}

    def account(self) -> AccountState:
        return AccountState(equity=self._equity, cash=self._equity,
                            buying_power=self._equity, account_id="test-paper-acct")

    def clock(self) -> str:
        return self._clock_ts

    def account_activities_window(self, after: str, until: str) -> list[dict]:
        return list(self._activities)

    def snapshot(self, universe: list[str]) -> TickSnapshot:
        all_syms = set(universe) | set(self._positions)
        return TickSnapshot(
            equity=self._equity,
            market_values={s: self._positions.get(s, 0.0) * 100.0 for s in all_syms},
            qtys={s: self._positions.get(s, 0.0) for s in all_syms},
        )

    def cancel_open_orders(self) -> None:
        pass

    def close_positions(self, symbols: list[str]) -> None:
        for s in symbols:
            self._positions.pop(s, None)

    def submit_sized(self, intent, snap, coid=None, reserve=None) -> str:
        self._order_counter += 1
        broker_id = f"bo-{intent.symbol}-{self._order_counter}"
        self.submitted_orders.append({
            "coid": coid,
            "broker_id": broker_id,
            "symbol": intent.symbol,
        })
        return broker_id


def test_trade_tick_no_phantom_flatten_after_fill(monkeypatch, tmp_path):
    """#249 regression: a fill at the venue reconciles clean on the next tick.

    Old path: derive_positions (paper_fills) was always empty for the wall-clock lane, so
    broker-held positions caused RiskBreach('reconcile') — a phantom flatten of a healthy
    strategy.  New path: paper_venue_fills are ingested before each tick and paper_believed_
    positions is reconciled against broker.snapshot().qtys — fill lands → belief matches
    broker → reconcile_ok=True.
    """
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "p.db"))
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    _to_paper()

    broker = _PaperVenueTestBroker()
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings", lambda: broker)
    from algua.backtest._sample import SyntheticProvider
    monkeypatch.setattr("algua.cli.paper_cmd._select_provider",
                        lambda demo, snapshot: SyntheticProvider())

    # Tick 1: broker flat, no activities → strategy decides and submits orders
    result1 = runner.invoke(app, ["paper", "trade-tick", _NAME, "--snapshot", "snap1",
                                  "--start", _START, "--end", _END])
    assert result1.exit_code == 0, result1.stdout
    assert broker.submitted_orders, "Tick 1 must submit at least one order for the regression"

    # Simulate fills landing: for each submitted order the broker now reports a fill and holds
    # the position.  The fill amount is arbitrary; the test verifies parity, not sizing math.
    fill_qty = 100.0
    for i, order in enumerate(list(broker.submitted_orders)):
        broker._activities.append({
            "id": f"fill-act-{i}",
            "activity_type": "FILL",
            "side": "buy",
            "qty": str(fill_qty),
            "price": "100.0",
            "symbol": order["symbol"],
            "order_id": order["broker_id"],
            "transaction_time": "2024-01-15T14:30:00Z",
        })
        broker._positions[order["symbol"]] = fill_qty

    # Tick 2: fills ingested into paper_venue_fills → belief matches broker snapshot → clean
    result2 = runner.invoke(app, ["paper", "trade-tick", _NAME, "--snapshot", "snap1",
                                  "--start", _START, "--end", _END])
    assert result2.exit_code == 0, result2.stdout
    payload2 = json.loads(result2.stdout)
    assert payload2["ok"] is True
    assert payload2["reconcile_ok"] is True  # the phantom-flatten bug would give False / exit 1


def test_trade_tick_trips_on_orphan_holding(monkeypatch, tmp_path):
    """A broker position with no matching paper_venue_fills (orphan / manual holding) must trip
    RiskBreach('reconcile') — fail-closed, never trades on unattributed state."""
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "p.db"))
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    _to_paper()

    broker = _PaperVenueTestBroker()
    # Broker holds AAPL with NO matching paper_venue_fills (no orders recorded, no fills ingested)
    broker._positions["AAPL"] = 50.0
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings", lambda: broker)
    from algua.backtest._sample import SyntheticProvider
    monkeypatch.setattr("algua.cli.paper_cmd._select_provider",
                        lambda demo, snapshot: SyntheticProvider())

    result = runner.invoke(app, ["paper", "trade-tick", _NAME, "--snapshot", "snap1",
                                 "--start", _START, "--end", _END])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["kind"] == "reconcile"
    assert payload["kill_switch"] == "tripped"
