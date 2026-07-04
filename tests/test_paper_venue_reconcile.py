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
    _ingest_paper_venue(c, broker, broker.clock())
    assert paper_believed_positions(c, "s") == {"AAA": 5.0}
    assert fill_cursor(c, LedgerKind.PAPER) == "2026-01-02T00:00:00Z"  # = until (broker clock)


def test_ingest_fails_closed_on_transport_error(tmp_path):
    c = _conn(tmp_path)
    broker = FakeBroker({_FAR_PAST: RuntimeError("503")})
    with pytest.raises(RuntimeError):
        _ingest_paper_venue(c, broker, broker.clock())
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

    def get_positions(self):
        import pandas as pd
        return pd.Series(self._positions, dtype="float64")

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


def _seed_allocation_venue(name: str, capital: float = 100_000.0) -> None:
    """Insert a strategy_allocations row without going through the paper-allocate CLI."""
    from datetime import UTC, datetime

    from algua.config.settings import get_settings
    from algua.registry.db import connect, migrate
    with connect(get_settings().db_path) as conn:
        migrate(conn)
        rec = conn.execute("SELECT id FROM strategies WHERE name = ?", (name,)).fetchone()
        if rec is None:
            raise LookupError(f"strategy {name!r} not found")
        conn.execute(
            "INSERT INTO strategy_allocations(strategy_id, capital, effective_ts, actor) "
            "VALUES (?,?,?,?)",
            (rec["id"], capital, datetime.now(UTC).isoformat(), "agent"),
        )
        conn.commit()


def test_trade_tick_no_phantom_flatten_after_fill(monkeypatch, tmp_path):
    """#249 regression: a fill at the venue reconciles clean on the next tick.

    Old path: derive_positions (paper_fills) was always empty for the wall-clock lane, so
    broker-held positions caused RiskBreach('reconcile') — a phantom flatten of a healthy
    strategy.  New path: paper_venue_fills are ingested before each tick and the account
    reconcile checks attributed_paper_net vs broker net — fill lands → belief matches
    broker → account reconcile clean → reconcile_ok=True (no venue_belief in run_tick).
    """
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "p.db"))
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    _to_paper()
    _seed_allocation_venue(_NAME, capital=100_000.0)

    broker = _PaperVenueTestBroker()
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings", lambda: broker)
    from algua.backtest._sample import SyntheticProvider
    monkeypatch.setattr("algua.cli.paper_cmd._select_provider",
                        lambda demo, snapshot: SyntheticProvider())

    # A wall-clock trade-tick values the book off bar marks, which the #452 freshness wall requires
    # to be RECENT (<= 2 completed sessions old). Use a rolling window ending today so the synthetic
    # marks are fresh, rather than the frozen 2023 window (which would trip stale_marks).
    from datetime import UTC, datetime, timedelta
    _end = datetime.now(UTC).date()
    _start = (_end - timedelta(days=500)).isoformat()
    _end = _end.isoformat()

    # Tick 1: broker flat, no activities → strategy decides and submits orders
    result1 = runner.invoke(app, ["paper", "trade-tick", _NAME, "--snapshot", "snap1",
                                  "--start", _start, "--end", _end])
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
                                  "--start", _start, "--end", _end])
    assert result2.exit_code == 0, result2.stdout
    payload2 = json.loads(result2.stdout)
    assert payload2["ok"] is True
    assert payload2["reconcile_ok"] is True  # the phantom-flatten bug would give False / exit 1


def test_trade_tick_trips_on_orphan_holding(monkeypatch, tmp_path):
    """A broker position with no matching paper_venue_fills (orphan / manual holding) must block
    trading — fail-closed, never trades on unattributed state.

    Multi-tenant (new) path: account-level reconcile detects AAPL as pending (first cycle, within
    grace window) → defers instead of immediately tripping a RiskBreach. The key invariant stays:
    the strategy submits zero orders while a reconcile mismatch is open.

    (Old single-tenant path: immediate RiskBreach('reconcile') via venue_belief inside run_tick.
    Grace window added by #316a.)
    """
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "p.db"))
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    _to_paper()
    _seed_allocation_venue(_NAME, capital=100_000.0)

    broker = _PaperVenueTestBroker()
    # Broker holds AAPL with NO matching paper_venue_fills (no orders recorded, no fills ingested)
    broker._positions["AAPL"] = 50.0
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings", lambda: broker)
    from algua.backtest._sample import SyntheticProvider
    monkeypatch.setattr("algua.cli.paper_cmd._select_provider",
                        lambda demo, snapshot: SyntheticProvider())

    result = runner.invoke(app, ["paper", "trade-tick", _NAME, "--snapshot", "snap1",
                                 "--start", _START, "--end", _END])
    # Grace window: first cycle is a DEFER (pending), not a halt/breach. Exits 0.
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload.get("deferred") is True   # account reconcile mismatch → deferred
    # Fail-closed: no orders submitted while reconcile is not clean
    assert broker.submitted_orders == []


# ---------------------------------------------------------------------------
# Task-11: fail-closed evidence + clock-resilience (venue ingest refactor)
# ---------------------------------------------------------------------------

def test_trade_tick_fails_closed_on_ingest_fetch_failure(monkeypatch, tmp_path):
    """Venue fetch failure → exit 1 before run_tick; no tick snapshot recorded.

    Task 11: fail-closed evidence — if account_activities_window raises BrokerError,
    trade_tick must exit non-zero AND record NO tick snapshot (reconcile_ok=True is
    never fabricated for a tick whose fill-state is unknown).
    """
    from contextlib import closing

    from algua.config.settings import get_settings
    from algua.execution.alpaca_broker import BrokerError
    from algua.execution.order_state import latest_tick_snapshot
    from algua.registry.db import connect, migrate

    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "p.db"))
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    _to_paper()

    class _IngestFailBroker:
        def clock(self) -> str:
            return "2024-01-15T14:00:00Z"

        def account(self) -> AccountState:
            return AccountState(equity=100_000.0, cash=100_000.0,
                                buying_power=100_000.0, account_id="fail-acct")

        def account_activities_window(self, after: str, until: str) -> list:
            raise BrokerError("transport error: 503 Service Unavailable")

    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings",
                        lambda: _IngestFailBroker())
    monkeypatch.setattr("algua.cli.paper_cmd._select_provider",
                        lambda demo, snapshot: object())
    # run_tick must NOT be reached — the fail-closed exit happens before it

    result = runner.invoke(app, ["paper", "trade-tick", _NAME, "--snapshot", "snap1",
                                 "--start", _START, "--end", _END])
    assert result.exit_code == 1, result.stdout
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["kind"] == "venue_ingest_failed"

    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        snap = latest_tick_snapshot(conn, _NAME)
    assert snap is None, "tick snapshot must NOT be recorded when venue ingest fails"


def test_trade_tick_survives_broker_clock_failure(monkeypatch, tmp_path):
    """broker.clock() failure falls back to local timestamp; tick does NOT abort.

    Task 11 / clock-resilience: tick_clock catches broker.clock() failures and returns
    a local timestamp, which is then used as the window upper-bound for ingest.
    account_activities_window returns [] → ingest succeeds → tick proceeds → snapshot
    records clock_source='local'.
    """
    from contextlib import closing
    from datetime import UTC, datetime

    from algua.config.settings import get_settings
    from algua.execution.alpaca_broker import BrokerError
    from algua.execution.order_state import latest_tick_snapshot
    from algua.live.live_loop import TickResult
    from algua.registry.db import connect, migrate

    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "p.db"))
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    _to_paper()
    _seed_allocation_venue(_NAME, capital=50_000.0)

    class _ClockFailBroker:
        def clock(self) -> str:
            raise BrokerError("clock endpoint down")

        def account(self) -> AccountState:
            return AccountState(equity=50_000.0, cash=10_000.0,
                                buying_power=40_000.0, account_id="acct-cf")

        def account_activities_window(self, after: str, until: str) -> list:
            return []

        def get_positions(self):
            import pandas as pd
            return pd.Series(dtype="float64")

    fake_result = TickResult(
        decision_ts=datetime(2024, 1, 15, 14, 0, 0, tzinfo=UTC),
        target_weights={},
        positions_before={},
        submitted=[],
        equity=50_000.0,
        peak_equity=50_000.0,
    )
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings",
                        lambda: _ClockFailBroker())
    monkeypatch.setattr("algua.cli.paper_cmd._select_provider",
                        lambda demo, snapshot: object())
    monkeypatch.setattr("algua.cli.paper_cmd.run_tick", lambda *a, **k: fake_result)

    result = runner.invoke(app, ["paper", "trade-tick", _NAME, "--snapshot", "snap1",
                                 "--start", _START, "--end", _END])
    assert result.exit_code == 0, result.stdout

    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        snap = latest_tick_snapshot(conn, _NAME)
    assert snap is not None
    assert snap["clock_source"] == "local", "clock failure must fall back to local, never abort"
    assert snap["tick_ts"]  # a valid local timestamp was written


# ---------------------------------------------------------------------------
# Review-round fixes: FIX #1 (flatten resilient clock) + FIX #3 (non-BrokerError fail-closed)
# ---------------------------------------------------------------------------

def test_trade_tick_fails_closed_on_non_brokererror_ingest_failure(monkeypatch, tmp_path):
    """Non-BrokerError (RuntimeError) from account_activities_window → exit 1; no snapshot.

    FIX #3: the normal-path ingest catch is widened to Exception so a RuntimeError / OSError /
    JSONDecodeError from the broker transport also exits non-zero and emits a structured payload
    rather than crashing with an unhandled traceback (and skipping the audit_append).
    """
    from contextlib import closing

    from algua.config.settings import get_settings
    from algua.execution.order_state import latest_tick_snapshot
    from algua.registry.db import connect, migrate

    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "p.db"))
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    _to_paper()

    class _RuntimeFailBroker:
        def clock(self) -> str:
            return "2024-01-15T14:00:00Z"

        def account(self) -> AccountState:
            return AccountState(equity=100_000.0, cash=100_000.0,
                                buying_power=100_000.0, account_id="rt-fail-acct")

        def account_activities_window(self, after: str, until: str) -> list:
            raise RuntimeError("JSON decode error from broker transport")

    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings",
                        lambda: _RuntimeFailBroker())
    monkeypatch.setattr("algua.cli.paper_cmd._select_provider",
                        lambda demo, snapshot: object())

    result = runner.invoke(app, ["paper", "trade-tick", _NAME, "--snapshot", "snap1",
                                 "--start", _START, "--end", _END])
    assert result.exit_code == 1, result.stdout
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["kind"] == "venue_ingest_failed"

    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        snap = latest_tick_snapshot(conn, _NAME)
    assert snap is None, "tick snapshot must NOT be recorded when venue ingest fails"


def test_flatten_still_offsets_when_broker_clock_raises(monkeypatch, tmp_path):
    """FIX #1: flatten's ingest now uses tick_clock (resilient fallback), so a BrokerError
    from broker.clock() does NOT abort the offset loop — the position is still liquidated.
    """
    from contextlib import closing

    from algua.execution.alpaca_broker import BrokerError
    from algua.registry.db import connect, migrate

    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "p.db"))
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    _to_paper()

    # Seed a believed position so the offset loop has something to liquidate
    with closing(connect(tmp_path / "p.db")) as conn:
        migrate(conn)
        conn.execute(
            "INSERT INTO paper_venue_fills"
            "(activity_id, broker_order_id, strategy, symbol, qty, price, fill_ts)"
            " VALUES (?,?,?,?,?,?,?)",
            ("fill-a1", "boid-a1", _NAME, "AAA", 5.0, 100.0, "2024-01-01T00:00:00Z"),
        )
        conn.commit()

    offset_calls: list = []

    class _ClockDownBroker:
        def clock(self) -> str:
            raise BrokerError("clock outage")

        def account_activities_window(self, after: str, until: str) -> list:
            return []

        def cancel_open_orders(self) -> None:
            pass

        def submit_offset(self, sym: str, qty: float, coid: str) -> str:
            offset_calls.append((sym, qty, coid))
            return f"o-{sym}"

    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings",
                        lambda: _ClockDownBroker())

    result = runner.invoke(app, ["paper", "flatten", _NAME])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["liquidation_submitted"] is True
    assert any(sym == "AAA" for sym, _, _ in offset_calls), (
        "clock outage must not abort the offset loop — submit_offset must still be called"
    )
