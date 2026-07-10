import json
from datetime import UTC, datetime, timedelta

import pytest
from typer.testing import CliRunner

from algua.cli._common import resolve_wall_clock_window
from algua.cli.main import app
from algua.contracts.types import LiveAuthorization

runner = CliRunner()


# --- resolve_wall_clock_window tests (#452 Layer B) ---

def test_resolve_wall_clock_window_both_none():
    """When both start and end are None, resolve to a recent rolling window."""
    start, end = resolve_wall_clock_window(None, None)
    today = datetime.now(UTC).date()
    end_date = datetime.fromisoformat(end).date()
    start_date = datetime.fromisoformat(start).date()

    assert end_date == today, f"end should be today ({today}), got {end_date}"
    # start should be approximately 400 days ago
    expected_start = today - timedelta(days=400)
    assert start_date == expected_start, (
        f"start should be ~400 days ago ({expected_start}), got {start_date}")


def test_resolve_wall_clock_window_explicit_values():
    """When start and end are explicitly provided, pass them through unchanged."""
    explicit_start = "2024-01-01"
    explicit_end = "2024-12-31"
    start, end = resolve_wall_clock_window(explicit_start, explicit_end)

    assert start == explicit_start
    assert end == explicit_end


def test_resolve_wall_clock_window_partial_explicit():
    """When only one of start/end is provided, resolve only the missing one."""
    explicit_start = "2024-01-01"
    start, end = resolve_wall_clock_window(explicit_start, None)

    assert start == explicit_start
    today = datetime.now(UTC).date()
    end_date = datetime.fromisoformat(end).date()
    assert end_date == today


def test_resolve_wall_clock_window_only_end_explicit():
    """When only end is provided, resolve only start."""
    explicit_end = "2024-12-31"
    start, end = resolve_wall_clock_window(None, explicit_end)

    assert end == explicit_end
    # start should be resolved to ~400 days before today (not before end)
    today = datetime.now(UTC).date()
    start_date = datetime.fromisoformat(start).date()
    expected_start = today - timedelta(days=400)
    assert start_date == expected_start


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "p.db"))
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ALGUA_ALPACA_LIVE_API_KEY", "lk")
    monkeypatch.setenv("ALGUA_ALPACA_LIVE_API_SECRET", "ls")


def _to_live(name="cross_sectional_momentum", allocate=True):
    # bring a strategy to 'live' stage in the DB directly (the signed ceremony is tested elsewhere).
    # Under the #497 inverted flow a strategy enters `live` UNALLOCATED and the human allocates it
    # afterward; `live run-all` SKIPS an unallocated live strategy. So by default this helper also
    # seeds a live allocation (the normal post-`live allocate` state a run-all cycle ticks against).
    # Pass allocate=False to leave it unallocated (to exercise the skip path).
    from contextlib import closing
    from datetime import UTC, datetime

    from algua.config.settings import get_settings
    from algua.registry.db import connect, migrate
    from algua.registry.store import SqliteStrategyRepository
    assert runner.invoke(app, ["backtest", "run", name, "--demo", "--register",
                               "--start", "2022-01-01", "--end", "2023-12-31"]).exit_code == 0
    # CANDIDATE via human: scaffolding to live, not exercising the agent shortlist gate.
    for to, actor in (("candidate", "human"), ("paper", "agent")):
        runner.invoke(app, ["registry", "transition", name, "--to", to, "--actor", actor,
                            "--reason", "x"])
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        conn.execute("UPDATE strategies SET stage='live' WHERE name=?", (name,))
        if allocate:
            sid = SqliteStrategyRepository(conn).get(name).id
            conn.execute(
                "INSERT INTO strategy_allocations(strategy_id, capital, effective_ts, actor) "
                "VALUES (?,?,?,?)",
                (sid, 10_000.0, datetime.now(UTC).isoformat(), "human"))
        conn.commit()


def _auth():
    return LiveAuthorization(1, "c", "cf", "d", "lior", "t")


def _permissive_book(monkeypatch):
    """Stub the #389 book-level exposure build with a large-equity, empty long-only book so book
    caps never bind, AND no-op the #390 book-level LOSS circuit breaker — for run-all tests that
    assert the pre-existing tick/pool/breach behavior. (Book-exposure enforcement is covered by
    test_book_limits.py + test_live_book_limits.py; the loss breaker by test_book_breaker.py +
    test_live_book_breaker.py.)"""
    from algua.risk.book_limits import BookExposure, BookRiskLimits
    monkeypatch.setattr(
        "algua.cli.live_cmd._build_book_exposure",
        lambda *a, **k: (BookExposure(1e15, {}, BookRiskLimits()), None),
    )
    monkeypatch.setattr(
        "algua.cli.live_cmd._evaluate_book_loss_breaker", lambda *a, **k: None
    )



def test_run_all_no_live_strategies_is_noop(monkeypatch):
    r = runner.invoke(app, ["live", "run-all", "--snapshot", "x"])
    assert r.exit_code == 0
    assert json.loads(r.stdout)["strategies"] == []


def test_run_all_no_authorized_returns_without_building_broker(monkeypatch):
    # #253: the REAL authorization gate runs here (NOT monkeypatched). A live-stage strategy with no
    # valid live_authorizations row must be skipped; with zero verified strategies, run-all returns
    # "no authorized live strategies" and must NOT construct the broker or submit any order — the
    # last barrier before live trading. (Every other run-all test fakes a passing auth, so this is
    # the only coverage that the fail-closed wiring actually fails closed.)
    _to_live("cross_sectional_momentum")  # stage=live, but no signed authorization row exists
    built = {"broker": False}
    monkeypatch.setattr("algua.cli.live_cmd._alpaca_live_broker",
                        lambda auth: built.__setitem__("broker", True))
    r = runner.invoke(app, ["live", "run-all", "--snapshot", "x"])
    assert r.exit_code == 0, r.stdout
    payload = json.loads(r.stdout)
    assert payload["note"] == "no authorized live strategies"
    assert payload["strategies"] == []
    assert [s["strategy"] for s in payload["skipped"]] == ["cross_sectional_momentum"]
    assert built["broker"] is False  # early return precedes broker construction (no trade)


def test_run_all_skips_only_the_unauthorized_strategy(monkeypatch):
    _permissive_book(monkeypatch)
    # #253 sibling: with a mix, the unauthorized strategy is skipped and the authorized one still
    # trades — an unverified strategy must never leak into the traded set.
    from contextlib import closing

    from algua.config.settings import get_settings
    from algua.registry.db import connect, migrate
    from algua.registry.live_gate import LiveAuthorizationError
    _to_live("cross_sectional_momentum")  # the authorized one
    # a second live-stage strategy with no authorization (registry-only; skipped before load)
    assert runner.invoke(app, ["registry", "add", "phantom_live"]).exit_code == 0
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        conn.execute("UPDATE strategies SET stage='live' WHERE name='phantom_live'")
        conn.commit()

    def _selective(conn, repo, name, signers_path):
        if name == "cross_sectional_momentum":
            return _auth()
        raise LiveAuthorizationError(f"{name}: no valid authorization")

    monkeypatch.setattr("algua.cli.live_cmd.verify_live_authorization", _selective)
    monkeypatch.setattr("algua.cli.live_cmd._alpaca_live_broker", lambda auth: object())
    monkeypatch.setattr("algua.cli.live_cmd._select_provider", lambda demo, snapshot: object())
    monkeypatch.setattr("algua.cli.live_cmd.ingest_activities", lambda conn, acts, kind: None)
    monkeypatch.setattr("algua.cli.live_cmd.fill_cursor", lambda conn, kind: None)
    monkeypatch.setattr("algua.cli.live_cmd._broker_account_activities", lambda broker, after: [])
    monkeypatch.setattr("algua.cli.live_cmd._broker_net_positions", lambda broker: {})
    monkeypatch.setattr("algua.cli.live_cmd._broker_buying_power", lambda broker: 100_000.0)
    monkeypatch.setattr("algua.cli.live_cmd._run_strategy_tick",
                        lambda *a, **k: {"strategy": "cross_sectional_momentum", "submitted": []})
    r = runner.invoke(app, ["live", "run-all", "--snapshot", "x"])
    assert r.exit_code == 0, r.stdout
    payload = json.loads(r.stdout)
    traded = [s["strategy"] for s in payload["strategies"]]
    assert "cross_sectional_momentum" in traded
    assert [s["strategy"] for s in payload["skipped"]] == ["phantom_live"]
    assert "phantom_live" not in traded


def _allocate(name="cross_sectional_momentum", capital=10_000.0):
    from contextlib import closing

    from algua.config.settings import get_settings
    from algua.registry import allocations
    from algua.registry.db import connect, migrate
    from algua.registry.store import SqliteStrategyRepository
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        with conn:
            allocations.allocate_locked(
                conn, SqliteStrategyRepository(conn).get(name).id, capital, "human", 50_000.0)


def _bench_to_dormant(name="cross_sectional_momentum"):
    from contextlib import closing

    from algua.config.settings import get_settings
    from algua.contracts.lifecycle import Actor, Stage
    from algua.registry.db import connect, migrate
    from algua.registry.store import SqliteStrategyRepository
    from algua.registry.transitions import transition_strategy
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        transition_strategy(SqliteStrategyRepository(conn), name, Stage.DORMANT, Actor.AGENT,
                            reason="benched mid-cycle")


def test_still_live_allocated_false_after_bench(monkeypatch):
    # #281: the submit-time guard reflects a live->dormant bench (stage flips + allocation revoked).
    from contextlib import closing

    from algua.cli.live_cmd import _still_live_allocated
    from algua.config.settings import get_settings
    from algua.registry.db import connect, migrate
    _to_live()
    _allocate()
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        assert _still_live_allocated(conn, "cross_sectional_momentum") is True
    _bench_to_dormant()
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        assert _still_live_allocated(conn, "cross_sectional_momentum") is False


def test_run_all_halts_strategy_benched_to_dormant_mid_cycle(monkeypatch):
    _permissive_book(monkeypatch)
    # #281: a live->dormant bench landing MID-CYCLE must halt the tick via should_halt — never
    # submit/persist an order for the now-dormant strategy (which run-all, iterating Stage.LIVE,
    # would never flatten). Simulate the concurrent bench committing inside run_tick.
    from contextlib import closing

    from algua.config.settings import get_settings
    from algua.live.live_loop import TickHalted
    from algua.registry.db import connect, migrate
    _to_live()
    _allocate()
    monkeypatch.setattr("algua.cli.live_cmd.verify_live_authorization", lambda *a, **k: _auth())
    monkeypatch.setattr("algua.cli.live_cmd._alpaca_live_broker", lambda auth: object())
    monkeypatch.setattr("algua.cli.live_cmd._select_provider", lambda demo, snapshot: object())
    monkeypatch.setattr("algua.cli.live_cmd.ingest_activities", lambda conn, acts, kind: None)
    monkeypatch.setattr("algua.cli.live_cmd.fill_cursor", lambda conn, kind: None)
    monkeypatch.setattr("algua.cli.live_cmd._broker_account_activities", lambda b, a: [])
    monkeypatch.setattr("algua.cli.live_cmd._broker_net_positions", lambda b: {})
    monkeypatch.setattr("algua.cli.live_cmd._broker_buying_power", lambda b: 100_000.0)

    seen = {}

    def _fake_run_tick(strategy, broker, provider, start, end, hooks=None, max_drawdown=None):
        _bench_to_dormant()  # concurrent live->dormant bench commits mid-tick (revokes allocation)
        seen["halt"] = hooks.should_halt()
        if seen["halt"]:
            raise TickHalted("benched mid-cycle")
        raise AssertionError("should_halt must trip after a mid-cycle bench")  # would-be order path

    monkeypatch.setattr("algua.cli.live_cmd.run_tick", _fake_run_tick)
    r = runner.invoke(app, ["live", "run-all", "--snapshot", "x"])
    assert seen["halt"] is True          # the submit-time guard saw the mid-cycle bench
    assert r.exit_code == 1              # halted -> the cycle aborts (no orders)
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        n = conn.execute("SELECT COUNT(*) FROM live_orders WHERE strategy=?",
                         ("cross_sectional_momentum",)).fetchone()[0]
    assert n == 0  # no order persisted for the now-dormant strategy


def test_run_all_halts_on_unexplained_reconcile_drift(monkeypatch):
    from algua.risk import global_halt
    _to_live()
    monkeypatch.setattr("algua.cli.live_cmd.verify_live_authorization", lambda *a, **k: _auth())
    monkeypatch.setattr("algua.cli.live_cmd._alpaca_live_broker", lambda auth: object())
    monkeypatch.setattr("algua.cli.live_cmd._select_provider", lambda demo, snapshot: object())
    # broker activities -> none; positions -> an unexplained holding the books don't have
    monkeypatch.setattr("algua.cli.live_cmd.ingest_activities", lambda conn, acts, kind: None)
    monkeypatch.setattr("algua.cli.live_cmd.fill_cursor", lambda conn, kind: None)
    monkeypatch.setattr("algua.cli.live_cmd._broker_account_activities", lambda broker, after: [])
    monkeypatch.setattr("algua.cli.live_cmd._broker_net_positions", lambda broker: {"ZZZ": 99.0})
    # --grace-cycles 0 forces the mismatch straight to unexplained -> assert global halt engaged
    r = runner.invoke(app, ["live", "run-all", "--snapshot", "x", "--grace-cycles", "0"])
    assert r.exit_code == 1
    payload = json.loads(r.stdout)
    assert payload["ok"] is False and payload["reconcile"]["halt"] is True
    assert payload["skipped_unallocated"] == []  # #497 F1: threaded into reconcile-halt
    from contextlib import closing

    from algua.config.settings import get_settings
    from algua.registry.db import connect, migrate
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        assert global_halt.is_engaged(conn)


def test_run_all_ticks_strategy_when_clean(monkeypatch):
    _permissive_book(monkeypatch)
    _to_live()
    monkeypatch.setattr("algua.cli.live_cmd.verify_live_authorization", lambda *a, **k: _auth())
    monkeypatch.setattr("algua.cli.live_cmd._alpaca_live_broker", lambda auth: object())
    monkeypatch.setattr("algua.cli.live_cmd._select_provider", lambda demo, snapshot: object())
    monkeypatch.setattr("algua.cli.live_cmd.ingest_activities", lambda conn, acts, kind: None)
    monkeypatch.setattr("algua.cli.live_cmd.fill_cursor", lambda conn, kind: None)
    monkeypatch.setattr("algua.cli.live_cmd._broker_account_activities", lambda broker, after: [])
    monkeypatch.setattr("algua.cli.live_cmd._broker_net_positions", lambda broker: {})
    monkeypatch.setattr("algua.cli.live_cmd._broker_buying_power", lambda broker: 100_000.0)
    monkeypatch.setattr("algua.cli.live_cmd._run_strategy_tick",
                        lambda *a, **k: {"strategy": "cross_sectional_momentum", "submitted": []})
    r = runner.invoke(app, ["live", "run-all", "--snapshot", "x"])
    assert r.exit_code == 0
    payload = json.loads(r.stdout)
    assert payload["reconcile"]["clean"] is True
    assert payload["strategies"][0]["strategy"] == "cross_sectional_momentum"


def _force_live_stage(name: str) -> None:
    """Force a fake (module-less) strategy to stage='live' directly in the DB. Under #497 the
    LIVE-gate on `live allocate` requires stage=='live', so a fake strategy must be forced live
    before it can be allocated."""
    from contextlib import closing

    from algua.config.settings import get_settings
    from algua.registry.db import connect, migrate
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        conn.execute("UPDATE strategies SET stage='live' WHERE name=?", (name,))
        conn.commit()


def test_live_allocate_records_and_enforces_sum(monkeypatch):
    from contextlib import closing

    from algua.config.settings import get_settings
    from algua.registry.db import connect, migrate
    monkeypatch.setattr("algua.cli.live_cmd._live_account_equity", lambda: 50_000.0)
    assert runner.invoke(app, ["registry", "add", "s1"]).exit_code == 0
    _force_live_stage("s1")
    r = runner.invoke(app, ["live", "allocate", "s1", "--capital", "10000"])
    assert r.exit_code == 0, r.stdout
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        n = conn.execute("SELECT COUNT(*) FROM strategy_allocations WHERE revoked_ts IS NULL"
                         ).fetchone()[0]
        assert n == 1
    # over-commit refused
    runner.invoke(app, ["registry", "add", "s2"])
    _force_live_stage("s2")
    r2 = runner.invoke(app, ["live", "allocate", "s2", "--capital", "45000"])
    assert r2.exit_code == 1 and json.loads(r2.stdout)["ok"] is False


def test_run_all_rejects_bad_max_drawdown():
    r = runner.invoke(app, ["live", "run-all", "--snapshot", "x", "--max-drawdown", "1.5"])
    assert r.exit_code == 1 and json.loads(r.stdout)["ok"] is False


def test_run_all_breach_liquidates_per_strategy(monkeypatch):
    from algua.live.live_loop import RiskBreach
    name = "cross_sectional_momentum"
    _to_live(name)
    # Seed the live ledger with AAA position so reconciliation is clean before the breach
    _seed_live_fills(name, {"AAA": 5.0})
    _permissive_book(monkeypatch)
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
            self.offsets.append((symbol, qty))
            return "off"
        def account(self):
            from algua.execution.alpaca_broker import AccountState
            # last_equity == equity so the #390 book loss breaker passes; this test exercises the
            # PER-STRATEGY breach path (run_tick raising), not the book-level breaker.
            return AccountState(equity=100_000.0, cash=100_000.0, buying_power=100_000.0,
                                last_equity=100_000.0)

    broker = _LiqBroker()
    monkeypatch.setattr("algua.cli.live_cmd._alpaca_live_broker", lambda auth: broker)
    monkeypatch.setattr("algua.cli.live_cmd._select_provider", lambda demo, snapshot: object())
    monkeypatch.setattr("algua.cli.live_cmd._broker_account_activities", lambda b, a: [])
    # #449: _broker_net_positions now feeds the held-cap in flatten_strategy, so mock it to return
    # the actual held position (matching the believed position so the offset is submitted).
    monkeypatch.setattr("algua.cli.live_cmd._broker_net_positions", lambda b: {"AAA": 5.0})
    # #336: the believed-position read moved into the single-sourced flatten helper, so patch it
    # where the helper resolves it (algua.execution.flatten), not the old live_cmd call site.
    monkeypatch.setattr("algua.execution.flatten.believed_positions",
                        lambda conn, name, kind: {"AAA": 5.0})  # strategy believes it holds 5 AAA
    monkeypatch.setattr("algua.cli.live_cmd.active_allocation",
                        lambda conn, sid: {"capital": 10_000.0})
    monkeypatch.setattr("algua.cli.live_cmd.run_tick",
                        lambda *a, **k: (_ for _ in ()).throw(RiskBreach("drawdown", "dd")))
    r = runner.invoke(app, ["live", "run-all", "--snapshot", "x"])
    assert r.exit_code == 1
    assert broker.offsets == [("AAA", 5.0)]  # offset sized to the believed qty (capped to held)
    # the offset is RECORDED in the books (+backfilled) so its fill attributes back to the strategy
    # and believed_positions can drop to flat — else the resume gate blocks resume forever (codex)
    from contextlib import closing

    from algua.config.settings import get_settings
    from algua.registry.db import connect, migrate
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        row = conn.execute(
            "SELECT side, broker_order_id FROM live_orders "
            "WHERE strategy = 'cross_sectional_momentum' AND symbol = 'AAA'"
        ).fetchone()
        assert row["side"] == "sell" and row["broker_order_id"] == "off"


def test_run_all_flattens_not_skips_on_ledger_nav_wipe(monkeypatch):
    # #452 regression: a live book whose LEDGER NAV resolves to <= 0 off TRUSTWORTHY, FRESH marks
    # is a genuine economic WIPE — it must route to trip + FLATTEN (economic-breach handler), NOT to
    # a silent LiveSizingError {"skipped": ...} envelope that leaves the position dangling. Drives
    # the REAL CLI lane (_run_strategy_tick -> run_tick -> build_live_sizing_snapshot): the sizing
    # snapshot RETURNS the non-positive-equity book, run_tick's uniform guard raises
    # RiskBreach('non_positive_equity'), and the handler flattens. (Before the fix, build_sizing_
    # snapshot raised LiveSizingError here and the lane SKIPPED instead of flattening.)
    import pandas as pd

    from algua.calendar.market_calendar import MarketCalendar
    name = "cross_sectional_momentum"
    _to_live(name)
    # Held 5 AAA at $100 cost (seed price). At a fresh $1 mark unrealized = 5*(1-100) = -$495, so
    # with a $100 allocation NAV = 100 - 495 = -$395 <= 0 -> the sizing equity is a genuine wipe.
    _seed_live_fills(name, {"AAA": 5.0})
    _permissive_book(monkeypatch)
    monkeypatch.setattr("algua.cli.live_cmd.verify_live_authorization", lambda *a, **k: _auth())

    # A fresh (last completed session), positive, finite mark: passes assert_marks_usable so the
    # book is TRUSTED, isolating the NAV<=0 economic path (not a stale/unvaluable dark-feed HALT).
    cal = MarketCalendar()
    bar_day = cal.previous_session(datetime.now(UTC).date())
    bar_ts = datetime(bar_day.year, bar_day.month, bar_day.day, tzinfo=UTC)

    class _FreshBarProvider:
        def get_bars(self, symbols, start, end, timeframe):
            return pd.DataFrame(
                [{"timestamp": bar_ts, "symbol": "AAA", "open": 1.0, "high": 1.0, "low": 1.0,
                  "close": 1.0, "adj_close": 1.0, "volume": 1000}]
            ).set_index("timestamp").sort_index()

    class _LiqBroker:
        def __init__(self):
            self.offsets = []
        def account_activities(self, after=None):
            return []
        def get_positions(self):
            return pd.Series(dtype=float)
        def list_open_orders(self):
            return []
        def cancel_order(self, oid):
            pass
        def submit_offset(self, symbol, qty, coid):
            self.offsets.append((symbol, qty))
            return "off"
        def account(self):
            from algua.execution.alpaca_broker import AccountState
            return AccountState(equity=100_000.0, cash=100_000.0, buying_power=100_000.0,
                                last_equity=100_000.0)

    broker = _LiqBroker()
    monkeypatch.setattr("algua.cli.live_cmd._alpaca_live_broker", lambda auth: broker)
    monkeypatch.setattr("algua.cli.live_cmd._select_provider",
                        lambda demo, snapshot: _FreshBarProvider())
    monkeypatch.setattr("algua.cli.live_cmd._broker_account_activities", lambda b, a: [])
    monkeypatch.setattr("algua.cli.live_cmd._broker_net_positions", lambda b: {"AAA": 5.0})
    monkeypatch.setattr("algua.execution.flatten.believed_positions",
                        lambda conn, name, kind: {"AAA": 5.0})
    monkeypatch.setattr("algua.cli.live_cmd.active_allocation",
                        lambda conn, sid: {"capital": 100.0})

    r = runner.invoke(app, ["live", "run-all", "--snapshot", "x"])
    assert r.exit_code == 1, r.stdout
    payload = json.loads(r.stdout)
    strategies = payload["strategies"]
    assert len(strategies) == 1
    marker = strategies[0]
    # FLATTEN, not skip: an economic-breach marker with the non_positive_equity kind + liquidation.
    assert marker["ok"] is False
    assert marker["kind"] == "non_positive_equity"
    assert marker["liquidation_submitted"] is True
    assert "skipped" not in marker  # NOT a LiveSizingError skip envelope
    assert broker.offsets == [("AAA", 5.0)]  # the wiped book was actually liquidated


def test_run_all_book_loss_breaker_halts_and_flattens_whole_account(monkeypatch):
    # #390: a whole-account daily loss past the cap engages the global halt and flattens the ENTIRE
    # account (close_all_positions), BEFORE any strategy ticks. run_tick must never be reached.
    _to_live()
    monkeypatch.setattr("algua.cli.live_cmd.verify_live_authorization", lambda *a, **k: _auth())

    class _LossBroker:
        def __init__(self):
            self.closed_all = False
        def account_activities(self, after=None):
            return []
        def get_positions(self):
            import pandas as pd
            return pd.Series(dtype=float)
        def close_all_positions(self):
            self.closed_all = True
        def account(self):
            from algua.execution.alpaca_broker import AccountState
            # equity 10% below prior-session close (last_equity) — past the 5% daily-loss cap.
            return AccountState(equity=90_000.0, cash=0.0, buying_power=0.0,
                                last_equity=100_000.0)

    broker = _LossBroker()
    monkeypatch.setattr("algua.cli.live_cmd._alpaca_live_broker", lambda auth: broker)
    monkeypatch.setattr("algua.cli.live_cmd._select_provider", lambda demo, snapshot: object())
    monkeypatch.setattr("algua.cli.live_cmd._broker_account_activities", lambda b, a: [])
    monkeypatch.setattr("algua.cli.live_cmd._broker_net_positions", lambda b: {})
    monkeypatch.setattr("algua.cli.live_cmd.active_allocation",
                        lambda conn, sid: {"capital": 10_000.0})

    def _no_tick(*a, **k):
        raise AssertionError("run_tick must not run once the book loss breaker trips")
    monkeypatch.setattr("algua.cli.live_cmd.run_tick", _no_tick)

    r = runner.invoke(app, ["live", "run-all", "--snapshot", "x"])
    assert r.exit_code == 1
    import json
    payload = json.loads(r.stdout)
    assert payload["ok"] is False
    assert payload["book_breach"]["kind"] == "book_daily_loss"
    assert payload["global_halt"] == "set"
    assert payload["liquidation_submitted"] is True
    assert broker.closed_all is True
    assert payload["skipped_unallocated"] == []  # #497 F1: threaded into book-loss-halt

    # the global halt is now persisted: a subsequent cycle refuses at the top-of-cycle check
    from contextlib import closing

    from algua.config.settings import get_settings
    from algua.registry.db import connect, migrate
    from algua.risk import global_halt
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        assert global_halt.is_engaged(conn) is True


def test_run_all_breach_preserves_already_ticked_results(monkeypatch):
    _permissive_book(monkeypatch)
    # #270: when a LATER strategy breaches, run-all must still surface the siblings already ticked
    # this cycle in one envelope (not discard them by emitting only the breach payload).
    from contextlib import closing

    from algua.cli._common import breach_payload
    from algua.config.settings import get_settings
    from algua.registry.db import connect, migrate
    _to_live("cross_sectional_momentum")
    # A second live strategy row (the module is never loaded — verify_live_authorization and
    # _run_strategy_tick are both mocked; we only need two rows for repo.list_strategies(LIVE)).
    assert runner.invoke(app, ["registry", "add", "second_live"]).exit_code == 0
    with closing(connect(get_settings().db_path)) as conn:
        from datetime import UTC, datetime

        from algua.registry.store import SqliteStrategyRepository
        migrate(conn)
        conn.execute("UPDATE strategies SET stage='live' WHERE name='second_live'")
        # #497: run-all skips an UNALLOCATED live strategy; seed an allocation so this second live
        # strategy is actually ticked (and breaches) rather than skipped.
        sid = SqliteStrategyRepository(conn).get("second_live").id
        conn.execute(
            "INSERT INTO strategy_allocations(strategy_id, capital, effective_ts, actor) "
            "VALUES (?,?,?,?)",
            (sid, 10_000.0, datetime.now(UTC).isoformat(), "human"))
        conn.commit()
    monkeypatch.setattr("algua.cli.live_cmd.verify_live_authorization", lambda *a, **k: _auth())
    monkeypatch.setattr("algua.cli.live_cmd._alpaca_live_broker", lambda auth: object())
    monkeypatch.setattr("algua.cli.live_cmd._select_provider", lambda demo, snapshot: object())
    monkeypatch.setattr("algua.cli.live_cmd._broker_account_activities", lambda b, a: [])
    monkeypatch.setattr("algua.cli.live_cmd._broker_net_positions", lambda b: {})
    monkeypatch.setattr("algua.cli.live_cmd._broker_buying_power", lambda b: 100_000.0)

    calls: list[str] = []

    def _fake_tick(conn, name, auth, broker, provider, max_drawdown, start=None, end=None,
                   reserve_buy=None, cancel=None):
        calls.append(name)
        if len(calls) == 1:
            return {"strategy": name, "venue": "live", "submitted": []}  # first ticks clean
        return breach_payload("dd", strategy=name, kind="drawdown")       # second breaches

    monkeypatch.setattr("algua.cli.live_cmd._run_strategy_tick", _fake_tick)
    r = runner.invoke(app, ["live", "run-all", "--snapshot", "x"])
    assert r.exit_code == 1
    payload = json.loads(r.stdout)
    assert payload["ok"] is False
    strategies = payload["strategies"]
    assert len(strategies) == 2  # the clean sibling is preserved ALONGSIDE the breaching one
    assert strategies[0].get("ok") is not False and strategies[0]["strategy"] == calls[0]
    assert strategies[1]["ok"] is False and strategies[1]["error"] == "dd"
    assert strategies[1]["strategy"] == calls[1]


def test_run_all_reserves_buying_power_across_strategies(monkeypatch):
    _permissive_book(monkeypatch)
    _to_live()
    captured = {}
    monkeypatch.setattr("algua.cli.live_cmd.verify_live_authorization", lambda *a, **k: _auth())
    monkeypatch.setattr("algua.cli.live_cmd._alpaca_live_broker", lambda auth: object())
    monkeypatch.setattr("algua.cli.live_cmd._select_provider", lambda demo, snapshot: object())
    monkeypatch.setattr("algua.cli.live_cmd._broker_account_activities", lambda b, a: [])
    monkeypatch.setattr("algua.cli.live_cmd._broker_net_positions", lambda b: {})
    monkeypatch.setattr("algua.cli.live_cmd._broker_buying_power", lambda b: 30_000.0)

    def _fake_tick(conn, name, auth, broker, provider, max_drawdown, start=None, end=None,
                   reserve_buy=None, cancel=None):
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


def test_live_allocate_rejects_dormant(monkeypatch):
    from contextlib import closing

    from algua.config.settings import get_settings
    from algua.contracts.lifecycle import Actor, Stage
    from algua.registry.db import connect, migrate
    from algua.registry.store import SqliteStrategyRepository
    from algua.registry.transitions import transition_strategy

    # Make the equity/broker call blow up: the dormant guard must reject BEFORE it is reached,
    # so this must never fire (proves the stage check precedes the network call).
    def _boom() -> float:
        raise AssertionError("_live_account_equity must not be called for a dormant strategy")
    monkeypatch.setattr("algua.cli.live_cmd._live_account_equity", _boom)
    # register a strategy and drive it to paper via the legal chain
    assert runner.invoke(app, ["registry", "add", "s1"]).exit_code == 0
    for to, actor in (("backtested", "human"), ("candidate", "human"), ("paper", "agent")):
        r = runner.invoke(app, ["registry", "transition", "s1", "--to", to,
                                "--actor", actor, "--reason", "x"])
        assert r.exit_code == 0, r.stdout
    # move to dormant directly via the python API (paper -> dormant, any actor, requires reason)
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        repo = SqliteStrategyRepository(conn)
        transition_strategy(repo, "s1", Stage.DORMANT, Actor.AGENT, reason="bench")
    # live allocate must refuse
    r = runner.invoke(app, ["live", "allocate", "s1", "--capital", "10000"])
    assert r.exit_code != 0
    assert "dormant" in r.output.lower()


def test_run_all_skips_unallocated_live_strategy(monkeypatch):
    # #497 inverted flow: a strategy enters `live` UNALLOCATED. `live run-all` must SKIP it (not
    # crash on the no-allocation raise), report it under skipped_unallocated, NOT tick it, exit 0.
    name = "cross_sectional_momentum"
    _permissive_book(monkeypatch)
    _to_live(name, allocate=False)  # drives to live but does NOT allocate
    monkeypatch.setattr("algua.cli.live_cmd.verify_live_authorization", lambda *a, **k: _auth())
    monkeypatch.setattr("algua.cli.live_cmd._alpaca_live_broker", lambda auth: object())
    monkeypatch.setattr("algua.cli.live_cmd._select_provider", lambda demo, snapshot: object())
    monkeypatch.setattr("algua.cli.live_cmd.ingest_activities", lambda conn, acts, kind: None)
    monkeypatch.setattr("algua.cli.live_cmd.fill_cursor", lambda conn, kind: None)
    monkeypatch.setattr("algua.cli.live_cmd._broker_account_activities", lambda broker, after: [])
    monkeypatch.setattr("algua.cli.live_cmd._broker_net_positions", lambda broker: {})
    monkeypatch.setattr("algua.cli.live_cmd._broker_buying_power", lambda broker: 100_000.0)

    def _must_not_tick(*a, **k):
        raise AssertionError("an unallocated live strategy must never be ticked")
    monkeypatch.setattr("algua.cli.live_cmd._run_strategy_tick", _must_not_tick)

    r = runner.invoke(app, ["live", "run-all", "--snapshot", "x"])
    assert r.exit_code == 0, r.stdout
    payload = json.loads(r.stdout)
    assert payload["skipped_unallocated"] == [name]
    assert payload["strategies"] == []


def test_run_all_reconcile_halt_carries_unallocated_sibling(monkeypatch):
    # #497 F1: the reconcile-halt envelope must carry the skipped_unallocated list, not just the
    # loop-completes path — an unallocated live sibling stays visible even when the cycle halts.
    from contextlib import closing

    from algua.config.settings import get_settings
    from algua.registry.db import connect, migrate
    _to_live("cross_sectional_momentum")  # allocated
    assert runner.invoke(app, ["registry", "add", "unalloc_sib"]).exit_code == 0
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        conn.execute("UPDATE strategies SET stage='live' WHERE name='unalloc_sib'")
        conn.commit()
    monkeypatch.setattr("algua.cli.live_cmd.verify_live_authorization", lambda *a, **k: _auth())
    monkeypatch.setattr("algua.cli.live_cmd._alpaca_live_broker", lambda auth: object())
    monkeypatch.setattr("algua.cli.live_cmd._select_provider", lambda demo, snapshot: object())
    monkeypatch.setattr("algua.cli.live_cmd.ingest_activities", lambda conn, acts, kind: None)
    monkeypatch.setattr("algua.cli.live_cmd.fill_cursor", lambda conn, kind: None)
    monkeypatch.setattr("algua.cli.live_cmd._broker_account_activities", lambda broker, after: [])
    monkeypatch.setattr("algua.cli.live_cmd._broker_net_positions", lambda broker: {"ZZZ": 99.0})
    r = runner.invoke(app, ["live", "run-all", "--snapshot", "x", "--grace-cycles", "0"])
    assert r.exit_code == 1
    payload = json.loads(r.stdout)
    assert payload["reconcile"]["halt"] is True
    assert payload["skipped_unallocated"] == ["unalloc_sib"]


def test_run_all_forwards_start_end_to_tick(monkeypatch):
    _permissive_book(monkeypatch)
    # operator-supplied --start/--end must reach the per-strategy tick (codex C2: were ignored)
    _to_live()
    captured = {}
    monkeypatch.setattr("algua.cli.live_cmd.verify_live_authorization", lambda *a, **k: _auth())
    monkeypatch.setattr("algua.cli.live_cmd._alpaca_live_broker", lambda auth: object())
    monkeypatch.setattr("algua.cli.live_cmd._select_provider", lambda demo, snapshot: object())
    monkeypatch.setattr("algua.cli.live_cmd._broker_account_activities", lambda b, a: [])
    monkeypatch.setattr("algua.cli.live_cmd._broker_net_positions", lambda b: {})
    monkeypatch.setattr("algua.cli.live_cmd._broker_buying_power", lambda b: 1_000.0)

    def _fake_tick(conn, name, auth, broker, provider, max_drawdown, start=None, end=None,
                   reserve_buy=None, cancel=None):
        captured["start"], captured["end"] = start, end
        return {"strategy": name}

    monkeypatch.setattr("algua.cli.live_cmd._run_strategy_tick", _fake_tick)
    r = runner.invoke(app, ["live", "run-all", "--snapshot", "x", "--start", "2021-01-01",
                            "--end", "2021-12-31"])
    assert r.exit_code == 0
    assert captured == {"start": "2021-01-01", "end": "2021-12-31"}


class _BreachThenFlatBroker:
    """Fake live broker for the full breach->flatten->resume chain. Before the offsets are
    submitted it reports the seeded holdings (AAA+5, ZZZ+3) and no new activities; once the breach
    handler submits the offsets it flips to flat and its activity feed returns the offset FILLs."""
    def __init__(self):
        self.offsets = []
        self._closed = False
    def account_activities(self, after=None):
        if not self._closed:
            return []
        return [
            {"id": f"act-off-{sym}", "activity_type": "FILL", "side": "sell",
             "qty": str(abs(qty)), "price": "100", "symbol": sym, "order_id": f"off-{sym}",
             "transaction_time": "2023-01-02T00:00:00Z"}
            for sym, qty in self.offsets
        ]
    def get_positions(self):
        import pandas as pd
        if self._closed:
            return pd.Series(dtype="float64")
        return pd.Series({"AAA": 5.0, "ZZZ": 3.0})
    def list_open_orders(self):
        return []
    def cancel_order(self, oid):
        pass
    def submit_offset(self, symbol, qty, coid):
        self.offsets.append((symbol, qty))
        self._closed = True
        return f"off-{symbol}"
    def account(self):
        from algua.execution.alpaca_broker import AccountState
        return AccountState(equity=100_000.0, cash=100_000.0, buying_power=100_000.0)


class _ScriptedReadOnlyBroker:
    """Read-only live broker stub: fixed activities + broker net positions (for resume)."""
    def __init__(self, activities, positions):
        self._activities = activities
        self._positions = positions
    def account_activities(self, after=None):
        return self._activities
    def get_positions(self):
        import pandas as pd
        return pd.Series(self._positions, dtype="float64")


def _seed_live_fills(name, fills):
    from contextlib import closing

    from algua.config.settings import get_settings
    from algua.registry.db import connect, migrate
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        for sym, qty in fills.items():
            conn.execute(
                "INSERT INTO live_fills(activity_id, broker_order_id, strategy, symbol, qty, "
                "price, fill_ts) VALUES (?,?,?,?,?,?,?)",
                (f"seed-{sym}", f"bo-seed-{sym}", name, sym, qty, 100.0,
                 "2023-01-01T00:00:00Z"),
            )
        conn.commit()


def test_breach_flatten_resume_end_to_end(monkeypatch):
    _permissive_book(monkeypatch)
    from algua.live.live_loop import RiskBreach
    name = "cross_sectional_momentum"
    _to_live(name)
    # strategy believes it holds AAA (5) and ZZZ (3); ZZZ is held-but-dropped (not in universe)
    _seed_live_fills(name, {"AAA": 5.0, "ZZZ": 3.0})

    broker = _BreachThenFlatBroker()
    monkeypatch.setattr("algua.cli.live_cmd.verify_live_authorization", lambda *a, **k: _auth())
    monkeypatch.setattr("algua.cli.live_cmd._alpaca_live_broker", lambda auth: broker)
    monkeypatch.setattr("algua.cli.live_cmd._select_provider", lambda demo, snapshot: object())
    monkeypatch.setattr("algua.cli.live_cmd.active_allocation",
                        lambda conn, sid: {"capital": 10_000.0})
    monkeypatch.setattr("algua.cli.live_cmd.run_tick",
                        lambda *a, **k: (_ for _ in ()).throw(RiskBreach("drawdown", "dd")))
    # resume reads the SAME fake broker (read-only path)
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_live_readonly_from_settings", lambda: broker)

    # 1) run-all breaches -> kill-switch trips, offsets submitted over BOTH believed symbols
    r = runner.invoke(app, ["live", "run-all", "--snapshot", "x"])
    assert r.exit_code == 1
    assert sorted(broker.offsets) == [("AAA", 5.0), ("ZZZ", 3.0)]  # held-but-dropped ZZZ included

    from contextlib import closing

    from algua.config.settings import get_settings
    from algua.execution.live_ledger import LedgerKind, believed_positions
    from algua.registry.db import connect, migrate
    from algua.risk import kill_switch
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        assert kill_switch.is_tripped(conn, name)
        # operator-trap: offset fills have NOT been ingested yet -> belief still non-flat
        assert believed_positions(conn, name, LedgerKind.LIVE) == {"AAA": 5.0, "ZZZ": 3.0}

    # 2) resume ingests the offset fills, reconciles to flat, clears the kill-switch (zero drift)
    r2 = runner.invoke(app, ["paper", "resume", name])
    assert r2.exit_code == 0, r2.stdout
    assert json.loads(r2.stdout)["kill_switch"] == "reset"
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        assert not kill_switch.is_tripped(conn, name)
        assert believed_positions(conn, name, LedgerKind.LIVE) == {}   # ledger flat after resume


def test_resume_sibling_holds_same_symbol_does_not_block(monkeypatch):
    """Strategy A is flat; sibling B (a registered LIVE strategy) legitimately holds AAPL — a symbol
    in A's OWN universe, so the explains-path is actually exercised. The account-wide reconcile
    attributes the broker's AAPL to B's live ledger, so resuming A is NOT blocked."""
    name = "cross_sectional_momentum"
    _to_live(name)
    # a REAL live sibling: only a currently-live, attributed strategy may explain broker exposure
    assert runner.invoke(app, ["registry", "add", "sibling_live"]).exit_code == 0
    from contextlib import closing

    from algua.config.settings import get_settings
    from algua.registry.db import connect, migrate
    from algua.risk import kill_switch
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        conn.execute("UPDATE strategies SET stage='live' WHERE name='sibling_live'")
        # sibling B holds AAPL (5) in its own ledger; A holds nothing. AAPL is in A's universe.
        conn.execute(
            "INSERT INTO live_fills(activity_id, broker_order_id, strategy, symbol, qty, price, "
            "fill_ts) VALUES (?,?,?,?,?,?,?)",
            ("sib-aapl", "bo-sib", "sibling_live", "AAPL", 5.0, 100.0, "2023-01-01T00:00:00Z"),
        )
        kill_switch.trip(conn, name, reason="manual", actor="system")
        conn.commit()

    # broker shows AAPL+5 (B's), no activities; A's own ledger is empty
    broker = _ScriptedReadOnlyBroker(activities=[], positions={"AAPL": 5.0})
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_live_readonly_from_settings", lambda: broker)

    r = runner.invoke(app, ["paper", "resume", name])
    assert r.exit_code == 0, r.stdout
    assert json.loads(r.stdout)["kill_switch"] == "reset"


def test_resume_refuses_when_broker_holds_orphan_position(monkeypatch):
    """Fail-closed (GATE-2 CRITICAL): the broker holds AAPL (in A's universe) but NO live strategy's
    ledger explains it. An orphan fill (strategy NULL — a manual/external trade ingested but never
    mapped to an order) must NOT cancel out the broker exposure. A's own ledger is empty, yet resume
    must REFUSE because the broker position is unattributed."""
    name = "cross_sectional_momentum"
    _to_live(name)
    from contextlib import closing

    from algua.config.settings import get_settings
    from algua.registry.db import connect, migrate
    from algua.risk import kill_switch
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        # orphan fill: ingested from a manual/external trade, never attributed to a strategy
        conn.execute(
            "INSERT INTO live_fills(activity_id, broker_order_id, strategy, symbol, qty, price, "
            "fill_ts) VALUES (?,?,?,?,?,?,?)",
            ("orphan-aapl", "bo-orphan", None, "AAPL", 5.0, 100.0, "2023-01-01T00:00:00Z"),
        )
        kill_switch.trip(conn, name, reason="manual", actor="system")
        conn.commit()

    broker = _ScriptedReadOnlyBroker(activities=[], positions={"AAPL": 5.0})
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_live_readonly_from_settings", lambda: broker)

    r = runner.invoke(app, ["paper", "resume", name])
    assert r.exit_code == 1
    payload = json.loads(r.stdout)
    assert payload["ok"] is False and "not flat" in r.stdout.lower()
    assert "AAPL" in str(payload)            # the unexplained broker residual is surfaced


# ---------------------------------------------------------------------------
# live flatten (#449) — emergency single-strategy close + kill-switch trip
# ---------------------------------------------------------------------------


class _FlattenBroker:
    """Fake live broker for the `live flatten` E2E tests. get_positions() feeds the held-cap."""

    def __init__(self, positions):
        self._positions = positions
        self.offsets = []

    def account_activities(self, after=None):
        return []

    def get_positions(self):
        import pandas as pd
        return pd.Series(self._positions, dtype="float64")

    def list_open_orders(self):
        return []

    def cancel_order(self, oid):
        pass

    def submit_offset(self, symbol, qty, coid):
        self.offsets.append((symbol, qty))
        return "off"


def _is_tripped(name):
    from contextlib import closing

    from algua.config.settings import get_settings
    from algua.registry.db import connect, migrate
    from algua.risk import kill_switch
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        return kill_switch.is_tripped(conn, name)


def test_live_flatten_delegates_to_flatten_strategy(monkeypatch):
    from algua.execution.flatten import FlattenResult
    from algua.execution.live_ledger import LedgerKind
    name = "cross_sectional_momentum"
    _to_live(name)
    monkeypatch.setattr("algua.cli.live_cmd.verify_live_authorization", lambda *a, **k: _auth())
    monkeypatch.setattr("algua.cli.live_cmd._alpaca_live_broker", lambda auth: object())

    captured = {}

    def _spy(conn, broker, sname, kind, *, lane, cancel, ingest, held=None, **k):
        captured["kind"] = kind
        captured["lane"] = lane
        captured["callables"] = (callable(cancel), callable(ingest), callable(held))
        captured["tripped_at_call"] = _is_tripped(sname)  # trip must precede the helper
        return FlattenResult(n_offsets=1, flatten_error=None)

    monkeypatch.setattr("algua.cli.live_cmd.flatten_strategy", _spy)
    r = runner.invoke(app, ["live", "flatten", name])
    assert r.exit_code == 0, r.stdout
    assert captured["kind"] is LedgerKind.LIVE
    assert captured["lane"] == "live"
    assert captured["callables"] == (True, True, True)   # cancel/ingest/held all injected
    assert captured["tripped_at_call"] is True           # kill-switch tripped BEFORE the helper
    assert json.loads(r.stdout) == {
        "ok": True, "strategy": name, "kill_switch": "tripped",
        "liquidation_submitted": True, "offsets_submitted": 1,
    }


def test_live_flatten_non_live_stage_refused_no_trip(monkeypatch):
    # Fork E (load-bearing): a non-LIVE strategy cannot be emergency-flattened via the live surface,
    # and the DB kill-switch must NOT be written (no cross-lane DoS).
    name = "paper_only"
    assert runner.invoke(app, ["registry", "add", name]).exit_code == 0
    for to, actor in (("backtested", "human"), ("candidate", "human"), ("paper", "agent")):
        assert runner.invoke(app, ["registry", "transition", name, "--to", to,
                                   "--actor", actor, "--reason", "x"]).exit_code == 0

    def _no_helper(*a, **k):
        raise AssertionError("flatten_strategy must not run for a non-LIVE strategy")
    monkeypatch.setattr("algua.cli.live_cmd.flatten_strategy", _no_helper)

    r = runner.invoke(app, ["live", "flatten", name])
    assert r.exit_code == 1
    payload = json.loads(r.stdout)
    assert payload["ok"] is False
    assert payload["kill_switch"] == "not_tripped"
    assert _is_tripped(name) is False  # the paper strategy's switch was never written


def test_live_flatten_already_flat_reports_not_submitted(monkeypatch):
    from algua.execution.flatten import FlattenResult
    name = "cross_sectional_momentum"
    _to_live(name)
    monkeypatch.setattr("algua.cli.live_cmd.verify_live_authorization", lambda *a, **k: _auth())
    monkeypatch.setattr("algua.cli.live_cmd._alpaca_live_broker", lambda auth: object())
    monkeypatch.setattr("algua.cli.live_cmd.flatten_strategy",
                        lambda *a, **k: FlattenResult(n_offsets=0, flatten_error=None))
    r = runner.invoke(app, ["live", "flatten", name])
    assert r.exit_code == 0, r.stdout
    payload = json.loads(r.stdout)
    assert payload["liquidation_submitted"] is False
    assert payload["offsets_submitted"] == 0
    assert payload["kill_switch"] == "tripped"
    assert _is_tripped(name) is True


def test_live_flatten_error_stays_tripped(monkeypatch):
    from algua.execution.flatten import FlattenResult
    name = "cross_sectional_momentum"
    _to_live(name)
    monkeypatch.setattr("algua.cli.live_cmd.verify_live_authorization", lambda *a, **k: _auth())
    monkeypatch.setattr("algua.cli.live_cmd._alpaca_live_broker", lambda auth: object())
    monkeypatch.setattr("algua.cli.live_cmd.flatten_strategy",
                        lambda *a, **k: FlattenResult(n_offsets=2, flatten_error="boom"))
    r = runner.invoke(app, ["live", "flatten", name])
    assert r.exit_code == 1
    payload = json.loads(r.stdout)
    assert payload["ok"] is False
    assert payload["kill_switch"] == "tripped"
    assert payload["error"] == "boom"
    assert payload["liquidation_submitted"] is False
    assert payload["offsets_submitted"] == 2
    assert _is_tripped(name) is True  # STILL tripped after a failed flatten


def test_live_flatten_revoked_live_trips_then_fails_closed(monkeypatch):
    # Fork A: a LIVE strategy whose authorization does NOT verify. The kill-switch trips FIRST
    # (fail-safe), then verify raises -> no broker built, helper never called, payload names the
    # raw-broker break-glass path AND #478.
    from algua.registry.live_gate import LiveAuthorizationError
    name = "cross_sectional_momentum"
    _to_live(name)

    def _revoked(*a, **k):
        raise LiveAuthorizationError("authorization revoked")
    monkeypatch.setattr("algua.cli.live_cmd.verify_live_authorization", _revoked)

    def _no_broker(auth):
        raise AssertionError("no live broker may be built for a revoked strategy")
    monkeypatch.setattr("algua.cli.live_cmd._alpaca_live_broker", _no_broker)

    def _no_helper(*a, **k):
        raise AssertionError("flatten_strategy must not run when authorization fails")
    monkeypatch.setattr("algua.cli.live_cmd.flatten_strategy", _no_helper)

    r = runner.invoke(app, ["live", "flatten", name])
    assert r.exit_code == 1
    payload = json.loads(r.stdout)
    assert payload["ok"] is False
    assert payload["kill_switch"] == "tripped"      # (1) trip precedes verify
    assert payload["liquidation_submitted"] is False
    assert "478" in payload["note"]                 # names #478
    assert "raw broker" in payload["note"].lower()  # names the raw-broker break-glass
    assert _is_tripped(name) is True


def test_live_flatten_missing_strategy(monkeypatch):
    def _no_trip(*a, **k):
        raise AssertionError("kill-switch must not be written for an unknown strategy")
    monkeypatch.setattr("algua.risk.kill_switch.trip", _no_trip)
    r = runner.invoke(app, ["live", "flatten", "does_not_exist"])
    assert r.exit_code == 1
    payload = json.loads(r.stdout)
    assert payload["ok"] is False


def test_live_flatten_offsets_and_trips(monkeypatch):
    # END-TO-END through the real helper: believed AAA+5, broker holds AAA+5 -> submit_offset AAA 5,
    # live_orders row side='sell' with broker_order_id backfilled.
    name = "cross_sectional_momentum"
    _to_live(name)
    _seed_live_fills(name, {"AAA": 5.0})
    broker = _FlattenBroker({"AAA": 5.0})
    monkeypatch.setattr("algua.cli.live_cmd.verify_live_authorization", lambda *a, **k: _auth())
    monkeypatch.setattr("algua.cli.live_cmd._alpaca_live_broker", lambda auth: broker)

    r = runner.invoke(app, ["live", "flatten", name])
    assert r.exit_code == 0, r.stdout
    payload = json.loads(r.stdout)
    assert payload["liquidation_submitted"] is True
    assert payload["offsets_submitted"] == 1
    assert broker.offsets == [("AAA", 5.0)]

    from contextlib import closing

    from algua.config.settings import get_settings
    from algua.registry.db import connect, migrate
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        row = conn.execute(
            "SELECT side, broker_order_id FROM live_orders WHERE strategy=? AND symbol='AAA'",
            (name,),
        ).fetchone()
    assert row["side"] == "sell" and row["broker_order_id"] == "off"


def test_live_flatten_belief_exceeds_held_caps_offset(monkeypatch):
    # END-TO-END: believed AAA+10 but broker holds only AAA+3 -> the injected held caps the offset
    # to 3, proving the command's held callable reaches the helper.
    name = "cross_sectional_momentum"
    _to_live(name)
    _seed_live_fills(name, {"AAA": 10.0})
    broker = _FlattenBroker({"AAA": 3.0})
    monkeypatch.setattr("algua.cli.live_cmd.verify_live_authorization", lambda *a, **k: _auth())
    monkeypatch.setattr("algua.cli.live_cmd._alpaca_live_broker", lambda auth: broker)

    r = runner.invoke(app, ["live", "flatten", name])
    assert r.exit_code == 0, r.stdout
    assert broker.offsets == [("AAA", 3.0)]  # capped to held, not the believed 10


# ---------------------------------------------------------------------------
# live halt-all (#449) — global halt ONLY, no account close
# ---------------------------------------------------------------------------


def _global_halt_engaged():
    from contextlib import closing

    from algua.config.settings import get_settings
    from algua.registry.db import connect, migrate
    from algua.risk import global_halt
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        return global_halt.is_engaged(conn)


def test_live_halt_all_engages_halt_only(monkeypatch):
    # #449: halt-all engages ONLY the global halt — it builds NO broker, closes NO position.
    _to_live()

    def _no_broker(auth):
        raise AssertionError("halt-all must NOT build a live broker")

    monkeypatch.setattr("algua.cli.live_cmd._alpaca_live_broker", _no_broker)

    r = runner.invoke(app, ["live", "halt-all", "--reason", "market crash",
                            "--actor", "agent"])
    assert r.exit_code == 0, r.stdout
    payload = json.loads(r.stdout)
    assert payload["ok"] is True
    assert payload["global_halt"] == "set"
    assert payload["liquidation_submitted"] is False
    assert "478" in payload["note"]
    assert "raw broker" in payload["note"].lower()
    assert _global_halt_engaged() is True


def test_live_halt_all_human_same_as_agent(monkeypatch):
    # --actor human is byte-identical halt-only behaviour: engage the global halt, build no broker.
    _to_live()

    def _no_broker(auth):
        raise AssertionError("halt-all must NOT build a live broker")

    monkeypatch.setattr("algua.cli.live_cmd._alpaca_live_broker", _no_broker)

    r = runner.invoke(app, ["live", "halt-all", "--reason", "human decision",
                            "--actor", "human"])
    assert r.exit_code == 0, r.stdout
    payload = json.loads(r.stdout)
    assert payload["ok"] is True
    assert payload["global_halt"] == "set"
    assert payload["liquidation_submitted"] is False
    assert _global_halt_engaged() is True


def test_live_halt_all_no_close_account_flag():
    # The whole-account close is unreachable here: --close-account is an unknown option.
    r = runner.invoke(app, ["live", "halt-all", "--reason", "x", "--close-account"])
    assert r.exit_code != 0


# ---------------------------------------------------------------------------
# Additional flatten & halt-all tests for task 3 of #449
# ---------------------------------------------------------------------------


def test_live_flatten_rejects_non_live(monkeypatch):
    # (b) Bring a strategy only to 'paper' stage (NOT 'live'), verify flatten rejects it.
    # The REAL authorization check should raise, since no valid authorization exists.
    # Assert exit 1, ok=False, kill-switch NOT tripped (no cross-lane DoS).
    from contextlib import closing

    from algua.config.settings import get_settings
    from algua.registry.db import connect, migrate

    name = "paper_only"
    assert runner.invoke(app, ["registry", "add", name]).exit_code == 0
    # Bring to paper (via backtested -> candidate -> paper)
    for to, actor in (("backtested", "human"), ("candidate", "human"), ("paper", "agent")):
        assert runner.invoke(app, ["registry", "transition", name, "--to", to,
                                   "--actor", actor, "--reason", "x"]).exit_code == 0

    # Do NOT monkeypatch verify_live_authorization, so the REAL check raises
    # Assert broker is never built
    def _boom(auth):
        raise AssertionError("no broker for non-LIVE strategy")
    monkeypatch.setattr("algua.cli.live_cmd._alpaca_live_broker", _boom)

    r = runner.invoke(app, ["live", "flatten", name])
    assert r.exit_code == 1
    payload = json.loads(r.stdout)
    assert payload["ok"] is False
    # Kill-switch must NOT be tripped (no cross-lane DoS)
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        from algua.risk import kill_switch
        assert kill_switch.is_tripped(conn, name) is False


def test_live_flatten_skips_subtol(monkeypatch):
    # (c) Believed position is subtol (5e-7); broker holds it; offset should NOT be submitted.
    # Kill-switch still tripped (fail-safe), but liquidation_submitted=False, offsets_submitted=0.
    name = "cross_sectional_momentum"
    _to_live(name)
    # Seed with subtol qty
    _seed_live_fills(name, {"AAA": 5e-7})
    # Broker also reports the same subtol qty
    broker = _FlattenBroker({"AAA": 5e-7})
    monkeypatch.setattr("algua.cli.live_cmd.verify_live_authorization", lambda *a, **k: _auth())
    monkeypatch.setattr("algua.cli.live_cmd._alpaca_live_broker", lambda auth: broker)

    r = runner.invoke(app, ["live", "flatten", name])
    assert r.exit_code == 0, r.stdout
    payload = json.loads(r.stdout)
    assert payload["liquidation_submitted"] is False
    assert payload["offsets_submitted"] == 0
    assert payload["kill_switch"] == "tripped"
    # No offset should have been submitted
    assert broker.offsets == []


def test_live_flatten_close_failure_stays_tripped(monkeypatch):
    # (d) Flatten strategy raises BrokerError during submit_offset; assert exit 1,
    # ok=False, and DB kill-switch STILL tripped (fail-safe, no rollback).
    from algua.execution.alpaca_broker import BrokerError

    name = "cross_sectional_momentum"
    _to_live(name)
    _seed_live_fills(name, {"AAA": 5.0})

    class _FailingBroker(_FlattenBroker):
        def submit_offset(self, symbol, qty, coid):
            raise BrokerError("broker offline")

    broker = _FailingBroker({"AAA": 5.0})
    monkeypatch.setattr("algua.cli.live_cmd.verify_live_authorization", lambda *a, **k: _auth())
    monkeypatch.setattr("algua.cli.live_cmd._alpaca_live_broker", lambda auth: broker)

    r = runner.invoke(app, ["live", "flatten", name])
    assert r.exit_code == 1
    payload = json.loads(r.stdout)
    assert payload["ok"] is False
    # Kill-switch must remain tripped (fail-safe, no rollback on error)
    assert payload["kill_switch"] == "tripped"
    assert _is_tripped(name) is True


def test_live_halt_all_engages_and_closes(monkeypatch):
    # (e) halt-all with verify_live_authorization monkeypatched to _auth().
    # halt-all is halt-only: builds NO broker, closes NO positions.
    # Assert exit 0, payload global_halt=="set", liquidation_submitted=False,
    # and DB global_halt.is_engaged()==True. Verify broker is never built.
    _to_live()

    def _no_broker(auth):
        raise AssertionError("halt-all must NOT build a live broker")

    monkeypatch.setattr("algua.cli.live_cmd.verify_live_authorization", lambda *a, **k: _auth())
    monkeypatch.setattr("algua.cli.live_cmd._alpaca_live_broker", _no_broker)

    r = runner.invoke(app, ["live", "halt-all", "--reason", "panic", "--actor", "agent"])
    assert r.exit_code == 0, r.stdout
    payload = json.loads(r.stdout)
    assert payload["global_halt"] == "set"
    assert payload["liquidation_submitted"] is False
    assert _global_halt_engaged() is True


def test_live_halt_all_no_authorized_still_halts(monkeypatch):
    # (f) _to_live() with NO authorization row. REAL verify_live_authorization would raise.
    # halt-all is halt-only and succeeds regardless: it engages the global halt without
    # building a broker. The halt-first fail-safe means we still succeed (exit 0).
    # Assert exit 0, liquidation_submitted=False, DB global_halt is engaged.
    _to_live()
    # No manual authorization injection, so verify_live_authorization fails naturally

    sentinel = {"built": False}
    def _broker_sentinel(auth):
        sentinel["built"] = True
        raise AssertionError("broker must not be built")
    monkeypatch.setattr("algua.cli.live_cmd._alpaca_live_broker", _broker_sentinel)

    r = runner.invoke(app, ["live", "halt-all", "--reason", "panic", "--actor", "agent"])
    assert r.exit_code == 0
    payload = json.loads(r.stdout)
    assert payload["liquidation_submitted"] is False
    assert sentinel["built"] is False  # broker never constructed
    # Global halt is engaged (halt-first fail-safe)
    assert _global_halt_engaged() is True


def test_live_halt_all_close_failure_stays_engaged(monkeypatch):
    # (g) halt-all is halt-only, so it never calls close_all_positions.
    # Verify that the global halt is engaged even if authorization fails.
    # This test ensures halt-first fail-safe means we engage the halt even when
    # authorization is missing or verify would fail.
    _to_live()

    def _auth_fails(*a, **k):
        from algua.registry.live_gate import LiveAuthorizationError
        raise LiveAuthorizationError("revoked")

    monkeypatch.setattr("algua.cli.live_cmd.verify_live_authorization", _auth_fails)

    def _boom(auth):
        raise AssertionError("no broker built for halt-all")

    monkeypatch.setattr("algua.cli.live_cmd._alpaca_live_broker", _boom)

    r = runner.invoke(app, ["live", "halt-all", "--reason", "panic", "--actor", "agent"])
    assert r.exit_code == 0  # halt-first succeeds even when verify fails
    payload = json.loads(r.stdout)
    assert payload["liquidation_submitted"] is False
    # Global halt is still engaged (fail-safe)
    assert _global_halt_engaged() is True


# --- #452 HIGH#3: dark-feed breach HALTS (no flatten) vs economic breach flattens ---


class _AliveBroker:
    """A broker that is fully alive (account readable, no positions) — the dark-feed scenario is a
    dead BAR feed, not a dead broker, so the account/positions calls all succeed."""

    def account_activities(self, after=None):
        return []

    def get_positions(self):
        import pandas as pd
        return pd.Series(dtype=float)

    def list_open_orders(self):
        return []

    def cancel_order(self, oid):
        pass

    def account(self):
        from algua.execution.alpaca_broker import AccountState
        # equity == last_equity so the #390 book-loss breaker (patched off anyway) never fires.
        return AccountState(equity=100_000.0, cash=100_000.0, buying_power=100_000.0,
                            last_equity=100_000.0)


def _wire_alive_run_all(monkeypatch, broker):
    monkeypatch.setattr("algua.cli.live_cmd.verify_live_authorization", lambda *a, **k: _auth())
    monkeypatch.setattr("algua.cli.live_cmd._alpaca_live_broker", lambda auth: broker)
    monkeypatch.setattr("algua.cli.live_cmd._select_provider", lambda demo, snapshot: object())
    monkeypatch.setattr("algua.cli.live_cmd._broker_account_activities", lambda b, a: [])
    monkeypatch.setattr("algua.cli.live_cmd._broker_net_positions", lambda b: {})
    monkeypatch.setattr("algua.cli.live_cmd._broker_buying_power", lambda b: 100_000.0)
    monkeypatch.setattr("algua.cli.live_cmd.active_allocation",
                        lambda conn, sid: {"capital": 10_000.0})


@pytest.mark.parametrize("kind", ["stale_marks", "unvaluable_marks"])
def test_run_all_dark_feed_breach_halts_without_flatten(monkeypatch, kind):
    # #452 HIGH#3: a stale / unvaluable mark (dark BAR feed, broker still alive) must ENGAGE THE
    # GLOBAL HALT and PRESERVE positions — it must NOT flatten the book at unknown prices.
    from algua.live.live_loop import RiskBreach
    name = "cross_sectional_momentum"
    _to_live(name)
    _permissive_book(monkeypatch)
    broker = _AliveBroker()
    _wire_alive_run_all(monkeypatch, broker)

    flatten_calls: list = []
    monkeypatch.setattr(
        "algua.cli.live_cmd.flatten_strategy",
        lambda *a, **k: flatten_calls.append(k.get("lane", a)),
    )
    monkeypatch.setattr(
        "algua.cli.live_cmd.run_tick",
        lambda *a, **k: (_ for _ in ()).throw(RiskBreach(kind, f"{kind} dark feed")),
    )

    r = runner.invoke(app, ["live", "run-all", "--snapshot", "x"])
    assert r.exit_code == 1
    payload = json.loads(r.stdout)
    assert payload["ok"] is False
    strat = payload["strategies"][0]
    assert strat["kind"] == kind
    assert strat["halted"] is True
    assert strat["global_halt"] == "set"
    assert strat["liquidation_submitted"] is False
    assert flatten_calls == []  # dark feed HALTS — flatten_strategy is never called
    assert _global_halt_engaged() is True  # systemic: whole account halted


@pytest.mark.parametrize("kind", ["drawdown", "gross_exposure_realized"])
def test_run_all_economic_breach_still_flattens(monkeypatch, kind):
    # #452 HIGH#3: an ECONOMIC/integrity breach (drawdown, gross_exposure_realized, ...) keeps the
    # UNCHANGED trip + scoped-flatten path — only the dark-feed kinds are diverted to halt-only.
    from algua.execution.flatten import FlattenResult
    from algua.live.live_loop import RiskBreach
    name = "cross_sectional_momentum"
    _to_live(name)
    _permissive_book(monkeypatch)
    broker = _AliveBroker()
    _wire_alive_run_all(monkeypatch, broker)

    flatten_calls: list = []

    def _fake_flatten(conn, broker_, name_, kind_, lane, cancel, ingest, held):
        flatten_calls.append(name_)
        return FlattenResult(n_offsets=1, flatten_error=None)

    monkeypatch.setattr("algua.cli.live_cmd.flatten_strategy", _fake_flatten)
    monkeypatch.setattr(
        "algua.cli.live_cmd.run_tick",
        lambda *a, **k: (_ for _ in ()).throw(RiskBreach(kind, f"{kind} economic")),
    )

    r = runner.invoke(app, ["live", "run-all", "--snapshot", "x"])
    assert r.exit_code == 1
    payload = json.loads(r.stdout)
    assert payload["ok"] is False
    strat = payload["strategies"][0]
    assert strat["kind"] == kind
    assert strat["liquidation_submitted"] is True
    assert flatten_calls == [name]  # economic breach DID flatten (scoped)
    assert strat.get("halted") is not True  # not a halt-only marker
    assert strat.get("global_halt") != "set"  # economic breach does not globally halt


# --- #452 HIGH#2: _build_book_exposure account-book freshness wall (halt, no flatten) ---


class _BookProvider:
    """Provider returning a fixed bars frame for the account-book valuation fetch."""

    def __init__(self, bars):
        self._bars = bars

    def get_bars(self, symbols, start, end, timeframe):
        return self._bars


class _BookBroker:
    def __init__(self, equity=100_000.0):
        self._equity = equity

    def account(self):
        from algua.execution.alpaca_broker import AccountState
        return AccountState(equity=self._equity, cash=self._equity,
                            buying_power=self._equity, last_equity=self._equity)


_BOOK_DATES = [datetime(2023, 1, d, tzinfo=UTC) for d in (2, 3, 4)]
_BOOK_NOW = datetime(2023, 1, 5, tzinfo=UTC)  # all three sessions closed => staleness 0


def _book_bars(symbol_closes):
    import pandas as pd
    rows = []
    for sym, closes in symbol_closes.items():
        for ts, px in zip(_BOOK_DATES, closes, strict=True):
            rows.append({"timestamp": ts, "symbol": sym, "open": px, "high": px,
                         "low": px, "close": px, "adj_close": px, "volume": 1000})
    return pd.DataFrame(rows).set_index("timestamp").sort_index()


def test_build_book_exposure_fresh_account_builds():
    from algua.cli.live_cmd import _build_book_exposure
    from algua.risk.book_limits import BookExposure
    provider = _BookProvider(_book_bars({"AAA": [10.0, 11.0, 12.0]}))
    book, reason = _build_book_exposure(
        _BookBroker(), provider, {"AAA": 5.0}, "2023-01-02", "2023-01-04", now=_BOOK_NOW
    )
    assert reason is None
    assert isinstance(book, BookExposure)


def test_build_book_exposure_stale_mark_raises_riskbreach():
    from algua.cli.live_cmd import _build_book_exposure
    from algua.risk.limits import RiskBreach
    provider = _BookProvider(_book_bars({"AAA": [10.0, 11.0, 12.0]}))
    # now is far in the future -> the newest 2023-01-04 bar is many sessions stale.
    with pytest.raises(RiskBreach) as exc:
        _build_book_exposure(
            _BookBroker(), provider, {"AAA": 5.0}, "2023-01-02", "2023-01-04",
            now=datetime(2024, 1, 5, tzinfo=UTC),
        )
    assert exc.value.kind == "stale_marks"


def test_build_book_exposure_absent_mark_raises_riskbreach():
    from algua.cli.live_cmd import _build_book_exposure
    from algua.risk.limits import RiskBreach
    # provider returns bars only for BBB, but the account holds AAA -> AAA has no mark at all.
    provider = _BookProvider(_book_bars({"BBB": [10.0, 11.0, 12.0]}))
    with pytest.raises(RiskBreach) as exc:
        _build_book_exposure(
            _BookBroker(), provider, {"AAA": 5.0}, "2023-01-02", "2023-01-04", now=_BOOK_NOW
        )
    assert exc.value.kind == "stale_marks"  # no_mark => infinite staleness


def test_build_book_exposure_infinite_mark_raises_riskbreach():
    from algua.cli.live_cmd import _build_book_exposure
    from algua.risk.limits import RiskBreach
    provider = _BookProvider(_book_bars({"AAA": [10.0, 11.0, float("inf")]}))
    with pytest.raises(RiskBreach) as exc:
        _build_book_exposure(
            _BookBroker(), provider, {"AAA": 5.0}, "2023-01-02", "2023-01-04", now=_BOOK_NOW
        )
    assert exc.value.kind == "unvaluable_marks"


def test_build_book_exposure_short_precondition_benign_defer():
    from algua.cli.live_cmd import _build_book_exposure
    provider = _BookProvider(_book_bars({"AAA": [10.0, 11.0, 12.0]}))
    book, reason = _build_book_exposure(
        _BookBroker(), provider, {"AAA": -5.0}, "2023-01-02", "2023-01-04", now=_BOOK_NOW
    )
    assert book is None
    assert "short position" in reason  # policy defer, NOT a data-integrity RiskBreach


def test_build_book_exposure_seed_breach_benign_defer():
    from algua.cli.live_cmd import _build_book_exposure
    # A small equity vs a large single-name notional makes the seed book already breach a cap.
    provider = _BookProvider(_book_bars({"AAA": [10.0, 11.0, 12.0]}))
    book, reason = _build_book_exposure(
        _BookBroker(equity=1.0), provider, {"AAA": 1000.0}, "2023-01-02", "2023-01-04",
        now=_BOOK_NOW,
    )
    assert book is None
    assert "already breaches" in reason


def test_run_all_book_stale_marks_halts_without_close_all(monkeypatch):
    # #452 HIGH#2: a stale account-book mark makes run-all engage the GLOBAL HALT and NOT call
    # close_all_positions (halt-only — the broker-truth book-loss breaker already ran this cycle).
    name = "cross_sectional_momentum"
    _to_live(name)
    _seed_live_fills(name, {"AAA": 5.0})  # ledger matches the broker net so reconcile is clean
    monkeypatch.setattr("algua.cli.live_cmd.verify_live_authorization", lambda *a, **k: _auth())
    monkeypatch.setattr("algua.cli.live_cmd._evaluate_book_loss_breaker", lambda *a, **k: None)

    class _StaleBookBroker:
        def __init__(self):
            self.closed_all = False

        def account_activities(self, after=None):
            return []

        def get_positions(self):
            import pandas as pd
            return pd.Series(dtype=float)

        def close_all_positions(self):
            self.closed_all = True

        def account(self):
            from algua.execution.alpaca_broker import AccountState
            return AccountState(equity=100_000.0, cash=100_000.0, buying_power=100_000.0,
                                last_equity=100_000.0)

    broker = _StaleBookBroker()
    monkeypatch.setattr("algua.cli.live_cmd._alpaca_live_broker", lambda auth: broker)
    monkeypatch.setattr("algua.cli.live_cmd._select_provider",
                        lambda demo, snapshot: _BookProvider(_book_bars({"AAA": [10.0, 11.0,
                                                                                 12.0]})))
    monkeypatch.setattr("algua.cli.live_cmd._broker_account_activities", lambda b, a: [])
    # the reconciled account holds AAA; the bar frame is far stale relative to real "now".
    monkeypatch.setattr("algua.cli.live_cmd._broker_net_positions", lambda b: {"AAA": 5.0})
    monkeypatch.setattr("algua.cli.live_cmd.active_allocation",
                        lambda conn, sid: {"capital": 10_000.0})

    def _no_tick(*a, **k):
        raise AssertionError("run_tick must not run once the book stale-marks wall trips")
    monkeypatch.setattr("algua.cli.live_cmd.run_tick", _no_tick)

    r = runner.invoke(app, ["live", "run-all", "--snapshot", "x"])
    assert r.exit_code == 1
    payload = json.loads(r.stdout)
    assert payload["ok"] is False
    assert payload["book_breach"]["kind"] == "stale_marks"
    assert payload["global_halt"] == "set"
    assert payload["liquidation_submitted"] is False
    assert payload["skipped_unallocated"] == []  # #497 F1: threaded into stale-marks-halt
    assert broker.closed_all is False  # halt-only: NEVER flatten off a dead feed
    assert _global_halt_engaged() is True
