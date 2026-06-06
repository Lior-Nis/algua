import json
from datetime import UTC, datetime

import pytest
from typer.testing import CliRunner

from algua.cli.main import app
from algua.contracts.types import LiveAuthorization
from algua.live.live_loop import TickResult
from algua.risk.limits import RiskBreach

runner = CliRunner()



@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "p.db"))
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ALGUA_ALPACA_LIVE_API_KEY", "lk")
    monkeypatch.setenv("ALGUA_ALPACA_LIVE_API_SECRET", "ls")


def _to_live(name="cross_sectional_momentum"):
    # bring a strategy to 'live' stage in the DB directly (the signed ceremony is tested elsewhere)
    from contextlib import closing

    from algua.config.settings import get_settings
    from algua.registry.db import connect, migrate
    assert runner.invoke(app, ["backtest", "run", name, "--demo", "--register",
                               "--start", "2022-01-01", "--end", "2023-12-31"]).exit_code == 0
    for to in ("shortlisted", "paper"):
        runner.invoke(app, ["registry", "transition", name, "--to", to, "--actor", "agent",
                            "--reason", "x"])
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        conn.execute("UPDATE strategies SET stage='live' WHERE name=?", (name,))
        conn.commit()


def _auth():
    return LiveAuthorization(1, "c", "cf", "d", "lior", "t")


def test_live_trade_tick_refused_without_authorization(monkeypatch):
    from algua.registry.live_gate import LiveAuthorizationError
    _to_live()
    monkeypatch.setattr("algua.cli.live_cmd.verify_live_authorization",
                        lambda *a, **k: (_ for _ in ()).throw(LiveAuthorizationError("nope")))
    r = runner.invoke(app, ["live", "trade-tick", "cross_sectional_momentum", "--snapshot", "x"])
    assert r.exit_code == 1 and json.loads(r.stdout)["ok"] is False


def test_live_trade_tick_refused_when_killed(monkeypatch):
    _to_live()
    monkeypatch.setattr("algua.cli.live_cmd.verify_live_authorization", lambda *a, **k: _auth())
    runner.invoke(app, ["paper", "kill", "cross_sectional_momentum", "--reason", "x"])
    r = runner.invoke(app, ["live", "trade-tick", "cross_sectional_momentum", "--snapshot", "x"])
    assert r.exit_code == 1 and json.loads(r.stdout)["ok"] is False


def test_live_trade_tick_missing_live_keys(monkeypatch):
    _to_live()
    monkeypatch.setenv("ALGUA_ALPACA_LIVE_API_KEY", "")
    monkeypatch.setattr("algua.cli.live_cmd.verify_live_authorization", lambda *a, **k: _auth())
    r = runner.invoke(app, ["live", "trade-tick", "cross_sectional_momentum", "--snapshot", "x"])
    assert r.exit_code == 1 and json.loads(r.stdout)["ok"] is False


def test_live_trade_tick_happy_path(monkeypatch):
    _to_live()
    ts = datetime(2023, 6, 1, tzinfo=UTC)
    fake = TickResult(decision_ts=ts, target_weights={"AAA": 1.0}, positions_before={},
                      submitted=[{"symbol": "AAA", "side": "buy", "target_weight": 1.0,
                                  "order_id": "o-1", "client_order_id": "c"}],
                      equity=50000.0, peak_equity=50000.0)
    monkeypatch.setattr("algua.cli.live_cmd.verify_live_authorization", lambda *a, **k: _auth())
    monkeypatch.setattr("algua.cli.live_cmd._alpaca_live_broker", lambda auth: object())
    monkeypatch.setattr("algua.cli.live_cmd._select_provider", lambda demo, snapshot: object())

    def _fake_run_tick(*a, **k):
        # mimic run_tick invoking on_submitted per accepted order (the immediate-audit path)
        from algua.live.live_loop import SubmittedOrder
        k["hooks"].on_submitted(SubmittedOrder(symbol="AAA", side="buy", target_weight=1.0,
                                               order_id="o-1", client_order_id="c", decision_ts=ts))
        return fake

    monkeypatch.setattr("algua.cli.live_cmd.run_tick", _fake_run_tick)
    r = runner.invoke(app, ["live", "trade-tick", "cross_sectional_momentum", "--snapshot", "x"])
    assert r.exit_code == 0, r.stdout
    payload = json.loads(r.stdout)
    assert payload["submitted"][0]["order_id"] == "o-1"
    # the live order was audited and a tick snapshot recorded
    from contextlib import closing

    from algua.config.settings import get_settings
    from algua.execution.order_state import latest_tick_snapshot
    from algua.registry.db import connect, migrate
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        assert latest_tick_snapshot(conn, "cross_sectional_momentum") is not None
        n = conn.execute("SELECT COUNT(*) FROM audit_log WHERE action='live_order'").fetchone()[0]
        assert n == 1
        # the order is recorded in the BOOKS (client_order_id primary + broker_order_id backfilled)
        # so fills can attribute + scoped cancel can find it (codex HIGH)
        order = conn.execute(
            "SELECT strategy, broker_order_id FROM live_orders WHERE client_order_id='c'"
        ).fetchone()
        assert order["strategy"] == "cross_sectional_momentum"
        assert order["broker_order_id"] == "o-1"


def test_live_trade_tick_breach_trips_and_flattens(monkeypatch):
    _to_live()
    monkeypatch.setattr("algua.cli.live_cmd.verify_live_authorization", lambda *a, **k: _auth())

    class _FlatBroker:
        def __init__(self):
            self.closed = None
        def cancel_open_orders(self):
            pass
        def close_positions(self, syms):
            self.closed = list(syms)

    broker = _FlatBroker()
    monkeypatch.setattr("algua.cli.live_cmd._alpaca_live_broker", lambda auth: broker)
    monkeypatch.setattr("algua.cli.live_cmd._select_provider", lambda demo, snapshot: object())
    monkeypatch.setattr("algua.cli.live_cmd.run_tick",
                        lambda *a, **k: (_ for _ in ()).throw(RiskBreach("drawdown", "dd")))
    r = runner.invoke(app, ["live", "trade-tick", "cross_sectional_momentum", "--snapshot", "x"])
    assert r.exit_code == 1
    payload = json.loads(r.stdout)
    assert payload["ok"] is False and payload["kind"] == "drawdown"
    assert broker.closed is not None  # scoped flatten ran on the live broker


def test_run_all_no_live_strategies_is_noop(monkeypatch):
    r = runner.invoke(app, ["live", "run-all", "--snapshot", "x"])
    assert r.exit_code == 0
    assert json.loads(r.stdout)["strategies"] == []


def test_run_all_halts_on_unexplained_reconcile_drift(monkeypatch):
    from algua.risk import global_halt
    _to_live()
    monkeypatch.setattr("algua.cli.live_cmd.verify_live_authorization", lambda *a, **k: _auth())
    monkeypatch.setattr("algua.cli.live_cmd._alpaca_live_broker", lambda auth: object())
    monkeypatch.setattr("algua.cli.live_cmd._select_provider", lambda demo, snapshot: object())
    # broker activities -> none; positions -> an unexplained holding the books don't have
    monkeypatch.setattr("algua.cli.live_cmd.ingest_activities", lambda conn, acts: None)
    monkeypatch.setattr("algua.cli.live_cmd.fill_cursor", lambda conn: None)
    monkeypatch.setattr("algua.cli.live_cmd._broker_account_activities", lambda broker, after: [])
    monkeypatch.setattr("algua.cli.live_cmd._broker_net_positions", lambda broker: {"ZZZ": 99.0})
    # --grace-cycles 0 forces the mismatch straight to unexplained -> assert global halt engaged
    r = runner.invoke(app, ["live", "run-all", "--snapshot", "x", "--grace-cycles", "0"])
    assert r.exit_code == 1
    payload = json.loads(r.stdout)
    assert payload["ok"] is False and payload["reconcile"]["halt"] is True
    from contextlib import closing

    from algua.config.settings import get_settings
    from algua.registry.db import connect, migrate
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        assert global_halt.is_engaged(conn)


def test_run_all_ticks_strategy_when_clean(monkeypatch):
    _to_live()
    monkeypatch.setattr("algua.cli.live_cmd.verify_live_authorization", lambda *a, **k: _auth())
    monkeypatch.setattr("algua.cli.live_cmd._alpaca_live_broker", lambda auth: object())
    monkeypatch.setattr("algua.cli.live_cmd._select_provider", lambda demo, snapshot: object())
    monkeypatch.setattr("algua.cli.live_cmd.ingest_activities", lambda conn, acts: None)
    monkeypatch.setattr("algua.cli.live_cmd.fill_cursor", lambda conn: None)
    monkeypatch.setattr("algua.cli.live_cmd._broker_account_activities", lambda broker, after: [])
    monkeypatch.setattr("algua.cli.live_cmd._broker_net_positions", lambda broker: {})
    monkeypatch.setattr("algua.cli.live_cmd._run_strategy_tick",
                        lambda *a, **k: {"strategy": "cross_sectional_momentum", "submitted": []})
    r = runner.invoke(app, ["live", "run-all", "--snapshot", "x"])
    assert r.exit_code == 0
    payload = json.loads(r.stdout)
    assert payload["reconcile"]["clean"] is True
    assert payload["strategies"][0]["strategy"] == "cross_sectional_momentum"


def test_live_allocate_records_and_enforces_sum(monkeypatch):
    from contextlib import closing

    from algua.config.settings import get_settings
    from algua.registry.db import connect, migrate
    monkeypatch.setattr("algua.cli.live_cmd._live_account_equity", lambda: 50_000.0)
    assert runner.invoke(app, ["registry", "add", "s1"]).exit_code == 0
    r = runner.invoke(app, ["live", "allocate", "s1", "--capital", "10000"])
    assert r.exit_code == 0, r.stdout
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        n = conn.execute("SELECT COUNT(*) FROM strategy_allocations WHERE revoked_ts IS NULL"
                         ).fetchone()[0]
        assert n == 1
    # over-commit refused
    runner.invoke(app, ["registry", "add", "s2"])
    r2 = runner.invoke(app, ["live", "allocate", "s2", "--capital", "45000"])
    assert r2.exit_code == 1 and json.loads(r2.stdout)["ok"] is False


def test_run_all_rejects_bad_max_drawdown():
    r = runner.invoke(app, ["live", "run-all", "--snapshot", "x", "--max-drawdown", "1.5"])
    assert r.exit_code == 1 and json.loads(r.stdout)["ok"] is False
