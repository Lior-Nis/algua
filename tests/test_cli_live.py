import json

import pytest
from typer.testing import CliRunner

from algua.cli.main import app
from algua.contracts.types import LiveAuthorization

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
    # CANDIDATE via human: scaffolding to live, not exercising the agent shortlist gate.
    for to, actor in (("candidate", "human"), ("paper", "agent")):
        runner.invoke(app, ["registry", "transition", name, "--to", to, "--actor", actor,
                            "--reason", "x"])
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        conn.execute("UPDATE strategies SET stage='live' WHERE name=?", (name,))
        conn.commit()


def _auth():
    return LiveAuthorization(1, "c", "cf", "d", "lior", "t")



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
    monkeypatch.setattr("algua.cli.live_cmd._broker_buying_power", lambda broker: 100_000.0)
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
            self.offsets.append((symbol, qty))
            return "off"
        def account(self):
            from algua.execution.alpaca_broker import AccountState
            return AccountState(equity=100_000.0, cash=100_000.0, buying_power=100_000.0)

    broker = _LiqBroker()
    monkeypatch.setattr("algua.cli.live_cmd._alpaca_live_broker", lambda auth: broker)
    monkeypatch.setattr("algua.cli.live_cmd._select_provider", lambda demo, snapshot: object())
    monkeypatch.setattr("algua.cli.live_cmd._broker_account_activities", lambda b, a: [])
    monkeypatch.setattr("algua.cli.live_cmd._broker_net_positions", lambda b: {})
    monkeypatch.setattr("algua.cli.live_cmd.believed_positions",
                        lambda conn, name: {"AAA": 5.0})  # strategy believes it holds 5 AAA
    monkeypatch.setattr("algua.cli.live_cmd.active_allocation",
                        lambda conn, sid: {"capital": 10_000.0})
    monkeypatch.setattr("algua.cli.live_cmd.run_tick",
                        lambda *a, **k: (_ for _ in ()).throw(RiskBreach("drawdown", "dd")))
    r = runner.invoke(app, ["live", "run-all", "--snapshot", "x"])
    assert r.exit_code == 1
    assert broker.offsets == [("AAA", 5.0)]  # offset sized to the believed qty
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


def test_run_all_reserves_buying_power_across_strategies(monkeypatch):
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


def test_run_all_forwards_start_end_to_tick(monkeypatch):
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
    from algua.execution.live_ledger import believed_positions
    from algua.registry.db import connect, migrate
    from algua.risk import kill_switch
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        assert kill_switch.is_tripped(conn, name)
        # operator-trap: offset fills have NOT been ingested yet -> belief still non-flat
        assert believed_positions(conn, name) == {"AAA": 5.0, "ZZZ": 3.0}

    # 2) resume ingests the offset fills, reconciles to flat, clears the kill-switch (zero drift)
    r2 = runner.invoke(app, ["paper", "resume", name])
    assert r2.exit_code == 0, r2.stdout
    assert json.loads(r2.stdout)["kill_switch"] == "reset"
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        assert not kill_switch.is_tripped(conn, name)
        assert believed_positions(conn, name) == {}   # ledger flat after resume


def test_resume_sibling_holds_same_symbol_does_not_block(monkeypatch):
    """Strategy A is flat after ingest; sibling B legitimately holds AAA. The account-wide reconcile
    explains the broker's AAA via B's ledger, so resuming A is NOT blocked."""
    name = "cross_sectional_momentum"
    _to_live(name)
    from contextlib import closing

    from algua.config.settings import get_settings
    from algua.registry.db import connect, migrate
    from algua.risk import kill_switch
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        # sibling B holds AAA (5) in its own ledger; A holds nothing
        conn.execute(
            "INSERT INTO live_fills(activity_id, broker_order_id, strategy, symbol, qty, price, "
            "fill_ts) VALUES (?,?,?,?,?,?,?)",
            ("sib-aaa", "bo-sib", "sibling_live", "AAA", 5.0, 100.0, "2023-01-01T00:00:00Z"),
        )
        kill_switch.trip(conn, name, reason="manual", actor="system")
        conn.commit()

    # broker shows AAA+5 (B's), no activities; A's own ledger is empty
    broker = _ScriptedReadOnlyBroker(activities=[], positions={"AAA": 5.0})
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_live_readonly_from_settings", lambda: broker)

    r = runner.invoke(app, ["paper", "resume", name])
    assert r.exit_code == 0, r.stdout
    assert json.loads(r.stdout)["kill_switch"] == "reset"
