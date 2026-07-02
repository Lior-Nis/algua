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


def _permissive_book(monkeypatch):
    """Stub the #389 book-level exposure build with a large-equity, empty long-only book so book
    caps never bind — for run-all tests that assert the pre-existing tick/pool/breach behavior.
    (Book-limit enforcement itself is covered by test_book_limits.py + test_live_book_limits.py.)"""
    from algua.risk.book_limits import BookExposure, BookRiskLimits
    monkeypatch.setattr(
        "algua.cli.live_cmd._build_book_exposure",
        lambda *a, **k: (BookExposure(1e15, {}, BookRiskLimits()), None),
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
        allocations.allocate(conn, SqliteStrategyRepository(conn).get(name).id,
                             capital=capital, actor="human", account_equity=50_000.0)


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
        migrate(conn)
        conn.execute("UPDATE strategies SET stage='live' WHERE name='second_live'")
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
