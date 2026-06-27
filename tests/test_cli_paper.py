import json
from datetime import UTC, datetime

import pytest
from typer.testing import CliRunner

from algua.cli.main import app
from algua.execution.alpaca_broker import AccountState, BrokerError
from algua.live.live_loop import SubmittedOrder, TickResult
from algua.risk.limits import RiskBreach

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "p.db"))
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))


def _to_paper(name="cross_sectional_momentum"):
    assert runner.invoke(app, ["backtest", "run", name, "--demo", "--register",
                               "--start", "2022-01-01", "--end", "2023-12-31"]).exit_code == 0
    assert runner.invoke(app, ["registry", "transition", name, "--to", "candidate",
                               "--actor", "human", "--reason", "ok"]).exit_code == 0
    assert runner.invoke(app, ["registry", "transition", name, "--to", "paper",
                               "--actor", "agent", "--reason", "paper"]).exit_code == 0


def test_paper_run_executes_and_reconciles():
    _to_paper()
    result = runner.invoke(app, ["paper", "run", "cross_sectional_momentum", "--demo",
                                 "--start", "2022-01-01", "--end", "2023-12-31"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["ok"] is True  # success envelope discriminator (mirrors {"ok": false} failures)
    assert payload["strategy"] == "cross_sectional_momentum"
    assert payload["reconcile_ok"] is True
    assert payload["orders"] >= 1
    show = json.loads(runner.invoke(app, ["paper", "show", "cross_sectional_momentum"]).stdout)
    assert show["ok"] is True
    assert show["n_orders"] >= 1


def test_paper_run_rejects_non_paper_stage():
    runner.invoke(app, ["registry", "add", "cross_sectional_momentum"])  # stage = idea
    result = runner.invoke(app, ["paper", "run", "cross_sectional_momentum", "--demo",
                                 "--start", "2022-01-01", "--end", "2023-12-31"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False



def test_dormant_strategy_not_run_by_paper_lane():
    """A dormant strategy is rejected by `paper run` with a non-zero exit.

    The stage guard in _load_gated_strategy fires before any heavy work: it checks that the
    strategy is at Stage.PAPER or Stage.FORWARD_TESTED. A dormant strategy is neither, so the
    command exits 1 with {"ok": false} containing a stage/eligibility message.
    """
    # Register and drive to paper, then bench to dormant.
    assert runner.invoke(app, ["backtest", "run", "cross_sectional_momentum", "--demo",
                               "--register",
                               "--start", "2022-01-01", "--end", "2023-12-31"]).exit_code == 0
    assert runner.invoke(app, ["registry", "transition", "cross_sectional_momentum",
                               "--to", "candidate", "--actor", "human",
                               "--reason", "ok"]).exit_code == 0
    assert runner.invoke(app, ["registry", "transition", "cross_sectional_momentum",
                               "--to", "paper", "--actor", "agent",
                               "--reason", "paper"]).exit_code == 0
    assert runner.invoke(app, ["registry", "transition", "cross_sectional_momentum",
                               "--to", "dormant", "--actor", "agent",
                               "--reason", "seasonal"]).exit_code == 0

    # paper run must reject the dormant strategy before any data/provider work.
    result = runner.invoke(app, ["paper", "run", "cross_sectional_momentum", "--demo",
                                 "--start", "2022-01-01", "--end", "2023-12-31"])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    # The guard message names the stage and the required stages.
    assert "dormant" in payload.get("error", "").lower()


def test_manual_kill_blocks_run_then_resume_allows(monkeypatch):
    _to_paper()
    assert runner.invoke(app, ["paper", "kill", "cross_sectional_momentum",
                               "--reason", "manual"]).exit_code == 0
    blocked = runner.invoke(app, ["paper", "run", "cross_sectional_momentum", "--demo",
                                  "--start", "2022-01-01", "--end", "2023-12-31"])
    assert blocked.exit_code == 1
    assert json.loads(blocked.stdout)["ok"] is False
    assert runner.invoke(app, ["paper", "resume", "cross_sectional_momentum"]).exit_code == 0
    ok = runner.invoke(app, ["paper", "run", "cross_sectional_momentum", "--demo",
                             "--start", "2022-01-01", "--end", "2023-12-31"])
    assert ok.exit_code == 0


def test_resume_rebases_drawdown_peak():
    # After a drawdown halt the account is flattened to a lower equity; resume must clear the
    # persisted peak, else the next tick re-trips the breaker against the stale pre-loss high.
    from contextlib import closing

    from algua.config.settings import get_settings
    from algua.execution.order_state import get_peak_equity, update_peak_equity
    from algua.registry.db import connect, migrate

    _to_paper()
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        update_peak_equity(conn, "cross_sectional_momentum", 200_000.0)
    runner.invoke(app, ["paper", "kill", "cross_sectional_momentum", "--reason", "drawdown"])
    assert runner.invoke(app, ["paper", "resume", "cross_sectional_momentum"]).exit_code == 0
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        assert get_peak_equity(conn, "cross_sectional_momentum") is None


def test_breach_trips_killswitch_and_persists_nothing(monkeypatch):
    _to_paper()

    def _boom(*a, **k):
        raise RiskBreach("drawdown", "drawdown 0.30 exceeds max_drawdown 0.10")

    monkeypatch.setattr("algua.cli.paper_cmd.run_paper", _boom)
    result = runner.invoke(app, ["paper", "run", "cross_sectional_momentum", "--demo",
                                 "--start", "2022-01-01", "--end", "2023-12-31",
                                 "--max-drawdown", "0.1"])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False and payload["kind"] == "drawdown"
    show = json.loads(runner.invoke(app, ["paper", "show", "cross_sectional_momentum"]).stdout)
    assert show["kill_switch"]["tripped"] is True
    assert show["n_orders"] == 0


def test_kill_rejects_unknown_strategy():
    result = runner.invoke(app, ["paper", "kill", "no_such_strategy", "--reason", "x"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False


@pytest.mark.parametrize(
    "argv",
    [
        ["paper", "kill", "whatever", "--reason", "x", "--actor", "humn"],
        ["paper", "flatten", "whatever", "--actor", "humn"],
        ["paper", "halt-all", "--reason", "x", "--actor", "humn"],
        ["paper", "resume-all", "--actor", "humn"],
    ],
)
def test_operational_commands_reject_bad_actor(argv):
    """A typo'd --actor fails closed via Actor() coercion before any switch/halt is touched (#259).

    The coercion is the first line of each command body, so an invalid actor is rejected
    before the DB/broker is reached — no mis-attributed audit/kill-switch row is written.
    """
    result = runner.invoke(app, argv)
    assert result.exit_code == 1, result.stdout
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert "humn" in payload["error"]  # the bad actor token surfaces, not an unrelated failure


def test_paper_account_missing_creds_errors(monkeypatch):
    # Empty env vars override any local .env (env > .env in pydantic-settings) so this stays
    # hermetic even on a developer machine that has real Alpaca keys in .env.
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "")
    result = runner.invoke(app, ["paper", "account"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False


def test_paper_account_emits_balances(monkeypatch):
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    monkeypatch.setattr(
        "algua.cli.paper_cmd.AlpacaPaperBroker.account",
        lambda self: AccountState(equity=100000.0, cash=50000.0, buying_power=150000.0),
    )
    result = runner.invoke(app, ["paper", "account"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["equity"] == 100000.0 and payload["cash"] == 50000.0


def test_trade_tick_rejects_non_paper_stage(monkeypatch):
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    runner.invoke(app, ["registry", "add", "cross_sectional_momentum"])  # stage = idea
    result = runner.invoke(app, ["paper", "trade-tick", "cross_sectional_momentum",
                                 "--snapshot", "snap1"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False


def test_trade_tick_refused_when_killed(monkeypatch):
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    _to_paper()
    runner.invoke(app, ["paper", "kill", "cross_sectional_momentum", "--reason", "x"])
    result = runner.invoke(app, ["paper", "trade-tick", "cross_sectional_momentum",
                                 "--snapshot", "snap1"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False


def test_trade_tick_old_name_removed(monkeypatch):
    # #28: the old `trade-live` name is gone (no alias) — invoking it must error.
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    result = runner.invoke(app, ["paper", "trade-live", "cross_sectional_momentum",
                                 "--snapshot", "snap1"])
    assert result.exit_code != 0


class _MinimalBroker:
    """Minimal broker stub with the account/clock methods trade-tick now requires."""
    def account(self):
        return AccountState(equity=99000.0, cash=10000.0, buying_power=89000.0,
                            account_id="test-acct")

    def clock(self):
        return "2023-06-01T14:00:00+00:00"


def test_trade_tick_submits_and_persists(monkeypatch):
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    _to_paper()
    ts = datetime(2023, 6, 1, tzinfo=UTC)
    fake_result = TickResult(
        decision_ts=ts, target_weights={"AAA": 1.0}, positions_before={},
        submitted=[{"symbol": "AAA", "side": "buy", "target_weight": 1.0, "order_id": "o-1",
                    "client_order_id": "c-1"}],
        peak_equity=100_000.0,
    )

    def _fake_run_tick(strategy, broker, provider, start, end, hooks=None, max_drawdown=None):
        # exercise the immediate-persist hook the CLI wires up (#18)
        hooks.on_submitted(SubmittedOrder(symbol="AAA", side="buy", target_weight=1.0,
                                          order_id="o-1", client_order_id="c-1", decision_ts=ts))
        return fake_result

    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings", _MinimalBroker)
    monkeypatch.setattr("algua.cli.paper_cmd._select_provider", lambda demo, snapshot: object())
    monkeypatch.setattr("algua.cli.paper_cmd.run_tick", _fake_run_tick)
    result = runner.invoke(app, ["paper", "trade-tick", "cross_sectional_momentum",
                                 "--snapshot", "snap1"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["ok"] is True  # success envelope discriminator
    assert payload["submitted"][0]["order_id"] == "o-1"
    show = json.loads(runner.invoke(app, ["paper", "show", "cross_sectional_momentum"]).stdout)
    assert show["n_orders"] == 1


class _FlattenBroker:
    def __init__(self, fail=False):
        self.fail = fail
        self.cancelled = False
        self.closed_symbols = None

    def cancel_open_orders(self):
        self.cancelled = True

    def close_positions(self, symbols):
        if self.fail:
            raise BrokerError("alpaca failed to close some positions: [...]")
        self.closed_symbols = list(symbols)


def test_paper_flatten_closes_and_trips(monkeypatch):
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    _to_paper()
    broker = _FlattenBroker()
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings", lambda: broker)
    result = runner.invoke(app, ["paper", "flatten", "cross_sectional_momentum"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["liquidation_submitted"] is True and payload["kill_switch"] == "tripped"
    # scoped to the strategy's universe (a symbol list), not an account-wide close
    assert broker.cancelled is True
    assert isinstance(broker.closed_symbols, list) and broker.closed_symbols
    show = json.loads(runner.invoke(app, ["paper", "show", "cross_sectional_momentum"]).stdout)
    assert show["kill_switch"]["tripped"] is True


def test_paper_flatten_allowed_at_forward_tested_stage(monkeypatch):
    """A certified forward_tested strategy still holds paper positions while awaiting the go-live
    signature — emergency flatten must work there too (#124 GATE-2)."""
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    name = "cross_sectional_momentum"
    _to_paper()
    assert runner.invoke(
        app, ["registry", "transition", name, "--to", "forward_tested",
              "--actor", "human", "--reason", "gate passed"]
    ).exit_code == 0
    broker = _FlattenBroker()
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings", lambda: broker)
    result = runner.invoke(app, ["paper", "flatten", name])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["liquidation_submitted"] is True and payload["kill_switch"] == "tripped"
    assert broker.cancelled is True
    assert isinstance(broker.closed_symbols, list) and broker.closed_symbols
    show = json.loads(runner.invoke(app, ["paper", "show", name]).stdout)
    assert show["kill_switch"]["tripped"] is True


def test_paper_flatten_rejects_non_paper_stage(monkeypatch):
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    runner.invoke(app, ["registry", "add", "cross_sectional_momentum"])  # idea
    result = runner.invoke(app, ["paper", "flatten", "cross_sectional_momentum"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False


def test_paper_flatten_close_failure_stays_tripped(monkeypatch):
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    _to_paper()
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings",
                        lambda: _FlattenBroker(fail=True))
    result = runner.invoke(app, ["paper", "flatten", "cross_sectional_momentum"])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False and payload["kill_switch"] == "tripped"
    show = json.loads(runner.invoke(app, ["paper", "show", "cross_sectional_momentum"]).stdout)
    assert show["kill_switch"]["tripped"] is True


def test_trade_tick_persists_snapshot(monkeypatch):
    from contextlib import closing

    from algua.config.settings import get_settings
    from algua.execution.order_state import latest_tick_snapshot
    from algua.registry.db import connect, migrate

    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    _to_paper()
    fake = TickResult(decision_ts=datetime(2023, 6, 1, tzinfo=UTC), target_weights={"AAA": 1.0},
                      positions_before={"AAA": 5.0}, submitted=[{"symbol": "AAA"}],
                      equity=99000.0, peak_equity=99000.0)
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings", _MinimalBroker)
    monkeypatch.setattr("algua.cli.paper_cmd._select_provider", lambda demo, snapshot: object())
    monkeypatch.setattr("algua.cli.paper_cmd.run_tick", lambda *a, **k: fake)
    result = runner.invoke(app, ["paper", "trade-tick", "cross_sectional_momentum",
                                 "--snapshot", "snap1"])
    assert result.exit_code == 0, result.stdout
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        snap = latest_tick_snapshot(conn, "cross_sectional_momentum")
    assert snap is not None and snap["equity"] == 99000.0
    assert snap["positions"] == {"AAA": 5.0} and snap["n_submitted"] == 1


def _seed_snapshot(name, *, equity, peak, reconcile_ok=True, positions=None):
    from contextlib import closing

    from algua.config.settings import get_settings
    from algua.execution.order_state import record_tick_snapshot, update_peak_equity
    from algua.registry.db import connect, migrate
    from algua.registry.store import SqliteStrategyRepository
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        update_peak_equity(conn, name, peak)
        rec = SqliteStrategyRepository(conn).get(name)
        record_tick_snapshot(conn, name, tick_ts="2023-06-01T00:00:00+00:00",
                             decision_ts="2023-05-31T00:00:00+00:00", equity=equity,
                             peak_equity=peak, positions=positions or {}, n_submitted=0,
                             reconcile_ok=reconcile_ok, lane="paper", strategy_id=rec.id,
                             code_hash="c", config_hash="g", dependency_hash=None,
                             account_id="test", cash=0.0, clock_source="local")


def test_show_consolidated_view():
    _to_paper()
    _seed_snapshot("cross_sectional_momentum", equity=90.0, peak=100.0, positions={"AAA": 3.0})
    payload = json.loads(runner.invoke(app, ["paper", "show", "cross_sectional_momentum"]).stdout)
    assert payload["stage"] == "paper"
    assert payload["drawdown"]["peak_equity"] == 100.0
    assert payload["drawdown"]["last_equity"] == 90.0
    assert abs(payload["drawdown"]["drawdown"] - 0.10) < 1e-9
    assert payload["last_tick"]["positions"] == {"AAA": 3.0}
    assert payload["health"] == "ok"
    assert "recent_orders" in payload


def test_show_health_halted():
    _to_paper()
    runner.invoke(app, ["paper", "kill", "cross_sectional_momentum", "--reason", "x"])
    payload = json.loads(runner.invoke(app, ["paper", "show", "cross_sectional_momentum"]).stdout)
    assert payload["health"] == "halted"


def test_show_health_drift():
    _to_paper()
    _seed_snapshot("cross_sectional_momentum", equity=90.0, peak=100.0, reconcile_ok=False)
    payload = json.loads(runner.invoke(app, ["paper", "show", "cross_sectional_momentum"]).stdout)
    assert payload["health"] == "drift"


def test_show_health_idle_no_ticks():
    _to_paper()
    payload = json.loads(runner.invoke(app, ["paper", "show", "cross_sectional_momentum"]).stdout)
    assert payload["health"] == "idle" and payload["last_tick"] is None


def test_show_unknown_strategy_errors():
    result = runner.invoke(app, ["paper", "show", "no_such_strategy"])
    assert result.exit_code == 1 and json.loads(result.stdout)["ok"] is False


class _HaltBroker:
    def __init__(self, fail=False):
        self.fail = fail
        self.closed_all = False

    def close_all_positions(self):
        if self.fail:
            raise BrokerError("alpaca failed to close some positions: [...]")
        self.closed_all = True


def test_halt_all_engages_and_flattens(monkeypatch):
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    broker = _HaltBroker()
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings", lambda: broker)
    result = runner.invoke(app, ["paper", "halt-all", "--reason", "panic"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["global_halt"] == "set" and payload["liquidation_submitted"] is True
    assert broker.closed_all is True


def test_halt_all_close_failure_stays_engaged(monkeypatch):
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings",
                        lambda: _HaltBroker(fail=True))
    result = runner.invoke(app, ["paper", "halt-all", "--reason", "panic"])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False and payload["global_halt"] == "set"
    assert payload["liquidation_submitted"] is False
    from contextlib import closing

    from algua.config.settings import get_settings
    from algua.registry.db import connect, migrate
    from algua.risk import global_halt
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        assert global_halt.is_engaged(conn) is True  # still engaged (fail-safe)


def test_resume_all_clears_and_wipes_peaks_but_keeps_strategy_switch(monkeypatch):
    from contextlib import closing

    from algua.config.settings import get_settings
    from algua.execution.order_state import get_peak_equity, update_peak_equity
    from algua.registry.db import connect, migrate
    from algua.risk import global_halt, kill_switch

    _to_paper()
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        global_halt.engage(conn, reason="x", actor="human")
        update_peak_equity(conn, "cross_sectional_momentum", 100.0)
        kill_switch.trip(conn, "cross_sectional_momentum", reason="indiv", actor="human")
    result = runner.invoke(app, ["paper", "resume-all"])
    assert result.exit_code == 0, result.stdout
    assert json.loads(result.stdout)["global_halt"] == "reset"
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        assert global_halt.is_engaged(conn) is False
        assert get_peak_equity(conn, "cross_sectional_momentum") is None  # peaks wiped
        assert kill_switch.is_tripped(conn, "cross_sectional_momentum") is True  # untouched


def test_resume_all_default_actor_is_agent_in_audit():
    """resume-all's default --actor is 'agent' (matching its sibling halt commands), so the
    audit row isn't mislabeled 'human' when an agent invokes it with the default (#272)."""
    from contextlib import closing

    from algua.audit import log as audit_log
    from algua.config.settings import get_settings
    from algua.registry.db import connect, migrate
    from algua.risk import global_halt

    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        global_halt.engage(conn, reason="x", actor="human")
    result = runner.invoke(app, ["paper", "resume-all"])  # no --actor: use the default
    assert result.exit_code == 0, result.stdout
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        rows = audit_log.read(conn)
        resume_rows = [r for r in rows if r["action"] == "resume_all"]
        assert resume_rows, "expected a resume_all audit row"
        assert resume_rows[0]["actor"] == "agent"


def _engage_global_halt():
    from contextlib import closing

    from algua.config.settings import get_settings
    from algua.registry.db import connect, migrate
    from algua.risk import global_halt
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        global_halt.engage(conn, reason="halted", actor="human")


def test_trade_tick_refused_when_globally_halted(monkeypatch):
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    _to_paper()
    _engage_global_halt()
    result = runner.invoke(app, ["paper", "trade-tick", "cross_sectional_momentum",
                                 "--snapshot", "snap1"])
    assert result.exit_code == 1 and json.loads(result.stdout)["ok"] is False


def test_paper_run_refused_when_globally_halted():
    _to_paper()
    _engage_global_halt()
    result = runner.invoke(app, ["paper", "run", "cross_sectional_momentum", "--demo",
                                 "--start", "2022-01-01", "--end", "2023-12-31"])
    assert result.exit_code == 1 and json.loads(result.stdout)["ok"] is False


def test_show_reflects_global_halt():
    _to_paper()
    _engage_global_halt()
    payload = json.loads(runner.invoke(app, ["paper", "show", "cross_sectional_momentum"]).stdout)
    assert payload["health"] == "halted"
    assert payload["kill_switch"]["global_halt"] is True


def _seed_paper_order(db_path, strategy, symbol):
    from contextlib import closing

    from algua.registry.db import connect, migrate
    with closing(connect(db_path)) as conn:
        migrate(conn)
        cur = conn.execute(
            "INSERT INTO paper_orders(strategy, symbol, side, target_weight, decision_ts, "
            "submitted_ts, status, broker_order_id) VALUES (?,?,?,?,?,?,?,?)",
            (strategy, symbol, "buy", 0.5, "2023-01-01T00:00:00Z", "2023-01-01T00:00:00Z",
             "filled", f"bo-{strategy}-{symbol}"),
        )
        conn.execute(
            "INSERT INTO paper_fills(order_id, symbol, qty, price, fill_ts) VALUES (?,?,?,?,?)",
            (cur.lastrowid, symbol, 5.0, 100.0, "2023-01-01T00:00:00Z"),
        )
        conn.commit()


def test_paper_flatten_closes_dropped_symbol_not_siblings(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    _to_paper()
    db = tmp_path / "p.db"
    # the strategy holds ZZZ (a symbol no longer in its universe); a SIBLING strategy holds SIB
    _seed_paper_order(db, "cross_sectional_momentum", "ZZZ")
    _seed_paper_order(db, "sibling_strat", "SIB")

    broker = _FlattenBroker()
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings", lambda: broker)
    result = runner.invoke(app, ["paper", "flatten", "cross_sectional_momentum"])
    assert result.exit_code == 0, result.stdout

    closed = set(broker.closed_symbols)
    assert "ZZZ" in closed                     # held-but-dropped symbol IS closed
    assert "SIB" not in closed                 # sibling's symbol is NOT closed
    assert "ZZZ" in json.loads(result.stdout)["closed_symbols"]

    # the strategy's derived belief is reset to flat after the close
    show = json.loads(runner.invoke(app, ["paper", "show", "cross_sectional_momentum"]).stdout)
    assert show["positions"] == {}


# ---------------------------------------------------------------------------
# Task 6: ledger-flat resume gate
# ---------------------------------------------------------------------------

def _seed_live_killed_with_position(monkeypatch, tmp_path, name="cross_sectional_momentum"):
    """Bring a strategy to LIVE stage, trip its kill-switch, and insert a live_fills row so
    believed_positions is non-empty. Returns the strategy name."""
    from contextlib import closing

    from algua.config.settings import get_settings
    from algua.registry.db import connect, migrate
    from algua.risk import kill_switch

    # Advance through the registry to the paper stage first (CLI path).
    _to_paper(name)

    # Forcibly set stage = live directly in the DB (bypassing the signed go-live challenge).
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        conn.execute("UPDATE strategies SET stage = 'live' WHERE name = ?", (name,))
        conn.commit()

    # Trip the kill-switch (simulate a drawdown halt or manual stop).
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        kill_switch.trip(conn, name, reason="test-breach", actor="system")

    # Insert a live_fills row so believed_positions returns a non-empty dict.
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        conn.execute(
            "INSERT INTO live_fills"
            "(activity_id, broker_order_id, strategy, symbol, qty, price, fill_ts)"
            " VALUES (?,?,?,?,?,?,?)",
            ("act-1", "boid-1", name, "AAA", 5.0, 100.0, "2026-06-06T00:00:00+00:00"),
        )
        conn.commit()

    return name


def _clear_belief(tmp_path, name="cross_sectional_momentum"):
    """Delete all live_fills for the strategy so believed_positions returns empty (flat)."""
    from contextlib import closing

    from algua.config.settings import get_settings
    from algua.registry.db import connect, migrate

    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        conn.execute("DELETE FROM live_fills WHERE strategy = ?", (name,))
        conn.commit()


def test_resume_refused_while_live_strategy_not_flat(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_ALPACA_LIVE_API_KEY", "lk")
    monkeypatch.setenv("ALGUA_ALPACA_LIVE_API_SECRET", "ls")
    name = _seed_live_killed_with_position(monkeypatch, tmp_path)
    # broker still holds AAA (non-flat): ledger has a fill, broker confirms the position
    broker_with_position = _ReadOnlyLiveBroker(activities=[], positions={"AAA": 5.0})
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_live_readonly_from_settings",
                        lambda: broker_with_position)
    r = runner.invoke(app, ["paper", "resume", name])
    assert r.exit_code == 1 and "not flat" in r.stdout.lower()
    # once flat (belief cleared AND broker reports flat), resume succeeds
    _clear_belief(tmp_path, name)
    flat_broker = _ReadOnlyLiveBroker(activities=[], positions={})
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_live_readonly_from_settings",
                        lambda: flat_broker)
    assert runner.invoke(app, ["paper", "resume", name]).exit_code == 0


def test_resume_clears_live_nav_peak(monkeypatch, tmp_path):
    from contextlib import closing

    from algua.config.settings import get_settings
    from algua.execution.order_state import get_nav_peak, update_nav_peak
    from algua.registry.db import connect, migrate
    monkeypatch.setenv("ALGUA_ALPACA_LIVE_API_KEY", "lk")
    monkeypatch.setenv("ALGUA_ALPACA_LIVE_API_SECRET", "ls")
    name = _seed_live_killed_with_position(monkeypatch, tmp_path)
    _clear_belief(tmp_path, name)                       # make it flat so resume is allowed
    flat_broker = _ReadOnlyLiveBroker(activities=[], positions={})
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_live_readonly_from_settings",
                        lambda: flat_broker)
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        update_nav_peak(conn, name, 12_000.0)           # a stale pre-breach NAV peak
    assert runner.invoke(app, ["paper", "resume", name]).exit_code == 0
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        assert get_nav_peak(conn, name) is None          # cleared on resume (else it re-trips)


# ---------------------------------------------------------------------------
# Task 4 (C2): stage-aware paper show — live -> believed positions + NAV peak
# ---------------------------------------------------------------------------

def test_show_live_strategy_uses_believed_positions_and_nav_peak(monkeypatch, tmp_path):
    import json
    from contextlib import closing

    from algua.config.settings import get_settings
    from algua.execution.order_state import update_nav_peak
    from algua.registry.db import connect, migrate
    name = _seed_live_killed_with_position(monkeypatch, tmp_path)  # live stage + live_fills belief
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        update_nav_peak(conn, name, 12_345.0)
    payload = json.loads(runner.invoke(app, ["paper", "show", name]).stdout)
    assert payload["drawdown"]["peak_equity"] == 12_345.0  # NAV peak, not the (absent) paper peak
    assert payload["positions"]                             # believed positions, not empty paper


# ---------------------------------------------------------------------------
# Task 5 (#124): stamped writers — trade-tick persists provenance columns
# ---------------------------------------------------------------------------

def test_trade_tick_persists_provenance(monkeypatch):
    """Tick snapshot written by trade-tick carries lane, registry id, identity hashes,
    account_id, cash, clock_source, and tick_ts derived from the mocked broker clock."""
    from contextlib import closing

    from algua.config.settings import get_settings
    from algua.execution.order_state import latest_tick_snapshot
    from algua.registry.db import connect, migrate
    from algua.registry.store import SqliteStrategyRepository

    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    _to_paper()

    name = "cross_sectional_momentum"
    # Venue clock reports an EDT offset: the stamp must be the converted UTC instant, pinned as a
    # literal below — never recompute it with the same expression the implementation uses.
    clock_ts = "2026-06-11T10:00:00-04:00"
    fake_result = TickResult(
        decision_ts=datetime(2026, 6, 11, 14, 0, 0, tzinfo=UTC),
        target_weights={"AAA": 1.0},
        positions_before={},
        submitted=[],
        equity=50_000.0,
        peak_equity=50_000.0,
    )

    class _FakeBroker:
        def clock(self):
            return clock_ts

        def account(self):
            return AccountState(equity=50_000.0, cash=10_000.0, buying_power=40_000.0,
                                account_id="acct-xyz")

    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings", _FakeBroker)
    monkeypatch.setattr("algua.cli.paper_cmd._select_provider", lambda demo, snapshot: object())
    monkeypatch.setattr("algua.cli.paper_cmd.run_tick", lambda *a, **k: fake_result)

    result = runner.invoke(app, ["paper", "trade-tick", name, "--snapshot", "snap1"])
    assert result.exit_code == 0, result.stdout

    from algua.registry.approvals import compute_artifact_hashes
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        snap = latest_tick_snapshot(conn, name)
        rec = SqliteStrategyRepository(conn).get(name)

    identity = compute_artifact_hashes(name)
    assert snap is not None
    assert snap["lane"] == "paper"
    assert snap["strategy_id"] == rec.id
    assert snap["code_hash"] == identity.code_hash
    assert snap["config_hash"] == identity.config_hash
    assert snap["dependency_hash"] == identity.dependency_hash
    assert snap["account_id"] == "acct-xyz"
    assert snap["cash"] == 10_000.0
    assert snap["clock_source"] == "broker"
    # tick_ts is the broker clock ts (10:00-04:00), normalized to UTC: hour shifted, +00:00 offset
    assert snap["tick_ts"] == "2026-06-11T14:00:00+00:00"


def _raise_broker_error():
    raise BrokerError("clock unavailable")


@pytest.mark.parametrize("bad_clock", [
    _raise_broker_error,                      # venue clock endpoint failed
    lambda: "2026-06-11T14:00:00",            # tz-naive ts: tz_convert raises TypeError
    lambda: "not-a-timestamp",                # malformed ts: pd.Timestamp raises ValueError
], ids=["broker_error", "naive_ts", "malformed_ts"])
def test_trade_tick_unusable_broker_clock_falls_back_to_local(monkeypatch, bad_clock):
    """Any unusable venue clock — endpoint failure, naive ts, garbage ts — falls back to
    clock_source='local' and the tick is still recorded (never crash after orders went out)."""
    from contextlib import closing

    from algua.config.settings import get_settings
    from algua.execution.order_state import latest_tick_snapshot
    from algua.registry.db import connect, migrate

    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    _to_paper()

    name = "cross_sectional_momentum"
    fake_result = TickResult(
        decision_ts=datetime(2026, 6, 11, 14, 0, 0, tzinfo=UTC),
        target_weights={},
        positions_before={},
        submitted=[],
        equity=50_000.0,
        peak_equity=50_000.0,
    )

    class _ClockFailBroker:
        def clock(self):
            return bad_clock()

        def account(self):
            return AccountState(equity=50_000.0, cash=10_000.0, buying_power=40_000.0,
                                account_id="acct-xyz")

    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings", _ClockFailBroker)
    monkeypatch.setattr("algua.cli.paper_cmd._select_provider", lambda demo, snapshot: object())
    monkeypatch.setattr("algua.cli.paper_cmd.run_tick", lambda *a, **k: fake_result)

    result = runner.invoke(app, ["paper", "trade-tick", name, "--snapshot", "snap1"])
    assert result.exit_code == 0, result.stdout

    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        snap = latest_tick_snapshot(conn, name)

    assert snap is not None
    assert snap["clock_source"] == "local"
    assert snap["tick_ts"]  # some local timestamp was written


def test_trade_tick_allowed_at_forward_tested_stage(monkeypatch):
    """A strategy at stage forward_tested (human transition from paper) can still run trade-tick."""
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    _to_paper()

    name = "cross_sectional_momentum"
    # advance to forward_tested via a human transition
    assert runner.invoke(
        app, ["registry", "transition", name, "--to", "forward_tested",
              "--actor", "human", "--reason", "gate passed"]
    ).exit_code == 0

    fake_result = TickResult(
        decision_ts=datetime(2026, 6, 11, 14, 0, 0, tzinfo=UTC),
        target_weights={},
        positions_before={},
        submitted=[],
        equity=50_000.0,
        peak_equity=50_000.0,
    )

    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings", _MinimalBroker)
    monkeypatch.setattr("algua.cli.paper_cmd._select_provider", lambda demo, snapshot: object())
    monkeypatch.setattr("algua.cli.paper_cmd.run_tick", lambda *a, **k: fake_result)

    result = runner.invoke(app, ["paper", "trade-tick", name, "--snapshot", "snap1"])
    assert result.exit_code == 0, result.stdout
    assert json.loads(result.stdout)["ok"] is True


# ---------------------------------------------------------------------------
# Task 12 (#124): `algua paper promote` — the forward-test evidence gate CLI
# ---------------------------------------------------------------------------

_NAME = "cross_sectional_momentum"
_GATE_IDENT = None  # initialized lazily to avoid an import cost at collection


def _gate_ident():
    global _GATE_IDENT
    if _GATE_IDENT is None:
        from algua.registry.repository import ArtifactIdentity
        _GATE_IDENT = ArtifactIdentity(code_hash="c", config_hash="g", dependency_hash="d")
    return _GATE_IDENT


class _PromoteBroker:
    """Broker fake for the promote path: only the activities window is consulted."""

    def account_activities_window(self, after, until):
        return []


def _wire_promote(monkeypatch):
    """Pin identity the way tests/test_forward_promotion.py does (one IDENT for the recorded
    row AND the token-consume recheck), swap the heavy exchange calendar for weekday
    arithmetic, and stub the broker."""
    from tests.test_forward_promotion import FakeCalendar
    ident = _gate_ident()
    monkeypatch.setattr(
        "algua.registry.forward_promotion.compute_artifact_hashes", lambda name: ident)
    monkeypatch.setattr("algua.registry.transitions._compute_hashes", lambda name: ident)
    monkeypatch.setattr("algua.cli.paper_cmd.MarketCalendar", FakeCalendar)
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings", _PromoteBroker)


def _promote_conn():
    from algua.config.settings import get_settings
    from algua.registry.db import connect, migrate
    conn = connect(get_settings().db_path)
    migrate(conn)
    return conn


def _past_weekdays(n):
    """The n weekdays strictly before today (UTC), oldest first — every seeded tick_ts is in
    the past and the newest is at most one session stale, so the gate's now=datetime.now(UTC)
    needs no pinning."""
    from datetime import date, timedelta
    out, d = [], date.today() - timedelta(days=1)
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d -= timedelta(days=1)
    return list(reversed(out))


def _seed_passing_forward_window(name=_NAME, n=64):
    """64 admissible sessions (63 returns >= the floor) through the REAL tick writer, plus a
    qualified backtest gate row (holdout_sharpe=1.0 -> bar = max(.5*1.0, .3) = .5)."""
    from contextlib import closing
    from datetime import timedelta

    from algua.execution.order_state import record_tick_snapshot
    from algua.registry.store import SqliteStrategyRepository
    days = _past_weekdays(n)
    with closing(_promote_conn()) as conn:
        rec = SqliteStrategyRepository(conn).get(name)
        eq = 100.0
        for i, day in enumerate(days):
            decision = day - timedelta(days=1)
            while decision.weekday() >= 5:
                decision -= timedelta(days=1)
            record_tick_snapshot(
                conn, name,
                tick_ts=datetime(day.year, day.month, day.day, 20, tzinfo=UTC).isoformat(),
                decision_ts=datetime(decision.year, decision.month, decision.day, 20,
                                     tzinfo=UTC).isoformat(),
                equity=eq, peak_equity=None, positions={}, n_submitted=0, reconcile_ok=True,
                lane="paper", strategy_id=rec.id, code_hash="c", config_hash="g",
                dependency_hash="d", account_id="acct", cash=0.0, clock_source="broker")
            eq *= 1.004 if i % 2 == 0 else 0.999
        conn.execute(
            "INSERT INTO gate_evaluations(strategy_id, passed, n_funnel, own_lifetime_combos, "
            "windowed_total_combos, funnel_window_days, breadth_provenance, pit_ok, "
            "pit_override, holdout_n_bars, min_holdout_observations, code_hash, config_hash, "
            "dependency_hash, data_source, snapshot_id, period_start, period_end, "
            "holdout_frac, actor, decision_json, consumed, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (rec.id, 1, 1, 1, 1, 90, "measured", 1, 0, 100, 63, "c", "g", "d", "snapshot",
             None, "2026-01-01", "2026-06-01", 0.25, "agent",
             json.dumps({"checks": [{"name": "holdout_sharpe", "value": 1.0}]}), 0,
             "2026-06-10T00:00:00+00:00"))
        conn.commit()


def _stage_of(name=_NAME):
    from contextlib import closing
    with closing(_promote_conn()) as conn:
        return conn.execute("SELECT stage FROM strategies WHERE name=?", (name,)).fetchone()[0]


def test_paper_promote_happy_path_promotes(monkeypatch):
    _to_paper()
    _wire_promote(monkeypatch)
    _seed_passing_forward_window()
    result = runner.invoke(app, ["paper", "promote", _NAME])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["strategy"] == _NAME
    assert payload["passed"] is True
    assert payload["promoted"] is True
    assert isinstance(payload["decision"]["checks"], list) and payload["decision"]["checks"]
    assert isinstance(payload["excluded_ticks"], dict)
    assert all(v == 0 for v in payload["excluded_ticks"].values())
    assert payload["n_concurrent_forward"] == 1
    assert _stage_of() == "forward_tested"
    from contextlib import closing
    with closing(_promote_conn()) as conn:
        audit = conn.execute(
            "SELECT reason FROM audit_log WHERE action='paper_promote' AND strategy=?",
            (_NAME,)).fetchall()
    assert [r["reason"] for r in audit] == ["pass"]


def test_paper_promote_failing_gate_records_row_and_exits_1(monkeypatch):
    _to_paper()
    _wire_promote(monkeypatch)  # no ticks seeded: the window floor fails
    result = runner.invoke(app, ["paper", "promote", _NAME])
    assert result.exit_code == 1, result.stdout
    payload = json.loads(result.stdout)
    assert payload["ok"] is False  # the repo-wide exit-1 discriminator
    assert payload["passed"] is False
    assert payload["promoted"] is False
    assert isinstance(payload["excluded_ticks"], dict)
    assert _stage_of() == "paper"  # not transitioned
    from contextlib import closing
    with closing(_promote_conn()) as conn:
        rows = conn.execute(
            "SELECT passed, consumed FROM forward_gate_evaluations").fetchall()
        audit = conn.execute(
            "SELECT reason FROM audit_log WHERE action='paper_promote' AND strategy=?",
            (_NAME,)).fetchall()
    assert len(rows) == 1  # the failing evaluation WAS recorded
    assert rows[0]["passed"] == 0 and rows[0]["consumed"] == 0
    assert [r["reason"] for r in audit] == ["fail"]  # audited on fail too


def test_paper_promote_at_forward_tested_refreshes_certificate(monkeypatch):
    _to_paper()
    _wire_promote(monkeypatch)
    _seed_passing_forward_window()
    from contextlib import closing
    with closing(_promote_conn()) as conn:
        conn.execute("UPDATE strategies SET stage='forward_tested' WHERE name=?", (_NAME,))
        conn.commit()
    result = runner.invoke(app, ["paper", "promote", _NAME])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["passed"] is True
    assert payload["promoted"] is False  # certificate refresh, no stage change
    assert _stage_of() == "forward_tested"


def test_paper_promote_agent_relaxation_refused(monkeypatch):
    # Deliberately NO broker stub and NO creds: preflight must refuse a relaxation attempt
    # BEFORE the broker is even constructed (no credentials needed to be told no).
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "")
    _to_paper()
    result = runner.invoke(app, ["paper", "promote", _NAME, "--sharpe-floor", "0.2"])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert "sharpe_floor" in payload["error"] and "human" in payload["error"]
    from contextlib import closing
    with closing(_promote_conn()) as conn:  # refused BEFORE any evaluation row is minted
        assert conn.execute("SELECT COUNT(*) FROM forward_gate_evaluations").fetchone()[0] == 0


def test_paper_promote_agent_tightening_accepted(monkeypatch):
    _to_paper()
    _wire_promote(monkeypatch)
    _seed_passing_forward_window()
    result = runner.invoke(app, ["paper", "promote", _NAME, "--sharpe-floor", "0.5"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["passed"] is True and payload["promoted"] is True


def test_paper_promote_wrong_stage_refused(monkeypatch):
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    assert runner.invoke(app, ["backtest", "run", _NAME, "--demo", "--register",
                               "--start", "2022-01-01", "--end", "2023-12-31"]).exit_code == 0
    assert runner.invoke(app, ["registry", "transition", _NAME, "--to", "candidate",
                               "--actor", "human", "--reason", "ok"]).exit_code == 0
    result = runner.invoke(app, ["paper", "promote", _NAME])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert "candidate" in payload["error"]
    assert "paper or forward_tested" in payload["error"]


def test_paper_promote_missing_creds_json_error(monkeypatch):
    # Empty env vars override any local .env (env > .env in pydantic-settings) — hermetic, and
    # the same envelope discipline as trade-tick's missing-creds path.
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "")
    _to_paper()
    result = runner.invoke(app, ["paper", "promote", _NAME])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert "ALGUA_ALPACA_API_KEY" in payload["error"]


def test_trade_tick_breach_flattens_dropped_symbol_and_clears_belief(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    _to_paper()
    # the strategy holds ZZZ, a symbol no longer in its universe
    _seed_paper_order(tmp_path / "p.db", "cross_sectional_momentum", "ZZZ")

    class _BreachTickBroker(_MinimalBroker):
        def __init__(self):
            self.closed_symbols = None
        def cancel_open_orders(self):
            pass
        def close_positions(self, symbols):
            self.closed_symbols = list(symbols)

    broker = _BreachTickBroker()
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings", lambda: broker)
    monkeypatch.setattr("algua.cli.paper_cmd._select_provider", lambda demo, snapshot: object())
    monkeypatch.setattr("algua.cli.paper_cmd.run_tick",
                        lambda *a, **k: (_ for _ in ()).throw(RiskBreach("drawdown", "dd")))
    result = runner.invoke(app, ["paper", "trade-tick", "cross_sectional_momentum",
                                 "--snapshot", "snap1"])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["kind"] == "drawdown" and payload["kill_switch"] == "tripped"
    assert "ZZZ" in payload["closed_symbols"]      # held-but-dropped symbol was flattened
    assert "ZZZ" in broker.closed_symbols
    # belief cleared after the successful flatten
    show = json.loads(runner.invoke(app, ["paper", "show", "cross_sectional_momentum"]).stdout)
    assert show["positions"] == {}


# ---------------------------------------------------------------------------
# Task 5 (issue 163): resume (Stage.LIVE) — ingest + reconcile against broker truth
# ---------------------------------------------------------------------------

class _ReadOnlyLiveBroker:
    """Fake read-only live broker: scripted activities + broker net positions for resume tests."""
    def __init__(self, activities, positions):
        self._activities = activities
        self._positions = positions
    def account_activities(self, after=None):
        return self._activities
    def get_positions(self):
        import pandas as pd
        return pd.Series(self._positions, dtype="float64")


def _seed_live_killed(db_path, name, fills):
    """Seed a tripped live strategy with believed fills (symbol -> qty)."""
    from contextlib import closing

    from algua.registry.db import connect, migrate
    from algua.risk import kill_switch
    with closing(connect(db_path)) as conn:
        migrate(conn)
        conn.execute("UPDATE strategies SET stage='live' WHERE name=?", (name,))
        for i, (sym, qty) in enumerate(fills.items()):
            conn.execute(
                "INSERT INTO live_fills(activity_id, broker_order_id, strategy, symbol, qty, "
                "price, fill_ts) VALUES (?,?,?,?,?,?,?)",
                (f"seed-{i}", f"bo-{i}", name, sym, qty, 100.0, "2023-01-01T00:00:00Z"),
            )
        kill_switch.trip(conn, name, reason="flatten", actor="system")
        conn.commit()


def test_resume_live_refuses_when_broker_still_holds(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_ALPACA_LIVE_API_KEY", "lk")
    monkeypatch.setenv("ALGUA_ALPACA_LIVE_API_SECRET", "ls")
    name = "cross_sectional_momentum"
    _to_paper()
    _seed_live_killed(tmp_path / "p.db", name, {"AAA": 5.0})
    # broker still reports AAA held, no new activities -> ledger non-flat AND broker exposed
    broker = _ReadOnlyLiveBroker(activities=[], positions={"AAA": 5.0})
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_live_readonly_from_settings", lambda: broker)
    r = runner.invoke(app, ["paper", "resume", name])
    assert r.exit_code == 1
    assert json.loads(r.stdout)["ok"] is False
    assert "not flat" in r.stdout.lower()


def test_resume_live_refuses_when_creds_missing(monkeypatch, tmp_path):
    name = "cross_sectional_momentum"
    _to_paper()
    _seed_live_killed(tmp_path / "p.db", name, {"AAA": 5.0})
    # explicitly clear any ambient live creds -> cannot confirm flat -> refuse
    monkeypatch.delenv("ALGUA_ALPACA_LIVE_API_KEY", raising=False)
    monkeypatch.delenv("ALGUA_ALPACA_LIVE_API_SECRET", raising=False)
    r = runner.invoke(app, ["paper", "resume", name])
    assert r.exit_code == 1
    assert json.loads(r.stdout)["ok"] is False


# ---------------------------------------------------------------------------
# Task 6 (issue 163): resume-all ingests live activities before not_flat check
# ---------------------------------------------------------------------------

def test_resume_all_ingests_before_warning(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_ALPACA_LIVE_API_KEY", "lk")
    monkeypatch.setenv("ALGUA_ALPACA_LIVE_API_SECRET", "ls")
    from algua.risk import global_halt
    name = "cross_sectional_momentum"
    _to_paper()
    # one live strategy holding AAA; an ingest delivers the offsetting fill so it nets flat
    _seed_live_killed(tmp_path / "p.db", name, {"AAA": 5.0})
    from contextlib import closing

    from algua.config.settings import get_settings
    from algua.execution.live_ledger import backfill_broker_order_id, record_live_order
    from algua.registry.db import connect, migrate
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        # record the order so the ingested offset fill attributes back to the strategy
        record_live_order(conn, name, "AAA", "sell", None, "coid-off")
        backfill_broker_order_id(conn, "coid-off", "bo-off")
        global_halt.engage(conn, reason="halt-all", actor="agent")

    offset_fill = [{"id": "act-off", "activity_type": "FILL", "side": "sell", "qty": "5",
                    "price": "100", "symbol": "AAA", "order_id": "bo-off",
                    "transaction_time": "2023-01-02T00:00:00Z"}]
    broker = _ReadOnlyLiveBroker(activities=offset_fill, positions={})
    monkeypatch.setattr("algua.cli.paper_cmd._maybe_live_readonly", lambda: broker)

    r = runner.invoke(app, ["paper", "resume-all"])
    assert r.exit_code == 0, r.stdout
    payload = json.loads(r.stdout)
    assert payload["global_halt"] == "reset"
    # after ingest the offset fill landed -> strategy is flat -> NOT listed as not_flat
    assert "live_not_flat" not in payload


def test_resume_all_survives_malformed_activity(monkeypatch, tmp_path):
    # A malformed activity in the ingest stream must not crash resume-all (#250): it is quarantined
    # and the command still emits a clean JSON result rather than a raw traceback.
    monkeypatch.setenv("ALGUA_ALPACA_LIVE_API_KEY", "lk")
    monkeypatch.setenv("ALGUA_ALPACA_LIVE_API_SECRET", "ls")
    from contextlib import closing

    from algua.config.settings import get_settings
    from algua.registry.db import connect, migrate
    from algua.risk import global_halt
    name = "cross_sectional_momentum"
    _to_paper()
    _seed_live_killed(tmp_path / "p.db", name, {"AAA": 5.0})
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        global_halt.engage(conn, reason="halt-all", actor="agent")

    poison = [{"id": "bad-1", "activity_type": "FILL", "side": "hold", "qty": "5",
               "price": "100", "symbol": "AAA", "order_id": "bo-x",
               "transaction_time": "2023-01-02T00:00:00Z"}]
    broker = _ReadOnlyLiveBroker(activities=poison, positions={"AAA": 5.0})
    monkeypatch.setattr("algua.cli.paper_cmd._maybe_live_readonly", lambda: broker)

    r = runner.invoke(app, ["paper", "resume-all"])
    assert r.exit_code == 0, r.stdout
    assert json.loads(r.stdout)["global_halt"] == "reset"
    with closing(connect(get_settings().db_path)) as conn:
        assert conn.execute(
            "SELECT activity_id FROM live_activity_quarantine"
        ).fetchone()["activity_id"] == "bad-1"
