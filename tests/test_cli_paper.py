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
    assert runner.invoke(app, ["registry", "transition", name, "--to", "shortlisted",
                               "--actor", "agent", "--reason", "ok"]).exit_code == 0
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

    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings", lambda: object())
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
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings", lambda: object())
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
