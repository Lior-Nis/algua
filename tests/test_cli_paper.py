import json
from datetime import UTC, datetime

import pytest
from typer.testing import CliRunner

from algua.cli.main import app
from algua.execution.alpaca_broker import AccountState, BrokerError
from algua.live.live_loop import TickResult
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
    assert payload["strategy"] == "cross_sectional_momentum"
    assert payload["reconcile_ok"] is True
    assert payload["orders"] >= 1
    show = json.loads(runner.invoke(app, ["paper", "show", "cross_sectional_momentum"]).stdout)
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


def test_trade_live_rejects_non_paper_stage(monkeypatch):
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    runner.invoke(app, ["registry", "add", "cross_sectional_momentum"])  # stage = idea
    result = runner.invoke(app, ["paper", "trade-live", "cross_sectional_momentum",
                                 "--snapshot", "snap1"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False


def test_trade_live_refused_when_killed(monkeypatch):
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    _to_paper()
    runner.invoke(app, ["paper", "kill", "cross_sectional_momentum", "--reason", "x"])
    result = runner.invoke(app, ["paper", "trade-live", "cross_sectional_momentum",
                                 "--snapshot", "snap1"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False


def test_trade_live_submits_and_persists(monkeypatch):
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    _to_paper()
    ts = datetime(2023, 6, 1, tzinfo=UTC)
    fake_result = TickResult(
        decision_ts=ts, target_weights={"AAA": 1.0}, positions_before={},
        submitted=[{"symbol": "AAA", "side": "buy", "target_weight": 1.0, "order_id": "o-1"}],
    )
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings",
                        lambda: _AccountBroker(equity=100.0))
    monkeypatch.setattr("algua.cli.paper_cmd._select_provider", lambda demo, snapshot: object())
    monkeypatch.setattr("algua.cli.paper_cmd.run_tick",
                        lambda strategy, broker, provider, start, end: fake_result)
    result = runner.invoke(app, ["paper", "trade-live", "cross_sectional_momentum",
                                 "--snapshot", "snap1"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
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


class _AccountBroker(_FlattenBroker):
    """_FlattenBroker (cancel_open_orders + close_positions) plus an account() stub."""
    def __init__(self, equity, fail=False):
        super().__init__(fail=fail)
        self._equity = equity

    def account(self):
        return AccountState(equity=self._equity, cash=self._equity, buying_power=self._equity)


def _seed_peak(name, value):
    from contextlib import closing

    from algua.config.settings import get_settings
    from algua.registry import store
    from algua.registry.db import connect, migrate
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        store.set_equity_peak(conn, name, value)


def test_trade_live_drawdown_trips_and_flattens(monkeypatch):
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    _to_paper()
    _seed_peak("cross_sectional_momentum", 100.0)
    broker = _AccountBroker(equity=80.0)  # 20% drawdown
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings", lambda: broker)
    monkeypatch.setattr("algua.cli.paper_cmd.run_tick",
                        lambda *a, **k: pytest.fail("run_tick should not run on breach"))
    result = runner.invoke(app, ["paper", "trade-live", "cross_sectional_momentum",
                                 "--snapshot", "x", "--max-drawdown", "0.10"])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["kind"] == "drawdown" and payload["kill_switch"] == "tripped"
    assert payload["liquidation_submitted"] is True
    assert broker.closed_symbols  # flattened to the strategy universe
    show = json.loads(runner.invoke(app, ["paper", "show", "cross_sectional_momentum"]).stdout)
    assert show["kill_switch"]["tripped"] is True


def test_trade_live_new_high_persists_peak(monkeypatch):
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    _to_paper()
    _seed_peak("cross_sectional_momentum", 100.0)
    broker = _AccountBroker(equity=130.0)  # new high
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings", lambda: broker)
    monkeypatch.setattr("algua.cli.paper_cmd.run_tick",
                        lambda *a, **k: TickResult(None, {}, {}, []))
    result = runner.invoke(app, ["paper", "trade-live", "cross_sectional_momentum",
                                 "--snapshot", "x", "--max-drawdown", "0.10"])
    assert result.exit_code == 0, result.stdout
    from contextlib import closing

    from algua.config.settings import get_settings
    from algua.registry import store
    from algua.registry.db import connect, migrate
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        assert store.get_equity_peak(conn, "cross_sectional_momentum") == 130.0


def test_trade_live_drawdown_disabled_default(monkeypatch):
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    _to_paper()
    _seed_peak("cross_sectional_momentum", 100.0)
    broker = _AccountBroker(equity=10.0)  # huge drawdown, but breaker disabled (default 1.0)
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings", lambda: broker)
    monkeypatch.setattr("algua.cli.paper_cmd.run_tick",
                        lambda *a, **k: TickResult(None, {}, {}, []))
    result = runner.invoke(app, ["paper", "trade-live", "cross_sectional_momentum",
                                 "--snapshot", "x"])  # no --max-drawdown -> 1.0
    assert result.exit_code == 0, result.stdout
