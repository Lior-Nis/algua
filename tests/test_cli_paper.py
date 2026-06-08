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


def _seed_snapshot(name, *, equity, peak, reconcile_ok=True, positions=None):
    from contextlib import closing

    from algua.config.settings import get_settings
    from algua.execution.order_state import record_tick_snapshot, update_peak_equity
    from algua.registry.db import connect, migrate
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        update_peak_equity(conn, name, peak)
        record_tick_snapshot(conn, name, tick_ts="2023-06-01T00:00:00+00:00",
                             decision_ts="2023-05-31T00:00:00+00:00", equity=equity,
                             peak_equity=peak, positions=positions or {}, n_submitted=0,
                             reconcile_ok=reconcile_ok)


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
    name = _seed_live_killed_with_position(monkeypatch, tmp_path)
    r = runner.invoke(app, ["paper", "resume", name])
    assert r.exit_code == 1 and "not flat" in r.stdout.lower()
    # once flat (belief cleared), resume succeeds
    _clear_belief(tmp_path, name)
    assert runner.invoke(app, ["paper", "resume", name]).exit_code == 0


def test_resume_clears_live_nav_peak(monkeypatch, tmp_path):
    from contextlib import closing

    from algua.config.settings import get_settings
    from algua.execution.order_state import get_nav_peak, update_nav_peak
    from algua.registry.db import connect, migrate
    name = _seed_live_killed_with_position(monkeypatch, tmp_path)
    _clear_belief(tmp_path, name)                       # make it flat so resume is allowed
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
