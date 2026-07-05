import json
from contextlib import closing
from datetime import UTC, date, datetime, timedelta

import pandas as pd
import pytest
from typer.testing import CliRunner

from algua.backtest._sample import SyntheticProvider
from algua.cli._common import resolve_wall_clock_window
from algua.cli.main import app
from algua.config.settings import get_settings
from algua.execution.alpaca_broker import AccountState, BrokerError
from algua.execution.order_state import latest_tick_snapshot
from algua.live.live_loop import SubmittedOrder, TickResult
from algua.registry.db import connect, migrate
from algua.registry.store import SqliteStrategyRepository
from algua.risk import kill_switch
from algua.risk.limits import RiskBreach
from tests._human_actor_helpers import install_human_actor_anchor, promote_signed

runner = CliRunner()

# --- #329 authenticated --actor human plumbing ---------------------------------------------------
_HUMAN_KEY = None
_TMP_PATH = None


def _promote(args):
    """Drop-in for `runner.invoke(app, args)` at gated promote call sites: when `args` requests
    `--actor human`, run the signed challenge dance; otherwise a plain invoke."""
    if "human" in args:
        return promote_signed(runner, app, args, _HUMAN_KEY, _TMP_PATH)
    return runner.invoke(app, args)

_SNAP = "snap1"


# --- resolve_wall_clock_window tests (#452 Layer B) ---

def test_resolve_wall_clock_window_defaults_to_rolling_window():
    """When both start and end are None, resolve to a rolling window ending today."""
    start, end = resolve_wall_clock_window(None, None)
    today = datetime.now(UTC).date()
    end_date = datetime.fromisoformat(end).date()
    start_date = datetime.fromisoformat(start).date()

    assert end_date == today
    expected_start = today - timedelta(days=400)
    assert start_date == expected_start


def test_resolve_wall_clock_window_respects_explicit_values():
    """Explicit --start/--end values should pass through unchanged."""
    explicit_start = "2023-06-01"
    explicit_end = "2024-06-01"
    start, end = resolve_wall_clock_window(explicit_start, explicit_end)

    assert start == explicit_start
    assert end == explicit_end


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "p.db"))
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))
    global _HUMAN_KEY, _TMP_PATH
    _HUMAN_KEY = install_human_actor_anchor(monkeypatch, tmp_path)
    _TMP_PATH = tmp_path


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

    def account_activities_window(self, after, until):
        return []

    def get_positions(self):
        return pd.Series(dtype="float64")

    def get_order_by_client_order_id(self, coid):
        # #312 recovery: default stub has no orders at the venue (a NULL row is preserved).
        return None


def test_trade_tick_submits_and_persists(monkeypatch):
    from contextlib import closing

    from algua.config.settings import get_settings
    from algua.contracts.types import OrderIntent, Side
    from algua.registry.db import connect, migrate

    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    _to_paper()
    _seed_allocation("cross_sectional_momentum")
    ts = datetime(2023, 6, 1, tzinfo=UTC)
    intent = OrderIntent(symbol="AAA", side=Side.BUY, target_weight=1.0, decision_ts=ts)
    fake_result = TickResult(
        decision_ts=ts, target_weights={"AAA": 1.0}, positions_before={},
        submitted=[{"symbol": "AAA", "side": "buy", "target_weight": 1.0, "order_id": "o-1",
                    "client_order_id": "c-1"}],
        peak_equity=100_000.0,
    )

    def _fake_run_tick(strategy, broker, provider, start, end, hooks=None, max_drawdown=None):
        # exercise the intent-before-submit hook (#249) then the broker-id backfill hook (#18)
        hooks.before_submit(intent, "c-1")
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
    # wall-clock orders now go into paper_venue_orders (not paper_orders)
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        row = conn.execute(
            "SELECT client_order_id, broker_order_id FROM paper_venue_orders "
            "WHERE strategy = 'cross_sectional_momentum' AND symbol = 'AAA'"
        ).fetchone()
    assert row is not None, "paper_venue_orders row missing after trade-tick"
    assert row["client_order_id"] == "c-1"
    assert row["broker_order_id"] == "o-1"


@pytest.mark.parametrize("kind", ["stale_marks", "unvaluable_marks"])
def test_trade_tick_dark_feed_breach_halts_without_flatten(monkeypatch, kind):
    # #452 HIGH#3 (paper lane): a stale / unvaluable mark (dark BAR feed, broker still alive) must
    # ENGAGE THE GLOBAL HALT and PRESERVE positions — it must NOT flatten the book at unknown
    # prices. Mirrors test_cli_live.test_run_all_dark_feed_breach_halts_without_flatten.
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    _to_paper()
    _seed_allocation("cross_sectional_momentum")

    flatten_calls: list = []
    monkeypatch.setattr("algua.cli.paper_cmd.flatten_strategy",
                        lambda *a, **k: flatten_calls.append(k.get("lane", a)))
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings", _MinimalBroker)
    monkeypatch.setattr("algua.cli.paper_cmd._select_provider", lambda demo, snapshot: object())
    monkeypatch.setattr(
        "algua.cli.paper_cmd.run_tick",
        lambda *a, **k: (_ for _ in ()).throw(RiskBreach(kind, f"{kind} dark feed")),
    )
    result = runner.invoke(app, ["paper", "trade-tick", "cross_sectional_momentum",
                                 "--snapshot", "snap1"])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["kind"] == kind
    assert payload["halted"] is True
    assert payload["global_halt"] == "set"
    assert payload["liquidation_submitted"] is False
    assert flatten_calls == []  # dark feed HALTS — flatten_strategy is never called
    from algua.risk import global_halt
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        assert global_halt.is_engaged(conn) is True  # systemic: whole account halted


@pytest.mark.parametrize("kind", ["drawdown", "gross_exposure_realized"])
def test_trade_tick_economic_breach_still_flattens(monkeypatch, kind):
    # #452 HIGH#3 (paper lane): an ECONOMIC/integrity breach keeps the UNCHANGED trip +
    # scoped-flatten path — only the dark-feed kinds divert to halt-only.
    from algua.execution.flatten import FlattenResult
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    _to_paper()
    _seed_allocation("cross_sectional_momentum")

    flatten_calls: list = []

    def _fake_flatten(*a, **k):
        flatten_calls.append(k.get("lane"))
        return FlattenResult(n_offsets=1, flatten_error=None)

    # the economic branch passes cancel=broker.cancel_open_orders (evaluated eagerly)
    monkeypatch.setattr(_MinimalBroker, "cancel_open_orders", lambda self: None, raising=False)
    monkeypatch.setattr("algua.cli.paper_cmd.flatten_strategy", _fake_flatten)
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings", _MinimalBroker)
    monkeypatch.setattr("algua.cli.paper_cmd._select_provider", lambda demo, snapshot: object())
    monkeypatch.setattr(
        "algua.cli.paper_cmd.run_tick",
        lambda *a, **k: (_ for _ in ()).throw(RiskBreach(kind, f"{kind} economic")),
    )
    result = runner.invoke(app, ["paper", "trade-tick", "cross_sectional_momentum",
                                 "--snapshot", "snap1"])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["kind"] == kind
    assert payload["liquidation_submitted"] is True
    assert flatten_calls == ["paper"]  # economic breach DID flatten (scoped)
    assert payload.get("halted") is not True  # not a halt-only marker
    assert payload.get("global_halt") != "set"  # economic breach does not globally halt
    from algua.risk import global_halt
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        assert global_halt.is_engaged(conn) is False


def test_trade_tick_venue_order_recording(monkeypatch):
    """After a trade-tick that submits one order:
    - a paper_venue_orders row exists with client_order_id (pre-submit intent, crash-safe)
    - broker_order_id is backfilled (post-submit accept)
    """
    from contextlib import closing

    from algua.config.settings import get_settings
    from algua.contracts.types import OrderIntent, Side
    from algua.registry.db import connect, migrate

    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    _to_paper()
    _seed_allocation("cross_sectional_momentum")

    ts = datetime(2023, 6, 1, tzinfo=UTC)
    dropped_sym = "ZZZ"  # NOT in cross_sectional_momentum's universe
    fake_intent = OrderIntent(symbol=dropped_sym, side=Side.BUY, target_weight=0.1, decision_ts=ts)

    def _fake_run_tick(strategy, broker, provider, start, end, hooks=None, max_drawdown=None):
        # Fire before_submit (intent recording) BEFORE broker call, then on_submitted (backfill).
        hooks.before_submit(fake_intent, "c-venue-1")
        hooks.on_submitted(SubmittedOrder(symbol=dropped_sym, side="buy", target_weight=0.1,
                                          order_id="o-venue-1", client_order_id="c-venue-1",
                                          decision_ts=ts))
        return TickResult(
            decision_ts=ts, target_weights={dropped_sym: 0.1}, positions_before={},
            submitted=[{"symbol": dropped_sym, "side": "buy", "target_weight": 0.1,
                        "order_id": "o-venue-1", "client_order_id": "c-venue-1"}],
            peak_equity=100_000.0,
        )

    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings", _MinimalBroker)
    monkeypatch.setattr("algua.cli.paper_cmd._select_provider", lambda demo, snapshot: object())
    monkeypatch.setattr("algua.cli.paper_cmd.run_tick", _fake_run_tick)

    result = runner.invoke(app, ["paper", "trade-tick", "cross_sectional_momentum",
                                 "--snapshot", "snap1"])
    assert result.exit_code == 0, result.stdout

    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        row = conn.execute(
            "SELECT client_order_id, broker_order_id FROM paper_venue_orders "
            "WHERE strategy = 'cross_sectional_momentum' AND symbol = ?",
            (dropped_sym,),
        ).fetchone()

    assert row is not None, "paper_venue_orders intent row missing"
    assert row["client_order_id"] == "c-venue-1"
    assert row["broker_order_id"] == "o-venue-1"


def test_trade_tick_noop_retracts_fresh_phantom_intent(monkeypatch):
    """#311: a sizing that resolves to noop/skipped fires before_submit (records the crash-safe
    intent) then on_noop. Because THIS tick freshly recorded the row, on_noop must retract it so no
    phantom paper_venue_orders row is left behind (would inflate the venue count + flip the display
    branch)."""
    from contextlib import closing

    from algua.config.settings import get_settings
    from algua.contracts.types import OrderIntent, Side
    from algua.registry.db import connect, migrate

    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    _to_paper()
    _seed_allocation("cross_sectional_momentum")

    ts = datetime(2023, 6, 1, tzinfo=UTC)
    intent = OrderIntent(symbol="AAA", side=Side.BUY, target_weight=1.0, decision_ts=ts)

    def _fake_run_tick(strategy, broker, provider, start, end, hooks=None, max_drawdown=None):
        # record the intent (crash-safe), then submit_sized reports noop -> on_noop retracts it
        hooks.before_submit(intent, "c-noop-1")
        hooks.on_noop(intent, "c-noop-1")
        return TickResult(decision_ts=ts, target_weights={"AAA": 1.0}, positions_before={},
                          submitted=[], peak_equity=100_000.0)

    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings", _MinimalBroker)
    monkeypatch.setattr("algua.cli.paper_cmd._select_provider", lambda demo, snapshot: object())
    monkeypatch.setattr("algua.cli.paper_cmd.run_tick", _fake_run_tick)

    result = runner.invoke(app, ["paper", "trade-tick", "cross_sectional_momentum",
                                 "--snapshot", "snap1"])
    assert result.exit_code == 0, result.stdout
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        row = conn.execute(
            "SELECT 1 FROM paper_venue_orders WHERE client_order_id = 'c-noop-1'"
        ).fetchone()
    assert row is None, "phantom intent row for a noop was not retracted"


def test_trade_tick_noop_preserves_preexisting_null_intent(monkeypatch):
    """#311 crash-safety: a pre-existing NULL-broker_order_id row is INDISTINGUISHABLE from a real
    order that POSTed then crashed before backfill. When this tick sizes the same coid to a noop,
    before_submit's INSERT is IGNORED (not fresh), so on_noop must NOT delete the pre-existing row —
    preserving the #249 durable intent."""
    from contextlib import closing

    from algua.config.settings import get_settings
    from algua.contracts.types import OrderIntent, Side
    from algua.registry.db import connect, migrate

    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    _to_paper()
    _seed_allocation("cross_sectional_momentum")

    # Simulate a prior run's real order: an intent row already exists for this coid, NULL broker id.
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        sid = conn.execute(
            "SELECT id FROM strategies WHERE name = 'cross_sectional_momentum'"
        ).fetchone()["id"]
        conn.execute(
            "INSERT INTO paper_venue_orders(strategy, symbol, side, intended_notional, "
            "client_order_id, strategy_id, status, submitted_ts) VALUES (?,?,?,?,?,?,?,?)",
            ("cross_sectional_momentum", "AAA", "buy", None, "c-crash-1", sid, "submitted",
             "2023-05-31T00:00:00Z"),
        )
        conn.commit()

    ts = datetime(2023, 6, 1, tzinfo=UTC)
    intent = OrderIntent(symbol="AAA", side=Side.BUY, target_weight=1.0, decision_ts=ts)

    def _fake_run_tick(strategy, broker, provider, start, end, hooks=None, max_drawdown=None):
        hooks.before_submit(intent, "c-crash-1")   # INSERT OR IGNORE -> ignored -> NOT fresh
        hooks.on_noop(intent, "c-crash-1")          # must NOT delete the pre-existing row
        return TickResult(decision_ts=ts, target_weights={"AAA": 1.0}, positions_before={},
                          submitted=[], peak_equity=100_000.0)

    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings", _MinimalBroker)
    monkeypatch.setattr("algua.cli.paper_cmd._select_provider", lambda demo, snapshot: object())
    monkeypatch.setattr("algua.cli.paper_cmd.run_tick", _fake_run_tick)

    result = runner.invoke(app, ["paper", "trade-tick", "cross_sectional_momentum",
                                 "--snapshot", "snap1"])
    assert result.exit_code == 0, result.stdout
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        row = conn.execute(
            "SELECT 1 FROM paper_venue_orders WHERE client_order_id = 'c-crash-1'"
        ).fetchone()
    assert row is not None, "a pre-existing (possibly real, crash-orphaned) intent was deleted"


class _StrandedRecoveryBroker(_MinimalBroker):
    """#312: the venue HAS the crash-stranded order; get_positions matches the recovered fill so the
    account reconciles clean and the tick proceeds."""
    def get_positions(self):
        return pd.Series({"AAA": 10.0}, dtype="float64")

    def get_order_by_client_order_id(self, coid):
        if coid == "c-crash-1":
            return {"id": "broker-9", "client_order_id": "c-crash-1", "symbol": "AAA",
                    "status": "filled"}
        return None


def test_trade_tick_recovers_stranded_fill(monkeypatch):
    """#312 end-to-end: a prior run crashed after accept but before backfill -> a NULL broker id
    row plus a fill already ingested under the real broker id with strategy NULL. The trade-tick
    recovery pass (after ingest, before reconcile) must backfill the broker id and attribute the
    fill, so it is no longer an unexplained residual."""
    from algua.registry.db import connect, migrate

    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    _to_paper()
    _seed_allocation("cross_sectional_momentum")

    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        sid = conn.execute(
            "SELECT id FROM strategies WHERE name = 'cross_sectional_momentum'").fetchone()["id"]
        conn.execute(
            "INSERT INTO paper_venue_orders(strategy, symbol, side, intended_notional, "
            "client_order_id, strategy_id, status, submitted_ts) VALUES (?,?,?,?,?,?,?,?)",
            ("cross_sectional_momentum", "AAA", "buy", None, "c-crash-1", sid, "submitted",
             "2023-05-31T00:00:00Z"))  # NULL broker_order_id
        conn.execute(
            "INSERT INTO paper_venue_fills(activity_id, broker_order_id, strategy, symbol, qty, "
            "price, fill_ts) VALUES (?,?,?,?,?,?,?)",
            ("act-1", "broker-9", None, "AAA", 10.0, 100.0, "2023-05-31T00:00:00Z"))  # orphan fill
        conn.commit()

    ts = datetime(2023, 6, 1, tzinfo=UTC)

    def _fake_run_tick(strategy, broker, provider, start, end, hooks=None, max_drawdown=None):
        return TickResult(decision_ts=ts, target_weights={}, positions_before={"AAA": 10.0},
                          submitted=[], peak_equity=100_000.0)

    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings",
                        _StrandedRecoveryBroker)
    monkeypatch.setattr("algua.cli.paper_cmd._select_provider", lambda demo, snapshot: object())
    monkeypatch.setattr("algua.cli.paper_cmd.run_tick", _fake_run_tick)

    result = runner.invoke(app, ["paper", "trade-tick", "cross_sectional_momentum",
                                 "--snapshot", "snap1"])
    assert result.exit_code == 0, result.stdout
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        order = conn.execute(
            "SELECT broker_order_id FROM paper_venue_orders WHERE client_order_id='c-crash-1'"
        ).fetchone()
        fill = conn.execute(
            "SELECT strategy FROM paper_venue_fills WHERE broker_order_id='broker-9'").fetchone()
    assert order["broker_order_id"] == "broker-9", "stranded order id was not backfilled"
    assert fill["strategy"] == "cross_sectional_momentum", "stranded fill was not attributed"


class _FlattenBroker:
    def __init__(self, fail=False):
        self.fail = fail
        self.cancelled = False
        self.offset_calls: list = []   # (sym, qty, coid) tuples

    def cancel_open_orders(self):
        self.cancelled = True

    def clock(self):
        return "2023-06-01T14:00:00+00:00"

    def account_activities_window(self, after, until):
        return []

    def submit_offset(self, sym, qty, coid):
        if self.fail:
            raise BrokerError("alpaca failed to submit offset: [...]")
        self.offset_calls.append((sym, qty, coid))
        return f"o-offset-{sym}"


def test_paper_strategy_scoped_offset_flatten(monkeypatch, tmp_path):
    """Core sibling-isolation contract (#249 Task 10):
    - flatten on strategy A believed-holding {AAA: 5} submits exactly submit_offset("AAA", 5, ...)
    - sibling SIB (held by sibling_strat in paper_venue_fills) is NEVER offset
    - payload reports liquidation_submitted: True
    - a believed qty <= 1e-6 is skipped (no submit_offset call)
    """
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    _to_paper()
    db = tmp_path / "p.db"
    # strategy A holds AAA (qty=5) and a near-zero residual sub-tol position
    _seed_paper_venue_fill(db, "cross_sectional_momentum", "AAA", qty=5.0)
    _seed_paper_venue_fill(db, "cross_sectional_momentum", "NEARZERO", qty=5e-7)
    # sibling holds SIB — must never be touched by the strategy A flatten
    _seed_paper_venue_fill(db, "sibling_strat", "SIB", qty=10.0)

    broker = _FlattenBroker()
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings", lambda: broker)
    result = runner.invoke(app, ["paper", "flatten", "cross_sectional_momentum"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["liquidation_submitted"] is True
    assert payload["kill_switch"] == "tripped"

    offset_syms = [sym for sym, _, _ in broker.offset_calls]
    assert "AAA" in offset_syms           # believed position IS offset
    assert "SIB" not in offset_syms       # sibling's symbol is NEVER offset
    assert "NEARZERO" not in offset_syms  # sub-tolerance qty is skipped


def test_paper_flatten_closes_and_trips(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    _to_paper()
    # seed a paper-venue fill so paper_believed_positions returns something to offset
    _seed_paper_venue_fill(tmp_path / "p.db", "cross_sectional_momentum", "AAA")
    broker = _FlattenBroker()
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings", lambda: broker)
    result = runner.invoke(app, ["paper", "flatten", "cross_sectional_momentum"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["liquidation_submitted"] is True and payload["kill_switch"] == "tripped"
    # scoped to the strategy's believed positions via paper_venue_fills, not account-wide close
    assert broker.cancelled is True
    assert any(sym == "AAA" for sym, _, _ in broker.offset_calls)
    show = json.loads(runner.invoke(app, ["paper", "show", "cross_sectional_momentum"]).stdout)
    assert show["kill_switch"]["tripped"] is True


def test_paper_flatten_allowed_at_forward_tested_stage(monkeypatch, tmp_path):
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
    # seed a paper-venue fill so paper_believed_positions returns something to offset
    _seed_paper_venue_fill(tmp_path / "p.db", name, "AAA")
    broker = _FlattenBroker()
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings", lambda: broker)
    result = runner.invoke(app, ["paper", "flatten", name])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["liquidation_submitted"] is True and payload["kill_switch"] == "tripped"
    assert broker.cancelled is True
    assert any(sym == "AAA" for sym, _, _ in broker.offset_calls)
    show = json.loads(runner.invoke(app, ["paper", "show", name]).stdout)
    assert show["kill_switch"]["tripped"] is True


def test_paper_flatten_rejects_non_paper_stage(monkeypatch):
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    runner.invoke(app, ["registry", "add", "cross_sectional_momentum"])  # idea
    result = runner.invoke(app, ["paper", "flatten", "cross_sectional_momentum"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False


def test_paper_flatten_close_failure_stays_tripped(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    _to_paper()
    # seed a fill so the offset loop runs and submit_offset(fail=True) raises BrokerError
    _seed_paper_venue_fill(tmp_path / "p.db", "cross_sectional_momentum", "AAA")
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
    _seed_allocation("cross_sectional_momentum")
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
        # Fresh tick_ts (now) so the liveness/staleness rollup (#399/#400) reads this seeded tick
        # as current — the health verdict under test is then driven by reconcile_ok/kill-switch,
        # not by accidental staleness of a hardcoded past date.
        now = datetime.now(UTC).isoformat()
        record_tick_snapshot(conn, name, tick_ts=now,
                             decision_ts=now, equity=equity,
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
    assert payload["staleness_sessions"] == 0  # #399: fresh tick -> liveness proven
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


def _seed_allocation(name: str, capital: float = 10_000.0) -> None:
    """Insert a strategy_allocations row directly (no paper-allocate CLI dependency)."""
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        rec = SqliteStrategyRepository(conn).get(name)
        conn.execute(
            "INSERT INTO strategy_allocations(strategy_id, capital, effective_ts, actor) "
            "VALUES (?,?,?,?)",
            (rec.id, capital, datetime.now(UTC).isoformat(), "agent"),
        )
        conn.commit()


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


def _seed_paper_venue_fill(db_path, strategy, symbol, qty=5.0):
    """Seed a paper_venue_fills row so paper_believed_positions returns the given qty for tests.
    This is the paper-ledger analogue of seeding live_fills for the live lane."""
    from contextlib import closing

    from algua.registry.db import connect, migrate
    with closing(connect(db_path)) as conn:
        migrate(conn)
        conn.execute(
            "INSERT INTO paper_venue_fills"
            "(activity_id, broker_order_id, strategy, symbol, qty, price, fill_ts)"
            " VALUES (?,?,?,?,?,?,?)",
            (f"fill-{strategy}-{symbol}", f"boid-{strategy}-{symbol}",
             strategy, symbol, qty, 100.0, "2023-01-01T00:00:00Z"),
        )
        conn.commit()


def _seed_paper_venue_order(db_path, strategy, symbol):
    """Seed a paper_venue_orders row simulating a wall-clock trade-tick submission."""
    from contextlib import closing

    from algua.registry.db import connect, migrate
    with closing(connect(db_path)) as conn:
        migrate(conn)
        row = conn.execute(
            "SELECT id FROM strategies WHERE name = ?", (strategy,)
        ).fetchone()
        strategy_id = row["id"] if row else 1
        conn.execute(
            "INSERT OR IGNORE INTO paper_venue_orders"
            "(strategy, symbol, side, intended_notional, client_order_id, broker_order_id,"
            " strategy_id, status, submitted_ts) VALUES (?,?,?,?,?,?,?,?,?)",
            (strategy, symbol, "buy", None, f"coid-{strategy}-{symbol}",
             f"boid-{strategy}-{symbol}", strategy_id, "submitted",
             "2023-01-01T00:00:00Z"),
        )
        conn.commit()


def test_paper_flatten_closes_dropped_symbol_not_siblings(monkeypatch, tmp_path):
    """The flatten offset loop iterates paper_believed_positions (paper_venue_fills attributed
    to THIS strategy), so it offsets ZZZ (held by this strategy) but never touches SIB
    (held by sibling_strat on the same account)."""
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    _to_paper()
    db = tmp_path / "p.db"
    # cross_sectional_momentum holds ZZZ via paper_venue_fills; sibling_strat holds SIB.
    # The offset loop only iterates cross_sectional_momentum's believed positions — SIB is
    # invisible to it because it is not attributed to this strategy in paper_venue_fills.
    _seed_paper_venue_fill(db, "cross_sectional_momentum", "ZZZ")
    _seed_paper_venue_fill(db, "sibling_strat", "SIB")

    broker = _FlattenBroker()
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings", lambda: broker)
    result = runner.invoke(app, ["paper", "flatten", "cross_sectional_momentum"])
    assert result.exit_code == 0, result.stdout

    offset_syms = [sym for sym, _, _ in broker.offset_calls]
    assert "ZZZ" in offset_syms           # held-but-dropped symbol IS offset
    assert "SIB" not in offset_syms       # sibling's symbol is NOT offset

    # flatten submits an offset order → paper_venue_orders has a row → show uses venue belief.
    # The fake broker never ingests fills, so the offset fill hasn't landed:
    # paper_believed_positions still shows ZZZ:5.0 (initial seeded fill). SIB not in our books.
    show = json.loads(runner.invoke(app, ["paper", "show", "cross_sectional_momentum"]).stdout)
    assert show["positions"] == {"ZZZ": 5.0}


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
    _seed_allocation("cross_sectional_momentum")

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

        def account_activities_window(self, after, until):
            return []

        def get_positions(self):
            return pd.Series(dtype="float64")

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
    _seed_allocation("cross_sectional_momentum")

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

        def account_activities_window(self, after, until):
            return []

        def get_positions(self):
            return pd.Series(dtype="float64")

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
    _seed_allocation("cross_sectional_momentum")

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
    """On a RiskBreach, the handler submits strategy-scoped submit_offset calls for each symbol
    in paper_believed_positions (paper_venue_fills). A held-but-dropped symbol (ZZZ, not in
    universe) is offset; paper show reflects derive_positions (paper_fills) which is empty."""
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    _to_paper()
    _seed_allocation("cross_sectional_momentum")
    # seed a paper_venue_fills row so paper_believed_positions returns ZZZ:5
    _seed_paper_venue_fill(tmp_path / "p.db", "cross_sectional_momentum", "ZZZ")

    class _BreachTickBroker(_MinimalBroker):
        def __init__(self):
            self.offset_calls: list = []   # (sym, qty, coid) tuples
        def get_positions(self):
            # broker agrees with the seeded paper_venue_fills for ZZZ so reconcile passes
            return pd.Series({"ZZZ": 5.0}, dtype="float64")
        def cancel_open_orders(self):
            pass
        def submit_offset(self, sym, qty, coid):
            self.offset_calls.append((sym, qty, coid))
            return f"o-offset-{sym}"

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
    assert payload["liquidation_submitted"] is True
    # ZZZ was in paper_believed_positions → exactly one submit_offset call for it
    offset_syms = [sym for sym, _, _ in broker.offset_calls]
    assert offset_syms == ["ZZZ"]
    # breach handler submits an offset order → paper_venue_orders has a row → show uses venue
    # belief. The fake broker never ingests fills, so the offset fill hasn't landed:
    # paper_believed_positions still shows ZZZ:5.0 (the initial seeded fill).
    show = json.loads(runner.invoke(app, ["paper", "show", "cross_sectional_momentum"]).stdout)
    assert show["positions"] == {"ZZZ": 5.0}


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


# ---------------------------------------------------------------------------
# Task 12 (#249): paper show — venue vs sim branch
# ---------------------------------------------------------------------------

def test_show_venue_strategy_reports_venue_positions(tmp_path):
    """A strategy with paper_venue_orders (wall-clock traded) → paper show uses
    paper_believed_positions (venue ledger) and venue order count, NOT the sim view."""
    from contextlib import closing

    from algua.config.settings import get_settings
    from algua.registry.db import connect, migrate

    _to_paper()
    db = tmp_path / "p.db"
    # Seed a venue fill so paper_believed_positions returns AAA:5
    _seed_paper_venue_fill(db, "cross_sectional_momentum", "AAA", qty=5.0)
    # Seed a venue order row — this is what the probe SELECT sees
    _seed_paper_venue_order(db, "cross_sectional_momentum", "AAA")

    payload = json.loads(runner.invoke(app, ["paper", "show", "cross_sectional_momentum"]).stdout)
    assert payload["ok"] is True
    # Positions come from paper_believed_positions (paper_venue_fills), not sim derive_positions
    assert payload["positions"] == {"AAA": 5.0}
    # n_orders counts paper_venue_orders
    assert payload["n_orders"] == 1
    # recent_orders comes from the venue lane
    assert len(payload["recent_orders"]) == 1
    assert payload["recent_orders"][0]["symbol"] == "AAA"

    # Confirm: no sim fills were seeded, so derive_positions would return {} — venue path differs
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        from algua.execution.order_state import derive_positions
        assert derive_positions(conn, "cross_sectional_momentum") == {}


def test_show_sim_strategy_still_reports_sim_positions():
    """A strategy with ONLY sim paper_orders/paper_fills (no venue orders) → paper show uses
    derive_positions (the sim view), not the venue ledger."""
    _to_paper()
    # Run the sim broker to seed paper_orders + paper_fills
    result = runner.invoke(app, ["paper", "run", "cross_sectional_momentum", "--demo",
                                 "--start", "2022-01-01", "--end", "2023-12-31"])
    assert result.exit_code == 0, result.stdout
    sim_n_orders = json.loads(result.stdout)["orders"]

    payload = json.loads(runner.invoke(app, ["paper", "show", "cross_sectional_momentum"]).stdout)
    assert payload["ok"] is True
    # n_orders is from the sim paper_orders table (no venue orders seeded)
    assert payload["n_orders"] == sim_n_orders
    # recent_orders comes from sim paper_orders
    assert len(payload["recent_orders"]) <= 10


def test_paper_broker_net_drops_zero_positions():
    import pandas as pd

    from algua.cli.paper_cmd import _paper_broker_net

    class _B:
        def get_positions(self):
            return pd.Series({"AAA": 10.0, "BBB": 0.0, "CCC": -3.0})

    assert _paper_broker_net(_B()) == {"AAA": 10.0, "CCC": -3.0}


# ---------------------------------------------------------------------------
# Task 2 (#316a): multi-tenant trade-tick — NAV sizing, reconcile defer, breach
# ---------------------------------------------------------------------------

class _FakePaperBroker:
    """Fake paper broker for multi-tenant trade-tick tests.

    `force_breach=True` causes `cancel_open_orders()` to raise RiskBreach (called
    unconditionally by run_tick after decide() so the breach fires regardless of whether
    the strategy generates any intents — i.e. works even with short date ranges where the
    lookback signal is all-NaN).
    """

    def __init__(self, account_equity: float, positions: dict, marks: dict,
                 force_breach: bool = False) -> None:
        self.account_equity = account_equity
        self._positions = positions
        self._marks = marks
        self.force_breach = force_breach
        self.submitted: list = []
        self.account_wide_cancels: int = 0

    def account(self) -> AccountState:
        return AccountState(equity=self.account_equity, cash=self.account_equity,
                            buying_power=self.account_equity, account_id="fake-acct")

    def clock(self) -> str:
        return "2026-01-02T14:00:00Z"

    def account_activities_window(self, after: str, until: str) -> list:
        return []

    def get_positions(self) -> pd.Series:
        return pd.Series(self._positions, dtype="float64")

    def cancel_open_orders(self) -> None:
        self.account_wide_cancels += 1
        if self.force_breach:
            raise RiskBreach("drawdown", "forced breach for testing")

    def submit_sized(self, intent, snap, coid=None, reserve=None) -> str:
        order_id = f"o-{intent.symbol}"
        self.submitted.append(order_id)
        return order_id

    def submit_offset(self, sym: str, qty: float, coid: str) -> str:
        return f"o-offset-{sym}"


def _paper_strategy_with_allocation(
    monkeypatch, tmp_path, *, capital: float, account_equity: float,
    name: str = "cross_sectional_momentum",
) -> str:
    """Advance a strategy to paper stage, seed a capital allocation, and monkeypatch the
    data provider to SyntheticProvider so trade-tick doesn't need a real snapshot."""
    _to_paper(name)
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        rec = SqliteStrategyRepository(conn).get(name)
        conn.execute(
            "INSERT INTO strategy_allocations(strategy_id, capital, effective_ts, actor) "
            "VALUES (?,?,?,?)",
            (rec.id, capital, datetime.now(UTC).isoformat(), "agent"),
        )
        conn.commit()
    monkeypatch.setattr("algua.cli.paper_cmd._select_provider",
                        lambda demo, snapshot: SyntheticProvider())
    return name


def _latest_tick(name: str) -> dict | None:
    """Return the latest tick snapshot for a strategy from the test DB."""
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        return latest_tick_snapshot(conn, name)


def kill_switch_is_tripped(name: str) -> bool:
    """True iff the strategy's kill-switch is currently tripped."""
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        return kill_switch.is_tripped(conn, name)


def test_trade_tick_sizes_off_allocation_not_account(monkeypatch, tmp_path):
    # A paper strategy at PAPER stage with a $10k allocation, account funded at $1M.
    # Orders must target the $10k allocation/NAV, not the $1M account equity.
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    name = _paper_strategy_with_allocation(monkeypatch, tmp_path, capital=10_000.0,
                                           account_equity=1_000_000.0)
    broker = _FakePaperBroker(account_equity=1_000_000.0, positions={}, marks={"AAA": 100.0})
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings", lambda: broker)
    # A wall-clock trade-tick values the book off bar marks, which the #452 freshness wall requires
    # to be RECENT (<= 2 completed sessions old). Use a rolling window ending today so the synthetic
    # marks are fresh rather than months-stale.
    from datetime import timedelta
    _end = datetime.now(UTC).date()
    _start = (_end - timedelta(days=90)).isoformat()
    result = runner.invoke(app, ["paper", "trade-tick", name, "--snapshot", _SNAP,
                                 "--start", _start, "--end", _end.isoformat()])
    assert result.exit_code == 0, result.output
    # the recorded tick snapshot equity is the per-strategy NAV (~allocation), not 1_000_000
    snap = _latest_tick(name)
    assert snap["equity"] <= 10_000.0


def test_trade_tick_defers_on_unattributable_holding(monkeypatch, tmp_path):
    # Broker shows a holding no paper strategy owns -> reconcile not clean -> defer, no orders.
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    name = _paper_strategy_with_allocation(monkeypatch, tmp_path, capital=10_000.0,
                                           account_equity=100_000.0)
    broker = _FakePaperBroker(account_equity=100_000.0,
                              positions={"ZZZ": 5.0}, marks={"AAA": 100.0})
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings", lambda: broker)
    result = runner.invoke(app, ["paper", "trade-tick", name, "--snapshot", _SNAP,
                                 "--start", "2026-01-01", "--end", "2026-02-01"])
    payload = json.loads(result.stdout)  # stdout is the JSON contract; logs go to stderr (#346)
    assert payload.get("traded") is False
    assert payload.get("deferred") is True
    assert broker.submitted == []   # nothing traded on an unreconciled account
    assert _latest_tick(name) is None  # a deferred tick records NO snapshot (no gate coverage)


def test_trade_tick_halts_after_grace_expiry(monkeypatch, tmp_path):
    # After DEFAULT_GRACE_CYCLES (=3) the persistent unattributable holding escalates from
    # deferred to halt.  Invocations 1-3 defer (exit 0); invocation 4 hits
    # cycle - first_seen == 3 >= 3, engages the global halt, and exits non-zero.
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    name = _paper_strategy_with_allocation(monkeypatch, tmp_path, capital=10_000.0,
                                           account_equity=100_000.0)
    broker = _FakePaperBroker(account_equity=100_000.0,
                              positions={"ZZZ": 5.0}, marks={"AAA": 100.0})
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings", lambda: broker)
    args = ["paper", "trade-tick", name, "--snapshot", _SNAP,
            "--start", "2026-01-01", "--end", "2026-02-01"]

    # Cycles 1-3: mismatch is "pending" (within the grace window) -> defer, exit 0.
    for i in range(3):
        r = runner.invoke(app, args)
        assert r.exit_code == 0, f"cycle {i + 1} unexpectedly non-zero: {r.output}"
        p = json.loads(r.stdout)  # stdout is the JSON contract; logs go to stderr (#346)
        assert p.get("deferred") is True, f"cycle {i + 1} did not defer"

    # Cycle 4: cycle - first_seen == 3 >= DEFAULT_GRACE_CYCLES -> unexplained -> halt.
    result = runner.invoke(app, args)
    assert result.exit_code != 0, f"expected halt exit-code, got 0: {result.output}"
    payload = json.loads(result.stdout)  # stdout is the JSON contract; logs go to stderr (#346)
    assert payload.get("halted") is True

    from algua.risk import global_halt
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        assert global_halt.is_engaged(conn) is True


def test_trade_tick_breach_trips_and_scoped_flattens(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    name = _paper_strategy_with_allocation(monkeypatch, tmp_path, capital=10_000.0,
                                           account_equity=100_000.0)
    broker = _FakePaperBroker(account_equity=100_000.0, positions={}, marks={"AAA": 100.0},
                              force_breach=True)
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings", lambda: broker)
    result = runner.invoke(app, ["paper", "trade-tick", name, "--snapshot", _SNAP,
                                 "--start", "2026-01-01", "--end", "2026-02-01",
                                 "--max-drawdown", "0.01"])
    assert result.exit_code != 0
    assert kill_switch_is_tripped(name)


def test_run_paper_strategy_tick_breach_uses_scoped_cancel(monkeypatch, tmp_path):
    # A caller-supplied scoped cancel (run-all passes a per-strategy one) must be used on a breach
    # flatten — NOT the broker's account-wide cancel, which would nuke a sibling's resting orders.
    from algua.cli import paper_cmd
    from algua.cli._common import registry_conn
    from algua.registry.gating import load_gated_strategy

    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    name = _paper_strategy_with_allocation(monkeypatch, tmp_path, capital=10_000.0,
                                           account_equity=100_000.0)
    broker = _FakePaperBroker(account_equity=100_000.0, positions={}, marks={"AAA": 100.0},
                              force_breach=True)
    # Force the tick to breach deterministically (no dependence on the synthetic signal).
    monkeypatch.setattr(paper_cmd, "run_tick",
                        lambda *a, **k: (_ for _ in ()).throw(RiskBreach("drawdown", "forced")))

    scoped_cancels = {"n": 0}

    def _scoped_cancel() -> None:
        scoped_cancels["n"] += 1

    with registry_conn() as conn:
        rec = SqliteStrategyRepository(conn).get(name)
        strategy, _rec = load_gated_strategy(conn, name, "trade-tick")
        out = paper_cmd._run_paper_strategy_tick(
            conn, name, strategy, rec, broker, SyntheticProvider(), 0.01,
            "tick-ts", "broker", broker.account(), cancel=_scoped_cancel,
            start="2026-01-01", end="2026-02-01")

    assert out["ok"] is False
    assert scoped_cancels["n"] >= 1                 # the scoped cancel WAS used
    assert broker.account_wide_cancels == 0          # the account-wide cancel was NOT
