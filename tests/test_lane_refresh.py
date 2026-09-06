"""Cycle plan + lane bars refresh for run-all --refresh (#556)."""
from __future__ import annotations

import dataclasses
from contextlib import closing
from datetime import date, timedelta

import pandas as pd
import pytest
from typer.testing import CliRunner

import algua.cli.lane_refresh as lane_refresh
from algua.cli.lane_refresh import (
    CyclePlan,
    build_cycle_plan,
    cycle_start,
    lane_symbols,
    refresh_lane_snapshot,
)
from algua.cli.main import app
from algua.config.settings import get_settings
from algua.data.contracts import ProviderBars
from algua.execution.live_ledger import LedgerKind
from algua.registry.db import connect, migrate
from tests._gate_row_helpers import seed_passing_gate

runner = CliRunner()
_S = "cross_sectional_momentum"
_CONFIG_UNIVERSE = {"AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"}


def _default_window_start(end: str) -> str:
    """Mirrors ``lane_refresh``'s end-anchored 400-day default (NOT
    ``resolve_wall_clock_window``, which defaults an unset start off real wall-clock "now"
    regardless of the ``end`` it is given — wrong for a cycle window, which must stay
    deterministic given (end, min_rows) alone; see the deviation note in the task report)."""
    return (date.fromisoformat(end) - timedelta(days=400)).isoformat()


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "p.db"))
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))


def _register(name=_S):
    assert runner.invoke(app, ["backtest", "run", name, "--demo", "--register",
                               "--start", "2022-01-01", "--end", "2023-12-31"]).exit_code == 0


def _conn():
    conn = connect(get_settings().db_path)
    migrate(conn)
    return conn


def test_plan_resolves_gate_bound_universe_held_and_history_floor(tmp_path):
    _register()
    seed_passing_gate(_S)
    with closing(_conn()) as conn:
        conn.execute("INSERT INTO paper_venue_fills(activity_id, strategy, symbol, qty, price, "
                     "fill_ts) VALUES (?,?,?,?,?,?)",
                     ("act-1", _S, "TSLA", 2.0, 100.0, "2023-01-01T00:00:00Z"))
        conn.commit()
        plan = build_cycle_plan(conn, names=[_S], kind=LedgerKind.PAPER,
                                data_dir=get_settings().data_dir)
    assert plan.names == [_S] and plan.skipped == []
    assert set(plan.universes[_S]) == _CONFIG_UNIVERSE
    assert plan.held[_S] == ["TSLA"]
    # feature_lookback=60 for this strategy -> floor = max(60, warmup) + 1 on every universe name;
    # a held-only symbol carries no floor.
    assert all(plan.min_rows[s] >= 61 for s in _CONFIG_UNIVERSE)
    assert "TSLA" not in plan.min_rows


def test_plan_isolates_a_strategy_with_no_gate_row(tmp_path):
    _register()  # registered, NO passing gate row -> resolve_operational_universe raises
    with closing(_conn()) as conn:
        plan = build_cycle_plan(conn, names=[_S], kind=LedgerKind.PAPER,
                                data_dir=get_settings().data_dir)
    assert plan.names == []
    assert plan.skipped[0]["strategy"] == _S and plan.skipped[0]["traded"] is False


def test_plan_skips_undeclared_feature_lookback_never_as_zero(monkeypatch, tmp_path):
    _register()
    seed_passing_gate(_S)
    real_load = lane_refresh.load_tradable_strategy

    def _undeclared(name):
        loaded = real_load(name)
        new_config = loaded.config.model_copy(update={"feature_lookback": None})
        return dataclasses.replace(loaded, config=new_config)
    monkeypatch.setattr(lane_refresh, "load_tradable_strategy", _undeclared)
    with closing(_conn()) as conn:
        plan = build_cycle_plan(conn, names=[_S], kind=LedgerKind.PAPER,
                                data_dir=get_settings().data_dir)
    assert plan.names == [] and "undeclared_feature_lookback" in plan.skipped[0]["skipped"]


def test_cycle_start_widens_for_a_deep_lookback():
    from algua.calendar.market_calendar import MarketCalendar
    default_start = _default_window_start("2026-06-15")
    assert cycle_start(end="2026-06-15", min_rows={"AAPL": 61}) == default_start
    deep = cycle_start(end="2026-06-15", min_rows={"AAPL": 337})
    assert deep < default_start                       # earlier than 400 days back
    cal = MarketCalendar("XNYS")
    expected = cal.previous_session(date(2026, 6, 15))
    n = len(cal.sessions_in_range(date.fromisoformat(deep), expected))
    assert n >= 337 + 5                                # exact session count, not a ratio


def test_plan_floor_covers_the_capacity_adv_window(monkeypatch, tmp_path):
    _register()
    seed_passing_gate(_S)
    from algua.contracts.types import CapacityLimit  # the ExecutionContract.capacity type
    real_load = lane_refresh.load_tradable_strategy

    def _with_capacity(name):
        loaded = real_load(name)
        cap = CapacityLimit(reference_aum=1_000_000.0, max_participation_rate=0.1,
                            adv_window_bars=200)
        new_exec = dataclasses.replace(loaded.execution, capacity=cap)
        new_config = loaded.config.model_copy(update={"execution": new_exec})
        return dataclasses.replace(loaded, config=new_config)
    monkeypatch.setattr(lane_refresh, "load_tradable_strategy", _with_capacity)
    with closing(_conn()) as conn:
        plan = build_cycle_plan(conn, names=[_S], kind=LedgerKind.PAPER,
                                data_dir=get_settings().data_dir)
    assert all(plan.min_rows[s] >= 201 for s in _CONFIG_UNIVERSE)


def test_plan_rejects_a_non_tradable_strategy_like_the_tick_does(monkeypatch, tmp_path):
    _register()
    seed_passing_gate(_S)

    def _refuse(name):
        raise ValueError(f"{name}: needs_fundamentals has no paper/live lane")
    monkeypatch.setattr(lane_refresh, "load_tradable_strategy", _refuse)
    with closing(_conn()) as conn:
        plan = build_cycle_plan(conn, names=[_S], kind=LedgerKind.PAPER,
                                data_dir=get_settings().data_dir)
    assert plan.names == [] and "needs_fundamentals" in plan.skipped[0]["skipped"]


def test_lane_symbols_is_the_union_with_broker_net():
    plan = CyclePlan(universes={"a": ["MSFT", "AAPL"], "b": ["NVDA"]},
                     held={"a": ["TSLA"], "b": []}, min_rows={}, skipped=[])
    assert lane_symbols(plan, {"ORPH": 3.0, "ZERO": 0.0}) == ["AAPL", "MSFT", "NVDA", "ORPH",
                                                              "TSLA"]


def _provider_with(dates: list[str]):
    class _Provider:
        name = "fake"

        def __init__(self):
            self.request = None

        def get_bars(self, request):
            self.request = request
            n = len(dates)
            return ProviderBars(frame=pd.DataFrame({
                "ts": [f"{d}T00:00:00+00:00" for d in dates], "symbol": ["AAPL"] * n,
                "open": [1.0] * n, "high": [1.0] * n, "low": [1.0] * n, "close": [1.0] * n,
                "adj_close": [1.0] * n, "volume": [1.0] * n}), source_metadata={})
    return _Provider()


def test_refresh_lane_snapshot_requires_previous_session_bar(monkeypatch, tmp_path):
    provider = _provider_with(["2023-06-13", "2023-06-14"])
    monkeypatch.setattr(lane_refresh, "get_provider", lambda name, settings: provider)
    info = refresh_lane_snapshot(["AAPL"], end="2023-06-15", min_rows={"AAPL": 2},
                                 kind=LedgerKind.PAPER)
    assert info["require_bar_on"] == "2023-06-14"   # previous XNYS session before 06-15
    assert info["refreshed"] is True and info["symbols"] == 1
    assert provider.request.end == "2023-06-15" and info["end"] == "2023-06-15"
    assert provider.request.start == info["start"] == _default_window_start("2023-06-15")


def test_refresh_lane_snapshot_maps_weekend_to_friday(monkeypatch, tmp_path):
    provider = _provider_with(["2023-06-16"])
    monkeypatch.setattr(lane_refresh, "get_provider", lambda name, settings: provider)
    info = refresh_lane_snapshot(["AAPL"], end="2023-06-17", min_rows={}, kind=LedgerKind.PAPER)
    assert info["require_bar_on"] == "2023-06-16"


def test_live_refresh_provider_must_be_set_explicitly(monkeypatch, tmp_path):
    monkeypatch.delenv("ALGUA_BARS_REFRESH_PROVIDER_LIVE", raising=False)
    with pytest.raises(ValueError, match="live_refresh_provider_unset"):
        refresh_lane_snapshot(["AAPL"], end="2023-06-15", min_rows={}, kind=LedgerKind.LIVE)


def test_live_refresh_uses_the_live_provider_setting(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_BARS_REFRESH_PROVIDER_LIVE", "fake-live")
    seen = {}
    provider = _provider_with(["2023-06-14"])

    def _get(name, settings):
        seen["name"] = name
        return provider
    monkeypatch.setattr(lane_refresh, "get_provider", _get)
    refresh_lane_snapshot(["AAPL"], end="2023-06-15", min_rows={}, kind=LedgerKind.LIVE)
    assert seen["name"] == "fake-live"
