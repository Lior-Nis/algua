"""Normative Story 1.3a tests at the public Phase A / Phase B seams."""

from __future__ import annotations

import json
from dataclasses import fields, replace
from datetime import UTC, datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from algua.contracts.types import ExecutionContract
from algua.execution.alpaca_broker import TickSnapshot
from algua.live.live_loop import PlannerContext, TickHooks, run_tick
from algua.live.planner import phase_a, phase_b
from algua.live.planner_binding import phase_a_binding
from algua.live.planner_contract import (
    BOUNDARY_VERSION,
    CapturedStrategyState,
    Decision,
    EarlyNoDecision,
    EarlyPlannerInput,
    LatePlannerInput,
    PhaseBindingFailure,
    PlannerInputFailure,
    PlannerRiskFailure,
    PlannerState,
    SnapshotRequired,
    VenueBeliefDisabled,
    VenueBeliefEnabled,
    VenueBeliefPending,
    VenueBeliefRequired,
)
from algua.strategies.base import LoadedStrategy, StrategyConfig, config_hash

NOW = datetime(2023, 1, 5, tzinfo=UTC)


def _identity(scores, view, params):
    return scores


def _strategy(
    *, warmup: int = 0, decisions: list[str] | None = None, params: dict | None = None
) -> LoadedStrategy:
    def signal(view, params):
        if decisions is not None:
            decisions.append("decision")
        return pd.Series({"AAA": 0.5})

    return LoadedStrategy(
        config=StrategyConfig(
            name="test",
            universe=["AAA"],
            execution=ExecutionContract(rebalance_frequency="1d", warmup_bars=warmup),
            params={} if params is None else params,
            construction="top_k_equal_weight",
            construction_params={"top_k": 1},
        ),
        signal_fn=signal,
        construct_fn=_identity,
    )


def _bars(*, reverse: bool = False) -> pd.DataFrame:
    rows = [
        {
            "timestamp": datetime(2023, 1, day, tzinfo=UTC),
            "symbol": symbol,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "adj_close": price,
            "volume": 100.0,
        }
        for day in (2, 3, 4)
        for symbol, price in (("AAA", 10.0), ("OLD", 5.0))
    ]
    if reverse:
        rows.reverse()
    return pd.DataFrame(rows).set_index("timestamp")


def _early(
    strategy: LoadedStrategy,
    *,
    bars: pd.DataFrame | None = None,
    positions: dict[str, float] | None = None,
) -> EarlyPlannerInput:
    resolved = json.dumps(
        strategy.config.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return EarlyPlannerInput(
        boundary_version=BOUNDARY_VERSION,
        request_id="0123456789abcdef0123456789abcdef",
        strategy_name="test",
        deployment_id=None,
        artifact_id=None,
        manifest_digest=None,
        config_hash=config_hash(strategy),
        resolved_config_json=resolved,
        now=NOW,
        timeframe="1d",
        calendar_code="XNYS",
        raw_bars=_bars() if bars is None else bars,
        early_positions={} if positions is None else positions,
        gate_universe=("AAA",),
        max_drawdown=0.1,
    )


def _captured(*, belief=None, sizing: float = 100.0) -> CapturedStrategyState:
    return CapturedStrategyState(
        request_id="0123456789abcdef0123456789abcdef",
        sizing_equity=sizing,
        drawdown_equity=100.0,
        quantities={"OLD": 2.0},
        market_values={"OLD": 20.0},
        persisted_peak_equity=100.0,
        venue_belief=VenueBeliefDisabled() if belief is None else VenueBeliefEnabled(belief),
    )


def test_early_input_has_only_the_normative_data_fields():
    assert [field.name for field in fields(EarlyPlannerInput)] == [
        "boundary_version",
        "request_id",
        "strategy_name",
        "deployment_id",
        "artifact_id",
        "manifest_digest",
        "config_hash",
        "resolved_config_json",
        "now",
        "timeframe",
        "calendar_code",
        "raw_bars",
        "early_positions",
        "gate_universe",
        "max_drawdown",
    ]
    forbidden = {
        "registry",
        "provider",
        "broker",
        "hook",
        "connection",
        "credential",
        "callback",
        "persistence",
    }
    assert not forbidden.intersection(field.name for field in fields(EarlyPlannerInput))
    assert not forbidden.intersection(field.name for field in fields(CapturedStrategyState))


def test_phase_a_returns_early_no_decision_without_late_state_for_flat_warmup():
    strategy = _strategy(warmup=3)
    result = phase_a(strategy, _early(strategy, positions={"ZERO": 0.0}))
    assert isinstance(result, EarlyNoDecision)
    assert result.reason == "warming"
    assert result.state.positions_before == (("ZERO", 0.0),)
    assert result.state.decision_ts == datetime(2023, 1, 4, tzinfo=UTC)


def test_phase_a_binding_is_order_sensitive_for_bars_but_not_mappings():
    strategy = _strategy()
    baseline = phase_a(strategy, _early(strategy, positions={"A": 0.0, "B": -0.0}))
    reordered_map = phase_a(strategy, _early(strategy, positions={"B": 0.0, "A": -0.0}))
    reordered_bars = phase_a(
        strategy, _early(strategy, bars=_bars(reverse=True), positions={"A": 0.0, "B": -0.0})
    )
    assert isinstance(baseline, SnapshotRequired)
    assert isinstance(reordered_map, SnapshotRequired)
    assert isinstance(reordered_bars, SnapshotRequired)
    assert baseline.phase_a_binding == reordered_map.phase_a_binding
    assert baseline.phase_a_binding != reordered_bars.phase_a_binding
    assert len(baseline.phase_a_binding) == 64


def test_phase_a_binding_matches_the_locked_canonical_example():
    strategy = _strategy()
    result = phase_a(strategy, _early(strategy, positions={"OLD": 2.0}))
    assert isinstance(result, SnapshotRequired)
    assert result.phase_a_binding == (
        "d405bd0c34d784c6db0f381fde5745f670dc04dc4516f60926a120885c6491ab"
    )


def test_phase_b_rejects_binding_mismatch_before_late_risk_or_decision():
    decisions: list[str] = []
    strategy = _strategy(decisions=decisions)
    early = _early(strategy, positions={"OLD": 2.0})
    first = phase_a(strategy, early)
    assert isinstance(first, SnapshotRequired)
    result = phase_b(
        strategy,
        LatePlannerInput(early, "0" * 64, _captured(sizing=0.0)),
    )
    assert result == PhaseBindingFailure(
        "phase_a_binding_mismatch", "Phase A binding does not match"
    )
    assert decisions == []


def test_enabled_empty_venue_belief_is_distinct_from_disabled():
    strategy = _strategy()
    early = _early(strategy, positions={"OLD": 2.0})
    first = phase_a(strategy, early)
    assert isinstance(first, SnapshotRequired)

    disabled = phase_b(strategy, LatePlannerInput(early, first.phase_a_binding, _captured()))
    enabled_empty = phase_b(
        strategy, LatePlannerInput(early, first.phase_a_binding, _captured(belief={}))
    )

    assert isinstance(disabled, Decision)
    assert [intent.symbol for intent in disabled.ordered_intents] == ["AAA", "OLD"]
    assert isinstance(enabled_empty, PlannerRiskFailure)
    assert enabled_empty.kind == "reconcile"


def test_phase_b_requests_venue_belief_only_after_equity_and_drawdown_pass():
    strategy = _strategy()
    early = _early(strategy, positions={"OLD": 2.0})
    first = phase_a(strategy, early)
    assert isinstance(first, SnapshotRequired)

    pending = replace(_captured(), venue_belief=VenueBeliefPending())
    assert phase_b(strategy, LatePlannerInput(early, first.phase_a_binding, pending)) == (
        VenueBeliefRequired()
    )
    failed = replace(pending, sizing_equity=0.0)
    result = phase_b(strategy, LatePlannerInput(early, first.phase_a_binding, failed))
    assert isinstance(result, PlannerRiskFailure)
    assert result.kind == "non_positive_equity"


@pytest.mark.parametrize("binding", [None, 1, "x" * 64, "0" * 63, "A" * 64])
def test_phase_b_returns_typed_failure_for_malformed_binding(binding):
    strategy = _strategy()
    early = _early(strategy, positions={"OLD": 2.0})
    result = phase_b(strategy, LatePlannerInput(early, binding, _captured()))
    assert result == PhaseBindingFailure(
        "invalid_phase_a_binding", "binding must be 64 lowercase hex chars"
    )


@pytest.mark.parametrize(
    "change",
    [
        {"sizing_equity": float("inf")},
        {"drawdown_equity": float("nan")},
        {"persisted_peak_equity": float("inf")},
        {"quantities": {"OLD": float("nan")}},
        {"market_values": {"OLD": float("inf")}},
        {"venue_belief": VenueBeliefEnabled({"OLD": float("nan")})},
    ],
)
def test_phase_b_rejects_non_finite_captured_economics(change):
    strategy = _strategy()
    early = _early(strategy, positions={"OLD": 2.0})
    first = phase_a(strategy, early)
    assert isinstance(first, SnapshotRequired)
    result = phase_b(
        strategy, LatePlannerInput(early, first.phase_a_binding, replace(_captured(), **change))
    )
    expected = PlannerRiskFailure if "sizing_equity" in change else PlannerInputFailure
    assert isinstance(result, expected)


def test_phase_b_rejects_inconsistent_snapshot_symbol_sets():
    strategy = _strategy()
    early = _early(strategy, positions={"OLD": 2.0})
    first = phase_a(strategy, early)
    assert isinstance(first, SnapshotRequired)
    captured = replace(_captured(), market_values={})
    result = phase_b(strategy, LatePlannerInput(early, first.phase_a_binding, captured))
    assert isinstance(result, PlannerInputFailure)
    assert "symbol sets" in result.detail


def test_phase_b_rejects_unknown_venue_belief_variant():
    strategy = _strategy()
    early = _early(strategy, positions={"OLD": 2.0})
    first = phase_a(strategy, early)
    assert isinstance(first, SnapshotRequired)
    captured = replace(_captured(), venue_belief=object())
    result = phase_b(strategy, LatePlannerInput(early, first.phase_a_binding, captured))
    assert isinstance(result, PlannerInputFailure)
    assert result.code == "invalid_venue_belief"


@pytest.mark.parametrize(
    "change",
    [
        {"request_id": None},
        {"config_hash": None},
        {"now": "2023-01-05"},
        {"gate_universe": ["AAA"]},
        {"early_positions": []},
        {"max_drawdown": float("nan")},
    ],
)
def test_phase_a_returns_typed_failure_for_malformed_early_values(change):
    strategy = _strategy()
    assert isinstance(phase_a(strategy, replace(_early(strategy), **change)), PlannerInputFailure)


def test_phase_a_rejects_non_utc_clock_and_bar_index():
    strategy = _strategy()
    local_now = NOW.astimezone(timezone(timedelta(hours=2)))
    assert isinstance(
        phase_a(strategy, replace(_early(strategy), now=local_now)), PlannerInputFailure
    )
    local_bars = _bars()
    local_bars.index = local_bars.index.tz_convert("Asia/Jerusalem")
    assert isinstance(
        phase_a(strategy, replace(_early(strategy), raw_bars=local_bars)), PlannerInputFailure
    )


def test_phase_a_canonicalizes_numeric_dtypes_and_rejects_unicode_collisions():
    strategy = _strategy()
    integer_bars = _bars()
    integer_bars["volume"] = integer_bars["volume"].astype("int64")
    canonical = phase_a(strategy, _early(strategy))
    integer = phase_a(strategy, replace(_early(strategy), raw_bars=integer_bars))
    assert isinstance(canonical, SnapshotRequired)
    assert isinstance(integer, SnapshotRequired)
    assert integer.phase_a_binding == canonical.phase_a_binding
    collision = {"\u00e9": 1.0, "e\u0301": 2.0}
    assert isinstance(
        phase_a(strategy, replace(_early(strategy), early_positions=collision)),
        PlannerInputFailure,
    )
    collision_bars = _bars()
    symbol_column = collision_bars.columns.get_loc("symbol")
    collision_bars.iloc[0, symbol_column] = "\u00e9"
    collision_bars.iloc[1, symbol_column] = "e\u0301"
    assert isinstance(
        phase_a(strategy, replace(_early(strategy), raw_bars=collision_bars)),
        PlannerInputFailure,
    )


def test_gate_universe_overlay_preserves_authored_deployment_identity():
    authored = _strategy()
    effective = replace(authored, config=authored.config.model_copy(update={"universe": ["OLD"]}))
    early = replace(_early(authored), gate_universe=("OLD",))
    result = phase_a(effective, early)
    assert isinstance(result, SnapshotRequired)


def test_supervisor_accepts_gate_universe_overlay_with_deployment_identity():
    authored = _strategy()
    effective = replace(
        authored,
        config=authored.config.model_copy(update={"universe": ["OLD"]}),
        signal_fn=lambda view, params: pd.Series({"OLD": 0.5}),
    )
    resolved = json.dumps(
        authored.config.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    hooks = TickHooks(
        live_positions=lambda: {},
        live_snapshot=lambda bars: (
            TickSnapshot(100.0, {"OLD": 0.0}, {"OLD": 0.0}),
            100.0,
        ),
        cancel=lambda: None,
        planner_context=PlannerContext(1, 2, "a" * 64, config_hash(authored), resolved, "XNYS"),
    )
    broker = SimpleNamespace(submit_sized=lambda *args, **kwargs: "noop")
    provider = SimpleNamespace(get_bars=lambda *args, **kwargs: _bars())

    result = run_tick(effective, broker, provider, NOW, NOW, now=NOW, hooks=hooks)

    assert result.target_weights == {"OLD": 0.5}


def test_decision_output_contains_only_state_and_ordered_portfolio_intents():
    assert [field.name for field in fields(Decision)] == ["state", "ordered_intents"]
    assert [field.name for field in fields(PlannerState)] == [
        "decision_ts",
        "target_weights",
        "positions_before",
        "equity",
        "peak_equity",
        "reconcile_ok",
        "realized_gross",
    ]
    broker_fields = {"order_id", "client_order_id", "broker_response", "submission_result"}
    assert not broker_fields.intersection(field.name for field in fields(Decision))


def test_any_behavior_affecting_early_change_changes_the_binding():
    strategy = _strategy()
    early = _early(strategy, positions={"OLD": 2.0})
    baseline = phase_a(strategy, early)
    assert isinstance(baseline, SnapshotRequired)
    changes = [
        replace(early, request_id="fedcba9876543210fedcba9876543210"),
        replace(early, deployment_id=1, artifact_id=2, manifest_digest="a" * 64),
        replace(early, calendar_code="XNAS"),
        replace(early, now=NOW.replace(hour=1)),
        replace(early, max_drawdown=0.2),
        replace(early, early_positions={"OLD": 3.0}),
        replace(early, raw_bars=_bars().assign(close=lambda frame: frame["close"] + 1.0)),
    ]
    bindings = []
    for changed in changes:
        result = phase_a(strategy, changed)
        assert isinstance(result, SnapshotRequired)
        bindings.append(result.phase_a_binding)
    assert all(binding != baseline.phase_a_binding for binding in bindings)
    assert phase_a_binding(early, decision_ts=baseline.decision_ts, warming=True) != (
        baseline.phase_a_binding
    )
    assert (
        phase_a_binding(
            replace(early, gate_universe=("OLD",)),
            decision_ts=baseline.decision_ts,
            warming=False,
        )
        != baseline.phase_a_binding
    )

    changed_strategy = _strategy(params={"lookback": 2})
    changed_config = _early(changed_strategy, positions={"OLD": 2.0})
    changed_result = phase_a(changed_strategy, changed_config)
    assert isinstance(changed_result, SnapshotRequired)
    assert changed_result.phase_a_binding != baseline.phase_a_binding
