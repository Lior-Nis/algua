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


def _unsorted_bars_with_today() -> pd.DataFrame:
    """Raw captured bars as a provider could deliver them: newest first, a same-day (unclosed)
    bar on `NOW`'s date, an NFD-spelled out-of-universe symbol and an integer volume column."""
    rows = [
        (datetime(2023, 1, 5, tzinfo=UTC), "AAA", 11.0),
        (datetime(2023, 1, 4, tzinfo=UTC), "é", 3.0),
        (datetime(2023, 1, 4, tzinfo=UTC), "OLD", 5.0),
        (datetime(2023, 1, 4, tzinfo=UTC), "AAA", 10.0),
        (datetime(2023, 1, 3, tzinfo=UTC), "OLD", 5.0),
        (datetime(2023, 1, 3, tzinfo=UTC), "AAA", 10.0),
        (datetime(2023, 1, 2, tzinfo=UTC), "OLD", 5.0),
        (datetime(2023, 1, 2, tzinfo=UTC), "AAA", 10.0),
    ]
    frame = pd.DataFrame(
        [
            {"timestamp": ts, "symbol": symbol, "open": price, "high": price, "low": price,
             "close": price, "adj_close": price, "volume": 100}
            for ts, symbol, price in rows
        ]
    ).set_index("timestamp")
    assert str(frame["volume"].dtype) == "int64"
    return frame


def _expected_closed_bars() -> pd.DataFrame:
    rows = [
        (datetime(2023, 1, 2, tzinfo=UTC), "OLD", 5.0),
        (datetime(2023, 1, 2, tzinfo=UTC), "AAA", 10.0),
        (datetime(2023, 1, 3, tzinfo=UTC), "OLD", 5.0),
        (datetime(2023, 1, 3, tzinfo=UTC), "AAA", 10.0),
        (datetime(2023, 1, 4, tzinfo=UTC), "é", 3.0),
        (datetime(2023, 1, 4, tzinfo=UTC), "OLD", 5.0),
        (datetime(2023, 1, 4, tzinfo=UTC), "AAA", 10.0),
    ]
    return pd.DataFrame(
        [
            {"timestamp": ts, "symbol": symbol, "open": price, "high": price, "low": price,
             "close": price, "adj_close": price, "volume": 100.0}
            for ts, symbol, price in rows
        ]
    ).set_index("timestamp")


def test_phase_a_closed_bars_golden_selection():
    """Pinned before the closed-bar selection was carved out of `prepare_early` (Story 1.3c T6):
    NFC symbols, float64 values, a STABLE timestamp sort and closed sessions only."""
    from algua.live.planner import phase_a_closed_bars

    strategy = _strategy()
    early = _early(strategy, bars=_unsorted_bars_with_today(), positions={"OLD": 2.0})
    pd.testing.assert_frame_equal(
        phase_a_closed_bars(strategy, early), _expected_closed_bars(), check_exact=True
    )


def test_closed_bar_selection_is_strategy_free_and_matches_phase_a():
    """The supervisor side of a frozen tick recomputes the closed bars without a strategy and
    checks the child's decision timestamp against them (contract §3)."""
    from algua.live.planner import phase_a_closed_bars
    from algua.live.planner_early import closed_universe_bars

    strategy = _strategy()
    raw = _unsorted_bars_with_today()
    pristine = raw.copy()
    early = _early(strategy, bars=raw, positions={"OLD": 2.0})

    closed = closed_universe_bars(raw, NOW, ("AAA",))

    pd.testing.assert_frame_equal(closed.bars, phase_a_closed_bars(strategy, early))
    pd.testing.assert_frame_equal(
        closed.universe_bars, _expected_closed_bars().query("symbol == 'AAA'"), check_exact=True
    )
    first = phase_a(strategy, early)
    assert isinstance(first, SnapshotRequired)
    assert closed.decision_ts == first.decision_ts == datetime(2023, 1, 4, tzinfo=UTC)
    pd.testing.assert_frame_equal(raw, pristine)  # the captured input is never mutated


def test_closed_bar_selection_without_a_closed_universe_session_has_no_decision_time():
    from algua.live.planner_early import closed_universe_bars

    raw = _unsorted_bars_with_today()
    only_today = closed_universe_bars(raw.iloc[:1], NOW, ("AAA",))
    assert only_today.bars.empty and only_today.universe_bars.empty
    assert only_today.decision_ts is None
    out_of_universe = closed_universe_bars(raw[raw["symbol"] == "OLD"], NOW, ("AAA",))
    assert len(out_of_universe.bars) == 3 and out_of_universe.universe_bars.empty
    assert out_of_universe.decision_ts is None


def test_a_reconcile_breach_reads_the_same_whatever_the_mapping_order():
    # A frozen child receives every mapping sorted by symbol (wire v1), the in-process planner in
    # broker order; the breach text must not depend on which (Story 1.3c parity).
    strategy = _strategy()
    early = _early(strategy, positions={"OLD": 2.0})
    first = phase_a(strategy, early)
    assert isinstance(first, SnapshotRequired)

    def breach(belief: dict[str, float]) -> str:
        result = phase_b(
            strategy, LatePlannerInput(early, first.phase_a_binding, _captured(belief=belief)))
        assert isinstance(result, PlannerRiskFailure) and result.kind == "reconcile"
        return result.detail

    assert breach({"ZZZ": 1.0, "OLD": 5.0}) == breach({"OLD": 5.0, "ZZZ": 1.0})


# --- golden master: every strategy-free verdict, pinned before it was shared (Story 1.3c) --------
#
# The early and late strategy-free verdicts were carved out of `phase_a`/`phase_b` so the frozen
# supervisor computes them with the same code. These digests were pinned against the planner BEFORE
# that carve, over each outcome's exact canonical form (types, field order, tuple order, float.hex
# numbers, ISO timestamps), so the carve is proven behaviour-identical, text included.


def _canon(value: object) -> object:
    from dataclasses import is_dataclass
    from enum import Enum

    if is_dataclass(value) and not isinstance(value, type):
        fields_ = {field.name: _canon(getattr(value, field.name)) for field in fields(value)}
        return [type(value).__name__, fields_]
    if isinstance(value, datetime):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, float):
        return value.hex()
    if isinstance(value, (tuple, list)):
        return [_canon(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    return value


def _golden_digest(result: object) -> tuple[str, str]:
    import hashlib

    text = json.dumps(_canon(result), sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:16], text


def _shifted(symbol: str, days: int) -> pd.DataFrame:
    bars = _bars()
    bars.index = pd.DatetimeIndex(
        [ts - timedelta(days=days) if sym == symbol else ts
         for ts, sym in zip(bars.index, bars.symbol, strict=True)],
        name="timestamp",
    )
    return bars


def _nan_close(symbol: str) -> pd.DataFrame:
    bars = _bars()
    bars.loc[(bars.symbol == symbol) & (bars.index == datetime(2023, 1, 4, tzinfo=UTC)),
             "close"] = float("nan")
    return bars


def _golden_early(case: str) -> tuple[LoadedStrategy, EarlyPlannerInput]:
    ready, warm = _strategy(), _strategy(warmup=3)
    held = {"OLD": 2.0}
    return {
        "a_snapshot": (ready, _early(ready, positions=held)),
        "a_snapshot_flat": (ready, _early(ready)),
        "a_no_bars_held": (ready, _early(ready, bars=_bars().iloc[0:0], positions=held)),
        "a_no_bars_flat": (ready, _early(ready, bars=_bars().iloc[0:0])),
        "a_warming_flat": (warm, _early(warm, positions={"ZERO": 0.0})),
        "a_warming_held": (warm, _early(warm, positions=held)),
        "a_stale_held": (ready, _early(ready, bars=_shifted("OLD", 21), positions=held)),
        "a_unvaluable_held": (ready, _early(ready, bars=_nan_close("OLD"), positions=held)),
        "a_no_mark_held": (ready, _early(ready, positions={"ZZZ": 1.0})),
        "a_input_failure": (ready, replace(_early(ready), request_id="xyz")),
    }[case]


def _golden_late(case: str) -> tuple[LoadedStrategy, LatePlannerInput]:
    ready, warm = _strategy(), _strategy(warmup=3)
    heavy = replace(ready, construct_fn=lambda scores, view, params: scores * 3.0)
    pending = VenueBeliefPending()
    captured = _captured()
    strategy, early, late_captured, binding = {
        "b_pending": (ready, None, replace(captured, venue_belief=pending), None),
        "b_decision_disabled": (ready, None, captured, None),
        "b_decision_enabled": (ready, None, _captured(belief={"OLD": 2.0}), None),
        "b_reconcile": (ready, None, _captured(belief={"OLD": 3.0, "ZZZ": 1.0}), None),
        "b_drawdown": (ready, None, replace(captured, persisted_peak_equity=200.0), None),
        "b_drawdown_pending": (
            ready, None, replace(captured, persisted_peak_equity=200.0, venue_belief=pending),
            None,
        ),
        "b_non_positive": (ready, None, _captured(sizing=0.0), None),
        "b_realized_gross": (ready, None, replace(captured, market_values={"OLD": 200.0}), None),
        "b_invalid_captured": (ready, None, replace(captured, drawdown_equity=float("nan")), None),
        "b_bad_belief_tag": (
            ready, None, replace(captured, venue_belief=VenueBeliefEnabled({}, tag="x")), None,
        ),
        "b_binding_mismatch": (ready, None, captured, "0" * 64),
        "b_universe_stale": (
            ready, _early(ready, bars=_shifted("AAA", 21), positions={"OLD": 2.0}), captured, None,
        ),
        "b_weights_breach": (heavy, None, captured, None),
        "b_warming_held": (warm, None, captured, None),
        "b_warming_held_drawdown": (
            warm, None, replace(captured, persisted_peak_equity=200.0), None,
        ),
    }[case]
    early = _early(strategy, positions={"OLD": 2.0}) if early is None else early
    first = phase_a(strategy, early)
    assert isinstance(first, SnapshotRequired)
    return strategy, LatePlannerInput(
        early, first.phase_a_binding if binding is None else binding, late_captured
    )


#: scenario -> (outcome type, sha256[:16] of its canonical form), pinned pre-carve.
GOLDEN = {
    "a_snapshot": ("SnapshotRequired", "1f1d20ad591d2992"),
    "a_snapshot_flat": ("SnapshotRequired", "631b7e2483de0510"),
    "a_no_bars_held": ("PlannerRiskFailure", "858bb5064e82c01c"),
    "a_no_bars_flat": ("EarlyNoDecision", "b1771f868b9c6cec"),
    "a_warming_flat": ("EarlyNoDecision", "bb0dcb30ba14651b"),
    "a_warming_held": ("SnapshotRequired", "978d7c24486ad0d6"),
    "a_stale_held": ("PlannerRiskFailure", "80f82a7811e04630"),
    "a_unvaluable_held": ("PlannerRiskFailure", "fb16aa86875ac800"),
    "a_no_mark_held": ("PlannerRiskFailure", "474c55468f396b87"),
    "a_input_failure": ("PlannerInputFailure", "08831f79d01bd999"),
    "b_pending": ("VenueBeliefRequired", "ad2382c95cea9cde"),
    "b_decision_disabled": ("Decision", "0e90f3a129edb111"),
    "b_decision_enabled": ("Decision", "0e90f3a129edb111"),
    "b_reconcile": ("PlannerRiskFailure", "7e86ce38ea231d97"),
    "b_drawdown": ("PlannerRiskFailure", "bb61415480eea80c"),
    "b_drawdown_pending": ("PlannerRiskFailure", "bb61415480eea80c"),
    "b_non_positive": ("PlannerRiskFailure", "ad3cedd9f9c6b5e5"),
    "b_realized_gross": ("PlannerRiskFailure", "b626e5a71ad4b101"),
    "b_invalid_captured": ("PlannerInputFailure", "5604a0fb550ab0c9"),
    "b_bad_belief_tag": ("PlannerInputFailure", "3d2bd1fed1f25ee3"),
    "b_binding_mismatch": ("PhaseBindingFailure", "15d9b6631dffa427"),
    "b_universe_stale": ("PlannerRiskFailure", "539599bdcd17dfc0"),
    "b_weights_breach": ("PlannerRiskFailure", "3da69b86839fc1ad"),
    "b_warming_held": ("LateNoDecision", "c7ccb16f106f6c48"),
    "b_warming_held_drawdown": ("PlannerRiskFailure", "bb61415480eea80c"),
}


@pytest.mark.parametrize("case", sorted(GOLDEN))
def test_every_planner_outcome_matches_its_pre_carve_golden(case):
    if case.startswith("a_"):
        result: object = phase_a(*_golden_early(case))
    else:
        result = phase_b(*_golden_late(case))
    digest, text = _golden_digest(result)
    assert (type(result).__name__, digest) == GOLDEN[case], text


def test_realized_gross_is_summed_in_symbol_order_whatever_the_broker_order():
    # A frozen child receives captured values sorted by symbol while the in-process planner sees
    # broker order; the state (and the realized-gross wall) must be the same either way (Story 1.3c
    # parity). The sum runs in symbol order; Python's compensated sum() also keeps it exact here.
    from algua.live.planner_late import _late_state

    def gross(order: list[str]) -> float:
        values = {"A": 10.0, "B": 20.0, "C": 30.0}
        captured = CapturedStrategyState(
            request_id="0123456789abcdef0123456789abcdef", sizing_equity=100.0,
            drawdown_equity=100.0, quantities={symbol: 1.0 for symbol in order},
            market_values={symbol: values[symbol] for symbol in order},
            persisted_peak_equity=100.0, venue_belief=VenueBeliefDisabled())
        return _late_state(captured, None)[0].realized_gross

    import math

    assert gross(["C", "B", "A"]) == gross(["A", "B", "C"]) == gross(["B", "C", "A"])
    assert gross(["A", "B", "C"]) == math.fsum([0.1, 0.2, 0.3])
