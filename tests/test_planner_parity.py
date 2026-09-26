"""Baseline characterization at the approved tick/provider/broker/hook seams."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

from algua.contracts.types import ExecutionContract
from algua.execution.alpaca_broker import TickSnapshot
from algua.live.live_loop import PlannerContext, TickHalted, TickHooks, run_tick
from algua.risk.limits import RiskBreach

NOW = datetime(2023, 1, 5, tzinfo=UTC)
START = datetime(2023, 1, 3, tzinfo=UTC)
DECISION = datetime(2023, 1, 4, tzinfo=UTC)


def scenario(*, held=None, qtys=None, warmup=0, equity=100.0, nav=100.0):
    events = []
    bars = pd.DataFrame(
        [
            {
                "timestamp": ts,
                "symbol": sym,
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 1.0,
                "adj_close": 1.0,
                "volume": 1.0,
            }
            for ts in (START, DECISION, NOW)
            for sym in ("AAA", "OLD")
        ]
    ).set_index("timestamp")
    snap = TickSnapshot(equity=equity, qtys=qtys or {}, market_values=qtys or {})

    def signal(view):
        events.append("decision")
        assert set(view.symbol) == {"AAA"}
        assert view.index.max() == DECISION
        return pd.Series({"AAA": 0.5})

    def fetch(symbols, start, end, timeframe):
        events.append(("fetch", tuple(symbols)))
        return bars[bars.symbol.isin(symbols)]

    def positions():
        events.append("positions")
        return held or {}

    def snapshot(view):
        events.append("snapshot")
        assert view.index.max() == DECISION
        return snap, nav

    def belief():
        events.append("belief")
        return qtys or {}

    def halt():
        events.append("halt")
        return False

    def submit(intent, snapshot, coid, reserve=None):
        assert snapshot is snap
        events.append(("submit", intent.symbol, intent.target_weight))
        return "noop" if intent.symbol == "OLD" else "accepted"

    strategy = SimpleNamespace(
        name="test",
        universe=["AAA"],
        target_weights=signal,
        execution=ExecutionContract(rebalance_frequency="1d", warmup_bars=warmup),
    )
    hooks = TickHooks(
        live_positions=positions,
        live_snapshot=snapshot,
        venue_belief=belief,
        should_halt=halt,
        cancel=lambda: events.append("cancel"),
        client_order_id_for=lambda name, ts, sym: sym,
        before_submit=lambda intent, coid: events.append(("before", coid)),
        on_submitted=lambda order: events.append(("persist", order.symbol)),
        on_noop=lambda intent, coid: events.append(("noop", coid)),
    )
    args = dict(
        strategy=strategy,
        broker=SimpleNamespace(submit_sized=submit),
        provider=SimpleNamespace(get_bars=fetch),
        start=START,
        end=NOW,
        now=NOW,
        hooks=hooks,
    )
    return args, events, bars


def test_decision_and_effect_order_including_dropped_holding_and_noop():
    args, events, _ = scenario(held={"OLD": 10.0}, qtys={"OLD": 20.0})
    result = run_tick(**args)
    assert events == [
        "positions",
        ("fetch", ("AAA", "OLD")),
        "snapshot",
        "belief",
        "decision",
        "halt",
        "cancel",
        "halt",
        "halt",
        ("before", "AAA"),
        ("submit", "AAA", 0.5),
        ("persist", "AAA"),
        "halt",
        ("before", "OLD"),
        ("submit", "OLD", 0.0),
        ("noop", "OLD"),
    ]
    assert result.positions_before == {"OLD": 20.0}
    assert result.target_weights == {"AAA": 0.5}
    assert result.realized_gross == 0.2
    assert [order["symbol"] for order in result.submitted] == ["AAA"]


@pytest.mark.parametrize("empty", [False, True])
def test_flat_early_return_does_not_acquire_snapshot(empty):
    args, events, bars = scenario(warmup=2)
    if empty:
        bars.drop(bars.index, inplace=True)
    result = run_tick(**args)
    assert events == ["positions", ("fetch", ("AAA",))]
    assert result.decision_ts == (None if empty else DECISION)
    assert result.target_weights == {} and result.submitted == []
    assert result.equity == 0 and result.peak_equity is None


def test_held_warmup_still_values_distinct_nav_and_snapshot_holdings():
    args, events, _ = scenario(
        held={"OLD": 10.0}, qtys={"OLD": 20.0}, warmup=2, equity=100.0, nav=120.0
    )
    result = run_tick(**args)
    assert events == ["positions", ("fetch", ("AAA", "OLD")), "snapshot", "belief"]
    assert result.equity == result.peak_equity == 120.0
    assert result.positions_before == {"OLD": 20.0}
    assert result.realized_gross == 0.2 and result.submitted == []


@pytest.mark.parametrize(
    "case,kind,tail",
    [
        ("mark", "unvaluable_marks", []),
        ("stale", "stale_marks", []),
        ("equity", "non_positive_equity", ["snapshot"]),
        ("drawdown", "drawdown", ["snapshot"]),
        ("reconcile", "reconcile", ["snapshot", "belief"]),
        ("gross", "gross_exposure_realized", ["snapshot", "belief"]),
    ],
)
def test_failure_stages_preserve_reads_and_prevent_downstream_effects(case, kind, tail):
    args, events, bars = scenario(
        held={"OLD": 10.0},
        qtys={"OLD": 200.0 if case == "gross" else 10.0},
        equity=0.0 if case == "equity" else 100.0,
    )
    if case == "mark":
        bars.loc[(bars.index == DECISION) & (bars.symbol == "OLD"), "close"] = float("nan")
    if case == "stale":
        bars.index = pd.DatetimeIndex(
            [
                ts - timedelta(days=21) if sym == "OLD" else ts
                for ts, sym in zip(bars.index, bars.symbol, strict=True)
            ],
            name="timestamp",
        )
        assert bars[bars.symbol == "AAA"].index.max() == NOW
        assert bars[bars.symbol == "OLD"].index.max() < START
    if case == "drawdown":
        args["hooks"].peak_equity = 200.0
        args["max_drawdown"] = 0.1
    if case == "reconcile":

        def empty_belief():
            events.append("belief")
            return {}

        args["hooks"].venue_belief = empty_belief
    with pytest.raises(RiskBreach) as exc:
        run_tick(**args)
    assert exc.value.kind == kind
    if case == "stale":
        assert "OLD" in exc.value.detail and "stale(" in exc.value.detail
    assert events == ["positions", ("fetch", ("AAA", "OLD")), *tail]


@pytest.mark.parametrize("halt_number", [1, 2, 3, 4])
def test_halt_stops_at_each_execution_boundary(halt_number):
    args, events, _ = scenario(held={"OLD": 10.0}, qtys={"OLD": 10.0})
    calls = 0

    def halt():
        nonlocal calls
        calls += 1
        events.append("halt")
        return calls == halt_number

    args["hooks"].should_halt = halt
    with pytest.raises(TickHalted):
        run_tick(**args)
    assert calls == halt_number
    assert ("cancel" in events) == (halt_number > 1)
    assert (("persist", "AAA") in events) == (halt_number == 4)
    assert ("before", "OLD") not in events


def test_non_daily_rejected_before_any_input_acquisition():
    args, events, _ = scenario()
    with pytest.raises(ValueError, match="only 1d"):
        run_tick(**args, timeframe="1h")
    assert events == []


def test_protocol_mismatch_prevents_decision_cancel_and_submit(monkeypatch):
    from algua.live import planner_decision

    args, events, _ = scenario()
    # The compatibility request is stamped v1; emulate a dispatcher/planner mismatch.
    monkeypatch.setattr(planner_decision, "PLANNER_PROTOCOL_VERSION", 2)
    with pytest.raises(ValueError, match="unsupported planner protocol"):
        run_tick(**args)
    assert events == ["positions", ("fetch", ("AAA",)), "snapshot", "belief"]


def test_supervisor_routes_pre_cancel_work_through_two_stateless_phases(monkeypatch):
    """The approved boundary runs after early capture and before any execution effect.

    This is the tracer bullet for Story 1.3a.  The detailed branch fixtures above remain the
    golden master for what each phase must preserve; this assertion pins where the new public
    phase seams sit in the already-characterized effect trace.
    """
    from algua.live import planner

    args, events, _ = scenario(held={"OLD": 10.0}, qtys={"OLD": 20.0})
    args["hooks"].planner_context = PlannerContext(
        deployment_id=7,
        artifact_id=11,
        manifest_digest="a" * 64,
        config_hash="b" * 32,
        resolved_config_json='{"name":"test"}',
        calendar_code="XNYS",
    )
    captured_early = []
    original_phase_a = planner.phase_a
    original_phase_b = planner.phase_b

    def phase_a(*phase_args, **phase_kwargs):
        events.append("phase_a")
        captured_early.append(phase_args[1])
        return original_phase_a(*phase_args, **phase_kwargs)

    def phase_b(*phase_args, **phase_kwargs):
        events.append("phase_b")
        return original_phase_b(*phase_args, **phase_kwargs)

    monkeypatch.setattr(planner, "phase_a", phase_a)
    monkeypatch.setattr(planner, "phase_b", phase_b)

    run_tick(**args)

    assert events == [
        "positions",
        ("fetch", ("AAA", "OLD")),
        "phase_a",
        "snapshot",
        "phase_b",
        "phase_a",
        "belief",
        "phase_b",
        "phase_a",
        "decision",
        "halt",
        "cancel",
        "halt",
        "halt",
        ("before", "AAA"),
        ("submit", "AAA", 0.5),
        ("persist", "AAA"),
        "halt",
        ("before", "OLD"),
        ("submit", "OLD", 0.0),
        ("noop", "OLD"),
    ]
    assert all(item.deployment_id == 7 for item in captured_early)
    assert all(item.artifact_id == 11 for item in captured_early)
    assert all(item.manifest_digest == "a" * 64 for item in captured_early)
