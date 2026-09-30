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


# --- Story 1.3c T6: the planner port (contract §3) -------------------------------------------
# `TickHooks.planner` is the one seam a frozen dispatcher replaces. Unset, run_tick must make
# exactly today's in-process calls (the golden masters above stay the proof of that); set, every
# planner call must go through it, with the venue-belief handshake and error mapping unchanged.


class _RecordingPort:
    """Delegates to the in-process adapter and records each port call into the effect trace."""

    def __init__(self, strategy, events):
        from algua.live.planner import InProcessPlanner

        self.inner = InProcessPlanner(strategy)
        self.events = events
        self.early: list = []
        self.late: list = []

    def phase_a(self, early):
        self.events.append("port.phase_a")
        self.early.append(early)
        return self.inner.phase_a(early)

    def closed_bars(self, early):
        self.events.append("port.closed_bars")
        self.early.append(early)
        return self.inner.closed_bars(early)

    def phase_b(self, late):
        self.events.append("port.phase_b")
        self.late.append(late)
        return self.inner.phase_b(late)


_DECISION_TAIL = [
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


def test_in_process_adapter_makes_exactly_todays_facade_calls(monkeypatch):
    from algua.live import planner

    calls = []
    sentinel = object()
    for name in ("phase_a", "phase_a_closed_bars", "phase_b"):
        monkeypatch.setattr(
            planner, name, lambda s, v, name=name: calls.append((name, s, v)) or sentinel
        )
    strategy, early, late = object(), object(), object()
    adapter = planner.InProcessPlanner(strategy)
    assert adapter.phase_a(early) is sentinel
    assert adapter.closed_bars(early) is sentinel
    assert adapter.phase_b(late) is sentinel
    assert calls == [
        ("phase_a", strategy, early),
        ("phase_a_closed_bars", strategy, early),
        ("phase_b", strategy, late),
    ]


def test_run_tick_routes_every_planner_call_through_the_port_live_snapshot_path():
    from algua.live.planner_contract import VenueBeliefEnabled, VenueBeliefPending

    args, events, _ = scenario(held={"OLD": 10.0}, qtys={"OLD": 20.0})
    port = _RecordingPort(args["strategy"], events)
    args["hooks"].planner = port

    result = run_tick(**args)

    assert events == [
        "positions",
        ("fetch", ("AAA", "OLD")),
        "port.phase_a",
        "port.closed_bars",
        "snapshot",
        "port.phase_b",
        "belief",
        "port.phase_b",
        *_DECISION_TAIL,
    ]
    assert len({id(early) for early in port.early}) == 1
    assert all(late.early is port.early[0] for late in port.late)
    pending, resolved = port.late
    assert pending.captured.venue_belief == VenueBeliefPending()
    assert resolved.captured.venue_belief == VenueBeliefEnabled({"OLD": 20.0})
    assert resolved.phase_a_binding == pending.phase_a_binding
    assert resolved.captured.request_id == pending.captured.request_id == port.early[0].request_id
    assert result.target_weights == {"AAA": 0.5}
    assert result.positions_before == {"OLD": 20.0}


def test_run_tick_routes_through_the_port_on_the_broker_snapshot_path():
    args, events, _ = scenario(held={"OLD": 10.0}, qtys={"OLD": 20.0})
    snap = TickSnapshot(equity=100.0, qtys={"OLD": 20.0}, market_values={"OLD": 20.0})

    def broker_snapshot(universe):
        events.append(("broker.snapshot", tuple(universe)))
        return snap

    args["broker"].snapshot = broker_snapshot
    args["hooks"].live_snapshot = None
    args["hooks"].planner = _RecordingPort(args["strategy"], events)
    args["broker"].submit_sized = lambda intent, s, coid, reserve=None: (
        events.append(("submit", intent.symbol, intent.target_weight))
        or ("noop" if intent.symbol == "OLD" else "accepted")
    )

    run_tick(**args)

    assert events == [
        "positions",
        ("fetch", ("AAA", "OLD")),
        "port.phase_a",
        ("broker.snapshot", ("AAA",)),
        "port.phase_b",
        "belief",
        "port.phase_b",
        *_DECISION_TAIL,
    ]


def test_unset_planner_and_explicit_in_process_adapter_are_indistinguishable():
    from algua.live.planner import InProcessPlanner

    outcomes = []
    for explicit in (False, True):
        args, events, _ = scenario(held={"OLD": 10.0}, qtys={"OLD": 20.0})
        if explicit:
            args["hooks"].planner = InProcessPlanner(args["strategy"])
        outcomes.append((repr(run_tick(**args)), events))
    assert outcomes[0] == outcomes[1]


class _ScriptedPort:
    """A port that never runs a strategy: canned results, as a frozen dispatcher would return."""

    def __init__(self, events, phase_a, closed_bars=None, phase_b=()):
        self.events = events
        self._a, self._bars, self._b = phase_a, closed_bars, list(phase_b)
        self.late: list = []

    def phase_a(self, early):
        self.events.append("port.phase_a")
        return self._a

    def closed_bars(self, early):
        self.events.append("port.closed_bars")
        return self._bars

    def phase_b(self, late):
        self.events.append("port.phase_b")
        self.late.append(late)
        return self._b.pop(0)


def _scripted_decision():
    from algua.contracts.types import OrderIntent, Side
    from algua.live.planner_contract import Decision, PlannerState

    state = PlannerState(DECISION, (("AAA", 0.25),), (("OLD", 20.0),), 100.0, 100.0, True, 0.2)
    return Decision(state, (OrderIntent("AAA", Side.BUY, 0.25, DECISION),))


def test_an_injected_port_is_the_only_planner_the_tick_consults(monkeypatch):
    from algua.live import planner
    from algua.live.planner_contract import SnapshotRequired, VenueBeliefRequired

    def forbidden(*_args, **_kwargs):
        raise AssertionError("run_tick bypassed the injected planner port")

    for name in ("phase_a", "phase_a_closed_bars", "phase_b", "plan"):
        monkeypatch.setattr(planner, name, forbidden)
    args, events, bars = scenario(held={"OLD": 10.0}, qtys={"OLD": 20.0})
    args["hooks"].planner = _ScriptedPort(
        events,
        SnapshotRequired(DECISION, False, "f" * 64),
        bars[bars.index < NOW],
        [VenueBeliefRequired(), _scripted_decision()],
    )

    result = run_tick(**args)

    assert events == [
        "positions",
        ("fetch", ("AAA", "OLD")),
        "port.phase_a",
        "port.closed_bars",
        "snapshot",
        "port.phase_b",
        "belief",
        "port.phase_b",
        "halt",
        "cancel",
        "halt",
        "halt",
        ("before", "AAA"),
        ("submit", "AAA", 0.25),
        ("persist", "AAA"),
    ]
    assert result.decision_ts == DECISION
    assert result.target_weights == {"AAA": 0.25}
    assert result.positions_before == {"OLD": 20.0}


def _port_failure_cases():
    from algua.live.planner_contract import (
        EarlyNoDecision,
        LateNoDecision,
        PhaseBindingFailure,
        PlannerInputFailure,
        PlannerRiskFailure,
        PlannerState,
        VenueBeliefRequired,
    )

    state = PlannerState(DECISION, (), (), 0.0, None, True, 0.0)
    head = ["positions", ("fetch", ("AAA", "OLD")), "port.phase_a"]
    late = [*head, "port.closed_bars", "snapshot", "port.phase_b"]
    final = [*late, "belief", "port.phase_b"]
    risk = PlannerRiskFailure("drawdown", "dd detail", False)
    return [
        ("a-risk", [risk], RiskBreach, "dd detail", head),
        ("a-input", [PlannerInputFailure("invalid_now", "bad now")], ValueError, "bad now", head),
        ("a-unknown", [object()], RuntimeError, "unknown Phase A result", head),
        ("a-no-decision", [EarlyNoDecision("warming", state)], None, None, head),
        ("b1-risk", [risk], RiskBreach, "dd detail", late),
        ("b1-binding", [PhaseBindingFailure("phase_a_binding_mismatch", "mixed")], ValueError,
         "mixed", late),
        ("b1-early-decision", [LateNoDecision("warming", state)], RuntimeError,
         "did not request venue belief", late),
        ("b2-risk", [VenueBeliefRequired(), risk], RiskBreach, "dd detail", final),
        ("b2-input", [VenueBeliefRequired(), PlannerInputFailure("invalid_captured_state", "cap")],
         ValueError, "cap", final),
        ("b2-repeat", [VenueBeliefRequired(), VenueBeliefRequired()], RuntimeError,
         "unknown Phase B result", final),
    ]


@pytest.mark.parametrize("case", _port_failure_cases(), ids=lambda case: case[0])
def test_port_results_keep_todays_error_mapping_and_stop_before_effects(case):
    from algua.live.planner_contract import SnapshotRequired

    label, results, error, message, trace = case
    args, events, bars = scenario(held={"OLD": 10.0}, qtys={"OLD": 20.0})
    if label.startswith("a-"):
        port = _ScriptedPort(events, results[0])
    else:
        port = _ScriptedPort(
            events, SnapshotRequired(DECISION, False, "f" * 64), bars[bars.index < NOW], results
        )
    args["hooks"].planner = port
    if error is None:
        result = run_tick(**args)
        assert result.decision_ts == DECISION and result.submitted == []
    else:
        with pytest.raises(error) as exc:
            run_tick(**args)
        assert type(exc.value) is error
        assert message in str(exc.value)
        if error is RiskBreach:
            assert exc.value.kind == "drawdown"
    assert events == trace
