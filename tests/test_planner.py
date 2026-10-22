"""Public in-process planner contract; operational adapters remain outside this seam."""

from datetime import UTC, datetime
from types import SimpleNamespace

import grimp
import pandas as pd
import pytest

from algua.contracts.types import ExecutionContract, OrderIntent, Side
from algua.risk.limits import RiskBreach


def test_planner_returns_validated_weights_and_sorted_rebalance_intents():
    from algua.live.planner import PlannerInput, plan

    ts = datetime(2023, 1, 4, tzinfo=UTC)
    weights = pd.Series({"BBB": 0.25, "AAA": 0.5})
    view = pd.DataFrame({"symbol": ["AAA", "BBB"]})
    strategy = SimpleNamespace(
        name="test", universe=["AAA", "BBB"],
        execution=ExecutionContract(rebalance_frequency="1d"),
        target_weights=lambda bars: weights,
    )
    request = PlannerInput(view, {"OLD": 0.2, "AAA": 0.5}, ts)
    result = plan(strategy, request)
    # Validated and emitted once as float64 in symbol order (Story 1.3c: the supervisor re-runs
    # the weight rules on exactly this data).
    pd.testing.assert_series_equal(result.weights, weights.astype("float64").sort_index())
    assert result.intents == [
        OrderIntent("BBB", Side.BUY, 0.25, ts),
        OrderIntent("OLD", Side.SELL, 0.0, ts),
    ]
    assert result.protocol_version == 1
    again = plan(strategy, request)
    pd.testing.assert_series_equal(again.weights, result.weights)
    assert again.intents == result.intents


@pytest.mark.parametrize("version", [0, 2, -1, True, 1.0, "1", None])
def test_unsupported_protocol_fails_before_strategy_access(version):
    from algua.live.planner import PlannerInput, plan

    request = PlannerInput(pd.DataFrame(), {}, datetime(2023, 1, 4, tzinfo=UTC), version)
    with pytest.raises(ValueError, match="unsupported planner protocol"):
        plan(object(), request)


def test_both_loops_share_the_planner_compatibility_surface():
    from algua.live import live_loop, paper_loop, planner

    assert live_loop.decide is planner.decide
    assert paper_loop.decide is planner.decide
    assert paper_loop.build_intents is planner.build_intents


@pytest.mark.parametrize("weights,kind", [
    ({"AAA": float("nan")}, "non_finite_weight"),
    ({"OTHER": 0.5}, "out_of_universe"),
    ({"AAA": -0.5}, "long_only"),
    ({"AAA": 1.1}, "max_weight_per_symbol"),
    ({"AAA": 0.6, "BBB": 0.6}, "gross_exposure"),
])
def test_planner_retains_shared_weight_rails(weights, kind):
    from algua.live.planner import PlannerInput, plan

    strategy = SimpleNamespace(
        name="test", universe=["AAA", "BBB"],
        execution=ExecutionContract(rebalance_frequency="1d"),
        target_weights=lambda view: pd.Series(weights),
    )
    with pytest.raises(RiskBreach) as exc:
        plan(strategy, PlannerInput(pd.DataFrame(), {}, datetime(2023, 1, 4, tzinfo=UTC)))
    assert exc.value.kind == kind


def _run_planner_entry(entry: str, current: dict, target: float) -> None:
    """Call ``build_intents`` or ``plan`` with one finite string target and ``current`` holdings."""
    from algua.live.planner import PlannerInput, build_intents, plan

    ts = datetime(2023, 1, 4, tzinfo=UTC)
    if entry == "build_intents":
        build_intents(pd.Series({"AAPL": target}), current, ts)
        return
    strategy = SimpleNamespace(
        name="test", universe=["AAPL"],
        execution=ExecutionContract(rebalance_frequency="1d"),
        target_weights=lambda view: pd.Series({"AAPL": target}),
    )
    plan(strategy, PlannerInput(pd.DataFrame(), current, ts))


@pytest.mark.parametrize("entry", ["build_intents", "plan"])
def test_nan_current_weight_with_nonzero_target_fails_before_intents(entry):
    # A NaN holding must not compare equal to every target and silently emit no order.
    with pytest.raises(ValueError, match="must be a finite number"):
        _run_planner_entry(entry, {"AAPL": float("nan")}, 0.5)


@pytest.mark.parametrize("entry", ["build_intents", "plan"])
def test_nan_current_weight_on_exit_fails_before_intents(entry):
    # An exit (target 0) against a NaN holding is the same contract error, not a skipped symbol.
    with pytest.raises(ValueError, match="must be a finite number"):
        _run_planner_entry(entry, {"AAPL": float("nan")}, 0.0)


@pytest.mark.parametrize("entry", ["build_intents", "plan"])
@pytest.mark.parametrize("holding", [float("inf"), float("-inf")])
@pytest.mark.parametrize("target", [0.0, 0.5])
def test_infinite_current_weight_fails_before_intents(entry, holding, target):
    with pytest.raises(ValueError, match="must be a finite number"):
        _run_planner_entry(entry, {"AAPL": holding}, target)


@pytest.mark.parametrize("entry", ["build_intents", "plan"])
@pytest.mark.parametrize("current", [{1: 0.1}, {"": 0.1}])
def test_non_str_or_empty_current_weight_key_fails_before_intents(entry, current):
    with pytest.raises(ValueError, match="non-empty strings"):
        _run_planner_entry(entry, current, 0.5)


def test_mixed_target_labels_raise_non_finite_weight():
    # A zero-weight non-string label used to pass the rails and TypeError inside build_intents.
    from algua.live.planner import PlannerInput, plan

    strategy = SimpleNamespace(
        name="test", universe=["AAPL"],
        execution=ExecutionContract(rebalance_frequency="1d"),
        target_weights=lambda view: pd.Series({1: 0.0, "AAPL": 0.5}),
    )
    with pytest.raises(RiskBreach) as exc:
        plan(strategy, PlannerInput(pd.DataFrame(), {}, datetime(2023, 1, 4, tzinfo=UTC)))
    assert exc.value.kind == "non_finite_weight"
    assert "non-string symbol label" in exc.value.detail


def test_valid_inputs_keep_sorted_delta_intents():
    # Legal string labels and finite weights stay the sorted delta intents from baseline a3c71fa.
    from algua.live.planner import build_intents

    ts = datetime(2023, 1, 4, tzinfo=UTC)
    intents = build_intents(
        pd.Series({"BBB": 0.25, "AAA": 0.5}), {"OLD": 0.2, "AAA": 0.5}, ts,
    )
    assert intents == [
        OrderIntent("BBB", Side.BUY, 0.25, ts),
        OrderIntent("OLD", Side.SELL, 0.0, ts),
    ]


def test_planner_dependency_closure_has_no_operational_authority():
    # Reuse the installed import-linter graph engine. Include TYPE_CHECKING edges so
    # importing a ledger-coupled DTO only for annotations cannot sneak across this seam.
    graph = grimp.build_graph("algua", include_external_packages=True, cache_dir=None)
    dependencies = graph.find_upstream_modules("algua.live.planner")
    pure_prefixes = ("algua.contracts", "algua.features", "algua.portfolio")
    allowed_leaves = {
        "algua.calendar.market_calendar", "algua.live.planner_binding",
        "algua.live.planner_contract", "algua.live.planner_decision",
        "algua.live.planner_early", "algua.live.planner_late",
        "algua.live.planner_validation", "algua.risk.limits",
        "algua.strategies.base",
    }
    forbidden = {
        name for name in dependencies if name.startswith("algua.")
        and name not in allowed_leaves
        and not any(name == prefix or name.startswith(prefix + ".") for prefix in pure_prefixes)
    }
    assert not forbidden, f"planner reaches operational modules: {sorted(forbidden)}"
    # The LoadedStrategy annotation reaches the existing feature catalogue's discovery
    # imports, but does not execute them. Check external capabilities on runtime edges.
    # This is a static boundary ratchet, not a sandbox proof for authored callbacks or pandas.
    runtime = grimp.build_graph("algua", include_external_packages=True,
                               exclude_type_checking_imports=True, cache_dir=None)
    assert not runtime.find_upstream_modules("algua.live.planner").intersection({
        "os", "pathlib", "sqlite3", "socket", "subprocess", "requests", "httpx",
        "time", "builtins",
    })
