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
    pd.testing.assert_series_equal(result.weights, weights)
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


def test_planner_dependency_closure_has_no_operational_authority():
    # Reuse the installed import-linter graph engine. Include TYPE_CHECKING edges so
    # importing a ledger-coupled DTO only for annotations cannot sneak across this seam.
    graph = grimp.build_graph("algua", include_external_packages=True, cache_dir=None)
    dependencies = graph.find_upstream_modules("algua.live.planner")
    pure_prefixes = ("algua.contracts", "algua.features", "algua.portfolio")
    allowed_leaves = {"algua.risk.limits", "algua.strategies.base"}
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
        "time", "importlib", "builtins",
    })
