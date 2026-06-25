"""Issue #208: static-mode observation parity. A misbehaving provider that returns an UNDECLARED
symbol (one not in strategy.universe) must never have that symbol's data reach the strategy's view,
panel, weights/grid, or fundamentals/news sidecars. Mirror of #179, which closed the out-of-universe
WEIGHT path; this closes the OBSERVATION path."""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd
import pytest

from algua.backtest._sample import SyntheticProvider
from algua.backtest.engine import BacktestError, run, simulate, verify_signal_panel_parity
from algua.contracts.types import ExecutionContract
from algua.data.fundamentals_schema import to_fundamentals_schema
from algua.data.news_schema import to_news_schema
from algua.strategies.base import LoadedStrategy, StrategyConfig

START = datetime(2024, 1, 1, tzinfo=UTC)
END = datetime(2024, 4, 1, tzinfo=UTC)


def _passthrough(scores: pd.Series, view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """Identity construction: scores ARE the desired raw weights."""
    return scores


class _ExtraSymbolProvider:
    """A misbehaving provider: returns bars for the requested symbols PLUS an undeclared `extra`."""

    def __init__(self, extra: str = "ZZZ", seed: int = 0) -> None:
        self.extra = extra
        self.seed = seed

    def get_bars(self, symbols, start, end, timeframe):  # noqa: ANN001
        requested = list(symbols)
        return SyntheticProvider(seed=self.seed).get_bars(
            requested + [self.extra], start, end, timeframe
        )


class _ViewRecorder:
    """A 2-arg signal that records every symbol it is shown and returns FLAT weights (so the
    observation check is isolated from the #179 weight-rejection path)."""

    def __init__(self) -> None:
        self.seen: set[str] = set()

    def __call__(self, view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        self.seen.update(view["symbol"].unique())
        return pd.Series(dtype="float64")


def _loop_strategy(signal) -> LoadedStrategy:  # noqa: ANN001
    cfg = StrategyConfig(
        name="obs_loop", universe=["AAA", "BBB"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1, warmup_bars=0),
        params={}, construction="passthrough",
    )
    return LoadedStrategy(config=cfg, signal_fn=signal, construct_fn=_passthrough)


def test_loop_view_excludes_undeclared_symbol() -> None:
    recorder = _ViewRecorder()
    strat = _loop_strategy(recorder)
    run(strat, _ExtraSymbolProvider(extra="ZZZ", seed=3), START, END)
    assert recorder.seen, "signal was never invoked"
    assert "ZZZ" not in recorder.seen
    assert recorder.seen <= {"AAA", "BBB"}


class _PanelRecorder:
    """A signal_panel that records the symbols it is handed and returns a FLAT (all-zero) scores
    matrix. Paired with a flat 2-arg signal so the fast-path parity guard (which compares the panel
    against the per-bar loop) holds."""

    def __init__(self) -> None:
        self.seen: set[str] = set()

    def __call__(self, bars: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        self.seen.update(bars["symbol"].unique())
        adj = bars.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
        return pd.DataFrame(0.0, index=adj.index, columns=adj.columns)


class _ConstructViewRecorder:
    """An identity construction policy that ALSO records the symbols present in the per-bar
    construction view (`view_t`). Used to pin that the fast-path construct step never sees an
    undeclared symbol either (not just signal_panel)."""

    def __init__(self) -> None:
        self.seen: set[str] = set()

    def __call__(self, scores: pd.Series, view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        self.seen.update(view["symbol"].unique())
        return scores


def _fast_strategy(panel: _PanelRecorder, construct=None) -> LoadedStrategy:  # noqa: ANN001
    cfg = StrategyConfig(
        name="obs_fast", universe=["AAA", "BBB"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1, warmup_bars=0),
        params={}, construction="passthrough",
    )

    # Flat loop twin so the fast path's parity guard agrees with the panel (both produce 0.0).
    def flat_loop(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        syms = sorted(view["symbol"].unique())
        return pd.Series(0.0, index=syms) if syms else pd.Series(dtype="float64")

    return LoadedStrategy(
        config=cfg, signal_fn=flat_loop, signal_panel_fn=panel,
        construct_fn=construct if construct is not None else _passthrough,
    )


def test_fast_path_panel_excludes_undeclared_symbol() -> None:
    panel = _PanelRecorder()
    construct = _ConstructViewRecorder()
    strat = _fast_strategy(panel, construct=construct)
    # run() drives simulate(), which selects the fast path (signal_panel_fn set, static mode).
    run(strat, _ExtraSymbolProvider(extra="ZZZ", seed=3), START, END)
    assert panel.seen, "signal_panel was never invoked"
    assert "ZZZ" not in panel.seen
    assert panel.seen <= {"AAA", "BBB"}
    # The per-bar construct view (view_t) is also projected — undeclared symbol never reaches it.
    assert construct.seen, "construct was never invoked"
    assert "ZZZ" not in construct.seen
    assert construct.seen <= {"AAA", "BBB"}


def test_verify_signal_panel_parity_panel_excludes_undeclared_symbol() -> None:
    panel = _PanelRecorder()
    strat = _fast_strategy(panel)
    # verify_signal_panel_parity fetches its own bars and runs both the panel and the loop.
    verify_signal_panel_parity(strat, _ExtraSymbolProvider(extra="ZZZ", seed=3), START, END)
    assert panel.seen, "signal_panel was never invoked"
    assert "ZZZ" not in panel.seen
    assert panel.seen <= {"AAA", "BBB"}


class _FundRecorder:
    """A 3-arg fundamentals signal that records the symbols present in the as-of fundamentals frame
    it is handed, and returns FLAT weights."""

    def __init__(self) -> None:
        self.seen: set[str] = set()

    def __call__(
        self, view: pd.DataFrame, params: dict[str, Any], fundamentals: pd.DataFrame
    ) -> pd.Series:
        self.seen.update(fundamentals["symbol"].unique())
        return pd.Series(dtype="float64")


class _ExtraFundamentalsProvider:
    """Returns fundamentals for the requested symbols PLUS the undeclared `extra` (misbehaving)."""

    def __init__(self, extra: str = "ZZZ") -> None:
        self.extra = extra

    def get_fundamentals(self, symbols, end):  # noqa: ANN001
        rows = [
            [s, "2023-12-31", "eps_diluted", 1.0, "2023-12-31T00:00:00Z", "v"]
            for s in list(symbols) + [self.extra]
        ]
        raw = pd.DataFrame(rows, columns=[
            "symbol", "fiscal_period_end", "metric", "value", "knowable_at", "source",
        ])
        return to_fundamentals_schema(raw)


def test_fundamentals_sidecar_excludes_undeclared_symbol() -> None:
    recorder = _FundRecorder()
    cfg = StrategyConfig(
        name="obs_funds", universe=["AAA", "BBB"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1, warmup_bars=0),
        params={}, construction="passthrough", needs_fundamentals=True,
    )
    strat = LoadedStrategy(config=cfg, fundamentals_signal_fn=recorder, construct_fn=_passthrough)
    # Bars provider returns ZZZ too: without #208 projection, adj.columns would include ZZZ, so the
    # loop's `allowed = set(columns)` would NOT mask ZZZ out of the fundamentals frame.
    run(
        strat, _ExtraSymbolProvider(extra="ZZZ", seed=3), START, END,
        fundamentals_provider=_ExtraFundamentalsProvider(extra="ZZZ"),
    )
    assert recorder.seen, "fundamentals signal was never invoked"
    assert "ZZZ" not in recorder.seen


class _NewsRecorder:
    def __init__(self) -> None:
        self.seen: set[str] = set()

    def __call__(
        self, view: pd.DataFrame, params: dict[str, Any], news: pd.DataFrame
    ) -> pd.Series:
        self.seen.update(news["symbol"].unique())
        return pd.Series(dtype="float64")


class _ExtraNewsProvider:
    def __init__(self, extra: str = "ZZZ") -> None:
        self.extra = extra

    def get_news(self, symbols, end):  # noqa: ANN001
        raw = pd.DataFrame([
            {
                "source": "src", "article_id": "art-" + s, "symbol": s,
                "published_at": "2023-01-01T00:00:00Z", "knowable_at": "2023-01-01T00:00:00Z",
                "headline": "headline", "url": None, "body": None, "retracted": False,
            }
            for s in list(symbols) + [self.extra]
        ])
        return to_news_schema(raw)


def test_news_sidecar_excludes_undeclared_symbol() -> None:
    recorder = _NewsRecorder()
    cfg = StrategyConfig(
        name="obs_news", universe=["AAA", "BBB"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1, warmup_bars=0),
        params={}, construction="passthrough", needs_news=True,
    )
    strat = LoadedStrategy(config=cfg, news_signal_fn=recorder, construct_fn=_passthrough)
    run(
        strat, _ExtraSymbolProvider(extra="ZZZ", seed=3), START, END,
        news_provider=_ExtraNewsProvider(extra="ZZZ"),
    )
    assert recorder.seen, "news signal was never invoked"
    assert "ZZZ" not in recorder.seen


def test_compliant_provider_is_a_noop() -> None:
    """A compliant provider (returns exactly the declared universe) runs cleanly through the
    projection and produces the standard metric keys (no projection-induced break)."""

    def ew(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        syms = sorted(view["symbol"].unique())
        return pd.Series(1.0 / len(syms), index=syms) if syms else pd.Series(dtype="float64")

    strat = _loop_strategy(ew)
    res = run(strat, SyntheticProvider(seed=3), START, END)
    for key in ["total_return", "sharpe", "n_rebalances", "avg_gross_exposure"]:
        assert key in res.metrics


def test_compliant_provider_preserves_weight_columns_and_order() -> None:
    """Projection is order-preserving and a strict no-op for a compliant provider: the effective
    weights cover exactly the declared universe, in adj-column order."""

    def ew(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        syms = sorted(view["symbol"].unique())
        return pd.Series(1.0 / len(syms), index=syms) if syms else pd.Series(dtype="float64")

    strat = _loop_strategy(ew)
    _pf, weights_eff, _forced = simulate(strat, SyntheticProvider(seed=3), START, END)
    assert list(weights_eff.columns) == ["AAA", "BBB"]


def test_provider_returns_only_undeclared_fails_closed() -> None:
    """All declared symbols missing (provider returns only an undeclared symbol) → fail closed."""

    class _OnlyWrongProvider:
        def get_bars(self, symbols, start, end, timeframe):  # noqa: ANN001
            return SyntheticProvider(seed=0).get_bars(["ZZZ"], start, end, timeframe)

    strat = _loop_strategy(_ViewRecorder())
    with pytest.raises(BacktestError, match="no fetched price data for any symbol"):
        run(strat, _OnlyWrongProvider(), START, END)


def test_empty_declared_universe_fails_closed_if_provider_returns_data() -> None:
    """Empty declared universe + a (contract-violating) provider that returns data for an empty
    request → fail closed (operating universe is empty). #208 reversed the prior 'show full panel'
    behavior; this is a no-op for compliant providers (empty request → no bars → earlier guard)."""

    class _DataForEmptyRequestProvider:
        def get_bars(self, symbols, start, end, timeframe):  # noqa: ANN001
            # Ignore the (empty) request and return data anyway — a double contract violation.
            return SyntheticProvider(seed=0).get_bars(["AAA", "BBB"], start, end, timeframe)

    cfg = StrategyConfig(
        name="obs_empty", universe=[],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
        params={}, construction="passthrough",
    )
    strat = LoadedStrategy(config=cfg, signal_fn=_ViewRecorder(), construct_fn=_passthrough)
    with pytest.raises(BacktestError, match="no fetched price data for any symbol"):
        run(strat, _DataForEmptyRequestProvider(), START, END)
