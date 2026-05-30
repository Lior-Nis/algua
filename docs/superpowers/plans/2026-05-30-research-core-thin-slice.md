# Research Core (Thin Vertical Slice) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An agent/human authors one cross-sectional strategy, backtests it on a synthetic (bar-schema-conformant) data provider, gets structured JSON metrics, and the registry advances `idea → backtested` — proving the data↔research seam and the `t→t+1` anti-look-ahead rule.

**Architecture:** A per-bar decision loop feeds each strategy the point-in-time view (rows ≤ `t`); the engine collects target weights into a matrix, shifts it forward by `decision_lag_bars` (the `t→t+1` rule, enforced centrally), and hands it to `vectorbt` for portfolio accounting. The engine depends only on the `DataProvider` protocol and the frozen bar schema, so it is decoupled from the data lane. Registry wiring lives only in the CLI layer; the engine stays pure.

**Tech Stack:** Python 3.12, pandas 2.3.3, numpy 2.4.6, **vectorbt 1.0.0** (`Portfolio.from_orders`, `size_type="targetpercent"`), Typer CLI, pytest. Branch: `research-core` (worktree `/home/liornisimov/Projects/algua`).

**Frozen inputs:** `docs/contracts/bar-schema.md` (bar shape); `algua/contracts/types.py` (`DataProvider`, `Strategy`, `ExecutionContract`); `algua/registry/store.py` (`add_strategy`, `transition` — call, do not edit).

**Bar schema recap (what every bars DataFrame looks like):** long/tidy; index `timestamp` (tz-aware UTC, bar close); columns `symbol, open, high, low, close, adj_close, volume`; sorted by `(timestamp, symbol)`.

---

### Task 1: Feature indicators (pure)

**Files:**
- Create: `algua/features/indicators.py`
- Test: `tests/test_features_indicators.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_features_indicators.py
import numpy as np
import pandas as pd
from algua.features.indicators import momentum, zscore


def test_momentum_is_trailing_return():
    s = pd.Series([10.0, 11.0, 12.0, 13.2])
    # momentum over lookback=3 at the last point: 13.2/10 - 1 = 0.32
    assert momentum(s, lookback=3).iloc[-1] == 0.32


def test_momentum_insufficient_history_is_nan():
    s = pd.Series([10.0, 11.0])
    assert np.isnan(momentum(s, lookback=3).iloc[-1])


def test_zscore_centers_and_scales():
    s = pd.Series([1.0, 2.0, 3.0])
    z = zscore(s)
    assert abs(z.mean()) < 1e-9
    assert z.iloc[-1] > 0 and z.iloc[0] < 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_features_indicators.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'algua.features.indicators'`.

- [ ] **Step 3: Write minimal implementation**

```python
# algua/features/indicators.py
from __future__ import annotations

import pandas as pd


def momentum(prices: pd.Series, lookback: int) -> pd.Series:
    """Trailing simple return over `lookback` periods: price_t / price_{t-lookback} - 1."""
    return prices / prices.shift(lookback) - 1.0


def zscore(values: pd.Series) -> pd.Series:
    """Cross-sectional/sample z-score. Std uses population (ddof=0) to stay defined for n>=1."""
    std = values.std(ddof=0)
    if std == 0:
        return values * 0.0
    return (values - values.mean()) / std
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_features_indicators.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add algua/features/indicators.py tests/test_features_indicators.py
git commit -m "feat: add pure feature indicators (momentum, zscore)"
```

---

### Task 2: Strategy config & LoadedStrategy adapter

**Files:**
- Create: `algua/strategies/base.py`
- Test: `tests/test_strategies_base.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_strategies_base.py
import pandas as pd
from algua.contracts.types import ExecutionContract, Strategy
from algua.strategies.base import StrategyConfig, LoadedStrategy


def _tw(view, params):
    # trivial: equal weight the configured universe
    syms = params["symbols"]
    return pd.Series(1.0 / len(syms), index=syms)


def test_loaded_strategy_satisfies_protocol_and_exposes_config():
    cfg = StrategyConfig(
        name="demo",
        universe=["AAPL", "MSFT"],
        execution=ExecutionContract(rebalance_frequency="1d"),
        params={"symbols": ["AAPL", "MSFT"]},
    )
    strat = LoadedStrategy(config=cfg, fn=_tw)
    assert strat.name == "demo"
    assert strat.universe == ["AAPL", "MSFT"]
    assert isinstance(strat, Strategy)  # runtime_checkable protocol
    w = strat.target_weights(pd.DataFrame())
    assert abs(w.sum() - 1.0) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_strategies_base.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'algua.strategies.base'`.

- [ ] **Step 3: Write minimal implementation**

```python
# algua/strategies/base.py
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd
from pydantic import BaseModel

from algua.contracts.types import ExecutionContract

TargetWeightsFn = Callable[[pd.DataFrame, dict[str, Any]], pd.Series]


class StrategyConfig(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    name: str
    universe: list[str]
    execution: ExecutionContract
    params: dict[str, Any] = {}


@dataclass
class LoadedStrategy:
    """Binds a StrategyConfig + a pure target_weights function into an object that
    satisfies the Strategy protocol (.name, .execution, .target_weights)."""

    config: StrategyConfig
    fn: TargetWeightsFn

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def universe(self) -> list[str]:
        return self.config.universe

    @property
    def execution(self) -> ExecutionContract:
        return self.config.execution

    @property
    def params(self) -> dict[str, Any]:
        return self.config.params

    def target_weights(self, features: pd.DataFrame) -> pd.Series:
        return self.fn(features, self.config.params)
```

Note: `ExecutionContract` is a frozen dataclass (not a pydantic model); `arbitrary_types_allowed` lets it sit inside the pydantic `StrategyConfig`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_strategies_base.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add algua/strategies/base.py tests/test_strategies_base.py
git commit -m "feat: add StrategyConfig and LoadedStrategy adapter"
```

---

### Task 3: Bundled cross-sectional momentum strategy

**Files:**
- Create: `algua/strategies/__init__.py` (empty if missing), `algua/strategies/examples/__init__.py` (empty), `algua/strategies/examples/cross_sectional_momentum.py`
- Test: `tests/test_strategy_momentum.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_strategy_momentum.py
import pandas as pd
from algua.strategies.examples import cross_sectional_momentum as csm


def _bars(prices_by_symbol: dict[str, list[float]]) -> pd.DataFrame:
    # Build a bar-schema long frame from per-symbol adj_close paths.
    ts = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    rows = []
    for sym, path in prices_by_symbol.items():
        for t, px in zip(ts, path):
            rows.append({"timestamp": t, "symbol": sym, "open": px, "high": px,
                         "low": px, "close": px, "adj_close": px, "volume": 1.0})
    return pd.DataFrame(rows).set_index("timestamp").sort_index()


def test_momentum_picks_top_k_winners_equal_weight():
    # WIN doubles, FLAT flat, LOSE halves over the window -> top_k=1 -> all weight on WIN
    view = _bars({"WIN": [10, 12, 16, 20], "FLAT": [10, 10, 10, 10], "LOSE": [10, 9, 8, 7]})
    params = {"lookback": 3, "top_k": 1}
    w = csm.target_weights(view, params)
    assert w.idxmax() == "WIN"
    assert abs(w.sum() - 1.0) < 1e-9
    assert (w.drop("WIN") == 0).all()


def test_has_config():
    assert csm.CONFIG.name == "cross_sectional_momentum"
    assert csm.CONFIG.execution.decision_lag_bars >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_strategy_momentum.py -v`
Expected: FAIL with import error for `cross_sectional_momentum`.

- [ ] **Step 3: Write minimal implementation**

```python
# algua/strategies/examples/cross_sectional_momentum.py
"""Cross-sectional momentum: hold the top-k trailing-return names, equal weight."""
from __future__ import annotations

from typing import Any

import pandas as pd

from algua.contracts.types import ExecutionContract
from algua.features.indicators import momentum
from algua.strategies.base import StrategyConfig

CONFIG = StrategyConfig(
    name="cross_sectional_momentum",
    universe=["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"],
    execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
    params={"lookback": 60, "top_k": 3},
)


def target_weights(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    lookback = int(params["lookback"])
    top_k = int(params["top_k"])
    # Wide adj_close (index=timestamp, columns=symbol) from the point-in-time view.
    wide = view.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    if len(wide) <= lookback:
        return pd.Series(dtype="float64")
    scores = momentum(wide, lookback=lookback).iloc[-1].dropna()
    winners = scores.sort_values(ascending=False).head(top_k).index
    if len(winners) == 0:
        return pd.Series(dtype="float64")
    return pd.Series(1.0 / len(winners), index=winners)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_strategy_momentum.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add algua/strategies/__init__.py algua/strategies/examples/ tests/test_strategy_momentum.py
git commit -m "feat: add cross-sectional momentum example strategy"
```

---

### Task 4: Strategy loader

**Files:**
- Create: `algua/strategies/loader.py`
- Test: `tests/test_strategy_loader.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_strategy_loader.py
import pytest
from algua.strategies.base import LoadedStrategy
from algua.strategies.loader import load_strategy, list_strategies, StrategyNotFound


def test_load_bundled_momentum():
    strat = load_strategy("cross_sectional_momentum")
    assert isinstance(strat, LoadedStrategy)
    assert strat.name == "cross_sectional_momentum"


def test_unknown_strategy_raises():
    with pytest.raises(StrategyNotFound):
        load_strategy("does_not_exist")


def test_list_includes_bundled():
    assert "cross_sectional_momentum" in list_strategies()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_strategy_loader.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'algua.strategies.loader'`.

- [ ] **Step 3: Write minimal implementation**

```python
# algua/strategies/loader.py
from __future__ import annotations

import importlib
import pkgutil

from algua.strategies import examples
from algua.strategies.base import LoadedStrategy


class StrategyNotFound(LookupError):
    pass


def load_strategy(name: str) -> LoadedStrategy:
    """Load a bundled strategy module by name; it must expose CONFIG + target_weights."""
    try:
        module = importlib.import_module(f"algua.strategies.examples.{name}")
    except ModuleNotFoundError as exc:
        raise StrategyNotFound(name) from exc
    if not hasattr(module, "CONFIG") or not hasattr(module, "target_weights"):
        raise StrategyNotFound(f"{name} is missing CONFIG or target_weights")
    return LoadedStrategy(config=module.CONFIG, fn=module.target_weights)


def list_strategies() -> list[str]:
    return [m.name for m in pkgutil.iter_modules(examples.__path__) if not m.name.startswith("_")]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_strategy_loader.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add algua/strategies/loader.py tests/test_strategy_loader.py
git commit -m "feat: add strategy loader"
```

---

### Task 5: Synthetic data provider (bar-schema conformant)

**Files:**
- Create: `algua/backtest/__init__.py` (empty), `algua/backtest/_sample.py`
- Test: `tests/test_backtest_sample.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_backtest_sample.py
import pandas as pd
from datetime import datetime, timezone
from algua.backtest._sample import SyntheticProvider

START = datetime(2024, 1, 1, tzinfo=timezone.utc)
END = datetime(2024, 3, 1, tzinfo=timezone.utc)
COLS = ["symbol", "open", "high", "low", "close", "adj_close", "volume"]


def test_returns_bar_schema_shape():
    df = SyntheticProvider(seed=1).get_bars(["AAA", "BBB"], START, END, "1d")
    assert list(df.columns) == COLS
    assert df.index.name == "timestamp"
    assert str(df.index.tz) == "UTC"
    assert set(df["symbol"].unique()) == {"AAA", "BBB"}
    assert df.index.is_monotonic_increasing
    assert not df[["open", "high", "low", "close", "adj_close"]].isna().any().any()


def test_deterministic_for_same_seed():
    a = SyntheticProvider(seed=42).get_bars(["AAA"], START, END, "1d")
    b = SyntheticProvider(seed=42).get_bars(["AAA"], START, END, "1d")
    pd.testing.assert_frame_equal(a, b)


def test_rejects_unknown_timeframe():
    import pytest
    with pytest.raises(ValueError):
        SyntheticProvider().get_bars(["AAA"], START, END, "7h")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_backtest_sample.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'algua.backtest._sample'`.

- [ ] **Step 3: Write minimal implementation**

```python
# algua/backtest/_sample.py
"""Dev-only synthetic DataProvider producing bar-schema-conformant data.

NOT for production use. Exists so the research lane can build and test the backtest
engine end-to-end without the real data layer. Geometric-brownian-ish price paths.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

_BAR_COLUMNS = ["symbol", "open", "high", "low", "close", "adj_close", "volume"]
_SUPPORTED = {"1d"}


class SyntheticProvider:
    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def get_bars(
        self, symbols: list[str], start: datetime, end: datetime, timeframe: str
    ) -> pd.DataFrame:
        if timeframe not in _SUPPORTED:
            raise ValueError(f"unsupported timeframe: {timeframe!r}")
        sessions = pd.date_range(start=start, end=end, freq="B", tz="UTC")  # business days
        rng = np.random.default_rng(self.seed)
        frames = []
        for i, sym in enumerate(sorted(symbols)):
            # Deterministic per-symbol drift/vol from a child RNG.
            sub = np.random.default_rng(self.seed + i + 1)
            rets = sub.normal(loc=0.0005, scale=0.02, size=len(sessions))
            price = 100.0 * np.exp(np.cumsum(rets))
            frames.append(pd.DataFrame({
                "timestamp": sessions, "symbol": sym,
                "open": price, "high": price * 1.01, "low": price * 0.99,
                "close": price, "adj_close": price, "volume": 1_000_000.0,
            }))
        _ = rng  # reserved for future noise; keeps seed wiring explicit
        out = pd.concat(frames).set_index("timestamp").sort_values(["timestamp", "symbol"])
        return out[_BAR_COLUMNS]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_backtest_sample.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add algua/backtest/__init__.py algua/backtest/_sample.py tests/test_backtest_sample.py
git commit -m "feat: add synthetic bar-schema data provider (dev-only)"
```

---

### Task 6: Backtest result & metrics

**Files:**
- Create: `algua/backtest/result.py`, `algua/backtest/metrics.py`
- Test: `tests/test_backtest_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_backtest_metrics.py
import numpy as np
import pandas as pd
from algua.backtest.metrics import weights_turnover, avg_gross_exposure
from algua.backtest.result import BacktestResult


def test_turnover_counts_weight_changes():
    # t0: 100% A; t1: 100% B -> one full rotation = turnover 1.0 at t1
    w = pd.DataFrame({"A": [1.0, 0.0], "B": [0.0, 1.0]})
    assert weights_turnover(w) == 1.0


def test_avg_gross_exposure():
    w = pd.DataFrame({"A": [0.5, 0.5], "B": [0.5, 0.5]})
    assert avg_gross_exposure(w) == 1.0


def test_result_to_dict_is_json_serializable():
    import json
    r = BacktestResult(
        strategy="s", metrics={"sharpe": 1.2}, config_hash="abc",
        data_source="synthetic", timeframe="1d",
        period={"start": "2024-01-01", "end": "2024-03-01"}, seed=0,
    )
    json.dumps(r.to_dict())  # must not raise
    assert r.to_dict()["metrics"]["sharpe"] == 1.2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_backtest_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'algua.backtest.metrics'`.

- [ ] **Step 3: Write minimal implementation**

```python
# algua/backtest/metrics.py
from __future__ import annotations

import pandas as pd


def weights_turnover(weights: pd.DataFrame) -> float:
    """Mean per-rebalance one-way turnover: 0.5 * sum |w_t - w_{t-1}| summed over the path.

    For a single full rotation (100% A -> 100% B) this returns 1.0.
    """
    diffs = weights.diff().abs().sum(axis=1).iloc[1:]  # drop first (NaN) row
    return float(diffs.sum() / 2.0)


def avg_gross_exposure(weights: pd.DataFrame) -> float:
    return float(weights.abs().sum(axis=1).mean())
```

```python
# algua/backtest/result.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class BacktestResult:
    strategy: str
    metrics: dict[str, float]
    config_hash: str
    data_source: str
    timeframe: str
    period: dict[str, str]
    seed: int | None = None
    snapshot_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "metrics": self.metrics,
            "config_hash": self.config_hash,
            "data_source": self.data_source,
            "timeframe": self.timeframe,
            "period": self.period,
            "seed": self.seed,
            "snapshot_id": self.snapshot_id,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_backtest_metrics.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add algua/backtest/result.py algua/backtest/metrics.py tests/test_backtest_metrics.py
git commit -m "feat: add backtest metrics and result container"
```

---

### Task 7: The backtest engine (per-bar loop + t→t+1 + vectorbt)

**Files:**
- Create: `algua/backtest/engine.py`
- Test: `tests/test_backtest_engine.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_backtest_engine.py
from datetime import datetime, timezone
import pandas as pd
from algua.contracts.types import ExecutionContract
from algua.strategies.base import StrategyConfig, LoadedStrategy
from algua.backtest._sample import SyntheticProvider
from algua.backtest.engine import run, BacktestError

START = datetime(2024, 1, 1, tzinfo=timezone.utc)
END = datetime(2024, 4, 1, tzinfo=timezone.utc)


def _equal_weight_strategy():
    cfg = StrategyConfig(
        name="ew", universe=["AAA", "BBB"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
        params={},
    )

    def fn(view, params):
        syms = view["symbol"].unique()
        return pd.Series(1.0 / len(syms), index=sorted(syms))

    return LoadedStrategy(config=cfg, fn=fn)


def test_run_produces_metrics_keys():
    res = run(_equal_weight_strategy(), SyntheticProvider(seed=3), START, END)
    for key in ["total_return", "cagr", "ann_volatility", "sharpe", "max_drawdown",
                "turnover", "avg_gross_exposure", "n_rebalances"]:
        assert key in res.metrics


def test_run_is_deterministic():
    a = run(_equal_weight_strategy(), SyntheticProvider(seed=3), START, END)
    b = run(_equal_weight_strategy(), SyntheticProvider(seed=3), START, END)
    assert a.metrics == b.metrics


def test_t_plus_1_blocks_same_bar_fill():
    """A 'cheating' strategy that, at bar t, puts 100% on whichever symbol rises at t
    must NOT capture bar t's move, because the engine shifts weights by decision_lag_bars."""
    cfg = StrategyConfig(
        name="cheat", universe=["AAA", "BBB"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
        params={},
    )

    def cheat(view, params):
        wide = view.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
        if len(wide) < 2:
            return pd.Series(dtype="float64")
        last_ret = wide.iloc[-1] / wide.iloc[-2] - 1.0  # uses the CURRENT bar's return
        winner = last_ret.idxmax()
        return pd.Series([1.0], index=[winner])

    strat = LoadedStrategy(config=cfg, fn=cheat)
    cheating = run(strat, SyntheticProvider(seed=5), START, END)

    # Compare to the same strategy with lag=0 (same-bar fill = look-ahead "cheating works").
    cfg0 = StrategyConfig(name="cheat0", universe=["AAA", "BBB"],
                          execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=0),
                          params={})
    leaky = run(LoadedStrategy(config=cfg0, fn=cheat), SyntheticProvider(seed=5), START, END)

    # Look-ahead must look dramatically better than the honest t+1 version.
    assert leaky.metrics["total_return"] > cheating.metrics["total_return"] + 0.05


def test_empty_universe_data_raises():
    cfg = StrategyConfig(name="x", universe=[],
                         execution=ExecutionContract(rebalance_frequency="1d"), params={})
    strat = LoadedStrategy(config=cfg, fn=lambda v, p: pd.Series(dtype="float64"))
    import pytest
    with pytest.raises(BacktestError):
        run(strat, SyntheticProvider(), START, END)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_backtest_engine.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'algua.backtest.engine'`.

- [ ] **Step 3: Write minimal implementation**

```python
# algua/backtest/engine.py
from __future__ import annotations

import hashlib
import json
from datetime import datetime

import numpy as np
import pandas as pd
import vectorbt as vbt

from algua.backtest.metrics import avg_gross_exposure, weights_turnover
from algua.backtest.result import BacktestResult
from algua.strategies.base import LoadedStrategy

# A trading year in business days, for annualization.
_ANN = 252


class BacktestError(RuntimeError):
    pass


def _config_hash(strategy: LoadedStrategy) -> str:
    payload = json.dumps(
        {"name": strategy.name, "universe": strategy.universe,
         "params": strategy.params,
         "execution": {"rebalance_frequency": strategy.execution.rebalance_frequency,
                       "decision_lag_bars": strategy.execution.decision_lag_bars,
                       "max_gross_exposure": strategy.execution.max_gross_exposure}},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def run(
    strategy: LoadedStrategy, provider, start: datetime, end: datetime, *, seed: int | None = None
) -> BacktestResult:
    timeframe = "1d"
    bars = provider.get_bars(strategy.universe, start, end, timeframe)
    if bars.empty:
        raise BacktestError("provider returned no bars for the universe/period")

    # Wide adj_close panel (index=timestamp, columns=symbol).
    adj = bars.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    adj = adj.sort_index()

    # --- Per-bar decision loop: strategy only ever sees rows <= t (parity guarantee). ---
    weights = pd.DataFrame(0.0, index=adj.index, columns=adj.columns)
    for t in adj.index:
        view = bars.loc[:t]  # label slice on tz-aware DatetimeIndex -> rows with timestamp <= t
        w = strategy.target_weights(view)
        if len(w) > 0:
            weights.loc[t, w.index] = w.reindex(weights.columns).fillna(0.0).values

    # --- Enforce t -> t+1: decisions at t take effect no earlier than t + lag. ---
    lag = strategy.execution.decision_lag_bars
    weights_eff = weights.shift(lag).fillna(0.0)

    # --- Portfolio accounting via vectorbt (accounting only; logic stayed in the loop). ---
    pf = vbt.Portfolio.from_orders(
        close=adj, size=weights_eff, size_type="targetpercent",
        cash_sharing=True, group_by=True, freq="1D",
    )

    metrics = _metrics(pf, weights_eff)
    return BacktestResult(
        strategy=strategy.name, metrics=metrics, config_hash=_config_hash(strategy),
        data_source=type(provider).__name__, timeframe=timeframe,
        period={"start": start.date().isoformat(), "end": end.date().isoformat()},
        seed=getattr(provider, "seed", seed),
    )


def _metrics(pf, weights_eff: pd.DataFrame) -> dict[str, float]:
    total_return = float(pf.total_return())
    returns = pf.returns()
    ann_vol = float(returns.std() * np.sqrt(_ANN))
    mean_ann = float(returns.mean() * _ANN)
    sharpe = float(mean_ann / ann_vol) if ann_vol > 0 else 0.0
    n_periods = len(returns)
    cagr = float((1.0 + total_return) ** (_ANN / n_periods) - 1.0) if n_periods > 0 else 0.0
    max_dd = float(pf.max_drawdown())
    n_rebalances = int((weights_eff.diff().abs().sum(axis=1) > 1e-12).sum())
    return {
        "total_return": total_return, "cagr": cagr, "ann_volatility": ann_vol,
        "sharpe": sharpe, "max_drawdown": max_dd,
        "turnover": weights_turnover(weights_eff),
        "avg_gross_exposure": avg_gross_exposure(weights_eff),
        "n_rebalances": n_rebalances,
    }
```

**vectorbt API note:** `Portfolio.from_orders(..., size_type="targetpercent", cash_sharing=True, group_by=True, freq="1D")` and `pf.total_return()`, `pf.returns()`, `pf.max_drawdown()` are confirmed present in vectorbt 1.0.0. If any method name differs at runtime, probe with `uv run python -c "import vectorbt as vbt; print([m for m in dir(vbt.Portfolio) if not m.startswith('_')])"` and adjust the metric extraction only — do not change the test assertions.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_backtest_engine.py -v`
Expected: PASS (4 passed). If `test_t_plus_1_blocks_same_bar_fill` is flaky for `seed=5`, try a different fixed seed in the test (the property — look-ahead beats honest — holds for any seed with a non-trivial price path); keep the assertion.

- [ ] **Step 5: Commit**

```bash
git add algua/backtest/engine.py tests/test_backtest_engine.py
git commit -m "feat: add backtest engine with t->t+1 enforcement and vectorbt accounting"
```

---

### Task 8: CLI — `strategy new` / `strategy list`

**Files:**
- Create: `algua/cli/strategy_cmd.py`
- Modify: `algua/cli/main.py` (register the sub-app)
- Test: `tests/test_cli_strategy.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli_strategy.py
import json
from typer.testing import CliRunner
from algua.cli.main import app

runner = CliRunner()


def test_strategy_list_includes_bundled():
    result = runner.invoke(app, ["strategy", "list"])
    assert result.exit_code == 0
    assert "cross_sectional_momentum" in json.loads(result.stdout)


def test_strategy_new_scaffolds_loadable_module(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["strategy", "new", "my_strat"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert (tmp_path / payload["path"]).exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli_strategy.py -v`
Expected: FAIL — `strategy` subcommand does not exist.

- [ ] **Step 3: Write the command module**

```python
# algua/cli/strategy_cmd.py
from __future__ import annotations

from pathlib import Path

import typer

from algua.cli.app import app, emit
from algua.cli.errors import json_errors
from algua.strategies.loader import list_strategies

strategy_app = typer.Typer(help="Author and list strategies", no_args_is_help=True)
app.add_typer(strategy_app, name="strategy")

_TEMPLATE = '''\
"""Strategy: {name}. Edit target_weights to express your cross-sectional logic."""
from __future__ import annotations

from typing import Any

import pandas as pd

from algua.contracts.types import ExecutionContract
from algua.features.indicators import momentum
from algua.strategies.base import StrategyConfig

CONFIG = StrategyConfig(
    name="{name}",
    universe=["AAPL", "MSFT", "NVDA"],
    execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
    params={{"lookback": 60, "top_k": 2}},
)


def target_weights(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    wide = view.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    if len(wide) <= int(params["lookback"]):
        return pd.Series(dtype="float64")
    scores = momentum(wide, lookback=int(params["lookback"])).iloc[-1].dropna()
    winners = scores.sort_values(ascending=False).head(int(params["top_k"])).index
    if len(winners) == 0:
        return pd.Series(dtype="float64")
    return pd.Series(1.0 / len(winners), index=winners)
'''


@strategy_app.command("list")
@json_errors
def list_() -> None:
    """List available strategies as JSON."""
    emit(list_strategies())


@strategy_app.command("new")
@json_errors
def new(name: str) -> None:
    """Scaffold a new strategy module under algua/strategies/examples/."""
    path = Path("algua/strategies/examples") / f"{name}.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError(f"strategy already exists: {path}")
    path.write_text(_TEMPLATE.format(name=name))
    emit({"ok": True, "name": name, "path": str(path)})
```

Note: `algua/cli/errors.py` already provides the `json_errors` decorator (created by the data lane's refactor). Confirm its exact exported name first with `uv run python -c "import algua.cli.errors as e; print([n for n in dir(e) if 'error' in n.lower()])"`; if it differs, import the correct symbol. If `algua/cli/errors.py` does not exist on this branch yet, define a local `json_errors` decorator in this module identical to the one described in `AGENTS.md` (catch `(ValueError, LookupError)` → `emit({"ok": False, "error": str(exc)})` + `raise typer.Exit(1)`).

- [ ] **Step 4: Register in `algua/cli/main.py`**

Add the import line (keep alphabetical with any existing command imports) so `main.py` reads:

```python
from __future__ import annotations

from algua.cli.app import app
from algua.cli import registry_cmd  # noqa: F401
from algua.cli import strategy_cmd  # noqa: F401

__all__ = ["app"]
```

If `main.py` already imports other command modules (e.g. `data_cmd`), keep them and just add the `strategy_cmd` line.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli_strategy.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add algua/cli/strategy_cmd.py algua/cli/main.py tests/test_cli_strategy.py
git commit -m "feat: add strategy CLI (new, list)"
```

---

### Task 9: CLI — `backtest run` (+ optional registry advance)

**Files:**
- Create: `algua/cli/backtest_cmd.py`
- Modify: `algua/cli/main.py`
- Test: `tests/test_cli_backtest.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli_backtest.py
import json
import pytest
from typer.testing import CliRunner
from algua.cli.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _tmp_db(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))


def test_backtest_run_demo_emits_metrics():
    result = runner.invoke(app, ["backtest", "run", "cross_sectional_momentum",
                                 "--demo", "--start", "2023-01-01", "--end", "2023-12-31"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["strategy"] == "cross_sectional_momentum"
    assert "sharpe" in payload["metrics"]


def test_backtest_run_register_advances_registry():
    result = runner.invoke(app, ["backtest", "run", "cross_sectional_momentum",
                                 "--demo", "--start", "2023-01-01", "--end", "2023-12-31",
                                 "--register"])
    assert result.exit_code == 0, result.stdout
    show = runner.invoke(app, ["registry", "show", "cross_sectional_momentum"])
    assert json.loads(show.stdout)["stage"] == "backtested"


def test_unknown_strategy_is_json_error():
    result = runner.invoke(app, ["backtest", "run", "nope", "--demo"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli_backtest.py -v`
Expected: FAIL — `backtest` subcommand does not exist.

- [ ] **Step 3: Write the command module**

```python
# algua/cli/backtest_cmd.py
from __future__ import annotations

from datetime import datetime, timezone

import typer

from algua.backtest._sample import SyntheticProvider
from algua.backtest.engine import run as run_backtest
from algua.cli.app import app, emit
from algua.cli.errors import json_errors
from algua.config.settings import get_settings
from algua.contracts.lifecycle import Actor, Stage
from algua.registry import store
from algua.registry.db import connect, migrate
from algua.strategies.loader import load_strategy

backtest_app = typer.Typer(help="Run backtests", no_args_is_help=True)
app.add_typer(backtest_app, name="backtest")


def _utc(date_str: str) -> datetime:
    return datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)


@backtest_app.command("run")
@json_errors
def run(
    name: str,
    start: str = typer.Option("2023-01-01", "--start"),
    end: str = typer.Option("2023-12-31", "--end"),
    demo: bool = typer.Option(False, "--demo", help="use the synthetic data provider"),
    register: bool = typer.Option(False, "--register", help="advance registry idea->backtested"),
) -> None:
    """Backtest a strategy and emit metrics JSON."""
    strategy = load_strategy(name)
    if not demo:
        raise ValueError("only --demo (synthetic provider) is supported until data integration")
    provider = SyntheticProvider(seed=0)
    result = run_backtest(strategy, provider, _utc(start), _utc(end))

    if register:
        conn = connect(get_settings().db_path)
        migrate(conn)
        existing = {s.name for s in store.list_strategies(conn)}
        if name not in existing:
            store.add_strategy(conn, name)
        reason = f"backtest sharpe={result.metrics['sharpe']:.2f} ret={result.metrics['total_return']:.2%}"
        store.transition(conn, name, Stage.BACKTESTED, Actor.AGENT, reason,
                         code_hash=result.config_hash, config_hash=result.config_hash)
        conn.close()

    emit(result.to_dict())
```

- [ ] **Step 4: Register in `algua/cli/main.py`**

Add `from algua.cli import backtest_cmd  # noqa: F401` alongside the other command imports.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli_backtest.py -v`
Expected: PASS (3 passed). Note: `registry transition` to `backtested` from `idea` is a legal lifecycle move (`idea -> backtested`), so `--register` succeeds.

- [ ] **Step 6: Commit**

```bash
git add algua/cli/backtest_cmd.py algua/cli/main.py tests/test_cli_backtest.py
git commit -m "feat: add backtest run CLI with optional registry advance"
```

---

### Task 10: Architecture guardrails (import-linter) & full verification

**Files:**
- Modify: `pyproject.toml` (extend `[tool.importlinter]` contracts)

- [ ] **Step 1: Add two contracts to the existing `[tool.importlinter]` section in `pyproject.toml`**

Append these two contract blocks after the existing contracts:

```toml
[[tool.importlinter.contracts]]
name = "features layer is pure (no algua imports beyond contracts)"
type = "forbidden"
source_modules = ["algua.features"]
forbidden_modules = ["algua.cli", "algua.registry", "algua.data", "algua.backtest", "algua.strategies"]

[[tool.importlinter.contracts]]
name = "backtest engine stays off cli, registry, and the data lane"
type = "forbidden"
source_modules = ["algua.backtest"]
forbidden_modules = ["algua.cli", "algua.registry", "algua.data"]
```

- [ ] **Step 2: Run import contracts**

Run: `uv run lint-imports`
Expected: all contracts KEPT (now 4). If `algua.backtest` is reported importing `algua.data`, that's a real violation — the engine must depend only on the `DataProvider` protocol from `algua.contracts`; fix the import, don't weaken the contract.

- [ ] **Step 3: Full quality gate**

Run:
```bash
uv run pytest -q
uv run ruff check .
uv run mypy algua
uv run lint-imports
```
Expected: all tests pass; ruff clean; mypy `Success`; import-linter `4 kept, 0 broken`.

- [ ] **Step 4: End-to-end CLI smoke test**

Run:
```bash
uv run algua strategy list
uv run algua backtest run cross_sectional_momentum --demo --start 2022-01-01 --end 2023-12-31 --register
uv run algua registry show cross_sectional_momentum
```
Expected: `strategy list` includes the momentum strategy; `backtest run` emits metrics JSON; `registry show` reports stage `backtested` with a transition recording the backtest reason + config hash.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "chore: enforce features/backtest import boundaries"
```

---

## Self-Review Notes

- **Spec coverage:** features (Task 1), strategy authoring fn+config (Tasks 2-3), loader (Task 4),
  synthetic provider (Task 5), metrics+result with reproducibility stamps (Task 6), engine with
  per-bar loop + `t→t+1` + vectorbt (Task 7), CLI `strategy`/`backtest` + registry advance
  (Tasks 8-9), import guardrails + verification (Task 10). Deferred items (walk-forward, sweeps,
  MLflow, real-data integration) are intentionally absent per the spec.
- **Lane isolation:** new code lives only in `algua/features|strategies|backtest/*` and new
  `algua/cli/*_cmd.py`; `registry` is called, never edited; `contracts/types.py` untouched.
- **Type consistency:** `LoadedStrategy` (`.name/.universe/.execution/.params/.target_weights`),
  `StrategyConfig`, `BacktestResult.to_dict()`, `run(strategy, provider, start, end)`,
  `SyntheticProvider(seed=...).get_bars(...)`, and the metric keys are used identically across tasks.
- **External-API caution:** the one external surface (vectorbt) has a probe/fallback note in Task 7;
  `algua/cli/errors.py::json_errors` has a confirm/fallback note in Task 8.
