# Walk-Forward Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `algua backtest walk-forward <strategy>` — run a strategy once over the full history, then segment the return series into a reserved holdout + K equal windows and report per-segment metrics + cross-window stability.

**Architecture:** Refactor the engine to expose `_build_portfolio` (reused without behavior change), then a new `algua/backtest/walkforward.py` builds the portfolio once, slices `pf.returns()` by bar count (holdout tail + K windows), and computes return-based metrics + stability. A new CLI command emits the result as JSON. Not parameter optimization — the engine is already point-in-time, so this measures stability across sub-periods + reserves a holdout.

**Tech Stack:** Python 3.12, pandas 2.3.3, numpy, vectorbt 1.0.0, Typer, pytest.

**Key existing code (branch `research-walkforward`):**
- `algua/backtest/engine.py`: `run(strategy, provider, start, end, *, seed=None) -> BacktestResult`, `BacktestError`, `_config_hash(strategy)`, `_metrics(pf, weights_eff)`, module const `_SUPPORTED_CADENCES = {"1d"}`. `run` does: cadence check → `provider.get_bars` → empty check → pivot `adj_close` to wide `adj` → per-bar loop building `weights` (with gross-exposure check) → `weights_eff = weights.shift(lag).fillna(0.0)` → `vbt.Portfolio.from_orders(close=adj, size=weights_eff, size_type="targetpercent", cash_sharing=True, group_by=True, freq="1D")` → `_metrics` → `BacktestResult`.
- `algua/backtest/_sample.py::SyntheticProvider(seed=...)`.
- `algua/cli/backtest_cmd.py`: `run` command with `--demo`/`--snapshot` selection, `_utc(date_str)`, `@json_errors(ValueError, LookupError, BacktestError)`, imports `SyntheticProvider`, `StoreBackedProvider`, `DataStore`, `get_settings`, `load_strategy`.
- `algua/contracts/types.py::DataProvider`.

---

### Task 1: Refactor engine to expose `_build_portfolio` (no behavior change)

**Files:**
- Modify: `algua/backtest/engine.py`

- [ ] **Step 1: Extract `_build_portfolio` and have `run` call it**

Replace the body of `run` so the portfolio-construction logic moves into a new helper. The file's `run` currently inlines everything; change it to:

```python
def _build_portfolio(
    strategy: LoadedStrategy, provider: DataProvider, start: datetime, end: datetime
) -> tuple[vbt.Portfolio, pd.DataFrame]:
    """Fetch bars, run the per-bar decision loop (enforcing gross exposure), apply the t->t+1
    shift, and simulate. Returns (portfolio, effective-weights). Shared by run() and walk_forward()."""
    timeframe = "1d"
    cadence = strategy.execution.rebalance_frequency.lower()
    if cadence not in _SUPPORTED_CADENCES:
        raise BacktestError(
            f"rebalance_frequency {strategy.execution.rebalance_frequency!r} not supported; "
            f"this slice rebalances daily only ({sorted(_SUPPORTED_CADENCES)})"
        )
    try:
        bars = provider.get_bars(strategy.universe, start, end, timeframe)
    except Exception as exc:
        raise BacktestError(f"provider error: {exc}") from exc
    if bars.empty:
        raise BacktestError("provider returned no bars for the universe/period")

    adj = bars.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    adj = adj.sort_index()

    weights = pd.DataFrame(0.0, index=adj.index, columns=adj.columns)
    for t in adj.index:
        view = bars.loc[:t]
        w = strategy.target_weights(view)
        if len(w) > 0:
            row = w.reindex(weights.columns).fillna(0.0)
            gross = float(row.abs().sum())
            max_gross = strategy.execution.max_gross_exposure
            if gross > max_gross + 1e-9:
                raise BacktestError(
                    f"strategy '{strategy.name}' targeted gross exposure {gross:.4f} at {t} "
                    f"exceeding max_gross_exposure={max_gross}"
                )
            weights.loc[t] = row.values

    lag = strategy.execution.decision_lag_bars
    weights_eff = weights.shift(lag).fillna(0.0)
    pf = vbt.Portfolio.from_orders(
        close=adj,
        size=weights_eff,
        size_type="targetpercent",
        cash_sharing=True,
        group_by=True,
        freq="1D",
    )
    return pf, weights_eff


def run(
    strategy: LoadedStrategy,
    provider: DataProvider,
    start: datetime,
    end: datetime,
    *,
    seed: int | None = None,
) -> BacktestResult:
    pf, weights_eff = _build_portfolio(strategy, provider, start, end)
    metrics = _metrics(pf, weights_eff)
    return BacktestResult(
        strategy=strategy.name,
        metrics=metrics,
        config_hash=_config_hash(strategy),
        data_source=type(provider).__name__,
        timeframe="1d",
        period={"start": start.date().isoformat(), "end": end.date().isoformat()},
        seed=getattr(provider, "seed", seed),
        snapshot_id=getattr(provider, "snapshot_id", None),
    )
```

Leave `_config_hash`, `_metrics`, `BacktestError`, and the imports/constants unchanged.

- [ ] **Step 2: Verify existing tests still pass (refactor safety net)**

Run: `uv run pytest tests/test_backtest_engine.py tests/test_cli_backtest.py -q`
Expected: all pass unchanged (this is a pure refactor — `run`'s output is identical).

- [ ] **Step 3: Gate + commit**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all green; 4 import contracts kept.
```bash
git add algua/backtest/engine.py
git commit -m "refactor: extract _build_portfolio from engine.run (no behavior change)"
```

---

### Task 2: `metrics_from_returns`

**Files:**
- Create: `algua/backtest/walkforward.py`
- Test: `tests/test_walkforward_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_walkforward_metrics.py
import numpy as np
import pandas as pd
from algua.backtest.walkforward import metrics_from_returns


def test_empty_returns_are_zero():
    m = metrics_from_returns(pd.Series(dtype="float64"))
    assert m == {"total_return": 0.0, "ann_return": 0.0, "ann_volatility": 0.0,
                 "sharpe": 0.0, "max_drawdown": 0.0}


def test_zero_volatility_returns_zero_sharpe():
    m = metrics_from_returns(pd.Series([0.0, 0.0, 0.0]))
    assert m["sharpe"] == 0.0
    assert m["ann_volatility"] == 0.0


def test_total_return_and_drawdown():
    # +10%, -50%, then flat: total = 1.1 * 0.5 - 1 = -0.45; max drawdown = -0.5
    m = metrics_from_returns(pd.Series([0.1, -0.5, 0.0]))
    assert abs(m["total_return"] - (-0.45)) < 1e-9
    assert abs(m["max_drawdown"] - (-0.5)) < 1e-9


def test_positive_series_has_positive_sharpe():
    m = metrics_from_returns(pd.Series([0.01, 0.02, 0.015, 0.005]))
    assert m["sharpe"] > 0
    assert m["ann_volatility"] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_walkforward_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'algua.backtest.walkforward'`.

- [ ] **Step 3: Write the implementation**

```python
# algua/backtest/walkforward.py
from __future__ import annotations

import numpy as np
import pandas as pd

_ANN = 252  # trading days/year


def metrics_from_returns(returns: pd.Series) -> dict[str, float]:
    """Return-based metrics for one segment. Safe on empty / zero-vol input (-> zeros)."""
    r = returns.dropna()
    if len(r) == 0:
        return {
            "total_return": 0.0, "ann_return": 0.0, "ann_volatility": 0.0,
            "sharpe": 0.0, "max_drawdown": 0.0,
        }
    total_return = float((1.0 + r).prod() - 1.0)
    ann_return = float(r.mean() * _ANN)
    ann_vol = float(r.std() * np.sqrt(_ANN))
    sharpe = float(ann_return / ann_vol) if ann_vol > 0 else 0.0
    equity = (1.0 + r).cumprod()
    max_drawdown = float((equity / equity.cummax() - 1.0).min())
    return {
        "total_return": total_return, "ann_return": ann_return, "ann_volatility": ann_vol,
        "sharpe": sharpe, "max_drawdown": max_drawdown,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_walkforward_metrics.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add algua/backtest/walkforward.py tests/test_walkforward_metrics.py
git commit -m "feat: add metrics_from_returns for walk-forward segments"
```

---

### Task 3: Segmentation `_segment_bounds`

**Files:**
- Modify: `algua/backtest/walkforward.py`
- Test: `tests/test_walkforward_segment.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_walkforward_segment.py
import pytest
from algua.backtest.engine import BacktestError
from algua.backtest.walkforward import _segment_bounds


def test_even_split():
    windows, holdout = _segment_bounds(100, windows=4, holdout_frac=0.2)
    assert holdout == (80, 100)
    assert windows == [(0, 20), (20, 40), (40, 60), (60, 80)]


def test_remainder_goes_to_last_window():
    windows, holdout = _segment_bounds(102, windows=4, holdout_frac=0.2)
    # holdout_n = int(102*0.2)=20 -> train=82, base=20, last window absorbs remainder
    assert holdout == (82, 102)
    assert windows[-1] == (60, 82)
    assert windows[0] == (0, 20)


def test_full_coverage_no_overlap():
    windows, holdout = _segment_bounds(97, windows=3, holdout_frac=0.25)
    covered = []
    for s, e in windows:
        covered.extend(range(s, e))
    covered.extend(range(holdout[0], holdout[1]))
    assert covered == list(range(97))  # contiguous, no gaps/overlap


def test_invalid_windows():
    with pytest.raises(ValueError):
        _segment_bounds(100, windows=1, holdout_frac=0.2)


def test_invalid_holdout_frac():
    with pytest.raises(ValueError):
        _segment_bounds(100, windows=4, holdout_frac=1.0)


def test_too_few_bars_raises_backtest_error():
    with pytest.raises(BacktestError):
        _segment_bounds(20, windows=4, holdout_frac=0.2)  # train=16, base=4 < _MIN_WINDOW_BARS(5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_walkforward_segment.py -v`
Expected: FAIL with `ImportError: cannot import name '_segment_bounds'`.

- [ ] **Step 3: Add to `algua/backtest/walkforward.py`**

Add the import and helper (place the import with the existing ones; `BacktestError` comes from the engine):

```python
from algua.backtest.engine import BacktestError

_MIN_WINDOW_BARS = 5


def _segment_bounds(
    n: int, windows: int, holdout_frac: float
) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    """Partition n bars (by index) into K equal windows + a final holdout, as half-open ranges.

    Holdout = the last int(n*holdout_frac) bars. The remaining bars split into `windows` equal
    pieces; any integer-division remainder goes to the LAST window.
    """
    if windows < 2:
        raise ValueError("windows must be >= 2")
    if not (0.0 < holdout_frac < 1.0):
        raise ValueError("holdout_frac must be in (0, 1)")
    holdout_n = int(n * holdout_frac)
    train_n = n - holdout_n
    base = train_n // windows
    if base < _MIN_WINDOW_BARS:
        raise BacktestError(
            f"not enough bars: {train_n} train bars / {windows} windows is "
            f"< {_MIN_WINDOW_BARS} bars/window; widen the period or lower --windows"
        )
    bounds: list[tuple[int, int]] = []
    s = 0
    for i in range(windows):
        e = train_n if i == windows - 1 else s + base
        bounds.append((s, e))
        s = e
    return bounds, (train_n, n)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_walkforward_segment.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add algua/backtest/walkforward.py tests/test_walkforward_segment.py
git commit -m "feat: add walk-forward bar-count segmentation"
```

---

### Task 4: `WalkForwardResult` + `walk_forward`

**Files:**
- Modify: `algua/backtest/walkforward.py`
- Test: `tests/test_walkforward.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_walkforward.py
from datetime import UTC, datetime

from algua.backtest._sample import SyntheticProvider
from algua.backtest.walkforward import WalkForwardResult, walk_forward
from algua.contracts.types import ExecutionContract
from algua.strategies.base import LoadedStrategy, StrategyConfig
import pandas as pd

START = datetime(2022, 1, 1, tzinfo=UTC)
END = datetime(2023, 12, 31, tzinfo=UTC)


def _equal_weight():
    cfg = StrategyConfig(
        name="ew", universe=["AAA", "BBB"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1), params={},
    )
    return LoadedStrategy(config=cfg, fn=lambda v, p: pd.Series(
        1.0 / len(v["symbol"].unique()), index=sorted(v["symbol"].unique())))


def test_walk_forward_shape_and_stamps():
    res = walk_forward(_equal_weight(), SyntheticProvider(seed=3), START, END,
                       windows=4, holdout_frac=0.2)
    assert isinstance(res, WalkForwardResult)
    d = res.to_dict()
    assert d["windows"] == 4
    assert len(d["window_metrics"]) == 4
    assert {"start", "end", "n_bars", "sharpe", "total_return"} <= set(d["holdout_metrics"])
    assert {"mean_sharpe", "std_sharpe", "min_sharpe", "pct_positive_windows"} == set(d["stability"])
    assert 0.0 <= d["stability"]["pct_positive_windows"] <= 1.0
    assert d["config_hash"] and d["data_source"] == "SyntheticProvider"


def test_walk_forward_is_deterministic():
    a = walk_forward(_equal_weight(), SyntheticProvider(seed=3), START, END)
    b = walk_forward(_equal_weight(), SyntheticProvider(seed=3), START, END)
    assert a.to_dict() == b.to_dict()


def test_windows_and_holdout_cover_all_bars():
    res = walk_forward(_equal_weight(), SyntheticProvider(seed=3), START, END,
                       windows=4, holdout_frac=0.2)
    total = sum(w["n_bars"] for w in res.window_metrics) + res.holdout_metrics["n_bars"]
    # equals the number of return bars (one per session in the period)
    assert total > 0
    assert res.holdout_metrics["n_bars"] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_walkforward.py -v`
Expected: FAIL with `ImportError: cannot import name 'WalkForwardResult'`.

- [ ] **Step 3: Add to `algua/backtest/walkforward.py`**

Add imports (`dataclass`, `datetime`, and the engine helpers) and the result + driver:

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from algua.backtest.engine import _build_portfolio, _config_hash
from algua.strategies.base import LoadedStrategy


@dataclass
class WalkForwardResult:
    strategy: str
    config_hash: str
    data_source: str
    snapshot_id: str | None
    period: dict[str, str]
    windows: int
    holdout_frac: float
    window_metrics: list[dict[str, Any]]
    holdout_metrics: dict[str, Any]
    stability: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "config_hash": self.config_hash,
            "data_source": self.data_source,
            "snapshot_id": self.snapshot_id,
            "period": self.period,
            "windows": self.windows,
            "holdout_frac": self.holdout_frac,
            "window_metrics": self.window_metrics,
            "holdout_metrics": self.holdout_metrics,
            "stability": self.stability,
        }


def _segment_record(returns: pd.Series, start_i: int, end_i: int) -> dict[str, Any]:
    seg = returns.iloc[start_i:end_i]
    rec: dict[str, Any] = {
        "start": str(seg.index[0].date()) if len(seg) else None,
        "end": str(seg.index[-1].date()) if len(seg) else None,
        "n_bars": int(len(seg)),
    }
    rec.update(metrics_from_returns(seg))
    return rec


def walk_forward(
    strategy: LoadedStrategy,
    provider,
    start: datetime,
    end: datetime,
    *,
    windows: int = 4,
    holdout_frac: float = 0.2,
) -> WalkForwardResult:
    """Run the strategy once, then segment its return series into K windows + a final holdout."""
    pf, _weights = _build_portfolio(strategy, provider, start, end)
    returns = pf.returns()
    bounds, holdout = _segment_bounds(len(returns), windows, holdout_frac)

    window_metrics = [
        {"index": i, **_segment_record(returns, s, e)} for i, (s, e) in enumerate(bounds)
    ]
    holdout_metrics = _segment_record(returns, holdout[0], holdout[1])

    sharpes = [w["sharpe"] for w in window_metrics]
    positive = sum(1 for w in window_metrics if w["total_return"] > 0)
    stability = {
        "mean_sharpe": float(np.mean(sharpes)),
        "std_sharpe": float(np.std(sharpes)),
        "min_sharpe": float(np.min(sharpes)),
        "pct_positive_windows": float(positive / len(window_metrics)),
    }

    return WalkForwardResult(
        strategy=strategy.name,
        config_hash=_config_hash(strategy),
        data_source=type(provider).__name__,
        snapshot_id=getattr(provider, "snapshot_id", None),
        period={"start": start.date().isoformat(), "end": end.date().isoformat()},
        windows=windows,
        holdout_frac=holdout_frac,
        window_metrics=window_metrics,
        holdout_metrics=holdout_metrics,
        stability=stability,
    )
```

Note: importing `_build_portfolio`/`_config_hash` from `algua.backtest.engine` is within the
`backtest` package (allowed by import-linter). `walkforward` must NOT import `algua.cli`,
`algua.registry`, or `algua.data`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_walkforward.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Gate + commit**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all green; 4 contracts kept.
```bash
git add algua/backtest/walkforward.py tests/test_walkforward.py
git commit -m "feat: add walk_forward driver and WalkForwardResult"
```

---

### Task 5: CLI `backtest walk-forward`

**Files:**
- Modify: `algua/cli/backtest_cmd.py`
- Test: `tests/test_cli_walkforward.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli_walkforward.py
import json
import pytest
from typer.testing import CliRunner
from algua.cli.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _tmp(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))


def test_walk_forward_demo_emits_result():
    result = runner.invoke(app, ["backtest", "walk-forward", "cross_sectional_momentum",
                                 "--demo", "--start", "2022-01-01", "--end", "2023-12-31",
                                 "--windows", "4", "--holdout-frac", "0.2"])
    assert result.exit_code == 0, result.stdout
    d = json.loads(result.stdout)
    assert len(d["window_metrics"]) == 4
    assert "mean_sharpe" in d["stability"]
    assert d["holdout_metrics"]["n_bars"] > 0


def test_walk_forward_requires_a_data_source():
    result = runner.invoke(app, ["backtest", "walk-forward", "cross_sectional_momentum"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False


def test_walk_forward_too_few_bars_is_json_error():
    result = runner.invoke(app, ["backtest", "walk-forward", "cross_sectional_momentum",
                                 "--demo", "--start", "2023-12-01", "--end", "2023-12-10",
                                 "--windows", "4"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli_walkforward.py -v`
Expected: FAIL — no `walk-forward` command.

- [ ] **Step 3: Edit `algua/cli/backtest_cmd.py`**

Add the import for the driver (with the other `algua.backtest` imports):

```python
from algua.backtest.walkforward import walk_forward
```

Extract the provider-selection logic into a helper (DRY — used by both commands). Add this near `_utc`:

```python
def _select_provider(demo: bool, snapshot: str | None):
    if demo and snapshot:
        raise ValueError("pass only one of --demo or --snapshot")
    if demo:
        return SyntheticProvider(seed=0)
    if snapshot:
        return StoreBackedProvider(DataStore(get_settings().data_dir), snapshot)
    raise ValueError("pass one of --demo (synthetic) or --snapshot <id> (real data)")
```

In the existing `run` command, replace the inline `if demo and snapshot / if demo / elif snapshot / else` block with:
```python
    provider = _select_provider(demo, snapshot)
```
(Keep the rest of `run` — the `result = run_backtest(...)`, `--register` block, and `emit` — unchanged.)

Then add the new command:

```python
@backtest_app.command("walk-forward")
@json_errors(ValueError, LookupError, BacktestError)
def walk_forward_cmd(
    name: str,
    start: str = typer.Option("2023-01-01", "--start"),
    end: str = typer.Option("2023-12-31", "--end"),
    demo: bool = typer.Option(False, "--demo", help="use the synthetic data provider"),
    snapshot: str = typer.Option(None, "--snapshot", help="backtest an ingested bars snapshot id"),
    windows: int = typer.Option(4, "--windows", help="number of equal out-of-sample windows"),
    holdout_frac: float = typer.Option(0.2, "--holdout-frac", help="fraction reserved as holdout"),
) -> None:
    """Walk-forward (out-of-sample-across-time) evaluation: per-window + holdout metrics + stability."""
    strategy = load_strategy(name)
    provider = _select_provider(demo, snapshot)
    result = walk_forward(strategy, provider, _utc(start), _utc(end),
                          windows=windows, holdout_frac=holdout_frac)
    emit(result.to_dict())
```

`BacktestError` is already imported in `backtest_cmd.py` (from the earlier `--snapshot` error fix). If it is not, add `from algua.backtest.engine import BacktestError`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli_walkforward.py tests/test_cli_backtest.py -q`
Expected: PASS (new walk-forward tests + the existing `run` tests, since `_select_provider` preserves behavior).

- [ ] **Step 5: Commit**

```bash
git add algua/cli/backtest_cmd.py tests/test_cli_walkforward.py
git commit -m "feat: add 'backtest walk-forward' CLI command"
```

---

### Task 6: Full verification & smoke

**Files:** none (verification only)

- [ ] **Step 1: Full quality gate**

Run:
```bash
uv run pytest -q
uv run ruff check .
uv run mypy algua
uv run lint-imports
```
Expected: all tests pass; ruff clean; mypy `Success`; import-linter `4 kept, 0 broken` (the engine + walkforward stay off `algua.data`; `walkforward` only imports within `backtest` + `contracts`/`strategies`).

- [ ] **Step 2: CLI smoke (synthetic)**

Run:
```bash
uv run algua backtest walk-forward cross_sectional_momentum --demo --start 2021-01-01 --end 2023-12-31 --windows 4 --holdout-frac 0.2
```
Expected: JSON with 4 `window_metrics` entries (each with `sharpe`/`total_return`/`n_bars`), a `holdout_metrics` block, and a `stability` block (`mean_sharpe`, `std_sharpe`, `min_sharpe`, `pct_positive_windows`).

- [ ] **Step 3: Final commit (if any verification fixes were needed)**

```bash
git add -A
git commit -m "test: verify walk-forward end to end" --allow-empty
```

---

## Self-Review Notes

- **Spec coverage:** engine refactor exposing `_build_portfolio` (Task 1), `metrics_from_returns`
  (Task 2), bar-count segmentation with holdout + remainder-to-last + too-few-bars guard (Task 3),
  `WalkForwardResult` + `walk_forward` with per-window/holdout metrics + stability + stamps
  (Task 4), CLI `walk-forward` with `--demo`/`--snapshot`/`--windows`/`--holdout-frac` (Task 5),
  verification (Task 6). Out-of-scope (sweeps, MLflow, promotion gates, optimization/anchored
  windows) intentionally absent.
- **Boundary:** all new code in `algua/backtest/*` + the CLI; `walkforward` imports only from
  `backtest`/`contracts`/`strategies`; engine + walkforward never import `algua.data` (4 contracts
  stay green).
- **Type consistency:** `_build_portfolio -> (pf, weights_eff)`, `metrics_from_returns(returns) ->
  dict`, `_segment_bounds(n, windows, holdout_frac) -> (list[(s,e)], (s,e))`, `_MIN_WINDOW_BARS`,
  `WalkForwardResult.to_dict()` keys, and `walk_forward(..., windows=, holdout_frac=)` are used
  identically across tasks and tests.
