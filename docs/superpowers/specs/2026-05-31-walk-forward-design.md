# Walk-Forward (Out-of-Sample-Across-Time) Evaluation — Design

**Date:** 2026-05-31
**Branch:** `research-walkforward`. **Status:** Approved (design); plan to follow.
**Sub-project:** 3 (research core) — "research depth", slice 1 of 4 (walk-forward; sweeps,
MLflow, promotion gates follow as later slices).

## 1. Goal

Turn a single in-sample-looking backtest into a **trustworthy out-of-sample-across-time**
evaluation: run a strategy once over the full history (point-in-time as always), then segment the
result to measure whether performance **holds across time windows**, and reserve an untouched
**holdout** period. This catches "only worked in one regime" overfitting — the #1 gap now.

```
algua backtest walk-forward cross_sectional_momentum --snapshot <id> --start D --end D
```

**Not** parameter optimization. The engine is already point-in-time (no fitting), so a single run
is out-of-sample everywhere; this slice measures *stability across sub-periods* + reserves a
holdout. True walk-forward optimization (re-fitting per window) arrives with the sweeps slice.

## 2. Approach

1. Build the portfolio once over `[start, end]` via the existing engine (per-bar loop + `t→t+1`
   shift + vectorbt). One run ⇒ one warmup ⇒ every later segment is fully warm.
2. Take the per-bar portfolio **return series** (`pf.returns()`).
3. Partition the series **by bar count**:
   - **Holdout:** the final `holdout_frac` (default `0.2`) of bars — reported separately, labeled
     holdout (the discipline anchor future promotion gates will require to be untouched).
   - **Windows:** the remaining bars split into `K` equal consecutive windows (default `K=4`).
4. Compute return-based metrics per window and on the holdout; compute aggregate **stability**
   stats across the windows.

## 3. Components

### 3.1 Engine refactor (no behavior change)
Extract `_build_portfolio(strategy, provider, start, end) -> tuple[vbt.Portfolio, pd.DataFrame]`
(the existing fetch → per-bar decision loop → gross-exposure check → `t→t+1` shift → `from_orders`
→ `(pf, weights_eff)`). `run()` calls it and builds `BacktestResult` exactly as today. This
exposes the portfolio for reuse. Existing `run` behavior and all current tests are unchanged.
`engine.py` stays free of `algua.data` imports.

### 3.2 `algua/backtest/walkforward.py`
- `metrics_from_returns(returns: pd.Series) -> dict[str, float]` — pure; returns
  `{total_return, ann_return, ann_volatility, sharpe, max_drawdown}` computed from a return series
  (Sharpe = mean/std × √252; max_drawdown from the cumulative product; safe on empty/zero-vol →
  metrics default to `0.0`). Reused for every segment.
- `WalkForwardResult` dataclass with `to_dict()` (stable JSON):
  - `strategy`, `config_hash`, `data_source`, `snapshot_id`, `period`, `windows` (K), `holdout_frac`
  - `window_metrics`: list of `{index, start, end, n_bars, **metrics}` per window
  - `holdout_metrics`: `{start, end, n_bars, **metrics}`
  - `stability`: `{mean_sharpe, std_sharpe, min_sharpe, pct_positive_windows}` (over windows only,
    holdout excluded)
- `walk_forward(strategy, provider, start, end, *, windows=4, holdout_frac=0.2) -> WalkForwardResult`:
  builds the portfolio once, takes `pf.returns()`, splits by bar count (holdout tail first, then K
  equal windows on the remainder; any remainder bars from integer division go to the **last**
  window), computes per-segment metrics + stability, stamps `config_hash`/`snapshot_id` (via
  `getattr(provider, "snapshot_id", None)`).

### 3.3 CLI
`algua/cli/backtest_cmd.py` gains a `walk-forward` command (sibling to `run`):
`algua backtest walk-forward <name> (--demo | --snapshot <id>) [--start D] [--end D]
[--windows K] [--holdout-frac F]` → emits `WalkForwardResult` JSON. Provider selection and the
`--demo`/`--snapshot` mutual-exclusion mirror `run`. Errors render as JSON (`@json_errors(...)`,
incl. `BacktestError`).

## 4. Validation & error handling
- `windows >= 2`, `0 < holdout_frac < 1` → else `ValueError` (JSON).
- After reserving the holdout, the remaining bars must give each window `>= _MIN_WINDOW_BARS`
  (e.g. 5) → else `BacktestError` ("not enough bars for K windows; widen the period or lower K").
- Empty/no-data and gross-exposure violations surface from the engine as `BacktestError` (already
  JSON-rendered).

## 5. Testing
- `metrics_from_returns`: known return series → expected Sharpe/total_return/maxDD; empty and
  zero-vol series → zeros, no exception.
- Segmentation: K windows + holdout exactly cover the series with no overlap; bar counts sum to
  the total; remainder lands in the last window; holdout is the final `holdout_frac` of bars.
- `walk_forward` end-to-end on the synthetic provider: deterministic (same seed → identical
  result), `stability` keys present, `pct_positive_windows` in `[0,1]`, holdout reported.
- Engine refactor: existing `tests/test_backtest_engine.py` stays green (proves `run` unchanged).
- CLI: `backtest walk-forward --demo` emits valid JSON with `window_metrics`/`holdout_metrics`/
  `stability`; `--snapshot <id>` works against an ingested snapshot and stamps `snapshot_id`;
  too-few-bars and bad-flag paths render JSON errors.
- Full gate green: `pytest`, `ruff`, `mypy`, `lint-imports` (4 contracts; `backtest` off the data
  lane; `walkforward.py` lives in `backtest/`).

## 6. Out of scope (later research-depth slices)
- Parameter **sweeps** / walk-forward **optimization** (re-fitting params per window).
- **MLflow** experiment tracking.
- **Statistical promotion gates** (using holdout + search-breadth to gate `idea→backtested→
  shortlisted`). This slice only *produces* the holdout + stability numbers those gates will read.
- Calendar/anchored window schemes; per-window turnover/exposure (return-based metrics only here).
