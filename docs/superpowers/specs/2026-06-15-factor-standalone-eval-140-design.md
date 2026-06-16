# Standalone factor evaluation — IC/IR + 1-factor backtest adapter (#140, slice B)

**Status:** design approved (brainstorming), pre-implementation
**Issue:** #140 — composable factor/signal registry
**Predecessor:** slices A+C merged (PR#202, `7849081`) — the `@factor` catalogue + derived
lineage. This is **slice B**: individual factor evaluation.
**Date:** 2026-06-15

## Problem

The factor catalogue (slice A) holds *library helpers* — `momentum(prices, lookback)`,
`zscore(values)` — pure functions with **arbitrary signatures**, used *inside* a strategy's
`signal()`. Issue #140 point #2 wants each factor judged **on its own**: "is news sentiment
predictive by itself?" Today that is impossible — there is no uniform contract by which an
arbitrary catalogued factor can be evaluated, and a raw factor must never go live (only
lifecycle-managed strategies do).

The issue's own framing: "Evaluate a factor as a standalone strategy = wrap it in a **trivial
1-factor adapter** (factor→weights) for backtest. Factors are *promotable to* strategies via an
adapter, never unified." Slice B builds that adapter plus a construction-free predictive-power
metric (IC/IR).

## Decisions taken (brainstorming)

1. **Modality:** a standalone 1-factor **backtest adapter** (reuse the existing
   engine) **AND** an **IC/IR** quality block from the same score panel. Not IC-only; not
   backtest-only.
2. **Bridge:** introduce a **uniform alpha-factor shape** — a standalone-evaluable factor is
   signal-shaped `(view, params) -> scores`. Transforms (`zscore`) and helpers (`momentum`) are
   **not** standalone-evaluable. A generic adapter wraps any standalone factor.
3. **Persistence/FDR:** **implemented in #219 (slice E).** Each `factor eval` invocation is
   recorded in the `factor_evaluations` ledger. The IC t-stat is multiple-testing corrected:
   breadth haircut `sqrt(2·ln N)` + DSR-confidence AND-check (`significant` in the `fdr` block).
   `fdr_corrected: true` in the CLI output reflects this.
4. **Rewire demo:** included — extract `cross_sectional_momentum`'s alpha into a catalogued
   standalone factor and have the strategy compose it (exercises slice-A/C lineage end to end).

## Architecture

Three layers, respecting the import-linter contracts (`features` is pure; `backtest` may import
`features`/`strategies`/`portfolio`/`engine` but not `cli`/`registry`/`data`):

```
features/catalogue.py   (pure)  — the standalone-evaluable contract + callable resolver
features/alphas.py      (pure)  — seed standalone alpha (xs_trailing_return)
backtest/factor_eval.py         — adapter (factor→synthetic LoadedStrategy) + IC/IR + orchestration
cli/factor_cmd.py               — `algua factor eval` (mirrors `backtest run` inputs)
```

### 1. Evaluable contract — `algua/features/catalogue.py` (pure)

A factor is **standalone-evaluable** iff it is **signal-shaped**:
`(view: bar-schema DataFrame, params: dict) -> pd.Series` of cross-sectional scores (the exact
shape of a strategy's `signal`).

- `FactorSpec` gains `standalone: bool` (defaults to `False`; tuple/frozen as today).
- `@factor` gains a `standalone: bool = False` kwarg.
- When `standalone=True`, the decorator **validates the shape at decoration time** (fail closed):
  exactly two parameters, both `POSITIONAL_OR_KEYWORD`, no `*args/**kwargs`. A transform like
  `zscore` (one arg) or a helper like `momentum` (two args but `(prices, lookback)` semantics —
  see note) cannot be silently mis-flagged.
  - Note: the shape check is structural (arity/kind), not semantic — it cannot tell `(view,
    params)` from `(prices, lookback)`. Authoring a `standalone` factor is a deliberate act; the
    arity gate stops the obvious mistakes (transforms, varargs), and the seed alpha is the
    worked example. This is the same "best-effort, author-asserted" posture as slice A/C lineage.
- New pure helper `load_factor_callable(spec: FactorSpec) -> Callable[..., Any]`: importlib-resolve
  `spec.import_path` (`"module:qualname"`) back to the function object. The catalogue scan already
  imports the module, so resolution is import-safe; resolves the qualname via attribute walk and
  asserts the resolved object carries the matching `__factor_spec__` (fail closed on drift).
- `momentum`/`zscore` stay `standalone=False`.

### 2. Seed alpha + composition demo

- New pure `algua/features/alphas.py` with one seed standalone alpha:

  ```python
  @factor(standalone=True, kind=FactorKind.MOMENTUM, tags=[...], summary=...)
  def xs_trailing_return(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
      """Cross-sectional trailing return per symbol (the cross_sectional_momentum alpha)."""
      # pivot view to wide adj_close, momentum(wide, lookback), take last row, dropna
  ```

  Built on `indicators.momentum` (composition within the pure layer).
- **Rewire `algua/strategies/momentum/cross_sectional_momentum.py`**: `signal` delegates to
  `xs_trailing_return` (behavior-identical — the body moves, the output is byte-identical). The
  top-level import means slice-A/C lineage (`factor uses cross_sectional_momentum`, `factor
  dependents xs_trailing_return`) and the live-gate code_hash closure now both cover the factor.
  Nothing is live → the one-time closure/identity shift is harmless (same as slice A's note).
  `signal_panel` is updated in lockstep so the engine's fast-path parity guard still holds.

### 3. Adapter + IC/IR — `algua/backtest/factor_eval.py`

**Adapter.** `build_factor_strategy(spec, *, universe, params, construction, construction_params,
execution) -> LoadedStrategy`:
- Rejects a non-`standalone` factor with a clear error.
- Resolves the callable via `load_factor_callable`, binds it as `signal_fn`.
- Builds a `StrategyConfig` named `__factor__:<spec.name>` (reserved `__factor__:` prefix —
  registry-inert; never persisted, never given a gate token, never run through
  walk_forward/holdout). Construction is **required** (no default) so no hidden weighting bias.
- Returns a `LoadedStrategy(config, construct_fn=get_construction_policy(construction),
  signal_fn=fn)`.

**Backtest.** The synthetic strategy is fed straight to the **existing `engine.run`** →
standard `BacktestResult` metrics. No engine change.

**IC/IR.** `factor_ic(score_panel, fwd_returns, *, min_cross_section)`:
- **Score panel** `score_panel[t, symbol]`: built by iterating the factor over an **expanding PIT
  window** — for each decision bar `t`, `view = bars[:t]`, `scores_t = fn(view, params)`. This is
  the faithful, look-ahead-free construction (mirrors `_canonical_row`). Vectorization via a
  factor `signal_panel` form is **deferred**.
- **Forward returns** `fwd_returns[t, symbol]`: from `adj_close`, the return over `horizon` bars
  **offset by the execution decision lag** — a score known at `t` is tradable at `t+lag`, so the
  label is `adj_close_{t+lag+h} / adj_close_{t+lag} - 1`. Lag and horizon are explicit, never
  implicit.
- **Per-bar rank IC**: Spearman correlation between `score_panel` row `t` and `fwd_returns` row
  `t`, over the symbols present (finite) in both. Bars with a cross-section narrower than
  `min_cross_section` (default 3) are skipped.
- **Aggregates**: `mean_ic`, `ic_std` (sample, ddof=1), `ir = mean_ic / ic_std`,
  `t_stat = ir * sqrt(n_obs)`, `hit_rate = mean(ic > 0)`, `n_obs`. A degenerate run (n_obs < 2,
  or zero IC variance) returns explicit NaN/None rather than a misleading number.
- Spearman (rank) IC is the single method — robust to monotone rescaling, the standard factor
  metric. Pearson IC is not added (YAGNI).

**Orchestration.** `evaluate_factor(...)` fetches provider bars for the window once, runs
`engine.run` for the backtest block, builds the score panel + forward returns from the same bars
for the IC block, and returns a merged result object whose `to_dict()` is the JSON envelope.

### 4. CLI — `algua factor eval` (extends `cli/factor_cmd.py`)

Mirrors `backtest run`: `name` (factor), `--start`, `--end`, `--demo`, `--snapshot`,
`--universe` (PIT opt-in), plus:
- `--construction <policy_id>` (**required**), `--construction-param k=v` (repeatable),
- `--param k=v` (repeatable; the factor params, e.g. `lookback=60`),
- `--horizon <int>` (forward-return horizon in bars; default 1),
- `--track` (optional MLflow logging — harmless, useful for the report skill).
- **No `--register`** — factor eval writes nothing to the registry.

Reuses `resolve_universe_inputs` and the provider/period resolution from `_common`. Param parsing
reuses the sweep `_coerce` value coercion. Emits one `ok(...)` JSON envelope:

```json
{
  "factor": "xs_trailing_return",
  "standalone": true,
  "params": {"lookback": 60},
  "construction": "top_k_equal_weight",
  "horizon": 1,
  "backtest": { "metrics": {...}, "config_hash": "...", "period": {...}, "universe_name": ... },
  "ic": { "method": "spearman", "mean_ic": 0.04, "ic_std": 0.21, "ir": 0.19,
          "t_stat": 1.7, "hit_rate": 0.55, "n_obs": 84, "min_cross_section": 3 }
}
```

## Non-goals (explicit, deferred)

- **No walk-forward / sweep for factors.** Those are strategy-*promotion* tools tied to the
  holdout-burn integrity surface (#161/#192/#193); a factor is non-promotable, so factor eval is a
  plain in-sample backtest only. Param sweeps over a factor are a possible later add.
- **FDR / persistence implemented in #219 (slice E).** `factor eval` now records each evaluation
  in the `factor_evaluations` ledger (SCHEMA_VERSION 25) and emits a `fdr` block with
  `fdr_corrected: true`. The `significant` field in the `fdr` block is the honest verdict after
  breadth haircut + DSR correction. Report-only — `research promote` and `gates.py` are unchanged.
  The `interpret-results` skill reflects this.
- **Factors never become tradable.** The only path from a promising factor to live is an operator
  authoring a real strategy module that composes it — preserving the live-gate model.
- **No vectorized IC.** The score panel is built by the per-bar expanding-window loop.

## Testing (TDD)

- **Contract:** `@factor(standalone=True)` accepts a 2-arg signal-shaped fn; rejects 1-arg
  (`zscore`-like), 3-arg, and `*args`/`**kwargs` fns at decoration time. `load_factor_callable`
  round-trips a spec to its function and fails closed on a stamp mismatch.
- **Adapter:** `build_factor_strategy` produces a valid `LoadedStrategy` (name prefix, bound
  signal_fn, resolved construction); rejects a non-standalone factor; requires construction.
- **IC math:** a synthetic strictly-monotone factor (score ∝ realized forward return) yields
  `mean_ic ≈ 1`; a constant/noise factor yields `mean_ic ≈ 0`; a sign-flipped factor yields
  `mean_ic ≈ -1`. Degenerate windows return NaN/None, not a fake number.
- **PIT/lag:** the score panel at `t` is invariant to bars after `t` (no look-ahead); the forward
  return is offset by the declared decision lag (a unit test pins the index alignment).
- **CLI:** `factor eval` emits the documented envelope; missing `--construction` errors; param
  coercion works; `--universe` provenance flows through; non-standalone factor errors cleanly in
  the JSON contract.
- **Rewire parity:** `cross_sectional_momentum` produces a **byte-identical** backtest result
  (metrics + config_hash) before vs after delegating to `xs_trailing_return`; lineage commands
  now report the dependency.

**Gate:** `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`.
