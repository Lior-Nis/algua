# Parameter Sweeps — Design

**Date:** 2026-05-31
**Branch:** `research-sweeps`. **Status:** Approved (design); plan to follow.
**Sub-project:** 3 (research core) — "research depth", slice 2 of 4 (sweeps; MLflow + promotion
gates follow). Builds directly on slice 1 (walk-forward).

## 1. Goal

Run a strategy across a **parameter grid**, score each combo **out-of-sample** via walk-forward,
and emit a **ranked** result plus the **search-breadth count** the promotion gate will need.

```
algua backtest sweep cross_sectional_momentum --snapshot <id> \
    --param lookback=20,40,60 --param top_k=1,3,5 --rank-by mean_sharpe --top 10
```

## 2. Scoring discipline (the whole point)

Each combo is evaluated with **`walk_forward`** (slice 1): rank by an **out-of-sample window
metric** (default `mean_sharpe` across windows). The reserved **holdout is carried through per
combo but NEVER used for ranking** — it stays untouched for the final promotion gate (slice 4).
Ranking the best of N combos on in-sample data is the classic overfitting trap; scoring on
walk-forward windows mitigates it, and recording `n_combos` lets the gate penalize search breadth.

## 3. Components (new `algua/backtest/sweep.py`, backtest lane)

- `_parse_grid(params: list[str]) -> dict[str, list[Any]]` — turns `["lookback=20,40,60",
  "top_k=1,3,5"]` into `{"lookback": [20, 40, 60], "top_k": [1, 3, 5]}`. Each value is coerced
  int → float → str (first that parses). Malformed entries (no `=`, empty key, empty value list)
  → `ValueError`.
- `_combos(grid: dict[str, list]) -> list[dict[str, Any]]` — Cartesian product of the grid into a
  list of param dicts. Guarded by `_MAX_COMBOS = 200`: more → `BacktestError` ("grid too large:
  N combos > 200; narrow the grid"). (combos × K windows backtests can explode.)
- `_override(strategy, combo) -> LoadedStrategy` — `cfg = strategy.config.model_copy(update=
  {"params": {**strategy.config.params, **combo}})`; returns `LoadedStrategy(config=cfg,
  fn=strategy.fn)`. Must NOT mutate the base strategy/config.
- `SweepResult` dataclass + `to_dict()`:
  - sweep-level: `strategy`, `data_source`, `snapshot_id`, `timeframe`, `seed`, `period`, `grid`,
    `n_combos` (**search breadth**), `rank_by`, `ranked` (list, sorted by `rank_by` descending),
    `best` (the top combo's `{params, score}`).
  - each `ranked` entry: `{params, config_hash, n_windows, stability{mean_sharpe, std_sharpe,
    min_sharpe, pct_positive_windows}, holdout{n_bars, sharpe, total_return, max_drawdown},
    score}` where `score = stability[rank_by]`. (Compact per-combo summary — not the full
    per-window breakdown.)
- `sweep(strategy, provider, start, end, *, grid, windows=4, holdout_frac=0.2,
  rank_by="mean_sharpe") -> SweepResult` — drives `_combos` → `_override` → `walk_forward` per
  combo, builds records, sorts by `score` descending. `rank_by` ∈ `{"mean_sharpe", "min_sharpe"}`
  (window/stability metrics only — never a holdout metric); invalid → `ValueError`.

## 4. CLI

`algua/cli/backtest_cmd.py` gains a `sweep` command (sibling to `run`/`walk-forward`), reusing the
existing `_select_provider`:
`algua backtest sweep <name> (--demo | --snapshot <id>) [--start D] [--end D] [--windows K]
[--holdout-frac F] --param KEY=v1,v2,... [--param ...] [--rank-by mean_sharpe] [--top N]`
→ emits `SweepResult` JSON. `--param` is repeatable (Typer `list[str]`). `--top` limits the
printed `ranked` rows (default 20; the full `n_combos` is always reported). Errors render as JSON
via `@json_errors(ValueError, LookupError, BacktestError)`.

## 5. Error handling
- No `--param` given → `ValueError` ("provide at least one --param KEY=v1,v2,...").
- Malformed `--param` → `ValueError` (from `_parse_grid`).
- Grid > `_MAX_COMBOS` → `BacktestError`.
- Bad `--rank-by` → `ValueError`.
- Data-source / too-few-bars / empty-data errors propagate from `walk_forward`/engine as
  `BacktestError` (already JSON-rendered).

## 6. Testing
- `_parse_grid`: int/float/str coercion; multiple params; malformed (`"x"`, `"=1"`, `"k="`) → raise.
- `_combos`: Cartesian count = product of value counts; `_MAX_COMBOS` guard raises.
- `_override`: produces a strategy whose `params` merge combo over defaults; the base strategy's
  config/params are unchanged (no mutation).
- `sweep` end-to-end on the synthetic provider: `n_combos` correct; `ranked` sorted descending by
  `score`; `score == stability[rank_by]`; each entry carries holdout but holdout is not the sort
  key; deterministic (same seed → identical `to_dict()`); bad `rank_by` raises.
- CLI: `backtest sweep --demo --param ...` emits ranked JSON with `n_combos`/`best`; `--top N`
  limits rows; no-`--param`, malformed-`--param`, and too-large-grid render JSON errors.
- Full gate green: `pytest`, `ruff`, `mypy`, `lint-imports` (4 contracts; `sweep`/`walkforward`
  off the data lane).

## 7. Out of scope (later slices)
- **MLflow** experiment tracking (slice 3) — recording the now-many runs/artifacts.
- **Promotion gates** (slice 4) — consuming `n_combos` (search-breadth penalty) + holdout +
  stability to gate `idea→backtested→shortlisted`.
- **Full walk-forward optimization** (re-fit best params per train window, evaluate next window) —
  a heavier refinement; this slice ranks whole-history walk-forward scores, it does not re-fit
  per window.
- Random/Bayesian search, parallel execution — grid + sequential only for now.
