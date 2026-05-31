# MLflow Experiment Tracking — Design

**Date:** 2026-05-31
**Branch:** `research-mlflow`. **Status:** Approved (design); plan to follow.
**Sub-project:** 3 (research core) — "research depth", slice 3 of 4 (tracking; promotion gates
follow). Builds on slices 1–2 (walk-forward, sweeps).

## 1. Goal

Persist research runs to **MLflow** so the now-many runs (backtests, walk-forwards, sweeps) are
queryable, comparable, and auditable. Opt-in per command:

```
algua backtest run         <strat> --snapshot <id> --track
algua backtest walk-forward <strat> --snapshot <id> --track
algua backtest sweep        <strat> --snapshot <id> --param ... --track
```

## 2. Decisions (from brainstorm)
- **Full MLflow** (verified co-installs with pinned pandas 2.3.3 / numpy 2.4.6 / vectorbt). The
  `mlflow ui --backend-store-uri <uri>` browser UI is available for free.
- **Local file backend** (`mlruns/`, gitignored), configurable URI.
- **Run structure:** one run per `run`/`walk-forward`; a `sweep` = a **parent** run with a
  **nested child run per combo**.
- **Opt-in `--track`** flag — no behavior change when off; no surprise `mlruns/` writes.

## 3. Components

### 3.1 Config
Add to `algua/config/settings.py`: `mlflow_tracking_uri: str = "mlruns"` (env-overridable as
`ALGUA_MLFLOW_TRACKING_URI`). Tests point it at a tmp dir.

### 3.2 `algua/tracking/mlflow_tracker.py` (new)
Imported only by the CLI. `import mlflow` happens INSIDE the functions (lazy) so non-tracking
commands and unrelated tests don't pay MLflow's import cost.

- `log_backtest(result: BacktestResult, params: dict[str, Any], *, tracking_uri: str) -> str`
  Sets the tracking URI, `set_experiment(result.strategy)`, starts a run, logs:
  - **params:** the strategy `params` (flattened, `param.<k>`), plus `config_hash`, `snapshot_id`,
    `seed`, `period_start`, `period_end`, `timeframe`.
  - **metrics:** `result.metrics` (sharpe, cagr, …).
  - **tags:** `kind=backtest`, `strategy`, `config_hash`, `snapshot_id`, `timeframe`.
  - **artifact:** `result.to_dict()` as `result.json` (via `mlflow.log_dict`).
  Returns the MLflow `run_id`.
- `log_walk_forward(result: WalkForwardResult, params: dict[str, Any], *, tracking_uri: str) -> str`
  Same shape; logs `stability` + `holdout` (flattened, e.g. `holdout.sharpe`) as metrics and
  `windows`/`holdout_frac` as params; `kind=walk_forward`.
- `log_sweep(result: SweepResult, *, tracking_uri: str) -> str`
  Parent run: params `grid` (flattened `grid.<k>` as comma-joined), `n_combos`, `rank_by`,
  `windows`, `holdout_frac`, sweep-level stamps; metric `best_score`; tag `kind=sweep`; artifact
  `sweep.json` (`result.to_dict()`). Then a **nested child run per `ranked` entry**
  (`mlflow.start_run(nested=True)`): child params = the combo's `params` + `config_hash`; child
  metrics = `score` + the combo's `stability`/`holdout` (flattened); tag `kind=sweep_combo`.
  Returns the parent `run_id`.

A small private `_flatten(prefix, d)` helper turns nested dicts into `prefix.key` metric/param
names (MLflow keys must be flat strings; metric values must be numeric — non-numeric holdout
fields like `start`/`end` are logged as tags/params, not metrics).

### 3.3 CLI (`algua/cli/backtest_cmd.py`)
Add `track: bool = typer.Option(False, "--track", help="log this run to MLflow")` to `run`,
`walk_forward_cmd`, and `sweep_cmd`. After producing the result and BEFORE/around `emit`, when
`track` is set call the matching tracker with `get_settings().mlflow_tracking_uri`:
- `run`: `log_backtest(result, strategy.config.params, tracking_uri=...)`
- `walk-forward`: `log_walk_forward(result, strategy.config.params, tracking_uri=...)`
- `sweep`: `log_sweep(result, tracking_uri=...)`
The emitted JSON is unchanged (optionally include the returned `run_id` under a `"mlflow_run_id"`
key — yes, include it so the caller can find the run). Tracking errors are NOT swallowed silently
beyond the existing `@json_errors` handling; an MLflow failure surfaces as a JSON error.

### 3.4 Boundary
New import-linter contract: `algua.backtest` is forbidden from importing `algua.tracking` (the
engine/sweep/walkforward stay pure of tracking). `tracking` may import `algua.backtest` result
types + `algua.contracts`; the CLI imports `tracking`. (4 → 5 contracts.)

## 4. Error handling
- `--track` with an unreachable/invalid tracking URI → MLflow raises → rendered as JSON error.
- Non-numeric values never passed to `log_metric` (the `_flatten` split handles this).
- Existing data-source / grid / cadence errors are unchanged (raised before tracking).

## 5. Testing
- `_flatten`: nested dict → flat `a.b` keys; numeric vs non-numeric separation helper behaves.
- `log_backtest` / `log_walk_forward`: with `tracking_uri=tmp`, after logging, an `MlflowClient`
  (or `mlflow.search_runs(experiment_names=[strategy])`) finds exactly one run with the expected
  metrics (e.g. `sharpe`), params (`config_hash`), and `kind` tag; the `result.json` artifact
  exists.
- `log_sweep`: finds one parent run (`kind=sweep`, `n_combos` param) and exactly `len(ranked)`
  nested child runs (`kind=sweep_combo`) under it (filter by `tags.mlflow.parentRunId`).
- CLI: `backtest run --demo --track` (tmp URI via `ALGUA_MLFLOW_TRACKING_URI`) → exit 0, emitted
  JSON has `mlflow_run_id`, and a run exists in the store; WITHOUT `--track` → no run logged
  (experiment absent / empty). Same smoke for `walk-forward --track` and `sweep --track`
  (parent + children).
- Existing CLI tests (no `--track`) stay green and write nothing to `mlruns/`.
- Full gate green: `pytest`, `ruff`, `mypy`, `lint-imports` (now 5 contracts; `backtest` off
  `tracking` and off `data`).

## 6. Out of scope (later)
- **Promotion gates** (slice 4) — consuming holdout + stability + search-breadth to gate
  `idea→backtested→shortlisted`. Tracking only *records*; it does not gate.
- A tracking server / remote backend, model registry, run retention/pruning policies (local file
  store only for now; the architecture spec flags a future migration trigger).
- Auto-logging by default (this slice is opt-in `--track`).
