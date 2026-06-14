# Dormant re-eval sweep — Slice B of #125 — issue #198

**Date:** 2026-06-13
**Issue:** #198 (Slice B of #125 — the dormant lifecycle stage shipped in Slice A, PR #196)
**Scope:** A read-only CLI command that screens the `dormant` pool for recovery and prints a ranked, advisory report. No state change, no ledger writes, **no holdout peek**.

## Problem

Slice A gave validated-but-resting strategies a home (`dormant`) and a recovery edge (`dormant -> paper`). What's missing is the **monitor**: a cheap way to answer "which dormant strategies look like they're working again on recent data?" without standing up a per-strategy paper/live loop. #125's answer is a periodic batch that re-evaluates the whole dormant pool on the latest bars and reports recovery candidates.

## Decision (from brainstorming + GATE-1)

- **Advisory STABILITY screen, NOT a gate, NOT a holdout peek.** The single-use walk-forward holdout is sacred: `walk_forward.holdout_metrics` are sensitive and only `research promote` may reveal them, and only after atomically burning the holdout (`backtest sweep` withholds them for exactly this reason — peek-without-consume). So the sweep **screens on the walk-forward WINDOW/stability metrics** (mean/min window Sharpe, pct-positive windows — the same class `backtest sweep` already reveals) and **never reads, evaluates, reveals, reserves, or burns the holdout**. The walk-forward holdout segment is computed in-memory by `walk_forward` but is dropped on the floor here.
  - It writes nothing (no `holdout_evaluations`, no `gate_evaluations`, no transition, no token), so it is **repeatable** nightly and scales linearly to hundreds of strategies.
- **It is a prioritization screen, not a promise.** A screen pass means "its walk-forward windows look healthy again — worth re-auditioning," NOT "it will clear re-promotion or the #124 forward gate." Re-promotion still burns a fresh single-use holdout; the forward gate still demands live/paper forward evidence. The report says this explicitly.
- **Report-only.** Never transitions. A human/agent acts via the existing `registry transition <name> --to paper --reason "..."`.
- **Eval inputs:** one common `--start/--end` window for the whole pool ("the latest bars"); each strategy re-evaluated on its **own static `config.universe`**. An optional single `--universe NAME` applies one universe to all (resolved ONCE, outside the loop). PIT status is reported as information; it is **not** a gate check here.
- **Per-strategy isolation:** each strategy runs in its own `try/except Exception` (re-raising `KeyboardInterrupt`/`SystemExit`); any failure is recorded and the sweep continues.
- **Fundamentals strategies are skipped** (the walk-forward lane can't thread fundamentals yet) with an explicit reason.

## Why a stability screen is the right primitive (and statistically honest)

Re-evaluating N degraded strategies and re-promoting whichever looks good is a multiple-comparisons search — "recovery" is often noise (#125's guard). Two properties keep this honest:

1. **It grants no progression and leaks no holdout.** The screen cannot move a strategy toward live and cannot reveal the untouched OOS holdout. A strategy only re-reaches live via `dormant -> paper -> forward_tested (#124, fresh live/paper evidence) -> live (human sign)`, and any future `backtested -> candidate` re-promotion burns its own fresh holdout. The screen sits entirely outside those single-use ledgers.
2. **It is labelled as a screen, not a gate.** The window/stability metrics it reports are the *robustness-across-sub-periods* signal `backtest sweep` already exposes — never the holdout verdict. **Residual multiple-testing risk remains** (running a wide screen and acting on the top-ranked is itself a search), and forward-gate multiple-comparison enforcement is currently *recorded, not enforced* (#124). So the report carries the pool size + rank and an explicit caution that acting on top-ranked names is a human judgement, not an automated entitlement.

## Architecture

New command `research dormant-sweep` in `algua/cli/research_cmd.py` (under `research_app`), reusing `promote`'s input helpers but none of its holdout/gate/transition machinery and none of `evaluate_gate` (which is holdout-coupled).

```
research dormant-sweep
  --start D --end D            # common window for the whole pool (defaults match promote)
  [--demo | --snapshot ID]     # data source (same as promote)
  [--universe NAME]            # optional single universe applied to all; else per-strategy static
  [--windows N] [--holdout-frac F]   # walk_forward shape; holdout is computed but NOT revealed
  [--min-window-sharpe X]      # screen threshold on MEAN window Sharpe (stability)
  [--min-pct-positive Y]       # screen threshold on pct-positive windows
  [--top N]                    # optional: cap the reported passed/failed lists
```

Flow (one `registry_conn()`):

1. Validate flags; parse `start/end` once. Build the data provider ONCE via the shared `_common` provider selector (so `--snapshot` doesn't rebuild a `DataStore` per strategy). If `--universe` is set, resolve `universe_by_date, universe_prov = resolve_universe_inputs(universe, start_dt, end_dt)` ONCE; else both are `None`.
2. `dormant = repo.list_strategies(Stage.DORMANT)`.
3. For each `rec` in `dormant`, in `try/except Exception` (re-raise `KeyboardInterrupt`/`SystemExit`):
   - `strategy = load_strategy(rec.name)`.
   - If `getattr(strategy.config, "needs_fundamentals", False)` → append `{strategy, reason: "needs_fundamentals: walk-forward lane not wired"}` to `skipped`; continue.
   - `wf = walk_forward(strategy, provider, start_dt, end_dt, windows=windows, holdout_frac=holdout_frac, universe_by_date=universe_by_date, universe_name=universe, universe_snapshots=universe_prov)`.
   - Read **only** `wf.stability` (mean_sharpe, min_sharpe, std_sharpe, pct_positive_windows) and `wf.window_metrics`. **Do not read `wf.holdout_metrics`.**
   - `screen_passed = (wf.stability["mean_sharpe"] >= min_window_sharpe) and (wf.stability["pct_positive_windows"] >= min_pct_positive)`.
   - Build a result: `{strategy, screen_passed, stability: wf.stability, windows: wf.window_metrics, config_hash: wf.config_hash, universe_name: wf.universe_name, universe_snapshots: wf.universe_snapshots, pit: bool(universe_prov)}`; route to `passed`/`failed` by `screen_passed`.
   - On exception → `errors.append({strategy: rec.name, error: f"{type(e).__name__}: {e}"})`.
4. Sort `passed` (and `failed`) by `stability.mean_sharpe` descending; apply `--top` if set.
5. `emit(ok({...}))`.

The command never calls `repo.apply_transition`, `reserve_holdout`/`finalize`/`release`, `record_gate_evaluation`, or `evaluate_gate`. It opens the registry connection only to `list_strategies(Stage.DORMANT)`.

## Output (single JSON object)

```json
{
  "ok": true,
  "note": "advisory stability screen over walk-forward windows; NOT the holdout gate. A pass means the strategy's windows look healthy again — worth re-auditioning via `registry transition --to paper` — it does NOT guarantee it will clear re-promotion (which burns a fresh holdout) or the #124 forward gate. Residual multiple-testing risk: acting on top-ranked names is a human judgement.",
  "period": {"start": "2023-01-01", "end": "2023-12-31"},
  "data_source": "SyntheticProvider",
  "snapshot_id": null,
  "thresholds": {"min_window_sharpe": 0.0, "min_pct_positive": 0.6},
  "total_dormant": 5,
  "evaluated": 4,
  "passed": [
    {"strategy": "mom_a", "screen_passed": true,
     "stability": {"mean_sharpe": 1.2, "min_sharpe": 0.4, "std_sharpe": 0.3, "pct_positive_windows": 0.75},
     "windows": [...], "config_hash": "...", "universe_name": null, "pit": false}
  ],
  "failed": [
    {"strategy": "rev_b", "screen_passed": false, "stability": {...}, "windows": [...], "config_hash": "..."}
  ],
  "skipped": [
    {"strategy": "fund_c", "reason": "needs_fundamentals: walk-forward lane not wired"}
  ],
  "errors": [
    {"strategy": "broken_d", "error": "StrategyNotFound: ..."}
  ]
}
```

`evaluated` = `len(passed) + len(failed)`. **No `holdout` key appears anywhere** (the load-bearing invariant). Follows the `emit(ok({...}))` envelope, like `backtest sweep`.

## Error handling

- The command is wrapped with `@json_errors(ValueError, LookupError, BacktestError, sqlite3.OperationalError)` (matching `promote`) for clean top-level JSON errors.
- Per-strategy failures are caught with a broad `except Exception` (re-raising `KeyboardInterrupt`/`SystemExit`) and collected into `errors[]`, recording `type(e).__name__` + message — one bad strategy never aborts the sweep.
- An empty dormant pool returns `ok:true`, `total_dormant: 0`, empty lists (not an error).

## Testing

`tests/test_cli_research.py` (or a new `tests/test_dormant_sweep.py`):

- **Empty pool:** no dormant strategies → `ok:true`, `total_dormant: 0`, all lists empty.
- **Pass/fail split:** seed two dormant strategies (drive each through the legal chain to `dormant`), tune `--min-window-sharpe`/`--min-pct-positive` so one screens pass and one fail deterministically on the synthetic provider → one in `passed`, one in `failed`, ranked by mean window Sharpe.
- **No holdout leak (load-bearing):** assert no `holdout`/`holdout_metrics` key appears anywhere in the output JSON.
- **No side effects (load-bearing):** after a sweep, strategy stages unchanged (still `dormant`); `holdout_evaluations` and `gate_evaluations` row counts unchanged.
- **Repeatable:** running the sweep twice on the same window yields identical pass/fail (no holdout exhaustion).
- **Fundamentals skipped:** a `needs_fundamentals=True` dormant strategy lands in `skipped` (with reason), not `errors`, and doesn't abort the sweep.
- **Per-strategy isolation:** a dormant strategy that raises during load/eval lands in `errors[]` while a healthy one still appears in `passed`/`failed`.
- **Non-dormant excluded:** a `paper`/`live` strategy is never evaluated.

## Out of scope

- **Scheduling** — the command is the unit; the operator crons it. No scheduler.
- **Auto-transition / auto-reactivation** (per #125) — report only.
- **A holdout-burning formal re-gate** — explicitly rejected in brainstorming; the screen is window/stability-only.
- **Per-strategy stored recovery windows/universes** (new schema) — common-window + per-strategy-static-universe.
- **Structured `dormancy_reason`** — unchanged from Slice A's deferral.
- **Fundamentals re-eval** — skipped until the fundamentals walk-forward lane exists.
- **Experiment-tracker/MLflow recording** — the JSON report is the artifact; curation is the `report-experiments` skill's job.

## Gate

`uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports` green.
