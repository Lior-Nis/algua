# Backtest series-emit CLI seam + report-experiments series plots (#181)

**Status:** design (revised after GATE-1 multi-model review — Codex + OpenCode)
**Issue:** #181 (deferred from #133 / PR #175 `report-experiments` skill v1)
**Date:** 2026-06-20

## Problem

The `report-experiments` skill (v1) plots **tracked-data only** (sweep heatmap/sensitivity/
leaderboard, walk-forward per-window stability, cross-run `mean_sharpe`). The most-wanted *series*
plots were deferred:

- equity curve
- drawdown
- rolling Sharpe
- return distribution

They need the portfolio **return series**, which `BacktestResult` carries internally
(`returns: pd.Series | None`) but `to_dict()` excludes (`algua/backtest/result.py:57-62`), so **no
CLI/MLflow surface emits it**. Producing it from the skill would mean reaching into `algua.backtest`
internals, violating the golden rule *"never reach into modules to bypass the CLI."* This spec adds
the emit surfaces, then teaches the skill to consume them.

## Goals

1. Expose the decision-time portfolio return series, deterministically + provenance-stamped, two ways:
   - **File seam:** `backtest run --emit-series PATH` writes a parquet (operator/agent ad-hoc use).
   - **MLflow artifact:** when `--track`, `log_backtest` logs the same series as a run artifact, so the
     skill consumes the **logged run's own** series with no re-run.
2. Extend `report-experiments` to render equity / drawdown / rolling-Sharpe / return-distribution
   SVGs from that artifact, preserving the skill's discipline (deterministic SVG, provenance stamp,
   never surface the holdout).
3. Declare `matplotlib` as a direct dependency.

## Holdout safety (the load-bearing analysis — corrected after review)

GATE-1 correctly flagged that the first draft's "a plain backtest run has no holdout, so plotting it
is safe **by construction**" reasoning was **wrong**. The single-use holdout's identity is the OOS
**calendar interval** `[holdout_start, holdout_end]`, **provenance-independent** (#205,
`registry/store.py:500-530`), not the command shape. A `backtest run` over a window overlapping a
reserved/burned holdout interval *does* aggregate the holdout bars.

The corrected, honest position (verified against the code):

- This change introduces **no new *kind* of holdout risk**. Today `backtest run` already (a) emits
  scalar metrics over the entire `[start, end]` with **no holdout guard** (`backtest_cmd.py:42-137`,
  `engine.py:686-709`), and (b) **persists the entire daily return series** to the registry via
  `persist_backtest_returns` for any **registered** strategy (#222 clustering,
  `backtest_cmd.py:113-128`, `store.py:1288`). The window selection that determines holdout overlap
  is identical whether you read the scalar Sharpe, the DB-persisted series, or the new file/artifact.
- It **does** add a new *sanctioned, fine-grained* surface: for an **unregistered** strategy the
  daily path was previously nowhere on disk (only scalar metrics), and a return *path* is strictly
  more granular than a blended scalar. So the honest framing is "no new holdout *mechanism* to defend,
  not a literal zero-delta exposure." Declining an emit-only guard is still correct (guarding only
  this path while metrics + `persist_backtest_returns` stay open is inconsistent theater); the proper
  fix, if ever warranted, is a **global** `backtest run` holdout-overlap policy — explicitly deferred
  (see Decisions).
- `backtest run` is intentionally a **free exploration tool**; the holdout's single-use *integrity*
  is enforced at the **gate** (`research promote` reserves+burns the interval atomically;
  `walk_forward`/`sweep` **withhold** `holdout_metrics`). This change touches **none** of that.
- Adding a holdout-overlap guard to **only** `--emit-series` would be inconsistent security-theater
  (metrics + `persist_backtest_returns` would stay equally open) and is therefore **out of scope**.
  If "free exploration over a reserved holdout window" is ever judged a real risk, the fix is to
  guard **all of `backtest run`** (a new public `list_held_intervals(strategy_id)` query + a
  pre-execution overlap check) — a separate, larger decision, noted as a deferred follow-up.

What this design **does** guarantee:

- The series is sourced for the skill **only from a plain `backtest run` artifact** (MLflow
  `kind="backtest"`). `log_sweep`/`log_walk_forward` log **no** series artifact, so a
  walk-forward/sweep return vector (which *does* contain the reserved holdout tail —
  `walkforward.py`, and `WalkForwardResult.holdout_returns` is SENSITIVE per #211 S1) is **never**
  written to MLflow or a file. A test asserts walk-forward/sweep tracked runs carry no series
  artifact. `series_frame()` (below) only accepts a `BacktestResult`, so it is type-impossible to
  feed it a `WalkForwardResult`; an invariant comment documents the rule.

## Architecture

### Shared serialization (no duplication between the two emit surfaces)

- **`series_frame(result: BacktestResult) -> tuple[pd.DataFrame, dict[str, str]]`** in
  `algua/backtest/result.py` (pure pandas + stdlib, no I/O). Builds the `[date, ret]` frame
  (`date` = ISO-8601 of each `result.returns.index` timestamp; `ret` = float daily return,
  **canonicalized `-0.0 -> +0.0`** so a flat day can't perturb bytes — mirrors `logical_bars_hash`)
  and a **fully-stamped** provenance metadata dict. To avoid an incomplete-provenance file seam, the
  metadata embeds the **entire** identity: a single key `algua.result_json` =
  `json.dumps(result.to_dict(), sort_keys=True, default=str)` (which already excludes `returns` and
  carries `strategy`, `config_hash`, `code_hash`, `dependency_hash`, `snapshot_id`, `seed`,
  `timeframe`, `period`, `data_source`, `universe_name`/`universe_snapshots`,
  `fundamentals_snapshot`, `news_snapshot`, `delisting_snapshot`, `forced_exits`, and `metrics`).
  Sorted keys keep it byte-deterministic. Callers guard
  `result.returns is not None and len(result.returns) > 0` before calling.
- **`frame_to_parquet_bytes(frame, metadata: dict[str, str] | None = None)`** —
  extend the existing primitive (`algua/data/files.py:105`). Today it strips schema metadata for
  content-hash determinism; the new optional `metadata` (default `None` = strip, **existing callers
  unchanged**) attaches the dict to the arrow schema with **sorted keys**, utf-8 encoded, so output
  stays byte-deterministic for a fixed `dependency_hash`. (Determinism is scoped to identical
  `dependency_hash`/pyarrow — a writer/dep bump may change bytes, which is fine: the file is not
  content-addressed.)
- **`write_bytes_atomic(data: bytes, dest: Path) -> None`** in `algua/data/files.py` — same-dir
  temp + `os.replace` (no fsync; this is an ephemeral plotting input, not a durable snapshot). Honors
  the "never a partial file" promise even on mid-write crash.

### Part A — `backtest run --emit-series PATH`

Add the flag to `backtest run` (`algua/cli/backtest_cmd.py`). When set:

- **Fail closed** if `result.returns is None` (engine set it for non-finite returns,
  `result.py:54-55`, `engine.py:714-717`) **or** the series is empty (`len == 0`): raise
  `BacktestError` ("backtest produced no finite return series; nothing to emit") → standard
  `@json_errors` envelope, non-zero exit, **no file written**.
- Else `frame, meta = series_frame(result)`; `write_bytes_atomic(frame_to_parquet_bytes(frame, meta), PATH)`.
- The stdout payload (otherwise byte-identical to today) gains a `"series"` descriptor:
  `{path, n, code_hash, dependency_hash, config_hash, snapshot_id, seed, data_source, start, end, timeframe}`.
  Absent when `--emit-series` is not passed.

### Part B — series artifact at `--track` time

In `log_backtest` (`algua/tracking/mlflow_tracker.py:84`), after the existing
`log_dict(result.to_dict(), "result.json")`: if `result.returns is not None and len(result.returns) > 0`,
log a `series.parquet` artifact. MLflow logs an artifact under its **basename**, so the temp file must
be named exactly `series.parquet` and cleaned up deterministically:
`with tempfile.TemporaryDirectory() as d: p = Path(d) / "series.parquet"; p.write_bytes(frame_to_parquet_bytes(*series_frame(result))); mlflow.log_artifact(str(p))`.
Tracking is best-effort (`_track` wraps failures), so a missing/empty series simply skips the
artifact — never fails the run. `log_sweep`/`log_walk_forward` are **unchanged** (no series artifact —
see Holdout safety).

### Part C — skill: series plots in `report-experiments`

`.codex/skills/report-experiments/SKILL.md` (single file, embedded reference script). Purely
**additive read** logic — the skill drives **no** backtest; it reads the artifact:

1. Select the backtest run providing the series, preferring identity coherence with the rest of the
   report (so the equity/drawdown can't silently belong to a different config than the reported
   sweep/walk-forward):
   - if `--backtest-run-id` is given, use that run;
   - else prefer the newest `kind == "backtest"` run whose `series.parquet` downloads **and** whose
     `config_hash` + `snapshot_id` match the walk-forward (or sweep) run the report is built on;
   - else fall back to the newest `kind == "backtest"` run with a `series.parquet`, and the report
     **explicitly labels which `config_hash` / `period` / `snapshot_id` the series is from** (not just
     "as logged") so a reader can't assume it matches the sweep's best combo.
   Download via `mlflow.artifacts.download_artifacts`, read with pandas. Provenance comes from the
   **same run's** `result.json` — the series plots carry the **"identity AS LOGGED BY THE RUN"** stamp,
   consistent with the existing plots (no re-run, no current-code drift, no conflation — the H1/H2
   review findings resolved by construction). No backtest run with a `series.parquet` → series plots
   omitted with a one-line "run a `--track`ed `backtest run` first" note.
2. Render four SVGs from `ret`, all derived:
   - **equity** = `(1 + ret).cumprod()`
   - **drawdown** = `equity / equity.cummax().clip(lower=1.0) - 1` — peak floored at starting
     capital to match the canonical `metrics.py` max-drawdown convention (`metrics.py:40`).
   - **rolling Sharpe** — fixed window **63** (≈ one quarter; documented in the prose), `mean/std × √252`;
     the warm-up prefix (`n < window`) renders as **NaN gaps**, and any window with `std == 0` is
     **NaN** (never a fake `0` or `inf`). If the whole series has `n < window`, the plot is skipped.
   - **return distribution** — histogram of `ret`.

Determinism unchanged (same `svg.hashsalt` / `MPLCONFIGDIR` / no-date pinning); deterministic
returns → byte-identical SVGs.

### Part D — dependency hygiene

Add `matplotlib` to `[project].dependencies` in `pyproject.toml` (currently transitive via `mlflow`).
Drop the skill preflight's `uv add matplotlib` fallback note (keep a friendly importability check).

## Testing

- `tests/test_cli_backtest.py`:
  - `--emit-series` writes a parquet at PATH; `n == len(returns)`, columns `[date, ret]`, schema
    metadata carries the provenance keys; stdout `series` descriptor matches the file.
  - determinism: two emits (same snapshot+seed) → byte-identical parquet.
  - omitting `--emit-series` leaves the payload identical (no `series` key).
  - fail-closed: non-finite/empty returns + `--emit-series` → error envelope, non-zero exit, no file.
- `tests/` data-files module: `frame_to_parquet_bytes(metadata=…)` round-trip + deterministic bytes
  + sorted-key independence; `-0.0` canonicalization; `write_bytes_atomic` leaves no partial file.
- tracker tests: a `--track`ed `backtest run` logs `series.parquet`; **`sweep`/`walk-forward`
  tracked runs do NOT** (holdout-tail guard).
- skill: validated by a real dry-run (tracked `backtest run --demo --track`, run the script, confirm
  the four series SVGs render from the artifact with the logged-identity provenance stamp) — recorded
  in the PR.

## Build order (all safe alone — no new exposure, see Holdout safety)

1. Shared plumbing: `series_frame`, `frame_to_parquet_bytes(metadata=…)`, `write_bytes_atomic` (+ tests).
2. Part A `--emit-series` flag (+ tests).
3. Part B series artifact in `log_backtest` (+ tests incl. sweep/wf negative).
4. Part D matplotlib dep.
5. Part C skill series plots (+ dry-run).

## Quality gate

`uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports` green at every
commit; the skill's series plots validated by a real dry-run.

## Decisions / deferred

- **No `--emit-series` holdout guard** — no new holdout *mechanism* (window selection that determines
  overlap is identical to the existing scalar metrics + `persist_backtest_returns`); a guard on only
  this path would be inconsistent theater. **Deferred follow-up (separate issue):** the series emit is
  a new fine-grained surface (notably for *unregistered* strategies, whose daily path was previously
  nowhere on disk) — if "free exploration over a reserved holdout window" is judged a real risk, the
  fix is a **global** `backtest run` holdout-overlap policy (a new public `list_held_intervals(strategy_id)`
  registry query + a pre-execution overlap check), not an emit-only patch.
- **No MLflow-artifact → no series plots** for runs tracked before this lands; going forward every
  `--track`ed backtest carries the artifact. (Issue's "out of scope: persist as MLflow artifact" was
  reframed by GATE-1 from a perf optimization to a correctness fix and adopted with user sign-off —
  it removes the re-run's code/input-drift dishonesty entirely.)
- **Series-only (no equity column)** — equity/drawdown/rolling-Sharpe/dist all derive from returns;
  one source of truth, no drift.
- **Plain file export, not a managed snapshot** — ephemeral, reproducible plotting input; the
  DataStore manifest/content-address/fsync machinery would be overkill.
