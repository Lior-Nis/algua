# Agent skill: organized experiment reports with plots (#133)

**Status:** design · **Date:** 2026-06-09 · **Issue:** #133

## Problem

Experiment tracking is raw: MLflow logs scalar `metrics`, `params`, stamps, and
JSON blobs (`result.json` / `sweep.json` via `log_dict`) — **no figures, no
time-series, no interpreted narrative**. The kb sync writes only a flat
latest-metrics block per strategy note (`render_results_block`). There is no
organized, plotted report for a strategy's experiments or for an HPO/sweep.

## Decision

A **new agent skill** — `report-experiments` — and **no code change**. The skill
instructs the agent to read MLflow-tracked data, render data-science plots as SVG,
and author an organized, graph-linked markdown report into a dedicated kb area. It
is the curated/interpreted layer *on top of* MLflow, not a re-store of it.

### v1 scope — tracked data only (decided)

v1 reports are **research/HPO-phase** reports (sweep + walk-forward search
evidence). Plots are generated **only from data MLflow already holds**, so no
backtest re-run and no new CLI surface are needed:

- **Sweep / HPO:** parameter heatmap (score over a 2-param grid), per-parameter
  sensitivity curves (marginal score vs each swept param), top-N leaderboard
  (table + bar) — sourced from the logged `sweep.json` + run params/metrics.
- **Walk-forward:** per-window **in-sample** stability bars (e.g. Sharpe/return per
  window) — sourced from the `windows` segment records in `result.json`.
  **NO holdout-vs-insample plot (GATE-1 HIGH, Codex):** `mlflow_tracker` explicitly
  `pop`s `holdout_metrics` from the persisted `result.json` ("it lives only in
  `research promote`"), so the holdout is *not* in MLflow — it is single-use and
  burned at promotion. Plotting it from MLflow is impossible by design; surfacing
  burned holdout evidence from the gate record is a possible future enhancement,
  out of scope here.
- **Cross-run:** a leaderboard / metric-comparison across the strategy's runs over
  time — sourced from `mlflow.search_runs` filtered by strategy name.

**Deferred to a follow-up issue:** equity curve, drawdown, rolling Sharpe, return
distribution. Those need the **portfolio return series**, which no CLI command
emits today (`BacktestResult` carries only scalar `metrics`; `pf.returns()` stays
internal). Producing them from a skill alone would require reaching into
`algua.backtest` internals — a violation of the "never bypass the CLI" golden rule.
The clean enablement is a small future CLI change to emit the series through the
seam; that is **out of scope here** and noted as the follow-up.

### How the agent gets the data (CLI-respecting)

Reading MLflow is **not** bypassing the algua command surface — MLflow is a separate
experiment store, and the platform already reads it
(`algua/knowledge/metrics.py:latest_run_metrics` queries runs by strategy name via
the `mlflow_tracking_uri` setting). The skill directs the agent to:

1. Resolve the tracking URI the same way the platform does (the `mlflow_tracking_uri`
   setting / `MLFLOW_TRACKING_URI`, default `mlruns`).
2. Use the `mlflow` Python API (`search_runs`, `artifacts.download_artifacts`) to
   pull the run metrics/params and the `result.json` / `sweep.json` artifacts.
3. Plot with **matplotlib** (present transitively via the core `mlflow` dependency;
   verified importable in-env). Transitive availability is not a stable contract, so
   the skill **preflight-checks** `python -c "import matplotlib"` and, if missing,
   stops with a clear instruction (`uv add matplotlib`) rather than failing
   mid-report (GATE-1). Anything that *drives the system* still goes through
   `uv run algua ...` — the skill only *reads* tracked results.

The skill carries a **reference plotting script** (in the SKILL.md, as
documentation — not code added to `algua/`) that the agent adapts per run, so output
is consistent and reproducible rather than hand-rolled each time.

### Output location & shape

- Reports live at **`kb/strategies/<name>/reports/<run-or-timestamp>/`**:
  `report.md` + sibling `*.svg` figures. The kb sync globs `strategies/*.md`
  non-recursively, so the `<name>/` companion folder does not pollute `_index.md`
  / family rosters. (Per-strategy locality chosen over a central `kb/experiments/`
  so a strategy's knowledge stays in one subtree.)
- The report `[[wikilink]]`s back to the strategy note (`kb/strategies/<name>.md`)
  and the family note; the skill adds a link line into the strategy note's
  free-text area (never into the registry-owned synced blocks).

### Discipline the skill must enforce

1. **Not an MLflow duplicate.** MLflow stays the raw run store. The report is the
   curated, interpreted, graph-linked reading — it pulls *from* MLflow, never
   re-stores raw runs.
2. **Reproducibility stamp.** Every report header records the MLflow run id(s),
   `snapshot_id`, `code_hash`, `config_hash` it was generated from (all already in
   the tracked params / `result.json`). These are labeled **"identity as logged by
   the run"** — they pin the *run's* artifact, not the strategy's current source
   (which may have moved on), so a reader never misattributes provenance (GATE-1). A
   plot without provenance is a liability.
3. **Binaries in git.** The kb is version-controlled. Prefer **SVG** (text,
   diffable); keep figures small; one report dir per experiment. Emit SVG
   **deterministically** so re-runs don't churn git: set a fixed
   `mpl.rcParams["svg.hashsalt"]`, strip the date metadata
   (`savefig(..., metadata={"Date": None})`), **fix figure size / style / font
   family**, **sort all inputs deterministically** (MLflow runs by
   `(start_time, run_id)`; grids/leaderboards by a stable key), and point
   `MPLCONFIGDIR` at a report-local writable cache during rendering (GATE-1 round 2
   — these close the remaining SVG nondeterminism beyond hashsalt/Date). (Full
   content-addressing / git-LFS is over-engineering for a dev vault — deterministic
   SVG is the proportionate fix.)

## Components

- `.claude/skills/report-experiments/SKILL.md` and
  `.codex/skills/report-experiments/SKILL.md` — mirrored, like the other project
  skills. Frontmatter `name` + `description` (for trigger matching), body =
  playbook + the reference plotting script + the discipline checklist.
- Cross-links: `run-the-research-loop` names this as a natural **step-7 (Record)**
  extension; `interpret-results` is the judgement input the narrative draws on.
  (Editing `run-the-research-loop` to point at it is a small, optional touch — keep
  it to a one-line pointer if done.)

## Verification (no test suite for a skill)

Because there is no code, the gate is **a real dry-run**: pick the existing
strategy with tracked runs (or generate runs via `backtest sweep` /
`walk-forward`), follow the skill end-to-end, and confirm it produces a
`kb/strategies/<name>/reports/.../report.md` with valid SVGs, a correct
reproducibility stamp, and working wikilinks — committed to the kb. The skill text
must be specific enough that this succeeds without improvisation. The repo quality
gate (`pytest && ruff && mypy && lint-imports`) still must stay green (it will,
since no Python changes).

## Out of scope

- Persisting equity/returns/figures as MLflow artifacts (a separate small code
  change if re-running ever proves too heavy).
- Series-based plots (equity/drawdown/rolling-Sharpe/return-dist) and the CLI
  series-emit they require — the deferred follow-up.
- Declaring `matplotlib` as an explicit dependency (it's transitively guaranteed by
  `mlflow`; an explicit declaration is an optional rigor follow-up, not needed for
  the skill to work).

## Gate

A successful skill dry-run producing a committed report; repo quality gate
(`pytest && ruff check . && mypy algua && lint-imports`) stays green.
