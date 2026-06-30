# CLI `--summary` field projection (context-rot defense) — #349

## Problem

Algua's agent-facing JSON is emitted as full `indent=2` payloads with no field
projection. The heavy commands return everything:

- `backtest walk-forward` returns every per-window metric (`window_metrics`) plus the
  stability block.
- `backtest sweep` returns the full ranked combo list plus the parameter grid.
- `research promote` returns the entire `decision.to_dict()` — all FDR/regime/family/
  breadth/shadow-audit fields (~40 keys).

Only `data inspect` has a `--summary`. For an unattended operator ingesting hundreds of
runs/day into a finite, attention-degrading context window, this is the KB's named
**context-rot** failure mode (Tool Design for Agents — token-efficient responses;
Context Engineering — tool-output filtering).

## Goal

Give the operator a way to request only the **decision-relevant scalars** from the heavy
commands, with the full payload still available on demand.

## Design

### Fork decisions

1. **`--summary` (curated), not `--fields` (generic).** A generic `--fields a,b,c`
   projector pushes the knowledge of *which fields matter* onto the caller, is a footgun
   (typos, nested paths), and is YAGNI. A curated `--summary` directly serves the goal and
   matches the existing `data inspect --summary` precedent. `--fields` is dropped.

2. **Opt-in flag; default behavior unchanged (non-breaking).** Default emits the full
   payload exactly as today; `--summary` opts into the projection. Existing tests, MLflow
   logging, and downstream parsers are unaffected. The operator skill opts in by passing
   `--summary` (matching the issue's "have the operator skill request summaries").

3. **Top-level key projection, no nesting.** The decision-relevant sub-blocks
   (`stability`, `holdout`, `checks`, `best`) are already compact scalar dicts/lists, so
   selecting top-level keys suffices. No nested-path projector.

4. **One shared mechanism, per-command keep-list.** A single helper in `cli/_common.py`:

   ```python
   def project(payload: dict, keep: Collection[str]) -> dict:
       """Project a success payload to its decision-relevant subset for --summary
       (context-rot defense, #349). Always preserves the 'ok' discriminator, stamps
       'summary': True so a consumer can tell a projected payload from a full one, and
       keeps only the listed keys that are present."""
       return {k: v for k, v in payload.items() if k == "ok" or k in keep} | {"summary": True}
   ```

   Each command defines its own `_SUMMARY_KEYS` tuple. **Keep-lists, not drop-lists** — so
   any future diagnostic field added to a result is excluded-by-default from the summary
   (the whole point for `research promote`).

   **Success payloads only.** `project()` is applied to the assembled success payload
   inside the command body, immediately before `emit(ok(...))`. The error envelope
   (`{"ok": false, "error": ...}`) is produced by the `@json_errors` decorator and never
   reaches `project()`, so `--summary` can never strip `error`/diagnostics from a failure.
   A consumer distinguishes projected from full by the `summary: true` marker.

### Per-command summary content

- **`backtest walk-forward`** — keep: `strategy`, `data_source`, `snapshot_id`,
  `timeframe`, `seed`, `period`, `windows`, `holdout_frac`, `stability`, `code_hash`,
  `dependency_hash`, `config_hash`, `universe_name`, `universe_snapshots`,
  `fundamentals_snapshot`, `news_snapshot`, `mlflow_run_id`. Drops the bulky per-window
  `window_metrics` list (its scalar summary is `stability`). `mlflow_run_id` is kept so a
  `--summary --track` run still surfaces the trace handle. The holdout is already withheld
  from this command and that is unchanged.

- **`backtest sweep`** — keep: `strategy`, `n_combos`, `rank_by`, `best`,
  `trial_sharpe_count`, `trial_sharpe_mean`, `trial_sharpe_var_ann`, `recorded_breadth`,
  `code_hash`, `dependency_hash`, `universe_name`, `universe_snapshots`,
  `fundamentals_snapshot`, `news_snapshot`, `mlflow_run_id`. Drops the per-combo `ranked`
  list and the `grid`. `best` is the single headline combo.

- **`research promote`** — keep the decision essence: `promoted`, `strategy`, `passed`,
  `checks`, `n_combos`, `n_funnel`, `breadth_provenance`, `base_min_holdout_sharpe`,
  `effective_min_holdout_sharpe`, `pit_ok`, `pit_override`, `dsr_binding`, `fdr_binding`,
  `regime_robustness_binding`, `holdout`, `stability`, `config_hash`, `snapshot_id`,
  `universe_name`, `universe_snapshots`, `fundamentals_snapshot`, `news_snapshot`,
  `holdout_reuse`. Drops the ~25 deep `dsr_*` internals, `fdr_*` internals,
  `per_regime_sharpes`, `n_regimes_*`, `regime_method`, and the shadow-audit fields
  (`haircut_would_have_blocked`, `phase3_component_mask`, `own_lifetime_combos`,
  `windowed_total_combos`, `funnel_window_days`, `returns_available`, `dsr_n_eff`, etc.).
  `checks` already carries each gate's name/value/threshold/pass — the decision summary.

The projection is applied as the LAST step before `emit(...)`, after all payload assembly
(including `--track` mlflow run id and sweep's `--top` capping), so a `--summary` run still
records exactly what a full run does — only the printed surface shrinks.

### Out of scope

- `backtest run`, `factor eval`, the paper/live emits — the issue names walk-forward,
  sweep, and research promote as the heavy context-rot offenders; keep scope tight.
- A generic `--fields` projector (fork 1).
- Changing the default output to summary (fork 2).

## Testing

Per command (typer `CliRunner`, demo provider where possible):

1. `--summary` output **omits** the bulky/deep keys (`window_metrics` / `ranked`+`grid` /
   the `dsr_*` internals) and **retains** the decision keys (`stability` / `best` /
   `checks`+`promoted`), carries `ok: true` and `summary: true`, and a kept scalar matches
   the full run (cross-check, e.g. `stability` / `best` equal between full and summary).
2. Default (no `--summary`) output is unchanged — the bulky/deep keys are present and there
   is no `summary` marker.

## Docs

- Note `--summary` on the relevant CLAUDE.md command-surface lines.
- Note `--summary` in the `operating-algua` skill (`.codex/skills/operating-algua/SKILL.md`).
