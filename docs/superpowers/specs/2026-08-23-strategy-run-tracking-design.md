# Strategy run tracking — a W&B for strategies

**Status:** spec, locked via a `grill-me` session (2026-08-23).
**Builds on:** `2026-08-15-monitor-dashboard-redesign.md` (tab structure, ICP, brand guardrails —
all still binding). This spec replaces that spec's **Research** screen and adds the store behind it.

---

## 1. Why

The monitor is accurate and it is textual. There is exactly one chart in the entire application —
`EquityChart` (uPlot, equity + peak), on `StrategyDetail`. Every other screen is `micro-label`
headings, text rows, `MetricTile` scalars and `<details>` lists.

The obvious diagnosis — "add charts" — is wrong. Counted against `data/algua.db` on 2026-08-22:

| Table | Rows | Visualized today |
|---|---|---|
| `tick_snapshots` | **8** | the only chart |
| `paper_fills` / `paper_orders` | **0 / 0** | — |
| `forward_gate_evaluations` | **0** | — |
| `ideas` | **0** | a full Research panel |
| `backtest_returns` | 9 (each a full return series) | nothing |
| `holdout_returns` | 9 (series + bar dates) | nothing |
| `search_trials` | 10 | nothing |
| `gate_evaluations` | 9 (~50 scalars each) | text dump |
| `stage_transitions` | 38 | `<details>` list |

**The monitor is instrumented on the operational lane, and all the data is in the research lane.**
It charts the emptiest table in the database and text-dumps the richest ones. It reads as textual
because text is the honest rendering of eight data points.

A second finding, from the same pass: `algua/tracking/` defines a full `ExperimentTracker`
Protocol (`log_backtest`, `log_sweep`, `log_walk_forward`) with an MLflow adapter and a
`NoopTracker`. `--track` on `backtest run` defaults to **`False`**, the research-loop skill never
passes it, and **zero MLflow runs exist anywhere** — `mlruns/0` and all 32 `.runs/*/mlruns/0`
directories contain nothing but `meta.yaml`. Two months of autonomous research have recorded not
one tracked run.

Meanwhile the metrics are already computed. One row of `gate_evaluations.decision_json`:

```
holdout_sharpe         0.0250   >= 2.6765   FAIL (advisory)
holdout_return        -0.0320   >  0        FAIL (advisory)
pct_positive_windows   0.0      >= 0.60     FAIL (advisory)
min_window_sharpe     -0.7374   >= 0        FAIL (advisory)
min_holdout_obs      452        >= 63       PASS (binding)
holdout_sharpe_floor   0.0250   >  0        PASS (binding)
dsr_evidence           0.0964   >= 0.95     FAIL (advisory)
dsr_bootstrap          0.0013   >= 0.95     FAIL (advisory)
regime_robustness      null     >= 0        FAIL (advisory)
idiosyncratic_alpha   -0.0942   >= 0.30     FAIL (advisory)
```

plus ~40 more scalars (`market_beta`, `ir_residual_vol_ann`, `dsr_skew`, `dsr_raw_kurtosis`,
`n_combos`, `effective_min_holdout_sharpe`, per-regime Sharpes). That is already a run summary with
a `name / op / threshold / value / passed / advisory` metric table and provenance columns
(`code_hash`, `config_hash`, `dependency_hash`, `snapshot_id`, `universe_name`) stronger than what
a general-purpose tracker provides. It is stored in a TEXT column, so nothing can sort, filter or
compare across it.

**So this is not "build a tracking platform." It is "normalise a table we already own, then draw
it."** If this work grows a run daemon or a metric ingestion API, it has gone wrong.

## 2. Two layers, one edge

A strategy will eventually compose several trainable components — a sentiment model, a chart
model, possibly an LLM agent — alongside rules. Those components must be tracked. They must not be
tracked *here*.

| | MLflow — component layer | Algua run store — economic layer |
|---|---|---|
| Unit | one fit / train / eval of one component | one evaluation of one strategy against a data interval |
| Metric denomination | loss, AUC, IC, perplexity | money and risk — Sharpe, drawdown, vol, alpha |
| Schema | free-form key-value, by design | **fixed and versioned**, by design |
| Cardinality | thousands, disposable | dozens to thousands, each one costly |
| Deletable | yes | **no — it is an audit record** |
| Reliability | best-effort side effect | transactional |

The decisive reason for the split is not metric vocabulary. It is that **a strategy evaluation in
algua is not telemetry**: it burns a single-use holdout, mints a gate token, writes an FDR ledger
row, stamps family membership, and can advance a lifecycle stage. The codebase already refuses to
let a tracker near that — `backtest_cmd._record_tracking` is best-effort precisely so "a tracker
failure must NOT discard a completed evaluation," and `NoopTracker` refuses to invent a run id
because "fabricating one would make the payload claim a run succeeded when nothing was logged."
Putting the economic record inside a mutable, deletable artifact store would invert that guarantee.

**The edge is one-directional.** A strategy run records the component runs it consumed — MLflow run
ids / registered-model versions — and folds their artifact digests into its identity. MLflow never
learns that strategies exist. Consequence: MLflow can be wiped, or replaced with W&B, and every
strategy record stays valid and reproducible, because it kept the digest, not just the pointer.

### 2.1 What already exists (and what does not)

Issue #376 already built this edge. `algua/models/registry.py` plus a `ModelRef(name, version,
digest, training_as_of, provenance_digest)` pinned in the strategy config; `strategies/loader.py`
resolves the pin against the model registry and **fails closed** if digest, `training_as_of`, or
provenance digest mismatch. `BacktestResult.model_ref` carries it as sidecar provenance beside
fundamentals/news/delisting. `training_as_of` is a genuine training-window PIT pin.

The gap is **arity**. `strategies/base.py` enforces:

```
at most one of needs_fundamentals / needs_news / needs_model may be True
```

and `model_ref` is singular; `signal_panel` is additionally unsupported with `needs_model`. A
strategy composed of a sentiment model *and* a chart model *and* news cannot be expressed today.

Fixing that is out of scope here. It constrains this spec in one binding way: **a run's component
lineage is a list from day one**, never a single nullable column, so the singular assumption is not
baked into the run store as well.

## 3. Locked decisions

Each resolved one at a time, with a recommendation, in the grilling session.

| # | Decision | Resolution |
|---|---|---|
| Q1 | Subject | **Research** performance (backtest / holdout), not realized equity. Realized equity stays a thin, honest strip that is empty until it is not. |
| Q2 | What is a run | **Any evaluation**: `backtest`, `walk_forward`, `gate`, **and every `sweep_trial`** as a child run of a parent sweep. |
| Q3 | Metric representation | **Fixed, versioned vocabulary**, sample-suffixed. Free-form config JSON. Overflow key-value for the long tail. |
| Q4 | Store location | **Registry DB**, schema v42. Trial rows batched into ONE `executemany`, in its own transaction on the sweep's connection (not inside `search_trials`'s own transaction — see §4.4's known follow-up). |
| Q5 | Views | Ranked run list, IS-vs-OOS scatter, funnel-wide trial distribution, return-series overlay, gate bullet card. Defer parallel coordinates, Sankey funnel, correlation heatmap. |
| Q6 | Read surface | Three pure reads: `runs list` / `runs show` / `runs series`. Runs carry `derived_from` and `components[]`. |
| Q7 | Placement | **Research becomes the runs surface.** Idea pool demoted. Tab count stays four. |
| Q8 | Backfill | **None.** The store ships empty and accumulates natively. |

### 3.1 Why "no bare `sharpe`" is a correctness rule, not a naming preference

There are at least four Sharpes in this system and they are not the same number:

| Metric | Sample | Honesty |
|---|---|---|
| backtest `sharpe` | in-sample, full period | inflated by construction |
| sweep trial `sharpe` | in-sample, one grid point | **most inflated** — selected as a maximum |
| `holdout_sharpe` | out-of-sample, single-use | the real one |
| forward realized Sharpe | broker-clocked, live wall | the only one denominated in money |

A single sortable `sharpe` column ranks the most overfit number in the system at the top. That is
the precise failure the gate exists to prevent, reintroduced at the UI layer. Hence: every metric
name carries its sample (`sharpe_is`, `sharpe_oos`, `sharpe_realized`), and **runs are comparable
only within a sample class**. The UI enforces it — default sort is `sharpe_oos`, in-sample metrics
are visually demoted, and a cross-kind comparison on an in-sample axis must be asked for.

## 4. Schema (registry DB, v42)

Registry DB posture, measured: WAL, `busy_timeout=5000`, `SCHEMA_VERSION = 41`, migrations are an
idempotent bootstrap (`CREATE TABLE IF NOT EXISTS` + guarded `_add_missing_columns` ALTERs). Adding
tables is the established pattern. The DB is 1.8 MB.

### 4.1 `runs`

One row per evaluation. Columns, grouped:

- **Identity:** `id`, `kind` (`backtest` | `walk_forward` | `sweep_trial` | `gate`),
  `strategy_name`, `strategy_id`, `created_at`.
- **Lineage:** `derived_from` (JSON list of parent run ids — a gate run points at the backtest and
  walk-forward runs it evaluated; a `sweep_trial` points at its parent sweep), `components` (JSON
  list of component refs: model name / version / digest / `training_as_of`). **Both are lists.**
- **Provenance (mirrors `BacktestResult`):** `code_hash`, `config_hash`, `dependency_hash`,
  `data_source`, `snapshot_id`, `universe_name`, `fundamentals_snapshot`, `news_snapshot`,
  `delisting_snapshot`, `seed`, `timeframe`, `period_start`, `period_end`.
- **Config:** `config_json` — free-form. Grid points are heterogeneous by nature; freezing them is
  impossible and pointless.
- **Fixed metrics:** the closed vocabulary below, as real columns.
- **`metric_schema_version`** — so the vocabulary can evolve without silently changing what an
  existing chart means.

### 4.2 Fixed metric vocabulary v1

Derived from what `algua/backtest/metrics.py` already computes (`total_return`, `ann_volatility`,
`max_drawdown`, `sharpe`, `sortino`, `cagr`, `calmar`) and what `walk_forward` adds (`mean_sharpe`,
`std_sharpe`, `min_sharpe`, `pct_positive_windows`), re-expressed with explicit sample class:

```
sharpe_is          sharpe_oos          sharpe_realized
sortino_is         sortino_oos
total_return_is    total_return_oos
max_drawdown_is    max_drawdown_oos
ann_vol_is         ann_vol_oos
cagr_is            calmar_is
n_obs_is           n_obs_oos
mean_window_sharpe  std_window_sharpe  min_window_sharpe  pct_positive_windows
```

A metric absent for a run kind is NULL, never zero. `metrics.py` currently returns `sharpe: 0.0`
for a degenerate series; the run writer must map that to NULL rather than record a zero that sorts
above a genuine negative.

### 4.3 `run_metrics` (overflow)

`(run_id, key, value)` for the long tail — the ~40 DSR / IR / regime diagnostics and whatever a
future model-bearing run wants. Queryable, but explicitly **not** part of the fixed vocabulary and
**not** offered as a default chart axis.

### 4.4 `sweep_trial` persistence and the `MAX_N_COMBOS` bomb

`search_trials` today records `n_combos`, `grid_json`, and aggregate `trial_sharpe_mean` /
`trial_sharpe_var_ann` — the per-trial results are discarded. Under Q2 each trial becomes a run.

**Write pattern, as actually implemented:** one batched `executemany` insert of all trial rows, in
its OWN transaction, on the SAME connection that writes the `search_trials` row — NOT inside that
row's transaction. `sweep()` is parallelised across processes; 70 concurrent writers against the
governance DB is the wrong shape, so the trial-row batch is still one writer, one transaction, at
the end (this part holds). This is also an integrity gain: `n_combos` stops being an unverifiable
assertion and becomes a count you can `SELECT`.

**Known follow-up, deliberately not taken in this slice:** the trial-row batch and the
`search_trials` aggregate are two separate commits on one connection, not one atomic unit — a crash
between them can commit one without the other. Folding them into a single transaction would change
`record_search_trial`'s transaction semantics, and that function is a CODEOWNERS-protected
governance writer feeding the promotion gate's multiple-testing defense (the LORD++/FDR ledger).
That is not a tail-of-slice change, so it is left as a documented residual rather than silently
implied by an "inside the same transaction" claim this spec no longer makes.

**Bound:** `MAX_N_COMBOS = 1_000_000_000` exists for `SUM` overflow-safety in the family-lifetime
breadth seed, not as a realistic grid size. Harmless while a trial is a scalar; once trials are
rows it is a row-count bomb. The writer needs a real persistence cap — `MAX_PERSISTED_TRIALS`,
**default 10 000 per sweep**, a CODEOWNERS-protected constant alongside `MAX_N_COMBOS`. Beyond the
cap: keep the `search_trials` aggregate (which still governs breadth), stop writing trial rows, and
**record the truncation explicitly** on the parent sweep run as a `trials_truncated_at` field. A
silently truncated trial set would make the funnel-wide distribution lie about the breadth it
depicts, so view 3 must render a truncation notice rather than a partial histogram.

## 5. CLI surfaces

Three pure reads. No broker call, no writes, no locks. All emit JSON under the existing
`ok()` / `@json_errors` envelope, so `web/backend/algua_cli.py` consumes them unchanged, and all
are mounted at the composition root in `algua/cli/main.py`.

```
algua runs list  [--kind backtest|walk_forward|sweep_trial|gate]
                 [--strategy S] [--family F] [--sort METRIC] [--limit N]
    -> scalar rows only. Feeds the ranked list AND the scatter.

algua runs show  <run_id>
    -> one run: full metric set, config, gate checks, lineage.
       Feeds the gate card. Absorbs what `registry gates` does today,
       since under Q2 a gate evaluation IS a run.

algua runs series <run_id> [--run-id ...]
    -> return series for the named runs only. Feeds the overlay.
       NEVER returned by `list`.
       IN-SAMPLE backtest series only (per-bar, from `backtest_returns`). For a run with a
       holdout leg, returns the OOS interval (`holdout_start`/`holdout_end`) and `n_bars`
       ONLY — never a per-bar holdout vector. `holdout_returns.returns_blob` is SENSITIVE
       (see `algua/registry/db/holdout.py`'s DDL comment): the ONLY method allowed to read it
       is the sibling-only `overlapping_holdout_return_streams`, which explicitly never
       returns the requesting strategy's own vector. Handing a strategy its own per-bar OOS
       vector through a "get my own series" endpoint would re-open exactly the single-use
       best-of-N surface `sweep()`'s holdout burn exists to prevent — a scalar (`sharpe_oos`
       etc., already on the gate run row via `runs show`) leaks far less than the full vector.
```

The split is not stylistic. The ranked list and scatter want wide-and-shallow (N runs x M scalars);
the overlay wants narrow-and-deep (one run x hundreds of floats). One command returning both blows
up a payload that crosses a subprocess-and-JSON-parse seam — the lesson `--summary` (#349) already
encodes.

Backend, following the existing `run_cli(..., ttl_s=)` convention exactly:
`/api/runs` (60 s), `/api/runs/{id}` (30 s), `/api/runs/series?ids=` (60 s).

## 6. Views

Five ship. Each is preset — no axis pickers, no configuration.

1. **Ranked run list.** One sort metric at a time, chosen from a chip row; each row is name + the
   sort metric + a sparkline + pass/fail. Default sort `sharpe_oos`.
2. **IS-vs-OOS scatter.** `mean_window_sharpe` on x, `sharpe_oos` on y, diagonal drawn. Points near
   the diagonal are honest; points far above it were mined. **The highest-value chart in the
   system** and nearly free — one glance answers "is any of this real," which is currently
   unanswerable.
   The x-axis is deliberately the **walk-forward mean-window Sharpe**, not the full-period
   `sharpe_is`: it is the better predictor of out-of-sample survival and the more honest axis.
   `sharpe_is` is the more dramatic one and is available as an alternate x, never the default.
3. **Funnel-wide trial distribution + deflated bar.** All trials the funnel spent in the 90-day
   window, with this strategy's holdout result marked and `effective_min_holdout_sharpe` drawn as
   a line. This renders the argument that killed the most recent gate evaluation — holdout Sharpe
   0.025 against a deflated bar of 2.677 — as something inspectable rather than asserted. Nothing
   in a general-purpose tracker does this, because none of them have a concept of breadth
   deflation.
4. **Return-series overlay.** In-sample backtest curve, up to ~4 runs overlaid; a run with a
   holdout leg gets its OOS interval drawn as a **shaded region** (start/end from `runs series`)
   labelled with its scalar OOS metrics (`sharpe_oos` etc., from `runs show`) — not a plotted
   per-bar OOS curve. `holdout_returns.returns_blob` is a SENSITIVE single-use out-of-sample
   vector (`algua/registry/db/holdout.py`); no "get my own vector" read is allowed to exist,
   including this view's, because it would re-open the single-use best-of-N surface `sweep()`'s
   holdout burn exists to prevent. Region + scalar label is the honest ceiling here — do not
   "restore" a per-bar OOS plot.
5. **Gate bullet card.** The 11 checks as horizontal bullet bars, value against threshold, binding
   and advisory visually separated. Replaces the densest text dump in the application.

**Deferred, with reasons:** parallel coordinates (with grids of 5-6 combos, view 3 says more per
pixel); Sankey funnel (38 transitions render fine as a list; a Sankey is decoration at this
volume); cross-strategy correlation heatmap (not yet a decision input — it becomes one when there
is a live book).

### 6.1 Scale reality

Per-sweep grids measured today are small — `{"lookback": [60,90,120,180,252]}` is 5 combos,
`{"disagreement_penalty": [.25,.5,.75], "short_lookback": [126,189]}` is 6. The gate's
`n_combos: 70` is **funnel-wide accumulated breadth** (`own_lifetime_combos 24`,
`windowed_total_combos 64`), not one sweep's grid. View 3 is therefore funnel-wide by construction;
a per-sweep rendering would be six dots.

## 7. Mobile-first

The app is already built this way: a `max-width: 680px` container with **zero media queries**.

A general-purpose tracker is exploratory because ML researchers do not know which metric matters —
hence wide sortable tables, arbitrary axis pickers, hover tooltips. That question is already
answered here: does out-of-sample Sharpe survive the breadth tax. The mobile constraint therefore
forces the views to be **conclusive rather than exploratory**, which is the better instrument
regardless of screen size.

**Hard rules:**

- **No chart may require hover to be readable.** Direct-label lines and outliers; no legends.
- **Tap means navigate, never tooltip.**
- One chart per viewport width. No side-by-side small multiples.
- The runs *table* dies; the ranked *list* survives. Multi-column tables are a desktop luxury and
  the semantics were always a ranking.

**Charting:** no new dependency. `uplot@^1.6.32` is already in `web/frontend/package.json` and
covers scatter (`paths.points`) and histograms (bars). Bullet bars and sparklines are SVG/CSS — a
library would outweigh what it draws.

**Unsolved, and real design work rather than a token pick:** there is no categorical series
palette. Electric `#3982ff` is reserved as the rare "one active thing" signal; green/red/amber/
violet are reserved for status; Slate is non-text only. A 4-series overlay currently has no legal
colours. This must be solved against the brand tokens during slice 3, not hand-waved.

## 8. Slices

| # | Slice | Gate |
|---|---|---|
| 1 | v42 schema (`runs`, `run_metrics`), the fixed vocabulary, and the **write path** — backtest / walk-forward / gate / batched sweep trials, with the truncation cap | Root pytest, ruff, mypy, lint-imports |
| 2 | `runs list` / `runs show` / `runs series` + `/api/runs*` endpoints | Root gate + web backend pytest |
| 3 | Research screen rebuild: ranked list, scatter; `StrategyDetail` gains gate card, trial distribution, overlay. Includes solving the series palette | Frontend check + build |

**Slice 1 is urgent, not merely first.** With no backfill (Q8), every day the write path is not
landed is a day of runs discarded. Slices 2 and 3 can lag and will then be built against real
accumulated data instead of nine reconstructed rows.

## 9. Out of scope

- **Any write action.** The monitor stays strictly read-only (inherited from the 2026-08-15 spec).
- **Backfill.** Deliberately none. Per-trial Sharpes were never persisted and are unrecoverable —
  only `grid_json` and mean/variance survived — so view 3 has no history to reconstruct in any case.
- **Multi-component strategies.** The `needs_fundamentals` / `needs_news` / `needs_model` mutual
  exclusion and the singular `model_ref` are a real gap (§2.1) but a separate piece of work. This
  spec only refuses to bake the singular assumption into the run store.
- **Turning MLflow on.** The component layer is not instrumented by this work.
- **Retention policy.** YAGNI at this volume, beyond the trial persistence cap.

## 10. A hazard to decide before the planned reset

The operator intends to reset the system after the current refactor. The DS-integrity guarantees
are enforced by **DB rows, not files**: wiping `algua.db` while keeping the strategy modules in
`algua/strategies/` resets `holdout_evaluations`, so a strategy that has already burned its
single-use holdout can be re-gated on the same data, and family breadth restarts at zero. That is
fine if the strategies are discarded alongside the DB. It is not fine if they are kept and
re-promoted. Decide deliberately rather than discovering it.
