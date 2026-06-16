---
name: interpret-results
description: How to read algua backtest, walk-forward, sweep, and gate JSON — what the metrics mean, what "good" looks like, the promotion-gate criteria, and the overfitting/look-ahead pitfalls to watch. Use when judging results and deciding promote vs discard.
---

# Interpreting algua results

Every command emits JSON on stdout. Judge a strategy on **out-of-sample** evidence, not the
in-sample backtest. Be skeptical: a great single backtest is the easiest thing to produce by
accident.

## What the outputs contain

- **`backtest run`** → `metrics`: `sharpe`, `total_return`, `max_drawdown`, plus `config_hash`,
  `snapshot_id`, data provenance. This is **in-sample** — necessary, not sufficient.
- **`backtest walk-forward`** → `window_metrics` (per out-of-sample window) and `stability`
  (`pct_positive_windows` = fraction of the K windows with positive return; `min_sharpe` = the worst
  window's Sharpe). The **holdout is WITHHELD here** — `research promote` is the only command that
  reveals it (and burns it, single-use). Stability tells you whether performance is consistent or
  driven by one lucky window; the burned holdout (seen at promote time) is the headline evidence.
- **`backtest sweep`** → ranked combos by `mean_sharpe` or `min_sharpe`. Useful for exploration,
  but see "search breadth" below.

## The promotion gate (what `research promote` checks)

The gate passes only if **all** hold (defaults in `algua/research/gates.py`):
- `holdout_sharpe >= 0.5`
- `holdout_total_return > 0.0` (strict)
- `pct_positive_windows >= 0.6`
- `min_window_sharpe >= 0.0` (the worst window isn't badly negative)

A pass is a genuine signal of out-of-sample robustness; a fail is informative — read which check
failed. **Never lower these thresholds to force a pass.** If a strategy fails, discard it (or form
a better hypothesis) — that's the gate doing its job.

## What "good" looks like

- Holdout Sharpe comfortably above the threshold, not scraping it.
- Most windows positive AND the worst window not deeply negative (consistency, not one big win).
- Holdout performance in the same ballpark as in-sample (a large in-sample/holdout gap is an
  overfitting smell).

## Pitfalls to watch

- **Search breadth / overfitting.** The more parameter combos you sweep, the more likely one looks
  good by chance. Record the number searched (`promote --n-combos K`); treat a thin holdout margin
  after a wide search as weak. The holdout is your defense against search-induced overfitting.
- **Look-ahead.** The engine enforces the `t→t+1` decision lag centrally, so a *correctly authored*
  strategy can't peek — but if results look too good to be true, re-read `compute_weights` (and
  `compute_weights_panel` if present) for any accidental use of current-bar information.
- **Tiny samples / degenerate periods.** Very short windows or flat data produce noisy metrics;
  prefer a multi-year `--start/--end`.
- **The judgment layer.** `kb/principles/research-methodology.md` explains *why* these walls exist,
  the leakage vectors no wall catches, and how to read an in-sample↔holdout gap honestly.

## Reading `algua factor eval` IC (issues #140 slice B + #219 slice E)

`algua factor eval <factor>` evaluates a single catalogued factor on its own: a PIT backtest (via
a 1-factor→weights adapter) plus a construction-free **IC** block. In the `ic` block:

- `mean_ic` — average per-bar cross-sectional rank (Spearman) correlation between the factor's
  score and the forward return. ~0 means no monotone cross-sectional predictiveness.
- `ir = mean_ic / ic_std`; `t_stat = ir * sqrt(n_obs)`; `hit_rate` — share of bars with IC > 0.
- `n_obs` — usable bars. A stable small-but-positive IC over many observations is more credible
  than a large IC over few bars — always check `n_obs` (a `None` mean_ic means too few usable bars).

**`fdr_corrected: true` (#219 slice E).** Each evaluation is recorded in the `factor_evaluations`
ledger. The JSON now carries a top-level `fdr` block with the multiple-testing-corrected verdict:

- `n_hypotheses` — effective funnel breadth (distinct factor hypotheses evaluated in the rolling
  90-day window). Higher N inflates the expected-max t-stat benchmark.
- `breadth_benchmark_t` — `sqrt(2·ln N)` expected-max inflator on the t-stat.
- `breadth_significant` — whether `t_stat >= z_alpha + breadth_benchmark_t`.
- `dsr_binding` / `dsr_confidence` — DSR (Bailey–LdP) layer binds when N≥2 and pooled IR
  dispersion is measurable. Tighten-only AND with the breadth check.
- **`significant`** — the honest verdict. This is the field to act on; the raw `t_stat` is
  inflated by search effort and should not be used in isolation.
- `n_dependents` — strategies that compose this factor (blast radius). A factor with many
  dependents that fails FDR is a systemic exposure, not just one bad idea.

A factor never goes live — only a *strategy* that composes it does, through the normal
walk-forward/holdout gates. `significant: false` means the factor has not yet demonstrated
edge beyond the expected noise of the search process; add more out-of-sample evidence or
discard it.

## Your recommendation

When delegated a results set, return a clear **promote** or **discard** with one or two sentences
of reasoning grounded in the holdout + stability numbers and the gate decision. You do not run
state-changing commands or author code — you judge.
