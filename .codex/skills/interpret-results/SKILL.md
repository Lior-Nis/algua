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
  strategy can't peek — but if results look too good to be true, re-read the `target_weights` for
  any accidental use of current-bar information.
- **Tiny samples / degenerate periods.** Very short windows or flat data produce noisy metrics;
  prefer a multi-year `--start/--end`.
- **The judgment layer.** `kb/principles/research-methodology.md` explains *why* these walls exist,
  the leakage vectors no wall catches, and how to read an in-sample↔holdout gap honestly.

## Your recommendation

When delegated a results set, return a clear **promote** or **discard** with one or two sentences
of reasoning grounded in the holdout + stability numbers and the gate decision. You do not run
state-changing commands or author code — you judge.
