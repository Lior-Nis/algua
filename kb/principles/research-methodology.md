# Research methodology — designing for generalization, above the walls

This note is algua's **data-science judgment layer**. It sits *above* the code walls that enforce
research integrity. It is **advisory** — and it lives by three rules:

1. **It never restates a wall as the enforcement.** Where a control is walled in code, the code is
   the source of truth; this note only explains it and points at it.
2. **It never claims a control the code has not built.** If it says something is enforced, it is.
   The "Not yet enforced" section below is the honest line.
3. **It never frames a wall as optional.** The advisory tone is about *judgment*, never about the
   floor.

The walls are a **backstop**: they catch only the leaks they can enumerate, *after* a strategy is
authored — during evaluation and at promotion. This note is **prevention** — it covers the unbounded space of leaks no wall
can catch, which is exactly where the agent (the researcher most able to fool itself) does the
damage.

## The walls are the floor (enforced today)

| Leakage vector | How the platform blocks it | Code wall (`path::symbol`) | Where the wall STOPS |
|---|---|---|---|
| **Execution-timing look-ahead** | Central `t→t+1` decision lag; the optional `signal_panel` fast path is guarded by a fail-closed **weight-level** parity check against the per-bar `construct(signal(view), view)` path. | `algua/backtest/engine.py::simulate` (shifts by `decision_lag_bars`), `algua/backtest/engine.py::_assert_parity`; contract floor `algua/contracts/types.py::ExecutionContract` (`decision_lag_bars >= 1`) | **Execution timing only.** It does NOT catch feature/data leakage inside your signal — see "Leaks no wall can catch". |
| **Survivorship** | Point-in-time universe: membership is masked as-of each decision date, with snapshot provenance recorded. | `algua/backtest/engine.py::_members_as_of`, `algua/backtest/engine.py::_decision_weights`; CLI resolution + provenance `algua/cli/_common.py::resolve_universe_inputs` | **Opt-in only.** Absence of `--universe` selects a static universe; PIT is not the default (see "Not yet enforced"). |
| **Holdout peeking** | `holdout_metrics` is withheld from every command except `research promote`, which reveals AND burns it (single-use). The audited override is `--allow-holdout-reuse`. | `algua/backtest/walkforward.py` (`holdout_metrics` is the SENSITIVE field; stripped by `algua/cli/backtest_cmd.py::walk_forward_cmd` and `algua/tracking/mlflow_tracker.py`); burn identity in `algua/cli/research_cmd.py::promote` (`overlapping_holdout_evaluations` → `record_holdout_evaluation`) | **Re-evaluating the same window.** Nothing stops you forming a *new* hypothesis after seeing a burn result, or peeking at the holdout "to debug". |
| **Multiple testing (per-strategy only)** | The holdout-Sharpe bar is deflated by `sqrt(2·ln N)·sqrt(ANN)/sqrt(T)`. Breadth N is **measured** from recorded sweep trials (preferred) or **declared** via `--n-combos` (audited as less trustworthy); promotion refuses if neither exists; a degenerate (0-bar) holdout fails closed (`inf`). | `algua/research/gates.py::sharpe_haircut`, `algua/research/gates.py::evaluate_gate`; `algua/cli/backtest_cmd.py::_record_search_breadth`, `algua/cli/research_cmd.py::_resolve_breadth` | **N counts recorded sweep trials for THIS strategy name** — not the breadth of the whole research funnel (see "Not yet enforced"). |

## Not yet enforced — judgment is the only guard (gaps, sibling #137)

These are **not built**. Do not assume coverage:

- **Funnel-level multiple-testing deflation.** The haircut above counts only one strategy's own
  combos. Across many hypotheses, family-wide / false-discovery control is *not* enforced by code.
  The more ideas the funnel tries, the more one will pass by luck — that discipline is yours
  ("Search-breadth honesty" below).
- **PIT-by-default.** The static universe is the default; you must opt into the point-in-time
  universe per run. A run with no `--universe` is survivorship-exposed.
- **Minimum-sample floor.** A per-window minimum (5 bars) and the degenerate-holdout fail-closed
  exist, but there is no global minimum-sample gate. A multi-year period is on you.

## Leaks no wall can catch (the unbounded space)

A wall can only enforce what it can enumerate. These do not trip any wall — only a
methodologically-aware author avoids them:

- **Full-sample fitting.** Computing a feature's normalization, ranking, or threshold over the
  *whole* sample instead of a trailing window leaks the future into every bar.
- **Target leakage inside a custom feature.** A feature that reads a future bar (e.g. a forward
  return, an off-by-one `shift`) is look-ahead the execution lag never sees.
- **Panel fast-path parity-vs-validity trap.** `signal_panel` — the optional scores twin of
  `signal` — runs over the whole period, so it is the easiest place to compute full-sample
  normalizers, `shift(-1)`, or fitted thresholds. The **weight-level** parity guard proves the
  panel-derived weights *equal* the per-bar `construct(signal(view), view)` weights — it does
  **not** prove either is scientifically valid. A bug present in *both* paths passes parity silently.
- **`adj_close` / data-provenance leaks.** Using adjustment factors, vendor-restated bars, or
  split-adjusted history that was not knowable as of the decision date smuggles the future in
  through the data, not the code.
- **Universe-selection leaks.** Choosing membership from price availability or "the top symbols
  over the whole period" is survivorship by hand — use the opt-in PIT universe instead.
- **Peeking at the holdout "to debug".** The burn is single-use for a reason; one look contaminates
  it as out-of-sample evidence.

## Reading results honestly

- **In-sample ↔ holdout gap** is the overfitting smell. A backtest that sparkles in-sample and
  collapses out-of-sample was fit to noise.
- **Parameter stability** matters more than a single peak: `pct_positive_windows` and the worst
  window's `min_sharpe` tell you whether the edge is consistent or one lucky window.
- **Deflated vs raw Sharpe.** After a wide search, the raw winner is inflated; the deflated bar is
  the honest one. A thin margin over the deflated bar is weak evidence.
- **Small samples lie.** Short or flat periods produce noisy metrics — prefer multi-year data.

## Design for generalization

- Prefer **simple, robust** signals over many-knobbed ones — fewer parameters, fewer ways to overfit.
- Be **regime-aware**: an edge that only exists in one regime should be understood as such (and may
  belong dormant rather than retired — ties #125).
- Be **turnover/cost realistic**: an edge that evaporates under realistic costs is not an edge.

## Search-breadth honesty

- **Record every combo and hypothesis you try.** The deflated bar can only correct for the breadth
  it is told about; an unrecorded search silently lowers the bar.
- **Never re-run a refuted idea.** Capture the verdict in the strategy/family doc (`## Open
  questions` / `## Verdict & next`) so the funnel moves forward, not in circles (ties #126).
- **Breadth raises the bar by design.** Every extra combo makes a lucky winner more likely, so the
  honest holdout it must clear is higher (the haircut row above).

## Related principles

- [[risk-conventions]] — weight-space risk conventions: sizing and protection in allocation space
  (inverse-vol sizing, drawdown-based weight decay, conviction sizing) — judgment above the #135 walls.
