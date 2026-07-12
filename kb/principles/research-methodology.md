# Research methodology — designing for generalization, above the walls

This note is algua's **data-science judgment layer**. It sits *above* the code walls that enforce
research integrity. It is **advisory** — and it lives by three rules:

1. **It never restates a wall as the enforcement.** Where a control is walled in code, the code is
   the source of truth; this note only explains it and points at it.
2. **It never claims a control the code has not built.** If it says something is enforced, it is.
   The "Residual judgment gaps" section below is the honest line.
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
| **Survivorship** | Point-in-time universe: membership is masked as-of each decision date, with snapshot provenance recorded. | `algua/backtest/engine.py::_members_as_of`, `algua/backtest/engine.py::_decision_weights`; CLI resolution + provenance `algua/cli/_common.py::resolve_universe_inputs` | **Static backtests stay opt-out.** A raw backtest with no `--universe` runs a static universe — but the *agent* `research promote` gate REQUIRES `--universe` (non-PIT fails closed; `--allow-non-pit` is human-only), so no agent promotion can rest on a survivorship-exposed run. |
| **Holdout peeking** | `holdout_metrics` is withheld from every command except `research promote`, which reveals AND burns it (single-use). The audited override is `--allow-holdout-reuse`. | `algua/backtest/walkforward.py` (`holdout_metrics` is the SENSITIVE field; stripped by `algua/cli/backtest_cmd.py::walk_forward_cmd` and `algua/tracking/mlflow_tracker.py`); burn identity in `algua/cli/research_cmd.py::promote` (`overlapping_holdout_evaluations` → `record_holdout_evaluation`) | **Re-evaluating the same window.** Nothing stops you forming a *new* hypothesis after seeing a burn result, or peeking at the holdout "to debug". |
| **Multiple testing (funnel-wide)** | The holdout-Sharpe bar is deflated by `sqrt(2·ln N)·sqrt(ANN)/sqrt(T)`, where N is the **funnel-wide effective breadth** — the tighten-only max of this strategy's measured sweep trials, the windowed funnel total, and the family-lifetime breadth (#137/#222), **measured** from recorded trials (declared `--n-combos` is human-audited as weaker); promotion refuses if neither exists; a degenerate (0-bar) holdout fails closed (`inf`). A second, INDEPENDENT **LORD++ online-FDR** AND-check then requires the per-strategy DSR p-value to clear a running alpha-wealth level (`FDR_ALPHA=0.05`, `W0=0.025`). | `algua/research/gates.py::sharpe_haircut`, `::effective_funnel_breadth`, `::lord_plus_plus_level`, `::evaluate_gate`; applied in `algua/registry/promotion.py`; breadth recording `algua/cli/backtest_cmd.py::_record_search_breadth` | **The FDR AND-check binds only MEASURED breadth.** The funnel-wide haircut applies to *every* promotion (declared breadth just contributes the own-count input to N, which the 3-way `effective_funnel_breadth` max can still raise via windowed/family breadth), but the LORD++ FDR check needs a finite DSR p-value, so it binds only when breadth is measured — declared `--n-combos` (human-only) skips it. Binding is keyed on measured *provenance*, not the actor. A thesis deliberately split across families to dodge family-lifetime breadth is caught only by the ADVISORY `research family-audit` (#228), not a gate. |
| **Underpowered holdout** | A minimum holdout-observations floor — an OOS tail too short to carry statistical weight fails closed rather than passing on noise. | `algua/research/gates.py::MIN_HOLDOUT_OBSERVATIONS` (63), enforced in `evaluate_gate` / `algua/registry/promotion.py` | **A power floor, not a power guarantee.** 63 observations clears the floor; whether your period is *economically* long enough (regimes, turnover) is still judgment. |

## Residual judgment gaps — what the walls still do NOT cover

The three gaps this section once listed — funnel-level multiple-testing, PIT-by-default, and a
minimum-sample floor — are **now code walls** (built by #137, extended by #211/#220's LORD++ FDR and
#222's family breadth); see the table above. What the walls still leave to your judgment:

- **Declared breadth bypasses FDR.** The funnel-wide *haircut* applies to every run, but the LORD++
  FDR AND-check binds only **measured** breadth (it needs a finite DSR p-value). A `--n-combos`-declared
  count — which is human-only — skips the FDR ledger; the honesty of that declared breadth is then
  yours. Binding is keyed on measured *provenance*, not on the actor: a human who runs a measured
  sweep still enters the ledger.
- **Cross-family gaming is advisory-only.** Family-lifetime breadth (#222) is evaded by deliberately
  splitting one thesis across families. `research family-audit` (#228) *flags* such clusters but
  transitions nothing — acting on it is a human call, not a gate.
- **Everything in "Leaks no wall can catch" below.** The unbounded space — full-sample fitting,
  in-feature look-ahead, provenance leaks — is still prevention, not enforcement.

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
- **Raw vs PIT-adjusted vs restated — the price-provenance trichotomy.** The choice is NOT "raw
  (safe) vs adjusted (leaky)". It is **three-way**, and only the middle arm is correct:
  1. **PIT-correct adjusted** — split/dividend-adjusted *as-of the decision date* — is **CORRECT**.
     A price/return/momentum signal should read the point-in-time-adjusted series.
  2. **Future / vendor-restated adjusted** — adjustment factors, or restated history, that were
     **not knowable as of the decision date** — is the provenance **LEAK**: it smuggles the future
     in through the data, not the code.
  3. **RAW `close` / `volume`** — is corporate-action **CONTAMINATED**, **not** leak-avoidance.
     Reaching for raw to "avoid the adjustment leak" trades a leak for a correctness defect.
  Why raw is a defect, not a virtue: a split inside a trailing-return lookback window
  (`close[t]/close[t-N] - 1`) fabricates catastrophic **fake momentum** (a 4:1 split reads as a
  ~75% crash), and dividends distort the return; raw share **`volume`** jumps mechanically at a
  split (the share count changes), so a "volume surge" can be reading a corporate action, not flow.
  State it plainly: **raw close/volume is a correctness defect, not a leak-avoidance virtue.**
  What the platform provides **today** (honestly — this note never claims an unbuilt control):
  `adj_close` (see `docs/contracts/bar-schema.md`) is the split+dividend-adjusted price series, and
  it is what backtests use for returns — so price/return/momentum signals should read **`adj_close`,
  NOT raw `close`**. Two residual gaps remain **your judgment**, not a wall:
  (a) `adj_close` today reflects the **latest** snapshot's adjustment, not a true as-of PIT read —
  the bar schema's `get_bars` has **no `as_of` parameter yet** (deferred), so across data revisions
  it can carry mild restatement (arm 2 in miniature); and
  (b) there is **no adjusted-volume column** — raw `volume` still carries split discontinuities, so a
  volume signal that spans a split must normalize it deliberately or be understood as such.
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
