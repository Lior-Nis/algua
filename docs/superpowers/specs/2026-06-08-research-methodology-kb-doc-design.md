# Research-methodology / anti-leakage standards doc (issue #138)

**Status:** design approved, pre-implementation
**Date:** 2026-06-08
**Issue:** #138 — kb: data-science methodology / anti-leakage standards doc
**Type:** kb doc + skill references. **No code, no tests** (prose only).

## Problem

algua's data-science integrity rules are encoded as **code walls** (the `t→t+1` lag, PIT
universe masking, single-use holdout burn, per-strategy deflated-Sharpe haircut) but are **not
articulated anywhere as standards the agent reads**. The authoring/operating/interpreting agent
has no methodology reference explaining the leakage vectors, why the walls exist, or how to design
for generalization.

This doc is the **judgment layer that sits *above* the walls** — prevention, not a second copy of
the enforcement. The walls are a backstop that catches only the *enumerable* leaks they can check;
this doc covers the *unbounded judgment space* the walls structurally cannot.

## Governing discipline (the reason this doc exists)

1. **Never restate a wall as if it were the enforcement.** The doc points to the code wall as the
   non-negotiable floor; it does not re-implement the rule in prose as though prose enforced it.
2. **Never drift from / lie to the agent.** If the doc claims a control the code has not built, it
   is lying. Anything walled: **the code is the source of truth and the doc only explains it.**
3. **Never restate a wall as optional.** Advisory tone applies to the *judgment* layer, never to
   the walls themselves.

## Decisions (settled in brainstorming + GATE-1 design review)

- **Standalone doc**, not co-located with #136. Create the `kb/principles/` domain with one note;
  reserve a dangling `[[risk-conventions]]` link for #136, explicitly marked "not yet authored."
  (GATE-1 #9: co-location is not required by anything.)
- **Cite walls by `path::symbol`, not `file:line`.** Line numbers rot on the first reformat;
  symbol names survive refactors. (GATE-1 #8.) The enforced-vs-aspirational **table** is still the
  structural drift-guard the user asked for — it just anchors on stable symbols.
- **One pointer per skill, never a copy of the table.** Duplicating the wall table into three
  skills multiplies the drift surface. (GATE-1 #10.)

## Deliverables

### 1. `kb/principles/research-methodology.md` (new)

Plain hand-authored Obsidian-vault prose (H1 + sections, **no frontmatter, no synced/generated
blocks** — unlike `kb/strategies/`). Sections:

1. **Where this sits** — prevention vs. the code-wall backstop; advisory above the floor; the three
   discipline rules above in one tight paragraph.

2. **The walls are the floor (enforced today)** — the explicit table. Columns: *leakage vector →
   how the platform blocks it → the code wall (`path::symbol`) → where the wall stops*. Rows
   (corrected per GATE-1):
   - **Execution-timing look-ahead** (GATE-1 #4 — renamed from "look-ahead" to scope it precisely)
     → central `t→t+1` decision lag; optional `compute_weights_panel` guarded by a fail-closed
     parity check → `algua/backtest/engine.py::simulate` (shifts by `decision_lag_bars`),
     `engine.py::_assert_parity`; contract floor `ExecutionContract` (`decision_lag_bars >= 1`) →
     **stops at execution timing only**; feature/data leakage is NOT covered here (see §4).
   - **Survivorship** → point-in-time universe as-of masking → engine enforcement
     `algua/backtest/engine.py::_members_as_of`, `engine.py::_decision_weights`; CLI resolution +
     provenance `algua/cli/_common.py::resolve_universe_inputs`, `universe_snapshots` →
     **opt-in only**: absence of `--universe` selects static mode (PIT-by-default is NOT built — §3).
   - **Holdout peeking** → `holdout_metrics` is withheld from every command except `research
     promote`, which reveals AND burns it (single-use), audited override `--allow-holdout-reuse`
     (GATE-1 #2 — corrected flag name) → `algua/backtest/walkforward.py` (SENSITIVE field; stripped
     by `algua/cli/backtest_cmd.py::walk_forward_cmd` and `algua/tracking/mlflow_tracker.py`),
     burn identity in `algua/cli/research_cmd.py::promote`
     (`overlapping_holdout_evaluations` → `record_holdout_evaluation`) → **stops at re-evaluation of
     the same window**; nothing stops you forming a new hypothesis after seeing the burn result.
   - **Multiple testing (per-strategy only)** → deflated-Sharpe haircut raising the holdout-Sharpe
     bar by `sqrt(2·ln N)·sqrt(ANN)/sqrt(T)`; breadth N **measured** from recorded sweep trials
     (preferred) or **declared** via `--n-combos` (audited as less trustworthy); promotion refuses
     if neither; degenerate holdout fails closed (`inf`) →
     `algua/research/gates.py::sharpe_haircut`, `gates.py::evaluate_gate`;
     `algua/cli/backtest_cmd.py::_record_search_breadth`, `research_cmd.py::_resolve_breadth` →
     **stops at "recorded sweep trials for *this strategy name*"** (GATE-1 #5) — it does NOT see the
     breadth of the whole research funnel (§3).

3. **Not yet enforced — judgment is the only guard (gaps, sibling #137)** — explicitly marked NOT
   built, so the agent never assumes coverage:
   - **Funnel-level multiple-testing deflation.** The haircut counts only one strategy's own combos.
     Across many hypotheses, family-wide / FDR control is *not* enforced by code — the agent must
     self-discipline (§7).
   - **PIT-by-default.** Static universe is the default; PIT must be opted into per run.
   - **Minimum-sample floor.** A per-window minimum (5 bars) and degenerate-holdout fail-closed
     exist, but there is no global minimum-sample gate.

4. **Leaks no wall can catch (the unbounded space)** — concrete, algua-specific examples
   (expanded per GATE-1 #6, #7):
   - Fitting feature normalization / ranking / thresholds over the **full sample** instead of a
     trailing window.
   - **Target leakage inside a custom feature** (e.g. a feature that peeks at a future bar).
   - **Panel fast-path leaks** (GATE-1 #6): `compute_weights_panel` operating on the whole period
     can compute full-sample normalizers, `shift(-1)`, or fitted thresholds; the parity guard
     proves it *equals* `compute_weights`, **not** that either is scientifically valid — a shared
     bug passes parity.
   - **`adj_close` / data-provenance leaks** (GATE-1 #7): using adjustment factors, vendor-restated
     bars, or split-adjusted history not knowable as of the decision date.
   - **Universe-selection leaks** (GATE-1 #7): selecting membership from price availability or
     "top symbols over the whole period" instead of the opt-in PIT universe.
   - **Peeking at the holdout "to debug."**

5. **Reading results honestly** — in-sample↔holdout gap as the overfitting smell; parameter
   stability (`pct_positive_windows`, `min_sharpe`); deflated vs raw Sharpe; small-sample
   skepticism. (Explains the *why* behind `interpret-results`; does not duplicate its tables.)

6. **Design for generalization** — prefer simple/robust signals; fewer parameters; regime-awareness
   (ties #125 dormancy); turnover/cost realism.

7. **Search-breadth honesty** — record every combo and hypothesis tried; never re-run a refuted
   idea (ties #126; in practice via the strategy/family doc `## Open questions` / `## Verdict`);
   why breadth raises the bar (links back to the §2 haircut row).

8. **Related principles** — dangling `[[risk-conventions]]` (reserved for #136, not yet authored).

### 2. `kb/README.md` (edit)

Register the `principles/` domain under "Domains": one note today
(`research-methodology.md`), hand-authored prose, **no tooling behind it** (distinct from the
CLI-managed `strategies/` domain).

### 3. Skill references — one pointer line each (`.codex/skills/`)

Reference the doc **by path** (`kb/principles/research-methodology.md`) — Obsidian `[[wikilinks]]`
resolve only inside the vault, not from the Codex skills.

- **`author-a-strategy/SKILL.md`** — near the look-ahead/purity rules: a pointer to read the
  methodology doc for the leakage vectors no wall catches (esp. the panel fast-path and `adj_close`
  cautions).
- **`run-the-research-loop/SKILL.md`** — in the Ideate + search-breadth steps.
- **`interpret-results/SKILL.md`** — in the Pitfalls section, AND **fix the existing inaccuracy**
  (GATE-1 #3): the skill currently claims `backtest walk-forward` output contains `holdout_metrics`.
  It does not — `walk_forward_cmd` strips it. Correct it to: `backtest walk-forward` exposes window
  metrics + stability only; `research promote` is the sole command that reveals and burns the
  holdout. This correction is mandatory — leaving it teaches the agent the opposite of the holdout
  wall.

`operating-algua` is intentionally **out of scope** (the issue names only the three research
skills).

## Out of scope / deferred

- #136 risk-conventions content (separate issue; only the dangling link here).
- #137 walls (funnel-level deflation, PIT-by-default, min-sample floor) — this doc *describes the
  gap*, it does not build the control.
- Any tooling/index for the `principles/` domain (YAGNI until there's more than one note).

## Verification

- Full quality gate stays green (no Python touched):
  `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`.
- Every `path::symbol` cited in the table resolves to a real symbol in the named file (manual check
  during authoring — all were verified against the current tree during design).
- The three skills each carry exactly one pointer (no copied table); `interpret-results` no longer
  claims `walk-forward` exposes the holdout.
