# Research-methodology kb doc (issue #138) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `kb/principles/research-methodology.md` — the data-science judgment layer above algua's code walls — and wire three research skills to it (fixing one existing inaccuracy).

**Architecture:** One hand-authored Obsidian-vault prose note in a new `kb/principles/` domain; cites each enforced wall by stable `path::symbol`; explicitly separates *enforced today* from *not-yet-built (#137)* from *unbounded leaks no wall catches*. Three Codex skills get a one-line pointer (by path); `interpret-results` additionally gets a factual correction. No Python, no tests.

**Tech Stack:** Markdown only. Verification via the repo quality gate (`pytest`/`ruff`/`mypy`/`lint-imports`, all unaffected) plus a manual `grep` that every cited symbol resolves.

> **Adaptation note (prose, not code):** there is no failing-test/passing-test cycle. Each task's "verify" step instead confirms cited anchors resolve and/or the file renders as intended. The quality gate is run once at the end (Task 6) since no Python is touched.

**Spec:** `docs/superpowers/specs/2026-06-08-research-methodology-kb-doc-design.md`

---

## File Structure

- **Create:** `kb/principles/research-methodology.md` — the methodology note (sole responsibility: the judgment layer above the walls).
- **Modify:** `kb/README.md` — register the new `principles/` domain.
- **Modify:** `.codex/skills/author-a-strategy/SKILL.md` — one pointer line.
- **Modify:** `.codex/skills/run-the-research-loop/SKILL.md` — one pointer line.
- **Modify:** `.codex/skills/interpret-results/SKILL.md` — one pointer line + fix the holdout-output inaccuracy.

All work happens on a worktree branch off `origin/main` (created by the build phase). The spec, this plan, and all deliverables are committed on that branch.

---

### Task 1: Create the methodology doc

**Files:**
- Create: `kb/principles/research-methodology.md`

- [ ] **Step 1: Write the file verbatim**

````markdown
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
authored, at the gate. This note is **prevention** — it covers the unbounded space of leaks no wall
can catch, which is exactly where the agent (the researcher most able to fool itself) does the
damage.

## The walls are the floor (enforced today)

| Leakage vector | How the platform blocks it | Code wall (`path::symbol`) | Where the wall STOPS |
|---|---|---|---|
| **Execution-timing look-ahead** | Central `t→t+1` decision lag; the optional `compute_weights_panel` fast path is guarded by a fail-closed parity check against the per-bar path. | `algua/backtest/engine.py::simulate` (shifts by `decision_lag_bars`), `algua/backtest/engine.py::_assert_parity`; contract floor `algua/contracts/types.py::ExecutionContract` (`decision_lag_bars >= 1`) | **Execution timing only.** It does NOT catch feature/data leakage inside your signal — see "Leaks no wall can catch". |
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
- **Panel fast-path parity-vs-validity trap.** `compute_weights_panel` runs over the whole period,
  so it is the easiest place to compute full-sample normalizers, `shift(-1)`, or fitted thresholds.
  The parity guard proves the panel output *equals* `compute_weights` — it does **not** prove either
  is scientifically valid. A bug present in *both* paths passes parity silently.
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

- [[risk-conventions]] — weight-space risk conventions *(reserved for issue #136; not yet authored)*.
````

- [ ] **Step 2: Verify every cited symbol resolves**

Run:
```bash
cd /path/to/worktree
for sig in \
  "algua/backtest/engine.py:def simulate" \
  "algua/backtest/engine.py:def _assert_parity" \
  "algua/backtest/engine.py:def _members_as_of" \
  "algua/backtest/engine.py:def _decision_weights" \
  "algua/contracts/types.py:ExecutionContract" \
  "algua/cli/_common.py:def resolve_universe_inputs" \
  "algua/cli/backtest_cmd.py:def walk_forward_cmd" \
  "algua/cli/backtest_cmd.py:def _record_search_breadth" \
  "algua/cli/research_cmd.py:def promote" \
  "algua/cli/research_cmd.py:def _resolve_breadth" \
  "algua/research/gates.py:def sharpe_haircut" \
  "algua/research/gates.py:def evaluate_gate" \
  "algua/tracking/mlflow_tracker.py:holdout_metrics" ; do
    f="${sig%%:*}"; p="${sig#*:}"; grep -q "$p" "$f" && echo "OK  $sig" || echo "MISS $sig"
done
```
Expected: every line prints `OK`. Any `MISS` means a citation is wrong — fix the doc before committing.

- [ ] **Step 3: Commit**

```bash
git add kb/principles/research-methodology.md
git commit -m "docs(kb): add research-methodology principles doc (#138)"
```

---

### Task 2: Register the `principles/` domain in the kb README

**Files:**
- Modify: `kb/README.md` (the "## Domains" list)

- [ ] **Step 1: Add the domain entry**

In `kb/README.md`, under `## Domains`, add a bullet immediately after the `strategies/` bullet (before the "Future domains" paragraph):

```markdown
- **`principles/`** — hand-authored methodology / standards notes (no tooling behind them, unlike
  `strategies/`). Start with `principles/research-methodology.md` — the data-science judgment layer
  above the code walls.
```

- [ ] **Step 2: Verify**

Run: `grep -n "principles/" kb/README.md`
Expected: the new bullet appears under Domains.

- [ ] **Step 3: Commit**

```bash
git add kb/README.md
git commit -m "docs(kb): register principles/ domain in README (#138)"
```

---

### Task 3: Pointer from `author-a-strategy`

**Files:**
- Modify: `.codex/skills/author-a-strategy/SKILL.md`

- [ ] **Step 1: Add a pointer bullet**

In `.codex/skills/author-a-strategy/SKILL.md`, find the `Rules that matter:` list under `## The contract`. Append this as the FINAL bullet of that list (after the "Keep weights sane" bullet):

```markdown
- **Read the methodology before authoring.** `kb/principles/research-methodology.md` covers the
  leakage vectors no wall catches — full-sample fitting, target leakage inside a custom feature,
  `adj_close`/provenance leaks, and the `compute_weights_panel` parity-vs-validity trap. The rules
  here are the floor, not the whole job.
```

- [ ] **Step 2: Verify**

Run: `grep -n "research-methodology" .codex/skills/author-a-strategy/SKILL.md`
Expected: one match.

- [ ] **Step 3: Commit**

```bash
git add .codex/skills/author-a-strategy/SKILL.md
git commit -m "docs(skills): point author-a-strategy at research-methodology (#138)"
```

---

### Task 4: Pointer from `run-the-research-loop`

**Files:**
- Modify: `.codex/skills/run-the-research-loop/SKILL.md`

- [ ] **Step 1: Extend the Ideate step**

In `.codex/skills/run-the-research-loop/SKILL.md`, in step `1. **Ideate.**`, find the sentence:

```
First read the knowledge base: `kb/strategies/_index.md` and
   `kb/strategies/_families.md`.
```

Replace it with:

```
First read the knowledge base: `kb/strategies/_index.md` and
   `kb/strategies/_families.md`, plus the research methodology
   `kb/principles/research-methodology.md` (leakage vectors, search-breadth honesty, designing for
   generalization).
```

- [ ] **Step 2: Verify**

Run: `grep -n "research-methodology" .codex/skills/run-the-research-loop/SKILL.md`
Expected: one match.

- [ ] **Step 3: Commit**

```bash
git add .codex/skills/run-the-research-loop/SKILL.md
git commit -m "docs(skills): point run-the-research-loop at research-methodology (#138)"
```

---

### Task 5: Pointer + holdout-output fix in `interpret-results`

**Files:**
- Modify: `.codex/skills/interpret-results/SKILL.md`

- [ ] **Step 1: Fix the walk-forward output inaccuracy**

In `.codex/skills/interpret-results/SKILL.md`, under `## What the outputs contain`, find the `backtest walk-forward` bullet, which currently reads:

```
- **`backtest walk-forward`** → `holdout_metrics` (`sharpe`, `total_return` on a reserved holdout
  segment) and `stability` (`pct_positive_windows` = fraction of the K out-of-sample windows with
  positive return; `min_sharpe` = the worst window's Sharpe). The **holdout is the headline
  evidence**; stability tells you whether performance is consistent or driven by one lucky window.
```

Replace it with:

```
- **`backtest walk-forward`** → `window_metrics` (per out-of-sample window) and `stability`
  (`pct_positive_windows` = fraction of the K windows with positive return; `min_sharpe` = the worst
  window's Sharpe). The **holdout is WITHHELD here** — `research promote` is the only command that
  reveals it (and burns it, single-use). Stability tells you whether performance is consistent or
  driven by one lucky window; the burned holdout (seen at promote time) is the headline evidence.
```

- [ ] **Step 2: Add a methodology pointer in Pitfalls**

In the same file, under `## Pitfalls to watch`, append this as the final bullet:

```markdown
- **The judgment layer.** `kb/principles/research-methodology.md` explains *why* these walls exist,
  the leakage vectors no wall catches, and how to read an in-sample↔holdout gap honestly.
```

- [ ] **Step 3: Verify**

Run:
```bash
grep -n "research-methodology" .codex/skills/interpret-results/SKILL.md
grep -n "WITHHELD here" .codex/skills/interpret-results/SKILL.md
grep -c "holdout_metrics" .codex/skills/interpret-results/SKILL.md
```
Expected: the pointer matches once; "WITHHELD here" matches once; `holdout_metrics` count is `0` in the walk-forward bullet's old form (the word may still appear elsewhere — confirm the walk-forward bullet no longer claims it as output).

- [ ] **Step 4: Commit**

```bash
git add .codex/skills/interpret-results/SKILL.md
git commit -m "docs(skills): fix holdout-output claim + point interpret-results at methodology (#138)"
```

---

### Task 6: Final verification

**Files:** none (verification only)

- [ ] **Step 1: Quality gate (must stay green — no Python touched)**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all pass (unchanged from main).

- [ ] **Step 2: Confirm the three skills each carry exactly one pointer**

Run: `grep -rc "research-methodology" .codex/skills/`
Expected: `author-a-strategy`, `run-the-research-loop`, `interpret-results` each report `1`.

- [ ] **Step 3: Confirm no stray uncommitted changes**

Run: `git status --short`
Expected: clean (everything committed across Tasks 1–5).

---

## Self-Review

- **Spec coverage:** doc (Task 1) ✓; README domain (Task 2) ✓; three skill pointers (Tasks 3–5) ✓; `interpret-results` holdout fix (Task 5) ✓; cite-by-symbol + enforced/aspirational/unbounded split (Task 1 content) ✓; standalone + dangling `[[risk-conventions]]` (Task 1) ✓; verification gate (Task 6) ✓.
- **Placeholders:** none — full doc and full edit strings are inline.
- **Consistency:** all `path::symbol` citations in Task 1 match the verification list in Task 1 Step 2 and were checked against the current tree during design.
