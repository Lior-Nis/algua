# Strategy Run Tracking — Slice 3 (the views) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **ALSO REQUIRED, for every task that draws anything:** load the `dataviz` skill before writing chart code. Its procedure (form → color → *validate with the script* → marks → accessibility → **render it and look at it**) is what this plan was written against; the palette decisions below are its validator's output, not opinion.

**Goal:** Turn the run ledger into the five preset views the spec promises, so the monitor answers "is any of this real?" at a glance instead of in paragraphs.

**Architecture:** Five presets, no axis pickers. `algua/registry` and the CLI are untouched — slice 2 already ships `runs list` / `runs show` / `runs series` and `/api/runs*`. All work is in `web/frontend/` plus one dev-only seeding script.

### Renderer split — decided before execution, and it decides what the tests can assert

**uPlot renders to `<canvas>`.** jsdom cannot inspect canvas contents, which is why the existing `EquityChart.test.tsx` asserts only on placeholders and footnote text — never on marks. Any test in this plan of the form "the diagonal is present" or "this point is above the line" is **unwritable against a canvas renderer**.

So:

| View | Renderer | Why |
|---|---|---|
| 2 IS-vs-OOS scatter | **inline SVG** | ~10–70 points. No library needed, marks are real DOM nodes, direct labels are trivial, and the geometry is assertable. |
| 3 Trial distribution | **inline SVG** | ~70 points + one threshold rule + one marker. Same reasoning. |
| 5 Gate bullet card | **SVG/CSS** | 11 rows of value-vs-threshold. Never needed a library. |
| 1 Sparkline | **inline SVG** | one tiny series per row. |
| 4 Return overlay | **uPlot** | a real multi-point time series — this is what uPlot is for, and the only place it earns its weight. |

**Consequence for Task 7 (the only canvas view):** follow `EquityChart.test.tsx`'s established pattern — assert on the empty/placeholder states, the labels, the small-multiples *structure* (panel count), and the prepared data handed to uPlot. Do **not** attempt to assert rendered marks.

**Do not add a charting dependency.** uPlot is already present; everything else here is hand-written SVG, which is less code than a library wrapper at these data sizes.

**Tech Stack:** React 19, TypeScript, uPlot 1.6, Vite, vitest. Standalone `web/` uv project for the backend.

**Spec:** `docs/superpowers/specs/2026-08-23-strategy-run-tracking-design.md` §6 (views), §7 (mobile-first), and the 2026-08-15 monitor spec for the brand guardrails.

**Base:** `origin/main` @ `cf2a20c` (slices 1 and 2 merged; schema v44).

## Global Constraints

- Frontend gate on EVERY task: `cd web/frontend && npm run check && npm run build`. When a task touches the backend, also `uv run --project web pytest web/backend/tests -q`.
- **Never add web dependencies to the ROOT project.** `web/` is a separate uv project on purpose — the root `uv.lock` is `dependency_hash` identity. The frontend's own `package.json` may gain nothing either: **uPlot is already there and is sufficient.** If a task seems to need a charting library, stop and report.
- `git add` scoped to named files — **never `git add -A`.**
- Commits end with: `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`
- **Read-only.** The monitor may show that something needs attention; it never acts.

### Brand guardrails (fixed — not this slice's to renegotiate)

Obsidian `#000000` ground; hairline Mist borders; **Electric `#3982ff` marks the ONE active thing**; semantic green/red/amber/violet are reserved for **status** and never become "series N"; Inter for text, IBM Plex Mono for numerics. Tokens live in `web/frontend/src/theme.css`.

### The series palette — computed, not chosen

The `dataviz` validator was run against the real tokens on the real surface. Results that bind this plan:

- A four-step **neutral** ramp **FAILS**: `#5d6675`↔`#727c8b` is ΔE **7.5** for *normal* vision (floor is 15). Four peer series cannot be built from this brand's neutrals.
- **`#3982ff` + `#727c8b` PASSES** every distinguishability check: lightness band (both inside L 0.48–0.67), CVD separation ΔE 17.6 protan / 11.1 tritan, normal-vision ΔE 18.0, contrast ≥3:1 on black. Its one FAIL is the chroma floor ("`#727c8b` reads gray") — intended: that check governs categorical *peer* slots, and this is figure/ground.

**Therefore, binding for this slice:**

1. **Multi-series = focus + context.** One focused run in Electric, one context run in `#727c8b`, **both direct-labelled at the line end**. Identity is never colour-alone.
2. **Three or more runs = small multiples**, one series per panel, monochrome — no categorical palette required.
3. **Do not introduce a new hue.** If a view seems to need a third series colour, it needs small multiples instead. Amending the brand kit is explicitly out of scope (the 2026-08-15 spec resolved that conflict in the brand's favour).

Add the two series tokens to `theme.css` as `--series-focus` and `--series-context` so no component hard-codes a hex.

### Mobile-first rules (spec §7, binding)

- **No chart may require hover to be readable.** Direct-label; no legend boxes.
- **Tap means navigate, never tooltip.**
- One chart per viewport width; no side-by-side small multiples — stack them.
- The container is `max-width: 680px` with **zero media queries**. Keep it that way.
- **Never a misleading zero baseline for an empty series.** An empty view renders an honest empty state, not an axis with nothing on it.

### The reality every view must handle first

The `runs` table on `main` has **0 rows**, and per spec Q8 there is **no backfill** — it fills only when the operator loop restarts. So:

- **Every view's FIRST rendered state is empty**, and that is the state the user will see on day one.
- An empty state must say *why* ("no runs recorded yet"), never draw axes around nothing, and never imply performance that does not exist.
- Task 1 exists so the other tasks can be built and *looked at* against plausible data.

---

## File Structure

| File | Responsibility |
|---|---|
| `scripts/seed_runs_dev.py` | **Create.** Dev-only. Seeds a SCRATCH db with plausible runs so the views can be developed and eyeballed. Never touches `data/algua.db`. |
| `web/frontend/src/theme.css` | **Modify.** Add `--series-focus`, `--series-context`. |
| `web/frontend/src/api.ts`, `src/types.ts` | **Modify.** Types + fetch for `/api/runs*`. |
| `web/frontend/src/components/ChartFrame.tsx` | **Create.** Shared shell: title, honest empty state, fixed height, no-hover guarantee. Every chart mounts inside it. |
| `web/frontend/src/components/ScatterISOOS.tsx` | **Create.** View 2. |
| `web/frontend/src/components/Sparkline.tsx` | **Create.** View 1's per-row mark (SVG, no library). |
| `web/frontend/src/components/RunList.tsx` | **Create.** View 1. |
| `web/frontend/src/components/GateBulletCard.tsx` | **Create.** View 5 (SVG/CSS, no library). |
| `web/frontend/src/components/TrialDistribution.tsx` | **Create.** View 3. |
| `web/frontend/src/components/ReturnOverlay.tsx` | **Create.** View 4 (focus + context; small multiples ≥3). |
| `web/frontend/src/screens/Research.tsx` | **Modify.** Becomes the runs surface (spec Q7). |
| `web/frontend/src/screens/StrategyDetail.tsx` | **Modify.** Gains views 3–5. |

---

## Task 1: Dev seeding, so the views can be looked at

**Files:** Create `scripts/seed_runs_dev.py`; Test: none (a dev tool — but it must be *safe*).

**Why first:** the `dataviz` procedure ends with "render it and look at it," and every later task's self-review depends on being able to. With 0 rows there is nothing to render.

- [ ] **Step 1: Write the script**

A CLI that seeds a scratch DB via the **public repository API only** (`SqliteStrategyRepository.record_run` etc.) — never raw INSERTs, so seeded rows obey the same validation as real ones and the seed cannot drift from the schema.

It must generate enough shape to exercise every view:
- ~8 strategies across 2 families, some registered, some not
- per strategy: a `backtest`, a `walk_forward`, a `sweep` + 5–8 `sweep_trial` children, and a `gate`
- metrics spread so the IS-vs-OOS scatter has points **on both sides of the diagonal** (the whole point of view 2 is showing mined vs honest)
- a few runs with NULL metrics, so empty/partial handling is exercised rather than assumed

- [ ] **Step 2: Make it impossible to point at the real DB**

**Hard requirement.** The script must refuse to run against the operator's registry. Resolve the target path and abort unless it is explicitly passed and is NOT `data/algua.db`. Print the path it will write to and require `--yes`. A seeding tool that can overwrite live strategy state is a genuine incident waiting to happen.

- [ ] **Step 3: Verify** — seed a scratch DB, then confirm via the real CLI:

```bash
ALGUA_DB_PATH=/tmp/slice3-dev.db uv run python scripts/seed_runs_dev.py --db /tmp/slice3-dev.db --yes
ALGUA_DB_PATH=/tmp/slice3-dev.db uv run algua runs list --limit 5
ALGUA_DB_PATH=/tmp/slice3-dev.db uv run algua runs list --kind sweep_trial --limit 5
```

- [ ] **Step 4: Root gate** (`uv run pytest -q` as its OWN Bash call with `timeout: 600000`; then ruff/mypy/lint-imports separately) — the script is in-repo, so it must lint and typecheck.

- [ ] **Step 5: Commit** — `chore(dev): seeding script for run-ledger view development`

**How to run the frontend against it:** point the backend at the scratch DB (`ALGUA_DB_PATH=/tmp/slice3-dev.db`) and run `npm run dev`. Record the exact working command in the commit message — every later task needs it.

---

## Task 2: `ChartFrame` + the series tokens

**Files:** Modify `web/frontend/src/theme.css`; create `src/components/ChartFrame.tsx`; test `src/components/ChartFrame.test.tsx`.

**Interfaces produced:** `<ChartFrame title, isEmpty, emptyLabel, height, children>`.

Every chart in this slice mounts inside it, so the honest-empty-state rule and the fixed-height/no-layout-shift rule are enforced in **one** place rather than remembered five times.

- [ ] **Step 1: Failing test** — renders the empty label and **no `<svg>`/`<canvas>`** when `isEmpty`; renders children when not; keeps the same height in both states (no layout shift when data arrives).
- [ ] **Step 2: Watch it fail.**
- [ ] **Step 3: Implement**, and add to `theme.css`:

```css
  --series-focus:   #3982ff; /* Electric — the ONE active series (dataviz validator: PASS) */
  --series-context: #727c8b; /* neutral context line; reads gray BY DESIGN (figure/ground) */
```

- [ ] **Step 4: Tests pass. Step 5: `npm run check && npm run build`. Step 6: Commit** — `feat(web): ChartFrame + validated series tokens`

---

## Task 3: View 2 — the IS-vs-OOS scatter

**Files:** create `src/components/ScatterISOOS.tsx` + test; modify `src/api.ts`, `src/types.ts`.

**This is the highest-value chart in the system.** One glance answers "is any of this real": points near the diagonal are honest, points far above it were mined. Build it first — everything after is additive.

- x = `mean_window_sharpe`, y = `sharpe_oos`, from `/api/runs?kind=gate`
- **Draw the y=x diagonal** — without it the chart means nothing
- Points: ≥8px (mobile tap target sizing, even though tap does nothing here)
- **No hover.** Direct-label only the outliers (furthest above the diagonal); never a label on every point
- Runs missing either metric are **excluded and counted** ("3 runs lack an OOS metric") — never plotted at zero

- [ ] **Step 1: Failing test** — diagonal present; a point above vs below is distinguishable in the DOM; a run with a NULL metric is excluded and reported, not plotted at 0.
- [ ] **Step 2: Watch it fail. Step 3: Implement. Step 4: Tests pass.**
- [ ] **Step 5: LOOK AT IT** — run the frontend against the seeded DB and confirm both regions are populated and legible at 390px. The `dataviz` procedure requires this step; say in the report what you saw.
- [ ] **Step 6: `npm run check && npm run build`. Step 7: Commit** — `feat(web): IS-vs-OOS scatter`

---

## Task 4: View 1 — ranked run list, and Research becomes the runs surface

**Files:** create `src/components/Sparkline.tsx`, `src/components/RunList.tsx` + tests; modify `src/screens/Research.tsx`.

Spec Q7: **Research becomes the runs surface.** Today it spends its largest panel on an idea pool with **zero rows** and renders the funnel as `<details>` lists — that is the weakest screen in the app occupying the right real estate.

New Research, top to bottom: funnel counts as a **one-line strip** → the IS-vs-OOS scatter (Task 3) → the ranked run list. Idea pool drops to a **collapsed count** until it has rows.

- Sort via a **chip row**, not column headers. Default `sharpe_oos`.
- Each row: strategy name · the sort metric (mono) · sparkline · pass/fail mark.
- Sparkline is inline SVG, **monochrome** — one series per row needs no palette.
- Tapping a row **navigates** to `StrategyDetail`. Never a tooltip.
- Honest empty state: "no runs recorded yet — the ledger fills when the operator loop runs."

- [ ] **Step 1: Failing tests** — chip changes sort order; NULL sort-metric rows sort last, never first; empty ledger renders the empty state and no table chrome; row tap navigates.
- [ ] **Step 2: Watch them fail. Step 3: Implement. Step 4: Tests pass.**
- [ ] **Step 5: LOOK AT IT** at 390px — report what you saw.
- [ ] **Step 6: `npm run check && npm run build`. Step 7: Commit** — `feat(web): ranked run list; Research becomes the runs surface`

---

## Task 5: View 5 — the gate bullet card

**Files:** create `src/components/GateBulletCard.tsx` + test; modify `src/screens/StrategyDetail.tsx`.

Replaces the densest text dump in the application. The 11 checks are natively a bullet chart: value against threshold, one row each.

- Source: `/api/runs/{id}` for a `gate` run — the checks arrive **allowlist-projected** (`GATE_DECISION_ALLOWLIST`); render what you are given and never request more.
- **Binding vs advisory must be visually distinct and not by colour alone** — advisory checks are the ones that compute but do not veto, and conflating them misreads the whole gate. Use position/weight/label, with status colour as reinforcement.
- Status colours (green/red) are legitimate here — this is pass/fail, exactly their reserved job.
- A check with a NULL value renders as "not evaluated", never as 0.

- [ ] **Step 1: Failing test** — a binding fail and an advisory fail are distinguishable **with colour stripped** (assert on text/attributes, not fill); a NULL-value check reads "not evaluated".
- [ ] **Step 2: Watch it fail. Step 3: Implement. Step 4: Tests pass.**
- [ ] **Step 5: LOOK AT IT. Step 6: `npm run check && npm run build`. Step 7: Commit** — `feat(web): gate bullet card`

---

## Task 6: View 3 — funnel trial distribution + the deflated bar

**Files:** create `src/components/TrialDistribution.tsx` + test; modify `src/screens/StrategyDetail.tsx`.

This renders the argument that kills most strategies — holdout Sharpe **0.025** against a deflated bar of **2.677** — as something inspectable rather than asserted. Nothing in a general-purpose tracker does this.

- **Funnel-wide, not per-sweep** (spec §6.1): a single sweep is 5–6 combos; the gate's `n_combos: 70` is accumulated funnel breadth. Source: `/api/runs?kind=sweep_trial`.
- At N≈70, prefer a **dot/strip plot over a histogram** — it shows individual trials honestly at this scale. Use judgement and say which you chose and why.
- Draw `effective_min_holdout_sharpe` as a **threshold rule**, and mark this strategy's holdout result.
- Both the rule and the marker are **direct-labelled** — the chart is meaningless if you cannot tell which line is the bar.

- [ ] **Step 1: Failing test** — threshold rule rendered and labelled; the strategy's own marker distinguishable from trial points; empty trial set renders the empty state, not an axis around nothing.
- [ ] **Step 2: Watch it fail. Step 3: Implement. Step 4: Tests pass.**
- [ ] **Step 5: LOOK AT IT. Step 6: `npm run check && npm run build`. Step 7: Commit** — `feat(web): funnel trial distribution with the deflated bar`

---

## Task 7: View 4 — return overlay (focus + context)

**Files:** create `src/components/ReturnOverlay.tsx` + test; modify `src/screens/StrategyDetail.tsx`.

**Read the "series palette" section of this plan before writing a line of this.** It is the only view with a multi-series problem and the palette answer is already computed.

- **1 focused run** in `--series-focus`; **1 context run** in `--series-context`; **both direct-labelled at the line end.**
- **3+ runs → small multiples**, stacked (never side by side), one monochrome series per panel.
- **Do not introduce a third series hue.** If it feels necessary, that is the small-multiples case.
- The in-sample curve comes from `/api/runs/series`. A run with a holdout leg gets its OOS interval drawn as a **shaded region**, labelled with its scalar OOS metrics from `/api/runs/{id}`.

**THE HARD RULE — do not undo a fix that took an integrity review to find.** `runs series` returns the holdout **interval and `n_bars` only**, never a per-bar OOS vector. `holdout_returns.returns_blob` is SENSITIVE (`algua/registry/db/holdout.py`): exposing a strategy's own OOS vector re-opens the single-use best-of-N surface the promotion gate depends on. **Region + scalar label is the honest ceiling.** If this view seems to want a plotted OOS curve, that is the feeling the rule exists to override — stop and report.

- [ ] **Step 1: Failing tests** — this is the **canvas** view, so assert the way `EquityChart.test.tsx` does: on placeholders, labels and structure, never on rendered marks. Cover: two runs produce two prepared series and both end-labels; three runs produce **three panels** (small multiples), not one chart with three series; the OOS leg is prepared as a region (interval), and **no array of per-bar OOS values reaches the component at all** — assert on the prepared data, which is stronger than a DOM check would have been.
- [ ] **Step 2: Watch them fail. Step 3: Implement. Step 4: Tests pass.**
- [ ] **Step 5: LOOK AT IT** — confirm the focus line reads as the active one and the context line recedes. Report what you saw.
- [ ] **Step 6: `npm run check && npm run build`. Step 7: Commit** — `feat(web): return overlay — focus plus context`

---

## Task 8: Verify the slice end to end

**Files:** none modified.

- [ ] **Step 1:** Frontend gate from clean: `cd web/frontend && npm run check && npm run build`. Backend suite: `uv run --project web pytest web/backend/tests -q`. Root gate: `uv run pytest -q` as its OWN Bash call with `timeout: 600000`, then ruff/mypy/lint-imports separately.
- [ ] **Step 2:** Run the frontend against the **seeded** scratch DB and walk all five views at 390px width. Report what each looked like.
- [ ] **Step 3:** Run it against an **EMPTY** DB and confirm every view renders its honest empty state — no axes around nothing, no zero baselines, no implied performance. **This is the state the operator sees on day one**, so it gets the same scrutiny as the populated case.
- [ ] **Step 4:** Confirm no per-bar OOS series appears anywhere in the rendered DOM.
- [ ] **Step 5:** Clean up the scratch DB; report `git log --oneline origin/main..HEAD` and confirm a clean tree.

---

## Out of scope

- **Any write action.** The monitor stays read-only.
- **Amending the brand kit.** Resolved in the brand's favour by the 2026-08-15 spec; this slice works within it.
- **New frontend dependencies.** uPlot is present and sufficient.
- **Deferred views:** parallel coordinates, Sankey funnel, cross-strategy correlation heatmap (spec §6).
- **Restarting the operator loop.** Operational, human-only — and until it happens these views are correct but empty.
