# Design — PIT-correct adjusted prices & volume, not raw or restated (#521)

**Status:** Design — **round 6** (GATE-1 pending re-review). Rounds 1–3 blocking findings resolved;
round-4/5 closed 5 HIGH + 4 MEDIUM integration/provenance/panel seams. **Round-6 Codex (gpt-5.5,
read-only adversarial pass) verdict BLOCK, 3 HIGH + 3 MEDIUM** — the enforcement was still opt-in
(leaky by default), had no real data path, and under-audited the waiver. This revision closes them;
see "Round-6 findings" at the end. **The load-bearing round-6 change: the raw price/volume
withholding is now DEFAULT-ON for every research strategy view, not gated on the `needs_corpactions`
flag** — the `whale_volume_momentum` raw-close defect is thereby fixed unbypassably for a strategy
that declares *nothing*, which is exactly the "no flag declared" leak round-5 left open.
**Date:** 2026-07-12. **Issue:** #521 (`[ds]`, severity: high).

### Implementation acceptance criteria (carried + round-6)
- **Raw price/volume is withheld from the research strategy view BY DEFAULT (round-6 H1).** The
  runtime wall does **not** depend on a strategy declaring `needs_corpactions`: raw
  `open/high/low/close/volume` are dropped from *every* strategy `view` the research/backtest lane
  serves, and the split-invariant derived `dollar_volume` is added to *every* view. A strategy that
  declares nothing therefore **cannot read raw `close`/`volume` at all** — the #521 defect is
  unrepresentable on the default path, not merely undeclared. `needs_corpactions` is the *upgrade*
  that additionally swaps the ratio-only `adj_close` for the full PIT-correct `pit_adj_*` level basis
  and withholds `adj_close` too (see §3, §4).
- `knowable_at` MUST be source-derived (vendor declaration/announcement date). If a source lacks it,
  **fail closed** — never synthesize `knowable_at = ex_date`. This forces a **NEW** parser + schema
  object that actually carries `knowable_at` (Slice 2.1); the existing `Split`/`Dividend`/
  `parse_databento_corp_actions` have **no such field** and are NOT reused for the PIT lane.
- **A concrete declaration-date-bearing corpaction SOURCE + importer is an acceptance criterion, not
  a Non-goal (round-6 H2).** Slice 2 ships `data import-corpactions --file <csv>` accepting a
  strictly-specified PIT corpaction CSV (schema pinned in §2.1 / round-6 M6) so the lane has a real,
  ingestible data path *today* for any declaration-bearing source (a curated CSV or a future vendor
  feed). The design ALSO documents honestly (§ "Real-data reach") that the specific in-tree Databento
  auto-feed carries no declaration column, so the PIT-*level* half of the defect on that feed stays
  unresolved until a declaration-bearing Databento source lands — while the raw-contamination half of
  the defect is fixed end-to-end on Databento data *today* by the default withholding above (which
  needs no corpaction snapshot).
- `pit_adj_open`/`pit_adj_high`/`pit_adj_low` MUST use the exact same per-event factor path as
  `pit_adj_close` (a whole bar scales uniformly), verified by test. The single canonical function
  that produces all of them is **`build_pit_price_volume_view` (round-6 M4)** — see §2.2.
- The human waiver MUST relax **both** enforcement layers together (re-expose the withheld columns
  AND skip the promote-time AST scan); neither alone. It does **not** re-enable the `signal_panel`
  fast path for a `needs_corpactions` strategy (see §3 / §4). **The waiver state is durable, audited
  provenance (round-6 H3):** stamped on `BacktestResult`/`WalkForwardResult`/`SweepResult`, the CLI
  JSON, and `gate_evaluations.decision_json`, and surfaced in the go-live signature challenge/context
  so a raw-contaminated candidate is distinguishable from a clean PIT-only one at every downstream
  gate (§4 Waiver provenance).
- A `needs_corpactions` strategy **forbids `signal_panel`** (fail-closed at load) and runs the
  canonical per-bar loop only — so both the backtest fast path and the promote-time
  `verify_signal_panel_parity` gate are disabled for it (§3, §4, finding H1/H4).
- The corpaction snapshot id is **first-class provenance**: threaded through
  `BacktestResult`/`WalkForwardResult`/`SweepResult`, the CLI JSON projections, and the
  `gate_evaluations` audit row, mirroring the #132 fundamentals/news stamping (finding H3).
- **Snapshot presence is a RUNTIME precondition checked at each entry point, NOT a load-time
  `StrategyConfig` check (round-6 M5).** `base.py`/`loader.py` govern only static config coherence
  (the `signal_panel` forbid + flag well-formedness). Requiring a corpaction provider/snapshot for a
  `needs_corpactions` strategy is enforced fail-closed at every runtime entry point — CLI
  `backtest run`/`walk-forward`/`sweep`, `promotion_preflight`, and the paper/live load guards (§3).

## The defect (verbatim scope)
A supervised research strategy (`whale_volume_momentum`, 2026-07-12) used raw `close`/`volume`
*"deliberately, to avoid `adj_close` provenance leakage"* and the run-report framed that as a
virtue. It is a correctness DEFECT: raw-close momentum over a split fabricates fake momentum, and
raw share-volume mechanically jumps at a split. The authoring guidance actively steers toward it.

## GATE-1 resolution: this design commits to (B), not (A)
GATE-1 (Codex) BLOCKED the issue-as-written because its item 3 ("provide/point to the right
accessor") is satisfiable by a **docs-only relabel** that points authors at the existing
`adj_close` column — which is **not** PIT-correct. This design **rejects (A) and specifies (B): a
genuine point-in-time adjustment accessor.** The rejection is load-bearing, so state the honest
model first.

### The honest three-way model (this is the correctness core)
For a decision made at date **T**, a price/volume series is one of:

1. **PIT-forward-adjusted (✅ correct).** Each bar `t ≤ T` reflects **only** corporate actions with
   `ex_date ≤ t`. Nothing after the bar's own date is baked in. This is what authors must use.
2. **Restated / vendor-anchored adjusted (❌ leak).** The value at historical bar `t` reflects
   corporate actions with `ex_date` anywhere up to the **ingest/pull date** — including actions
   *after* `t`. This is exactly the `adj_close`/provenance leak `research-methodology.md` already
   warns about.
3. **Raw (❌ contaminated).** A split inside a lookback window fabricates momentum; share volume
   jumps at a split.

**The stored `adj_close` column is flavor (2), not (1).** `back_adjust()`
(`algua/data/corpactions.py`) anchors the adjustment at the **most recent bar of the imported
frame**; it is invoked once at ingest (`algua/data/importers/databento.py`) over the full historical
frame, and the result is frozen into an immutable snapshot (`docs/contracts/bar-schema.md`: `get_bars`
has **no `as_of` parameter** and returns the latest snapshot). So `adj_close[t]` read during a
walk-forward carries adjustment factors from actions years after `t`. **This design must never call
the stored `adj_close` column PIT-correct anywhere in kb, the skill, or the schema doc.**

### The precise leak surface (why (A) is a placebo, and what actually needs fixing)
A subtle but decisive fact governs the whole design:

- **Only degree-0-homogeneous (pure ratio/return) features are restatement-invariant — nothing
  broader.** For a strictly homogeneous-of-degree-zero function of prices — the canonical case
  `adj[t]/adj[t-N]` — any event with `ex_date > t` scales numerator and denominator by the *same*
  global factor and **cancels**, so the value computed on ingest-anchored `adj_close` equals the PIT
  value. This is why the backtest engine's P&L, which computes **returns** from `adj_close`
  (`algua/backtest/engine.py::adj_grid`), is *not* the leak and does not change in this work. The
  invariance claim is deliberately narrow: it holds ONLY for functions where a common multiplicative
  rescale of every input cancels. It does **not** extend to anything level-sensitive below.
- **Every non-homogeneous / level-sensitive feature DOES leak.** Absolute price level, a
  z-score/threshold on price (the additive mean/σ constants are not homogeneous), a price vs its own
  long moving average measured in absolute terms, any cross-sectional comparison of price *levels*,
  and any arbitrary transform that is not degree-0 homogeneous — these read the global anchor factor,
  which restatement (2) moves. A future split rescales the entire historical `adj_close` series, so
  `adj_close[t]` measured today ≠ the value knowable at T. When in doubt, a feature is assumed
  level-sensitive (fail-safe default), not ratio-invariant.
- **Volume leaks at the split itself.** Raw share volume jumps at a split regardless of anchor;
  split-adjusting fixes the in-window jump. Dividends do **not** affect volume.

Therefore (A) — "point authors at `view['adj_close']`" — is a placebo for level features and a
false-confidence hazard (it would tell authors the column is "the correct one now" while the leak
`research-methodology.md` documents stays intact). The real fix is a PIT-forward-adjusted accessor
for **both** price and volume, plus enforcement.

## The efficiency key: forward adjustment is PIT-correct in O(n)
The naive reading of (B) ("recompute a back-adjustment anchored at each decision date T") is O(n²)
across a walk-forward. We avoid it entirely.

**Multiplier orientation (this is a correctness detail, not a reuse of `back_adjust`'s factor).**
`back_adjust` scales bars *before* an event's date DOWN onto the latest basis — for a 2:1 split it
multiplies pre-split closes by `m_e = 0.5`. A forward series anchored at the *earliest* bar must do
the opposite: leave the earliest bars alone and scale bars *on/after* the event's date onto the
earliest basis — for a 2:1 split it multiplies post-split closes by the **forward-basis factor**
`f_e = 1/m_e = 2.0`. For a dividend the split-multiplier is `1` and the dividend factor is likewise
inverted relative to the back-adjust convention. **The accessor computes `f_e` explicitly (the
reciprocal of `back_adjust`'s per-event multiplier); it does NOT reuse `back_adjust`'s `m_e`
verbatim.** A property test pins `f_e * m_e == 1` per event.

**Why a single O(n) series IS the correct point-in-time object here — and the invariant that
licenses it.** A truly general bitemporal accessor is 2-D: the value for bar `s` as-of decision time
`D` is `∏ f_e over events with ex_date ≤ s AND knowable_at ≤ D`. When `knowable_at > ex_date` for
some event, that 2-D surface genuinely cannot collapse to one universal series indexed by bar-date
(the historical values get *restated* as later decisions learn the event) — a per-decision-date
recompute (O(n²)) or an as-of parameter would be required, and applying the event at `knowable_at`
instead of `ex_date` (an earlier idea) is **wrong**: it would leave a raw split discontinuity in the
`ex_date..knowable_at` window even for a decision that fully knows the event.

We avoid the 2-D problem by **enforcing the real-world invariant `knowable_at ≤ ex_date`** — splits
and dividends are *declared before* they go ex, so the declaration/knowable date precedes the
ex-date. The `validate_corpactions` validator makes this a hard, fail-closed check. Under it, any
event with `ex_date ≤ s` also has `knowable_at ≤ ex_date ≤ s ≤ D` for any decision `D ≥ s`, so the
`knowable_at ≤ D` condition is *automatically satisfied* for every bar the series serves and the 2-D
surface collapses **exactly** to the single ex-date-gated series:

> **Anchor at the *earliest* bar and accumulate each event from its `ex_date` forward.** The forward
> factor is `FF[t] = ∏ f_e over events with ex_date ≤ t`; the PIT-forward-adjusted close is
> `pit[t] = raw_close[t] * FF[t]` — a **single O(n) series** (sort events by `ex_date`, one
> cumulative sweep).

`knowable_at` is therefore **consumed as the validated precondition that makes ex-date gating
PIT-correct** (not provenance-only, and not a wrong `max()`): late-reported or vendor-restated events
(`knowable_at > ex_date`) are the genuinely-2-D hard case and are an explicit **fail-closed
non-goal** — the validator rejects them and a follow-up issue tracks a full as-of accessor. This is
the same fail-closed discipline the #132 lanes use, and it keeps the whole efficiency thesis honest.

This series is then PIT-correct for **every** window simultaneously: for `pit[t]/pit[t-N]`, events
with `ex_date ∈ (t-N, t]` are applied to `pit[t]` only (correct), events with `ex_date ≤ t-N` cancel
(correct), events with `ex_date > t` are absent from both (correct — not yet ex, and by the invariant
not yet knowable). Because levels are anchored at the fixed earliest bar (not the moving ingest
date), level features are PIT-correct too. **One O(n) pass per symbol, no per-decision-date recompute,
no O(n²).** Forward vs back adjustment differ only by a global constant that cancels in ratios but not
in levels — which is precisely why forward is the PIT flavor and the ingest-anchored back-adjusted
`adj_close` is not.

## Split into two slices (GATE-1 finding #5/#6: one PR is too broad)

### Slice 1 — honest docs + secondary bug fixes (small, non-CODEOWNERS, ships first)
No new data plumbing. Does **not** claim an accessor that does not yet exist; tells the truth in the
interim.

1. **`kb/principles/research-methodology.md`** — under "Leaks no wall can catch", replace the single
   `adj_close`/provenance bullet with the explicit **three-way trichotomy** above. State plainly:
   raw `close`/`volume` is a **contamination defect, not leak-avoidance**; the **stored `adj_close`
   column is ingest-time-anchored (vendor/import-restated) and is itself the leak vector for any
   level-sensitive, pre-ingest-date decision**; only a PIT-forward-adjusted series (Slice 2) is
   correct. Note the restatement-invariance of pure return/ratio features so the guidance is
   precise, not alarmist. Explicitly: **until Slice 2 lands there is no sanctioned PIT-adjusted
   accessor** — interim guidance is "prefer return/ratio features on `adj_close`; do not build
   absolute-level features on `adj_close`; never use raw `close`/`volume` in a signal."
2. **`.claude/skills/author-a-strategy/SKILL.md` (+ the `.codex/skills/author-a-strategy/SKILL.md`
   mirror)** — stop presenting `values="adj_close"` as unconditionally safe and stop any framing of
   raw as leak-avoidance. Document the trichotomy, the ratio-vs-level nuance, and the interim rule.
   Point forward to the Slice-2 accessor (`pit_adj_close`/`pit_adj_volume`) as the future default,
   without claiming it exists yet. **Note the Slice-2 default-view change:** once Slice 2 lands, raw
   `open/high/low/close/volume` disappear from *every* strategy view (not just `needs_corpactions`
   ones), so "never use raw `close`/`volume`" is enforced by absence, not merely advised — Slice 1's
   interim rule is the honest bridge until that wall exists.
3. **Secondary strategy bug fixes** (Codex review, same origin) in the offending strategy and any
   shared author default/template: (a) `sort_index()` before positional `.iloc` indexing; (b) a
   full-count guard so `.mean()`'s silent `skipna` cannot compute a momentum/vol stat over a
   sparse/short window (require the full expected observation count, else no-opinion); (c) warmup
   floor `max(lookback+1, vol_long, vol_recent)` — the current `max(lookback+1, vol_long)` omits
   `vol_recent`; (d) guard zero / non-finite volume denominators (turnover ratios) → no-opinion, not
   `inf`/`NaN`.
4. **Tests** for (3) — the four secondary fixes, each with a red→green case.

**Slice-1 gate/CODEOWNERS:** none touched (kb + skills + a strategy module + tests). Auto-mergeable
if CI green.

### Slice 2 — the PIT corporate-action lane + accessor + fail-closed gate (larger, CODEOWNERS)
Mirrors the #132 fundamentals/news bitemporal PIT lane precedent.

1. **PIT corporate-action data lane** (`algua/data/*`, non-CODEOWNERS):
   - **NEW bitemporal value objects + parser (finding H5 — the current types do NOT carry
     `knowable_at`).** `algua/data/corpactions.py`'s existing `Split`/`Dividend` frozen dataclasses
     carry only `ex_date`, and `parse_databento_corp_actions` (in `importers/databento.py`) parses a
     `(symbol, ex_date, kind, value)` schema with **no declaration/announcement date at all**. The PIT
     lane therefore adds a **new** parser and a **new** typed record rather than reusing them:
     - `algua/data/corpactions_schema.py` defines a `PitCorpAction` record (or a thin
       `KnowableCorpAction` wrapper `(symbol, ex_date, action_type: {"split","dividend"}, value,
       knowable_at)`) that **carries and preserves `knowable_at`** end-to-end (parse → snapshot →
       read → accessor). `knowable_at` is a **required, non-null** field.
     - `parse_corpactions_pit(path)` (new) parses a source that **must** provide a per-event
       declaration/announcement date column. **If the source lacks a declaration date, the parser
       FAILS CLOSED** (raises) — it NEVER synthesizes `knowable_at = ex_date`. (The existing
       `parse_databento_corp_actions` stays untouched; the databento CSV has no declaration column,
       so it is not a valid PIT source until the vendor feed carries one — see § "Real-data reach".)
     - **PIT corpaction SOURCE SCHEMA — pinned as strictly as `bar-schema.md` (round-6 M6).** A
       load-bearing bitemporal field cannot be loosely parsed, so the accepted file is specified
       exactly and everything else is rejected fail-closed:
       - **Required columns, EXACT header names (no aliasing, no positional guessing):** `symbol`,
         `ex_date`, `action_type`, `value`, `declared_date`. Optional: `event_id`. **Any missing
         required header → raise; any *unexpected* extra column → raise** (so an unrelated vendor date
         column, e.g. `record_date`/`pay_date`/`announce_dt`, can NEVER be silently misparsed as
         `declared_date` — the parser only ever reads the literal `declared_date` header).
       - **`action_type`** ∈ {`split`, `dividend`} exactly (lowercase); any other token → raise.
       - **`value`** must be finite and strictly `> 0` (split ratio and dividend amount are both
         positive; a `≤0`, NaN, or inf value → raise). Split `value` is the ratio (2:1 → `2.0`);
         dividend `value` is the cash amount per share in `adj_close` units.
       - **Date columns (`ex_date`, `declared_date`) — timezone normalization identical to the daily
         bar rail (#262):** parsed as calendar session dates and normalized to **UTC midnight**; a
         naive date is interpreted under that documented convention; a value carrying a *conflicting*
         explicit tz offset → raise (never silently shift). `knowable_at` is set from the normalized
         `declared_date` (this is the ONLY thing `declared_date` becomes — it is renamed to
         `knowable_at` on parse so no other column can populate it).
       - **Null handling:** any null/blank in a required column → raise (no defaulting, no forward
         fill). `event_id`, if the column is present, must be non-null and unique across the file.
       - **Duplicate / revision semantics — immutable, mirroring the universe-snapshot rule:** the
         natural key is `(symbol, ex_date, action_type)`. Two rows sharing that key → **raise**,
         whether the `value`/`declared_date` agree (exact dup) or conflict (a revision). Revisions are
         NOT merged or last-wins'd; a correction requires a fresh curated file → a new immutable
         snapshot (a same-key conflict aborts before any write, exactly like `import-universe`). If
         `event_id` is supplied it is an additional uniqueness constraint, not an override of the
         natural-key rule.
     - The back-adjust engine keeps consuming the old `Split`/`Dividend` at *ingest* time (unchanged);
       the PIT accessor (§2.2) consumes the new `knowable_at`-bearing records. The two are deliberately
       separate value objects so the ex-only types can never leak into a lane that must be bitemporal.
   - `validate_corpactions` validator on the new schema: `knowable_at` required/non-null; enforces the
     hard, fail-closed invariant **`knowable_at ≤ ex_date`** (declaration precedes ex — the real-world
     norm for splits and dividends). That invariant is exactly what licenses the single-series ex-date
     gate above: an event's adjustment enters bar `t` iff `ex_date ≤ t`, and under the invariant it is
     provably already knowable by then. **Late-reported / vendor-restated events (`knowable_at >
     ex_date`) are rejected (fail closed)** — the genuinely-2-D case deferred to the as-of-accessor
     follow-up, not silently mis-adjusted.
   - `DataStore.ingest_corpactions` / `read_corpactions` + a `data import-corpactions --file <csv>`
     CLI (the concrete real-data path, round-6 H2), and `hindsight.py::query_corpactions`
     (full-hindsight post-mortem read), following the fundamentals/news method signatures exactly.
     `data import-corpactions` runs `parse_corpactions_pit` → `validate_corpactions` → immutable
     snapshot; a same-key conflict aborts before any write (§2.1). **This gives the lane a working
     ingestion path today** for any declaration-bearing source (a hand-curated corpactions CSV or a
     future vendor export), so the PIT accessor is exercisable end-to-end on real data — the source
     is an acceptance criterion, not a deferred Non-goal. The corpaction **data lane** is a new
     snapshot dataset kind (file-based manifest), **not** a registry-DB change — so the lane itself
     needs **no `db.py` SCHEMA_VERSION bump** (confirm the dataset-kind enum is additive). (The single
     deliberate `db.py` bump in this work is the `gate_evaluations.corpactions_snapshot` audit column
     — see §5 Provenance below.)
   - Import-time consistency: reuse `check_adj_close_consistent` to assert the vendor `adj_close`
     matches the event list, so the lane and the stored column can't silently diverge.
2. **Pure PIT accessor** (`algua/data/corpactions.py`, non-CODEOWNERS — additive, alongside
   `back_adjust`):
   - **`build_pit_price_volume_view(raw, events) -> DataFrame[ts, pit_adj_open, pit_adj_high,
     pit_adj_low, pit_adj_close, pit_adj_volume, pit_adj_factor]` is the ONE canonical function that
     produces the entire PIT basis (round-6 M4).** It exists precisely so implementers can't diverge:
     the acceptance criterion "`pit_adj_open/high/low` use the exact same per-event factor path as
     `pit_adj_close`" is guaranteed *by construction* because this single function computes `pit_adj_factor
     = FF[t] = ∏ f_e over events with ex_date ≤ t` once and multiplies **all four** OHLC columns by
     that identical factor, then attaches `pit_adj_volume` (from `split_adjusted_volume`) and — where
     the view builder composes it — `dollar_volume`. `f_e = 1/m_e` is the **forward-basis factor**
     (the reciprocal of `back_adjust`'s per-event multiplier); ex-date gating is PIT-correct by the
     validated `knowable_at ≤ ex_date` invariant; anchored at the earliest bar.
   - `forward_adjust(series, events) -> DataFrame[ts, pit_adj, pit_adj_factor]` remains as the
     internal **single-series primitive** (one price column → its adjusted series + factor) that
     `build_pit_price_volume_view` calls once per OHLC column with the shared factor; it is not a
     separate public surface for the view — everything the strategy view consumes flows through the
     canonical `build_pit_price_volume_view`. Property-tested against `back_adjust`: `f_e * m_e == 1`
     per event; `pit[t]/pit[t-N]` ratios equal the back-adjusted ratios; levels differ by the single
     global anchor constant; and all four `pit_adj_{open,high,low,close}` share one factor per bar.
   - `split_adjusted_volume(raw, events) -> Series`: raw share volume divided onto a single share-count
     basis by applying only **splits** with `ex_date ≤ t` (a 2:1 split doubles raw volume, so the
     adjusted series divides post-split volume by 2 — the split-count reciprocal of the price factor;
     dividends do **not** touch volume). Ex-date gating is again licensed by the `knowable_at ≤ ex_date`
     invariant.
   - `dollar_volume(raw) -> Series`: the market dollar volume `raw_close[t] * raw_volume[t]`.
     **Because price halves and shares double at a split, this product is split-invariant by
     construction** — so it is itself a PIT-safe *level* series and is served as its own derived
     column (see §3), which is what makes the recommended liquidity/ADV feature reachable even though
     the raw `close`/`volume` columns are withheld from the strategy view. The doc also records:
     - the mixed-basis product `pit_adj_close * raw_volume` is **NOT** invariant — never use it;
     - **adjusted-basis dollar volume `pit_adj_close * pit_adj_volume`** equals `dollar_volume` up to
       the fixed global anchor constant (available if a consistent adjusted basis is already in play);
     - **share turnover `pit_adj_volume / shares_outstanding`** when a PIT shares-outstanding series
       exists (a documented follow-up if not).
3. **Serve the series through ONE pure function so every lane that serves it is identical** (GATE-1:
   "how do all three get the same series"):
   - **The default research strategy view withholds raw OHLCV for EVERY strategy (round-6 H1) — the
     wall is NOT opt-in.** The serving/view layer applies two independent transforms:
     - **(default, always-on, no flag):** drop raw `open/high/low/close/volume` from the `view` handed
       to `signal()`/`construct()`, and add the derived, split-invariant `dollar_volume` (computed
       internally from raw — the raw columns are consumed by the view builder and then withheld). What
       remains for a plain strategy is `adj_close` (degree-0-homogeneous ratio-safe interim) plus
       `dollar_volume`. This alone makes the `whale_volume_momentum` raw-close/raw-volume defect
       **unrepresentable for a strategy that declares nothing** — closing round-5's "no flag declared"
       leak. It needs **no corpaction snapshot** (`dollar_volume = raw_close*raw_volume` is
       split-invariant by construction), so it works on today's Databento data immediately.
     - **(upgrade, when `needs_corpactions` is declared):** additionally swap `adj_close` for the full
       PIT-correct level basis `pit_adj_open/high/low/close` + `pit_adj_volume` (from the canonical
       `build_pit_price_volume_view` over the PIT corpaction snapshot) **and withhold `adj_close`
       too** (pit dominates it — same ratios, correct levels; see §4). This is the only path to a
       PIT-correct *level* feature.
   - The PIT columns are **derived signal-view columns, NOT stored snapshot / `get_bars` contract
     columns.** The immutable bar snapshot and `get_bars` are unchanged (raw OHLCV + `adj_close`); the
     derived columns are computed in the *serving/view layer* by the single canonical
     `build_pit_price_volume_view` / `split_adjusted_volume` / `dollar_volume` functions, then added
     to the `view`. Any lane that constructs the signal view calls the *same pure functions over the
     same event source*, so it gets an identical series **by construction** — that is the mechanism,
     independent of how many lanes are wired today.
   - **Migration note (the default withholding is intentionally breaking):** any in-tree strategy,
     template, or test that reads raw `open/high/low/close/volume` off the *view* will now fail
     closed — that is the correctness fix, not a regression. The offending `whale_volume_momentum`
     strategy and the shared author default/template are already ported in Slice 1; Slice 2 includes a
     repo-wide sweep porting every remaining view-side raw read to `adj_close` (ratio use),
     `dollar_volume` (liquidity), or `needs_corpactions` + `pit_adj_*` (levels), each with a red→green
     test. The engine's *internal* `bars` frame (raw `close` for sizing) is untouched — see below.
   - **`needs_corpactions` is ORTHOGONAL to the fundamentals/news/model mutual-exclusion group
     (MEDIUM — sidecar interaction).** `algua/strategies/base.py::__post_init__` enforces "at most one
     of `needs_fundamentals`/`needs_news`/`needs_model`" because those select which *signal* function
     is bound. `needs_corpactions` selects nothing about the signal function — it only reshapes the
     **price/volume basis of the bar view** — so it is a **separate axis** and is **NOT** added to that
     exclusive set. A strategy may declare `needs_corpactions` alongside the default price signal OR
     alongside any one of fundamentals/news/model: the PIT view-reshaping is applied regardless of
     which signal lane is active. The base.py exclusivity check for the trio is unchanged.
   - **What is checked WHERE — load-time config coherence vs runtime snapshot presence (round-6
     M5).** `StrategyConfig`/`LoadedStrategy` at **load time** cannot know whether a CLI/runtime
     corpaction snapshot argument was supplied, so `base.py::__post_init__` / `loader.py` govern
     **only static config coherence**: (a) `needs_corpactions` is a well-formed bool orthogonal to the
     fundamentals/news/model trio; (b) the `signal_panel` forbid (a `needs_corpactions` strategy
     declaring a `signal_panel_fn` is rejected at load, §3 below). **They do NOT check snapshot
     presence.** Requiring an actual corpaction provider/snapshot for a `needs_corpactions` strategy
     is a **runtime precondition, enforced fail-closed at every entry point that loads a strategy for
     execution**, each of which must refuse to run a `needs_corpactions` strategy without one:
     - **CLI `backtest run` / `backtest walk-forward` / `backtest sweep`** (`algua/cli/backtest_cmd.py`):
       require `--corpactions-snapshot` when the loaded strategy declares `needs_corpactions`; missing
       → fail closed (and reject `--corpactions-snapshot` when the strategy does NOT declare it — the
       same consistency guard the `--fundamentals-snapshot`/`--news-snapshot` options use).
     - **`promotion_preflight`** (`algua/registry/promotion.py`): thread the corpaction provider into
       the preflight and fail closed if a `needs_corpactions` strategy reaches promotion without a
       snapshot.
     - **paper/live load points**: `assert_tradable_without_corpactions` fail-closed guard (§3 below).
   - **The `signal_panel` fast path is FORBIDDEN for a `needs_corpactions` strategy (finding H1).**
     `signal_panel` receives the whole-period `bars` frame at once (not a per-decision `view`), which
     makes the panel's basis ambiguous and its promote-time parity verifier (§ enforcement) blind to
     the PIT-adjusted columns. Rather than build a vectorized PIT-safe panel + a corpaction-aware
     parity path now, we **fail closed at load** (`algua/strategies/loader.py`/`base.py`): a strategy
     that declares BOTH `needs_corpactions` and a `signal_panel_fn` is rejected. Every
     `needs_corpactions` strategy therefore runs the **canonical per-bar loop** (`_decision_weights`),
     over the PIT-adjusted `view`, exclusively. A vectorized PIT panel + parity path is an explicit
     deferred follow-up (Non-goals).
   - **The `needs_corpactions` view exposes a COMPLETE PIT-safe basis and NOTHING leaky.** `signal()`
     receives `pit_adj_open`/`pit_adj_high`/`pit_adj_low`/`pit_adj_close` (a whole bar scales
     uniformly, exactly as the engine's `adj_open`), `pit_adj_volume`, and the split-invariant
     `dollar_volume` — enough to build any price, return, volume, or liquidity feature correctly. It
     **withholds every leaky level column: raw `close`/`open`/`high`/`low`/`volume` AND the
     ingest-anchored `adj_close`.** `adj_close` is dropped from the strategy view deliberately:
     `pit_adj_close` gives the *same* ratios (they differ only by the global anchor that cancels) and
     *correct* levels, so it strictly dominates `adj_close`; keeping `adj_close` "for ratio use" would
     reopen the level-leak through a column the runtime cannot police (it can't tell
     `adj_close.pct_change()` from `adj_close.mean()`). Removing it makes the leak unreachable rather
     than merely discouraged. (The engine keeps using `adj_close` for its own P&L grid — that is the
     *engine's* internal frame, not the strategy `view`, and is untouched.)
   - **Split the `signal()`-visible view from the construction/risk view — capacity/ADV consume
     `dollar_volume`, NOT raw `close`/`volume` (finding H2).** Two distinct frames exist and must not
     be conflated:
     - The **engine's internal sizing/notional grid** (`adj_grid`/`_adj_open_grid` in
       `algua/backtest/engine.py`) reads raw `close` from the engine's *own* `bars` frame to size
       positions and compute next-bar-open fills. This is engine-internal, never the strategy `view`,
       and is **untouched** — withholding raw columns from the strategy view does not starve it.
     - The **strategy view** is handed to BOTH `signal(view)` AND `construct(scores, view)` (they get
       the *same* `view` in `_decision_weights`). So any capacity/ADV/liquidity logic a *construction
       policy* runs must read the **derived, split-invariant `dollar_volume`** column (and/or
       `pit_adj_volume` when `needs_corpactions`) that **every** view provides (round-6 H1: raw is
       withheld by default, so `dollar_volume` is served to all views, not just `needs_corpactions`
       ones) — it must **not** reach for raw `close`/`volume`, which are withheld.
       `dollar_volume = raw_close*raw_volume` is split-invariant
       by construction (§2.3), so it is the correct PIT-safe ADV/turnover denominator, and it is
       reachable in the view precisely so the recommended liquidity feature survives the withholding.
       Any construction policy or ADV helper in-tree that currently reads raw `close`/`volume` is
       ported to `dollar_volume`/`pit_adj_*` as part of this slice (with a red→green test).
   - **Which lanes are wired in Slice 2: backtest/research ONLY.** Paper/live are fail-closed (below),
     so there is no "byte-identical across all three lanes" claim to make yet — the guarantee is that
     *whenever* paper/live are wired (the follow-up) they call the identical function and therefore
     inherit the identical series. Slice 2 delivers one wired lane plus the shared function that makes
     the others free.
   - `docs/contracts/bar-schema.md` (contract doc; keep the three-in-sync change-control rule):
     document `pit_adj_close`/`pit_adj_volume` as **derived signal-view columns** (explicitly noting
     they are NOT part of the `get_bars`/snapshot contract, so no snapshot-schema change); **annotate
     `adj_close` as back-adjusted / ingest-anchored, PIT-invariant for degree-0-homogeneous
     returns/ratios ONLY, not for levels**; and keep the engine's return computation on `adj_close`
     unchanged (it is ratio-based → PIT-safe).
   - **Backtest/research lane first; paper/live fail closed** until corpaction serving is wired,
     via a new `assert_tradable_without_corpactions` fail-closed guard mirroring
     `assert_tradable_without_fundamentals`/`assert_tradable_without_news` in
     `algua/strategies/base.py` (called at every trading load point). This bounds Slice 2 while still fixing the lane where the defect was
     caught (`research promote`). A follow-up issue tracks paper/live corpaction serving.
   - `algua/backtest/engine.py` (**CODEOWNERS**) exposes the PIT columns in the served view / threads
     the corpaction snapshot → PR stays open for human merge.
   - **Corpaction snapshot id = first-class provenance, threaded end-to-end (finding H3).** Mirror the
     #132 fundamentals/news stamping exactly, so an audit can name which corpaction snapshot fed any
     `needs_corpactions` run:
     - **Result objects:** add `corpactions_snapshot: str | None = None` to `BacktestResult`
       (`algua/backtest/result.py`), `WalkForwardResult` (`algua/backtest/walkforward.py`), and
       `SweepResult` (`algua/backtest/sweep.py`) — the exact siblings of the existing
       `fundamentals_snapshot`/`news_snapshot` fields (walk-forward populates it, sweep copies it from
       `wf`, round-trips through the sweep meta dict).
     - **CLI JSON projections:** add `corpactions_snapshot` to the `backtest`/`walk-forward`/`sweep`
       result payloads and to the `--summary` projection key sets in `algua/cli/backtest_cmd.py`
       (alongside `fundamentals_snapshot`/`news_snapshot`), and add the `--corpactions-snapshot`
       option + the same `needs_corpactions`-consistency guard the fundamentals/news options use
       (reject a snapshot id when the strategy does not declare the flag, and — new — fail closed when
       a `needs_corpactions` strategy is run on a lane without a corpaction snapshot).
     - **Gate audit row (`gate_evaluations`):** stamp `corpactions_snapshot` on the gate row next to
       the existing `fundamentals_snapshot`/`news_snapshot` keys in `algua/registry/promotion.py`
       (**CODEOWNERS**). Mirroring #132 exactly means a new **`corpactions_snapshot` column** on
       `gate_evaluations` — the **single deliberate `db.py` SCHEMA_VERSION bump** in this work (own it,
       and confirm no other bump is in flight before merging; the data lane itself stays schema-free).
       This is the honest choice: folding it into `decision_json` would break the column-per-snapshot
       audit convention the #132 lanes established.
   - **`promotion_preflight` signal-panel-parity serves corpactions by DISABLING it (finding H4).**
     `promotion_preflight` (`algua/registry/promotion.py`) calls
     `verify_signal_panel_parity(_loaded, provider, start, end)`, which reads bars through the provider
     and compares the fast `signal_panel` grid against the per-bar loop over a `_static_operating_view`
     built from **raw** bars — it does **not** thread a corpaction provider or the PIT view builder. We
     do not teach it to: because §3 **forbids `signal_panel` for `needs_corpactions`**,
     `verify_signal_panel_parity` sees `signal_panel_fn is None` and is a **no-op** for exactly these
     strategies — there is no non-PIT panel comparison to run, and no stale-basis parity check to get
     wrong. The design records this dependency explicitly: **if** a future follow-up adds a vectorized
     PIT `signal_panel`, it MUST simultaneously thread the corpaction provider / PIT view builder into
     `verify_signal_panel_parity` (and into `_decision_weights_fast`), or the parity gate would compare
     PIT-adjusted loop weights against a raw-basis panel and fail spuriously. Until then, the forbid is
     the clean, fail-closed answer.
4. **Enforcement = two layers, the primary one unbypassable at RUNTIME** (GATE-1 findings on
   advisory-only AND on the AST scan's false-negative surface; CLAUDE.md precedent: gates change
   agent behavior, advisories don't; severity: high). A static AST scan alone is porous — aliases,
   helper functions, transitive imports, `getattr`, dynamic `df[col]`, `.eval`, precomputed feature
   columns, and shared templates can all hide a raw `close`/`volume` read. So the AST scan is NOT the
   load-bearing control:
   - **Primary (runtime, unbypassable, DEFAULT-ON for every strategy — round-6 H1): raw
     `open/high/low/close/volume` are withheld from the served strategy `view` regardless of whether
     the strategy declares `needs_corpactions`.** This is the structural core of the round-6 fix: the
     serving layer drops the raw level columns from *every* research/backtest view and adds
     split-invariant `dollar_volume` (§3), so a strategy that declares **nothing** still cannot read
     raw `close`/`volume` — a read fails at runtime regardless of spelling (alias, helper, `getattr`,
     dynamic `df[col]`), because the column is absent. A `needs_corpactions` strategy gets the strictly
     stronger view: `adj_close` is **also** withheld and replaced by `pit_adj_open/high/low/close` +
     `pit_adj_volume` (mirroring the #132 dual-access-mode data wall — the wrong series is simply not
     reachable). This makes the illegal state unrepresentable rather than merely un-declared, closing
     the entire false-negative surface **on the default path, not just the opt-in path**.
   - **Secondary (promote-time, fail-closed early signal, DEFAULT-ON for every promotion): an AST scan
     at `research promote` preflight**, modeled on the #277 data-wall scanner (`algua/research/` +
     wired through `algua/registry/promotion.py` — **CODEOWNERS**), runs for **all** strategies and
     **FAILS CLOSED** on a *statically detectable* view-side raw `close`/`volume` read, so authors get
     a clear, early, actionable rejection instead of a runtime `KeyError` — irrespective of the
     `needs_corpactions` flag. Direct `adj_close` access is flagged with a steer to `pit_adj_close`
     (advisory within the failure message; for a `needs_corpactions` strategy, where `adj_close` is
     withheld, an `adj_close` read is a hard fail). The scan is explicitly best-effort — its false
     negatives are backed by the runtime guarantee above, and the design says so rather than
     pretending AST is complete.
   - The waiver is **human-only** (a `--allow-raw-price`-style flag rejected on the agent path),
     consistent with every other integrity relaxation flag. Its semantics are exact (MEDIUM — waiver
     for construction/fast-path):
     - It relaxes **both** enforcement layers together — re-expose the withheld raw columns
       (`open/high/low/close/volume` + `adj_close`) in the view AND skip the promote-time AST scan.
       When re-exposed, **both** `signal(view)` **and** `construct(scores, view)` see the raw columns
       (they share the one `view`); the derived `pit_adj_*`/`dollar_volume` columns remain present too,
       so a waived strategy is strictly additive, never column-starved.
     - It does **NOT** re-enable the `signal_panel` fast path for a `needs_corpactions` strategy: that
       forbid (§3) is a separate, always-on structural rule (a strategy wanting the fast path simply
       must not declare `needs_corpactions`). The waiver is only about the raw-column withholding + the
       AST scan.
   - **Waiver provenance — the waiver is durable, audited state threaded through the SAME path as
     every other human-only relaxation (round-6 H3).** A waived (raw-contaminated) candidate must be
     durably distinguishable from a clean PIT-only one at every downstream gate, so `--allow-raw-price`
     does not just relax enforcement silently — it stamps a `raw_price_waived: bool` provenance flag:
     - **Result objects:** carried on `BacktestResult`/`WalkForwardResult`/`SweepResult` next to
       `corpactions_snapshot` (walk-forward populates it, sweep copies from `wf`).
     - **CLI JSON + `--summary`:** `raw_price_waived` appears in the `backtest`/`walk-forward`/`sweep`
       payloads and the `--summary` projection (a relaxation flag belongs in even the trimmed view).
     - **Gate audit row:** recorded in `gate_evaluations.decision_json` alongside the other human-only
       relaxation flags (`--allow-non-pit`, `--allow-holdout-reuse`, `--degradation-factor`, …) in
       `algua/registry/promotion.py` (**CODEOWNERS**) — it rides the *existing* relaxation-flags record,
       so it needs **no new column** (unlike `corpactions_snapshot`, which mirrors the per-snapshot
       column convention).
     - **Bound into the signed go-live context:** the go-live challenge/context assembly
       (`algua/registry/transitions.py` / `live_gate.py`, **CODEOWNERS**) — which already summarizes the
       forward certificate — additionally surfaces a `raw_price_waived` flag pulled from the candidate's
       promotion audit lineage, so the human signature demonstrably covers "this candidate was promoted
       under a raw-price contamination waiver." A raw-contaminated candidate therefore cannot reach live
       without the signer being explicitly, durably informed. (Tests assert the flag round-trips
       result→CLI→`decision_json` and appears in the go-live challenge payload.)
   - Justification for gate-not-advisory is recorded in the spec: the issue explicitly says the
     defect "silently degrades EVERY authored strategy; the gate can't catch it (it's inside the
     feature)" — withholding the raw column at the serving boundary makes it catchable and
     unbypassable, so it is enforced, not advised.
5. **Tests** (the reviewer's required scenarios):
   - **Forward-factor orientation**: for a 2:1 split, post-split bars scale UP by `f_e = 2.0` and
     `f_e * m_e == 1` against `back_adjust`'s `m_e`; same reciprocal check for a dividend event.
   - Post-decision **split** and **dividend**: prove the `build_pit_price_volume_view` output's ratio &
     level are PIT-correct and that the stored `adj_close` level leaks (differs) while its ratio does not.
   - **Bitemporal invariant (validator)**: a normal event (`knowable_at ≤ ex_date`) validates and the
     ex-date-gated series is PIT-correct; a **late-known/restated event (`knowable_at > ex_date`) is
     REJECTED fail-closed** by `validate_corpactions`; a null `knowable_at` is rejected. (True
     late-known as-of handling is the deferred follow-up, not a silent mis-adjustment.)
   - **Volume**: raw volume jumps at a split; `split_adjusted_volume` does not; dividend leaves volume
     unchanged; the served **`dollar_volume = raw_close*raw_volume` is split-invariant**, and the
     mixed-basis `pit_adj_close*raw_volume` is asserted **NOT** invariant (regression guard against
     the earlier wrong formula).
   - **Default-on withholding (primary enforcement, round-6 H1)**: for a strategy that declares
     **NOTHING** (no `needs_corpactions`), the served view already has **no** raw
     `open/high/low/close/volume` — a read of any of them raises regardless of spelling (alias, helper,
     `getattr`, dynamic `df[col]`), while `adj_close` and the derived `dollar_volume` are present. For
     a `needs_corpactions` strategy the view *additionally* withholds `adj_close` and exposes
     `pit_adj_open/high/low/close` + `pit_adj_volume`. This is the test that proves the no-flag path is
     no longer leaky.
   - **Canonical single-factor view (round-6 M4)**: `build_pit_price_volume_view` scales
     `pit_adj_{open,high,low,close}` by one identical `pit_adj_factor` per bar (a whole bar scales
     uniformly) — asserted directly, not via four independent calls.
   - **PIT source schema (round-6 M6)**: `parse_corpactions_pit` accepts the exact-header CSV and
     rejects, fail-closed, each of: missing `declared_date`; an *extra*/aliased date column (e.g.
     `record_date`) offered in its place; a bad `action_type` token; a `value ≤ 0`/NaN/inf; a null in
     any required column; a `(symbol, ex_date, action_type)` duplicate (both exact-dup and
     conflicting-revision); a tz-conflicting date. A well-formed file parses with `knowable_at` set
     from `declared_date`.
   - **Real-data path (round-6 H2)**: `data import-corpactions` ingests a curated declaration-bearing
     CSV into an immutable snapshot and the PIT accessor runs end-to-end over it; a separate assertion
     documents that the in-tree Databento CSV (no declaration column) fails closed through
     `parse_corpactions_pit` (so it is not silently accepted).
   - **Gate fail-closed (secondary, default-on)**: a statically detectable view-side raw
     `close`/`volume` read is rejected at `research promote` for a strategy that declares **nothing**;
     the human waiver re-exposes columns AND skips the scan; a `pit_adj_close`-only (or `adj_close`
     ratio-only) signal passes.
   - **Waiver provenance (round-6 H3)**: a `--allow-raw-price` run stamps `raw_price_waived: true` on
     the result object, the CLI JSON (incl. `--summary`), and `gate_evaluations.decision_json`; the
     go-live challenge/context for that candidate surfaces `raw_price_waived`; a clean candidate shows
     `false` everywhere. The flag is rejected on the agent path (human-only).
   - **O(n)** / walk-forward equivalence (positive): under the validated `knowable_at ≤ ex_date`
     invariant, the single forward-adjusted series matches a per-date as-of recompute on a bounded
     sample (no O(n²)).
   - **Negative O(n)-equivalence for late-known events (MEDIUM):** construct a synthetic event with
     `knowable_at > ex_date` and assert (a) `validate_corpactions` REJECTS it fail-closed, AND (b) a
     white-box check that *if* it were admitted, the single ex-date-gated O(n) series would **differ**
     from the true per-decision-date as-of recompute in the `ex_date..knowable_at` window — proving the
     collapse-to-one-series is unsound for late-known events and that the fail-closed rejection is
     load-bearing, not incidental.
   - **`signal_panel` forbidden (H1):** a strategy declaring both `needs_corpactions` and a
     `signal_panel_fn` is rejected at load; a `needs_corpactions` strategy with no panel loads and runs
     the per-bar loop; `verify_signal_panel_parity` is a no-op for it (H4).
   - **Provenance stamping (H3):** a `needs_corpactions` backtest/walk-forward/sweep carries
     `corpactions_snapshot` on the result object and in the CLI JSON (incl. `--summary`); a
     `research promote` run stamps `corpactions_snapshot` on the `gate_evaluations` row; the CLI
     rejects a `--corpactions-snapshot` for a non-`needs_corpactions` strategy AND fails closed when a
     `needs_corpactions` strategy is run without one.
   - **Construction/ADV uses `dollar_volume`, not raw (H2):** a construction policy / ADV helper on a
     `needs_corpactions` view reads `dollar_volume`/`pit_adj_volume` and produces the correct
     split-invariant capacity figure; a red→green test proves the ported helper no longer reads raw
     `close`/`volume` (which are absent) and that the engine's internal sizing grid (raw `close`) is
     unaffected by the withholding.
   - **paper/live fail-closed** for `needs_corpactions` until wired (`assert_tradable_without_corpactions`).

**Slice-2 gate/CODEOWNERS:** touches `algua/backtest/engine.py` and `algua/registry/promotion.py`
(and possibly `algua/research/gates.py`), plus the single `db.py` SCHEMA_VERSION bump for the
`gate_evaluations.corpactions_snapshot` audit column → **PR stays open for human merge**. This is
correct: the change alters the anti-look-ahead surface, the promotion gate, and the audit schema.

## Real-data reach — what #521 actually fixes end-to-end (round-6 H2)
An honest statement of practical resolution, so reviewers/stakeholders are not misled about what
Slice 2 delivers on the real (Databento-sourced) data where the defect was reported:

- **The raw-contamination half of the defect is fixed end-to-end on Databento data TODAY.** The
  reported `whale_volume_momentum` failure was raw `close`/`volume` fabricating split-momentum and a
  split-jump in share volume. The default-on withholding (round-6 H1) removes raw `open/high/low/close/
  volume` from *every* strategy view and serves split-invariant `dollar_volume` — this needs **no
  corpaction snapshot at all**, so it resolves the raw-contamination defect on the existing Databento
  feed immediately.
- **The restatement / PIT-level half needs a declaration-bearing source, which the lane now accepts
  but the in-tree Databento auto-feed does not yet provide.** `data import-corpactions` gives the PIT
  lane a real, working ingestion path *today* for any declaration-bearing source (a curated CSV or a
  future vendor export), and the accessor is exercised end-to-end over it. But the specific in-tree
  Databento corp-action CSV carries **no** declaration column, so `parse_corpactions_pit` fails closed
  on it: **PIT-correct *level* features on Databento-sourced data remain unavailable until a
  declaration-bearing Databento source lands** (tracked as the source follow-up below). Slice 2 does
  not paper over this — the fail-closed parse is the honest boundary, and this paragraph is the
  reviewer-facing disclosure that the level half is not yet resolved on that particular feed.

## Non-goals / deferred
- Paper/live corpaction serving (a follow-up issue; fail-closed until then).
- **Monitoring / shadow-eval (#392) / standalone factor-eval (#219) lanes get NO corpaction serving
  in this work (MEDIUM — non-goal scoping).** They keep their current price/volume basis; wiring the
  shared `build_pit_price_volume_view`/`split_adjusted_volume`/`dollar_volume` functions into them (they'd get the
  identical series by construction, since it's the same pure functions) is a documented follow-up.
  Slice 2 wires exactly one decision lane — **backtest/research** — plus the shared pure functions.
- A vectorized PIT-safe `signal_panel` + a corpaction-aware `verify_signal_panel_parity` /
  `_decision_weights_fast` path — deferred; `needs_corpactions` forbids `signal_panel` and runs the
  per-bar loop only (§3/§4).
- **Late-known / vendor-restated corporate actions (`knowable_at > ex_date`)** — rejected fail-closed
  by the validator; a true as-of accessor (event inclusion gated by decision time, application by
  `ex_date`) that handles them without collapsing to one timeline is a documented follow-up.
- A PIT shares-outstanding series for exact share-turnover (dollar-volume/ADV suffice in the interim).
- **A declaration-bearing DATABENTO feed** (the generic importer + a strict CSV schema ARE in scope —
  round-6 H2 — so the lane has a real data path today; what is deferred is specifically wiring the
  in-tree *Databento* corp-action export to carry a declaration/announcement date, since its current
  CSV does not and `parse_corpactions_pit` fails closed on it). Auto-populating the PIT lane from
  Databento is the prerequisite follow-up for PIT-level features on that feed (§ Real-data reach).
- Other corporate-action *source vendors* beyond a strict PIT CSV / the Databento files.
- Re-adjusting the engine's P&L basis — unchanged; `adj_close` returns are already PIT-invariant.
- **`db.py` SCHEMA_VERSION:** the corpaction **data lane** is a new snapshot dataset (no registry-DB
  change). The **one** deliberate bump in this work is the `gate_evaluations.corpactions_snapshot`
  audit column (§3 Provenance / §5), mirroring #132 — own it as the single in-flight bump.

## Why this clears every GATE-1 blocking finding
### Round-1 findings
- **CRITICAL (blesses leaky product):** we explicitly refuse to call stored `adj_close` PIT-correct
  anywhere and build a real forward-adjusted accessor.
- **HIGH (ambiguous scope closable by docs-only):** the fix is (B) by construction — a docs-only
  close is impossible because Slice 1 states no accessor exists yet and the gate/lane land in Slice 2.
- **HIGH (contract conflict):** bar-schema documents `pit_adj_*` as *derived signal-view* columns
  (NOT `get_bars`/snapshot contract), annotates `adj_close` as ratio-only-PIT, keeps the ratio-based
  engine returns — an honest, consistent contract with no snapshot-schema change.
- **HIGH (advisory too weak):** enforcement is primarily runtime (raw level columns withheld from the
  served view — unbypassable) plus a fail-closed promote-time AST early-signal, human-waiver-only.
- **HIGH (volume under-specified):** `split_adjusted_volume` + *corrected* dollar-volume
  (`raw_close*raw_volume`, split-invariant) / turnover guidance + explicit dividend-does-not-affect-
  volume rule.
- **HIGH (single PR too broad):** two slices — honest docs+bugfixes, then the CODEOWNERS-scoped lane.

### Round-2 findings (this revision)
- **Multiplier orientation:** the accessor uses the **forward-basis factor `f_e = 1/m_e`**, explicitly
  the reciprocal of `back_adjust`'s multiplier (property-tested `f_e*m_e==1`), not `m_e` verbatim.
- **`knowable_at` unused → not bitemporal:** the accessor now gates each event on `d_e =
  max(ex_date, knowable_at)`, consuming both temporal axes; `knowable_at` is required/validated;
  late-reported events cannot retroactively adjust.
- **Volume formula wrong:** market dollar volume is `raw_close*raw_volume` (invariant); the mixed-basis
  `pit_adj_close*raw_volume` is called out as NOT invariant with a regression test.
- **Porous AST gate:** the load-bearing control is runtime column-withholding (illegal state
  unrepresentable); the AST scan is demoted to an explicitly best-effort early signal backed by it.
- **Contract-column ambiguity:** `pit_adj_*` are stated to be derived view columns, not snapshot
  contract columns.
- **Overreaching byte-identical claim:** dropped — Slice 2 wires backtest/research only; the shared
  pure function is the mechanism that makes future paper/live lanes identical when wired.
- **Over-broad ratio-invariance:** narrowed to strictly degree-0-homogeneous features, with a
  fail-safe "assume level-sensitive when in doubt" default and an enumerated unsafe-class list.

### Round-3 findings (this revision)
- **`max(ex_date, knowable_at)` gate was wrong:** replaced with a pure ex-date gate *licensed by* a
  validated, fail-closed `knowable_at ≤ ex_date` invariant; genuinely-2-D late-known/restated events
  are rejected and deferred to an as-of-accessor follow-up (no artificial jump at the knowledge date,
  no silent mis-adjustment).
- **`adj_close` reopened the level-leak:** it is now *withheld* from the `needs_corpactions` strategy
  view entirely (`pit_adj_close` dominates it — same ratios, correct levels); the engine's internal
  P&L grid still uses `adj_close` and is untouched.
- **Withholding vs the dollar-volume recipe:** the serving layer provides a derived, split-invariant
  `dollar_volume` column so the recommended liquidity feature is reachable without the raw columns.

### Round-4/round-5 findings (this revision) — the 5 HIGH gaps + 4 MEDIUM items
**HIGH:**
- **H1 — `signal_panel` fast-path unaddressed for `needs_corpactions`.** The design withheld raw
  columns from the per-bar `view` but never said what happens to the whole-period `signal_panel` fast
  path, whose input frame is not the PIT view. **Resolved:** `signal_panel` is **forbidden**
  (fail-closed at load) for a `needs_corpactions` strategy; it runs the canonical per-bar loop over the
  PIT view exclusively. A vectorized PIT-safe panel + parity path is an explicit deferred follow-up
  (§3, Non-goals).
- **H2 — `signal()` view vs construction/risk view not split; capacity/ADV would read raw.** Withholding
  raw `close`/`volume` from the shared `view` would starve any construction/ADV logic that needs them.
  **Resolved:** the engine's internal sizing/notional grid keeps reading raw `close` from its *own*
  `bars` frame (untouched); the strategy `view` (shared by `signal` AND `construct`) routes all
  capacity/ADV/liquidity through the derived split-invariant `dollar_volume`/`pit_adj_volume` columns,
  and any in-tree helper is ported off raw with a red→green test (§3).
- **H3 — corpaction snapshot id not first-class provenance.** **Resolved:** `corpactions_snapshot` is
  threaded through `BacktestResult`/`WalkForwardResult`/`SweepResult`, the CLI JSON + `--summary`
  projections, and stamped on the `gate_evaluations` row (a new column — the single deliberate `db.py`
  bump), mirroring the #132 fundamentals/news stamping exactly (§3 Provenance).
- **H4 — `promotion_preflight` signal-panel-parity undefined for corpactions.** **Resolved:** because
  H1 forbids `signal_panel` for `needs_corpactions`, `verify_signal_panel_parity` sees no panel and is
  a no-op for these strategies — no raw-vs-PIT parity mismatch is possible. The design records that any
  future PIT panel MUST simultaneously thread the corpaction provider/view builder into the parity
  verifier and the fast path (§3).
- **H5 — claimed reuse of `parse_databento_corp_actions`/`Split`/`Dividend`, which carry no
  `knowable_at`.** **Resolved:** Slice 2.1 defines a **NEW** `knowable_at`-bearing record + a **new**
  `parse_corpactions_pit` parser that **fails closed if the source lacks a declaration date** (never
  synthesizes `knowable_at = ex_date`). The ex-only types stay in the ingest-time back-adjust engine,
  never leaking into the bitemporal lane.

**MEDIUM:**
- **Sidecar mutual-exclusivity interaction:** `needs_corpactions` is **orthogonal** to the
  fundamentals/news/model exclusive trio (it reshapes the price basis, not the signal function), so it
  is not added to that set; allowed alongside any signal lane (§3).
- **Waiver semantics for construction/fast-path:** the human waiver re-exposes raw columns to BOTH
  `signal` and `construct` AND skips the AST scan (additive, never column-starved), but does **not**
  re-enable `signal_panel` for `needs_corpactions` (§4).
- **Non-goal scoping of monitoring/shadow/factor lanes:** explicitly out of scope; they keep their
  current basis and are a documented follow-up (Non-goals).
- **Negative O(n)-equivalence test for late-known events:** added — a `knowable_at > ex_date` event is
  rejected by the validator AND a white-box check shows the single O(n) series would diverge from the
  as-of recompute if admitted, proving the fail-closed rejection is load-bearing (§5 tests).

### Round-6 findings (this revision) — Codex gpt-5.5 BLOCK, 3 HIGH + 3 MEDIUM
**HIGH:**
- **H1 — opt-in enforcement gap (leaky by default).** Round-5 withheld raw columns only when a
  strategy declared `needs_corpactions`, leaving the "no flag declared" path exactly as leaky as
  before Slice 2 — precisely the `whale_volume_momentum` case. **Resolved:** the raw
  `open/high/low/close/volume` withholding is now **DEFAULT-ON for every research strategy view**
  (option (a) — withhold by default), with split-invariant `dollar_volume` served to every view and
  the promote-time AST scan running for every promotion. `needs_corpactions` is the *upgrade* that
  additionally withholds `adj_close` and serves the PIT `pit_adj_*` level basis. The default path is
  no longer leaky; a strategy that declares nothing cannot read raw `close`/`volume` (§ acceptance,
  §3, §4).
- **H2 — no real-data path.** Round-5 deferred the corpaction *source* to Non-goals, so on the
  Databento-sourced data where the defect was reported, nothing changed end-to-end. **Resolved via
  both (a) and (b):** (a) a concrete `data import-corpactions --file <csv>` importer + a strictly
  specified PIT CSV schema is now a Slice-2 acceptance criterion, giving the lane a working data path
  today for any declaration-bearing source; (b) a § "Real-data reach" section discloses honestly that
  the raw-contamination half of the defect is fixed on Databento data *today* (default withholding,
  no snapshot needed) while the PIT-*level* half stays unresolved on the in-tree Databento feed until
  a declaration-bearing Databento source lands (that feed has no declaration column, so
  `parse_corpactions_pit` fails closed on it).
- **H3 — waiver provenance not audited.** Round-5's `--allow-raw-price` waiver relaxed enforcement
  without leaving a durable trace. **Resolved:** a `raw_price_waived` flag is threaded through the
  same audit path as every other human-only relaxation — stamped on
  `BacktestResult`/`WalkForwardResult`/`SweepResult`, the CLI JSON + `--summary`, and
  `gate_evaluations.decision_json`, and surfaced in the signed go-live challenge/context — so a
  waived (raw-contaminated) candidate is durably distinguishable from a clean PIT-only one at every
  downstream gate (§4 Waiver provenance).

**MEDIUM:**
- **M4 — `forward_adjust` signature returned only close+factor while acceptance required identical
  OHLC scaling.** **Resolved:** one canonical `build_pit_price_volume_view(raw, events)` returns all
  `pit_adj_{open,high,low,close}` + `pit_adj_volume` + `pit_adj_factor`, computing a single factor per
  bar and multiplying all four OHLC columns by it — implementers cannot diverge; `forward_adjust`
  demoted to the internal single-series primitive it composes (§2.2).
- **M5 — misplaced load-time snapshot-presence claim.** `StrategyConfig`/`LoadedStrategy` at load
  time cannot know about a CLI/runtime snapshot argument. **Resolved:** `base.py`/`loader.py` govern
  only static config coherence (flag well-formedness + the `signal_panel` forbid); snapshot presence
  is a **runtime** precondition enumerated and enforced fail-closed at every entry point — CLI
  `backtest run`/`walk-forward`/`sweep`, `promotion_preflight`, and the paper/live load guards (§3).
- **M6 — PIT source schema under-specified for a load-bearing bitemporal field.** **Resolved:** the
  accepted CSV is pinned as strictly as `bar-schema.md` — exact required headers
  (`symbol,ex_date,action_type,value,declared_date`), extra/aliased columns rejected (so an unrelated
  vendor date can't be misparsed as `knowable_at`), `action_type` enum, positive-finite `value`,
  UTC-midnight tz normalization (#262 convention) with tz-conflict rejection, null rejection, and
  immutable `(symbol,ex_date,action_type)` duplicate/revision rejection (§2.1).
