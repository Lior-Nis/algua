# Issue #228 — Advisory Cross-Family Gaming Detector

**Part of:** `docs/superpowers/specs/2026-06-15-streaming-funnel-multiple-testing-issue-211-design.md` (Phase 4 follow-up)
**Builds directly on:** #222 (`docs/superpowers/specs/2026-06-16-hierarchical-family-budgets-anti-gaming-issue-222-design.md`)
**Status:** GATE-1 reviewed (Codex/gpt-5.5 + OpenCode/GLM-5.1, round 1; Gemini CLI deauthorized). All
accepted findings folded in below; see "GATE-1 findings summary" at the end.

---

## Problem

#222 shipped **assignment-time** clustering: when a *new* strategy enters the funnel,
`family_similarity` (code-ancestry + factor-lineage + return-correlation axes) classifies it
MERGE / PARENTAGE / NOVEL into a family, and `family_lifetime_effective` accumulates breadth
across the family DAG. That defends against *naive* family laundering (a clone gets MERGE'd into
its incumbent) and sibling-spam (lifetime combos accumulate).

It does **not** defend against the **deliberate-split** adversary: hypotheses spread across
*several separate* families that each passed as NOVEL at their own entry point but collectively
behave as one thesis. #222's check only ever compares one entering strategy against one family;
nothing ever asks *"are families A and B secretly the same thesis?"* Once two families exist
unlinked, their breadth is accounted **separately** — each draws its own multiple-testing budget.

This is the gap #228 closes: a post-hoc, **cross-family** detector that scans the whole family
graph for convergence the entry-point check could not see, and surfaces it for a human to collapse.

### What "budget" means today (scope grounding)

The hierarchical alpha-budget that #228 nominally defends is, per #222's Stratum A/B split, still
an **interface only** — `FamilyBudgetLedger` has just the `InMemoryFamilyBudgetLedger` fake; the
real LORD++ binding is deferred to Phase 2 (#220). The mechanism that is **live today** is
`family_lifetime_effective` → breadth haircut + DSR `N` in the gate. So "splitting a thesis across
families to dodge the budget" concretely means **dodging breadth accumulation** right now. #228
targets that live mechanism. When the real per-family alpha-wealth ledger lands (#220), the same
detector's clusters apply to it unchanged — collapse unifies whichever budget is live.

---

## Scope

**In scope (build now):** a **read-only advisory** command
`uv run algua research family-audit` that detects and ranks suspected cross-family gaming and
**recommends** a collapse. It mirrors `research dormant-sweep`: it reads, ranks, emits JSON, and
**mutates nothing** — no holdout peek/burn, no ledger writes, no transitions, no return-series
recompute. The collapse remedy remains the **existing human-governed** `add_parent_edge` (which
already unifies `family_lifetime_effective` across the DAG via BFS).

**Explicitly out of scope:**
- Any automatic / agent-driven family-graph mutation (preserves #222's invariant: agents never
  mint or collapse families).
- A new mutating `family collapse` command (Option 2). Deferred until the detector demonstrates
  real convergent clusters exist; collapse today is the raw human `add_parent_edge`.
- Inline auto-collapse at promotion (Option 3) — irreversible automation against a statistical
  signal; rejected.
- New protected gate constants. The command is advisory; thresholds are report filters, not a gate.

---

## Design

### 1. Architecture & placement — pure core takes precomputed data, CLI orchestrates

Three layers, keeping **all** I/O out of the pure core (GATE-1 M4 / "pure-core" finding — the pure
function must not call a store callback mid-algorithm, or determinism would depend on I/O timing):

- **CLI command `research family-audit` in `algua/cli/` (mirrors `research dormant-sweep`).**
  Opens **one** store connection and reads everything under a **single consistent snapshot** (one
  read transaction) so a concurrent `add_parent_edge`/assignment/return-persist cannot produce a
  mixed view (GATE-1 snapshot-consistency finding): family member profiles
  (`all_families_with_member_profiles()`), each strategy's return series + its period
  (`load_backtest_returns()`, batch-loaded **once** per strategy — O(M), not O(M²)), per-family
  lifetime breadth, and the ancestry map. It then calls the pure core and prints JSON to stdout.
  Adds **no** store-mutation methods.
- **`algua/research/family_audit.py` (new, pure — no I/O, no DB).** A pure pipeline of three
  functions interleaved with two CLI breadth-fetch steps, so the evasion-skip gets the pairwise
  union breadth it needs **before** grouping (GATE-1-R2 ordering finding) while the pure core never
  touches the DB:
  1. `flag_edges(profiles, returns) -> candidate_edges` — similarity-only pass (max-linkage audit
     score per family pair, §2). No breadth needed. Produces every **similar** family pair with its
     score/axes/status.
  2. *(CLI)* compute `pair_breadth = {frozenset({a,b}): lifetime_combos_for_families({a,b})}` for the
     **candidate** pairs only (few — not all O(F²)), under the read snapshot.
  3. `build_components(candidate_edges, *, pair_breadth, individual_breadth) -> components, kept_edges`
     — applies the **evasion-based skip** (§4) using `pair_breadth`, then groups survivors into
     connected components.
  4. *(CLI)* compute `component_breadth = {frozenset(component): lifetime_combos_for_families(component)}`.
  5. `rank_clusters(components, kept_edges, *, component_breadth, individual_breadth) -> report` —
     `family_breadth_delta` ranking + per-edge/per-cluster assembly (§5).

  All inputs are fully-materialized immutable data; no callbacks, no I/O in `family_audit.py`.
- **`algua/research/clustering.py` (protected, additive + one behavior-preserving refactor).** Adds
  the `pairwise_axes` primitive and routes `family_similarity` through it (see §2).

### 2. Detection — max-linkage with an audit score that makes return-correlation authoritative

For every **unordered pair of families (A, B)** with non-zero evasion (see §4):

- Compute the cross-family signal by **max-linkage (single-linkage)** over `a∈A × b∈B`. For each
  strategy pair, `pairwise_axes` returns the per-axis dict `{"code", "factor", "return"}`
  (`"return"` is `None` when the axis is not evaluable — < 63 shared dates or a series missing).

- **`pairwise_axes(code_a, factors_a, returns_a, code_b, factors_b, returns_b) -> (blended, axes)`**
  is added to the **protected** `clustering.py`, and `family_similarity`'s inner loop is **refactored
  to call it** so there is exactly **one** axis-math implementation (GATE-1 H2 dual-path finding —
  two copies could silently drift). The refactor is behavior-preserving: it reproduces the existing
  empty-hash→0, empty-factor-sets→0, negative-correlation→0, non-finite→fail-closed rules, and the
  existing `family_similarity` reference tests must stay **byte-identical** green. `pairwise_axes`
  reuses the protected weights (code 0.50 / factor 0.30 / return 0.20) for `blended`.
  `clustering_version()` is unchanged (it hashes config constants, and the constants/axis-availability
  are unchanged).

- **The audit score is NOT the plain blend (GATE-1 C1, the central correctness fix).** With the
  assignment-time weights, a pure return-clone (code 0, factor 0, return 1.0) blends to only
  `0.20` — below the `0.50` flag floor — so the blend **cannot** catch the "obfuscate code +
  relabel factors" thesis-split that #228 exists to stop. `family_audit` therefore computes, per
  strategy pair, an **audit score** that lets the return axis flag *independently*:

  ```
  audit_score(a, b) = max(
      blended(a, b),                                   # multi-axis agreement
      axes["return"] if axes["return"] is not None     # return-correlation, authoritative on its own
                      else 0.0,
  )
  ```

  A pair is **flagged** when EITHER condition holds (two thresholds, because the two paths carry
  different baseline noise — GATE-1-R2 false-positive finding):

  ```
  flagged = (blended >= AUDIT_FLAG_THRESHOLD)            # multi-axis agreement, floor 0.50
            or (return_axis is not None
                and provenance_comparable                 # §6 — overlapping period, ≥63 shared dates
                and return_axis >= RETURN_INDEPENDENT_THRESHOLD)   # return-only, HIGH floor 0.85
  ```

  - `AUDIT_FLAG_THRESHOLD = 0.50` (= `PARENTAGE_THRESHOLD`, imported) — multi-axis blends flag at the
    same floor #222 uses; a 0.50 blend already requires real code/factor agreement.
  - `RETURN_INDEPENDENT_THRESHOLD = 0.85` (= `MERGE_THRESHOLD`, imported) — the return axis flags on
    its **own** only at near-duplicate strength. A 0.50 *return* correlation is common between
    unrelated long-equity strategies via market beta, so the independent return path uses the HIGH
    floor to avoid beta-driven false positives while still catching the obfuscate-code-and-relabel
    near-clone (whose returns track ~1.0).

  Both thresholds are imported from `clustering.py`; `family_audit.py` adds no protected constants.
  `audit_score = max(blended, return_axis or 0.0)` is reported for ranking/tier display.
  Max-linkage = the `argmax over a∈A, b∈B` of the flagging strength. The winning pair is the edge's
  `representative_pair` and supplies its `axis_breakdown`.

### 3. Grouping — connected components, but report and recommend per EDGE

Flagged family pairs (edges) form an undirected graph; each **connected component** with ≥ 2
families is one reported **cluster**. But because max-linkage + transitivity can chain dissimilar
families together (A~B, B~C, A≁C) and the remedy is graph mutation, the report does **not** collapse
a component blindly (GATE-1 over-collapse finding):

- Each cluster lists **`flagged_edges[]`** — every flagged pair with its `audit_score`,
  `axis_breakdown`, per-edge `status` (§6), and `representative_pair`. The human sees exactly which
  links are direct vs transitive.
- **`recommended_remediation`** is honest, not a prescriptive edge list (GATE-1-R2 remedy finding).
  `add_parent_edge(child, parent)` is **directional**: it makes `child` inherit `parent`'s lifetime
  trials, **not** the reverse (breadth flows ancestor→descendant via the BFS in
  `family_lifetime_combos`). So N−1 parent edges do **not** symmetrically unify a cluster — the
  chosen parent keeps dodging the children's trials — and "child = low / parent = high breadth" can
  even propose an edge from an existing ancestor to its descendant (a cycle `add_parent_edge` would
  reject). The detector therefore recommends the **governed remedy** without over-claiming: flag the
  cluster for human review and consolidate the split families into a single canonical family by
  **reassigning members** (`assign_strategy_to_family`, human-governed) so future promotions land in
  one family and face the pooled lifetime breadth. The output names the highest-breadth family as the
  natural consolidation target and lists the cluster's family ids; it does **not** auto-prescribe
  edges that could under-fix or cycle. (Building a streamlined `family collapse` command remains the
  deferred Option-2 follow-up.)

Each edge is tagged `tier: "merge"` (audit_score ≥ `MERGE_THRESHOLD` 0.85) or `tier: "parentage"`
(≥ floor, < 0.85) — a descriptive similarity tier, replacing the misleading blunt `confidence`
label (GATE-1 M3); evidential weight is carried by `status` + `axis_breakdown`.

### 4. Scope — skip pairs with zero actual evasion (not "shared ancestor")

The earlier "skip if they share an ancestor" rule was **wrong** (GATE-1 C2): two sibling families
under a common parent each inherit the parent's trials but **not each other's own** trials, so a
thesis split between siblings still dodges breadth. The correct, self-checking rule is **evasion-
based**:

```
skip (do not flag) the pair (A, B) iff  breadth_of({A, B}) == max(breadth(A), breadth(B))
```

i.e. unifying them would add no breadth to the larger family — the only case where there is nothing
to detect. This subsumes the true ancestor/descendant case (where one already contains the other's
trials) **and** correctly keeps sibling pairs in scope. `breadth_of({A,B})` is the `pair_breadth`
the CLI precomputes for candidate pairs in pipeline step 2 (§1) and passes into `build_components` —
not a live callback, so the skip stays inside the pure core with deterministic inputs.

### 5. Severity — `family_breadth_delta` (honest about the 3-way gate)

Per cluster:

```
unified_breadth        = lifetime_combos over the UNION of the cluster's families (+ancestors, deduped once)
max_individual_breadth = max over f in cluster of family_lifetime_combos(f)
family_breadth_delta   = unified_breadth − max_individual_breadth
```

Renamed from "evasion_magnitude" (GATE-1 H3): the live gate is
`effective_funnel_breadth = max(own_lifetime, windowed_total, family_lifetime_effective)`, so the
family-term delta is **not necessarily** the realized gate penalty — if own-lifetime or
funnel-windowed already dominates, collapsing changes nothing at the gate. The output documents this
explicitly; `family_breadth_delta` is the breadth the split dodges *in the family term*, the term
#228 governs. Clusters are ranked by `family_breadth_delta` descending (tie-break: lowest min family
id, deterministic). A family with zero lifetime combos contributes nothing to the delta but is still
shown in the cluster for similarity (GATE-1 L4).

**Breadth computation.** A new read-only store accessor `lifetime_combos_for_families(family_ids)
-> int` collects the **union** of all input families' ancestor sets *first* (dedup the family-id
set), then runs the same DISTINCT-strategy query as `family_lifetime_combos` once over that union
(GATE-1 H3-union: a shared ancestor is counted exactly once). `family_lifetime_combos(fid)` becomes
the singleton case and delegates to it. The CLI calls this for each component's family-set to build
the immutable `breadth_map` it passes to `rank_clusters` — so the pure core never touches the DB.

### 6. Coverage status — never silently clean, never silently authoritative

The independent return-flag (§2) is sound **only** when the two return series are genuinely
comparable. `load_backtest_returns` gives each strategy's most-recent blob with its
`(period_start, period_end)`; the schema carries no universe/snapshot/construction identity (and
#228 adds none — read-only, no schema change). So `provenance_comparable` is defined conservatively
from what IS available: ≥ `_RETURN_CORRELATION_MIN_OVERLAP` (63) **shared** dates AND the shared
window covering a material fraction of both series (not a thin tail overlap). When provenance is NOT
confirmed comparable, the return axis may still **corroborate** the blended score but may **not**
independently flag — this is what stops a beta-driven or period-mismatched correlation from minting
a false positive (GATE-1-R2 provenance finding). Each flagged edge carries a `status`:

- **`flagged`** — return axis evaluated, provenance comparable, return ≥ 0.85 (independent) or it
  raised a blend ≥ 0.50. Strongest evidence; not gameable by code/factor relabeling.
- **`flagged_code_factor`** — flagged on the code/factor **blend** only (returns unavailable, thin,
  or provenance not comparable). **Weaker and gameable** (an adversary who obfuscates code can
  suppress this); the operator is told so. Without an authoritative return axis, the only way past
  the 0.50 blend floor is (near-)exact code-hash match or very high factor overlap — a documented
  blind spot, surfaced not hidden.
- **`inconclusive`** — pair has thin/incomparable returns AND a sub-floor code/factor blend: not
  flagged, but listed under a cluster's `inconclusive_edges[]` when it sits inside a flagged
  component, so a return-data blind spot is **visible** rather than dropped.

Edges report `return_overlap_days`, the two series' period spans, and `provenance_comparable` so the
human can judge the evidence. This replaces the binary "evaluated | insufficient_overlap" with an
honest taxonomy.

**Detection runs over ACTIVE members** (`all_families_with_member_profiles()` returns active members
only) while breadth is **lifetime** (includes removed members) — a deliberate asymmetry: a removed
member's *current* similarity is not meaningful, but its past trials still counted toward breadth.
The output reports per-family `active_member_count` and `lifetime_combos` so the asymmetry is
explicit (GATE-1 active-vs-lifetime finding).

### 7. Output — JSON on stdout

```json
{
  "clusters": [
    {
      "families": [
        {"id": 7, "name": "mom_a", "lifetime_combos": 480, "active_member_count": 3},
        {"id": 11, "name": "mom_b", "lifetime_combos": 440, "active_member_count": 2}
      ],
      "unified_breadth": 920,
      "max_individual_breadth": 480,
      "family_breadth_delta": 440,
      "flagged_edges": [
        {
          "family_a": 7, "family_b": 11,
          "audit_score": 0.91, "tier": "merge", "status": "flagged",
          "axis_breakdown": {"code": 0.0, "factor": 0.33, "return": 0.91},
          "return_overlap_days": 240, "provenance_comparable": true,
          "representative_pair": {"strategy_a": "mom_a_v2", "strategy_b": "mom_b_v1"}
        }
      ],
      "inconclusive_edges": [],
      "consolidation_target_family_id": 7,
      "recommended_remediation": "human review: consolidate families [7, 11] into family 7 by reassigning members (assign_strategy_to_family, --actor human) so future promotions face the pooled lifetime breadth. NOTE: add_parent_edge is directional and does not symmetrically unify breadth."
    }
  ],
  "n_families_scanned": 12,
  "n_pairs_evaluated": 34,
  "n_pairs_skipped_zero_evasion": 4,
  "wall_time_seconds": 0.42,
  "clustering_version": "…",
  "config": {
    "audit_flag_threshold": 0.50,
    "merge_threshold": 0.85,
    "return_correlation_min_overlap": 63
  }
}
```

`clusters: []` = no detected gaming — the **expected** result today (single operator, funnel not at
1000+ hypotheses), and itself the empirical "is the failure mode even real yet" signal the issue
asked for before any enforcement is built. `wall_time_seconds` + the `n_*` counters give the
operator scaling telemetry (GATE-1 M1/L3).

### 8. Read-only guarantees (mirror `dormant-sweep`)

No holdout reservation/peek/burn; no `gate_evaluations`/`family_events` writes; no transitions; no
return-series recompute (reads stored blobs only). All reads happen under one consistent snapshot.
The command's only effect is stdout JSON. Collapse stays the existing human-governed
`add_parent_edge`. **Scaling:** the scan is `O(F² × M²)` pairwise; for the current funnel that is
trivial. Cheap pruning (skip families with no usable profile; the zero-evasion skip in §4 drops
already-unified pairs) and the telemetry counters bound surprise. Hard caps / truncation are
deferred (YAGNI) until the funnel actually reaches the scale this issue's reopen condition names.

---

## Threat model

| Threat | #222 (assignment-time) | #228 (this detector) |
|--------|------------------------|----------------------|
| Relabel a clone as a fresh family | MERGE at entry assigns it to incumbent | If entry-time check missed it, cross-family max-linkage (code axis) flags it post-hoc |
| Deliberate split: one thesis across N NOVEL families | **not covered** | **covered** — the audit score lets return-correlation flag the convergent set independently; ranked by dodged family breadth |
| Obfuscate code + relabel factors to stay NOVEL | partial | **covered via the audit score** (§2): the return axis flags **on its own** at the HIGH floor (≥ 0.85, comparable provenance) — NOT via the 0.20-weighted blend, which could never reach the floor |
| Hide one near-duplicate inside an otherwise-diverse family | diluted by average | **max-linkage** flags on the single hottest cross-family pair |
| Split between sibling families under a shared parent | n/a | **covered** — evasion-based skip (§4) keeps sibling pairs in scope (they don't share each other's own trials) |

**Residual gaps (documented, not hidden):**
- A truly diverse-but-coordinated set with low pairwise return correlation evades (fundamental
  limit of correlation-based detection).
- Edges flagged `flagged_code_factor` (returns thin/unavailable) are **gameable** by code
  obfuscation — surfaced with that status, not treated as authoritative (§6).
- Members with < 63 shared return-dates or mismatched provenance are reported `inconclusive`, not
  silently passed (§6).
- Detection is advisory; an operator who ignores a flagged cluster leaves the evasion in place
  (by design — collapse is a governed human action, matching #222's autonomy boundary).

---

## Footprint

| File | Protected | Change |
|------|-----------|--------|
| `algua/research/family_audit.py` *(new)* | no | pure detector: `group_families` (max-linkage audit score, evasion-based skip, connected components, per-edge status) + `rank_clusters` (family_breadth_delta ranking); takes precomputed immutable data, no I/O |
| `algua/research/clustering.py` | **YES** (CODEOWNERS) | add `pairwise_axes(...)` primitive **and** refactor `family_similarity`'s inner loop to call it (behavior-preserving; reference tests stay byte-identical green). Constants + `clustering_version()` unchanged |
| `algua/registry/store.py` | **YES** | new read-only `lifetime_combos_for_families(family_ids)` (union-of-ancestors dedup); `family_lifetime_combos` delegates to it (singleton case) |
| `algua/registry/repository.py` | no | Protocol entry for the new accessor |
| `algua/cli/` (research command group) | no | `research family-audit` command (mirrors `dormant-sweep`): single-snapshot reads, batch-load returns, builds `breadth_map`, JSON out |

No schema change (read-only over #222's tables). No new gate constants (audit thresholds are
report filters in the unprotected `family_audit.py`, importing `clustering.py`'s floor).

---

## Testing strategy

Pure-function (`family_audit.py`) tests:
- **Return-authoritative flag (C1)**: a pair with code 0 / factor 0 / return 0.91 → `audit_score
  0.91`, flagged, `status: "flagged"`. The blend alone (0.18) would NOT flag — asserts the audit
  score, not the blend, drives detection.
- **Max-linkage picks the hidden clone**: a diverse family with one member highly return-correlated
  to a member of another family → flagged on that pair; the diverse siblings don't dilute it.
- **Sibling split stays in scope (C2)**: two sibling families under one parent, each with their own
  trials → NOT skipped (their union breadth > each individual), and flagged if similar.
- **Zero-evasion skip (C2)**: a true ancestor/descendant pair (one already contains the other's
  trials) → `breadth_of({A,B}) == max(...)` → skipped.
- **Connected components + per-edge report**: A~B, B~C, A≁C → one cluster `{A,B,C}` with two
  `flagged_edges` (A-B, B-C); A-C absent; output carries `recommended_remediation` +
  `consolidation_target_family_id` (the highest-breadth family), and prescribes no parent edges.
- **family_breadth_delta math**: union breadth with shared-ancestor dedup; `delta = unified −
  max_individual`; delta 0 when one family subsumes the others.
- **Coverage taxonomy**: returns thin/missing → `flagged_code_factor` (when code/factor reach the
  floor) or `inconclusive` (when they don't, listed under `inconclusive_edges`), never dropped.
- **Empty result**: mutually dissimilar families → `clusters: []`.
- **Ranking + determinism**: clusters sorted by `family_breadth_delta` desc; tie-break lowest min
  family id; pure functions are deterministic given identical inputs.

`clustering.py` tests: `pairwise_axes` per-axis reference values; symmetry `axes(a,b)==axes(b,a)`;
non-finite → fail-closed; **equivalence test** — `family_similarity` (now routed through
`pairwise_axes`) reproduces its existing reference values byte-identically (guards the H2 refactor).

Store test: `lifetime_combos_for_families` counts a strategy reachable via two cluster families
exactly once (union-of-ancestors dedup); equals `family_lifetime_combos` for a single-family input.

CLI smoke test: read-only (DB byte-identical before/after); reads occur under one snapshot; valid
JSON of the documented shape; exit 0 on empty result.

---

## Quality gate

`uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`

All four must pass before merge. Protected walls (`clustering.py`, `store.py`) require human review
on the PR.

---

## Deferred (out of scope for #228)

- Governed mutating `family collapse` command (Option 2) — build only if the detector surfaces real
  clusters worth a streamlined remedy.
- Holdings-overlap axis (inherited #222 deferral).
- Binding the detector to the real per-family alpha-wealth ledger (arrives with #220; clusters apply
  unchanged).
- Scheduled/automated audit runs — the command is operator-invoked, like `dormant-sweep`.
- Hard scale caps / truncation mode for the O(F²×M²) scan (GATE-1 M1, declined as YAGNI) — revisit
  when the funnel reaches the scale this issue's reopen condition names.
- Reporting a realized `effective_gate_delta` (own/windowed terms) alongside `family_breadth_delta`
  — the family-term delta is the governed quantity; the full gate delta needs per-strategy own +
  windowed terms and is deferred.

---

## GATE-1 findings summary (Codex/gpt-5.5 + OpenCode/GLM-5.1, round 1, 2026-06-19)

Gemini CLI was deauthorized (tier migration) so the panel was two-lineage. Both models
**independently** raised the two CRITICALs — high-confidence real.

| Finding | Severity | Models | Resolution |
|---------|----------|--------|------------|
| Return-correlation (0.20 weight) can't reach the 0.50 floor → "authoritative" claim false; the headline obfuscate-code+relabel-factors split is undetectable | CRITICAL | Codex + OpenCode | **Audit score** `max(blended, return_axis)` with the return axis flagging independently (§2); threat model corrected |
| "Skip if share an ancestor" wrongly suppresses sibling-split evasion | CRITICAL | Codex + OpenCode | **Evasion-based skip**: skip iff `breadth_of({A,B}) == max(breadth)` (§4) |
| `pairwise_axes` vs `family_similarity` dual code path can drift | HIGH | Codex + OpenCode | `family_similarity` **refactored to call** `pairwise_axes`; byte-identical equivalence test (§2) |
| `family_breadth_delta` overstates dodged budget under the 3-way `max` gate | HIGH | Codex | Renamed + documented as the family-term delta, not realized gate delta (§5) |
| Return provenance: most-recent blob may span different periods/universes | HIGH | Codex | `status` taxonomy + `return_overlap_days`/period spans; mismatch → not authoritative (§6) |
| Missing-return fallback gameable / silently absent | HIGH | Codex + OpenCode | `flagged` / `flagged_code_factor` / `inconclusive` taxonomy, never dropped (§6) |
| Connected-component collapse can over-merge via weak transitive chains | HIGH | Codex | Per-edge `flagged_edges[]` + `recommended_edges[]`; recommend per edge, not blanket (§3) |
| Pure core takes a store-backed `breadth_of` callback → I/O-timing dependence | MEDIUM | Codex + OpenCode | Restructured: pure `group_families`/`rank_clusters` take precomputed immutable data; CLI builds `breadth_map` (§1, §5) |
| Read snapshot consistency under concurrent writes | MEDIUM | Codex | Single consistent read snapshot in the CLI (§1, §8) |
| Active-vs-lifetime member asymmetry | MEDIUM | Codex | Documented; `active_member_count` + `lifetime_combos` reported (§6) |
| `confidence` label uninformative | MEDIUM | OpenCode | Replaced by `tier` (merge/parentage) + `status` + `axis_breakdown` (§3) |
| O(F²×M²) scaling, no telemetry | MEDIUM/LOW | Codex + OpenCode | `wall_time_seconds` + `n_*` counters + cheap pruning; hard caps deferred (YAGNI) (§8) |
| Zero-breadth families, batch-load returns, docstring note | LOW | both | Folded into §5/§6/§1 |

**Declined (with rationale):** weight-redistribution when returns missing (superseded by the
independent return threshold — cleaner, doesn't muddy the protected blend); hard `--max-families`
caps (YAGNI at current single-operator scale — revisit at the funnel-scale reopen condition).

### Round 2 (Codex/gpt-5.5 re-review of the revised spec, 2026-06-19)

Confirmed round-1 fixes sound (audit_score, evasion-skip logic, `pairwise_axes` refactor,
`family_breadth_delta`, status taxonomy). Three new findings, all folded in:

| Finding | Severity | Resolution |
|---------|----------|------------|
| Evasion-skip needs pairwise union breadth, but breadth was computed per-component *after* grouping — ordering hole | HIGH | Pure pipeline `flag_edges → (CLI pair_breadth) → build_components → (CLI component_breadth) → rank_clusters` (§1); skip uses precomputed `pair_breadth` (§4) |
| `add_parent_edge` is directional → N−1 edges don't symmetrically unify; "child=low/parent=high" can propose an ancestor→descendant cycle | HIGH | Dropped prescriptive `recommended_edges`; honest `recommended_remediation` = governed member-reassignment into a canonical family; names a consolidation target, prescribes no cycling edges (§3, §7) |
| Independent return-flag at the 0.50 floor is beta-gameable; provenance comparability not guaranteed by stored data | MEDIUM | Independent return-flag raised to `MERGE_THRESHOLD` 0.85 AND gated on `provenance_comparable` (≥63 shared dates + material window overlap); otherwise return only corroborates the blend (§2, §6) |
