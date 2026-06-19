# Issue #228 — Advisory Cross-Family Gaming Detector

**Part of:** `docs/superpowers/specs/2026-06-15-streaming-funnel-multiple-testing-issue-211-design.md` (Phase 4 follow-up)
**Builds directly on:** #222 (`docs/superpowers/specs/2026-06-16-hierarchical-family-budgets-anti-gaming-issue-222-design.md`)
**Status:** Design — pre-GATE-1 (multi-model design review pending).

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

### 1. Architecture & placement

Two pieces, splitting pure logic from I/O exactly like #222's `clustering.py` (pure) vs
`promotion.py` (I/O):

- **`algua/research/family_audit.py` (new, pure — no I/O, no DB).** The detection + grouping +
  evasion-ranking core. Deterministic given its inputs; the testable heart.
- **CLI command `research family-audit` in `algua/cli/` (mirrors `research dormant-sweep`).**
  Pulls families (`all_families_with_member_profiles()`), return series
  (`load_backtest_returns()`), and breadth (read-only store accessors), calls the pure function,
  prints JSON to stdout. Adds **no** store-mutation methods.

### 2. Detection — max-linkage, return-correlation authoritative

For every **unordered pair of families (A, B)** not already DAG-connected (see §4):

- Compute cross-family similarity by **max-linkage (single-linkage)**: the most-similar
  cross-family strategy pair, `argmax over a∈A, b∈B of sim(a, b)`.
- `sim(a, b)` reuses #222's **protected** similarity math via **one new pure helper** added to
  `clustering.py`: `pairwise_axes(code_a, factors_a, returns_a, code_b, factors_b, returns_b)
  -> (score, axes)` — the strategy-to-strategy primitive returning both the blended score
  (same weights: code 0.50 / factor 0.30 / return 0.20, same fail-closed rule) and the per-axis
  dict `{"code", "factor", "return"}` (where `"return"` is `None` when the axis was not evaluable).
  `family_audit.py` iterates `a∈A × b∈B`, tracks the arg-max by `score` (that is max-linkage), and
  the winning pair yields the cluster's `representative_pair`, `axis_breakdown`, and `return_axis`
  coverage in one pass. No similarity math is reimplemented in `family_audit.py`.
- The existing `family_similarity` (the assignment-time strategy-vs-family wrapper) is **not
  refactored** — its behavior and reference-value tests are untouched; `pairwise_axes` is purely
  additive. `clustering_version()` hashes config constants, not the function set, so the digest is
  unchanged.
- **Return-correlation is authoritative**: an adversary can rename code and swap factor labels, but
  a hidden near-duplicate's *returns* still correlate. Code/factor axes corroborate. This falls out
  of the weight blend (the return axis dominates only when present and high); see §5/§6 for the
  honest-coverage handling when return data is thin.

**Per-axis breakdown for triage.** Reporting the winning pair's per-axis breakdown lets the human
judge *why* a pair flagged — a return-correlation flag is strong evidence; a pure factor-Jaccard
flag is weak/coincidental. The breakdown is the `axes` dict `pairwise_axes` already returns for the
arg-max pair, so no extra computation is needed.

### 3. Grouping — connected components

A pair is **flagged** if its max-linkage similarity ≥ `PARENTAGE_THRESHOLD` (0.50, imported from
`clustering.py`); a pair ≥ `MERGE_THRESHOLD` (0.85) is tagged **high-confidence**. Flagged pairs
form an undirected graph over families; each **connected component** with ≥2 families is one
reported **cluster**. (If A~B and B~C are both flagged, report `{A,B,C}` — the human remedy of
parent edges would unify them transitively anyway.) Thresholds are imported, not redeclared, so
`clustering.py` remains the single source; the audit command introduces no protected constants.

### 4. Scope — skip already-unified families

Before evaluating a pair, skip it if A and B are already DAG-connected — i.e. one is an
ancestor/descendant of the other (`family_ancestry()`), or they share an ancestor such that
`family_lifetime_combos` already pools their trials. Already-linked families' breadth is **already**
unified via the gate's BFS, so there is no evasion to detect. This avoids flagging legitimate
PARENTAGE-minted lineages from #222.

### 5. Severity — evasion magnitude (breadth dodged)

Per cluster:

```
unified_breadth      = lifetime_combos over the UNION of the cluster's families (+ancestors, deduped)
max_individual_breadth = max over f in cluster of family_lifetime_combos(f)
evasion_magnitude    = unified_breadth − max_individual_breadth
```

`evasion_magnitude` is the breadth the split currently dodges — the extra multiple-testing penalty
even the *largest* family in the cluster escapes by staying separate (a conservative lower bound on
total evasion; it is 0 only when one family already subsumes all the others' trials). Clusters are
ranked by `evasion_magnitude` descending. Similarity (≥ threshold) is the **confidence filter** that
admits a cluster; evasion is the **rank** among admitted clusters — directly the thing #228 defends.

Computed via a new read-only store accessor `lifetime_combos_for_families(family_ids) -> int`: the
union of `_family_member_strategies` across all given families (+ transitive ancestors), summed once
over `search_trials.n_combos` (same DISTINCT-strategy dedup as `family_lifetime_combos`, which may
delegate to it). The pure function receives this as an injected `breadth_of: Callable[[frozenset[int]],
int]` callback (store-backed in the CLI, dict-backed in tests) so evasion + ranking stay inside the
pure, testable core.

### 6. Missing return-data — flag on code/factor, report coverage (never silently clean)

Return-correlation needs ≥ `_RETURN_CORRELATION_MIN_OVERLAP` (63) shared dates. When a pair lacks
sufficient overlap (or a member has no stored `backtest_returns`), the detector does **not** treat
the pair as clean: it still computes the code/factor axes and flags on those, **and** the cluster's
JSON reports `return_axis: "evaluated" | "insufficient_overlap"` per the flagging pair. A
return-data blind spot is therefore **visible**, not hidden — consistent with the platform's
no-silent-caps / honest-coverage ethos. (A pure code/factor flag is reported as lower-confidence so
the human weights it accordingly.)

### 7. Output — JSON on stdout

```json
{
  "clusters": [
    {
      "families": [
        {"id": 7, "name": "mom_a", "lifetime_combos": 480},
        {"id": 11, "name": "mom_b", "lifetime_combos": 440}
      ],
      "unified_breadth": 920,
      "max_individual_breadth": 480,
      "evasion_magnitude": 440,
      "max_similarity": 0.91,
      "confidence": "high",
      "return_axis": "evaluated",
      "axis_breakdown": {"code": 0.0, "factor": 0.33, "return": 0.91},
      "representative_pair": {"strategy_a": "mom_a_v2", "strategy_b": "mom_b_v1", "similarity": 0.91},
      "recommended_action": "human add_parent_edge to unify breadth (e.g. registry … --actor human)"
    }
  ],
  "n_families_scanned": 12,
  "n_pairs_evaluated": 38,
  "n_pairs_skipped_already_linked": 4,
  "clustering_version": "…",
  "config": {
    "merge_threshold": 0.85,
    "parentage_threshold": 0.50,
    "return_correlation_min_overlap": 63
  }
}
```

`clusters: []` = no detected gaming. That is the **expected** result today (single operator, funnel
not at 1000+ hypotheses) — and is itself the empirical "is the failure mode even real yet" signal
the issue asked for before any enforcement is built.

### 8. Read-only guarantees (mirror `dormant-sweep`)

No holdout reservation/peek/burn; no `gate_evaluations`/`family_events` writes; no transitions; no
return-series recompute (reads stored blobs only via `load_backtest_returns`). The command's only
effect is stdout JSON. Collapse stays the existing human-governed `add_parent_edge`.

---

## Threat model

| Threat | #222 (assignment-time) | #228 (this detector) |
|--------|------------------------|----------------------|
| Relabel a clone as a fresh family | MERGE at entry assigns it to incumbent | If entry-time check missed it, cross-family max-linkage (code axis) flags it post-hoc |
| Deliberate split: one thesis across N NOVEL families | **not covered** | **covered** — return-correlation max-linkage flags the convergent set; ranked by dodged breadth |
| Obfuscate code + relabel factors to stay NOVEL | partial | return-correlation axis catches behavioral convergence code/factor relabeling can't hide |
| Hide one near-duplicate inside an otherwise-diverse family | diluted by average | **max-linkage** flags on the single hottest cross-family pair |

**Residual gaps (documented, not hidden):**
- A truly diverse-but-coordinated set with low pairwise return correlation evades (fundamental
  limit of correlation-based detection).
- Members with < 63 shared return-dates fall back to gameable code/factor axes — reported as
  `insufficient_overlap`, not silently passed (§6).
- Detection is advisory; an operator who ignores a flagged cluster leaves the evasion in place
  (by design — collapse is a governed human action, matching #222's autonomy boundary).

---

## Footprint

| File | Protected | Change |
|------|-----------|--------|
| `algua/research/family_audit.py` *(new)* | no | pure detector: pairwise max-linkage, connected-component grouping, evasion ranking; `breadth_of` injected |
| `algua/research/clustering.py` | **YES** (CODEOWNERS) | **additive only**: `pairwise_axes(...)` pure helper for per-axis breakdown; existing `family_similarity` + constants + `clustering_version()` untouched |
| `algua/registry/store.py` | **YES** | new read-only `lifetime_combos_for_families(family_ids)`; `family_lifetime_combos` may delegate to it |
| `algua/registry/repository.py` | no | Protocol entry for the new accessor |
| `algua/cli/` (research command group) | no | `research family-audit` command (mirrors `dormant-sweep`); JSON out |

No schema change (read-only over #222's tables). No new gate constants.

---

## Testing strategy

Pure-function (`family_audit.py`) tests:
- **Max-linkage picks the hidden clone**: a diverse family with one member highly return-correlated
  to a member of another family → flagged on that pair; the diverse siblings don't dilute it.
- **Connected-component grouping**: A~B, B~C (not A~C) → one cluster `{A,B,C}`.
- **Evasion math**: union breadth with shared-ancestor dedup; `evasion = unified − max_individual`;
  evasion 0 when one family subsumes the others.
- **Already-linked pairs excluded**: a PARENTAGE lineage from #222 is never flagged.
- **Missing return-data**: < 63 overlap → falls back to code/factor, `return_axis:
  "insufficient_overlap"`, confidence reflects it.
- **Empty result**: mutually dissimilar families → `clusters: []`.
- **Ranking**: clusters sorted by evasion descending; deterministic tie-break (lowest min family id).

`clustering.py` `pairwise_axes` tests: per-axis reference values; symmetry `axes(a,b)==axes(b,a)`;
non-finite → fail-closed; existing `family_similarity` reference tests still pass unchanged.

Store test: `lifetime_combos_for_families` dedups a strategy reachable via two cluster families;
equals `family_lifetime_combos` for a single-family input.

CLI smoke test: read-only (DB byte-identical before/after); valid JSON of the documented shape;
exit 0 on empty result.

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
