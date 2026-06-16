# Phase 4: Hierarchical Family Alpha-Budgets + Anti-Gaming Clustering — Issue #222

**Part of:** `docs/superpowers/specs/2026-06-15-streaming-funnel-multiple-testing-issue-211-design.md`
**Status:** GATE-1 reviewed (Codex+OpenCode/GLM-5.1, round 1); findings folded in below.

---

## Problem

The research funnel's multiple-testing defense rests on `effective_funnel_breadth`, which uses a
two-way max: `max(own_lifetime, windowed_funnel)`. "Family" is a free-text column on strategies
with no FK, no effect on the gate, and no breadth accounting. This creates two exploitable gaps:

1. **Family laundering** — relabel a clone under a fresh `family` slug. Breadth resets to the
   clone's own tiny count, dodging the multiple-testing penalty.
2. **Sibling-spam within a family** — a family's accumulated sweeps don't raise the breadth bar
   for new members; all siblings share the funnel-wide pool, which dilutes per-family signal.

Phase 4 closes both gaps without requiring Phase 2 (#220, LORD++) or Phase 3 (#221, calibration).

---

## Scope — Stratum A (build now) vs Stratum B (interface until Phase 2 ships)

**Stratum A** — substrate buildable today, Phase-2-independent:
- Canonical family registry + parentage DAG (FK-backed, cycle-guarded, atomic writes)
- Empirical clustering for family assignment — code-ancestry + factor-lineage axes
- Governed family creation (agent may NOT mint new families)
- **Family-lifetime breadth** as the Phase-2-independent budget proxy: a crowded family → larger N
  → stricter haircut and DSR gate immediately, before Phase 2 exists

**Stratum B** — alpha-budget allocation layer, INTERFACE ONLY until Phase 2 ships:
- `FamilyBudgetLedger` Protocol (partition of global LORD++ wealth, `reserve`/`settle`)
- `InMemoryFamilyBudgetLedger` fake for contract tests
- Real LORD++ binding deferred to Phase 2 (#220)

---

## Design

### 1. Family registry + parentage DAG

Four new tables in `db.py`:

```sql
families           (id, name, created_at, created_by_actor, created_by_strategy)
family_members     (id, family_id FK, strategy_name, joined_at, joined_by_actor)
family_parents     (id, child_family_id FK, parent_family_id FK)
                   -- UNIQUE constraint on (child_family_id, parent_family_id)
family_events      (id, event_type, family_id FK, strategy_name,
                    actor, clustering_verdict, similarity_score, clustering_version,
                    axis_json, matched_family_id, created_at)
```

`family_events` is the governance ledger: every family-creation, member-assignment, and
parent-edge event is written here with its full clustering context (verdict, similarity score,
`clustering_version` hash of active weights/thresholds/axis_availability, matched family id,
axis breakdowns). This is how human review can reconstruct why a strategy ended up in a family.

#### Atomic cycle guard — `add_parent_edge` (CRITICAL fix, F2)

The cycle-check + INSERT **must be atomic**. A Python-side BFS before INSERT is insufficient:
two concurrent sessions can each see no cycle and insert A→B and B→A, corrupting the DAG.

`add_parent_edge` uses `BEGIN IMMEDIATE` (same pattern as `reserve_holdout`):

```
BEGIN IMMEDIATE
  BFS from proposed child → reject if parent is reachable from child
  INSERT INTO family_parents (child_family_id, parent_family_id)
COMMIT
```

`UNIQUE (child_family_id, parent_family_id)` is a second-line guard; `family_ancestry()`
traversal always uses a visited-set to defend against any future inconsistency.

Schema bump: **SCHEMA_VERSION 25 → 26** (feat(219) already used 24→25).

### 2. Clustering similarity — `algua/research/clustering.py` (new, **CODEOWNERS-protected**, F8)

`family_similarity(strategy, family_members, *, returns_lookup) → (SimVerdict, float)` is a pure
function analogous to `dsr_confidence`. Three verdicts:

- `MERGE (≥ MERGE_THRESHOLD)` — assign strategy into the incumbent family
- `PARENTAGE (≥ PARENTAGE_THRESHOLD, < MERGE_THRESHOLD)` — see governed-creation rules below
- `NOVEL (< PARENTAGE_THRESHOLD)` — strategy is genuinely novel

**Axes (introduced in order of substrate availability):**
- *code-ancestry* (`compute_artifact_hashes().code_hash`, exact match or shared history)
- *factor-lineage* (`lineage.factors_used_by()`, Jaccard similarity over factor sets)
- *return-correlation* (Task 7, deferred until `backtest_returns` table lands)

**Protected constants** (`clustering.py` is CODEOWNERS-protected alongside `gates.py`):
- `MERGE_THRESHOLD`, `PARENTAGE_THRESHOLD`, per-axis weights
- Pin reference-value tests exactly (same discipline as `dsr_confidence`)

**Clustering version (F10):** `clustering_version = sha256(repr(thresholds+weights+axis_availability))[:16]`,
recorded in every `family_events` row. Thresholds may ONLY be tightened (MERGE_THRESHOLD ↓ →
more strategies merge → stricter gate). No retroactive re-evaluation on threshold change.

### 3. Family-lifetime breadth (core anti-gaming mechanism, F3+F4)

**Key insight:** `windowed_family_combos` (rolling-window, family-filtered) is structurally dead
in the 3-way max because `windowed_funnel ≥ windowed_family` always (the family is a subset of
the funnel). Using windowed counts for family breadth also means the anti-reset mechanism decays:
after 90 days, ancestor trials age out and a child family inherits zero.

**Solution — lifetime family combos:**

```
family_lifetime_effective(family_id) = Σ SUM(n_combos) of search_trials
                                       FOR ALL strategies IN (BFS over family + ancestor families)
```

**Crucially: no `family_id` column on `search_trials`** (F4). Instead, compute via JOIN:

```sql
SELECT COALESCE(SUM(st.n_combos), 0)
FROM search_trials st
WHERE st.strategy_name IN (
    SELECT fm.strategy_name FROM family_members fm
    WHERE fm.family_id IN (<set of family_id + ancestor_family_ids>)
)
```

This means ALL of a strategy's trials count toward its current family at query time, regardless
of when they were recorded. No backfill, no denormalized column to keep in sync.

**New 3-way max in `effective_funnel_breadth` (F5):**

```python
def effective_funnel_breadth(
    own_lifetime: int,
    windowed_total: int,
    family_lifetime_effective: int = 0,   # NEW, default = 0 for backward compat
) -> int:
    return max(int(own_lifetime), int(windowed_total), int(family_lifetime_effective))
```

Default of 0 preserves the 2-arg call sites in `factor_cmd.py` and `factor_fdr.py` unchanged.
The promotion path passes `family_lifetime_effective` explicitly.

**Tighten-only proof:** `N_new = max(own, windowed_funnel, family_lifetime_effective) ≥ max(own, windowed_funnel) = N_old` ∀ values. Guaranteed by the `max` operation.

**Breadth snapshotted in `run_gate` (F7, Codex HIGH):** The family assignment is classified in
`promotion_preflight` (which family the strategy belongs to), but `family_lifetime_effective` is
computed immediately before `evaluate_gate` in `run_gate`. This avoids stale values from concurrent
sweeps between preflight and gate evaluation. The `gate_evaluations` row records the snapshotted
value.

### 4. Governed family creation (agent autonomy boundary, F1)

The policy for agent actor and each clustering verdict:

| Verdict | Agent actor | Human actor |
|---------|-------------|-------------|
| MERGE | Auto-assign to incumbent family | Same |
| PARENTAGE | **Auto-assign to parent family (MERGE direction)** — NOT allowed to mint child | Mint child family with parent edge + `family_events` audit row |
| NOVEL | **ValueError** — new family requires `--actor human` | Mint family with `--new-family <slug>` flag |

**F1 fix (CRITICAL):** Agents can only MERGE (assign to an existing family). Both PARENTAGE and
NOVEL are human-only paths. For an agent facing a PARENTAGE verdict, the system assigns the
strategy to the *closest ancestor family* — the stricter direction (more breadth inherited) rather
than letting the agent start a new family with lower initial breadth.

This mirrors `guard_agent_relaxations` in `promotion.py`. The human `--new-family` flag creates a
`family_events` row with `event_type='family_created'`, `actor='human'`, verdict, and score.

**Orphan families (OpenCode M1):** A family created in preflight (before gate evaluation) persists
even if the gate fails — the clustering verdict represents thesis similarity, which is independent
of gate pass/fail. Orphan families with zero members have zero breadth contribution and are
harmless. A future human-triggered cleanup command can remove them if they accumulate.

### 5. Wire family breadth into the gate — `gates.py` (PROTECTED), `promotion.py` (PROTECTED)

`run_gate` resolves the strategy's `family_id` (set in preflight), then:
1. Computes `family_lifetime_effective` via `repo.family_lifetime_combos(family_id)` BFS
2. Passes it as the 3rd arg to `effective_funnel_breadth` (new default arg, no existing caller breakage)
3. The result feeds BOTH the haircut AND the DSR `N` — same seam as before

The `gate_evaluations` row gains three new audit columns: `family_id`, `family_lifetime_effective`,
`windowed_family_strategy_count` (informational; count of strategies whose trials were included).

### 6. Parentage budget inheritance — `family_lifetime_effective` IS the anti-reset

Because `family_lifetime_effective` = own + transitive ancestors' combos (all LIFETIME, never
windowed), a PARENTAGE-minted family's new strategy immediately inherits the parent's full
accumulated breadth. There is no window expiry:

- Parent has 500 lifetime combos → child inherits 500 immediately
- 90 days later: parent's windowed combos = 0, but child's `family_lifetime_effective` STILL
  includes parent's 500 lifetime combos (they're permanent)

This closes the rolling-window anti-reset bypass (OpenCode C1, Codex HIGH-5) completely.

Multi-parent DAG: `family_lifetime_effective` BFS uses a visited-set on family_id to avoid
double-counting shared ancestors. Each trial is counted once.

### 7. Persist backtest return series (Task 7 — deferred return-correlation axis)

`engine.py:674` computes `returns = pf.returns()` and discards it. Task 7 surfaces it:

- New `backtest_returns` blob table: `(id, strategy_name, period_start, period_end, returns_json, created_at)`
- `BacktestResult.returns: pd.Series | None`
- Return-correlation axis in `clustering.py` (min-overlap floor like `MIN_HOLDOUT_OBSERVATIONS`; below it, axis omitted → conservative behavior)
- Engine is a PROTECTED wall; change is minimal (surface already-computed value)

Until Task 7 ships, clustering uses code-ancestry + factor-lineage only. Strategies assigned
under 2-axis clustering are grandfathered (acknowledged gap, same as DSR Phase 1 gap for older
promotions). The `clustering_version` field in `family_events` records which axes were active,
enabling retroactive audit.

### 8. `FamilyBudgetLedger` Protocol — Stratum B interface

```python
class FamilyBudgetLedger(Protocol):
    def global_cap(self) -> float: ...       # total alpha wealth (global LORD++ W)
    def family_wealth(self, family_id: int) -> float: ...   # α_f · W
    def reserve(self, family_id: int, gate_eval_id: int, p_value: float,
                actor: str) -> bool: ...     # returns False if budget exhausted
    def settle(self, family_id: int, gate_eval_id: int, final_p: float) -> None: ...
```

**Reservation sequencing (F6 — Codex/OpenCode HIGH):** `reserve()` is called in `run_gate`
AFTER `evaluate_gate` returns DSR confidence. `p = 1 − dsr_confidence` is known only at that
point. Sequence:

1. Preflight: no ledger mutation
2. Walk-forward + holdout evaluation
3. `evaluate_gate` → compute DSR → know `dsr_confidence`
4. If all non-ledger checks pass → call `reserve(family_id, gate_eval_id, 1-dsr_confidence, actor)`
5. If reservation fails → gate FAIL (append `budget_ledger` check = FAIL)
6. If gate passes + reservation succeeds → record `gate_evaluations` row + transition

`settle()` is called when a strategy is retired/dormant (alpha wealth released back to the family).
`InMemoryFamilyBudgetLedger` fake (no persistence) is the only implementation until Phase 2.

Budget is a PARTITION of one global wealth: `Σ α_f ≤ 1`. Spawning a new family is zero-sum.

---

## Threat model

| Threat | Defense |
|--------|---------|
| Sibling-spam within family | `family_lifetime_effective` accumulates — all member sweeps count |
| Family laundering (relabel clone as new family) | Clustering assigns it to the incumbent via MERGE |
| PARENTAGE minting to get lower breadth | Agents can't mint (PARENTAGE → MERGE-to-parent); humans must use `--new-family` |
| Novel-family minting to dodge cap | Blocked by governed creation; blocked post-Phase-2 by global cap |
| Wait-90-days-then-spawn anti-reset bypass | CLOSED: lifetime combos never decay |
| Threshold miscalibration (looser thresholds → NOVEL instead of MERGE) | Forward-only policy: thresholds can only tighten; `clustering_version` audits which version was active |
| Cycle in DAG → infinite BFS | Atomic `BEGIN IMMEDIATE` prevents cycle creation; visited-set guards traversal |

**Residual gaps (documented, not hidden):**
- Holdings-overlap axis deferred (returns can be gamed if positions are shuffled; mitigated by return-correlation axis in Task 7)
- 2-axis grandfathering for pre-Task-7 assignments (same pattern as DSR Phase 1 gap)
- Lifetime floor only prevents reset; a truly novel-to-family strategy still starts with own_lifetime only

---

## Footprint

| File | Protected | Change |
|------|-----------|--------|
| `algua/registry/db.py` | no | schema 25→26; 4 new tables + `backtest_returns` |
| `algua/research/clustering.py` *(new)* | **YES** (CODEOWNERS, F8) | pure `family_similarity` + `SimVerdict`; protected constants |
| `algua/registry/store.py` | **YES** | `family_lifetime_combos` + parentage BFS accessors; `add_parent_edge` (BEGIN IMMEDIATE); return-series persistence |
| `algua/registry/repository.py` | no | Protocol for new accessors |
| `algua/registry/promotion.py` | **YES** | governed creation in `promotion_preflight`; family breadth in `run_gate` |
| `algua/research/gates.py` | **YES** | 3rd arg `family_lifetime_effective=0` to `effective_funnel_breadth`; new audit fields |
| `algua/backtest/engine.py` | **YES** | surface `pf.returns()` at line ~674 |
| `algua/backtest/result.py` | no | `BacktestResult.returns` field |
| `algua/registry/family_budget.py` *(new)* | no | `FamilyBudgetLedger` Protocol + `InMemoryFamilyBudgetLedger` |
| `algua/cli/` (promote, strategy) | no | `--new-family` human flag; surface family verdict in output |

`clustering.py` is added to CODEOWNERS because its constants directly determine the promotion
gate outcome. An unprotected threshold change could allow strategies to escape family breadth.

---

## Testing strategy

- Schema migration: idempotent; test `_add_missing_columns` patterns, UNIQUE constraints
- Cycle guard: concurrent-session test (`BEGIN IMMEDIATE` holds the lock); visited-set test
- `family_lifetime_effective`: 2-level ancestry; shared-ancestor dedup; cycle-guard fallback
- `effective_funnel_breadth`: **strong tighten-only property** — `N_new ≥ N_old` for all combos
  of args; `windowed_total` dominates when family empty (backward compat); family term dominates
  when established family has large lifetime count
- Governed creation: agent+NOVEL → `ValueError`; agent+PARENTAGE → MERGE-to-parent; human+NOVEL → family_events row; clustering_version recorded
- Atomic cycle guard race: two concurrent `add_parent_edge` calls trying to create A→B and B→A;
  exactly one succeeds, other raises
- Return series: `BacktestResult.returns` carries finite date-indexed series; store round-trips blob; engine still raises on non-finite returns
- `FamilyBudgetLedger` contract: `Σ α_f ≤ 1`; reserve/settle replenish/tighten; exhausted budget → FAIL

---

## Quality gate

`uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`

All four must pass before merge. Protected walls require human review on PR.

---

## Deferred (out of scope for #222)

- Real LORD++ binding in `FamilyBudgetLedger` — Phase 2 (#220)
- Holdings-overlap axis + `backtest_holdings` table
- Retroactive re-clustering of already-promoted strategies (when return correlation lands)
- Lifetime (un-windowed) funnel breadth to close window-evasion for the global term
- Orphan-family cleanup command
- `--family-id` as explicit human override in `registry transition`

---

## GATE 1 panel findings summary (Codex + OpenCode/GLM-5.1, 2026-06-16)

All findings folded into this design. Key changes from the pre-review draft:

| Finding | Severity | Resolution |
|---------|----------|------------|
| Agent PARENTAGE mints a family | CRITICAL | Agent PARENTAGE → MERGE-to-parent; human-only mints |
| Cycle guard not atomic (DAG corruption race) | CRITICAL | `add_parent_edge` = `BEGIN IMMEDIATE` + unique index |
| Windowed family breadth is structurally dead (subset of windowed_funnel) | CRITICAL | Switched to LIFETIME combos for family; windowed_family term removed |
| Anti-reset decays after 90 days | CRITICAL | Resolved by lifetime combos (never decay) |
| `family_id` on `search_trials` under-counts pre-assignment trials | CRITICAL | No column on `search_trials`; JOIN through `family_members` at query time |
| `effective_funnel_breadth` signature breaks factor-FDR caller | HIGH | New 3rd arg `family_lifetime_effective=0` (default preserves 2-arg sites) |
| FamilyBudgetLedger `reserve()` at preflight — DSR not known yet | HIGH | `reserve()` in `run_gate` AFTER `evaluate_gate`; no ledger mutation in preflight |
| Breadth computed at preflight, stale at gate time | HIGH | Recomputed immediately before `evaluate_gate` in `run_gate` |
| `clustering.py` thresholds in unprotected module | MEDIUM | `clustering.py` added to CODEOWNERS |
| `family_events` schema unspecified | MEDIUM | Explicit columns specified above |
| Threshold miscalibration / no versioning | MEDIUM | Forward-only threshold policy; `clustering_version` in all `family_events` rows |
| Schema 24→25 stale (already 25 from feat(219)) | LOW | Phase 4 bumps **25→26** |
