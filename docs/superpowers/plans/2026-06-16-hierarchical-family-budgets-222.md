# TDD Plan — #222 Phase 4: Hierarchical Family Budgets + Anti-Gaming Clustering

**Spec:** `docs/superpowers/specs/2026-06-16-hierarchical-family-budgets-anti-gaming-issue-222-design.md`
**Base:** `origin/main` at `5651f9a` (SCHEMA_VERSION 25; Phase 1 DSR gate + factor FDR merged)

**Pre-flight check before Task 1:**
```bash
uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports
```
Must be green on base before any task begins.

**Protected files (require human review on PR):**

| File | Tasks touching it |
|------|-------------------|
| `algua/research/gates.py` | 5 |
| `algua/research/clustering.py` *(new, CODEOWNERS-protected)* | 2, 7 |
| `algua/registry/store.py` | 1, 3, 6, 7 |
| `algua/registry/promotion.py` | 4, 5 |
| `algua/backtest/engine.py` | 7 |

**Quality gate between every task:**
`uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`

---

## Task 1 — Family registry + parentage DAG (schema 25→26)

**Files:** `algua/registry/db.py`, `algua/registry/store.py`, `algua/registry/repository.py`

Add four new tables and their idempotent migration. Add atomic `add_parent_edge` and ancestry BFS.

### Schema additions in `db.py`

```sql
-- families: canonical family registry (FK-backed; replaces free-text strategies.family)
CREATE TABLE IF NOT EXISTS families (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    name             TEXT NOT NULL UNIQUE,
    created_at       TEXT NOT NULL,
    created_by_actor TEXT NOT NULL,   -- 'human' | 'system'
    created_by_strategy TEXT           -- strategy that triggered creation (if any)
);

-- family_members: many-many, strategy→family assignment (current assignment is the latest row)
CREATE TABLE IF NOT EXISTS family_members (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    family_id    INTEGER NOT NULL REFERENCES families(id),
    strategy_name TEXT NOT NULL,
    joined_at    TEXT NOT NULL,
    joined_by_actor TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_family_members_strategy
    ON family_members(strategy_name);  -- one family per strategy at a time

-- family_parents: parentage DAG (multi-parent allowed; cycle-guarded)
CREATE TABLE IF NOT EXISTS family_parents (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    child_family_id  INTEGER NOT NULL REFERENCES families(id),
    parent_family_id INTEGER NOT NULL REFERENCES families(id)
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_family_parents
    ON family_parents(child_family_id, parent_family_id);

-- family_events: governance audit log
CREATE TABLE IF NOT EXISTS family_events (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type           TEXT NOT NULL,  -- 'family_created'|'strategy_merged'|'parent_edge_added'|'strategy_assigned'
    family_id            INTEGER REFERENCES families(id),
    strategy_name        TEXT,
    actor                TEXT NOT NULL,
    clustering_verdict   TEXT,           -- 'MERGE'|'PARENTAGE'|'NOVEL'|NULL (non-clustering events)
    similarity_score     REAL,
    clustering_version   TEXT,           -- sha256[:16] of weights+thresholds+axis_availability
    axis_json            TEXT,           -- JSON of per-axis scores
    matched_family_id    INTEGER REFERENCES families(id),  -- incumbent matched against
    created_at           TEXT NOT NULL
);
```

`SCHEMA_VERSION` bumps 25 → 26.

### Store methods (all in `store.py`)

- `create_family(name, actor, created_by_strategy=None) -> int` — INSERT + `family_events` row
- `assign_strategy_to_family(strategy_name, family_id, actor, *, verdict, similarity_score, clustering_version, axis_json, matched_family_id=None)` — INSERT into `family_members` (UPSERT on unique strategy) + `family_events` row
- `strategy_family(strategy_name) -> int | None` — current family_id for a strategy (latest member row)
- `family_ancestry(family_id) -> list[int]` — BFS from family_id over `family_parents`, returns ALL ancestor family_ids (visited-set, cycle-safe)
- `add_parent_edge(child_family_id, parent_family_id)` — **ATOMIC** `BEGIN IMMEDIATE`: BFS checks no cycle → INSERT → COMMIT; raises `ValueError` on cycle
- `family_lifetime_combos(family_id) -> int` — JOIN `search_trials` through `family_members` for `family_id` + transitive ancestors (no `family_id` column on `search_trials`; see below)

`family_lifetime_combos` query pattern:
```python
ancestor_ids = [family_id] + self.family_ancestry(family_id)
placeholders = ",".join("?" * len(ancestor_ids))
row = self._conn.execute(
    f"SELECT COALESCE(SUM(st.n_combos), 0) FROM search_trials st"
    f" JOIN family_members fm ON fm.strategy_name = st.strategy_name"
    f" WHERE fm.family_id IN ({placeholders})",
    ancestor_ids,
).fetchone()
return int(row[0])
```

No `family_id` column on `search_trials` (GATE-1 finding: adding it would under-count
pre-assignment trials and require backfill; JOIN through `family_members` is correct).

### Tests

```python
def test_create_family_roundtrips():
    # create_family("momentum") -> id; name unique; family_events row with event_type='family_created'

def test_assign_strategy_upserts():
    # assign_strategy_to_family("strat_a", family_id=1, ...) -> strategy_family("strat_a") == 1
    # reassigning to family 2 -> strategy_family("strat_a") == 2 (UPSERT)
    # family_events has 2 rows for strat_a

def test_family_ancestry_linear():
    # A -> B -> C; family_ancestry(C) == [B, A]

def test_family_ancestry_multi_parent():
    # C has parents A and B; family_ancestry(C) == {A, B} (order-independent)

def test_family_ancestry_visited_set_no_double_count():
    # A -> B, A -> C, B -> D, C -> D (diamond); family_ancestry(D) = {B, C, A}; D not in it

def test_add_parent_edge_cycle_rejected():
    # A -> B; add_parent_edge(A, B) raises ValueError (A reachable from B through A)

def test_add_parent_edge_unique_constraint():
    # add_parent_edge(A, B) twice; second raises (UNIQUE)

def test_add_parent_edge_atomic_race():
    # Two threads: one adds A->B, one adds B->A; exactly one raises ValueError
    # (validates BEGIN IMMEDIATE atomicity)

def test_family_lifetime_combos_empty():
    # family with no members -> 0

def test_family_lifetime_combos_own_only():
    # strat_a in family F; record_search_trial("strat_a", n_combos=50) -> family_lifetime_combos(F) == 50

def test_family_lifetime_combos_with_ancestors():
    # parent family P has strat_p (100 combos); child family C has strat_c (30 combos)
    # add_parent_edge(C, P); family_lifetime_combos(C) == 130

def test_family_lifetime_combos_deduped_on_diamond():
    # Diamond DAG: D->B->A, D->C->A; strat_a in A has 100 combos
    # family_lifetime_combos(D) == 100 (strat_a counted once)

def test_schema_migration_idempotent():
    # migrate() twice; no error; SCHEMA_VERSION == 26
```

---

## Task 2 — Pure clustering similarity + verdict (`algua/research/clustering.py`, **CODEOWNERS-protected**)

**Files:** `algua/research/clustering.py` *(new, CODEOWNERS-protected)*

Pure module: no I/O, no DB access. Analogous to `factor_fdr.py`.

### `SimVerdict` enum + constants

```python
from enum import Enum

class SimVerdict(Enum):
    MERGE = "merge"
    PARENTAGE = "parentage"
    NOVEL = "novel"

# Protected constants — in CODEOWNERS file alongside gates.py
MERGE_THRESHOLD = 0.85       # similarity >= this -> MERGE
PARENTAGE_THRESHOLD = 0.50   # similarity >= this (but < MERGE) -> PARENTAGE

WEIGHT_CODE_ANCESTRY = 0.50
WEIGHT_FACTOR_LINEAGE = 0.30
WEIGHT_RETURN_CORRELATION = 0.20  # 0 until Task 7 activates return axis
```

`clustering_version() -> str`: `sha256(repr({...thresholds+weights+axis_availability}))[:16]`

### `family_similarity(strategy_code_hash, strategy_factors, family_members, *, returns_lookup=None) -> tuple[SimVerdict, float]`

- Compute per-axis similarity against each family member; take the MAX across members (best-match family member determines the verdict)
- Code-ancestry axis: `1.0` if `code_hash` matches; `0.0` otherwise (add partial-match when ancestry graph exists)
- Factor-lineage axis: Jaccard similarity of factor sets
- Return-correlation axis: stubbed `0.0` until Task 7; omitted if `min_overlap` floor not met
- Weighted sum → similarity score → `SimVerdict` by threshold
- Empty family → `(NOVEL, 0.0)` (no members to compare against → first strategy is always NOVEL)
- Non-finite intermediate → fail-closed to `NOVEL` (conservative)
- Axis availability recorded in the returned verdict context (for `clustering_version`)

### Tests

```python
def test_identical_code_hash_is_merge():
    # same code_hash -> score near 1.0 -> MERGE

def test_disjoint_factors_no_code_match_is_novel():
    # different code_hash, no factor overlap -> low score -> NOVEL

def test_threshold_boundary_pinned():
    # score == MERGE_THRESHOLD -> MERGE; score == PARENTAGE_THRESHOLD -> PARENTAGE
    # score just below PARENTAGE_THRESHOLD -> NOVEL

def test_verdict_determinism():
    # same inputs -> same output across multiple calls

def test_empty_family_returns_novel():
    # family_members=[] -> (NOVEL, 0.0)

def test_non_finite_score_fails_to_novel():
    # Inject NaN weight -> result is NOVEL (conservative)

def test_weight_monotonicity():
    # score increases as more axes agree

def test_clustering_version_changes_when_constants_change():
    # Mutate MERGE_THRESHOLD in test -> clustering_version() differs (validates the hash)

def test_factor_lineage_jaccard_reference_value():
    # {A,B,C} vs {B,C,D} -> Jaccard = 2/4 = 0.5 -> check score within tolerance
```

---

## Task 3 — Family-scoped breadth accessor (`store.py`, `repository.py`)

**Files:** `algua/registry/store.py` (PROTECTED), `algua/registry/repository.py`

Add `windowed_family_combos` (for completeness / informational auditing) and ensure
`family_lifetime_combos` (Task 1) is wired into `repository.py`.

`windowed_family_combos(family_id, window_days) -> int` — like `family_lifetime_combos` but
filtered by `created_at >= cutoff`. Used for informational output and `gate_evaluations` audit
field; NOT used in the 3-way max (which uses lifetime).

Update `Repository` Protocol (or ABC) to declare:
- `family_lifetime_combos(family_id: int) -> int`
- `windowed_family_combos(family_id: int, window_days: int) -> int`
- `strategy_family(strategy_name: str) -> int | None`
- `family_ancestry(family_id: int) -> list[int]`

### Tests

```python
def test_windowed_family_combos_filters_by_cutoff():
    # Trial recorded 100 days ago: not in windowed_family_combos(window_days=90)
    # Trial recorded today: included

def test_windowed_family_combos_empty_family():
    # Family with no strategies -> 0

def test_windowed_family_combos_ancestor_included():
    # Parent family's recent trial is included in child's windowed count

def test_family_lifetime_vs_windowed():
    # lifetime includes old trials; windowed only includes recent ones

def test_null_family_strategy_returns_zero():
    # strategy with no family assignment; family_lifetime_combos of any family omits it
```

---

## Task 4 — Governed family creation guard in `promotion_preflight` (`promotion.py`, PROTECTED)

**Files:** `algua/registry/promotion.py` (PROTECTED)

Insert the clustering verdict step at the start of `promotion_preflight`, BEFORE holdout
reservation. This classifies the strategy's family. Breadth numbers are NOT finalized here —
they are snapshotted in `run_gate` (Task 5).

Pseudocode:
```python
# In promotion_preflight, after guard_agent_relaxations():
if strategy.family_id is None:
    verdict, score, version, axis_json = clustering.classify(strategy, repo, *, returns_lookup=None)
else:
    verdict = SimVerdict.MERGE  # already assigned; no re-classification
    score = 1.0

if verdict == SimVerdict.MERGE:
    repo.assign_strategy_to_family(name, matched_family_id, actor, ...)
elif verdict == SimVerdict.PARENTAGE:
    if actor == Actor.AGENT:
        # F1 fix: agent cannot mint a child family; resolve to MERGE toward parent
        repo.assign_strategy_to_family(name, matched_family_id, actor, ...)  # parent family
    else:
        # human: create child family + parent edge
        new_fam_id = repo.create_family(f"{name}_derived", actor=actor.value, ...)
        repo.add_parent_edge(new_fam_id, matched_family_id)
        repo.assign_strategy_to_family(name, new_fam_id, actor, ...)
elif verdict == SimVerdict.NOVEL:
    if actor == Actor.AGENT:
        raise ValueError(
            f"strategy {name!r} has no matching family (clustering verdict: NOVEL). "
            "Assign to a family or use --actor human with --new-family <slug>.")
    else:
        # human: create new family via --new-family flag
        new_fam_id = repo.create_family(new_family_slug, actor=actor.value, ...)
        repo.assign_strategy_to_family(name, new_fam_id, actor, ...)

# family_id is now set on the strategy; record in BreadthContext for run_gate
ctx.family_id = repo.strategy_family(name)
```

### Tests

```python
def test_agent_novel_verdict_raises():
    # New strategy, no existing families, actor=agent -> ValueError

def test_agent_parentage_resolves_to_merge_toward_parent():
    # Existing parent family; new strategy is PARENTAGE similar; actor=agent
    # -> assigned to PARENT family (not a new child family)
    # -> family_events records 'strategy_assigned' with verdict='parentage', assigned to parent

def test_human_novel_creates_family():
    # actor=human, --new-family "momentum_v2" -> family created + family_events row

def test_human_parentage_creates_child_with_parent_edge():
    # actor=human, PARENTAGE verdict -> new child family + parent edge in family_parents
    # -> family_events row event_type='family_created' + 'parent_edge_added'

def test_agent_merge_assigns_to_incumbent():
    # actor=agent, MERGE verdict -> assigned to matched family; no new family

def test_borderline_resolved_stricter():
    # Score exactly at PARENTAGE_THRESHOLD for agent -> treated as MERGE (stricter)
    # (the cluster function handles this; here we test the preflight wiring)

def test_family_events_row_has_clustering_version():
    # After any assignment, family_events row has non-None clustering_version

def test_already_assigned_strategy_skips_reclassification():
    # Strategy already has family_id set -> skips clustering, keeps existing assignment
```

---

## Task 5 — Wire family breadth into the gate (`gates.py` PROTECTED, `promotion.py` PROTECTED)

**Files:** `algua/research/gates.py` (PROTECTED), `algua/registry/promotion.py` (PROTECTED)

### `gates.py` change

```python
def effective_funnel_breadth(
    own_lifetime: int,
    windowed_total: int,
    family_lifetime_effective: int = 0,  # NEW — default 0 preserves all 2-arg callers
) -> int:
    """3-way max (tighten-only): own lifetime, funnel-wide windowed, family+ancestor lifetime.
    family_lifetime_effective=0 (default) is byte-identical to the prior 2-arg behavior."""
    return max(int(own_lifetime), int(windowed_total), int(family_lifetime_effective))
```

No other change to `gates.py` — the existing 2-arg callers (`factor_cmd.py`, `factor_fdr.py`)
continue to work unchanged (family_lifetime_effective defaults to 0 → 2-way max, as before).

### `promotion.py` change — `run_gate`

Immediately before calling `evaluate_gate`, recompute family breadth (F7 fix):
```python
family_id = repo.strategy_family(name)   # from preflight classification
family_lifetime_effective = repo.family_lifetime_combos(family_id) if family_id else 0
```

Pass it to `effective_funnel_breadth` as the 3rd arg. Record `family_id` and
`family_lifetime_effective` in the `gate_evaluations` row (requires two new audit columns).

### New `gate_evaluations` columns (via `_add_missing_columns`)

```python
_add_missing_columns(conn, "gate_evaluations", [
    ("family_id", "INTEGER"),
    ("family_lifetime_effective", "INTEGER"),
])
```

### Tests

```python
def test_effective_funnel_breadth_3way_max():
    # max(10, 5, 8) == 10; max(10, 5, 15) == 15; max(10, 15, 8) == 15

def test_effective_funnel_breadth_default_backward_compat():
    # effective_funnel_breadth(10, 15) == 15  (2-arg, family=0 default)

def test_tighten_only_strong_property():
    # For any (own, windowed_total, family_lifetime_effective):
    # effective_funnel_breadth(own, windowed_total, family_lifetime_effective)
    #   >= effective_funnel_breadth(own, windowed_total)  (N_new >= N_old always)
    # Test with 50 random combinations

def test_crowded_family_raises_bar():
    # Family with 1000 lifetime combos; strategy own=50, windowed_funnel=200
    # -> effective breadth = 1000 (family dominates)

def test_empty_family_fallback_unchanged():
    # family_id=None -> family_lifetime_effective=0
    # -> max(own, windowed_funnel, 0) == max(own, windowed_funnel) (same as before)

def test_gate_evaluations_row_records_family_fields():
    # After a full promotion run: gate_evaluations row has family_id and
    # family_lifetime_effective matching the snapshotted values

def test_breadth_snapshotted_in_run_gate_not_preflight():
    # Add combos to family AFTER preflight but BEFORE evaluate_gate call
    # -> gate_evaluations row reflects the LATER (higher) value
    # (validates that breadth is read in run_gate, not carried stale from preflight)
```

---

## Task 6 — Parentage budget inheritance is already complete (anti-reset)

`family_lifetime_combos` from Task 1 already does the right thing: it sums ALL combos of ALL
member strategies across ALL ancestor families via BFS. Because it uses LIFETIME (not windowed)
counts, the anti-reset is permanent — ancestor combos never decay.

**Task 6 is therefore verification-only:** write the property test that confirms the inheritance
holds after the 90-day window would have expired.

### Tests

```python
def test_lifetime_inheritance_survives_window_expiry():
    # record_search_trial("strat_parent", n_combos=500) with created_at = 200 days ago
    # Parent family P has strat_parent; child family C has strat_child (5 combos, today)
    # add_parent_edge(C, P)
    # windowed_family_combos(C, window_days=90) == 5  (parent's old trial out of window)
    # family_lifetime_combos(C) == 505               (lifetime includes parent's 500)
    # effective_funnel_breadth(5, 5, 505) == 505      (anti-reset holds)

def test_two_level_lifetime_inheritance():
    # A -> B -> C; each family has 100 combos
    # family_lifetime_combos(C) == 300

def test_diamond_ancestry_no_double_count():
    # D->B->A, D->C->A; A has 100 combos
    # family_lifetime_combos(D) == 100 (A counted once despite two paths)

def test_cycle_guard_prevents_infinite_bfs():
    # Manually corrupt family_parents to create a cycle (bypass add_parent_edge)
    # family_lifetime_combos() must not hang (visited-set protection)
```

---

## Task 7 — Persist backtest return series + return-correlation axis

**Files:** `algua/backtest/engine.py` (PROTECTED), `algua/backtest/result.py`,
`algua/registry/db.py`, `algua/registry/store.py` (PROTECTED), `algua/research/clustering.py` (PROTECTED)

### Schema additions

```sql
CREATE TABLE IF NOT EXISTS backtest_returns (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_name  TEXT NOT NULL,
    period_start   TEXT NOT NULL,
    period_end     TEXT NOT NULL,
    returns_json   BLOB NOT NULL,   -- JSON array of [date_str, float] pairs
    created_at     TEXT NOT NULL
);
```

### `engine.py` change (PROTECTED — minimal diff)

At the existing line that computes `returns = pf.returns()` and discards it (~line 674):
- Surface it on `BacktestResult` as `returns: pd.Series | None`
- If returns has any non-finite values → set to `None` (same fail-closed convention)

### Return-correlation axis in `clustering.py`

```python
def _return_correlation_axis(strategy_returns, family_members_returns, *, min_overlap=63) -> float:
    """Max correlation between strategy_returns and any family member's returns over the
    shared date range. Returns 0.0 if any pair has fewer than min_overlap shared dates
    (axis omitted → conservative, same as missing axis)."""
```

Correlation axis is active when `returns_lookup` is provided and overlap ≥ `min_overlap`.
`WEIGHT_RETURN_CORRELATION = 0.20` is now non-zero. Update `clustering_version()`.

### Tests

```python
def test_backtest_result_carries_returns():
    # Run simulate(); result.returns is a finite date-indexed pd.Series

def test_backtest_returns_table_roundtrip():
    # Persist and reload returns; values match to float precision

def test_engine_still_raises_on_nonfinite_returns():
    # Mock pf.returns() to return NaN -> result.returns is None (not surfaced)

def test_return_correlation_reference_value():
    # Perfectly correlated returns -> axis score 1.0
    # Anticorrelated returns -> axis score near 0.0 (or however axis maps it)

def test_return_correlation_below_min_overlap_omitted():
    # Only 30 shared dates (< 63 min_overlap) -> axis contributes 0.0
    # verdict is determined by code+factor axes only

def test_clustering_version_changes_when_return_axis_activated():
    # Before: WEIGHT_RETURN_CORRELATION = 0 in clustering_version hash
    # After Task 7: weight = 0.20 -> different clustering_version
```

---

## Task 8 — `FamilyBudgetLedger` Protocol + `InMemoryFamilyBudgetLedger` (Stratum B)

**Files:** `algua/registry/family_budget.py` *(new)*

```python
from typing import Protocol

class FamilyBudgetLedger(Protocol):
    """Stratum B: partition of global LORD++ alpha wealth across families.
    Real implementation deferred to Phase 2 (#220). p = 1 - dsr_confidence (Phase 1 convention).
    reserve() is called in run_gate AFTER evaluate_gate computes DSR; NOT in promotion_preflight."""

    def global_cap(self) -> float:
        """Total alpha wealth W_global."""

    def family_wealth(self, family_id: int) -> float:
        """Current unallocated alpha wealth for family_id (α_f · W_global remaining)."""

    def reserve(
        self, family_id: int, gate_eval_id: int, p_value: float, actor: str
    ) -> bool:
        """Allocate p_value of alpha from family_id's budget for gate_eval_id.
        Returns True if budget allows; False if exhausted (promotion blocked)."""

    def settle(self, family_id: int, gate_eval_id: int, final_p: float) -> None:
        """Release or adjust the reservation after strategy is retired/dormant."""


class InMemoryFamilyBudgetLedger:
    """Fake implementation for tests. No persistence. Single global wealth pool."""

    def __init__(self, global_cap: float = 1.0) -> None:
        self._cap = global_cap
        self._reservations: dict[int, float] = {}  # gate_eval_id -> p_value

    def global_cap(self) -> float:
        return self._cap

    def family_wealth(self, family_id: int) -> float:
        allocated = sum(self._reservations.values())
        return max(0.0, self._cap - allocated)

    def reserve(self, family_id: int, gate_eval_id: int, p_value: float, actor: str) -> bool:
        allocated = sum(self._reservations.values())
        if allocated + p_value > self._cap + 1e-10:
            return False
        self._reservations[gate_eval_id] = p_value
        return True

    def settle(self, family_id: int, gate_eval_id: int, final_p: float) -> None:
        if gate_eval_id in self._reservations:
            del self._reservations[gate_eval_id]
```

### Tests

```python
def test_reserve_within_cap():
    ledger = InMemoryFamilyBudgetLedger(global_cap=1.0)
    assert ledger.reserve(1, 101, p_value=0.3, actor="agent") is True
    assert ledger.family_wealth(1) == pytest.approx(0.7)

def test_reserve_exhausts_budget():
    ledger = InMemoryFamilyBudgetLedger(global_cap=0.2)
    ledger.reserve(1, 101, p_value=0.15, actor="agent")
    assert ledger.reserve(1, 102, p_value=0.10, actor="agent") is False  # 0.25 > 0.20

def test_settle_releases_wealth():
    ledger = InMemoryFamilyBudgetLedger(global_cap=1.0)
    ledger.reserve(1, 101, p_value=0.5, actor="agent")
    ledger.settle(1, 101, final_p=0.5)
    assert ledger.family_wealth(1) == pytest.approx(1.0)

def test_sum_alpha_never_exceeds_cap():
    # 5 concurrent reserves summing to exactly cap -> all succeed; one more fails
    ledger = InMemoryFamilyBudgetLedger(global_cap=1.0)
    for i in range(5):
        ledger.reserve(1, i, p_value=0.2, actor="agent")
    assert ledger.reserve(1, 99, p_value=0.01, actor="agent") is False

def test_protocol_structural_conformance():
    # InMemoryFamilyBudgetLedger satisfies FamilyBudgetLedger Protocol
    from typing import get_type_hints
    ledger: FamilyBudgetLedger = InMemoryFamilyBudgetLedger()  # type: ignore[assignment]
    # mypy checks this; test just confirms instantiation
```

---

## Task 9 — Quality gate + umbrella spec + docs

### Quality gate

```bash
uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports
```

All four green. Check that `clustering.py` appears in `.github/CODEOWNERS` alongside `gates.py`.

### Umbrella spec update

In `docs/superpowers/specs/2026-06-15-streaming-funnel-multiple-testing-issue-211-design.md`,
expand the Phase 4 paragraph (line ~288) to link the new design spec and confirm it is built.

### CLAUDE.md update

Add `family_lifetime_combos` and governed-creation semantics to the command surface description
of `research promote`.

---

## Self-Review Checklist

- [ ] `effective_funnel_breadth` has `family_lifetime_effective: int = 0` default; all 2-arg callers unchanged
- [ ] `add_parent_edge` uses `BEGIN IMMEDIATE`; unique index on `(child_family_id, parent_family_id)`
- [ ] No `family_id` column added to `search_trials`; `family_lifetime_combos` JOINs through `family_members`
- [ ] Agent + NOVEL → `ValueError`; agent + PARENTAGE → MERGE-to-parent; human + NOVEL → new family
- [ ] `family_lifetime_effective` computed in `run_gate` (not carried stale from preflight)
- [ ] `FamilyBudgetLedger.reserve()` called in `run_gate` AFTER `evaluate_gate`, never in preflight
- [ ] `clustering.py` in CODEOWNERS
- [ ] `family_events` has `clustering_version`, `axis_json`, `actor`, `similarity_score`
- [ ] SCHEMA_VERSION = 26 (25 was feat(219) factor_evaluations); migration idempotent
- [ ] All 4 quality checks green

---

## GATE-1 findings summary (for reviewer context)

GATE-1 panel: Codex (GPT-5-codex) + OpenCode (GLM-5.1), 2026-06-16. All findings folded into
this plan. See the design spec for the full triage table and rationale for each decision.

Major pre-review-to-post-review changes: switched from windowed to lifetime combos for family
breadth (resolves anti-reset decay AND windowed_family dead-term problem); removed `family_id`
from `search_trials` (JOIN through `family_members` instead); made `add_parent_edge` atomic with
`BEGIN IMMEDIATE`; moved agent PARENTAGE to MERGE-to-parent (can't mint); added `family_lifetime_effective=0`
default to `effective_funnel_breadth` (preserves factor-FDR caller); moved `FamilyBudgetLedger.reserve()`
call to `run_gate` post-DSR; added `clustering.py` to CODEOWNERS; specified `family_events` schema.
