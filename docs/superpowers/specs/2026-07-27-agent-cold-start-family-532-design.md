# Agent cold-start: let an agent found family #0 on an empty registry (#532)

Status: IMPLEMENTED (post-GATE-2 fixes) — Findings 1, 3a, 4 landed; §4 files-touched and §3 Finding 4
reflect the as-shipped test layout (coverage co-located in existing suites, NO dedicated test module)
Issue: #532 — "Agent cold-start: let an agent found family #0 on an empty registry (extend #524 deferred-mint)"
Depends on: #524 (agent-NOVEL deferred pass-time mint + seeded lifetime prior), #529 (gate recalibration)
CODEOWNERS: touches `algua/registry/promotion.py` + `algua/registry/store.py` → PR stays OPEN for human merge (no auto-merge).

---

## 1. Problem

On a clean/empty family registry an agent `research promote` fails closed at
`algua/registry/promotion.py:277-284`: the NOVEL + `actor is AGENT` branch raises
`ValueError("… the family registry is empty — no existing family to merge into …")` and demands a
human run `research promote --actor human --new-family <slug>` first. #524 made autonomous
NOVEL-family creation work only when families already exist; the **cold-start (founding family #0)
was intentionally left human-only**. This is the last human-in-the-loop dependency for a fully
autonomous research loop, and it re-appears on every funnel reset.

Founding family #0 is the single case with **nothing to game**: there are no sibling families to
dodge a multiplicity tax against, and the return-correlation classifier has no comparison set, so a
NOVEL verdict on a truly empty registry is trivially correct. Letting an agent found #0 therefore
weakens no multiple-testing defense.

## 2. Core change (unchanged from the reviewed design)

Extend #524's **deferred pass-time mint** to the empty-registry case: the agent-NOVEL path falls
through to the **same** `PendingNovelFamily` deferral the non-empty NOVEL-agent path already uses
(`promotion.py:285-303`). The family is minted **only at the pass moment** inside the atomic promote
transaction by the existing `_mint_agent_novel_family` (store.py) — **no new mint code path**. The
seed is the same funnel-lifetime prior (`agent_novel_mint_seed()`), the rate cap
(`AGENT_NOVEL_MINT_CAP`) applies unchanged, and the human `--new-family` cold-start path is
untouched.

This revised design (round 2) exists to resolve the **four GATE-1 blocking findings** below.
Everything in §2's mechanism that GATE-1 did not fault is inherited as-is from #524.

---

## 3. Resolution of GATE-1 findings

### Finding 1 (BLOCKING) — precise empty-registry check, NOT a blanket deletion

**Problem.** The reviewed design deleted the `if not _has_any_family: raise` block outright.
`_has_any_family` is a **byproduct of the clustering rank loop over ACTIVE members**
(`promotion.py:198-210` `any_family`, sourced from `all_families_with_member_profiles()`, which
returns "all families **with active members**" — store/repository). So `_has_any_family` is `False`
in **two distinct** states:
  (a) a true cold-start — the `families` table is genuinely empty; and
  (b) families exist but **every** family has zero active members (all members removed / orphaned).

State (b) is NOT covered by the issue's "nothing to game" argument: real (if dead) sibling families
exist, their historical breadth/lineage is on the graph, and letting an agent silently found a fresh
near-zero-prior family alongside them is exactly the multiplicity-evasion #524 defends against.

**Resolution.** Add a dedicated table-emptiness read `repo.family_count()` (a fresh
`SELECT COUNT(*) FROM families`) and gate on it precisely. Replace the blanket refusal with:

```python
# NOVEL verdict
if actor is Actor.AGENT:
    if not _has_any_family and repo.family_count() > 0:
        # NOT a cold-start: families exist but none has an ACTIVE member (all removed/orphaned).
        # Real (if dead) sibling families are on the graph, so the founding-#0 "nothing to game"
        # argument does NOT hold — fail closed and require a human, as pre-#532.
        raise ValueError(
            f"strategy {name!r}: the family registry has {repo.family_count()} family(ies) but "
            "none with an active member. An agent cannot found a new family alongside existing "
            "(if dormant) families; a human must intervene via "
            "`research promote --actor human --new-family <slug>`."
        )
    # Cold-start (family_count()==0) OR normal non-empty (has active members): fall through to the
    # #524 deferred PendingNovelFamily mint. No branch on emptiness in the seed or the mint.
    fp_after = repo.family_graph_fingerprint()
    ...  # unchanged #524 CAS + PendingNovelFamily construction
```

Only state (b) raises. State (a) (cold-start) and the ordinary non-empty case both fall through to
the identical deferred mint.

**Why the `family_count()` read is race-safe (no new TOCTOU).** The families-table count is already
**fingerprint component 0**: `family_graph_fingerprint()` (store.py:708) begins with
`_cm("SELECT COUNT(*), COALESCE(MAX(id),0) FROM families")`. `fp_before` is captured at the top of
classification and `fp_after == fp_before` is asserted before the `PendingNovelFamily` is built, and
re-asserted in `_revalidate_pending_novel` and under the promote write lock. So a concurrent mint
between the `family_count()` read and the fingerprint CAS strictly bumps component 0 and trips the
existing CAS (`FamilyGraphDriftError`, fail-closed) — the cold-start branch decision is therefore
covered by the same CAS that already guards the NOVEL verdict. `family_count()` is read at
classification time, inside the fp_before…fp_after window, so it observes the same snapshot the
verdict was computed on.

`family_count()` is a new **read-only** method on the `StrategyRepository` Protocol
(`algua/registry/repository.py`, NOT CODEOWNERS-protected) + its `SqliteStrategyRepository`
implementation (`algua/registry/store.py`, CODEOWNERS). `_has_any_family` remains in use (it still
distinguishes state (b) from cold-start), so it is NOT dropped.

### Finding 2 (BLOCKING) — founder-tax asymmetry: option (a), inherited from #524, nothing new

The concern: the founder's gate is evaluated at `family_lifetime_effective = 0` (its family does not
exist yet) while its family is retroactively seeded non-zero for future siblings.

**Confirmed: #524 already evaluated and ACCEPTED exactly this asymmetry for the non-empty agent-NOVEL
case.** From the #524 design (`docs/superpowers/specs/2026-07-18-novel-family-agent-seed-524-design.md`):

- §4 "What the seed buys": *"Because the agent family is created only after the gate passes, it does
  not exist during gate evaluation. So for the FOUNDING promotion the family arm is 0 and
  `n_funnel = max(own, windowed_total)` — exactly today's bar, and identical to what a human-created
  fresh family founder pays. The seed therefore has zero effect on the founding pass … The seed's
  entire, load-bearing purpose is future siblings."*
- §6A: a **durable funnel-lifetime founder tax** (a per-founder statistical bar) was **deliberately
  NOT built**; the repeated-founder attack (mint N one-founder families >90 days apart) is bounded
  **SOLELY by the per-window rate cap** `AGENT_NOVEL_MINT_CAP` (R5-HIGH-2, made load-bearing).

**#532 introduces nothing new here.** A cold-start founder is evaluated at family arm 0 exactly like
every other agent-NOVEL founder, and family #0 is seeded with the funnel-lifetime prior for its
future siblings exactly like any agent-NOVEL family. Cold-start is in fact **strictly more benign**:
family #0 has no sibling families whose breadth it could dodge, and it is the first row so
`windowed_total`/`own` fully describe its own search effort. The repeated-founder surface is
identical to #524's and bounded by the SAME rate cap.

**We take option (a): document and accept.** We explicitly REJECT option (b) (folding the pending
mint's seed into the founder's own gate-evaluated breadth) because it would:
  - diverge the agent founder from the **human** fresh-family founder, who also pays arm 0 (a new
    agent/human asymmetry, the opposite of what #524 worked to remove); and
  - contradict the value actually recorded in the founder's gate row (`family_lifetime_effective=0`,
    `family_id=NULL`), reintroducing the Finding-3c inconsistency below.

The founder-tax governance is therefore **wholly inherited from #524** — #532 changes only *which
registry states* reach the deferred mint (Finding 1), not the tax model.

### Finding 3 (MEDIUM, addressed in this pass)

**3a — move the `agent_novel_mint_seed() > 0` guard into preflight (avoid a burned holdout).**
Today the seed-positivity guard fires only inside `_mint_agent_novel_family` (store.py:2239-2243),
which runs under the promote write lock **after** the holdout peek/burn. A promote that is certain to
fail the seed guard (e.g. a founder whose funnel has no well-typed in-range `search_trials`) would
still burn its single-use holdout before the mint rejects it.

Resolution: add a pure-read seed-positivity check to `_revalidate_pending_novel`
(`promotion.py:438-453`), which already runs as the LAST preflight step before the holdout peek and
is documented as "all pure DB reads (no holdout, no gate row), safe to call both pre-peek and at the
atomic burn":

```python
def _revalidate_pending_novel(repo, name, pending) -> None:
    repo.check_agent_novel_mint_bounds()
    if repo.agent_novel_mint_seed() <= 0:            # #532 (3a): fail closed BEFORE the holdout peek
        raise ValueError(
            f"strategy {name!r}: agent-NOVEL mint requires a strictly-positive funnel-lifetime "
            "breadth seed; the funnel has no well-typed in-range search_trials to seed (fail closed)")
    if repo.strategy_family(name) is not None:
        ...  # unchanged
    if repo.family_graph_fingerprint() != pending.graph_fingerprint:
        ...  # unchanged
```

The authoritative under-lock guard in `_mint_agent_novel_family` is **kept unchanged** as
defense-in-depth (the preflight read is advisory-early, not a replacement — a concurrent change to
`search_trials` between preflight and commit is still caught under the lock). This improvement
applies to ALL agent-NOVEL mints, not just cold-start; #524 deferred it, #532 lands it since it is on
the same code path and cheap.

Note the intended cold-start scenario (families-empty but `search_trials` non-empty from the
founder's own `backtest sweep`) has `seed > 0` and is unaffected. A **literally** first-ever promote
with zero recorded search breadth anywhere fails closed at preflight now, with no holdout burned —
which is the desired behavior (a strategy cannot reach `research promote` without a prior sweep that
records `search_trials`).

**3b — document that the rate cap is scoped to the connected registry DB.**
`check_agent_novel_mint_bounds()` counts rows in the `families` table of the **currently connected
SQLite registry** (`created_by_actor='agent'`, canonical-UTC `created_at` in the last
`AGENT_NOVEL_MINT_WINDOW_DAYS`). It is therefore **per-DB-file**, not global across environments. The
design records this explicitly:
  - staging/CI must use a **separate registry DB file** from prod (already the norm — each
    environment owns its `registry.sqlite`); an agent promote in staging consumes staging's rate-cap
    budget only.
  - if a DB file is ever wiped/rotated, the rate-cap window resets with it (the cap counts live rows,
    not an append-only external ledger). This is acceptable because the rate cap is a *throughput
    bound on autonomous minting within one registry*, not a cross-environment audit trail; a wiped
    prod registry is a catastrophic operator event with far larger consequences than a reset mint
    window. No code change — this is a documented operating constraint (added to the design and to the
    `check_agent_novel_mint_bounds` docstring context; see §5).

**3c — the founding gate row keeps `family_id = NULL` (inherited #524 decision, no change).**
The reviewed design asked whether the `gate_evaluations` row for a founding pass should carry the
resulting `family_id`. #524 explicitly considered and **rejected** this (design finding R3-MED-1):
*"stamping the created `family_id`/seed onto the just-evaluated gate row contradicts the evaluated
`family_id=None`/`family_lifetime_effective=0` in `n_funnel`/`decision_json`."* The gate row must
faithfully record **what was evaluated**, and the founder was evaluated with no family (family arm 0).

Audit/analytics continuity is already provided by the **reverse** link:
`families.founder_gate_id` is a FK to `gate_evaluations(id)` (db.py:589-591), stamped by
`_mint_agent_novel_family` (store.py:2257) at mint time. So the founding gate row ← family linkage is
queryable (`SELECT * FROM families WHERE founder_gate_id = ?`) without corrupting the gate row's
recorded evaluation. #532 **keeps `family_id = NULL`** on the founder's gate row and adds a design
note that the reverse FK is the intended audit join. (`gate_evaluations` is not append-only-triggered,
so a back-stamp is *possible*, but it is deliberately not done for the consistency reason above.)

### Finding 4 (BLOCKING) — tests

**As shipped: no dedicated `tests/registry/test_agent_cold_start_family_532.py` module was created.**
The cold-start coverage is co-located with the code it exercises in the two existing suites that
already own these seams (`tests/test_registry_store.py` for the store-level atomic-mint tx and the
`family_count()` accessor; `tests/registry/test_family_creation_guard.py` for the
`promotion_preflight` / `_classify_and_assign_family` / `_revalidate_pending_novel` seams), plus
updates to three existing tests that incidentally asserted the old raise — see §6. The seven behaviors
below are all covered:

1. **Truly-empty families table → agent PASS founds family #0.** `test_cold_start_pass_mints_family_zero`
   (`tests/test_registry_store.py`): empty `families`, `search_trials` seeded (`seed > 0`), agent
   NOVEL, gate PASS → founds family #0 at the pass moment, founder assigned as the founding member,
   `created_by_actor='agent'`, `founder_gate_id` = the founding gate row, `seeded_prior_combos` = the
   funnel-lifetime prior, exactly one `families` row.
2. **Empty families table, gate FAIL → mints nothing.** `test_cold_start_fail_mints_nothing`
   (`tests/test_registry_store.py`): deferred-mint semantics unchanged; strategy stays BACKTESTED,
   zero `families` rows.
3. **Families non-empty but zero active members → agent still fail-closed (Finding 1).**
   `test_agent_novel_families_exist_but_no_active_member_fails_closed`
   (`tests/registry/test_family_creation_guard.py`): a family whose only member is soft-deleted so
   `_has_any_family` is `False` while `family_count() > 0`; preflight raises the "none with an active
   member" `ValueError` (before the holdout is touched) and mints no family.
4. **Concurrent cold-start → loser gets `FamilyGraphDriftError`, no double-found.**
   `test_cold_start_concurrent_double_founding_prevented` (`tests/test_registry_store.py`): the pending
   spec captures the empty-graph fingerprint, a concurrent founder mutates the graph, and the
   under-lock CAS in `record_gate_with_fdr_and_maybe_promote` raises `FamilyGraphDriftError`, rolls
   back (no gate row written), leaving exactly one family and no double-found.
5. **Founder seed = 0 rejected at PREFLIGHT, not just mint (Finding 3a).**
   `test_agent_novel_seed_zero_rejected_at_preflight_before_holdout`
   (`tests/registry/test_family_creation_guard.py`): calls `_revalidate_pending_novel` (the last
   preflight step BEFORE the holdout peek) with an empty-funnel repo (`seed == 0`) and asserts the
   strictly-positive-seed `ValueError` — pure DB reads, no holdout burned. (Through full
   `promotion_preflight` the empty-`search_trials` case is shadowed by the earlier "no recorded search
   breadth" breadth gate; this pre-peek seed guard is the defense that ALSO covers a migrated-DB
   corrupt/overlarge `search_trials` row — counted by `total_search_combos` but excluded from the
   WHERE-filtered lifetime seed — so it is exercised directly at its preflight pre-peek position.)
6. **Human `--new-family` cold-start path unchanged.** Covered by the pre-existing
   `test_human_novel_creates_family` / `test_human_novel_family_has_zero_seed`
   (`tests/registry/test_family_creation_guard.py`) — a human NOVEL on an empty registry still creates
   a fresh zero-prior root family in preflight.
7. **`family_count()` unit test.** `test_family_count_raw_table_count`
   (`tests/test_registry_store.py`): returns 0 on a fresh DB, increments as families are created,
   counts a family whose members are all soft-deleted (a table count, not an active-member count).

Also added: `test_agent_novel_cold_start_defers_on_empty_registry`
(`tests/registry/test_family_creation_guard.py`) — the preflight-level assertion that an agent NOVEL
on a truly-empty registry returns the deferred `PendingNovelFamily` spec (no family created in
preflight).

Not implemented as a separate test: the full-`promotion_preflight` variant of #1's PASS-mints-#0
path (it would require driving `walk_forward` + the holdout burn end-to-end); the mint mechanism is
instead exercised at the store `record_gate_with_fdr_and_maybe_promote` seam (#1/#2/#4 above), which
is where the atomic pass-moment mint actually lives.

Existing tests to update (§6): the three that use the old cold-start raise as an incidental terminal.

---

## 4. Files touched

| File | Change | CODEOWNERS |
|---|---|---|
| `algua/registry/promotion.py` | Finding 1 precise `family_count()` gate replacing the blanket raise; Finding 3a seed>0 check in `_revalidate_pending_novel`. | YES → PR open |
| `algua/registry/store.py` | New `family_count()` read impl; docstring note for 3b (rate cap per-DB scope). | YES → PR open |
| `algua/registry/repository.py` | Add `family_count()` to the `StrategyRepository` Protocol. | no |
| `tests/test_registry_store.py` | Finding 4 tests 1, 2, 4 (store atomic-mint tx) + test 7 (`family_count()`). | no |
| `tests/registry/test_family_creation_guard.py` | Finding 4 tests 3, 5 + the cold-start preflight-deferral test; test 6 is pre-existing. NO new dedicated `test_agent_cold_start_family_532.py` module — coverage co-located with the seams it exercises. | no |
| `tests/test_fundamentals_guards.py`, `tests/test_promotion_needs_fundamentals.py`, `tests/test_promotion_needs_news.py` | Update the incidental `match="family registry is empty"` assertions (§6). | no |
| this design doc + plan | docs. | no |

No schema change (no `db.py` SCHEMA_VERSION bump). `family_count()` reads an existing table; all
#524 columns/triggers/FKs are reused as-is.

## 5. Documentation notes carried into the design

- **Founder tax (Finding 2):** the founder pays family arm 0 and family #0 is seeded for its future
  siblings — identical to any #524 agent-NOVEL founder and to a human fresh-family founder; bounded
  by the rate cap, not a durable per-founder statistical tax (inherited, §3 Finding 2).
- **Rate-cap scope (Finding 3b):** per connected registry DB file; staging/CI must not share a
  registry DB with prod; a DB wipe resets the window. Documented in this spec and reflected near
  `check_agent_novel_mint_bounds`.
- **Audit join (Finding 3c):** founder gate row keeps `family_id=NULL`; join families→gate via
  `families.founder_gate_id`.

## 6. Existing-test migration

`tests/test_fundamentals_guards.py:41`, `tests/test_promotion_needs_fundamentals.py:69`,
`tests/test_promotion_needs_news.py:68` currently assert `pytest.raises(ValueError, match="family
registry is empty")` — they used an empty registry purely as a convenient terminal to prove preflight
"got past" the needs_fundamentals / needs_news PIT check. After #532 an agent preflight on an empty
registry no longer raises that; depending on whether they seed `search_trials` it will either (i)
return a `BreadthContext` carrying a `pending_novel_family` (seed>0), or (ii) raise the new
strictly-positive-seed `ValueError` (seed==0, no sweep recorded). Each test is updated to assert the
guard it actually intends to test — either by seeding a non-matching family + `search_trials` and
asserting the pending spec, or by matching the new seed error — so it no longer depends on the
removed cold-start refusal. This is a mechanical migration, not a behavior regression.

## 7. Acceptance (from the issue, unchanged)

- Agent `research promote` on an empty registry, NOVEL, gate PASS → founds family #0 at the pass
  moment, assigns the founder, consumes one rate-cap mint. Gate FAIL → no family created, holdout
  burned as usual.
- Concurrent cold-start does not double-found (drift → re-run).
- Rate cap still bounds cold-start mints.
- Human `--new-family` cold-start path unchanged.
- Families-exist-but-no-active-member state still fails closed (Finding 1).
- Seed==0 rejected at preflight without burning the holdout (Finding 3a).
- CODEOWNERS gate-core (`promotion.py`/`store.py`) → PR stays open for human merge.

## 8. Task list

1. **Task 0 — re-run GATE-1 (Codex, read-only)** on this revised design before implementation.
2. **Task 1 — `family_count()` accessor** (repository Protocol + store impl) + unit test (Finding 4.7).
3. **Task 2 — promotion.py Finding 1** precise cold-start gate replacing the blanket raise; keep
   `_has_any_family` for state (b). Tests 1, 3 (+ human path regression test 6).
4. **Task 3 — promotion.py Finding 3a** seed>0 check in `_revalidate_pending_novel`; store 3b/3c
   doc notes. Tests 2, 5.
5. **Task 4 — concurrency test** (Finding 4.4) + migrate the three incidental existing tests (§6).
6. **Task 5 — integration:** FULL gate (`uv run pytest -q && uv run ruff check . && uv run mypy
   algua && uv run lint-imports`) + open the CODEOWNERS PR (stays open for human merge).
