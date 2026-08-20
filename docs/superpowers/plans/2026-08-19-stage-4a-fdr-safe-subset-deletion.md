# Stage 4a — FDR Safe-Subset Deletion + Constant Relocation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete the LORD++ online-FDR "binding" machinery that is provably dead in production (`research/fdr_lord.py`'s math + the `if fdr_binding:` branch it feeds inside `registry/store.py`'s `record_gate_with_fdr_and_maybe_promote`), while relocating the two symbols (`fdr_cohort_position`, `FDR_COHORT_SIZE`) that a genuinely live, run-on-every-bootstrap `db.py` migration still needs. The `gate_evaluations` table schema, its historical rows, and the surviving non-binding promotion path are untouched.

**Architecture:** This is Stage 4a of the `registry/store.py` per-Protocol carve program (Stage 4b/4c follow as separate plans). Unlike Stage 3's pure structural carve, this is a **kill-list-style deletion** (matching Stage 1's shape): dead code + its exclusively-dead tests are removed, one still-live-but-misplaced pair of symbols is relocated (not deleted), and — because the deletion touches a shared method that many tests call — a careful per-test triage separates tests whose *subject* is the deleted branch (delete) from tests that merely passed an incidental non-`None` `p_value` while testing something else that survives untouched (fix, don't delete). That triage was done exhaustively during this plan's research and is given verbatim below — do not re-derive it from scratch, but do re-verify each cited line against the file in front of you (line numbers are a snapshot as of this plan's writing).

**Tech Stack:** Python 3.12, uv, pytest, ruff, mypy, import-linter, sqlite3.

**Spec:** `docs/superpowers/specs/2026-08-18-system-simplification-design.md` §3 (kill-list KEEP paragraph: "Frozen LORD++ FDR machinery") and §6 item 2 ("the `store.py → research.gates` constants import is fixed").

## Global Constraints

- Quality gate on EVERY task before commit: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`. All four must pass.
- **No schema change.** `gate_evaluations`'s `fdr_*` columns and every historical row stay exactly as they are (spec: "The ledger schema and recorded rows remain"). Do not bump `SCHEMA_VERSION` in `algua/registry/db.py` — nothing in this plan adds/removes/renames a column, table, or index; only Python-level code (functions, imports, call signatures) moves or is deleted.
- **`fdr_stream_state` (the surviving read path) and the `#339` funnel-drift CAS (`_verify_funnel_snapshot`, which runs unconditionally, before any binding check) are untouched in behavior.** Only the `if fdr_binding:` branch inside `record_gate_with_fdr_and_maybe_promote` — and the LORD++ math it alone calls — is deleted.
- **`FunnelSnapshot.dsr_binding` is a DIFFERENT concept from `fdr_binding`/`p_value` and is NOT part of this deletion.** `dsr_binding` gates the DSR-statistics variance CAS inside `_verify_funnel_snapshot` (`algua/registry/store.py:1442`) and belongs to the advisory DSR stack, which the spec explicitly keeps ("Advisory statistical stack ... kept"). Do not touch anything gated on `dsr_binding`. Test names containing the word "binding" are NOT a reliable signal for which concept they mean — the per-test disposition table below was verified by reading each test's actual assertions, not by name pattern-matching; follow the table, not the names.
- `git add` is always scoped to the named files — never `git add -A`.
- Commits end with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Known pre-existing worktree hazard: some test writes a demo strategy file into the real `algua/strategies/momentum/` directory as a side effect. If `git status` shows an untracked file there after running tests, delete it before staging — don't commit it.
- **Process rule from prior stages' execution — read this before dispatching or resuming any subagent**: if a background command (e.g. the full test suite, ~6-7 minutes on this repo) is started, the implementer/reviewer MUST actively poll/re-check its own output with real tool calls in a loop. There is no notification that wakes a dispatched subagent when a background command finishes — ending a turn to "wait" for one stalls the task indefinitely until manually resumed. This has happened repeatedly across every prior stage's execution (Stage 0-2 and Stage 3 both), including to implementers whose dispatch prompt already contained this exact warning — budget for needing a manual resume roughly every other dispatch regardless.
- No import-linter contract needs to change: no contract in `pyproject.toml` names `algua.registry.store`, `algua.registry.db`, or `algua.research.fdr_lord`/`algua.research.gates` specifically — all matching contracts use the broader `algua.registry`/`algua.research` prefixes, and this plan does not cross any contract-forbidden boundary (the FDR-constant relocation moves `fdr_cohort_position`/`FDR_COHORT_SIZE` from `research → registry` direction *of ownership*, and the one remaining `store.py` import from `research.gates` — `MIN_FUNNEL_FLOOR_STRATEGIES` — is the same already-legal `registry → research` direction the codebase uses throughout).

---

### Task 1: Relocate `fdr_cohort_position` + `FDR_COHORT_SIZE` into `registry/db.py`

**Files:**
- Modify: `algua/registry/db.py`
- Modify: `algua/registry/store.py`
- Modify: `tests/test_registry_db.py`
- Modify: `tests/test_registry_store.py`

**Interfaces:**
- Produces: `algua.registry.db.FDR_COHORT_SIZE` (int, value `8`), `algua.registry.db.fdr_cohort_position(k: int) -> tuple[int, int]` — both importable from `algua.registry.db` (joining the existing `algua.registry.db.MAX_N_COMBOS`).
- Consumes: nothing from earlier tasks (this is the first task).

**Why this task is separate and goes first:** `fdr_cohort_position`/`FDR_COHORT_SIZE` are the ONLY two symbols from `research/fdr_lord.py` with a genuinely live, non-test production consumer — `algua/registry/db.py`'s `_backfill_fdr_cohorts` and `_relabel_fdr_cohorts_for_current_size`, both called unconditionally on every `migrate()` run. Moving them first, in isolation, lets this relocation be verified green before Tasks 2-3 touch the much larger and more novel dead-branch deletion.

- [ ] **Step 1: Move `FDR_COHORT_SIZE` and `fdr_cohort_position` into `algua/registry/db.py`**

Read `algua/research/fdr_lord.py` in full first (179 lines) to copy the exact current text — the excerpt below is what to insert, verbatim including the docstring and the "WHY N=8" comment block, but re-verify against the live file since these are safety-critical protected-constant rationale comments that must move exactly, not be paraphrased.

In `algua/registry/db.py`, insert directly after the `MAX_N_COMBOS = 1_000_000_000` line (line 25) and before `_SCHEMA = """` (line 27):

```python

# Count-triggered cohort restarts (#324) + budget-derived recalibration (#529), relocated from
# research/fdr_lord.py (simplification stage 4a — this is registry-owned state: the FDR ledger's
# cohort partitioning is consumed by this module's own migrate()-invoked backfills below, not by
# any research-layer code). The LORD++ stream is partitioned into consecutive, non-overlapping
# COHORTS of exactly FDR_COHORT_SIZE binding tests, assigned by ARRIVAL ORDER. Protected constant —
# changing it re-scopes every historical cohort boundary; see _relabel_fdr_cohorts_for_current_size
# below for what a change requires.
FDR_COHORT_SIZE = 8


def fdr_cohort_position(k: int) -> tuple[int, int]:
    """Map a 1-based GLOBAL binding-test ordinal ``k`` to its ``(cohort_index, within_cohort_t)``.

    ``cohort_index = (k − 1) // FDR_COHORT_SIZE`` (0-based); ``within_cohort_t`` runs 1..
    FDR_COHORT_SIZE and is the position fed to the LORD++ level function for that cohort's
    independent stream. Fails closed (``ValueError``) on ``k < 1`` — a binding ordinal is
    always ≥ 1 by construction, so a non-positive value is a caller bug, not a silent-0 default.
    """
    if k < 1:
        raise ValueError(f"binding-test ordinal k must be >= 1, got {k}")
    return (k - 1) // FDR_COHORT_SIZE, (k - 1) % FDR_COHORT_SIZE + 1

```

No new import is needed in `db.py` for this — `fdr_cohort_position`'s body uses only `//`/`%` (stdlib operators), and `db.py` already has no non-stdlib imports at module level (confirm this is still true before and after your edit — it should stay that way).

- [ ] **Step 2: Remove the two now-redundant lazy imports inside `db.py` itself**

In `_backfill_fdr_cohorts` (around line 986) and `_relabel_fdr_cohorts_for_current_size` (around line 1025), delete the line `from algua.research.gates import fdr_cohort_position` in each — `fdr_cohort_position` is now a sibling function in the same module, called directly with no import needed. Leave every other line in both functions (including their docstrings and the surrounding `migrate()` call sequence) completely untouched — this task does not touch `migrate()`'s call ordering in any way.

- [ ] **Step 3: Update `algua/registry/store.py`'s imports**

Find this line near the top of the file (around line 20):
```python
from algua.registry.db import MAX_N_COMBOS
```
Change to:
```python
from algua.registry.db import FDR_COHORT_SIZE, MAX_N_COMBOS
```

Find this import block (around lines 93-96):
```python
from algua.research.gates import (
    FDR_COHORT_SIZE, FDR_NEAR_TERM_BINDING_BUDGET, FDR_THROTTLE_WINDOW_DAYS,
    MIN_FUNNEL_FLOOR_STRATEGIES,
)
```
Remove `FDR_COHORT_SIZE` from it (it now comes from `algua.registry.db` per the edit above) — the block becomes:
```python
from algua.research.gates import (
    FDR_NEAR_TERM_BINDING_BUDGET, FDR_THROTTLE_WINDOW_DAYS, MIN_FUNNEL_FLOOR_STRATEGIES,
)
```
(`FDR_NEAR_TERM_BINDING_BUDGET`/`FDR_THROTTLE_WINDOW_DAYS` are still needed here — they're only removed in Task 2, which deletes the code that uses them. Do not remove them in this task.)

Do not touch anything else in `store.py` in this task — `fdr_stream_state` (which uses `FDR_COHORT_SIZE`) needs no other change since the import now resolves to the same value from a different, correct home.

- [ ] **Step 4: Update `tests/test_registry_db.py`'s two lazy imports**

This file has two function-local imports of `FDR_COHORT_SIZE` from `algua.research.gates`:
- Around line 398, inside a test function testing `_backfill_fdr_cohorts` (search for `from algua.research.gates import FDR_COHORT_SIZE`).
- Around line 463, inside a test function testing `_relabel_fdr_cohorts_for_current_size`.

Change both to `from algua.registry.db import FDR_COHORT_SIZE`. Nothing else in either test changes — they exercise the real, unchanged `_backfill_fdr_cohorts`/`_relabel_fdr_cohorts_for_current_size` behavior end-to-end and must keep passing identically.

- [ ] **Step 5: Update `tests/test_registry_store.py`'s two lazy imports**

- The helper `_seed_cohort0` (around line 1686-1693) has `from algua.research.gates import FDR_COHORT_SIZE` at its first line (around line 1688) — change to `from algua.registry.db import FDR_COHORT_SIZE`.
- `test_fdr_stream_cohort_seals_and_restarts_at_boundary` (around line 1696-1705) has the same import at its first line (around line 1698) — change to `from algua.registry.db import FDR_COHORT_SIZE`.

Do not touch any other test function in this file in this task — Task 2 handles the rest of this file's FDR-related tests.

- [ ] **Step 6: Full quality gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`

Expected: all four pass, with the SAME pass count as the pre-task baseline (no test was added, removed, or behaviorally changed in this task — only import paths moved). Check `git status` for the known momentum-strategy test-fixture hazard; delete if present, don't stage it.

- [ ] **Step 7: Commit**

```bash
git add algua/registry/db.py algua/registry/store.py tests/test_registry_db.py tests/test_registry_store.py
```

```bash
git commit -m "$(cat <<'EOF'
refactor: relocate fdr_cohort_position + FDR_COHORT_SIZE into registry/db.py (stage 4a.1)

These two symbols are the only research/fdr_lord.py exports with a genuinely live production
consumer: db.py's migrate()-invoked backfills (_backfill_fdr_cohorts,
_relabel_fdr_cohorts_for_current_size). Moving them into registry/db.py — where they're actually
consumed — lets the rest of fdr_lord.py's provably-dead LORD++ binding machinery be deleted
cleanly in the next task without touching any live migration path. Zero behavior change: pure
relocation, no schema change, no SCHEMA_VERSION bump.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Delete the dead FDR-binding branch in `registry/store.py` + update `promotion.py`

**Files:**
- Modify: `algua/registry/store.py`
- Modify: `algua/registry/promotion.py`
- Modify: `tests/test_registry_store.py`

**Interfaces:**
- Consumes: `algua.registry.db.FDR_COHORT_SIZE` (Task 1) — already wired, no new consumption here.
- Produces: `SqliteStrategyRepository.record_gate_with_fdr_and_maybe_promote`'s new signature — `(self, rec: StrategyRecord, *, gate_row: dict[str, Any], funnel: FunnelSnapshot, actor: Actor, reason: str | None = None, pending_novel_family: PendingNovelFamily | None = None) -> FdrGateOutcome` (drops `p_value`, `level_fn`, `fdr_alpha`). This signature is Stage 4b's concern to further reshape (e.g. whether it stays a `GateLedger` Protocol method) — this task only removes the three dead parameters and the branch they drove; it does not touch the Protocol declaration in `repository.py` at all.

**Why `record_gate_with_fdr_and_maybe_promote` shrinks its signature (not just its body):** `p_value`/`level_fn`/`fdr_alpha` exist ONLY to drive the `if fdr_binding:` branch being deleted — `run_gate` (`promotion.py`, the method's only production caller) hardcodes `p_value = None` unconditionally, so `fdr_binding = p_value is not None and math.isfinite(p_value)` is always `False` today. Once the branch these parameters feed is gone, keeping the parameters around unused would be exactly the kind of dead-but-present surface this program exists to remove. This is a real, deliberate API-signature change — not an incidental side effect — call it out as such if a reviewer asks.

- [ ] **Step 1: Read the current method in full**

Read `algua/registry/store.py`'s `record_gate_with_fdr_and_maybe_promote` (currently lines 1459-1747) and `_windowed_binding_test_count` (currently lines 1383-1401) in full before editing — re-verify the line numbers and exact text below against what you see, since these are safety-critical, carefully-commented transaction internals.

- [ ] **Step 2: Delete `_windowed_binding_test_count`**

This method (currently lines 1383-1401, docstring included) has exactly one caller: the `if fdr_binding:` branch being deleted in the next step. Delete the whole method.

- [ ] **Step 3: Rewrite `record_gate_with_fdr_and_maybe_promote`**

Replace the entire method (currently lines 1459-1747) with:

```python
    def record_gate_with_fdr_and_maybe_promote(
        self,
        rec: StrategyRecord,
        *,
        gate_row: dict[str, Any],
        funnel: FunnelSnapshot,
        actor: Actor,
        reason: str | None = None,
        pending_novel_family: PendingNovelFamily | None = None,
    ) -> FdrGateOutcome:
        # #524: coerce the actor BEFORE the pending-mint boundary guard (callers may pass a raw
        # string; an identity test against an un-coerced string would mis-evaluate).
        actor = Actor(actor)
        # #524: the mint is an AGENT-only capability and THIS method is the safety boundary, not
        # just the (trusted) caller path. Fail-closed unless the method actor is the agent AND the
        # pending spec's own actor/verdict are internally consistent (a caller bug must not write
        # mismatched audit/classification metadata). 'novel' is a plain-string literal so the store
        # never imports algua.research.SimVerdict (registry→research boundary stays clean).
        if pending_novel_family is not None and not (
            actor is Actor.AGENT
            and pending_novel_family.actor == Actor.AGENT.value
            and pending_novel_family.verdict == "novel"
        ):
            raise ValueError(
                "record_gate_with_fdr_and_maybe_promote: a pending_novel_family mint requires "
                "actor=agent and a consistent agent/novel pending spec")
        # TOP-LEVEL ONLY — mirrors reserve_holdout's contract. A manual BEGIN IMMEDIATE inside an
        # already-open transaction would raise "cannot start a transaction within a transaction";
        # catching that instead of pre-checking would leave the caller's surrounding tx open in a
        # rolled-back state. Fail loudly so the contract is enforced, not assumed.
        if self._conn.in_transaction:
            raise RuntimeError(
                "record_gate_with_fdr_and_maybe_promote must be called at top level,"
                " not inside an open transaction")

        # #339 — bind the funnel snapshot to the strategy being promoted. The in-lock CAS re-reads
        # the funnel-wide state keyed by funnel.strategy_name; if that name did not match rec (a
        # caller passing the wrong snapshot), the CAS would validate a DIFFERENT strategy's inputs
        # and the promoted strategy's own drift could escape. Fail closed on mismatch — the store
        # method is the safety boundary, not just the run_gate caller.
        if funnel.strategy_name != rec.name:
            raise FunnelDriftError(
                f"funnel snapshot is for {funnel.strategy_name!r} but the promotion is for "
                f"{rec.name!r}; the funnel CAS must verify the strategy being promoted")

        provisional_passed = bool(gate_row.get("passed"))
        final_passed: bool

        # BEGIN IMMEDIATE takes the write lock up front so the gate row INSERT and the optional
        # stage CAS are one atomic critical section.
        # BaseException (not Exception) so KeyboardInterrupt/SystemExit still rolls back.
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            # #524 step 1 — pending-NOVEL CAS UNDER the write lock (R3-HIGH + R5-HIGH-1). Before any
            # other work, prove the agent-NOVEL verdict is still valid against the graph
            # at commit: (a) the strategy is STILL unassigned (no concurrent assignment landed), and
            # (b) the full-classifier-read-set fingerprint EQUALS the one the NOVEL verdict was
            # computed on (captured before==after classification, §7.1). A mismatch means a family
            # was minted / a member (re)assigned or removed / a parentage edge added / a member's
            # returns refreshed since classification, so the verdict is stale and might now be
            # MERGE/PARENTAGE — fail closed (the CLI re-runs preflight against the current graph).
            if pending_novel_family is not None:
                if self.strategy_family(rec.name) is not None:
                    raise FamilyGraphDriftError(
                        f"strategy {rec.name!r} was assigned to a family since NOVEL "
                        "classification; re-run promote", axis="still_assigned")
                if self.family_graph_fingerprint() != pending_novel_family.graph_fingerprint:
                    raise FamilyGraphDriftError(
                        f"family graph changed since {rec.name!r} was classified NOVEL; re-run "
                        "promote", axis="graph_fingerprint")
            # #339 — serializability CAS: re-read the funnel-wide MUTABLE state the (lock-free)
            # decision was computed against and abort if any of it drifted, so a committed decision
            # is provably a pure function of ONE funnel snapshot. Runs BEFORE the INSERT/stage-CAS,
            # inside this one write-locked critical section.
            self._verify_funnel_snapshot(funnel)
            # Factory soft gate (#529): the LORD++ FDR-binding branch itself was deleted
            # (simplification stage 4a) — it was provably dead in production (run_gate never
            # supplied a real p_value). final_passed is now always the provisional integrity-floor
            # verdict. The FDR ledger machinery this branch used to write — fdr_stream_state's read
            # path, the gate_evaluations fdr_* columns, db.py's cohort backfills — is preserved for
            # future re-tightening; only the write path (this branch) and its LORD++ math
            # (research/fdr_lord.py) are gone.
            final_passed = provisional_passed

            # Patch decision_json so the stored audit record reflects final_passed — only known
            # after this transaction's checks run.
            raw_decision = json.loads(gate_row.get("decision_json") or "{}")
            raw_decision["passed"] = final_passed
            decision_json = json.dumps(raw_decision)

            cur = self._conn.execute(
                "INSERT INTO gate_evaluations"
                "(strategy_id, passed, n_funnel, own_lifetime_combos, windowed_total_combos,"
                " funnel_window_days, breadth_provenance, pit_ok, pit_override, holdout_n_bars,"
                " min_holdout_observations, code_hash, config_hash, dependency_hash, data_source,"
                " snapshot_id, period_start, period_end, holdout_frac, actor, decision_json,"
                " consumed, created_at,"
                " fdr_binding, fdr_p_value, fdr_alpha_level, fdr_rejected, fdr_test_index,"
                " fdr_cohort,"
                " family_id, family_lifetime_effective,"
                " fundamentals_snapshot, news_snapshot, attempt_token, universe_name)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                # Agent passing rows are born-consumed: the stage has already advanced inside
                # this transaction, so the token is spent. Leaving consumed=0 would let a
                # future `registry transition --to candidate --actor agent` reuse the old row
                # after a back-step, bypassing the gate re-run requirement.
                (rec.id, int(final_passed),
                 gate_row["n_funnel"], gate_row["own_lifetime_combos"],
                 gate_row["windowed_total_combos"], gate_row["funnel_window_days"],
                 gate_row["breadth_provenance"], int(gate_row["pit_ok"]),
                 int(gate_row.get("pit_override", False)),
                 gate_row["holdout_n_bars"], gate_row["min_holdout_observations"],
                 gate_row["code_hash"], gate_row["config_hash"], gate_row.get("dependency_hash"),
                 gate_row["data_source"], gate_row.get("snapshot_id"),
                 gate_row["period_start"], gate_row["period_end"], gate_row["holdout_frac"],
                 actor.value, decision_json,
                 int(actor is Actor.AGENT and final_passed), _now(),
                 # fdr_binding, fdr_p_value, fdr_alpha_level, fdr_rejected, fdr_test_index,
                 # fdr_cohort: the binding-write path is retired (stage 4a) — every binding row
                 # is NULL going forward. Historical binding rows (written before this change)
                 # keep their real values; fdr_stream_state's WHERE fdr_binding=1 filter already
                 # excludes NULL rows, so it continues to see only genuine historical binding
                 # evidence.
                 None, None, None, None, None, None,
                 gate_row.get("family_id"), gate_row.get("family_lifetime_effective"),
                 gate_row.get("fundamentals_snapshot"), gate_row.get("news_snapshot"),
                 gate_row.get("attempt_token"), gate_row.get("universe_name")),
            )
            gate_id = cur.lastrowid
            assert gate_id is not None

            updated_rec: StrategyRecord | None = None
            if final_passed:
                # #246: re-assert the source-stage invariant INSIDE this locked critical section.
                # The CAS below only pins WHERE stage=rec.stage (the caller's post-walk_forward
                # re-read). If a concurrent transition drifted the strategy off BACKTESTED before
                # that re-read — e.g. to a terminal RETIRED — the CAS would happily apply a
                # forbidden RETIRED->CANDIDATE edge (ALLOWED_TRANSITIONS forbids it). Mirror
                # promotion_preflight here, atomic with the CAS, so the gate can never resurrect a
                # drifted/terminal stage. Require exactly BACKTESTED (not validate_transition, which
                # permits the legal PAPER->CANDIDATE back-step) — promotion_preflight's reasoning.
                if rec.stage is not Stage.BACKTESTED:
                    raise TransitionError(
                        f"research promote requires stage backtested at promote time, got "
                        f"{rec.stage.value} (source drifted off backtested under concurrency since "
                        f"preflight); refusing to promote")
                updated_rec = self._apply_transition_locked(
                    rec, Stage.CANDIDATE, actor, reason,
                    code_hash=gate_row["code_hash"], config_hash=gate_row["config_hash"],
                    dependency_hash=gate_row.get("dependency_hash"),
                    consume_gate_id=None, consume_forward_gate_id=None, now=_now())
                # #524 step 5 — mint the seeded agent-NOVEL family, IN THIS SAME TRANSACTION, only
                # on a pass. All-or-nothing with the gate row + stage CAS: a crash before commit
                # leaves no orphan family/membership.
                if pending_novel_family is not None:
                    self._mint_agent_novel_family(pending_novel_family, rec.name, gate_id)

            self._conn.commit()
        except BaseException:
            self._conn.rollback()
            raise

        return FdrGateOutcome(
            gate_id=gate_id,
            fdr_binding=False,
            fdr_test_index=None,
            fdr_p_value=None,
            fdr_alpha_level=None,
            fdr_rejected=None,
            final_passed=final_passed,
            updated_rec=updated_rec,
            fdr_cohort=None,
            fdr_cohorts_completed=None,
            fdr_binding_tests=None,
            fdr_discoveries=None,
            fdr_expected_false_discoveries=None,
            fdr_throttle_window_binding=None,
            fdr_throttle_tripped=None,
            fdr_active_cohort_position=None,
            fdr_active_cohort_applied_alpha=None,
            fdr_expected_false_discoveries_incl_active=None,
        )
```

Note what stayed verbatim (actor coercion, both guards, the funnel-drift check, the pending-NOVEL CAS, `_verify_funnel_snapshot`, the INSERT's SQL text/columns, the stage-CAS + family-mint tail, the commit/rollback wrapper) versus what changed (signature; `fdr_binding`'s computation is gone, `final_passed = provisional_passed` unconditionally; the `if fdr_binding:`-only variable declarations, the `fdr_checks` append block, and the six now-always-`None` INSERT values are collapsed to literals; the return statement's `fdr_*` fields are literals instead of conditionals).

- [ ] **Step 4: Update `store.py`'s now-stale `research.gates` import**

The import block Task 1 left as:
```python
from algua.research.gates import (
    FDR_NEAR_TERM_BINDING_BUDGET, FDR_THROTTLE_WINDOW_DAYS, MIN_FUNNEL_FLOOR_STRATEGIES,
)
```
`FDR_NEAR_TERM_BINDING_BUDGET` and `FDR_THROTTLE_WINDOW_DAYS` were used only inside the now-deleted branch and `_windowed_binding_test_count`. Change to a single-symbol import:
```python
from algua.research.gates import MIN_FUNNEL_FLOOR_STRATEGIES
```

- [ ] **Step 5: Update `algua/registry/promotion.py`'s `run_gate`**

Find this block (around lines 597-605):
```python
    # Factory soft gate (2026-08-10 spec): the statistical stack is ADVISORY, so the LORD++
    # BINDING stream is never consumed on this path — p_value is ALWAYS None, the store writes an
    # fdr_binding=NULL row (invisible to the stream reader and the windowed throttle count), and
    # final_passed collapses to the provisional integrity-floor verdict. dsr_confidence is still
    # computed and recorded in the decision above (advisory telemetry); the whole LORD++/throttle
    # ledger machinery is PRESERVED in the store for future re-tightening.
    p_value = None
    decision.fdr_binding = False
    decision.fdr_skip_reason = "stats_advisory"
```
Replace with:
```python
    # Factory soft gate (2026-08-10 spec): the statistical stack is ADVISORY. The LORD++
    # FDR-binding branch itself was deleted (simplification stage 4a — it was provably dead here,
    # this path never supplied a real p_value); record_gate_with_fdr_and_maybe_promote's
    # final_passed is now always the provisional integrity-floor verdict. dsr_confidence is still
    # computed and recorded in the decision above (advisory telemetry).
    decision.fdr_binding = False
    decision.fdr_skip_reason = "stats_advisory"
```

Find this block (around lines 684-688):
```python
    # Atomic FDR-test-and-maybe-promote. For non-binding rows (p_value=None), the method
    # behaves identically to the old record_gate_evaluation + conditional transition_strategy
    # pair, but always uses BEGIN IMMEDIATE for consistency (negligible overhead for ≤ a few
    # thousand gate_evaluations rows).
    level_fn = functools.partial(lord_plus_plus_level, alpha=FDR_ALPHA, w0=FDR_W0)
```
Replace with:
```python
    # Atomic gate-record-and-maybe-promote — always uses BEGIN IMMEDIATE for consistency
    # (negligible overhead for ≤ a few thousand gate_evaluations rows).
```

Find the call site (around lines 711-716):
```python
    fdr_outcome = repo.record_gate_with_fdr_and_maybe_promote(
        rec, gate_row=gate_row, p_value=p_value, funnel=funnel, level_fn=level_fn,
        fdr_alpha=FDR_ALPHA, actor=actor,
        reason=(_gate_reason(decision) + reason_suffix) if decision.passed else None,
        pending_novel_family=breadth.pending_novel_family,  # #524: minted only on pass, in-tx
    )
```
Replace with:
```python
    fdr_outcome = repo.record_gate_with_fdr_and_maybe_promote(
        rec, gate_row=gate_row, funnel=funnel, actor=actor,
        reason=(_gate_reason(decision) + reason_suffix) if decision.passed else None,
        pending_novel_family=breadth.pending_novel_family,  # #524: minted only on pass, in-tx
    )
```

Find the trailing comment (around lines 718-720):
```python
    # With p_value=None the store skipped the LORD++/throttle logic entirely, so final_passed ==
    # provisional_passed (the integrity floor). Kept as an assignment (not an assert) so the
    # decision always mirrors what the store committed.
```
Replace with:
```python
    # With the LORD++ binding branch retired (stage 4a), final_passed == provisional_passed (the
    # integrity floor) unconditionally. Kept as an assignment (not an assert) so the decision
    # always mirrors what the store committed.
```

- [ ] **Step 6: Update `promotion.py`'s imports**

Find (around lines 35-51):
```python
from algua.research.gates import (
    DSR_ALPHA,
    DSR_BOOTSTRAP_LOWER_QUANTILE,
    DSR_BOOTSTRAP_RESAMPLES,
    FDR_ALPHA,
    FDR_W0,
    FUNNEL_WINDOW_DAYS,
    MIN_CORR_OVERLAP_BARS,
    MIN_N_EFF_SIBLINGS,
    RHO_BAR_SHRINKAGE_K,
    GateCriteria,
    GateDecision,
    dsr_sr_star_annualized,
    effective_funnel_breadth,
    evaluate_gate,
    lord_plus_plus_level,
)
```
Remove `FDR_ALPHA`, `FDR_W0`, `lord_plus_plus_level` (all three now unused in this file — confirm with a grep for each name across the whole file before removing, in case this plan's snapshot missed a usage):
```python
from algua.research.gates import (
    DSR_ALPHA,
    DSR_BOOTSTRAP_LOWER_QUANTILE,
    DSR_BOOTSTRAP_RESAMPLES,
    FUNNEL_WINDOW_DAYS,
    MIN_CORR_OVERLAP_BARS,
    MIN_N_EFF_SIBLINGS,
    RHO_BAR_SHRINKAGE_K,
    GateCriteria,
    GateDecision,
    dsr_sr_star_annualized,
    effective_funnel_breadth,
    evaluate_gate,
)
```
Also remove `import functools` from the top of the file (around line 3) — `functools.partial` at the now-deleted line 688 was its only use in this file; confirm with `grep -n "functools\." algua/registry/promotion.py` that nothing else uses it before removing the import.

- [ ] **Step 7: Update `tests/test_registry_store.py` — the exhaustive per-test disposition**

Every test function in this file that calls `record_gate_with_fdr_and_maybe_promote(` was individually classified during this plan's research by reading its actual assertions (not by name pattern). Three dispositions:

**(A) DELETE — 17 functions genuinely testing the deleted branch.** Delete these whole functions (docstring, body, everything) and, for the throttle-budget group, the shared helper `_land_binding_rows` they alone use:

`test_fdr_gate_drift_rollback_does_not_advance_fdr_stream`, `test_fdr_gate_binding_accept_promotes_when_provisional_passes`, `test_fdr_gate_binding_reject_blocks_promotion`, `test_fdr_gate_provisional_fail_skips_fdr_promotion`, `test_fdr_gate_db_passed_column_reflects_final_not_provisional`, `test_fdr_gate_db_passed_column_true_on_accept`, `test_fdr_gate_binding_row_appends_fdr_checks_to_decision_json`, `test_fdr_gate_binding_reject_carries_failing_fdr_check`, `test_fdr_gate_stream_grows_for_binding_rows`, `test_fdr_gate_discovery_increments_test_index_and_replenishes`, `test_fdr_gate_surfaces_cohort_and_exposure`, `test_fdr_throttle_blocks_promotion_beyond_budget`, `test_fdr_throttle_out_of_window_row_not_counted`, `test_fdr_throttle_has_no_override`, `test_fdr_active_cohort_exposure_surfaced`, `test_fdr_gate_concurrent_distinct_t_values`, `test_fdr_gate_anti_scaling_alpha_never_collapses_across_cohort` (this one never calls the method at all — it seeds rows via `_insert_fdr_row` and imports `lord_plus_plus_level`/`FDR_ALPHA`/`FDR_W0`/`FDR_COHORT_SIZE`/`fdr_cohort_position` directly as an oracle; delete it because its subject, the LORD++ level computation, is itself deleted).

Also delete the now-fully-unused helper functions `_level_accept`, `_level_reject` (currently around lines 1226-1231) and `_land_binding_rows` (currently around line 1480) — after the deletions and fixes in this step, grep the file yourself to confirm none of these three names appear anywhere else before deleting them (`grep -n "_level_accept\|_level_reject\|_land_binding_rows" tests/test_registry_store.py`).

**(B) FIX — 17 functions whose real subject survives untouched; only their call needs updating.** For each of these, remove the now-nonexistent `p_value=..., level_fn=..., fdr_alpha=...,` keyword arguments from their call(s) to `record_gate_with_fdr_and_maybe_promote` (whether the value was `p_value=None` or a real float — the parameters don't exist on the new signature either way) — nothing else in these functions changes:

`test_fdr_gate_non_binding_passes_through_on_provisional_pass`, `test_fdr_gate_non_binding_fails_on_provisional_fail`, `test_fdr_gate_refuses_promote_from_drifted_source_stage`, `test_fdr_gate_promotes_from_backtested_source`, `test_cold_start_pass_mints_family_zero`, `test_cold_start_fail_mints_nothing`, `test_cold_start_concurrent_double_founding_prevented`, `test_funnel_cas_passes_when_snapshot_matches`, `test_funnel_cas_aborts_on_wrong_strategy_snapshot`, `test_funnel_cas_aborts_on_windowed_breadth_drift`, `test_funnel_cas_aborts_on_family_drift`, `test_fdr_gate_top_level_only_guard` (tests the top-level-only guard, unrelated to binding), `test_fdr_gate_rollback_on_stage_cas_failure` (tests the stage-CAS rollback, unrelated to binding), `test_funnel_cas_aborts_on_search_trials_insert` (tests the unconditional #339 CAS), `test_funnel_cas_concurrent_stale_promote_aborts_order_independent` (tests #339 CAS order-independence), `test_fdr_gate_agent_pass_is_born_consumed` (tests the `consumed` column formula, unrelated to binding).

`test_funnel_cas_aborts_on_variance_drift_when_binding` is ALSO in this fix-not-delete group DESPITE its name and its `p_value=0.01` call — read `algua/registry/store.py:_verify_funnel_snapshot` (around line 1442) yourself to confirm: the variance-check block it exercises is gated on `funnel.dsr_binding`, a `FunnelSnapshot` field the DSR statistical stack sets — a completely different concept from this method's own (now-deleted) `p_value`-driven `fdr_binding`. The test constructs its `FunnelSnapshot` with `dsr_binding=True` directly (via `_live_funnel(repo, "s", dsr_binding=True)`), so this test's real subject is fully independent of the deletion. Fix its call the same way as the others (drop the three kwargs); consider renaming the function to `test_funnel_cas_aborts_on_variance_drift_when_dsr_binding` while you're there to stop the name misleading the next reader, but this rename is optional polish, not required.

**(C) DELETE-WITH-SALVAGE — 1 function.** `test_fdr_gate_agent_fail_and_human_pass_not_consumed` mixes a first half that is genuinely about the deleted branch (an agent row that provisionally passes but fails the hard LORD++ gate — impossible to construct once `p_value`/`level_fn` are gone) with a second half testing a surviving, otherwise-uncovered property (a HUMAN actor's passing row must have `consumed=0`, since human rows are never consumable promotion tokens). Delete the whole function, and extend the fixed `test_fdr_gate_agent_pass_is_born_consumed` (from group B) to also cover the human case — after applying its group-B fix, its body should end up as:

```python
def test_fdr_gate_agent_pass_is_born_consumed(repo):
    """Agent passing rows must be born consumed=1 so a back-step cannot replay the token."""
    rec = _at_backtested(repo)
    repo.record_gate_with_fdr_and_maybe_promote(
        rec, funnel=_EMPTY_FUNNEL, gate_row=_make_gate_row(passed=True), actor=Actor.AGENT,
    )
    row = repo._conn.execute(
        "SELECT consumed FROM gate_evaluations ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert row["consumed"] == 1

    # Human passing row -> also consumed=0 (human rows are never consumable tokens).
    rec2 = _at_backtested(repo, "s2")
    repo.record_gate_with_fdr_and_maybe_promote(
        rec2, funnel=_EMPTY_FUNNEL._replace(strategy_name="s2"),
        gate_row=_make_gate_row(passed=True), actor=Actor.HUMAN,
    )
    row_human = repo._conn.execute(
        "SELECT consumed FROM gate_evaluations ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert row_human["consumed"] == 0
```

(This drops the deleted test's FIRST assertion block — the agent-provisionally-passes-but-LORD++-rejects case — since that scenario can no longer be constructed; it keeps the human-row check, adapted to the new no-`p_value` signature.)

- [ ] **Step 8: Full quality gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`

Expected: all four pass. The total test count drops by exactly 17 (group A deletions) + 1 (the group-C deletion, `test_fdr_gate_agent_fail_and_human_pass_not_consumed`) = 18 relative to Task 1's ending count — verify this exact delta, not just "tests pass," since an unexpected delta means either a stray test was missed or one was double-counted. Check `git status` for the momentum-strategy hazard.

- [ ] **Step 9: Commit**

```bash
git add algua/registry/store.py algua/registry/promotion.py tests/test_registry_store.py
```

```bash
git commit -m "$(cat <<'EOF'
refactor: delete the dead FDR-binding branch in record_gate_with_fdr_and_maybe_promote (stage 4a.2)

The LORD++ FDR-binding branch was provably dead in production: run_gate (promotion.py) has always
hardcoded p_value=None (factory soft gate, #529), so the `if fdr_binding:` branch could never
execute. Deletes the branch, its dedicated helper (_windowed_binding_test_count), and the three
now-meaningless parameters (p_value, level_fn, fdr_alpha) from the method's signature. The
surviving non-binding path, the #339 funnel-drift CAS, the stage-CAS + family-mint transaction
tail, and fdr_stream_state's read path are byte-for-byte unchanged. gate_evaluations' fdr_*
columns and every historical row are untouched.

18 tests deleted (17 exercised only the deleted branch; 1 deleted-with-salvage — see below);
17 tests fixed in place (dropped the three removed kwargs — their real subjects, e.g. the
top-level-only guard, the #339 CAS, the dsr_binding-gated variance check, all survive untouched);
the deleted-with-salvage test's still-relevant human-actor assertion was folded into a surviving
sibling test rather than lost.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Delete `research/fdr_lord.py` + update `research/gates.py`'s re-export facade

**Files:**
- Delete: `algua/research/fdr_lord.py`
- Modify: `algua/research/gates.py`
- Modify: `tests/test_research_gates.py`

**Interfaces:**
- Consumes: nothing new (Task 1 already relocated the two symbols this file's deletion would otherwise have broken; Task 2 already removed every remaining production consumer of this file's other symbols).
- Produces: nothing new — this task only removes now-fully-dead code and its facade re-export.

- [ ] **Step 1: Confirm zero remaining consumers before deleting**

Run `grep -rn "fdr_lord\|lord_plus_plus_level\|_compute_lord_gamma\|_LORD_GAMMA\b" algua/ --include='*.py' | grep -v "algua/research/fdr_lord.py\|algua/research/gates.py"` and confirm it returns nothing (Task 2 should have already removed every production reference; if this grep finds something, stop and report it rather than deleting under it). Separately run the same grep against `tests/` (excluding `tests/test_research_gates.py`, which this task edits directly) and confirm it also returns nothing.

- [ ] **Step 2: Delete `algua/research/fdr_lord.py`**

```bash
git rm algua/research/fdr_lord.py
```

- [ ] **Step 3: Update `algua/research/gates.py`'s docstring**

Remove the `algua.research.fdr_lord` bullet from the module docstring's list of sibling modules (currently lines 21-24):
```python
- ``algua.research.fdr_lord``  — LORD++ online-FDR γ-sequence, cohort restarts, and α_t level
  (recalibrated #529; the promote path now records ``fdr_binding`` NULL rows — the LORD++ ledger
  machinery is preserved for future re-tightening but the stream is not consumed while the
  statistical stack is advisory).
```
Delete this whole bullet — the module no longer exists. Leave the other four sibling-module bullets (`regime`, `dsr`, `haircut`, `_constants`) untouched.

- [ ] **Step 4: Remove the `fdr_lord` import block from `gates.py`**

Delete:
```python
from algua.research.fdr_lord import (
    _LORD_GAMMA,
    FDR_ALPHA,
    FDR_COHORT_SIZE,
    FDR_NEAR_TERM_BINDING_BUDGET,
    FDR_THROTTLE_WINDOW_DAYS,
    FDR_W0,
    _compute_lord_gamma,
    fdr_cohort_position,
    lord_plus_plus_level,
)
```
entirely (currently lines 61-71).

- [ ] **Step 5: Remove the corresponding `__all__` entries**

From `gates.py`'s `__all__` list, remove: `"FDR_ALPHA"`, `"FDR_COHORT_SIZE"`, `"FDR_NEAR_TERM_BINDING_BUDGET"`, `"FDR_THROTTLE_WINDOW_DAYS"`, `"FDR_W0"`, `"_LORD_GAMMA"`, `"_compute_lord_gamma"`, `"fdr_cohort_position"`, `"lord_plus_plus_level"` (9 entries total). Leave every other entry untouched.

- [ ] **Step 6: Delete the FDR-math tests in `tests/test_research_gates.py`**

Delete these 14 test functions entirely (they test `fdr_lord.py`'s deleted symbols directly): `test_fdr_constants`, `test_gamma_weights_normalized_over_cohort_size`, `test_gamma_weights_are_positive`, `test_gamma_weights_are_decreasing_across_cohort`, `test_lord_no_discoveries_alpha_decreasing`, `test_lord_cohort_spends_full_budget`, `test_lord_all_null_first_discovery_probability`, `test_lord_retry_surface_within_near_term_budget`, `test_lord_first_discovery_bumps_alpha`, `test_lord_multiple_discoveries_replenish`, `test_lord_manual_recursion_check`, `test_lord_fail_closed_guards`, `test_lord_injected_params`, `test_lord_t1_no_discoveries_equals_gamma1_times_w0`.

Do NOT touch `test_gate_decision_has_fdr_fields_with_defaults`, `test_gate_decision_to_dict_includes_fdr_keys`, `test_gate_decision_to_dict_fdr_fields_populated`, `test_gate_decision_to_dict_non_binding_fdr` — these test the plain `GateDecision` dataclass's `fdr_*` audit fields, which have nothing to do with `fdr_lord.py` and are entirely unaffected by this deletion.

- [ ] **Step 7: Update `test_research_gates.py`'s import block**

Find (currently lines 8-26):
```python
from algua.research.gates import (
    _LORD_GAMMA,
    DSR_ALPHA,
    EULER_MASCHERONI,
    FDR_ALPHA,
    FDR_COHORT_SIZE,
    FDR_NEAR_TERM_BINDING_BUDGET,
    FDR_THROTTLE_WINDOW_DAYS,
    FDR_W0,
    FUNNEL_WINDOW_DAYS,
    MIN_HOLDOUT_OBSERVATIONS,
    GateCriteria,
    GateDecision,
    dsr_confidence,
    effective_funnel_breadth,
    evaluate_gate,
    lord_plus_plus_level,
    sharpe_haircut,
)
```
Remove `_LORD_GAMMA`, `FDR_ALPHA`, `FDR_COHORT_SIZE`, `FDR_NEAR_TERM_BINDING_BUDGET`, `FDR_THROTTLE_WINDOW_DAYS`, `FDR_W0`, `lord_plus_plus_level` (all seven confirmed used only inside the 14 deleted tests — re-confirm with a grep across the whole file before removing, per this plan's own re-verify discipline):
```python
from algua.research.gates import (
    DSR_ALPHA,
    EULER_MASCHERONI,
    FUNNEL_WINDOW_DAYS,
    MIN_HOLDOUT_OBSERVATIONS,
    GateCriteria,
    GateDecision,
    dsr_confidence,
    effective_funnel_breadth,
    evaluate_gate,
    sharpe_haircut,
)
```

- [ ] **Step 8: Full quality gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`

Expected: all four pass, test count down by exactly 14 from Task 2's ending count. Check `git status` for the momentum-strategy hazard.

- [ ] **Step 9: Commit**

```bash
git add algua/research/fdr_lord.py algua/research/gates.py tests/test_research_gates.py
```

```bash
git commit -m "$(cat <<'EOF'
refactor: delete research/fdr_lord.py — safe subset of the frozen FDR machinery (stage 4a.3)

Deletes the LORD++ online-FDR math module (γ-sequence, cohort-position mapper duplicate now
superseded by registry/db.py's copy from stage 4a.1, and the alpha-level function) now that
nothing production or test-side references it: stage 4a.1 relocated the two symbols db.py's live
migrations still need (fdr_cohort_position, FDR_COHORT_SIZE), and stage 4a.2 deleted the only
production call site that fed this module's math a real p_value. research/gates.py's re-export
facade and its module docstring are updated to match — the fdr_lord.py bullet is removed, its
import block and __all__ entries are dropped. gate_evaluations' fdr_* columns, historical rows,
and GateDecision's plain fdr_* audit fields (unrelated to this module) are all untouched.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Close-out verification

**Files:** none expected (verification only; fix anything found)

- [ ] **Step 1: Confirm zero stray references anywhere**

```bash
grep -rn "fdr_lord\|lord_plus_plus_level\|_compute_lord_gamma\|_LORD_GAMMA\b\|_windowed_binding_test_count\|_land_binding_rows\|\blevel_fn\b" algua/ tests/ --include='*.py'
```
Expected: no hits anywhere. Any hit is a stale reference this plan's tasks missed.

- [ ] **Step 2: Confirm the relocated symbols resolve from their new home**

```bash
uv run python -c "from algua.registry.db import FDR_COHORT_SIZE, fdr_cohort_position; print(FDR_COHORT_SIZE, fdr_cohort_position(9))"
```
Expected: prints `8 (1, 1)` (9th binding-test ordinal is cohort 1, within-cohort position 1, under `FDR_COHORT_SIZE=8`) and exits 0.

- [ ] **Step 3: Confirm `record_gate_with_fdr_and_maybe_promote`'s new signature**

```bash
uv run python -c "
import inspect
from algua.registry.store import SqliteStrategyRepository
sig = inspect.signature(SqliteStrategyRepository.record_gate_with_fdr_and_maybe_promote)
params = list(sig.parameters)
assert params == ['self', 'rec', 'gate_row', 'funnel', 'actor', 'reason', 'pending_novel_family'], params
print('OK', params)
"
```
Expected: prints `OK [...]` and exits 0. If this fails, the signature does not match what Task 2 specified — investigate before proceeding.

- [ ] **Step 4: Full quality gate one more time**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`

Expected: all four pass (this re-confirms all three prior tasks' gates on a clean tree, catching anything a `git status` oversight might have missed). The total test-count delta versus this plan's pre-Task-1 baseline should be exactly −32 (18 deletions in Task 2's step 8's accounting + 14 in Task 3 = 32; zero new tests were added anywhere in this plan).

- [ ] **Step 5: CLI smoke test**

```bash
uv run algua doctor
uv run algua registry list
```
Expected: both exit 0 (or a clean, unrelated non-zero on an empty/no-data worktree — anything mentioning `fdr_lord`, a missing import, or an attribute error on `SqliteStrategyRepository`/`GateDecision` is a real regression, not expected).

- [ ] **Step 6: Commit any fixes**

If steps 1-5 forced fixes, commit them (scoped `git add`, correct trailer). If nothing needed fixing, this task makes no commit — that's expected and consistent with how prior stages' equivalent close-out tasks landed.
