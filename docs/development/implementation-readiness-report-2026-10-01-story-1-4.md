---
stepsCompleted:
  - step-01-document-discovery
  - step-02-prd-analysis
  - step-03-epic-coverage-validation
  - step-04-ux-alignment
  - step-05-epic-quality-review
  - step-06-final-assessment
filesIncluded:
  - docs/development/stories/1-4-controlled-exit-of-the-legacy-paper-cohort.md
  - docs/development/specs/spec-story-1-4-legacy-cohort-exit/ (SPEC, legacy-exit-contract, decision log)
  - scratchpad/1-4/legacy-exit-runbook.md (draft runbook), scratchpad/1-4/cohort-2026-10-01.json
  - Stories 1.2 and 1.3a–1.3d, epics.md, AGENTS.md, CLAUDE.md, docs/architecture.md
assessmentScope: Story 1.4
baseline: 14621bf (main after PR #683, Story 1.3d)
---

# Implementation Readiness Assessment Report

**Date:** 2026-10-01 · **Project:** Algua — Story 1.4 · **Assessor:** independent reviewer (Claude),
BMAD method, non-interactive defaults. Code claims were checked at `14621bf`. Scratch probes ran
against registries built by the repo's `migrate()`; the production registry was read only through
`file:…/data/algua.db?mode=ro`. Targeted tests in `tests/test_deployments.py` (operator lock,
retirement matrix, legacy, candidate episode) pass at baseline.

**Probes** (scratch `1-4/readiness/probe.py`, a legacy member forced with `force_legacy_strategy`):

| Probe | What it exercised |
|---|---|
| P1 | Dust fills (`0.1 + 0.2 − 0.3`) under `believed_positions`, `_assert_flat_for_bench`, `flatten_strategy` and `apply_transition` |
| P2 | `apply_transition` inside a caller's `BEGIN IMMEDIATE` (a: revoke edge; b: non-revoke edge) |
| P3 | Two `_apply_transition_locked` calls in one transaction (a: rollback; b: commit) |
| P4 | `kill_switch.reset` inside the exit transaction |
| P5 | A nested `operator_run_lock` in one process |
| P6 | `dormant → retired` while holding a position and an allocation |

**Discovery:** the inputs are as listed, with no duplicates. UX does not apply (one CLI command).
The story traces FR7, FR9–FR10, NFR1 and NFR4–NFR8. The contract also bears on FR11 (long-only
account) through the drain's offsets.

## Requirements Traceability

| AC | Contract | Code seams (verified) | Status |
|---|---|---|---|
| 1 cohort, safe stages | §1 table | `legacy_deployment_strategies` is immutable and persists after exit (db/deployment.py:61-85). `active_deployment` (store/deployment.py:71-83), `is_legacy_deployment_strategy` (:110-113); `repo.get` raises `not_found` | covered, **m1** |
| 2 drain | §2 | `paper_scoped_cancel` (cli/paper_venue.py:63-66) via `owned_open_order_ids` (execution/live_ledger.py:605-620). `flatten_strategy` records each intent before submitting (execution/flatten.py:54-63, 139-143) and skips \|q\| ≤ 1e-6 (:117-119). `submit_offset` is share-based, 9 dp, DAY (alpaca_broker.py:368-389, :83). `paper flatten` cancels **account-wide** (paper_cmd.py:1302). A tripped switch stops ticks (gating.py:45-46, frozen_runtime.py:191-192, 268-278; live_loop.py:341-355) | **B2, M1, M2** |
| 3 atomic exit | §3.1–3.2 | `apply_transition` commits itself and is top-level-only on revoke edges (store/crud.py:279-317). `_apply_transition_locked` can be composed (store/base.py:27-46, 116-158). `_REVOKE_ON_EXIT` (transitions.py:30-33). The operator flock (operator/deployment_lock.py:15-42, transitions.py:94-96) | **B1, M3** |
| 4 clean re-admission state | §3.3 | `kill_switch.reset` commits (risk/kill_switch.py:24-27). `rebase_strategy_peak` → `clear_peak_equity` commits (execution/peaks.py:46-51, order_state.py:159-168). `audit.append` commits (audit/log.py:7-15). Intake ignores the switch and the peak (registry/intake.py). NAV is computed in live_sizing.py:52-102 | **B1**, m4 |
| 5 no false qualification | §3.4, §4 | Intake requires the newest gate for the manifest identity plus a candidate-episode row that carries that identity (store/deployment.py:157-186). Raw back-steps write NULL hashes (transitions.py:55-57; probe P3b). `candidate → paper` exists only through intake (transitions.py:49-51, crud.py:253-256). Research promote writes a hashed candidate row (store/gate.py:454-457) | holds, **m3** |
| 6 interruption | §4 | Rollback of a composed transaction verified (P3a). Trip-first ordering | **B1, M1**, m5, m10 |
| 7 visibility | §1 `--list` | Fleet rows carry no deployment, cohort or allocation field (execution/fleet_health.py:148-275) | covered via `--list`, m1, m9 |
| 8 contracts | §4–§5 | Type-keyed error registry (cli/errors.py:11-90, 145-147). cli independence contract (pyproject.toml:389-407). Pins: paper_cmd 1382/1409, live_ledger **620/620**, order_state 341/394, crud 345/393. CODEOWNERS `store/` directory rule; `INTEGRITY_CRITICAL_MODULES` (tests/test_repo_hygiene.py:237) | m2, m6 |

No clause adds a lifecycle edge, gate, schema change or authority. The command only reduces risk and
leaves the live wall untouched. FR7's "no ad hoc database stage edits" holds as long as the exit
goes through `_apply_transition_locked` (B1).

## Code Feasibility Verification

| Claim | Verdict | Evidence |
|---|---|---|
| Cohort membership = `legacy_deployment_strategies` row | Holds, persists | The UPDATE, DELETE and late-INSERT triggers abort (db/deployment.py:68-85). After exit the row remains (P3b). Membership alone therefore cannot mean "remaining" (m1) |
| Kill switch stops a legacy tenant from ticking | Holds | Preflight refuses it (`ValueError`), and it is isolated as `strategy_setup_error`; siblings tick on. Its own resting orders are **not** cancelled by run-all |
| `paper_scoped_cancel` exists, scoped to the tenant | Holds | Broker open orders whose `client_order_id` is in the tenant's `paper_venue_orders` |
| "Owned open orders in the paper order ledger" | **Fails** | The ledger has no order state: `status` is always `'submitted'` (live_ledger.py:274). Ownership comes from the ledger; openness comes only from the broker (M1) |
| `paper flatten` implements most of the drain | Partly | Same `flatten_strategy`, but with an account-wide cancel, no `recover_stranded`, and errors swallowed into `flatten_error` (flatten.py:145-149) |
| "The existing bench flatness check" | **Fails** for 6 tenants | `_assert_flat_for_bench` uses `believed_positions`, which keeps any `!= 0.0` (live_ledger.py:416-425). Float dust of ~1e-16 fails it forever, and `flatten_strategy` never offsets it (B2) |
| Exit in ONE `BEGIN IMMEDIATE` with "the existing transition machinery" | **Fails** | P2a: `apply_transition(revoke)` inside a transaction raises `RuntimeError`. P2b: a non-revoke `apply_transition` commits the caller's transaction (`with self._conn`) |
| Two transitions in one transaction | Feasible via `_apply_transition_locked` | P3: `paper → candidate` (revoke) then `candidate → backtested` in one `BEGIN IMMEDIATE`. Legacy members pass the deployment-retirement check (base.py:135-150). Both rows carry NULL hashes. Rollback leaves the tenant at paper and allocated |
| `rebase_strategy_peak` / `kill_switch.reset` in the same transaction | **Fails** | Both commit. P4: after `kill_switch.reset` inside the exit transaction, `in_transaction` is False. A later "rollback" left the tenant retired, unallocated and with the switch cleared |
| "Under the operator lock" + existing transitions | **Fails** | flock is per open-file description. A nested acquire in the same process raises `OperatorLockHeld` (P5). `transition_strategy` takes that lock for `paper → candidate` and `→ retired`, as pinned by tests/test_deployments.py:550-573 |
| `dormant → retired` via the machinery | **Unsafe** | It is not in `_REVOKE_ON_EXIT` and has no flatness check. P6: retired while holding 5 AAPL and an active allocation |
| Old gate never admissible | Holds, wrong reason | All 18 old gates are `consumed=1, actor=agent`, which is exactly the **eligible** shape (store/deployment.py:181-186). The real walls are the newest-gate, candidate-episode and identity checks (m3). All 18 drifted on all three hashes (recomputed at 14621bf) |
| Re-admission via research promote + frozen intake | Holds | Promote needs `backtested` and reuses the existing family (family_assignment.py:82-85). The holdout ledger is per `strategy_id`, so the burned window stays burned (store/holdout.py:56-75) |
| Run-all ingests during the migration | **Fails** | Once every remaining tenant is tripped, run-all exits `strategy_setup_failed` (paper_cmd.py:969-973) before its ingest (:980-981) (M1) |

## Production Sanity Check (read-only, user_version 48)

- **Cohort:** 18 rows in `legacy_deployment_strategies`. All are at `paper` (original stage `paper`),
  with no deployments and none at `dormant`, `forward_tested` or `live`. No global halt; orphan
  fills net to zero. There are 7 stranded order rows with a NULL broker id (ids 4, 7, 9×2, 10×2, 20).
- **Kill-switched:** exactly six, all gross-exposure trips: ids 1 (1.0009), 2 (1.0019), 4 (1.0010),
  5 (1.0252), 6 (1.0159), 7 (1.5786).
- **Holdout Sharpe below 0.2:** ids 4 (0.166), 6 (0.075), 8 (0.059), 10 (0.179), 13 (0.169).
- **O1 as recorded:** retire {1, 2, 4, 5, 6, 7, 8, 10, 13} (9) and move {9, 11, 12, 14, 17–21} (9) to
  backtested. This matches the draft runbook.
  - The runbook still holds ids 1 and 2 for owner confirmation. Their trips are 0.09% and 0.19%
    sizing overshoots, and both cleared 0.2. The plan is not final until Lior confirms.
- **Positions:**
  - 13 tenants hold real ledger positions (ids 4, 8–14, 17–21).
  - Five are flat except for float dust (ids 1, 2, 5, 6, 7), and id 20 carries dust beside real
    positions.
  - Id 4 has been tripped since 09-16 and still holds MRK 9.76 and NVDA 6.56: its breach flatten
    recorded an MRK offset with a NULL broker id and went no further.
- **Holdout:** all 21 gated strategies, including 3 outside the cohort, committed the same burn of
  [2024-07-18, 2026-09-04].

## Findings

### BLOCKER

**B1 — The §3 exit cannot be built from the named machinery.** Four facts break it:

- `apply_transition` owns its transaction (P2a, P2b).
- The hygiene helpers commit (P4).
- `transition_strategy` re-acquires the command's own flock (P5).
- `dormant → retired` neither revokes nor checks flatness (P6).

Followed literally, the contract cannot meet AC3, AC4 or AC6.

**Fix — replace §3 items 1–4 with:**

> The exit is one new store method, `exit_legacy_tenant(rec, to, actor, open_order_ids)`, on a new
> `LegacyExitMixin` (`algua/registry/store/legacy_exit.py`, joined to `SqliteStrategyRepository`). It
> never calls `transition_strategy` or `apply_transition`. Like `intake_candidate_to_paper`, it runs
> at top level only. In one `BEGIN IMMEDIATE`:
>
> 1. **Re-check.** The strategy is in `legacy_deployment_strategies` with no active deployment. Its
>    stage is allowed by §1. It is flat (§3a). `open_order_ids()` is empty: this is the broker
>    listing, called under the write lock as the live exit guard does (crud.py:340-345). Any failure
>    rolls back and raises the §5 exception.
> 2. **Transition.** Call `allocations.revoke_active_locked`, which is a no-op when nothing is
>    allocated. Then call `_apply_transition_locked`:
>    - once for `paper → retired` or `dormant → retired`;
>    - twice for backtested: `paper → candidate`, then `candidate → backtested` on the returned
>      record.
>
>    Every call passes `reason="legacy_migration"`, the actor, `None` for all three hashes and both
>    token ids, and one shared `now`.
> 3. **Hygiene, same transaction.** Delete the tenant's `kill_switches` and `strategy_peaks` rows.
>    Append audit rows `kill_switch_reset` and `legacy_exit` (reason `to=<stage>`). Use non-committing
>    `kill_switch.reset_locked`, `order_state.clear_peak_equity_locked` and `audit.log.append_locked`;
>    the existing committing functions delegate to them. `kill_switch.reset`, `rebase_strategy_peak`
>    and `audit.append` commit, so they must not be called inside the transaction.
> 4. **Commit.**

Keep the NULL-hash sentence. Mirror the change in AC3 by replacing "with the existing transition
machinery" with "through the transition body `_apply_transition_locked`".

**B2 — Float dust wedges 6 of the 18 tenants.** For 7 strategies, `believed_positions` returns
~1e-16 sums (P1a): ids 1, 2, 5, 6 and 7, which are retire-bound, and id 20. These values cause
two failures:

- They fail `_assert_flat_for_bench` forever (P1b, P1d).
- `flatten_strategy` never offsets them (P1c).

Every run would therefore return `pending_flat` with 0 offsets, or `legacy_exit_not_flat`, and O3
("exit the whole cohort") is unreachable. Offsets of fractional fills will create more dust.

**Fix — add §3a:**

> Flat means every `believed_positions(conn, name, PAPER)` quantity has
> \|q\| ≤ `DEFAULT_TOLERANCE` (`algua/execution/reconcile_core.py`, 1e-6 shares). This is the
> tolerance `flatten_strategy` uses to skip dust and the one reconcile compares at. The command uses
> this predicate for the phase decision, the exit re-check and `--list`. It does not use the strict
> `_assert_flat_for_bench`. Sub-tolerance residual fills stay in the ledger, unrewritten. They leave
> `attributed_paper_net` (paper_reconcile.py:35-49) when the stage leaves `paper`.

Add a test with a `0.1 + 0.2 − 0.3` fill triple.

### MAJOR

**M1 — The phase decision needs a fresh ingest and the broker, and no paper cycle will supply
either.** The ledger cannot say whether an order is open. Once the last tickable tenant is tripped,
`run-all` stops before it ingests. The runbook's "wait for the next paper cycle to ingest" therefore
never happens, and the exit never sees flat.

**Fix — prepend to §2 and §3:**

> Every invocation except `--list` runs these steps in order under the lock (§2a):
>
> 1. **Validate** read-only. An unknown name returns `not_found`.
> 2. **Trip the switch** if it is not already tripped (reason `legacy_migration`, audited
>    `kill_switch_trip`). An existing switch keeps its original reason.
> 3. **Ingest** exactly as `run-all` does: `ingest_paper_venue(conn, broker,
>    tick_clock(broker.clock)[0])`, then `recover_stranded(conn, broker, PAPER)`. The command never
>    relies on a paper cycle for ingest.
> 4. **Read owned open orders** with `owned_open_order_ids(conn, broker, name, kind=PAPER)`.
> 5. **Decide the phase.** At `paper`, if an owned order is open or a position exceeds the tolerance,
>    drain and report `pending_flat`. Otherwise exit and report `exited`. At `dormant` or `candidate`,
>    the command only exits; if the tenant is not flat it refuses with `legacy_exit_not_flat`.
>
> Cancel and ingest happen before the decision, so an exit always follows at least one full ingest
> after the last cancel.

**M2 — "Re-running is idempotent" is false for the drain, and its failure path is unspecified.**
`flatten_strategy` re-offsets the ingested belief on every call. A re-run while offsets rest cancels
and replaces them; a 422 on a just-filled order is a no-op (alpaca_broker.py:419-428). A fill not
yet in the activity feed can overshoot once. `flatten_error` swallows every exception, and
`breach_payload` carries no `code`.

**Fix — replace §2 items 2–3 and the last paragraph:**

> **Drain.** Call `flatten_strategy(..., PAPER, lane="paper", strategy_id=rec.id,
> cancel=<paper_scoped_cancel recording the cancelled ids>, ingest=<step 3>)`, which offsets the
> exact belief like `paper flatten`. Offsets are submitted only if `owned_open_order_ids` is empty
> after the cancel and ingest; otherwise the run reports `pending_flat` with no offsets.
>
> **Re-runs** are safe and convergent rather than no-ops:
>
> - resting offsets are cancelled and replaced from the fresh belief;
> - an expired or partially filled DAY offset leaves a residual that the next run offsets;
> - a sibling's order is never cancelled.
>
> **Failure.** A set `flatten_error` raises `BrokerError` (code `broker_error`). Recorded intents stay
> recorded, and the switch stays tripped.
>
> **Payload:** `strategy`, `phase`, `to`, `orders_cancelled` (ids), `offsets_submitted`, and the
> above-tolerance `positions`.

Do not adopt the LIVE `held=` cap. It takes the sign from the account, so a tenant short in a
long-held symbol would sell more (flatten.py:131-136). AC2's "Re-running is idempotent" becomes
"Re-running is safe and convergent".

**M3 — "The operator lock" is unnamed, and the lock-held outcome is unspecified.** The paper operator
holds `operator_run_lock(<git-dir>/operator.lock)` (operator_cmd.py:153-168) for up to 40 minutes per
fire (`TimeoutStartSec=2400`, every 20 minutes). The path is per git directory, so a run from a
worktree takes a different lock and silently gets no exclusion. `OperatorLockHeld` is not in the
error registry, so it would surface as `internal`.

**Fix — add §2a:**

> The command holds `operator_run_lock` on the path `deployment_lock._operator_lock_path()` resolves
> (expose it as a public `operator_lock()` in `algua/operator/deployment_lock.py`), with
> `job="legacy-exit"`. The lock is non-blocking and spans the whole invocation except `--list`.
>
> - **Contention** raises `OperatorLockHeld`, registered as `operator_lock_held` (retryable) with no
>   effect.
> - **Checkout:** the lock serializes with the paper operator only from the operator's checkout
>   (production: the main checkout, `algua-paper.service` WorkingDirectory).
> - **Never** call `transition_strategy` while holding it.

### MINOR

- **m1 State table.** Add these rows:
  - unknown name → `not_found`;
  - `--to` equal to the current stage → `exited`, idempotent (only then);
  - `backtested` with `--to retired` → `legacy_exit_unsupported_stage` (use `registry transition`);
  - a cohort member at `candidate` (reached by a raw back-step) → exit only, `candidate →
    backtested|retired`;
  - `dormant` or `candidate` not flat or owning open orders → `legacy_exit_not_flat`, with no drain
    (`paper flatten` refuses those stages and run-all never ticks them).

  Define the remaining cohort member as: a legacy row, no active deployment, and stage in {paper,
  forward_tested, live, dormant, candidate}. For each member, `--list` reports stage, `allocated`,
  kill-switch reason and above-tolerance positions.
- **m2 Codes.** Define `LegacyExitNotCohort`, `LegacyExitUnsupportedStage` and `LegacyExitNotFlat` as
  `TransitionError` subclasses in `algua/registry/legacy_exit.py`, which must not import `store` at
  module level. Register them before `(TransitionError, "wrong_stage")` and add rows to the envelope
  doc; all three are non-retryable. `--actor` is an audit label only, as in `paper flatten`.
- **m3 §4 rationale.** Replace "already consumed and identity-mismatched" with three facts:
  - intake admits only the newest passing gate for the manifest identity;
  - the latest candidate row must carry that identity, and after exit it is the NULL-hash back-step
    (store/deployment.py:164-180);
  - the 18 old identities no longer match the checkout.

  Consumed agent gates are the eligible kind (:181-186).
- **m4 NAV carry.** Re-admitted NAV = allocation + lifetime dividend credits + realized and
  unrealized P&L over the **full** fill history of each currently held symbol, dust symbols included.
  Sizing equity = min(allocation, NAV). The offset is therefore not a fixed "small realized offset".
- **m5 AC6 wording.** Membership is immutable, so "never paper without cohort membership" is vacuous
  for members. Replace it with "an exited member returns to `paper` only through frozen intake; no
  allocation remains on a non-paper strategy."
- **m6 Placement:**
  - The drain lives in `algua/cli/paper_migrate_cmd.py`; registry cannot import `cli.paper_venue`.
  - Mount `migrate_app` flat onto `paper_app` in `cli/main.py`, using the `data_refresh_cmd` idiom.
  - Add the module to the cli independence list.
  - Add `registry/legacy_exit.py`, `store/legacy_exit.py` and the CLI module to CODEOWNERS and
    `INTEGRITY_CRITICAL_MODULES`.
  - Add nothing to live_ledger.py, which is at its 620/620 pin.
  - Update docs/architecture.md:39 ("explicit unmigrated cohort").
- **m7 Bypass.** A raw `registry transition` on a member skips the order drain (registry_cmd.py:38-44),
  and `paper resume` re-arms a draining tenant. The runbook must forbid both for cohort members.
- **m8 Global halt.** State that, like `paper flatten`, the command ignores the halt and reports
  `global_halt`. `fleet health` stays red for draining members (operational and halted) until they
  exit.
- **m9 Runbook:**
  - The schema is v48, not v47.
  - Freeze the plan before step 2, keyed on the original trip reasons.
  - Stop `algua-paper.timer` (and wait for the service to go idle) before draining; restart it after
    verification. Otherwise an all-tripped book alerts `strategy_setup_failed` every 20 minutes.
  - Steps 3–4: re-run after the fills; the command ingests itself.
  - Replace the unverifiable checks ("fleet status shows no allocation", "registry list shows
    deployments") with `--list`.
  - Add a broker-flat check before any frozen re-admission. Unattributed residuals would make the
    first frozen cycle's reconcile defer and then halt.
  - For O2, choose `--end` and `--holdout-frac` so that holdout_start is after 2026-09-04 and there
    are at least 63 observations. The burn identity is the out-of-sample interval.
- **m10 Fault seams.** Name them in §4:
  - a raise after the switch trip;
  - a raise after an intent is recorded but before backfill (a stranded row, recovered on the next
    run);
  - a raise from `append_locked` after both transition rows (rollback leaves the tenant at paper,
    allocated and tripped).

### NOTE

- **N1** Under the exit, one broker listing call runs while the SQLite write lock is held. This
  matches the live precedent. With `busy_timeout` at 5 s, a slow call can briefly return
  `db_unavailable` to other writers.
- **N2** Name reuse (O6) and the environment re-epoch (O4) stay deferred, as the story says.

## Risk to Existing Behaviour

- **Working-tree, frozen and live lanes:** untouched.
- **Shared helpers:** the new `*_locked` variants are delegated to by the existing committing
  functions, so callers see no behaviour change.
- **Unchanged checks:** `_assert_flat_for_bench` and `_REVOKE_ON_EXIT` are not edited, so the
  existing bench and lane edges keep their current semantics. The dust weakness on `paper → dormant`
  remains, out of scope.
- **Production impact:** limited to the 18 cohort members, executed per runbook with the timer
  stopped.

## Summary and Verdict

**Overall readiness: NOT READY as drafted.** Two contract mechanisms fail against the real code:

- **B1:** the single-transaction exit cannot be built from `apply_transition` and the committing
  hygiene helpers, under a lock that `transition_strategy` would re-take.
- **B2:** the strict flatness check wedges 6 production tenants on float dust.

Three MAJOR gaps would force design choices:

- **M1:** the order and ingest source for the phase decision;
- **M2:** convergent drain semantics and error mapping;
- **M3:** lock identity and the lock-held code.

All fixes are text-only and exact; the composed transaction and its rollback were verified in
scratch (P3). After B1–B2 and M1–M3 are applied, the contract is **READY**. A delta check of §2–§3 is
enough; no further full review is needed. Fold m1–m10 in as convenient. Retiring tenants 1 and 2
needs Lior's confirmation before the runbook runs. Readiness never authorizes trading, capital use or
a production run.
