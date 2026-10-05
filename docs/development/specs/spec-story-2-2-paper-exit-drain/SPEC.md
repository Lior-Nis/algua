---
id: SPEC-story-2-2-paper-exit-drain
companions:
  - paper-exit-drain-contract.md
  - ../../stories/2-2-drain-resting-paper-orders-on-lane-exit.md
sources:
  - https://github.com/Lior-Nis/algua/issues/685
  - ../../epics.md
  - ../spec-story-1-4-legacy-cohort-exit/SPEC.md
---

> **Canonical contract.** This SPEC and the files in `companions:` are the complete contract for
> what to build, test and validate. Source documents remain narrative and traceability evidence;
> they do not override this contract.

# Story 2.2 Paper Exit Drain Contract

## Why

A book exit sheds the strategy's allocation and re-checks its ledger positions inside the exit
transaction (`algua/registry/store/crud.py:279-353`), but it looks for the strategy's resting
orders only when an `ExitLaneGuard` is injected, and only the live lane injects one
(`algua/cli/registry_cmd.py:38-93`). A paper strategy can therefore leave the paper book with a
resting order. When that order fills, the fill is a material belief on a strategy that is no longer
on the lane, which the paper account reconcile refuses to count
(`algua/execution/paper_reconcile.py:36-58`). The cycle defers and, after the grace window, engages
the global halt, which also stops `live run-all`. The signed go-live (`forward_tested -> live`) and
the unattended paper run both pass through paper-lane exits.

## Capabilities

- id: CAP-1
  intent: Every paper-lane book exit drains the strategy's own resting paper orders before the
    registry write lock.
  success: On each of the five paper-source edges in `_REVOKE_ON_EXIT` (`paper -> candidate |
    dormant | retired`, `forward_tested -> retired | live`) the transition syncs the paper venue
    (fill ingest plus stranded-order recovery up to the venue's clock); if the strategy holds no
    material paper position, it cancels only its own open paper orders, observes them again and
    syncs once more. A sibling's resting order survives and the account-wide cancel is never
    called.
- id: CAP-2
  intent: A paper-lane exit can succeed only through a completed drain.
  success: Under the write lock the exit is refused, with stage, allocation, deployment and go-live
    challenge untouched, when the strategy holds a material paper position, when one of its orders
    was still open after the cancel, when a fresh re-list shows one, or when the drain was skipped.
    The refusals reuse the existing "not flat" `TransitionError` messages.
- id: CAP-3
  intent: The drain fails closed and says why.
  success: Missing paper credentials refuse the exit with a `paper_exit_drain_unavailable` audit
    row. A clock, ingest, open-order read or cancel failure before the lock refuses it with a
    `paper_exit_drain_failed` audit row that names the step. A failure under the lock rolls back.
    No failure falls back to the positions-only check.
- id: CAP-4
  intent: One guard selection serves both lanes, and every allocation-shedding transition gets it.
  success: `transition_strategy` selects the source lane's guard itself, from the same record whose
    stage the compare-and-swap checks, for every edge in `_REVOKE_ON_EXIT` and no other.
    `registry_cmd.py` no longer selects and shrinks. Live drains, the account-credential fallback,
    their messages and their audit actions are unchanged. A structural test ties the edge set, the
    selector and the single production path into the store together.
- id: CAP-5
  intent: A paper-lane exit cannot interleave with the paper operator.
  success: Every paper-source exit takes the non-blocking `operator.lock`, as retirement already
    does, so a timer-driven paper cycle can neither run during the drain nor submit for the exiting
    strategy after the exit commits.

## Constraints

- `paper-exit-drain-contract.md` is normative: the edge set, call order, guard algorithm, messages,
  audit actions, failure classification, placement, protection and tests.
- The store is unchanged: `apply_transition`, `_assert_flat_for_bench`, the Story 1.4 dust rule, the
  positions check and both "not flat" messages stay as they are.
- No schema change, CLI command or flag, error code, lifecycle edge, gate or authority. Paper-lane
  exits stay agent-allowed; go-live stays human-only and signed.
- The guard performs no database write and no commit while the registry write lock is held
  (`audit.log.append` commits).
- The registry reaches the guards only through a lazy import in `transitions.py`. No import-linter
  exemption. `transitions.py` stays under 300 lines.
- The new and moved modules join CODEOWNERS and the integrity-critical set.

## Non-goals

- Changing the live guard's internals. Two live-lane residuals are recorded for a follow-up issue.
- Concurrency with paper commands run by hand outside the operator wrapper (operator discipline, as
  for `paper run-all` today).
- Activity-feed visibility lag at the venue (the existing cursor semantics of every paper ingest).
- #647 (paper whole-account loss breaker), #662, #677, cleaning stranded order rows, and a paper
  break-glass.

## Success signal

Each paper-source exit cancels the strategy's own resting paper orders and refuses while one remains
or a fill lands; it refuses, audited, when the venue cannot be reached; a sibling's order survives;
an exit attempted while liquidation offsets are still resting leaves them in place; live exits
behave and read exactly as before; the structural lane test and every mutation check pass; and the
full repository gate passes.
