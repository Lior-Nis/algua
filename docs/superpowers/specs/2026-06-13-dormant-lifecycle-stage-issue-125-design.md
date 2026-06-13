# Dormant lifecycle stage — Slice A (state model) — issue #125

**Date:** 2026-06-13
**Issue:** #125 — Add a non-terminal `dormant` lifecycle stage for validated-but-resting strategies (vs. terminal `retired`)
**Scope:** Slice A of 2. This slice ships the **state model** only. The nightly re-eval sweep is Slice B (a separate follow-up issue).

## Problem

`retired` is today the only "stop" state, and it is a terminal tombstone (irreversible by design). A `live` or `paper` strategy that *was* validated but stops working — seasonality, a regime shift, temporary signal degradation — has only two bad options:

- **retire it** — throws away something that earned its way up and may recover, or
- **demote to `paper`** — wrong semantics: `paper` means "new candidate auditioning toward live for the first time," not "validated strategy resting."

There is no state for **"valid, not trading now, may return."** That is the gap this slice fills.

## Decision (from the issue's design fork + brainstorming)

`dormant` is a **new lifecycle stage** (not a flag): a true non-terminal end-state a strategy can climb *out* of, so it earns distinct transition rules and first-class querying. This touches `algua/contracts/lifecycle.py` (CODEOWNERS-protected) and is therefore a human design decision, recorded here.

### Scope split (decided in brainstorming)

- **Slice A (this spec):** the state model — the `dormant` stage, its transition edges, entry rules, the bench-reason guard, and the tests. Driven entirely through the existing `registry transition` CLI; no new command.
- **Slice B (deferred):** the periodic re-eval sweep that re-runs the walk-forward gate over the dormant pool on the latest bars and surfaces recovery candidates. Deferred because it is the larger, riskier build and depends on this slice's state model.

## The state model

`algua/contracts/lifecycle.py` defines `_LIVE_TRANSITIONS` (the core acyclic edges) and derives `ALLOWED_TRANSITIONS` by auto-appending `-> RETIRED` to every non-`retired` stage. Adding `dormant` is therefore a localized change.

Add to the `Stage` enum:

```python
DORMANT = "dormant"
```

Wire edges in `_LIVE_TRANSITIONS`:

```python
Stage.PAPER:   {Stage.FORWARD_TESTED, Stage.CANDIDATE, Stage.DORMANT},
Stage.LIVE:    {Stage.PAPER, Stage.DORMANT},
Stage.DORMANT: {Stage.PAPER},          # -> RETIRED is auto-added by the comprehension
```

Resulting edges:

| Edge | Meaning |
|---|---|
| `live -> dormant` | bench a validated live strategy |
| `paper -> dormant` | bench a validated paper strategy |
| `dormant -> paper` | recovery: re-enter the audition |
| `dormant -> retired` | permanent give-up (auto-derived retire edge) |

Properties:

- **Entry only from `live` and `paper`.** No stage below `paper` (idea/backtested/candidate) gains a `-> dormant` edge — only previously-validated strategies can be benched. Pre-validation strategies just back-step or retire.
- **`dormant` is non-terminal** (it has the outgoing `-> paper` edge); `retired` remains the terminal tombstone (`ALLOWED_TRANSITIONS[RETIRED] == set()`).

## Actor gating

No new token plumbing this slice. All four edges flow through the existing plain `apply_transition` path in `transition_strategy` (`algua/registry/transitions.py`), which special-cases only `-> live`, agent `backtested -> candidate`, and `paper -> forward_tested`. The dormant edges match none of those, so they are open to **any actor**:

| Edge | Actor |
|---|---|
| `live -> dormant`, `paper -> dormant` (bench) | any |
| `dormant -> paper` (recovery) | any |
| `dormant -> retired` | any |
| `forward_tested -> live` (unchanged) | human, signed gate |

Actor-openness is orthogonal to the `live -> dormant` wind-down preconditions below (flat + allocation-release): those are safety walls that apply to *any* actor, agent or human. An agent may bench, but only a strategy that has already been wound down to flat — which keeps agent benching safe even though the agent can never put a strategy live.

### Why `dormant -> paper` open to agents is safe (the statistical guard)

The issue's non-negotiable guard is "recovery must re-clear the real gate; no fast-lane back to live." That guard is already enforced **downstream**, not on this edge: `paper` is not a privileged state. A recovered strategy sitting at `paper` must still honestly re-earn the #124 `forward_tested` gate (a fresh passing forward-gate token) and then a **human** must re-sign the live gate. So letting an agent move a dormant strategy back to `paper` grants no shortcut — it only re-opens the audition the strategy must re-win from the bottom of the trade ladder. The deferred Slice B sweep is therefore an *evidence/recommendation* tool, never a gate, which keeps it consistent with the issue's "no auto-reactivation engine" boundary.

### The bench-reason guard

Entering `dormant` must record *why*. Add one guard in `transition_strategy` (the platform's policy layer): if `target == Stage.DORMANT` and `reason` is missing/blank, raise `TransitionError`. Scoped to `target == DORMANT` only — recovery and retire edges keep `reason` optional, as today. The rationale is persisted in the existing `stage_transitions.reason` column (no schema change).

This sits in `transition_strategy`, not in the lower-level `repo.apply_transition`, deliberately: every existing gate (the human-only live wall, the shortlist token, the forward token) lives in `transition_strategy`, while `apply_transition` is the storage primitive and is intentionally policy-free. `transition_strategy` is the single operational mutation path (the CLI routes every stage change through it); a test or helper that calls `apply_transition` directly to construct a fixture bypasses *all* policy by design, not just this guard. Placing the reason guard here is therefore consistent with the established architecture.

## Winding down a live strategy before benching (safety preconditions)

`live -> dormant` must not strand real money. A live strategy that still holds positions or an open allocation would, once benched, vanish from the live loop (`run-all` only iterates `Stage.LIVE`) — no fill ingestion, reconcile, risk handling, or scoped cancel — while sitting in a state that is by definition non-tradeable. So the `live -> dormant` edge carries two preconditions, enforced in the CLI `registry transition` command (mirroring where the go-live path already does its allocation handling, `registry_cmd.py:119-125`):

Both are enforced **inside `transition_strategy`** — the single policy path that already runs `validate_transition` (legality) and will run the bench-reason guard. We deliberately do NOT add a second operational write path (an alternate composite repo method would bypass those guards); instead the allocation-revoke is threaded through `apply_transition` as a side-effect, exactly like the existing `consume_gate_id`/`consume_forward_gate_id` tokens.

1. **Must be flat.** When `rec.stage is LIVE and target is DORMANT`, `transition_strategy` checks `believed_positions(conn, name)`; if non-empty, raise `TransitionError` (flatten first). The transition does *not* auto-flatten — flattening is the existing risk/execution machinery's job; the bench edge only refuses until the strategy is wound down. `believed_positions` is imported lazily, the same pattern the live-certificate verifier already uses for its execution-layer imports in this very file (`transitions.py`), so this adds no new import-time `registry -> execution` coupling.
2. **Releases the allocation in the SAME transaction as the stage change.** Revoking the allocation and flipping the stage to `dormant` must be atomic — otherwise a partial failure leaves either a `live` strategy with no allocation (the live loop then errors on it) or a `dormant` strategy still holding a capital reservation. `apply_transition` already runs its token-consume + stage CAS + transition INSERT inside one `with self._conn:` block via `_apply_transition_locked` (store.py:270-291). The design:
   - factors the revoke write into a **non-committing** `allocations.revoke_active_locked(conn, strategy_id)` (the existing committing `deallocate` is refactored to call it, preserving its `is_flat` wrapper contract), and
   - adds a `revoke_allocation_id: int | None` parameter to `apply_transition`/`_apply_transition_locked` that runs that revoke inside the same transaction (rowcount-checked → whole transition rolls back if the allocation row vanished), mirroring how `consume_gate_id` is handled.

   For `live -> dormant`, after the flat check, `transition_strategy` looks up the active allocation and passes its id as `revoke_allocation_id`. The `is_flat` precondition that the committing `deallocate` enforces is satisfied here by step 1's flat check.

These apply only to the `live -> dormant` entry edge. `paper -> dormant` needs neither — paper strategies hold no live positions and no live allocation (paper positions are simulated) — so it falls through the ordinary `transition_strategy` path with no extra work. This makes the *new* `live -> dormant` edge strictly safer than the pre-existing `live -> paper` edge, which has the same (separately-tracked, deferred) wind-down gap; we are not fixing `live -> paper` in this slice, only ensuring the edge we add is sound.

## What does NOT change

- **No DB migration.** `stage` is a free TEXT column; `stage_transitions.reason` already records the bench rationale. `SCHEMA_VERSION` stays 21.
- **Trade-lane RUN guards unchanged.** The `paper`/`live` *run* guards are **allowlists** (only `PAPER`/`FORWARD_TESTED` run the paper lane, only `LIVE` is iterated by the live `run-all`). A `dormant` strategy matches no allowlist, so it is inherently non-runnable — no edits there; we add a test locking the property in.
- **One capital-guard edit:** `live allocate` (`live_cmd.py:75`) currently writes an allocation without checking stage. Add a guard so it rejects a `dormant` (resting) strategy — capital is only allocated to a strategy on its way to/at live, and re-allocating a recovered strategy happens only after it re-climbs `paper -> ... -> live`.
- **No new CLI command.** Benching, recovery, and give-up all use the existing `registry transition --to {dormant,paper,retired}`.

## Explicitly out of scope (→ follow-up issues)

- **Slice B: the re-eval sweep** — periodic batch that re-runs the walk-forward/research gate over the dormant pool on the latest bars and surfaces recovery candidates. The consumer of a structured stop-reason.
- **Structured `dormancy_reason` enum column** (e.g. seasonal | regime_shift | signal_degradation | other). Deferred with Slice B per YAGNI; until then the free-form transition `reason` captures it.
- **The pre-existing `live -> paper` wind-down gap.** This slice makes the *new* `live -> dormant` edge flat-and-deallocated-safe (see "Winding down a live strategy"), but does **not** retrofit the same preconditions onto the existing `live -> paper` edge — that remains the separately-tracked flatten/ledger-reconcile backlog item. The asymmetry is intentional: we harden the edge we add, without expanding scope into the existing one.
- **No auto-reactivation engine** — no auto-detect-regime, no auto-rebench, no auto-repromote (per the issue).

## Tests

`tests/test_lifecycle.py`:
- `live -> dormant`, `paper -> dormant`, `dormant -> paper`, `dormant -> retired` are legal.
- Illegal entries rejected: `idea -> dormant`, `backtested -> dormant`, `candidate -> dormant`, `forward_tested -> dormant`, `retired -> dormant`.
- `dormant` is non-terminal (`ALLOWED_TRANSITIONS[DORMANT]` non-empty); `retired` stays terminal.
- The transition table stays total; every non-retired stage (now including `dormant`) can reach `retired`.

`tests/test_cli_registry.py` and/or `tests/test_e2e_lifecycle.py`:
- e2e `live -> dormant -> paper` and `paper -> dormant -> retired` via the CLI, with `actor=agent`.
- Benching to `dormant` without a `--reason` is rejected; with a reason it succeeds and the reason lands in transition history.
- `paper -> dormant` succeeds with no flat/allocation precondition (paper has no live state).

Live wind-down preconditions (`transition_strategy`-level in `tests/test_registry_store.py`/`test_cli_registry.py`, plus CLI e2e):
- `live -> dormant` is **rejected** when the strategy has believed live positions (not flat).
- `live -> dormant` on a flat strategy with an active allocation **succeeds and revokes the allocation atomically** (`active_allocation` is `None` afterward), with `--actor agent`.
- **Atomic rollback:** if the stage CAS/INSERT is forced to fail, the allocation revoke is rolled back too (allocation still active, stage still `live`) — the revoke never lands without the stage change, nor vice versa.
- The bench-reason guard and `validate_transition` legality still fire on this path: blank-reason `live -> dormant` rejected; an illegal source into `dormant` rejected — confirming the threaded-revoke path did not bypass the policy guards.
- `live allocate` is **rejected** for a `dormant` strategy.

Trade-lane test (in the paper/live CLI test module):
- A `dormant` strategy is rejected by the `paper` run guard and is not iterated by the live `run-all`.

Policy-placement note (not a bypass to fix): a direct `repo.apply_transition(..., Stage.DORMANT, reason=None)` is *expected* to skip the reason guard, exactly as it skips every other gate — the guard is enforced at the operational `transition_strategy`/CLI layer, which the tests above exercise.

## Gate

`uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports` green.
