---
id: SPEC-story-2-1-unrelaxed-live-qualification
companions:
  - live-qualification-contract.md
  - ../../stories/2-1-refuse-live-qualification-on-relaxed-gates.md
sources:
  - https://github.com/Lior-Nis/algua/issues/624 (owner decision of 2026-10-04, item 2)
  - https://github.com/Lior-Nis/algua/issues/682 (owner decision of 2026-10-04)
  - ../spec-story-1-3d-frozen-evidence-and-qualification/SPEC.md
---

> **Canonical contract.** This SPEC and the files in `companions:` are the complete contract for
> what to build, test and validate. Source documents remain narrative and traceability evidence;
> they do not override this contract.

# Story 2.1 Unrelaxed Live Qualification Contract

## Why

Signed relaxations exist so a human can explore: declare breadth, reuse a holdout, accept a non-PIT
universe, loosen a threshold. The owner decided on 2026-10-04 that such exploration may never
authorize go-live: experimental live capital is authorized only by a research gate and a forward
certificate that both ran at their protected defaults, and nobody, human included, can waive that.
Today the rows do not record what was relaxed, the live wall never asks, and two raw transitions
reach `candidate` and `forward_tested` without any gate. This story records the exact relaxation
set on every gate row, judges it once at go-live, and removes the two raw edges.

## Capabilities

- id: CAP-1
  intent: Which inputs relax a gate is defined once, in a closed vocabulary.
  success: One protected module names every relaxation token and computes the research and forward
    sets from a run's inputs with two pure functions; every option of `research promote`,
    `promote_task` and `paper promote` is classified in an explicit table, and a test fails if one is
    added without a classification.
- id: CAP-2
  intent: Every new gate row records its relaxation set, permanently.
  success: Every research and forward evaluation row written after v49, pass or fail, carries a
    canonical `relaxations_json` (`[]` when unrelaxed); the schema refuses a new row without one, a
    malformed or out-of-vocabulary value, and any later change to it.
- id: CAP-3
  intent: Rows written before v49 are classified once, conservatively.
  success: The v49 migration derives a set only where the row's own columns prove it (an agent row
    with recoverable thresholds and a known data source); every other row stays NULL, which reads as
    `unrecorded`. It changes no other column, runs once, and is idempotent. On the production
    registry of 2026-10-05 it classifies all 21 research rows as `[]`; there are no forward rows.
- id: CAP-4
  intent: Go-live is refused unless both qualifying rows are unrelaxed.
  success: One function runs the certificate verifier and then the qualification predicate, at
    challenge issuance (before any `live_challenges` row) and at completion (before signature
    verification or challenge consumption), for the injected and the default verifier alike. A
    relaxed or unrecorded research gate or certificate refuses with the stable, non-retryable code
    `live_qualification_relaxed`, naming each relaxation and the row that carried it. A valid human
    signature does not override it; the issued challenge shows both rows' (empty) sets.
- id: CAP-5
  intent: Exploration behaves exactly as before.
  success: Signed relaxed `research promote` and `paper promote` runs produce byte-identical
    challenges, verdicts, tokens and stage moves; a relaxed human research gate still anchors a paper
    deployment; a relaxed certificate still promotes `paper -> forward_tested` and refreshes there.
- id: CAP-6
  intent: The gate commands are the only ways into `candidate` and `forward_tested`.
  success: A raw `backtested -> candidate` or `paper -> forward_tested` transition is refused for
    every actor, human included; every other raw edge, including the `paper -> candidate` back-step,
    is unchanged.

## Constraints

- The vocabulary, predicates, DDL, triggers, migration rules, call sites, refusal codes and
  messages in `live-qualification-contract.md` are normative; its §2 is this story's protected
  schema review.
- There is no in-band waiver: no flag, actor, signature or configuration disables the predicate.
- No gate threshold, default, verdict, token, stage move, human-actor challenge byte or go-live
  signed payload changes. Frozen go-live stays `frozen_live_unsupported`. No capital, allocation or
  live-activation behaviour changes. Agents gain no authority.
- A row's relaxation set is computed from the run's own inputs when the row is written; it is never
  derived later, except once, by the v49 migration, for rows that predate it.
- Every new module is CODEOWNERS-protected and in the integrity-critical set; pinned modules are
  carved, not grown; `algua/contracts` stays pure; the full root gate passes.
- v49 is forward-only: v48 code cannot write a gate row on a v49 registry (the insert trigger
  refuses it), which fails closed.

## Non-goals

- Binding live authorization to the frozen deployment, the signed go-live payload, or frozen go-live
  (Story 2.3).
- Removing the legacy-cohort certificate branch and the legacy tick paths (the Story 1.4
  follow-up); this story makes the branch unable to authorize go-live.
- Removing the token-consume parameters of the store's `apply_transition` primitive, which no
  production caller uses once the raw edges are gone (recorded follow-up).
- Showing relaxation sets in `registry gates`, `research promote` or `paper promote` output.
- Restricting or re-pricing any relaxation for research or paper.

## Success signal

Every gate row written after v49 carries its relaxation set and cannot be changed; the 21 existing
production research rows read `[]`; a go-live whose research gate or certificate was relaxed or is
unrecorded is refused with `live_qualification_relaxed` before any challenge is issued or any
signature is checked, whatever verifier is injected and whoever signs; signed exploration is
byte-for-byte unchanged; raw transitions into `candidate` and `forward_tested` are refused for
every actor; and the full repository gate passes.
