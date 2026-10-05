---
baseline_commit: e875b6d
---

# Story 2.1: Refuse live qualification built on relaxed gates

Status: backlog

Prepared: 2026-10-04. Baseline: `e875b6d` (main after PR #686). Epic: 2.
Requirements: FR14 (signed-relaxation policy), FR8 (live-wall preconditions), NFR2, NFR4–NFR6,
NFR8.
Owner decision: [#624, comment of 2026-10-04, item 2](https://github.com/Lior-Nis/algua/issues/624).
Depends on: Stories 1.2 and 1.3d (done). No Epic 2 predecessor.
Readiness: 2026-10-05,
[implementation readiness report](../implementation-readiness-report-2026-10-05-story-2-1.md):
READY WITH CONDITIONS (0 blockers, 1 major, 7 minor, 6 notes). All conditions (M1, m1–m7) are
applied to the contract, and the notes are recorded, in the decision log entry "2026-10-05 —
Readiness corrections". No further full review is required. Status stays `backlog` until the
contract PR merges.

## Story

As Algua's owner,
I want go-live to refuse any strategy whose research gate or forward certificate used a relaxation,
so that signed relaxations stay available for exploration while experimental live capital is only
ever authorized by gates at their protected defaults.

## Context

The owner decided on 2026-10-04: signed relaxations remain available for research and paper
exploration; a relaxed research gate or forward certificate can never authorize go-live;
experimental live qualification must pass every gate at its protected default; agents receive no
waiver authority.

What exists today:

- Relaxations are human-only and authenticated (#329). The code enumerates them in
  `algua/registry/human_actor.py:17-20`: declared breadth `--n-combos`, `--allow-holdout-reuse`,
  `--allow-non-pit`, `--assume-terminal-last-close`, the NOVEL/PARENTAGE `--new-family` mint, and
  the `paper promote` threshold relaxations. Agents are refused by `guard_agent_relaxations`
  (`algua/registry/promotion.py:37-54`), the terminal-last-close guard
  (`algua/registry/promote_run.py:128-137`) and `guard_forward_relaxations`
  (`algua/registry/forward_promotion.py:47-70`).
- The rows do not record what was relaxed. A research row (`gate_evaluations`,
  `algua/registry/db/gate.py:25-62`) keeps `pit_override` and `breadth_provenance` (`'declared'`
  under `--n-combos`, `promotion.py:186-190`) but not holdout reuse (only
  `holdout_evaluations.reused`, which no gate row links to), terminal-last-close, `--new-family` or
  loosened advisory thresholds (only inside `decision_json`). A forward row
  (`forward_gate_evaluations`, `algua/registry/db/forward_gate.py:20-51`) keeps six of its eight
  thresholds as columns; `min_session_coverage` lives only in `decision_json` and
  `forward_sharpe_confidence` only in a check's detail text.
- A human-signed research gate can anchor a deployment: paper intake accepts `actor='human'` with
  `consumed=0` (`algua/registry/store/deployment.py:181-186`).
- The live wall runs actor, then certificate, then approval
  (`algua/registry/transitions.py:116-157`). The certificate verifier
  (`algua/registry/live_certificate.py:36-192`) is called at challenge issuance
  (`algua/cli/registry_cmd.py:215-217`) and at completion (`transitions.py:142-143`). It selects the
  newest certificate (`live_certificate.py:83-95`) but never asks how that certificate or the
  deployment's research gate were produced. A programmatic caller may also inject its own
  `forward_certificate_verifier` (`transitions.py:43`), so a check placed only inside the default
  verifier is bypassable.
- Production on 2026-10-04 (read-only): 21 research rows, all `actor='agent'`,
  `breadth_provenance='measured'`, `pit_override=0`; no forward rows; no deployments; no holdout
  reuse.

Timing matters. The research, merge-back and frozen-intake path runs unattended. A deployment
admitted before this story lands anchors on a research row with no recorded relaxation set.
Recording must start, and existing rows must be classified once, before any such deployment can
approach go-live (at least 63 forward sessions away).

## Scope and authority

In scope: record the exact relaxation set on every new research and forward evaluation row
(append-only, immutable); classify pre-existing rows once at migration; enforce one qualification
predicate on the go-live wall at challenge issuance and at completion; close the legacy-cohort
go-live branch, which has no deployment-bound research gate to judge (the cohort is retired by
Story 1.4).

Must not: change any gate threshold, default, verdict, token or stage move; remove or restrict any
relaxation for research or paper; add any flag, actor or signature that waives the predicate (like
the certificate wall, there is no in-band waiver); change the signed go-live payload (Story 2.3);
enable frozen go-live (it stays `frozen_live_unsupported` until Story 2.3); touch capital,
allocations or live activation; give agents any new authority.

## Normative contract

The [Story 2.1 machine contract](../specs/spec-story-2-1-unrelaxed-live-qualification/SPEC.md)
and its [field-level companion](../specs/spec-story-2-1-unrelaxed-live-qualification/live-qualification-contract.md)
are normative. The companion's §2 (exact DDL, triggers and migration order) is this story's
protected schema review and its §7 is the entry-point inventory. Implementers and reviewers must
read both and the [decision log](../specs/spec-story-2-1-unrelaxed-live-qualification/.decision-log.md),
which records the calls the acceptance criteria left open.

## Acceptance criteria

1. **Closed relaxation vocabulary.** One protected module defines a closed vocabulary and two pure
   functions. `research_relaxations(...)` returns `declared_breadth` (`--n-combos` given),
   `allow_holdout_reuse` (flag given, whether or not an overlap was found), `allow_non_pit`,
   `assume_terminal_last_close`, `new_family`, and `threshold:<field>` for each `GateCriteria` field
   (`algua/research/gates.py:98-106`) looser than its protected default. `forward_relaxations(...)`
   returns `threshold:<field>` for each of the eight `ForwardGateCriteria` fields looser than its
   default, using exactly the direction table of `guard_forward_relaxations`. A stricter value is
   not a relaxation. Output is sorted and unique; the empty tuple means unrelaxed. The contract also
   classifies `--demo` (recommended: `demo_data`, because a gate on synthetic bars is not a gate at
   its protected default).
2. **Every flag is classified.** A test enumerates every option of `research promote` and
   `paper promote` and requires each to appear in an explicit table as either a relaxation or a
   non-relaxation input, so a new relaxation flag cannot ship unclassified.
3. **Recorded at evaluation, immutable.** Every new research and forward row, pass or fail, from
   every writer (`algua/registry/store/gate.py:85`, `:386`; the one INSERT at
   `algua/registry/store/forward_gate.py:108` behind both forward record paths, `:18` and `:129`)
   records `relaxations_json`, the canonical JSON array of its set (`[]` when unrelaxed). The schema
   refuses a new row without it and any later change to it. Agents keep being refused exactly as
   today; an agent row that tightens a threshold records `[]`.
4. **One-time classification of existing rows.** The migration classifies each pre-existing row
   once, before the immutability trigger exists. An `actor='agent'` research row records the set
   derivable from its columns and `decision_json` thresholds (human-only relaxations are impossible
   for an agent). A forward row records the set derived from its threshold columns and
   `decision_json`. An `actor='human'` research row, or any row whose thresholds cannot be parsed,
   stays NULL and reads as `unrecorded`. The migration is idempotent, changes no verdict, token or
   other column, and is rehearsed on a copy of production (the 21 agent rows become `[]`).
5. **One predicate, unbypassable.** `live_qualification_relaxations(...)` returns the union of the
   deployment's `research_gate_id` row set and the selected certificate row set, with `unrecorded`
   for NULL and for a strategy with no active deployment. It runs in a single function that both the
   challenge-issuance path and `_validate_live_gate` call after the certificate verifier returns the
   certificate id, so an injected verifier cannot skip it. Go-live proceeds only when the set is
   empty.
6. **Refused before any signing step.** A non-empty set refuses issuance before a `live_challenges`
   row is written and refuses completion before signature verification or challenge consumption. The
   refusal is a `TransitionError` subclass with stable code `live_qualification_relaxed` that names
   each relaxation and the row that carried it; it is registered in `algua/cli/errors.py` and
   `docs/contracts/cli-error-envelope.md`.
7. **No human override.** A fully authenticated human go-live with a valid signature and an
   otherwise valid certificate is refused when either row is relaxed or unrecorded. The issued
   challenge's certificate summary reports both rows' relaxation sets so the signer sees the
   verdict.
8. **Exploration unchanged.** Signed relaxations on `research promote` and `paper promote` behave
   exactly as before: same challenge bytes, verdicts, tokens and stage moves. A relaxed human
   research gate can still anchor a paper deployment, and a relaxed certificate can still move
   `paper -> forward_tested` and refresh at `forward_tested`.
9. **Protected and green.** The vocabulary module, the predicate and the schema change are
   CODEOWNERS-protected and in the integrity-critical set (`tests/test_repo_hygiene.py:237`). Pinned
   modules at exactly their size (`promote_run.py` 348, `promotion.py` 542, `store/gate.py` 594,
   `forward_gates.py` 392) are carved, not grown. Every predicate branch is mutation-checked. The
   full root gate passes.
10. **No raw way in (#682).** `registry transition --to forward_tested` and `--to candidate` are
   refused for every actor, human included, with a stable code, so `paper promote` and `research
   promote` are the only ways into those stages (owner decision 2026-10-04 on #682). Back-steps and
   other raw edges are unchanged; tests cover both actors and both edges.

## Tasks / subtasks

- [ ] Contract and readiness: field-level companion with DDL, triggers, vocabulary table, migration
      rules and an entry-point inventory of every row writer and every go-live path (AC1–AC6).
- [ ] Vocabulary module and pure functions, test-first, including the CLI-option table test
      (AC1–AC2).
- [ ] Schema: `relaxations_json` on both tables, insert-requires and no-update triggers, the
      classification migration, `SCHEMA_VERSION` bump (`algua/registry/db/constants.py:36`)
      (AC3–AC4).
- [ ] Thread inputs: `promote_task` -> `run_gate` -> both research writers; `paper promote` ->
      `run_forward_gate` -> both forward writers. Carve to keep pins (AC3, AC9).
- [ ] Qualification predicate and its single call site for issuance and completion; error code and
      envelope docs (AC5–AC7).
- [ ] Regression proof that exploration is unchanged (AC8).
- [ ] Rehearse the migration on a copy of the production registry; record the result (AC4).
- [ ] Refuse the raw forward_tested and candidate edges for every actor (AC10).
- [ ] Protection, mutation checks, full gate, independent review (AC9).

## Dev notes

### Seams

| Seam | Today | Change |
|---|---|---|
| `algua/registry/promote_run.py:195-224` | Builds the signed run context and calls `promotion_preflight` | Compute the research set from the same inputs; pass it down |
| `algua/registry/promotion.py:266`, `:416-452` | `run_gate` builds `gate_row` | Carry `relaxations_json` |
| `algua/registry/store/gate.py:51-101`, `:289-410` | Two INSERTs | Write the column |
| `algua/registry/store/forward_gate.py:18-129` | One INSERT behind both record paths | Write the column |
| `algua/registry/forward_promotion.py:47-70` | Guard computes the relaxed list for agents only | Extract a pure `forward_relaxations`; guard reuses it |
| `algua/registry/forward_promotion.py:221-267` | `gate_row` and the two record paths | Carry `relaxations_json` |
| `algua/registry/live_certificate.py:83-99` | Selects the certificate row | No change: the summary already returns the row `id` (`:186`), which the predicate reads and requires to be the deployment's newest forward row (contract §5 step 4) |
| `algua/registry/transitions.py:116-157`, `algua/cli/registry_cmd.py:216-217` | Certificate check at completion and issuance | Both call one `verify_live_qualification` that runs the verifier, then the predicate |
| `algua/registry/transitions.py:36-113` | Raw forward edges consume agent tokens; humans pass freely | Refuse both forward edges for every actor; delete the unreachable token branches and helpers (contract §6) |

The companion (§4–§6, §9) is authoritative where it is more specific than this table: it adds the
carves (`capture_gate_fail_experience` to `gate_fail_capture.py`, `guard_agent_relaxations` to
`relaxations.py`), deletes the two `find_consumable_*` finders whose only callers §6 removes, and
lowers every touched pin to its new size. `registry_cmd.py` has two lines of headroom under its 446
pin; replace the issuance call rather than adding one. The legacy-cohort branch
(`live_certificate.py:91-95`) stops authorizing go-live because the predicate reads a strategy
without an active deployment as `unrecorded`; the branch itself goes with the legacy tick paths
(the Story 1.4 follow-up).

### Contract decisions beyond the acceptance-criteria text

Recorded in the decision log; the owner may revisit the first two like the `--demo` call.

- Every human research row records `agent_walls_waived`: a human run skips the agent-only walls
  (reproducible source, cost floor, feature lookback, gated universe, seeded family path) with no
  flag. A human who wants a live-eligible research gate runs `research promote --actor agent`.
- `--demo` records `demo_data`; `--new-family` records `new_family` only for a human (an agent's is
  ignored by the code).
- A non-finite threshold is a relaxation; the forward agent guard reuses the same function, so an
  agent's `NaN` forward threshold is now refused at preflight (finite inputs: byte-identical).
- Human forward rows written before v49 stay `unrecorded` (their confidence is recorded nowhere);
  agent forward rows need the LCB check to be classified.
- The raw-edge refusal is a plain `TransitionError` (`wrong_stage`), like the intake refusal; the
  `paper -> candidate` back-step stays.
- `cli/registry_cmd.py` and the new modules become CODEOWNERS-protected and integrity-critical.

### Traps

- Put the predicate after the certificate verifier, not inside it: `transition_strategy` accepts an
  injected verifier, and tests already inject fakes. The predicate judges the deployment's newest
  forward row and refuses unless the verifier returned that id, so an injected verifier cannot steer
  it to an older clean certificate (readiness m1).
- `allow_holdout_reuse` is a relaxation when the flag is given, even if no overlap existed. The rule
  is "a relaxation flag was signed", not "a relaxation changed the outcome".
- Advisory thresholds count. An agent may pass a looser `--min-holdout-sharpe` today; that row is
  then recorded as relaxed and its deployment is not live-eligible. This is the owner's "every gate
  at its protected default", not a new agent restriction.
- Run the migration's UPDATE before creating the no-update trigger. SQLite cannot add `NOT NULL` to
  an existing table, so enforce "required on insert" with a trigger, as the v47/v48 contracts do.
- Keep the v49 work out of `db/migrate.py`: it is 291 lines and unpinned, and the inline block
  measured 303. `db/relaxations.py::apply_relaxation_schema` holds the ALTERs, the classification
  and the triggers; `migrate()` gains one import and one call (readiness M1).
- The recorded triggers open with an append-only check (`INSERT OR REPLACE` could otherwise rewrite
  a relaxed set), and the canonical test refuses any backslash (readiness m2). Copy contract §2
  verbatim.
- Deploy v49 with the contract §2 roll-forward: stop the merge-back drain and research timers and
  let running units exit, migrate once, restart. A v48 promote still running when v49 migrates
  would burn its holdout and then have its row refused (readiness m3).
- Story 2.2 edits the same transition and go-live code; whichever merges second rebases and keeps
  contract §9's ordering (readiness m7).
- Do not derive at runtime for new rows. The derivation exists only inside the one-time migration.

### Test matrix

Vocabulary per flag and per threshold direction; tightening is not relaxing; every writer records;
insert without the column and any update are refused; migration classifies agent, human and
unparseable rows and is idempotent; predicate for relaxed research row, relaxed certificate,
unrecorded row, no deployment and both clean; a certificate id that is not the deployment's newest
forward row is refused; `INSERT OR REPLACE` over an existing row and a JSON-escaped token are
refused; issuance writes no challenge on refusal; completion refuses before `ssh-keygen` runs and
before consumption; a valid human signature is still refused; signed relaxed research and paper
promotions behave as before.

## Owner decisions

None open for this story. The policy was decided on 2026-10-04 (#624). The `--demo` classification
in AC1 is a conservative design call the contract records; the owner may revisit it.

## Verification

```bash
uv run pytest -q
uv run ruff check .
uv run mypy algua
uv run lint-imports
```

## References

- [#624 owner decisions](https://github.com/Lior-Nis/algua/issues/624); [epics.md](../epics.md)
  FR14, FR8
- `docs/PRD.md` §§19, 21; `CLAUDE.md` (authenticated `--actor human`, live wall)
- [Story 1.2](1-2-record-working-tree-deployments.md) (deployment-bound certificates),
  [Story 1.3d](1-3d-bind-operational-evidence-and-qualification.md) (frozen promotion chokepoint)
- Todoist:
  [Bind signed live authorization to exact deployment](https://app.todoist.com/app/task/bind-signed-live-authorization-to-exact-deployment-6hfCrg4FrJjHwgPG)

## Dev Agent Record

### Agent Model Used

### Completion Notes

### File List
