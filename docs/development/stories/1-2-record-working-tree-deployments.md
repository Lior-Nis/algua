---
baseline_commit: 118558c3b2ccd36827cffd988c848a43aa88a614
---

# Story 1.2: Record working-tree deployments and evaluate one explicit epoch

Status: ready-for-dev

Prepared: 2026-09-24. Baseline: Story 1.1 merge `118558c` (PR #667).
Epic: 1. Requirements: FR5 and the record-only portion of FR4/FR12.
Constraints: NFR1–NFR6, NFR8. Upstream: #661, artifact-freeze design slice 3.

## Story

As Algua's operator,
I want each newly admitted paper strategy and its completed ticks bound to an explicit deployment
epoch,
so that forward evidence cannot be back-credited, mixed across deployments or silently reused after
redeployment while the repository continues to evolve.

## Scope and authority

This story records and enforces deployment identity before executable artifacts are frozen. A new
paper admission atomically records a canonical working-tree deployment descriptor, opens one
append-only strategy deployment epoch and performs the existing allocation/stage transition. Paper
ticks verify and stamp that deployment. Forward qualification evaluates exactly that epoch.

Execution still loads the current in-process working tree. The descriptor is evidence about that
tree and environment; it is **not** a recoverable immutable executable artifact. Repository drift
therefore stops a deployment-aware tick until the later frozen-execution story removes that
temporary limitation. Do not describe this slice as satisfying FR4's recoverability requirement.

No exported tree/wheel, planner subprocess or IPC, artifact store, retention/garbage collection,
existing-fleet backfill, migration/requalification command, live signed-payload redesign, hourly
execution, hash-policy narrowing, capital change or live activation. A fixed cohort of tenants that
already exists when the schema migrates remains temporarily readable/runnable, but cannot acquire
new forward evidence without explicit deployment records. Missing deployment state alone must never
create legacy status, and no post-migration admission can enter the cohort. Do not infer or backfill
deployment IDs.

This story tightens one protected wall: forward certificates for a deployment-aware strategy must
be selected by deployment ID as well as artifact identity. Otherwise redeploying identical hashes
could reuse an earlier epoch's certificate. This tightening requires explicit protected-code review;
it does not grant new live authority or replace the later deployment-bound signing ceremony.

## Acceptance criteria

1. **Atomic paper adoption.** Given a candidate with the exact qualifying research gate whose
   code/config/dependency identity matches the clean current working tree, when paper intake
   succeeds, then the artifact descriptor, deployment activation, allocation, candidate-to-paper
   CAS and transition audit commit in one top-level `BEGIN IMMEDIATE` transaction. A fault or stale
   predicate at any step rolls all five back.
2. **Canonical complete descriptor.** Given an adoption request, when its manifest is constructed,
   then canonical JSON records exact research identity, authored behavior-affecting
   `StrategyConfig`, the gate-bound `universe_name`/binding (not its evolving membership list),
   dependency digest, planner protocol, Python implementation and version, ABI/cache tag, platform
   tag, `source_kind="working_tree"`, and exact git commit/source reference. `manifest_digest` is
   SHA-256 over those canonical bytes. One centralized verifier requires every tracked repository
   file to equal recorded `HEAD`, rejects non-generated untracked files beneath `algua/`, and records
   and verifies the path plus content digest of every config-referenced model/strategy asset outside
   the tracked tree. Only cache artifacts mechanically derived from tracked source may be excluded,
   through one explicit constant allowlist. Admission refuses a missing dependency digest, dirty or
   incomplete read set, missing/mismatched gate, unsupported planner protocol or incomplete
   environment fingerprint before changing registry state. Point-in-time universe membership remains
   a per-tick input/provenance value resolved as of that tick.
3. **Immutable records and epochs.** Given descriptor/deployment rows, when writes are attempted,
   then descriptor insert is idempotent only for a byte-identical manifest with the same digest;
   digest collision or mutation fails closed, and no `REPLACE` path exists. The database permits at
   most one active deployment per strategy. Retirement is one-way. Redeploying even the identical
   descriptor creates a new `strategy_deployment.id` and never reactivates an old epoch, but in this
   story activation is reachable only through qualified `candidate -> paper` intake. There is no
   general replace/redeploy command.
4. **Gate-bound activation.** Given a deployment activation, when committed, then it records
   `strategy_id`, `artifact_id`, exact `research_gate_id`, `activated_at`, nullable `retired_at` and
   nullable `superseded_by`. Under the write lock, the gate is rechecked as the newest passing row
   for that strategy and exact identity that justified its current candidate state and has never
   anchored another deployment: `actor='agent'`
   requires `consumed=1`; an authenticated human promotion is eligible with `actor='human'` and its
   audit-only `consumed=0`, including any signed PIT override already recorded by that gate. No other
   actor/consumption combination is eligible. A unique constraint (or equivalent under-lock hard
   predicate) permanently limits each `research_gate_id` to one committed deployment. A rolled-back
   intake leaves no reference and may retry; a committed deployment burns that gate as an activation
   anchor. The strategy and intake predicates are also rechecked.
5. **Verify before paper effects.** Given a deployment-aware paper tenant, when a tick begins, then
   the supervisor resolves its one active deployment and recomputes/verifies working-tree source,
   environment, protocol, resolved configuration and bound gate universe before provider, venue,
   cancellation, submission or hook effects. Missing, corrupt, retired or drifted deployment state
   is a per-strategy setup failure and is never stamped as the old deployment.
6. **Tick provenance.** Given a deployment-aware paper tick for which the existing execution path
   writes a snapshot, when that snapshot is persisted, then `deployment_id` identifies that
   strategy's still-active deployment and the
   deployment artifact identity equals the tick's three hashes. The guarded insert and retirement
   serialization prevent a tick from claiming an already retired deployment. Every public path
   that can retire a paper deployment acquires the same operator lock used by paper execution before
   changing stage/deployment state; contention tests prove retirement cannot interleave with broker
   effects and snapshot persistence. Pre-story rows and ticks for the fixed migration-time legacy
   cohort retain `NULL`; absence of a deployment does not enroll another tenant. A wrong-strategy,
   retired or identity-mismatched deployment fails closed. Empty/flat warm-up behavior is unchanged
   and does not gain a synthetic snapshot.
7. **One explicit evidence epoch.** Given an active deployment, when the forward gate assembles
   evidence, then it requires that deployment and selects only paper ticks carrying its exact ID.
   It does not infer an epoch from contiguous hashes. A new ID starts with zero return observations
   even for identical hashes; no pre-activation or legacy-null tick is credited. Inadmissible ticks
   carrying the ID remain inside that deployment's hygiene/integrity universe. The evaluation row
   records the same `deployment_id`. Existing identity-scoped repeated-look protection is not reset
   merely by opening a new deployment.
8. **Stable operational universe binding.** Given an active deployment, when its strategy input
   universe is resolved, then the universe name comes from that deployment's exact
   `research_gate_id`; a newer ambient passing gate cannot change the binding. Membership is still
   resolved point-in-time as of the tick and recorded with its normal input provenance; it is not
   frozen into the descriptor.
9. **Explicit retirement; qualified re-admission only.** Given a deployment leaves its
   evidence-bearing life, when it is retired, then retirement is operator-lock serialized,
   CAS-guarded and append-preserving. `paper -> candidate` and `-> retired` retire the deployment;
   pause/dormant and paper/forward/live lane movement retain it unless the reviewed lifecycle matrix
   explicitly abandons it. Opening a successor requires the strategy to return through the normal
   qualification path to `candidate` and then the atomic intake in AC1 with a newly applicable exact
   research gate; this story exposes no arbitrary active-stage replacement path.
10. **Certificate cannot cross epochs.** Given two deployments with identical artifact hashes, when
    the current deployment seeks live eligibility, then a forward certificate earned by the other
    deployment is rejected. Deployment-aware certificate lookup requires both exact identity and
    exact `deployment_id`; it can never fall back to an identity-only row. A strategy in the fixed
    migration-time legacy cohort with no deployment temporarily retains today's identity-only
    verifier; a no-deployment strategy outside that cohort is denied. The signed live authorization
    payload itself remains unchanged until its dedicated story.
11. **No behavioral or authority drift.** Given existing paper/live execution and gates, when this
    slice lands, then the `t -> t+1` rule, risk/effect ordering, allocation/count caps, authenticated
    human boundary, live approval rules and CLI JSON contract remain unchanged. The full quality
    gate passes without weaker assertions, type suppressions, import exemptions or raised size pins.

## Tasks / Subtasks

- [ ] Characterize the current adoption/evidence/certificate paths before schema edits (AC 1, 5–11).
  - [ ] Add red tests proving same-hash redeployment currently risks old-epoch evidence/certificate
    reuse; preserve current legacy behavior explicitly where migration is deferred.
  - [ ] Record paper setup/effect traces so deployment drift is proven to fail before provider,
    venue, cancellation, submission and downstream hooks.
  - [ ] Record the intended transition-to-retirement matrix for protected review before wiring it.
- [ ] Add immutable deployment schema and typed records (AC 2–4, 6–7).
  - [ ] Add `algua/registry/db/deployment.py` with `deployment_artifacts` and
    `strategy_deployments`; include foreign keys, lookup indexes, append-preserving guards and a
    partial unique index for one active deployment per strategy, plus permanent uniqueness of
    `research_gate_id` across deployments.
  - [ ] Add nullable `deployment_id` foreign keys to both fresh and migrated `tick_snapshots` and
    `forward_gate_evaluations`; bump the schema marker and prove idempotent migration. Leave legacy
    values null and atomically capture the fixed pre-migration tenant cohort through an explicit,
    auditable marker that later admissions cannot obtain.
  - [ ] Add small immutable manifest/record values and canonical digest/environment/source capture.
    Centralize the complete working-tree read-set rule: tracked tree equals recorded HEAD, no
    non-generated untracked file beneath `algua/`, and every external config-referenced executable/
    model asset path and digest is part of the manifest. Keep git/filesystem/environment I/O outside
    SQL transactions.
  - [ ] Make artifact resolution insert-or-verify, never replace; detect byte disagreement under an
    existing digest as corruption.
- [ ] Make new paper admission open one atomic deployment epoch (AC 1–4, 8–9).
  - [ ] Resolve the exact qualified research gate and gate-bound universe; never use a
    merely latest passing row as the deployment anchor, and reject a gate already referenced by any
    committed deployment. Cover both agent-consumed and human audit-only promotion rows.
  - [ ] Extend or carve the intake store so artifact insert, activation, allocation and stage CAS
    share its existing `BEGIN IMMEDIATE`, with fault-injection rollback tests after each write.
  - [ ] Serialize retirement and enforce the reviewed transition matrix. Inventory every public
    paper-deployment retirement path and make it acquire the paper operator lock before the stage/
    deployment transaction. Do not hold the SQLite write lock across git, provider, broker or other
    external I/O, and do not add a general replacement command.
  - [ ] Prove concurrent admissions/activations produce one valid result or a domain failure, not
    partial state or a leaked SQLite lock error.
- [ ] Bind operational paper ticks to the active deployment (AC 5–6, 8, 11).
  - [ ] Resolve and verify the deployment during setup before current side effects. Keep the legacy
    path explicit and temporary only for the migration-time cohort; missing records are not proof of
    legacy status.
  - [ ] Resolve the operational universe through `deployment.research_gate_id` for deployment-aware
    ticks, while retaining the current warning/fallback only for truly legacy rows.
  - [ ] Thread `deployment_id` into the tick writer and guarded persistence. If module ratchets require
    a carve, preserve compatibility imports rather than growing pinned modules.
  - [ ] Test drift, missing deployment, wrong strategy, operator-lock retirement contention and
    identity mismatch without weakening existing planner/lane/risk parity assertions.
- [ ] Replace inferred epochs with explicit deployment evidence (AC 7, 10).
  - [ ] Require one active deployment in forward promotion and select its exact tick rows. Remove the
    contiguous-hash epoch inference; retain identity checks as independent corruption detection.
  - [ ] Anchor return, reconciliation, defect, breaker, activity, concurrency and optional-stopping
    windows to the same deployment epoch. A bad tick may be excluded as a return observation but not
    disappear from its integrity window.
  - [ ] Store `deployment_id` on pass and fail evaluations and make same-artifact new deployments
    begin with no return observations from their predecessor. Preserve at least the current
    identity-scoped bounded repeated-look count so redeployment is not a tax-reset escape hatch.
  - [ ] Tighten protected certificate selection so an identity-matching row from another deployment
    cannot authorize the current one. Obtain explicit protected-code review for this subtask.
- [ ] Verify and document the temporary boundary (AC 2, 5, 9–11).
  - [ ] Add focused schema, deployment store, tick provenance, universe binding, forward promotion,
    certificate, CLI paper, concurrency and parity tests listed below.
  - [ ] Run the focused suites and then the full sequential root gate. Record exact results.
  - [ ] Obtain independent review of schema safety, epoch selection, transaction rollback and the
    protected certificate tightening before marking implemented/reviewed.
  - [ ] Update architecture/reconciliation docs to say execution still uses the mutable working tree,
    existing tenants remain unmigrated and deployment materialization is the next slice.

## Dev Notes

### Required record semantics

Use repository naming conventions (`deployment_artifacts`, `strategy_deployments`) while retaining
the design's conceptual names. The minimum fresh-schema shape is:

- `deployment_artifacts`: primary key, unique `manifest_digest`, three artifact hashes with non-null
  dependency hash, canonical authored-config JSON plus the gate-bound universe name (not membership),
  environment digest and its interpreter/ABI/platform components, planner protocol version, source
  kind/ref, creation timestamp.
- `strategy_deployments`: primary key, strategy/artifact/research-gate foreign keys, activation and
  retirement timestamps, optional self-referencing `superseded_by`; one active row per strategy and
  one permanent deployment reference per research gate.
- `tick_snapshots.deployment_id` and `forward_gate_evaluations.deployment_id`: nullable only for
  honest legacy compatibility. New deployment-aware writes must provide them.
- An explicit migration-time legacy marker: populated only for the pre-existing operational cohort
  in the schema migration. Do not derive exemption from a nullable deployment field or current stage.

SQLite cannot add all constraints to existing tables through `ALTER TABLE`. Enforce new-write
discipline in guarded repository methods and test both fresh DDL and migration. Append-preserving
triggers may permit only the reviewed null-to-value retirement/supersession update; no general
artifact mutation. `PRAGMA user_version` remains a schema marker, not a migration cursor.

`manifest_digest` addresses the canonical descriptor, not executable bytes. `source_ref` must prove
the inspected clean commit; it must not imply that the current checkout will remain there. Define
cleanliness once and reuse it at activation and tick setup: all tracked paths equal recorded HEAD;
non-generated untracked files under `algua/` are forbidden even if gitignored; cache exclusions are
an explicit constant, not caller discretion; and every external model/strategy asset referenced by
the resolved configuration has its path and byte digest in the manifest. Do not silently ignore a
dynamic import or model path. A future frozen-artifact record may extend this descriptor or create a
new descriptor, but this story must not populate a fictitious artifact path/digest.

### Transaction and ordering traps

Compute the manifest and perform git/environment inspection before acquiring SQLite's write lock,
then recheck every database predicate inside `BEGIN IMMEDIATE`. Do not put broker, provider, git or
filesystem work under that lock. The exact manifest bytes supplied to the transaction are immutable
input; re-verify source drift again at tick setup.

The current tick persists its snapshot after broker effects. Therefore every public retirement path
must acquire the same paper operator lock before changing stage/deployment state, while the snapshot
insert also rechecks active deployment state transactionally. This is an enforceable requirement,
not an informal operator convention. The operator lock spans the existing tick; the SQLite write
lock does not span broker/provider calls. This story does not redesign execution into a distributed
transaction. Prove lock contention cannot retire the epoch between broker effects and persistence.

Artifact insert, activation, allocation and transition must be one commit. Do not use
`INSERT OR REPLACE`; with foreign keys it can delete/reinsert authority records. On failure, a retry
may reuse a verified identical descriptor but must open a fresh deployment only after the complete
intake succeeds.

### Evidence and certificate traps

Delete `_epoch_start_id` only after every deployment-aware evidence query is explicitly scoped. Keep
the three-hash check: `deployment_id` proves the epoch, while the hashes detect corrupt/mis-stamped
rows. The integrity universe is all rows of that deployment from its start, including rows that fail
return-admissibility filters. Never filter out bad rows before reconciliation/defect accounting.

All temporal windows that currently anchor on the first admissible tick must be reviewed together:
returns, reconciliation/defects, kill-switch events, broker activities and concurrent breadth.
Repeated-look accounting is different: retain at least today's bounded strategy+identity scope
across deployment IDs, or make it stricter. Opening a same-artifact epoch must not erase the current
optional-stopping tax.

The current live certificate lookup is identity-based. Adding explicit same-hash epochs without
changing that lookup creates an authorization hole. Tighten lookup/verification for
deployment-aware strategies now, but do not change challenge contents, signer namespaces, approval
hash algorithms or capital authority. A later story changes the ceremony to sign deployment ID plus
manifest digest and promote the exact paper artifact without rebuilding.

### Current files and likely change surfaces

| File | Expected treatment |
|---|---|
| `algua/registry/db/deployment.py` | New deployment DDL context |
| `algua/registry/db/schema.py`, `constants.py`, `migrate.py` | Assemble schema, bump marker, add legacy nullable columns |
| `algua/registry/deployment.py` | New canonical manifest/source/environment logic; no SQL |
| `algua/registry/store/deployment.py` | New guarded artifact/deployment operations and atomic intake support |
| `algua/registry/store/crud.py`, `registry/intake.py` | Carve/extend candidate intake without growing pinned code |
| `algua/execution/order_state.py` | Thread provenance; carve tick snapshot operations if required by its size pin |
| `algua/cli/paper_cmd.py` | Minimal setup verification/stamping; extract orchestration rather than exceed its pin |
| `algua/registry/universe_binding.py` | Exact research-gate resolution for deployment-aware paths |
| `algua/registry/forward_evidence.py`, `forward_promotion.py` | Explicit epoch selection and evaluation stamping |
| `algua/registry/store/forward_gate.py`, `live_certificate.py` | Persist and require deployment-bound certificates |
| `tests/test_module_size_ratchet.py` | Ratchet down after honest carves; never raise pins |

Do not grow already pinned `repository.py`, `store/crud.py`, `order_state.py`, `paper_cmd.py` or
`forward_evidence.py` merely for convenience. Prefer focused modules plus compatibility re-exports
where callers already import a public symbol.

### Test matrix

- `tests/test_deployments.py`: canonical digest, complete config/environment, tracked edits, staged
  edits, untracked/ignored importable source, external asset drift, ref and gate/identity mismatch,
  descriptor idempotency/collision, one-active and one-gate constraints, one-way retirement,
  same-artifact requalification/new epoch and rollback/retry behavior.
- Registry DB/migration tests: fresh foreign keys/indexes/guards, legacy nullable columns, an exact
  fixed legacy cohort, no post-migration enrollment, idempotent schema-marker migration and no tick
  or evaluation backfill.
- Concurrency tests: racing intake/activation and tick-versus-retirement serialization.
- Tick/order-state tests: deployment round-trip and guarded wrong-strategy/retired/hash-mismatch
  failures while legacy null rows survive.
- Forward promotion tests: deployment A never credits B, even with equal hashes; bad B rows remain
  visible to hygiene; evaluation row records B; no active deployment and legacy null fail closed;
  same-identity redeployment does not reset bounded repeated-look accounting.
- Universe binding tests: the deployment's gate name beats a newer ambient gate while point-in-time
  membership can evolve normally between ticks.
- CLI/planner/lane parity tests: setup drift precedes all effects; valid admitted ticks stamp the ID;
  legacy compatibility is explicit and temporary.
- Forward certificate tests: an earlier deployment's certificate cannot authorize the current one;
  exact active deployment, fixed-cohort legacy and unmarked no-deployment branches are distinct.

### Verification

No new dependency or external API is required. Use the repository's Python/SQLite/git tooling and
locked environment; no web research or package upgrade is justified for this slice.

Run focused tests chosen from the final touched set, including at least:

```bash
uv run pytest -q tests/test_deployments.py tests/test_registry_db.py \
  tests/test_order_state.py tests/test_forward_promotion.py tests/test_forward_certificate.py \
  tests/test_universe_binding.py tests/test_cli_paper.py tests/test_concurrency.py \
  tests/test_planner_parity.py tests/test_lane_parity.py tests/test_module_size_ratchet.py
```

Then run the full gate sequentially (pytest creates temporary source fixtures):

```bash
uv run pytest -q
uv run ruff check .
uv run mypy algua
uv run lint-imports
```

### Deferred decisions and recovery boundary

The following remain separate stories/owner decisions: executable representation and transport,
artifact storage/location/retention/GC, missing/corrupt frozen-artifact supervisor policy, migration
and honest requalification of existing tenants, deployment-bound signed payload shape, planner
subprocess isolation and live capital/account policy. No deployment or timer is restarted here.

Rollback before frozen execution is code rollback plus retirement of any newly opened epoch; never
delete deployment/evidence rows or transplant old tick IDs. A failed adoption is transactionally
absent. A successfully opened epoch is permanent history even if immediately retired.

### References

- [Canonical PRD](../../PRD.md), §§5–7, 9–10, 15–19, 24–26.
- [Current architecture](../../architecture.md) and [reconciliation](../../vision-reconciliation.md).
- [Approved epic outcomes and requirements](../epics.md), especially FR4–FR5 and NFR1–NFR6.
- [Artifact-freeze design](../../superpowers/specs/2026-09-22-artifact-freeze-design.md), records,
  strict fix policy, back-crediting analysis and decomposition slice 3.
- [Story 1.1](1-1-extract-in-process-decision-planner.md), the versioned in-process planner seam.
- [Issue #661](https://github.com/Lior-Nis/algua/issues/661), parent implementation direction.
- Repository `AGENTS.md`, `CLAUDE.md`, `docs/agent/operating.md` and frozen
  `docs/contracts/bar-schema.md` remain binding during implementation.

## Dev Agent Record

### Agent Model Used

To be completed by the implementing agent.

### Debug Log References

To be completed by the implementing agent.

### Completion Notes List

To be completed by the implementing agent.

### File List

To be completed by the implementing agent.
