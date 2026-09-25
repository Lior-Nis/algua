---
baseline_commit: 34ab0d77c61af9ac20637c59545874720bd15fc9
---

# Story 1.3: Materialize and execute frozen planner artifacts

Status: decomposed

Prepared: 2026-09-24. Baseline: Story 1.2 merge `34ab0d7` (PR #669).
Epic: 1. Requirements: FR4, FR6, FR9–FR10 and the paper portion of FR1/FR12.
Constraints: NFR1–NFR8. Upstream: #661, artifact-freeze design slices 4–5.

## Decomposition status

The 2026-09-25 implementation-readiness review found that this complete outcome is too large for one
implementation and review cycle. This file remains the approved parent requirement record; it is not
an executable development story. The approved children are:

1. [Story 1.3a](1-3a-complete-two-phase-planner-boundary-in-process.md) — complete the two-phase
   behavior boundary in-process;
2. [Story 1.3b](1-3b-materialize-and-verify-recoverable-planner-artifacts.md) — materialize and
   verify recoverable bundle/environment content;
3. [Story 1.3c](1-3c-execute-frozen-planners-in-paper.md) — dispatch frozen planners in paper;
4. [Story 1.3d](1-3d-bind-operational-evidence-and-qualification.md) — bind evidence and permit
   qualification from verified immutable content.

Parent acceptance coverage maps as follows:

| Parent acceptance criteria | Child ownership |
|---|---|
| AC6 plus in-process portions of AC9–AC10 | Story 1.3a |
| AC1–AC2, AC4–AC5, AC12–AC13 materialization/retention portions | Story 1.3b |
| AC3 future model assets | Superseded for this cycle: 1.3b reserves the manifest shape but rejects non-empty assets pending a separately reviewed tradable model lane |
| AC6–AC11 transport/dispatch/failure portions; AC14 compatibility dispatch | Story 1.3c |
| AC11–AC14 evidence, restart and qualification portions | Story 1.3d |

Shared parity, authority, compatibility and full-gate constraints apply to every child. The parent is
complete only after all four children are done; approval of this decomposition does not authorize
implementation or deployment.

## Story

As Algua's operator,
I want each newly admitted paper strategy's planner to execute from recoverable immutable content,
so that it can accumulate trustworthy evidence while the repository and current supervisor continue
to evolve.

## Scope and authority

This story replaces the temporary working-tree per-strategy decision path for **new** paper
admissions with a content-addressed source bundle, copied model assets and a shared environment
selected by the full environment fingerprint. One short-lived planner subprocess per strategy
receives exact supervisor-prepared inputs through Parquet plus a versioned request and returns one
strictly validated JSON result. The frozen boundary includes the behavior-affecting timing, warm-up, mark-freshness and
per-strategy risk semantics currently surrounding Story 1.1's pure weight/intent computation; merely
putting `planner.plan()` in a child would leave credited behavior mutable and does not satisfy this
story. The current supervisor remains the sole owner of the registry, migrations, provider and
broker clients, credentials, reconciliation, account state, book risk, cancellation, submission,
buying-power reservation, hooks, audit and tick persistence.

The subprocess boundary is reproducibility and capability hygiene, not a malicious-code sandbox.
The accepted runtime-sandbox residual in `AGENTS.md` remains. Read-only permissions and a scrubbed
environment protect against accidental drift and inherited authority; they do not stop hostile code
running as the same OS user from using the filesystem or network. Do not claim otherwise.

Existing `source_kind="working_tree"` deployments and the fixed migration-time legacy cohort are
not converted, repointed or backfilled. They retain their explicit temporary behavior until Story
1.4's controlled migration. A deployment record is immutable, so only future qualified intake may
mint `source_kind="frozen"`. There is no automatic rebuild from current `HEAD`, arbitrary redeploy
command, live-authorization change, capital change, hash-policy narrowing, hourly execution,
artifact garbage collection, microservice, generic plugin framework or dependency upgrade here.

The approved failure policy is per strategy. Missing/corrupt/unsupported planner content fails that
tenant before its cancel or order effects; another independently valid `run-all` tenant may proceed.
SQLite faults, global halt, reconciliation and other account-wide authority/risk failures remain
systemic and abort the lane. A single-strategy command exits nonzero. `run-all` returns nonzero only
under its existing breach/systemic semantics; isolated setup failures remain explicit entries in
its JSON envelope.

## Acceptance criteria

1. **Deterministic executable bundle.** Given an eligible candidate and its verified Story 1.2
   descriptor, when frozen materialization runs, then it exports the exact clean recorded Git tree
   needed by the planner from `source_ref` (not a second read of mutable working-tree source), adds
   canonical resolved configuration and any already-verified model bytes, and computes a full
   SHA-256 tree digest over normalized relative path, file kind/mode and exact bytes. Directory
   enumeration order, timestamps and host paths do not affect the digest; any source, asset,
   configuration, planner-protocol or environment-identity change does. The existing
   code/config/dependency hashes remain gate identities and are not substituted for this bundle
   checksum or narrowed.
2. **Safe atomic publication.** Given materialization, when content is staged, then it uses a unique
   private directory on the artifact store's filesystem, accepts regular files/directories only,
   rejects symlinks, hardlinks, devices/FIFOs, absolute/parent-traversing paths and normalized or
   case-colliding duplicates, fsyncs before publication where supported, and atomically publishes
   without overwrite beneath a digest-derived path in `Settings.data_dir`. If that digest already
   exists, byte-for-byte/canonical inventory verification is required. Concurrent builders yield
   one identical immutable artifact or a domain failure—never mixed content. Published files are
   non-writable and verified before dispatch. No `.env`, registry database, credentials, trust
   anchor, mutable worktree path or operational configuration is copied.
3. **Model assets are frozen from verified bytes (superseded for this cycle).** Given a strategy with a future supported model
   lane, when its artifact is prepared, then model content is copied from the atomically resolved
   `ModelHandle.artifact_bytes`, not reopened through an external path after validation. Its stable
   relative location, full digest and pinned model metadata are in the canonical artifact manifest.
   Today's paper tradability restrictions remain unchanged; this criterion closes the materializer
   contract without enabling an unsupported model/sidecar lane. The approved decomposition narrows
   Story 1.3b to source-only artifacts: it reserves the canonical asset inventory but rejects
   non-empty assets until a separately reviewed tradable model lane exists.
4. **Frozen records are append-only.** Given successful materialization, when candidate intake
   commits, then the artifact row records `source_kind="frozen"`, bundle digest and stable relative
   locator plus the complete Story 1.2 descriptor, and the deployment/allocation/stage transition
   remains one atomic transaction. Materialization and environment preparation occur before the
   SQLite write transaction. A failure commits no artifact row, deployment, allocation or stage
   change; an already published but unreferenced content object may remain and is safely reusable.
   Existing artifact/deployment rows are never mutated or repointed.
5. **Complete shared environment identity.** Given deployments with the same dependency digest,
   Python implementation/full version, ABI/cache tag and platform tag, when their planner
   environment is resolved, then they may share one content-addressed environment only for that
   complete fingerprint. It is built atomically from the existing lock with no project/workspace or
   other local/editable install, no network resolution at tick time, and no `algua` package bound to
   the current checkout. Its interpreter and installed inventory are recorded, sealed read-only and
   verified against the expected fingerprint before dispatch. Any mismatch or corruption fails
   closed; different fingerprints never share an environment. No new package versions are selected.
6. **Two-phase, versioned, bound transport.** Given a frozen tick, when the planner is invoked, then
   Phase A receives raw fetched bars plus explicit `now`, timeframe, early held positions, bounds,
   gate-bound universe and deployment/request identity and returns only a validated early result or
   `snapshot_required`. Only after `snapshot_required` may the current supervisor acquire the
   sizing/NAV and venue-belief values needed by Phase B. Phase B receives those captured values,
   preserves `None` versus an empty belief, revalidates/recomputes Phase A under the same request,
   and returns a no-decision result, typed risk breach or decision. This keeps closed-bar selection,
   held/universe filtering, warm-up, freshness/calendar semantics, decision timestamp and
   per-strategy equity/drawdown/realized-gross/reconciliation decisions inside immutable code while
   retaining external data acquisition and all effects in the supervisor. A unique private
   invocation directory contains Parquet/Arrow bars with the exact canonical bar schema and a
   request binding request ID/digest, phase, strategy, deployment/artifact IDs,
   artifact/environment digests, snapshot ID, config identity and all captured values. The
   supervisor records/hashes exact input bytes. Both sides validate UTC index, names, column
   order/types, sort/uniqueness, permitted nulls and empty behavior; no pickle/object payload crosses
   the boundary. Index preservation is explicit rather than dependent on a library default.
7. **Short-lived planner with no inherited authority.** Given a request, when dispatch occurs, then
   the supervisor invokes the verified environment's absolute Python interpreter with fixed argv,
   `shell=False`, isolated/no-user-site flags where compatible, no stdin, `close_fds=True`, a fresh
   process group, a bounded timeout and bounded stdout/stderr. `cwd` and import roots resolve the
   frozen bundle, never the mutable checkout. The child receives an explicit minimal environment
   allowlist; it cannot inherit `ALGUA_*`, `ALPACA_*`, DB/data paths, `PYTHONPATH`, `PYTHONHOME`,
   active-venv/uv variables, cloud tokens, proxy credentials or dynamic-loader injection variables.
   Timeout kills and reaps the process group. Stderr is bounded diagnostic evidence, never an
   instruction or authorization source.
8. **Strict response contract.** Given a child result, when the supervisor parses it, then stdout is
   exactly one bounded UTF-8 JSON document using the supported protocol. The parser rejects duplicate
   keys, trailing data, excessive nesting/size, unknown or missing fields, booleans masquerading as
   numbers, NaN/Infinity, mismatched echoed request/strategy/deployment/artifact/timestamp values,
   duplicate or out-of-universe symbols, non-finite weights, invalid sides and inconsistent intents.
   Output order is deterministic. The supervisor reruns current decision-weight and intent
   validation as defense in depth before any cancellation or broker effect.
9. **Complete frozen-boundary parity and determinism.** Given every representative planner
   fixture—including warm-up,
   held-symbol valuation, dropped universe members, empty/no-op decisions, fractional rules,
   overlays, capacity and gross-utilization limits—when the same captured input is evaluated by the
   frozen two-phase subprocess and the baseline `run_tick` path, then early-return metadata, breach
   kind, decision timestamp, target weights, ordered intents, domain errors, effect trace and
   next-bar simulation behavior are identical. Cover stale/missing/non-finite marks, invalid equity,
   drawdown, reconciliation and gross breaches as well as normal decisions. Repeating a request in
   fresh child processes is deterministic. The `t -> t+1` rule and effect ordering do not change.
10. **Supervisor remains current and singular.** Given `paper trade-tick` or `paper run-all`, when a
    frozen tenant is processed, then one current supervisor performs the existing shared ingest,
    snapshot refresh, reconcile, account/buying-power and book-risk sequence and dispatches only
    per-strategy decision/risk work. It supplies captured broker/ledger/account values only after a
    Phase A request proves they are needed. The child cannot open/migrate the registry, construct a
    broker/provider, independently inspect account state, cancel, submit, reserve capital, invoke
    hooks, write audit/ticks or change authority. One supervisor dispatches N independent planner
    calls; no frozen checkout runs the account-wide command.
11. **Fail closed before tenant effects.** Given a missing/deleted/corrupt bundle, path violation,
    environment mismatch, unsupported protocol, timeout/signal/nonzero exit, excessive output,
    malformed/mismatched JSON or planner validation error, when setup/decision runs, then that
    strategy performs zero cancel, submission and downstream hook effects, writes no successful
    tick, and records a stable deployment-bound audit/failure code without leaking paths or secrets.
    `trade-tick` exits nonzero. In `run-all`, tenant A's such failure does not prevent independently
    valid tenant B from deciding/trading; a shared-environment failure affects tenants using that
    environment but cannot contaminate another environment. `KeyboardInterrupt`/`SystemExit`,
    SQLite failures, global halt and account-wide reconciliation/risk failures are not isolated.
12. **Artifact is the execution source.** Given an active frozen deployment, when unrelated tracked
    working-tree files or the current strategy implementation change after activation, then repeated
    frozen requests continue importing and executing the recorded bundle and produce the recorded
    behavior. Missing content is never rebuilt from mutable `HEAD`. A process restart resolves the
    same artifact from registry identity without a Git checkout. Rebuilding is allowed only from the
    exact recorded source/content and must reproduce the same digest.
13. **Retention and lifecycle.** Given retirement or supersession, when deployment state changes,
    then its bundle, environment and manifest remain addressable indefinitely in Phase 1. Startup
    and ticks never delete unknown/unreferenced content. Crash-safe cleanup may remove only private
    invocation/staging directories whose ownership and incompleteness are proven. Garbage collection
    requires a later retention design and is not introduced here.
14. **Compatibility and authority preservation.** Given a Story 1.2 working-tree deployment or fixed
    legacy tenant, when this story lands, then it is not silently converted and retains the explicitly
    documented compatibility path until Story 1.4. Frozen dispatch is selected only by a valid
    `source_kind="frozen"` record. Forward evidence and promotion verify the active frozen deployment
    and its content rather than recomputing identity from the ambient checkout, so unrelated repo
    evolution cannot stop its clock. Live activation/signing remains unchanged and does not treat a
    new frozen artifact as authorized; deployment-bound signing is Story 2 work. All CLI responses
    remain parseable JSON, protected walls and imports remain intact, module-size ratchets are not
    raised, and the full root quality gate passes.

## Tasks / Subtasks

- [ ] Characterize and lock transport/runtime parity before changing intake (AC 6, 8–11, 14).
  - [ ] Add red golden-master tests around complete `run_tick` behavior and the exact point before
    cancellation; cover Phase A early exits, snapshot acquisition ordering, per-strategy breaches,
    weights, ordered intents, errors and hook traces. Freezing `plan_decision` alone is insufficient.
  - [ ] Add strict bar-frame round-trip tests for canonical daily frames, empty/warm-up frames,
    held/dropped symbols, UTC timestamps, ordering and invalid schemas.
  - [ ] Specify stable failure codes and `trade-tick`/`run-all` JSON/exit behavior before wiring the
    subprocess; preserve current systemic-versus-tenant fault classification.
- [ ] Add the content-addressed frozen artifact store (AC 1–4, 12–13).
  - [ ] Add focused immutable values for bundle inventory/manifest and a materializer that exports
    the exact Git object tree and copies verified asset bytes into a same-filesystem staging dir.
  - [ ] Implement canonical tree hashing, path/type/collision validation, atomic no-replace publish,
    permission sealing, verification/lease behavior and crash/concurrency tests. Do not use a
    manifest-provided absolute path to resolve stored content.
  - [ ] Extend fresh and migrated deployment schema/store records additively for a frozen bundle
    digest and stable relative locator. Preserve immutable triggers and complete manifest-vs-column
    verification; bump the schema marker and prove idempotence.
  - [ ] Change future candidate intake to prepare frozen content before its existing transaction and
    commit only the frozen descriptor. Preserve an unreferenced published object on a later DB fault;
    do not mutate a working-tree epoch.
- [ ] Provision and verify shared planner environments (AC 5, 7, 12–13).
  - [ ] Key environments by the existing complete environment fingerprint and build from the locked
    dependency set with local/project/editable installs excluded. Provisioning may use existing
    cached resources before activation; dispatch must never resolve/download dependencies.
  - [ ] Record and verify interpreter facts and installed inventory, publish atomically, seal
    read-only and prove the mutable checkout is absent from imports/sys.path.
  - [ ] Test concurrent provision, interrupted publish, wrong interpreter/ABI/platform, inventory
    drift and environment sharing/separation.
- [ ] Implement the two-phase planner wire protocol and frozen child entry point (AC 6–9).
  - [ ] Define typed/versioned Phase A/Phase B request and response values with canonical JSON,
    duplicate-key and finite-number rejection, bounded sizes and exact echoed bindings.
  - [ ] Serialize bars with Arrow/Parquet using explicit index preservation and validate the frozen
    child's reconstructed frame before calling the extracted planner.
  - [ ] Move timeframe/closed-bar selection, held/universe filtering, warm-up, freshness/calendar,
    decision timestamp and per-strategy risk semantics behind the frozen protocol without moving
    account-wide acquisition or effects. Preserve the current early-return/snapshot ordering.
  - [ ] Load strategy/config/assets only from the bundle, return deterministic weights/intents, and
    prove the child imports no mutable/local `algua` installation.
- [ ] Add the current-supervisor dispatcher and paper integration (AC 7–12, 14).
  - [ ] Launch the absolute verified interpreter with fixed argv, explicit allowlisted environment,
    closed descriptors, process-group timeout/reap and bounded binary output capture.
  - [ ] Refactor `run_tick` around the two-phase seam: keep provider/broker/ledger acquisition,
    account/book risk and hooks in the supervisor while frozen code owns all per-strategy
    behavior-affecting semantics. Do not duplicate them in mutable and frozen branches.
  - [ ] Resolve frozen artifacts during paper preflight before provider/venue effects. Keep the
    explicit working-tree/legacy compatibility path and route only valid frozen records to dispatch.
  - [ ] Persist input/output digests and stable deployment-bound failure evidence needed for the
    order trace without granting child DB access. Preserve guarded tick/deployment persistence.
  - [ ] Make forward promotion resolve/verify the active frozen deployment identity and content;
    never compare a frozen epoch to ambient checkout hashes. Keep live authorization on its current
    wall until the deployment-bound signing story.
- [ ] Prove failure isolation, immutability and operational recovery (AC 2, 5, 7–14).
  - [ ] Fault-inject every staging/publish/DB boundary and assert only an unreferenced complete object
    or a complete active deployment is reachable—never a partial or active missing artifact.
  - [ ] Exercise bundle/env deletion and corruption, protocol mismatch, malformed/oversized output,
    timeout and forked-child cleanup before cancel/submit/hook effects.
  - [ ] Test `run-all` with one bad and one valid tenant, plus systemic DB/global-halt/reconcile cases;
    verify exact JSON and exit semantics and no secret/path leakage.
  - [ ] Prove restart without Git, mutable-worktree independence, no GC on retirement and frozen
    versus in-process parity across the current planner corpus.
  - [ ] Run focused suites and the full sequential root gate; obtain independent review of artifact
    identity/publication, environment/import isolation, IPC validation, per-tenant failure ordering
    and immutable schema changes before marking implemented/reviewed.

## Dev Notes

### Proposed module seams

Keep the implementation a modular monolith. Prefer small modules under existing contexts rather
than growing `paper_cmd.py` or creating a service:

- `algua/registry/frozen_artifact.py`: typed artifact inventory/materialization/verification; no DB.
- `algua/registry/planner_environment.py`: complete environment-key resolution/provision/verification.
- `algua/live/planner_protocol.py`: typed two-phase request/response validation and Parquet/JSON codecs.
- `algua/live/frozen_planner.py`: supervisor dispatcher and child entry point split into focused
  modules if the size ratchet requires it.
- Existing `algua/registry/db/deployment.py`, `registry/store/deployment.py`,
  `registry/deployment.py`, `registry/intake.py`, `registry/paper_runtime.py`,
  `live/live_loop.py` and `cli/paper_cmd.py`: minimal integration changes only.

Names are guidance, not an abstraction mandate. Preserve the current import-linter layering. The
frozen child may import bundle code; contracts/features remain pure and must not gain registry,
configuration or execution dependencies.

`live_loop.py`, `paper_cmd.py` and `live_cmd.py` are already at their module-size pins, while
`registry/deployment.py` is near the unpinned threshold. Carve behavior into focused modules rather
than raising pins. Extend `CODEOWNERS`/repository hygiene checks so artifact identity, wire
validation and frozen-dispatch authority code receive the same protected review as current gates.

### Transaction and filesystem boundary

Prepare/verify potentially slow Git, filesystem and environment content before `BEGIN IMMEDIATE`.
Publish immutable content first, then let the existing atomic intake transaction insert-or-verify
its descriptor and activate it. If the transaction loses a race or rolls back, the content remains
an unreferenced complete object. Never hold a SQLite lock across Git, `uv`, process launch, Parquet
I/O or fsync walks. Never overwrite or repair an active artifact in place.

Use stable relative locators beneath configured artifact/environment roots. Registry data must not
contain host-absolute paths as identity. Resolve and containment-check paths from the trusted store
root, refuse symlink traversal and keep a lease/handle across verification and child lifetime so a
cooperating cleanup cannot replace the verified object. Same-UID hostile replacement remains part
of the accepted no-sandbox threat model until deployment hardening supplies stronger ownership or
mount isolation.

### Wire and process details

The current in-process planner protocol is `1`; the new two-phase wire shape requires an explicit
new supported version and must not reinterpret old manifests. Use Arrow's explicit
`preserve_index=True`, then validate the
round trip rather than trusting pandas metadata defaults. The repo already pins `pyarrow`; do not
upgrade it for this story.

Use `subprocess.Popen` where bounded streaming and process-group teardown require it. The Python
3.12 API supports an explicit replacement `env`, `close_fds`, full executable paths and
`start_new_session`; use those controls rather than `preexec_fn`. Avoid `uv run` at tick time because
it may inspect/sync project state. Invoke the already provisioned environment's absolute Python
directly with isolated/no-user-site flags compatible with importing the bundle.

Do not log raw child stderr or exceptions into user-visible/audit payloads. Map them to stable codes
and retain bounded sanitized diagnostics through the existing observability path. Runtime evidence
never grants authority.

### Required tests and verification

Add focused tests beside the new modules plus integration coverage in the existing deployment,
paper-runtime, live-loop, CLI paper, lane-parity, schema/migration and module-size suites. At minimum
prove:

- stable digest under order/mtime changes; digest changes for every behavior/content identity input;
- path/symlink/hardlink/special-file/collision rejection and concurrent/crash-safe publication;
- exact model bytes are copied without path reread;
- no partial DB/filesystem state across every failure boundary;
- environment key equality/separation, inventory verification and no checkout/editable import;
- canonical frame round trips and malformed frame refusal;
- strict JSON fuzz cases, request binding, timeout/process-tree cleanup and bounded output;
- in-process/frozen parity and determinism across representative strategies;
- no provider, venue, cancel, submit or hook effect for an invalid frozen tenant;
- valid sibling continuation and systemic failure propagation in `run-all`;
- mutable-working-tree independence, restart without Git and indefinite retention.

Run, sequentially from the repository root:

```bash
uv run pytest -q
uv run ruff check .
uv run mypy algua
uv run lint-imports
```

Do not weaken a test, add `# type: ignore`, introduce an import exemption or raise a module-size pin.

### References

- [Canonical PRD](../../PRD.md), especially §§5, 7, 9–10, 15–16, 24–26.
- [Architecture](../../architecture.md), especially artifact/runtime isolation and module boundaries.
- [Artifact-freeze design](../../superpowers/specs/2026-09-22-artifact-freeze-design.md).
- [Story 1.1](1-1-extract-in-process-decision-planner.md) and
  [Story 1.2](1-2-record-working-tree-deployments.md).
- [Bar schema](../../contracts/bar-schema.md).
- [Python 3.12 subprocess documentation](https://docs.python.org/3.12/library/subprocess.html).
- [Apache Arrow pandas/index integration](https://arrow.apache.org/docs/python/pandas.html) and
  [Parquet I/O](https://arrow.apache.org/docs/python/parquet.html).
- [uv locked syncing and partial installs](https://docs.astral.sh/uv/concepts/projects/sync/).

## Dev Agent Record

### Agent Model Used

To be recorded during implementation.

### Debug Log References

To be recorded during implementation.

### Completion Notes List

To be recorded during implementation.

### File List

To be recorded during implementation.

## Change Log

- 2026-09-25: Status changed to `decomposed` after implementation-readiness review. Stories
  1.3a–1.3d and their coverage map now govern implementation; unsupported model assets were
  explicitly deferred. No runtime, authority, schema or deployment behavior changed.
- 2026-09-24: Story prepared from the approved Story 1.3 architecture decision; status set to
  `ready-for-dev`. No runtime, authority, schema or deployment behavior changed by this document.
