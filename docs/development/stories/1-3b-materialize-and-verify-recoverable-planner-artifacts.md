---
baseline_commit: 24c4a2bc138822c5a74ea0ed91d6d5f03c402d97
---

# Story 1.3b: Materialize and verify recoverable planner artifacts

Status: review

Prepared: 2026-09-27. Baseline: Story 1.3a merge `24c4a2b` (PR #674).
Epic: 1. Parent: Story 1.3. Requirements: FR4, FR6, FR9–FR10 and NFR1, NFR3–NFR8.
Depends on: Story 1.3a reviewed and merged.

## Story

As Algua's operator,
I want a qualified candidate's planner source and matching dependency environment materialized as
recoverable content-addressed objects,
so that I can verify exactly what may run before any deployment is activated or trading behavior
changes.

## Scope and authority

This story builds, publishes, records and verifies immutable content; it never executes that content
for a tick. `algua deployment prepare NAME` may publish a bundle/environment and insert or
byte-verify one append-only artifact descriptor. `algua deployment verify MANIFEST_DIGEST` is an
offline read-only verifier. Neither command activates or retires a deployment, allocates capital,
changes lifecycle stage, consumes a research gate, invokes a planner, contacts a provider/broker or
changes paper/live behavior.

Preparation accepts only the exact current clean full-OID `HEAD` whose recomputed artifact identity
matches the candidate's newest eligible gate and entry. Source and build-input bytes come from Git
objects, never checkout files. The source bundle is not a checkout or an installed wheel. The shared
environment contains locked third-party runtime dependencies and no local/editable Algua install.

Current paper-tradable strategies are source-only. A model handle or non-empty asset inventory fails
before any external asset path or bytes are read. Existing working-tree descriptors remain byte-
compatible and their execution/verification path is unchanged.

## Normative artifact contract

The [Story 1.3b machine contract](../specs/spec-story-1-3b-artifact-environment-contract/SPEC.md) and
its [field-level companion](../specs/spec-story-1-3b-artifact-environment-contract/artifact-environment-contract.md)
are normative. Implementers and reviewers must read both. They define all digest domains, schemas,
limits, Git/path rules, environment flags, publication/recovery behavior, persistence semantics,
command payloads and error codes.

| Identity | Exact role |
|---|---|
| `bundle_digest` | exported Git source plus generated resolved configuration and protocol metadata |
| `build_inputs_digest` | exact committed `pyproject.toml`, `uv.lock` and `.python-version` blobs |
| `environment_key` | build inputs, dependency hash, complete interpreter/platform facts and installer policy |
| `installed_inventory_digest` | normalized installed distributions and importable files |
| `environment_digest` | environment key, installed inventory and verified interpreter identity |
| `manifest_digest` | outer frozen descriptor binding qualified identity, bundle, environment, config, assets and protocols |

These identities are non-cyclic. An environment-only change does not change `bundle_digest`.

## Acceptance criteria

1. **Qualification is exact and race-safe.** Given `deployment prepare NAME`, preparation requires
   the current candidate, exact recomputed code/config/dependency identity, newest eligible
   unanchored research gate and current clean full-OID `HEAD`. It rejects tracked drift and
   untracked source shadowing. After slow work, a short `BEGIN IMMEDIATE` transaction revalidates
   all predicates and the same HEAD before recording only the descriptor; drift produces
   `frozen_source_drift` and no row/state change.
2. **Source bytes come from Git objects.** Every accepted tracked regular blob under `algua/` is
   enumerated from the recorded commit using binary/NUL-safe parsing and read by object ID. Only
   modes `100644` and `100755` are accepted. Symlinks, gitlinks, special entries, unsafe/invalid or
   colliding paths and protected-limit violations fail with `frozen_source_invalid`. Mutable
   checkout bytes, `.git`, credentials, operational data and host-absolute paths are never copied.
3. **Bundle identity is deterministic and non-cyclic.** The bundle contains exact exported source,
   canonical `resolved-config.json` and canonical `protocol.json`. Its ordered inventory binds path,
   logical mode, size and full file digest under the normative domain. Order, root and timestamps do
   not affect it; any source/mode/config/protocol change does. Literal golden digest vectors pass.
4. **Publication is atomic and immutable.** Bundle and environment use unique private same-filesystem
   staging, complete canonical verification, sealing, bottom-up fsync, a per-digest lock and
   no-overwrite rename into exact digest-derived relative locators. A race loser verifies the
   winner. Existing valid content is reused; corrupt/partial/writable/mis-typed content fails closed
   without overwrite or repair. Failures clean only the exact owned staging path.
5. **Environment inputs and key are complete.** Exact committed `pyproject.toml`, `uv.lock` and
   `.python-version` bytes form the build-input digest. The environment key additionally binds the
   existing dependency hash, implementation/full Python version, cache tag, SOABI, platform/arch,
   uv version and complete normative create/sync policy. Equal environments share only when this
   full key matches; each relevant change separates them.
6. **Provisioning is locked and checkout-independent.** Preparation creates a relocatable
   environment from the exact current interpreter and committed lock with the normative locked,
   no-project/workspace/local/editable/build/download and copy-link flags. It may acquire a
   compatible wheel already selected by the lock, but never resolves/upgrades or downloads Python.
   Path/VCS/local/editable/source-only dependencies fail as incompatible; a temporarily unavailable
   compatible locked wheel alone yields retryable `frozen_environment_unavailable`.
7. **Published environment verifies independently.** Before and after final-location publication,
   the environment's interpreter proves its keyed identity, exact installed inventory, absence of
   an installed/importable `algua`, absence of checkout/cache hardlink leakage and compliance with
   the narrow interpreter-link and read-only permission policy. Verification after publication
   succeeds without Git, uv, network or the current checkout; any mismatch fails closed.
8. **Assets remain unsupported.** A source-only candidate records canonical assets `[]`. A model
   handle or other non-empty inventory yields `frozen_assets_unsupported` before external asset
   dereference and leaves no object or descriptor row. No model lane or paper-tradability rule is
   changed.
9. **Frozen descriptor is typed and append-only.** The canonical version-1 manifest records the
   exact qualified identity, full source commit, bundle/environment identities and locators,
   resolved config, universe, empty assets, existing integer planner protocol, boundary version and
   named `frozen-planner` wire version. Existing `deployment_artifacts` columns exactly agree with
   it. Insert-or-byte-verification is concurrency-safe; existing rows are never mutated/repointed;
   working-tree manifest bytes remain valid; no schema change occurs unless separately reviewed.
10. **Commands are bounded JSON and non-activating.** Prepare and verify return the exact normative
    JSON fields or stable typed error envelopes with correct retry flags, bounded sanitized
    diagnostics and relative locators only. No absolute path, raw uv stderr, URL, credential or
    traceback leaks. Tests prove zero deployment, allocation, transition, gate-consumption, planner,
    provider and broker effects.
11. **Recovery is offline and retained.** `deployment verify MANIFEST_DIGEST` loads the descriptor
    by digest and verifies all descriptor relations, inventories, permissions and the final
    interpreter from the trusted store root only. Missing/corrupt content is not rebuilt. Complete
    unreferenced objects left by a database failure are reusable; retry records the same bytes and
    ID. Retirement, startup and repeated verification never garbage-collect content.
12. **Repository and authority walls remain intact.** New artifact identity/publication/
    verification/error-policy modules are CODEOWNERS-protected and pinned by the repository hygiene
    test. Existing planner parity, working-tree behavior, live authority, capital controls, JSON
    contract, import boundaries and module-size ratchets remain unchanged, and the full root gate
    passes.

## Tasks / subtasks

- [x] Lock the pure descriptor and digest contracts with failing tests (AC3, AC5, AC9).
  - [x] Add typed/versioned frozen bundle, environment and manifest values with strict unknown/
    missing-field and canonical-JSON validation.
  - [x] Add literal golden vectors for bundle, build inputs, environment key/inventory/environment
    and outer manifest, including correct layer sensitivity and order/mtime/root invariance.
  - [x] Preserve existing working-tree manifest canonical bytes and verification tests.
- [x] Export and validate source/build inputs from exact Git objects (AC1–AC3).
  - [x] Add binary-safe `ls-tree` parsing and blob reads for accepted source and root build inputs;
    never read their bytes through checkout paths.
  - [x] Enforce clean current full-OID HEAD, source-shadow detection, modes, path normalization/
    collision/portability rules and every protected size/count bound.
  - [x] Reject unsupported assets before external path/byte access and prove no publication/row.
- [x] Implement the immutable content store (AC4, AC11).
  - [x] Add exact digest-derived locator resolution beneath trusted `Settings.data_dir` with
    component/type/containment checks.
  - [x] Implement private same-filesystem staging, canonical verification, sealing, fsync, flock,
    no-overwrite rename and final verification without replacement helpers.
  - [x] Cover same/different-digest races, winner reuse, corrupt existing targets and fault injection
    at every write/fsync/seal/rename boundary; delete only owned staging content.
- [x] Provision and verify the shared planner environment (AC5–AC7).
  - [x] Build from private committed inputs using exact tested uv argv/environment and current
    interpreter; reject resolution, project/local installs, unsupported lock entries and downloads.
  - [x] Generate/verify the complete installed inventory, keyed interpreter facts, permitted
    interpreter links, absence of hardlinks/writable content and absence of Algua/checkout imports.
  - [x] Re-run verification with the published interpreter at its final locator and test offline
    recovery plus sharing/separation cases.
- [x] Record and retrieve the frozen descriptor without activation (AC1, AC9, AC11).
  - [x] Reuse the current immutable ledger and denormalized fields; add typed frozen dispatch without
    changing working-tree parsing. Add fetch-by-manifest-digest.
  - [x] Revalidate stage, candidate entry, newest gate, identity and HEAD in one short write
    transaction; insert or byte-verify only the artifact row.
  - [x] Prove no Git/uv/filesystem walk runs under the transaction and DB rollback leaves complete
    reusable unreferenced objects.
- [x] Add the thin deployment command surface (AC10–AC11).
  - [x] Add a focused `algua/cli/deployment_cmd.py`, mount it only at the CLI composition root and
    implement exact prepare/verify success schemas.
  - [x] Add domain exception mapping and the normative retry allowlist entry only for environment
    unavailability; test every stable error envelope and disclosure bound.
  - [x] Prove verify performs no Git, uv, network, checkout or mutation access.
- [x] Protect and review the new integrity surface (AC12).
  - [x] Add new identity, verifier, publication, descriptor-recording and error-policy modules to
    root `CODEOWNERS` and `tests/test_repo_hygiene.py`; do not broaden existing allowlists.
  - [x] Run focused contract/store/environment/CLI tests, Story 1.3a parity, then the full sequential
    root gate.
  - [x] Obtain independent adversarial, edge-case and acceptance review before marking done. Any
    schema, activation/intake, paper/live runtime, broker, capital or authority change requires
    explicit rescoping rather than an incidental patch.

### Review Findings

- [x] [Review][Patch] Normalize the uv-created `lib64 -> lib` link before strict inventory so the
  normative Linux provisioning path can succeed [algua/registry/planner_environment.py:269]
- [x] [Review][Patch] Reject model-backed strategies before loader-side model artifact dereference
  [algua/registry/artifact_preparation.py:104]
- [x] [Review][Patch] Move the final identity/clean-HEAD callback immediately against `BEGIN
  IMMEDIATE`, while keeping Git and identity work outside the transaction
  [algua/registry/store/artifacts.py:139]
- [x] [Review][Patch] Inspect every store-locator ancestor without following symlinks
  [algua/registry/artifact_store.py:21]
- [x] [Review][Patch] Enforce generated-file, aggregate-bundle and one-MiB manifest bounds
  [algua/registry/artifact_preparation.py:65]
- [x] [Review][Patch] Bind and require the exact permitted interpreter-link inventory
  [algua/registry/planner_environment.py:175]
- [x] [Review][Patch] Sanitize and bound unexpected frozen-command diagnostics
  [algua/cli/deployment_cmd.py:35]
- [x] [Review][Patch] Reject untracked directory entries except a cache directory containing only
  recognized generated bytecode [algua/registry/frozen_source.py:150]
- [x] [Review][Patch] Complete the required publication fault, concurrency, offline and zero-effect
  verification matrix [tests/test_artifact_store.py:67]
- [x] [Review][Patch] Validate every locked wheel entry has canonical registry URL and hash data
  [algua/registry/planner_environment.py:93]
- [x] [Review][Patch] Recompute provisioning-key inputs and recheck the uv version immediately before
  execution [algua/registry/planner_environment.py:245]
- [x] [Review][Patch] Bound both uv provisioning subprocesses with explicit timeouts
  [algua/registry/planner_environment.py:267]
- [x] [Review][Patch] Bound Git tree/blob output before buffering it in memory
  [algua/registry/frozen_source.py:49]
- [x] [Review][Patch] Reject oversized bundle entries and undeclared empty directories during
  offline verification [algua/registry/artifact_store.py:65]
- [x] [Review][Patch] Hash environment files incrementally instead of reading unbounded files into
  memory [algua/registry/planner_environment.py:178]
- [x] [Review][Patch] Replace inherited `HOME` with a private/non-authoritative home for uv and
  interpreter subprocesses [algua/registry/planner_environment.py:131]
- [x] [Review][Patch] Canonicalize the retained resolved-config object, not only its emitted JSON
  bytes [algua/registry/artifact_preparation.py:109]
- [x] [Review][Patch] Classify a missing or unusable local uv executable as incompatible, leaving
  retryable unavailable for compatible locked-wheel acquisition only
  [algua/registry/planner_environment.py:48]
- [x] [Review][Patch] Force-refresh the strategy module before retaining CONFIG so a warm process
  cannot freeze stale configuration [algua/strategies/loader.py:212]
- [x] [Review][Patch] Stop a Git subprocess once its stdout exceeds the protected limit instead of
  bounding memory only after unbounded temporary-file output [algua/registry/frozen_source.py:60]
- [x] [Review][Patch] Require artifact sizes to be exact non-negative integers
  [algua/registry/artifact_contract.py:85]
- [x] [Review][Patch] Require bundle counts to be exact bounded integers
  [algua/registry/artifact_contract.py:144]
- [x] [Review][Patch] Apply the canonical relative-path rules to every ArtifactFile contract value
  [algua/registry/artifact_contract.py:85]
- [x] [Review][Patch] Reject empty or malformed interpreter identity fields at construction
  [algua/registry/artifact_contract.py:144]
- [x] [Review][Patch] Validate non-empty installer identity/argv and exact argument element types
  [algua/registry/artifact_contract.py:165]
- [x] [Review][Patch] Validate and uniquely order installed distribution and interpreter-link
  identities [algua/registry/artifact_contract.py:191]
- [x] [Review][Patch] Make manifest parsing total and strictly typed for versions, counts and argv
  rather than accepting bool/float equality aliases [algua/registry/artifact_manifest.py:42]
- [x] [Review][Patch] Require direct FrozenManifest construction to retain only a configuration
  object and string-or-null universe [algua/registry/frozen_manifest_contract.py:33]
- [x] [Review][Patch] Reject Git assume-unchanged and skip-worktree flags that can hide tracked
  source drift from status checks [algua/registry/frozen_source.py:187]
- [x] [Review][Patch] Purge timestamp-valid cached bytecode before refreshing a warm strategy
  module so same-size, same-mtime source changes cannot retain stale CONFIG
  [algua/strategies/loader.py:221]
- [x] [Review][Patch] Reload the strategy-family module closure in dependency-first order rather
  than relying on the incorrect assumption that `sys.modules` insertion order is dependency-first
  [algua/strategies/loader.py:72]
- [ ] [Review][Patch] Require NFC-normalized installer identity and argument strings before they
  enter the content-addressed environment key [algua/registry/environment_contract.py:77]
- [ ] [Review][Patch] Translate lone-surrogate manifest text into the parser's stable ValueError
  contract instead of leaking UnicodeEncodeError [algua/registry/artifact_manifest.py:58]
- [ ] [Review][Patch] Canonicalize installed distribution names across runs of hyphen, underscore
  and dot before forbidden-name and uniqueness checks
  [algua/registry/planner_environment_inventory.py:45]
- [ ] [Review][Patch] Normalize and retain universe names before canonical JSON and denormalized
  ledger projection so a descriptor cannot disagree with its own stored columns
  [algua/registry/frozen_manifest_contract.py:42]
- [ ] [Review][Patch] Enforce the bundle file-count bound before constructing or hashing the
  inventory payload [algua/registry/artifact_contract.py:157]
- [ ] [Review][Patch] Inspect repository-wide hidden Git index flags without imposing the
  source-entry aggregate byte bound on the complete repository index
  [algua/registry/frozen_source.py:201]

## Development notes

- Prefer focused modules: `registry/artifact_contract.py`, `registry/frozen_source.py`,
  `registry/artifact_store.py`, `registry/planner_environment.py`,
  `registry/artifact_preparation.py` and `cli/deployment_cmd.py`. Keep each below 300 lines and do
  not grow the already size-pinned `registry/deployment.py` or command modules.
- Reuse `algua/primitives/atomic_io.py` fsync helpers and `algua/primitives/flock.py`; do not use
  `write_bytes_durable` because replacement semantics are forbidden for immutable objects.
- Registry code must not import `algua.live`. Shared protocol constants belong in the pure
  `algua/contracts/planner.py` seam if an additional constant is required.
- The trusted root is operational configuration; manifest locators are data and must exactly equal
  their digest-derived form. Do not expose resolved absolute paths in output.
- Same-UID hostile replacement remains inside the accepted no-sandbox residual. Do not describe
  read-only permissions or flock as a security sandbox.
- A ready story authorizes implementation and review only. It does not authorize activation,
  deployment, capital use or live operation.

## Verification

```bash
uv run pytest -q
uv run ruff check .
uv run mypy algua
uv run lint-imports
```

## References

- [Parent Story 1.3](1-3-materialize-and-execute-frozen-planner-artifacts.md)
- [Normative Story 1.3b machine contract](../specs/spec-story-1-3b-artifact-environment-contract/SPEC.md)
- [Normative artifact/environment companion](../specs/spec-story-1-3b-artifact-environment-contract/artifact-environment-contract.md)
- [Story 1.3a](1-3a-complete-two-phase-planner-boundary-in-process.md)
- [Approved Sprint Change Proposal](../sprint-change-proposal-2026-09-25.md)
- [Artifact-freeze design](../../superpowers/specs/2026-09-22-artifact-freeze-design.md)
- [Canonical PRD](../../PRD.md), §§5–7, 10, 15, 24–26.
- [Current architecture](../../architecture.md)
- Repository `AGENTS.md`, `CLAUDE.md`, `docs/agent/operating.md` and frozen
  `docs/contracts/bar-schema.md` remain binding during implementation.

## Dev Agent Record

### Implementation Plan

- Implement the story in its approved task order using a red-green-refactor cycle per task.
- Keep canonical identity values and strict manifest parsing separate from Git/filesystem/uv I/O.
- Preserve current working-tree descriptor bytes and verify the full repository after each task.

### Debug Log References

- Story preparation used repository inspection plus independent requirements, implementation and
  security review. The initial backlog draft was not admitted until digest, Git export, environment,
  atomicity, persistence, CLI, recovery and authority semantics were closed normatively.
- Task 1 red phase failed on the absent `artifact_contract` module. Green phase added typed values,
  strict canonical parsing and literal vectors; 33 focused tests and the full 4,086-test suite pass.
- Task 2 red phase failed on the absent `frozen_source` module. Green phase added binary Git-object
  export, strict path/mode/bound checks, clean-HEAD shadow detection and pre-dereference asset refusal;
  21 focused tests and the full 4,098-test suite pass.
- Task 3 red phase failed on the absent `artifact_store` module. Green phase added canonical locator
  resolution, sealed same-parent staging, durable locked no-overwrite publication and full inventory
  verification; 11 focused tests and the full 4,109-test suite pass.
- Task 4 provisioning/inventory red tests locked exact uv policy, scrubbed environment, lock-source
  rejection, installed inventory and isolated interpreter verification. Twelve focused tests and the
  full 4,121-test suite pass; atomic environment publication remains unchecked.
- Task 4 final publication added sealed atomic environment reuse and post-rename execution checks;
  26 combined focused tests and the full 4,124-test suite pass.
- Task 5 red tests locked typed ledger round-trips, candidate/gate revalidation, non-activation,
  rollback and slow-work transaction boundaries. The first full run exposed the module-size ratchet;
  splitting the artifact ledger into its own mixin fixed the structural regression. Thirty-seven
  focused tests and the full 4,133-test suite pass.
- Task 6 red tests failed on the absent error, verification and command modules. Green added the
  composition-root command mount, exact payload projection, offline descriptor/content verification
  and all nine bounded frozen error envelopes. Seventy-six focused tests and the full 4,155-test
  suite pass; lint, types and all 28 import contracts are green.
- Task 7 protected every new integrity surface and completed the adversarial, edge-case and
  acceptance review. All 18 accepted review patches were implemented test-first; the focused
  artifact plus Story 1.3a parity matrix passes, and the full sequential root gate is green.
- Second review round: each of the 11 accepted patches first failed red (100 contract/manifest
  cases, 6 Git bound/index-flag cases and the warm-module CONFIG case). Validating installed
  distribution identities exposed that METADATA parsing read the description body (the locked
  `vectorbt` README overrides its `Name`); header-only parsing was added test-first. All 153 locked
  distributions and 25,596 files of the repository environment satisfy the stricter identities.
  Every new guard was mutation-checked; a redundant duplicate count check was removed. The full
  4,325-test suite, ruff, mypy and all 28 import contracts pass.

### Completion Notes

- Task 1 complete: non-cyclic bundle/build-input/environment/manifest identities are typed and
  versioned, parsing rejects noncanonical/duplicate/unknown input, and working-tree compatibility is
  unchanged. No filesystem, database, deployment or runtime behavior is introduced yet.
- Task 2 complete: source and build inputs are read from the exact commit object database, unsafe or
  ambiguous trees fail closed, and unsupported model content is refused before external access.
- Task 3 complete: bundle publication is atomic and reusable under races, detects corruption and
  permission/link drift without repair, and leaves only complete published objects across faults.
- Task 4 complete: private exact-lock provisioning, complete environment inventory, sealed atomic
  publication, final-locator interpreter execution and offline reuse all fail closed on drift.
- Task 5 complete: preparation publishes deterministic bundle/environment content outside SQLite,
  repeats identity and clean-HEAD checks, then atomically inserts or byte-verifies only the frozen
  descriptor after revalidating the candidate episode and gate. Digest lookup is activation-free.
- Task 6 complete: `deployment prepare` and `deployment verify` expose only canonical relative
  identities, offline verification performs no qualification or rebuilding, and retry policy is
  false for every frozen failure except temporary locked-environment acquisition.
- Task 7 complete: real Linux venv-link behavior, pre-dereference asset refusal, transaction
  adjacency, bounded Git/filesystem input, immutable-store ancestry, exact environment inventory,
  sanitized diagnostics and the complete publication/recovery fault matrix are now enforced.
- Second review round complete: the declared-config read force-refreshes the strategy closure;
  Git reads stream and kill the subprocess once output exceeds its bound; any assume-unchanged or
  skip-worktree index flag fails clean-HEAD qualification; artifact files, bundle counts,
  interpreter/installer/distribution/link identities and direct `FrozenManifest` fields are
  strictly typed and bounded at construction; and manifest parsing is total with exact-integer
  version, count and argv typing. Environment identity values moved to the protected
  `environment_contract.py` to keep both contract modules below the size floor. Golden digest
  vectors, working-tree descriptors, schema and every authority wall are unchanged.

### File List

- `algua/registry/artifact_contract.py`
- `algua/registry/environment_contract.py`
- `algua/registry/artifact_errors.py`
- `algua/registry/artifact_manifest.py`
- `algua/registry/frozen_manifest_contract.py`
- `algua/registry/artifact_preparation.py`
- `algua/registry/artifact_recording.py`
- `algua/registry/artifact_store.py`
- `algua/registry/artifact_verification.py`
- `algua/cli/deployment_cmd.py`
- `algua/cli/errors.py`
- `algua/cli/main.py`
- `algua/registry/environment_store.py`
- `algua/registry/frozen_source.py`
- `algua/registry/planner_environment.py`
- `algua/registry/planner_environment_errors.py`
- `algua/registry/planner_environment_inventory.py`
- `algua/registry/store/__init__.py`
- `algua/registry/store/artifacts.py`
- `algua/registry/store/deployment.py`
- `algua/strategies/loader.py`
- `CODEOWNERS`
- `docs/development/sprint-status.yaml`
- `docs/contracts/cli-error-envelope.md`
- `docs/development/stories/1-3b-materialize-and-verify-recoverable-planner-artifacts.md`
- `tests/test_frozen_artifact_contract.py`
- `tests/test_frozen_artifact_ledger.py`
- `tests/test_cli_deployment.py`
- `tests/test_frozen_artifact_verification.py`
- `tests/test_artifact_store.py`
- `tests/test_environment_store.py`
- `tests/test_frozen_source.py`
- `tests/test_planner_environment.py`
- `tests/test_repo_hygiene.py`
- `tests/test_strategy_loader.py`

### Change Log

- 2026-09-27: Rebased on Story 1.3a, added the normative artifact/environment contract and prepared
  the bounded non-activating implementation story for readiness review.
- 2026-09-27: Passed BMAD implementation readiness and moved to `ready-for-dev`.
- 2026-09-27: Started implementation and completed the pure frozen identity/manifest contract.
- 2026-09-27: Added exact Git-object source/build-input export and source-only admission checks.
- 2026-09-27: Added durable content-addressed bundle publication and offline verification.
- 2026-09-27: Added pinned uv provisioning policy and isolated environment inventory verification.
- 2026-09-27: Completed atomic environment publication and final-locator verification.
- 2026-09-27: Added non-activating frozen preparation, typed immutable-ledger recording and
  digest-based retrieval with candidate/gate revalidation and transaction-boundary tests.
- 2026-09-27: Added bounded deployment prepare/verify JSON commands, offline recovery verification
  and the stable frozen-artifact error/retry taxonomy.
- 2026-09-27: Addressed all 18 accepted independent-review patches, protected the expanded
  integrity surface and moved Story 1.3b to review with the full root gate green.
- 2026-09-28: Addressed all 11 second-round review patches test-first (warm CONFIG refresh, bounded
  Git streaming, hidden index flags, strict artifact/environment/manifest typing), carved the
  protected environment identity contract and kept the full root gate green.
