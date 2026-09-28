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
- [x] [Review][Patch] Require NFC-normalized installer identity and argument strings before they
  enter the content-addressed environment key [algua/registry/environment_contract.py:77]
- [x] [Review][Patch] Translate lone-surrogate manifest text into the parser's stable ValueError
  contract instead of leaking UnicodeEncodeError [algua/registry/artifact_manifest.py:58]
- [x] [Review][Patch] Canonicalize installed distribution names across runs of hyphen, underscore
  and dot before forbidden-name and uniqueness checks
  [algua/registry/planner_environment_inventory.py:45]
- [x] [Review][Patch] Normalize and retain universe names before canonical JSON and denormalized
  ledger projection so a descriptor cannot disagree with its own stored columns
  [algua/registry/frozen_manifest_contract.py:42]
- [x] [Review][Patch] Enforce the bundle file-count bound before constructing or hashing the
  inventory payload [algua/registry/artifact_contract.py:157]
- [x] [Review][Patch] Inspect repository-wide hidden Git index flags without imposing the
  source-entry aggregate byte bound on the complete repository index
  [algua/registry/frozen_source.py:201]
- [x] [Review][Patch] Purge cached bytecode for newly imported strategy-family helpers before any
  warm refresh executes current source [algua/primitives/module_refresh.py:24]
- [x] [Review][Patch] Detect cyclic strategy-family import components and fail closed instead of
  claiming an arbitrary DFS order is dependency-safe [algua/primitives/module_refresh.py:36]
- [x] [Review][Patch] Remove globals deleted from current source when refreshing a module rather
  than retaining them through `importlib.reload` dictionary reuse
  [algua/primitives/module_refresh.py:31]
- [x] [Review][Patch] Roll back the complete strategy-family module state if any closure member
  fails during refresh so later callers cannot observe a mixed-version closure
  [algua/primitives/module_refresh.py:31]
- [x] [Review][Patch] Reject surrogate-bearing installer arguments, universe names and resolved
  configuration during direct typed construction with stable ValueError failures
  [algua/registry/environment_contract.py:85]
- [x] [Review][Patch] Add the identity-critical module refresh seam to CODEOWNERS and the repository
  hygiene protection set [algua/primitives/module_refresh.py:18]
- [x] [Review][Patch] Serialize the complete process-global module refresh transaction so concurrent
  callers cannot observe or retain an absent or partially rebuilt strategy family
  [algua/primitives/module_refresh.py:44]
- [x] [Review][Patch] Remove newly imported external modules and their parent bindings when a failed
  refresh would otherwise leave references to discarded fresh family objects
  [algua/primitives/module_refresh.py:50]
- [x] [Review][Patch] Reject symlinked package/source paths before discovery or bytecode purge so a
  symlinked helper cannot execute timestamp-valid stale bytecode as current source
  [algua/primitives/module_refresh.py:93]
- [x] [Review][Patch] Snapshot the parent package binding directly from its dictionary so a dynamic
  `__getattr__` cannot manufacture state that rollback later installs
  [algua/primitives/module_refresh.py:48]
- [x] [Review][Patch] Replace recursive cycle traversal with an explicit stack so a valid large
  acyclic source closure cannot fail with `RecursionError`
  [algua/primitives/module_refresh.py:126]
- [x] [Review][Patch] Remove the CPython global import lock from strategy refresh because its lock
  ordering can deadlock against an already-running import; serialize only supported loader/refresh
  callers and document the import-quiescent process precondition
  [algua/primitives/module_refresh.py:52]
- [x] [Review][Patch] Snapshot and restore every affected module's direct parent binding exactly so
  rollback cannot delete a pre-existing binding when attempted code replaces its child module
  [algua/primitives/module_refresh.py:88]
- [x] [Review][Patch] Reject every symlink entry in the complete family tree before purge/import so
  a dynamic `importlib`/`__import__` edge cannot reach stale bytecode outside the static closure
  [algua/primitives/module_refresh.py:146]
- [x] [Review][Patch] Reject lexical parent traversal before normalization so a `symlink/..` search
  path cannot erase the symlink component before validation
  [algua/primitives/module_refresh.py:107]
- [x] [Review][Patch] Snapshot module state before package discovery and restore it on every
  preflight failure so a cold nested-parent import is not left behind
  [algua/primitives/module_refresh.py:60]
- [x] [Review][Patch] Keep beyond-recursion coverage in the in-memory graph test but use a small
  fixed filesystem closure for integration so the normal root gate does not create thousands of
  source files or scale with a mutable interpreter recursion limit
  [tests/test_module_refresh.py:264]
- [x] [Review][Patch] Restore parent bindings changed by transient imports even when attempted code
  removes its own child entry from `sys.modules` before failing
  [algua/primitives/module_refresh.py:116]
- [x] [Review][Patch] Fail preflight on importable sourceless bytecode or other non-source family
  entries that a dynamic import could execute outside the statically validated source closure
  [algua/primitives/module_refresh.py:146]
- [x] [Review][Patch] Constrain fresh package search locations to the exact prevalidated family
  roots before any child import so package `__init__` cannot extend `__path__` into an unscanned tree
  [algua/primitives/module_refresh.py:91]
- [x] [Review][Patch] Reject a nested refresh transaction on the same thread while retaining safe
  same-thread re-entrancy for ordinary serialized imports
  [algua/primitives/module_refresh.py:43]
- [x] [Review][Patch] Reinitialize the private refresh lock in a forked child so a vanished owner
  cannot leave every later strategy load permanently blocked
  [algua/primitives/module_refresh.py:44]
- [x] [Review][Patch] Translate family-tree inspection `OSError` failures into the stable
  `ModuleRefreshError` contract with bounded non-host-specific diagnostics
  [algua/primitives/module_refresh.py:150]
- [x] [Review][Patch] Limit rollback namespace snapshots to package/direct-parent binding state so
  refresh cost does not scale with every global in every loaded scientific module
  [algua/primitives/module_refresh.py:102]
- [x] [Review][Patch] Replace the deadlock regression's scheduling sleep with an event handshake
  proving the refresher reached the contested import before the blocked importer is released
  [tests/test_module_refresh.py:613]
- [x] [Review][Patch] Resolve and validate every family spec against its exact deterministic
  in-root source path so a poisoned path-entry finder or stale loaded-package spec cannot make
  preflight inspect one tree while execution loads another
  [algua/primitives/module_refresh.py:152]
- [x] [Review][Patch] Recover a forked child's complete inherited refresh transaction state by
  restoring its pre-refresh modules/bindings, removing inherited guards, resetting transaction
  state and replacing the vanished owner's lock before later loads proceed
  [algua/primitives/module_refresh.py:55]
- [x] [Review][Patch] Validate the committed fresh family graph so every reached entry remains a
  source-backed `ModuleType`, is identically bound in `sys.modules` and on its direct parent, and
  every package retains its exact confined `__path__`
  [algua/primitives/module_refresh.py:165]
- [x] [Review][Patch] Translate source stat/read/parse failures into bounded path-free
  `ModuleRefreshError` diagnostics rather than leaking raw `OSError` or absolute-path `SyntaxError`
  [algua/primitives/module_source_scan.py:116]
- [x] [Review][Patch] Reject non-regular importable source entries such as FIFOs, sockets and
  devices during preflight so a dynamic source import cannot block or execute unsupported content
  [algua/primitives/module_source_scan.py:50]
- [x] [Review][Patch] Translate bytecode-purge failures into the stable bounded refresh error
  contract and explicitly prove partial cache deletion cannot commit or mutate the module graph
  [algua/primitives/module_refresh.py:217]

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
- Third review round: all 8 accepted patches first failed red. The closure-order finding was
  sharper than stated: completed imports and reloads both move a module to the end of
  `sys.modules`, so order is dependency-first until a helper loaded earlier is edited to import a
  module that another family member imported independently; the red case reproduces exactly that.
  The stale-bytecode red case writes a timestamp pyc, then restores the old size and mtime. The
  scan of a streamed index is proven chunk-boundary independent and bounded to under 1 MiB of
  retained memory for a 32 MiB record. Every new guard was mutation-checked (17 mutations, all
  killed; one survivor exposed an unexercised single-chunk retention path, now covered). All 152
  locked third-party distributions stay unique under PEP 503 canonicalization.
- Fourth review round: all 6 accepted patches first failed red. With temporary stubs for the new
  names, the current refresh returned a newly imported helper's stale timestamp-valid bytecode
  value, accepted two-, three-, lazy-import and package-`__init__` cycles, retained a deleted
  global, silently re-imported a helper name removed from source, and left a `(2, 2)` mixed
  closure after a later member failed; nine surrogate cases passed construction and only failed
  as `UnicodeEncodeError` when first encoded; the hygiene set lacked `module_refresh.py`. Every
  new guard was mutation-checked (25 mutations, all killed). One initially survived because
  rollback masked a cycle check moved after the purge/drop; the cycle test now also proves a
  refused refresh purges nothing. All 25 bundled strategies resolve to acyclic closures, and the
  23 source-only strategies keep identical code/config/dependency hashes cold, after the
  declared-config refresh and after `reload=True`.
- Fifth review round: all 5 accepted patches first failed red (10 new cases): a competing
  thread imported a family member while the refresh was mid-transaction; a new external module
  holding the discarded fresh family stayed in `sys.modules` and bound on its parent; a parent
  `__getattr__` value was installed by rollback; all four symlink shapes (linked helper file,
  linked subpackage, linked package directory, linked `sys.path` ancestor) refreshed without
  refusal; and the recursive cycle check raised `RecursionError` on a valid closure deeper than
  the recursion limit, including a real lazy helper chain. Every new guard was mutation-checked
  (12 mutations, all killed). Three initially survived: popping a parent attribute without the
  identity check, and DFS without path release or memoization; the external-module case now
  asserts a restored parent keeps its previous binding, and a layered-diamond case proves each
  module is expanded exactly once. A dedicated case proves the search-location check precedes
  source discovery, which the origin check would otherwise mask. All 25 bundled strategies still
  refresh through `load_strategy_config`.
- Sixth review round: patches 1–5 first failed red (13 new cases). A child-process probe
  deadlocked (`returncode 3`, both threads alive) when the refresh held the global import lock
  while another thread, initializing a module the refresh imports, needed a new import; the
  supported serialized import and the cold `load_strategy` path had no shared serialization;
  both parent-binding shapes (a replaced child module, a pre-existing value shadowed by a new
  child) lost their binding (`KeyError`) on rollback; all four family-tree symlink shapes
  (dynamically imported subpackage and module, unreached nested file, dangling link) refreshed
  without refusal; a `link/..` search path passed the symlink check; and all four cold/warm-parent
  preflight failures (cycle and `KeyboardInterrupt`) left the cold nested parent imported. Patch 6
  is test-only: the filesystem case wrote 1,202 files at the default recursion limit; the fixed
  seven-helper chain is pinned by a lazy-traversal mutation. Sixteen mutations were run: fifteen
  are killed and one (reinstating an entry only when absent after the unconditional pop) is
  equivalent. Two initially survived and were closed test-first: the tree scan moved after source
  discovery (the tree case now forbids discovery before refusal) and a trailing `..` ignored (a
  direct raw-component case). All 25 bundled strategies still refresh through
  `load_strategy_config`.
- Seventh review round: patches 1–7 first failed red (20 new cases). With the transient child
  imported twice and its own `sys.modules` entry popped each time, both parent shapes kept the
  discarded module bound (absent-before and a pre-existing `'pre-existing'` value); discovery ran
  before any refusal for a sourceless `dyn.pyc`, an extension-suffixed file, a bytecode-only
  subpackage and `__pycache__/evil.pyc`; a fresh `__init__` (append, prepend-and-shadow), a
  subpackage, a member rebinding its own `__path__` and a deferred subpackage import all executed
  code outside the scanned root; a same-thread nested refresh dropped the module the outer refresh
  was executing (`KeyError`); a forked child could not acquire a lock held by a vanished thread
  (child exit 7); `scandir`, `DirEntry` and ancestor `lstat` failures leaked a raw
  `PermissionError` naming the host path, including through `load_strategy_config`; and both a
  successful and a failed refresh read an unrelated warm module's `__dict__`. The same-thread
  `serialized_import` case is a preservation proof and passed before and after. Patch 8 is
  test-only: reinstating the global import lock still deadlocks the handshake probe (`returncode
  3`), and five clean runs pass without any sleep. Patch 2 pushed `module_refresh.py` to 310 lines,
  over the size-ratchet floor, so the static non-executing preflight moved unchanged into the
  protected `module_source_scan.py` before patch 3's green. Twenty-eight mutations were run:
  twenty-five are killed and three are equivalent and were removed from the code (a parent-identity
  condition on recording, a list-type check on search paths, resolution through an already-equal
  given path). Two guard branches (no fallback to later finders, source-loader only) initially
  survived and were closed with a later-finder and a namespace-directory case. All 25 bundled
  strategies still refresh through `load_strategy_config`.
- Eighth review round: 32 new cases. Patch 1 first failed red in all three shapes (a poisoned
  cached path-entry finder, a poisoned path hook and a stale loaded package `__spec__` each executed
  outside code and served its value); a namespace, plain-module and absent family shape is a
  preservation case. Patch 5 failed red for a FIFO and a socket `dyn.py`, a FIFO subpackage
  `__init__.py` and an unreached FIFO (discovery ran before any refusal), and exact spec
  construction treated a FIFO source as absent; FIFO fixtures keep a releasable writer so a
  regression reads EOF instead of hanging. Patch 4 failed red for a syntax error, a NUL byte and
  undecodable source (raw `SyntaxError` naming the host path) and an unreadable source (raw
  `PermissionError`); the stat case already passed because patch 1's exact spec stat runs inside
  the bounded inspection wrapper. Patch 6 failed red with a raw `IsADirectoryError` after the
  top-level cache was already deleted, and a further case proves an unwalkable directory is
  refused, not silently skipped by `os.walk`. Patch 3 failed red (no refusal) for a non-module
  entry, a root replacing itself, a foreign entry, a self-removed entry, a rebound parent
  attribute, an unbound family package, a foreign `__spec__` and a module gaining an in-root
  `__path__`; a parent replaced by a non-module was added to kill a surviving mutation. Patch 2
  failed red: a child forked while another thread was parked mid-refresh inherited the partially
  rebuilt family (child exit 11). The first green attempt then hung in the child on CPython's own
  per-module import lock for the module the vanished thread was still executing; that lock is
  outside this patch and the import-quiescent precondition, so it is documented and the probe's
  later child refresh goes through a root the vanished thread was not initializing. Owner-thread
  fork (continues the transaction) and post-refresh fork (keeps the commit) are preservation
  probes. Twenty-one mutations were run and all are killed; three first survived and were closed
  with the stale-record fork probe, the non-module parent case and an explicit regular-package
  location check with a pinned refusal message. All 25 bundled strategies still refresh twice
  through `_reload_strategy_closure`.

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
- Third review round complete: the warm strategy-closure refresh purges each module's cached
  bytecode and reloads in the dependency-first order of its current source's static family
  imports, the strategy module last. That machinery is the new stdlib-only
  `algua/primitives/module_refresh.py` leaf, carved out so `loader.py` stays below the size floor.
  Installer identity and argv strings must be NFC; lone-surrogate manifest text fails as the
  parser's plain `ValueError`; `FrozenManifest` retains its NFC universe name, so JSON and ledger
  columns agree. A non-NFC gate universe still fails recording closed through the existing exact
  qualification comparison. Distribution names are PEP 503 canonical before forbidden-name and
  uniqueness checks, and a canonical collision is `EnvironmentIncompatible`. The bundle count and
  size bounds precede inventory payload work. The index-flag scan streams the whole repository
  index with bounded retention, without the source aggregate cap. Golden digest vectors, schema,
  working-tree descriptors and every live, authority, deployment and capital wall are unchanged.
- Fourth review round complete: the warm refresh no longer uses `importlib.reload`. It statically
  resolves the strategy's current family closure without executing it and fails closed on any
  cyclic component (`ModuleRefreshError`, an `ImportError` the loader maps to `StrategyNotFound`,
  hence frozen preparation's existing fail-closed drift envelope) before anything is purged or
  executed. It then purges the cached bytecode of every source under the family package, loaded
  or not, drops the warm family entries and re-imports the strategy as fresh module objects, so a
  deleted global or removed helper name cannot survive. Any failure, including a
  `BaseException`, restores every previous family module object and the parent binding, whose
  dictionaries were never touched. Refresh paths no longer pre-import the warm module. Canonical
  JSON, installer argv and the retained universe name reject lone surrogates at construction with
  a plain `ValueError`. `module_refresh.py` is CODEOWNERS- and hygiene-protected. Golden digest
  vectors, schema, working-tree descriptors and every live, authority, deployment and capital
  wall are unchanged.
- Fifth review round complete: the whole refresh transaction (preflight, bytecode purge,
  `sys.modules` drop, fresh import and rollback) holds the CPython global import lock
  (`_imp.acquire_lock`), so another thread's import of a module not already loaded waits for
  the final state; the refresh returns the fresh root module, so the loader no longer re-reads
  `sys.modules` after releasing the lock. Rollback snapshots every `sys.modules` entry, drops each
  entry the attempt introduced (inside or outside the family) plus any parent attribute bound to
  that exact discarded object, then reinstates the previous entries and the family parent
  binding, which is now read from and written to `parent.__dict__` directly. Every package search
  location and reachable source origin whose lexical path or any existing ancestor is a symlink
  fails closed before discovery, purge or import. The cycle check is an iterative DFS over
  explicit `(node, iterator)` frames with unchanged deterministic cycle naming. Static cycles
  still fail closed, and namespace and compiled modules stay unsupported. Accepted residual: a
  thread initializing a module the refresh also imports, that itself needs a new import while
  the refresh holds the lock, can deadlock against it, so warm refreshes belong in processes that
  do not import concurrently. Golden digest vectors, schema, working-tree descriptors and every
  live, authority, deployment and capital wall are unchanged.
- Sixth review round complete: the refresh no longer takes CPython's global import lock. A
  private re-entrant module lock serializes only the supported callers, `refresh_package_closure`
  and the new `serialized_import`, which the loader now uses for its cold path too. Correctness
  requires an import-quiescent process for everything else: a direct family import from another
  thread, or a supported call made from inside a module another thread is initializing, during a
  refresh is unsupported; no arbitrary `importlib` concurrency safety is claimed. Module state and
  a direct `__dict__` snapshot of every module namespace are taken before package discovery, and
  any failure from discovery through the fresh import, including a `BaseException`, reinstates
  exactly the previous `sys.modules` entries and restores each affected module's direct parent
  binding to its prior value or absence. Before discovery, every raw `..` search/source component
  fails closed and the complete family tree is scanned without following links, refusing any
  symlink entry. The beyond-recursion-limit proof stays in memory; the integration chain is fixed
  and small. Static cycles still fail closed, the scope stays source modules only, and the
  same-UID hostile filesystem swap remains an accepted residual. Golden digest vectors, schema,
  working-tree descriptors and every live, authority, deployment and capital wall are unchanged.
- Seventh review round complete: one first `sys.meta_path` guard spans the whole transaction.
  Before the import system loads any module it records that module's direct parent binding as
  first seen (read from the parent's own dictionary), and rollback restores every recorded binding
  to its prior value or absence even when attempted code removed the child's own `sys.modules`
  entry; the whole-namespace snapshot of every loaded module is gone, so rollback state is
  `sys.modules` plus those bindings. Once preflight passes, the guard resolves every family module
  itself from the exact scanned root: a child whose parent's search path deviates is refused
  before it executes, a family name missing from the root is never supplied by another finder,
  and a non-source (including namespace) entry is refused; a fresh family package whose
  `__path__` still deviates at commit fails closed, so a later lazy import cannot use it. The
  family must be a regular single-location package. The tree scan also refuses any importable
  sourceless or extension entry (a loader suffix on an undotted stem), while tagged
  `__pycache__` bytecode and data files stay admissible. A same-thread nested refresh fails with
  `ModuleRefreshError` while a same-thread `serialized_import` stays re-entrant; the private lock
  is reinitialized in a forked child. Tree-inspection `OSError`s become a bounded
  `ModuleRefreshError` naming only the error class and errno, without a chained host path, which
  the loader maps to `StrategyNotFound`. The static preflight now lives in the protected
  `algua/primitives/module_source_scan.py`. The import-quiescent precondition, source-only and
  static-cycle fail-closed policy and the same-UID hostile swap residual are unchanged; arbitrary
  import concurrency, sandboxing, mount cycles, memory exhaustion and external import side
  effects stay out of scope. Golden digest vectors, schema, working-tree descriptors and every
  live, authority, deployment and capital wall are unchanged.
- Eighth review round complete: the family location is found on the filesystem from the parent's
  search path (`sys.path` for a top-level family) and must be a regular source package, and every
  family spec, in preflight and in the guard, is a `SourceFileLoader` spec constructed from the
  exact expected in-root `__init__.py` or module `.py`, so no path hook, cached path-entry finder
  or stale loaded `__spec__` can redirect inspection or execution; namespace and non-regular
  shapes fail closed. The tree scan refuses every FIFO, socket or device node, and the preflight
  read opens without blocking or following a final link and requires a regular file. Source
  stat, read and parse failures (including `SyntaxError`) and bytecode-purge failures (including
  a directory the walk cannot enter) are bounded path-free `ModuleRefreshError`s naming only the
  error class and an errno or line; the purge precedes any module-graph change, so a partial cache
  deletion neither commits nor mutates the graph. Commit requires every fresh family entry and
  every guard-resolved module to be a `ModuleType` carrying its exact source spec, bound
  identically in `sys.modules` and on its direct parent, with a package keeping exactly its
  confined `__path__` and a module having none. A child forked while another thread runs a
  refresh restores the pre-refresh modules and recorded direct-parent bindings, removes inherited
  guards, resets the transaction record and replaces the lock; a transaction owned by the forking
  thread continues in the child. Residual: CPython's own per-module import locks held by the
  vanished thread are not reset, so re-importing exactly the modules it was initializing in that
  child is unsupported under the import-quiescent precondition. The in-process/no-sandbox threat
  model and every live, authority, deployment and capital wall are unchanged.

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
- `algua/primitives/module_refresh.py`
- `algua/primitives/module_source_scan.py`
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
- `tests/test_module_refresh.py`

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
- 2026-09-28: Addressed all 8 third-round review patches test-first (stale-bytecode and
  dependency-first closure refresh, NFC installer strings, surrogate-safe parsing, PEP 503
  distribution names, retained NFC universe, bound-first bundle inventory, streamed index-flag
  scan); the full 4,363-test root gate, ruff, mypy and all 28 import contracts pass.
- 2026-09-28: Addressed all 6 fourth-round review patches test-first (fresh, acyclic,
  all-or-nothing strategy-closure refresh with package-wide bytecode purge; surrogate-safe typed
  construction; protected refresh seam); the full 4,385-test root gate, ruff, mypy and all 28
  import contracts pass.
- 2026-09-28: Addressed all 5 fifth-round review patches test-first (import-lock-serialized
  refresh returning the fresh root, complete rollback of newly introduced modules and their
  parent bindings, symlinked source-path refusal, dictionary-read parent binding, iterative
  cycle check); the full 4,397-test root gate, ruff, mypy and all 28 import contracts pass.
- 2026-09-28: Addressed all 6 sixth-round review patches test-first (private-lock serialization
  of supported refresh/loader callers instead of the global import lock, exact parent-binding
  rollback, complete no-follow family-tree symlink refusal, raw `..` refusal, pre-discovery
  snapshot with preflight rollback, fixed-size lazy-chain integration case); the full 4,413-test
  root gate, ruff, mypy and all 28 import contracts pass.
- 2026-09-28: Addressed all 8 seventh-round review patches test-first (import-guard-recorded
  direct-parent rollback without whole-namespace copies, importable non-source entry refusal,
  exact-root confinement of fresh family imports, same-thread nested-refresh refusal, fork-safe
  refresh lock, bounded tree-inspection `OSError` mapping, deterministic deadlock handshake) and
  carved the protected `module_source_scan.py`; the full 4,436-test root gate, ruff, mypy and all
  28 import contracts pass.
- 2026-09-28: Addressed all 6 eighth-round review patches test-first (exact in-root spec
  construction, fork-child transaction recovery, commit-time fresh-graph validation, bounded
  source stat/read/parse and bytecode-purge errors, non-regular node refusal); the full
  4,468-test root gate, ruff, mypy and all 28 import contracts pass.
