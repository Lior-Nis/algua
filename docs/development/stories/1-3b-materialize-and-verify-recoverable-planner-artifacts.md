---
baseline_commit: 24c4a2bc138822c5a74ea0ed91d6d5f03c402d97
---

# Story 1.3b: Materialize and verify recoverable planner artifacts

Status: in-progress

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
- [x] [Review][Patch] Canonicalize the selected package root before preflight and import so a
  relative parent search entry cannot change meaning if refreshed code changes the process working
  directory
  [algua/primitives/module_source_scan.py:111]
- [x] [Review][Patch] Validate immutable expected source-spec facts and corresponding module
  metadata at commit so in-place mutation of the `ModuleSpec` handed to executed code cannot retain
  a forged loader, origin or package search location
  [algua/primitives/module_refresh.py:190]
- [x] [Review][Patch] Preserve refresh serialization when the refreshing thread itself forks so a
  second thread in the child cannot acquire a replacement lock and enter the still-active partial
  transaction before its owner commits or rolls back
  [algua/primitives/module_refresh.py:72]
- [x] [Review][Patch] Pin the exact concrete `ModuleSpec` and original `SourceFileLoader` identity
  and state handed to the import system so executed code cannot commit a forged spec subclass or a
  same-shaped replacement loader with altered behavior
  [algua/primitives/module_commit_check.py:62]
- [x] [Review][Patch] Require every committed family entry and direct parent to be exactly
  `ModuleType`, not a subclass whose attribute access can conceal a divergent `__path__` or other
  import metadata from dictionary-based validation
  [algua/primitives/module_commit_check.py:51]
- [x] [Review][Patch] Refuse a second resolution of the same family name in one transaction so a
  strategy cannot retain a stale first module object while a replacement is the object certified
  in `sys.modules` and on its parent
  [algua/primitives/module_refresh.py:180]
- [x] [Review][Patch] Snapshot and validate the behavior-relevant `ModuleSpec` bookkeeping state,
  including loader state, location state, uninitialized submodules and the required final
  initialization flag, so a committed spec cannot retain forged import machinery state
  [algua/primitives/module_commit_check.py:21]
- [x] [Review][Patch] Latch any duplicate family-resolution attempt as a transaction-wide
  violation so executed code cannot catch the refusal, restore apparently valid bindings and still
  commit a transaction that violated the one-resolution invariant
  [algua/primitives/module_refresh.py:173]
- [x] [Review][Patch] Distinguish an absent `submodule_search_locations` attribute from its valid
  `None` value for ordinary modules so deletion of required nullable spec metadata always refuses
  commit and rolls back
  [algua/primitives/module_commit_check.py:99]
- [x] [Review][Patch] Check absent-only module metadata by key membership rather than comparing
  with the importable `_ABSENT` sentinel so executed family code cannot install that sentinel as
  a present `__cached__` or `__path__` value and satisfy an absence requirement
  [algua/primitives/module_commit_check.py:105]
- [x] [Review][Patch] Correct the checked thirteenth-round finding's implementation pointer to the
  membership guard that now lives at `module_commit_check.py:105`
  [docs/development/stories/1-3b-materialize-and-verify-recoverable-planner-artifacts.md:386]
- [x] [Review][Patch] Make rollback presence-aware instead of using the importable `_ABSENT`
  sentinel as dictionary state: detect `sys.modules` changes by key membership and snapshot direct
  parent bindings as an explicit present bit plus value, so a sentinel-valued entry cannot survive
  rollback or erase a pre-existing binding
  [algua/primitives/module_refresh.py:176]
- [x] [Review][Patch] Make the fourteenth-round pointer correction internally consistent by naming
  the actual membership guard at `module_commit_check.py:105`, not the `optional` tuple at line 104
  [docs/development/stories/1-3b-materialize-and-verify-recoverable-planner-artifacts.md:391]
- [x] [Review][Patch] Publish bundle and environment directories with an atomic no-replace primitive
  so the final operation itself cannot replace a destination created after the existence check;
  verify an `EEXIST` winner and fail closed when no supported no-replace primitive exists
  [algua/registry/artifact_store.py:165]
- [x] [Review][Patch] Make bundle and environment tree walks fail closed on traversal errors and
  enforce the protected bundle file-count bound while inventorying, before memory can grow past it
  [algua/registry/artifact_store.py:82]
- [x] [Review][Patch] Make installed-environment inventory complete and resource-bounded: reject
  uninventoried empty directories, cap file count/per-file/aggregate bytes, and read distribution
  metadata through an explicit byte bound
  [algua/registry/planner_environment_inventory.py:88]
- [x] [Review][Patch] Run the final interpreter probe without installed startup hooks, bound both
  output streams, explicitly expose only the environment import roots needed for the `algua`
  check, and reject non-object or non-canonical probe JSON through `EnvironmentIncompatible`
  [algua/registry/planner_environment_inventory.py:146]
- [x] [Review][Patch] Bound uv version/create/sync output and normalize launch-time `OSError`
  failures so no provisioning command can exhaust memory or escape the typed incompatibility
  boundary despite its timeout
  [algua/registry/planner_environment.py:49]
- [x] [Review][Patch] Classify retryable environment unavailability only from positive evidence of
  a locked compatible-wheel acquisition outage using bounded stdout and stderr; ambiguous timeout,
  checksum, TLS/configuration and malformed wheel-URL failures must default to non-retryable
  incompatibility
  [algua/registry/planner_environment.py:97]
- [x] [Review][Patch] Exercise real offline final-locator verification without checkout, Git, uv or
  network, plus per-digest-lock and post-publication final-verification fault boundaries, instead
  of satisfying the mandatory matrix through verifier stubs
  [tests/test_frozen_artifact_offline.py:187]
- [x] [Review][Patch] Reject non-normative `.pth` path entries and executable startup hooks before
  accepting an installed environment, so the `-I -S` probe cannot certify an environment whose
  ordinary site startup would expose an external or checkout `algua`
  [algua/registry/planner_environment_inventory.py:37]
- [x] [Review][Patch] Replace `os.walk` inventory traversal with a fail-closed streaming walk that
  bounds every discovered entry and retained directory before allocation can exceed the protected
  limit, including empty-directory fanout and one directory with an enormous child list
  [algua/primitives/bounded_walk.py:50; consumers algua/registry/artifact_store.py:31,
  algua/registry/environment_store.py:34, algua/registry/planner_environment_inventory.py:154]
- [x] [Review][Patch] Require every locked registry package to have canonical non-empty `name` and
  `version` identities and make locked-wheel extraction total, so malformed committed locks stay
  inside the non-retryable `EnvironmentIncompatible` boundary instead of raising `KeyError`
  [algua/registry/planner_environment.py:111]
- [x] [Review][Patch] Validate each environment file's canonical bounded relative path before
  hashing its content, so an overlong or malformed path cannot force a large read before refusal
  [algua/registry/planner_environment_inventory.py:159]
- [x] [Review][Patch] Correct the offline-evidence finding's implementation pointer to the real
  end-to-end acceptance test in `test_frozen_artifact_offline.py`
  [docs/development/stories/1-3b-materialize-and-verify-recoverable-planner-artifacts.md:425]
- [ ] [Review][Patch] Pin a `uv sync` argv that the repository's uv 0.9.26 accepts: `--no-env-file`
  is a `uv run` option, so the normative sync argv exits with `unexpected argument
  '--no-env-file'` and real locked provisioning can never succeed. Correcting it changes the keyed
  sync argv, the environment-key golden vector and the normative companion together, so it needs
  its own reviewed contract change rather than an incidental patch
  [algua/registry/planner_environment.py:47]

#### Review round against `7877b8a` (2026-09-28)

- [x] [Review][Patch] Enforce `MAX_FILE_BYTES` while streaming each bundle file's digest, so a
  file that grows after its `lstat` size check fails closed instead of being read without bound
  [algua/registry/artifact_store.py:112]
- [x] [Review][Patch] Reject locked wheel URLs containing control or other non-printable
  characters and require an exact canonical URL round-trip before insertion and duplicate
  detection [algua/registry/planner_environment.py:145]
- [x] [Review][Patch] Canonicalize every environment entry, directories included, before retaining
  or using it, so a malformed directory path fails before its subtree is read
  [algua/registry/planner_environment_inventory.py:171]
- [x] [Review][Patch] Close every stacked `scandir` iterator even when one close raises, preserving
  the correct active or cleanup exception [algua/primitives/bounded_walk.py:66]
- [x] [Review][Patch] Translate dangling or unreadable required interpreter-link failures to
  `EnvironmentIncompatible` instead of leaking raw `OSError`/`RuntimeError`
  [algua/registry/planner_environment_inventory.py:177]
- [x] [Review][Patch] Enforce `MAX_PYVENV_CFG_BYTES` before hashing `pyvenv.cfg` and avoid a
  needless second bounded read of the same bytes
  [algua/registry/planner_environment_inventory.py:190]
- [x] [Review][Patch] Synchronize the protected Story 1.3b companion with `MAX_BUNDLE_DIRECTORIES`
  (10,000), `MAX_ENVIRONMENT_DIRECTORIES` (25,000), `MAX_PYVENV_CFG_BYTES` (64 KiB) and the
  uv 0.9.26 startup-file policy, including version and retention behavior for recorded
  environments; documentation alignment only, no authority or scope change
  [docs/development/specs/spec-story-1-3b-artifact-environment-contract/artifact-environment-contract.md]
- [x] [Review][Patch] Repair the stale checked `strict_walk.py` pointer to `bounded_walk.py` and its
  consumer locations
  [docs/development/stories/1-3b-materialize-and-verify-recoverable-planner-artifacts.md:436]
- [x] [Review][Defer] The normative `uv sync` argv's `--no-env-file` flag is rejected by uv 0.9.26.
  Pre-existing, already tracked by the open finding above and intentionally outside this round:
  `SYNC_FLAGS`, the golden vectors and the normative sync argv are unchanged. Recorded in
  `docs/development/stories/deferred-work.md` [algua/registry/planner_environment.py:47]
- Dismissed (not implemented): fd/`openat` hardening against hostile same-UID path replacement;
  the normative contract explicitly accepts same-UID hostile replacement as the no-sandbox
  residual.

#### Review round against `635e1f6` (2026-09-28)

- [x] [Review][Patch] Validate each locked wheel URL against one explicit canonical HTTPS identity
  (lowercase scheme and hostname, no credentials or fragment, no whitespace or non-ASCII/control
  characters, valid non-default port, canonical percent escapes and authority/path/query form) and
  key duplicate-wheel ownership by that identity, because the `urlunsplit(urlsplit(url))` check
  accepts uppercase-host, default-port, whitespace, malformed-escape and invalid-port aliases
  [algua/registry/planner_environment.py:148]
- [x] [Review][Patch] Make bounded-walk cleanup exhaustive: retain the first (deepest) failure of
  any kind, keep closing every remaining listing, preserve an active traversal error, and do not
  drop an exhausted listing before a failed close has had a cleanup retry
  [algua/primitives/bounded_walk.py:40]
- [x] [Review][Patch] Keep a consumer's typed validation error primary when closing the walk also
  fails, through one shared scoped-walk seam that every production `bounded_walk` consumer uses,
  while still closing every listing [algua/primitives/bounded_walk.py:50]
- [x] [Review][Patch] Accept every valid HTTPS wheel URL supported by the pinned uv acquisition
  path while deriving a separate canonical identity for duplicate ownership, rather than rejecting
  authoritative query-bearing, IPv6 or RFC sub-delimiter URLs
  [algua/registry/planner_environment_lock.py:33]
- [x] [Review][Patch] Give every non-exhausted stacked directory listing a bounded cleanup retry
  after a close fails before releasing it, without stopping cleanup of the remaining listings
  [algua/primitives/bounded_walk.py:41]
- [x] [Review][Patch] Select a recorded cleanup interrupt by explicit presence rather than exception
  truthiness, so a falsey `BaseException` subclass cannot be demoted
  [algua/primitives/bounded_walk.py:60]
- [x] [Review][Patch] Surface a listing close that raises `GeneratorExit` during generator
  abandonment instead of allowing `generator.close()` to suppress the cleanup failure
  [algua/primitives/bounded_walk.py:95]
- [x] [Review][Patch] Translate ordinary scoped-walk close failures after otherwise successful
  traversal into each consumer's stable artifact/environment error taxonomy while preserving typed
  consumer refusals and real interrupts [algua/registry/artifact_store.py:100]
- [x] [Review][Patch] Replace the regex-only direct-call guard with AST/import-aware enforcement so
  an aliased or module-qualified `bounded_walk` call cannot bypass the required `scoped_walk` seam
  [tests/primitives/test_scoped_walk.py:99]

#### Review round against `529f6d5..463cb7d` (2026-09-28)

- [ ] [Review][Patch] Keep an arbitrary digit-string port, including one longer than Python's
  4,300-digit integer conversion limit, inside `EnvironmentIncompatible` instead of a raw
  `ValueError` (Todoist 6hfP4pGmv6GfHhCp) [algua/registry/planner_environment_lock.py:81]
- [ ] [Review][Patch] Translate only a `WalkCleanupError` that originates from scoped-walk
  cleanup at the bundle store, environment store and inventory boundaries, so a consumer
  operation that itself raises one stays primary and is not mislabeled (Todoist
  6hfP4pM7xgQqWXpG) [algua/registry/artifact_store.py:45]
- [ ] [Review][Patch] Canonicalize the wheel URL in a uv failure report and compare its identity
  with an identity-indexed owner derived from the raw lock, keeping the exact single-report and
  cause rules, so uv's normalized spelling of an accepted raw URL is still recognized
  (Todoist 6hfP4pQC2Mvgq32p) [algua/registry/planner_environment_outage.py:68]
- [ ] [Review][Patch] Follow safe simple assignment propagation and rebinding of imported walk
  module/function references in the scoped-walk repository guard, without false positives;
  dynamic `__import__`/`vars` access stays out of scope (Todoist 6hfP4pQr5H8cM3qp)
  [tests/primitives/test_scoped_walk.py:172]
- [ ] [Review][Patch] Surface an inner-walk cleanup failure when an entered `scoped_walk`
  context-manager generator is itself abandoned, instead of letting `generator.close()` suppress
  it, while preserving consumer-error and interrupt precedence (Todoist 6hfP4pVJxJ2jhRhp)
  [algua/primitives/bounded_walk.py:142]

#### Review round against `9da4fd4` (2026-09-28)

- [ ] [Review][Patch] Conservatively merge possible binding states across `if`/`while`,
  zero-or-more-iteration `for`/`async for`, `try` handlers/`else`/`finally` and `match` cases,
  so the scoped-walk guard flags a raw walk whenever any feasible path keeps or reaches the
  imported alias and no branch inherits a sibling's impossible-path state; reproduced false
  negatives: conditional, `try` and `match` rebinds and a zero-iteration loop (Todoist
  6hfP9ww7Mx3rcx8p) [tests/primitives/test_scoped_walk.py:394]
- [ ] [Review][Patch] Model Python function binding and call timing for simple static cases: a
  call made before an enclosing rebind still reaches the alias, and a name that is a lexical
  local of its function (assignment, import, loop/`with`/`except`/match target anywhere in the
  body) is never resolved to an enclosing binding, honoring `global`/`nonlocal`; no arbitrary
  call-graph soundness is claimed (Todoist 6hfP9wrVj8P5QG9p)
  [tests/primitives/test_scoped_walk.py:271]
- [ ] [Review][Patch] Preserve right-hand-side-before-target evaluation and propagate every safe
  simple alias: self-assignment, chained plain assignment (`a = b = w`), annotated
  self-assignment, alias-producing walrus and `type` alias target rebinding, retaining every
  unrelated-rebind negative (Todoist 6hfP9wwJr96P4fqG) [tests/primitives/test_scoped_walk.py:375]
- [ ] [Review][Patch] Inspect executable parameter/return and annotated-assignment annotations,
  honoring postponed `from __future__ import annotations` evaluation, keep a lambda deferred
  inside a comprehension in that comprehension's lexical environment, and bind a comprehension
  walrus in the containing scope (Todoist 6hfP9x2F8GCg8Wmp)
  [tests/primitives/test_scoped_walk.py:316]

#### Review findings against `47062b9` (2026-09-28)

- [ ] [Review][Patch] Terminate analyzed loop paths at `break`/`continue`, carry those transfers
  through enclosing `finally` suites, and exclude impossible zero-iteration/`else` paths for
  statically unconditional loops (Todoist 6hfPJjCg4JQrc6Rp)
  [tests/primitives/test_scoped_walk.py:321]
- [ ] [Review][Patch] Inspect executable match-pattern expressions and preserve exact capture,
  guard-fallthrough and irrefutable-pattern state without retaining impossible aliases (Todoist
  6hfPJj6JjXppmjpG) [tests/primitives/test_scoped_walk.py:442]
- [x] [Review][Defer] Model exception-suppressing `with`/`async with` flow so an exception raised
  before rebinding can preserve a reachable imported alias (Todoist 6hfPJj8H8RXq3vmG)
  [tests/primitives/test_scoped_walk.py:517] — deferred, pre-existing
- [x] [Review][Defer] Honor simple class-body `global` writes to the containing module while
  keeping ordinary class locals isolated (Todoist 6hfPJj9J9jQVvpFG)
  [tests/primitives/test_scoped_walk.py:486] — deferred, pre-existing

#### Review findings against `762f984` (2026-09-28)

- [ ] [Review][Patch] Classify statically constant `while` conditions so truthy constants do not
  retain impossible zero-iteration/`else` paths and falsey constants do not execute an unreachable
  body (Todoist 6hfPWRXC8MMwMrcG) [tests/primitives/test_scoped_walk.py:419]
- [ ] [Review][Patch] Detect a deferred function's raw walk when it is called before a later
  enclosing alias rebind, rather than analyzing the body only against the scope's final bindings
  (Todoist 6hfPWX5RW3Hpvw7p) [tests/primitives/test_scoped_walk.py:331]
- [ ] [Review][Patch] Join match-guard named-expression bindings only onto the guard-success path,
  preserving the feasible guard-failure state without retaining a stale alias after success
  (Todoist 6hfPWX55FXhj52Jp) [tests/primitives/test_scoped_walk.py:486]
- [ ] [Review][Patch] Cover nested and conditional `finally` transfer forwarding, a continue-only
  unconditional loop, and an actual deferred-function call from the endless-loop fixture
  (Todoist 6hfPWRXGjPGr9qgp) [tests/primitives/test_scoped_walk.py:833]

#### Review findings against `3b80ed9` (2026-09-28)

- [ ] [Review][Patch] Classify every side-effect-free literal and unary-literal `while` condition,
  not only `ast.Constant`, so unreachable bodies and impossible zero-iteration exits do not enter
  the binding join (Todoist 6hfPWRXC8MMwMrcG) [tests/primitives/test_scoped_walk.py:275]
- [ ] [Review][Patch] Preserve separate truthy and falsey states through `and`/`or` and conditional
  expressions so match-guard success cannot inherit failure-only aliases and later cases receive
  only feasible false-guard bindings (Todoist 6hfPfM2QXfxpvjvp)
  [tests/primitives/test_scoped_walk.py:416]
- [ ] [Review][Patch] Give lambdas the bounded same-scope call identity used for plain definitions,
  so a call before a later enclosing rebind cannot hide a raw walk (Todoist 6hfPfM48RRh6Mc2G)
  [tests/primitives/test_scoped_walk.py:396]
- [ ] [Review][Patch] Record import-time calls from class bodies against their enclosing bindings
  while keeping class-local names isolated (Todoist 6hfPfM3X26Qxj9FG)
  [tests/primitives/test_scoped_walk.py:573]
- [ ] [Review][Patch] Model generator advancement and coroutine awaiting at execution time, or use
  a bounded conservative state model that cannot miss an alias becoming raw after object creation
  (Todoist 6hfPfM2HWPqcrXCG) [tests/primitives/test_scoped_walk.py:439]
- [ ] [Review][Patch] Make the callback/endless-loop fixture prove the mechanism named by the test,
  rather than passing only because the loop fallback retains the same alias independently of the
  opaque callback (Todoist 6hfPfJxg8X2mrGpp) [tests/primitives/test_scoped_walk.py:817]

#### Review findings against `bdaf792` (2026-09-28)

- [ ] [Review][Patch] Complete the side-effect-free literal evaluator for unary booleans and
  nested safe literals, so every admitted statically true/false loop condition removes its
  impossible body or exit instead of retaining a false-positive alias path (Todoist
  6hfPWRXC8MMwMrcG) [tests/primitives/test_scoped_walk.py:276]
- [ ] [Review][Patch] Keep lazy-object existence and accumulated bindings correlated across
  mutually exclusive branches, so a generator or coroutine created only on a safe path cannot
  inherit a raw alias reached only on a sibling path (Todoist 6hfPmMP337jqHpGG)
  [tests/primitives/test_scoped_walk.py:446]
- [ ] [Review][Patch] Propagate later caller-scope states into a lazy object returned by a followed
  same-scope factory, so a generator, generator expression or coroutine advanced after the caller
  alias becomes raw cannot evade the guard (Todoist 6hfPmMMPmhrfqvjG)
  [tests/primitives/test_scoped_walk.py:426]
- [ ] [Review][Patch] Seed a nested class body from its lexical/module bindings rather than the
  outer class namespace, which Python does not close over, while preserving ordinary class-local
  isolation (Todoist 6hfPmMQ6h2xXcJ8G) [tests/primitives/test_scoped_walk.py:720]
- [ ] [Review][Patch] Thread only a comprehension filter's truthy state into later filters and the
  element, so a rebind required for evaluation cannot be merged with an impossible false path
  (Todoist 6hfPmMMpfmMvppxG) [tests/primitives/test_scoped_walk.py:495]

#### Review findings against `48f9934` (2026-09-28)

- [x] [Review][Patch] Keep a lazy body's entry state path-correlated instead of joining the
  enclosing scope's final bindings into every generator and coroutine, so a raw alias reached
  only on a sibling path where the object does not exist cannot create a false positive (Todoist
  6hfPmMP337jqHpGG) [tests/primitives/test_scoped_walk.py:446]
- [x] [Review][Patch] Preserve a returned lazy object's captured closure environment separately
  from later caller bindings, so a factory-local shadow cannot be overwritten by a caller-local
  rebind while genuine globals still retain Python's late-binding behavior (Todoist
  6hfPmMMPmhrfqvjG) [tests/primitives/test_scoped_walk.py:535]
- [x] [Review][Patch] Propagate returned lazy objects from feasible return states rather than a
  syntax-only name-to-one-node scan: ignore unreachable returns, resolve simple aliases and
  conditional results, and retain every definition that can reach a return (Todoist
  6hfPvjHrFwhGvmxp) [tests/primitives/test_scoped_walk.py:535]
- [x] [Review][Patch] Keep lazy objects created only as discarded comprehension temporaries
  confined to comprehension evaluation while preserving objects that actually escape in the
  result or through a modeled side effect (Todoist 6hfPvjP4PqJ74J2G)
  [tests/primitives/test_scoped_walk.py:573]
- [x] [Review][Patch] Evaluate a class comprehension's first iterable in class state and its body
  in lexical/module state, matching Python's nested comprehension scope so class-local shadows
  cannot hide a raw module alias (Todoist 6hfPvjR73hhFw8fG)
  [tests/primitives/test_scoped_walk.py:573]
- [x] [Review][Patch] Separate per-object lazy environments from the shared bindings map and cache
  repeated syntax summaries where needed, preventing the measured superlinear scan growth as live
  lazy objects accumulate (Todoist 6hfPvjhRQ6jVqfgp)
  [tests/primitives/test_scoped_walk.py:516]
- [x] [Review][Patch] Classify long unary-`not` chains iteratively or with an explicit safe bound,
  so a valid module cannot crash repository hygiene with `RecursionError` (Todoist
  6hfPWRXC8MMwMrcG) [tests/primitives/test_scoped_walk.py:305]

#### Review findings against `962fb2e` (2026-09-28)

- [x] [Review][Patch] Preserve a post-scope creation state for externally or cross-scope invoked
  lazy functions without reintroducing impossible sibling-path states into objects already
  created in the defining scope (Todoist 6hfQ8rC7977WmvXp)
  [tests/primitives/test_scoped_walk.py:558]
- [x] [Review][Patch] Restrict Boolean return propagation to operands that can actually be the
  expression result, including the always-truthy identity of generator and coroutine objects
  (Todoist 6hfQ933hvw6PVvVG) [tests/primitives/test_scoped_walk.py:1053]
- [x] [Review][Patch] Include assignment-expression targets owned by a lambda when deriving its
  closure locals, without collecting targets owned by nested scopes (Todoist 6hfQ932C86Xm2fGp)
  [tests/primitives/test_scoped_walk.py:380]
- [x] [Review][Patch] Treat lazy objects passed to unknown named calls as escaping unless the
  callee is explicitly proven non-retaining, so a callee cannot stash an object until after a raw
  rebind (Todoist 6hfQ94C4qGggxjWp) [tests/primitives/test_scoped_walk.py:436]
- [x] [Review][Patch] Make the lazy-flow complexity regression measure actual object-state copy
  work rather than the number of binding keys, and remove the remaining quadratic live-object
  update behavior (Todoist 6hfQ949RRWWvrqvp) [tests/primitives/test_scoped_walk.py:635]
- [x] [Review][Patch] Honor `global` and `nonlocal` declarations when deriving the actual free
  variables of a returned lazy object, so a declaration cannot be mistaken for a safe factory
  local (Todoist 6hfQ9Rm5JgMc3Mrp) [tests/primitives/test_scoped_walk.py:651]
- [x] [Review][Patch] Propagate relevant declared writes from a followed same-scope helper call,
  with recursion and complexity bounds, before checking subsequent direct or lazy walk use
  (Todoist 6hfQ9Rqc6X7wf35G) [tests/primitives/test_scoped_walk.py:664]
- [x] [Review][Patch] Distinguish constructing or truth-testing a temporary lazy object from
  advancing it, so `bool(gen())` does not execute the generator body (Todoist
  6hfQ9VrR8mjGPv4p) [tests/primitives/test_scoped_walk.py:626]
- [x] [Review][Patch] Traverse deeply nested conditional lazy-value expressions iteratively under
  an explicit node bound rather than crashing repository hygiene with `RecursionError` (Todoist
  6hfQ9Rq8R3RWMr3G) [tests/primitives/test_scoped_walk.py:389]
- [x] [Review][Patch] Project return-summary cache keys onto names relevant to the summarized
  function so unrelated accumulated definitions cannot cause quadratic cache growth (Todoist
  6hfQ9RmmQ7HwXJ3p) [tests/primitives/test_scoped_walk.py:643]

#### Review findings against `3c338bc` (2026-09-28)

- [ ] [Review][Patch] Make alive-state materialization genuinely near-linear; current flattened
  caches every growing frozenset prefix, measured about 1.38 GB at 8,000 objects (Todoist
  6hfQqH2m97ch7g8p) [tests/primitives/test_scoped_walk.py:260]
- [ ] [Review][Patch] Prove non-retaining builtins by binding and argument position; shadowed
  bool/next/any/all retain, and builtin next may return its default argument (Todoist
  6hfQqGxWrc4QpJqp) [tests/primitives/test_scoped_walk.py:585]
- [ ] [Review][Patch] Drive deferred cross-scope discovery to a bounded fixpoint; fixed two
  passes miss reverse-ordered call chains (Todoist 6hfQqH5p7CXJJCvG)
  [tests/primitives/test_scoped_walk.py:714]
- [ ] [Review][Patch] Resolve cross-scope calls from the callee's defining lexical environment
  rather than filtering the current caller state (Todoist 6hfQqH3pfGW6W7Vp)
  [tests/primitives/test_scoped_walk.py:888]
- [ ] [Review][Patch] Include lambda bodies in cross-scope call propagation with their locals
  (Todoist 6hfQqH2qxwxhVWHG) [tests/primitives/test_scoped_walk.py:725]
- [ ] [Review][Patch] Apply declared global/nonlocal effects to their real lexical owner so
  caller locals are not overwritten (Todoist 6hfQqGxfgHcvPFXp)
  [tests/primitives/test_scoped_walk.py:888]
- [ ] [Review][Patch] Compose transitive declared helper effects safely with recursion-bounded/
  fixpoint summaries (Todoist 6hfQqH53pjrMWgxp) [tests/primitives/test_scoped_walk.py:857]
- [ ] [Review][Patch] Bind positional, keyword, and default arguments in helper effect summaries
  (Todoist 6hfQqH4P4QC6r8Rp) [tests/primitives/test_scoped_walk.py:824]
- [ ] [Review][Patch] Propagate definite clearing in helper effects rather than retaining a stale
  raw alias (Todoist 6hfQqGxXJRvpH62G) [tests/primitives/test_scoped_walk.py:881]
- [ ] [Review][Patch] Fail closed when lazy-value traversal hits MAX_IFEXP_CHAIN instead of
  returning partial escape results (Todoist 6hfQqH5HcRMg38jp)
  [tests/primitives/test_scoped_walk.py:549]
- [ ] [Review][Patch] Exclude all function-owned locals, not only parameters, from return/effect
  summary cache entries (Todoist 6hfQqH4rMMvxq9JG) [tests/primitives/test_scoped_walk.py:824]

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
- Ninth review round: 24 new cases. Patch 1 failed red for a relative `sys.path` entry and a
  relative parent `__path__` entry: refreshed code that changed directory mid-import made the
  guard resolve, and execute, a same-shaped tree under the new directory (its marker was
  written). A `./src` entry (pins canonical normalization) and a vanished working directory (pins
  the bounded `FileNotFoundError` refusal now that every relative entry reads the working
  directory) were added after green to kill surviving mutations. Patch 2 failed red for 16 of 17
  in-place mutation cases (loader subclass type, loader name and path, spec name, origin, cache,
  the root's own origin, a package spec's rebound search locations, module `__name__`,
  `__package__` for a module and a package, `__loader__`, `__file__` changed, removed or an equal
  `str` subclass, `__cached__`); a module spec gaining search locations was already refused only
  because the old check derived its expected `__path__` from the live spec, and is now caught by
  the recorded facts. An in-place `str`-subclass element of a package `__path__` was added after
  green to pin element-wise exact typing. Every case proves complete rollback of family entries
  and parent bindings. Patch 3 failed red in both outcomes (commit and rollback): a second thread
  in the owner-forked child took the replacement lock mid-transaction (child exit 41). The probe
  is deterministic: a non-blocking acquire from the second thread proves the lock is held
  mid-transaction, and a blocked `serialized_import` records that no transaction was active when
  it finally entered; five repeated runs pass. Twenty-two mutations were run and all are killed:
  freezing and normalization of the root and the bounded working-directory read; each of the
  thirteen recorded spec/module fact checks, exact typing, element-wise list typing and the whole
  facts check; and each of the three fork branches (the owner keeps its lock, a vanished owner and
  no transaction still reset). Loader name and path are read with `getattr`, so a deleted
  attribute also fails closed. `module_refresh.py` stays at 299 lines, under the size-ratchet
  floor, by tightening prose rather than raising a pin. All 25 bundled strategies still refresh
  twice through `_reload_strategy_closure`.
- Tenth review round: 7 new cases. The fix pushed `module_refresh.py` to 306 lines, over the
  size-ratchet floor, so the commit-time validation (recorded facts, exact binding and metadata
  checks) first moved unchanged into the protected `module_commit_check.py` (CODEOWNERS and
  hygiene list) with the full refresh suite green. On that carved baseline 5 of 6 initial cases
  failed red: a `__class__` swap of a member's spec and of the root's own spec to a `ModuleSpec`
  subclass, a same-shaped `SourceFileLoader` (same name and path) bound to both the spec and
  `__loader__`, and an instance `get_code` or `get_data` attribute injected onto the original
  loader all committed. A loader `__class__` swap to a subclass was already refused by the exact
  loader type and is kept as a regression; a same-shaped loader bound only to the spec was added
  after green to pin the spec-loader identity independently of `__loader__`. Every case proves
  complete rollback of family entries and parent bindings. Six mutations were run and all are
  killed: dropping the exact spec class, the spec-loader identity, the module-loader identity,
  the exact loader class or the loader-state snapshot check, and re-reading the loader from the
  live spec instead of the recorded original. `module_refresh.py` is 246 lines and
  `module_commit_check.py` 90. The full 4,499-test root gate (`-p no:randomly`), ruff, mypy and
  all 28 import contracts pass.
- Eleventh review round: 28 new cases; all 27 refusal cases fail on the baseline (`ee88840`). Patch
  1 (exact `ModuleType`): 6 cases failed red (a member, the root, the package and the direct
  parent swapped through `__class__` to a `ModuleType` subclass, a direct parent replaced by a
  subclass instance, and a package subclass whose `__getattribute__('__path__')` exposes an
  alternate tree). A standalone probe on the baseline confirmed that last case commits and a
  post-commit lazy import of a new family name then loads from the alternate tree; the fixed
  refresh rolls back and the restored family cannot reach it. Patch 2 (one resolution per name):
  3 cases failed red (a helper imported, its entry deleted and re-imported through
  `importlib.import_module` or `__import__`, and a subpackage member likewise), each committing a
  stale first object beside the certified replacement; a case that swallows the refusal (it is an
  `ImportError`) was added after green to prove the commit check still rejects the handed-out,
  unbound module. A positive case (distinct names including a subpackage member, a repeated probe
  for an absent name, and a second transaction over the same names) pins that only a second
  hand-out is refused; its probe first used `from . import`, whose plain `ImportError` hid a
  surviving mutation, and now uses `importlib.import_module` so the `ModuleNotFoundError`-only
  handler exposes it. Patch 3 (spec bookkeeping): 14 cases failed red (loader state set, removed
  or set on the root's own spec; `has_location` cleared, `_set_fileattr` an equal `int` or
  removed; a forged pending submodule on a member or the package, the package's list replaced,
  the list removed; `_initializing` stuck `True`, an equal `0` or removed, and the package's stuck
  `True`); a removed `_cached` or `origin` escaped as a raw `AttributeError`, so spec fields are
  now read from the instance dictionary (2 cases); an empty `list` subclass with a lying
  `__contains__` was added after green because the package-replacement case is caught by the
  original list's emptiness rather than identity. Fifteen mutations were run and all are
  killed: each of the two exact-type checks (member, holder); dropping the second-resolution
  check, keying it on recorded parent bindings, counting a miss as a resolution and sharing the
  record across transactions; and for the bookkeeping, dropping the loader-state or location
  check, equality instead of identity for the pending list, dropping its emptiness, a falsy
  instead of exact `False` initialization flag, requiring the pre-load absence instead of the
  completed state (21 ordinary refreshes fail), reading `cached` or `origin` through the
  attribute, and recording a copy of the pending list. `module_refresh.py` is 252 lines and
  `module_commit_check.py` 111. The full 4,527-test root gate (`-p no:randomly`), ruff, mypy and
  all 28 import contracts pass.
- Twelfth review round: 18 new cases; 8 fail on the baseline (`ef3f280`). Patch 1 (latched
  duplicate resolution): 3 cases failed red. Refreshed code that imports a helper (or a
  subpackage member), drops its entry, catches the `ModuleRefreshError` from the second
  resolution and then reinstates the FIRST object in `sys.modules` and on its direct parent
  committed on the baseline; both now roll back completely. A positive case proves the latch
  does not outlive its transaction: after that rolled-back refresh, a refresh of the same names
  from clean source commits with the fresh helper bound on its parent; the existing distinct-name,
  repeated-miss and second-transaction case stays green. Patch 2 (missing versus `None`): an audit
  of every commit-check lookup found the reported ambiguity plus a second one. Deleting an
  ordinary module's spec `submodule_search_locations`, from a member or from the root's own spec,
  committed (2 cases red). With no cache tag a source spec's valid cache is `None` and its module
  has no `__cached__`, so deleting the spec's `_cached` (member or own spec) or adding a `None`
  `__cached__` also committed (3 cases red). Nine deletion regressions that already refused are
  pinned: the package's search locations, spec `name` and `loader`, and the module's `__spec__`,
  `__loader__`, `__name__`, `__package__`, `__cached__` and the package `__path__`. Every refusal
  case proves complete rollback of family entries and parent bindings, and a positive case proves
  a cache-less family still refreshes, with a present `None` `_cached` and no `__cached__`. Seven
  mutations were run and all are killed: never setting the latch, dropping the commit-time latch
  check, clearing the latch on a duplicate, reverting the search-locations or spec `_cached`
  default to `None`, defaulting `__cached__` to `None`, and expecting `__cached__` to equal the
  recorded `None` cache instead of being absent. The remaining lookups now also default to
  `_ABSENT` for consistency; their expected values are never `None`, so reverting any of them is
  behavior-equivalent (an equivalent mutant, not a surviving one). `module_refresh.py` is 260 lines and
  `module_commit_check.py` 120. The full 4,545-test root gate (`-p no:randomly`), ruff, mypy and
  all 28 import contracts pass.
- Thirteenth review round: 5 new cases, all failing on the baseline (`b3d9ef9`). With no cache
  tag, refreshed code that imports `module_commit_check._ABSENT` and installs it as a present
  `__cached__` on a member, on itself or on the package, or as a present `__path__` on a member or
  on itself, committed on the baseline (`DID NOT RAISE`); each now refuses and rolls back every
  family entry and parent binding. The two absence requirements are checked by key membership;
  present values keep the exact-type comparison. Five mutations were run and all are killed:
  dropping the membership check, restoring sentinel equality, reading membership through
  `getattr`, and fixing only `__cached__` (2 cases red) or only `__path__` (3 cases red). An
  audit of every other commit-check `_ABSENT` default found none absent-only: each compares with
  a spec, loader, module, `bool`, list, string or `None` that can never be the sentinel. The audit
  also reproduced a same-class rollback defect outside this finding: `_restore_modules` compares
  with `module_refresh._ABSENT`, so a NEW `sys.modules` entry whose value is that sentinel (inside
  or outside the family) survives a failed refresh, while a plain `object()` is dropped. It is
  left unchanged and reported for review triage. `module_commit_check.py` is 126 lines. The full
  4,550-test root gate (`-p no:randomly`), ruff, mypy and all 28 import contracts pass.
- Fourteenth review round: 24 new cases, 6 failing on the baseline (`5125a02`). Each rollback
  regression runs for a family and an external name with `None`, an ordinary object and every
  bare `object()` sentinel the refresh seam exposes (discovered at collection). On the baseline a
  failed refresh that installed `module_refresh._ABSENT` as a new family or external
  `sys.modules` entry left it behind, one that removed a pre-existing family or external entry
  valued `_ABSENT` never reinstated it, and a pre-existing `_ABSENT`-valued parent binding of the
  family package or of a new external module was deleted; the 18 other cases passed, pinning
  ordinary rollback. Changed entries are now found by key membership, then identity when present
  on both sides, and each parent binding is snapshotted as `_Binding` (parent, presence bit,
  exact value), the one type of the guard's `bindings` and of `_restore_modules`, which fork
  recovery shares. With an unused `_ABSENT` re-added all 24 cases pass; the sentinel is then
  deleted, leaving 18. Ten mutations are killed: a `None`-default lookup, a shared commit-check
  sentinel default, membership without identity, presence inferred from a `None` value at the
  snapshot or at restore, an inverted presence bit, always setting, always popping, fork recovery
  skipping the bindings (the existing fork probe fails, so no distinct fork regression is
  needed) and a literal revert to the baseline file (discovery re-adds its sentinel; 6 red). The
  thirteenth-round pointer now names the membership test (`:104` binds the absent-only keys,
  `:105` tests them); the rollback finding names the snapshot line. `module_refresh.py` is 264
  lines. The full 4,568-test root gate (`-p no:randomly`), ruff, mypy and all 28 import
  contracts pass.
- Publication review chunk 2 (baseline `932cf2e`): all 7 accepted patches first failed red.
  No-replace: the primitive was absent; a winner inserted after the existence check made the
  final `os.rename` fail `ENOTEMPTY` instead of verifying it, and an empty destination directory
  was silently replaced, for both stores. Walks: a smuggled bundle subtree verified clean, an
  environment whose `lib/` listing failed was published unsealed before failing, a writable file
  and an omitted inventory subtree went unseen; listing faults are injected at `os.scandir`, not
  through permission bits. Inventory: three empty-directory shapes were accepted and non-UTF-8
  METADATA leaked `UnicodeDecodeError`. The locked no-dev environment built with the normative
  flags (minus `--no-env-file`, see the new finding) has 23,849 files, 861.4 MiB, a 160.0 MiB
  largest file, 115.9 KiB largest METADATA and zero empty directories, so no staging
  normalization is added; the protected bounds are 100,000 entries, 512 MiB per file, 4 GiB
  aggregate and 1 MiB METADATA, and the dev superset (25,593 files, 949.7 MiB) is proven to fit.
  Probe: a malicious `.pth` executed, and uv's own `_virtualenv.pth` imported its shim and wrote
  `__pycache__` into the build environment (`-I` ignores `PYTHONDONTWRITEBYTECODE`), failing the
  next inventory. uv: launch `OSError`s and output overflow escaped raw and non-zero venv/version
  exits passed. Retry: malformed locked wheel URLs leaked raw `ValueError`; the recognizer's
  matrix (8 retryable, 22 not) is anchored on one verbatim uv 0.9.26 connection-refused report.
  Offline: a child process imports the verifier from a package copy under an audit hook that
  refuses checkout/Git access, network and every subprocess except the published interpreter,
  whose `-I -S` probe must run exactly once at its final locator. 84 mutations were run: 81 are
  killed, one is equivalent (CPython constructs `FileExistsError` from `OSError(EEXIST)`) and two
  exposed redundant guards that were removed (a METADATA size pre-check subsumed by the bounded
  read, a duplicate-key hook subsumed by canonical equality); eleven initial survivors were
  closed with new cases. The full 4,751-test root gate (`-p no:randomly`), ruff, mypy and all 28
  import contracts pass.
- Chunk-2 patch review (range `932cf2e..ae3c4aa`, 5 admitted follow-ups): every code patch first
  failed red. Lock: 29 malformed name/version identities were accepted, extraction leaked raw
  `KeyError`, `TypeError`, `UnicodeDecodeError` and `TOMLDecodeError`, and `provision_environment`
  ran uv for a lock missing `version` and then escaped `KeyError: 'version'`. Paths: six
  malformed shapes (overlong, undecodable, non-NFC, trailing space or dot, backslash) were hashed
  first and then leaked a plain `ValueError`. Traversal: `os.walk` pulled all 300 names of one
  fanout directory against bounds of about ten, and empty-directory fanout surfaced only after
  full traversal. Startup: 32 cases were accepted, including an environment whose ordinary
  startup demonstrably imports an external `algua` through a `.pth` path entry. uv 0.9.26's
  `_virtualenv.pth` is `import _virtualenv` with no trailing newline, and its 4,342-byte shim
  only patches distutils/setuptools install configuration; both are pinned by digest, a
  conformance test compares them with a real `uv venv` when uv 0.9.26 is present, and the real
  uv-built locked environment (23,849 files, 142 distributions) is accepted. New directory bounds
  are 25,000 for environments (3,147 locked, 3,302 dev superset) and 10,000 for bundles (36 in the
  source tree). Implementations: `planner_environment.py:104` (`locked_wheels`),
  `planner_environment_inventory.py:175` (path before read), `bounded_walk.py:36` and
  `planner_environment_startup.py:36`. 47 mutations were run: 44 are killed, two exposed
  redundant guards that were removed (a lock type check subsumed by the identity constructor, a
  directory path check subsumed by entry paths) and one is equivalent (the sync-failure path only
  ever sees a lock already revalidated at the key recheck); four initial traversal survivors
  were closed with new cases. The full 4,863-test root gate (`-p no:randomly`), ruff, mypy, all
  28 import contracts and `git diff --check` pass.
- Review round against `7877b8a` (8 approved patches, 1 defer): every code patch first failed
  red. Bundle digest: a file grown after its `lstat` check was streamed in full (3 MiB) and
  failed only on the descriptor comparison. Wheel URLs: 13 cases were accepted, including tab,
  newline, DEL, zero-width and line-separator characters, scheme-case, empty query/fragment and
  leading-space aliases, and two aliases that evaded duplicate detection; all 1,785 repository
  wheel URLs are printable and round-trip exactly. Directories: all five malformed directory
  shapes had their subtree listed before refusal. Walk cleanup: an abandoned walk left a stacked
  listing open when another close failed, and a failing close replaced both an active traversal
  limit and an active listing error. Interpreter links: dangling, looping and unreadable
  `python`/`python3`/`python3.12` links leaked raw `FileNotFoundError`, `RuntimeError` and
  `PermissionError` with host paths (9 cases). Small files: oversized `pyvenv.cfg` and `METADATA`
  were hashed through the 512 MiB path before refusal and accepted ones were opened twice; the
  single bounded read covers both, and the interpreter probe was carved unchanged into
  `planner_environment_probe.py` to keep the inventory module under the 300-line ratchet. Every
  bound and both startup digests written into the companion were checked against the code. 20
  mutations were run and all are killed; one initial survivor (which close failure is reported
  when two fail) was closed with a new case. The full 4,900-test root gate (`-p no:randomly`),
  ruff, mypy, all 28 import contracts and `git diff --check` pass.
- Review round against `635e1f6` (3 consolidated patches): every code patch first failed red.
  Wheel URLs: 30 malformed single URLs (uppercase or trailing-dot host, underscore or IPv6
  host, default/empty/zero/leading-zero/out-of-range/non-numeric port, whitespace, non-ASCII,
  short, non-hex, lowercase or unreserved-encoding escapes, raw `+` and `@`, query, empty or dot
  segments, backslash, angle brackets, no path, non-wheel) and 7 alias pairs (host case, default
  port, escape case, escaped unreserved, dot segment, trailing-dot host, raw versus escaped `+`)
  were accepted; all 1,785 repository wheel URLs remain their own accepted identities, and the
  lock policy moved into the protected `planner_environment_lock.py` for the 300-line ratchet.
  Walk cleanup: a `RuntimeError`, `ValueError`, `KeyboardInterrupt` or `SystemExit` close
  failure stopped cleanup (the root listing stayed open) and replaced the active traversal
  error, and an exhausted listing whose close failed before releasing was never retried.
  Consumers: with a failing close, a `RuntimeError` escaped the bundle store, published-seal
  check, sealing and inventory typed error mapping, and an `OSError` displaced each typed
  refusal. 30 mutations were run: 28 are killed and two exposed redundant guards that were
  removed (an explicit empty-segment check subsumed by the one-or-more segment pattern, a
  `?`/`#` exclusion subsumed by the segment character class). The full 4,973-test root gate
  (`-p no:randomly`), ruff, mypy, all 28 import contracts and `git diff --check` pass.
- Review follow-ups recorded in `529f6d5` (6 patches): every change first failed red. Wheel
  URLs: 58 cases failed on the previous validator; it rejected valid URLs uv can acquire
  (IPv6 authorities, queries, RFC path sub-delimiters, uppercase scheme or host, explicit or
  zero-padded default ports, lowercase or unreserved escapes, dot segments) while accepting
  numeric hosts uv's WHATWG parser reads as IPv4 aliases (`01.2.3.4`, `1.2.3`, `0x7f.0.0.1`,
  `16909060`) and a bare `.whl`, and it misreported alias pairs as malformed rather than as
  duplicate owners. Raw URLs are now kept for uv and outage evidence while ownership uses a
  separate RFC 3986 identity; all 1,785 repository URLs are their own identities. A mutation
  survivor showed the raw-character check is load-bearing (KELVIN SIGN lowercases to ASCII `k`),
  now pinned. Walk cleanup: a listing that failed to close before releasing its handle stayed
  open after abandonment, a consumer error or an active traversal error (6 cases); a falsey
  interrupt subclass was demoted to an ordinary failure; a listing close raising
  `GeneratorExit` was silently discarded by `generator.close()` (directly and after an early
  `scoped_walk` exit), leaked raw into the consumer's loop and displaced an active error.
  Taxonomy: with a failing listing close a `RuntimeError` escaped the bundle store, staging,
  published-environment check, sealing, inventory and provisioning, and `deployment verify`
  reported `frozen_descriptor_conflict` for both a bundle and an environment listing. Guard:
  the regex missed aliased, unused-alias, star, relative, module-alias-reference and `getattr`
  bypasses and falsely flagged a docstring mention and an unrelated local function; a planted
  aliased and module-qualified bypass in a production store is now caught. 53 mutations were
  run: 49 are killed (two after new cases closed initial survivors) and four exposed redundant
  guards that were removed (URL fragment, credential and leading-slash checks subsumed by the
  URL grammars, and a bare-name reference check subsumed by the import check). The full
  5,065-test root gate (`-p no:randomly`), ruff, mypy, all 28 import contracts and
  `git diff --check` pass.
- Review findings against `48f9934` (7 test-only patches, `tests/primitives/test_scoped_walk.py`):
  62 new cases (241 to 303 guard-file tests). Run against the `48f9934` analyzer, 41 fail and 21
  pass before and after as preservation proofs. Every fixture is transient raw-then-safe: the
  alias is raw only between a call and a later safe rebind, so a final-state join cannot supply
  the flag. Entry correlation (`:550`): a generator, handler, `match` case and generator
  expression made only on the branch opposite a final raw alias were flagged (4 false
  positives). A lazy body now starts only from what its objects saw. A lazy function no object
  of which is made in its scope still reads the final names, which is pinned by a positive case.
  Closures (`:376`, `:606`, `:626`): factory-local assignment, parameter, import, match capture,
  `def` and handler shadows, plus a shadow while the caller alias was raw at the call, were all
  overwritten by the caller (8 false positives). A returned object now reads its factory's own
  names from the closure. `global` names, comprehension targets and a caller-scope generator
  returned by a shadowing factory stay late-bound. Returns (`:651`, `_ReturnFlow` `:1011`): an
  alias, callee alias, conditional, `and`, walrus and same-name conditional definition missed
  the raw caller alias (6 false negatives). A return after a return, under a false literal or
  replaced by a `finally` return manufactured an object (3 false positives). Returned objects now
  come from feasible `return` states, and a `return` leaves through every `finally`.
  Comprehension temporaries (`:427`): a generator only passed to `bool`/`next`, tested by a
  filter, unpacked by `*` or passed to `any` escaped (7 false positives, including a class
  comprehension). The result, a walrus or a method argument still escapes: tuple, list, set and
  dict displays, conditional and `and` results, nested results, dict values, and a stored call
  after a same-function temporary. Class comprehensions (`:684`): a module alias the class
  shadows was missed in the element, filter, later iterable and a nested comprehension (4 false
  negatives), and a class-only alias was flagged in the body (1 false positive). The first
  iterable still reads the class. Complexity: lazy views no longer live in the shared bindings.
  `_live_generators` sums the bindings entries each statement reads, which is deterministic, not
  wall-clock. At 100/200/400 live generator expressions the sum was 10,709/41,409/162,809
  (quadratic) and is now 309/609/1,209. A local 400-object scan took 2.76 s and now takes 0.011 s
  (0.006/0.008/0.011 s at 200/300/400). `_runs_later`, comprehension escapes and return
  summaries are cached per node. Recursion (`:311`, `:366`, `:714`): under a pinned default
  recursion limit (1,000, because a plugin raises it to 3,000 under pytest), 1,100 nested `not`s
  raised `RecursionError` in `while`/`if` tests, an assignment, a function body, a filter and a
  literal display (7 cases). `_static_truth`, `_static_number`, `_condition`, `_expression` and
  `_own_nodes` now iterate, and the 1,101-`not` case keeps its body reachable. 46 mutations
  were run on copies outside the checkout, each with a 90 s cap. An earlier in-place run was
  killed by the environment before it finished; its partial output is not claimed, and the one
  mutant it left applied was restored before re-verification. In the first bounded pass, 38
  mutants failed an assertion, 7 survived and one (a views-in-bindings mutant that also copied
  shadow keys) timed out. Seven new cases closed the survivors: a closure snapshot at the call,
  closure names applied to a caller generator, a literal `if`/`and` return, per-site object
  identity, deferral inside a class comprehension and a recursive literal `not`. A faithful
  views-in-bindings mutant fails the near-linear assertion. All 46 now fail an assertion. The
  full 5,378-test root gate (`-p no:randomly`), ruff, mypy, all 28 import contracts and
  `git diff --check` pass.
- Review findings against `962fb2e` (10 patches, `tests/primitives/test_scoped_walk.py` only):
  16 new cases (303 to 319 guard-file tests). 6hfQ8rC7977WmvXp: a lazy object called from
  another function's body, not the scope that defines it, was invisible, since only same-scope
  calls were followed; `_called` now also follows a call from a resolved function's own scope,
  forwarding only the names that function does not shadow with a local of its own (`_shadow`),
  and `_scope`'s deferred loop runs a second pass so a sibling discovered only while resolving a
  later sibling still reaches the earlier one (new positive
  `generator-invoked-from-another-function-sees-a-feasible-sibling-alias`; the impossible
  sibling-path negative is unchanged). 6hfQ933hvw6PVvVG: `gen() and None` wrongly propagated the
  always-truthy generator as a possible return; `_ReturnFlow._truth` treats a call whose every
  feasible callee only ever hands back a lazy object (no `__bool__`/`__len__`) as always truthy,
  and an operand that definitely continues the chain is dropped unless nothing follows it (new
  `factory-and-discards-an-always-truthy-generator` negative and
  `factory-returning-the-last-operand-after-a-decided-boolean` positive). 6hfQ932C86Xm2fGp:
  `_locals` gave a lambda only its parameters, so a walrus bound inside it leaked as a free name a
  returned generator expression would read from the caller; a lambda's one-expression body is now
  walked like a function's statements, picking up the walrus through the existing Store-context
  `Name` handling (new `lambda-walrus-hides-a-caller-alias-from-a-generator-expression` negative).
  6hfQ94C4qGggxjWp and 6hfQ9VrR8mjGPv4p: `_escaping` treated any plain-named call (`stash(gen())`)
  as non-retaining, same as `bool(gen())`; a narrow `NON_RETAINING_CALLS = {bool, next, any, all}`
  allowlist now decides retention, so an unknown callee is conservatively retaining (new
  `generator-stashed-by-an-unknown-named-callee-escapes` positive) while `bool`/`next`/`any`
  truth-testing/advancing stays a comprehension temporary (new
  `generator-only-bool-tested-in-a-comprehension-stays-inside` negative). 6hfQ949RRWWvrqvp: the
  prior regression counted `len(bindings)`, near-constant regardless of live-object count, so a
  `frozenset | {oid}` rebuild on every object made (O(n) copy per creation, O(n^2) total) was
  invisible; a standalone reproduction of that rebuild measured 4,950/19,900/79,800/319,600 at
  100/200/400/800 creations (~4x per doubling). ALIVE's value is now `_Alive`, a persistent
  cons/join structure (O(1) `added`/`_alive_union`, nothing copied) whose `flattened()`/`__eq__`/
  iteration walk an explicit stack, not recursion (a first recursive draft raised
  `RecursionError` near 1,000 live objects); the rewritten test instruments actual node creation
  and iteration and now measures 200/400/800/1,600 at the same counts (exactly linear; separately
  verified to 3,200/6,400 with no `RecursionError`). `_Alive` is deliberately not a `frozenset`
  subclass: `set(x)`/`frozenset(x)`/`set.update(x)` special-case an actual frozenset instance and
  copy its table directly in C, bypassing any overridden `__iter__`; that was tried, silently
  broke 14 escaping-comprehension tests, and was reverted before this design. 6hfQ9Rm5JgMc3Mrp: a
  nested generator's own `global m` was still excluded from its returned closure whenever the
  enclosing factory had a safe local `m`, since exclusion used the factory's blanket `_locals`
  regardless of what the nested object itself declares; `_summarize` now subtracts each returned
  object's own `_global`-declared names from that closure (new
  `generator-declaring-global-bypasses-a-safe-factory-local` positive;
  `generator-declaring-nonlocal-resolves-to-the-factory-local` pins that plain `nonlocal` is
  unaffected). 6hfQ9Rqc6X7wf35G: a followed helper's `global`/`nonlocal` writes were never
  reflected in the caller's flow state; `_effect` computes, once per function and cached, the
  targets a call may leave a declared name holding across every feasible exit, using a bounded
  single-level `_ReturnFlow` pass (a call it makes is never followed, so a self- or
  mutually-recursive helper's in-flight call summarizes as having no effect, closing recursion
  without an explicit depth counter), merged into the caller's bindings via `_set` (new
  `helper-declaring-global-writes-into-the-caller-state` positive,
  `self-recursive-global-writing-helper-still-reports-its-own-write` recursion-boundary positive,
  `helper-writing-a-plain-local-leaves-the-caller-state-safe` and
  `helper-writing-a-shadowing-parameter-leaves-the-caller-state-safe` negatives). 6hfQ9Rq8R3RWMr3G:
  a comprehension result built as a deep right-nested ternary chain raised `RecursionError` in
  both `_outcomes`'s `IfExp` branch and `_values`, confirmed at depths 500 and 900; both now
  flatten the `orelse` chain with an explicit worklist bounded by `MAX_IFEXP_CHAIN` (10,000) AST
  nodes instead of recursing one frame per `else` (new
  `deep-ternary-chain-in-a-comprehension-result-is-reached`/`...-stays-clean` at depth 520; both
  the flagged and clean branch were verified correct, not just non-crashing, up to depth 5,000).
  6hfQ9RmmQ7HwXJ3p: `_returned`/`_effect` keyed their cache on the caller's entire scope state
  minus parameters, so an unrelated function defined anywhere else in the module inflated every
  other function's key; a reproduction with `count` independent generator/factory pairs measured
  cache-entry size summed across all `_summarize` calls at 2,650/10,300/40,600/161,200 for
  50/100/200/400 pairs (~4x per doubling). `_read_names`/`_entry` now project the cache key onto
  only the plain names a function's own top-level code actually loads, and the same reproduction
  now measures 50/100/200/400/800 (exactly linear; rewritten
  `test_return_summary_cache_keys_stay_near_linear_in_the_factories` asserts the same ≤2.1x bound
  with this deterministic, non-wall-clock instrumentation). All 319 tests in
  `tests/primitives/test_scoped_walk.py` pass. The full 5,394-test root gate
  (`-p no:randomly`, 527.47 s), ruff, mypy (332 source files), all 28 import contracts and
  `git diff --check` pass.

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
- Ninth review round complete: a relative `sys.path` or parent `__path__` entry is frozen against
  the working directory once, before preflight, and the family location is absolute and
  canonical, so refreshed code that changes directory cannot redirect the guard's later exact
  source resolution; a vanished working directory is the bounded refusal. The guard records each
  handed-out spec's immutable facts (name, origin, cache, search locations) at resolution, and
  commit requires the same spec object to still carry them with an exact `SourceFileLoader` whose
  name and path match, and its module to carry the matching `__name__`, `__package__`,
  `__loader__` (the spec's loader), `__file__`, `__cached__` and `__path__`, compared with exact
  types; any in-place mutation rolls the complete transaction back. A child forked by the thread
  that owns the active transaction keeps the inherited lock, so a second child thread waits until
  that owner commits or rolls back; no-transaction and vanished-owner forks keep the existing
  reset and recovery. The import-quiescent precondition, the accepted in-process/no-sandbox
  residual and every live, authority, deployment and capital wall are unchanged; malicious
  strategy sandboxing remains out of scope.
- Tenth review round complete: when the guard hands a family spec to the import system it records
  the original `SourceFileLoader` object and an immutable snapshot of its complete instance
  state. Commit requires the handed-out spec object to be exactly a `ModuleSpec`, both its
  `loader` and the module's `__loader__` to be that original loader object, still exactly a
  `SourceFileLoader`, and its instance state to equal the snapshot with exact key and value
  types, alongside the existing recorded spec/module metadata checks; only identity and
  exact-type comparisons are used. A spec class swap, a same-shaped replacement loader or an
  injected loader operation rolls the complete transaction back. The commit validation lives in
  the carved, protected `module_commit_check.py`. Mutation of the shared stdlib classes
  themselves stays inside the accepted in-process/no-sandbox residual; the import-quiescent
  precondition and every live, authority, deployment and capital wall are unchanged.
- Eleventh review round complete: at commit every family entry and every direct parent it is
  bound on must be exactly `ModuleType`, checked before any dictionary is read, so a subclass
  (including one swapped in through `__class__`) cannot answer `__path__` or other import
  metadata differently from the validated dictionary. The guard hands out each family name at
  most once per transaction; a second resolution after executed code dropped the first entry
  fails closed before a second spec exists, and even a swallowed refusal cannot commit. Before
  handing a spec to importlib the guard records its bookkeeping (no loader state, the location
  flag, its own empty uninitialized-submodules list object), and commit requires the completed
  state: the same bookkeeping, the original list still empty and `_initializing` exactly `False`
  (importlib sets it `True` while executing and `False` once the load completes), with every
  spec field read from the instance dictionary and compared by identity or exact type. A
  parent outside the family whose class was swapped in place is refused at commit but not
  reverted by rollback; like any other in-process mutation of non-family state, that stays in
  the accepted in-process/no-sandbox residual. The import-quiescent precondition and every live,
  authority, deployment and capital wall are unchanged.
- Twelfth review round complete: the guard latches the first family name resolved twice before
  raising its refusal, never clears it, and the commit seam refuses whenever the latch is set.
  Code that swallows the refusal and restores the first object's entry and parent binding
  therefore cannot commit; the latch lives on the per-transaction guard, so a later transaction
  is unaffected. Every spec, module, `sys.modules` and parent lookup in the commit check defaults
  to `_ABSENT`, never `None`, so a deleted entry never satisfies a valid `None`: an ordinary
  module's `None` search locations and a cache-less spec's `None` cache must be present, while
  `__path__` and `__cached__`, which the import system does not set in those cases, must be
  absent. Executed code that reaches the guard object itself and resets its latch is an
  in-process mutation of transaction state and stays in the accepted in-process/no-sandbox
  residual. The import-quiescent precondition and every live, authority, deployment and capital
  wall are unchanged.
- Thirteenth review round complete: an ordinary module's `__path__` and a cache-less module's
  `__cached__` must be absent by dictionary key membership, never by comparison with the
  importable `_ABSENT` sentinel, so executed family code that installs that sentinel as a present
  value refuses commit and rolls back. Every other commit-check lookup compares with a value that
  can never be the sentinel. The import-quiescent precondition, the accepted in-process/no-sandbox
  residual and every live, authority, deployment and capital wall are unchanged.
- Fourteenth review round complete: rollback decides presence by key membership alone. A changed
  `sys.modules` entry is present on one side only or bound to a different object, and each direct
  parent binding is recorded as an explicit presence bit plus its exact prior value. The
  importable `module_refresh._ABSENT` sentinel is gone, so no value executed code installs,
  `None` included, is mistaken for absence; fork recovery uses the same restore. The checked
  thirteenth-round finding points at the membership test. The import-quiescent precondition, the
  accepted in-process/no-sandbox residual and every live, authority, deployment and capital wall
  are unchanged.
- Publication review chunk 2 complete: final bundle and environment publication uses the
  protected `renameat2(RENAME_NOREPLACE)` primitive (an `EEXIST` winner is verified, a missing
  primitive fails closed with no replacement fallback); every verification and sealing walk
  propagates traversal errors and the bundle file-count bound precedes list growth; the
  environment inventory is complete (every directory implied by an entry) and bounded before
  every read; the final probe runs `-I -S` with only the direct site-packages root and accepts
  one canonical identity object through a protected bounded-subprocess seam that kills the
  process group; uv version/venv/sync output is bounded and launch failures are incompatible;
  retry requires exact positive evidence of a locked HTTPS wheel outage; and offline
  verification plus lock and post-publication faults run the real verifier. Because the probe no
  longer processes `.pth` files, an `algua` reachable only through a `.pth` path entry is outside
  its `find_spec` check (an installed `algua` distribution is still refused by metadata); Story
  1.3c's runtime import policy should decide whether site processing is permitted. The keyed
  argv, identity schemas, golden digests and every live, authority, deployment and capital wall
  are unchanged; the `--no-env-file` incompatibility is recorded as an open finding.
- Chunk-2 patch review complete: committed locks must name every registry package canonically
  and extraction is total, so malformed locks stay `EnvironmentIncompatible` and never reach uv;
  environment paths are validated before any read; every artifact and environment traversal
  streams entries through the protected `bounded_walk`, counting each file and directory and the
  encoded path length before retaining or descending; and an environment is accepted only if
  ordinary site startup exposes nothing beyond its own site-packages (the pinned uv shim, no
  other `.pth`, no customize hooks, system site-packages disabled), which closes the `.pth`
  residual recorded for the previous round. The offline-evidence finding now points at the real
  end-to-end test. The keyed argv, identity schemas, golden digests and every live, authority,
  deployment and capital wall are unchanged; the `--no-env-file` finding remains open.
- Review round against `7877b8a` complete: bundle digests are bounded while streaming; locked
  wheel URLs are printable and canonical before keying; every environment entry is canonical
  before it is retained or descended into; the walk closes every listing and keeps the correct
  error; broken interpreter links are `EnvironmentIncompatible`; `pyvenv.cfg` and `METADATA` are
  bounded before any read and read once; the normative companion now states the directory,
  `pyvenv.cfg` and other environment bounds and the uv 0.9.26 startup-file policy with its
  version and additive-retention rules; and the checked traversal finding points at
  `bounded_walk.py` and its consumers. The `--no-env-file` defect is deferred (recorded in
  `docs/development/stories/deferred-work.md`); same-UID `openat` hardening was dismissed as the
  accepted no-sandbox residual. The keyed argv, identity schemas, golden digests and every live,
  authority, deployment and capital wall are unchanged.
- Review round against `635e1f6` complete: each locked wheel URL must be one canonical HTTPS
  identity in which every character has exactly one spelling, and duplicate-wheel ownership is
  keyed by it; bounded-walk cleanup closes every listing whatever a close raises, reports the
  first (deepest) failure without ever dropping an interrupt, keeps an active traversal error
  primary and retries a failed exhausted-listing close; and every production consumer walks
  through the protected `scoped_walk` seam, which keeps a consumer's typed refusal primary
  when closing also fails while still reporting cleanup failures after a normal or early exit.
  The `--no-env-file` defect stays deferred. The keyed argv, identity schemas, golden digests
  and every live, authority, deployment and capital wall are unchanged.
- Review follow-ups of `529f6d5` complete: every valid HTTPS wheel URL the lock may name is
  accepted under its exact spelling, while duplicate ownership is decided by a separate RFC 3986
  identity (lowercase scheme/host, canonical IP, default port omitted, escapes normalized with
  unreserved decoded and reserved kept, dot segments removed); ambiguous numeric hosts are
  refused. Walk cleanup retries each failed listing exactly once, selects an interrupt by
  presence, and reports every ordinary listing-close failure, `GeneratorExit` included, as
  `WalkCleanupError` caused by it; interrupts are never wrapped. The bundle store, environment
  store and inventory translate exactly that error into their typed failures, so
  `deployment verify` keeps `frozen_bundle_corrupt` / `frozen_environment_corrupt` and
  provisioning `frozen_environment_incompatible`. An import-aware AST guard enforces the
  `scoped_walk` seam. Residual: outage evidence matches the exact locked URL, so a URL the lock
  spells non-canonically (uv reports its own normalized form) is conservatively classified
  non-retryable. The `--no-env-file` defect stays deferred, and the keyed argv, identity
  schemas, golden digests and every live, authority, deployment and capital wall are unchanged.

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
- `algua/primitives/module_commit_check.py`
- `algua/primitives/no_replace.py`
- `algua/primitives/bounded_walk.py`
- `algua/primitives/bounded_subprocess.py`
- `algua/registry/planner_environment_outage.py`
- `algua/registry/planner_environment_startup.py`
- `algua/registry/planner_environment_probe.py`
- `algua/registry/planner_environment_lock.py`
- `CODEOWNERS`
- `docs/development/sprint-status.yaml`
- `docs/contracts/cli-error-envelope.md`
- `docs/development/stories/1-3b-materialize-and-verify-recoverable-planner-artifacts.md`
- `docs/development/stories/deferred-work.md`
- `docs/development/specs/spec-story-1-3b-artifact-environment-contract/artifact-environment-contract.md`
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
- `tests/_venv_fixture.py`
- `tests/_walk_faults.py`
- `tests/primitives/test_bounded_subprocess.py`
- `tests/primitives/test_bounded_walk.py`
- `tests/primitives/test_scoped_walk.py`
- `tests/fixtures/uv-0.9.26-virtualenv-shim.py.txt`
- `tests/test_frozen_artifact_offline.py`
- `tests/test_no_replace.py`
- `tests/test_planner_environment_bounds.py`
- `tests/test_planner_environment_lock.py`
- `tests/test_planner_environment_outage.py`
- `tests/test_planner_environment_probe.py`
- `tests/test_planner_environment_startup.py`
- `tests/test_planner_environment_uv.py`

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
- 2026-09-28: Addressed all 3 ninth-round review patches test-first (canonical frozen package
  root, recorded source-spec facts and module metadata validated at commit, owner-fork lock
  preservation); the full 4,492-test root gate, ruff, mypy and all 28 import contracts pass.
- 2026-09-28: Addressed the tenth-round review patch test-first (exact handed-out `ModuleSpec`
  class and original `SourceFileLoader` identity and instance state validated at commit) and
  carved the protected `module_commit_check.py`; the full 4,499-test root gate, ruff, mypy and
  all 28 import contracts pass.
- 2026-09-28: Addressed all 3 eleventh-round review patches test-first (exact `ModuleType` family
  modules and direct parents, one resolution per family name per transaction, completed
  `ModuleSpec` bookkeeping validated at commit); the full 4,527-test root gate, ruff, mypy and
  all 28 import contracts pass.
- 2026-09-28: Addressed both twelfth-round review patches test-first (a duplicate family
  resolution latched for the whole transaction and refused at commit, missing spec and module
  metadata distinguished from a valid `None` at commit); the full 4,545-test root gate, ruff,
  mypy and all 28 import contracts pass.
- 2026-09-28: Addressed the thirteenth-round review patch test-first (absent-only `__cached__`
  and `__path__` checked by key membership at commit); the full 4,550-test root gate, ruff, mypy
  and all 28 import contracts pass.
- 2026-09-28: Addressed both fourteenth-round review patches test-first (presence-aware rollback
  of `sys.modules` entries and direct parent bindings without the importable sentinel, corrected
  thirteenth-round pointer); the full 4,568-test root gate, ruff, mypy and all 28 import contracts
  pass.
- 2026-09-28: Addressed the fifteenth-round review patch (documentation only: the fourteenth-round
  pointer correction now names the membership guard at `module_commit_check.py:105`);
  `git diff --check` and `tests/test_repo_hygiene.py` (8 tests) pass.
- 2026-09-28: Addressed all 7 publication review chunk-2 patches test-first (atomic no-replace
  publication, fail-closed bounded walks, complete bounded environment inventory, isolated
  bounded interpreter probe, bounded typed uv subprocesses, positive-evidence retry, real offline
  and fault acceptance) with three protected primitives and a protected outage recognizer;
  recorded the uv 0.9.26 `--no-env-file` argv incompatibility as an open finding; the full
  4,751-test root gate, ruff, mypy and all 28 import contracts pass.
- 2026-09-28: Addressed all 5 chunk-2 patch-review follow-ups test-first (canonical committed-lock
  identities with total extraction, environment paths validated before reading, a protected
  streaming bounded traversal replacing `strict_walk`, a protected pinned site-startup policy,
  corrected offline-evidence pointer); the full 4,863-test root gate, ruff, mypy, all 28 import
  contracts and `git diff --check` pass.
- 2026-09-28: Addressed all 8 patches of the review round against `7877b8a` test-first (bounded
  bundle digest streaming, canonical printable wheel URLs, canonical directories before descent,
  fault-tolerant walk cleanup, typed broken interpreter links, single bounded `pyvenv.cfg` and
  `METADATA` reads with the probe carved out, companion bound and startup-policy alignment,
  repaired traversal pointer) and recorded the `--no-env-file` defer; the full 4,900-test root
  gate, ruff, mypy, all 28 import contracts and `git diff --check` pass.
- 2026-09-28: Addressed all 3 consolidated patches of the review round against `635e1f6`
  test-first (canonical locked wheel-URL identity in a protected lock-policy module, exhaustive
  bounded-walk cleanup, a shared `scoped_walk` seam keeping consumer errors primary); the full
  4,973-test root gate, ruff, mypy, all 28 import contracts and `git diff --check` pass.
- 2026-09-28: Addressed all 6 review follow-ups of `529f6d5` test-first (valid wheel URLs with a
  separate RFC ownership identity, bounded cleanup retry, presence-based interrupt selection,
  visible cleanup `GeneratorExit`, frozen error taxonomy for listing-close failures, an
  import-aware scoped-walk guard); the full 5,065-test root gate, ruff, mypy, all 28 import
  contracts and `git diff --check` pass.
- 2026-09-28: Addressed all 7 review findings against `48f9934` test-first (path-correlated lazy
  entry, closure-captured names of returned objects, feasible-return object propagation,
  confined comprehension temporaries, Python class-comprehension scope, near-linear lazy
  observation, iterative `not`-chain classification); the full 5,378-test root gate, ruff, mypy,
  all 28 import contracts and `git diff --check` pass.
- 2026-09-28: Addressed all 10 review findings against `962fb2e` test-first, in
  `tests/primitives/test_scoped_walk.py` only (cross-scope lazy-call visibility, always-truthy
  Boolean return pruning, lambda-walrus closure locals, unknown-callee retention behind a narrow
  non-retaining allowlist, an O(1) persistent alive-set replacing the quadratic frozenset rebuild,
  per-object `global`/`nonlocal` closure exclusion, bounded same-scope helper effect propagation
  with a recursion-closing guard, iterative deep-ternary-chain classification, and return-summary/
  effect cache keys projected onto a function's actual reads); 16 new cases (303 to 319 guard-file
  tests), all passing; the full 5,394-test root gate, ruff, mypy, all 28 import contracts and
  `git diff --check` pass.
- 2026-09-28: Recorded 11 findings from an independent BMAD review against `3c338bc` and mirrored
  them to Todoist; not yet fixed. Story moved back to `in-progress`.
