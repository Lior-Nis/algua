---
id: SPEC-story-1-3b-artifact-environment-contract
companions:
  - artifact-environment-contract.md
  - ../../stories/1-3b-materialize-and-verify-recoverable-planner-artifacts.md
sources: []
---

> **Canonical contract.** This SPEC and the files in `companions:` are the complete,
> preservation-validated contract for what to build, test and validate. Source documents remain
> narrative and traceability evidence; they do not override this contract.

# Story 1.3b Artifact and Environment Contract

## Why

Algua must be able to recover and verify the exact planner source and third-party runtime that a
qualified candidate may later execute. Story 1.3b creates those immutable objects and their
append-only descriptor without activating a deployment or changing trading behavior.

## Capabilities

- id: CAP-1
  intent: An operator can materialize the exact qualified source/configuration/protocol input from
    one clean Git commit without reading source bytes from the mutable checkout.
  success: The published source bundle has a deterministic digest and canonical inventory; source,
    mode, resolved-configuration or protocol changes alter it while order, timestamps and host paths
    do not.
- id: CAP-2
  intent: An operator can provision a recoverable third-party Python environment from the exact
    committed lock inputs without installing Algua or selecting new dependency versions.
  success: The environment key captures every build input and interpreter/platform fact, its sealed
    inventory verifies, and the published interpreter runs after relocation without Git, uv, the
    checkout or network access.
- id: CAP-3
  intent: Concurrent or interrupted builders can publish and reuse immutable objects safely.
  success: Publication is same-filesystem, no-overwrite and atomic; an existing object is reused only
    after complete verification, corruption is never repaired in place, and a failure exposes no
    partially published object.
- id: CAP-4
  intent: The append-only artifact ledger can bind the qualified identity, source bundle and
    environment while remaining compatible with existing working-tree descriptors.
  success: Preparation inserts or byte-verifies one canonical frozen descriptor after revalidating
    qualification; it creates no deployment epoch, allocation or lifecycle transition and does not
    alter existing descriptor bytes.
- id: CAP-5
  intent: An operator can verify a recorded frozen descriptor and its content after the repository
    and build tooling are unavailable.
  success: Verification uses only the descriptor database, trusted artifact-store root and published
    objects, fails closed on any identity/inventory/permission drift, and emits bounded JSON with
    relative locators only.

## Constraints

- The exact canonical schemas, digest domains, path rules, publication protocol, environment build
  policy, error taxonomy and command behavior in `artifact-environment-contract.md` are normative.
- Preparation accepts only the current clean `HEAD`, represented by its full commit object ID, and
  only when the newest eligible research gate and candidate entry match the recomputed complete
  artifact identity.
- Source bytes and build-input bytes come from Git objects at that commit. Checkout paths are used
  only to establish clean-HEAD eligibility and detect tracked drift or untracked source shadowing.
- `bundle_digest`, `environment_key`, `environment_digest` and `manifest_digest` are separate,
  non-cyclic identities. An environment change does not alter `bundle_digest`.
- Published objects are content-addressed, retained indefinitely for Phase 1 and never repaired,
  overwritten or garbage-collected by this story.
- Source-only candidates have canonical assets `[]`. Any model handle or non-empty asset inventory
  fails before external asset bytes or paths are dereferenced.
- Environment acquisition may fetch only compatible distributions already selected by the committed
  lock. It never resolves or upgrades, never installs project/workspace/local/editable content and
  never runs during verification or a trading tick.
- The object store and read-only process boundary reduce accidental drift; they are not a hostile-code
  sandbox and do not strengthen same-UID authority.
- Existing working-tree records, paper/live dispatch, activation, lifecycle, allocation, live signing,
  capital limits and planner behavior remain unchanged.

## Non-goals

- Executing a planner, defining request/response bytes or introducing a child process.
- Activating or retiring a strategy deployment or admitting a candidate to paper.
- Copying model assets or making model-backed strategies paper-tradable.
- Migrating or rebuilding working-tree/legacy deployments.
- Recording invocation evidence or qualifying frozen paper evidence.
- Garbage collection, remote artifact storage, containers, sandboxing or distributed builders.
- Changing the existing live authorization ceremony or any authority boundary.

## Success signal

`algua deployment prepare NAME` produces or reuses a verified bundle, environment and append-only
frozen descriptor for the exact current qualified commit; `algua deployment verify DIGEST` proves
that descriptor offline; all named corruption, race, crash and drift cases fail closed; no deployment
or trading state changes; and the full repository gate passes.
