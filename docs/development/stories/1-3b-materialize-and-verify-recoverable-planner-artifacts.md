---
baseline_commit: dc2a222ef811dc1c3a835d656a411de58423796c
---

# Story 1.3b: Materialize and verify recoverable planner artifacts

Status: backlog

Prepared: 2026-09-25. Baseline: Story 1.3 readiness baseline `dc2a222` (PR #671).
Epic: 1. Parent: Story 1.3. Requirements: FR4, FR6, FR9–FR10 and NFR1, NFR3–NFR8.
Depends on: Story 1.3a reviewed and merged.

## Story

As Algua's operator,
I want a qualified candidate's planner source and matching dependency environment materialized as
recoverable content-addressed objects,
so that I can verify what will run before any deployment is activated or trading behavior changes.

## Scope and authority

This story builds and verifies immutable content but does not execute it for a tick. It adds an
operator-visible JSON command that prepares or verifies an artifact descriptor without activating a
strategy deployment, allocating capital, changing lifecycle stage or changing paper/live behavior.

The artifact is an exported source bundle, not a worktree and not an installed Algua wheel. The
environment contains locked third-party runtime dependencies and no local/editable Algua install.
Current paper-tradable strategies are source-only. Non-empty model assets remain unsupported and do
not become tradable through this story.

## Acceptance criteria

1. **Exact source inventory.** Given a clean recorded Git `source_ref`, materialization reads Git
   object bytes for every tracked regular file beneath `algua/`, plus generated canonical resolved
   configuration and manifest/protocol metadata. Mutable working-tree bytes are never the source.
   `.git`, `.env`, databases, credentials, trust anchors, logs, datasets, docs, tests, web files and
   host-absolute paths are excluded.
2. **Deterministic digest.** The tree digest covers normalized relative path, file kind/mode and exact
   bytes. Enumeration order, timestamps and host paths cannot change it; any source, configuration,
   protocol or environment-identity change does. Links, special files, traversal and normalized or
   case-colliding paths are rejected.
3. **Atomic immutable publication.** Bundle/environment content is built in a unique private staging
   directory on the target filesystem, verified, fsynced where supported and atomically published
   without overwrite beneath digest-derived paths in `Settings.data_dir`. A pre-existing digest is
   accepted only after canonical byte/inventory verification. Concurrent builders yield one
   identical object or a domain failure, never mixed content.
4. **Environment build inputs and identity.** `pyproject.toml`, `uv.lock` and `.python-version` are
   read from the same Git commit as build inputs, not copied into the executable bundle. Sharing is
   allowed only for the complete dependency digest, Python implementation/full version, ABI/cache
   tag and platform tag. Installed inventory and interpreter identity are recorded and verified.
5. **Admission-time acquisition only.** Preparation may download distributions already selected by
   the committed lockfile using locked/no-project-install semantics. It may not resolve, upgrade or
   select versions. A missing locked distribution fails with a stable retryable code before any
   activation or stage change. Tick-time resolution, download and `uv` execution remain forbidden.
6. **No checkout-bound package.** The prepared environment contains no editable/local `algua`
   distribution and cannot resolve `algua` from the mutable checkout during verification. Published
   bundle and environment files are non-writable.
7. **Current asset boundary.** Source-only candidates use an empty canonical asset inventory.
   Non-empty assets fail with a stable unsupported-lane code. The manifest may reserve asset entries,
   but this story does not copy a path, enable a model sidecar or weaken paper tradability gates.
8. **Append-only descriptor.** The existing immutable deployment-artifact ledger records or
   byte-verifies the frozen descriptor. Manifest data includes source commit, bundle digest, stable
   digest-derived locator, environment identity and canonical resolved configuration. No absolute
   host path is identity. Existing rows are never mutated or repointed.
9. **Read-only operator command.** A typed JSON-emitting preparation/verification command succeeds
   with artifact/environment identities and verification state or fails with a stable code. It does
   not create `strategy_deployments`, transition a strategy, allocate capital, invoke a planner or
   contact a broker.
10. **Retention and recovery.** Published complete objects remain addressable indefinitely. Failed
    transactions may leave only complete unreferenced objects that are safe to reuse. Cleanup may
    remove only provably owned incomplete staging directories; no garbage collection is introduced.
11. **Quality and authority preservation.** Slow Git/filesystem/environment work occurs outside a
    SQLite write transaction. CLI JSON, import boundaries, module-size ratchets, live authority and
    the full repository gate remain unchanged/green.

## Tasks / subtasks

- [ ] Add red digest/inventory/path-safety and concurrent-publication tests (AC1–AC3).
- [ ] Implement source export and canonical manifest in focused non-CLI modules (AC1–AC3).
- [ ] Add full environment-key calculation, locked provisioning and inventory verification (AC4–AC6).
- [ ] Enforce the source-only asset boundary (AC7).
- [ ] Record/verify the append-only descriptor without activation (AC8).
- [ ] Add the JSON preparation/verification command as a thin composition layer (AC9).
- [ ] Cover crash, race, cache-miss, corruption and unreferenced-content recovery (AC3, AC5, AC10).
- [ ] Obtain protected schema/identity review if persistence changes are required (AC8, AC11).

## Development notes

- Reuse `deployment_artifacts` and `DeploymentManifest` where their immutable semantics suffice;
  prefer canonical manifest fields over a schema expansion unless queries require first-class data.
- The authoritative source set matches current clean-tree verification: tracked `algua/` files.
- Never hold `BEGIN IMMEDIATE` across Git, uv, filesystem walks, fsync or environment provisioning.
- A digest-derived relative locator is data; the trusted store root is operational configuration.
- Same-UID hostile replacement remains within the accepted no-sandbox residual. Do not claim this
  story creates a security sandbox.

## Verification

```bash
uv run pytest -q
uv run ruff check .
uv run mypy algua
uv run lint-imports
```

## References

- [Parent Story 1.3](1-3-materialize-and-execute-frozen-planner-artifacts.md)
- [Approved Sprint Change Proposal](../sprint-change-proposal-2026-09-25.md)
- [Artifact-freeze design](../../superpowers/specs/2026-09-22-artifact-freeze-design.md)

