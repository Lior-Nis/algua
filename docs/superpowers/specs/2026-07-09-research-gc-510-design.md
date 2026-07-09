# `research gc` — advisory-first, lifecycle-tied cleanup of retired-strategy files & stale reports (#510)

## Problem
The strategy lifecycle has a terminal `retired` state and a resting `dormant` state, but
retiring/benching a strategy does **not** clean up its FILES. The strategy module
(`algua/strategies/<family>/<name>.py`) and its synced vault doc (`<kb>/strategies/<name>.md`)
persist in the active tree forever. Over many experiments the research surface silts up with a
mix of live and long-dead artifacts, and the loader keeps indexing modules for strategies that
can never trade again.

This is the cleanup companion to the hygiene GATE #509: #509 is a structural CI gate that
*prevents new mess* (placement/provenance/junk); #510 *reaps accumulated mess*, tied to the
state model that already exists. The two share one invariant — **nothing referenced by a
non-terminal lifecycle state is ever touched** — so they cannot fight each other.

## Solution — a `research gc` command, ADVISORY-FIRST
Mirror the two existing read-only research advisories (`research dormant-sweep`,
`research family-audit`): a `algua research gc` subcommand on the existing `research` typer
group that emits JSON on stdout, ranked worst-offender-first.

- **Default = read-only advisory scan.** Walk the reap-eligible file surfaces, cross-reference
  the registry, and REPORT what is safe to reap and why. Writes nothing, moves nothing,
  transitions nothing, reads/burns no holdout, writes no gate/FDR ledger. Safe to run
  unattended on the systemd clock as a fleet-health-style signal.
- **`--archive` = governed cleanup (human-only).** Move the reaped files into an `archive/`
  tree (never delete), recording a manifest of what moved. Fail-closed for an agent/system
  actor. Never deletes a DB row — the registry stays the immutable system of record; only the
  *derived FILES* are reaped.

### Why archive-not-delete, and why only files
The registry (`strategies`, `stage_transitions`, gate/allocation/FDR ledgers) is the governed,
immutable record of what was tried and decided. GC must never mutate it — a retired strategy's
row and its transition history stay forever. GC only relocates the *derived files* those rows
generated, and even then it archives rather than deletes, so a mistaken reap is fully
reversible (the archive tree + manifest are a redo log).

## Reap-eligible surfaces (in scope for this slice)
A file is a **reap candidate** only if it resolves unambiguously to a single strategy *name*:

1. **Strategy modules** — `algua/strategies/<family>/<name>.py`. The module stem *is* the bare
   strategy name (the loader's `_index()` maps `mod.name -> dotted module`). Archiving the file
   removes it from the loader index, which is correct: a retired strategy must not be loadable.
2. **Per-strategy vault docs** — `<knowledge_dir>/strategies/<name>.md` (the always-on synced
   strategy doc; `strategy_doc_path`). The stem is the strategy name.

Both map file → strategy by **exact stem == registry `name`**. The map is total and
unambiguous; anything that does not resolve to exactly one name is left alone.

## Explicitly-protected surfaces (NEVER reaped in this slice)
- `<kb>/experience/` — negative-result / refuted-hypothesis notes. These are a permanent
  curation ledger (the whole point of #332); you *want* to keep the record of what failed.
  Excluded wholesale.
- `<kb>/strategies/families/*.md` — family docs. A family doc can be referenced by many
  strategies across stages; reaping it safely needs whole-family liveness reasoning. Deferred.
- `<kb>/principles/`, `.obsidian/`, any vault infra.
- MLflow runs (`mlruns/`) and `artifacts/` — governed by their own store/retention; out of scope.
- Any path that does not resolve to exactly one strategy name (ambiguous / shared).
- Anything mapping to a strategy in a **non-terminal** stage, or **retired but within the
  retention window** (see below).

## Registry cross-reference — classification rules
For each candidate file, resolve its strategy `name`, look up the registry row, and classify.
Only two reasons make a file reapable; everything else is `protected`:

| reason | condition |
| --- | --- |
| `retired_expired` | a registry row exists AND current stage is `RETIRED` AND the retirement transition is older than the retention window |
| `orphaned` | NO registry row for `name` AND the file's mtime is older than the retention window (the mtime floor prevents sweeping a just-authored, not-yet-`registry add`ed module) |
| `protected` (not reapable) | row exists and stage is non-terminal (`idea`/`backtested`/`candidate`/`paper`/`forward_tested`/`live`/`dormant`); OR retired but within window; OR orphan newer than the mtime floor; OR path unresolvable |

- **Non-terminal protection is derived, not enumerated.** Reapable requires stage **is exactly
  `RETIRED`** (the sole terminal stage per `lifecycle.py`). Every other stage — including
  `dormant`, which is explicitly *non-terminal* — is protected by construction. If a new
  non-terminal stage is ever added, it is protected automatically (no allowlist to forget).
- **`retired_at`** = the `created_at` of the newest `stage_transitions` row whose `to_stage`
  is `RETIRED` for that strategy. Age = wall-clock days from `retired_at` to `now`. (Wall-clock,
  not trading sessions: file staleness is a disk-housekeeping question, not a market-time one.)
- **Orphan mtime floor** closes the author→register race: a module written by the
  author-a-strategy step but not yet `registry add`ed looks orphaned; requiring
  `file mtime older than retention` means GC cannot sweep in-flight authoring work.

## Retention window
- `--retention-days N`, default **90** (conservative). Applies to both `retired_expired`
  (age since retirement) and `orphaned` (age since file mtime).
- The **advisory scan** accepts any `--retention-days` freely — it is read-only and cannot
  destroy anything, so there is no relaxation to gate.
- The **`--archive` cleanup** is human-only regardless of retention (see below), so a small
  retention can never be weaponized by an agent to reap recently-retired work.

## Governed cleanup mode (`--archive`)
- **Explicit flag + verified human actor.** `--archive` alone is not enough; the effective
  actor must be `human`, verified through the existing #329 signed-challenge chokepoint — **not**
  a bare `--actor human`, which #329 established is forgeable by an agent. An agent/system actor
  invoking `--archive` **fails closed** (non-zero exit, nothing moved). The challenge payload is
  bound to a digest of the sorted reap-set + retention + run-context, so a signature cannot be
  replayed against a different set of files.
  - *Proportionality note (design decision):* GC-archive is reversible and touches no DB row and
    no money, so a bespoke weaker gate was tempting. We deliberately reuse the *same* verified
    human-actor chokepoint as every other governed mutation rather than reintroduce the exact
    forgeable-`--actor` defect #329 fixed. Consistency + non-forgeability beats a one-off gate.
- **Archive, never delete.** Each reaped file is moved under a run-scoped archive root
  `archive/<UTC-run-id>/<original-relative-path>` (e.g.
  `archive/20260709T2014Z-ab12/strategies/momentum/dead_mom.py`). Preserving the original
  relative structure + a per-run subdir means a later re-created/re-retired same-named file
  never collides with an earlier archived copy.
- **Manifest.** Every run that moves anything appends a JSON manifest row per move to
  `archive/manifest.jsonl` (append-only): `{run_id, ts, from, to, strategy, reason,
  stage_at_reap, retired_at, bytes}`. The manifest is the audit trail + redo log.
- **DB immutability.** Cleanup calls **no** registry writer — no `remove_strategy`, no
  `apply_transition`, no ledger write. It only relocates files. Asserted by a test that runs
  `--archive` against a temp registry and checks the `strategies` + `stage_transitions` row
  counts are byte-for-byte unchanged.
- **No hard delete / prune in this slice.** Archive is the only mutation. A `--prune`
  (permanent delete of already-archived items past a second, longer horizon) is deferred.

## Idempotency
- The advisory scan is naturally idempotent (pure read).
- Cleanup is idempotent by the **source-absence** rule: a file is reaped only if it is still
  present at its live path. After a run, a reaped module/doc no longer exists at the live path
  (it lives under `archive/`, which is **excluded from the scan roots**), so a second run finds
  nothing more to reap for it and moves nothing. The run-scoped archive subdir guarantees no
  target collision. Moves use `os.replace` (atomic within a filesystem; cross-fs falls back to
  copy → fsync → unlink). A crash mid-run leaves either the source or the archived copy intact
  (never neither), and re-running completes the remainder — no duplicate manifest row is written
  for a file already moved because it is no longer discoverable at the live path.

## Module structure & boundaries
- **`algua/research/gc.py` — pure classifier (no I/O, no DB, no filesystem).** Precedent:
  `research/family_audit.py` is a pure advisory core. Signature roughly:
  `classify(candidates, registry_state, *, now, retention_days) -> GcReport`, where
  `candidates` is a list of `FileCandidate(path, kind, strategy_name, mtime, size)` and
  `registry_state` is `{name: (stage, retired_at | None)}`. Returns the ranked reapable list +
  protected summary. 100% unit-testable with no fixtures.
- **`algua/cli/research_cmd.py` — orchestration (the only I/O).** The new `gc` command does the
  filesystem walk (discover candidates, follow no symlinks, contain to the two scan roots),
  reads the registry via `SqliteStrategyRepository.list_strategies()` + `list_transitions()`,
  calls the pure classifier, emits JSON, and — only under `--archive` + verified human — moves
  files + writes the manifest.
- **Path-containment safety.** Archive targets are built through the same
  containment guard `knowledge/sync.py::_safe_path` uses (target must resolve under the
  archive root), so a crafted strategy name cannot escape the archive tree. The walk resolves
  symlinks and refuses to descend or reap through them.
- **Import boundaries.** `gc.py` imports nothing from `algua.data`/`algua.cli`; the command
  lives in the already-registered `research` group, so no new `independence`-contract module
  registration is needed. `uv run lint-imports` must stay green.
- **CODEOWNERS.** None of the touched paths (`algua/research/gc.py` [new],
  `algua/cli/research_cmd.py`, tests, `CLAUDE.md`) are CODEOWNERS-protected → the PR is
  auto-merge eligible iff CI is green.

## JSON shape
Advisory (`research gc`):
```json
{
  "now": "2026-07-09T20:14:00+00:00",
  "retention_days": 90,
  "reapable": [
    {"strategy": "dead_mom_v3", "reason": "orphaned", "stage": null,
     "retired_at": null, "age_days": 412,
     "files": [{"path": "algua/strategies/momentum/dead_mom_v3.py", "kind": "module", "bytes": 2841}],
     "total_bytes": 2841},
    {"strategy": "old_meanrev", "reason": "retired_expired", "stage": "retired",
     "retired_at": "2026-01-02T…", "age_days": 188,
     "files": [{"path": "…/old_meanrev.py", "kind": "module", "bytes": 3120},
               {"path": "kb/strategies/old_meanrev.md", "kind": "vault_doc", "bytes": 1044}],
     "total_bytes": 4164}
  ],
  "protected_summary": {"n_strategies": 37, "n_files": 74, "by_stage": {"live": 3, "paper": 5, "dormant": 2, "retired_within_window": 4, "…": 0}},
  "summary": {"n_reapable": 2, "n_files": 3, "total_bytes": 7005}
}
```
Ranking = worst-first: `orphaned` before `retired_expired`, then by `total_bytes` desc, then
`age_days` desc, then name (stable). Cleanup (`--archive`) adds
`"archived": [{"from","to","strategy","reason","bytes"}], "manifest_path": "archive/manifest.jsonl"`.
`--summary` (the #349 projection) drops the per-file lists and `protected_summary`, keeping only
`summary` + the reapable strategy/reason/total_bytes scalars.

## Non-goals / deferred
- **Family docs, experience notes, MLflow/artifacts** — out of scope (protected). Family-doc
  reaping needs whole-family liveness; experience notes are a permanent ledger.
- **Hard delete / `--prune`** — archive-only this slice; permanent deletion of already-archived
  items past a longer horizon is a follow-up.
- **No gate/ledger/holdout interaction, no schema bump** — GC is pure housekeeping over files.
- **Report-artifact surfaces** (e.g. report-experiments SVGs/parquet keyed by strategy) — the
  classifier + `kind` enum are designed to extend to them, but this slice ships modules +
  strategy docs only.
- **No auto-archive on retire** — GC stays a separate, explicitly-invoked pass; wiring a reap
  into the `-> retired` transition would couple file I/O into a registry mutation (rejected).

## Implementation task list
1. **Pure classifier `algua/research/gc.py`** — `FileCandidate`, `ReapItem`, `GcReport`
   dataclasses + `classify(candidates, registry_state, *, now, retention_days)`; the
   reason/protected rules and the worst-first ranking above. No I/O. Unit tests
   `tests/test_research_gc.py`: retired-expired vs retired-within-window; orphan past vs under
   mtime floor; every non-terminal stage (esp. `dormant`) is protected; a new hypothetical
   stage is protected-by-default; ambiguous/unresolvable path left alone; ranking order.
2. **Filesystem discovery** (in `research_cmd.py` or a small `gc.py`-adjacent I/O helper) —
   walk the two scan roots, resolve symlinks, build `FileCandidate`s (skip `_`-prefixed/private
   modules exactly as the loader does; skip the archive root). Tests for the walk (temp tree).
3. **`research gc` CLI command** — wire onto the `research` typer group: `--retention-days`
   (default 90), `--archive`, `--actor`/`--actor-signature`, `--summary`. Advisory path reads
   registry (`list_strategies` + per-name `list_transitions` for `retired_at`), calls the
   classifier, `emit`s JSON. Tests `tests/test_cli_research_gc.py` for the advisory JSON.
4. **Governed cleanup path** — behind `--archive`: verify effective human actor via the #329
   chokepoint (fail-closed for agent/system, forged, replayed, or reap-set-mismatched
   signatures), move each reapable file to `archive/<run-id>/<rel-path>` via atomic replace
   (cross-fs copy+fsync+unlink fallback) through the `_safe_path` containment guard, append the
   `archive/manifest.jsonl` rows. Tests: agent `--archive` fails closed & moves nothing;
   verified human archives + writes manifest; **DB-immutability** test (row counts unchanged);
   **idempotency** test (second run moves nothing, no dup manifest row); path-containment test
   (crafted name cannot escape archive root).
5. **`.gitignore`** — add `/archive/` so archived files + the manifest never re-enter
   `git status`/an accidental `git add` (coordinates with #509's junk rules).
6. **CLAUDE.md command-surface entry** — add a `uv run algua research gc [--retention-days N]
   [--archive --actor human --actor-signature …] [--summary]` bullet: advisory-first
   lifecycle-tied file GC; read-only default (registry cross-ref, retention window, orphan +
   retired-expired detection); reaps NOTHING referenced by a non-terminal state; `--archive` is
   human-only (verified actor), archives-not-deletes, writes no DB row, idempotent. Note it is
   the cleanup companion to the #509 hygiene gate.
7. **FAST per-task check** during each task:
   `uv run ruff check . && uv run mypy algua && uv run lint-imports && uv run pytest -q
   tests/test_research_gc.py tests/test_cli_research_gc.py`. **FULL gate** at integration/finish:
   `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`.
8. **PR** — branch pushed ALONE; `gh pr create` separately. No CODEOWNERS path touched →
   auto-merge iff CI green.

## Review
GATE-1 (Codex, design): first pass CHANGES-REQUESTED — **the design/spec step produced no
readable design file** (the reviewer had nothing to review). Resolved by authoring and
committing this spec, which pins down: the read-only advisory scan (registry cross-reference,
conservative default retention window, orphan detection with an mtime floor), the governed
cleanup mode (archive-not-delete, explicit `--archive` flag + **verified** human actor per #329,
DB-row immutability), idempotency (source-absence rule + run-scoped archive + atomic move), and
the non-terminal-state protection derived from the single terminal `RETIRED` stage (companion to
the #509 hygiene gate). Re-run GATE-1 against this file inline.
