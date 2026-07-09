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
  tree (never delete), reporting the moved set in the command's stdout JSON. Fail-closed for an
  agent/system actor. Never deletes a DB row — the registry stays the immutable system of record;
  only the *derived FILES* are reaped.

### Why archive-not-delete, and why only files
The registry (`strategies`, `stage_transitions`, gate/allocation/FDR ledgers) is the governed,
immutable record of what was tried and decided. GC must never mutate it — a retired strategy's
row and its transition history stay forever. GC only relocates the *derived files* those rows
generated, and even then it archives rather than deletes, so a mistaken reap is fully
reversible (the archive tree is a browsable redo log; the moved set is echoed in the run's JSON).

## Reap-eligible surfaces (in scope for this slice)
A reap candidate is a file that resolves unambiguously to a single strategy *name*. **As shipped,
two surfaces are scanned** (each candidate is an individual FileItem — reports are NOT grouped into
a whole-tree unit):

1. **Strategy modules** — `algua/strategies/<family>/<name>.py`. The module stem *is* the bare
   strategy name (the loader's `_index()` maps `mod.name -> dotted module`). Archiving the file
   removes it from the loader index, which is correct: a retired strategy must not be loadable.
   Private `_*.py` helpers and `__init__.py` are skipped exactly as the loader does.
2. **Per-strategy generated report FILES** — every regular file under
   `<knowledge_dir>/strategies/<name>/reports/<stamp>/…` (report-experiments writes
   `report.md` + SVGs there; the `<name>` DIRECTORY component is the strategy key). This is the
   "stale reports whose strategy is long dead" surface the issue names. Each report file is scanned,
   classified, and archived individually (mirroring its `reports/<stamp>/…` sub-path under the run
   dir), NOT as an atomic whole-directory move.

**Explicitly NOT a reap surface (shipped behavior):** the top-level `<knowledge_dir>/strategies/*.md`
files are kb-sync-OWNED and are never scanned — this includes both the `_*` router pages
(`_index`/`_by-stage`/`_by-date`/`_families`) AND every per-strategy live synced note at
`strategy_doc_path()` (`<kb>/strategies/<name>.md`). The walk iterates only per-strategy
DIRECTORIES (and skips `families/`), which structurally excludes every top-level file without
name-by-name special-casing. (An earlier draft of this spec proposed reaping the per-strategy vault
doc as a third surface; the implementation deliberately does NOT, because a live synced note must
never be mistaken for a disposable artifact — the doc description here matches the code.)

Both scanned surfaces map candidate → strategy by **exact stem/dirname == registry `name`**. The map
is total and unambiguous; anything that does not resolve to exactly one name is left alone.

## Explicitly-protected surfaces (NEVER reaped in this slice)
- `<kb>/experience/` — negative-result / refuted-hypothesis notes. These are a permanent
  curation ledger (the whole point of #332); you *want* to keep the record of what failed.
  Excluded wholesale.
- `<kb>/strategies/families/*.md` and the `<kb>/strategies/families/` dir — family docs. A family
  doc can be referenced by many strategies across stages; reaping it safely needs whole-family
  liveness reasoning. Deferred, and `families` is a **hard-excluded reserved dirname** in the
  report-tree walk so it is never mistaken for a strategy-named report directory.
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
- **Archive, never delete — atomic same-filesystem rename ONLY.** Each reaped file is moved under a
  run-scoped archive dir `<archive-dir>/<run-id>/<mirrored-source-path>` (e.g.
  `archive/20260709T2014Z-ab12f0c9/home/u/algua/strategies/momentum/dead_mom.py` — the source's
  absolute path with its anchor `/` stripped so the join nests instead of collapsing). The move is a
  single `os.replace` (POSIX `rename(2)` — atomic for a file on one filesystem). **There is no
  cross-filesystem copy fallback:** a copy→fsync→unlink could crash with *both* source and target
  present (an ambiguous partial-archive state), so a genuinely cross-filesystem destination (EXDEV)
  is **skipped and surfaced** (`cross_filesystem`), never copied — keeping the invariant "at most
  one of {source, target} exists" true by construction.
  - **Single archive dir, not co-located dual roots (shipped).** `--archive-dir` (default
    `archive/`, repo-relative) is one root for all surfaces. A source on a *different* filesystem
    than the archive dir (e.g. a `<kb>` report when `--archive-dir` sits on another mount) is not
    copied — it is skipped `cross_filesystem`, so the operator points `--archive-dir` at the same
    filesystem as the surface they intend to reap. The run-id carries a `uuid4` suffix so two runs
    in the same UTC second get distinct run dirs and can never `os.replace` onto each other. The
    archive dir is excluded from the scan surfaces and belongs in `.gitignore`.
  - **Source hardening (TOCTOU/symlink/content).** Each source is opened once
    `O_RDONLY|O_NOFOLLOW` and never re-resolved by path: the whole parent chain is resolved and
    required to land under a scan root (`escaped_scan_root` else — catches an intermediate symlinked
    dir `O_NOFOLLOW`'s final-component check misses); `fstat` rejects a non-regular file; the bytes
    are hashed off the fd and must equal the sha256 the human SIGNED (`content_changed_since_authorization`
    else — point-of-use enforcement, not merely challenge-time); an `lstat` (st_dev,st_ino) recheck
    just before the move rejects a raced-in replacement (`replaced_before_move`).
  - **Destination hardening (symmetric).** Before AND after the `mkdir`, the destination is verified
    to contain no symlinked run-dir/mirrored component and to resolve under the archive root
    (`archive_dest_unsafe` else) — a planted symlink component cannot redirect a reaped file out of
    the archive tree via `mkdir(exist_ok=True)` + `os.replace`.
  - **Registry re-check at point of use.** Immediately before `os.replace` the strategy's CURRENT
    registry stage is re-read; a retired-expired item whose strategy was un-retired, or an orphan
    whose name was `registry add`ed, since the classify snapshot is skipped `registry_stage_changed`
    — closing the in-process TOCTOU between computing `reap` and moving.
- **Crash-safety: what actually ships (NO durable archive record / rebuildable index).** The atomic
  rename is the only crash-safety mechanism: after any crash, **at most one of {source, target}
  exists** for each artifact (never both, because there is no copy path; never a torn write). There
  is **no per-move sidecar, no `manifest.jsonl` rollup, and no reconcile-first recovery pass** — the
  record of what moved is emitted only in the command's stdout JSON (`archived`/`archive_skipped`/
  `archive_run_dir`), which is NOT a durable, rebuildable on-disk index. A crash mid-run therefore
  loses the audit line for any already-moved file (the file is safely under `archive/<run-id>/…` but
  no on-disk manifest records it); recovery is manual (the archive tree itself is browsable). This
  is an accepted limitation for an advisory, reversible, file-only housekeeping pass: nothing is
  deleted, no DB row moves, and the immutable registry remains the system of record. A durable
  sidecar+`manifest.jsonl`+reconcile protocol was designed (see the review history below) but
  deliberately **deferred** — it is not implemented, and this slice does not claim it.
- **DB immutability.** Cleanup calls **no** registry writer — no `remove_strategy`, no
  `apply_transition`, no ledger write. It only relocates files. Asserted by a test that runs
  `--archive` against a temp registry and checks the `strategies` + `stage_transitions` row
  counts are byte-for-byte unchanged.
- **No hard delete / prune in this slice.** Archive is the only mutation. A `--prune`
  (permanent delete of already-archived items past a second, longer horizon) is deferred.

## Idempotency
- The advisory scan is naturally idempotent (pure read).
- Cleanup is idempotent by the **source-absence** rule: an artifact is reaped only if it is still
  present at its live path. After a run, a reaped module/doc/report-tree no longer exists at the
  live path (it lives under `archive/`, which is **excluded from the scan roots**), so a second
  run finds nothing more to reap for it and moves nothing. The run-scoped archive subdir (UTC stamp
  + uuid suffix) guarantees no target collision. Moves are a single atomic `os.replace` on one
  filesystem; there is **no** non-atomic copy fallback, so after any crash exactly one of {source,
  target} exists per artifact.
- **Crash behavior (shipped, no reconcile pass):** because the move is atomic and there is no copy
  path, a crash mid-run leaves either the source (move never committed) or the archived copy (move
  committed) intact — never both, never neither. There is **no** manifest/sidecar to repair on the
  next run, so the next run simply re-classifies from live file + registry state: an already-moved
  file is gone from the scan surface (no-op), a not-yet-moved source is re-evaluated fresh. The only
  loss is the *audit line* for a file moved just before a crash (the file is safe under
  `archive/<run-id>/…`, but the stdout JSON that would have recorded it never printed) — an accepted
  limitation, not a data-loss or double-reap bug.

## Module structure & boundaries
- **`algua/research/lifecycle_gc.py` — pure classifier (no I/O, no DB, no filesystem).** Precedent:
  `research/family_audit.py` is a pure advisory core. Shipped signature:
  `classify(items, registry, *, now, retention_days) -> list[Classified]`, where `items` is a list
  of `FileItem(path, strategy, surface, size_bytes, mtime)` — `surface ∈ {strategy_module, report}`,
  each report FILE its own item (not a grouped tree) — and `registry` is `{name:
  RegistryEntry(stage, retired_at | None)}`. It also exposes `reapable()` (rank worst-first),
  `archive_manifest()` (canonical content-addressed description of the exact reap set), and
  `build_gc_archive_challenge()` (the #329 signed-challenge bytes). 100% unit-testable, no fixtures.
- **`algua/cli/research_cmd.py` — orchestration (the only I/O).** The `gc` command walks the two
  scan surfaces (`_gc_inventory`, symlink-refusing, contained to the two scan roots via
  `_gc_scan_roots`), reads the registry via `SqliteStrategyRepository.list_strategies()` +
  `list_transitions()`, calls the pure classifier, emits JSON, and — only under `--archive` +
  verified human — moves files (`_gc_archive`). There is **no manifest file written to disk**; the
  archived/skipped lists are reported in the command's stdout JSON only.
- **Path-containment safety.** Both the source parent chain and the archive destination are resolved
  and required to stay under their respective roots (mirroring `knowledge/sync.py::_safe_path`), so
  a crafted strategy name or a planted symlink component cannot escape either the scan roots or the
  archive tree. The walk resolves symlinks and refuses to descend or reap through them.
- **Import boundaries.** `lifecycle_gc.py` imports nothing from `algua.data`/`algua.cli`; the command
  lives in the already-registered `research` group, so no new `independence`-contract module
  registration is needed. `uv run lint-imports` must stay green.
- **CODEOWNERS.** None of the touched paths (`algua/research/lifecycle_gc.py` [new],
  `algua/cli/research_cmd.py`, tests, `CLAUDE.md`) are CODEOWNERS-protected → the PR is
  auto-merge eligible iff CI is green.

## JSON shape (as shipped)
Advisory (`research gc`) — `reapable` is a FLAT list of per-file entries (one per module / report
file), not grouped per strategy:
```json
{
  "note": "advisory lifecycle GC: read-only by default; …",
  "dry_run": true,
  "retention_days": 90,
  "total_files_scanned": 74,
  "reapable_count": 2,
  "reclaimable_bytes": 23408,
  "by_reason": {"untracked_module": 40, "protected_non_terminal": 30,
                "retired_expired": 1, "orphaned_report": 1},
  "reapable": [
    {"path": "…/kb/strategies/old_meanrev/reports/20260102-000000/report.md",
     "strategy": "old_meanrev", "surface": "report", "reason": "orphaned_report",
     "size_bytes": 20488, "age_days": 188.0, "stage": null},
    {"path": "…/algua/strategies/momentum/dead_mom_v3.py",
     "strategy": "dead_mom_v3", "surface": "strategy_module", "reason": "retired_expired",
     "size_bytes": 2920, "age_days": 412.0, "stage": "retired"}
  ],
  "archived": [], "archive_skipped": [], "archive_run_dir": null
}
```
Ranking = reclaimable `size_bytes` DESC, then `age_days` DESC, then `path` (stable). Cleanup
(`--archive --actor human` with a valid `--actor-signature`) populates `archived`
(`[{src,dest,strategy,surface,reason,size_bytes}]`), `archive_skipped`
(`[{src,strategy,surface,reason}]` — e.g. `cross_filesystem`, `archive_dest_unsafe`,
`registry_stage_changed`, `content_changed_since_authorization`, `replaced_before_move`,
`escaped_scan_root`, `refused_non_regular_file`), and `archive_run_dir` (the timestamp+uuid run
dir). **There is no `manifest_path`** — no manifest file is written; the archived/skipped lists live
only in this JSON. Running `--archive --actor human` WITHOUT a signature instead emits an
`action: "human_actor_challenge"` payload (the challenge to sign + the reapable preview) and moves
nothing.

## Non-goals / deferred
- **Family docs, experience notes, MLflow/artifacts** — out of scope (protected). Family-doc
  reaping needs whole-family liveness; experience notes are a permanent ledger.
- **Hard delete / `--prune`** — archive-only this slice; permanent deletion of already-archived
  items past a longer horizon is a follow-up.
- **No gate/ledger/holdout interaction, no schema bump** — GC is pure housekeeping over files.
- **MLflow runs / emit-series parquet under `mlruns/` + `artifacts/`** — governed by the tracking
  store's own retention; out of scope. The `surface` enum can extend to a strategy-keyed MLflow
  retention pass later, but this slice ships strategy modules and the per-strategy report FILES
  under `<kb>/strategies/<name>/reports/…` only (top-level vault docs are NOT reaped).
- **Durable archive record / rebuildable index** — the per-move sidecar + `manifest.jsonl` rollup +
  reconcile-first recovery protocol described in the review history is DEFERRED; this slice ships
  atomic `os.replace` only, with the archived/skipped record living in the command's stdout JSON.
- **No auto-archive on retire** — GC stays a separate, explicitly-invoked pass; wiring a reap
  into the `-> retired` transition would couple file I/O into a registry mutation (rejected).

## Implementation task list
1. **Pure classifier `algua/research/lifecycle_gc.py`** — `FileItem`, `RegistryEntry`, `Classified`
   dataclasses + `classify(items, registry, *, now, retention_days)` + `reapable()` ranking +
   `archive_manifest()`/`build_gc_archive_challenge()` (#329 challenge bytes). No I/O. Unit tests
   `tests/test_research_gc.py`: retired-expired vs retired-within-window; orphan past vs under
   mtime floor; every non-terminal stage (esp. `dormant`) is protected; a new hypothetical
   stage is protected-by-default; untracked module kept; ranking order.
2. **Filesystem discovery** (`_gc_inventory` in `research_cmd.py`) — walk the two scan surfaces
   (`algua/strategies/<family>/*.py`; per-strategy report FILES under
   `<kb>/strategies/<name>/reports/…`), refuse symlinks, build `FileItem`s (skip `_`-prefixed/private
   modules + `__init__.py` exactly as the loader does; iterate only per-strategy DIRECTORIES so
   `families/` and all top-level `*.md` are structurally excluded). Tests for the walk on a temp
   tree (module + report file for one name; top-level `.md` never scanned).
3. **`research gc` CLI command** — wire onto the `research` typer group: `--retention-days`
   (default 90), `--archive`, `--actor`/`--actor-signature`, `--archive-dir`, `--top`. Advisory path
   reads registry (`list_strategies` + per-name `list_transitions` for `retired_at`), calls the
   classifier, `emit`s JSON. Tests `tests/test_cli_research_gc.py` for the advisory JSON.
4. **Governed cleanup path (`_gc_archive`)** — behind `--archive`: FIRST verify effective human
   actor via the #329 chokepoint (fail-closed for agent/system, forged, replayed, or
   reap-set-mismatched signatures; no signature → print the manifest-bound challenge and move
   nothing) — no file moves before this gate — THEN per file: source hardening (`O_NOFOLLOW` fd,
   parent-chain containment to scan roots, signed-content-hash match, inode recheck), **destination
   hardening** (`_archive_dest_safe`: reject a symlinked run-dir/mirrored component or a dest
   resolving outside the archive root, checked before+after `mkdir`), **registry stage re-check**
   (skip `registry_stage_changed` if the strategy un-retired / the orphan was added since classify),
   then a single atomic `os.replace` into `<archive-dir>/<run-id>/<mirrored-path>` (`run-id` =
   UTC stamp + uuid suffix, collision-resistant; NO copy fallback — cross-fs skipped
   `cross_filesystem`). Tests: agent `--archive` fails closed & moves nothing; verified human
   archives module+report + reports the moved set; **DB-immutability** (`strategies` +
   `stage_transitions` counts unchanged); **idempotency** (second run moves nothing); source symlink
   refusal; intermediate-symlinked-dir escape refusal; **destination symlinked-component refusal**;
   content-changed refusal; atomic-replace (no both-paths window); cross-fs skip;
   **registry-stage-recheck** skip; run-id collision-resistance.
5. **`.gitignore`** — add `/archive/` (the default archive root) so archived artifacts never
   re-enter `git status`/an accidental `git add` (coordinates with #509's junk rules). The archive
   root is also excluded from the GC scan surfaces.
6. **CLAUDE.md command-surface entry** — add a `uv run algua research gc [--retention-days N]
   [--archive --actor human --actor-signature … --archive-dir DIR --top N]` bullet: advisory-first
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

Second GATE-1 pass CHANGES-REQUESTED on the committed spec, two material gaps, both folded in:
(1) the issue title includes "stale reports" but the first draft deferred report artifacts —
resolved by adding the `<kb>/strategies/<name>/` report tree as an in-scope `report_tree` reap
surface (with `families/` hard-excluded); (2) crash idempotency was incomplete — a crash after the
move but before the manifest append orphaned an archived artifact and the source-absence rule
blocked repair — resolved by the pre-move sidecar + reconcile-first recovery protocol that makes
the archive self-describing and the manifest a rebuildable rollup. Everything else was affirmed
covered (read-only default, registry cross-ref, 90-day retention, orphan mtime floor,
archive-not-delete, verified human actor per #329, DB immutability, protection-by-construction for
every non-`RETIRED` stage including `dormant`).

Third GATE-1 pass CHANGES-REQUESTED, one new material defect: the crash-idempotency claim was
undermined by the allowed cross-filesystem `copy → fsync → unlink` (and non-atomic directory
copy) fallback — a crash after copy but before unlink can leave *both* source and target present,
a state the reconcile matrix did not cover. Resolved by (a) dropping the copy fallback entirely
and mandating a single atomic same-filesystem `os.replace` (files and whole directories), (b)
co-locating the archive root per source root (`<repo>/archive/` for modules, `<kb>/archive/` for
vault artifacts) so rename is always same-fs, (c) skipping+surfacing any genuinely cross-fs
artifact (`skipped_cross_fs`) rather than copying it, and (d) completing the reconcile matrix with
the both-present case (impossible under atomic rename → skip + flag for human, never
double-reaped). Re-run GATE-1 against this file inline.

Fourth GATE-1 pass CHANGES-REQUESTED, one new material defect: the reconcile case `target absent,
source present → redo the move` could archive a source whose eligibility was decided in a prior,
now-stale intent (an intervening `registry add` making the orphan a live/protected strategy, a
source edit, or a retention change). Resolved by making a **pre-commit intent non-authoritative**:
the atomic rename is the sole commit point, so a never-committed move is discarded on reconcile and
the source is re-evaluated from scratch under CURRENT registry + file state (plus a
`src_content_hash` tamper fingerprint that can only downgrade to skip+surface, never re-archive).
Re-run GATE-1 against this file inline.

GATE-2 (Codex, code) reconciliation — spec-vs-reality. The GATE-1 review history above records a
progressively richer *design* for a durable, crash-recoverable archive (per-move sidecar +
`manifest.jsonl` rollup + reconcile-first recovery + co-located dual archive roots + a whole-tree
`report_tree`/`vault_doc` surface). **The shipped implementation deliberately does NOT build that
machinery.** To avoid the spec asserting guarantees the code does not provide (a GATE-2 blocking
finding), the body of this spec was rewritten to describe what actually ships: (1) two scan surfaces
— strategy modules and per-strategy report FILES; the top-level `<kb>/strategies/*.md` vault docs
are NEVER reaped (kb-sync-owned); (2) a SINGLE `--archive-dir` (not co-located dual roots), with a
cross-filesystem source skipped `cross_filesystem`; (3) archive = a single atomic `os.replace` into
a UTC+uuid run dir, with source hardening (`O_NOFOLLOW`/containment/signed-hash/inode), destination
hardening (`_archive_dest_safe` symlink+containment), and a point-of-use registry-stage re-check;
(4) **no sidecar, no `manifest.jsonl`, no reconcile pass** — crash-safety is limited to the
atomic-rename "at most one of {source,target} exists" invariant, and the audit record lives only in
the command's stdout JSON. The durable-record protocol is captured above as a DEFERRED follow-up,
not a shipped guarantee. The additional GATE-2 hardening (collision-resistant run id, destination
containment, registry re-check) is folded into the governed-cleanup section and task list.
