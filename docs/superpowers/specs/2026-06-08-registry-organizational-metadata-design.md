# Registry organizational metadata — design

**Issue:** #122 — Registry has no organizational metadata (family / tags / author / hypothesis) — pool isn't filterable at scale
**Date:** 2026-06-08
**Status:** approved design, pre-implementation (GATE 1 design review folded in)

## Problem

The registry is the lifecycle source of truth the agent queries, but `strategies` is just
`id, name, stage, created_at, updated_at`; the only filter is `registry list --stage S`.
Organizational metadata (thesis **family**, **tags**, **author**, **hypothesis_status**,
**derived_from**, a short **description**) lives **only** in hand-curated Obsidian kb frontmatter
(`kb/strategies/`), which is synced one-way *from* the registry and is **not queryable**. At
hundreds of strategies the agent cannot answer questions it must reason over the pool with, e.g.
*"all `untested` ideas in the `mean-reversion` family authored by the agent."*

## Goal

Promote organizational metadata to **first-class, queryable registry fields**, make the registry
the single source of truth for them, and have the kb frontmatter **derive** from the registry
(not diverge). Extend `registry list` with filters and ensure the write paths populate the fields.

## Non-goals (deferred)

- Auto-deriving `hypothesis_status` from research gates (`research promote/discard`). Manual
  `registry set` is the mutation path for now; gate-driven auto-status is a follow-up on top of it.
- The `strategies/examples/` folder restructure (paired issue, separate change).
- A metadata-change audit-log table, kb↔registry drift detection in `doctor`, and
  `--no-family`/`--any-tag` filter variants. All YAGNI until a concrete need appears.

---

## 1. Data model

### Schema (registry/db.py)

Bump `SCHEMA_VERSION` 16 → 17. Add six columns to `strategies` via a new `_add_missing_columns`
call in `migrate()`. **All columns are added NULL — no SQL `DEFAULT`** (this is what
`_add_missing_columns` already does; existing rows get NULL):

| Column | Type | New-row default (repo layer) | Notes |
|---|---|---|---|
| `family` | TEXT | NULL | bare slug, e.g. `mean-reversion` |
| `tags` | TEXT | `'[]'` | JSON array (canonicalized — see below) |
| `author` | TEXT | `'agent'` | enum `agent\|human` |
| `hypothesis_status` | TEXT | `'untested'` | enum `untested\|supported\|refuted\|inconclusive` |
| `derived_from` | TEXT | NULL | parent strategy **name** (bare) |
| `description` | TEXT | NULL | short one-liner |

**Why NULL-not-DEFAULT (GATE-1 MEDIUM-1):** a SQL `DEFAULT 'agent'`/`'untested'` would stamp every
*existing* row at migration time, so the backfill would see no NULLs and skip the kb-authored
values — the exact data it exists to recover. Defaults therefore live in the repository's `add()`
for **new** rows only; existing rows stay NULL until the backfill (or a final default-fill) touches
them.

### Enums (new file `algua/contracts/registry_metadata.py`)

`Author` and `HypothesisStatus` are registry-taxonomy concepts, **not** lifecycle transitions, so
they go in their own contracts module rather than muddying `contracts/lifecycle.py` (GATE-1
MEDIUM-3). Both are `StrEnum`, pure, no I/O:

```python
class Author(StrEnum):
    AGENT = "agent"
    HUMAN = "human"

class HypothesisStatus(StrEnum):
    UNTESTED = "untested"
    SUPPORTED = "supported"
    REFUTED = "refuted"
    INCONCLUSIVE = "inconclusive"
```

### Tag canonicalization contract (GATE-1 MEDIUM-4)

SQLite won't enforce shape, so the repository owns it. A single helper canonicalizes any tag list
before storage and is the only writer of the `tags` column:

- trim whitespace, lowercase, **reject empty strings**, **dedupe**, **sort**, then `json.dumps`.
- reads parse with a `json_valid(tags)`-guarded query (fall back to `[]` if somehow invalid).

`--tag` filtering uses `EXISTS (SELECT 1 FROM json_each(strategies.tags) WHERE value = ?)`.

### `StrategyRecord` (registry/repository.py)

Gains: `family: str | None`, `tags: list[str]`, `author: Author`,
`hypothesis_status: HypothesisStatus`, `derived_from: str | None`, `description: str | None`.
`_row_to_record` parses `tags` JSON and coerces the two enums on read: a NULL `author`/
`hypothesis_status`/`tags` (an existing, not-yet-backfilled row) reads as its default
(`agent`/`untested`/`[]`) — which **is** the semantic meaning of "no value recorded." New rows
always write concrete values. To keep this display coercion consistent with filtering (so a row
shown as `agent` is also matched by `--author agent`), the **filters use `COALESCE`** on these
three columns (see §3). `family`/`derived_from`/`description` have no default — NULL means
genuinely unset, displayed as `null`, and never matched by a provided filter value.

---

## 2. Write paths (CLI)

### `registry add` — THE metadata write path

`add` gains options: `--family`, `--tag` (repeatable), `--author` (default `agent`),
`--hypothesis-status` (default `untested`), `--derived-from`, `--description`. The repository
`add()` signature gains these as keyword args (with the new-row defaults above), writes them in the
same transaction as the existing insert + initial transition row.

**Validation (at the CLI/repo boundary):**
- `author` / `hypothesis_status` must be valid enum values (Typer/`StrEnum` rejects others).
- `family` must match the existing slug regex `^[a-z0-9][a-z0-9-]*$`.
- `derived_from`, if given, **must reference an existing registered strategy** and **must not be the
  strategy itself** (GATE-1 MEDIUM-2, light). We keep `derived_from` a bare free-text **name** (not
  a self-FK) to match the codebase's deliberate denormalization philosophy where references survive
  deletion; we reject typos/non-existent/self-reference but do **not** build multi-hop cycle
  detection (YAGNI for single-user lineage).

### `registry set` — mutate metadata post-creation (new command)

`registry set <name>` mutates organizational metadata and **never touches stage** (stage stays on
`transition`). Options: `--family`, `--add-tag`/`--remove-tag` (repeatable), `--author`,
`--hypothesis-status`, `--derived-from`, `--description`. Same validation as `add`. Repository gains
a focused `update_metadata(name, ...)` that updates only the provided fields, bumps `updated_at`,
and re-canonicalizes tags. After the DB write, the command **re-syncs the kb doc** for that
strategy. Emits **before/after** values of the changed fields in its JSON response (GATE-1 LOW-2,
light) — cheap provenance without a separate audit table.

### `strategy new` — now registers (GATE-1 HIGH-2, atomicity)

`strategy new` currently scaffolds module + kb doc and never touches the registry. It now becomes a
registering command so the kb doc can derive from the registry record. To avoid partial states
across DB + filesystem:

1. **Preflight everything before any write:** name is a valid non-keyword identifier; family slug
   valid; module path does not exist; kb doc path does not exist; `derived_from` (if given) exists;
   **registry name not already taken**.
2. **Register** via `registry add` (passing `--family`/`--derived-from`/etc.).
3. **Scaffold** the module file, kb doc, family hub.
4. **Rollback on scaffold failure:** if any filesystem write fails after the registry insert,
   delete the just-created registry row and emit a partial-state JSON error. Registration is
   ordered first (transactional, fast) so a failure in the flaky FS step leaves nothing registered.

This adds the only strategy-deletion path in the codebase — scoped strictly to the rollback of a
row this same call just created (not a general delete surface).

---

## 3. Read / filter (CLI)

- `registry list` gains `--family`, `--tag` (repeatable), `--author`, `--hypothesis-status`
  filters, composable and **AND-ed** with each other and with the existing `--stage`. Repository
  `list_strategies` grows optional filter params; the query builds a parametrized `WHERE` from the
  ones supplied.
- **Filter semantics (GATE-1 LOW-1):** an omitted filter means no constraint; `--family X` is exact
  match on the canonical slug; repeated `--tag a --tag b` means **all-of** (AND). `--no-family` /
  `--any-tag` are deferred until needed.
- **NULL handling in filters:** `--author` and `--hypothesis-status` filter on
  `COALESCE(column, '<default>')` so a not-yet-backfilled NULL row (semantically the default) is
  matched consistently with how it displays; `--tag` filters `json_each(COALESCE(tags, '[]'))`.
  `--family` is plain equality (no default), so it never matches a NULL-family row.
- `registry list` stays a **bare JSON array** (the documented collection exception); row objects
  gain the new keys — an **additive** change. Existing consumers reading `id/name/stage` keep
  working (GATE-1 MEDIUM-5). `registry show` adds the same fields to its envelope.

---

## 4. kb sync — the inversion (GATE-1 HIGH-1)

`sync_strategy_doc` becomes the point where registry metadata flows into kb frontmatter. The CLI
seam passes a plain metadata dict in (exactly as it passes `stage` today), so the knowledge layer
**stays registry-free**.

**Frontmatter ownership contract** (to avoid silently clobbering human edits):
- The set of **registry-owned keys** is exactly `{family, tags, author, hypothesis_status,
  derived_from, description, stage, mlflow_run}`. `sync_strategy_doc` overwrites **only** these and
  **preserves every other frontmatter key** (the current parse-mutate-render already preserves
  unknown keys — we keep that property and extend the owned set).
- `family` and `derived_from` are written wrapped as `[[wikilink]]` for Obsidian; the registry
  stores bare names. `_unwikilink` (already present) is the inverse for reads/backfill.
- The contract is documented in the spec and in the scaffold/sync docstrings: **metadata is edited
  via `registry set`, not by hand in the kb.** Overwriting the owned keys is therefore intended, not
  accidental. We deliberately **decline** the heavier `algua_*` key-prefix and `--force` diff-gate
  (GATE-1 suggested alternatives) as overkill for a single-user system — they'd uglify Obsidian
  dataview queries for a workflow that `registry set` already replaces.

`scaffold_strategy_doc` and `strategy new` keep writing initial frontmatter, but the values now come
from the registry record created in the same `strategy new` call (single source).

---

## 5. Backfill (`registry backfill-from-kb`, new command) (GATE-1 HIGH-3)

A one-shot CLI command that recovers the kb-authored metadata into the freshly-NULL registry
columns. It lives in the **CLI layer** (orchestrating both registry + knowledge) so `migrate()`
stays pure-SQL and the registry↔knowledge boundary holds.

For each registered strategy with a kb doc:
- read frontmatter; map `family`/`derived_from` through `_unwikilink` to bare names; read `tags`,
  `author`, `hypothesis_status`, `description`.
- **Fill where NULL only.** Because the migration adds the columns NULL and the backfill runs once
  before any registry writes populate them, there are no pre-existing non-NULL values to conflict
  with — so plain fill-NULL is correct and idempotent (re-running is a no-op). We deliberately
  **decline** the full `--prefer-kb/--prefer-registry` reconciliation engine; it solves a conflict
  case that cannot occur on the one run this command is for.
- **Enum mapping is explicit:** a kb `hypothesis_status` not in the enum (or absent) is reported as
  `unmappable` and left for the operator, not silently coerced. Same for an invalid `author`.
- **Report, don't hide:** emit a JSON report — `{processed: [...], unmappable: [...],
  kb_docs_without_registry_row: [...], registry_rows_without_kb_doc: [...]}` — so nothing is
  silently dropped. `processed` = strategies whose kb doc was found and reconciled (fill-only-NULL
  means already-populated columns are untouched, so this is not a "changed" list).
- **Final default-fill:** after kb recovery, any row still NULL on `author`/`hypothesis_status`/
  `tags` gets the standard defaults (`agent`/`untested`/`[]`) so the columns are never left NULL on
  an existing row.

---

## Build order (slices)

Each slice ends green on the gate: `pytest && ruff check . && mypy algua && lint-imports`.

1. **Schema + model + enums.** New `contracts/registry_metadata.py`; `SCHEMA_VERSION`→17 with the
   six NULL columns; `StrategyRecord` + `_row_to_record` gain the fields; tag-canonicalization
   helper; repository `add()` accepts the new kwargs (new-row defaults). `registry list`/`show`
   emit the new fields. Migration is safe to ship before the backfill (columns are NULL/defaulted).
2. **Write paths.** `registry add` options; `registry set` command + `update_metadata` + before/
   after emit; `derived_from` existence/self-reference validation. kb re-sync wired into `set`.
3. **Read / filter.** `list_strategies` filter params; `registry list` filter options with the
   AND/exact semantics.
4. **kb inversion.** `sync_strategy_doc` writes the owned-key set from the passed-in metadata dict;
   ownership contract documented; CLI seam passes the metadata dict.
5. **Backfill.** `registry backfill-from-kb` report-first fill-NULL + default-fill.
6. **`strategy new` coupling — last.** Preflight + register + scaffold + rollback, after the
   write-path + rollback contract are proven by the earlier slices.

## Boundaries / invariants

- `algua/contracts/*` and `algua/features/*` stay pure. New enums in `contracts/registry_metadata.py`
  import nothing with I/O.
- Knowledge layer never imports the registry; registry never imports knowledge. The CLI orchestrates
  both at the seam (`backfill-from-kb`, `strategy new`, `registry set` re-sync). `lint-imports` must
  stay green.
- `registry set` and `transition` are disjoint: stage only ever moves through `transition`;
  organizational metadata only ever moves through `add`/`set`/`backfill`.

## Testing

- `test_registry_db`: schema version 17; the six columns exist after migrate; idempotent re-run;
  existing rows are NULL (not defaulted) post-migration.
- `test_registry_store`: `add` with metadata round-trips; new-row defaults; tag canonicalization
  (trim/lowercase/dedupe/sort/reject-empty); `update_metadata` partial updates; filter queries
  (family exact, multi-tag AND, author, hypothesis_status, composability); `derived_from` existence
  + self-reference rejection.
- `test_cli_registry`: `add`/`set`/`list --filters`/`show` JSON; `set` before/after; bare-array
  shape preserved; `backfill-from-kb` report (filled / unmappable / orphans) on a seeded kb.
- `test_cli_strategy`: `strategy new` registers + scaffolds; preflight rejections; rollback leaves
  no registry row when a scaffold write fails.
- `test_knowledge_sync`: `sync_strategy_doc` writes the owned keys from a metadata dict, wraps
  `family`/`derived_from`, and preserves a foreign frontmatter key.
