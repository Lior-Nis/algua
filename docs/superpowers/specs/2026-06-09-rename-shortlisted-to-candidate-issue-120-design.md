# Rename lifecycle stage `shortlisted` → `candidate` (#120)

**Status:** design · **Date:** 2026-06-09 · **Issue:** #120

## Problem

`shortlisted` is the odd stage out. The other stages name the artifact's
state/runtime (`idea`, `backtested`, `paper`, `live`, `retired`); `shortlisted`
names a *relative selection* and doesn't convey what it represents — having
passed the `research promote` walk-forward + stability gate. `candidate` reads
naturally in the chain `backtested → candidate → paper` and means "cleared the
gate, worth paper-trading."

`forwardtested` was rejected: it collides with the canonical algotrading meaning
of "forward testing" (= paper trading), and the gate is walk-forward = *out-of-
sample backtesting*, still 100% historical. It would mislabel a stage that is
immediately followed by the real forward test (`paper`).

## Decision

A **hard rename** of the enum value `Stage.SHORTLISTED = "shortlisted"` →
`Stage.CANDIDATE = "candidate"`. No backwards-compat alias, no dual-accept of the
old string — the agent code carries no compat cruft. The wire value emitted on
stdout JSON and stored in the DB changes from `"shortlisted"` to `"candidate"` in
one cut, with a DB migration that rewrites existing rows.

## Scope

### Code (behavioral)
- `algua/contracts/lifecycle.py` — rename the enum member and value; update the
  `_LIVE_TRANSITIONS` keys/values (`BACKTESTED → CANDIDATE`, `CANDIDATE → PAPER`,
  `PAPER → CANDIDATE`). The derived retire edge and `ALLOWED_TRANSITIONS` follow
  automatically.
- `algua/cli/research_cmd.py`, `algua/research/gates.py`,
  `algua/registry/transitions.py`, `algua/registry/promotion.py` — every
  reference to the `SHORTLISTED` member or the `"shortlisted"` literal becomes
  `CANDIDATE` / `"candidate"`. (The shortlist *gate* keeps its name where it
  refers to the gate mechanism, not the stage; see "Naming boundary" below.)
- `algua/registry/db.py` — DB migration (below).

### DB migration
The `strategies.stage` and `stage_transitions.from_stage`/`to_stage` columns are
plain `TEXT` (no CHECK constraint), so only a **data** rewrite is needed, no table
rebuild. Following the established `_migrate_*`-before-bootstrap pattern
(`_rekey_search_trials_to_name`):

- Add `_migrate_shortlisted_to_candidate(conn)` that runs **before** the
  `CREATE TABLE IF NOT EXISTS` bootstrap and `UPDATE`s:
  - `strategies SET stage='candidate' WHERE stage='shortlisted'`
  - `stage_transitions SET from_stage='candidate' WHERE from_stage='shortlisted'`
  - `stage_transitions SET to_stage='candidate' WHERE to_stage='shortlisted'`
- **Guard each table independently** with its own `sqlite_master`/table-exists
  check (GATE-1: a fresh DB has neither table when this runs *before* the bootstrap,
  so an unguarded `UPDATE` would raise `no such table`). The `strategies`-only and
  both-tables cases are handled by the per-table guard.
- Naturally idempotent (the `WHERE` matches nothing on a second run).
- **Does NOT gate on `user_version`** — `migrate()` never does (the marker is a
  schema-generation stamp, not a migration cursor). The rewrite therefore *always*
  runs: a DB already stamped `20` that still holds `shortlisted` rows (e.g. created
  fresh, then written by old code) is still corrected on the next open.
- Bump `SCHEMA_VERSION` 19 → 20 (a value migration that changes stored data MUST
  bump the marker, per the db.py header note).
- Update the `gate_evaluations` schema comment in db.py that reads
  `BACKTESTED->SHORTLISTED` to `BACKTESTED->CANDIDATE` (keep the prose name
  "shortlist gate" — that names the *mechanism*; see "Naming boundary").

### Tests
Repoint every `shortlisted`/`SHORTLISTED` reference in:
`test_e2e_lifecycle`, `test_cli_live`, `test_cli_registry`, `test_promotion`,
`test_lifecycle`, `test_registry_store`, `test_cli_paper`,
`test_registry_approvals`, `test_cli_research`, `test_shortlist_gate`.
Migration tests (GATE-1 hardening):
- Seed a DB with a `shortlisted` strategy + a `shortlisted` transition row, run
  `migrate`, assert both read `candidate` and `user_version == 20`.
- **Fresh empty DB** (no tables): `migrate` runs clean, no `no such table`.
- **`strategies`-only DB** (no `stage_transitions`): per-table guard skips the
  missing table, the present table is rewritten.
- **Stale stamped DB**: `PRAGMA user_version=20` with `shortlisted` rows still
  present → the rewrite still runs and corrects them (no version gating).

### User-facing docs / agent surface
- `CLAUDE.md`, `AGENTS.md`, `docs/agent/research-lifecycle.md`.
- Skills under `.claude/skills/` and `.codex/skills/` (operating-algua,
  run-the-research-loop) and `.codex/scripts/run-research-loop.sh`.
- Regenerate `docs/algua-architecture.html` / `docs/algua-lifecycle.html` if they
  are generated from a source; otherwise edit the stage label in place.

### KB frontmatter resync (GATE-1 HIGH)
The Obsidian vault stores a strategy's `stage` in YAML frontmatter
(`sync_strategy_doc`), and `generate_indexes` / `sync_family_doc` count and group by
that raw string. That frontmatter is a **registry projection**, not the source of
truth — but the DB migration does not touch it, so after a migration a
formerly-`shortlisted` doc reads `stage: shortlisted` while the registry says
`candidate`, and `doctor`'s `kb_check` flags it as drift until a resync. Therefore
**the rename's rollout includes `uv run algua strategy doc --all`** to regenerate
the synced blocks from the (migrated) registry. The migration test set asserts
`kb_check` passes after a post-migration `sync_all`. (Locally the vault has no
`shortlisted` docs, so this is a no-op here, but it is part of the documented
rollout for other registries.)

## Intentionally NOT rewritten (historical / denormalized — GATE-1)

These hold the string for *historical* reasons and are deliberately left as-written;
rewriting them would falsify an audit trail and none participate in stage-logic
joins:
- `audit_log` (free-text action/reason) and `stage_transitions.reason` — immutable
  trail; the *typed* `from_stage`/`to_stage` columns that drive `can_transition`
  ARE migrated, the prose reason is not.
- `gate_evaluations.decision_json` — the gate-decision payload (sharpe/breadth/…);
  it does not encode the stage string in any field used for matching.
- MLflow tags/artifacts — verified to tag by strategy `name` + `kind`
  (`backtest`/`sweep`/`walk_forward`/`sweep_combo`), never the stage; no
  `shortlisted` literal lives there.

## Naming boundary (avoid over-rewriting)

The agent promotion gate is internally called the **"shortlist gate"** /
`gate_evaluations` (schema, table, `test_shortlist_gate`). That name refers to the
*gate mechanism that selects candidates*, not to the stage string. The rename
touches only the **stage value** and the symbols that denote the stage. The gate
table name, the `gate_evaluations` rows, and prose describing "the shortlist gate"
stay as-is — renaming them is out of scope and would be churn. Where docs say "the
strategy is *shortlisted*" (the state), that becomes "is a *candidate*."

## Out of scope

Historical specs/plans under `docs/superpowers/` — dated design records, left as
written. Worktree copies under `.claude/worktrees/` (other branches' workspaces).

## Risks

- **Missed reference.** A stray `"shortlisted"` literal that the rename misses
  fails loudly: `Stage("shortlisted")` raises `ValueError` (no such enum value),
  and the gate green-light requires every test to pass. Grep-driven sweep + the
  test suite catch it.
- **External DB with live `shortlisted` rows.** The migration rewrites them on next
  open. Local DB has none; the migration is for other registries.

## Gate

`uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports` green.
