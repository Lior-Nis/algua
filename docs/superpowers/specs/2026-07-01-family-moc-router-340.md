# Spec — #340: family hubs as real MOCs + index router layer

GATE-1 (Codex) APPROVED 2026-07-01. No CRITICALs; HIGH/MEDIUM folded in.

## Problem
`algua/knowledge/sync.py` generates the Obsidian vault's synced blocks:
- `sync_family_doc` rendered the MEMBERS block as a **count string** (`"3 members: idea 3"`) —
  zero `[[member]]` wikilinks, so family hubs are navigation dead-ends despite being marketed as
  MOC hubs in `kb/README.md`.
- `generate_indexes` emitted a **flat** `_index.md` (one bullet per strategy). No router, only one
  grouping axis (family), no date axis. At hundreds/day the flat list is an unnavigable wall.

No code parses these markdown files (only Obsidian humans read them), so format changes are safe.
`algua.knowledge` may import `algua.contracts` (import-linter verified).

## Change 1 — linked, stage-grouped member roster
Pure `render_members_block(members: list[tuple[str, str]]) -> str` (mirrors `render_results_block`):
- Empty → `_No members yet._` (matches scaffold placeholder).
- Else: `**{total} members**` line, then `### {stage} ({n})` sections, each `- [[stem]]` (sorted).
- Stage order = canonical lifecycle order from `algua.contracts.lifecycle.Stage`. An unknown stage
  is never dropped — it renders under its own `### {stage}` heading after all known stages.
- Links target `doc.stem` (canonical filename), never frontmatter `name`.
`sync_family_doc` collects `(stem, stage)` pairs and calls the renderer.

## Change 2 — router + axis pages
`generate_indexes` does one non-recursive `base.glob("*.md")` scan (skipping `_`-prefixed and
`type: family`) and writes:
- `_index.md` → **router**: wikilinks to the three axis pages + a `N strategies across M families`
  summary. Bounded by #axes, not #strategies.
- `_by-stage.md` → status axis (lifecycle stage): `## {stage}` groups (lifecycle order, unknown
  last), each `- [[stem]] — {hypothesis_status} · [[family]]` (hypothesis_status preserved).
- `_by-date.md` → date axis: `## {YYYY-MM}` groups (newest first; missing/invalid → `## undated`),
  each `- [[stem]] — {stage} · [[family]]`.
- `_families.md` → unchanged (links use `doc.stem`).
Pure `_created_month(value) -> str` normalizes the date bucket (ISO str / date / datetime / else
`undated`).

## Deferred
Per-month file-sharding of `_by-date` (and stage-sharding of `_by-stage`): `##` headings give
Obsidian fold + newest-first ordering, enough for the issue's horizon at current vault size; the
router can point at a subfolder later without conceptual rework.

## Files
`algua/knowledge/sync.py`, `tests/test_knowledge_sync.py`, `tests/test_cli_strategy.py`,
`kb/README.md`. No CODEOWNERS-protected path.
