# Knowledge base

This directory is **algua's knowledge base** — open it as an [Obsidian](https://obsidian.md)
vault. It is version-controlled alongside the code: the notes here *are* the knowledge, not a
generated export.

The "graph" you see in Obsidian is **emergent** — it's drawn from the `[[wikilinks]]` between
notes. There's no graph database and no graph-rendering code; Obsidian renders it for free.
Because links resolve across the whole vault, a note in one domain can link straight to a note
in another, and they connect in a single graph.

## Domains

The vault is organised by domain. Each domain is a subdirectory:

- **`strategies/`** — one note per trading strategy, plus thesis-family hubs under
  `strategies/families/`. This is the only domain wired to tooling today (see below).
- **`principles/`** — hand-authored methodology / standards notes (no tooling behind them, unlike
  `strategies/`). Start with `principles/research-methodology.md` — the data-science judgment layer
  above the code walls.

Future domains (research reports, a news field, pivot/decision logs, …) get their own
subdirectory when they have real content and tooling behind them — we don't pre-create empty
folders.

## The `strategies/` domain

Managed by the `algua` CLI — you generally don't hand-create these:

- `uv run algua strategy new <name> --family <slug> --derived-from <parent>` scaffolds a
  strategy note (and its family hub if new).
- `uv run algua strategy doc [<name> | --all]` regenerates the **synced** blocks in each note
  (lifecycle stage from the registry, metrics from MLflow) between
  `<!-- ALGUA:RESULTS -->` / `<!-- ALGUA:MEMBERS -->` markers. Everything outside those markers
  is hand-authored prose and is never touched. A family hub's `MEMBERS` block is a true MOC — a
  stage-grouped roster of `[[member]]` wikilinks, not a count string — so you can click straight
  into members.
- The **generated** roll-ups (don't edit by hand): `strategies/_index.md` is a small **router**
  that links to the per-axis pages — `_by-stage.md` (by lifecycle stage), `_by-date.md` (by created
  month) and `_families.md` (by thesis family). The router stays bounded as the vault grows.

`uv run algua doctor` reports a `knowledge_base` check that flags strategies missing a note or
with a stale synced stage.

The vault root is configurable via `ALGUA_KNOWLEDGE_DIR` (defaults to `kb`).
