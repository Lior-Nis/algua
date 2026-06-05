# Strategy Knowledge Base — Design

**Date:** 2026-06-03
**Status:** Approved (design); implementation plan to follow.
**Slice of:** the human-facing observability surface (sub-project 4/5 adjacent). This is the
first slice; a later slice adds the web dashboard that sits on top of this KB.

## 1. Purpose

Give the platform a **living knowledge base of strategies** that serves two operators at once:

- **The human** — full observability of the research space as an emergent graph (opened in
  Obsidian): which theses are being explored, which are barren, how strategies derive from
  one another, and what the agent intends to try next.
- **The agent** — durable context for *what to try next*. Today, the reasoning behind a
  strategy (why it was tried, why it failed, what it implies) evaporates into MLflow runs
  nobody re-reads. The KB captures that prose so knowledge **compounds** across the research
  loop instead of being re-derived.

The knowledge graph the human wants is **emergent**, not engineered: it is whatever Obsidian
renders from `[[wikilinks]]` between markdown docs. There is no graph database and no
graph-visualisation code to build.

## 2. Model: living KB, doc-primary

Each strategy and each thesis-family is a **markdown file** the agent authors and reads
(the same substrate as the agent memory system: per-item files + frontmatter + wikilinks +
a generated index). Each doc has two zones:

- **Authored prose** — hypothesis, derivation, verdict, "what to try next", family thesis,
  exploration state. **This knowledge exists nowhere else** and is the compounding asset.
- **Synced facts** — stage, metrics, repro stamps, member rosters. **Generated** from the
  registry + MLflow into marker-delimited blocks; never hand-edited.

Rejected: *doc-as-export* (DB-primary, docs are dead read-only renders). It avoids drift but
has no home for the prose knowledge that makes the KB worth having.

## 3. Source-of-truth boundaries (the anti-drift rule)

Every fact has exactly one owner. The KB is a purely **additive** layer — **the registry
needs zero schema changes.**

| Concern | Authoritative owner | Appears in the doc as |
|---|---|---|
| stage, transitions, approvals | registry (SQLite) | synced badge |
| metrics, repro stamps, run id | MLflow | synced Results block |
| hypothesis, derivation, verdict, next | the doc (authored) | the prose |
| family thesis, exploration state, `status` | the doc (authored) | the prose |
| lineage (`derived_from`, `family`) | the doc frontmatter (authored) | wikilinks → graph |
| `_index`, `_families`, member rosters | derived | generated files/blocks |

Lineage lives in **doc frontmatter**, not a SQL table. The agent declares lineage when it
authors a strategy; nothing queries a lineage table because the agent reads the docs directly.

## 4. Artifacts

### 4.1 Vault layout

> **Updated 2026-06-05 (PR #111):** the vault root moved from `docs/strategies/` to a
> top-level `kb/` so the knowledge base can hold domains beyond strategies (research, news,
> pivots) in one Obsidian graph. Strategy docs now live under `kb/strategies/`. The structure
> below is otherwise unchanged. `knowledge_dir` (default `kb`) is the vault root.

```
kb/                         # the Obsidian vault (mount this)
└── strategies/             # the strategy domain
    ├── _index.md           # generated: one line per strategy
    ├── _families.md        # generated: families + status + best result
    ├── families/
    │   ├── dark-pool-leadlag.md
    │   └── 13f-momentum.md
    ├── dark-pool-leadlag-v1.md
    ├── dark-pool-leadlag-v2.md
    └── ...
```

`[[wikilinks]]` resolve across folders — the layout is for human tidiness; the graph is
folder-agnostic. The `_` prefix sorts the generated indices to the top.

### 4.2 Strategy doc

Frontmatter:

```yaml
---
name: dark-pool-leadlag-v2
stage: backtested            # synced from registry
family: "[[dark-pool-leadlag]]"
derived_from: "[[dark-pool-leadlag-v1]]"   # omit/empty for a root strategy
hypothesis_status: refuted   # untested | testing | supported | refuted
mlflow_run: a3f9c1
created: 2026-06-03
---
```

Body sections:

- `## Hypothesis` — authored. What edge is being claimed and why.
- `## Derivation` — authored. What this was forked from and what changed; links to parent
  and family.
- `## Results` — **synced**, between `<!-- ALGUA:RESULTS -->` / `<!-- /ALGUA:RESULTS -->`.
  Train vs walk-forward metric table, gate verdict, dataset snapshot id, seed.
- `## Verdict & next` — authored. What was learned, and the next idea — including a
  `[[dangling-link]]` to a not-yet-authored strategy. A dangling link renders in Obsidian as
  a clickable future-idea node; this is how "suggestions rise from what we tested" with no
  recommender engine.

### 4.3 Thesis-family doc

Frontmatter:

```yaml
---
type: family
name: dark-pool-leadlag
status: exploring            # exploring | promising | exhausted | parked
created: 2026-05-28
---
```

Body sections:

- `## Thesis` — authored. The hypothesis for the whole family, stated once.
- `## Members` — **synced**, between `<!-- ALGUA:MEMBERS -->` / `<!-- /ALGUA:MEMBERS -->`.
  Counts by stage + best walk-forward metric, from the registry.
- `## State of exploration` — authored. Bulleted members with one-line outcome + status.
- `## Open questions / next` — authored. Which axes are exhausted, which remain, when to
  park. This section is the engine of the agent's next-idea selection and the human's
  "promising vs barren" read.

The family `status` field is the single most valuable human-glanceable signal in the KB.

## 5. The marker / sync contract

Synced blocks are delimited by HTML-comment markers (`<!-- ALGUA:RESULTS -->` …
`<!-- /ALGUA:RESULTS -->`, `<!-- ALGUA:MEMBERS -->` … `<!-- /ALGUA:MEMBERS -->`). The
generator rewrites **only** the bytes between a marker pair and never touches prose. This
mirrors the existing `GENERATED_BY` additions-only discipline used for strategy code, so the
contract is already familiar. A doc missing a marker pair gets one inserted at generation
time in the canonical section position.

## 6. Commands & lifecycle

The KB hooks into the existing command surface; it adds exactly one new command and one
`doctor` check. No existing command gains a hidden side effect.

1. **`algua strategy new <name> [--family F] [--derived-from P]`** — *extended* to also
   scaffold `docs/strategies/<name>.md`: frontmatter populated from the flags, empty
   `## Hypothesis` / `## Derivation` for the agent to fill, and empty synced blocks. If
   `--family` names a family with no doc yet, scaffold the family doc too.
2. **`algua strategy doc [<name> | --all]`** — **new.** A pure projection: read registry +
   MLflow, rewrite the marker-delimited blocks in the target doc(s), and regenerate
   `_index.md` and `_families.md` by scanning all doc frontmatter joined with stage +
   metrics. Idempotent; emits JSON describing what it wrote. Touches no prose.
3. **`algua doctor`** — *extended* with a KB check: registry strategies with no doc, docs
   whose synced block is stale vs the registry, docs with malformed/duplicate frontmatter.
   Reports as a `doctor` check result.

`backtest run` / `walk-forward` / `research promote` stay pure compute→JSON commands. The KB
sync lives in `algua/knowledge/` (new module) which may read registry + tracking but is
imported by neither `backtest/` nor `live/`; import-linter gains a rule to enforce this.

## 7. The research loop closes through the KB

`run-the-research-loop` skill integration (the skill is edited; this spec specifies the beats):

1. **Orient** — read `_index.md` + `_families.md`; pick a family that is `exploring`/
   `promising` with an open axis in its `## Open questions`.
2. **Ideate** — choose the most promising open axis; if a new thesis, scaffold a family doc.
3. **Author** — `strategy new --family --derived-from`; write Hypothesis + Derivation.
4. **Test** — `backtest run` / `walk-forward` / `research promote` (unchanged).
5. **Record** — run `strategy doc` to sync; write `## Verdict & next`; set
   `hypothesis_status`; update the family `## State of exploration` + `status`; drop a
   `[[dangling-link]]` for the next idea.

The KB is the loop's input (where to look) and output (what was learned) — knowledge
compounds instead of evaporating.

## 8. The human's view

For this slice, **Obsidian is the UI.** Open `docs/strategies/` as a vault:

- **Graph view** is the live map — family hubs, strategy clusters, dangling future-ideas,
  parked dead-ends.
- **Read** family docs for the narrative; `status` and `hypothesis_status` drive Obsidian
  node colours/tags.
- Zero custom UI code in this slice.

## 9. Scope

**In scope**

- Strategy-doc and family-doc schema + the marker/sync contract.
- `strategy new` doc scaffolding; new `strategy doc` projection command; `_index` /
  `_families` generation; `doctor` KB check.
- New `algua/knowledge/` module + import-linter rule.
- `run-the-research-loop` skill integration.

**Out of scope (later slices)**

- Web dashboard, PnL aggregation across paper/live/backtest, approval cockpit, agent
  reasoning feed. These sit on top of this KB + registry + MLflow and get their own
  brainstorm (including their own framework choice).
- Any registry schema change.
- Data/feature/run-report nodes in the graph (option C) — deferred unless a real need
  appears; provenance is already captured structurally.

## 10. Risks / open questions

- **Prose staleness.** Synced facts can't drift (regenerated), but authored prose can go
  stale relative to the family's real state. Mitigation: the loop's Record step updates the
  family doc every cycle; `doctor` can later flag families whose newest member post-dates
  the last `## State of exploration` edit. Start with the loop discipline; add the check if
  staleness shows up.
- **Wikilink hygiene.** A typo'd `[[link]]` silently creates a phantom node. The `doctor`
  KB check should list dangling links so intentional future-ideas can be distinguished from
  typos (e.g. by a convention: future-ideas live only in a `## Verdict & next` section).
- **Index generation cost** at large strategy counts — `strategy doc --all` rescans the
  vault. Fine at the scale of this slice; revisit if the vault reaches thousands of docs.
