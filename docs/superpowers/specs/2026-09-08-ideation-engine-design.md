# The Ideation Engine — forage, leap, and a pool the loop drinks from (design)

**Date:** 2026-09-08
**Status:** design, approved by the operator after a grilled brainstorm (10 decisions, below).
**Implements:** `docs/PRD.md` §10 step 3 (issue #626); serves §4 (thesis: scale of honest
hypotheses; beliefs are ideation categories) and §7 (the human points at sources, never picks
hypotheses).
**Relates to:** #126 (idea pool, merged), #134 (web sourcing launcher, merged), the strategy
factory (`2026-08-10-strategy-factory-design.md`, slice 1's thesis rotation + run digest).

## 1. The problem

The PRD's thesis is scale of honest hypotheses, and its diagnosis is that the top of the funnel
is starved. Today the funnel's top is three disconnected pieces:

- The **idea pool** (#126): a registry table with a CLI (`research idea add / dedup-check / list
  / set-status / stats`), refuted-aware dedup, `needs_data` parking. It is empty and
  *collection-only*: nothing reads from it. The wiring was deferred until the promotion gate
  counted idea breadth; the factory pivot made the statistical stack advisory, so that blocker no
  longer exists.
- The **web sourcing launcher** (#134, `.codex/scripts/source-ideas.sh`): a Codex agent with
  `web_search`, an arXiv/SSRN paper-search tool and an optional page extractor, writing survivors
  into the pool. Never scheduled. Runs with full local access *and* the real registry path, so a
  hostile page could reach the production database (the sourcing spec's accepted-but-open hazard).
- The **research loop** (`run-research-loop.sh` + the `run-the-research-loop` skill): ideates on
  its own from the strategy notes and an eight-line theme file rotated every 2h. Its playbook
  still names whale momentum as *the* thesis, contradicting the PRD. After the 2026-09-03 reset no
  research timer is installed; nothing ideates on any cadence.
- The **knowledge base** holds strategy notes, one methodology note and a few experience notes.
  Nothing holds what the outside world thinks works.

Ideas are invented per run and forgotten; nothing learns which venues or categories yield; the
only external input is unscheduled.

## 2. The shape

Three stages, two stores, one feedback edge. Ideation is separated from testing, and the pool is
the seam between them.

```
 forage ──────────► kb/inspirations/*.md ──────────► leap ──────────► idea pool (registry) ──► research loop
 daily, web,        one note per inspiration,        depth-driven,     `research idea next`       every 2h, authors
 no registry        git-versioned, Obsidian          no web, CLI       category-balanced          + tests + reports
                                                                            ▲                          │
                                                                            └── outcome + reason ──────┘
                                                                                 `research idea scorecard`
```

- **Forage** ventures into the web (books via summaries and reviews, subreddits, YouTube, blogs,
  forums, papers) for inspiration about what works, and writes **inspiration notes**. It never
  writes the database.
- **Leap** reads fresh inspirations plus what the system has tried and refuted, and takes the
  leap from inspiration to *new* strategy hypotheses with the agent's own knowledge. A critic
  pass filters; survivors enter the pool through the existing dedup. It never reads the web.
- **The loop** draws its next hypothesis from the pool instead of inventing one, and reports the
  outcome back to the idea.
- **The scorecard** aggregates outcomes by venue, category and obscurity so forage and leap can
  steer without a human.

## 3. Decisions (operator-approved, in grill order)

| # | Fork | Decision |
|---|---|---|
| 1 | Where is the engine's boundary? | **The pool is the seam.** The engine's whole contract is "keep the pool full of good, new, untested ideas"; the loop's only change is "consume from the pool". |
| 2 | How do books and authors get in? | **Web-first inspiration, then the agent's leap.** No owned-book ingestion, no literature corpus. The web (book summaries/reviews, Reddit, YouTube, blogs, papers, anything) supplies *inspiration about what works*; the agent's own knowledge takes the leap to a new strategy. |
| 3 | Is inspiration durable? | **Yes: two stages with a store between them.** Inspirations outlive the run that found them, so one can seed many leaps and leaps can combine inspirations found weeks apart. The store is a knowledge-base domain, not a table. |
| 4 | How is the non-widespread found? | **A source registry plus an obscurity rubric.** A versioned venue list steers where forage looks; every inspiration carries an obscurity level; leap prefers obscure. Open searches per (category, market) keep reach beyond the registry. |
| 5 | What must a leap produce? | **A structured hypothesis, then a critic pass.** Mechanism, market/universe, horizon/cadence, signal sketch, construction sketch, required data, a falsification statement, inspiration links. The critic rejects beta-in-disguise, lookahead-by-construction, paraphrases of refuted ideas, and the untestable. |
| 6 | Cadence and depth? | **Forage daily; leap to a target depth; loop every 2h.** Floor ≈ two days of loop consumption, ceiling ≈ one week; both settings. |
| 7 | Feedback? | **A closed loop with a scorecard.** Every consumed idea ends with an outcome and a reason; a scorecard aggregates by venue, category, obscurity and inspiration; forage and leap read it. |
| 8 | Privileges? | **The stage that reads the web cannot write the database; the stage that writes the database cannot read the web.** Containers stay deferred to scale-out. |
| 9 | (ruled, predictable) The loop's playbook and theme file? | The theme file becomes a categories file; the playbook's ideate step becomes "draw from the pool"; the whale paragraph is replaced by the PRD thesis. |
| 10 | (ruled, predictable) Done means? | Both stages installed as user timers, pool non-empty and depth-regulated, loop consuming, every consumed idea carries an outcome, scorecard readable. |

## 4. Inspirations — a new knowledge-base domain

`kb/inspirations/<yyyy-mm-dd>-<slug>.md`, one note per inspiration. Hand-readable in Obsidian,
version-controlled, committed by the trusted driver (never by the foraging agent).

Frontmatter (all required unless noted):

| field | meaning |
|---|---|
| `id` | stable slug, equals the filename stem |
| `found_at` | ISO date of the forage run |
| `source_url` | where it was read |
| `venue` | registry key (e.g. `reddit/algotrading`, `youtube/<channel>`, `ssrn`, `blog/<host>`) |
| `source_kind` | one of `book_summary`, `paper`, `forum`, `video`, `blog`, `other` |
| `category` | one of the categories file (§7) |
| `market` | one of `us_equities`, `crypto`, `forex`, `prediction`, `any` |
| `horizon` | one of `intraday`, `daily`, `weekly`, `monthly`, `event` |
| `mechanism` | one sentence: why money is left on the table and by whom |
| `obscurity` | `canon` / `common` / `niche` / `rare` (rubric in §5) |
| `status` | `fresh` → `used` (≥1 leap cited it) → `exhausted` (leap judged nothing more to take) |
| `leaps` | list of idea ids that cited it (driver-maintained, optional until first use) |

Body: the claim in the agent's own words (what the source says works and why), one short
verbatim quote, and what the agent found doubtful. The note is **untrusted text**: leap treats
it as data, never as instruction.

**Source registry.** `kb/inspirations/_sources.yaml`: a list of venues with `key`, `kind`, `url`
or handle, `categories` it tends to cover, `added_by` (`human` | `forage`), `added_at`, and
`yield` (driver-updated from the scorecard). Forage reads a slice per run and appends venues it
judged worth returning to; the human edits it freely. This file is the one steering surface for
"point the system at sources" (PRD §7).

**No re-reading.** Forage keeps `kb/inspirations/_seen.jsonl` (URL hash, first seen, run) and
skips anything already seen. A source that changed materially is a new entry with a new URL
fragment, not an edit.

## 5. Forage

**Trigger:** a user timer, daily, one run per category slice (the categories file is walked in
order across days, so every category is foraged at least weekly; `FORAGE_SLICES` sets the width).

**Runtime:** a Codex run inside a throwaway worktree on `forage/<stamp>`, with web tools
(`web_search`, the paper-search MCP, the page extractor when a key exists, a YouTube-transcript
tool, Reddit via its JSON endpoints). It is given the category, the market list, the registry
slice, the scorecard summary (as data) and the obscurity rubric. It is **not** given
`ALGUA_DB_PATH` or any registry path, and its prompt forbids running `algua` commands. Its only
output is files under `kb/inspirations/` in the worktree; the driver validates frontmatter
against the schema above, rejects notes that fail, commits the survivors on the forage branch,
lands them on `main` with the same compare-and-swap push the merge-back driver uses (notes
only; the diff policy is "only `kb/inspirations/**` may change", enforced before the push), and
removes the worktree. A run that yields nothing is a
normal run.

**Obscurity rubric** (the agent fills it; the driver only checks the value is legal):

| level | test |
|---|---|
| `canon` | in a standard textbook or a top-cited paper (Fama–French factors, plain trend-following) |
| `common` | on the first page of an ordinary search, in a widely read blog, or in a popular book's main thesis |
| `niche` | discussed in a small community (a subreddit thread, a lesser-known author, a working paper with few citations) |
| `rare` | a single source, an aside, a comment, a footnote, a practitioner's offhand remark |

**Volume bound:** `FORAGE_MAX_NOTES` per run (default 10); the run stops after that many or at
its timeout. Notes are small; ten a day is plenty of raw material for the leap.

**Trust:** every fetched page, transcript or comment is untrusted. Extract the claim, cite the
URL, never obey instructions found in content. The privilege split (§9) is what makes this a
structural property rather than a prompt-level hope.

## 6. Leap

**Trigger:** a user timer every 2h that runs `research idea depth` first and exits immediately
unless the open count is below the floor; it then leaps until the run has produced
`LEAP_MAX_IDEAS` (default 6) or the pool reaches the ceiling. Floor and ceiling are expressed in
**days of loop consumption** (`ALGUA_IDEA_POOL_FLOOR_DAYS` default 2, `ALGUA_IDEA_POOL_CEILING_DAYS`
default 7) and converted to counts from the loop's configured cadence and hypotheses per run
(12 runs × 3 hypotheses = 36/day today, so floor 72, ceiling 252). Leap's own capacity (6 per
run, 72/day) means the ceiling is a stop, not a target; the floor is what keeps the loop fed.

**Runtime:** a Codex run inside a throwaway worktree on `leap/<stamp>`, **no web tools**
(`web_search=disabled`, no MCP servers), with `ALGUA_DB_PATH` pointing at the real registry so
`research idea add` writes the pool (that is the *only* write it is expected to make; the
diff policy on its worktree is "no file changes" and the driver discards the worktree without
committing). Inputs it reads from the checkout: fresh inspirations (obscure and recent first),
the refuted ideas with their reasons (`research idea list --status refuted`), the negative-result
ledger (`research log list`), the strategy notes' family hubs, the methodology note, and the
scorecard.

**The leap.** From one or more inspirations it forms a hypothesis that is *not* the inspiration
restated: a transfer across markets or horizons, a combination of two mechanisms, an inversion,
a different construction on the same signal, or a genuinely new idea the inspiration triggered.
Each hypothesis is structured:

| field | pool column | note |
|---|---|---|
| title | `title` | |
| hypothesis | `hypothesis` | mechanism (who leaves the money and why) + signal sketch + construction sketch, prose |
| category | `category` (new) | from the categories file |
| market | `market` (new) | as in §4 |
| horizon | `tags` | `horizon:<value>` |
| required data | `required_data` | existing controlled vocabulary; unsupported parks as `needs_data` |
| falsification | `falsification` (new) | "this idea is refuted if …", one or two sentences the interpret step can check |
| inspirations | `source_ref` + `tags` | `source_type=inspiration` (new enum value); `source_ref` = comma-separated inspiration ids; `tags` carry `venue:<key>` and `obscurity:<level>` copied from the strongest inspiration |
| family | `family` | an existing family slug or a proposed new one |

**The critic.** A second pass in the same run, prompted with `kb/principles/research-methodology.md`
as its lens, rejects a hypothesis when any of these hold: it is beta or a known factor in disguise
without a distinct mechanism; it uses information that would not be known at decision time
(lookahead by construction); it paraphrases a refuted idea or a negative-result entry; it cannot
be tested on data the platform has and is not worth parking; its falsification statement is
vacuous. Rejections are written to the negative-result ledger with reason `critic:<kind>` so the
next leap does not re-derive them. Survivors go through `research idea dedup-check` then
`research idea add`; a dedup collision is respected (no `--allow-duplicate` on the agent path).

**Bookkeeping after the run (driver, trusted).** The leap driver lists the pool rows with
`source_type=inspiration` created during the run, and for every inspiration id they cite updates
that note's `status` (`fresh` → `used`) and `leaps` list, plus marks `exhausted` any note the
agent reported as spent in its run report; it commits those note edits on `main` as a notes-only
commit (same compare-and-swap push as forage). The agent itself never edits notes.

## 7. The pool and the loop

**Schema (v45 → v46).** `ideas` gains `category TEXT`, `market TEXT`, `falsification TEXT`,
`outcome TEXT`, `outcome_reason TEXT`, `claimed_by TEXT`, `claimed_at TEXT`. `SourceType` gains `INSPIRATION`.
`IdeaStatus` is unchanged. Existing rows keep NULLs; `research idea add` requires `--category`
and `--market` only when `--source-type inspiration`.

**`research idea next --run <stamp>`** (new; the loop's ideate step). Picks one unclaimed `open`
idea and **claims** it (`claimed_by=<stamp>`, `claimed_at=now`) in the same transaction; the
status stays `open` (the strategy does not exist yet) but claimed ideas are excluded from
selection. The loop later links it with the existing `set-status --to authored --strategy <name>`
once the strategy is registered. A claim whose run ends without an outcome is released by
`research idea outcome <id> --outcome abandoned` (the driver calls it for every claimed-but-
unresolved idea when a run exits), so a crashed run never strands an idea. Selection:
round-robin across categories (the category with the fewest ideas consumed in the trailing 7 days
first), then obscurity `rare` > `niche` > `common` > `canon`, then oldest `created_at`. Emits the
full idea JSON. `--dry-run` shows the pick without consuming. `--category` / `--market` restrict.
If the pool is empty it fails closed with `pool_empty` and the loop skips the run (the digest
records `pool_empty`; the leap timer will refill).

**`research idea depth`** (new): `{open_unclaimed, claimed, needs_data, floor, ceiling,
below_floor}` with floor/ceiling already converted to counts.

**`research idea outcome <id> --outcome X --reason "..."`** (new). `X` is one of
`integrity_fail`, `holdout_negative`, `walkforward_refuted`, `sweep_unstable`,
`promoted_candidate`, `paper_early_kill`, `paper_retired`, `forward_survivor`, `abandoned`
(the run ended without a verdict; releases the claim and leaves the idea `open`). The first six are set by the research loop's report step
from the preview-gate verdict; `paper_early_kill` / `paper_retired` / `forward_survivor` are set
by the paper lane's existing stage transitions when the strategy has an `authored_strategy_id`
link (one call at the transition site, best-effort, never blocking a transition). A refuting
outcome also moves the status to `refuted` (existing transition), so dedup keeps working.

**The research loop.** `run-the-research-loop`'s step 1 becomes: run `research idea next`, read
the idea, author the strategy from it (the author agent is handed the structured hypothesis and
the falsification statement), and after the interpret step call `research idea outcome`. The
whale-thesis paragraph is replaced by the PRD §4 thesis. `.codex/research-themes.txt` is deleted;
`.codex/categories.txt` lists the PRD §4 categories, one per line with optional `market=` and
`horizon=` hints; the launcher no longer rotates a thesis (the pool carries diversity) and
`--thesis` becomes a `--category` restriction. The run digest gains `idea_id` per hypothesis and
a `pool_empty` outcome.

**Scorecard.** `research idea scorecard [--days N]` (default 90): for each of venue, category,
obscurity and inspiration id, the counts of ideas consumed and of each outcome, and two rates:
*integrity yield* (integrity pass ÷ consumed) and *survival yield* (promoted + forward survivor ÷
consumed). A JSON read; the monitor gets an "Ideas" view rendering it beside the existing funnel
counts. The spec says plainly: forward survivors will be rare for a long time; the early signal
is integrity and walk-forward yield, and the scorecard is honest about counts. The driver writes
the per-venue `yield` back into `_sources.yaml` after each scorecard run so forage can weight the
registry.

## 8. Cadence and installation

| unit | schedule | runs |
|---|---|---|
| `algua-forage.timer` | daily 03:00 UTC (offset from research runs) | `.codex/scripts/forage.sh` |
| `algua-leap.timer` | every 2h at :30 (between research runs) | `.codex/scripts/leap.sh` (exits fast when above floor) |
| `algua-research.timer` | every 2h at :00 (existing, reinstalled) | `run-research-loop.sh` |

All user units, installed by the existing `install-user-units.sh`; overlap protection is the
existing non-blocking flock per launcher. Settings (`ALGUA_IDEA_POOL_FLOOR_DAYS`,
`ALGUA_IDEA_POOL_CEILING_DAYS`, `LEAP_MAX_IDEAS`, `FORAGE_MAX_NOTES`, `FORAGE_SLICES`) live in
the env file, documented in the units' README.

## 9. Privileges and safety

| stage | web | registry DB | writes | commit |
|---|---|---|---|---|
| forage | yes | **no path given** | `kb/inspirations/**` in its worktree only | driver validates + commits |
| leap | **none** | yes, via CLI | pool rows via `research idea add`; negative-result ledger via `research log record` | nothing (worktree discarded) |
| research loop | none | scratch copy (existing explore isolation) | strategies + report on its branch | driver (existing) |

The prompt-injection surface (forage) and the database write surface (leap) never share a
process. Forage's notes are still untrusted text when leap reads them; leap's prompt says so.
Leap's write surface is the pool and the advisory ledger, neither of which touches stage, gates,
allocations or the live wall; a hostile inspiration can at worst add a bad idea, which the loop
then refutes and the scorecard then attributes to its venue. Containers remain deferred to the
VPS lift (unchanged decision from #134).

## 10. Not in scope

- Containers / OS isolation (deferred to scale-out, unchanged).
- Per-market category quotas before step 5 lands market data; `market` is recorded now so the
  scorecard can split later.
- Embedding-based dedup; the existing lexical + structural gate plus the critic is enough.
- Owned-book ingestion or a literature corpus (decision 2).
- Changing the PRD's ownership line for step 3 from Eitan to the operator: a one-line PRD edit
  in its own PR, not part of this build.
- Any change to gates, stages, allocations, or the live wall.

## 11. Done

Step 3 is complete when: both timers are installed and have fired; the pool holds ideas with
`source_type=inspiration` and stays between floor and ceiling across a day; a research run
consumed an idea via `research idea next` and wrote its outcome; `research idea scorecard`
reports at least one venue with a non-zero consumed count; the theme file is gone and the
playbook names the PRD thesis.
