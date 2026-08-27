# Monitor redesign — a human-facing instrument

**Status:** design approved 2026-08-27, not yet implemented
**Supersedes nothing.** Builds on the slice-3 run-ledger views
(`2026-08-23-strategy-run-tracking-design.md`), which stay as they are.

## 1. The problem, measured

The monitor was built as one surface for two audiences and serves neither. Measured on
the deployed instance at a 390px viewport (iPhone 13, real data, 2026-08-27):

| tab | height | words | real marks |
|---|---|---|---|
| Now | 4,134px (~6 screens) | 365 | 0 |
| Fleet | 1,603px | 129 | 0 |
| Money | 850px | 80 | 0 |
| Research | 853px | 57 | 0 |

Zero visualizations across four screens. The Now tab's bulk is a `RECENT ACTIVITY`
feed rendering raw merge-back payloads — `base_sha`, `branch`, `merge_sha`, event
JSON — verbatim into the DOM.

**The decision that drives this whole spec:** that content has exactly one real
consumer, and it is not a human. Agents read the same facts as JSON through the CLI,
which is the documented seam (`CLAUDE.md`: "Every data command emits JSON on stdout").
Rendering it again as prose serves nobody and costs six screens.

**So: the monitor is a HUMAN-ONLY surface.** No log dumps, no SHAs, no event streams,
no JSON. Anything an agent needs stays in the CLI/API and leaves the UI entirely. This
is a subtraction, not a migration — nothing needs a new home.

### 1.1 A confirmed bug, and what it reveals

`/money` forces a 500px layout viewport at 390px: `.data-table` measures 483px (five
columns, long monospace strategy names such as `cadenced_tail_risk_relative_strength`)
and *sizes* its `.table-scroll` parent rather than scrolling inside it, so the page
grows instead of the table clipping. Now, Fleet and Research all sit at exactly 390.

The fix is not a CSS patch. A five-column table of long identifiers is the wrong mark
for a 390px screen; §4.2 replaces it, which removes the cause. §5.1 then makes the
class of bug unrepresentable.

## 2. What the monitor is for

The operator answers three questions, in this priority order:

1. **Do I need to act?**
2. **Is it making money?**
3. **Is the funnel producing?**

Ranked by urgency — but ranking is expressed by a **fixed skeleton with one variable
slot**, not by reordering. Bands always appear in the same order so the operator builds
positional memory; only the top slot changes shape (all-clear mark vs. exception
cards). A layout that reorders itself forces the reader to *read* to locate things,
which recreates the verbosity problem structurally, and makes a good day
indistinguishable from a different screen.

**The trade this accepts:** on a bad day the second and third problems sit below the
fold in their usual places rather than being promoted. The attention slot carries a
count and the exceptions are tappable, which is judged sufficient.

### 2.1 Validated before speccing

The Now tab was built as a rendered mock at 390px against a rich fixture and measured:
**664px, 42 words, 3 marks** — and, critically, **identical at 10 and at 65
strategies** (cells wrap; no second mode needed at scale). Against the current Now
tab: 4,134px → 664px, 365 words → 42.

## 3. Structure

Three tabs, down from four. Fleet folds into Now, because both answered "is the system
healthy?" and the fleet is better as a compact grid than a screen.

```
Now       do I need to act?      attention slot · fleet grid · deltas
Money     is it making money?    equity hero · capacity · contribution
Research  is the funnel working? funnel · IS-vs-OOS · ranked runs
```

Strategy detail remains a drill-down reached from any of the three.

## 4. The screens

### 4.1 Now

**Attention slot (the one variable band).** Either a single all-clear mark with a
one-line basis, or a count plus one compact card per exception (strategy name and the
cause).

> **Amended 2026-08-27 (slice 1).** This originally specified a *severity bar* and an
> *age* per card. `/api/triage` can supply neither: `web/backend/triage.py` hardcodes
> `since: None` for `strategy`-kind items, and their severity is the constant
> `SEVERITY['strategy']` (3) — there is no gradient to encode. Designing those two
> encodings would have produced a card that renders uniform and blank against real
> data, which slice 1's fixture work caught before slice 2 built it. Ranking between
> *kinds* (loop_down, global_halt, capital_stranded, queue_wedged, strategy) is real
> and still available; ranking *within* the strategy kind is not. If an age is wanted
> later, it is a backend change first.

**The all-clear must be bound to the verdict `fleet health` already computes** — the
fail-closed one that exits non-zero on `stale`/`drift`/`idle`/`halted`, global halt, or
a corrupt fleet row. The UI must NOT compute its own notion of "fine". A UI-side
definition will eventually show green while a loop is dead, which is strictly worse
than today's noisy screen because the operator will have learned to trust it.

**Fleet grid.** One cell per *fielded* strategy (live / paper / forward_tested /
dormant — research stages belong to §4.3). Position groups by stage; colour encodes the
health verdict: trading, stale, halted, resting. Verified to 65 cells without layout
change.

> **Amended 2026-08-27 (slice 1).** This paragraph originally called that set
> *operational*. It is not: `algua/execution/fleet_health.py:66` defines
> `OPERATIONAL_STAGES = {live, paper, forward_tested}` and explicitly EXCLUDES dormant,
> while `fleet_status()` emits a row for every registry strategy at any stage. The
> DISPLAY choice — dormant cells rendered as resting, research stages omitted from this
> grid — stands and is unchanged; only the word was wrong, and it collided with a
> codebase constant that means something narrower. Slice 2 must filter by stage
> explicitly rather than trusting the term, and must sort worst-severity-first, because
> that is the order the API returns (`fleet_health.py:263`).

**Deltas, not events.** What moved since the last look: equity change with a
zero-anchored sparkline, and counts for promoted / benched / gates run. No event list,
no timeline. If something needed attention it is already in the attention slot.

### 4.2 Money

- **Equity hero** — attributed equity curve with the drawdown envelope shaded. The one
  mark on this screen worth real vertical space.
- **Capacity** — `8/64` drawn as 64 slots with 8 filled, reusing the fleet-grid cell
  idiom rather than inventing a second one. Headroom becomes visible instead of being
  a sentence ("56 more tenants fit").
- **Contribution** — the book-slices table becomes horizontal bars sorted by P&L
  contribution: name, bar, value. Long names ellipsis; nothing forces page width.

### 4.3 Research

- **Funnel** — stage counts as an actual funnel with conversion between stages. This is
  what the platform is *for*, and it currently renders as a single chip reading
  `IDEA (6)`.
- **Unchanged from slice 3:** IS-vs-OOS scatter, ranked run list with degradation
  sparklines, and on strategy detail the gate bullet card, funnel trial distribution
  and return overlay. These are already visual and already reviewed. The work here is
  trimming the prose around them, not touching them.

## 5. Executable invariants

This repo enforces structure rather than trusting discipline — import-linter contracts,
the module-size ratchet, the data-wall AST scanner. Two invariants follow that pattern,
and exist because both failures already happened once.

### 5.1 No horizontal overflow

Every route, rendered at 390px against the fixture, asserts
`documentElement.scrollWidth <= window.innerWidth`. This turns the Money bug into a
test that fails if the class of defect reappears anywhere.

### 5.2 A word budget per screen

Every screen, rendered at 390px against the fixture, asserts a word ceiling and a
minimum mark count. Verbosity does not return in one commit; it returns one helpful
sentence at a time, which is how 365 words accumulated. A budget is the only mechanism
that resists that, and it makes "less text, more visualization" a property the CI can
hold rather than an intention.

Ceilings are set from the delivered screens with modest headroom, and raising one is a
deliberate, visible act in review — the same contract as the module-size ratchet.

## 6. The mock clone

**Form:** demo mode of the real app, built static. The same React components with a
fixture data source swapped in at build time — no backend, no database, no tailnet.
Highest fidelity (every pixel is the real code), most portable, and it doubles as the
harness the §5 invariants render against.

**Fixture:** one rich steady state — strategies spread across stages, populated funnel,
real history depth, a couple of live exceptions. Shaped exactly like the API envelopes
so the seam is a swap, not an adapter with logic in it.

**Two guards:**
- the production build must never bundle fixtures;
- the demo build must never reach the network.

**Empty states are still designed deliberately**, even though only one fixture is
built. Thin data reading as a wall of text is the exact failure that prompted this
work, and the run-ledger views will render empty until the operator loop restarts.

## 7. Decisions

**Electric's double duty.** `--series-focus` and `--electric` are the same value, so
data and chrome currently compete: slice 3 shipped an Electric equity mark next to an
Electric active-tab. Electric belongs to **data**. The active tab drops it for bright
ink plus weight. (Touches slice-3 code.)

**Counting.** "57 operating" beside "65 strategies" is a discrepancy waiting to
mislead. Trading and resting are counted separately in one line, matching the grid's
colours.

**Amber.** Amber means advisory-failure elsewhere in this app. "Benched" is a normal
outcome and renders in neutral ink.

## 8. Non-goals

- No change to CLI or API output. The JSON seam is untouched — that is the audience
  this redesign removes from the UI, not one it degrades.
- No backfill (unchanged from the slice-3 spec, Q8).
- No switchable demo scenarios — one fixture.
- No change to the slice-3 chart internals.
- No new data collection. Every value shown already exists.

## 9. Risks

- **The all-clear is a trust surface.** §4.1 binds it to `fleet health`; any drift
  between the two re-opens the failure. This is the single highest-stakes item here.
- **The word budget can be gamed** by moving words into images or tooltips. It is a
  ratchet, not a proof; review still has to care.
- **Deleting the activity feed is irreversible in practice.** If the operator later
  wants "what happened", the answer is the CLI. That is the accepted consequence of
  the §1 audience decision, not an oversight.
