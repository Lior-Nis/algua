# Algua monitor — dashboard redesign

**Status:** spec, locked via a `grill-me` session (2026-08-15).
**Supersedes the screen inventory in:** `2026-08-09-algua-monitor-pwa-design.md` (that spec's
architecture — tailnet-only, read-only, CLI-via-subprocess seam — is unchanged and still binding).

---

## 1. Why

The monitor is code-complete and deployed, and it is *accurate*. It is not *useful enough*, and
there is direct evidence for that claim rather than an opinion.

On 2026-08-15 the dashboard was green while both of the following were true:

1. **`algua-research` had been failing every hour and would keep failing for five days** — the
   Codex account hit its usage limit. The autonomous funnel, the top of the whole system, was
   dead. The dashboard had no concept of it.
2. **`liquidity_stable_quality_momentum` sat at stage `paper` with no book allocation.** It went
   `paper -> dormant` (which atomically releases the slice) and back via `dormant -> paper`; the
   stage was restored, the capital was not. The paper operator skips an unallocated strategy, so
   it has never ticked and never will. The dashboard showed it as `idle` — factually correct, and
   useless, because the cause was invisible.

Both are things only the human can fix. Neither had anywhere to appear.

The root cause is that the current screens are organised around **the CLI commands that produce
them** (`fleet health` -> Home, `registry list` -> Funnel, `audit log` -> Activity, `research idea
list` -> Ideas). That is an inventory, not an instrument. The system being monitored is not a set
of strategies; it is a **machine** — loops, a queue, a book, and a funnel — of which the strategies
are one output.

## 2. Who this is for (ICP)

Derived from `CLAUDE.md`, the memory index, the systemd unit topology, and the live fleet — not
assumed.

| | |
|---|---|
| **Role** | Solo operator and sole author of an autonomous fund-in-a-box. Capable engineer; prefers mature OSS to custom plumbing; treats the AI as a design partner and asks to be told what he is missing. |
| **Context** | The loop runs unattended: `algua-research` hourly, `algua-mergeback-drain` half-hourly, `algua-paper` daily 00:30. He is asleep or elsewhere while it runs. |
| **State today** | 10 strategies, all pre-live. Money is currently a *research* problem, not an engineering one. |
| **Scarce resource** | **Attention, not information.** He opens this on a phone to answer one question and close it. He is not browsing. |
| **His exclusive authority** | Go-live (signed ceremony), capital allocation, and relaxation flags. The system is deliberately built so an agent cannot do these. Everything else runs without him. |
| **Anti-goal** | Cruft, additive compat layers, dashboards that show everything and mean nothing. |

The design consequence: **every pixel competes with him closing the tab.** A screen that is
usually all-clear must cost nothing to read when it is all-clear, and must be unmissable when it
is not.

## 3. Locked decisions

Each was resolved one at a time, with a recommendation, in the grilling session.

| # | Decision | Resolution |
|---|---|---|
| Q1 | Primary job | **All three jobs coexist, tabbed**: "does the machine need me", "how is my money doing", "what is the funnel producing". Not narrowed to one. |
| Q2 | Landing screen | **Cross-cutting triage that routes.** Home spans all three jobs, ranks anything needing him, and deep-links into the owning tab. Collapses to one line plus three headline numbers when clear. |
| Q3 | Machine-health data | **New read-only `algua ops status`.** Algua already records what is needed; the monitor keeps consuming only the CLI. |
| Q4 | Capital data | **New read-only `algua book status`.** Serves both the triage alert and the money tab from one read. No broker call. |
| Q5 | Visual scope | **Explore 3–4 directions, brand tokens fixed.** Electric-as-rare-signal, true-black Obsidian and Inter are hard guardrails; typographic *scale*, density and status encoding are explored, then narrowed and polished. |
| Q6 | Tabs | **Now / Fleet / Money / Research.** Activity folds into Now as a "since you last looked" feed; Ideas folds into Research. |

### A conflict recorded rather than silently resolved

The brand kit (`docs/brand/README.md`, shipped 2026-08-14) mandates **Inter** as primary sans. The
design doctrine adopted here (Impeccable) treats Inter-everywhere as the canonical AI-slop tell.
These genuinely disagree. **The brand kit wins** — it is a deliberate identity decision, and
"de-slopping" a brand out from under its owner is not a design improvement. Inter stays as the
face; the exploration is confined to scale, weight and hierarchy.

## 4. New CLI surfaces

Both are **pure reads**: no broker call, no writes, no locks. Both emit JSON on stdout under the
existing `ok()`/`@json_errors` envelope, so `web/backend/algua_cli.py` consumes them unchanged.
Both are mounted at the composition root in `algua/cli/main.py`, never by a sibling import.

Crucially, both read **artifacts algua already writes**. Neither introduces a new data source, and
neither touches systemd — that was considered and rejected for coupling the monitor to the host's
init system.

### 4.1 `algua ops status`

Machine liveness. Reads:

- `data/research-runs.jsonl` — per-run digest. Already carries `stamp`, `branch`, `outcome`,
  `exit_code`, `timed_out`, `wall_s`, `n_strategy_files`, `preview_gate`, `trailer_parse_error`
  and — decisively — **`rate_limited`**. The five-day outage was a first-class recorded field the
  whole time.
- `data/operator_sessions.json` — the paper operator's last `command`, `rc`, `session`, `recorded_at`.
- `data/mergeback-queue.json` — queue depth and per-item `attempts`.

```json
{
  "ok": false,
  "loops": {
    "research":  {"health": "rate_limited", "last_ok_at": "...", "runs_since_ok": 34,
                  "consecutive_failures": 34, "detail": "codex usage limit"},
    "paper":     {"health": "ok", "last_rc": 0, "session": "2026-08-14", "recorded_at": "..."},
    "mergeback": {"health": "ok", "queue_depth": 3, "max_attempts": 1, "oldest_enqueued_at": "..."}
  }
}
```

**Health vocabulary** mirrors the existing fail-closed convention: a loop whose artifact is
missing, unparseable, or older than its expected cadence is never silently `ok`. `ok` requires
positive, fresh, parseable evidence — the same rule `strategy_health` already applies to ticks.

### 4.2 `algua book status`

Capital. Reads the `allocations` table plus persisted tick-snapshot equity (never the broker).

```json
{
  "ok": false,
  "capacity": 64,
  "allocated": 7,
  "count_headroom": 57,
  "sum_allocations": 10999.94,
  "unallocated_operational": [
    {"strategy": "liquidity_stable_quality_momentum", "stage": "paper",
     "since": "2026-08-14T12:12:43.110561+00:00", "ever_ticked": false}
  ],
  "slices": [{"strategy": "cadenced_tail_risk_relative_strength", "stage": "paper",
              "capital": 1571.42, "last_equity": 1571.42, "effective_ts": "...",
              "actor": "agent", "equity_error": null}]
}
```

**Correction to the design sketch: only COUNT headroom is reported, not capital headroom.** The
grilling sketch showed a capital `headroom` field; building it revealed that it requires the
account equity, and the only path to that is `paper account`, which calls the broker — the one
thing this view must not do. Deriving it from summed tick-snapshot equity would be inventing a
number that is not the account's. It is omitted rather than approximated.

`unallocated_operational` is the generalisation of the bug that motivated this: **any** strategy in
an operational stage (`live`/`paper`/`forward_tested`) with no active allocation. It is a standing
condition, not a one-off.

## 5. Screens

### Now (landing)

Answers "does anything need me", spanning all three jobs.

1. **NEEDS YOU** — ranked, worst first. Each row states the condition, its age, and the single
   fact that makes it actionable, and deep-links to the owning tab. Sources: `ops status` (dead or
   rate-limited loop, wedged queue), `book status` (operational-but-unallocated, no headroom),
   `fleet health` (alerting rows). **Empty state is a single line**, not an empty panel.
2. **Since you last looked** — the audit feed, absorbed from the Activity tab, bounded to what
   changed since the last visit. This is what "what did the machine do overnight" actually means.
3. **Three headline numbers** — one per job: fleet ok/total, book allocated/capacity, funnel
   output over the window.

### Fleet

Today's per-strategy rollup, worst-first, retained nearly as-is — it is already correct after
PR#567. Gains the allocation slice per row, so `idle` and `unallocated` are never again confused.

### Money

New. Book-level capital: capacity, Σ allocations, headroom, per-strategy slice against its current
equity, and portfolio equity over time. Honest empty state while nothing is live — it must not
imply performance that does not exist.

### Research

Funnel throughput plus the absorbed Ideas pool: stage counts, what the last research runs produced,
gate pass/fail with the binding-vs-advisory distinction from PR#567, and the idea pool by status.

## 6. Visual system

**Fixed (brand kit, not negotiable here):** Obsidian `#000000` ground and panel; hairline Mist
borders doing the separation; Electric `#3982FF` as a *rare* signal marking the one active thing;
semantic green/red/amber/violet reserved for status so blue never has to mean "problem"; Inter
sans, IBM Plex Mono for numerics.

**In play:** typographic scale and hierarchy, density, how status is encoded (badge vs. rule vs.
position), panel treatment, and the empty-state grammar.

**Method** (from the channel's technique, adapted): do not one-shot. Build 3–4 genuinely different
directions of the *Now* screen, view them side by side, choose one, generate a small number of
variations of that direction, then narrow and finish with Impeccable's audit/polish/typeset over
the result. The prompt payload for each direction carries aesthetic, reference, intent and
guardrails explicitly.

**Guardrails, stated as nevers:** never a purple/blue gradient; never Electric used decoratively;
never a colour as the *only* carrier of a status (accessibility); never an almost-black surface;
never a chart that renders a misleading zero baseline for an empty series.

## 7. Slices

| # | Slice | Gate | Status |
|---|---|---|---|
| 1 | `algua ops status` + `algua book status` — domain readers, CLI commands, tests | Root pytest, ruff, mypy, lint-imports | **done** |
| 2 | Backend endpoints `/api/ops`, `/api/book`; triage assembly is a pure, unit-tested function | Web backend pytest | **done** |
| 3 | Tab restructure to Now / Fleet / Money / Research; Activity and Ideas absorbed | Frontend check | **done** |
| 4 | Visual exploration → pick → narrow → Impeccable polish | Frontend check + build | **done** |

Slice 1 is load-bearing: slices 2–4 all consume it.

### Slice 4 outcome

Four directions of the *Now* screen were built side by side at phone width against the real
2026-08-15 state and reviewed as rendered pages, per the method: **A Ledger** (hairline rules, 2px
severity stripe), **B Stack** (card per condition), **C Numeric** (count as hero), **D Band**
(verdict sentence first). **A was chosen** — it was already what slice 3 shipped, so slice 4 became
refinement rather than replacement.

The mechanical slop detector returned zero findings before and after. Both real defects came from
MEASURING rather than looking:

1. `--text-fade` (brand Slate `#5d6675`) on Obsidian is **3.62:1** — under the 4.5:1 AA floor for
   the small text it carried in 13 rules. The brand kit itself assigns Slate to *light* surfaces
   and Fog to dark, so it was wrong on both counts. Repointed to `#727c8b` (4.97:1), the quietest
   step toward Fog that clears AA, so three tiers survive instead of collapsing into two. Brand
   Slate is retained as `--slate` for non-text, where the 3:1 graphical floor applies and it passes.
2. There were **no focus styles at all** — every interactive element fell back to a UA outline that
   is near-invisible on pure black. Added a `:focus-visible` ring in Electric, consistent with the
   brand's rule that Electric marks the one active thing on screen.

## 8. Out of scope

- **Any write action.** The monitor stays strictly read-only. It will show that a strategy needs
  allocating; it will never allocate. Capital and go-live remain terminal-only, human-only.
- **systemd/journalctl integration** — considered and rejected (§4).
- **Amending the brand kit** — recorded as a conflict, resolved in the brand's favour (§3).
- **Scroll-world / hero imagery / generated assets** — evaluated from the channel and rejected:
  it produces scroll-driven 3D camera-flight marketing pages, which is the opposite of a
  phone-glanceable operations instrument.
