---
name: run-the-research-loop
description: The autonomous research loop playbook — work the ideas claimed from the pool for this run, author a strategy per idea, backtest/walk-forward/sweep it, gate it with research promote, and record a run report whose v2 trailer reports each idea's outcome. Use when running an autonomous research session.
---

# Running the research loop

You operate algua autonomously to take strategy ideas to **candidate**, then hand back a branch
for human review. Read `operating-algua` first for the golden rules. Delegate authoring to the
`author` subagent and results-judgement to the `interpret` subagent.

## The thesis

**Scale is the moat** (PRD §4). Algua tests as many uncorrelated hypotheses as it can — cheaply,
fast and honestly — and lets harsh forward selection keep the few that survive. No single market
belief is privileged. The beliefs live one level down, as the **ideation categories** the idea
engine rotates through (`.codex/categories.txt`): momentum, mean-reversion, seasonality,
volatility structure, value/quality proxies, liquidity and microstructure, event-driven, and
institutional / whale flow (the original 2026-05 thesis, now one category among several).

You do not pick the belief. The driver claims ideas from the pool before your run and names them
in your goal; your job is to test them honestly and report what happened.

## The loop (repeat for each claimed idea, then stop)

For each claimed idea:

1. **Work the claimed ideas.** Your goal names them (`idea_id`, title, hypothesis, category,
   market, horizon, `falsification`, inspirations). Do not invent a hypothesis; if an idea is
   untestable, report `outcome: run_error` with the reason. Read
   `kb/principles/research-methodology.md` before authoring (leakage vectors, search-breadth
   honesty, designing for generalization), plus `kb/strategies/_index.md` and
   `kb/strategies/_families.md` for what the relevant family already learned. Pick a unique
   strategy name; skip names already in `uv run algua registry list`.
2. **Author.** Scaffold with `uv run algua strategy new <name> --family <slug> --derived-from
   <parent>` (creates the module *and* the KB doc + family hub). Delegate to the `author`
   subagent (it follows `author-a-strategy`) to write `algua/strategies/<family>/<name>.py`,
   then fill in the doc's `## Hypothesis` and `## Derivation` prose. Confirm it loads:
   `uv run algua backtest run <name> --demo`.
3. **Backtest + register.** `uv run algua backtest run <name> --demo --register` (advances `idea →
   backtested`).
4. **Out-of-sample evidence.** `uv run algua backtest walk-forward <name> --demo` (K windows +
   stability; the holdout is withheld until `research promote`). Optionally `uv run algua backtest sweep <name> --demo --param KEY=v1,v2,...`
   to scan parameters — but remember every combo you search raises the bar the holdout must clear
   (see `interpret-results` on search breadth).
5. **Interpret.** Delegate the results JSON to the `interpret` subagent for a promote/discard
   recommendation with reasoning, and assess the idea's own `falsification` statement against the
   walk-forward evidence -> `falsification_assessment` (`refuted` | `survived` | `untested`).
6. **Gate.** Run `uv run algua research promote <name> --demo` (record the combos you searched with
   `--n-combos K`). The gate advances `backtested → candidate` **only on pass**; on a fail it
   reports why and leaves the stage unchanged. Trust the gate — do not lower thresholds to force a pass.
7. **Record.** Sync the synced fact blocks: `uv run algua strategy doc <name>`. Then write
   the doc's `## Verdict & next` (what was learned + the next idea as a `[[dangling-link]]`),
   set `hypothesis_status`, and update the family doc's `## State of exploration` and
   `status`. Finally append the hypothesis, params, key metrics, the gate decision, and your
   candidate/discard rationale to your run report (see "Finishing a run" below). For a strategy
   worth a deeper write-up, the `report-experiments` skill turns its tracked sweep/walk-forward
   runs into a plotted, provenance-stamped report in the vault.

## Stopping

Work exactly the **claimed ideas** in your goal (at most N), then stop. If you are running low on
time, stop early — but always finish by writing the report, and give every claimed idea you never
got to its own trailer entry with `outcome: run_error`. An idea missing from the trailer is
recorded as `run_error` / `missing_from_trailer` anyway; saying why yourself is strictly better.

## Boundaries

- Operate **only** through `uv run algua ...`. Never go past `candidate` in an autonomous research
  session — do not attempt `registry transition --to paper/forward_tested/live` or `registry
  approve`; advancing to paper or beyond is outside the research loop scope.
- Never edit the human-owned safety/integrity files (see `operating-algua`).
- Author only **new** files under `algua/strategies/<family>/` via `strategy new --family <slug>`.

## Finishing a run

1. Ensure every authored strategy file is committed on the current `research-run/<stamp>` branch.
2. Write your report to `kb/research-runs/<stamp>.md` (create the directory if it doesn't exist):
   one section per hypothesis (name, params, backtest + walk-forward + gate results, decision +
   why), then a summary of what you promoted to candidate.
3. The report MUST end with a machine-readable trailer — one fenced ```json block with **exactly
   one `hypotheses[]` entry per claimed idea** (trailer **v2**):
   - `idea_id` — the claimed idea's id. Required; an entry without it teaches the pool nothing.
   - `title` — <= 120 chars, plain ASCII.
   - `outcome` — what actually happened, written straight into the authoritative idea ledger:
     `integrity_fail` (preflight/integrity refused it) | `holdout_negative` (holdout Sharpe <= 0) |
     `walkforward_refuted` (out-of-sample windows refute it) | `sweep_unstable` (only isolated
     combos work) | `candidate_preview_pass` (the preview gate passed) | `run_error` (you could not
     test it). Be honest — this is the record the funnel learns from.
   - `reason` — <= 300 chars, why that outcome.
   - `falsification_assessment` — `refuted` | `survived` | `untested`, judging the idea's own
     falsification statement against your walk-forward evidence.
   - `verdict` — `"discarded"|"candidate-preview-pass"|"error"` (kept for the digest readers).
   - `merge_back` — ONLY when `verdict` is `"candidate-preview-pass"`: `idea_id`, `strategy`,
     `universe`, `start`, `end`, `eval_context`, naming the exact strategy module you authored and
     its promote window.

   The launcher parses this into the durable run digest, writes one `record-outcome` per claimed
   idea, and for every valid `merge_back` enqueues the REAL, authoritative `paper merge-back` for
   the automated drainer to run — the merge-back you'd otherwise hand a human is now automatic; do
   not run it yourself.
4. Commit the report (the launcher does this for you). Review the branch afterward with
   `git diff main...<branch>` if you want to double-check what shipped.
