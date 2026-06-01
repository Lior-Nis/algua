---
name: run-the-research-loop
description: The autonomous research loop playbook — ideate a hypothesis, author a strategy, backtest/walk-forward/sweep it, gate it with research promote, shortlist or discard, and record a run report. Use when running an autonomous research session.
---

# Running the research loop

You operate algua autonomously to take strategy ideas to a **shortlist**, then hand back a branch
for human review. Read `operating-algua` first for the golden rules. Delegate authoring to the
`author` subagent and results-judgement to the `interpret` subagent.

## The thesis

algua's research thesis is **riding institutional / "whale" momentum** — strategies that follow
the moves big players make rather than fighting them. The concrete signal isn't fixed; that's what
research explores. Ground your hypotheses in this thesis unless told otherwise.

## The loop (repeat for N hypotheses, then stop)

For each hypothesis:

1. **Ideate.** Form one concrete, testable hypothesis (e.g. "longer-lookback cross-sectional
   momentum with a tighter top-k concentrates into stronger trends"). Pick a unique strategy name;
   skip names already in `uv run algua registry list`.
2. **Author.** Delegate to the `author` subagent (it follows `author-a-strategy`) to write
   `algua/strategies/examples/<name>.py`. Confirm it loads: `uv run algua backtest run <name> --demo`.
3. **Backtest + register.** `uv run algua backtest run <name> --demo --register` (advances `idea →
   backtested`).
4. **Out-of-sample evidence.** `uv run algua backtest walk-forward <name> --demo` (holdout + K
   windows + stability). Optionally `uv run algua backtest sweep <name> --demo --param KEY=v1,v2,...`
   to scan parameters — but remember every combo you search raises the bar the holdout must clear
   (see `interpret-results` on search breadth).
5. **Interpret.** Delegate the results JSON to the `interpret` subagent for a promote/discard
   recommendation with reasoning.
6. **Gate.** Run `uv run algua research promote <name> --demo` (record the combos you searched with
   `--n-combos K`). The gate advances `backtested → shortlisted` **only on pass**; on a fail it
   reports why and leaves the stage unchanged. Trust the gate — do not lower thresholds to force a pass.
7. **Record.** Append the hypothesis, params, key metrics, the gate decision, and your
   shortlist/discard rationale to `run-report.md`.

## Stopping

Evaluate exactly **N hypotheses** (N is given in your goal), then stop. If you are running low on
time, stop early — but always finish by committing your work and writing the report.

## Boundaries

- Operate **only** through `uv run algua ...`. Never go past `shortlisted` — do not attempt
  `registry transition --to paper/live` or `registry approve`; that's the human's call.
- Never edit the human-owned safety/integrity files (see `operating-algua`).
- Author only **new** files under `algua/strategies/examples/`.

## Finishing a run

1. Ensure every authored strategy file is committed on the current `research-run/<stamp>` branch.
2. Write `run-report.md` at the repo root: one section per hypothesis (name, params, backtest +
   walk-forward + gate results, decision + why), then a summary of what you shortlisted.
3. Commit the report. The human reviews the branch (`git diff main...<branch>`) and merges what's worth keeping.
