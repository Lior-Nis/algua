# Algua — Agent Operating Guide

Algua is an agent-first algotrading platform. You (an agent) and the human operator
drive the system through the **same** CLI. Every data command emits JSON on stdout.

## Orientation — where to look
- **Architecture & roadmap (source of truth):** `docs/superpowers/specs/2026-05-29-algua-platform-architecture-design.md`
- **Why the rules exist (detail):** `docs/agent/operating.md`
- **How this foundation was built (task plan):** `docs/superpowers/plans/2026-05-29-foundation-command-surface.md`
- **Reviewing/fixing the system?** Read `AGENTS.md` first (review mandate + invariants + deferred scope).
- **Data contract (frozen):** `docs/contracts/bar-schema.md` — the shape of bars crossing the
  data↔research seam (`DataProvider.get_bars`).
- **Current state:** Sub-project 1 (foundation) merged. Sub-project 2 (data layer) in progress.
  Work is split across two agents in parallel — **Codex: data lane** (`algua/data/*`); **Claude:
  research lane** (`algua/strategies|features|backtest|tracking/*`), meeting at the bar-schema
  contract. The 6-sub-project roadmap is in the spec above.

## Golden rules
- Drive the system through `uv run algua ...`. Never reach into modules to bypass the CLI.
- You may operate the lifecycle autonomously **up to and including paper**.
- You may **never** put a strategy live. The `paper -> live` transition requires a
  verified human approval and a human actor; the system enforces this.
- Keep `algua/contracts` and `algua/features` pure (no I/O, no cross-module imports
  beyond contracts). Import-linter enforces boundaries; run `uv run lint-imports`.

## Command surface
- `uv run algua version` — version JSON.
- `uv run algua doctor` — environment readiness; non-zero exit means a failed check.
- `uv run algua registry add <name>` — register a strategy (stage `idea`).
- `uv run algua registry list [--stage S]` — list strategies.
- `uv run algua registry show <name>` — strategy + transition history.
- `uv run algua registry transition <name> --to S --actor agent --reason "..."` — advance stage.
- `uv run algua registry approve <name> --code-hash H --config-hash H --by NAME` — human-only.

## Lifecycle stages
`idea -> backtested -> shortlisted -> paper -> live -> retired`
(plus allowed back-steps and `-> retired`). See `algua/contracts/lifecycle.py`.

## Quality gates before committing
`uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
