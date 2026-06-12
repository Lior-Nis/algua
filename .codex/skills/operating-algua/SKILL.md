---
name: operating-algua
description: Orientation for operating the algua algotrading platform — the golden rules, the CLI/JSON seam, the strategy lifecycle, and where to look. Read this first when running or operating algua.
---

# Operating algua

algua is an **agent-first** algotrading research platform. You operate it the same way the human
operator does: through one CLI, `uv run algua ...`, where **every data command emits JSON on
stdout**. You drive research; the system enforces the safety boundary.

## Golden rules (non-negotiable)

1. **Drive everything through `uv run algua ...`.** Never import or call algua modules directly to
   bypass the CLI. The CLI is the contract.
2. **You may operate the lifecycle autonomously up to and including `forward_tested`.** You may
   **never** put a strategy `live`. The `forward_tested → live` transition requires a verified
   human approval, a human actor, AND a fresh forward-test certificate; the system enforces this
   and you must not attempt to route around it.
3. **Do not weaken safety or integrity code.** Never edit `algua/registry/store.py`,
   `algua/contracts/lifecycle.py`, `algua/backtest/engine.py`, or `algua/research/gates.py` to make
   something pass. These are human-owned (CODEOWNERS); changing them is out of scope for a run.
4. **Keep the quality gate green** after any code you add:
   `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`.

## The lifecycle

```
idea → backtested → candidate → paper → forward_tested → live → retired
```
(plus allowed back-steps and `→ retired`). As an operator you take a strategy from `idea` to
`forward_tested`. Stage lives in the SQLite registry — `uv run algua registry show <name>` reports it.

## Command surface (the ones you use most)

- `uv run algua doctor` — environment readiness (non-zero exit = a failed check).
- `uv run algua registry list [--stage S]` / `registry show <name>` — inspect strategies + history.
- `uv run algua backtest run <name> --demo --register` — backtest + register + advance to `backtested`.
- `uv run algua backtest walk-forward <name> --demo` — K windows + stability (out-of-sample evidence); the holdout is withheld until `research promote`.
- `uv run algua backtest sweep <name> --demo --param KEY=v1,v2` — bounded parameter grid, ranked.
- `uv run algua research promote <name> --demo` — gate `backtested → candidate`; promotes only on pass.
- `uv run algua paper promote <name>` — gate `paper → forward_tested`; requires ≥63 broker-clocked daily return observations, ≥90% coverage, realized Sharpe ≥ max(0.5×holdout, 0.3), vol/DD bounds, clean integrity + hygiene, evidence ≤5 sessions stale. Relaxation flags are human-only.
- `uv run algua data inspect --summary` — what data snapshots exist.

`--demo` uses the synthetic data provider (offline, deterministic). Swap in real bars with
`--snapshot <id>` from `data ingest-bars` when you want real data.

## Where to look

- **The research loop you run:** the `run-the-research-loop` skill.
- **Writing a strategy:** the `author-a-strategy` skill.
- **Reading results / deciding promote-or-discard:** the `interpret-results` skill.
- **The canonical command walkthrough:** `docs/agent/research-lifecycle.md`.
- **Design intent / source of truth:** `docs/superpowers/specs/2026-05-29-algua-platform-architecture-design.md`.
- **The data contract crossing data↔research:** `docs/contracts/bar-schema.md`.

## How you're run

You typically run autonomously inside an isolated git worktree on a `research-run/<stamp>` branch.
Commit your authored files and a `run-report.md` to that branch; a human reviews the branch before
anything merges. Operate freely within the worktree — it's contained and reviewable.
