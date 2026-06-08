# Algua — Agent Operating Guide

Algua is an agent-first algotrading platform. You (an agent) and the human operator
drive the system through the **same** CLI. Every data command emits JSON on stdout.

## Orientation — where to look
- **Architecture & roadmap (source of truth):** `docs/superpowers/specs/2026-05-29-algua-platform-architecture-design.md`
- **Why the rules exist (detail):** `docs/agent/operating.md`
- **How this foundation was built (task plan):** `docs/superpowers/plans/2026-05-29-foundation-command-surface.md`
- **Reviewing/fixing the system?** Read `AGENTS.md` first (review mandate + invariants + deferred scope).
- **Data contract:** `docs/contracts/bar-schema.md` — the shape of bars crossing the
  data↔research seam.
- **Current state:** Sub-project 1 (foundation) merged. Sub-project 2 (data layer) is implemented:
  provider-backed bars, parquet snapshots, provenance manifest, and universe snapshots. The
  6-sub-project roadmap is in the spec above.

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
- `uv run algua registry transition <name> --to live --actor human` — step 1 of go-live: prints a
  challenge. Sign it, then re-run with `--signature <file>.sig` to complete the transition.
- `uv run algua research promote <name> --universe NAME --start D --end D` — gate
  `backtested -> shortlisted` on the walk-forward holdout + stability and promote on pass. For an
  agent: `--universe` is required (PIT — non-PIT fails closed); search breadth must be MEASURED via
  `backtest sweep` (declaring it with `--n-combos` is human-only); the holdout-Sharpe bar is
  DEFLATED by funnel-wide search breadth (multiple-testing defense); a minimum holdout-observations
  floor (63) applies (underpowered holdouts fail closed). A passing run is the ONLY way an agent
  reaches `shortlisted` — there is no raw `registry transition --to shortlisted` shortcut for an
  agent (`--allow-non-pit`, `--allow-holdout-reuse`, `--n-combos`, and the raw shortlist transition
  are all human-only).
- `uv run algua data ingest ... --from-file PATH` — register a local immutable snapshot.
- `uv run algua data ingest-bars --provider yfinance --symbols AAPL --start D --end D` — fetch
  historical bars into a parquet snapshot.
- `uv run algua data ingest-universe NAME --symbols AAPL,MSFT --effective-date D` — record
  point-in-time universe membership.
- `uv run algua data import-bars --vendor firstrate --raw-dir DIR --adjusted-dir DIR --as-of TS` —
  bulk-import local vendor files (FirstRateData: per-symbol unadjusted + adjusted), normalized to
  the bar-schema as one consolidated snapshot. Streamed (bounded RAM); `adj_close` from the adjusted
  file (no corporate-action math yet).
- `uv run algua data inspect [--summary|--dataset NAME|--snapshot-id ID]` — inspect data snapshots.

## Lifecycle stages
`idea -> backtested -> shortlisted -> paper -> live -> retired`
(plus allowed back-steps and `-> retired`). See `algua/contracts/lifecycle.py`.
For an agent, the `backtested -> shortlisted` edge is gated: it requires a fresh passing
`research promote` (an identity-matched, single-use gate token), not a raw `registry transition`.

## Quality gates before committing
`uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
