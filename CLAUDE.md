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
- You may operate the lifecycle autonomously **up to and including `forward_tested`**.
- You may **never** put a strategy live. The `forward_tested -> live` transition requires a
  verified human approval AND a fresh forward-test certificate; the system enforces this.
- Keep `algua/contracts` and `algua/features` pure (no I/O, no cross-module imports
  beyond contracts). Import-linter enforces boundaries; run `uv run lint-imports`.

## Command surface
- `uv run algua version` — version JSON.
- `uv run algua doctor` — environment readiness; non-zero exit means a failed check.
- `uv run algua registry add <name>` — register a strategy (stage `idea`).
- `uv run algua registry list [--stage S]` — list strategies.
- `uv run algua registry show <name>` — strategy + transition history.
- `uv run algua registry transition <name> --to S --actor agent --reason "..."` — advance stage.
- `uv run algua paper promote <name>` — gate `paper -> forward_tested` on ≥63 broker-clocked
  daily return observations (≥90% session coverage), realized Sharpe ≥ max(0.5×holdout, 0.3),
  vol/drawdown bounds, clean integrity + account hygiene, evidence ≤5 sessions stale; relaxation
  flags (`--degradation-factor`, `--sharpe-floor`, `--min-observations`, `--min-coverage`,
  `--min-vol`, `--max-drawdown`, `--max-staleness`) are human-only. A passing run is the ONLY
  agent path to `forward_tested`; re-running at `forward_tested` refreshes the live-wall
  certificate without changing the stage.
- `uv run algua registry transition <name> --to live --actor human` — step 1 of go-live: prints a
  challenge (includes forward certificate summary). Sign it, then re-run with
  `--signature <file>.sig` to complete the transition. Requires a fresh, matching forward
  certificate (newest evaluation for current identity+strategy must be a PASS, ≤10 sessions old,
  clean record + account hygiene since).
- `uv run algua research promote <name> --universe NAME --start D --end D` — gate
  `backtested -> candidate` on the walk-forward holdout + stability and promote on pass. For an
  agent: `--universe` is required (PIT — non-PIT fails closed); search breadth must be MEASURED via
  `backtest sweep` (declaring it with `--n-combos` is human-only); the holdout-Sharpe bar is
  DEFLATED by funnel-wide search breadth (multiple-testing defense); a minimum holdout-observations
  floor (63) applies (underpowered holdouts fail closed). A passing run is the ONLY way an agent
  reaches `candidate` — there is no raw `registry transition --to candidate` shortcut for an
  agent (`--allow-non-pit`, `--allow-holdout-reuse`, `--n-combos`, and the raw shortlist transition
  are all human-only).
- `uv run algua research dormant-sweep --start D --end D` — ADVISORY stability screen over the
  `dormant` pool: re-runs walk-forward per dormant strategy and reports which ones' window/stability
  metrics look healthy again, ranked. Read-only — never reads/burns the holdout, writes no ledger
  rows, and transitions nothing. A pass is a prioritization signal for re-auditioning (`registry
  transition --to paper`), NOT a gate and NOT a guarantee of re-promotion/forward-gate clearance.
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
`idea -> backtested -> candidate -> paper -> forward_tested -> live -> retired`
(plus allowed back-steps and `-> retired`). See `algua/contracts/lifecycle.py`.
`dormant` is a NON-terminal rest state for validated-but-resting strategies (entered only from
`live`/`paper`; recovers via `dormant -> paper`; gives up via `dormant -> retired`). Benching to
`dormant` needs a reason; `live -> dormant` requires the strategy be flat and atomically releases
its allocation. Unlike `retired` (the terminal tombstone), a `dormant` strategy can climb back out.
For an agent, BOTH the `backtested -> candidate` edge (research promote) AND the
`paper -> forward_tested` edge (paper promote) are token-gated: each requires a fresh passing
run that mints an identity-matched, single-use gate token, not a raw `registry transition`.

## Quality gates before committing
`uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
