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
- `uv run algua fleet status` — fleet-wide health rollup: every strategy's stage, kill-switch/
  global-halt, drawdown, last tick, and a fail-closed tick-staleness/health verdict in ONE read,
  worst-offender-first. Pure read (no broker call); always exits 0.
- `uv run algua fleet health` — loop-liveness / heartbeat GATE for an external watchdog (systemd
  `OnFailure=`, cron, k8s liveness): same rollup as `fleet status` but EXITS NON-ZERO iff an
  operator loop is dead/stalled/drifted/never-started (an operational strategy —
  live/paper/forward_tested — that is `stale`/`drift`/`idle`/`halted`), the account is globally
  halted, or a fleet row is corrupt. Cadence is COMPLETED NYSE sessions since the last tick (never
  wall-clock), so a weekend/holiday gap never false-alarms; a benched/retired strategy's ancient
  tick never wedges it red.
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
- `uv run algua paper merge-back --branch B --strategy S --universe U --start D --end D` — autonomous
  research-cycle merge-back (#485): one repo-global-locked cycle that gated-merges a candidate branch
  onto `main` (preview-merge → FULL quality gate on the staged tree → commit only on green), runs the
  metered strict-agent `research promote`, and on a PASS runs the FIFO paper intake to allocate a book
  slice; a proven promote FAILURE reverts the merge (`main` untouched). The branch's change set is
  gated by an allowlist/CODEOWNERS-denylist DIFF POLICY BEFORE any merge; recovery is driven by a
  durable per-strategy journal keyed on the branch-tip SHA, and promote attribution binds to a
  per-attempt `attempt_token` stamped on the gate row (not the ambient stage). Terminal `status` is
  `already_done` | `diff_policy_rejected` | `gate_failed` | `promote_failed` | `promoted_allocated` |
  `promoted_queued` (a not-promoted cycle is `ok`, not an error). Mutually exclusive with `paper
  trade-tick`/`run-all` BY OPERATOR DISCIPLINE (like #316); concurrent merge-backs are hard-serialized
  by a `merge_back.lock` flock. The push is a real remote compare-and-swap to `refs/heads/main`.
- **Authenticated `--actor human` (#329).** `--actor human` on `research promote` / `paper promote`
  is NOT a bare string anymore: run once with `--actor human` to print a single-use challenge, sign
  it (`ssh-keygen -Y sign -n algua-human-actor -f <key> <file>`), then re-run with `--actor-signature
  <file>.sig`. The signature binds the command + strategy + artifact identity + the exact run
  (every flag incl. the relaxation set) + a nonce, so it cannot be replayed onto another run/artifact
  /relaxation. A bare `--actor human` unlocks NOTHING. Enroll a human-actor key with `registry
  enroll-approver --namespace human-actor` (or `--namespace both` for a key that also does go-live);
  the anchor is `approvers/allowed_signers`, shared with the go-live gate under a distinct namespace.
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
  floor (63) applies (underpowered holdouts fail closed); a funnel-wide **LORD++ FDR alpha-wealth
  ledger** (#220, schema v26; **#324 cohort restarts**, schema v33) applies a SECOND tighten-only
  AND-check on measured runs — the per-strategy DSR p-value (p = 1 − dsr_confidence) must ≤ α_t
  (LORD++ level); FDR is an operating target (FDR_ALPHA=0.05, W0=0.025); declared/human breadth and
  missing dsr_confidence skip FDR. The stream is **partitioned into cohorts of FDR_COHORT_SIZE=64
  binding tests by arrival order**, each an independent LORD++ stream (fresh W0), so FDR is
  controlled **PER COHORT of 64 binding tests, NOT per lifetime**. This defeats anti-scaling: a
  single lifetime stream let a dry spell of failed attempts drive α_t → 0 (testing more garbage
  lowered everyone's bar); bounding the count floors the worst-case dry-spell level at γ_64·W0
  independent of throughput. Cumulative exposure over K completed cohorts is ≈ FDR_ALPHA·K, surfaced
  as audit-only `fdr_*` exposure fields. A passing run is the ONLY way an agent reaches `candidate`
  — there is no
  raw `registry transition --to candidate` shortcut for an agent (`--allow-non-pit`,
  `--allow-holdout-reuse`, `--n-combos`, and the raw shortlist transition are all human-only).
  **Family governance (#222):** at preflight, the strategy is empirically classified into a family
  via code-ancestry + factor-lineage + return-correlation clustering; MERGE verdict → assign into the
  incumbent family (inherits its breadth); PARENTAGE → new family but inherits the incumbent's
  accumulated breadth via a parent edge; NOVEL → **agent fail-closed** (new family requires
  `--new-family` + `--actor human`). Family-scoped lifetime breadth feeds the 3-way
  `effective_funnel_breadth(own, windowed_total, family_lifetime_effective)` tighten-only max.
- `uv run algua research dormant-sweep --start D --end D` — ADVISORY stability screen over the
  `dormant` pool: re-runs walk-forward per dormant strategy and reports which ones' window/stability
  metrics look healthy again, ranked. Read-only — never reads/burns the holdout, writes no ledger
  rows, and transitions nothing. A pass is a prioritization signal for re-auditioning (`registry
  transition --to paper`), NOT a gate and NOT a guarantee of re-promotion/forward-gate clearance.
- `uv run algua research family-audit` — ADVISORY cross-family gaming detector. Scans the family DAG
  for separate families that empirically behave as one thesis (deliberate-split breadth evasion #222's
  assignment-time clustering can't see), using return-correlation as the authoritative axis; ranks
  flagged clusters by family-term breadth dodged and recommends a human-governed consolidation
  (member reassignment). READ-ONLY: no holdout, no ledger writes, no transitions, no graph mutation —
  a prioritization signal, not a gate.
- `uv run algua research gc [--retention-days N --archive --actor human --archive-dir DIR --top N]` —
  ADVISORY reaper of dead strategy artifacts: it classifies the on-disk strategy modules
  (`algua/strategies/<family>/*.py`) and the per-strategy report-experiments subtree
  (`<knowledge_dir>/strategies/<name>/reports/<stamp>/…`, keyed by the `<name>` DIRECTORY) against
  the registry and lists what is safely reapable — a strategy RETIRED for more than `--retention-days`
  (default 90), or a report file with no registry row (orphaned) older than the window. The
  kb-sync-OWNED top-level `<knowledge_dir>/strategies/*.md` files (the `_*` router pages AND every
  per-strategy live synced note at `strategy_doc_path`) and the `families/` subtree are NEVER
  scanned or reaped. READ-ONLY by default (a listing is a prioritization signal, NOT a transition).
  Fail-safe: an untracked module, a non-terminal strategy, a retired-without-timestamp, and a
  fresh/undatable orphan report are ALWAYS kept. `--archive --actor human` (human-only, #329 signed
  challenge) MOVES the reapable files into a timestamped `archive/` tree via a single atomic
  `os.replace`; it NEVER deletes and NEVER touches the immutable registry DB row.
- `uv run algua research pbo <name> --param KEY=v1,v2,... [--windows N --rank-by mean_sharpe|min_sharpe]`
  — ADVISORY Probability-of-Backtest-Overfitting (PBO) via CSCV (#467) over the SAME grid a
  `backtest sweep` runs. Answers a question the DSR/FDR/breadth stack never asks: does the selection
  rule "pick the in-sample-best combo" GENERALIZE out-of-sample? Builds the trials×windows OOS-Sharpe
  matrix (holdout excluded by construction), then reports `pbo` = the fraction of combinatorially-
  symmetric IS/OOS splits where the IS-best combo lands below the OOS median. A HIGH pbo (≳0.5) flags
  a sweep fitting noise. This is a REAL grid search, so — like `backtest sweep` — it RECORDS its
  measured breadth (metered: repeated `pbo` runs self-penalize the eventual promotion Sharpe bar);
  it burns NO holdout STATISTIC (the holdout bars are read as part of the full-period sweep fetch but
  never scored/burned — `compute_holdout=False`), transitions nothing, and writes no gate/FDR ledger
  row. Output is AGGREGATE-ONLY (pbo, split/trial/window/subperiod counts, rank_by, warnings, and a
  fully-reconstructable `provenance` block: base `config_hash`, full `grid_hash`, delisting inputs) —
  never the raw matrix, ranked combos, or per-split logits. Fails closed (`pbo: null` + warning,
  exit 0) on <2 combos, <4 windows, or a non-finite matrix. See the `interpret-results` skill for how
  to read it. Orthogonal to promotion: a winner can pass DSR yet have a high PBO.
- `uv run algua data ingest ... --from-file PATH` — register a local immutable snapshot.
- `uv run algua data ingest-bars --provider yfinance --symbols AAPL --start D --end D` — fetch
  historical bars into a parquet snapshot.
- `uv run algua data ingest-universe NAME --symbols AAPL,MSFT --effective-date D` — record
  point-in-time universe membership.
- `uv run algua data import-universe NAME --file constituents.csv` — bulk-import a PIT constituents
  CSV (`symbol,add_date,drop_date`; add inclusive, drop exclusive; multiple rows/symbol for
  re-additions, including delisted tickers) into the universe-snapshot timeline (one snapshot per
  change date). Universes are IMMUTABLE — a same-date membership conflict aborts before any write
  (corrections need a new name); an empty-membership change date is rejected (deferred limitation).
- `uv run algua data import-delistings --file delistings.csv` — import per-symbol terminal prices
  (`symbol,delisting_date,delisting_value`; value = per-share terminal proceeds in adj_close units,
  strictly > 0) as a point-in-time delistings snapshot. Backtests opt in with `--delistings NAME`: a
  held name whose bars end mid-backtest is realized at its terminal price and removed (no silent
  survivorship drop); a held-into-gap name WITHOUT a record fails closed. `--assume-terminal-last-close`
  realizes such a name at its last close instead, but is HUMAN-ONLY (rejected on the agent
  `research promote` path).
- `uv run algua data import-bars --vendor firstrate --raw-dir DIR --adjusted-dir DIR --as-of TS` —
  bulk-import local vendor files (FirstRateData: per-symbol unadjusted + adjusted), normalized to
  the bar-schema as one consolidated snapshot. Streamed (bounded RAM); `adj_close` from the adjusted
  file (no corporate-action math yet).
- `--summary` (context-rot defense #349) — `backtest walk-forward`, `backtest sweep`, and
  `research promote` accept `--summary` to emit ONLY the decision-relevant scalars (drops the
  per-window/per-combo lists and the deep dsr_*/fdr_*/regime gate diagnostics); the projected
  payload carries `"summary": true`. Prefer it for unattended operation; omit for full detail.
- `uv run algua data inspect [--summary|--dataset NAME|--snapshot-id ID]` — inspect data snapshots.
- `uv run algua data verify [--snapshot-id ID]` — power-loss backstop: read each snapshot's
  payload back from disk (full read-back) and check it against its record; emits per-snapshot
  JSON and exits non-zero if any snapshot is damaged.
- `uv run algua factor list [--kind K]` — catalogue of known factors.
- `uv run algua factor show <name>` — one factor's full spec.
- `uv run algua factor eval <name> --symbols S,S --construction POLICY --demo` — evaluate ONE
  standalone factor: PIT backtest + FDR-corrected rank IC/IR (#219 slice E). Emits an `fdr`
  block: `breadth_benchmark_t`, `dsr_confidence`, **`significant`** (the honest verdict after
  funnel-wide multiple-testing correction), and `n_dependents` (blast radius). Records the eval
  in the `factor_evaluations` ledger (SCHEMA_VERSION 25) — `factor eval` is the ONLY agent path
  to a multiple-testing-honest factor verdict. See `interpret-results` skill for how to read the
  output. Factors are NEVER gate-tokened or live-pathed.
- `uv run algua factor dependents <name>` — strategies that compose a factor (blast radius).
- `uv run algua factor uses <strategy>` — factors a strategy composes.

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
