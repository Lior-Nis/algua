# Algua — Agent Operating Guide

Algua is an agent-first algotrading platform. You (an agent) and the human operator
drive the system through the **same** CLI. Every data command emits JSON on stdout.

## Orientation — where to look
- **START HERE if you are adding anything:** `docs/architecture.md` — the one-page module map: what
  each package owns, and the registration seam for adding a provider / importer / broker / tracker /
  calendar / strategy / command — new module + one registration line, not a core-file rewrite.
  It also lists the walls (PIT,
  single-use holdout, the paper→live gate, lane parity, executable CODEOWNERS) so you know why a
  change might be refused.
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
  halted, or a fleet row is corrupt. Cadence is COMPLETED sessions of the CONFIGURED exchange
  (`ALGUA_EXCHANGE`, default XNYS) since the last tick (never
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
  `backtested -> candidate` on the **INTEGRITY FLOOR** and promote on pass (**factory soft gate**,
  see `docs/superpowers/specs/2026-08-10-strategy-factory-design.md`). BINDING for everyone: PIT
  universe required (`--universe`; non-PIT fails closed), a minimum holdout-observations floor
  (63 — underpowered holdouts fail closed), and **raw holdout Sharpe > 0**
  (`holdout_sharpe_floor`), plus the preflight raises (measured breadth via `backtest sweep`,
  default-on costs, declared feature_lookback, reproducible source, signal-panel parity, delisting
  handling). The whole statistical stack — the breadth-DEFLATED holdout-Sharpe bar, window
  stability, DSR evidence + bootstrap, regime robustness, idiosyncratic alpha — still computes and
  is RECORDED on every run as **ADVISORY** (`"advisory": true` on the check; `*_binding` fields
  mean armed/evaluated, NOT veto). The **LORD++ FDR ledger is FROZEN**: rows land with
  `fdr_binding` NULL + skip reason `stats_advisory`, the throttle is no longer consulted, and
  `--fdr-throttle-override` is REMOVED (the ledger machinery is preserved for re-tightening — FDR
  mechanics live in the #529 spec). The **FORWARD gate is the harsh threshold** (its Sharpe bar is
  0.5× the holdout Sharpe recorded here, so overfit inflation self-punishes).
  A passing run is the ONLY way an agent reaches `candidate`
  — there is no
  raw `registry transition --to candidate` shortcut for an agent (`--allow-non-pit`,
  `--allow-holdout-reuse`, `--n-combos`, and the raw shortlist transition are all human-only).
  **Family governance (#222, #524):** at preflight, the strategy is empirically classified into a
  family via code-ancestry + factor-lineage + return-correlation clustering; MERGE verdict → assign
  into the incumbent family (inherits its breadth); PARENTAGE → new family but inherits the
  incumbent's accumulated breadth via a parent edge; NOVEL + agent → **the family is NOT created in
  preflight**: classification returns a deferred spec and the seeded family is minted **only if the
  gate passes, AT THE PASS MOMENT**, inside the atomic promote transaction — seeded with the
  funnel-wide **LIFETIME** search total (a durable prior that survives the 90-day window roll, so a
  future sibling can never wait out the window to reset the tax). The founder itself pays family arm
  0 (a no-op, symmetric with a human fresh family). Agent minting is bounded SOLELY by an automatic
  per-window rate cap (`AGENT_NOVEL_MINT_CAP` ≈ 8 mints / 90 days, fail-closed, canonical-UTC,
  CODEOWNERS-protected `algua/registry/store/family.py` constant — no human budget, no human
  top-up: zero-human autonomy).
  The rate cap is the retained count-bound because the deferred pass-time seed alone can't stop a
  repeated-founder attack (the founder passes and escapes the tax before its family is seeded). The
  mint re-checks a still-NOVEL family-graph fingerprint CAS under the write lock (drift →
  `FamilyGraphDrift` re-run; no holdout burned for drift caught at/before the `on_peek` burn — the
  post-peek/pre-lock re-check is a narrowed, monitored residual, release-on-failure + WARNING audit,
  not fully closed). NOVEL + human still creates a fresh 0-prior family in preflight (requires
  `--new-family` + `--actor human`). Family-scoped lifetime breadth feeds the 3-way
  `effective_funnel_breadth(own, windowed_total, family_lifetime_effective)` tighten-only max.
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

Two of those enforce structure, not style, and are worth knowing before you fight them:
- **`lint-imports`** — 28 contracts fixing the package layering (`cli` composes everything; nothing
  composes `cli`). Adding an import that breaks one is a design decision; do not add an exemption
  without saying why in the commit.
- **`tests/test_module_size_ratchet.py`** — a shrink-only size ratchet over every module ≥300 lines.
  Carved files cannot regrow and new god-files cannot appear silently. When it fails, prefer putting
  the change where it belongs, or carving; raising a pin is the last option and is meant to be
  visible in review.

When a change touches `web/` (the monitor PWA — a STANDALONE uv project; NEVER add web deps to
the root project, the root `uv.lock` is dependency_hash identity), also run:
`uv run --project web pytest web/backend/tests -q` and `cd web/frontend && npm run check && npm run build`.

## Company context

Business context for this project — what Nix is, who the client is, what the
current bet is — lives in the company knowledgebase at `~/Projects/nix`.

- Venture note: `ventures/Algua.md`
- Company: `company/Nix.md`, `company/Current Bet.md`

Do not duplicate that context here, and do not write project research there —
see `~/Projects/nix/_meta/project-kb-boundary.md`.
