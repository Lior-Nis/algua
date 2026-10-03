> **Superseded 2026-10-01, never implemented.** The owner chose to retire the whole cohort
> ("kill all strats"), so the requalification-preserving `migrate-legacy` command below was not
> built. The readiness review that judged it NOT READY is
> `docs/development/implementation-readiness-report-2026-10-01-story-1-4.md`. The rescoped contract
> is [SPEC.md](SPEC.md).

# Story 1.4 legacy exit contract

Normative companion to [SPEC.md](SPEC.md).

## 1. Command

`algua paper migrate-legacy NAME --to backtested|retired [--actor agent|human]`, JSON on stdout,
agent-allowed (it only reduces risk). One invocation performs exactly one phase:

| Tenant state | Effect | JSON `phase` | Exit |
|---|---|---|---|
| Not a legacy-cohort member, or has an active deployment | none | error `legacy_exit_not_cohort` | 1 |
| Cohort member at `forward_tested` or `live` | none | error `legacy_exit_unsupported_stage` | 1 |
| Cohort member at `dormant` and `--to backtested` | none | error `legacy_exit_unsupported_stage` | 1 |
| Already exited (stage `backtested`/`retired`, not allocated) | none | `exited` (idempotent) | 0 |
| Cohort member at `paper` holding ledger positions or owning open orders | drain (§2) | `pending_flat` | 0 |
| Cohort member at `paper` (or `dormant` for `--to retired`), flat, no owned open orders | exit (§3) | `exited` | 0 |

The payload always carries `strategy`, `phase`, `to` and, for `pending_flat`, the offsets submitted
and orders cancelled. `algua paper migrate-legacy --list` prints the remaining cohort members and
their stages (read-only).

## 2. Drain (per strategy, never account-wide)

Under the operator lock:
1. Trip the tenant's kill switch (`reason=legacy_migration`) first, so no paper tick re-enters it.
2. Cancel only the tenant's own open orders (`paper_scoped_cancel`), never `cancel_open_orders`.
3. Submit exact-quantity offsets for its ledger positions through the existing per-strategy
   liquidation (`submit_offset`, share-based) with the scoped cancel injected, recording the offset
   intents as `paper flatten` does.
4. Report `pending_flat`; no stage, allocation, peak or gate change. Offsets fill at the next venue
   session; the operator re-runs the command after fills are ingested.

A refused offset (venue error) aborts the phase with the existing broker error code; the kill
switch stays tripped, which is safe.

## 3. Exit (one transaction)

Under the operator lock, one `BEGIN IMMEDIATE` transaction:
1. Re-check: cohort member; stage `paper` (or `dormant` for `--to retired`); ledger flat
   (the existing bench flatness check); no owned open orders in the paper order ledger. Any failure
   rolls back with `legacy_exit_not_flat` (or the membership/stage codes) and no effect.
2. Revoke the allocation and advance the stage with the existing transition machinery:
   `--to backtested` records two audited rows, `paper → candidate` then `candidate → backtested`;
   `--to retired` records `paper → retired` (or `dormant → retired`). Reason: `legacy_migration`.
3. State hygiene in the same transaction: rebase the tenant's drawdown peak
   (`rebase_strategy_peak`) and clear its kill switch (`kill_switch.reset`) with an audit row
   (owner decision O5). The realized-P&L history in the paper ledger is kept; a later re-admission's
   NAV carries that small realized offset, which is documented rather than rewritten.
4. Commit. The candidate row written on the way to `backtested` carries NULL hashes, as every raw
   back-step does, so it can never satisfy intake's candidate-episode check.

## 4. Invariants the tests must prove

- The only route back to `paper` for an exited tenant is a fresh passing `research promote` at its
  current identity followed by frozen `paper intake`; its old gate is never admissible (already
  consumed and identity-mismatched), and a raw `paper → candidate` is never admissible.
- Interruption at every boundary — during drain, between drain and exit, inside the exit transaction
  (fault injection before commit) — leaves either a legacy paper tenant with its kill switch tripped
  or a fully exited, unallocated strategy with a reset peak and cleared switch; never `paper` without
  cohort membership or an active deployment; never an allocation on a non-paper strategy.
- Another tenant's resting orders are never cancelled (a sibling order survives a drain).
- Re-running any phase is idempotent.
- Working-tree/frozen tenants and the paper/live lanes are unchanged; JSON and exit codes follow the
  CLI error envelope; new codes are registered as stable, non-retryable.

## 5. Placement and protection

- A new registry module `algua/registry/legacy_exit.py` (the exit transaction and checks) and a thin
  CLI command in a new `algua/cli/paper_migrate_cmd.py` (or a small addition to `paper_cmd.py` only if
  it stays under its pin). Both join CODEOWNERS and the hygiene set.
- No schema change. Codes `legacy_exit_not_cohort`, `legacy_exit_unsupported_stage`,
  `legacy_exit_not_flat` are added to the CLI error registry and the envelope doc.

## 6. Runbook (docs/agent/legacy-exit-runbook.md)

Records the owner decisions (2026-10-01): exit the whole cohort once this story ships; retire the six
kill-switched tenants and those with gate holdout Sharpe below about 0.2; move the rest to
`backtested`; wait for fresh out-of-sample data (about 2026-12-04) before re-running
`research promote`, with no signed holdout reuse; the kill switch is cleared at exit. It lists the
per-tenant plan from the production registry at execution time, the two-run drain/exit sequence
across a venue session, and how to verify the cohort is empty.
