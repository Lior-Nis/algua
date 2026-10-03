---
baseline_commit: 14621bf
---

# Story 1.4: Controlled exit of the legacy paper cohort

Status: review

Prepared: 2026-10-01; rescoped the same day. Baseline: Story 1.3d merge `14621bf` (PR #683; schema
v48 applied in production 2026-10-01). Epic: 1. Requirements: FR7, FR9–FR10 and NFR1, NFR4–NFR8.
Depends on: Stories 1.2 and 1.3a–1.3d, reviewed and merged.

## Story

As Algua's operator,
I want every legacy paper tenant stopped, liquidated and retired,
so that the only strategies that trade in paper from now on are frozen deployments admitted through
the qualified path.

## Context

All 18 production paper tenants belonged to the Story 1.2 legacy cohort: they ticked the mutable
checkout with `deployment_id` NULL, could never be promoted, and carried no creditable evidence.
They all share one Alpaca paper account.

The first version of this story designed a requalification-preserving `migrate-legacy` command. Its
readiness review returned NOT READY
([report](../implementation-readiness-report-2026-10-01-story-1-4.md)), and the same day the owner
decided to kill the whole cohort instead. Retirement is terminal, so the requalification machinery
is not needed; the existing emergency `paper flatten` and the `paper -> retired` transition do the
job once three defects are fixed.

## Normative contract

[SPEC](../specs/spec-story-1-4-legacy-cohort-exit/SPEC.md) is the complete contract. The superseded
first design is kept beside it as the record the readiness review judged.

## Scope and authority

Three fixes to existing paths and an operational exit. No schema change, lifecycle edge, gate, CLI
command, error code or authority change. Retiring paper strategies and flattening them are
agent-allowed, risk-reducing actions; the owner asked for both.

## Acceptance criteria

1. **Sibling-safe flatten.** `paper flatten NAME` cancels only that tenant's own open orders, never
   the account's; a sibling's resting order survives (test).
2. **Tradeable offsets only.** On the paper lane, a long offset never sells more than the paper
   ledger says the shared account holds; a residual below the venue minimum ($1 at the symbol's
   latest recorded fill) or the reconcile tolerance is skipped rather than refused mid-loop; a
   material short is bought back in full (tests, including the production UNH pair).
3. **Retirable when flat.** A paper-lane bench transition accepts sub-minimum residuals and still
   refuses, naming the symbols, when a tradeable position remains; the live lane's rule is
   unchanged (tests).
4. **One protected rule.** The rule lives in one leaf module the registry may import, protected by
   CODEOWNERS and the integrity-critical set; import contracts, size pins and the full gate pass.
5. **No halt from retirement.** A strategy that left the paper lane with dust keeps that dust
   explained in the account reconcile; a material non-lane belief never explains a position (tests).
6. **Honest emergency flatten.** One symbol's venue refusal does not stop the rest; unsold material
   beliefs are reported; the scoped cancel never acts on a truncated open-order list (tests).
7. **The cohort is gone.** Every legacy tenant is kill-switched, flattened and retired with audited
   transitions, its allocation revoked, and the paper operator resumes with no legacy tenant.

## Owner decision (Lior, 2026-10-01)

"Kill all strats": retire all 18 legacy tenants; requalify none. This supersedes the earlier
decisions to requalify the stronger tenants after fresh holdout data (O1–O3) and to clear kill
switches at exit (O5). Deferred, unchanged: the name-reuse laundering policy and deleting the legacy
and working-tree tick paths now that the cohort is empty.

## Tasks / subtasks

- [x] Contract, readiness review and rescope.
- [x] Scope the `paper flatten` cancel to the strategy's own orders (AC1).
- [x] Shared paper dust rule and ledger-net offset cap in the flatten loop (AC2).
- [x] Accept sub-minimum paper residuals in the bench flatness check (AC3).
- [x] Protection, mutation checks, full gate (AC4).
- [x] Independent review round and fixes (AC5–AC6).
- [ ] Merge and deploy.
- [ ] Flatten and retire the cohort; resume the paper operator (AC7).

### Review Findings

Independent review round 1 (2026-10-03; Codex out of quota, so the story-delivery fallback ran a
combined Blind Hunter + Edge Case Hunter + Acceptance Auditor pass and applied the rule to the
production ledger read-only).

- [x] [Review][Patch] HIGH: retiring with dust dropped the residual out of the paper account
  reconcile while the broker still held it (or its counterpart), deferring every cycle and engaging
  the global halt; left-lane dust is now counted [algua/execution/paper_reconcile.py]
- [x] [Review][Patch] The ledger-net cap cannot see sells queued by earlier flattens, so a later
  tenant's offset could still be refused and abort its loop; a per-symbol venue refusal now
  continues with the remaining symbols [algua/execution/flatten.py]
- [x] [Review][Patch] The scoped cancel listed only Alpaca's default 50 open orders; it now reads up
  to 500 and fails closed on a full page [algua/execution/alpaca_broker.py]
- [x] [Review][Patch] A capped or skipped material belief was reported as success; `paper flatten`
  now reports it as `unsold` [algua/execution/flatten.py, algua/cli/paper_cmd.py]
- [x] [Review][Patch] The protected rule read unprotected sizing/reconcile constants; `dust.py` owns
  its constants [algua/execution/dust.py]
- [x] [Review][Defer] Paper-lane bench exits have no open-order re-check (live has one) — #685; the
  operator flattens and waits for fills before retiring
- [x] [Review][Defer] A stale last-fill price can misjudge a residual within about $1 of the
  minimum — bounded, recorded

## Development notes

- 2026-10-01 production state before the exit: all 18 tenants kill-switched by `paper flatten` at
  14:49 UTC with the paper timer stopped. 16 flattened cleanly. `distributed_gains_quality_momentum`
  had a sub-$1 UNH buy-back refused (-0.000123 shares), and `liquidity_stable_quality_momentum`'s UNH
  sale (3.552855) was refused because the account held 3.55273207: the two beliefs sum to the
  account. Six tenants carried float residuals near 1e-16 shares.
- Ledger-derived cap, never broker holdings: in a shared account the broker's net is not one
  tenant's, and capping to it can sell in the wrong direction.

## Verification

```bash
uv run pytest -q
uv run ruff check .
uv run mypy algua
uv run lint-imports
```

## References

- [Story 1.2](1-2-record-working-tree-deployments.md) (the legacy cohort)
- [Readiness report](../implementation-readiness-report-2026-10-01-story-1-4.md)
- #677 (notional-sized fills leave cross-tenant residuals), #662 (wash trades)
