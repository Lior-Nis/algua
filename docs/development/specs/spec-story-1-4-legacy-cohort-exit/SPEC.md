---
id: SPEC-story-1-4-legacy-cohort-exit
companions:
  - ../../stories/1-4-controlled-exit-of-the-legacy-paper-cohort.md
sources:
  - legacy-exit-contract.md
  - ../../implementation-readiness-report-2026-10-01-story-1-4.md
---

> **Canonical contract.** This SPEC is the complete contract for what to build, test and validate.
> `legacy-exit-contract.md` is the superseded first design, kept as the record the readiness review
> judged.

# Story 1.4 Legacy Cohort Exit Contract (rescoped 2026-10-01)

## Why

Every production paper tenant was a Story 1.2 legacy-cohort member: it traded the mutable checkout,
could never be promoted, and carried no creditable evidence. On 2026-10-01 the owner decided to kill
the whole cohort rather than requalify any of it. The existing emergency `paper flatten` and the
`paper -> retired` transition already do that per strategy, except that three defects left the
cohort impossible to exit cleanly on the shared paper account.

## Capabilities

- id: CAP-1
  intent: `paper flatten NAME` stops and liquidates one tenant without touching its siblings.
  success: It cancels only the tenant's own open orders (the scoped cancel `paper run-all` uses),
    never the whole account's, so flattening tenants one after another while the market is closed
    does not cancel the offsets queued by the previous ones.
- id: CAP-2
  intent: A tenant can always be flattened to what the venue can trade.
  success: On the paper lane, a long offset sells at most what the paper ledger says the shared
    account holds (the sum of every recorded paper fill in that symbol), never broker holdings; a
    residual worth less than the venue's minimum order (`MIN_NOTIONAL`, $1) at the symbol's latest
    recorded fill, or within the reconcile tolerance, is skipped instead of being refused by the
    venue mid-loop.
- id: CAP-3
  intent: A flattened tenant can retire.
  success: The `paper -> retired` (and every other paper-lane bench) flatness check treats the same
    sub-minimum residuals as flat; a tradeable residual still blocks and names its symbols. The live
    lane's rule is unchanged.

## Constraints

- One shared rule: `algua/execution/dust.py` (`paper_dust`, `paper_ledger_net`) is a leaf module so
  the registry's bench check and the flatten loop use the same rule without the registry importing
  the live lane. It is CODEOWNERS-protected and in the integrity-critical set, because the protected
  bench check trusts it.
- No schema change, lifecycle edge, gate, authority, CLI command or error code. The broker's
  `submit_offset` stays strict: an emergency offset the account cannot fully fill still raises.
- The paper `trade-tick` breach keeps its account-wide cancel (single tenant by construction).

## Non-goals

- The requalification-preserving `migrate-legacy` command, drawdown-peak rebase and kill-switch
  clearing (the strategies are retired, a terminal stage).
- Fixing notional-sized fills that create cross-tenant residuals in the first place (#677).
- Deleting the legacy and working-tree tick paths (a follow-up now that the cohort is empty).

## Success signal

All 18 legacy tenants are kill-switched, flattened and retired with audited transitions; the paper
account holds nothing they own; the paper operator runs with no legacy tenant; the full repository
gate passes.
