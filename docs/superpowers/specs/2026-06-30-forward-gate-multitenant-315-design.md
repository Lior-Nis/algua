# Forward gate: attribution-cleanliness replaces single-tenant (#315) — design

**Status:** Draft (design approved in brainstorming; pending human review → implementation plan)
**Date:** 2026-06-30
**Issue:** #315 (epic #318 — multi-tenant attributed paper forward-testing).
**Builds on:** #249 (`paper_venue_orders`, on main), #314/#316a/#316b (per-strategy NAV recorded in
`tick_snapshots`). ⚠️ **CODEOWNERS-protected** promotion-integrity gate (`forward_promotion.py`,
`forward_gates.py`) — needs an authorized merge + adversarial (Codex) review.

## 1. Problem

The forward gate forbids co-tenancy: `single_tenant_ok = (n_siblings == 0)` fails a strategy if any
other strategy traded the same `account_id` in its window (`forward_promotion.py` ~270-283), and
`_classify_activities` marks a FILL attributable only if it matches **this** strategy's orders — so a
sibling's fill is "unattributable" and trips `no_unattributable_fills`. Together they block the
multi-tenant paper book the epic restores. This issue relaxes exactly those two points to an
**attribution-cleanliness** rule, keeping every other integrity guarantee.

**Why it's safe to relax:** each strategy's measured **return series / Sharpe** comes from its own
per-strategy NAV equity in `tick_snapshots` (#314 + #316a/b), clean by construction — independent of
how many strategies share the account. So co-tenancy never contaminates a strategy's evidence.

## 2. Changes

### (a) `single_tenant_ok` → `single_account_ok` (keep the mixed-account half)
Today's `single_tenant_ok` bundles two checks: (i) the strategy's admissible ticks all come from ONE
`account_id` (`len(distinct_accounts) == 1` — mixed-account evidence is a tenancy violation), and
(ii) no siblings on that account (`n_siblings == 0`). Drop **(ii)**, keep **(i)**:

- `forward_promotion.py`: compute `single_account_ok = (account_id is not None and
  len({r["account_id"] for r in admissible}) == 1)`; **delete the `n_siblings` query**.
- `ForwardGateEvidence`: rename field `single_tenant_ok` → `single_account_ok`.
- `forward_gates.py`: the `single_tenant` check (line ~226) becomes `single_account`
  (`evidence.single_account_ok`); the emitted decision key renames `single_tenant` → `single_account`.

### (b) Account-level fill attribution
`_classify_activities(conn, strategy_id, acts)` → `_classify_activities(conn, acts)`: a FILL is
attributable iff it matches **any** recorded paper-venue order
(`SELECT 1 FROM paper_venue_orders WHERE broker_order_id = ?`), not specifically this strategy's. A
fill matching no recorded order is truly external/manual → unattributable → still fails. Drop the
now-unused `strategy_id` param at both call sites (the evidence assembly and the certificate
re-verification path). `n_external_cash_flows` logic unchanged.

### Unchanged
`reconcile_ok`, the realized-Sharpe bar, observations/coverage/vol/drawdown/staleness, the identity
and qualified-holdout checks, `n_concurrent_forward` (stays **recorded, advisory** — no concurrency
deflation added; a deliberate YAGNI choice given the absolute Sharpe bar + upstream LORD++ FDR).

## 3. Testing

- **Relaxation (headline):** a strategy with a **sibling** recording paper ticks on the same
  `account_id` during its window now **passes** (was a `single_tenant` fail).
- **Mixed-account still fails:** admissible ticks spanning two `account_id`s → `single_account_ok`
  False → gate fails.
- **Account-level attribution:** a sibling's FILL (matching the sibling's `paper_venue_order`) →
  attributable → `no_unattributable_fills` **passes**; a truly orphan FILL (`broker_order_id` in no
  recorded order) → unattributable → **fails**; an external cash flow → `no_external_cash_flows`
  **fails** (unchanged).
- **Certificate re-verify path** uses the account-level `_classify_activities` — a re-verify test.
- **Regression / faithful updates:** a single-strategy window (no siblings) still passes; existing
  `test_forward_promotion.py` / `test_forward_gates.py` single-tenant tests are **re-expressed** to
  the new semantics (sibling → pass, mixed-account → fail) — never weakened; the decision-key rename
  (`single_tenant` → `single_account`) is asserted.
- TDD; `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports` green.

## 4. Non-goals

- No concurrency-based multiple-testing deflation (`n_concurrent_forward` stays advisory) — defer
  until real traffic shows concurrent forward-tests inflating false promotions.
- No change to `reconcile_ok`, the Sharpe bar, or any other gate criterion.
- No `run_tick`/CLI/run-all change (those are #316a/b); no schema change.

## 5. Risk

- **Protected promotion-integrity surface.** The relaxation must not open an attribution hole. The two
  remaining account-integrity guarantees — `single_account_ok` (one account per strategy's evidence)
  and account-level `no_unattributable_fills` (every fill maps to a recorded paper order) +
  `no_external_cash_flows` — must jointly ensure no orphan/manual fill or cross-account contamination
  can sneak into a certified series. The final review goes to Codex with an explicit
  prove-no-attribution-hole mandate.
- **Merge order:** land after #316a (#322) + #316b (#323) — co-tenancy is only *enabled* by run-all,
  and the clean per-strategy series depends on the #316a/b NAV recording. CODEOWNERS-authorized merge.
