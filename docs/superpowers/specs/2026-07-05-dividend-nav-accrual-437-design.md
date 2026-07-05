# Dividend NAV credit — close the live/paper vs backtest total-return parity break (#437)

**Date:** 2026-07-05
**Issue:** #437 — live sizing/drawdown NAV excludes dividend cash while the backtest equity uses
total-return `adj_close`; the live/paper ledger NAV is a raw-close price-return basis, systematically
understating the sizing denominator and tripping the drawdown breaker early on any dividend-paying
long.
**Lane:** qf. **Severity:** high.

---

## What this PR ships (the minimal slice)

A single per-strategy dividend-credit derivation, `strategy_cash_credit(conn, strategy, kind)`
(`algua/execution/live_ledger.py`), wired additively into both NAV paths — `strategy_nav`
(live_ledger) and `build_sizing_snapshot` (live_sizing, both `LedgerKind.LIVE` and `PAPER`). Neither
lane's mark embeds dividends (both mark at raw `close`), so the credit is purely additive with no
double-count.

The credit is derived from the broker's own `DIV` cash activities already ingested into
`{live,paper_venue}_activities`. There is **no new table and no schema bump.** Attribution is:

- **Dividend-only.** Only `type = 'DIV'` rows are read. Non-dividend cash (interest, fees, journals,
  deposits) and NULL-amount / symbol-less rows are excluded — they are not per-security total-return
  and have no place in a parity-with-`adj_close` NAV.
- **Signed and per-side, from each strategy's OWN position.** Each `DIV` row is attributed
  individually (never summed across rows first). A **positive** amount is a long-side credit, split
  across the strategies **long** the symbol pro-rata by their long shares; a **negative** amount is a
  short-dividend debit, split across the strategies **short** the symbol pro-rata by their short
  shares. A long is credited and a short debited **independently** — there is no shared long+short
  divisor, so an offsetting book (A long, B short) does not cancel to ~0 and is not sign-blind.
- **Bounded to the dividend's own date.** A strategy's entitled share base is its signed position
  reconstructed from fills on or before the activity date (`date(fill_ts) <= date(ts)`). A past
  dividend's attributed credit is therefore **deterministic** and does not drift as later, unrelated
  fills accumulate.
- **Never divides by zero.** A row contributes only when its same-side share base is positive; a row
  whose entitled side is empty in the ledger stays an unattributed account-level residual.

Tests (`tests/test_live_ledger_pnl.py`, `tests/test_live_ledger_orders.py`,
`tests/test_live_sizing.py`) cover: a single long holder taking the full credit; a long-side credit
split by long shares; a **short position debited** (negative credit) on a dividend; an **offsetting
long/short book** where both sides get nonzero, oppositely-signed attribution; **determinism** of a
historical credit under later unrelated fills; and exclusion of non-`DIV` / symbol-less / untraded
rows.

### Known limitations of the slice (deliberate — not bugs)

1. **The per-share figure is IMPLIED from account cash, not declared.** The credit uses the broker's
   net `DIV` cash `amount`, which is post-withholding/fees, not the gross declared dividend that
   `adj_close` reinvests. Over the withholding haircut, live NAV is credited slightly *under* the
   backtest's gross total return. This is a smaller residual parity error than the fully-excluded
   dividend it replaces, but it is not exact.
2. **A single broker-netted row cannot recover an internal long/short split.** If the broker books
   one net row for a symbol the ledger holds both long and short internally, this slice attributes it
   to the long side only; the true per-side split needs the declaration. It still conserves the
   booked cash and never blows up.
3. **The entitlement bound is the activity date, not a calendar ex-date/record-date rule.** For the
   current regular-hours daily rail this is a close approximation; it is not the exchange
   record-date convention. The bound is deliberately **date-level, not full-timestamp**: broker
   `DIV` rows carry a date-only `date` field (no intraday time), so a finer bound would be false
   precision. A direct consequence — accepted, not a bug — is that **a fill placed the same day the
   dividend posts, even one timestamped strictly *after* the `DIV` row's own instant, is treated as
   entitled** (it shares the activity's date). The exact ex-date/record-date rule is deferred to the
   declaration-sourced design below. Asserted by `test_same_day_fill_after_dividend_is_entitled`.
4. **A `DIV` row with a NULL or non-ISO-parseable `ts` is failed closed to an account-level
   residual.** Such a row carries no entitlement window we can trust (a malformed `ts` sorting below
   every fill date would otherwise credit an unbounded all-fills window), so it is excluded from
   attribution (`AND ts IS NOT NULL` plus a parse guard) and left unattributed rather than
   silently zero-credited by an empty-string fallback. Asserted by
   `test_dividend_with_null_ts_is_residual` and `test_dividend_with_malformed_ts_is_residual`.

These are acceptable for closing the *gross* parity break (dividend excluded → phantom drawdown) that
#437 is about. The exact, declaration-sourced accrual is deferred (below).

---

## Deferred — the exact declaration-sourced design (NOT in this PR)

The exact fix for the residual errors above is to stop implying the dividend from account cash and
instead **accrue a declared gross per-share at the ex-date, attributed independently per strategy by
entitled quantity**, reconciling the payment-date net cash separately. Sketch of the target design,
recorded so the follow-up has a starting point — **none of it is implemented here, and nothing in
this PR should be read as delivering it:**

- A pure `CorporateActionDeclaration` contract in `algua/contracts/` (symbol, stable
  `corp_action_id`, ex-date, **declared gross per-share** in raw-close units, payment date,
  normalized vendor) sourced from corporate-action / dividend-declaration metadata — never derived
  from net account cash.
- Ex-date accrual into a `dividend_accruals` table (`strategy_accrual(s) = gross_per_share ·
  entitled_qty_at(cutoff, s, symbol)`, signed, independent per strategy), with a payment-date
  reconciliation that flips the accrual to paid, records the net cash, and surfaces the
  withholding/fee residual — without ever re-crediting NAV.
- A `DeclarationSource` port with a live (broker corporate-actions announcements) adapter and a paper
  (snapshot CA-event-manifest) adapter, plus a source-check / basis-parity gate on the forward
  certificate (protected `forward_gates.py` / `forward_promotion.py`).

That design carries its own schema bump, protected-path changes, and a CA-event manifest, and is a
separate, larger change. It should land as its own PR once an implementation matches it.
