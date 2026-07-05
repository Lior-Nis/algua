# Dividend NAV accrual — close the live/paper vs backtest total-return parity break (#437)

**Date:** 2026-07-05
**Issue:** #437 — live sizing/drawdown NAV excludes dividend cash while the backtest equity uses total-return `adj_close`; the live/paper ledger NAV is a raw-close price-return basis, systematically understating the sizing denominator and tripping the drawdown breaker early on any dividend-paying long.
**Lane:** qf. **Severity:** high.
**Supersedes:** the issue's own one-line "Recommendation" (sum cash activities per strategy into NAV). GATE-1 (Codex, read-only, grounded in `live_ledger.py` / `live_sizing.py` / `registry/db.py` / `bar-schema.md`) **blocked** that recommendation: the activities tables carry no strategy attribution, a broker dividend lands at payment date (not the ex-date `adj_close` uses), account-level cash noise (deposits/withdrawals/journals/fees) would leak into sizing, and the shared paper venue would double-credit multi-tenant same-symbol holdings. This spec replaces it with an explicit cash-attribution / ex-date-accrual model.

---

## Problem (grounded)

- The backtest simulates on `adj_close` (`bar-schema.md:49`) — a **total-return** series with ordinary cash dividends reinvested retroactively at the **ex-date** (`corpactions.py`: `div_mult = (p_prev - cash) / p_prev` applied at the ex-date boundary).
- The live/paper ledger sizing path is a **price-return** basis: `build_sizing_snapshot` (`live_sizing.py:43-93`) marks positions at **raw** `close` (`_latest_marks` reads `bars["close"]`, the raw column per the schema) and computes `NAV = allocation + Σ position_pnl(fills, mark)` — `position_pnl` (`live_ledger.py:55-83`) is derived **solely from trade fills and the mark**. It never reads the dividend rows in `live_activities` / `paper_venue_activities`.
- `run_tick` (`live_loop.py:268-295`) uses this NAV as **both** the sizing denominator (`equity = min(allocation, NAV)`) **and** the drawdown basis (`check_drawdown(drawdown_equity, peak, max_drawdown)`, peak persisted via `update_nav_peak`).
- **Consequence:** for any dividend-paying long, live/paper NAV drifts below the backtest's total-return equity by the cumulative dividend yield — smaller sized notional, and a drawdown breaker that can halt a healthy book on a phantom drawdown. This corrupts the forward-test certificate, which compares live ticks to the qualifying backtest.

Because **both** the live path (`build_live_sizing_snapshot`) and the multi-tenant paper-venue path (`build_paper_sizing_snapshot`, wired at `paper_cmd.py:418`) go through `build_sizing_snapshot` with raw-close marks, the fix must cover **both lanes** (`LedgerKind.LIVE` and `LedgerKind.PAPER`). Neither lane's mark already embeds dividends, so a dividend accrual credited to NAV is purely **additive** — there is no double-count against an already-total-return mark.

---

## Decision 0 — the NAV basis definition (resolves GATE-1 finding 5)

**Live/paper ledger NAV is defined as `research-equivalent gross accrued total return`.** It tracks the same accounting basis as `adj_close`:

- Ordinary cash dividends are **accrued gross of withholding tax** at the **ex-date**, matching `adj_close`'s timing and gross basis (`adj_close` reinvests the full declared cash, pre-tax).
- The broker's **payment-date net cash** (post-withholding, post-fees) is a **reconciliation target**, not the NAV input. The withholding/fee gap between the gross accrual and the net cash is an explicitly-tracked **reconciliation residual** routed to suspense — it is NEVER folded into the sizing/drawdown NAV.

**Gross per-share is sourced from corporate-action / dividend-declaration metadata, never derived from account cash (fail-closed).** The NAV accrual's `gross_per_share` MUST come from the declared corporate-action record — the dividend-declaration / corporate-action-announcement metadata that states the **gross, pre-withholding** cash amount per share (plus its stable corp-action id, ex-date, and payment date). The **payment-date account cash is net** (post-withholding/fees) and is deliberately NOT a source for `gross_per_share`: dividing net cash by share count would silently bake the withholding haircut into NAV — exactly the opposite parity break Decision 0 exists to prevent. **If no gross per-share is resolvable from corporate-action metadata for an ex-date event, the event fails closed to suspense** (no accrual is written, NAV is untouched, the operator sees an unresolved-declaration residual). We never fall back to deriving the accrual from net cash received. The declared metadata is the single authoritative gross input; net account cash appears only later, as the payment-date reconciliation target (§2).

**Rationale.** The north star is backtest↔live parity: the frictionless backtest gates candidate promotion and the forward certificate compares live ticks to that backtest. The backtest equity is gross total return (no withholding modeled). If live NAV credited only the net-of-withholding cash, we would trade one parity break (dividends excluded) for the opposite one (dividends under-credited by the tax haircut). Defining NAV on the backtest's own basis makes the sizing denominator and drawdown basis directly comparable to the qualifying backtest. The real withholding cash cost is genuine but belongs in a reconciliation/attribution report, not in the parity-critical NAV.

Everything below follows from Decision 0.

---

## Design

### 1. Strategy attribution for non-fill cash — derivation rule, NOT a `strategy_id` column (resolves finding 1)

A dividend cannot carry a `strategy_id` the way `paper_venue_orders` does: the broker account is a **netted custodian** (`live_ledger.py` module docstring — "The broker account is the netted custodian; this ledger is the source of truth for per-strategy attribution"). The broker emits **one account-level** DIV activity per symbol per ex-date; it has no idea which virtual sub-strategy holds those shares. So attribution must be **derived** from the ledger, exactly as position attribution already is.

**Attribution rule.** For a dividend on `symbol` with ex-date `X` and **declared gross per-share `g`** (from corporate-action metadata — Decision 0; if `g` is unresolvable the whole event fails closed to suspense and nothing below runs):

```
strategy_accrual(s)  = g * entitled_qty_at(X, s, symbol)      # signed, per strategy, INDEPENDENT
```

Every strategy is attributed **independently** from the declared gross `g` and its **own** entitled quantity. There is NO division by a shared, signed net-quantity total — so an offsetting long/short book (e.g. tenant A long 100, tenant B short 100, netting to 0) never zeroes out or distorts attribution: A accrues `+100·g`, B accrues `−100·g`, each correct on its own book. (The old design divided the account cash by `total_ledger_qty_at`, a signed sum that collapses to ~0 exactly when longs and shorts offset — a divide-by-near-zero that both blew up the per-share figure and mislabeled a fully-attributable event as "non-attributable." That divisor is gone.)

- `entitled_qty_at(X, s, symbol)` is the strategy's signed position **at the entitlement cutoff** (defined below), reconstructed from the append-only fills ledger: `Σ fills.qty WHERE strategy = s AND symbol = ? AND fill_ts <= entitlement_cutoff(X)`. (The fills table is append-only with `fill_ts`; historical positions are exactly reconstructable — the same sum `believed_positions` computes, with a time filter at the cutoff.)
- **Sign:** a long (`qty > 0`) is **credited** (+), a short (`qty < 0`) is **debited** (−) — a short borrower owes the dividend. `strategy_accrual` inherits the sign of `entitled_qty_at` automatically. Longs and shorts are separately-signed terms in an independent per-strategy computation; they are never summed into a divisor.

**Entitlement cutoff (pinned).** `entitlement_cutoff(X)` is **the close of the last exchange-calendar session strictly before the ex-date `X`**, expressed as a single canonical instant in **one timezone** — the exchange calendar's `America/New_York`, materialized to a UTC instant for storage/comparison against `fill_ts` (which is stored UTC). Concretely: take the exchange trading calendar, find the last session whose date `< X`, take that session's regular close (16:00 `America/New_York`, DST-aware), convert to UTC — that instant is the cutoff. A holder **at that close** carries the position into the ex-date and is entitled (`fill_ts <= cutoff`); a fill after it (i.e. on or after the ex-date session) is **not** entitled. This matches the backtest exactly: `corpactions.py` applies `div_mult` at the ex-date bar boundary, crediting the position established by fills on sessions *before* ex-date; the daily UTC-midnight bar rail (#262) places the prior session close strictly between the prior bar and the ex-date bar, so "held at prior session close" and "held going into the ex-date bar" denote the same position. The cutoff is a single deterministic instant — no wall-clock ambiguity, no per-fill timezone guessing.

- **`total_ledger_qty_at(X, symbol)` is retained ONLY for reconciliation, never for NAV attribution.** It is the ledger's net entitled share count and is compared against the broker's declared/paid share count at payment-date reconciliation (§2) to surface orphan/manual holdings and attribution drift. It is not a divisor and not on the NAV path.

**Why derivation, not a column.** A `strategy_id` column on `live_activities`/`paper_venue_activities` would have to be populated by the same ledger derivation anyway (the broker does not supply it), so the column adds a denormalized copy with no new information and a new corruption surface. The derivation reads the one source of truth (fills) directly.

**Non-attributable / orphan residual.** Attribution no longer depends on a nonzero net total, so an offsetting book is fully attributed. The remaining orphan case is a share the **account holds but the ledger attributes to no strategy** (a manual/external holding): the broker's declared/paid share count exceeds the ledger's net long entitled qty. That excess is **not attributable** — its cash is a reconciliation-report residual (surfaces manual holdings / attribution drift) and never touches any strategy NAV. Attribution writes exactly `Σ_s g·entitled_qty_at(X, s, symbol)` across ledger strategies; the orphan gap is visible only at payment-date reconciliation (§2), not in `dividend_accruals`.

### 2. Ex-date accrual + payment-date reconciliation (resolves finding 2)

A dividend is modeled as an **ex-date receivable**, not payment-date cash. Two events, weeks apart:

1. **Ex-date `X`:** the entitlement exists and `adj_close` steps. We accrue the gross receivable into NAV **as of `X`**, independent of when the broker actually pays.
2. **Payment date `P`:** the broker's cash activity lands. It does **not** re-credit NAV (that would double-count the accrual). It **reconciles** against the open accrual: mark it paid, record the net cash received, surface `gross_accrued − net_cash` as the withholding/fee residual.

**New table `dividend_accruals`** (one `db.py` schema bump: `SCHEMA_VERSION 35 -> 36`; the only bump this change makes — see the operating rule on one bump in flight):

```sql
CREATE TABLE IF NOT EXISTS dividend_accruals (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    lane                TEXT NOT NULL,             -- 'live' | 'paper' (LedgerKind.value)
    strategy            TEXT NOT NULL,             -- attributed sub-strategy
    symbol              TEXT NOT NULL,
    corp_action_id      TEXT NOT NULL,             -- stable declaration id (the reconciliation key)
    ex_date             TEXT NOT NULL,             -- ex-date calendar date (America/NY session)
    entitlement_cutoff  TEXT NOT NULL,             -- UTC instant: last session close strictly before ex_date
    gross_per_share     REAL NOT NULL,             -- DECLARED gross (corp-action metadata; never net-derived)
    accrued_qty         REAL NOT NULL,             -- entitled_qty_at(cutoff) (signed)
    gross_amount        REAL NOT NULL,             -- gross_per_share * accrued_qty (signed; NAV input)
    payment_date        TEXT,                      -- broker payment date when known
    status              TEXT NOT NULL,             -- 'accrued' | 'paid'
    reconciled_cash     REAL,                      -- net cash matched at payment (attributed share)
    source_activity_id  TEXT,                      -- broker activity that drove accrual/payment
    created_ts          TEXT NOT NULL,
    updated_ts          TEXT NOT NULL,             -- bumped on a correcting upsert of an 'accrued' row
    UNIQUE(lane, strategy, symbol, ex_date)        -- one accrual per strategy/symbol/ex-date
);
CREATE INDEX IF NOT EXISTS ix_dividend_accruals_lane_strategy
    ON dividend_accruals(lane, strategy);
CREATE INDEX IF NOT EXISTS ix_dividend_accruals_corp_action
    ON dividend_accruals(lane, corp_action_id);   -- payment-date reconciliation lookup
```

**NAV formula change.** `build_sizing_snapshot` (`live_sizing.py`) and `strategy_nav` (`live_ledger.py`) gain one additive term:

```
NAV = allocation
    + Σ_symbol (pnl.realized + pnl.unrealized)          # unchanged, raw-close price return
    + accrued_dividends(conn, strategy, lane, as_of)    # NEW: Σ gross_amount WHERE ex_date <= as_of
```

`accrued_dividends` sums `gross_amount` for the strategy's accruals with `ex_date <= as_of` (both `accrued` and `paid` rows — once accrued it stays in NAV; the payment-date reconciliation changes `status`/`reconciled_cash`, not the NAV contribution). This is the whole parity fix: NAV now carries the same ex-date-timed total-return dividend contribution as `adj_close`.

**Two independent triggers — accrual is driven by the DECLARATION feed at ex-date, NOT by the payment-date cash activity.** This is the crux of the timing fix and must not be conflated. The broker's DIV **account cash activity typically lands on the payment date `P`, weeks after `X`** — driving the accrual off that activity would leave NAV wrong for the entire `X..P` window (the exact parity break this issue exists to close). So the two events have two separate triggers on two separate feeds:

1. **Accrual trigger — the corporate-action / dividend-declaration feed (announcements), polled independently every run cycle.** This feed carries `(symbol, corp_action_id, ex_date, gross_per_share, payment_date)` for declared/upcoming dividends **at or before ex-date**, independent of when cash pays. Each cycle, for every declaration whose `ex_date` has passed since the last poll (and `ex_date >= cutover_ts`), we compute `entitlement_cutoff` and the per-strategy entitled qty and write the ex-date accrual. Because `accrued_dividends(as_of)` dates the contribution at `ex_date` (`ex_date <= as_of`), NAV reflects the dividend from `X` onward even though the poll that wrote the row runs on/just after `X` — never waiting for `P`.
2. **Reconciliation trigger — the payment-date DIV cash activity.** `_ingest_one_activity` recognizes the whitelisted DIV cash activity and uses it ONLY to reconcile the already-open accrual (match by `corp_action_id`, flip `status='paid'`, record `reconciled_cash`, surface residuals). It NEVER writes or re-credits an accrual — the accrual already exists from trigger 1.

- **Live:** the accrual (trigger 1) reads Alpaca's **corporate-actions announcements** endpoint (declared gross cash per share + ex/payment dates + stable id) — the authoritative gross, pre-withholding declaration. The payment-date DIV account activity (trigger 2, net cash) reconciles it. **Fail-closed on any missing declared input:** a declaration lacking `gross_per_share`, `ex_date`, or `corp_action_id` is not accrued (routed to suspense); we never guess `ex_date = payment_date` (timing break) and never derive `gross_per_share` from net account cash (withholding break). A payment-date DIV cash activity that matches NO open accrual (no announcement was seen) is an **unmatched-payment discrepancy** surfaced to the reconciliation report — it is NOT back-filled into an accrual from net cash.
- **Paper venue:** the shared paper venue must emit declaration events (trigger 1) at ex-date for the accrual to fire. If the venue does not synthesize them today (it fills on a price grid only), that is a **known gap** this issue closes for paper parity: the venue emits a synthetic **declaration** per ex-date (carrying `corp_action_id`, `gross_per_share`, `ex_date`) from the same corporate-action data the sim already applies, shaped identically to the live declaration so ONE accrual path serves both lanes. (See Task list — a spike task confirms whether the venue emits declarations today and scopes the synthesizer.)

**Accrual insertion is a correction-aware upsert, NOT `INSERT OR IGNORE`.** A plain `INSERT OR IGNORE` on the `UNIQUE(lane, strategy, symbol, ex_date)` key would satisfy idempotency but would **silently preserve a stale/incorrect accrual** when the broker later corrects the declaration (revised gross per-share, corrected ex-date qty, restated corp-action). Instead:

```sql
INSERT INTO dividend_accruals (...) VALUES (...)
ON CONFLICT(lane, strategy, symbol, ex_date) DO UPDATE SET
    corp_action_id = excluded.corp_action_id,
    entitlement_cutoff = excluded.entitlement_cutoff,
    gross_per_share = excluded.gross_per_share,
    accrued_qty = excluded.accrued_qty,
    gross_amount = excluded.gross_amount,
    updated_ts = excluded.created_ts
WHERE dividend_accruals.status = 'accrued'          -- only an OPEN accrual may be corrected
  AND ( dividend_accruals.gross_per_share <> excluded.gross_per_share
     OR dividend_accruals.accrued_qty     <> excluded.accrued_qty
     OR dividend_accruals.corp_action_id  <> excluded.corp_action_id );
```

- An **identical re-pull** matches no changed column → the `WHERE` is false → no write. Idempotent (same guarantee `INSERT OR IGNORE` gave) without a spurious `updated_ts` bump.
- A **genuine correction to a still-open (`accrued`) row** overwrites the stale figures and bumps `updated_ts` — the NAV term self-heals to the corrected declaration.
- A correction arriving **after the row is `paid`** (`status='paid'`) is guarded out by the `status='accrued'` predicate: the settled row is **not** silently rewritten. The conflicting corrected event is instead routed to the reconciliation report as a **post-payment restatement discrepancy** for human attention (fail-closed — a restated dividend on an already-settled, already-reconciled accrual is an exception, not a silent NAV edit).

This mirrors — and hardens — the existing `activity_id`-dedup discipline in `ingest_activities`: dedup on identity, but let a corrected upstream record correct the derived row rather than freezing the first-seen (possibly wrong) value.

### 3. Whitelist of NAV-eligible activity types; everything else fails closed to suspense (resolves finding 3)

Only **ordinary cash dividends** are eligible to touch NAV. A hardcoded allowlist gates it:

```python
NAV_ELIGIBLE_ACTIVITY_TYPES = frozenset({"DIV"})   # ordinary cash dividend only
```

- **Interest (`INT`)** is deliberately **excluded**: interest on a cash balance is an account-level artifact, not a total-return component of any held asset, and it is not in `adj_close`. Crediting it would create a fresh parity break. (Documented as an explicit non-inclusion, not an oversight.)
- **Special / return-of-capital / liquidating distributions** are excluded to stay symmetric with the backtest, which *rejects* them (`corpactions.py` raises when `cash >= p_prev`). A DIV whose declared `gross_per_share >= prior close`, or a non-ordinary DIV subtype (e.g. `DIVROC`, `DIVCGL`, `DIVCGS`, `DIVNRA`), routes to suspense.
- **All other cash** — deposits (`CSD`), withdrawals (`CSW`), journals (`JNLC`/`JNLS`), transfers (`ACATC`/`ACATS`), fees (`FEE`), and any **unrecognized/unknown type** — routes to suspense and can never reach sizing or drawdown. This is the fail-closed default: an activity type must be *explicitly* whitelisted to affect NAV.

**Suspense.** Non-eligible cash is recorded (as today) in the `*_activities` table, which is **already excluded from NAV** (`position_pnl` / `accrued_dividends` never read it). No behavioral leak exists today; the design's job is to *keep* it that way when we start reading dividend rows — i.e. the accrual reads ONLY whitelisted DIV events, never the raw `*_activities` sum. A `cash_suspense` view/report over the non-eligible `*_activities` rows (plus the withholding residual from §2 and the non-attributable residual from §1) gives the operator visibility without any NAV path. No new suspense table is required — the existing `*_activities` table is the suspense ledger; the guarantee is that NAV's dividend term is sourced from `dividend_accruals` (whitelisted, attributed, ex-date-accrued), not from a raw activities sum.

### 4. Paper multi-tenancy — per-`strategy` accrual in the shared venue (resolves finding 4)

The shared paper venue is a single account with multiple tenant strategies; a `LedgerKind.PAPER` DIV for `AAPL` is **one account-level event** even when three tenants hold `AAPL`. The §1 derivation rule handles this natively and **independently per tenant**: each tenant `s` accrues `g · entitled_qty_at(X, s, AAPL)` from the same declared gross `g`, producing one `dividend_accruals` row **per tenant strategy**. Because each row is computed from `g` and that tenant's own entitled qty — with **no shared divisor** — offsetting tenant books never interfere: a tenant long `AAPL` and a tenant short `AAPL` in the same venue accrue a credit and a debit respectively, both correct, even if the venue nets flat. `accrued_dividends(conn, strategy, 'paper', as_of)` reads only that strategy's rows, so each tenant's NAV, sizing, and drawdown see only their own dividend share. This is the paper analog of how `believed_positions` already scopes fills per `strategy` in the shared `paper_venue_fills` table. The account-vs-ledger share reconciliation (orphan detection) happens once per event at payment date (§1 orphan residual / §2), not per tenant.

### 5. Parity test asserting Decision 0, plus the edge cases (resolves finding 5)

Tests assert the NAV basis is **gross accrued total return** and that it matches the backtest's `adj_close` basis:

1. **Basis parity (the core assertion).** A single long held across an ex-date: the ledger NAV growth from `T-1` to ex-date `X` (raw-close price return **+** the gross dividend accrual) equals the backtest total-return equity growth on the same `adj_close` bars over the same interval, within tolerance. This is the assertion the issue asked for, pinned to the **gross** definition.
2. **Ex-date vs payment-date separation.** Accrual hits NAV at ex-date `X`; the later payment-date cash activity does **not** change NAV (no double-count), only flips `status` to `paid` and records `reconciled_cash`. NAV between `X` and `P` already reflects the dividend.
3. **Multi-strategy same-symbol holding.** Two tenants holding the same symbol across one account-level DIV each accrue `g · own_entitled_qty`; neither sees the other's share; the two gross accruals sum to `g · (qty_A + qty_B)`.
4. **Offsetting long/short book does not zero out (finding 2).** Tenant A long `N` and tenant B short `N` of the same symbol across one DIV (venue nets flat): A accrues `+g·N`, B accrues `−g·N`, both nonzero and correct; no divide-by-near-zero, and the event is NOT mislabeled non-attributable despite the zero net total. Asserts attribution is independent per strategy, not divided by a signed net total.
5. **Short-position debit sign.** A short across the ex-date is **debited** (`gross_amount < 0`), lowering NAV — matching `adj_close` (a short's total return pays the dividend).
6. **Entitlement-cutoff boundary (finding 3).** For an ex-date `X` with a pinned `entitlement_cutoff` (last `America/NY` session close strictly before `X`, as a UTC instant): a buy filled **just before** the cutoff is entitled (accrues); a buy filled **exactly at** the cutoff instant is entitled (`fill_ts <= cutoff`); a buy filled **just after** the cutoff / on the ex-date session is **not** entitled (no accrual). A sell just before the cutoff removes entitlement. Table-driven over the three boundary offsets; asserts a DST-affected ex-date resolves the close correctly in the single canonical timezone.
7. **Gross sourced from declaration, fail-closed on net (finding 1).** A DIV whose corporate-action metadata supplies a declared `gross_per_share` accrues `g·qty`; a DIV with **no resolvable declared gross** (or missing `ex_date`/`corp_action_id`) produces **no** accrual and routes to suspense — the accrual is NEVER derived from the net payment cash. Assert that given a net cash strictly below `g·qty`, NAV still reflects the **gross** `g·qty`, never the net-implied per-share.
8. **Rejection of non-attributable / non-whitelisted cash.** A deposit / withdrawal / journal / fee / interest activity and a DIV on a symbol with zero ledger entitled qty produce **no** NAV change (routed to suspense); an unknown activity type is fail-closed (no NAV change).
9. **Withholding residual + reconciliation key.** When net payment cash < gross accrual, NAV keeps the **gross** accrual and the residual is surfaced in the reconciliation report, not deducted from NAV. The payment activity is matched to its open accrual by **`corp_action_id`** (not a fuzzy date window): a mismatched/absent `corp_action_id` leaves the accrual open and surfaces an unmatched-payment discrepancy.
10. **Idempotency + correction upsert.** Re-ingesting the same DIV window does not double-accrue and does not bump `updated_ts` (upsert `WHERE` predicate is false). A **corrected declaration on a still-open (`accrued`) row** overwrites `gross_per_share`/`accrued_qty`/`gross_amount` and bumps `updated_ts` (NAV self-heals). A **corrected declaration arriving after the row is `paid`** does NOT rewrite the settled row; it surfaces a post-payment restatement discrepancy instead.

### 6. Rollout policy for the retroactive NAV step-change (resolves finding 6)

The fix raises NAV for every dividend-paying long. Applied retroactively to a running book it would step NAV up (and, if peaks ratchet, could **mask** a subsequent real drawdown) — or, if peaks were rebased down, could **trip** one. Chosen policy: **prospective-only accrual from a per-strategy cutover, plus an accounting-basis stamp; no peak rebase.**

- **Prospective cutover — gated on the entitlement INSTANT, not a date-vs-timestamp compare.** `cutover_ts` is the per-strategy deploy **instant** (UTC), persisted on first tick under the new basis. The accrual filter is `entitlement_cutoff(X) > cutover_ts` — **instant vs instant** (both UTC; `entitlement_cutoff` is the pinned §1 last-session-close instant). We deliberately do NOT compare the calendar `ex_date` against a timestamp: that is a type/semantics mismatch, and it would wrongly accrue a dividend whose entitlement crystallized **before** the cutover but whose ex-date session merely happens to fall on/after the deploy day (e.g. deploy at 10:00 on an ex-date whose entitlement cutoff was the prior session close — pre-cutover). Gating on `entitlement_cutoff > cutover_ts` guarantees the entire economic event (the moment you had to hold to be entitled) arose under the new basis, so there is **no retroactive NAV step** — nothing to rebase, no phantom trip/mask on deploy. Any dividend whose entitlement crystallized at or before the cutover stays excluded (it was also excluded from the peak that gated the running book, so the book stays internally consistent).
- **Accounting-basis version stamp + explicit cross-basis ineligibility.** NAV/certificates carry an `accounting_basis_version` (bumped to mark the dividend-accrual basis). Every forward certificate and paper-promote evidence window is annotated with the basis it was measured under. **Explicit rule (not merely "annotated"): evidence measured under an older `accounting_basis_version` than the strategy's current live basis is INELIGIBLE to satisfy any post-cutover parity/promotion comparison — the gate treats a basis mismatch as fail-closed (equivalent to no valid certificate), never silently comparing pre-cutover (old-basis) evidence against a post-cutover requirement.** An evidence window that **straddles** the cutover (mixed-basis observations) is likewise ineligible; only a window whose observations are entirely under the current basis qualifies. A strategy seeking a fresh certificate under the new basis therefore re-runs the paper→forward evidence window from the cutover forward (clean, single-basis evidence). The gate already requires fresh evidence, so the freshness machinery carries most of this; the added, explicit piece is the **basis-equality precondition** the gate must assert so a stale old-basis PASS can never be reused across the cutover.
- **No peak rebase.** Because accrual is prospective, `live_nav_peaks` / `strategy_peaks` need no migration — the peak simply begins ratcheting on the new (dividend-inclusive) NAV from the cutover, and the first post-cutover ex-date raises NAV and peak together (no drawdown artifact).
- **Explicitly rejected alternatives:** (a) retroactive back-credit since inception + peak rebase — larger blast radius, must touch peak tables, and any rebase error directly arms/masks the drawdown breaker; (b) apply retroactively but leave peaks — guarantees a mask or a trip on deploy. Prospective + basis-stamp is the minimal, safe path and keeps the drawdown breaker honest.

---

## Scope / non-goals

- **In scope:** dividend (ordinary cash) accrual into ledger NAV for both `LedgerKind.LIVE` and `LedgerKind.PAPER`; ex-date accrual + payment-date reconciliation; per-strategy derivation attribution; whitelist + suspense fail-closed; multi-tenant paper split; prospective rollout + basis stamp; the parity + edge-case tests.
- **Out of scope / deferred:** withholding-tax modeling *inside* NAV (deliberately excluded per Decision 0 — it belongs in a reconciliation report, and the backtest doesn't model it either); interest/borrow-cost accrual; a first-class `cash_suspense` table (the existing `*_activities` table is the suspense ledger; a richer report is a follow-up); splits/other corporate actions through the ledger (already handled by the mark grid, not cash).
- **CODEOWNERS:** the bulk of this change targets **non-protected** files — `algua/execution/live_ledger.py`, `algua/execution/live_sizing.py`, `algua/execution/order_state.py`, `algua/registry/db.py` (schema), `algua/cli/live_cmd.py`, `algua/cli/paper_cmd.py`, and the paper-venue broker. It deliberately does **not** touch `backtest/engine.py`: the fix moves the *live/paper* basis onto the backtest's existing `adj_close` total-return basis, not the reverse. **One caveat:** the §6 basis-equality precondition (reject old-/mixed-basis evidence at the gate) is likely to touch a CODEOWNERS-protected file — `algua/research/forward_gates.py` and/or `algua/registry/forward_promotion.py` (and possibly `algua/registry/live_gate.py`). Task 8 isolates that gate edit so the reviewer can see it; **if the implemented diff touches any protected path the PR must stay OPEN for human merge** (no auto-merge), per the operating rule. If the basis-equality check can be satisfied purely by stamping the certificate and reading the stamp in an already-non-protected read path, the auto-merge path is preserved — the spike in task 8 determines which.

---

## Task list

Ordered; each task is independently testable. FAST per-task check during Implement
(`uv run ruff check . && uv run mypy algua && uv run lint-imports && uv run pytest -q <this task's tests>`);
the FULL gate runs only at integration and after any GATE-2 fix.

1. **Spike — declaration feeds (live + paper).** Confirm (a) that Alpaca exposes a corporate-actions **announcements** feed carrying declared gross cash-per-share + ex/payment dates + a stable id **before/at ex-date** (the trigger-1 source), reachable from the execution lane without importing `algua.data`; and (b) whether the shared paper venue broker emits ex-date **declaration** events today (grep the paper-venue/SimBroker adapter + `_ingest_paper_venue`). Output: a one-paragraph finding — does the live announcements feed give gross-per-share + ex-date, and does the paper venue emit declarations (→ task 6 wires ingestion) or not (→ task 6 adds the synthetic declaration emitter). No code. Gates the shape of tasks 5 and 6.
2. **Schema — `dividend_accruals` table + `accounting_basis_version`.** Add the table (columns per §2: incl. `corp_action_id`, `entitlement_cutoff`, `gross_per_share`, `updated_ts`) and both indexes to `db.py` (`SCHEMA_VERSION 35 -> 36`; migration note in the version comment block). Add the `accounting_basis_version` constant/stamp surface. Tests: fresh-DB schema present; `PRAGMA user_version == 36`; UNIQUE constraint rejects a duplicate `(lane, strategy, symbol, ex_date)`; the correction-aware upsert (§2) overwrites an `accrued` row on a changed declared figure, is a no-op on an identical re-pull, and does NOT rewrite a `paid` row.
3. **Pure derivation + entitlement-cutoff helpers (`live_ledger.py`).** `resolve_entitlement_cutoff(ex_date) -> utc_instant` (last exchange-calendar session close strictly before `ex_date`, `America/NY` → UTC, DST-aware; uses a calendar available to the execution lane — NOT `algua.data`); `entitled_qty_at(conn, strategy, symbol, kind, cutoff)` (fills with `fill_ts <= cutoff`); `total_ledger_qty_at(conn, symbol, kind, cutoff)` (reconciliation only); and pure `attribute_dividend(gross_per_share, per_strategy_qtys) -> {strategy: gross_amount}` — attributes each strategy **independently** as `gross_per_share * qty` (signed; long credit / short debit), with **no shared divisor**. Side-effect-free. Tests: long/short sign; multi-tenant independent split; **offsetting long/short book (nets flat) stays fully attributed, no divide-by-zero, not mislabeled non-attributable**; cutoff resolution incl. a DST-boundary ex-date; entitlement boundary just-before / exactly-at (`<=`) / just-after the cutoff instant.
4. **Whitelist + suspense classification (`live_ledger.py`).** `NAV_ELIGIBLE_ACTIVITY_TYPES = {"DIV"}`; a classifier mapping an activity to `eligible-dividend | suspense`. A DIV is eligible ONLY if it is ordinary AND its **declared `gross_per_share`, `ex_date`, and `corp_action_id` all resolve from corporate-action metadata**; otherwise → suspense (special/ROC/`gross_per_share >= prior_close` → suspense; unknown type → suspense; deposits/withdrawals/journals/fees/interest → suspense; missing declared gross → suspense — never net-derived). Tests: each type routes correctly; unknown fails closed; a DIV with no declared gross fails closed to suspense.
5. **Accrual write path — declaration-feed poll (trigger 1, NOT payment-activity ingestion).** A per-cycle poll of the corporate-action **announcements** feed (task 1): for each declaration whose `ex_date` has passed since the last poll and whose `entitlement_cutoff(X) > cutover_ts` (instant vs instant — task 8), resolve `(corp_action_id, gross_per_share, ex_date, payment_date)`, compute `entitlement_cutoff` (task 3), derive per-strategy accruals (task 3), and **correction-aware-upsert** (§2 `ON CONFLICT DO UPDATE ... WHERE status='accrued'`, NOT `INSERT OR IGNORE`) into `dividend_accruals`. This writes the accrual **at/just-after ex-date, independent of the payment date** — `_ingest_one_activity` does NOT write accruals (it only reconciles, task 7). A declaration with no resolvable declared gross/ex_date/corp_action_id → suspense (fail-closed). Idempotent on re-poll; a corrected open accrual self-heals. Tests: accrual written from a declaration whose ex-date passed while cash is still unpaid (NAV correct across `X..P`); cutover-gated; no-declared-gross → suspense; correction updates an open row; identical re-poll is a no-op.
6. **Paper-venue dividend events (depends on task 1).** If the venue does not emit DIV: add a synthetic per-ex-date DIV emitter carrying a stable `corp_action_id`, declared `gross_per_share`, and `ex_date` (from the same corporate-action data the sim applies), shaped identically to the live DIV declaration so ONE ingestion path (task 5) serves both lanes; if it already emits, wire `_ingest_paper_venue` through the same accrual path. Tests: a paper ex-date produces per-tenant independent accruals via the shared path.
7. **Payment-date reconciliation.** On the payment-date DIV cash activity, match the open `accrued` rows **strictly by `corp_action_id`** (the stable key — never a lane/symbol/ex-date window). A payment whose `corp_action_id` is absent or matches no open accrual is an **unmatched-payment discrepancy** surfaced to the reconciliation report (fail-closed) — NOT fuzzy-matched or back-filled into an accrual from net cash. On a match, set `status='paid'`, record `reconciled_cash` (attributed share of net cash), compute the withholding residual and the orphan (account-vs-ledger qty) residual — WITHOUT re-crediting NAV. A post-payment corrected declaration is a restatement discrepancy (task 2 upsert guard), not a silent NAV edit. Tests: payment does not change NAV; matched strictly by `corp_action_id`; missing/unknown id → unmatched-payment discrepancy (no fuzzy match, no accrual back-fill); status/residual recorded; residual surfaced not deducted; post-payment-restatement discrepancy surfaced.
8. **Prospective rollout + basis stamp + cross-basis gate.** Persist per-strategy `cutover_ts` (UTC instant) on first tick under the new basis; accrual (task 5/6) filters `entitlement_cutoff(X) > cutover_ts` (instant vs instant — never a calendar `ex_date` vs a timestamp). Stamp `accounting_basis_version` on the NAV snapshot / forward certificate. **Enforce the §6 basis-equality precondition:** the forward/promotion gate rejects (fail-closed) any certificate or evidence window whose `accounting_basis_version` is older than the strategy's current live basis, or that straddles the cutover — pre-cutover / mixed-basis evidence can NEVER satisfy a post-cutover comparison. **First determine (spike) whether this can be enforced by reading the stamp in a non-protected read path or requires editing `forward_gates.py`/`forward_promotion.py`/`live_gate.py` (CODEOWNERS — then the PR stays OPEN for human merge).** Tests: a dividend whose `entitlement_cutoff <= cutover_ts` is NOT accrued (incl. the boundary case where `ex_date` falls on/after the deploy day but its entitlement cutoff was the pre-cutover prior-session close); a dividend whose `entitlement_cutoff > cutover_ts` is accrued; certificate carries the basis version; an old-basis and a straddling evidence window are both rejected by the gate; no peak-table migration required.
9. **NAV wiring — `accrued_dividends` term into `build_sizing_snapshot` + `strategy_nav`.** Add `accrued_dividends(conn, strategy, lane, as_of) -> float` and fold it into the NAV total in `live_sizing.build_sizing_snapshot` and `live_ledger.strategy_nav`; thread the `as_of` mark timestamp from the tick. Tests: NAV includes ex-date accrual for long/short; unchanged when no accruals.
10. **Parity + edge-case test suite (§5).** The basis-parity assertion (ledger NAV growth across an ex-date == backtest `adj_close` total-return growth, gross basis) plus items 2-10 of §5 (incl. offsetting long/short non-zeroing, the entitlement-cutoff boundary table, gross-not-net sourcing, and the reconciliation-by-`corp_action_id` / correction cases). This is the acceptance test the issue demanded, pinned to the gross definition.
11. **Integration — FULL gate + reconciliation report surface.** Run the full gate; add/extend the operator reconciliation view (suspense cash + withholding residual + non-attributable residual) so the excluded cash is visible without a NAV path. Confirm `fleet_health` / `paper show` NAV read paths pick up the new term consistently.
