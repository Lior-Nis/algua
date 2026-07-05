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

**Rationale.** The north star is backtest↔live parity: the frictionless backtest gates candidate promotion and the forward certificate compares live ticks to that backtest. The backtest equity is gross total return (no withholding modeled). If live NAV credited only the net-of-withholding cash, we would trade one parity break (dividends excluded) for the opposite one (dividends under-credited by the tax haircut). Defining NAV on the backtest's own basis makes the sizing denominator and drawdown basis directly comparable to the qualifying backtest. The real withholding cash cost is genuine but belongs in a reconciliation/attribution report, not in the parity-critical NAV.

Everything below follows from Decision 0.

---

## Design

### 1. Strategy attribution for non-fill cash — derivation rule, NOT a `strategy_id` column (resolves finding 1)

A dividend cannot carry a `strategy_id` the way `paper_venue_orders` does: the broker account is a **netted custodian** (`live_ledger.py` module docstring — "The broker account is the netted custodian; this ledger is the source of truth for per-strategy attribution"). The broker emits **one account-level** DIV activity per symbol per ex-date; it has no idea which virtual sub-strategy holds those shares. So attribution must be **derived** from the ledger, exactly as position attribution already is.

**Attribution rule.** For a dividend on `symbol` with ex-date `X`:

```
per_share            = account_dividend_cash / total_ledger_qty_at(X, symbol)
strategy_accrual(s)  = per_share * believed_qty_at(X, s, symbol)      # signed
```

- `believed_qty_at(X, s, symbol)` is the strategy's signed position **as of the ex-date**, reconstructed from the append-only fills ledger: `Σ fills.qty WHERE strategy = s AND symbol = ? AND fill_ts <= X`. (The fills table is append-only with `fill_ts`; historical positions are exactly reconstructable — this is the same ledger `believed_positions` sums without the time filter.)
- `total_ledger_qty_at(X, symbol) = Σ_s believed_qty_at(X, s, symbol)` over all strategies of that `LedgerKind`. This is the sum the account-level cash is spread across.
- **Sign:** a long (`qty > 0`) is **credited** (+), a short (`qty < 0`) is **debited** (−) — a short borrower owes the dividend. `strategy_accrual` inherits the sign of `believed_qty_at` automatically.
- When the broker DIV activity already carries a trustworthy `per_share_amount`, prefer it and use `total_ledger_qty_at` only to **reconcile** (`per_share * total_ledger_qty ≈ account_cash`); a mismatch beyond tolerance routes the whole event to suspense (fail-closed) rather than silently mis-crediting.

**Why derivation, not a column.** A `strategy_id` column on `live_activities`/`paper_venue_activities` would have to be populated by the same ledger derivation anyway (the broker does not supply it), so the column adds a denormalized copy with no new information and a new corruption surface. The derivation reads the one source of truth (fills) directly.

**Non-attributable residual.** If `total_ledger_qty_at(X, symbol) == 0` (the account holds shares the ledger does not attribute to any strategy — a manual/orphan holding) the dividend is **not attributable**: it is routed to suspense in full and never touches any strategy NAV. `Σ_s strategy_accrual(s)` covers only the ledger-attributed shares; any remainder vs `account_dividend_cash` is a suspense residual (surfaces manual holdings / attribution drift).

### 2. Ex-date accrual + payment-date reconciliation (resolves finding 2)

A dividend is modeled as an **ex-date receivable**, not payment-date cash. Two events, weeks apart:

1. **Ex-date `X`:** the entitlement exists and `adj_close` steps. We accrue the gross receivable into NAV **as of `X`**, independent of when the broker actually pays.
2. **Payment date `P`:** the broker's cash activity lands. It does **not** re-credit NAV (that would double-count the accrual). It **reconciles** against the open accrual: mark it paid, record the net cash received, surface `gross_accrued − net_cash` as the withholding/fee residual.

**New table `dividend_accruals`** (one `db.py` schema bump: `SCHEMA_VERSION 35 -> 36`; the only bump this change makes — see the operating rule on one bump in flight):

```sql
CREATE TABLE IF NOT EXISTS dividend_accruals (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    lane               TEXT NOT NULL,              -- 'live' | 'paper' (LedgerKind.value)
    strategy           TEXT NOT NULL,              -- attributed sub-strategy
    symbol             TEXT NOT NULL,
    ex_date            TEXT NOT NULL,              -- UTC-midnight session date (matches corpactions)
    per_share          REAL NOT NULL,
    accrued_qty        REAL NOT NULL,              -- believed_qty_at(ex_date) (signed)
    gross_amount       REAL NOT NULL,              -- per_share * accrued_qty (signed; NAV input)
    payment_date       TEXT,                       -- broker payment date when known
    status             TEXT NOT NULL,              -- 'accrued' | 'paid'
    reconciled_cash    REAL,                       -- net cash matched at payment (attributed share)
    source_activity_id TEXT,                       -- broker activity that drove accrual/payment
    created_ts         TEXT NOT NULL,
    UNIQUE(lane, strategy, symbol, ex_date)        -- one accrual per strategy/symbol/ex-date
);
CREATE INDEX IF NOT EXISTS ix_dividend_accruals_lane_strategy
    ON dividend_accruals(lane, strategy);
```

**NAV formula change.** `build_sizing_snapshot` (`live_sizing.py`) and `strategy_nav` (`live_ledger.py`) gain one additive term:

```
NAV = allocation
    + Σ_symbol (pnl.realized + pnl.unrealized)          # unchanged, raw-close price return
    + accrued_dividends(conn, strategy, lane, as_of)    # NEW: Σ gross_amount WHERE ex_date <= as_of
```

`accrued_dividends` sums `gross_amount` for the strategy's accruals with `ex_date <= as_of` (both `accrued` and `paid` rows — once accrued it stays in NAV; the payment-date reconciliation changes `status`/`reconciled_cash`, not the NAV contribution). This is the whole parity fix: NAV now carries the same ex-date-timed total-return dividend contribution as `adj_close`.

**Ex-date data source (the seam).** The accrual needs an ex-date-timed dividend event `(symbol, ex_date, per_share | account_cash, payment_date)`. The live/execution lane is import-walled from `algua.data`, so the source is the **broker adapter**, not the research corpactions module:

- **Live:** Alpaca's account activities carry the DIV cash at payment date and the corporate-action metadata carries the ex-date. The activity ingestion (`_ingest_one_activity`) already parses non-fill activities; we extend it to recognize a whitelisted DIV, extract `ex_date` from the activity payload, and drive the accrual. **Fail-closed:** a DIV activity with no resolvable ex-date is routed to suspense (we never guess ex-date = payment date; that would reintroduce the timing break).
- **Paper venue:** the shared paper venue must emit dividend events for the accrual to fire. If the current paper venue broker does not synthesize DIV activities (it fills on a price grid only), that is a **known gap** this issue must close for paper parity: the paper venue emits a synthetic DIV activity per ex-date from the same corporate-action data the sim already applies, shaped identically to the live DIV activity so ONE ingestion/accrual path serves both lanes. (See Task list — a spike task confirms whether the paper venue emits DIV today and scopes the synthesizer.)

Accrual insertion is **idempotent** (`UNIQUE(lane, strategy, symbol, ex_date)` + `INSERT OR IGNORE`) so an overlap re-pull of the activity window never double-accrues — mirroring the existing `activity_id`-dedup discipline in `ingest_activities`.

### 3. Whitelist of NAV-eligible activity types; everything else fails closed to suspense (resolves finding 3)

Only **ordinary cash dividends** are eligible to touch NAV. A hardcoded allowlist gates it:

```python
NAV_ELIGIBLE_ACTIVITY_TYPES = frozenset({"DIV"})   # ordinary cash dividend only
```

- **Interest (`INT`)** is deliberately **excluded**: interest on a cash balance is an account-level artifact, not a total-return component of any held asset, and it is not in `adj_close`. Crediting it would create a fresh parity break. (Documented as an explicit non-inclusion, not an oversight.)
- **Special / return-of-capital / liquidating distributions** are excluded to stay symmetric with the backtest, which *rejects* them (`corpactions.py` raises when `cash >= p_prev`). A DIV whose `per_share >= prior close`, or a non-ordinary DIV subtype (e.g. `DIVROC`, `DIVCGL`, `DIVCGS`, `DIVNRA`), routes to suspense.
- **All other cash** — deposits (`CSD`), withdrawals (`CSW`), journals (`JNLC`/`JNLS`), transfers (`ACATC`/`ACATS`), fees (`FEE`), and any **unrecognized/unknown type** — routes to suspense and can never reach sizing or drawdown. This is the fail-closed default: an activity type must be *explicitly* whitelisted to affect NAV.

**Suspense.** Non-eligible cash is recorded (as today) in the `*_activities` table, which is **already excluded from NAV** (`position_pnl` / `accrued_dividends` never read it). No behavioral leak exists today; the design's job is to *keep* it that way when we start reading dividend rows — i.e. the accrual reads ONLY whitelisted DIV events, never the raw `*_activities` sum. A `cash_suspense` view/report over the non-eligible `*_activities` rows (plus the withholding residual from §2 and the non-attributable residual from §1) gives the operator visibility without any NAV path. No new suspense table is required — the existing `*_activities` table is the suspense ledger; the guarantee is that NAV's dividend term is sourced from `dividend_accruals` (whitelisted, attributed, ex-date-accrued), not from a raw activities sum.

### 4. Paper multi-tenancy — per-`strategy` accrual in the shared venue (resolves finding 4)

The shared paper venue is a single account with multiple tenant strategies; a `LedgerKind.PAPER` DIV for `AAPL` is **one account-level event** even when three tenants hold `AAPL`. The §1 derivation rule handles this natively: the account cash is split by `believed_qty_at(X, s, symbol) / total_ledger_qty_at(X, symbol)`, producing one `dividend_accruals` row **per tenant strategy**, summing back to the account cash across the attributed shares. There is **no** single cumulative total applied to every tenant. `accrued_dividends(conn, strategy, 'paper', as_of)` reads only that strategy's rows, so each tenant's NAV, sizing, and drawdown see only their own dividend share. This is the paper analog of how `believed_positions` already scopes fills per `strategy` in the shared `paper_venue_fills` table.

### 5. Parity test asserting Decision 0, plus the edge cases (resolves finding 5)

Tests assert the NAV basis is **gross accrued total return** and that it matches the backtest's `adj_close` basis:

1. **Basis parity (the core assertion).** A single long held across an ex-date: the ledger NAV growth from `T-1` to ex-date `X` (raw-close price return **+** the gross dividend accrual) equals the backtest total-return equity growth on the same `adj_close` bars over the same interval, within tolerance. This is the assertion the issue asked for, pinned to the **gross** definition.
2. **Ex-date vs payment-date separation.** Accrual hits NAV at ex-date `X`; the later payment-date cash activity does **not** change NAV (no double-count), only flips `status` to `paid` and records `reconciled_cash`. NAV between `X` and `P` already reflects the dividend.
3. **Multi-strategy same-symbol holding.** Two tenants holding the same symbol across one account-level DIV each accrue their `qty`-proportional share; the two accruals sum to the account cash; neither sees the other's share.
4. **Short-position debit sign.** A short across the ex-date is **debited** (`gross_amount < 0`), lowering NAV — matching `adj_close` (a short's total return pays the dividend).
5. **Rejection of non-attributable / non-whitelisted cash.** A deposit / withdrawal / journal / fee / interest activity and a DIV on a symbol with zero ledger qty produce **no** NAV change (routed to suspense); an unknown activity type is fail-closed (no NAV change).
6. **Withholding residual.** When net payment cash < gross accrual, NAV keeps the **gross** accrual and the residual is surfaced in the reconciliation report, not deducted from NAV.
7. **Idempotency.** Re-ingesting the same DIV activity window does not double-accrue (`UNIQUE(lane, strategy, symbol, ex_date)`).

### 6. Rollout policy for the retroactive NAV step-change (resolves finding 6)

The fix raises NAV for every dividend-paying long. Applied retroactively to a running book it would step NAV up (and, if peaks ratchet, could **mask** a subsequent real drawdown) — or, if peaks were rebased down, could **trip** one. Chosen policy: **prospective-only accrual from a per-strategy cutover, plus an accounting-basis stamp; no peak rebase.**

- **Prospective cutover.** Accrue only dividends whose `ex_date >= cutover_ts`, where `cutover_ts` is the per-strategy deploy timestamp persisted on first tick under the new basis. No historical back-credit → **no retroactive NAV step** → nothing to rebase and no phantom trip/mask on deploy. Dividends already paid under the old basis stay excluded (they were also excluded from the peak that gated the running book, so the book stays internally consistent).
- **Accounting-basis version stamp.** NAV/certificates carry an `accounting_basis_version` (bump to mark the dividend-accrual basis). The forward certificate minted under the old basis is comparable only within its own basis; a certificate is annotated with the basis it was measured under so pre/post are never silently compared. A strategy seeking a fresh certificate under the new basis re-runs the paper→forward evidence window from the cutover forward (clean, single-basis evidence) — the gate already requires fresh evidence, so this is a natural consequence, not a new mechanism.
- **No peak rebase.** Because accrual is prospective, `live_nav_peaks` / `strategy_peaks` need no migration — the peak simply begins ratcheting on the new (dividend-inclusive) NAV from the cutover, and the first post-cutover ex-date raises NAV and peak together (no drawdown artifact).
- **Explicitly rejected alternatives:** (a) retroactive back-credit since inception + peak rebase — larger blast radius, must touch peak tables, and any rebase error directly arms/masks the drawdown breaker; (b) apply retroactively but leave peaks — guarantees a mask or a trip on deploy. Prospective + basis-stamp is the minimal, safe path and keeps the drawdown breaker honest.

---

## Scope / non-goals

- **In scope:** dividend (ordinary cash) accrual into ledger NAV for both `LedgerKind.LIVE` and `LedgerKind.PAPER`; ex-date accrual + payment-date reconciliation; per-strategy derivation attribution; whitelist + suspense fail-closed; multi-tenant paper split; prospective rollout + basis stamp; the parity + edge-case tests.
- **Out of scope / deferred:** withholding-tax modeling *inside* NAV (deliberately excluded per Decision 0 — it belongs in a reconciliation report, and the backtest doesn't model it either); interest/borrow-cost accrual; a first-class `cash_suspense` table (the existing `*_activities` table is the suspense ledger; a richer report is a follow-up); splits/other corporate actions through the ledger (already handled by the mark grid, not cash).
- **CODEOWNERS:** this change targets `algua/execution/live_ledger.py`, `algua/execution/live_sizing.py`, `algua/execution/order_state.py`, `algua/registry/db.py` (schema), `algua/cli/live_cmd.py`, `algua/cli/paper_cmd.py`, and the paper-venue broker. **None are on the CODEOWNERS list** (that list is registry/store, contracts/lifecycle, backtest/engine, research/gates, research/clustering, approvers, registry/live_gate, registry/transitions, registry/promotion, research/forward_gates, registry/forward_promotion) — so the PR can auto-merge on green CI. It deliberately does **not** touch `backtest/engine.py`: the fix moves the *live/paper* basis onto the backtest's existing `adj_close` total-return basis, not the reverse.

---

## Task list

Ordered; each task is independently testable. FAST per-task check during Implement
(`uv run ruff check . && uv run mypy algua && uv run lint-imports && uv run pytest -q <this task's tests>`);
the FULL gate runs only at integration and after any GATE-2 fix.

1. **Spike — paper-venue dividend emission.** Confirm whether the shared paper venue broker emits DIV activities today (grep the paper-venue/SimBroker adapter + `_ingest_paper_venue`). Output: a one-paragraph finding that either (a) it already emits DIV → §2 paper path needs only ingestion, or (b) it does not → add the synthetic-DIV emitter task. No code. Gates the shape of task 6.
2. **Schema — `dividend_accruals` table + `accounting_basis_version`.** Add the table and index to `db.py` (`SCHEMA_VERSION 35 -> 36`; migration note in the version comment block). Add the `accounting_basis_version` constant/stamp surface. Tests: fresh-DB schema present; `PRAGMA user_version == 36`; UNIQUE constraint rejects a duplicate `(lane, strategy, symbol, ex_date)`.
3. **Pure derivation helpers (`live_ledger.py`).** `believed_qty_at(conn, strategy, symbol, kind, as_of)` (fills with `fill_ts <= as_of`), `total_ledger_qty_at(conn, symbol, kind, as_of)`, and pure `attribute_dividend(account_cash | per_share, per_strategy_qtys) -> {strategy: gross_amount}` (signed; long credit / short debit; zero-total → non-attributable). Side-effect-free. Tests: long/short sign, multi-tenant split summing to account cash, zero-qty non-attributable, per_share-vs-account_cash reconcile tolerance.
4. **Whitelist + suspense classification (`live_ledger.py`).** `NAV_ELIGIBLE_ACTIVITY_TYPES = {"DIV"}`; a classifier that maps an activity to `eligible-dividend | suspense` (special/ROC/`per_share >= prior_close` → suspense; unknown type → suspense; deposits/withdrawals/journals/fees/interest → suspense). Tests: each type routes correctly; unknown fails closed to suspense.
5. **Accrual write path — live ingestion (`live_ledger._ingest_one_activity` / `ingest_activities`).** On a whitelisted DIV with a resolvable ex-date: derive per-strategy accruals (task 3) and `INSERT OR IGNORE` into `dividend_accruals` (ex-date accrual), respecting the prospective `cutover_ts` (task 8). A DIV with no resolvable ex-date → suspense (fail-closed). Idempotent on window re-pull. Tests: accrual rows created once, ex-date-timed, cutover-gated, no-ex-date → suspense.
6. **Paper-venue dividend events (depends on task 1).** If the venue does not emit DIV: add a synthetic per-ex-date DIV emitter shaped identically to the live DIV activity so ONE ingestion path (task 5) serves both lanes; if it already emits, wire `_ingest_paper_venue` through the same accrual path. Tests: a paper ex-date produces per-tenant accruals via the shared path.
7. **Payment-date reconciliation.** On the payment-date DIV cash activity, match open `accrued` rows (by lane/symbol/ex-date window), set `status='paid'`, record `reconciled_cash` (attributed share of net cash), compute the withholding residual — WITHOUT re-crediting NAV. Tests: payment does not change NAV; status/residual recorded; residual surfaced not deducted.
8. **Prospective rollout + basis stamp.** Persist per-strategy `cutover_ts` on first tick under the new basis; accrual (task 5/6) filters `ex_date >= cutover_ts`. Stamp `accounting_basis_version` on the NAV snapshot / forward certificate; annotate certificates with the basis measured under. Tests: pre-cutover ex-date not accrued; post-cutover accrued; certificate carries the basis version; no peak-table migration required.
9. **NAV wiring — `accrued_dividends` term into `build_sizing_snapshot` + `strategy_nav`.** Add `accrued_dividends(conn, strategy, lane, as_of) -> float` and fold it into the NAV total in `live_sizing.build_sizing_snapshot` and `live_ledger.strategy_nav`; thread the `as_of` mark timestamp from the tick. Tests: NAV includes ex-date accrual for long/short; unchanged when no accruals.
10. **Parity + edge-case test suite (§5).** The basis-parity assertion (ledger NAV growth across an ex-date == backtest `adj_close` total-return growth, gross basis) plus items 2-7 of §5. This is the acceptance test the issue demanded, pinned to the gross definition.
11. **Integration — FULL gate + reconciliation report surface.** Run the full gate; add/extend the operator reconciliation view (suspense cash + withholding residual + non-attributable residual) so the excluded cash is visible without a NAV path. Confirm `fleet_health` / `paper show` NAV read paths pick up the new term consistently.
