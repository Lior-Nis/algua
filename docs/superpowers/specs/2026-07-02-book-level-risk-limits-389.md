# Book-Level Aggregate Risk Limits (#389)

## Problem

Every risk wall in algua is enforced **per-strategy** against that strategy's own target
weights, sized against its own virtual subaccount (`equity = min(allocation, NAV)`).
`validate_decision_weights` (`algua/risk/limits.py`) applies gross / concentration / short /
drawdown checks to a single strategy's weights only. The one account-level constraint,
`allocations.py`, caps `Σ(active capital) ≤ account_equity` — it bounds capital *committed*, not
*exposure* or *concentration*.

In the **shared** brokerage account the live `run-all` loop trades, per-strategy positions
**compound**: two strategies each holding 100% of their subaccount in AAPL (each individually
passing `max_weight_per_symbol`) leave the real account at 2× the intended single-name
concentration; N momentum strategies all long the same factor leave the book with unbounded
aggregate gross / net exposure. `run_all` (`algua/cli/live_cmd.py`) loops strategies with **no**
portfolio-level pre-trade aggregation. This is the canonical safe-scaling failure mode.

## Goal

Add a **book-level** pre-trade aggregation layer, enforced across ALL strategies sharing the
account at order-submit time in the live `run-all` loop, capping:

- **aggregate gross** exposure (`Σ|position notional| ≤ max_gross × equity`),
- **aggregate net** exposure (`|Σ position notional| ≤ max_net × equity`),
- **single-name concentration**, on two bases: fraction of the gross book
  (`|symbol| / gross ≤ max_symbol_concentration`) AND fraction of equity
  (`|symbol| ≤ max_symbol_notional × equity`).

Fail closed. Configurable with conservative defaults.

## Design

### Intercept point (KISS — reuse the existing reserve hook)

`run_all` already threads a **buying-power reservation** through every BUY: it builds a shared
closure `pool = {"available": broker_buying_power}` and `_reserve_for(name)` returns a
`reserve(symbol, notional) -> permitted_notional` closure passed into
`_run_strategy_tick → run_tick → broker.submit_sized(..., reserve=reserve_buy)`.
`submit_sized` calls `reserve` **only for BUY orders**, trims `amount = min(amount, permitted)`,
and skips on `permitted <= 0` or a trim below `MIN_NOTIONAL`.

The book-level layer composes **into this same hook** — no new hook, no change to `run_tick`,
`submit_sized`, or `live_loop.py`. `reserve(symbol, n)` becomes
`min(pool_permitted, book.permit_buy(symbol, min(n, pool_permitted)))` and audits book trims
alongside pool trims via `record_reservation`.

### Long-only precondition (the safety anchor)

algua's default execution contract is **long-only** (`ExecutionContract.allow_short=False`;
`check_short_policy` hard-breaches any negative target weight per strategy). The book layer makes
long-only an **explicit precondition** and **fails closed** if the reconciled account book
violates it. Under a guaranteed long-only book every quantity below is **exact and
monotone-increasing** in the buy notional, so the correctness proof is trivial and there is no
under-restriction. A short in the live account is an anomaly that must **stop** trading, not be
traded through with signed-book math (which is where the earlier design had a short-cover
concentration under-restriction bug — GATE-1).

### Seed (whole-account truth, after clean reconcile)

`run_all` seeds the book **after** `ingest_activities` + reconcile pass **clean** (the loop
already defers all trades when reconcile is not clean). Seed = `_broker_net_positions(broker)`
(the reconciled **broker** net positions `{symbol: signed qty}` — whole-account truth, including
non-ticked / dormant / orphan residuals) × latest marks.

Marks come from bars fetched once for the union of all reconciled-position symbols and all
verified strategy universes. Fail-closed guards at seed time (skip trading the whole cycle, emit
a note — never build a partially-valued book):

- any reconciled nonzero position with a **missing / non-finite / ≤ 0 mark** → fail closed;
- any reconciled position with **`qty < 0`** (a short) → fail closed (long-only precondition).

### `algua/risk/book_limits.py` (pure, no I/O)

```
@dataclass(frozen=True)
class BookRiskLimits:
    max_gross: float = 2.0                 # × equity
    max_net: float = 1.0                   # × equity
    max_symbol_concentration: float = 0.25 # fraction of gross book
    max_symbol_notional: float = 0.50      # × equity
    # __post_init__ validates: max_gross/max_net/max_symbol_notional >= 0,
    # 0 < max_symbol_concentration <= 1.

class BookExposure:
    """Sequential book accumulator. Seeded with equity + current signed book notionals
    (guaranteed long-only by the caller). permit_buy trims a BUY to the largest notional
    that keeps ALL caps satisfied AFTER the add, then MUTATES the book by that permitted
    amount so the next strategy's buys see the compounded book."""
    def __init__(self, equity, book_notionals, limits): ...
    def permit_buy(self, symbol, requested_notional) -> float: ...
```

For a long-only book (`symbol_before ≥ 0`, `net_before = gross_before ≥ 0`) and a BUY of
notional `p ≥ 0`:

- `symbol_after = symbol_before + p` (exact, ↑)
- `gross_after  = gross_before + p` (exact — no short to cover, ↑)
- `net_after    = net_before + p`   (exact, ↑)
- `concentration_after = (symbol_before + p)/(gross_before + p)` (↑, since `symbol_before ≤ gross`)

All monotone-increasing in `p`, so the max feasible `p` is the `min` of four independent
headrooms (each `max(0, …)`):

```
gross_headroom        : max_gross × equity − gross_before
net_headroom          : max_net   × equity − net_before
symbol_notional_hr    : max_symbol_notional × equity − symbol_before
concentration_hr      : (c·gross_before − symbol_before)/(1 − c)      # c = max_symbol_concentration, c<1
```

`permit_buy` returns `max(0, min(headrooms, requested_notional))` and, on a positive permit,
mutates `symbol += permitted`, `gross += permitted`, `net += permitted`.
`equity ≤ 0` → all headrooms 0 → deny (fail closed). `gross_before + p == 0` only when the symbol
is flat and `p == 0` → concentration vacuous.

**Already-breached book:** if the seed already breaches any cap, every headroom is ≤ 0 → all buys
denied this cycle. Under long-only, buys are strictly risk-increasing and SELLs (rebalance-down)
still flow, so the book heals. (The short-cover-to-de-risk exception cannot occur — guarded out.)

### Wiring in `run_all`

1. After clean reconcile, fetch marks for the union universe, build `BookExposure` with the
   conservative `BookRiskLimits` (from settings). On any fail-closed guard → emit a note, skip
   trading this cycle (like the reconcile-pending path).
2. `_reserve_for(name)` composes: `pool_permitted = min(notional, pool.available)`;
   `permitted = book.permit_buy(symbol, pool_permitted)`; decrement pool by `permitted`; audit a
   shortfall (`permitted < notional`) via `record_reservation`. Book mutates by the **final**
   permitted amount (after pool trim) — never by its own headroom.

Only touches `algua/cli/live_cmd.py` (run_all + `_reserve_for`), new `algua/risk/book_limits.py`,
`algua/config/settings.py` (4 conservative defaults, env-overridable via `ALGUA_BOOK_*`), and
tests. **No CODEOWNERS-protected file.**

## Non-goals / deferred (documented, not fixed here)

- **Concurrency / TOCTOU:** the book is in-memory per `run_all` cycle — the **same** property the
  pre-existing buying-power `pool` already has. Live `run_all` assumes a single runner per account
  (one broker, one account). Concurrent runners racing the account, and broker-asynchrony (a
  pre-existing open order filling after seed), are a pre-existing shared class, not introduced or
  resolved here.
- **SELLs bypass the hook.** Under the long-only precondition a SELL only reduces an existing long
  toward flat (never opens/increases a short), so ignoring SELLs cannot under-restrict — worst case
  it is slightly over-strict (does not free same-cycle budget). A future short-enabled contract
  must **replace** this module (signed-book math), not reuse it.
- **Per-factor / per-sector** book caps (issue "ideally") — deferred; gross/net/single-name first.
- **Paper `run-all`** is unmerged on `origin/main`; wiring is scoped to the live `run-all` driver.
