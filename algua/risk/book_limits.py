"""Book-level aggregate risk limits — a pure risk helper (no I/O, no DB, stdlib only).

Where `algua.risk.limits` caps a SINGLE strategy's decision-weight vector, this module caps
the BOOK: the aggregate exposure summed across every strategy/tenant on one account.

The public surface is a frozen `BookRiskLimits` config plus one pure function `evaluate_book`
that takes the WHOLE cycle at once — the reconciled per-symbol seed notionals, the aggregated
cross-strategy SELL and BUY intents — and returns a `BookVerdict`: whether to trade, and (on a
pass) the quantized per-(strategy, symbol) BUY notionals that keep every book-level cap satisfied
at EVERY reachable order-submission prefix, not merely at the terminal book (#389 design v4).

Prefix-safety crux (design v4, §v4-9). Concentration is enforced against a FIXED per-cycle
denominator `D = max(equity, ΣB)` where `B[s] = seed[s] − sell_total[s]` is the post-SELL lower
bound on the book (SELLs shrink gross; a SELL-reduced intermediate prefix can drive gross all the
way down to ΣB). Validating single-name concentration against `c·D` — NOT against the friendlier
terminal gross `ΣQ` — certifies `Q[s] ≤ c·max(equity, gross_I)` for every intermediate prefix `I`.
The equity floor in `D` is definitional, not a fudge: `symbol ≤ c·max(equity, gross)` measures a
name against total capital while the book is unlevered, so a SELL that shrinks *gross* can never
retroactively push a held name over the raw `symbol/gross` ratio (which the discarded raw-ratio
form would miss — Codex counterexample: equity=100, c=0.5, hold A=50 / B=50, sell all B leaves A
at 50/50 = 100% of gross but only 50% of equity).

Precondition: LONG-ONLY. All seed notionals, all SELL totals, and all BUY requests are >= 0; the
aggregate SELL in a name is bounded by its seed (no over-sell — a hard assert, step 0b), so
`B[s] >= 0` and no reachable intermediate book goes short.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

# Arithmetic dust ONLY (~1e-6 * equity): absorbs float round-off in the cap sums so a book exactly
# at a cap isn't spuriously flagged over it. It is NEVER used to reconcile a valuation basis
# difference — the caller values the entire book on ONE mark basis before calling (design v4,
# Finding 3), so `_EPS` here is pure round-off slack.
#
# ACCEPTED BOUND (#389 GATE-2, LOW): because the step-4 gross pass and the step-6 final validation
# permit up to `cap + eps`, the quantized OUTPUT book may sit at most `1e-6 * equity` over a cap
# (e.g. ~$1k over a $1e9 gross cap). This is accepted valuation dust — proportionally negligible and
# always below the dollar rounding step for any realistic account. `evaluate_book` is not yet wired
# into any driver (only the interim trimmer is), so this bound is dead today; when it is wired, if a
# strictly sub-cap output is ever required the permitting comparisons should switch to a fixed
# sub-cent absolute tolerance rather than this equity-scaled band.
_EPS_REL = 1e-6

# Over-sell (step 0b naked-short) tolerance: a FIXED tiny absolute notional, deliberately NOT the
# equity-scaled `_EPS_REL * equity` band. The over-sell assert underpins the B[s] = seed − sell >= 0
# lower bound the whole prefix-safety proof rests on, so a naked short must fail closed REGARDLESS
# of account size — a $500 over-sell on a $1e9 account is still a naked short, but an equity-scaled
# eps (~$1000 there) would wave it through. This absorbs ONLY float round-off in the seed−sell
# compare; it is dimensionally a raw notional, independent of equity.
_OVERSELL_ABS_EPS = 1e-6

# Default BUY notional quantization step (dollars) — BUYs are submitted as dollar-denominated
# notional orders, so the step is a DOLLAR step, not a share step (design v4, §v4-7).
DEFAULT_BUY_NOTIONAL_STEP = 0.01

# Small positive nudge so a float share that is mathematically an exact multiple of `step` (but
# lands microscopically under it after division) still floors to that multiple, not one step below.
_FLOOR_SLACK = 1e-9


def _finite(x: float) -> bool:
    """True iff x is a finite float (not NaN, not +/-inf)."""
    return math.isfinite(x)


@dataclass(frozen=True)
class BookRiskLimits:
    """Book-level caps, all expressed as multiples of account equity.

    - max_gross: cap on Σ|notional| / equity across the whole book.
    - max_net: cap on Σ notional / equity (signed).
    - max_symbol_concentration: cap on any single symbol's share of gross, i.e. sb/gross <= c.
      The denominator is FLOORED at equity: sb <= c * max(gross, equity). While the book is
      unlevered (gross <= equity) a single name is capped at c * equity (so a first fill into a
      flat/small book is not spuriously "100% of a tiny gross"); once the book levers past equity
      (gross > equity) the cap reverts to c * gross — the real diversification constraint on the
      deployed book. Must lie in (0, 1]; c == 1.0 disables the concentration cap.
    - max_symbol_notional: cap on any single symbol's notional / equity.
    """

    max_gross: float = 2.0
    max_net: float = 1.0
    max_symbol_concentration: float = 0.25
    max_symbol_notional: float = 0.50

    def __post_init__(self) -> None:
        for name in ("max_gross", "max_net", "max_symbol_concentration", "max_symbol_notional"):
            value = getattr(self, name)
            if not _finite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")
        if self.max_gross < 0:
            raise ValueError(f"max_gross must be >= 0, got {self.max_gross!r}")
        if self.max_net < 0:
            raise ValueError(f"max_net must be >= 0, got {self.max_net!r}")
        if self.max_symbol_notional < 0:
            raise ValueError(f"max_symbol_notional must be >= 0, got {self.max_symbol_notional!r}")
        if not (0 < self.max_symbol_concentration <= 1):
            raise ValueError(
                "max_symbol_concentration must satisfy 0 < c <= 1, "
                f"got {self.max_symbol_concentration!r}"
            )


@dataclass(frozen=True)
class BookVerdict:
    """Outcome of one whole-cycle book evaluation.

    - ok: True → trade this cycle (SELLs in full; BUYs per `permitted_buys`). False → de-risk only
      (permit SELLs, submit NO BUYs) — a hard fail-closed (degenerate equity, over-sell, or the
      final-net miss).
    - reason: a short machine token on a fail-closed or a global reduce-only (`"equity"`,
      `"oversell:<symbol>"`, `"seed_breach:<which>"`, `"final_net:<which>"`); None on a clean pass.
    - permitted_buys: the quantized BUY notionals to submit, keyed by (strategy, symbol). Empty on
      any fail-closed or global reduce-only. SELLs are ALWAYS permitted in full (not represented
      here) whenever `ok` is True.
    """

    ok: bool
    reason: str | None
    permitted_buys: dict[tuple[str, str], float] = field(default_factory=dict)


def _sorted_symbols(*maps: Mapping[str, float]) -> list[str]:
    """Deterministic union of symbol keys across the given maps."""
    seen: set[str] = set()
    for m in maps:
        seen.update(m.keys())
    return sorted(seen)


def _caps_ok(book: Mapping[str, float], cap_gross: float, cap_sym: float,
             conc_bound: float, eps: float) -> bool:
    """True iff `book` satisfies gross, per-symbol-notional, and (fixed-D) concentration within eps.

    `conc_bound` is `c·D` (the FIXED per-cycle denominator, or +inf when concentration is off)."""
    if sum(book.values()) > cap_gross + eps:
        return False
    for val in book.values():
        if val > cap_sym + eps or val > conc_bound + eps:
            return False
    return True


def evaluate_book(
    equity: float,
    seed_notionals: Mapping[str, float],
    sell_total: Mapping[str, float],
    buy_request: Mapping[tuple[str, str], float],
    limits: BookRiskLimits,
    *,
    buy_notional_step: Mapping[str, float] | float = DEFAULT_BUY_NOTIONAL_STEP,
    min_notional: float = 0.0,
) -> BookVerdict:
    """Evaluate the WHOLE cycle's aggregate book in one shot; return a `BookVerdict`.

    NOTIONAL-ONLY. Every input is a dollar notional valued on the caller's single mark basis; the
    share-quantity SELL bound and any notional<->share conversion live in the I/O layer, not here.

    Args:
      equity: account equity (already source/freshness/consistency-validated by the I/O layer).
      seed_notionals: reconciled per-symbol held notional across the whole account (>= 0).
      sell_total: aggregated cross-strategy SELL notional per symbol (>= 0).
      buy_request: aggregated per-(strategy, symbol) BUY notional request (>= 0), already
        pool-limited by the caller.
      limits: the book caps.
      buy_notional_step: DOLLAR quantization step for BUY legs — a per-symbol map or one scalar
        (default $0.01). BUY legs are floored (never rounded up) to this step.
      min_notional: a quantized BUY leg landing in (0, min_notional) is dropped to 0 (matches the
        downstream venue-minimum skip so PLAN and APPLY agree).

    Steps (design v4):
      0  degenerate equity → fail closed "equity".
      0b sell-bound HARD assert 0 <= sell_total[s] <= seed[s]+eps → over-sell fails "oversell:<s>".
      1  B[s] = seed − sell (>= 0 by 0b); D = max(equity, ΣB) — FIXED per-cycle constant.
      2  seed breach anywhere → GLOBAL reduce-only: P[s]=0 for every s, SELLs in full,
         ok=True reason "seed_breach:<which>", empty permitted_buys.
      3  per-symbol closed-form BUY bound P[s] = max(0, min(request, cap_sym−seed, c·D−seed)).
      4  gross cap in ONE proportional pass over P[s] > 0.
      5  rounding-safe NOTIONAL distribution across requesting strategies (floor to step; grant one
         step of remainder only if re-validation still passes; drop legs in (0, min_notional)).
      6  FINAL VALIDATION over the exact quantized q, concentration vs the FIXED D (not ΣQ);
         any miss → ok=False (SELLs only), reason "final_net:<which>".
    """
    # ---- step 0: degenerate equity ------------------------------------------------------------
    if not _finite(equity) or equity <= 0.0:
        return BookVerdict(ok=False, reason="equity", permitted_buys={})

    eps = _EPS_REL * equity
    symbols = _sorted_symbols(seed_notionals, sell_total)

    def _seed(s: str) -> float:
        return float(seed_notionals.get(s, 0.0))

    def _sell(s: str) -> float:
        return float(sell_total.get(s, 0.0))

    def _step(s: str) -> float:
        if isinstance(buy_notional_step, Mapping):
            step = float(buy_notional_step.get(s, DEFAULT_BUY_NOTIONAL_STEP))
        else:
            step = float(buy_notional_step)
        if not _finite(step) or step <= 0.0:
            return DEFAULT_BUY_NOTIONAL_STEP
        return step

    # ---- step 0b: sell-bound hard assert (no over-sell) ---------------------------------------
    # A negative sell or an aggregate sell exceeding the held notional is a caller bug / naked
    # short — fail closed, never silently clamp (clamping would hide the over-sell and break the
    # B[s] lower bound the prefix-proof rests on).
    for s in symbols:
        sell = _sell(s)
        seed = _seed(s)
        # FIXED absolute tolerance here (NOT the equity-scaled `eps`): a naked short must fail
        # closed regardless of account equity — an equity-scaled band would let a large account
        # over-sell a name by up to `_EPS_REL * equity` and silently break the B[s] >= 0 bound.
        if (not _finite(sell) or not _finite(seed)
                or sell < -_OVERSELL_ABS_EPS or sell > seed + _OVERSELL_ABS_EPS):
            return BookVerdict(ok=False, reason=f"oversell:{s}", permitted_buys={})

    # ---- step 1: denominator floor (FIXED per-cycle constant) ---------------------------------
    b = {s: max(0.0, _seed(s) - _sell(s)) for s in symbols}
    sum_b = sum(b.values())
    d = max(equity, sum_b)  # worst-case (minimum) gross denominator any prefix can reach

    lim = limits
    cap_gross = min(lim.max_gross, lim.max_net) * equity  # long-only ⇒ gross == net; take tighter
    cap_sym = lim.max_symbol_notional * equity
    c = lim.max_symbol_concentration
    conc_bound = math.inf if c >= 1.0 else c * d

    # ---- step 2: seed breach anywhere → GLOBAL reduce-only ------------------------------------
    sum_seed = sum(_seed(s) for s in symbols)
    seed_breach: str | None = None
    if sum_seed > cap_gross + eps:
        seed_breach = "gross"
    else:
        for s in symbols:
            if _seed(s) > cap_sym + eps:
                seed_breach = "symbol_notional"
                break
        if seed_breach is None and c < 1.0:
            for s in symbols:
                if _seed(s) > conc_bound + eps:
                    seed_breach = "concentration"
                    break
    if seed_breach is not None:
        # A book that breaches ANYWHERE at the account level must de-risk EVERYWHERE: no new
        # exposure to any name (even unbreached) once the aggregate is over a cap. SELLs still
        # permitted (they only shrink the book). Skip steps 3-6.
        return BookVerdict(ok=True, reason=f"seed_breach:{seed_breach}", permitted_buys={})

    # ---- step 3: per-symbol closed-form BUY bound (no iteration) -------------------------------
    # P[s] simultaneously clears the per-symbol-notional cap and the FIXED-denominator
    # concentration cap. Both `cap_sym − seed` and `c·D − seed` are fixed constants (D does not
    # move), so this is a single closed-form min — never a fixpoint loop.
    request_by_symbol: dict[str, float] = {}
    for (_strat, sym), req in buy_request.items():
        request_by_symbol[sym] = request_by_symbol.get(sym, 0.0) + max(0.0, float(req))
    p: dict[str, float] = {}
    for sym, req in request_by_symbol.items():
        seed = _seed(sym)
        bound = min(req, cap_sym - seed, conc_bound - seed)
        p[sym] = max(0.0, bound)

    # ---- step 4: gross cap — ONE proportional pass --------------------------------------------
    # Reducing P only relaxes the step-3 per-symbol / concentration bounds, so a single
    # proportional pass suffices (no iteration). Overflow is spread across P[s] > 0 in proportion
    # to P[s].
    sum_p = sum(p.values())
    overflow = sum_seed + sum_p - cap_gross
    if overflow > eps and sum_p > 0.0:
        scale = max(0.0, (sum_p - overflow) / sum_p)
        p = {sym: v * scale for sym, v in p.items()}

    # ---- step 5: rounding-safe NOTIONAL distribution (two-pass) --------------------------------
    # PASS 1 — base floors. Split each symbol's P across the requesting strategies proportional to
    # request and FLOOR every leg to the dollar step (never up → Σq ≤ P per symbol, so the floored
    # book satisfies every step-3/step-4 cap by construction).
    q: dict[tuple[str, str], float] = {}
    legs_by_symbol: dict[str, list[tuple[str, float]]] = {}
    granted_by_symbol: dict[str, dict[str, float]] = {}
    for sym in sorted(p):
        p_sym = p[sym]
        if p_sym <= 0.0:
            continue
        step = _step(sym)
        legs = sorted(
            ((strat, float(req)) for (strat, s), req in buy_request.items()
             if s == sym and float(req) > 0.0),
            key=lambda kv: kv[0],
        )
        total_req = sum(req for _, req in legs)
        if total_req <= 0.0:
            continue
        granted: dict[str, float] = {}
        for strat, req in legs:
            share = p_sym * (req / total_req)
            floored = math.floor(share / step + _FLOOR_SLACK) * step
            if floored < min_notional:
                floored = 0.0
            if floored > 0.0:
                granted[strat] = floored
        legs_by_symbol[sym] = legs
        granted_by_symbol[sym] = granted

    # The FULL floored book (seed + EVERY symbol's base grants — not a prefix). Every remainder
    # re-validation below starts from this whole book, so a remainder for one symbol is checked
    # against the base legs of the symbols processed BEFORE and AFTER it. That closes the MEDIUM:
    # a remainder can never combine with a not-yet-processed symbol's floored leg to overshoot the
    # aggregate gross cap, so step 6 can never hard-fail on a pure gross overshoot.
    book_now: dict[str, float] = {s: _seed(s) for s in symbols}
    for sym, granted in granted_by_symbol.items():
        book_now[sym] = book_now.get(sym, 0.0) + sum(granted.values())

    # PASS 2 — remainders. Hand at most ONE step of each symbol's floor-remainder to a single
    # strategy, and ONLY if re-adding it keeps the whole quantized book (`book_now`, incl. every
    # symbol's base legs + remainders already granted this pass) under every cap.
    for sym in sorted(granted_by_symbol):
        p_sym = p[sym]
        step = _step(sym)
        granted = granted_by_symbol[sym]
        remainder = p_sym - sum(granted.values())
        if remainder < step - _FLOOR_SLACK:
            continue
        for strat, _req in legs_by_symbol[sym]:
            candidate = granted.get(strat, 0.0) + step
            if candidate < min_notional:
                continue
            trial = book_now.get(sym, 0.0) + step
            trial_book = dict(book_now)
            trial_book[sym] = trial
            if _caps_ok(trial_book, cap_gross, cap_sym, conc_bound, eps):
                granted[strat] = candidate
                book_now[sym] = trial  # commit so later symbols' remainders see it
            break

    for sym, granted in granted_by_symbol.items():
        for strat, val in granted.items():
            if val > 0.0:
                q[(strat, sym)] = val

    # ---- step 6: FINAL VALIDATION over the EXACT quantized q, concentration vs FIXED D ---------
    final_book: dict[str, float] = {s: _seed(s) for s in symbols}
    for (_strat, sym), val in q.items():
        final_book[sym] = final_book.get(sym, 0.0) + val
    if sum(final_book.values()) > cap_gross + eps:
        return BookVerdict(ok=False, reason="final_net:gross", permitted_buys={})
    for val in final_book.values():
        if val > cap_sym + eps:
            return BookVerdict(ok=False, reason="final_net:symbol_notional", permitted_buys={})
        if c < 1.0 and val > conc_bound + eps:
            return BookVerdict(ok=False, reason="final_net:concentration", permitted_buys={})

    return BookVerdict(ok=True, reason=None, permitted_buys=q)
