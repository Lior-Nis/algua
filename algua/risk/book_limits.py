"""Book-level aggregate risk limits — a pure risk helper (no I/O, no DB, stdlib only).

Where `algua.risk.limits` caps a SINGLE strategy's decision-weight vector, this module caps
the BOOK: the aggregate exposure summed across every strategy/tenant on one account. A caller
seeds `BookExposure` with the current per-symbol notional across the whole account, then asks
`permit_buy(symbol, requested)` before submitting each incremental buy; the accumulator returns
the largest notional that keeps every book-level cap satisfied AFTER the add and folds the
grant into its running totals, so a sequence of buys across strategies can never collectively
breach a book cap (the first buyer consumes headroom the next one no longer sees).

Precondition: LONG-ONLY. All seeded notionals and all requested buys are >= 0; under it gross
(abs sum) == net (signed sum), but both are tracked independently for defense-in-depth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Absolute tolerance for a seed "already breached" comparison — absorbs float noise in the
# mark×qty book valuation so a book exactly at a cap isn't spuriously flagged as over it.
_EPS = 1e-6


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


class BookExposure:
    """Stateful accumulator over the account's aggregate long book.

    Seed with the current signed notional per symbol summed across the whole account, then call
    `permit_buy` per incremental buy. `permit_buy` returns the largest permitted notional and
    mutates the running book/gross/net on a positive grant, so subsequent calls see the effect.
    """

    def __init__(
        self, equity: float, book_notionals: dict[str, float], limits: BookRiskLimits
    ) -> None:
        self.equity = equity
        self.limits = limits
        # Mutable working copy; unseen symbols default to 0.0 on access.
        self.book: dict[str, float] = dict(book_notionals)
        # Running aggregates. gross = Σ|v| (abs sum), net = Σ v (signed sum). Under the long-only
        # precondition these coincide at seed, but we track them independently for defense.
        self.gross: float = sum(abs(v) for v in book_notionals.values())
        self.net: float = sum(book_notionals.values())

    def seed_breaches(self) -> list[str]:
        """Names of book caps ALREADY breached by the seed (before any buy). A non-empty result
        means the whole account book is over a limit at reconcile time — an anomaly the caller must
        treat as fail-closed (skip the cycle), because the per-buy monotone headroom only guarantees
        NO-worse; it cannot heal an already-breached OTHER symbol via a buy (Codex #389 GATE-2). An
        empty book (equity <= 0 or non-finite) is reported as breached ('equity') so a degenerate
        seed also fails closed. Concentration checked with the same equity-floored base as buys."""
        e = self.equity
        if not _finite(e) or e <= 0:
            return ["equity"]
        lim = self.limits
        breaches: list[str] = []
        if self.gross > lim.max_gross * e + _EPS:
            breaches.append("gross")
        if self.net > lim.max_net * e + _EPS:
            breaches.append("net")
        over_notional = [
            s for s, v in self.book.items() if abs(v) > lim.max_symbol_notional * e + _EPS
        ]
        if over_notional:
            breaches.append("symbol_notional")
        if lim.max_symbol_concentration < 1.0:
            denom = max(self.gross, e)
            over_conc = [
                s for s, v in self.book.items()
                if abs(v) > lim.max_symbol_concentration * denom + _EPS
            ]
            if over_conc:
                breaches.append("concentration")
        return breaches

    def permit_buy(
        self, symbol: str, requested_notional: float, *, min_notional: float = 0.0
    ) -> float:
        """Largest permitted BUY notional (0 <= permitted <= requested_notional) that keeps
        every book cap satisfied after the add; mutate the accumulator iff permitted > 0.

        `min_notional`: a venue minimum — if the trimmed permitted lands in (0, min_notional) the
        buy would be SKIPPED downstream (below the broker minimum), so return 0 WITHOUT mutating,
        keeping the accumulator's book in step with what actually reaches the venue (Codex #389).

        Fail-closed on: requested <= 0, non-finite requested, equity <= 0, non-finite equity.
        """
        if not _finite(requested_notional) or requested_notional <= 0:
            return 0.0
        e = self.equity
        if not _finite(e) or e <= 0:
            return 0.0

        lim = self.limits
        sb = self.book.get(symbol, 0.0)  # >= 0 under precondition
        g = self.gross
        n = self.net
        c = lim.max_symbol_concentration

        # Each headroom floored at 0 — an already-breached seed yields 0 headroom, not negative.
        gross_hr = max(0.0, lim.max_gross * e - g)
        net_hr = max(0.0, lim.max_net * e - n)
        symbol_notional_hr = max(0.0, lim.max_symbol_notional * e - sb)
        if c >= 1.0:
            concentration_hr = math.inf  # no binding concentration cap
        else:
            # Post-add (sb + p) <= c * max(g + p, e), equity-floored so a first fill into a
            # flat/small book isn't spuriously "100% of a tiny gross". `sb + p - c*max(g+p, e)`
            # is continuous & strictly increasing in p (slope 1 below the kink p*=e-g, 1-c above),
            # so the feasible set is [0, h] — closed form (Codex #389):
            if g >= e:
                concentration_hr = max(0.0, (c * g - sb) / (1 - c))  # already levered: gross base
            else:
                h_floor = c * e - sb                # regime gross+p <= e: base is equity
                kink = e - g                         # p* where gross+p crosses e
                if h_floor < kink:                   # binding while still unlevered
                    concentration_hr = max(0.0, h_floor)
                else:                                # headroom extends past the kink -> gross base
                    concentration_hr = max(0.0, (c * g - sb) / (1 - c))

        permitted = max(
            0.0,
            min(requested_notional, gross_hr, net_hr, symbol_notional_hr, concentration_hr),
        )

        # A trim that lands below the venue minimum is SKIPPED downstream (no order posted), so do
        # not burn book budget for a phantom fill — return 0 without mutating (Codex #389 GATE-2).
        if permitted < min_notional:
            return 0.0

        if permitted > 0.0:
            self.book[symbol] = sb + permitted
            self.gross = g + permitted
            self.net = n + permitted

        return permitted
