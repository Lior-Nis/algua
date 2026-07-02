"""Unit tests for algua.risk.book_limits — book-level aggregate risk limits.

Written TDD-first. The module is a pure risk helper (no I/O, no DB): a frozen
`BookRiskLimits` config plus a stateful `BookExposure` accumulator whose
`permit_buy` returns the largest permitted long notional that keeps every
book-level cap satisfied AFTER the add, then mutates the running totals.
"""

from __future__ import annotations

import math

import pytest

from algua.risk.book_limits import BookExposure, BookRiskLimits

TOL = 1e-9


# --------------------------------------------------------------------------- #
# BookRiskLimits validation
# --------------------------------------------------------------------------- #


def test_defaults_are_valid() -> None:
    lim = BookRiskLimits()
    assert lim.max_gross == 2.0
    assert lim.max_net == 1.0
    assert lim.max_symbol_concentration == 0.25
    assert lim.max_symbol_notional == 0.50


def test_frozen_dataclass_is_immutable() -> None:
    lim = BookRiskLimits()
    with pytest.raises(Exception):  # noqa: B017 - FrozenInstanceError (dataclasses)
        lim.max_gross = 3.0  # type: ignore[misc]


def test_concentration_one_is_accepted() -> None:
    # c == 1.0 is the upper boundary of (0, 1]; it means "no binding concentration cap".
    lim = BookRiskLimits(max_symbol_concentration=1.0)
    assert lim.max_symbol_concentration == 1.0


@pytest.mark.parametrize("bad", [-1e-9, -1.0, -100.0])
def test_negative_max_gross_rejected(bad: float) -> None:
    with pytest.raises(ValueError):
        BookRiskLimits(max_gross=bad)


@pytest.mark.parametrize("bad", [-1e-9, -1.0])
def test_negative_max_net_rejected(bad: float) -> None:
    with pytest.raises(ValueError):
        BookRiskLimits(max_net=bad)


@pytest.mark.parametrize("bad", [-1e-9, -1.0])
def test_negative_max_symbol_notional_rejected(bad: float) -> None:
    with pytest.raises(ValueError):
        BookRiskLimits(max_symbol_notional=bad)


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.0000001, 2.0, -1.0])
def test_concentration_out_of_range_rejected(bad: float) -> None:
    # Must satisfy 0 < c <= 1.
    with pytest.raises(ValueError):
        BookRiskLimits(max_symbol_concentration=bad)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_gross": float("nan")},
        {"max_gross": float("inf")},
        {"max_net": float("nan")},
        {"max_net": float("inf")},
        {"max_symbol_notional": float("nan")},
        {"max_symbol_notional": float("inf")},
        {"max_symbol_concentration": float("nan")},
        {"max_symbol_concentration": float("inf")},
    ],
)
def test_non_finite_caps_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        BookRiskLimits(**kwargs)


# --------------------------------------------------------------------------- #
# permit_buy — happy path
# --------------------------------------------------------------------------- #


def test_full_permit_when_book_empty_and_buy_small() -> None:
    # Concentration is disabled (c==1) so a single name into an empty book isn't blocked by
    # being 100% of gross; the buy is well under every remaining cap → fully permitted.
    lim = BookRiskLimits(max_symbol_concentration=1.0)  # gross 2.0, net 1.0, sym_notional 0.50
    book = BookExposure(equity=100.0, book_notionals={}, limits=lim)
    # A tiny buy well under every cap → fully permitted.
    permitted = book.permit_buy("AAPL", 5.0)
    assert permitted == pytest.approx(5.0, abs=TOL)
    # Accumulator mutated.
    assert book.book["AAPL"] == pytest.approx(5.0, abs=TOL)
    assert book.gross == pytest.approx(5.0, abs=TOL)
    assert book.net == pytest.approx(5.0, abs=TOL)


def test_concentration_floored_at_equity_permits_first_fill_from_flat() -> None:
    # The concentration denominator is FLOORED at equity: while the book is unlevered
    # (gross <= equity) a lone name is capped at c*equity, NOT "100% of a tiny gross". So a
    # first fill from a FLAT book of a name under c*equity is fully permitted (not spuriously
    # blocked). c=0.25, equity=100 → a lone name may reach 25; a 5.0 buy is well under.
    lim = BookRiskLimits(max_symbol_concentration=0.25)
    book = BookExposure(equity=100.0, book_notionals={}, limits=lim)
    assert book.permit_buy("AAA", 5.0) == pytest.approx(5.0, abs=TOL)
    assert book.book["AAA"] == pytest.approx(5.0, abs=TOL)
    # A first fill ABOVE c*equity is trimmed to c*equity (25) by the equity-floored cap.
    book2 = BookExposure(equity=100.0, book_notionals={}, limits=lim)
    assert book2.permit_buy("AAA", 40.0) == pytest.approx(25.0, abs=TOL)


def test_seed_is_tracked_in_gross_and_net() -> None:
    lim = BookRiskLimits()
    book = BookExposure(equity=100.0, book_notionals={"AAPL": 10.0, "MSFT": 20.0}, limits=lim)
    assert book.gross == pytest.approx(30.0, abs=TOL)
    assert book.net == pytest.approx(30.0, abs=TOL)


# --------------------------------------------------------------------------- #
# permit_buy — cap binds (net dominates gross for default limits, so we test
# each cap with limits chosen so THAT cap is the binding one).
# --------------------------------------------------------------------------- #


def test_gross_cap_binds_across_sequential_strategies() -> None:
    # Make gross the binding cap: raise net & concentration & symbol_notional out of the way.
    lim = BookRiskLimits(
        max_gross=1.0,
        max_net=10.0,
        max_symbol_concentration=1.0,
        max_symbol_notional=10.0,
    )
    equity = 100.0
    book = BookExposure(equity=equity, book_notionals={}, limits=lim)
    # Gross headroom = max_gross*equity = 100.0 total.
    # Strategy 1 asks for 80 of a name → fully permitted (under 100).
    p1 = book.permit_buy("AAA", 80.0)
    assert p1 == pytest.approx(80.0, abs=TOL)
    # Strategy 2 asks for 50 of another name → only 20 left of gross headroom.
    p2 = book.permit_buy("BBB", 50.0)
    assert p2 == pytest.approx(20.0, abs=TOL)
    assert book.gross == pytest.approx(100.0, abs=TOL)  # exactly at cap
    # Strategy 3 gets nothing.
    p3 = book.permit_buy("CCC", 10.0)
    assert p3 == pytest.approx(0.0, abs=TOL)
    assert book.gross == pytest.approx(100.0, abs=TOL)  # no over-fill


def test_net_cap_binds() -> None:
    # Default limits: net 1.0*equity is tighter than gross 2.0*equity (long-only), so net binds.
    lim = BookRiskLimits(
        max_gross=10.0,
        max_net=1.0,
        max_symbol_concentration=1.0,
        max_symbol_notional=10.0,
    )
    equity = 100.0
    book = BookExposure(equity=equity, book_notionals={}, limits=lim)
    # Net headroom = 100. Buy 150 → trimmed to 100.
    permitted = book.permit_buy("AAA", 150.0)
    assert permitted == pytest.approx(100.0, abs=TOL)
    assert book.net == pytest.approx(100.0, abs=TOL)


def test_symbol_notional_cap_binds() -> None:
    lim = BookRiskLimits(
        max_gross=10.0,
        max_net=10.0,
        max_symbol_concentration=1.0,
        max_symbol_notional=0.30,
    )
    equity = 100.0
    book = BookExposure(equity=equity, book_notionals={"AAA": 10.0}, limits=lim)
    # Per-symbol notional cap = 0.30*100 = 30. AAA already at 10 → 20 headroom.
    permitted = book.permit_buy("AAA", 50.0)
    assert permitted == pytest.approx(20.0, abs=TOL)
    assert book.book["AAA"] == pytest.approx(30.0, abs=TOL)


def test_concentration_cap_binds_on_same_name_when_levered() -> None:
    # On a LEVERED book (gross >= equity) the concentration denominator is gross, so the
    # gross-fraction cap binds: a further buy of the SAME name is limited to c of the deployed
    # book. Seed gross == equity so the floor at equity is not looser than gross.
    lim = BookRiskLimits(
        max_gross=100.0,
        max_net=100.0,
        max_symbol_concentration=0.50,
        max_symbol_notional=100.0,
    )
    equity = 100.0
    # Seed: AAA=40, BBB=60 → gross 100 == equity. AAA share = 40/100 = 0.40 < 0.50.
    book = BookExposure(equity=equity, book_notionals={"AAA": 40.0, "BBB": 60.0}, limits=lim)
    # Add AAA. gross(100) >= equity(100) so denom is gross: (40+p)/(100+p) <= 0.50
    #   40+p <= 50 + 0.5p  → 0.5p <= 10 → p <= 20.
    permitted = book.permit_buy("AAA", 1000.0)
    assert permitted == pytest.approx(20.0, abs=TOL)
    sb = book.book["AAA"]
    g = book.gross
    # Post-add concentration is at the cap within tolerance.
    assert sb / g == pytest.approx(0.50, abs=1e-9)
    assert sb / g <= lim.max_symbol_concentration + TOL


def test_concentration_floor_reverts_to_gross_once_levered() -> None:
    # Below 1x gross the cap is c*equity; buying past equity switches the base to gross. Seed a
    # near-flat book and buy a large amount of ONE name: it's trimmed at c*equity (the unlevered
    # regime binds before the book levers up), never exceeding c of the eventual book.
    lim = BookRiskLimits(
        max_gross=10.0, max_net=10.0, max_symbol_concentration=0.25, max_symbol_notional=10.0,
    )
    equity = 100.0
    book = BookExposure(equity=equity, book_notionals={}, limits=lim)
    # Flat book: h_floor = c*equity - sb = 25; kink = equity - gross = 100; 25 < 100 so the
    # unlevered (equity-floored) regime binds → permitted 25.
    permitted = book.permit_buy("AAA", 1000.0)
    assert permitted == pytest.approx(25.0, abs=TOL)
    assert book.book["AAA"] / max(book.gross, equity) <= lim.max_symbol_concentration + TOL


def test_concentration_never_binds_when_c_is_one() -> None:
    # c == 1.0 → concentration headroom is +inf; only gross/net/symbol_notional matter.
    lim = BookRiskLimits(
        max_gross=100.0,
        max_net=100.0,
        max_symbol_concentration=1.0,
        max_symbol_notional=0.10,  # this one binds instead
    )
    equity = 1000.0
    book = BookExposure(equity=equity, book_notionals={"AAA": 900.0, "BBB": 50.0}, limits=lim)
    # Even though AAA would dominate the book, concentration does not bind (c==1).
    # symbol_notional cap = 0.10*1000 = 100; AAA at 900 already over → 0 headroom for AAA.
    permitted = book.permit_buy("AAA", 500.0)
    assert permitted == pytest.approx(0.0, abs=TOL)
    # But a different small name is limited only by its own symbol_notional headroom.
    p2 = book.permit_buy("CCC", 500.0)
    assert p2 == pytest.approx(100.0, abs=TOL)  # 0.10*1000 - 0


# --------------------------------------------------------------------------- #
# permit_buy — fail-closed / degenerate inputs
# --------------------------------------------------------------------------- #


def test_already_breached_seed_returns_zero() -> None:
    lim = BookRiskLimits(max_gross=1.0, max_net=10.0, max_symbol_concentration=1.0,
                         max_symbol_notional=10.0)
    equity = 100.0
    # Seed gross 150 > max_gross*equity (100) — already over.
    book = BookExposure(equity=equity, book_notionals={"AAA": 150.0}, limits=lim)
    permitted = book.permit_buy("BBB", 10.0)
    assert permitted == 0.0
    assert book.gross == pytest.approx(150.0, abs=TOL)  # unchanged, no mutation


def test_non_positive_equity_returns_zero() -> None:
    lim = BookRiskLimits()
    for eq in (0.0, -100.0):
        book = BookExposure(equity=eq, book_notionals={}, limits=lim)
        assert book.permit_buy("AAA", 10.0) == 0.0
        assert book.book.get("AAA", 0.0) == 0.0


def test_non_finite_equity_returns_zero() -> None:
    lim = BookRiskLimits()
    for eq in (float("nan"), float("inf")):
        book = BookExposure(equity=eq, book_notionals={}, limits=lim)
        assert book.permit_buy("AAA", 10.0) == 0.0


@pytest.mark.parametrize("req", [0.0, -1.0, -100.0])
def test_non_positive_requested_returns_zero_no_mutation(req: float) -> None:
    lim = BookRiskLimits()
    book = BookExposure(equity=100.0, book_notionals={"AAA": 5.0}, limits=lim)
    assert book.permit_buy("AAA", req) == 0.0
    assert book.book["AAA"] == pytest.approx(5.0, abs=TOL)
    assert book.gross == pytest.approx(5.0, abs=TOL)
    assert book.net == pytest.approx(5.0, abs=TOL)


@pytest.mark.parametrize("req", [float("nan"), float("inf")])
def test_non_finite_requested_returns_zero_no_mutation(req: float) -> None:
    lim = BookRiskLimits()
    book = BookExposure(equity=100.0, book_notionals={"AAA": 5.0}, limits=lim)
    assert book.permit_buy("AAA", req) == 0.0
    assert book.book["AAA"] == pytest.approx(5.0, abs=TOL)
    assert book.gross == pytest.approx(5.0, abs=TOL)


# --------------------------------------------------------------------------- #
# Mutation compounding across sequential calls
# --------------------------------------------------------------------------- #


def test_mutation_compounds_same_name() -> None:
    lim = BookRiskLimits(max_gross=100.0, max_net=100.0, max_symbol_concentration=1.0,
                         max_symbol_notional=0.50)
    equity = 100.0
    book = BookExposure(equity=equity, book_notionals={}, limits=lim)
    # symbol_notional cap = 50.
    p1 = book.permit_buy("AAA", 30.0)
    assert p1 == pytest.approx(30.0, abs=TOL)
    # Second buy sees the running 30 → only 20 headroom left.
    p2 = book.permit_buy("AAA", 30.0)
    assert p2 == pytest.approx(20.0, abs=TOL)
    assert book.book["AAA"] == pytest.approx(50.0, abs=TOL)
    # Third buy → nothing.
    p3 = book.permit_buy("AAA", 10.0)
    assert p3 == pytest.approx(0.0, abs=TOL)


def test_mutation_compounds_across_names_on_gross() -> None:
    lim = BookRiskLimits(max_gross=0.5, max_net=10.0, max_symbol_concentration=1.0,
                         max_symbol_notional=10.0)
    equity = 200.0  # gross headroom = 0.5*200 = 100.
    book = BookExposure(equity=equity, book_notionals={}, limits=lim)
    assert book.permit_buy("AAA", 60.0) == pytest.approx(60.0, abs=TOL)
    assert book.permit_buy("BBB", 60.0) == pytest.approx(40.0, abs=TOL)  # 100-60
    assert book.gross == pytest.approx(100.0, abs=TOL)


# --------------------------------------------------------------------------- #
# Boundary — a permit that would exactly hit a cap is allowed to the boundary
# --------------------------------------------------------------------------- #


def test_exact_boundary_permit_allowed() -> None:
    lim = BookRiskLimits(max_gross=1.0, max_net=10.0, max_symbol_concentration=1.0,
                         max_symbol_notional=10.0)
    equity = 100.0
    book = BookExposure(equity=equity, book_notionals={}, limits=lim)
    # gross headroom exactly 100; ask exactly 100.
    permitted = book.permit_buy("AAA", 100.0)
    assert permitted == pytest.approx(100.0, abs=TOL)
    assert book.gross == pytest.approx(100.0, abs=TOL)
    assert not math.isnan(book.gross)


# --------------------------------------------------------------------------- #
# seed_breaches — already-breached seed detection (Codex #389 GATE-2)
# --------------------------------------------------------------------------- #


def test_seed_breaches_empty_for_compliant_book() -> None:
    lim = BookRiskLimits()  # gross 2.0, net 1.0, conc 0.25, sym_notional 0.50
    book = BookExposure(equity=1000.0, book_notionals={"AAA": 100.0, "BBB": 100.0}, limits=lim)
    assert book.seed_breaches() == []


def test_seed_breaches_flags_symbol_notional_on_other_name() -> None:
    # The hole Codex found: AAA over its notional cap must be flagged even when a buy targets BBB.
    lim = BookRiskLimits(max_gross=100.0, max_net=100.0, max_symbol_concentration=1.0,
                         max_symbol_notional=0.50)
    book = BookExposure(equity=1000.0, book_notionals={"AAA": 600.0}, limits=lim)
    assert "symbol_notional" in book.seed_breaches()


def test_seed_breaches_flags_gross_net_and_concentration() -> None:
    lim = BookRiskLimits(max_gross=0.5, max_net=0.5, max_symbol_concentration=0.25,
                         max_symbol_notional=10.0)
    # gross 900 > 0.5*1000=500; net 900 > 500; AAA 600/max(900,1000)=0.6 > 0.25.
    book = BookExposure(equity=1000.0, book_notionals={"AAA": 600.0, "BBB": 300.0}, limits=lim)
    flags = set(book.seed_breaches())
    assert {"gross", "net", "concentration"} <= flags


def test_seed_breaches_flags_degenerate_equity() -> None:
    assert BookExposure(0.0, {}, BookRiskLimits()).seed_breaches() == ["equity"]
    assert BookExposure(float("nan"), {}, BookRiskLimits()).seed_breaches() == ["equity"]


def test_seed_compliant_book_at_exactly_the_cap_is_not_breached() -> None:
    lim = BookRiskLimits(max_gross=1.0, max_net=10.0, max_symbol_concentration=1.0,
                         max_symbol_notional=10.0)
    # gross exactly at 1.0*100 = 100 -> not breached (tolerance).
    book = BookExposure(equity=100.0, book_notionals={"AAA": 100.0}, limits=lim)
    assert book.seed_breaches() == []


# --------------------------------------------------------------------------- #
# permit_buy min_notional floor — sub-minimum trim skips WITHOUT mutating
# --------------------------------------------------------------------------- #


def test_permit_buy_sub_min_notional_returns_zero_no_mutation() -> None:
    # Only 0.5 of headroom left but venue min is 1.0 -> the buy would be skipped downstream, so
    # permit_buy returns 0 and does NOT burn book budget (accounting stays in step with the venue).
    lim = BookRiskLimits(max_gross=1.0, max_net=10.0, max_symbol_concentration=1.0,
                         max_symbol_notional=10.0)
    book = BookExposure(equity=100.0, book_notionals={"AAA": 99.5}, limits=lim)  # gross hr = 0.5
    permitted = book.permit_buy("BBB", 10.0, min_notional=1.0)
    assert permitted == 0.0
    assert book.gross == pytest.approx(99.5, abs=TOL)  # unchanged
    assert "BBB" not in book.book


def test_permit_buy_at_or_above_min_notional_still_permits() -> None:
    lim = BookRiskLimits(max_gross=2.0, max_net=10.0, max_symbol_concentration=1.0,
                         max_symbol_notional=10.0)
    book = BookExposure(equity=100.0, book_notionals={}, limits=lim)
    # 5.0 >= min_notional 1.0 -> permitted and mutated.
    assert book.permit_buy("AAA", 5.0, min_notional=1.0) == pytest.approx(5.0, abs=TOL)
    assert book.book["AAA"] == pytest.approx(5.0, abs=TOL)
