"""Unit tests for algua.risk.book_limits — book-level aggregate risk limits (#389 design v4).

The module is a pure risk helper (no I/O, no DB): a frozen `BookRiskLimits` config plus one pure
whole-cycle function `evaluate_book(equity, seed_notionals, sell_total, buy_request, limits, *,
buy_notional_step, min_notional) -> BookVerdict`. These tests exercise the v4 evaluator directly:
each cap binding on the aggregate book, prefix-safe concentration under a gross-shrinking SELL, the
no-SELL-credit rule, the hard over-sell fail-closed, the flat-account full-basket bootstrap, the
global reduce-only on any seed breach, the closed-form (no-fixpoint) concentration bound, the
rounding-safe notional distribution, the fixed-`D` final validation, and the min_notional floor.
"""
from __future__ import annotations

import dataclasses
import math

import pytest

from algua.risk.book_limits import BookRiskLimits, BookVerdict, evaluate_book

# --------------------------------------------------------------------------- #
# BookRiskLimits config validation (unchanged four caps)
# --------------------------------------------------------------------------- #


def test_book_risk_limits_defaults():
    lim = BookRiskLimits()
    assert lim.max_gross == 2.0
    assert lim.max_net == 1.0
    assert lim.max_symbol_concentration == 0.25
    assert lim.max_symbol_notional == 0.50


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_gross": -1.0},
        {"max_net": -0.1},
        {"max_symbol_notional": -0.5},
        {"max_symbol_concentration": 0.0},
        {"max_symbol_concentration": 1.5},
        {"max_gross": float("nan")},
        {"max_net": float("inf")},
    ],
)
def test_book_risk_limits_rejects_bad_config(kwargs):
    with pytest.raises(ValueError):
        BookRiskLimits(**kwargs)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _relaxed(**over) -> BookRiskLimits:
    """A permissive limits object; override one cap to make it the binding one."""
    base = dict(max_gross=100.0, max_net=100.0, max_symbol_concentration=1.0,
                max_symbol_notional=100.0)
    base.update(over)
    return BookRiskLimits(**base)


# --------------------------------------------------------------------------- #
# step 0 / 0b — degenerate equity, over-sell fail-closed (§v4-1)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("equity", [0.0, -1.0, float("nan"), float("inf")])
def test_degenerate_equity_fails_closed(equity):
    v = evaluate_book(equity, {}, {}, {("s", "AAA"): 100.0}, BookRiskLimits())
    assert v.ok is False
    assert v.reason == "equity"
    assert v.permitted_buys == {}


def test_oversell_fails_closed_hard_not_clamped():
    # Aggregate SELL of AAA (150) exceeds held seed (100) -> HARD fail closed, NOT a max(0,.) clamp.
    v = evaluate_book(1000.0, {"AAA": 100.0}, {"AAA": 150.0}, {}, BookRiskLimits())
    assert v.ok is False
    assert v.reason == "oversell:AAA"
    assert v.permitted_buys == {}


def test_negative_sell_fails_closed():
    v = evaluate_book(1000.0, {"AAA": 100.0}, {"AAA": -1.0}, {}, BookRiskLimits())
    assert v.ok is False
    assert v.reason == "oversell:AAA"


def test_sell_exactly_at_seed_is_permitted():
    # sell == seed is within bound (B[s] = 0); no over-sell.
    v = evaluate_book(1000.0, {"AAA": 100.0}, {"AAA": 100.0}, {}, BookRiskLimits())
    assert v.ok is True
    assert v.reason is None


def test_naked_sell_on_large_equity_account_fails_closed():
    # #389 GATE-2: the step-0b over-sell assert must use a FIXED absolute tolerance, NOT an
    # equity-scaled eps. On a $1e9 account an equity-scaled band (~$1000) would wave through a
    # naked short of up to that size; a naked sell of ANY material size must fail closed regardless
    # of account equity, or the B[s] = seed − sell >= 0 prefix-safety bound breaks.
    equity = 1e9
    # sell exceeds seed by $500 — under the old equity-scaled eps ($1000) but a real naked short.
    v = evaluate_book(equity, {"AAA": 100.0}, {"AAA": 600.0}, {}, BookRiskLimits())
    assert v.ok is False
    assert v.reason == "oversell:AAA"
    assert v.permitted_buys == {}


# --------------------------------------------------------------------------- #
# each cap binding on the AGGREGATE book
# --------------------------------------------------------------------------- #


def test_per_symbol_notional_cap_binds_on_aggregate():
    # cap_sym = 0.5 * 1000 = 500. Two strategies each request 400 of AAA (aggregate 800) -> the
    # aggregate BUY is trimmed so seed(0)+Σq <= 500.
    lim = _relaxed(max_symbol_notional=0.5)
    v = evaluate_book(
        1000.0, {}, {},
        {("s1", "AAA"): 400.0, ("s2", "AAA"): 400.0},
        lim, buy_notional_step=0.01,
    )
    assert v.ok is True
    total = sum(v.permitted_buys.values())
    assert total <= 500.0 + 1e-6
    assert total == pytest.approx(500.0, abs=0.02)


def test_gross_cap_binds_on_aggregate():
    # cap_gross = min(max_gross, max_net) * equity. Both 1.0 => cap_gross = 1000. Requests across
    # names sum to 1500 -> trimmed to 1000 gross.
    lim = _relaxed(max_gross=1.0, max_net=1.0)
    v = evaluate_book(
        1000.0, {}, {},
        {("s1", "AAA"): 500.0, ("s2", "BBB"): 500.0, ("s3", "CCC"): 500.0},
        lim, buy_notional_step=0.01,
    )
    assert v.ok is True
    total = sum(v.permitted_buys.values())
    assert total <= 1000.0 + 1e-6
    assert total == pytest.approx(1000.0, abs=0.05)


def test_net_cap_is_the_tighter_of_gross_net():
    # max_gross 100 but max_net 0.3 => cap_gross = min * equity = 300.
    lim = _relaxed(max_gross=100.0, max_net=0.3)
    v = evaluate_book(
        1000.0, {}, {},
        {("s1", "AAA"): 500.0, ("s2", "BBB"): 500.0},
        lim, buy_notional_step=0.01,
    )
    assert v.ok is True
    assert sum(v.permitted_buys.values()) <= 300.0 + 1e-6


def test_concentration_cap_binds_on_aggregate():
    # c = 0.25, equity 1000, flat book => a single name capped at 0.25 * max(equity, ΣB) = 250.
    lim = _relaxed(max_symbol_concentration=0.25)
    v = evaluate_book(
        1000.0, {}, {},
        {("s1", "AAA"): 400.0, ("s2", "AAA"): 400.0},
        lim, buy_notional_step=0.01,
    )
    assert v.ok is True
    assert sum(v.permitted_buys.values()) <= 250.0 + 1e-6
    assert sum(v.permitted_buys.values()) == pytest.approx(250.0, abs=0.02)


# --------------------------------------------------------------------------- #
# prefix-safety of concentration (the ΣB / equity floor)
# --------------------------------------------------------------------------- #


def test_concentration_uses_equity_floor_not_raw_symbol_over_gross():
    # Codex counterexample: equity=100, c=0.5, hold A=50, B=50, SELL all of B.
    # D = max(equity=100, ΣB=50) = 100 => A's concentration bound is c*D = 50, exactly A's held 50,
    # so A is NOT a seed breach (50 <= c*D=50) and a BUY of A gets ZERO headroom — the raw
    # symbol/gross form (50 / post-sell gross 50 = 100%) is NOT what is enforced.
    v = evaluate_book(
        100.0,
        {"A": 50.0, "B": 50.0},
        {"B": 50.0},
        {("s", "A"): 100.0},  # wants to add to A
        BookRiskLimits(max_gross=100.0, max_net=100.0,
                       max_symbol_concentration=0.5, max_symbol_notional=100.0),
        buy_notional_step=0.01,
    )
    assert v.ok is True
    assert v.permitted_buys.get(("s", "A"), 0.0) == pytest.approx(0.0, abs=1e-9)


def test_sell_shrinking_gross_does_not_free_concentration_headroom():
    # equity 1000, c 0.25. Hold A=200, Z=200 (each within the seed concentration cap c*D=250); SELL
    # all of Z so ΣB drops to 200 (< equity). The BUY headroom of A is against the FIXED, equity-
    # floored D = max(equity, ΣB) = max(1000, 200) = 1000, NOT ΣB. Without the equity floor D would
    # be 200 and A (already 200) would falsely breach; with it, A may climb to c*D = 250 (add 50).
    v = evaluate_book(
        1000.0,
        {"A": 200.0, "Z": 200.0},
        {"Z": 200.0},
        {("s", "A"): 1000.0},
        BookRiskLimits(max_gross=100.0, max_net=100.0,
                       max_symbol_concentration=0.25, max_symbol_notional=100.0),
        buy_notional_step=0.01,
    )
    assert v.ok is True
    assert v.permitted_buys.get(("s", "A"), 0.0) == pytest.approx(50.0, abs=0.02)


# --------------------------------------------------------------------------- #
# no-SELL-credit: BUY headroom is against seed, not seed - sell
# --------------------------------------------------------------------------- #


def test_no_sell_credit_same_cycle():
    # cap_sym = 0.5*1000 = 500. Hold AAA=500 (at the per-symbol cap), SELL 300 of AAA. A same-cycle
    # BUY of AAA gets NO credit for the sell: its headroom is cap_sym - seed(500) = 0, not
    # cap_sym - (seed - sell). So the BUY is fully trimmed to 0.
    v = evaluate_book(
        1000.0,
        {"AAA": 500.0},
        {"AAA": 300.0},
        {("s", "AAA"): 400.0},
        _relaxed(max_symbol_notional=0.5),
        buy_notional_step=0.01,
    )
    assert v.ok is True
    assert v.permitted_buys.get(("s", "AAA"), 0.0) == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# flat-account full-basket bootstrap (D = equity dominates, no per-leg kink)
# --------------------------------------------------------------------------- #


def test_full_basket_bootstrap_into_flat_account():
    # Flat account, diversified basket well within every cap => no spurious concentration trim.
    lim = BookRiskLimits(max_gross=2.0, max_net=1.0,
                         max_symbol_concentration=0.25, max_symbol_notional=0.5)
    req = {("s", sym): 100.0 for sym in ("A", "B", "C", "D")}  # each 100 < 250 conc, Σ 400 < 1000
    v = evaluate_book(1000.0, {}, {}, req, lim, buy_notional_step=0.01)
    assert v.ok is True
    for sym in ("A", "B", "C", "D"):
        assert v.permitted_buys[("s", sym)] == pytest.approx(100.0, abs=0.02)


# --------------------------------------------------------------------------- #
# global reduce-only on ANY seed breach (§v4-4)
# --------------------------------------------------------------------------- #


def test_seed_breach_one_name_triggers_global_reduce_only():
    # cap_sym = 0.5*1000 = 500. Held AAA = 600 (breach). BBB = 100 (fine). A breach in ONE name =>
    # P[s] = 0 for EVERY symbol (incl. the unbreached BBB); SELLs still permitted; ok=True.
    v = evaluate_book(
        1000.0,
        {"AAA": 600.0, "BBB": 100.0},
        {},
        {("s", "BBB"): 200.0},  # would-be BUY of the UNBREACHED name
        BookRiskLimits(max_gross=100.0, max_net=100.0,
                       max_symbol_concentration=1.0, max_symbol_notional=0.5),
    )
    assert v.ok is True
    assert v.reason == "seed_breach:symbol_notional"
    assert v.permitted_buys == {}


def test_seed_breach_gross_triggers_global_reduce_only():
    lim = _relaxed(max_gross=1.0, max_net=1.0)  # cap_gross = 1000
    v = evaluate_book(
        1000.0,
        {"AAA": 700.0, "BBB": 700.0},  # Σseed = 1400 > 1000
        {},
        {("s", "CCC"): 100.0},
        lim,
    )
    assert v.ok is True
    assert v.reason == "seed_breach:gross"
    assert v.permitted_buys == {}


def test_seed_breach_concentration_triggers_global_reduce_only():
    # c 0.25, equity 1000, D=max(1000, ΣB=300)=1000. Held AAA=300 > c*D=250 -> concentration breach.
    v = evaluate_book(
        1000.0,
        {"AAA": 300.0},
        {},
        {("s", "BBB"): 100.0},
        BookRiskLimits(max_gross=100.0, max_net=100.0,
                       max_symbol_concentration=0.25, max_symbol_notional=100.0),
    )
    assert v.ok is True
    assert v.reason == "seed_breach:concentration"
    assert v.permitted_buys == {}


def test_seed_breach_still_permits_sells():
    # A seed breach returns ok=True (SELLs may proceed to heal) but zero BUYs.
    v = evaluate_book(
        1000.0,
        {"AAA": 600.0},
        {"AAA": 100.0},  # a de-risking SELL
        {("s", "AAA"): 50.0},
        BookRiskLimits(max_gross=100.0, max_net=100.0,
                       max_symbol_concentration=1.0, max_symbol_notional=0.5),
    )
    assert v.ok is True
    assert v.reason.startswith("seed_breach:")
    assert v.permitted_buys == {}


# --------------------------------------------------------------------------- #
# closed-form concentration bound — assert NO fixpoint iteration
# --------------------------------------------------------------------------- #


def test_no_fixpoint_iteration_path_exists():
    # The evaluator must be a single closed-form pass — no loop re-solves the concentration cap
    # against a moving denominator. Guard structurally: no `while` loop, D computed exactly once.
    import inspect

    from algua.risk import book_limits

    src = inspect.getsource(book_limits.evaluate_book)
    assert "while" not in src, "evaluate_book must not iterate to a fixpoint"
    assert src.count("d = max(equity") == 1


def test_concentration_bound_is_single_valued_closed_form():
    # With a fixed D, the per-symbol concentration headroom is exactly c*D - seed. Held AAA=100,
    # equity 1000, c 0.25 => bound = 250 - 100 = 150 additional. Request 500 -> granted 150.
    v = evaluate_book(
        1000.0,
        {"AAA": 100.0},
        {},
        {("s", "AAA"): 500.0},
        BookRiskLimits(max_gross=100.0, max_net=100.0,
                       max_symbol_concentration=0.25, max_symbol_notional=100.0),
        buy_notional_step=0.01,
    )
    assert v.ok is True
    assert v.permitted_buys[("s", "AAA")] == pytest.approx(150.0, abs=0.02)


# --------------------------------------------------------------------------- #
# rounding-safe NOTIONAL distribution (floor to step; Σq <= P; remainder guarded)
# --------------------------------------------------------------------------- #


def test_distribution_floors_each_leg_to_notional_step():
    # Two strategies split a per-symbol P; each leg is FLOORED to the dollar step so Σq <= P.
    lim = _relaxed(max_symbol_notional=0.25)  # cap_sym = 250 on equity 1000
    v = evaluate_book(
        1000.0, {}, {},
        {("s1", "AAA"): 100.0, ("s2", "AAA"): 100.0},  # aggregate 200 < 250, both fit
        lim, buy_notional_step=1.0,
    )
    assert v.ok is True
    for leg in v.permitted_buys.values():
        assert leg == pytest.approx(round(leg), abs=1e-9)  # integer multiple of the $1 step
    assert sum(v.permitted_buys.values()) <= 200.0 + 1e-6


def test_distribution_quantized_book_never_exceeds_cap():
    # A big step forces flooring; the quantized book must still satisfy the binding per-symbol cap.
    lim = _relaxed(max_symbol_notional=0.25)  # cap_sym = 250
    v = evaluate_book(
        1000.0, {}, {},
        {("s1", "AAA"): 137.0, ("s2", "AAA"): 149.0},  # aggregate 286 > 250 -> trimmed to <= 250
        lim, buy_notional_step=10.0,
    )
    assert v.ok is True
    assert sum(v.permitted_buys.values()) <= 250.0 + 1e-6


def test_remainder_step_granted_only_when_revalidation_passes():
    # Three strategies split P over one name; flooring each leg leaves a per-symbol remainder >= one
    # step. cap_sym = 0.4*1000 = 400, aggregate request 600 -> P = 400. Each leg share = 133.33,
    # floored to 100 (step) => Σ granted = 300, remainder = 100 >= step. The remainder step is
    # granted to ONE strategy (re-validation passes: 400 <= cap_sym), so the book lands exactly at
    # 400 with one leg getting the extra step.
    lim = _relaxed(max_symbol_notional=0.4)
    v = evaluate_book(
        1000.0, {}, {},
        {("s1", "AAA"): 200.0, ("s2", "AAA"): 200.0, ("s3", "AAA"): 200.0},
        lim, buy_notional_step=100.0,
    )
    assert v.ok is True
    legs = {k: val for k, val in v.permitted_buys.items()}
    assert sum(legs.values()) == pytest.approx(400.0, abs=1e-9)
    # exactly one strategy got 200, the other two got 100 each (the single remainder step).
    assert sorted(legs.values()) == [pytest.approx(100.0), pytest.approx(100.0),
                                     pytest.approx(200.0)]


def test_remainder_not_granted_when_it_would_breach():
    # cap_sym = 250; step 100; request 300 -> P trimmed to 250, floored to 200, remainder 50 < step
    # -> no grant, quantized book stays at 200 <= 250.
    lim = _relaxed(max_symbol_notional=0.25)
    v = evaluate_book(
        1000.0, {}, {},
        {("s", "AAA"): 300.0},
        lim, buy_notional_step=100.0,
    )
    assert v.ok is True
    assert v.permitted_buys[("s", "AAA")] <= 250.0 + 1e-6


def test_remainder_revalidation_accounts_for_every_symbol_floored_legs():
    # MEDIUM (design-v4 GATE-2): step-5 grants each symbol's remainder step over the WHOLE floored
    # book (every symbol's base legs, not a processed-so-far prefix), and step 6 re-validates the
    # exact quantized book — so a multi-symbol remainder distribution under a TIGHT aggregate gross
    # cap never hard-fails on gross. equity=1000, cap_gross = 1.0*equity = 1000. Two names, two
    # strategies each; step 100. Each name's aggregate request is 78 (per strat 39) *10 scaling ...
    # Concretely: request 780 per name (390/strat) -> step3 p = 500 per name (cap_sym 500) capped by
    # gross: sum_p 1000 == cap_gross, no scale. Floors to 400/name (two legs of 200 each), remainder
    # 100 each -> both remainders granted, landing the book exactly at the 1000 gross cap.
    lim = BookRiskLimits(max_gross=1.0, max_net=1.0, max_symbol_concentration=1.0,
                         max_symbol_notional=0.5)  # cap_gross=1000, cap_sym=500
    v = evaluate_book(
        1000.0, {}, {},
        {("s1", "AAA"): 390.0, ("s2", "AAA"): 390.0,
         ("s1", "BBB"): 390.0, ("s2", "BBB"): 390.0},
        lim, buy_notional_step=100.0,
    )
    assert v.ok is True  # never a step-6 gross hard-fail
    book: dict[str, float] = {}
    for (_strat, sym), val in v.permitted_buys.items():
        book[sym] = book.get(sym, 0.0) + val
        assert val == pytest.approx(round(val / 100.0) * 100.0, abs=1e-9)  # multiple of the step
    # both names land at their 500 cap_sym via the remainder grants; aggregate at the gross cap.
    assert book["AAA"] == pytest.approx(500.0)
    assert book["BBB"] == pytest.approx(500.0)
    assert sum(book.values()) <= 1000.0 + 1e-6  # gross cap respected on the quantized book


def test_min_notional_floor_drops_sub_minimum_legs():
    # A leg quantized into (0, min_notional) is dropped to 0 (PLAN/APPLY agreement).
    lim = _relaxed(max_symbol_notional=100.0)
    v = evaluate_book(
        1000.0, {}, {},
        {("s", "AAA"): 3.0},  # below min_notional
        lim, buy_notional_step=0.01, min_notional=5.0,
    )
    assert v.ok is True
    assert ("s", "AAA") not in v.permitted_buys


def test_leg_above_min_notional_survives():
    lim = _relaxed(max_symbol_notional=100.0)
    v = evaluate_book(
        1000.0, {}, {},
        {("s", "AAA"): 20.0},
        lim, buy_notional_step=0.01, min_notional=5.0,
    )
    assert v.ok is True
    assert v.permitted_buys[("s", "AAA")] == pytest.approx(20.0, abs=0.02)


# --------------------------------------------------------------------------- #
# fixed-D final validation (against D, not the terminal gross ΣQ)
# --------------------------------------------------------------------------- #


def test_final_validation_uses_fixed_D_not_terminal_gross():
    # Show a name bounded by c*D even though the terminal gross would permit MORE. equity 1000,
    # c 0.25 => c*D = c*max(equity, ΣB) = 250. Seed A=200; also BUY six other names (B..G) each to
    # 250 so the terminal gross balloons to ~1750 => 0.25*terminal_gross = 437.5 > 250. If the net
    # validated concentration against the terminal gross it would let A rise to 437; the FIXED-D net
    # holds A at 250 (add only 50) — certifying every SELL-reduced intermediate prefix.
    lim = BookRiskLimits(max_gross=2.0, max_net=2.0,
                         max_symbol_concentration=0.25, max_symbol_notional=1.0)
    req = {("s", "A"): 1000.0}
    for sym in ("B", "C", "D", "E", "F", "G"):
        req[("s", sym)] = 250.0
    v = evaluate_book(1000.0, {"A": 200.0}, {}, req, lim, buy_notional_step=0.01)
    assert v.ok is True
    final_a = 200.0 + v.permitted_buys.get(("s", "A"), 0.0)
    assert final_a == pytest.approx(250.0, abs=0.05)  # bounded by c*D, not 0.25*terminal_gross
    terminal_gross = 200.0 + sum(v.permitted_buys.values())
    assert 0.25 * terminal_gross > 250.0  # the friendlier basis WOULD have allowed more


def test_clean_pass_returns_ok_none_reason():
    v = evaluate_book(
        1_000_000.0, {}, {},
        {("s", "AAA"): 100.0},
        BookRiskLimits(),
        buy_notional_step=0.01,
    )
    assert isinstance(v, BookVerdict)
    assert v.ok is True
    assert v.reason is None
    assert v.permitted_buys[("s", "AAA")] == pytest.approx(100.0, abs=0.02)


def test_verdict_is_frozen():
    v = evaluate_book(1000.0, {}, {}, {}, BookRiskLimits())
    with pytest.raises(dataclasses.FrozenInstanceError):
        v.ok = False  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# empty inputs and infinite-concentration disable
# --------------------------------------------------------------------------- #


def test_empty_cycle_is_a_clean_no_op_pass():
    v = evaluate_book(1000.0, {}, {}, {}, BookRiskLimits())
    assert v.ok is True
    assert v.reason is None
    assert v.permitted_buys == {}


def test_concentration_disabled_at_c_equals_one():
    # c == 1.0 disables the concentration cap; only gross / per-symbol notional bind.
    lim = _relaxed(max_symbol_concentration=1.0, max_symbol_notional=100.0,
                   max_gross=100.0, max_net=100.0)
    v = evaluate_book(
        1000.0, {}, {},
        {("s", "AAA"): 900.0},
        lim, buy_notional_step=0.01,
    )
    assert v.ok is True
    assert v.permitted_buys[("s", "AAA")] == pytest.approx(900.0, abs=0.02)


def test_scalar_and_per_symbol_step_both_accepted():
    lim = _relaxed()
    scalar = evaluate_book(1000.0, {}, {}, {("s", "A"): 50.0}, lim, buy_notional_step=0.01)
    per_symbol = evaluate_book(
        1000.0, {}, {}, {("s", "A"): 50.0}, lim, buy_notional_step={"A": 0.01}
    )
    assert scalar.permitted_buys[("s", "A")] == pytest.approx(
        per_symbol.permitted_buys[("s", "A")]
    )
    assert not math.isnan(scalar.permitted_buys[("s", "A")])
