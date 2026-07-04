"""INTERIM book-level aggregate risk wiring in live `run-all` (#389 Task-1 slice).

Covers `_build_interim_book_headroom` (seed / fail-closed preconditions) and the compose into
`_reserve_for` (the interim trimmer caps a BUY across strategies on the shared account). The pure
whole-cycle cap math (incl. the prefix-safe single-name CONCENTRATION cap) is covered by
`tests/test_book_limits.py`; here we test the INTERIM CLI wiring — the two prefix-safe monotone
caps (aggregate gross/net + single-name notional) and the fail-closed cycle skip. The concentration
cap is deliberately NOT part of the interim path (prefix-unsafe BUY-by-BUY), so it is not tested
here — it lands with the `evaluate_book` wiring slice.
"""
from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from algua.cli import live_cmd
from algua.cli.live_cmd import _InterimBookHeadroom

# A fully-closed session date the seed marks are dated on, and a cycle clock strictly AFTER it so
# those bars survive the closed-session cutoff. Mirrors run-all threading a single per-cycle `now`.
_CLOSED = datetime(2023, 6, 1, tzinfo=UTC)
_NOW = datetime(2023, 6, 2, tzinfo=UTC)


class _Acct:
    def __init__(self, equity: float, buying_power: float) -> None:
        self.equity = equity
        self.buying_power = buying_power


class _Broker:
    def __init__(self, equity: float = 100_000.0, buying_power: float = 100_000.0) -> None:
        self._acct = _Acct(equity, buying_power)

    def account(self) -> _Acct:
        return self._acct


class _Provider:
    """Returns a bars frame for the requested symbols: ONE fully-closed bar per symbol dated at
    ``_CLOSED``, matching the real provider's DatetimeIndex + 'symbol'/'close' shape."""

    def __init__(self, marks: dict[str, float]) -> None:
        self._marks = marks

    def get_bars(self, symbols, start, end, timeframe):  # noqa: ANN001, ARG002
        rows = [
            {"symbol": s, "close": self._marks[s]}
            for s in symbols
            if s in self._marks
        ]
        return pd.DataFrame(rows, index=pd.DatetimeIndex([_CLOSED] * len(rows)))


class _PartialBarProvider:
    """Returns TWO bars per symbol: the last fully-closed bar (dated ``_CLOSED``) and a later,
    still-forming partial bar dated at/after the cycle cutoff. The partial bar carries a DIFFERENT
    (typically lower) close and MUST be dropped before valuation — otherwise it would look-ahead
    into an intraday price (#389 GATE-2)."""

    def __init__(self, closed: dict[str, float], partial: dict[str, float],
                 partial_date: datetime = _NOW) -> None:
        self._closed = closed
        self._partial = partial
        self._partial_date = partial_date

    def get_bars(self, symbols, start, end, timeframe):  # noqa: ANN001, ARG002
        rows = []
        index = []
        for s in symbols:
            if s in self._closed:
                rows.append({"symbol": s, "close": self._closed[s]})
                index.append(_CLOSED)
            if s in self._partial:
                rows.append({"symbol": s, "close": self._partial[s]})
                index.append(self._partial_date)
        return pd.DataFrame(rows, index=pd.DatetimeIndex(index))


# --------------------------------------------------------------------------- #
# _build_interim_book_headroom — seed + fail-closed preconditions
# --------------------------------------------------------------------------- #


def test_build_interim_book_seeds_from_reconciled_positions_and_equity():
    broker = _Broker(equity=100_000.0)
    provider = _Provider({"AAA": 10.0, "BBB": 20.0})
    net_positions = {"AAA": 100.0, "BBB": 50.0}  # 100*10 + 50*20 = 1000 + 1000 = 2000 gross
    book, reason = live_cmd._build_interim_book_headroom(
        broker, provider, net_positions, "2023-01-01", "2023-12-31", _NOW
    )
    assert reason is None
    assert book is not None
    assert book._gross == pytest.approx(2000.0)
    assert book._book["AAA"] == pytest.approx(1000.0)
    # cap_gross = min(2.0, 1.0) * 100_000 = 100_000; cap_sym = 0.5 * 100_000 = 50_000.
    assert book._cap_gross == pytest.approx(100_000.0)
    assert book._cap_sym == pytest.approx(50_000.0)


def test_build_interim_book_empty_account_is_valid_empty_book():
    book, reason = live_cmd._build_interim_book_headroom(
        _Broker(equity=50_000.0), _Provider({}), {}, "2023-01-01", "2023-12-31", _NOW
    )
    assert reason is None
    assert book is not None
    assert book._gross == 0.0


def test_build_interim_book_fails_closed_on_short_position():
    # A short in the live account violates the long-only precondition -> fail closed (skip cycle).
    book, reason = live_cmd._build_interim_book_headroom(
        _Broker(), _Provider({"AAA": 10.0}), {"AAA": -100.0}, "2023-01-01", "2023-12-31", _NOW
    )
    assert book is None
    assert reason is not None and "short" in reason.lower()


def test_build_interim_book_fails_closed_on_missing_mark():
    # A held symbol with no usable mark makes the whole book unvaluable -> fail closed.
    book, reason = live_cmd._build_interim_book_headroom(
        _Broker(), _Provider({}), {"AAA": 100.0}, "2023-01-01", "2023-12-31", _NOW
    )
    assert book is None
    assert reason is not None and "mark" in reason.lower()


@pytest.mark.parametrize("bad_mark", [0.0, -5.0, float("nan"), float("inf")])
def test_build_interim_book_fails_closed_on_non_positive_or_non_finite_mark(bad_mark):
    book, reason = live_cmd._build_interim_book_headroom(
        _Broker(), _Provider({"AAA": bad_mark}), {"AAA": 100.0}, "2023-01-01", "2023-12-31", _NOW
    )
    assert book is None
    assert reason is not None


@pytest.mark.parametrize("bad_equity", [0.0, -1.0, float("nan"), float("inf")])
def test_build_interim_book_fails_closed_on_unusable_equity(bad_equity):
    book, reason = live_cmd._build_interim_book_headroom(
        _Broker(equity=bad_equity), _Provider({}), {}, "2023-01-01", "2023-12-31", _NOW
    )
    assert book is None
    assert reason is not None and "equity" in reason.lower()


def test_build_interim_book_fails_closed_on_already_breached_symbol_notional():
    # Seed a single name over its per-symbol notional cap (0.5*equity): AAA notional = 600 on a
    # 1000 equity account (max_symbol_notional default 0.5 => 500). An already-breached book is an
    # anomaly -> fail closed (a buy of ANOTHER symbol must NOT proceed through a breached book).
    book, reason = live_cmd._build_interim_book_headroom(
        _Broker(equity=1000.0), _Provider({"AAA": 6.0}), {"AAA": 100.0},
        "2023-01-01", "2023-12-31", _NOW,
    )
    assert book is None
    assert reason is not None and "already breaches" in reason.lower()


def test_build_interim_book_fails_closed_on_already_breached_concentration():
    # A seed book over the single-name CONCENTRATION cap but UNDER both the gross and per-symbol
    # notional caps must still fail closed at reconcile. This is the prefix-INDEPENDENT t0 check on
    # the fixed reconciled book (mirrors evaluate_book's seed_breach:concentration branch), NOT the
    # deferred BUY-by-BUY concentration trim. equity 1000 -> cap_gross = min(2,1)*1000 = 1000;
    # cap_sym = 0.5*1000 = 500; conc denom = max(gross, equity) = max(400, 1000) = 1000;
    # conc_cap = 0.25*1000 = 250. AAA notional = 100*4 = 400: under gross (400<=1000) and under the
    # per-name notional cap (400<=500), but over concentration (400 > 250) -> fail closed.
    book, reason = live_cmd._build_interim_book_headroom(
        _Broker(equity=1000.0), _Provider({"AAA": 4.0}), {"AAA": 100.0},
        "2023-01-01", "2023-12-31", _NOW,
    )
    assert book is None
    assert reason is not None and "concentration" in reason.lower()


def test_build_interim_book_fails_closed_on_already_breached_gross():
    # cap_gross = min(2,1)*1000 = 1000; cap_sym = 0.5*1000 = 500. Three names each 450 notional
    # (under the 500 per-name cap) sum to 1350 gross > 1000 -> the aggregate gross cap trips first
    # (each name individually is fine), so the whole cycle fails closed at reconcile.
    book, reason = live_cmd._build_interim_book_headroom(
        _Broker(equity=1000.0), _Provider({"AAA": 4.5, "BBB": 4.5, "CCC": 4.5}),
        {"AAA": 100.0, "BBB": 100.0, "CCC": 100.0}, "2023-01-01", "2023-12-31", _NOW,
    )
    assert book is None
    assert reason is not None and "gross" in reason.lower()


# --------------------------------------------------------------------------- #
# closed-session cutoff — the seed valuation must ignore a partial current bar (#389 GATE-2)
# --------------------------------------------------------------------------- #


def test_seed_ignores_partial_current_bar_valuation():
    # A later, partial (lower-priced) bar for the held name must NOT change the seed valuation vs
    # the last CLOSED bar. Closed bar close = 10 -> notional 100*10 = 1000; a partial bar dated at
    # the cutoff carries close = 5 (would value the book at 500) but must be dropped.
    broker = _Broker(equity=100_000.0)
    provider = _PartialBarProvider(closed={"AAA": 10.0}, partial={"AAA": 5.0})
    book, reason = live_cmd._build_interim_book_headroom(
        broker, provider, {"AAA": 100.0}, "2023-01-01", "2023-12-31", _NOW
    )
    assert reason is None
    assert book is not None
    # Valued on the CLOSED bar (10), not the partial (5): notional 1000, not 500.
    assert book._book["AAA"] == pytest.approx(1000.0)
    assert book._gross == pytest.approx(1000.0)


def test_seed_over_cap_on_closed_bar_fails_even_when_partial_bar_would_hide_it():
    # A seed ALREADY over the single-name notional cap on its last CLOSED bar must fail the
    # reconcile-time seed check EVEN THOUGH a later partial (lower-priced) bar would value it under
    # the cap and hide the breach. equity 1000 -> cap_sym = 0.5*1000 = 500. Closed close = 6 ->
    # notional 100*6 = 600 (> 500, breach). A partial close = 4 -> 400 (< 500) would mask it.
    broker = _Broker(equity=1000.0)
    provider = _PartialBarProvider(closed={"AAA": 6.0}, partial={"AAA": 4.0})
    book, reason = live_cmd._build_interim_book_headroom(
        broker, provider, {"AAA": 100.0}, "2023-01-01", "2023-12-31", _NOW
    )
    assert book is None
    assert reason is not None and "already breaches" in reason.lower()


class _DescendingProvider:
    """Returns TWO fully-closed bars per symbol in DESCENDING timestamp order (latest row FIRST,
    oldest row LAST). Both bars are dated strictly before the cycle cutoff so
    `drop_open_session_bars` keeps both. Exercises the #389 GATE-2 look-ahead fix:
    `groupby(...).last()` returns the last row in FRAME order, so without sorting the seed would be
    valued off the OLDEST bar (last row here), not the true latest closed bar."""

    _LATER = datetime(2023, 5, 31, tzinfo=UTC)
    _EARLIER = datetime(2023, 5, 30, tzinfo=UTC)

    def __init__(self, latest: dict[str, float], earlier: dict[str, float]) -> None:
        self._latest = latest
        self._earlier = earlier

    def get_bars(self, symbols, start, end, timeframe):  # noqa: ANN001, ARG002
        rows = []
        index = []
        for s in symbols:
            # descending: emit the LATER (true-latest) bar first, then the EARLIER bar last
            if s in self._latest:
                rows.append({"symbol": s, "close": self._latest[s]})
                index.append(self._LATER)
            if s in self._earlier:
                rows.append({"symbol": s, "close": self._earlier[s]})
                index.append(self._EARLIER)
        return pd.DataFrame(rows, index=pd.DatetimeIndex(index))


def test_seed_uses_true_latest_bar_under_descending_provider_order():
    # #389 GATE-2: a descending-order provider frame must still be valued off the TRUE latest closed
    # bar. equity 1000 -> cap_sym = 0.5*1000 = 500. Latest closed close = 6 -> notional 100*6 = 600
    # (OVER the 500 cap => must fail closed). The earlier bar close = 4 -> 400 (under cap); if the
    # seed were (wrongly) valued off the last FRAME row (the earlier bar) it would pass and mask the
    # breach. The sort_index() fix in _latest_marks makes it fail closed.
    broker = _Broker(equity=1000.0)
    provider = _DescendingProvider(latest={"AAA": 6.0}, earlier={"AAA": 4.0})
    book, reason = live_cmd._build_interim_book_headroom(
        broker, provider, {"AAA": 100.0}, "2023-01-01", "2023-12-31", _NOW
    )
    assert book is None
    assert reason is not None and "already breaches" in reason.lower()


def test_build_interim_book_fails_closed_on_non_finite_reconciled_qty():
    # #389 GATE-2 (MEDIUM): a NaN reconciled qty survives the `!= 0.0` filter and the `q < 0` short
    # check, then yields a NaN notional that makes every seed-breach comparison return False — an
    # unvaluable book must fail closed, not hand back a usable trimmer. The mark is valid, so this
    # isolates the qty guard.
    broker = _Broker(equity=100_000.0)
    provider = _Provider({"AAA": 10.0})
    book, reason = live_cmd._build_interim_book_headroom(
        broker, provider, {"AAA": float("nan")}, "2023-01-01", "2023-12-31", _NOW
    )
    assert book is None
    assert reason is not None and "non-finite" in reason.lower()


# --------------------------------------------------------------------------- #
# _InterimBookHeadroom.permit_buy + compose into _reserve_for
# --------------------------------------------------------------------------- #


def test_interim_book_trims_second_strategy_buy_of_same_name(monkeypatch):
    """Two strategies each intend to BUY the same name. Individually each is within its own
    subaccount limit, but the book-level single-name NOTIONAL cap trims the SECOND strategy's buy so
    the aggregate never breaches. This is the #389 compounding failure the interim layer stops."""
    # Empty long-only book, cap_sym = 250 across BOTH strategies, cap_gross large (only the
    # single-name notional cap binds).
    book = _InterimBookHeadroom({}, cap_gross=1e12, cap_sym=250.0)
    recorded: list[tuple] = []

    # Replicate the run-all closure shape (pool huge so only the book binds).
    pool = {"available": 1e12}

    def _reserve_for(strategy_name):
        def _reserve(symbol: str, notional: float) -> float:
            pool_permitted = min(notional, max(0.0, pool["available"]))
            permitted = book.permit_buy(symbol, pool_permitted)
            pool["available"] -= permitted
            if permitted < notional:
                recorded.append((strategy_name, symbol, notional, permitted))
            return permitted
        return _reserve

    r1 = _reserve_for("s1")
    r2 = _reserve_for("s2")
    # Strategy 1 buys 200 of AAA -> fully permitted (under the 250 book cap).
    assert r1("AAA", 200.0) == pytest.approx(200.0)
    # Strategy 2 buys another 200 of AAA -> only 50 of headroom left (250 book cap), TRIMMED.
    assert r2("AAA", 200.0) == pytest.approx(50.0)
    assert book._book["AAA"] == pytest.approx(250.0)
    # Strategy 2's trim was audited as a shortfall.
    assert any(strat == "s2" and sym == "AAA" and permitted == pytest.approx(50.0)
               for strat, sym, _intended, permitted in recorded)


def test_interim_book_gross_cap_trims_across_symbols():
    """The aggregate gross cap binds across DIFFERENT names too — once the running book fills the
    gross headroom, a later BUY of an unrelated name is trimmed to zero."""
    book = _InterimBookHeadroom({}, cap_gross=300.0, cap_sym=1e12)
    assert book.permit_buy("AAA", 200.0) == pytest.approx(200.0)
    assert book.permit_buy("BBB", 200.0) == pytest.approx(100.0)  # only 100 gross headroom left
    assert book.permit_buy("CCC", 50.0) == pytest.approx(0.0)  # gross exhausted
    assert book._gross == pytest.approx(300.0)


def test_interim_book_and_pool_compose_take_the_min():
    """When the buying-power pool is the tighter constraint, the pool trim wins; the book mutates
    by the FINAL (post-pool) permitted amount, not the raw intended notional."""
    book = _InterimBookHeadroom({}, cap_gross=1e12, cap_sym=1e12)  # no book cap binds
    pool = {"available": 30.0}  # pool is the binding constraint

    def _reserve(symbol: str, notional: float) -> float:
        pool_permitted = min(notional, max(0.0, pool["available"]))
        permitted = book.permit_buy(symbol, pool_permitted)
        pool["available"] -= permitted
        return permitted

    # Ask 100; pool only has 30 -> permitted 30; book mutated by 30 (the final amount), not 100.
    assert _reserve("AAA", 100.0) == pytest.approx(30.0)
    assert book._book["AAA"] == pytest.approx(30.0)
    assert pool["available"] == pytest.approx(0.0)


def test_interim_book_permit_buy_rejects_non_positive_request():
    book = _InterimBookHeadroom({}, cap_gross=1e12, cap_sym=1e12)
    assert book.permit_buy("AAA", 0.0) == 0.0
    assert book.permit_buy("AAA", -5.0) == 0.0
    assert book._gross == 0.0
