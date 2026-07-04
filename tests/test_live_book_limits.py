"""Book-level aggregate risk wiring in live `run-all` (#389).

Covers `_build_book_exposure` (seed / fail-closed preconditions) and the compose into
`_reserve_for` (the book trims a BUY across strategies on the shared account). The pure cap math
is covered by `tests/test_book_limits.py`; here we test the CLI wiring + fail-closed cycle skip.
"""
from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from algua.cli import live_cmd
from algua.risk.limits import RiskBreach

# Fixed session dates + a `now` where all three sessions are fully closed (staleness 0) so the
# shared mark-freshness wall (#452 HIGH#2) passes deterministically regardless of the real clock.
_DATES = [datetime(2023, 1, d, tzinfo=UTC) for d in (2, 3, 4)]
_NOW = datetime(2023, 1, 5, tzinfo=UTC)


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
    """Returns a timestamp-indexed bars frame (one row per session per requested+known symbol) so
    the null-preserving latest-row selection + freshness wall (#452) read a real dated close."""

    def __init__(self, marks: dict[str, float]) -> None:
        self._marks = marks

    def get_bars(self, symbols, start, end, timeframe):  # noqa: ANN001, ARG002
        rows = [
            {"timestamp": ts, "symbol": s, "close": self._marks[s]}
            for s in symbols
            if s in self._marks
            for ts in _DATES
        ]
        if not rows:
            return pd.DataFrame(columns=["symbol", "close"])
        return pd.DataFrame(rows).set_index("timestamp")


# --------------------------------------------------------------------------- #
# _build_book_exposure — seed + fail-closed preconditions
# --------------------------------------------------------------------------- #


def test_build_book_exposure_seeds_from_reconciled_positions_and_equity():
    broker = _Broker(equity=100_000.0)
    provider = _Provider({"AAA": 10.0, "BBB": 20.0})
    net_positions = {"AAA": 100.0, "BBB": 50.0}  # 100*10 + 50*20 = 1000 + 1000 = 2000 gross
    book, reason = live_cmd._build_book_exposure(
        broker, provider, net_positions, "2023-01-01", "2023-12-31", now=_NOW
    )
    assert reason is None
    assert book is not None
    assert book.equity == pytest.approx(100_000.0)
    assert book.gross == pytest.approx(2000.0)
    assert book.net == pytest.approx(2000.0)
    assert book.book["AAA"] == pytest.approx(1000.0)


def test_build_book_exposure_empty_account_is_valid_empty_book():
    book, reason = live_cmd._build_book_exposure(
        _Broker(equity=50_000.0), _Provider({}), {}, "2023-01-01", "2023-12-31", now=_NOW
    )
    assert reason is None
    assert book is not None
    assert book.gross == 0.0 and book.net == 0.0


def test_build_book_exposure_fails_closed_on_short_position():
    # A short in the live account violates the long-only precondition -> BENIGN defer (skip cycle),
    # NOT a data-integrity RiskBreach (this is a policy/economic state, checked before the wall).
    book, reason = live_cmd._build_book_exposure(
        _Broker(), _Provider({"AAA": 10.0}), {"AAA": -100.0}, "2023-01-01", "2023-12-31", now=_NOW
    )
    assert book is None
    assert reason is not None and "short" in reason.lower()


def test_build_book_exposure_raises_on_missing_mark():
    # #452 HIGH#2: a held symbol with no bar at all is a DATA-INTEGRITY failure -> the shared wall
    # raises RiskBreach('stale_marks', no_mark => infinite staleness), which run-all HALTS on.
    with pytest.raises(RiskBreach) as exc:
        live_cmd._build_book_exposure(
            _Broker(), _Provider({}), {"AAA": 100.0}, "2023-01-01", "2023-12-31", now=_NOW
        )
    assert exc.value.kind == "stale_marks"


@pytest.mark.parametrize("bad_mark", [0.0, -5.0, float("nan"), float("inf")])
def test_build_book_exposure_raises_on_non_positive_or_non_finite_mark(bad_mark):
    # #452 HIGH#2: a non-positive / non-finite latest mark is unvaluable -> RiskBreach (dark feed),
    # routed by run-all to a HALT (no flatten), not a benign (None, reason) defer.
    with pytest.raises(RiskBreach) as exc:
        live_cmd._build_book_exposure(
            _Broker(), _Provider({"AAA": bad_mark}), {"AAA": 100.0},
            "2023-01-01", "2023-12-31", now=_NOW,
        )
    assert exc.value.kind == "unvaluable_marks"


def test_build_book_exposure_fails_closed_on_already_breached_seed():
    # Seed a single name over its per-symbol notional cap (0.5*equity): AAA notional = 600 on a
    # 1000 equity account (max_symbol_notional default 0.5 => 500). An already-breached book is an
    # anomaly -> BENIGN defer (a buy of ANOTHER symbol must NOT proceed through a breached book).
    # The mark is fresh + valuable, so this is an economic state, NOT a data-integrity RiskBreach.
    book, reason = live_cmd._build_book_exposure(
        _Broker(equity=1000.0), _Provider({"AAA": 6.0}), {"AAA": 100.0},
        "2023-01-01", "2023-12-31", now=_NOW,
    )
    assert book is None
    assert reason is not None and "already breaches" in reason.lower()


# --------------------------------------------------------------------------- #
# compose into _reserve_for — the book trims a BUY across strategies
# --------------------------------------------------------------------------- #


def test_book_trims_second_strategy_buy_of_same_name(monkeypatch):
    """Two strategies each intend to BUY the same name. Individually each is within its own
    subaccount limit, but the book-level single-name concentration cap trims the SECOND strategy's
    buy so the aggregate never breaches. This is the #389 compounding failure the layer stops."""
    from algua.risk.book_limits import BookExposure, BookRiskLimits

    # Empty long-only book, equity 1000. Concentration cap 0.25 with equity floor => a single name
    # is capped at 0.25*1000 = 250 total across BOTH strategies.
    book = BookExposure(1000.0, {}, BookRiskLimits(max_gross=10.0, max_net=10.0,
                                                   max_symbol_concentration=0.25,
                                                   max_symbol_notional=10.0))
    recorded: list[tuple] = []
    monkeypatch.setattr(live_cmd, "record_reservation",
                        lambda conn, cycle, strat, sym, intended, permitted:
                        recorded.append((strat, sym, intended, permitted)))

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
    assert book.book["AAA"] == pytest.approx(250.0)
    # Strategy 2's trim was audited as a shortfall.
    assert any(strat == "s2" and sym == "AAA" and permitted == pytest.approx(50.0)
               for strat, sym, _intended, permitted in recorded)


def test_book_and_pool_compose_take_the_min(monkeypatch):
    """When the buying-power pool is the tighter constraint, the pool trim wins; when the book is
    tighter, the book trim wins. Book mutates by the FINAL (post-pool) permitted amount."""
    from algua.risk.book_limits import BookExposure, BookRiskLimits

    book = BookExposure(1000.0, {}, BookRiskLimits(max_gross=10.0, max_net=10.0,
                                                   max_symbol_concentration=1.0,  # no conc cap
                                                   max_symbol_notional=10.0))
    monkeypatch.setattr(live_cmd, "record_reservation", lambda *a, **k: None)
    pool = {"available": 30.0}  # pool is the binding constraint

    def _reserve(symbol: str, notional: float) -> float:
        pool_permitted = min(notional, max(0.0, pool["available"]))
        permitted = book.permit_buy(symbol, pool_permitted)
        pool["available"] -= permitted
        return permitted

    # Ask 100; pool only has 30 -> permitted 30; book mutated by 30 (the final amount), not 100.
    assert _reserve("AAA", 100.0) == pytest.approx(30.0)
    assert book.book["AAA"] == pytest.approx(30.0)
    assert pool["available"] == pytest.approx(0.0)
