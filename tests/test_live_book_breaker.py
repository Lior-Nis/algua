"""Book-level LOSS circuit-breaker wiring in live `run-all` (#390).

Covers `_evaluate_book_loss_breaker` (the seam that reads one account snapshot, ratchets the account
high-water mark, sources the daily baseline from the broker's prior-session close, and validates
equity BEFORE mutating the peak) plus the resume-all book-peak re-base. The pure decision math is in
`tests/test_book_breaker.py`; the peak persistence in `tests/test_book_equity.py`.
"""
from __future__ import annotations

import pytest

from algua.cli import live_cmd
from algua.config.settings import get_settings
from algua.registry.db import connect, migrate
from algua.risk.book_breaker import BookBreach
from algua.risk.book_equity import get_book_peak, update_book_peak


class _Acct:
    def __init__(self, equity: float, last_equity: float) -> None:
        self.equity = equity
        self.last_equity = last_equity


class _Broker:
    def __init__(self, equity: float, last_equity: float) -> None:
        self._acct = _Acct(equity, last_equity)

    def account(self) -> _Acct:
        return self._acct


def _conn(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    return conn


def test_clean_book_ratchets_peak_and_returns_none(tmp_path):
    conn = _conn(tmp_path)
    broker = _Broker(equity=100_000.0, last_equity=100_000.0)
    assert live_cmd._evaluate_book_loss_breaker(conn, broker) is None
    # the high-water mark was ratcheted to include this cycle
    assert get_book_peak(conn) == 100_000.0


def test_drawdown_breach_from_ratcheted_peak(tmp_path):
    conn = _conn(tmp_path)
    s = get_settings()
    # seed a prior peak well above current equity so book drawdown trips (daily baseline high too)
    update_book_peak(conn, 200_000.0)
    equity = 200_000.0 * (1.0 - s.book_max_drawdown) - 1.0  # just past the drawdown cap
    breach = live_cmd._evaluate_book_loss_breaker(conn, _Broker(equity, 200_000.0))
    assert isinstance(breach, BookBreach)
    assert breach.kind == "book_drawdown"


def test_daily_loss_breach_from_last_equity(tmp_path):
    conn = _conn(tmp_path)
    s = get_settings()
    last_equity = 100_000.0
    equity = last_equity * (1.0 - s.book_max_daily_loss) - 1.0  # just past daily cap
    breach = live_cmd._evaluate_book_loss_breaker(conn, _Broker(equity, last_equity))
    assert isinstance(breach, BookBreach)
    assert breach.kind == "book_daily_loss"


@pytest.mark.parametrize("equity", [0.0, -1.0, float("nan"), float("inf")])
def test_unusable_equity_fails_closed_without_touching_peak(tmp_path, equity):
    conn = _conn(tmp_path)
    update_book_peak(conn, 100_000.0)  # a healthy prior peak
    breach = live_cmd._evaluate_book_loss_breaker(conn, _Broker(equity, 100_000.0))
    assert isinstance(breach, BookBreach)
    assert breach.kind == "book_equity_unusable"
    # GATE-1 correction: a bad equity read must NOT corrupt the high-water mark
    assert get_book_peak(conn) == 100_000.0


def test_unusable_baseline_fails_closed(tmp_path):
    conn = _conn(tmp_path)
    # broker gave no prior-session close -> daily baseline cannot be established -> fail closed
    breach = live_cmd._evaluate_book_loss_breaker(conn, _Broker(100_000.0, 0.0))
    assert isinstance(breach, BookBreach)
    assert breach.kind == "book_baseline_unusable"


def test_broker_account_read_failure_fails_closed(tmp_path):
    # A BrokerError reading/parsing the account must become a fail-closed breach (GATE-2), NOT fall
    # through to a retryable JSON error — an unvaluable book must engage the persistent halt.
    from algua.execution.alpaca_broker import BrokerError

    conn = _conn(tmp_path)

    class _BadBroker:
        def account(self):
            raise BrokerError("alpaca /v2/account: bad or missing field 'equity'")

    breach = live_cmd._evaluate_book_loss_breaker(conn, _BadBroker())
    assert isinstance(breach, BookBreach)
    assert breach.kind == "book_account_read_failed"
    # a failed read never mutates the high-water mark
    assert get_book_peak(conn) is None
