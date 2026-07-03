"""Tests for the account-wide high-water mark persistence (#390)."""

from __future__ import annotations

import math

import pytest

from algua.registry.db import connect, migrate
from algua.risk.book_equity import clear_book_peak, get_book_peak, update_book_peak


def _conn(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    return conn


def test_empty_peak_is_none(tmp_path):
    assert get_book_peak(_conn(tmp_path)) is None


def test_update_ratchets_up_only(tmp_path):
    conn = _conn(tmp_path)
    assert update_book_peak(conn, 100_000.0) == 100_000.0
    assert update_book_peak(conn, 120_000.0) == 120_000.0
    # a lower equity does NOT lower the peak
    assert update_book_peak(conn, 90_000.0) == 120_000.0
    assert get_book_peak(conn) == 120_000.0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.0, -1.0])
def test_update_rejects_unusable_equity(tmp_path, bad):
    conn = _conn(tmp_path)
    with pytest.raises(ValueError):
        update_book_peak(conn, bad)
    # nothing was written
    assert get_book_peak(conn) is None


def test_single_row(tmp_path):
    conn = _conn(tmp_path)
    update_book_peak(conn, 100_000.0)
    update_book_peak(conn, 110_000.0)
    n = conn.execute("SELECT COUNT(*) AS c FROM book_equity_peak").fetchone()["c"]
    assert n == 1


def test_clear(tmp_path):
    conn = _conn(tmp_path)
    update_book_peak(conn, 100_000.0)
    clear_book_peak(conn)
    assert get_book_peak(conn) is None
    # clearing an empty table is a no-op
    clear_book_peak(conn)
    assert get_book_peak(conn) is None
    assert math.isfinite(1.0)
