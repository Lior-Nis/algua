from datetime import date

import pytest

from algua.data.constituents import (
    ConstituentInterval,
    constituents_to_snapshots,
    parse_constituents_rows,
)


def test_open_interval_is_a_survivor():
    ivs = [ConstituentInterval("AAPL", date(1998, 1, 2), None)]
    assert constituents_to_snapshots(ivs) == [(date(1998, 1, 2), frozenset({"AAPL"}))]


def test_closed_interval_drops_at_drop_date_exclusive():
    ivs = [ConstituentInterval("ENRN", date(1998, 1, 2), date(2001, 11, 28))]
    tl = constituents_to_snapshots(ivs)
    assert tl == [
        (date(1998, 1, 2), frozenset({"ENRN"})),
        (date(2001, 11, 28), frozenset()),
    ]


def test_simultaneous_add_and_drop_is_one_snapshot():
    ivs = [
        ConstituentInterval("A", date(2000, 1, 1), date(2005, 1, 1)),
        ConstituentInterval("B", date(2005, 1, 1), None),
    ]
    tl = constituents_to_snapshots(ivs)
    assert tl == [
        (date(2000, 1, 1), frozenset({"A"})),
        (date(2005, 1, 1), frozenset({"B"})),
    ]


def test_re_addition_two_intervals():
    ivs = [
        ConstituentInterval("XYZ", date(2005, 3, 1), date(2009, 6, 15)),
        ConstituentInterval("XYZ", date(2012, 1, 1), None),
    ]
    tl = constituents_to_snapshots(ivs)
    assert tl == [
        (date(2005, 3, 1), frozenset({"XYZ"})),
        (date(2009, 6, 15), frozenset()),
        (date(2012, 1, 1), frozenset({"XYZ"})),
    ]


def test_no_op_change_dates_collapsed():
    ivs = [
        ConstituentInterval("A", date(2000, 1, 1), None),
        ConstituentInterval("B", date(2000, 1, 1), date(2001, 1, 1)),
        ConstituentInterval("B", date(2001, 1, 1), None),  # re-add same day it drops
    ]
    tl = constituents_to_snapshots(ivs)
    assert tl == [(date(2000, 1, 1), frozenset({"A", "B"}))]


def test_overlapping_intervals_rejected():
    ivs = [
        ConstituentInterval("A", date(2000, 1, 1), date(2003, 1, 1)),
        ConstituentInterval("A", date(2002, 1, 1), None),
    ]
    with pytest.raises(ValueError, match="overlap"):
        constituents_to_snapshots(ivs)


def test_open_interval_followed_by_another_rejected():
    ivs = [
        ConstituentInterval("A", date(2000, 1, 1), None),
        ConstituentInterval("A", date(2010, 1, 1), None),
    ]
    with pytest.raises(ValueError, match="overlap"):
        constituents_to_snapshots(ivs)


def test_add_after_drop_rejected():
    with pytest.raises(ValueError, match="add_date.*<=.*drop_date|add_date must be"):
        parse_constituents_rows(
            [{"symbol": "A", "add_date": "2005-01-01", "drop_date": "2004-01-01"}]
        )


def test_zero_length_interval_rejected():
    with pytest.raises(ValueError, match="zero-length|add_date == drop_date"):
        parse_constituents_rows(
            [{"symbol": "A", "add_date": "2005-01-01", "drop_date": "2005-01-01"}]
        )


def test_symbols_normalized_before_dedup():
    rows = [
        {"symbol": " aapl ", "add_date": "1998-01-02", "drop_date": ""},
        {"symbol": "AAPL", "add_date": "1998-01-02", "drop_date": ""},  # dup after normalize
    ]
    ivs = parse_constituents_rows(rows)
    assert ivs == [ConstituentInterval("AAPL", date(1998, 1, 2), None)]


def test_malformed_row_rejected():
    with pytest.raises(ValueError):
        parse_constituents_rows([{"symbol": "", "add_date": "1998-01-02", "drop_date": ""}])
    with pytest.raises(ValueError):
        parse_constituents_rows([{"symbol": "A", "add_date": "not-a-date", "drop_date": ""}])
    with pytest.raises(ValueError):
        parse_constituents_rows([{"symbol": "A", "add_date": "2005-01-01", "drop_date": "bad"}])
