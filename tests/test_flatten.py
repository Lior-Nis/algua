"""Unit tests for the single-sourced emergency flatten helper (#336 / #449).

The live-lane held-cap (Fork B, #449) is single-sourced in ``algua.execution.flatten`` so it is
tested here directly, driving ``flatten_strategy`` with an injected ``held`` callable and a fake
``OffsetBroker`` that records every ``submit_offset`` call. Believed positions are stubbed at the
helper's own resolution site (``algua.execution.flatten.believed_positions``)."""
from __future__ import annotations

import sqlite3
from contextlib import closing

import pytest

from algua.audit.log import read as audit_read
from algua.execution.flatten import flatten_strategy
from algua.execution.live_ledger import LedgerKind
from algua.registry.db import connect, migrate


@pytest.fixture
def conn(tmp_path):
    with closing(connect(tmp_path / "f.db")) as c:
        migrate(c)
        yield c


class _FakeOffsetBroker:
    """Records every submit_offset call; returns a deterministic broker order id."""

    def __init__(self):
        self.offsets: list[tuple[str, float, str]] = []

    def submit_offset(self, symbol: str, qty: float, coid: str) -> str:
        self.offsets.append((symbol, qty, coid))
        return f"off-{symbol}"


def _run(conn, broker, *, believed, held, monkeypatch):
    monkeypatch.setattr("algua.execution.flatten.believed_positions",
                        lambda conn, name, kind: dict(believed))
    return flatten_strategy(
        conn, broker, "cross_sectional_momentum", LedgerKind.LIVE, lane="live",
        cancel=lambda: None, ingest=lambda: None,
        held=held,
    )


def test_flatten_caps_offset_to_held_quantity(conn, monkeypatch):
    broker = _FakeOffsetBroker()
    res = _run(conn, broker, believed={"AAA": 10.0}, held=lambda: {"AAA": 3.0},
               monkeypatch=monkeypatch)
    assert res.n_offsets == 1
    assert res.flatten_error is None
    assert len(broker.offsets) == 1
    sym, qty, _coid = broker.offsets[0]
    assert sym == "AAA" and qty == 3.0  # capped to held, NOT 10


def test_flatten_skips_symbol_with_no_held_position(conn, monkeypatch):
    broker = _FakeOffsetBroker()
    res = _run(conn, broker, believed={"AAA": 10.0}, held=lambda: {},
               monkeypatch=monkeypatch)
    assert res.n_offsets == 0
    assert res.flatten_error is None
    # nothing actually held -> no fresh position opened off a stale belief
    assert broker.offsets == []


def test_flatten_sign_disagreement_reduces_exposure(conn, monkeypatch):
    broker = _FakeOffsetBroker()
    res = _run(conn, broker, believed={"AAA": 10.0}, held=lambda: {"AAA": -3.0},
               monkeypatch=monkeypatch)
    assert res.n_offsets == 1
    sym, qty, _coid = broker.offsets[0]
    # sign follows HELD: buys to cover the short, never sells into a deeper short
    assert sym == "AAA" and qty == -3.0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_flatten_non_finite_held_fails_closed(conn, monkeypatch, bad):
    # A non-finite broker-held qty must NOT fall back to the full believed qty (min(10, nan) -> 10
    # would bypass the cap). Fail closed loudly instead, submitting nothing.
    broker = _FakeOffsetBroker()
    res = _run(conn, broker, believed={"AAA": 10.0}, held=lambda: {"AAA": bad},
               monkeypatch=monkeypatch)
    assert res.n_offsets == 0
    assert res.flatten_error is not None and "non-finite" in res.flatten_error
    assert broker.offsets == []  # nothing submitted off an unverifiable held read
    rows = audit_read(conn, action="flatten_failed")
    assert len(rows) == 1


def test_flatten_shared_symbol_closes_only_attributed(conn, monkeypatch):
    broker = _FakeOffsetBroker()
    res = _run(conn, broker, believed={"AAA": 3.0}, held=lambda: {"AAA": 10.0},
               monkeypatch=monkeypatch)
    assert res.n_offsets == 1
    sym, qty, _coid = broker.offsets[0]
    # closes only the attributed 3, leaves the foreign 7 (per-strategy scope preserved)
    assert sym == "AAA" and qty == 3.0


def test_flatten_held_none_unchanged(conn, monkeypatch):
    broker = _FakeOffsetBroker()
    res = _run(conn, broker, believed={"AAA": 10.0}, held=None, monkeypatch=monkeypatch)
    assert res.n_offsets == 1
    sym, qty, _coid = broker.offsets[0]
    # held=None -> offsets the full believed qty exactly as before (paper-lane byte-identical)
    assert sym == "AAA" and qty == 10.0


def test_flatten_held_getter_failure_fails_safe(conn, monkeypatch):
    broker = _FakeOffsetBroker()

    def _boom() -> dict[str, float]:
        raise RuntimeError("held read failed")

    res = _run(conn, broker, believed={"AAA": 10.0}, held=_boom, monkeypatch=monkeypatch)
    assert res.n_offsets == 0
    assert res.flatten_error == "held read failed"       # captured, not propagated
    assert broker.offsets == []
    rows = audit_read(conn, action="flatten_failed")
    assert len(rows) == 1 and rows[0]["strategy"] == "cross_sectional_momentum"


class _FakeOffsetBrokerWithFailingSubmit:
    """Raises an exception during submit_offset to test the fail-safe exception-capture path."""

    def __init__(self, exc: Exception):
        self.exc = exc

    def submit_offset(self, symbol: str, qty: float, coid: str) -> str:
        raise self.exc


@pytest.mark.parametrize("exc", [
    sqlite3.OperationalError("database is locked"),
    ValueError("bad coid"),
])
@pytest.mark.parametrize("kind", [LedgerKind.LIVE, LedgerKind.PAPER])
def test_flatten_submit_exception_captured_both_lanes(conn, monkeypatch, exc, kind):
    """Verify that NON-BrokerError exceptions raised during submit_offset are captured into
    FlattenResult.flatten_error (never propagated), with lane parity for LIVE and PAPER.

    This tests the #450 scenario: a database lock or other system exception during the critical
    offset-submission step is caught by the ``except Exception`` wrapper and recorded in the
    books as a flatten_failed audit entry, allowing safe emergency exit.
    """
    # Stub out all record/backfill operations to isolate the submit exception path
    monkeypatch.setattr(
        "algua.execution.flatten.record_live_order",
        lambda *a, **k: None
    )
    monkeypatch.setattr(
        "algua.execution.flatten.record_paper_venue_order",
        lambda *a, **k: True
    )
    monkeypatch.setattr(
        "algua.execution.flatten.backfill_broker_order_id",
        lambda *a, **k: None
    )
    monkeypatch.setattr(
        "algua.execution.flatten.backfill_paper_venue_broker_order_id",
        lambda *a, **k: None
    )

    # Stub believed positions at the helper's resolution site
    monkeypatch.setattr(
        "algua.execution.flatten.believed_positions",
        lambda conn, name, kind: {"AAA": 5.0}
    )
    monkeypatch.setattr(
        "algua.execution.flatten.paper_believed_positions",
        lambda conn, name: {"AAA": 5.0}
    )

    broker = _FakeOffsetBrokerWithFailingSubmit(exc)
    lane = "live" if kind is LedgerKind.LIVE else "paper"
    strategy_id = None if kind is LedgerKind.LIVE else 1

    res = flatten_strategy(
        conn, broker, "cross_sectional_momentum", kind, lane=lane,
        cancel=lambda: None, ingest=lambda: None,
        strategy_id=strategy_id,
        held=None if kind is LedgerKind.LIVE else None,  # LIVE: use believed directly
    )

    # Assert the exception message was captured
    assert res.flatten_error is not None
    assert str(exc) in res.flatten_error or exc.args[0] in res.flatten_error

    # Assert no offsets were submitted (exception happened before any successful submit)
    assert res.n_offsets == 0

    # Assert the failure was audited
    rows = audit_read(conn, action="flatten_failed")
    assert len(rows) == 1
    assert rows[0]["strategy"] == "cross_sectional_momentum"
