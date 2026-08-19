"""registry.challenges — one challenge lifecycle for both signing namespaces (spec §4.4)."""
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from algua.registry.challenges import ChallengeSpec, build_payload, consume, find_pending, issue

SPEC = ChallengeSpec(
    table="test_challenges",
    namespace="algua-test",
    payload_fields=("strategy", "strategy_id", "code_hash"),
    column_fields=("strategy_id", "code_hash"),
    ttl=timedelta(minutes=10),
)

VALUES = {"strategy": "momo", "strategy_id": 7, "code_hash": "abc"}


@pytest.fixture()
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.execute(
        "CREATE TABLE test_challenges(nonce TEXT PRIMARY KEY, strategy_id INTEGER,"
        " code_hash TEXT, issued_at TEXT, expires_at TEXT, consumed_at TEXT)"
    )
    return c


def test_payload_format_is_namespace_kv_nonce_expiry():
    payload = build_payload(SPEC, VALUES, "n0nce", "2026-01-01T00:00:00+00:00")
    assert payload == (
        "algua-test\nstrategy=momo\nstrategy_id=7\ncode_hash=abc\n"
        "nonce=n0nce\nexpires_at=2026-01-01T00:00:00+00:00"
    )


def test_issue_then_find_then_consume_single_use(conn):
    issued = issue(conn, SPEC, VALUES)
    assert set(issued) == {"nonce", "expires_at", "challenge"}
    row = find_pending(conn, SPEC, VALUES)
    assert row is not None and row["nonce"] == issued["nonce"]
    assert consume(conn, SPEC, issued["nonce"]) is True
    assert consume(conn, SPEC, issued["nonce"]) is False  # single-use
    assert find_pending(conn, SPEC, VALUES) is None       # consumed -> not pending


def test_find_pending_respects_expiry(conn):
    old = datetime.now(UTC) - timedelta(hours=1)
    issue(conn, SPEC, VALUES, now=old)
    assert find_pending(conn, SPEC, VALUES) is None  # expired


def test_find_pending_matches_null_column_with_is(conn):
    spec = ChallengeSpec(
        table="test_challenges", namespace="algua-test",
        payload_fields=("strategy", "strategy_id", "code_hash"),
        column_fields=("strategy_id", "code_hash"), ttl=timedelta(minutes=10),
    )
    values = {"strategy": "momo", "strategy_id": 7, "code_hash": None}
    issue(conn, spec, values)
    row = find_pending(conn, spec, values)
    assert row is not None  # NULL-valued bound column matches via IS
