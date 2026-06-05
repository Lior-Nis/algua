from datetime import UTC, datetime, timedelta

from algua.registry import live_gate
from algua.registry.db import connect, migrate


def _conn(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    conn.execute("INSERT INTO strategies(id, name, stage, created_at, updated_at) "
                 "VALUES (1, 's', 'paper', '2026-01-01', '2026-01-01')")
    conn.commit()
    return conn


def test_build_challenge_is_deterministic():
    a = live_gate.build_challenge("s", 1, "ch", "cfg", "nonce123", "2026-06-05T00:10:00+00:00")
    b = live_gate.build_challenge("s", 1, "ch", "cfg", "nonce123", "2026-06-05T00:10:00+00:00")
    assert a == b and "nonce=nonce123" in a and a.startswith("algua-go-live")


def test_issue_then_find_then_consume(tmp_path):
    conn = _conn(tmp_path)
    now = datetime(2026, 6, 5, tzinfo=UTC)
    issued = live_gate.issue_challenge(conn, 1, "s", "ch", "cfg", now=now)
    assert "nonce" in issued and "challenge" in issued and "expires_at" in issued
    row = live_gate.find_pending_challenge(conn, 1, "ch", "cfg", now=now)
    assert row is not None and row["nonce"] == issued["nonce"]
    assert live_gate.consume_challenge(conn, issued["nonce"], now=now) is True
    assert live_gate.consume_challenge(conn, issued["nonce"], now=now) is False  # single-use
    assert live_gate.find_pending_challenge(conn, 1, "ch", "cfg", now=now) is None  # consumed


def test_find_pending_rejects_expired_and_wrong_hash(tmp_path):
    conn = _conn(tmp_path)
    now = datetime(2026, 6, 5, tzinfo=UTC)
    live_gate.issue_challenge(conn, 1, "s", "ch", "cfg", now=now)
    later = now + timedelta(hours=1)
    assert live_gate.find_pending_challenge(conn, 1, "ch", "cfg", now=later) is None  # expired
    assert live_gate.find_pending_challenge(conn, 1, "DIFFERENT", "cfg", now=now) is None
