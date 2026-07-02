from algua.audit.log import append, read
from algua.registry.db import connect, migrate


def _setup(tmp_path, n: int = 5) -> object:
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    for i in range(n):
        append(conn, actor="agent", action=f"action_{i}", strategy="s" if i % 2 == 0 else "t")
    return conn


def test_append_and_read_roundtrip(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    append(conn, actor="agent", action="paper_run", reason="2 orders", strategy="s")
    rows = read(conn, strategy="s")
    assert len(rows) == 1
    assert rows[0]["actor"] == "agent"
    assert rows[0]["action"] == "paper_run"
    assert rows[0]["strategy"] == "s"


def test_read_default_most_recent_first(tmp_path):
    """Default ordering is most-recent-first (DESC by id)."""
    conn = _setup(tmp_path, n=3)
    rows = read(conn)
    ids = [r["id"] for r in rows]
    assert ids == sorted(ids, reverse=True), "Default order should be most-recent-first"


def test_read_limit(tmp_path):
    """limit= caps the number of returned rows."""
    conn = _setup(tmp_path, n=5)
    rows = read(conn, limit=3)
    assert len(rows) == 3


def test_read_limit_offset(tmp_path):
    """limit + offset together page through the result set."""
    conn = _setup(tmp_path, n=5)
    all_rows = read(conn)  # all 5, most-recent-first
    page1 = read(conn, limit=2, offset=0)
    page2 = read(conn, limit=2, offset=2)
    assert page1[0]["id"] == all_rows[0]["id"]
    assert page2[0]["id"] == all_rows[2]["id"]
    assert len(page1) == 2
    assert len(page2) == 2


def test_read_strategy_filter_with_limit(tmp_path):
    """strategy= filter is compatible with limit/offset."""
    conn = _setup(tmp_path, n=6)  # 3 rows for 's', 3 for 't'
    rows = read(conn, strategy="s", limit=2)
    assert len(rows) == 2
    assert all(r["strategy"] == "s" for r in rows)


def test_read_limit_larger_than_table(tmp_path):
    """limit larger than total rows returns all rows without error."""
    conn = _setup(tmp_path, n=3)
    rows = read(conn, limit=100)
    assert len(rows) == 3


def test_read_no_limit_returns_all(tmp_path):
    """Passing limit=None (default) returns every row."""
    conn = _setup(tmp_path, n=4)
    rows = read(conn)
    assert len(rows) == 4


# --- validation tests ---

def test_read_limit_zero_raises(tmp_path):
    """limit=0 is rejected; it would return no rows and is meaningless."""
    import pytest
    conn = _setup(tmp_path, n=2)
    with pytest.raises(ValueError, match="limit"):
        read(conn, limit=0)


def test_read_limit_negative_raises(tmp_path):
    """Negative limit must be rejected; SQLite treats it as unlimited."""
    import pytest
    conn = _setup(tmp_path, n=2)
    with pytest.raises(ValueError, match="limit"):
        read(conn, limit=-1)


def test_read_offset_negative_raises(tmp_path):
    """Negative offset must be rejected."""
    import pytest
    conn = _setup(tmp_path, n=2)
    with pytest.raises(ValueError, match="offset"):
        read(conn, limit=10, offset=-1)


def test_read_offset_negative_no_limit_raises(tmp_path):
    """Negative offset is rejected even when no limit is supplied."""
    import pytest
    conn = _setup(tmp_path, n=2)
    with pytest.raises(ValueError, match="offset"):
        read(conn, offset=-5)


def test_read_valid_limit_and_offset_works(tmp_path):
    """Boundary-valid values (limit=1, offset=0) must not raise."""
    conn = _setup(tmp_path, n=3)
    rows = read(conn, limit=1, offset=0)
    assert len(rows) == 1


def test_read_valid_offset_without_limit_works(tmp_path):
    """offset=0 with no limit is valid (default call pattern)."""
    conn = _setup(tmp_path, n=3)
    rows = read(conn, offset=0)
    assert len(rows) == 3


# --- actor / action / time-range filter tests ---

def test_read_actor_filter(tmp_path):
    """actor= restricts to rows for that actor."""
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    append(conn, actor="agent", action="paper_run", strategy="s")
    append(conn, actor="human", action="flatten", strategy="s")
    rows = read(conn, actor="human")
    assert [r["actor"] for r in rows] == ["human"]


def test_read_action_filter(tmp_path):
    """action= restricts to rows for that action."""
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    append(conn, actor="agent", action="kill_switch_trip", strategy="s")
    append(conn, actor="agent", action="paper_run", strategy="s")
    rows = read(conn, action="kill_switch_trip")
    assert [r["action"] for r in rows] == ["kill_switch_trip"]


def test_read_combined_filters_are_anded(tmp_path):
    """strategy + actor + action filters AND together."""
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    append(conn, actor="agent", action="paper_run", strategy="s")
    append(conn, actor="human", action="paper_run", strategy="s")
    append(conn, actor="agent", action="paper_run", strategy="t")
    rows = read(conn, strategy="s", actor="agent", action="paper_run")
    assert len(rows) == 1
    assert rows[0]["strategy"] == "s"
    assert rows[0]["actor"] == "agent"


def test_read_since_until_range(tmp_path):
    """since (inclusive) / until (exclusive) bound the ts range."""
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    for ts in ("2026-01-01T00:00:00+00:00",
               "2026-01-02T00:00:00+00:00",
               "2026-01-03T00:00:00+00:00"):
        conn.execute(
            "INSERT INTO audit_log(ts, actor, action) VALUES (?,?,?)",
            (ts, "agent", "a"),
        )
    conn.commit()
    rows = read(conn, since="2026-01-02T00:00:00+00:00", until="2026-01-03T00:00:00+00:00")
    assert [r["ts"] for r in rows] == ["2026-01-02T00:00:00+00:00"]
