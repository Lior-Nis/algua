from algua.registry.db import connect, migrate
from algua.risk import global_halt


def _conn(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    return conn


def test_engage_is_engaged_clear(tmp_path):
    conn = _conn(tmp_path)
    assert global_halt.is_engaged(conn) is False
    global_halt.engage(conn, reason="panic", actor="human")
    assert global_halt.is_engaged(conn) is True
    info = global_halt.get(conn)
    assert info["reason"] == "panic" and info["actor"] == "human"
    assert global_halt.clear(conn) is True
    assert global_halt.is_engaged(conn) is False
    assert global_halt.clear(conn) is False  # already clear -> no row removed


def test_engage_is_single_row(tmp_path):
    conn = _conn(tmp_path)
    global_halt.engage(conn, reason="a", actor="agent")
    global_halt.engage(conn, reason="b", actor="human")  # upsert, not a second row
    assert conn.execute("SELECT COUNT(*) FROM global_halt").fetchone()[0] == 1
    assert global_halt.get(conn)["reason"] == "b"
