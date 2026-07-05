from algua.registry.db import connect, migrate
from algua.risk import kill_switch


def _conn(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    return conn


def test_trip_then_is_tripped(tmp_path):
    conn = _conn(tmp_path)
    assert kill_switch.is_tripped(conn, "s") is False
    kill_switch.trip(conn, "s", reason="boom", actor="system")
    assert kill_switch.is_tripped(conn, "s") is True
    info = kill_switch.get(conn, "s")
    assert info["reason"] == "boom" and info["actor"] == "system"


def test_reset_clears(tmp_path):
    conn = _conn(tmp_path)
    kill_switch.trip(conn, "s", reason="boom", actor="system")
    assert kill_switch.reset(conn, "s") is True
    assert kill_switch.is_tripped(conn, "s") is False
    assert kill_switch.reset(conn, "s") is False  # nothing to reset


def test_retrip_updates_reason(tmp_path):
    conn = _conn(tmp_path)
    kill_switch.trip(conn, "s", reason="first", actor="agent")
    kill_switch.trip(conn, "s", reason="second", actor="system")
    assert kill_switch.get(conn, "s")["reason"] == "second"


def test_list_tripped_returns_sorted_names(tmp_path):
    conn = _conn(tmp_path)
    assert kill_switch.list_tripped(conn) == []  # empty DB
    kill_switch.trip(conn, "zeta", reason="boom", actor="system")
    kill_switch.trip(conn, "alpha", reason="boom", actor="system")
    assert kill_switch.list_tripped(conn) == ["alpha", "zeta"]
