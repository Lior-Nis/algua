from algua.execution import live_reservations as R
from algua.registry.db import connect, migrate


def _conn(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    return conn


def test_record_reservation_trim_and_skip(tmp_path):
    conn = _conn(tmp_path)
    R.record_reservation(conn, cycle=1, strategy="s1", symbol="AAA",
                         intended=1000.0, permitted=400.0)   # partial -> trimmed
    R.record_reservation(conn, cycle=1, strategy="s2", symbol="BBB",
                         intended=500.0, permitted=0.0)       # none -> skipped
    rows = conn.execute(
        "SELECT strategy, reason, permitted_notional FROM live_reservations ORDER BY id"
    ).fetchall()
    assert (rows[0]["strategy"], rows[0]["reason"]) == ("s1", "trimmed")
    assert (rows[1]["strategy"], rows[1]["reason"], rows[1]["permitted_notional"]) == \
        ("s2", "skipped", 0.0)
