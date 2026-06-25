from algua.audit import log as audit
from algua.registry.db import connect, migrate
from algua.risk import kill_switch
from algua.risk.breach import trip_for_breach
from algua.risk.limits import RiskBreach


def _conn(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    return conn


def test_trip_for_breach_trips_and_audits(tmp_path):
    conn = _conn(tmp_path)
    trip_for_breach(conn, "s", RiskBreach("drawdown", "equity below floor"))
    assert kill_switch.is_tripped(conn, "s") is True
    rows = audit.read(conn, strategy="s")
    assert any(r["action"] == "kill_switch_trip" and "drawdown" in r["reason"] for r in rows)
