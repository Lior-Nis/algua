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


def test_trip_for_breach_records_non_drawdown_kind(tmp_path):
    # The audit reason must carry the breach KIND, not just the detail string, so a non-drawdown
    # breach (gross_exposure here) is attributable in the audit log.
    conn = _conn(tmp_path)
    trip_for_breach(conn, "s", RiskBreach("gross_exposure", "gross 1.5 exceeds 1.0"))
    assert kill_switch.is_tripped(conn, "s") is True
    rows = audit.read(conn, strategy="s")
    trips = [r for r in rows if r["action"] == "kill_switch_trip"]
    # Exact match: the reason must carry both kind AND detail (the f"{kind}: {detail}" contract),
    # so a dropped/corrupted detail can't slip through a prefix-only check.
    assert any(r["reason"] == "gross_exposure: gross 1.5 exceeds 1.0" for r in trips)


def test_trip_for_breach_retrip_is_idempotent(tmp_path):
    # Re-tripping an already-tripped strategy must not error and must leave it tripped; the
    # kill-switch row is upserted (ON CONFLICT DO UPDATE), so the latest reason wins.
    conn = _conn(tmp_path)
    trip_for_breach(conn, "s", RiskBreach("drawdown", "first breach"))
    trip_for_breach(conn, "s", RiskBreach("gross_exposure", "second breach"))
    assert kill_switch.is_tripped(conn, "s") is True
    # The kill-switch row is upserted (ON CONFLICT DO UPDATE), so the SECOND trip's reason must
    # win — proving the re-trip actually wrote, not silently no-op'd while only the audit appended.
    assert kill_switch.get(conn, "s")["reason"] == "second breach"
    trips = [r for r in audit.read(conn, strategy="s") if r["action"] == "kill_switch_trip"]
    assert len(trips) == 2  # audit appends one row per trip — both breaches are recorded
