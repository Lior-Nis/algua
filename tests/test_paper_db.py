from algua.registry.db import SCHEMA_VERSION, connect, migrate
from algua.registry.store import SqliteStrategyRepository


def _tables(conn):
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return {r["name"] for r in rows}


def test_migrate_creates_paper_tables(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    tables = _tables(conn)
    assert {"paper_orders", "paper_fills", "audit_log", "strategy_peaks"} <= tables


def test_migrate_is_idempotent(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    migrate(conn)  # second run must not raise
    assert conn.execute("PRAGMA user_version;").fetchone()[0] == SCHEMA_VERSION


def test_migrate_creates_kill_switches_table(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    assert "kill_switches" in _tables(conn)


def _fk_targets(conn, table):
    return {r["table"] for r in conn.execute(f"PRAGMA foreign_key_list({table})").fetchall()}


def test_operational_tables_are_denormalized_snapshots(tmp_path):
    """paper_orders / audit_log / kill_switches key off the free-text strategy
    NAME by design and carry NO foreign key into strategies.

    These are operational/audit snapshots that must survive strategy deletion
    (audit_log in particular is an immutable trail). Pinning the absence of an
    FK documents this as a deliberate denormalization, not an oversight.
    """
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    for table in ("paper_orders", "audit_log", "kill_switches"):
        assert "strategies" not in _fk_targets(conn, table)


def test_operational_rows_survive_strategy_deletion(tmp_path):
    """With foreign_keys=ON, the denormalized operational/audit tables impose no
    constraint of their own, so their rows survive a strategy's removal.

    The normalized core (stage_transitions/approvals) intentionally DOES block
    deletion via FK; here we delete those children first, then confirm the
    audit/operational snapshots remain — by design, not by accident.
    """
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    SqliteStrategyRepository(conn).add("alpha")
    conn.execute(
        "INSERT INTO audit_log(ts, actor, action, reason, strategy) VALUES (?,?,?,?,?)",
        ("2026-06-04T00:00:00+00:00", "agent", "noted", None, "alpha"),
    )
    conn.execute(
        "INSERT INTO kill_switches(strategy, reason, actor, created_at) VALUES (?,?,?,?)",
        ("alpha", None, "agent", "2026-06-04T00:00:00+00:00"),
    )
    conn.commit()

    # Clear the normalized children that legitimately reference strategies(id),
    # then remove the strategy itself.
    conn.execute(
        "DELETE FROM stage_transitions WHERE strategy_id IN "
        "(SELECT id FROM strategies WHERE name = ?)",
        ("alpha",),
    )
    conn.execute("DELETE FROM strategies WHERE name = ?", ("alpha",))
    conn.commit()

    # The denormalized snapshots are untouched.
    assert conn.execute(
        "SELECT COUNT(*) FROM audit_log WHERE strategy = ?", ("alpha",)
    ).fetchone()[0] == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM kill_switches WHERE strategy = ?", ("alpha",)
    ).fetchone()[0] == 1


def test_migrate_creates_tick_snapshots_table(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    assert "tick_snapshots" in _tables(conn)


def test_migrate_creates_global_halt_table(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    assert "global_halt" in _tables(conn)


def test_migrate_creates_live_challenges_table(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    assert "live_challenges" in _tables(conn)
