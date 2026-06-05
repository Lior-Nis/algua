from __future__ import annotations

import sqlite3
from pathlib import Path

# Identifies the current schema generation. This is a marker stamped into the
# DB's user_version, NOT a migration cursor: there is no per-version migration
# logic. `migrate()` is an idempotent bootstrap (CREATE TABLE/INDEX IF NOT EXISTS),
# so it can add new tables/indexes to an existing DB but CANNOT ALTER a populated one.
# Any column/constraint change to an existing table needs a real migration
# (write it explicitly when the need arrives) — not just a bump of this number.
SCHEMA_VERSION = 5

_SCHEMA = """
CREATE TABLE IF NOT EXISTS strategies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    stage TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS stage_transitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id INTEGER NOT NULL REFERENCES strategies(id),
    from_stage TEXT,
    to_stage TEXT NOT NULL,
    actor TEXT NOT NULL,
    reason TEXT,
    code_hash TEXT,
    config_hash TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS approvals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id INTEGER NOT NULL REFERENCES strategies(id),
    code_hash TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    -- dependency_hash is nullable on purpose: rows written before this column existed carry
    -- NULL and MUST never satisfy the live gate (fail-closed), and `has_valid_approval` refuses
    -- a NULL probe outright. New approvals always write a concrete hash.
    dependency_hash TEXT,
    approved_by TEXT NOT NULL,
    created_at TEXT NOT NULL,
    revoked_at TEXT
);
-- paper_orders / paper_fills / audit_log / kill_switches are DELIBERATELY
-- denormalized: they reference a strategy by its free-text NAME and carry no
-- foreign key into strategies(id). These are operational/audit snapshots, not
-- relational children of the registry. audit_log in particular is an immutable
-- trail that MUST survive a strategy's removal, and there is intentionally no
-- strategy-deletion path in the codebase. Keying by name (rather than id +
-- ON DELETE CASCADE) keeps these records readable and self-contained even after
-- the parent strategy is gone. The normalized core (stage_transitions,
-- approvals) keeps its integer FK to strategies(id) precisely because it is
-- relational state that should not outlive its strategy.
CREATE TABLE IF NOT EXISTS paper_orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    target_weight REAL NOT NULL,
    decision_ts TEXT NOT NULL,
    submitted_ts TEXT NOT NULL,
    status TEXT NOT NULL,
    broker_order_id TEXT NOT NULL
);
-- One broker order maps to at most one paper_orders row per strategy, so a crash/retry or a
-- duplicate Alpaca client_order_id path that re-returns the same order is an idempotent no-op
-- rather than a duplicate row (#18).
CREATE UNIQUE INDEX IF NOT EXISTS ux_paper_orders_strategy_broker
    ON paper_orders(strategy, broker_order_id);
CREATE TABLE IF NOT EXISTS paper_fills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id INTEGER NOT NULL REFERENCES paper_orders(id),
    symbol TEXT NOT NULL,
    qty REAL NOT NULL,
    price REAL NOT NULL,
    fill_ts TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    reason TEXT,
    strategy TEXT
);
CREATE TABLE IF NOT EXISTS kill_switches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy TEXT NOT NULL UNIQUE,
    reason TEXT,
    actor TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS strategy_peaks (
    strategy TEXT PRIMARY KEY,
    peak_equity REAL NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def migrate(conn: sqlite3.Connection) -> None:
    """Bootstrap the schema, then apply in-place column migrations; idempotent.

    The `CREATE TABLE IF NOT EXISTS` bootstrap brings a DB missing whole tables up to date but
    CANNOT add a column to an already-populated table. Adding a column to an existing table needs
    a dedicated `ALTER TABLE` — `_add_missing_columns` does exactly that, guarded by an
    introspection check so re-running is a no-op. We do not gate on user_version (doing so would
    falsely imply migration history and could skip needed table creation on a pre-stamped DB);
    we only stamp it afterward as a schema-generation marker.
    """
    conn.executescript(_SCHEMA)
    _add_missing_columns(conn, "approvals", {"dependency_hash": "TEXT"})
    conn.execute(f"PRAGMA user_version={SCHEMA_VERSION};")
    conn.commit()


def _add_missing_columns(
    conn: sqlite3.Connection, table: str, columns: dict[str, str]
) -> None:
    """Add any of ``columns`` (name -> column type) missing from ``table`` via ``ALTER TABLE``.

    Idempotent: existing columns are skipped. New columns are added without a default, so on a
    populated table the existing rows get NULL — which the live gate treats as fail-closed
    (a NULL ``dependency_hash`` can never match a recomputed concrete hash)."""
    existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
    for name, col_type in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {col_type}")
