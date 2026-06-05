from __future__ import annotations

import sqlite3
from pathlib import Path

# Identifies the current schema generation. This is a marker stamped into the
# DB's user_version, NOT a migration cursor: there is no per-version migration
# logic. `migrate()` is an idempotent bootstrap (CREATE TABLE/INDEX IF NOT EXISTS),
# so it can add new tables/indexes to an existing DB but CANNOT ALTER a populated one.
# Any column/constraint change to an existing table needs a real migration
# (write it explicitly when the need arrives) — not just a bump of this number.
SCHEMA_VERSION = 7

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
-- Append-only per-tick operability record (equity + positions per completed tick); the equity
-- time-series `paper show` and the future dashboard read. Permanent history — no pruning path yet
-- (`trade-tick` is wall-clock-per-invocation, so growth is modest); add retention when it matters.
CREATE TABLE IF NOT EXISTS tick_snapshots (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy     TEXT NOT NULL,
    tick_ts      TEXT NOT NULL,
    decision_ts  TEXT,
    equity       REAL NOT NULL,
    peak_equity  REAL,
    positions    TEXT NOT NULL,
    n_submitted  INTEGER NOT NULL,
    reconcile_ok INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_tick_snapshots_strategy_ts ON tick_snapshots(strategy, tick_ts);
CREATE TABLE IF NOT EXISTS global_halt (
    id         INTEGER PRIMARY KEY CHECK (id = 1),
    reason     TEXT,
    actor      TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS live_challenges (
    nonce        TEXT PRIMARY KEY,
    strategy_id  INTEGER NOT NULL REFERENCES strategies(id),
    code_hash    TEXT NOT NULL,
    config_hash  TEXT NOT NULL,
    issued_at    TEXT NOT NULL,
    expires_at   TEXT NOT NULL,
    consumed_at  TEXT
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
    """Bootstrap the schema; idempotent. NOT a versioned in-place migrator.

    This runs the full current `_SCHEMA` unconditionally. Every statement is
    `CREATE TABLE IF NOT EXISTS`, so re-running is a no-op and a DB missing only
    some tables is brought fully up to date — regardless of the recorded
    user_version. It does NOT (and cannot) ALTER existing tables: changing a
    column or constraint on a populated table requires a dedicated migration,
    not a bump of SCHEMA_VERSION. We do not gate on user_version (doing so would
    falsely imply migration history and could skip needed table creation on a
    pre-stamped DB); we only stamp it afterward as a schema-generation marker.
    """
    conn.executescript(_SCHEMA)
    conn.execute(f"PRAGMA user_version={SCHEMA_VERSION};")
    conn.commit()
