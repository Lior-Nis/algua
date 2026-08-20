"""Operational-lane context: the 23 paper/live trading tables.

Orders, fills, activity-ingestion cursors and dead-letter quarantines, reconcile state, cycle
counters, tick snapshots, kill switches, the global halt, and the drawdown high-water marks
(operated by ``algua/execution/`` and ``algua/live/``). Denormalized by strategy NAME on purpose --
the rationale is the comment above ``paper_orders`` below: these are operational/audit snapshots,
not relational children of the registry.
"""
from __future__ import annotations

SCHEMA = """
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
-- book_equity_peak is the ACCOUNT-WIDE high-water mark (single row id=1) that the book-level
-- drawdown circuit breaker (#390) measures against — the aggregate analog of strategy_peaks /
-- live_nav_peaks. Ratcheted up each live cycle; cleared by resume-all after a flatten-to-cash so
-- the account re-bases its drawdown denominator.
CREATE TABLE IF NOT EXISTS book_equity_peak (
    id         INTEGER PRIMARY KEY CHECK (id = 1),
    peak       REAL NOT NULL,
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
CREATE TABLE IF NOT EXISTS live_orders (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy          TEXT NOT NULL,
    symbol            TEXT NOT NULL,
    side              TEXT NOT NULL,
    intended_notional REAL,
    client_order_id   TEXT NOT NULL UNIQUE,
    broker_order_id   TEXT,
    status            TEXT NOT NULL,
    submitted_ts      TEXT NOT NULL
);
-- broker_order_id is the fill-attribution key: at most one order may own it (partial unique so the
-- many pre-backfill NULLs are allowed).
CREATE UNIQUE INDEX IF NOT EXISTS ux_live_orders_broker_order_id
    ON live_orders(broker_order_id) WHERE broker_order_id IS NOT NULL;
CREATE TABLE IF NOT EXISTS live_fills (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    activity_id     TEXT NOT NULL UNIQUE,
    broker_order_id TEXT,
    strategy        TEXT,
    symbol          TEXT NOT NULL,
    qty             REAL NOT NULL CHECK(qty != 0),
    price           REAL NOT NULL CHECK(price > 0),
    fill_ts         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_live_fills_strategy_symbol ON live_fills(strategy, symbol);
CREATE TABLE IF NOT EXISTS live_activities (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    activity_id  TEXT NOT NULL UNIQUE,
    type         TEXT NOT NULL,
    symbol       TEXT,
    amount       REAL,
    ts           TEXT,
    raw          TEXT
);
CREATE TABLE IF NOT EXISTS live_fill_cursor (
    name    TEXT PRIMARY KEY,
    cursor  TEXT
);
CREATE TABLE IF NOT EXISTS live_activity_quarantine (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    activity_id  TEXT NOT NULL UNIQUE,
    error        TEXT NOT NULL,
    raw          TEXT NOT NULL,
    quarantined_ts TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE TABLE IF NOT EXISTS paper_venue_orders (        -- crash-safe intent + attribution
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy          TEXT NOT NULL,
    symbol            TEXT NOT NULL,
    side              TEXT NOT NULL,
    intended_notional REAL,
    client_order_id   TEXT NOT NULL UNIQUE,             -- durable identity; idempotent re-submit
    broker_order_id   TEXT,                             -- backfilled on broker accept
    strategy_id       INTEGER NOT NULL,                 -- attribution for the forward gate
    status            TEXT NOT NULL,
    submitted_ts      TEXT NOT NULL
);
-- exactly one order may own a broker id (the fill-attribution key); many pre-backfill NULLs allowed
CREATE UNIQUE INDEX IF NOT EXISTS ux_paper_venue_orders_broker_order_id
    ON paper_venue_orders(broker_order_id) WHERE broker_order_id IS NOT NULL;
CREATE TABLE IF NOT EXISTS paper_venue_fills (          -- signed fills (≈ live_fills)
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    activity_id TEXT NOT NULL UNIQUE,
    broker_order_id TEXT,
    strategy TEXT,                                       -- nullable: orphan / pre-backfill
    symbol TEXT NOT NULL,
    qty REAL NOT NULL CHECK(qty != 0),
    price REAL NOT NULL CHECK(price > 0),
    fill_ts TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_paper_venue_fills_strategy_symbol
    ON paper_venue_fills(strategy, symbol);
CREATE TABLE IF NOT EXISTS paper_venue_activities (     -- non-fill (cash/div) rows
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    activity_id TEXT NOT NULL UNIQUE,
    type TEXT NOT NULL, symbol TEXT, amount REAL, ts TEXT, raw TEXT
);
CREATE TABLE IF NOT EXISTS paper_venue_fill_cursor (    -- ingestion cursor (≈ live_fill_cursor)
    name TEXT PRIMARY KEY, cursor TEXT
);
CREATE TABLE IF NOT EXISTS paper_venue_activity_quarantine ( -- #250 dead-letter
    activity_id TEXT PRIMARY KEY, error TEXT NOT NULL, raw TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS live_reconcile_state (
    symbol           TEXT PRIMARY KEY,
    expected_qty     REAL NOT NULL,
    broker_qty       REAL NOT NULL,
    first_seen_cycle INTEGER NOT NULL,
    status           TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS live_cycle (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    n  INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS paper_reconcile_state (
    symbol           TEXT PRIMARY KEY,
    expected_qty     REAL NOT NULL,
    broker_qty       REAL NOT NULL,
    first_seen_cycle INTEGER NOT NULL,
    status           TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS paper_cycle (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    n  INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS live_nav_peaks (
    strategy   TEXT PRIMARY KEY,
    peak       REAL NOT NULL,
    updated_ts TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS live_reservations (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle              INTEGER NOT NULL,
    strategy           TEXT NOT NULL,
    symbol             TEXT NOT NULL,
    intended_notional  REAL NOT NULL,
    permitted_notional REAL NOT NULL,
    reason             TEXT NOT NULL,
    ts                 TEXT NOT NULL
);
"""
