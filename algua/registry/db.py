from __future__ import annotations

import sqlite3
from pathlib import Path

# Identifies the current schema generation. This is a marker stamped into the
# DB's user_version, NOT a migration cursor: there is no per-version migration
# logic. `migrate()` is an idempotent bootstrap (CREATE TABLE/INDEX IF NOT EXISTS)
# that ALSO performs guarded in-place column additions via `_add_missing_columns`
# (PRAGMA table_info introspection + ALTER TABLE), so it can both add new
# tables/indexes AND add columns to an already-populated table. Adding a column
# is therefore the established pattern — but a SCHEMA_VERSION bump MUST be
# accompanied by the corresponding migration step (a new table/index in _SCHEMA
# and/or a new entry in the `_add_missing_columns` calls in `migrate()`); never
# bump this number without the migration that earns it.
SCHEMA_VERSION = 12

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
    -- dependency_hash mirrors code_hash/config_hash: it is the locked-dependency identity pinned
    -- by the live gate, recorded here so the "what was promoted to live" audit trail carries the
    -- full (code, config, dependency) identity. NULL for non-live transitions (no hashes), exactly
    -- as code_hash/config_hash are.
    dependency_hash TEXT,
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
-- search_trials records the MEASURED search breadth of each parameter sweep so the promotion
-- gate's multiple-testing defense can scale on the real count of combinations tried, not a
-- self-reported flag. One row per `backtest sweep`: n_combos is the actual size of that sweep's
-- grid; grid_json is the JSON grid for the audit trail. The promotion gate sums n_combos across
-- all rows for a strategy (cumulative trials searched in the family — the conservative, honest
-- count).
-- KEYED BY strategy NAME (free text), NOT a strategies(id) FK, ON PURPOSE: a sweep can run
-- BEFORE a strategy is registered (exploration precedes registration). Keying by id would force
-- pre-registration sweeps to record nothing, letting an agent search broadly first and then
-- promote a freshly-registered strategy under a smaller DECLARED breadth — defeating the gate.
-- Keying by name lets those measured trials persist and be summed at promotion. (Same
-- denormalized-by-name rationale as paper_orders/audit_log above.)
-- INTENTIONAL: there is no grid deduplication. Re-running an identical sweep inserts another row
-- and permanently raises the cumulative count — and therefore the promotion bar. This is the
-- conservative choice: exploratory re-runs are real search effort and should count; silently
-- deduplicating them would quietly weaken the multiple-testing defense.
CREATE TABLE IF NOT EXISTS search_trials (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_name TEXT NOT NULL,
    n_combos INTEGER NOT NULL,
    grid_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_search_trials_strategy ON search_trials(strategy_name);
-- holdout_evaluations burns a walk-forward holdout window on use, so it can be evaluated ONCE.
-- `research promote` carves the last holdout_frac of the period into an out-of-sample holdout and
-- gates on it; the promotion guarantee rests on that holdout being seen once. Each row records a
-- holdout that was looked at (regardless of gate pass/fail — looking consumes it). A later promote
-- whose (strategy, data identity, OVERLAPPING period, same holdout_frac) collides with a recorded
-- row is REFUSED unless the operator passes --allow-holdout-reuse, which writes a row with
-- reused=1 to make the statistical compromise auditable. Matching is on the WINDOW, not on
-- config_hash: re-gating the same out-of-sample window with a tweaked config is exactly the leak
-- being closed (config_hash is recorded as evidence only). Data identity = snapshot_id when both
-- sides have one, else data_source. FK into strategies(id) — relational state, not an audit
-- snapshot, so it should not outlive its strategy.
CREATE TABLE IF NOT EXISTS holdout_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id INTEGER NOT NULL REFERENCES strategies(id),
    data_source TEXT NOT NULL,
    snapshot_id TEXT,
    period_start TEXT NOT NULL,
    period_end TEXT NOT NULL,
    holdout_frac REAL NOT NULL,
    config_hash TEXT NOT NULL,
    reused INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_holdout_evaluations_strategy
    ON holdout_evaluations(strategy_id);
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
    nonce           TEXT PRIMARY KEY,
    strategy_id     INTEGER NOT NULL REFERENCES strategies(id),
    code_hash       TEXT NOT NULL,
    config_hash     TEXT NOT NULL,
    dependency_hash TEXT,
    issued_at       TEXT NOT NULL,
    expires_at      TEXT NOT NULL,
    consumed_at     TEXT
);
CREATE TABLE IF NOT EXISTS live_authorizations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id     INTEGER NOT NULL REFERENCES strategies(id),
    code_hash       TEXT NOT NULL,
    config_hash     TEXT NOT NULL,
    dependency_hash TEXT,
    challenge       TEXT NOT NULL,
    signature       TEXT NOT NULL,
    principal       TEXT NOT NULL,
    authorized_at   TEXT NOT NULL,
    revoked_at      TEXT
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
    _rekey_search_trials_to_name(conn)
    conn.executescript(_SCHEMA)
    _add_missing_columns(conn, "approvals", {"dependency_hash": "TEXT"})
    _add_missing_columns(conn, "stage_transitions", {"dependency_hash": "TEXT"})
    conn.execute(f"PRAGMA user_version={SCHEMA_VERSION};")
    conn.commit()


def _rekey_search_trials_to_name(conn: sqlite3.Connection) -> None:
    """Forward-migrate a dev DB whose ``search_trials`` is keyed by the old ``strategy_id`` FK to
    the name-keyed table, carrying each row's breadth across by resolving the id to a strategy
    name. Runs BEFORE the ``CREATE TABLE IF NOT EXISTS`` bootstrap (which would otherwise leave an
    old-shaped table untouched). Idempotent: a no-op once the table is already name-keyed (or
    absent — the bootstrap then creates it fresh)."""
    table_exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='search_trials'"
    ).fetchone()
    if table_exists is None:
        return
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(search_trials)")}
    if "strategy_name" in cols or "strategy_id" not in cols:
        return  # already migrated (or some other shape) — leave it alone
    # Rebuild the table name-keyed, joining through strategies to recover each row's name. Rows
    # whose strategy_id no longer resolves are dropped (the strategy is gone; its breadth is moot).
    conn.executescript(
        """
        ALTER TABLE search_trials RENAME TO _search_trials_old;
        CREATE TABLE search_trials (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_name TEXT NOT NULL,
            n_combos INTEGER NOT NULL,
            grid_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        INSERT INTO search_trials(strategy_name, n_combos, grid_json, created_at)
            SELECT s.name, o.n_combos, o.grid_json, o.created_at
            FROM _search_trials_old o JOIN strategies s ON s.id = o.strategy_id;
        DROP TABLE _search_trials_old;
        """
    )


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
