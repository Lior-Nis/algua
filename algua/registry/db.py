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
SCHEMA_VERSION = 23

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
-- is REFUSED if its OOS interval [holdout_start, holdout_end] — the exact bars walk_forward burns
-- (#192) — overlaps a recorded row's interval for the same strategy+data identity, unless the
-- operator passes --allow-holdout-reuse (writes reused=1, auditable). A NULL interval matches
-- unconditionally (fail closed). period_* and holdout_frac are recorded as evidence only; matching
-- is on the INTERVAL, not on config_hash (re-gating the same OOS window with a tweaked config is
-- exactly the leak being closed). Data identity = snapshot_id when both sides have one, else
-- data_source. FK into strategies(id) — relational state, not an audit snapshot.
CREATE TABLE IF NOT EXISTS holdout_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id INTEGER NOT NULL REFERENCES strategies(id),
    data_source TEXT NOT NULL,
    snapshot_id TEXT,
    period_start TEXT NOT NULL,
    period_end TEXT NOT NULL,
    holdout_frac REAL NOT NULL,
    config_hash TEXT NOT NULL,   -- '' while in-flight (placeholder); real hash written at finalize.
    reused INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    committed_at TEXT,           -- NULL = in-flight reservation (or a legacy burn predating this
                                 -- column); non-NULL = committed burn. Either way an overlapping
                                 -- row blocks fail-closed. Orphaned reservations (pending rows from
                                 -- a crashed run) are listable via WHERE committed_at IS NULL and
                                 -- are cleared only by a deliberate human --allow-holdout-reuse.
    holdout_start TEXT,          -- ISO date; OOS tail start (the matched single-use window, #192)
    holdout_end TEXT             -- ISO date; OOS tail end (last actual bar date)
);
CREATE INDEX IF NOT EXISTS ix_holdout_evaluations_strategy
    ON holdout_evaluations(strategy_id);
-- gate_evaluations records every promotion-gate evaluation (pass AND fail) for the audit trail,
-- AND is the single-use, AGENT-ONLY token the BACKTESTED->CANDIDATE transition consumes (the
-- shortlist gate, mirroring the live gate: trust the gate record, not the stage flag). A passing
-- AGENT row is minted by `research promote` (via the protected registry.promotion orchestrator)
-- stamped with the artifact identity recomputed by approvals.compute_artifact_hashes; the
-- transition consumes THAT row's id, in the same transaction as the stage change. A human/override
-- promote writes an actor='human' row that is NEVER an agent-consumable token (audit only). FK into
-- strategies(id) — relational state, not an audit snapshot.
CREATE TABLE IF NOT EXISTS gate_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id INTEGER NOT NULL REFERENCES strategies(id),
    passed INTEGER NOT NULL,
    n_funnel INTEGER NOT NULL,
    own_lifetime_combos INTEGER NOT NULL,
    windowed_total_combos INTEGER NOT NULL,
    funnel_window_days INTEGER NOT NULL,
    breadth_provenance TEXT NOT NULL,
    pit_ok INTEGER NOT NULL,
    pit_override INTEGER NOT NULL DEFAULT 0,
    holdout_n_bars INTEGER NOT NULL,
    min_holdout_observations INTEGER NOT NULL,
    code_hash TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    dependency_hash TEXT,
    data_source TEXT NOT NULL,
    snapshot_id TEXT,
    period_start TEXT NOT NULL,
    period_end TEXT NOT NULL,
    holdout_frac REAL NOT NULL,
    actor TEXT NOT NULL,
    decision_json TEXT NOT NULL,
    consumed INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_gate_evaluations_strategy ON gate_evaluations(strategy_id);
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
-- The signed payload is NEVER stored verbatim and re-verified — an agent with DB write could then
-- pair vetted-identity columns with a foreign signature (codex CRITICAL). We store only the
-- non-identity payload parts (nonce, expires_at); trade-time verification REBUILDS the canonical
-- challenge from the RECOMPUTED identity + strategy + these, so a signature is valid only over the
-- current artifact.
CREATE TABLE IF NOT EXISTS live_authorizations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id     INTEGER NOT NULL REFERENCES strategies(id),
    code_hash       TEXT NOT NULL,
    config_hash     TEXT NOT NULL,
    dependency_hash TEXT,
    nonce           TEXT NOT NULL,
    expires_at      TEXT NOT NULL,
    signature       TEXT NOT NULL,
    principal       TEXT NOT NULL,
    authorized_at   TEXT NOT NULL,
    revoked_at      TEXT
);
CREATE TABLE IF NOT EXISTS strategy_allocations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id   INTEGER NOT NULL REFERENCES strategies(id),
    capital       REAL NOT NULL,
    effective_ts  TEXT NOT NULL,
    actor         TEXT NOT NULL,
    revoked_ts    TEXT
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
-- ideas is the structured top-of-funnel pool (#126): externally-sourced, deduped,
-- provenance-stamped hypothesis records that climb the normal gated ladder. authored_strategy_id
-- is the relational link to the strategy an idea became (NULL until authored); the dedup gate
-- resolves a refuted strategy through this FK (a live join), so a refuted strategy blocks its
-- idea's near-duplicates without mutating idea rows. duplicate_of_idea_id records a deliberate
-- --allow-duplicate override (paired with override_reason).
CREATE TABLE IF NOT EXISTS ideas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    hypothesis TEXT NOT NULL,
    family TEXT,
    tags TEXT NOT NULL DEFAULT '[]',
    source_type TEXT NOT NULL,
    source_ref TEXT,
    source_date TEXT,
    source_note TEXT,
    required_data TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL,
    signature TEXT NOT NULL,
    authored_strategy_id INTEGER REFERENCES strategies(id),
    duplicate_of_idea_id INTEGER REFERENCES ideas(id),
    override_reason TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_ideas_status ON ideas(status);
CREATE INDEX IF NOT EXISTS ix_ideas_family ON ideas(family);
-- forward_gate_evaluations is the single-use AGENT-ONLY token ledger for the forward-test gate
-- (#124): each row records one gate evaluation (pass AND fail) and — for passing agent rows — is
-- the consumable token the PAPER->FORWARD_TESTED transition requires (mirroring gate_evaluations
-- for the BACKTESTED->CANDIDATE edge). A passing row is minted by the forward-gate run once the
-- strategy has accumulated enough forward-test observations; the transition consumes THAT row's id
-- in the same transaction as the stage change. Legacy NULL tick_snapshot rows (pre-v21) are
-- DELIBERATELY inadmissible as gate evidence — fail-closed, no backfill. FK into strategies(id).
-- NOTE: SQLite ALTER TABLE cannot add CHECK constraints, so lane/clock_source value discipline is
-- enforced by the writers (order_state.py), not the schema.
CREATE TABLE IF NOT EXISTS forward_gate_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id INTEGER NOT NULL REFERENCES strategies(id),
    passed INTEGER NOT NULL,
    n_forward_observations INTEGER NOT NULL,
    min_forward_observations INTEGER NOT NULL,
    session_coverage REAL,
    realized_sharpe REAL,
    holdout_sharpe REAL,
    degradation_factor REAL NOT NULL,
    sharpe_floor REAL NOT NULL,
    realized_vol REAL,
    min_forward_vol REAL NOT NULL,
    realized_max_drawdown REAL,
    max_forward_drawdown REAL NOT NULL,
    first_tick_id INTEGER,
    last_tick_id INTEGER,
    first_tick_ts TEXT,
    last_tick_ts TEXT,
    max_staleness_sessions INTEGER NOT NULL,
    n_reconcile_failures INTEGER NOT NULL,
    n_concurrent_forward INTEGER NOT NULL,
    account_id TEXT,
    code_hash TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    dependency_hash TEXT,
    actor TEXT NOT NULL,
    decision_json TEXT NOT NULL,
    consumed INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_forward_gate_strategy ON forward_gate_evaluations(strategy_id);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000;")  # WAL + busy_timeout = deliberate concurrency posture
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
    _migrate_shortlisted_to_candidate(conn)
    _rekey_search_trials_to_name(conn)
    conn.executescript(_SCHEMA)
    _add_missing_columns(conn, "approvals", {"dependency_hash": "TEXT"})
    _add_missing_columns(conn, "stage_transitions", {"dependency_hash": "TEXT"})
    _add_missing_columns(
        conn,
        "strategies",
        {
            "family": "TEXT",
            "tags": "TEXT",
            "author": "TEXT",
            "hypothesis_status": "TEXT",
            "derived_from": "TEXT",
            "description": "TEXT",
        },
    )
    # v21 (#124): stamp tick provenance onto existing tick_snapshots rows so the forward gate can
    # verify artifact identity (code_hash/config_hash/dependency_hash), lane, and account. Legacy
    # NULL rows are DELIBERATELY inadmissible as gate evidence — fail-closed, no backfill. SQLite
    # ALTER TABLE cannot add CHECK constraints, so lane/clock_source value discipline is enforced
    # by the writers (order_state.py); the gate rejects NULL lane/clock_source fail-closed.
    _add_missing_columns(
        conn,
        "tick_snapshots",
        {
            "lane": "TEXT",
            "code_hash": "TEXT",
            "config_hash": "TEXT",
            "dependency_hash": "TEXT",
            "strategy_id": "INTEGER",
            "account_id": "TEXT",
            "cash": "REAL",
            "clock_source": "TEXT",
            "recorded_at": "TEXT",
        },
    )
    # v21 (#124): link paper_orders to strategies(id) for forward-gate tick↔order attribution.
    # Legacy NULL rows are inadmissible gate evidence (fail-closed, no backfill).
    _add_missing_columns(conn, "paper_orders", {"strategy_id": "INTEGER"})
    # v22 (#161): committed_at distinguishes an in-flight holdout reservation (NULL) from a
    # committed burn (non-NULL). NO backfill: a legacy row that predates this column keeps
    # committed_at=NULL and is treated as a permanent reservation (blocks fail-closed). Backfilling
    # would introduce a migration race that could clobber a genuine concurrent reservation.
    # v23 (#192): holdout_start/holdout_end are the OOS interval matched by the single-use guard.
    # Legacy rows (pre-v23) are backfilled to the conservative full period [period_start,
    # period_end] — a guaranteed superset of any real OOS tail, so the guard fails closed.
    _add_missing_columns(
        conn,
        "holdout_evaluations",
        {"committed_at": "TEXT", "holdout_start": "TEXT", "holdout_end": "TEXT"},
    )
    _backfill_holdout_intervals(conn)
    conn.execute(f"PRAGMA user_version={SCHEMA_VERSION};")
    conn.commit()


def _migrate_shortlisted_to_candidate(conn: sqlite3.Connection) -> None:
    """Rewrite the renamed lifecycle stage value `shortlisted` -> `candidate` (#120) in the typed
    stage columns. Runs BEFORE the `CREATE TABLE IF NOT EXISTS` bootstrap, so each table is guarded
    independently — a fresh DB has neither table yet. Idempotent: the `WHERE` matches nothing on a
    second run, and it does NOT gate on `user_version`, so a DB already stamped at the new version
    but still holding `shortlisted` rows is still corrected.

    Only the typed `stage` / `from_stage` / `to_stage` columns are rewritten — the free-text audit
    trail (`audit_log`, `stage_transitions.reason`) and `gate_evaluations.decision_json` are
    immutable history and intentionally left as written."""
    def _has(table: str) -> bool:
        return (
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone()
            is not None
        )

    def _has_col(table: str, col: str) -> bool:
        return any(
            row[1] == col
            for row in conn.execute(f"PRAGMA table_info({table})")
        )

    if _has("strategies") and _has_col("strategies", "stage"):
        conn.execute("UPDATE strategies SET stage='candidate' WHERE stage='shortlisted'")
    if _has("stage_transitions"):
        if _has_col("stage_transitions", "from_stage"):
            conn.execute(
                "UPDATE stage_transitions SET from_stage='candidate'"
                " WHERE from_stage='shortlisted'"
            )
        if _has_col("stage_transitions", "to_stage"):
            conn.execute(
                "UPDATE stage_transitions SET to_stage='candidate' WHERE to_stage='shortlisted'"
            )


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


def _backfill_holdout_intervals(conn: sqlite3.Connection) -> None:
    """Backfill v23 holdout_start/holdout_end on legacy rows to the CONSERVATIVE full period
    [period_start, period_end]. The exact OOS tail cannot be recomputed at migration time (no data
    provider here), and the full period is a guaranteed superset of any real tail -> fail closed
    (may over-block a new run overlapping a legacy burn's period, the acceptable direction). Only
    touches rows missing an interval, so a row written by the new reserve path (interval already
    set) is never overwritten; deterministic, so concurrent/repeat runs converge. Idempotent."""
    conn.execute(
        "UPDATE holdout_evaluations SET holdout_start = period_start, holdout_end = period_end"
        " WHERE holdout_start IS NULL OR holdout_end IS NULL"
    )
    leftover = conn.execute(
        "SELECT COUNT(*) AS c FROM holdout_evaluations"
        " WHERE holdout_start IS NULL OR holdout_end IS NULL"
    ).fetchone()["c"]
    if leftover:
        raise RuntimeError(
            f"holdout interval backfill left {leftover} NULL-interval row(s); refusing to stamp v23"
        )


def _add_missing_columns(
    conn: sqlite3.Connection, table: str, columns: dict[str, str]
) -> None:
    """Add any of ``columns`` (name -> column type) missing from ``table`` via ``ALTER TABLE``.

    Idempotent and cross-process safe: existing columns are skipped via introspection, and a
    concurrent process that adds the same column between our introspection and our ALTER (the
    lost-ALTER race) makes our ALTER raise ``duplicate column name`` — which we swallow, since the
    column now exists either way. New columns are added without a default, so on a populated table
    the existing rows get NULL — which the live/forward gates treat as fail-closed (a NULL
    ``dependency_hash`` can never match a recomputed concrete hash)."""
    existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
    for name, col_type in columns.items():
        if name not in existing:
            try:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {col_type}")
            except sqlite3.OperationalError as exc:
                # Lost the concurrent-ALTER race: another process added it first. Idempotent.
                if "duplicate column name" not in str(exc):
                    raise
