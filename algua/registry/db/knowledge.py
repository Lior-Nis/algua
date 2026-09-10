"""Knowledge/audit context: ``audit_log`` and ``negative_results``.

The immutable actor trail (operated by ``algua/audit/log.py``; appended through it by several
CLI/risk/execution call sites and read back by ``algua/cli/audit_cmd.py``) and the advisory
failed-hypothesis experience log (#332, operated by ``algua/registry/negative_results.py``). Both
are deliberately keyed by strategy NAME rather than a ``strategies(id)`` FK so they survive their
strategy -- see the denormalization rationale above ``paper_orders`` in ``execution.py``.
"""
from __future__ import annotations

import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    reason TEXT,
    strategy TEXT
);
-- v32 (#332): negative_results is an ADVISORY experience log capturing failed/rejected hypotheses
-- (gate FAILs, discards, research dead-ends) so knowledge is not lost with the branch. It NEVER
-- gates promotion and NEVER touches the live/paper path; it is written best-effort as a side effect
-- of the reject path and via a manual CLI. `gate_evaluation_id` is a NULLABLE advisory back-link to
-- the authoritative gate_evaluations row (not a hard FK — the log survives even if the reference is
-- unknown). CHECK constraints keep `kind`/`source` to their known vocabularies.
CREATE TABLE IF NOT EXISTS negative_results (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at          TEXT NOT NULL,
    strategy_name       TEXT,
    gate_evaluation_id  INTEGER,
    kind                TEXT NOT NULL CHECK (kind IN ('gate_fail', 'discard', 'dead_end')),
    verdict             TEXT NOT NULL,
    actor               TEXT NOT NULL,
    reason              TEXT NOT NULL,
    hypothesis          TEXT,
    params_json         TEXT,
    tags                TEXT,
    source              TEXT NOT NULL
        CHECK (source IN ('auto:research_promote', 'auto:leap_critic', 'manual'))
);
CREATE INDEX IF NOT EXISTS ix_negative_results_strategy ON negative_results(strategy_name);
CREATE INDEX IF NOT EXISTS ix_negative_results_created ON negative_results(created_at);
CREATE INDEX IF NOT EXISTS ix_negative_results_kind ON negative_results(kind);
"""

# The column list here MUST mirror the CREATE TABLE above (kept as a literal copy under a
# rebuild-target name rather than assembled from SCHEMA -- easier to eyeball-diff against the
# real table when the two drift, which a rebuild is exactly the kind of change that risks).
_NEGATIVE_RESULTS_REBUILD_DDL = """
CREATE TABLE negative_results__new (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at          TEXT NOT NULL,
    strategy_name       TEXT,
    gate_evaluation_id  INTEGER,
    kind                TEXT NOT NULL CHECK (kind IN ('gate_fail', 'discard', 'dead_end')),
    verdict             TEXT NOT NULL,
    actor               TEXT NOT NULL,
    reason              TEXT NOT NULL,
    hypothesis          TEXT,
    params_json         TEXT,
    tags                TEXT,
    source              TEXT NOT NULL
        CHECK (source IN ('auto:research_promote', 'auto:leap_critic', 'manual'))
);
"""


def _rebuild_negative_results_if_stale(conn: sqlite3.Connection) -> None:
    """v46 (#626): rebuild ``negative_results`` if its ``source`` CHECK predates
    ``'auto:leap_critic'``.

    SQLite cannot ``ALTER`` a CHECK constraint in place, and ``executescript(SCHEMA)``'s
    ``CREATE TABLE IF NOT EXISTS`` is a no-op on a table that already exists -- so a DB that
    reached v45/v46 before this change landed keeps the OLD ``source`` CHECK forever unless
    something rebuilds it. Detect the stale CHECK by reading the table's own stored DDL back
    from ``sqlite_master`` (the single source of truth -- no separate "have we done this" flag
    to keep in sync), and rebuild via the standard SQLite recipe: create the current-shape table
    under a new name, copy every row across, drop the old table, rename the new one into place,
    then recreate its indexes (``DROP TABLE`` drops them too).

    Safe: ``negative_results`` declares no FK out (``gate_evaluation_id`` is a plain nullable
    INTEGER, a deliberate soft back-link -- see the SCHEMA comment above), and grepping this
    package confirms no OTHER table declares an FK into it -- only the three indexes below
    reference it, and they're recreated after the rename. Idempotent: a DB already on the new
    CHECK (every fresh DB, or one already rebuilt by an earlier ``migrate()`` call) sees its own
    current DDL already contain ``'auto:leap_critic'`` and returns immediately -- no second
    rebuild, so a steady-state DB's `negative_results` keeps the same sqlite rootpage across
    repeated ``migrate()`` calls.
    """
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='negative_results'"
    ).fetchone()
    if row is None or "'auto:leap_critic'" in (row["sql"] or ""):
        return  # brand-new DB (SCHEMA above just created the current shape) or already rebuilt
    conn.execute("DROP TABLE IF EXISTS negative_results__new")  # crash-safety: a prior half-run
    conn.executescript(_NEGATIVE_RESULTS_REBUILD_DDL)
    conn.execute(
        "INSERT INTO negative_results__new(id, created_at, strategy_name, gate_evaluation_id,"
        " kind, verdict, actor, reason, hypothesis, params_json, tags, source)"
        " SELECT id, created_at, strategy_name, gate_evaluation_id, kind, verdict, actor, reason,"
        " hypothesis, params_json, tags, source FROM negative_results"
    )
    conn.execute("DROP TABLE negative_results")
    conn.execute("ALTER TABLE negative_results__new RENAME TO negative_results")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_negative_results_strategy ON negative_results(strategy_name)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_negative_results_created ON negative_results(created_at)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_negative_results_kind ON negative_results(kind)"
    )
