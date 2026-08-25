"""The single ordered schema-bootstrap sequence.

``migrate()`` is deliberately kept as ONE unsplit function. Not because its ordering constraints
cross bounded contexts -- they do not. Three hard constraints are INTRA-context: the holdout
interval backfill after the holdout_evaluations ALTER; the legacy FDR index DROP before the cohort
backfill (and the composite index after it); the attempt_token index after its column. There is
also one defensive (not hard) ordering: the family_members UPDATE trigger after that table's
ALTER -- reversing it is provably benign (byte-identical schema, full registry/migration/family
test set passes, a legacy-shaped table migrates cleanly, and the trigger still fires correctly),
but the ALTER-then-trigger order is kept anyway as the more conservative default; see the
rationale in family.py. The only cross-context coupling is the single ``executescript(SCHEMA)``
barrier that every later step depends on.

It stays whole for a stronger reason: it is the single auditable place where the whole ordered
sequence is visible at once. The one historical production ordering bug -- the GATE-2 finding
recorded below at the ``DROP INDEX`` -- was itself intra-context, so a per-context split would not
have prevented it; it would only have scattered the sequence that makes such a bug reviewable.
"""
from __future__ import annotations

import sqlite3

from algua.registry.db._util import _add_missing_columns
from algua.registry.db.constants import SCHEMA_VERSION
from algua.registry.db.core import _migrate_shortlisted_to_candidate
from algua.registry.db.gate import _backfill_fdr_cohorts, _relabel_fdr_cohorts_for_current_size
from algua.registry.db.holdout import _backfill_holdout_intervals
from algua.registry.db.schema import SCHEMA


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
    conn.executescript(SCHEMA)
    # v40: advisory shadow lane deleted (simplification stage 1). Idempotent drop; the rows were
    # advisory-only (never gate evidence), so no export is taken.
    conn.execute("DROP TABLE IF EXISTS shadow_evaluations")
    # v41: standalone factor-eval layer deleted (simplification stage 1). Idempotent drop; the
    # rows were advisory-only (never gate evidence), so no export is taken.
    conn.execute("DROP TABLE IF EXISTS factor_evaluations")
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
    # v24 (#211): trial-Sharpe dispersion columns for the DSR evidence layer. NULL on pre-existing
    # rows — old sweep rows lack stats and the pooled accessor returns None (fail closed).
    _add_missing_columns(conn, "search_trials", {
        "trial_sharpe_count": "INTEGER",
        "trial_sharpe_mean": "REAL",
        "trial_sharpe_var_ann": "REAL",
    })
    # v25 (#219): factor_evaluations is a brand-new table — `executescript(SCHEMA)` above
    # creates it via `CREATE TABLE IF NOT EXISTS`. No _add_missing_columns needed.
    # (later dropped in v41 — see the DROP TABLE above)
    # v26 (#220): FDR accounting columns for the LORD++ alpha-wealth ledger. NULL on pre-existing
    # rows — legacy evaluations are excluded from the FDR stream by WHERE fdr_binding=1 (fail
    # closed). The partial unique index is created AFTER the columns exist (it references
    # fdr_test_index which isn't in the base DDL), so it lives here rather than in SCHEMA.
    _add_missing_columns(conn, "gate_evaluations", {
        "fdr_binding": "INTEGER",
        "fdr_p_value": "REAL",
        "fdr_alpha_level": "REAL",
        "fdr_rejected": "INTEGER",
        "fdr_test_index": "INTEGER",
    })
    # v33 (#324): count-triggered cohort restarts. fdr_test_index is no longer a GLOBAL lifetime
    # counter — it restarts at 1 within each cohort of FDR_COHORT_SIZE binding tests. Add fdr_cohort
    # and replace the global-unique index with a (cohort, index) composite so the same within-cohort
    # position in different cohorts does not false-conflict.
    _add_missing_columns(conn, "gate_evaluations", {"fdr_cohort": "INTEGER"})
    # DROP the old global-unique index BEFORE the backfill. On a legacy DB with > FDR_COHORT_SIZE
    # binding rows, the backfill rewrites the 65th row's fdr_test_index to 1 — which would collide
    # with row 1 under the still-present global unique index and abort the migration mid-write. The
    # new composite index is created AFTER the backfill (once every row carries its within-cohort
    # (cohort, index) pair), so no window exists where either index is violated. (GATE-2 finding.)
    conn.execute("DROP INDEX IF EXISTS ix_gate_evaluations_fdr_index")
    _backfill_fdr_cohorts(conn)
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ix_gate_evaluations_fdr_cohort_index"
        " ON gate_evaluations(fdr_cohort, fdr_test_index) WHERE fdr_binding=1"
    )
    # #529: FDR_COHORT_SIZE was recalibrated (64 -> 8), which re-scopes the cohort partition.
    # Binding rows labeled under the OLD size carry stale (fdr_cohort, fdr_test_index) — e.g. a row
    # at global ordinal 9 was (cohort=0, index=9) under N=64 but must be (cohort=1, index=1) under
    # N=8. A stale within-cohort index > the new size fails fdr_stream_state's check + wedges every
    # future binding promotion. Re-partition ALL binding rows to the CURRENT size (idempotent; a
    # steady-state DB is a no-op). MUST run after _backfill_fdr_cohorts (which fills NULL labels).
    _relabel_fdr_cohorts_for_current_size(conn)
    # v26 (#222): backtest_returns is a brand-new table; executescript creates it.
    # v26 (#222): family registry tables (families/family_members/family_parents/family_events).
    # All brand-new tables; executescript(SCHEMA) above creates them (CREATE TABLE IF NOT EXISTS).
    # No _add_missing_columns needed for new tables.
    # v26 (#222): family breadth audit columns on gate_evaluations (Task 5).
    _add_missing_columns(conn, "gate_evaluations", {
        "family_id": "INTEGER",
        "family_lifetime_effective": "INTEGER",
    })
    # v37 (#524): the durable family breadth prior + founder→gate audit link + persisted member
    # profile columns. Additive with DEFAULT/nullable, so SQLite's ALTER TABLE ADD COLUMN accepts
    # them on a populated table (every existing family row gets seeded_prior_combos=0/founder
    # NULL = a fresh zero-prior, non-agent-founded family — all current anti-reset numbers
    # unchanged). ALTER cannot add a CHECK, so the seeded_prior_combos>=0 invariant on legacy DBs is
    # enforced in app code (the §5.1 mint asserts seed>0 before its INSERT; the reader treats a
    # negative as corruption). The append-only TRIGGERS on the five classifier-read tables are
    # created by the executescript(SCHEMA) bootstrap ABOVE; but on a LEGACY DB with un-materialised
    # member profiles, the store-layer _materialise_legacy_member_profiles() one-time NULL→value
    # backfill runs AFTER migrate() (from the store bootstrap, where module loads are legal) — the
    # family_members UPDATE trigger explicitly permits that NULL→value flip, so ordering is safe.
    _add_missing_columns(conn, "families", {
        "seeded_prior_combos": "INTEGER NOT NULL DEFAULT 0",
        "founder_gate_id": "INTEGER",
    })
    _add_missing_columns(conn, "family_members", {
        "member_code_hash": "TEXT",
        "member_factors_json": "TEXT",
    })
    # v37 (#524, R9-H1): the family_members BEFORE UPDATE append-only trigger references the just-
    # added member_code_hash/member_factors_json columns, so it is created HERE (after the ALTER),
    # not in SCHEMA (where a legacy family_members table would not yet have those columns — SQLite
    # would not reject the CREATE, since it resolves trigger column references at fire time, but the
    # trigger would be unfireable until the ALTER landed). Permits exactly two one-way flips: the
    # removed_at tombstone (NULL→ts) and the one-time legacy profile materialisation (NULL→value).
    conn.execute(
        "CREATE TRIGGER IF NOT EXISTS trg_family_members_append_only_upd"
        " BEFORE UPDATE ON family_members WHEN NOT ("
        "  (OLD.removed_at IS NULL AND NEW.removed_at IS NOT NULL"
        "     AND NEW.member_code_hash IS OLD.member_code_hash"
        "     AND NEW.member_factors_json IS OLD.member_factors_json"
        "     AND NEW.family_id=OLD.family_id AND NEW.strategy_name=OLD.strategy_name"
        "     AND NEW.joined_at=OLD.joined_at AND NEW.joined_by_actor=OLD.joined_by_actor)"
        "  OR"
        "  (OLD.member_code_hash IS NULL AND NEW.member_code_hash IS NOT NULL"
        "     AND NEW.removed_at IS OLD.removed_at AND NEW.family_id=OLD.family_id"
        "     AND NEW.strategy_name=OLD.strategy_name AND NEW.joined_at=OLD.joined_at"
        "     AND NEW.joined_by_actor=OLD.joined_by_actor)"
        " ) BEGIN SELECT RAISE(ABORT,"
        " 'family_members: only removed_at or one-time profile materialise (#524)'); END;"
    )
    # v28 (#250): live_activity_quarantine is a brand-new dead-letter table; executescript(SCHEMA)
    # above creates it (CREATE TABLE IF NOT EXISTS). No _add_missing_columns needed.
    # v29 (#132): PIT sidecar snapshot provenance on the gate audit row. Additive nullable — legacy
    # rows stay NULL (no backfill; pre-#132 promotions had no PIT snapshot).
    _add_missing_columns(
        conn, "gate_evaluations",
        {"fundamentals_snapshot": "TEXT", "news_snapshot": "TEXT"})
    # v30 (#249): paper_venue_* (orders/fills/activities/cursor/quarantine) are brand-new tables;
    # executescript(SCHEMA) above creates them (CREATE TABLE IF NOT EXISTS).
    # v32 (#332): negative_results is a brand-new advisory table; executescript(SCHEMA) above
    # creates it (CREATE TABLE IF NOT EXISTS). No _add_missing_columns needed.
    # v33 (#324): fdr_cohort column + (cohort, index) composite unique index + legacy backfill
    # handled above (before the index swap so the new composite index sees the rewritten indices).
    # v34 (#392): shadow_evaluations is a brand-new ADVISORY table; executescript(SCHEMA) above
    # creates it (CREATE TABLE IF NOT EXISTS). No _add_missing_columns needed.
    # (later dropped in v40 — see the DROP TABLE above)
    # v35 (#329): actor_challenges is a brand-new table; executescript(SCHEMA) above creates it
    # (CREATE TABLE IF NOT EXISTS). No _add_missing_columns needed.
    # v36 (#485): attempt_token on gate_evaluations — the merge-back driver's per-attempt idem
    # key. Additive nullable (NULL for every existing/non-driver row). The partial unique index is
    # created AFTER the column exists (it references attempt_token, absent from an existing DB's
    # table until _add_missing_columns runs), so it lives here rather than in SCHEMA.
    _add_missing_columns(conn, "gate_evaluations", {"attempt_token": "TEXT"})
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ux_gate_evaluations_attempt_token"
        " ON gate_evaluations(strategy_id, attempt_token) WHERE attempt_token IS NOT NULL"
    )
    # v38 (merge-back authoritative intake): mergeback_evidence is a brand-new marker table;
    # executescript(SCHEMA) above creates it (CREATE TABLE IF NOT EXISTS). No _add_missing_columns
    # needed. Written only by algua.registry.mergeback_intake (the drainer's evidence chokepoint).
    # v39 (#559): universe_name on gate_evaluations — the PIT universe the gate evidence was
    # produced on, so paper deployment binds to the GATED universe, not the module's CONFIG.
    # Additive nullable — legacy rows stay NULL (tick binding falls back to CONFIG with a warning).
    _add_missing_columns(conn, "gate_evaluations", {"universe_name": "TEXT"})
    # v42 (strategy run tracking): runs + run_metrics are brand-new tables; executescript(SCHEMA)
    # above creates them (CREATE TABLE IF NOT EXISTS). No _add_missing_columns needed.
    # v43 (strategy run tracking, fix wave): runs.gate_id — the join a `gate` run needs back to
    # its own gate_evaluations row. The bootstrap CREATE TABLE cannot add a column to an
    # already-created v42 `runs` table, so a v42 DB needs the explicit ALTER. Additive nullable —
    # every non-`gate` row (and every pre-v43 `gate` row) stays NULL by design.
    _add_missing_columns(conn, "runs", {"gate_id": "INTEGER"})
    # v44 (slice 2): series pointers on runs. Additive nullable; every pre-v44 row stays NULL
    # (no backfill — provenance matching is non-unique across re-runs).
    _add_missing_columns(
        conn, "runs", {"series_backtest_id": "INTEGER", "series_holdout_id": "INTEGER"})
    conn.execute(f"PRAGMA user_version={SCHEMA_VERSION};")
    conn.commit()
