"""Gate context: ``gate_evaluations``.

The promotion-gate audit trail AND the single-use agent-only BACKTESTED->CANDIDATE token ledger
(operated by ``algua/registry/store/gate.py``), plus the LORD++ FDR cohort partitioning constants
and the two cohort-relabelling migrations.

Kept separate from ``forward_gate.py`` for a 1:1 correspondence with ``store/gate.py`` /
``store/forward_gate.py``. Cross-module coupling: ``families.founder_gate_id`` REFERENCES
``gate_evaluations(id)`` -- a further split must not separate this module from ``family.py``
carelessly.
"""
from __future__ import annotations

import sqlite3

SCHEMA = """
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
    fundamentals_snapshot TEXT,
    news_snapshot TEXT,
    period_start TEXT NOT NULL,
    period_end TEXT NOT NULL,
    holdout_frac REAL NOT NULL,
    actor TEXT NOT NULL,
    decision_json TEXT NOT NULL,
    consumed INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    -- v36 (#485): opaque per-attempt idempotency key stamped by the autonomous merge-back driver so
    -- promote-outcome attribution binds to the branch identity, not the ambient stage. NULL for
    -- every non-driver caller (backward-compatible). A partial unique index on non-null
    -- (strategy_id, attempt_token) makes a second insert of the same token a hard DB error.
    attempt_token TEXT,
    -- v39 (#559): the NAME of the PIT universe this evaluation was gated on (the --universe flag),
    -- so a paper/live deployment can be BOUND to the universe identity its gate evidence was
    -- produced on (resolve_operational_universe). NULL for legacy rows and non-universe runs;
    -- the tick binding treats NULL as config_legacy (CONFIG.universe + a loud warning).
    universe_name TEXT
);
CREATE INDEX IF NOT EXISTS ix_gate_evaluations_strategy ON gate_evaluations(strategy_id);
-- NOTE: the partial unique index on (strategy_id, attempt_token) is created in migrate() AFTER
-- _add_missing_columns adds the column, NOT here: on an existing DB the CREATE TABLE above is a
-- no-op and the column would not yet exist for the index to reference (same reason the FDR index
-- lives in migrate()).
"""


# Count-triggered cohort restarts (#324) + budget-derived recalibration (#529), relocated from
# research/fdr_lord.py (simplification stage 4a — this is registry-owned state: the FDR ledger's
# cohort partitioning is consumed by this module's own migrate()-invoked backfills below, not by
# any research-layer code). The LORD++ stream is partitioned into consecutive, non-overlapping
# COHORTS of exactly FDR_COHORT_SIZE binding tests, assigned by ARRIVAL ORDER. Protected constant —
# changing it re-scopes every historical cohort boundary; see _relabel_fdr_cohorts_for_current_size
# below for what a change requires.
FDR_COHORT_SIZE = 8


def fdr_cohort_position(k: int) -> tuple[int, int]:
    """Map a 1-based GLOBAL binding-test ordinal ``k`` to its ``(cohort_index, within_cohort_t)``.

    ``cohort_index = (k − 1) // FDR_COHORT_SIZE`` (0-based); ``within_cohort_t`` runs 1..
    FDR_COHORT_SIZE and is the position fed to the LORD++ level function for that cohort's
    independent stream. Fails closed (``ValueError``) on ``k < 1`` — a binding ordinal is
    always ≥ 1 by construction, so a non-positive value is a caller bug, not a silent-0 default.
    """
    if k < 1:
        raise ValueError(f"binding-test ordinal k must be >= 1, got {k}")
    return (k - 1) // FDR_COHORT_SIZE, (k - 1) % FDR_COHORT_SIZE + 1


def _backfill_fdr_cohorts(conn: sqlite3.Connection) -> None:
    """Re-partition legacy LORD++ binding rows into cohorts of FDR_COHORT_SIZE (#324).

    Pre-#324 binding rows carried a GLOBAL lifetime ``fdr_test_index`` (1, 2, 3, …) and a NULL
    ``fdr_cohort``. This one-time, idempotent backfill packs them, in ``id`` order, into
    consecutive cohorts: the g-th binding row (g 1-based) gets ``fdr_cohort = (g-1)//N`` and its
    ``fdr_test_index`` is REWRITTEN to the within-cohort position ``(g-1)%N + 1``.

    Guarded by "any binding row with NULL fdr_cohort" so it runs exactly once (a second migrate()
    finds every binding row already assigned and does nothing). Ordering by ``id`` (not by the old
    ``fdr_test_index``) makes the packing robust to any historical index gap.

    Stored ``fdr_alpha_level`` is LEFT FROZEN — it is the historical record of the decision made at
    the time (computed under the old lifetime formula); recomputing it under cohort scoping would
    falsify the audit trail. The stream reader validates only index contiguity + p/alpha finiteness
    + rejected∈{0,1}; it never recomputes a past α, so freezing is correct. The re-partition does
    not change any past pass/fail verdict (``passed`` is untouched); it only re-labels the ledger so
    the new (cohort, index) invariant and composite unique index hold over history. Rows imported
    from the migration MUST use the SAME FDR_COHORT_SIZE the reader/writer use — imported here.

    MUST run after the legacy global-unique fdr index is dropped in migrate() (GATE-2); see
    migrate() for the rationale.
    """
    binding_ids = [
        row["id"]
        for row in conn.execute(
            "SELECT id FROM gate_evaluations WHERE fdr_binding=1 AND fdr_cohort IS NULL ORDER BY id"
        )
    ]
    if not binding_ids:
        return
    # If SOME binding rows already have fdr_cohort (a partial prior backfill / mixed history), the
    # global ordinal must count ALL binding rows, not just the NULL ones, so the packing stays
    # contiguous. Count already-assigned binding rows to offset the global ordinal.
    already = conn.execute(
        "SELECT COUNT(*) AS c FROM gate_evaluations WHERE fdr_binding=1 AND fdr_cohort IS NOT NULL"
    ).fetchone()["c"]
    for offset, row_id in enumerate(binding_ids):
        g = already + offset + 1
        cohort, t = fdr_cohort_position(g)
        conn.execute(
            "UPDATE gate_evaluations SET fdr_cohort=?, fdr_test_index=? WHERE id=?",
            (cohort, t, row_id),
        )


def _relabel_fdr_cohorts_for_current_size(conn: sqlite3.Connection) -> None:
    """Re-partition ALL LORD++ binding rows into cohorts of the CURRENT ``FDR_COHORT_SIZE`` (#529).

    Changing ``FDR_COHORT_SIZE`` re-scopes the cohort partition, so binding rows labeled under a
    PRIOR size carry stale ``(fdr_cohort, fdr_test_index)`` — a within-cohort index greater than the
    new size would fail ``fdr_stream_state``'s integrity check and wedge every future binding
    promotion (``FDR stream integrity failure``). This one-time, idempotent migration re-labels
    every ``fdr_binding=1`` row, in ``id`` order, to ``(cohort, t) = fdr_cohort_position(g)`` under
    the CURRENT size. It is guarded by a mismatch scan, so a steady-state DB (labels already right)
    is a pure no-op. ``fdr_alpha_level`` / ``fdr_rejected`` / ``passed`` are LEFT FROZEN (the
    historical audit record), exactly like :func:`_backfill_fdr_cohorts`: only the partition labels
    move; no past verdict changes. The composite unique index is dropped BEFORE and recreated AFTER
    the rewrite so a row-by-row re-label can never hit a transient ``(cohort, index)`` collision.
    """
    rows = conn.execute(
        "SELECT id, fdr_cohort, fdr_test_index FROM gate_evaluations"
        " WHERE fdr_binding=1 ORDER BY id"
    ).fetchall()
    if not rows:
        return
    targets: list[tuple[int, int, int]] = []
    mismatch = False
    for g, row in enumerate(rows, start=1):
        cohort, t = fdr_cohort_position(g)
        targets.append((cohort, t, row["id"]))
        if row["fdr_cohort"] != cohort or row["fdr_test_index"] != t:
            mismatch = True
    if not mismatch:
        return
    conn.execute("DROP INDEX IF EXISTS ix_gate_evaluations_fdr_cohort_index")
    conn.executemany(
        "UPDATE gate_evaluations SET fdr_cohort=?, fdr_test_index=? WHERE id=?", targets)
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ix_gate_evaluations_fdr_cohort_index"
        " ON gate_evaluations(fdr_cohort, fdr_test_index) WHERE fdr_binding=1"
    )
