"""``HoldoutLedger`` — single-use holdout-window reservations/burns + OOS return-vector
persistence."""
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

import numpy as np

from algua.registry.store._util import _now


class HoldoutLedgerMixin:
    _conn: sqlite3.Connection

    def reserve_holdout(
        self,
        strategy_id: int,
        *,
        data_source: str,
        snapshot_id: str | None,
        period_start: str,
        period_end: str,
        holdout_frac: float,
        holdout_start: str,
        holdout_end: str,
        allow_reuse: bool,
    ) -> tuple[int, bool]:
        # TOP-LEVEL ONLY. A manual BEGIN IMMEDIATE inside an already-open transaction raises
        # "cannot start a transaction within a transaction", and a blanket rollback below could
        # roll back a caller's surrounding tx. Fail loudly so the contract is enforced, not assumed.
        if self._conn.in_transaction:
            raise RuntimeError(
                "reserve_holdout must be called at top level, not inside an open transaction")
        # Single-use identity is the OOS INTERVAL [holdout_start, holdout_end] for the strategy,
        # PROVENANCE-INDEPENDENT (#205): the same OOS calendar window is burn-once regardless of how
        # the bars were reached (snapshot S, a different snapshot S2, or a provider P). data_source/
        # snapshot_id are persisted as EVIDENCE only, never matched on (was: a per-provenance
        # bucket, which let the same physical window be burned twice across provenance — #205).
        # Defensive (GATE-1): an inverted incoming interval (start > end) would slip both the NULL
        # branch and the overlap test below and fail OPEN, so reject it. holdout_window yields a
        # well-formed interval (idx[train_n] <= idx[-1]); this guards the primitive vs. a caller.
        if holdout_start > holdout_end:
            raise ValueError(
                f"invalid holdout interval: start {holdout_start!r} > end {holdout_end!r}")
        # Match identity is the OOS INTERVAL [holdout_start, holdout_end] — the exact bars
        # walk_forward burns (#192), NOT (full-period overlap, holdout_frac): a different
        # --holdout-frac that lands on overlapping OOS bars must NOT escape the guard. The standard
        # interval-overlap test (a.start <= b.end AND b.start <= a.end). A row with a NULL interval
        # (legacy/old-code reservation, before this column existed) matches UNCONDITIONALLY — fail
        # closed. period_*/holdout_frac are persisted as EVIDENCE only, never matched on. Matches
        # ALL rows (pending reservation OR committed burn) — no committed_at filter — so a pending
        # row blocks too.
        # BEGIN IMMEDIATE takes the write lock up front so the overlap SELECT + INSERT are one
        # atomic critical section: two concurrent reserves can't both see "no overlap" and both
        # insert. BaseException (not Exception) so a KeyboardInterrupt/SystemExit still releases the
        # lock via rollback.
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            row = self._conn.execute(
                "SELECT 1 FROM holdout_evaluations WHERE strategy_id = ?"
                " AND (holdout_start IS NULL OR holdout_end IS NULL"
                "      OR (holdout_start <= ? AND ? <= holdout_end)) LIMIT 1",
                (strategy_id, holdout_end, holdout_start),
            ).fetchone()
            overlap = row is not None
            if overlap and not allow_reuse:
                raise ValueError(
                    "holdout already consumed: an overlapping out-of-sample window was already "
                    "evaluated. Use fresh out-of-sample data, or --allow-holdout-reuse "
                    "(--actor human) to override and accept the statistical cost.")
            reused = bool(overlap)  # only reachable here with allow_reuse when overlap is True
            cur = self._conn.execute(
                "INSERT INTO holdout_evaluations"
                "(strategy_id, data_source, snapshot_id, period_start, period_end, holdout_frac,"
                " config_hash, reused, created_at, committed_at, holdout_start, holdout_end)"
                " VALUES (?,?,?,?,?,?,?,?,?,NULL,?,?)",
                (strategy_id, data_source, snapshot_id, period_start, period_end, holdout_frac,
                 "", int(reused), _now(), holdout_start, holdout_end),
            )
            self._conn.commit()
        except BaseException:
            self._conn.rollback()
            raise
        rowid = cur.lastrowid
        assert rowid is not None  # a successful INSERT always sets lastrowid
        return rowid, reused

    def finalize_holdout_reservation(
        self, reservation_id: int, *, config_hash: str, strategy_id: int
    ) -> None:
        with self._conn:  # UPDATE + guard commit together or roll back
            cur = self._conn.execute(
                "UPDATE holdout_evaluations SET committed_at = ?, config_hash = ?"
                " WHERE id = ? AND strategy_id = ? AND committed_at IS NULL",
                (_now(), config_hash, reservation_id, strategy_id),
            )
            if cur.rowcount != 1:
                # Raise INSIDE the with so the mismatch rolls back (mirrors apply_transition's
                # gate-consume guard). Guards a double-finalize or a vanished/released row. Raise,
                # not assert — asserts strip under python -O.
                raise ValueError(
                    f"holdout reservation {reservation_id} is missing, already committed, "
                    f"or strategy_id mismatch")

    def release_holdout_reservation(self, reservation_id: int) -> None:
        with self._conn:
            # No guard: a release after a finalize/crash is a harmless no-op (rowcount 0). Never
            # touches a committed burn (committed_at IS NULL filter).
            self._conn.execute(
                "DELETE FROM holdout_evaluations WHERE id = ? AND committed_at IS NULL",
                (reservation_id,),
            )

    def record_holdout_returns(
        self, holdout_evaluation_id: int, strategy_id: int, *,
        holdout_start: str, holdout_end: str,
        returns: list[float], bar_dates: list[str],
    ) -> int:
        """Persist ONE OOS return vector for a committed holdout burn (#221 Slice 1). Separate
        transaction from the burn (the burn committed at on_peek). IDEMPOTENT on identical content:
        if a row already exists with the same (strategy_id, holdout_start, holdout_end, n_bars,
        returns_blob, bar_dates_blob), the existing row id is returned without error — enabling
        the "returns row exists, gate row missing → re-run" reconciliation path. A row that exists
        with DIFFERENT content raises ValueError (genuine conflicting double-write).
        Validation is fail-closed, INSIDE the transaction so check+insert are atomic (TOCTOU)."""
        n_bars = len(returns)
        if n_bars != len(bar_dates):
            raise ValueError(
                f"holdout_returns length mismatch: {n_bars} returns vs {len(bar_dates)} dates")
        if n_bars == 0:
            raise ValueError("holdout_returns must have at least one bar")
        # Encoding is CPU-only — do it OUTSIDE the transaction so we don't hold a lock during it.
        returns_blob = np.asarray(returns, dtype="<f8").tobytes()
        bar_dates_blob = "\n".join(bar_dates).encode("utf-8")
        try:
            with self._conn:
                # Concurrency safety here rests on UNIQUE(holdout_evaluation_id) + the caught
                # IntegrityError below + the idempotency check, NOT on these SELECTs: Python's
                # sqlite3 opens the transaction only on the first DML (the INSERT), so a read-only
                # validation runs in autocommit. That is acceptable — no supported API mutates a
                # committed burn's strategy_id/committed_at, and concurrent inserts of the same
                # holdout_evaluation_id are caught by the UNIQUE constraint. The checks are
                # caller-bug defense (missing / uncommitted / mismatched-strategy burn).
                row = self._conn.execute(
                    "SELECT strategy_id, committed_at FROM holdout_evaluations WHERE id = ?",
                    (holdout_evaluation_id,),
                ).fetchone()
                if row is None:
                    raise ValueError(f"holdout_evaluation {holdout_evaluation_id} does not exist")
                if row["committed_at"] is None:
                    raise ValueError(
                        f"holdout_evaluation {holdout_evaluation_id} is not a committed burn")
                if int(row["strategy_id"]) != int(strategy_id):
                    raise ValueError(
                        f"strategy_id {strategy_id} does not match holdout_evaluation "
                        f"{holdout_evaluation_id} (strategy_id {row['strategy_id']})")
                existing = self._conn.execute(
                    "SELECT id, strategy_id, holdout_start, holdout_end, n_bars,"
                    " returns_blob, bar_dates_blob"
                    " FROM holdout_returns WHERE holdout_evaluation_id = ?",
                    (holdout_evaluation_id,),
                ).fetchone()
                if existing is not None:
                    if (int(existing["strategy_id"]) == int(strategy_id)
                            and existing["holdout_start"] == holdout_start
                            and existing["holdout_end"] == holdout_end
                            and int(existing["n_bars"]) == n_bars
                            and bytes(existing["returns_blob"]) == returns_blob
                            and bytes(existing["bar_dates_blob"]) == bar_dates_blob):
                        # idempotent reconciliation: identical content already written
                        return int(existing["id"])
                    raise ValueError(
                        f"holdout_returns already written for holdout_evaluation "
                        f"{holdout_evaluation_id} with different content")
                cur = self._conn.execute(
                    "INSERT INTO holdout_returns"
                    "(holdout_evaluation_id, strategy_id, holdout_start, holdout_end, n_bars,"
                    " returns_blob, bar_dates_blob, created_at)"
                    " VALUES (?,?,?,?,?,?,?,?)",
                    (holdout_evaluation_id, strategy_id, holdout_start, holdout_end, n_bars,
                     returns_blob, bar_dates_blob, _now()),
                )
                rowid = cur.lastrowid
        except sqlite3.IntegrityError as e:
            # Concurrent insert raced between our existence check and INSERT — conflicting write.
            raise ValueError(
                f"holdout_returns already written for holdout_evaluation "
                f"{holdout_evaluation_id} (concurrent insert)") from e
        assert rowid is not None
        return int(rowid)

    def overlapping_holdout_return_streams(
        self, strategy_id: int, holdout_start: str, holdout_end: str, window_days: int
    ) -> list[tuple[list[float], list[str]]]:
        """SIBLING-ONLY cross-strategy read (#221 Slice 1 access control).

        Returns OTHER strategies' OOS return vectors whose holdout interval overlaps
        [holdout_start, holdout_end], burned within the trailing window_days. NEVER
        returns the requesting strategy's own vector. This is the ONLY method that reads
        returns_blob. The caller (promotion.run_gate) is trusted to pass the correct
        strategy_id so self-exclusion holds.

        NOTE: holdout_returns keys by strategy_id (FK to strategies.id) while
        search_trials keys by strategy_name — the asymmetry is intentional: returns
        vectors belong to a specific registered strategy row, whereas search trials are
        recorded before registration.
        """
        # Window filter uses he.created_at (burn time via holdout_evaluations.created_at),
        # NOT hr.created_at (write time of the vector row), so a late-written vector for an
        # out-of-window burn is correctly excluded.
        cutoff = (datetime.now(UTC) - timedelta(days=window_days)).isoformat()
        rows = self._conn.execute(
            "SELECT hr.n_bars AS n_bars, hr.returns_blob AS rb, hr.bar_dates_blob AS db "
            "FROM holdout_returns hr JOIN holdout_evaluations he"
            " ON hr.holdout_evaluation_id = he.id "
            "WHERE hr.strategy_id != ? AND he.created_at >= ?"
            "  AND hr.holdout_start <= ? AND ? <= hr.holdout_end",
            (strategy_id, cutoff, holdout_end, holdout_start),
        ).fetchall()
        out: list[tuple[list[float], list[str]]] = []
        for r in rows:
            vec = np.frombuffer(r["rb"], dtype="<f8")
            if len(vec) != int(r["n_bars"]):
                raise ValueError(
                    f"corrupt holdout_returns blob: {len(vec)} floats != n_bars {r['n_bars']}")
            dates = r["db"].decode("utf-8").split("\n")
            if len(dates) != int(r["n_bars"]):
                raise ValueError(
                    f"corrupt holdout_returns bar_dates_blob: "
                    f"{len(dates)} dates != n_bars {r['n_bars']}")
            out.append(([float(x) for x in vec], dates))
        return out
