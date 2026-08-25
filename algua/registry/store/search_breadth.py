"""``SearchBreadthLedger`` — search-trial breadth + trial-Sharpe dispersion (the #211/#221
multiple-testing inputs)."""
from __future__ import annotations

import math
import sqlite3
from datetime import UTC, datetime, timedelta

from algua.registry.db import MAX_N_COMBOS
from algua.registry.repository import FunnelFloor
from algua.registry.store._util import _now
from algua.research.dsr import MIN_FUNNEL_FLOOR_STRATEGIES


def _pool_trial_sharpe_var(triples: list[tuple[int, float, float]]) -> float | None:
    """Exact pooled SAMPLE variance (ddof=1) of trial Sharpes from ``(count, mean, var)`` triples.
    ``None`` for empty input; ``0.0`` for total count <= 1. Callers must pre-validate each triple
    (finite mean/var, count >= 1, var >= 0); this helper assumes clean triples."""
    if not triples:
        return None
    total_n = sum(n for n, _, _ in triples)
    if total_n <= 1:
        return 0.0
    grand_mean = sum(n * m for n, m, _ in triples) / total_n
    sse = sum((n - 1) * v + n * (m - grand_mean) ** 2 for n, m, v in triples)
    return sse / (total_n - 1)


def _validated_triples(rows) -> list[tuple[int, float, float]] | None:
    """Validate raw (n, mean, var) DB rows. Returns None (fail closed) if ANY row has a
    NULL/NaN/inf/negative stat — NULL rows are NEVER silently skipped."""
    triples: list[tuple[int, float, float]] = []
    for r in rows:
        n, mean, var = r["n"], r["mean"], r["var"]
        if n is None or mean is None or var is None:
            return None
        if not (math.isfinite(mean) and math.isfinite(var)) or int(n) < 1 or var < 0.0:
            return None
        triples.append((int(n), float(mean), float(var)))
    return triples


class SearchBreadthLedgerMixin:
    _conn: sqlite3.Connection

    def record_search_trial(
        self, strategy_name: str, n_combos: int, grid_json: str,
        *, trial_sharpe_count: int | None = None,
        trial_sharpe_mean: float | None = None,
        trial_sharpe_var_ann: float | None = None,
    ) -> int:
        # v37 (#524, R9-M3): type-safe + bounded so a corrupt/overlarge row can never overflow the
        # funnel-lifetime seed SUM. `type(n_combos) is int` (NOT isinstance) excludes `bool` — an
        # `int` subclass that would otherwise store True as 1 and pass a typeof='integer' check.
        if type(n_combos) is not int or not (1 <= n_combos <= MAX_N_COMBOS):
            raise ValueError(
                f"n_combos must be an int in [1, {MAX_N_COMBOS}], got {n_combos!r}")
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO search_trials(strategy_name, n_combos, grid_json, created_at,"
                " trial_sharpe_count, trial_sharpe_mean, trial_sharpe_var_ann)"
                " VALUES (?,?,?,?,?,?,?)",
                (strategy_name, n_combos, grid_json, _now(),
                 trial_sharpe_count, trial_sharpe_mean, trial_sharpe_var_ann),
            )
        rowid = cur.lastrowid
        assert rowid is not None  # a successful INSERT always sets lastrowid
        return rowid

    def pooled_trial_sharpe_var(self, strategy_name: str) -> float | None:
        """Exact pooled SAMPLE variance (ddof=1) of the strategy's trial Sharpes across all its
        search_trials rows. Returns None (fail closed) if there are no rows OR any contributing
        row has a NULL/NaN/inf/negative count|mean|var. NULL rows are NEVER silently skipped."""
        rows = self._conn.execute(
            "SELECT trial_sharpe_count AS n, trial_sharpe_mean AS mean,"
            " trial_sharpe_var_ann AS var FROM search_trials WHERE strategy_name=?"
            " ORDER BY id",  # deterministic row order -> bit-identical pooled var (#339 CAS)
            (strategy_name,),
        ).fetchall()
        if not rows:
            return None
        triples = _validated_triples(rows)
        if triples is None:
            return None
        return _pool_trial_sharpe_var(triples)

    def total_search_combos(self, strategy_name: str) -> int:
        # COALESCE so an empty result (no trials) reads as 0 rather than NULL.
        row = self._conn.execute(
            "SELECT COALESCE(SUM(n_combos), 0) AS total FROM search_trials WHERE strategy_name=?",
            (strategy_name,),
        ).fetchone()
        return int(row["total"])

    def funnel_lifetime_search_combos(self) -> int:
        # v37 (#524, R9-M3): funnel-wide LIFETIME search effort. WHERE-filtered to well-typed
        # in-range rows so each summand is <= MAX_N_COMBOS (overflow-safe) and a corrupt/overlarge
        # legacy row is EXCLUDED (contributes 0) rather than overflowing or coercing. This is the
        # EXACT summation the §5.1 mint seed uses, so the accessor and the seed agree. Always >= 0.
        row = self._conn.execute(
            "SELECT COALESCE(SUM(n_combos), 0) AS total FROM search_trials"
            " WHERE typeof(n_combos)='integer' AND n_combos BETWEEN 1 AND ?",
            (MAX_N_COMBOS,),
        ).fetchone()
        return int(row["total"])

    def search_trials_fingerprint(self) -> tuple[int, int]:
        # search_trials is append-only (INSERT-only, AUTOINCREMENT PK), so (COUNT(*), MAX(id))
        # strictly increases on every insert and uniquely fingerprints the whole row set — the
        # row-identity half of the #339 funnel CAS.
        row = self._conn.execute(
            "SELECT COUNT(*) AS n, COALESCE(MAX(id), 0) AS mx FROM search_trials",
        ).fetchone()
        return int(row["n"]), int(row["mx"])

    def windowed_search_combos(self, window_days: int) -> int:
        """Sum of ``n_combos`` across ALL strategies' search_trials recorded within the trailing
        ``window_days`` (funnel-wide search effort for Wall A). ISO-8601 UTC timestamps compare
        lexicographically in chronological order, so a string `>=` on created_at is correct."""
        cutoff = (datetime.now(UTC) - timedelta(days=window_days)).isoformat()
        row = self._conn.execute(
            "SELECT COALESCE(SUM(n_combos), 0) AS total FROM search_trials WHERE created_at >= ?",
            (cutoff,),
        ).fetchone()
        return int(row["total"])

    def funnel_trial_sharpe_var(self, window_days: int) -> FunnelFloor:
        """Per-strategy pooling FIRST (anti-gaming: one vote per strategy regardless of combo
        count), then MEAN across strategies with at least one search_trials row in the trailing
        ``window_days``. A selected strategy pools ALL its rows (the window selects strategies, it
        does NOT slice rows). A strategy with any NULL/NaN/inf stat row is excluded. Returns
        FunnelFloor(None, ...) when fewer than _MIN_FUNNEL_FLOOR_STRATEGIES finite variances exist
        (fail-open -> Phase-1 behavior). ISO-8601 UTC timestamps sort lexically, so a string `>=`
        on created_at is chronological."""
        cutoff = (datetime.now(UTC) - timedelta(days=window_days)).isoformat()
        # SELECT all rows of every strategy that has at least one in-window row. The window
        # filters which STRATEGIES are eligible; pooling then uses every row of each.
        rows = self._conn.execute(
            "SELECT strategy_name AS name, trial_sharpe_count AS n, trial_sharpe_mean AS mean,"
            " trial_sharpe_var_ann AS var FROM search_trials WHERE strategy_name IN"
            " (SELECT DISTINCT strategy_name FROM search_trials WHERE created_at >= ?)"
            " ORDER BY strategy_name, id",  # deterministic pooling order -> stable floor (#339 CAS)
            (cutoff,),
        ).fetchall()
        by_strategy: dict[str, list] = {}
        for r in rows:
            by_strategy.setdefault(r["name"], []).append(r)
        per_strategy_vars: list[float] = []
        total_rows = 0
        for name_rows in by_strategy.values():
            triples = _validated_triples(name_rows)
            if triples is None:
                continue  # excluded: a NULL/non-finite stat in any of this strategy's rows
            var_s = _pool_trial_sharpe_var(triples)
            if var_s is None or not math.isfinite(var_s):
                continue
            per_strategy_vars.append(var_s)
            total_rows += len(name_rows)
        n_strategies = len(per_strategy_vars)
        if n_strategies < MIN_FUNNEL_FLOOR_STRATEGIES:
            return FunnelFloor(None, n_strategies, total_rows)
        return FunnelFloor(sum(per_strategy_vars) / n_strategies, n_strategies, total_rows)
