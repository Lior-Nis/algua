"""``RunLedger`` — the economic-layer evaluation ledger (``runs`` + ``run_metrics``).

The only module that embeds ``runs`` SQL. Mirrors ``search_breadth.py``: plain mixin, caller-owned
connection, ``with self._conn:`` per write.
"""
from __future__ import annotations

import json
import math
import sqlite3
from typing import Any

from algua.registry.db import MAX_PERSISTED_TRIALS, METRIC_SCHEMA_VERSION
from algua.registry.store._util import _now

#: The fixed metric vocabulary v1 (spec §4.2). A key outside this set is a caller bug, not a
#: reason to widen the table: it either belongs in `extra_metrics` (the overflow tail) or it earns
#: a column and a METRIC_SCHEMA_VERSION bump.
METRIC_COLUMNS: tuple[str, ...] = (
    "sharpe_is", "sharpe_oos", "sharpe_realized",
    "sortino_is", "sortino_oos",
    "total_return_is", "total_return_oos",
    "max_drawdown_is", "max_drawdown_oos",
    "ann_vol_is", "ann_vol_oos",
    "cagr_is", "calmar_is",
    "n_obs_is", "n_obs_oos",
    "mean_window_sharpe", "std_window_sharpe", "min_window_sharpe", "pct_positive_windows",
)

#: The metrics a sweep TRIAL can carry. A trial is one grid point evaluated across the
#: walk-forward windows — it has no holdout segment (the holdout is withheld until the gate),
#: so the _oos and full-period _is metrics are not merely unstored here, they are undefined.
#: Validating trials against the FULL vocabulary would let a recognised-but-unpersisted key
#: pass validation and then vanish.
SWEEP_TRIAL_METRIC_COLUMNS: tuple[str, ...] = (
    "mean_window_sharpe", "std_window_sharpe", "min_window_sharpe", "pct_positive_windows",
)

#: Provenance columns a caller may set. Kept explicit (not `**kwargs` into SQL) so a typo is a
#: ValueError rather than a silently-dropped field.
PROVENANCE_COLUMNS: tuple[str, ...] = (
    "code_hash", "config_hash", "dependency_hash", "data_source", "snapshot_id",
    "universe_name", "fundamentals_snapshot", "news_snapshot", "delisting_snapshot",
    "seed", "timeframe", "period_start", "period_end",
)

_KINDS = frozenset({"backtest", "walk_forward", "sweep", "sweep_trial", "gate"})


def _finite_or_none(value: Any) -> float | None:
    """NaN/inf are not measurements. Store NULL rather than a value that poisons every
    aggregate downstream."""
    if value is None:
        return None
    v = float(value)
    return v if math.isfinite(v) else None


class RunLedgerMixin:
    _conn: sqlite3.Connection

    def record_run(  # noqa: PLR0913 — a ledger row genuinely has this many independent parts
        self,
        kind: str,
        strategy_name: str,
        *,
        strategy_id: int | None = None,
        derived_from: list[int] | None = None,
        components: list[dict[str, Any]] | None = None,
        provenance: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
        metrics: dict[str, float | int | None] | None = None,
        extra_metrics: dict[str, float | None] | None = None,
        passed: bool | None = None,
        trials_truncated_at: int | None = None,
    ) -> int:
        """Insert one run row (and its overflow metrics) and return its id, in its OWN transaction.

        `metrics` keys MUST come from `METRIC_COLUMNS`; anything else raises. `extra_metrics` is
        the free-form overflow tail and accepts any key. This is `_insert_run_locked` plus
        `with self._conn:` — the public entry point for a caller with no transaction of its own.
        """
        with self._conn:
            return self._insert_run_locked(
                kind, strategy_name,
                strategy_id=strategy_id, derived_from=derived_from, components=components,
                provenance=provenance, config=config, metrics=metrics,
                extra_metrics=extra_metrics, passed=passed,
                trials_truncated_at=trials_truncated_at,
            )

    def _insert_run_locked(self, kind: str, strategy_name: str, **kwargs: Any) -> int:
        """The `record_run` INSERT with NO transaction of its own — for callers that already hold
        one (the gate's `BEGIN IMMEDIATE` block; see `GateLedgerMixin
        .record_gate_with_fdr_and_maybe_promote`). `record_run` is this plus `with self._conn:`.

        Takes `**kwargs` (only the store itself calls this — never CLI/research code), but still
        validates every key against `record_run`'s own keyword set: an unexpected key would
        otherwise silently vanish into an unused dict entry instead of raising, and a typo in a
        gate's `run_row` dict must not pass silently.
        """
        allowed = {
            "strategy_id", "derived_from", "components", "provenance", "config",
            "metrics", "extra_metrics", "passed", "trials_truncated_at",
        }
        unknown = set(kwargs) - allowed
        if unknown:
            raise ValueError(
                f"_insert_run_locked() got unexpected keyword argument(s): {sorted(unknown)}")
        strategy_id: int | None = kwargs.get("strategy_id")
        derived_from: list[int] | None = kwargs.get("derived_from")
        components: list[dict[str, Any]] | None = kwargs.get("components")
        provenance: dict[str, Any] | None = kwargs.get("provenance")
        config: dict[str, Any] | None = kwargs.get("config")
        metrics: dict[str, float | int | None] | None = kwargs.get("metrics")
        extra_metrics: dict[str, float | None] | None = kwargs.get("extra_metrics")
        passed: bool | None = kwargs.get("passed")
        trials_truncated_at: int | None = kwargs.get("trials_truncated_at")

        if kind not in _KINDS:
            raise ValueError(f"unknown run kind {kind!r}; expected one of {sorted(_KINDS)}")
        prov = dict(provenance or {})
        for key in prov:
            if key not in PROVENANCE_COLUMNS:
                raise ValueError(f"{key!r} is not a provenance column")
        mets = dict(metrics or {})
        for key in mets:
            if key not in METRIC_COLUMNS:
                raise ValueError(
                    f"{key!r} is not in the fixed metric vocabulary; "
                    f"pass it via extra_metrics or give it a column")

        columns = ["kind", "strategy_name", "strategy_id", "created_at",
                   "metric_schema_version", "derived_from", "components", "config_json",
                   "passed", "trials_truncated_at"]
        values: list[Any] = [
            kind, strategy_name, strategy_id, _now(), METRIC_SCHEMA_VERSION,
            json.dumps(list(derived_from or [])),
            json.dumps(list(components or [])),
            json.dumps(config or {}, sort_keys=True, default=str),
            None if passed is None else int(passed),
            trials_truncated_at,
        ]
        for key in PROVENANCE_COLUMNS:
            if key in prov:
                columns.append(key)
                values.append(prov[key])
        for key in METRIC_COLUMNS:
            if key in mets:
                columns.append(key)
                # n_obs_* are integer counts; everything else is a float that must be finite.
                raw = mets[key]
                values.append(
                    (None if raw is None else int(raw)) if key.startswith("n_obs")
                    else _finite_or_none(raw)
                )
        placeholders = ",".join("?" for _ in columns)
        cur = self._conn.execute(
            f"INSERT INTO runs({','.join(columns)}) VALUES ({placeholders})", values)
        rowid = cur.lastrowid
        assert rowid is not None  # a successful INSERT always sets lastrowid
        if extra_metrics:
            self._conn.executemany(
                "INSERT OR REPLACE INTO run_metrics(run_id, key, value) VALUES (?,?,?)",
                [(rowid, k, _finite_or_none(v)) for k, v in extra_metrics.items()],
            )
        return rowid

    def record_sweep_trials(
        self, parent_run_id: int, strategy_name: str, trials: list[dict[str, Any]],
    ) -> tuple[int, int | None]:
        """Write up to ``MAX_PERSISTED_TRIALS`` child runs for a sweep, in ONE transaction.

        Returns ``(n_written, truncated_at)``; ``truncated_at`` is None when the whole trial set
        was persisted. The caller stamps ``truncated_at`` onto the parent run — a silently
        truncated set would make the funnel-wide distribution lie about the breadth it depicts.

        One writer, one transaction, at the END of the sweep: ``sweep()`` fans combos out across
        processes, and 70 concurrent writers against the governance DB is the wrong shape.
        """
        cap = MAX_PERSISTED_TRIALS
        kept = trials[:cap]
        truncated_at = cap if len(trials) > cap else None
        now = _now()
        lineage = json.dumps([parent_run_id])
        rows = []
        for trial in kept:
            mets = dict(trial.get("metrics") or {})
            for key in mets:
                if key not in SWEEP_TRIAL_METRIC_COLUMNS:
                    raise ValueError(
                        f"{key!r} is not a sweep-trial metric; "
                        f"expected one of {sorted(SWEEP_TRIAL_METRIC_COLUMNS)}")
            rows.append((
                strategy_name, now, lineage,
                json.dumps(trial.get("config") or {}, sort_keys=True, default=str),
                trial.get("config_hash"),
                _finite_or_none(mets.get("mean_window_sharpe")),
                _finite_or_none(mets.get("std_window_sharpe")),
                _finite_or_none(mets.get("min_window_sharpe")),
                _finite_or_none(mets.get("pct_positive_windows")),
            ))
        with self._conn:
            self._conn.executemany(
                "INSERT INTO runs(kind, strategy_name, created_at, metric_schema_version,"
                " derived_from, components, config_json, config_hash,"
                " mean_window_sharpe, std_window_sharpe, min_window_sharpe, pct_positive_windows)"
                f" VALUES ('sweep_trial',?,?,{METRIC_SCHEMA_VERSION},?,'[]',?,?,?,?,?,?)",
                rows,
            )
        return len(kept), truncated_at

    def stamp_trials_truncated(self, run_id: int, truncated_at: int) -> None:
        """Mark a `sweep` parent whose trial set was capped. Never inferred by a reader — a
        truncated distribution must announce itself."""
        with self._conn:
            self._conn.execute(
                "UPDATE runs SET trials_truncated_at=? WHERE id=?", (truncated_at, run_id))

    def get_run(self, run_id: int) -> sqlite3.Row | None:
        return self._conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()

    def list_runs(
        self, *, kind: str | None = None, strategy_name: str | None = None,
        sort: str | None = None, limit: int = 100,
    ) -> list[sqlite3.Row]:
        """Scalar run rows, newest first (or best-first when ``sort`` names a metric).

        ``sort`` is interpolated into the SQL — it MUST be allow-listed against the fixed
        vocabulary, never taken from caller input unchecked.
        """
        if sort is not None and sort not in METRIC_COLUMNS:
            raise ValueError(f"{sort!r} is not a sortable metric")
        clauses: list[str] = []
        params: list[Any] = []
        if kind is not None:
            clauses.append("kind=?")
            params.append(kind)
        if strategy_name is not None:
            clauses.append("strategy_name=?")
            params.append(strategy_name)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        # NULLS LAST: a run with no measurement must never outrank one with a real number.
        order = (f" ORDER BY {sort} IS NULL, {sort} DESC, id DESC"
                 if sort else " ORDER BY id DESC")
        params.append(int(limit))
        return list(self._conn.execute(
            f"SELECT * FROM runs{where}{order} LIMIT ?", params).fetchall())
