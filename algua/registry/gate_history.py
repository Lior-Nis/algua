"""Read-only accessors over the two gate-evaluation ledgers for the `registry gates` audit view.

PURE READS ONLY: this module issues explicit-column SELECTs against ``gate_evaluations`` and
``forward_gate_evaluations`` and nothing else. It is STRUCTURALLY unable to reach
``holdout_returns.returns_blob`` — no join, no reference, no ``SELECT *`` anywhere (the SENSITIVE
marker on ``holdout_returns`` in ``db.py`` is the authority: no CLI accessor may read that vector).

``decision_json`` is never emitted raw. Each row's blob is parsed and projected through a
TWO-LAYER allowlist: (1) the key must appear in a MATERIALIZED frozen constant hand-written from
the decision dataclass's fields (never derived at runtime — a new dataclass field must be
consciously reviewed into the allowlist, enforced by the golden drift test); (2) the value must be
scalar-shaped (scalar, dict of scalars, or — for ``checks`` — a list of dicts reduced to their
scalar name/op/threshold/value/passed fields). Anything failing either layer is dropped and
reported in ``decision_dropped_keys``, so a vector planted in the blob can never transit this view.
"""

from __future__ import annotations

import json
import math
import sqlite3
from typing import Any

# Hand-materialized from the fields of algua.research.gates.GateDecision (NEVER derived at runtime
# from the dataclass: a future field must fail the golden drift test until reviewed in here).
GATE_DECISION_ALLOWLIST: frozenset[str] = frozenset({
    "passed",
    "checks",
    "n_combos",
    "breadth_provenance",
    "base_min_holdout_sharpe",
    "effective_min_holdout_sharpe",
    "own_lifetime_combos",
    "windowed_total_combos",
    "funnel_window_days",
    "pit_ok",
    "pit_override",
    "dsr_binding",
    "dsr_confidence",
    "dsr_skip_reason",
    "dsr_n_trials",
    "dsr_trial_sr_var_ann",
    "dsr_t",
    "dsr_skew",
    "dsr_raw_kurtosis",
    "dsr_funnel_floor_var_ann",
    "dsr_funnel_floor_n_strategies",
    "dsr_funnel_floor_n_total_rows",
    "fdr_binding",
    "fdr_p_value",
    "fdr_alpha_level",
    "fdr_test_index",
    "fdr_rejected",
    "fdr_skip_reason",
    "fdr_cohort",
    "fdr_cohorts_completed",
    "fdr_binding_tests",
    "fdr_discoveries",
    "fdr_expected_false_discoveries",
    "fdr_throttle_window_binding",
    "fdr_throttle_tripped",
    "fdr_throttle_override",
    "fdr_active_cohort_position",
    "fdr_active_cohort_applied_alpha",
    "fdr_expected_false_discoveries_incl_active",
    "returns_available",
    "dsr_bootstrap_binding",
    "dsr_bootstrap_lower",
    "dsr_bootstrap_seed",
    "dsr_bootstrap_b",
    "dsr_bootstrap_block_len",
    "dsr_n_eff",
    "dsr_rho_bar",
    "dsr_n_siblings",
    "regime_method",
    "n_regimes_attempted",
    "n_regimes_surviving",
    "regime_robustness_binding",
    "ir_method",
    "ir_binding",
    "ir_overlap_n",
    "market_beta",
    "ir_alpha_ann",
    "ir_residual_vol_ann",
    "appraisal_ratio",
    "haircut_would_have_blocked",
    "phase3_component_mask",
})

# GateDecision fields deliberately NOT projected. per_regime_sharpes is a bare list of floats —
# the shape contract carries scalars, dicts-of-scalars, and the checks[] reduction only, and a
# bare-list carve-out is exactly the aperture a planted returns vector would ride through.
GATE_DECISION_EXCLUDED: frozenset[str] = frozenset({
    "per_regime_sharpes",
})

# Hand-materialized from the fields of algua.research.forward_gates.ForwardGateDecision.
FORWARD_DECISION_ALLOWLIST: frozenset[str] = frozenset({
    "passed",
    "checks",
})

FORWARD_DECISION_EXCLUDED: frozenset[str] = frozenset()

# Explicit column lists — never SELECT * (a future SENSITIVE column must not leak by default).
# fdr_* on gate_evaluations are migration-added and may be NULL on legacy rows (emitted as nulls).
_GATE_COLUMNS = (
    "id", "passed", "actor", "created_at", "consumed", "pit_ok", "pit_override",
    "holdout_n_bars", "min_holdout_observations", "holdout_frac", "n_funnel",
    "own_lifetime_combos", "windowed_total_combos", "funnel_window_days", "breadth_provenance",
    "data_source", "snapshot_id", "code_hash", "config_hash", "dependency_hash",
    "period_start", "period_end", "fdr_binding", "fdr_p_value", "fdr_alpha_level",
    "fdr_rejected", "fdr_test_index", "fdr_cohort",
)

_FORWARD_COLUMNS = (
    "id", "passed", "n_forward_observations", "min_forward_observations", "session_coverage",
    "realized_sharpe", "holdout_sharpe", "degradation_factor", "sharpe_floor", "realized_vol",
    "min_forward_vol", "realized_max_drawdown", "max_forward_drawdown", "first_tick_id",
    "last_tick_id", "first_tick_ts", "last_tick_ts", "max_staleness_sessions",
    "n_reconcile_failures", "n_concurrent_forward", "account_id", "code_hash", "config_hash",
    "dependency_hash", "actor", "consumed", "created_at",
)

_SCALAR_TYPES = (str, int, float, bool, type(None))

# Bulk-exfiltration bounds: a projected string longer than this, a dict with more entries, or a
# checks list longer than this is dropped whole. The legitimate writers (GateDecision /
# ForwardGateDecision serialization) sit far below every bound; only a smuggled payload hits them.
_MAX_STRING_LEN = 512
_MAX_DICT_ENTRIES = 32
_MAX_CHECKS = 64
# json.loads accepts integer literals thousands of digits long (interpreter cap ~4300), so an
# unbounded int is a bulk channel just like an unbounded string. Every legitimate decision int is
# a count/index/flag — 10**18 is orders of magnitude above all of them.
_MAX_INT_MAGNITUDE = 10**18

# The scalar fields kept from each checks[] entry (drops free-text `detail` and anything else).
# threshold/value must be finite-numeric-or-null and passed must be bool — a vector cannot be
# re-encoded as check fields.
_CHECK_FIELDS = ("name", "op", "threshold", "value", "passed")


def _bounded_scalar(value: Any) -> bool:
    if isinstance(value, str):
        return len(value) <= _MAX_STRING_LEN
    if isinstance(value, bool):
        return True
    if isinstance(value, int):
        return abs(value) <= _MAX_INT_MAGNITUDE
    if isinstance(value, float):
        return math.isfinite(value)
    return value is None


def _finite_number_or_none(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return value is None
    if isinstance(value, int):
        return abs(value) <= _MAX_INT_MAGNITUDE
    return isinstance(value, float) and math.isfinite(value)


def _check_field_ok(field: str, value: Any) -> bool:
    if field in ("threshold", "value"):
        return _finite_number_or_none(value)
    if field == "passed":
        return isinstance(value, bool)
    return isinstance(value, str) and len(value) <= _MAX_STRING_LEN  # name / op


def _shape(key: str, value: Any) -> tuple[bool, Any]:
    """Layer-2 shape guard: (survives, projected_value). Bounded scalars and small
    numeric-or-null dicts pass through; the ``checks`` list is reduced to its typed scalar
    fields; everything else — including oversized or non-numeric-dict payloads that could
    smuggle a vector under an allowlisted key — is dropped."""
    if isinstance(value, _SCALAR_TYPES):
        return (True, value) if _bounded_scalar(value) else (False, None)
    if (
        isinstance(value, dict)
        and len(value) <= _MAX_DICT_ENTRIES
        and all(
            isinstance(k, str) and len(k) <= _MAX_STRING_LEN and _finite_number_or_none(v)
            for k, v in value.items()
        )
    ):
        return True, value
    if (
        key == "checks"
        and isinstance(value, list)
        and len(value) <= _MAX_CHECKS
        and all(isinstance(c, dict) for c in value)
    ):
        return True, [
            {f: c[f] for f in _CHECK_FIELDS if f in c and _check_field_ok(f, c[f])}
            for c in value
        ]
    return False, None


def _project_decision(raw: str, allowlist: frozenset[str]) -> dict[str, Any]:
    """Parse + project one row's decision_json. A corrupt blob yields ``decision: None`` +
    ``decision_error`` (sibling rows unaffected); a parsed blob yields the two-layer-allowlisted
    ``decision`` + the sorted ``decision_dropped_keys`` it shed."""
    try:
        obj = json.loads(raw)
    except (TypeError, ValueError) as exc:
        return {"decision": None, "decision_error": str(exc)}
    if not isinstance(obj, dict):
        return {"decision": None, "decision_error": "decision_json is not a JSON object"}
    decision: dict[str, Any] = {}
    dropped: list[str] = []
    for key, value in obj.items():
        if key in allowlist:
            survives, projected = _shape(key, value)
            if survives:
                decision[key] = projected
                continue
        dropped.append(key)
    return {"decision": decision, "decision_dropped_keys": sorted(dropped)}


def _history(
    conn: sqlite3.Connection, table: str, columns: tuple[str, ...],
    allowlist: frozenset[str], strategy_id: int, limit: int,
) -> list[dict[str, Any]]:
    rows = conn.execute(
        f"SELECT {', '.join(columns)}, decision_json FROM {table}"
        " WHERE strategy_id = ? ORDER BY id DESC LIMIT ?",
        (strategy_id, limit),
    ).fetchall()
    return [
        {**{c: row[c] for c in columns}, **_project_decision(row["decision_json"], allowlist)}
        for row in rows
    ]


def gate_evaluation_history(
    conn: sqlite3.Connection, strategy_id: int, *, limit: int = 20,
) -> list[dict[str, Any]]:
    """Newest-first ``gate_evaluations`` rows for one strategy, decision_json projected."""
    return _history(conn, "gate_evaluations", _GATE_COLUMNS,
                    GATE_DECISION_ALLOWLIST, strategy_id, limit)


def forward_gate_evaluation_history(
    conn: sqlite3.Connection, strategy_id: int, *, limit: int = 20,
) -> list[dict[str, Any]]:
    """Newest-first ``forward_gate_evaluations`` rows for one strategy, decision_json projected."""
    return _history(conn, "forward_gate_evaluations", _FORWARD_COLUMNS,
                    FORWARD_DECISION_ALLOWLIST, strategy_id, limit)


def strategy_gate_history(
    conn: sqlite3.Connection, strategy_id: int, *, limit: int = 20,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Both ledgers read inside ONE explicit BEGIN DEFERRED transaction — a consistent snapshot
    (DEFERRED on purpose: this is a reader and must never take the write lock)."""
    conn.execute("BEGIN DEFERRED")
    try:
        gates = gate_evaluation_history(conn, strategy_id, limit=limit)
        forwards = forward_gate_evaluation_history(conn, strategy_id, limit=limit)
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    return gates, forwards
