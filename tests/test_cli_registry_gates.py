"""`registry gates` — read-only gate-evaluation history with allowlist-projected decision_json.

Hermetic CLI tests (CliRunner + ALGUA_DB_PATH monkeypatch, rows inserted via raw conn.execute —
the tests/test_fleet_health.py precedent). The load-bearing pins: a planted vector in
decision_json can NEVER transit the view, and the golden drift test forces a conscious review of
the projection whenever a decision dataclass grows a field.
"""

import dataclasses
import json
from contextlib import closing

from typer.testing import CliRunner

from algua.cli.main import app
from algua.config.settings import get_settings
from algua.registry.db import connect, migrate
from algua.registry.gate_history import (
    FORWARD_DECISION_ALLOWLIST,
    FORWARD_DECISION_EXCLUDED,
    GATE_DECISION_ALLOWLIST,
    GATE_DECISION_EXCLUDED,
)
from algua.registry.store import SqliteStrategyRepository
from algua.research.forward_gates import ForwardGateDecision
from algua.research.gates import GateDecision

runner = CliRunner()


def _conn():
    conn = connect(get_settings().db_path)
    migrate(conn)
    return conn


def _register(conn, name="strat"):
    return SqliteStrategyRepository(conn).add(name)


def _insert_gate_row(conn, strategy_id, *, decision_json="{}",
                     created_at="2026-01-01T00:00:00+00:00"):
    """A legacy-shaped gate_evaluations row: base NOT NULL columns only, so every migration-added
    column (fdr_*, family_*, attempt_token) stays NULL."""
    cols = {
        "strategy_id": strategy_id, "passed": 1, "n_funnel": 1, "own_lifetime_combos": 12,
        "windowed_total_combos": 30, "funnel_window_days": 90, "breadth_provenance": "measured",
        "pit_ok": 1, "holdout_n_bars": 100, "min_holdout_observations": 63, "code_hash": "c",
        "config_hash": "cfg", "dependency_hash": "dep", "data_source": "snapshot",
        "snapshot_id": "snap-1", "period_start": "2020-01-01", "period_end": "2024-01-01",
        "holdout_frac": 0.2, "actor": "agent", "decision_json": decision_json,
        "created_at": created_at,
    }
    cur = conn.execute(
        f"INSERT INTO gate_evaluations ({', '.join(cols)})"
        f" VALUES ({', '.join('?' * len(cols))})",
        tuple(cols.values()),
    )
    conn.commit()
    return cur.lastrowid


def _insert_forward_row(conn, strategy_id, *, decision_json="{}",
                        created_at="2026-01-01T00:00:00+00:00"):
    cols = {
        "strategy_id": strategy_id, "passed": 1, "n_forward_observations": 70,
        "min_forward_observations": 63, "session_coverage": 0.95, "realized_sharpe": 0.8,
        "holdout_sharpe": 1.0, "degradation_factor": 0.5, "sharpe_floor": 0.3,
        "realized_vol": 0.1, "min_forward_vol": 0.01, "realized_max_drawdown": -0.05,
        "max_forward_drawdown": -0.25, "first_tick_id": 1, "last_tick_id": 70,
        "first_tick_ts": "2026-01-01T00:00:00+00:00", "last_tick_ts": "2026-04-01T00:00:00+00:00",
        "max_staleness_sessions": 5, "n_reconcile_failures": 0, "n_concurrent_forward": 1,
        "account_id": "acct", "code_hash": "c", "config_hash": "cfg", "dependency_hash": "dep",
        "actor": "agent", "decision_json": decision_json, "created_at": created_at,
    }
    cur = conn.execute(
        f"INSERT INTO forward_gate_evaluations ({', '.join(cols)})"
        f" VALUES ({', '.join('?' * len(cols))})",
        tuple(cols.values()),
    )
    conn.commit()
    return cur.lastrowid


def _invoke(name="strat", *extra):
    result = runner.invoke(app, ["registry", "gates", name, *extra])
    return result, json.loads(result.stdout)


def test_newest_first_ordering_and_limit(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    with closing(_conn()) as conn:
        rec = _register(conn)
        ids = [_insert_gate_row(conn, rec.id) for _ in range(3)]
        fids = [_insert_forward_row(conn, rec.id) for _ in range(3)]
    result, payload = _invoke("strat", "--limit", "2")
    assert result.exit_code == 0, result.stdout
    assert payload["ok"] is True
    assert payload["strategy"] == "strat"
    assert [r["id"] for r in payload["gate_evaluations"]] == [ids[2], ids[1]]
    assert [r["id"] for r in payload["forward_gate_evaluations"]] == [fids[2], fids[1]]


def test_realistic_decision_round_trips_allowlisted_fields(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    decision = {
        "passed": True,
        "breadth_provenance": "measured",
        "effective_min_holdout_sharpe": 0.91,
        "dsr_confidence": 0.97,
        "fdr_p_value": 0.003,
        "fdr_rejected": True,
        "market_beta": 0.4,
        "appraisal_ratio": 1.1,
        "checks": [
            {"name": "holdout_sharpe", "op": ">=", "threshold": 0.91, "value": 1.2,
             "passed": True, "detail": None},
            {"name": "holdout_total_return", "op": ">", "threshold": 0.0, "value": 0.15,
             "passed": True, "detail": "free text that must be stripped"},
        ],
    }
    with closing(_conn()) as conn:
        rec = _register(conn)
        _insert_gate_row(conn, rec.id, decision_json=json.dumps(decision))
        _insert_forward_row(conn, rec.id, decision_json=json.dumps(
            {"passed": False, "checks": [{"name": "coverage", "op": ">=", "threshold": 0.9,
                                          "value": 0.8, "passed": False, "detail": "low"}]}))
    result, payload = _invoke()
    assert result.exit_code == 0, result.stdout
    d = payload["gate_evaluations"][0]["decision"]
    for key in ("passed", "breadth_provenance", "effective_min_holdout_sharpe", "dsr_confidence",
                "fdr_p_value", "fdr_rejected", "market_beta", "appraisal_ratio"):
        assert d[key] == decision[key]
    # each check keeps EXACTLY the scalar quintet — `detail` never transits
    for check in d["checks"]:
        assert set(check) == {"name", "op", "threshold", "value", "passed"}
    assert payload["gate_evaluations"][0]["decision_dropped_keys"] == []
    fd = payload["forward_gate_evaluations"][0]["decision"]
    assert fd["passed"] is False
    assert set(fd["checks"][0]) == {"name", "op", "threshold", "value", "passed"}


def test_planted_vector_key_never_appears(monkeypatch, tmp_path):
    """A returns vector planted in decision_json must be dropped by the shape guard even if a
    writer smuggled it in — the view is structurally aggregate-only."""
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    with closing(_conn()) as conn:
        rec = _register(conn)
        _insert_gate_row(conn, rec.id, decision_json=json.dumps(
            {"passed": True, "oos_returns": [0.1, 0.2]}))
    result, payload = _invoke()
    assert result.exit_code == 0, result.stdout
    row = payload["gate_evaluations"][0]
    assert "oos_returns" not in row["decision"]
    assert "oos_returns" in row["decision_dropped_keys"]
    assert "oos_returns" not in json.dumps(row["decision"])
    assert row["decision"]["passed"] is True


def test_non_allowlisted_scalar_key_dropped_and_listed(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    with closing(_conn()) as conn:
        rec = _register(conn)
        _insert_gate_row(conn, rec.id, decision_json=json.dumps(
            {"passed": True, "some_future_field": 1.5}))
    result, payload = _invoke()
    assert result.exit_code == 0, result.stdout
    row = payload["gate_evaluations"][0]
    assert "some_future_field" not in row["decision"]
    assert row["decision_dropped_keys"] == ["some_future_field"]


def test_corrupt_decision_json_isolated_per_row(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    with closing(_conn()) as conn:
        rec = _register(conn)
        good_id = _insert_gate_row(conn, rec.id, decision_json='{"passed": true}')
        bad_id = _insert_gate_row(conn, rec.id, decision_json="{not json")
    result, payload = _invoke()
    assert result.exit_code == 0, result.stdout
    by_id = {r["id"]: r for r in payload["gate_evaluations"]}
    assert by_id[bad_id]["decision"] is None
    assert by_id[bad_id]["decision_error"]
    assert by_id[good_id]["decision"] == {"passed": True}


def test_unknown_strategy_not_found(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    result = runner.invoke(app, ["registry", "gates", "nope"])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["code"] == "not_found"


def test_registered_strategy_zero_rows(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    with closing(_conn()) as conn:
        _register(conn)
    result, payload = _invoke()
    assert result.exit_code == 0, result.stdout
    assert payload["ok"] is True
    assert payload["gate_evaluations"] == []
    assert payload["forward_gate_evaluations"] == []


def test_returns_blob_never_in_payload(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    with closing(_conn()) as conn:
        rec = _register(conn)
        _insert_gate_row(conn, rec.id, decision_json='{"passed": true}')
        _insert_forward_row(conn, rec.id, decision_json='{"passed": true}')
    result, _ = _invoke()
    assert result.exit_code == 0, result.stdout
    assert "returns_blob" not in result.stdout


def test_golden_drift_allowlists_cover_dataclass_fields():
    """GOLDEN DRIFT TEST: any field added to a decision dataclass must be consciously reviewed
    into the projection (allowlist) or out of it (excluded) before this passes again."""
    gate_fields = {f.name for f in dataclasses.fields(GateDecision)}
    assert gate_fields == GATE_DECISION_ALLOWLIST | GATE_DECISION_EXCLUDED
    assert not GATE_DECISION_ALLOWLIST & GATE_DECISION_EXCLUDED
    forward_fields = {f.name for f in dataclasses.fields(ForwardGateDecision)}
    assert forward_fields == FORWARD_DECISION_ALLOWLIST | FORWARD_DECISION_EXCLUDED
    assert not FORWARD_DECISION_ALLOWLIST & FORWARD_DECISION_EXCLUDED


def test_vector_smuggled_as_dict_under_allowlisted_key_dropped(monkeypatch, tmp_path):
    """A returns vector re-encoded as a {date: return} dict under an ALLOWLISTED key must be
    dropped by the dict-entry cap — dict-of-scalars is bounded, not a bulk channel."""
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    smuggled = {f"2020-01-{i:02d}": 0.01 * i for i in range(1, 41)}  # 40 entries > cap
    with closing(_conn()) as conn:
        rec = _register(conn)
        _insert_gate_row(conn, rec.id, decision_json=json.dumps(
            {"passed": True, "dsr_confidence": smuggled}))
    result, payload = _invoke()
    assert result.exit_code == 0, result.stdout
    row = payload["gate_evaluations"][0]
    assert "dsr_confidence" not in row["decision"]
    assert "dsr_confidence" in row["decision_dropped_keys"]


def test_vector_smuggled_as_string_under_allowlisted_key_dropped(monkeypatch, tmp_path):
    """A serialized vector smuggled as one long string under an allowlisted key must be dropped
    by the string-length cap."""
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    smuggled = ",".join(f"{0.001 * i:.6f}" for i in range(500))  # far beyond the cap
    with closing(_conn()) as conn:
        rec = _register(conn)
        _insert_gate_row(conn, rec.id, decision_json=json.dumps(
            {"passed": True, "dsr_skip_reason": smuggled}))
    result, payload = _invoke()
    assert result.exit_code == 0, result.stdout
    row = payload["gate_evaluations"][0]
    assert "dsr_skip_reason" not in row["decision"]
    assert "dsr_skip_reason" in row["decision_dropped_keys"]


def test_vector_smuggled_as_checks_rows_dropped(monkeypatch, tmp_path):
    """A vector re-encoded as hundreds of fabricated checks rows must be dropped whole by the
    checks-count cap; and non-numeric check threshold/value fields are stripped."""
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    fabricated = [{"name": f"r{i}", "op": ">=", "threshold": 0.0, "value": 0.001 * i,
                   "passed": True} for i in range(200)]  # 200 > cap
    with closing(_conn()) as conn:
        rec = _register(conn)
        _insert_gate_row(conn, rec.id, decision_json=json.dumps(
            {"passed": True, "checks": fabricated}))
        _insert_gate_row(conn, rec.id, decision_json=json.dumps(
            {"passed": True,
             "checks": [{"name": "sharpe", "op": ">=", "threshold": "0.1,0.2,0.3",
                         "value": 1.2, "passed": True}]}))
    result, payload = _invoke()
    assert result.exit_code == 0, result.stdout
    capped_row = next(r for r in payload["gate_evaluations"]
                      if "checks" in r["decision_dropped_keys"])
    assert "checks" not in capped_row["decision"]
    typed_row = next(r for r in payload["gate_evaluations"]
                     if "checks" in r["decision"])
    (check,) = typed_row["decision"]["checks"]
    assert "threshold" not in check  # string threshold stripped, not passed through
    assert check["value"] == 1.2
    assert check["passed"] is True


def test_vector_smuggled_as_giant_int_dropped(monkeypatch, tmp_path):
    """json.loads accepts integer literals thousands of digits long — a vector packed into one
    giant int under an allowlisted key must be dropped by the magnitude bound."""
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    giant = int("42" * 1000)  # a 2000-digit payload channel
    with closing(_conn()) as conn:
        rec = _register(conn)
        _insert_gate_row(conn, rec.id, decision_json=json.dumps(
            {"passed": True, "dsr_n_trials": giant,
             "checks": [{"name": "sharpe", "op": ">=", "threshold": giant, "value": 1.2,
                         "passed": True}]}))
    result, payload = _invoke()
    assert result.exit_code == 0, result.stdout
    row = payload["gate_evaluations"][0]
    assert "dsr_n_trials" not in row["decision"]
    assert "dsr_n_trials" in row["decision_dropped_keys"]
    (check,) = row["decision"]["checks"]
    assert "threshold" not in check  # giant-int check field stripped
    assert check["value"] == 1.2


def test_legacy_row_null_fdr_columns_emit_as_nulls(monkeypatch, tmp_path):
    """A legacy-shaped row (inserted without the migration-added fdr_* columns) must emit them as
    JSON nulls, not crash."""
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    with closing(_conn()) as conn:
        rec = _register(conn)
        _insert_gate_row(conn, rec.id)
    result, payload = _invoke()
    assert result.exit_code == 0, result.stdout
    row = payload["gate_evaluations"][0]
    for col in ("fdr_binding", "fdr_p_value", "fdr_alpha_level", "fdr_rejected",
                "fdr_test_index", "fdr_cohort"):
        assert col in row
        assert row[col] is None
