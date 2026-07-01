"""Tests for the advisory negative-result experience log (#332).

Covers the record/list roundtrip, secret redaction across BOTH durable surfaces (the ledger row
and the rendered vault note), payload bounding, the top-level-only transaction guard, and the
pure gate-fail record builder. The log is advisory: these never assert a promotion outcome.
"""
from __future__ import annotations

import json

import pytest

from algua.knowledge.experience import render_experience_note
from algua.registry.db import connect, migrate
from algua.registry.negative_results import (
    build_gate_fail_record,
    list_negative_results,
    record_negative_result,
    sanitize_record,
)

_SECRET = "sk-abcdefghij1234567890"  # provider-style key shape the redactor must mask


def _conn(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    return conn


def test_record_and_list_roundtrip(tmp_path):
    conn = _conn(tmp_path)
    rid = record_negative_result(
        conn, kind="discard", verdict="DISCARD", actor="agent",
        reason="tried X, no edge", source="manual", strategy_name="s",
        hypothesis="X predicts Y", params={"a": 1}, tags="mean-reversion")
    assert rid == 1
    rows = list_negative_results(conn, strategy="s")
    assert len(rows) == 1
    r = rows[0]
    assert r["kind"] == "discard" and r["verdict"] == "DISCARD" and r["params"] == {"a": 1}


def test_secret_redacted_in_ledger_reason_and_params(tmp_path):
    conn = _conn(tmp_path)
    record_negative_result(
        conn, kind="discard", verdict="DISCARD", actor="agent",
        reason=f"leaked {_SECRET} oops", source="manual",
        params={"nested": {"api_key": _SECRET}})
    r = list_negative_results(conn)[0]
    assert _SECRET not in r["reason"]
    assert "[REDACTED]" in r["reason"]
    assert _SECRET not in json.dumps(r["params"])  # scrubbed inside nested params too


def test_secret_redacted_in_param_keys(tmp_path):
    # A secret pasted as a dict KEY (not value) must also be masked in ledger + note.
    conn = _conn(tmp_path)
    record_negative_result(
        conn, kind="discard", verdict="DISCARD", actor="agent", reason="k",
        source="manual", params={f"api_key={_SECRET}": "v"})
    r = list_negative_results(conn)[0]
    assert _SECRET not in json.dumps(r["params"])


def test_params_with_mixed_and_non_string_keys_does_not_crash(tmp_path):
    # A logging primitive must not raise on plausible input: mixed str/int keys would break
    # json.dumps(sort_keys=True); keys are coerced to str (JSON-object semantics).
    conn = _conn(tmp_path)
    rid = record_negative_result(
        conn, kind="discard", verdict="DISCARD", actor="agent", reason="k",
        source="manual", params={1: "x", "a": "y", (2, 3): "z"})
    r = list_negative_results(conn)[0]
    assert r["id"] == rid
    assert set(r["params"]) == {"1", "a", "(2, 3)"}


def test_params_json_is_bounded(tmp_path):
    conn = _conn(tmp_path)
    record_negative_result(
        conn, kind="discard", verdict="DISCARD", actor="agent", reason="big",
        source="manual", params={"blob": "x" * 1_000_000})
    r = list_negative_results(conn)[0]
    # a single pathological string leaf is capped, so the stored payload cannot bloat the row
    assert len(json.dumps(r["params"])) < 20_000


def test_transaction_guard_refuses_open_transaction(tmp_path):
    conn = _conn(tmp_path)
    conn.execute("BEGIN")
    conn.execute(
        "INSERT INTO strategies(name, stage, created_at, updated_at)"
        " VALUES ('s','idea','2026-01-01','2026-01-01')")
    with pytest.raises(RuntimeError, match="top level"):
        record_negative_result(
            conn, kind="discard", verdict="DISCARD", actor="agent",
            reason="x", source="manual")
    conn.rollback()


@pytest.mark.parametrize("bad", [
    {"kind": "gate_fail_typo"}, {"source": "auto:evil"}, {"actor": "root"}, {"verdict": "  "},
])
def test_boundary_validation_rejects_bad_fields(tmp_path, bad):
    conn = _conn(tmp_path)
    kwargs = dict(kind="discard", verdict="DISCARD", actor="agent", reason="x", source="manual")
    kwargs.update(bad)
    with pytest.raises(ValueError):
        record_negative_result(conn, **kwargs)


def test_sanitize_record_scrubs_note_surface():
    # The vault note is rendered from the SANITIZED record (call-site contract), so a secret in
    # reason/hypothesis/params never reaches kb markdown even though the ledger is redacted too.
    raw = {"strategy_name": "s", "kind": "discard", "verdict": "DISCARD", "actor": "agent",
           "source": "manual", "created_at": "2026-07-01T00:00:00+00:00",
           "reason": f"key {_SECRET}", "hypothesis": f"token {_SECRET}",
           "params": {"k": _SECRET}, "tags": None, "gate_evaluation_id": None}
    note = render_experience_note(sanitize_record(raw), record_id=7)
    assert _SECRET not in note
    assert "[REDACTED]" in note


def test_build_gate_fail_record_shape():
    decision = {"checks": [{"name": "holdout_sharpe", "passed": False},
                           {"name": "dsr", "passed": True}], "n_combos": 3}
    rec = build_gate_fail_record(
        "s", decision, actor="agent", period_start="2020-01-01", period_end="2021-01-01",
        holdout={"sharpe": 0.1}, stability=None)
    assert rec["kind"] == "gate_fail" and rec["verdict"] == "FAIL"
    assert rec["source"] == "auto:research_promote"
    assert "holdout_sharpe" in rec["reason"]  # names the failed check
    assert rec["params"]["checks"] == decision["checks"]
