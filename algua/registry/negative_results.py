"""Advisory negative-result / experience log (#332).

Captures failed or rejected hypotheses — a strategy that FAILED a promotion gate, a discarded
idea, or a research dead-end — into a durable, queryable ledger so the knowledge is not lost with
the branch. The next ideation pass can query it (``research log list``) instead of re-deriving
already-refuted axes and re-burning breadth/holdout.

This module is DELIBERATELY separate from the CODEOWNERS-protected ``store.py``: it owns its own
INSERT/SELECT against the (unprotected) ``negative_results`` table. It is ADVISORY ONLY — nothing
here gates a promotion or touches the live/paper path.

Design notes for the paranoid reader:
- ``record_negative_result`` validates ``kind``/``source``/``actor``/``verdict`` at the boundary
  (the table also CHECK-constrains ``kind``/``source``), redacts operator-supplied free text, and
  serializes ``params`` with non-finite floats nulled and ``allow_nan=False`` so the column is
  always valid JSON.
- The write is a single-row INSERT in its own transaction; callers on the reject path invoke it
  best-effort so a logging failure can never break the thing being logged.
"""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import UTC, datetime
from typing import Any

VALID_KINDS = frozenset({"gate_fail", "discard", "dead_end"})
VALID_SOURCES = frozenset({"auto:research_promote", "manual"})
VALID_ACTORS = frozenset({"agent", "human", "system"})

_MAX_TEXT = 8000  # cap any single free-text field so a pasted blob can't bloat the ledger row

# Secret-shaped substrings to mask out of operator-supplied free text before it is persisted.
# Best-effort defense-in-depth: the ledger should never become an exfiltration surface for a
# key/token accidentally pasted into a --reason/--hypothesis. Ordered longest/most-specific first.
_REDACTIONS: tuple[re.Pattern[str], ...] = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.DOTALL),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),                      # AWS access key id
    re.compile(r"\b(?:Bearer|bearer)\s+[A-Za-z0-9._\-]{16,}"),  # bearer token
    re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9]{16,}\b"),         # provider-style secret keys
    # key/secret/token/password = <value>  (value up to whitespace)
    re.compile(r"(?i)\b(?:api[_-]?key|secret|token|password|passwd|pwd)\b\s*[:=]\s*\S+"),
    re.compile(r"\b[A-Fa-f0-9]{32,}\b"),                      # long hex run (hashes/keys)
    re.compile(r"\b[A-Za-z0-9+/]{40,}={0,2}\b"),              # long base64 run
)

_REDACTED = "[REDACTED]"


def _redact(text: str | None) -> str | None:
    """Mask secret-shaped substrings and cap length. None passes through."""
    if text is None:
        return None
    out = text
    for pat in _REDACTIONS:
        out = pat.sub(_REDACTED, out)
    if len(out) > _MAX_TEXT:
        out = out[:_MAX_TEXT] + "…[truncated]"
    return out


def _sanitize(obj: Any) -> Any:
    """Recursively replace non-finite floats (NaN/Inf) with None so the payload is valid JSON.

    Mirrors how ``GateDecision.to_dict`` nulls non-finite metrics; applied here as a backstop for
    the raw holdout/stability dicts we fold into ``params``.
    """
    if isinstance(obj, float):
        return obj if obj == obj and obj not in (float("inf"), float("-inf")) else None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def _dump_params(params: dict[str, Any] | None) -> str | None:
    if params is None:
        return None
    return json.dumps(_sanitize(params), allow_nan=False, sort_keys=True, default=str)


def record_negative_result(
    conn: sqlite3.Connection,
    *,
    kind: str,
    verdict: str,
    actor: str,
    reason: str,
    source: str,
    strategy_name: str | None = None,
    hypothesis: str | None = None,
    params: dict[str, Any] | None = None,
    tags: str | None = None,
    gate_evaluation_id: int | None = None,
    created_at: str | None = None,
) -> int:
    """Insert one advisory negative-result row; return its id. Own transaction.

    Validates the constrained fields at the boundary and redacts the free-text fields. Raises
    ValueError on a bad ``kind``/``source``/``actor``/``verdict`` (the auto-capture caller wraps
    this best-effort so a validation slip never breaks a promote).
    """
    if kind not in VALID_KINDS:
        raise ValueError(f"kind must be one of {sorted(VALID_KINDS)}, got {kind!r}")
    if source not in VALID_SOURCES:
        raise ValueError(f"source must be one of {sorted(VALID_SOURCES)}, got {source!r}")
    if actor not in VALID_ACTORS:
        raise ValueError(f"actor must be one of {sorted(VALID_ACTORS)}, got {actor!r}")
    verdict = verdict.strip()
    if not verdict:
        raise ValueError("verdict must be a non-empty label")
    if len(verdict) > 64:
        raise ValueError("verdict must be <= 64 chars")
    if not reason or not reason.strip():
        raise ValueError("reason must be non-empty")

    row = (
        created_at or datetime.now(UTC).isoformat(),
        strategy_name,
        gate_evaluation_id,
        kind,
        verdict,
        actor,
        _redact(reason),
        _redact(hypothesis),
        _dump_params(params),
        _redact(tags),
        source,
    )
    with conn:
        cur = conn.execute(
            "INSERT INTO negative_results("
            "created_at, strategy_name, gate_evaluation_id, kind, verdict, actor, reason, "
            "hypothesis, params_json, tags, source) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            row,
        )
    rowid = cur.lastrowid
    assert rowid is not None  # AUTOINCREMENT INSERT always sets lastrowid
    return rowid


def list_negative_results(
    conn: sqlite3.Connection,
    *,
    strategy: str | None = None,
    kind: str | None = None,
    verdict: str | None = None,
    since: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Query the ledger newest-first with optional filters. ``params_json`` is parsed back to a
    dict under the ``params`` key. ``since`` is a normalized UTC ISO-8601 lower bound on
    ``created_at`` (caller normalizes)."""
    if limit < 1:
        raise ValueError("limit must be >= 1")
    clauses: list[str] = []
    args: list[Any] = []
    if strategy is not None:
        clauses.append("strategy_name = ?")
        args.append(strategy)
    if kind is not None:
        if kind not in VALID_KINDS:
            raise ValueError(f"kind must be one of {sorted(VALID_KINDS)}, got {kind!r}")
        clauses.append("kind = ?")
        args.append(kind)
    if verdict is not None:
        clauses.append("verdict = ?")
        args.append(verdict.strip())
    if since is not None:
        clauses.append("created_at >= ?")
        args.append(since)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    args.append(limit)
    rows = conn.execute(
        "SELECT id, created_at, strategy_name, gate_evaluation_id, kind, verdict, actor, reason, "
        "hypothesis, params_json, tags, source FROM negative_results"
        f"{where} ORDER BY id DESC LIMIT ?",
        args,
    ).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        d = dict(r)
        raw = d.pop("params_json", None)
        d["params"] = json.loads(raw) if raw else None
        out.append(d)
    return out


def build_gate_fail_record(
    name: str,
    decision: dict[str, Any],
    *,
    actor: str,
    period_start: str,
    period_end: str,
    holdout: dict[str, Any] | None,
    stability: dict[str, Any] | None,
) -> dict[str, Any]:
    """Turn a FAILED gate ``decision`` (``GateDecision.to_dict()``) into record fields.

    Pure — no I/O. The ``reason`` names the failed checks (the refutation, human-readable); the
    ``params`` carry the full checks list plus period/breadth/holdout so the evidence survives.
    """
    checks = decision.get("checks") or []
    failed = [c.get("name", "?") for c in checks if not c.get("passed", True)]
    reason = (
        "gate FAILED: " + ", ".join(failed) if failed else "gate FAILED (no failing check named)"
    )
    params = {
        "period": {"start": period_start, "end": period_end},
        "checks": checks,
        "n_funnel": decision.get("n_combos"),
        "breadth_provenance": decision.get("breadth_provenance"),
        "effective_min_holdout_sharpe": decision.get("effective_min_holdout_sharpe"),
        "dsr_confidence": decision.get("dsr_confidence"),
        "holdout": holdout,
        "stability": stability,
    }
    return {
        "strategy_name": name,
        "kind": "gate_fail",
        "verdict": "FAIL",
        "actor": actor,
        "reason": reason,
        "params": params,
        "source": "auto:research_promote",
    }
