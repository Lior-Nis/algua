# algua/registry/idea_scorecard.py
"""The feedback edge (spec §7): attempt outcomes + downstream stage, grouped four ways."""
from __future__ import annotations

import sqlite3
from collections import defaultdict
from datetime import UTC, datetime, timedelta

from algua.contracts.idea import AttemptOutcome

_MIN_N = 5
_PAST_INTEGRITY = frozenset(a.value for a in AttemptOutcome) - {
    AttemptOutcome.INTEGRITY_FAIL.value, AttemptOutcome.ABANDONED.value,
    AttemptOutcome.RUN_ERROR.value}
_PAST_WALKFORWARD = _PAST_INTEGRITY - {AttemptOutcome.HOLDOUT_NEGATIVE.value,
                                       AttemptOutcome.WALKFORWARD_REFUTED.value,
                                       AttemptOutcome.SWEEP_UNSTABLE.value}


def _bucket() -> dict:
    return {"n": 0, "outcomes": defaultdict(int), "stages": defaultdict(int), "survivals": 0}


def _derived_stage(stage: str | None) -> str | None:
    if stage is None:
        return None
    if stage in ("forward_tested", "live"):
        return "forward_survivor"
    return stage  # paper | retired | dormant | candidate | backtested


def _finalize(b: dict) -> dict:
    n = b["n"]
    past_i = sum(v for k, v in b["outcomes"].items() if k in _PAST_INTEGRITY)
    past_w = sum(v for k, v in b["outcomes"].items() if k in _PAST_WALKFORWARD)

    def rate(x: int) -> float | None:
        return (x / n) if n >= _MIN_N else None

    return {"n": n, "outcomes": dict(b["outcomes"]), "stages": dict(b["stages"]),
            "survivals": b["survivals"], "integrity_yield": rate(past_i),
            "walkforward_yield": rate(past_w), "survival_yield": rate(b["survivals"])}


def scorecard(conn: sqlite3.Connection, *, days: int) -> dict:
    since = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    rows = conn.execute(
        "SELECT a.id AS attempt_id, a.idea_id, a.outcome, i.category, s.stage AS stage,"
        " ins.inspiration_id, ins.venue, ins.obscurity FROM idea_attempts a"
        " JOIN ideas i ON i.id = a.idea_id"
        " LEFT JOIN strategies s ON s.id = i.authored_strategy_id"
        " LEFT JOIN idea_inspirations ins ON ins.idea_id = a.idea_id"
        " WHERE a.claimed_at >= ? AND a.outcome IS NOT NULL", (since,)).fetchall()
    groups: dict[str, dict[str, dict]] = {k: defaultdict(_bucket) for k in
                                          ("by_venue", "by_category", "by_obscurity",
                                           "by_inspiration")}
    seen_attempt_per_key: set[tuple] = set()
    for r in rows:
        stage = _derived_stage(r["stage"])
        # A single attempt survives if it was directly promoted OR its idea later reached
        # forward_tested/live — a UNION counted once per attempt, never both, so an idea that
        # was promoted and LATER also reached forward_tested cannot count twice.
        survives = r["outcome"] == AttemptOutcome.PROMOTED_CANDIDATE.value \
            or stage == "forward_survivor"
        keys = [("by_category", r["category"] or "legacy")]
        if r["venue"] is not None:
            keys += [("by_venue", r["venue"]), ("by_obscurity", r["obscurity"]),
                     ("by_inspiration", r["inspiration_id"])]
        for group, key in keys:
            # one attempt (identified by its OWN idea_attempts.id, not idea_id+outcome — two
            # attempts on the same idea can share an outcome) counts once per (group,key) even
            # with several inspiration rows fanning it out.
            marker = (group, key, r["attempt_id"])
            if marker in seen_attempt_per_key:
                continue
            seen_attempt_per_key.add(marker)
            b = groups[group][key]
            b["n"] += 1
            b["outcomes"][r["outcome"]] += 1
            if stage:
                b["stages"][stage] += 1
            if survives:
                b["survivals"] += 1
    return {"days": days, "min_n_for_rates": _MIN_N,
            **{g: {k: _finalize(b) for k, b in d.items()} for g, d in groups.items()}}
