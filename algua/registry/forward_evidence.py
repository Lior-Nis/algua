"""Forward-test evidence assembly (#124): DB + broker -> ``ForwardEvidence``.

The I/O half of the protected ``paper -> forward_tested`` gate, split out of
``registry/forward_promotion.py`` (mirroring ``registry/promotion.py`` / ``family_assignment.py``
for the shortlist gate). This module reads the paper-lane tick rows, partitions them into
admissible observations vs per-filter exclusions, builds the daily return series with the SHARED
backtest metrics (same Sharpe/vol/drawdown definitions as the holdout the gate compares against),
runs the integrity / single-tenant / broker-activities checks, and locates the qualified backtest
gate row. The pure judgement lives in ``algua.research.forward_gates`` — this module never
decides, only assembles. ``registry/live_certificate.py`` re-verifies an EARNED certificate using
the same admissibility parsing (``_parse_dt``) and activity classification
(``_classify_activities``) defined here, imported rather than duplicated.

Every clause here is a wall against an autonomous agent fabricating forward evidence (back-dated
ticks, identity drift, sibling contamination, manual fills, external capital). Each helper FAILS
CLOSED — ambiguity is never resolved in the strategy's favor.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any

import pandas as pd

from algua.backtest.metrics import metrics_from_returns
from algua.registry.repository import ArtifactIdentity
from algua.research.forward_gates import FORWARD_RELOOK_HORIZON_SESSIONS, ForwardEvidence
from algua.risk.global_halt import is_engaged
from algua.risk.kill_switch import is_tripped

if TYPE_CHECKING:
    from algua.registry.forward_promotion import SessionCalendar

# Alpaca account-activity types that move EXTERNAL capital in/out of the account (deposits,
# withdrawals, transfers, journals, ACATS). Any of these inside the evidence window invalidates
# the equity curve as strategy performance. DIV/INT/FEE are position-attributable and pass.
EXTERNAL_CAPITAL_TYPES = frozenset({"CSD", "CSW", "TRANS", "JNLC", "JNLS", "ACATC", "ACATS"})

# Decision-freshness bound (sessions from decision session to tick session). Honest wall-clock
# ticks decide on the latest closed bar, trailing by at most a data-staleness session or two;
# anything beyond is the historical-replay attack (`trade-tick --end <past>`).
_MAX_DECISION_LAG_SESSIONS = 2

# Per-filter exclusion keys, IN EVALUATION ORDER (first matching filter wins the count).
_EXCLUSION_FILTERS = ("local_clock", "identity_drift", "legacy_null", "bad_tick_ts",
                      "no_decision", "bad_decision_ts", "stale_decision")


# (after_iso, until_iso) -> raw activity dicts; exhaustively paginated by the broker layer,
# which raises on ANY failure or partial page (the gate then fails closed, never passes).
ActivitiesFetch = Callable[[str, str], list[dict[str, Any]]]


@dataclass(frozen=True)
class AssembledEvidence:
    """``ForwardEvidence`` plus the window anchors / breadth facts the evaluation row records."""

    evidence: ForwardEvidence
    first_tick_id: int | None
    last_tick_id: int | None
    first_tick_ts: str | None
    last_tick_ts: str | None
    account_id: str | None
    n_concurrent_forward: int
    n_prior_forward_looks: int
    excluded: dict[str, int]  # per-filter exclusion counts for the CLI payload


def _parse_dt(value: Any) -> datetime | None:
    """ISO-8601 -> aware-UTC datetime, or None on anything unparseable — INCLUDING a tz-naive
    timestamp. Every legitimate writer stamps an explicit offset (the tick clock normalizes to
    UTC and falls back to ``clock_source='local'`` on a naive venue clock; bar-schema decision
    timestamps are tz-aware UTC), so a naive string can only be a raw-write fabrication: it is
    rejected fail-closed rather than guessed at, and an aware-vs-naive comparison can never
    raise mid-gate. Aware values are normalized to UTC so ``.date()`` session arithmetic uses
    the UTC date — an exotic offset cannot shift a tick into a not-yet-traded session."""
    if not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    return dt.astimezone(UTC) if dt.tzinfo is not None else None


def _identity_matches(row: sqlite3.Row, identity: ArtifactIdentity) -> bool:
    """All three hashes must match. A NULL stored hash never matches (legacy rows fail closed);
    a None ``identity.dependency_hash`` (lockfile absent) matches NOTHING — same fail-closed
    rule as ``has_valid_approval``."""
    if identity.dependency_hash is None:
        return False
    return (row["code_hash"] == identity.code_hash
            and row["config_hash"] == identity.config_hash
            and row["dependency_hash"] == identity.dependency_hash)


def _inadmissible_reason(
    row: sqlite3.Row, identity: ArtifactIdentity, calendar: SessionCalendar, now_utc: datetime,
) -> str | None:
    """The FIRST failing admissibility filter (spec order), or None for an admissible tick."""
    if row["clock_source"] != "broker":
        return "local_clock"
    if not _identity_matches(row, identity):
        return "identity_drift"
    if row["account_id"] is None:
        return "legacy_null"
    tick_dt = _parse_dt(row["tick_ts"])
    if tick_dt is None or tick_dt > now_utc:
        return "bad_tick_ts"
    if row["decision_ts"] is None:
        return "no_decision"
    decision_dt = _parse_dt(row["decision_ts"])
    if decision_dt is None:
        return "bad_decision_ts"
    lag = calendar.sessions_between(decision_dt.date(), tick_dt.date())
    if not 0 <= lag <= _MAX_DECISION_LAG_SESSIONS:
        return "stale_decision"
    return None


def _classify_activities(
    conn: sqlite3.Connection, acts: list[dict[str, Any]],
) -> tuple[int, int]:
    """(n_external_cash_flows, n_unattributable_fills) over raw broker activities. External capital
    types always count; a FILL is attributable iff it reconciles to SOME recorded paper-venue order
    by broker order id (account-level — any current paper-book strategy's order; a missing or
    unmatched order_id is unattributable, fail closed). Everything else (DIV/INT/FEE/...) passes.
    Shared with the certificate re-verification path."""
    n_external = 0
    n_unattributable = 0
    for act in acts:
        activity_type = act.get("activity_type")
        if activity_type in EXTERNAL_CAPITAL_TYPES:
            n_external += 1
        elif activity_type == "FILL":
            order_id = act.get("order_id")
            matched = order_id is not None and conn.execute(
                "SELECT 1 FROM paper_venue_orders WHERE broker_order_id = ?",
                (order_id,),
            ).fetchone() is not None
            if not matched:
                n_unattributable += 1
    return n_external, n_unattributable


def qualified_holdout_sharpe(
    conn: sqlite3.Connection, strategy_id: int, identity: ArtifactIdentity,
) -> float | None:
    """RAW measured holdout Sharpe from the newest QUALIFIED backtest gate row: passed=1,
    pit_ok=1, pit_override=0, identity == current. None -> the forward gate fails closed."""
    if identity.dependency_hash is None:
        return None
    row = conn.execute(
        "SELECT decision_json FROM gate_evaluations WHERE strategy_id=? AND passed=1"
        " AND pit_ok=1 AND pit_override=0 AND code_hash=? AND config_hash=?"
        " AND dependency_hash=? ORDER BY id DESC LIMIT 1",
        (strategy_id, identity.code_hash, identity.config_hash, identity.dependency_hash),
    ).fetchone()
    if row is None:
        return None
    checks = json.loads(row["decision_json"]).get("checks", [])
    vals = [c.get("value") for c in checks if c.get("name") == "holdout_sharpe"]
    return float(vals[0]) if vals and vals[0] is not None else None


def assemble_forward_evidence(
    conn: sqlite3.Connection,
    *,
    strategy_id: int,
    name: str,
    identity: ArtifactIdentity,
    calendar: SessionCalendar,
    now: datetime,
    activities_fetch: ActivitiesFetch,
) -> AssembledEvidence:
    """Assemble one strategy's forward-test evidence from its paper-lane ticks, the audit
    trail, and the broker's account activities. Pure read path — writes nothing.

    Window bounds use the DB-assigned row ``id`` / writer-stamped ``recorded_at`` (never
    ``tick_ts`` — you cannot bound a universe by the very timestamp whose quality you're
    auditing). The integrity universe is WIDER than the observation set: every paper-lane row
    for this strategy_id from the first admissible row onward, so a bad tick cannot hide by
    being inadmissible."""
    # Normalize (not just tag) to UTC: an aware non-UTC `now` kept as-is would make `.date()`
    # session arithmetic use the LOCAL date, shifting session boundaries (e.g. staleness).
    now_utc = now.astimezone(UTC) if now.tzinfo is not None else now.replace(tzinfo=UTC)
    now_iso = now_utc.isoformat()

    # 1-2. Fetch in id order; partition into admissible ticks vs per-filter exclusions.
    rows = conn.execute(
        "SELECT id, tick_ts, decision_ts, equity, reconcile_ok, clock_source, code_hash,"
        " config_hash, dependency_hash, account_id, recorded_at"
        " FROM tick_snapshots WHERE lane='paper' AND strategy_id=? ORDER BY id",
        (strategy_id,),
    ).fetchall()
    excluded = dict.fromkeys(_EXCLUSION_FILTERS, 0)
    admissible: list[sqlite3.Row] = []
    for row in rows:
        reason = _inadmissible_reason(row, identity, calendar, now_utc)
        if reason is None:
            admissible.append(row)
        else:
            excluded[reason] += 1

    # 3. Observations: key by decision session; the last (max-id) admissible tick per session
    # wins; equity in session order -> daily simple returns -> the SHARED backtest metrics.
    by_session: dict[date, sqlite3.Row] = {}
    for row in admissible:  # id order, so later assignment == max id
        decision_dt = _parse_dt(row["decision_ts"])
        assert decision_dt is not None  # admissibility already proved it parses
        by_session[calendar.session_on_or_before(decision_dt.date())] = row
    sessions = sorted(by_session)
    equities = [float(by_session[s]["equity"]) for s in sessions]
    returns = pd.Series(equities, dtype=float).pct_change().dropna()
    m = metrics_from_returns(returns)

    # 4. Coverage: decided sessions over trading sessions in [first, last] observation session.
    if sessions:
        session_coverage = len(sessions) / len(
            calendar.sessions_in_range(sessions[0], sessions[-1]))
    else:
        session_coverage = 0.0

    # 5. Integrity universe: EVERY paper-lane row for this strategy from the first admissible
    # row onward — inadmissible rows cannot hide. Empty when there are no observations at all
    # (the gate fails on the missing observations anyway).
    n_reconcile_failures = 0
    n_defective_ticks = 0
    if admissible:
        first_admissible_id = admissible[0]["id"]
        for row in rows:
            if row["id"] < first_admissible_id:
                continue
            if not row["reconcile_ok"]:
                n_reconcile_failures += 1
            tick_dt = _parse_dt(row["tick_ts"])
            if tick_dt is None or tick_dt > now_utc:
                n_defective_ticks += 1

    # 6. Breakers: current kill/halt state, plus kill-switch trip EVENTS inside the window —
    # a tripped-then-resumed forward test is a failed forward test.
    kill_switch_tripped = is_tripped(conn, name)
    global_halt_engaged = is_engaged(conn)
    n_kill_trips_in_window = 0
    if admissible:
        window_start_recorded_at = admissible[0]["recorded_at"]
        n_kill_trips_in_window = conn.execute(
            "SELECT COUNT(*) FROM audit_log"
            " WHERE strategy=? AND action='kill_switch_trip' AND ts >= ?",
            (name, window_start_recorded_at),
        ).fetchone()[0]

    # 7. Single account: the admissible ticks must all share ONE account (mixed-account evidence
    # is a tenancy violation). Siblings on the same account are ALLOWED: the multi-tenant book
    # attributes each strategy's return series via its own per-strategy NAV ticks (#314/#316a-b)
    # and account-level fill attribution (_classify_activities), so a sibling cannot contaminate
    # it.
    account_id: str | None = None
    single_account_ok = True
    if admissible:
        account_id = admissible[-1]["account_id"]
        single_account_ok = len({row["account_id"] for row in admissible}) == 1

    # 8. Concurrency breadth (recorded, not yet enforced): distinct strategies with ANY
    # paper-lane ticks overlapping the window — failed/inadmissible siblings still inflated
    # the family-wise error rate.
    n_concurrent_forward = 0
    if admissible:
        n_concurrent_forward = conn.execute(
            "SELECT COUNT(DISTINCT strategy) FROM tick_snapshots"
            " WHERE lane='paper' AND recorded_at >= ? AND recorded_at <= ?",
            (admissible[0]["recorded_at"], now_iso),
        ).fetchone()[0]

    # 8b. Optional-stopping count (#431): PRIOR forward-gate evaluations of THIS strategy+identity
    # in the ledger, WITHIN a trailing FORWARD_RELOOK_HORIZON_SESSIONS window — the re-runnable
    # forward gate lets an agent take repeated looks at the same strategy, inflating the family-wise
    # error rate.
    #
    # HORIZON BOUND (the #324 anti-scaling fix): only looks whose `created_at` is on or after the
    # session FORWARD_RELOOK_HORIZON_SESSIONS trading sessions before `now` count. Without this the
    # live wall's MANDATORY periodic re-certification (a passing run must stay <= 10 sessions old)
    # would accumulate looks forever and eventually make the bar unpassable — taxing a strategy for
    # COMPLYING with the freshness wall. Bounded, burst-rate-limiting counting taxes clustered
    # re-runs and lets looks age out; it does NOT claim to control sequential false-pass over
    # unbounded time (that would require the very lifetime-cumulative penalty #324 removed).
    #
    # SCOPE — narrower v1 (identity-exact-match), stated honestly. The count keys on an EXACT
    # code+config+dependency hash match, NOT on a #222 family/lineage component. This is a
    # deliberate, documented v1 limitation, not the escape-hatch-proof lineage scope: an agent that
    # RE-REGISTERS a peeked strategy under a new name (new strategy_id) or edits any byte of code
    # (new code_hash) resets this count to 0. Widening to lineage-component scoping (walking the
    # #222 family DAG) is deferred follow-up work; it is NOT implemented here and this code makes no
    # claim to close that hatch. What v1 DOES tax honestly: repeated looks at the SAME fixed
    # artifact within the horizon (the dominant optional-stopping pattern — an agent re-running
    # `paper promote` each session hoping the growing window clears the bar). A code change
    # legitimately resets the count anyway — it forces a fresh `research promote` pass first. A None
    # dependency_hash matches nothing (SQL `= NULL` never matches, and the holdout check already
    # fails closed there) — leave 0.
    #
    # RESIDUAL RACE (documented, not closed in v1). This is a plain read; the row for the current
    # run is inserted later, in a SEPARATE write transaction (run_forward_gate ordering). Two
    # `paper promote` runs on the same identity that interleave read-read-insert-insert can both
    # observe the same look count L and both pass on a stale tax. This is a bounded, tighten-only
    # residual: the worst case UNDER-counts by the number of truly-concurrent racing promotes of one
    # identity (a rare operator pattern — a single agent drives one strategy's gate serially), it
    # can only make the tax too SMALL for that one race (never spuriously fail an honest run), and
    # the NEXT run counts both committed rows and re-taxes. Fully closing it (recompute-and-insert
    # in one BEGIN IMMEDIATE critical section, as `record_gate_with_fdr_and_maybe_promote` does for
    # the research gate) is deferred follow-up; v1 accepts this residual rather than shipping
    # an unbuilt in-lock accounting method.
    n_prior_forward_looks = 0
    if identity.dependency_hash is not None:
        horizon_session = calendar.session_on_or_before(now_utc.date())
        for _ in range(FORWARD_RELOOK_HORIZON_SESSIONS):
            horizon_session = calendar.previous_session(horizon_session)
        horizon_cutoff_iso = datetime(
            horizon_session.year, horizon_session.month, horizon_session.day, tzinfo=UTC
        ).isoformat()
        n_prior_forward_looks = conn.execute(
            "SELECT COUNT(*) FROM forward_gate_evaluations WHERE strategy_id=? AND code_hash=?"
            " AND config_hash=? AND dependency_hash=? AND created_at>=?",
            (strategy_id, identity.code_hash, identity.config_hash, identity.dependency_hash,
             horizon_cutoff_iso),
        ).fetchone()[0]

    # 9-10. Broker activities + staleness. With no admissible ticks there is no window: skip
    # the broker entirely (activities_ok=True, zeros — the gate already fails on observations)
    # and staleness is None (fail closed in the evaluator). Any fetch/classify failure means
    # the account is UNVERIFIABLE: partial history never passes.
    staleness_sessions: int | None = None
    activities_ok = True
    n_external_cash_flows = 0
    n_unattributable_fills = 0
    if admissible:
        last_tick_dt = _parse_dt(admissible[-1]["tick_ts"])
        assert last_tick_dt is not None  # admissibility already proved it parses
        staleness_sessions = calendar.sessions_between(last_tick_dt.date(), now_utc.date())
        # Alpaca's activities `after` bound is EXCLUSIVE: an external deposit stamped at
        # EXACTLY the first-tick instant would escape an `after == first_tick_ts` window, so
        # widen the start 1s earlier. The overlap errs fail-closed — an extra pre-window
        # capital movement can only FAIL the gate, never pass it; pre-window FILLs from ANY
        # paper-book strategy's order on this account remain attributable (account-level
        # attribution).
        first_tick_dt = _parse_dt(admissible[0]["tick_ts"])
        assert first_tick_dt is not None  # admissibility already proved it parses
        window_after = (first_tick_dt - timedelta(seconds=1)).isoformat()
        try:
            acts = activities_fetch(window_after, now_iso)
            n_external_cash_flows, n_unattributable_fills = _classify_activities(
                conn, acts)
        except Exception:
            activities_ok = False
            n_external_cash_flows = 0
            n_unattributable_fills = 0

    evidence = ForwardEvidence(
        n_return_observations=len(returns),
        session_coverage=float(session_coverage),
        realized_sharpe=float(m["sharpe"]),
        realized_skew=float(m["skewness"]),
        realized_kurtosis=float(m["kurtosis"]),
        realized_vol=float(m["ann_volatility"]),
        realized_max_drawdown=abs(float(m["max_drawdown"])),
        holdout_sharpe=qualified_holdout_sharpe(conn, strategy_id, identity),
        n_reconcile_failures=n_reconcile_failures,
        n_defective_ticks=n_defective_ticks,
        kill_switch_tripped=kill_switch_tripped,
        global_halt_engaged=global_halt_engaged,
        n_kill_trips_in_window=int(n_kill_trips_in_window),
        single_account_ok=single_account_ok,
        activities_ok=activities_ok,
        n_external_cash_flows=n_external_cash_flows,
        n_unattributable_fills=n_unattributable_fills,
        staleness_sessions=staleness_sessions,
        n_prior_forward_looks=int(n_prior_forward_looks),
        n_concurrent_forward=int(n_concurrent_forward),
    )
    return AssembledEvidence(
        evidence=evidence,
        first_tick_id=admissible[0]["id"] if admissible else None,
        last_tick_id=admissible[-1]["id"] if admissible else None,
        first_tick_ts=admissible[0]["tick_ts"] if admissible else None,
        last_tick_ts=admissible[-1]["tick_ts"] if admissible else None,
        account_id=account_id,
        n_concurrent_forward=int(n_concurrent_forward),
        n_prior_forward_looks=int(n_prior_forward_looks),
        excluded=excluded,
    )
