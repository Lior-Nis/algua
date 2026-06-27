"""Forward-test evidence assembly (#124): DB + broker -> ``ForwardEvidence``.

The protected orchestration layer for the ``paper -> forward_tested`` gate, mirroring
``registry/promotion.py`` for the shortlist gate. This module is the I/O half: it reads the
paper-lane tick rows, partitions them into admissible observations vs per-filter exclusions,
builds the daily return series with the SHARED backtest metrics (same Sharpe/vol/drawdown
definitions as the holdout the gate compares against), runs the integrity / single-tenant /
broker-activities checks, and locates the qualified backtest gate row. The pure judgement lives
in ``algua.research.forward_gates`` — this module never decides, only assembles.

CODEOWNERS-protected: every clause here is a wall against an autonomous agent fabricating
forward evidence (back-dated ticks, identity drift, sibling contamination, manual fills,
external capital). Each helper FAILS CLOSED — ambiguity is never resolved in the
strategy's favor.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any, Protocol

import pandas as pd

from algua.backtest.metrics import metrics_from_returns
from algua.contracts.lifecycle import Actor, Stage, TransitionError, validate_transition
from algua.registry.approvals import compute_artifact_hashes
from algua.registry.repository import ArtifactIdentity, StrategyRecord, StrategyRepository
from algua.research.forward_gates import (
    CERTIFICATE_FRESH_SESSIONS,
    ForwardEvidence,
    ForwardGateCriteria,
    ForwardGateDecision,
    evaluate_forward_gate,
)
from algua.risk.global_halt import is_engaged
from algua.risk.kill_switch import is_tripped

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


class SessionCalendar(Protocol):
    """The session arithmetic the assembly needs (satisfied by ``MarketCalendar``)."""

    def session_on_or_before(self, day: date) -> date: ...
    def sessions_between(self, a: date, b: date) -> int: ...
    def sessions_in_range(self, start: date, end: date) -> list[date]: ...


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
    conn: sqlite3.Connection, strategy_id: int, acts: list[dict[str, Any]],
) -> tuple[int, int]:
    """(n_external_cash_flows, n_unattributable_fills) over raw broker activities. External
    capital types always count; a FILL is attributable iff it reconciles to one of THIS
    strategy's persisted paper orders by broker order id AND strategy_id (a missing order_id
    is unattributable — fail closed). Everything else (DIV/INT/FEE/...) passes. Shared with
    the certificate re-verification path."""
    n_external = 0
    n_unattributable = 0
    for act in acts:
        activity_type = act.get("activity_type")
        if activity_type in EXTERNAL_CAPITAL_TYPES:
            n_external += 1
        elif activity_type == "FILL":
            order_id = act.get("order_id")
            matched = order_id is not None and conn.execute(
                "SELECT 1 FROM paper_venue_orders WHERE strategy_id = ? AND broker_order_id = ?",
                (strategy_id, order_id),
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

    # 7. Single tenant: the admissible ticks must share ONE account, and no other strategy may
    # have paper-lane ticks on that account inside [first admissible recorded_at, now].
    account_id: str | None = None
    single_tenant_ok = True
    if admissible:
        account_id = admissible[-1]["account_id"]
        distinct_accounts = {row["account_id"] for row in admissible}
        if len(distinct_accounts) > 1:
            single_tenant_ok = False  # mixed-account evidence is itself a tenancy violation
        else:
            n_siblings = conn.execute(
                "SELECT COUNT(*) FROM tick_snapshots WHERE lane='paper' AND account_id=?"
                " AND strategy_id != ? AND recorded_at >= ? AND recorded_at <= ?",
                (account_id, strategy_id, admissible[0]["recorded_at"], now_iso),
            ).fetchone()[0]
            single_tenant_ok = n_siblings == 0

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
        # capital movement can only FAIL the gate, never pass it; pre-window FILLs from this
        # strategy's own orders remain attributable.
        first_tick_dt = _parse_dt(admissible[0]["tick_ts"])
        assert first_tick_dt is not None  # admissibility already proved it parses
        window_after = (first_tick_dt - timedelta(seconds=1)).isoformat()
        try:
            acts = activities_fetch(window_after, now_iso)
            n_external_cash_flows, n_unattributable_fills = _classify_activities(
                conn, strategy_id, acts)
        except Exception:
            activities_ok = False
            n_external_cash_flows = 0
            n_unattributable_fills = 0

    evidence = ForwardEvidence(
        n_return_observations=len(returns),
        session_coverage=float(session_coverage),
        realized_sharpe=float(m["sharpe"]),
        realized_vol=float(m["ann_volatility"]),
        realized_max_drawdown=abs(float(m["max_drawdown"])),
        holdout_sharpe=qualified_holdout_sharpe(conn, strategy_id, identity),
        n_reconcile_failures=n_reconcile_failures,
        n_defective_ticks=n_defective_ticks,
        kill_switch_tripped=kill_switch_tripped,
        global_halt_engaged=global_halt_engaged,
        n_kill_trips_in_window=int(n_kill_trips_in_window),
        single_tenant_ok=single_tenant_ok,
        activities_ok=activities_ok,
        n_external_cash_flows=n_external_cash_flows,
        n_unattributable_fills=n_unattributable_fills,
        staleness_sessions=staleness_sessions,
    )
    return AssembledEvidence(
        evidence=evidence,
        first_tick_id=admissible[0]["id"] if admissible else None,
        last_tick_id=admissible[-1]["id"] if admissible else None,
        first_tick_ts=admissible[0]["tick_ts"] if admissible else None,
        last_tick_ts=admissible[-1]["tick_ts"] if admissible else None,
        account_id=account_id,
        n_concurrent_forward=int(n_concurrent_forward),
        excluded=excluded,
    )


def verify_forward_certificate(
    repo: StrategyRepository,
    conn: sqlite3.Connection,
    *,
    name: str,
    strategy_id: int,
    identity: ArtifactIdentity,
    calendar: SessionCalendar,
    now: datetime,
    activities_fetch: ActivitiesFetch,
    account_id_fetch: Callable[[], str],
) -> dict[str, Any]:
    """The evidence precondition of the live wall (#124). NOT waivable in-band: there is
    deliberately no flag; a human who must bypass owns the DB. Selection is the NEWEST row
    pass-or-fail so a newer failed re-evaluation invalidates an older pass.

    Checks, in order, each failing closed with an actionable ``TransitionError``: a certificate
    exists for THIS strategy + the current identity; its newest verdict is a pass; it is fresh
    (at most ``CERTIFICATE_FRESH_SESSIONS`` sessions old); the record since certification is
    clean (no reconcile-failed and no malformed/future-stamped paper ticks after
    ``last_tick_id`` — the gate's own defect rule — no kill-switch trips after ``created_at``,
    kill switch and global halt clear); the broker the live gate queries is the SAME account
    the certificate was earned on (``account_id_fetch`` vs the row — account drift makes
    hygiene continuity unverifiable); and account hygiene re-verified over ``[created_at,
    now]`` with the same activity-classification rules the gate itself uses.
    Returns the certificate summary the human signs against."""
    # Normalize (not just tag) to UTC — same rule as assemble_forward_evidence: `.date()`
    # freshness arithmetic must use the UTC date, not a local one.
    now_utc = now.astimezone(UTC) if now.tzinfo is not None else now.replace(tzinfo=UTC)
    row = repo.latest_forward_gate_row(
        strategy_id, identity.code_hash, identity.config_hash, identity.dependency_hash)
    if row is None:
        raise TransitionError(
            "go-live requires a forward-test certificate for the current "
            "code+config+dependency; run `algua paper promote`")
    if not row["passed"]:
        raise TransitionError(
            f"the newest forward-gate evaluation (id={row['id']}, created_at="
            f"{row['created_at']}) FAILED, which invalidates any prior pass; "
            "re-run `algua paper promote`")
    age = calendar.sessions_between(
        datetime.fromisoformat(row["created_at"]).date(), now_utc.date())
    if age > CERTIFICATE_FRESH_SESSIONS:
        raise TransitionError(
            f"the forward-test certificate is stale: {age} sessions old, max "
            f"{CERTIFICATE_FRESH_SESSIONS}; re-run `algua paper promote` to refresh it")
    ticks_since = conn.execute(
        "SELECT tick_ts, reconcile_ok FROM tick_snapshots WHERE lane='paper' AND strategy_id=?"
        " AND id > ?",
        (strategy_id, row["last_tick_id"] or 0),
    ).fetchall()
    n_bad_ticks = sum(1 for t in ticks_since if not t["reconcile_ok"])
    if n_bad_ticks:
        raise TransitionError(
            f"{n_bad_ticks} reconcile-failed paper tick(s) since certification; investigate,"
            " then re-run `algua paper promote`")
    n_malformed = sum(
        1 for t in ticks_since
        if (tick_dt := _parse_dt(t["tick_ts"])) is None or tick_dt > now_utc)
    if n_malformed:
        raise TransitionError(
            f"{n_malformed} malformed or future-stamped paper tick(s) since certification"
            " (the gate's defective-tick rule); investigate, then re-run `algua paper promote`")
    n_trips = conn.execute(
        "SELECT COUNT(*) FROM audit_log WHERE strategy=? AND action='kill_switch_trip'"
        " AND ts >= ?",
        (name, row["created_at"]),
    ).fetchone()[0]
    if n_trips:
        raise TransitionError(
            f"{n_trips} kill-switch trip(s) since certification; investigate, then re-run"
            " `algua paper promote`")
    if is_tripped(conn, name) or is_engaged(conn):
        raise TransitionError(
            "kill switch / global halt engaged; clear it before going live")
    # Account continuity: the hygiene re-check below queries whatever account the CURRENT broker
    # credentials point at, while the certificate's evidence lives on row["account_id"]. If they
    # differ (the operator switched paper accounts after certification), a deposit or manual fill
    # on the certified account would be invisible here — unverifiable continuity fails closed.
    if row["account_id"] is None:
        raise TransitionError(
            "the forward-test certificate records no account_id, so hygiene continuity since "
            "certification is unverifiable; re-run `algua paper promote`")
    try:
        current_account = account_id_fetch()
    except Exception as exc:
        raise TransitionError(
            f"could not verify the broker account id ({exc}); failing closed") from exc
    if current_account != row["account_id"]:
        raise TransitionError(
            f"the broker credentials point at account {current_account!r} but the certificate "
            f"was earned on account {row['account_id']!r}; hygiene continuity since "
            "certification is unverifiable — re-run `algua paper promote` on this account")
    # Alpaca's activities `after` bound is EXCLUSIVE: a capital movement stamped at EXACTLY
    # the certification instant would escape an `after == created_at` window, so widen the
    # start 1s earlier. The overlap errs fail-closed — an extra pre-window movement can only
    # FAIL the wall, never pass it; pre-window FILLs from this strategy's own orders remain
    # attributable.
    window_after = (datetime.fromisoformat(row["created_at"]) - timedelta(seconds=1)).isoformat()
    try:
        acts = activities_fetch(window_after, now_utc.isoformat())
    except Exception as exc:
        raise TransitionError(
            f"could not verify account activities since certification ({exc}); "
            "failing closed") from exc
    n_external, n_unattributable = _classify_activities(conn, strategy_id, acts)
    if n_external or n_unattributable:
        raise TransitionError(
            f"account hygiene failed since certification: {n_external} external capital "
            f"flow(s), {n_unattributable} unattributable fill(s); re-run `algua paper promote`")
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "realized_sharpe": row["realized_sharpe"],
        "holdout_sharpe": row["holdout_sharpe"],
        "n_forward_observations": row["n_forward_observations"],
        "n_concurrent_forward": row["n_concurrent_forward"],
    }


def guard_forward_relaxations(actor: Actor, criteria: ForwardGateCriteria) -> None:
    """Each threshold has a strict direction; an agent may only move it stricter (#124).
    Mirrors ``guard_agent_relaxations``: the agent only ever sees the strict gate."""
    if actor is Actor.HUMAN:
        return
    defaults = ForwardGateCriteria()
    higher_is_stricter = ("min_forward_observations", "min_session_coverage",
                          "degradation_factor", "sharpe_floor", "min_forward_vol")
    lower_is_stricter = ("max_forward_drawdown", "max_staleness_sessions")
    relaxed = [f for f in higher_is_stricter if getattr(criteria, f) < getattr(defaults, f)]
    relaxed += [f for f in lower_is_stricter if getattr(criteria, f) > getattr(defaults, f)]
    if relaxed:
        raise ValueError(
            "forward-gate relaxation requires --actor human: " + ", ".join(sorted(relaxed)))


def forward_promotion_preflight(
    repo: StrategyRepository, name: str, *, actor: Actor, criteria: ForwardGateCriteria,
) -> StrategyRecord:
    """Pre-work refusals (mirrors ``promotion_preflight``): actor legality, relaxation guard,
    stage legality. FORWARD_TESTED is legal because a re-evaluation refreshes the live-wall
    certificate without a stage change (#124)."""
    # SYSTEM would pass as "not human" (strict) yet mint a row it can never consume.
    if actor not in (Actor.AGENT, Actor.HUMAN):
        raise ValueError(f"paper promote requires --actor agent or human, got {actor.value}")
    guard_forward_relaxations(actor, criteria)
    rec = repo.get(name)
    if rec.stage not in (Stage.PAPER, Stage.FORWARD_TESTED):
        raise TransitionError(
            f"paper promote requires stage paper or forward_tested, got {rec.stage.value}")
    if rec.stage is Stage.PAPER:
        validate_transition(rec.stage, Stage.FORWARD_TESTED)
    return rec


@dataclass
class ForwardPromotionOutcome:
    decision: ForwardGateDecision
    promoted: bool
    assembled: AssembledEvidence


def run_forward_gate(
    repo: StrategyRepository,
    conn: sqlite3.Connection,
    *,
    name: str,
    actor: Actor,
    criteria: ForwardGateCriteria,
    calendar: SessionCalendar,
    now: datetime,
    activities_fetch: ActivitiesFetch,
) -> ForwardPromotionOutcome:
    """Assemble evidence -> evaluate -> record (pass AND fail) -> on pass from PAPER record AND
    promote in one transaction. At FORWARD_TESTED a passing run is the certificate refresh: a
    new row, no stage change.

    Identity is computed ONCE via ``compute_artifact_hashes`` and feeds the evidence
    admissibility filter, the evaluation row, AND the transition's pinned hashes — they can
    never disagree."""
    rec = repo.get(name)
    identity = compute_artifact_hashes(name)
    asm = assemble_forward_evidence(
        conn, strategy_id=rec.id, name=name, identity=identity, calendar=calendar, now=now,
        activities_fetch=activities_fetch)
    decision = evaluate_forward_gate(asm.evidence, criteria)
    gate_row: dict[str, Any] = {
        "passed": decision.passed,
        "n_forward_observations": asm.evidence.n_return_observations,
        "min_forward_observations": criteria.min_forward_observations,
        "session_coverage": asm.evidence.session_coverage,
        "realized_sharpe": asm.evidence.realized_sharpe,
        "holdout_sharpe": asm.evidence.holdout_sharpe,
        "degradation_factor": criteria.degradation_factor,
        "sharpe_floor": criteria.sharpe_floor,
        "realized_vol": asm.evidence.realized_vol,
        "min_forward_vol": criteria.min_forward_vol,
        "realized_max_drawdown": asm.evidence.realized_max_drawdown,
        "max_forward_drawdown": criteria.max_forward_drawdown,
        "first_tick_id": asm.first_tick_id, "last_tick_id": asm.last_tick_id,
        "first_tick_ts": asm.first_tick_ts, "last_tick_ts": asm.last_tick_ts,
        "max_staleness_sessions": criteria.max_staleness_sessions,
        "n_reconcile_failures": asm.evidence.n_reconcile_failures,
        "n_concurrent_forward": asm.n_concurrent_forward,
        "account_id": asm.account_id,
        "code_hash": identity.code_hash, "config_hash": identity.config_hash,
        "dependency_hash": identity.dependency_hash,
        "decision_json": json.dumps(decision.to_dict(), sort_keys=True),
    }
    promoted = False
    if decision.passed and rec.stage is Stage.PAPER:
        # "Record passing row + stage CAS + transition row" is ONE sqlite transaction (#124
        # GATE-2): the old record-then-transition shape committed a consumable token first, so
        # a raced/failed transition banked a consumed=0 pass an agent could spend within the
        # TTL after a later demotion — re-entry without a fresh gate run. Going through the
        # repository instead of ``transitions.transition_strategy`` drops exactly two policy
        # steps, both deliberately: ``validate_transition`` (paper -> forward_tested is a
        # statically legal edge by construction here — preflight checked it — and the CAS's
        # from_stage=paper predicate enforces the stage atomically) and the consumable-token
        # lookup (we promote on the very evidence row we insert, in the same transaction —
        # there is no token to find or consume; the row is born spent). The standalone token
        # path in ``transitions`` remains for tokens minted by earlier runs.
        repo.record_forward_pass_and_promote(
            rec, gate_row=gate_row, actor=actor, reason=_forward_gate_reason(decision))
        promoted = True
    else:
        repo.record_forward_gate_evaluation(
            rec.id, **gate_row, actor=actor.value,
            # A refresh at forward_tested must refresh the live certificate WITHOUT minting a
            # re-entry token (#124 GATE-2): only a run FROM paper writes a consumable row, so a
            # demote-then-re-promote can never bank a refresh — it always re-runs the full gate.
            consumable=rec.stage is Stage.PAPER)
    return ForwardPromotionOutcome(decision=decision, promoted=promoted, assembled=asm)


def _forward_gate_reason(decision: ForwardGateDecision) -> str:
    """Human-readable gate summary (mirrors ``promotion._gate_reason``). Metric checks render
    value/op/threshold; boolean checks render name=pass|fail."""
    parts: list[str] = []
    for c in decision.checks:
        if "value" in c and c.get("value") is not None and c.get("threshold") is not None:
            parts.append(f"{c['name']}={c['value']:.4g}{c['op']}{c['threshold']:.4g}")
        else:
            parts.append(f"{c['name']}={'pass' if c['passed'] else 'fail'}")
    return "forward gate pass: " + ", ".join(parts)
