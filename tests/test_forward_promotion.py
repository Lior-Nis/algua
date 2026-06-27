"""Task 7 (#124): forward-evidence assembly — DB + broker -> ForwardEvidence.

These tests use a FAKE weekday-arithmetic calendar (Mon-Fri sessions, no holidays) instead of
``MarketCalendar`` so the suite does not depend on exchange_calendars session data; the fake
mirrors the real wrapper's session_on_or_before / sessions_between / sessions_in_range contracts.
Admissible ticks are seeded through the REAL writer (``record_tick_snapshot``) with recorded_at
then pinned via UPDATE (the writer stamps wall-clock time); legacy/defective rows use raw INSERTs
because the writer refuses to produce them — exactly the adversarial shapes the gate must reject.
"""
import json
from datetime import UTC, date, datetime, timedelta, timezone

import pandas as pd
import pytest

from algua.backtest.metrics import metrics_from_returns
from algua.contracts.lifecycle import Actor, Stage, TransitionError
from algua.execution.live_ledger import (
    backfill_paper_venue_broker_order_id,
    record_paper_venue_order,
)
from algua.execution.order_state import record_tick_snapshot
from algua.registry.db import connect, migrate
from algua.registry.forward_promotion import (
    EXTERNAL_CAPITAL_TYPES,
    AssembledEvidence,
    assemble_forward_evidence,
    forward_promotion_preflight,
    guard_forward_relaxations,
    qualified_holdout_sharpe,
    run_forward_gate,
)
from algua.registry.repository import ArtifactIdentity
from algua.registry.store import SqliteStrategyRepository
from algua.registry.transitions import transition_strategy
from algua.research.forward_gates import FORWARD_TOKEN_TTL_DAYS, ForwardGateCriteria

# Friday 2026-06-12: June 2026 weekdays are Jun 1-5 (Mon-Fri) and Jun 8-12 (Mon-Fri).
NOW = datetime(2026, 6, 12, 21, 0, tzinfo=UTC)
IDENT = ArtifactIdentity(code_hash="c", config_hash="g", dependency_hash="d")

EXCLUSION_KEYS = {"local_clock", "identity_drift", "legacy_null", "bad_tick_ts",
                  "no_decision", "bad_decision_ts", "stale_decision"}


class FakeCalendar:
    """Weekday arithmetic standing in for MarketCalendar (no exchange_calendars data)."""

    def session_on_or_before(self, day: date) -> date:
        while day.weekday() >= 5:
            day -= timedelta(days=1)
        return day

    def sessions_in_range(self, start: date, end: date) -> list[date]:
        out, d = [], start
        while d <= end:
            if d.weekday() < 5:
                out.append(d)
            d += timedelta(days=1)
        return out

    def sessions_between(self, a: date, b: date) -> int:
        sa, sb = self.session_on_or_before(a), self.session_on_or_before(b)
        lo, hi = (sa, sb) if sb >= sa else (sb, sa)
        n = len(self.sessions_in_range(lo, hi)) - 1
        return n if sb >= sa else -n


CAL = FakeCalendar()


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "r.db")
    migrate(c)
    c.execute("INSERT INTO strategies(name, stage, created_at, updated_at) "
              "VALUES ('s', 'paper', 't', 't')")
    c.commit()
    return c


def _ts(day: date, hour: int = 20) -> str:
    return datetime(day.year, day.month, day.day, hour, tzinfo=UTC).isoformat()


def _prev_weekday(day: date) -> date:
    day -= timedelta(days=1)
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day


_AUTO = "AUTO"


def seed_tick(conn, day: date, equity: float, *, name="s", strategy_id=1, decision_ts=_AUTO,
              reconcile_ok=True, clock_source="broker", code_hash="c", config_hash="g",
              dependency_hash="d", account_id="acct", recorded_at=None, hour=20) -> int:
    """Seed via the REAL writer, then pin recorded_at (the writer stamps wall-clock time,
    which would fall outside the fixed test NOW)."""
    if decision_ts == _AUTO:
        decision_ts = _ts(_prev_weekday(day))
    record_tick_snapshot(
        conn, name, tick_ts=_ts(day, hour), decision_ts=decision_ts, equity=equity,
        peak_equity=None, positions={}, n_submitted=0, reconcile_ok=reconcile_ok,
        lane="paper", strategy_id=strategy_id, code_hash=code_hash, config_hash=config_hash,
        dependency_hash=dependency_hash, account_id=account_id, cash=0.0,
        clock_source=clock_source,
    )
    rid = conn.execute("SELECT max(id) FROM tick_snapshots").fetchone()[0]
    conn.execute("UPDATE tick_snapshots SET recorded_at=? WHERE id=?",
                 (recorded_at or _ts(day, hour), rid))
    conn.commit()
    return rid


def raw_tick(conn, *, name="s", strategy_id=1, tick_ts, decision_ts, equity=100.0,
             reconcile_ok=1, lane="paper", clock_source="broker", code_hash="c",
             config_hash="g", dependency_hash="d", account_id="acct", recorded_at=None) -> int:
    """Raw INSERT for legacy/defective rows the real writer refuses to produce."""
    cur = conn.execute(
        "INSERT INTO tick_snapshots(strategy, tick_ts, decision_ts, equity, positions, "
        "n_submitted, reconcile_ok, lane, strategy_id, code_hash, config_hash, "
        "dependency_hash, account_id, cash, clock_source, recorded_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (name, tick_ts, decision_ts, equity, "{}", 0, reconcile_ok, lane, strategy_id,
         code_hash, config_hash, dependency_hash, account_id, 0.0, clock_source,
         recorded_at or "2026-06-11T20:00:00+00:00"),
    )
    conn.commit()
    return cur.lastrowid


def assemble(conn, *, activities=lambda after, until: [], now=NOW,
             identity=IDENT) -> AssembledEvidence:
    return assemble_forward_evidence(
        conn, strategy_id=1, name="s", identity=identity, calendar=CAL, now=now,
        activities_fetch=activities,
    )


# ---------------------------------------------------------------------------
# Per-dimension admissibility exclusion (clause 2)
# ---------------------------------------------------------------------------

def _two_admissible(conn):
    seed_tick(conn, date(2026, 6, 10), 100.0)
    seed_tick(conn, date(2026, 6, 12), 101.0)


def test_local_clock_excluded(conn):
    _two_admissible(conn)
    seed_tick(conn, date(2026, 6, 11), 99.0, clock_source="local")
    res = assemble(conn)
    assert res.excluded["local_clock"] == 1
    assert res.evidence.n_return_observations == 1  # 2 admissible sessions -> 1 return


def test_identity_drift_excluded(conn):
    _two_admissible(conn)
    seed_tick(conn, date(2026, 6, 11), 99.0, code_hash="STALE")
    res = assemble(conn)
    assert res.excluded["identity_drift"] == 1
    assert res.evidence.n_return_observations == 1


def test_null_hash_never_matches_identity(conn):
    _two_admissible(conn)
    seed_tick(conn, date(2026, 6, 11), 99.0, dependency_hash=None)
    res = assemble(conn)
    assert res.excluded["identity_drift"] == 1
    assert res.evidence.n_return_observations == 1


def test_identity_with_none_dependency_hash_matches_nothing(conn):
    _two_admissible(conn)
    res = assemble(conn, identity=ArtifactIdentity("c", "g", None))
    assert res.excluded["identity_drift"] == 2
    assert res.evidence.n_return_observations == 0
    assert res.evidence.staleness_sessions is None


def test_legacy_null_account_excluded(conn):
    _two_admissible(conn)
    raw_tick(conn, tick_ts=_ts(date(2026, 6, 11)), decision_ts=_ts(date(2026, 6, 10)),
             account_id=None)
    res = assemble(conn)
    assert res.excluded["legacy_null"] == 1
    assert res.evidence.n_return_observations == 1


def test_bad_tick_ts_unparseable_excluded(conn):
    _two_admissible(conn)
    raw_tick(conn, tick_ts="not-a-timestamp", decision_ts=_ts(date(2026, 6, 10)))
    res = assemble(conn)
    assert res.excluded["bad_tick_ts"] == 1
    assert res.evidence.n_return_observations == 1


def test_future_tick_ts_excluded(conn):
    _two_admissible(conn)
    raw_tick(conn, tick_ts=_ts(date(2026, 6, 15)),  # Monday after NOW (Friday)
             decision_ts=_ts(date(2026, 6, 12)))
    res = assemble(conn)
    assert res.excluded["bad_tick_ts"] == 1
    assert res.evidence.n_return_observations == 1


def test_naive_tick_ts_excluded_and_defective(conn):
    # The writers always stamp an offset (tick clock normalizes to UTC; naive venue clocks fall
    # back to clock_source='local'), so a tz-NAIVE tick_ts is raw-write fabrication: it must be
    # rejected (not guessed as UTC) AND counted defective in the integrity universe — and the
    # aware-vs-naive comparison against `now` must not crash the gate.
    _two_admissible(conn)
    raw_tick(conn, tick_ts="2026-06-11T20:00:00", decision_ts=_ts(date(2026, 6, 10)))
    res = assemble(conn)
    assert res.excluded["bad_tick_ts"] == 1
    assert res.evidence.n_defective_ticks == 1
    assert res.evidence.n_return_observations == 1


def test_naive_decision_ts_excluded(conn):
    _two_admissible(conn)
    seed_tick(conn, date(2026, 6, 11), 99.0, decision_ts="2026-06-10T20:00:00")  # no offset
    res = assemble(conn)
    assert res.excluded["bad_decision_ts"] == 1
    assert res.evidence.n_return_observations == 1


def test_tick_ts_offset_normalized_to_utc_date(conn):
    # +12:00 offset whose UTC instant is in the past but whose LOCAL date is tomorrow: the
    # session math must use the UTC date (Jun 12), not let an exotic offset shift the tick into
    # a not-yet-traded session (which would mask staleness / fake decision lag).
    seed_tick(conn, date(2026, 6, 10), 100.0)
    raw_tick(conn, tick_ts="2026-06-13T08:00:00+12:00",  # == 2026-06-12T20:00:00Z, past NOW
             decision_ts=_ts(date(2026, 6, 11)), recorded_at=_ts(date(2026, 6, 12)))
    res = assemble(conn)
    assert res.excluded["bad_tick_ts"] == 0       # not future once normalized to UTC
    assert res.evidence.n_defective_ticks == 0
    assert res.evidence.staleness_sessions == 0   # measured from the UTC session (Jun 12)


def test_no_decision_excluded(conn):
    _two_admissible(conn)
    seed_tick(conn, date(2026, 6, 11), 99.0, decision_ts=None)
    res = assemble(conn)
    assert res.excluded["no_decision"] == 1
    assert res.evidence.n_return_observations == 1


def test_bad_decision_ts_excluded(conn):
    _two_admissible(conn)
    seed_tick(conn, date(2026, 6, 11), 99.0, decision_ts="garbage")
    res = assemble(conn)
    assert res.excluded["bad_decision_ts"] == 1
    assert res.evidence.n_return_observations == 1


def test_stale_decision_excluded(conn):
    _two_admissible(conn)
    # Decision Fri Jun 5, tick Thu Jun 11 -> 4 sessions apart, beyond the <=2 freshness bound.
    seed_tick(conn, date(2026, 6, 11), 99.0, decision_ts=_ts(date(2026, 6, 5)))
    res = assemble(conn)
    assert res.excluded["stale_decision"] == 1
    assert res.evidence.n_return_observations == 1


def test_decision_after_tick_is_stale(conn):
    _two_admissible(conn)
    seed_tick(conn, date(2026, 6, 9), 99.0, decision_ts=_ts(date(2026, 6, 10)))  # negative gap
    res = assemble(conn)
    assert res.excluded["stale_decision"] == 1
    assert res.evidence.n_return_observations == 1


def test_decision_two_sessions_back_is_admissible(conn):
    _two_admissible(conn)  # decision sessions Jun 9 and Jun 11
    # Exactly 2 sessions of lag (Jun 10 -> Jun 12), keyed to a distinct decision session.
    seed_tick(conn, date(2026, 6, 12), 99.0, decision_ts=_ts(date(2026, 6, 10)), hour=15)
    res = assemble(conn)
    assert res.excluded["stale_decision"] == 0
    assert res.evidence.n_return_observations == 2  # 3 decision sessions -> 2 returns


def test_first_matching_filter_wins(conn):
    _two_admissible(conn)
    # Fails BOTH local-clock and identity; counted once, under the FIRST filter only.
    seed_tick(conn, date(2026, 6, 11), 99.0, clock_source="local", code_hash="STALE")
    res = assemble(conn)
    assert res.excluded["local_clock"] == 1
    assert res.excluded["identity_drift"] == 0
    assert sum(res.excluded.values()) == 1


# ---------------------------------------------------------------------------
# Observations, returns, coverage (clauses 3-4)
# ---------------------------------------------------------------------------

def test_return_and_coverage_math_on_hand_path(conn):
    days = [date(2026, 6, d) for d in (1, 2, 3, 4, 5, 8, 9, 10, 11, 12)]
    equities = [100.0, 102.0, 101.0, 105.0, 104.0, 106.0, 108.0, 107.0, 110.0, 112.0]
    first_id = last_id = None
    for day, eq in zip(days, equities, strict=True):
        rid = seed_tick(conn, day, eq)
        first_id = first_id if first_id is not None else rid
        last_id = rid
    res = assemble(conn)
    m = metrics_from_returns(pd.Series(equities).pct_change().dropna())
    assert res.evidence.n_return_observations == 9
    assert res.evidence.realized_sharpe == pytest.approx(m["sharpe"])
    assert res.evidence.realized_vol == pytest.approx(m["ann_volatility"])
    assert res.evidence.realized_max_drawdown == pytest.approx(abs(m["max_drawdown"]))
    assert res.evidence.session_coverage == pytest.approx(1.0)
    assert res.evidence.staleness_sessions == 0  # last tick Jun 12 == NOW's session
    assert res.first_tick_id == first_id and res.last_tick_id == last_id
    assert res.first_tick_ts == _ts(days[0]) and res.last_tick_ts == _ts(days[-1])
    assert res.account_id == "acct"
    assert set(res.excluded) == EXCLUSION_KEYS
    assert all(v == 0 for v in res.excluded.values())


def test_session_coverage_counts_gaps(conn):
    # Ticks Jun 1, 2, 4, 5 -> decision sessions May 29, Jun 1, Jun 3, Jun 4: 4 of the
    # 5 sessions in [May 29, Jun 4].
    for day, eq in [(date(2026, 6, 1), 100.0), (date(2026, 6, 2), 101.0),
                    (date(2026, 6, 4), 102.0), (date(2026, 6, 5), 103.0)]:
        seed_tick(conn, day, eq)
    res = assemble(conn)
    assert res.evidence.session_coverage == pytest.approx(4 / 5)


def test_last_tick_per_decision_session_wins(conn):
    seed_tick(conn, date(2026, 6, 10), 100.0)
    seed_tick(conn, date(2026, 6, 11), 90.0, hour=15)    # same decision session (Jun 10)...
    seed_tick(conn, date(2026, 6, 11), 105.0, hour=20)   # ...max-id row wins
    seed_tick(conn, date(2026, 6, 12), 110.0)
    res = assemble(conn)
    m = metrics_from_returns(pd.Series([100.0, 105.0, 110.0]).pct_change().dropna())
    assert res.evidence.n_return_observations == 2
    assert res.evidence.realized_sharpe == pytest.approx(m["sharpe"])
    assert res.evidence.realized_max_drawdown == pytest.approx(abs(m["max_drawdown"]))


# ---------------------------------------------------------------------------
# Integrity universe (clause 5) + kill/halt (clause 6)
# ---------------------------------------------------------------------------

def test_integrity_universe_catches_inadmissible_bad_rows_in_window(conn):
    # Reconcile-failed local-clock row BEFORE the first admissible row: outside the universe.
    raw_tick(conn, tick_ts=_ts(date(2026, 6, 9)), decision_ts=_ts(date(2026, 6, 8)),
             clock_source="local", reconcile_ok=0)
    seed_tick(conn, date(2026, 6, 10), 100.0)
    # INADMISSIBLE (local clock) reconcile-failed row INSIDE the window: must still count.
    raw_tick(conn, tick_ts=_ts(date(2026, 6, 11)), decision_ts=_ts(date(2026, 6, 10)),
             clock_source="local", reconcile_ok=0)
    # Future-tick row inside the window: defective (also inadmissible as bad_tick_ts).
    raw_tick(conn, tick_ts=_ts(date(2026, 6, 16)), decision_ts=_ts(date(2026, 6, 11)))
    seed_tick(conn, date(2026, 6, 12), 101.0)
    res = assemble(conn)
    assert res.evidence.n_reconcile_failures == 1   # only the in-window failure
    assert res.evidence.n_defective_ticks == 1
    assert res.evidence.n_return_observations == 1  # bad rows never become observations


def test_kill_halt_state_and_trip_events_in_window(conn):
    seed_tick(conn, date(2026, 6, 10), 100.0)
    seed_tick(conn, date(2026, 6, 12), 101.0)
    conn.execute("INSERT INTO kill_switches(strategy, reason, actor, created_at) "
                 "VALUES ('s', 'dd', 'system', 't')")
    conn.execute("INSERT INTO global_halt(id, reason, actor, created_at) "
                 "VALUES (1, 'r', 'human', 't')")
    rows = [("2026-06-09T00:00:00+00:00", "s"),   # before window start -> not counted
            ("2026-06-11T00:00:00+00:00", "s"),   # inside -> counted
            ("2026-06-11T00:00:00+00:00", "other")]  # other strategy -> not counted
    for ts, strat in rows:
        conn.execute("INSERT INTO audit_log(ts, actor, action, reason, strategy) "
                     "VALUES (?, 'system', 'kill_switch_trip', 'dd', ?)", (ts, strat))
    conn.commit()
    res = assemble(conn)
    assert res.evidence.kill_switch_tripped is True
    assert res.evidence.global_halt_engaged is True
    assert res.evidence.n_kill_trips_in_window == 1


# ---------------------------------------------------------------------------
# Staleness (clause 10)
# ---------------------------------------------------------------------------

def test_staleness_sessions(conn):
    seed_tick(conn, date(2026, 6, 8), 100.0)
    seed_tick(conn, date(2026, 6, 9), 101.0)
    res = assemble(conn)
    assert res.evidence.staleness_sessions == 3  # Jun 9 -> Jun 12 (Tue->Fri)


def test_nonutc_aware_now_is_normalized_to_utc_date(conn):
    # Mon Jun 15 01:00 at +05:00 is still Sun Jun 14 20:00 UTC. Keeping the +05:00 tz as-is
    # would make `.date()` read the LOCAL Monday and measure staleness from session Jun 15
    # (1 session); normalized to UTC the session is Fri Jun 12 (0 sessions).
    _three_admissible(conn)  # last admissible tick: Fri Jun 12
    local_now = datetime(2026, 6, 15, 1, 0, tzinfo=timezone(timedelta(hours=5)))
    res = assemble(conn, now=local_now)
    assert res.evidence.staleness_sessions == 0


# ---------------------------------------------------------------------------
# Single-tenant + concurrency (clauses 7-8)
# ---------------------------------------------------------------------------

def test_single_tenant_violation_same_account_sibling(conn):
    seed_tick(conn, date(2026, 6, 10), 100.0)
    seed_tick(conn, date(2026, 6, 12), 101.0)
    raw_tick(conn, name="other", strategy_id=2, tick_ts=_ts(date(2026, 6, 11)),
             decision_ts=_ts(date(2026, 6, 10)), account_id="acct",
             recorded_at=_ts(date(2026, 6, 11)))
    res = assemble(conn)
    assert res.evidence.single_tenant_ok is False
    assert res.n_concurrent_forward == 2


def test_sibling_before_window_does_not_violate_single_tenant(conn):
    seed_tick(conn, date(2026, 6, 10), 100.0)
    seed_tick(conn, date(2026, 6, 12), 101.0)
    raw_tick(conn, name="other", strategy_id=2, tick_ts=_ts(date(2026, 6, 9)),
             decision_ts=_ts(date(2026, 6, 8)), account_id="acct",
             recorded_at=_ts(date(2026, 6, 9)))  # before first admissible recorded_at
    res = assemble(conn)
    assert res.evidence.single_tenant_ok is True
    assert res.n_concurrent_forward == 1


def test_mixed_account_admissible_ticks_fail_single_tenant(conn):
    seed_tick(conn, date(2026, 6, 10), 100.0, account_id="acct-A")
    seed_tick(conn, date(2026, 6, 12), 101.0, account_id="acct-B")
    res = assemble(conn)
    assert res.evidence.single_tenant_ok is False
    assert res.account_id == "acct-B"  # latest admissible tick's account


def test_sibling_on_other_account_counts_concurrent_but_keeps_single_tenant(conn):
    seed_tick(conn, date(2026, 6, 10), 100.0)
    seed_tick(conn, date(2026, 6, 12), 101.0)
    raw_tick(conn, name="other", strategy_id=2, tick_ts=_ts(date(2026, 6, 11)),
             decision_ts=_ts(date(2026, 6, 10)), account_id="other-acct",
             clock_source="local",  # inadmissible siblings still count (family-wise error)
             recorded_at=_ts(date(2026, 6, 11)))
    res = assemble(conn)
    assert res.evidence.single_tenant_ok is True
    assert res.n_concurrent_forward == 2


# ---------------------------------------------------------------------------
# Broker activities (clause 9)
# ---------------------------------------------------------------------------

def _three_admissible(conn):
    seed_tick(conn, date(2026, 6, 10), 100.0)
    seed_tick(conn, date(2026, 6, 11), 101.0)
    seed_tick(conn, date(2026, 6, 12), 102.0)


def test_external_capital_types_constant():
    assert EXTERNAL_CAPITAL_TYPES == frozenset(
        {"CSD", "CSW", "TRANS", "JNLC", "JNLS", "ACATC", "ACATS"})


def test_activities_window_starts_one_second_before_first_admissible_tick(conn):
    # Alpaca's `after` bound is EXCLUSIVE, so the window start is widened 1s before the first
    # admissible tick instant — otherwise a deposit stamped exactly at first_tick_ts escapes.
    _three_admissible(conn)
    seen = {}

    def fetch(after, until):
        seen["after"], seen["until"] = after, until
        return []

    assemble(conn, activities=fetch)
    assert seen["after"] == "2026-06-10T19:59:59+00:00"  # _ts(Jun 10) minus 1s
    assert seen["until"] == NOW.isoformat()


def test_deposit_stamped_exactly_at_window_start_is_counted(conn):
    # The fake mirrors the broker's EXCLUSIVE `after` filtering: with an unwidened
    # `after == first_tick_ts` the boundary-instant deposit would be dropped server-side and
    # the hygiene check would PASS on contaminated capital.
    _three_admissible(conn)
    deposit = {"activity_type": "CSD", "date": _ts(date(2026, 6, 10))}  # == first tick instant

    def fetch(after, until):
        return [a for a in [deposit] if a["date"] > after]  # broker-side exclusive `after`

    res = assemble(conn, activities=fetch)
    assert res.evidence.activities_ok is True
    assert res.evidence.n_external_cash_flows == 1


def test_external_cash_flow_counts(conn):
    _three_admissible(conn)
    res = assemble(conn, activities=lambda a, u: [{"activity_type": "CSD", "id": "1"}])
    assert res.evidence.activities_ok is True
    assert res.evidence.n_external_cash_flows == 1
    assert res.evidence.n_unattributable_fills == 0


def test_dividend_interest_fee_pass(conn):
    _three_admissible(conn)
    acts = [{"activity_type": t} for t in ("DIV", "INT", "FEE")]
    res = assemble(conn, activities=lambda a, u: acts)
    assert res.evidence.activities_ok is True
    assert res.evidence.n_external_cash_flows == 0
    assert res.evidence.n_unattributable_fills == 0


def test_fill_matched_to_own_paper_venue_order_passes(conn):
    _three_admissible(conn)
    record_paper_venue_order(conn, "s", "AAA", "buy", None, "coid-1", strategy_id=1)
    backfill_paper_venue_broker_order_id(conn, "coid-1", "ord-1")
    res = assemble(
        conn, activities=lambda a, u: [{"activity_type": "FILL", "order_id": "ord-1"}])
    assert res.evidence.n_unattributable_fills == 0


def test_unmatched_fill_counts(conn):
    _three_admissible(conn)
    # An order recorded for ANOTHER strategy_id must not attribute this fill.
    record_paper_venue_order(conn, "other", "AAA", "buy", None, "coid-2", strategy_id=2)
    backfill_paper_venue_broker_order_id(conn, "coid-2", "ord-2")
    res = assemble(
        conn, activities=lambda a, u: [{"activity_type": "FILL", "order_id": "ord-2"}])
    assert res.evidence.n_unattributable_fills == 1


def test_fill_missing_order_id_counts_unattributable(conn):
    _three_admissible(conn)
    res = assemble(conn, activities=lambda a, u: [{"activity_type": "FILL"}])
    assert res.evidence.n_unattributable_fills == 1


def test_activities_fetch_failure_fails_closed(conn):
    _three_admissible(conn)

    def boom(after, until):
        raise RuntimeError("broker down")

    res = assemble(conn, activities=boom)
    assert res.evidence.activities_ok is False


# ---------------------------------------------------------------------------
# No admissible ticks at all
# ---------------------------------------------------------------------------

def test_no_admissible_ticks_skips_broker_and_zeroes_evidence(conn):
    seed_tick(conn, date(2026, 6, 11), 100.0, clock_source="local")  # inadmissible only

    def must_not_call(after, until):
        raise AssertionError("broker must not be called with no admissible window")

    res = assemble(conn, activities=must_not_call)
    ev = res.evidence
    assert ev.staleness_sessions is None
    assert ev.activities_ok is True  # would be False had the fake been called
    assert ev.n_external_cash_flows == 0 and ev.n_unattributable_fills == 0
    assert ev.n_return_observations == 0 and ev.session_coverage == 0.0
    assert ev.n_reconcile_failures == 0 and ev.n_defective_ticks == 0
    assert ev.n_kill_trips_in_window == 0
    assert ev.single_tenant_ok is True
    assert res.n_concurrent_forward == 0
    assert res.first_tick_id is None and res.last_tick_id is None
    assert res.first_tick_ts is None and res.last_tick_ts is None
    assert res.account_id is None
    assert res.excluded["local_clock"] == 1


# ---------------------------------------------------------------------------
# qualified_holdout_sharpe (clause 11)
# ---------------------------------------------------------------------------

def seed_gate_row(conn, *, passed=1, pit_ok=1, pit_override=0, code_hash="c", config_hash="g",
                  dependency_hash="d", checks=_AUTO, value=1.2):
    if checks == _AUTO:
        checks = [{"name": "holdout_sharpe", "value": value}]
    conn.execute(
        "INSERT INTO gate_evaluations(strategy_id, passed, n_funnel, own_lifetime_combos, "
        "windowed_total_combos, funnel_window_days, breadth_provenance, pit_ok, pit_override, "
        "holdout_n_bars, min_holdout_observations, code_hash, config_hash, dependency_hash, "
        "data_source, snapshot_id, period_start, period_end, holdout_frac, actor, "
        "decision_json, consumed, created_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (1, passed, 1, 1, 1, 90, "measured", pit_ok, pit_override, 100, 63, code_hash,
         config_hash, dependency_hash, "snapshot", None, "2026-01-01", "2026-06-01", 0.25,
         "agent", json.dumps({"checks": checks}), 0, "2026-06-10T00:00:00+00:00"),
    )
    conn.commit()


def test_qualified_holdout_sharpe_found(conn):
    seed_gate_row(conn, value=1.2)
    assert qualified_holdout_sharpe(conn, 1, IDENT) == pytest.approx(1.2)


def test_qualified_holdout_sharpe_newest_qualified_wins(conn):
    seed_gate_row(conn, value=1.0)
    seed_gate_row(conn, value=1.5)
    assert qualified_holdout_sharpe(conn, 1, IDENT) == pytest.approx(1.5)


def test_qualified_holdout_sharpe_ignores_failed_row(conn):
    seed_gate_row(conn, passed=0, value=1.2)
    assert qualified_holdout_sharpe(conn, 1, IDENT) is None


def test_qualified_holdout_sharpe_ignores_non_pit_row(conn):
    seed_gate_row(conn, pit_ok=0, value=1.2)
    assert qualified_holdout_sharpe(conn, 1, IDENT) is None


def test_qualified_holdout_sharpe_ignores_pit_override_row(conn):
    seed_gate_row(conn, pit_override=1, value=1.2)
    assert qualified_holdout_sharpe(conn, 1, IDENT) is None


def test_qualified_holdout_sharpe_identity_drift_is_none(conn):
    seed_gate_row(conn, code_hash="STALE", value=1.2)
    assert qualified_holdout_sharpe(conn, 1, IDENT) is None


def test_qualified_holdout_sharpe_none_dependency_hash_is_none(conn):
    seed_gate_row(conn, value=1.2)
    assert qualified_holdout_sharpe(conn, 1, ArtifactIdentity("c", "g", None)) is None


def test_qualified_holdout_sharpe_missing_check_is_none(conn):
    seed_gate_row(conn, checks=[{"name": "other_check", "value": 9.9}])
    assert qualified_holdout_sharpe(conn, 1, IDENT) is None


def test_assemble_threads_holdout_sharpe_into_evidence(conn):
    _three_admissible(conn)
    seed_gate_row(conn, value=1.4)
    res = assemble(conn)
    assert res.evidence.holdout_sharpe == pytest.approx(1.4)


# ---------------------------------------------------------------------------
# Task 10: preflight + relaxation guard + run_forward_gate orchestration
# ---------------------------------------------------------------------------

@pytest.fixture
def repo(conn):
    return SqliteStrategyRepository(conn)


def _set_stage(conn, stage: str) -> None:
    conn.execute("UPDATE strategies SET stage=? WHERE id=1", (stage,))
    conn.commit()


def test_preflight_refuses_system_actor(repo):
    with pytest.raises(ValueError, match="agent or human"):
        forward_promotion_preflight(repo, "s", actor=Actor.SYSTEM,
                                    criteria=ForwardGateCriteria())


@pytest.mark.parametrize("stage", ["candidate", "live"])
def test_preflight_refuses_wrong_stage(conn, repo, stage):
    _set_stage(conn, stage)
    with pytest.raises(TransitionError, match="paper or forward_tested"):
        forward_promotion_preflight(repo, "s", actor=Actor.AGENT,
                                    criteria=ForwardGateCriteria())


@pytest.mark.parametrize("stage", ["paper", "forward_tested"])
def test_preflight_accepts_paper_and_forward_tested(conn, repo, stage):
    _set_stage(conn, stage)
    rec = forward_promotion_preflight(repo, "s", actor=Actor.AGENT,
                                      criteria=ForwardGateCriteria())
    assert rec.stage.value == stage


@pytest.mark.parametrize(("kwargs", "fieldname"), [
    ({"sharpe_floor": 0.2}, "sharpe_floor"),
    ({"max_forward_drawdown": 0.30}, "max_forward_drawdown"),
    ({"max_staleness_sessions": 6}, "max_staleness_sessions"),
    ({"min_session_coverage": 0.8}, "min_session_coverage"),
])
def test_guard_refuses_agent_relaxation_naming_field(kwargs, fieldname):
    with pytest.raises(ValueError, match=fieldname):
        guard_forward_relaxations(Actor.AGENT, ForwardGateCriteria(**kwargs))


def test_guard_allows_agent_tightening_and_defaults():
    guard_forward_relaxations(Actor.AGENT, ForwardGateCriteria())  # pure defaults
    guard_forward_relaxations(Actor.AGENT, ForwardGateCriteria(sharpe_floor=0.5))
    guard_forward_relaxations(Actor.AGENT, ForwardGateCriteria(max_forward_drawdown=0.20))
    guard_forward_relaxations(Actor.AGENT, ForwardGateCriteria(max_staleness_sessions=3))


def test_guard_human_may_relax_anything():
    guard_forward_relaxations(Actor.HUMAN, ForwardGateCriteria(
        min_forward_observations=1, min_session_coverage=0.1, degradation_factor=0.0,
        sharpe_floor=0.0, min_forward_vol=0.0, max_forward_drawdown=0.9,
        max_staleness_sessions=99))


def _pin_identity(monkeypatch):
    """One IDENT for BOTH the orchestrator's row and the token-consume recheck (in production
    both call algua.registry.approvals.compute_artifact_hashes)."""
    monkeypatch.setattr(
        "algua.registry.forward_promotion.compute_artifact_hashes", lambda name: IDENT)
    monkeypatch.setattr("algua.registry.transitions._compute_hashes", lambda name: IDENT)


def _weekdays_ending(n: int, end: date = date(2026, 6, 12)) -> list[date]:
    out: list[date] = []
    d = end
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d -= timedelta(days=1)
    return list(reversed(out))


def _seed_passing_window(conn):
    """64 consecutive sessions of admissible ticks (63 returns >= the floor) with a gently
    rising, non-degenerate equity path, plus a qualified holdout row (bar = max(.5*1.0, .3))."""
    eq = 100.0
    for i, day in enumerate(_weekdays_ending(64)):
        seed_tick(conn, day, eq)
        eq *= 1.004 if i % 2 == 0 else 0.999
    seed_gate_row(conn, value=1.0)


def _run(repo, conn, *, actor=Actor.AGENT, criteria=None):
    return run_forward_gate(
        repo, conn, name="s", actor=actor, criteria=criteria or ForwardGateCriteria(),
        calendar=CAL, now=NOW, activities_fetch=lambda a, u: [])


def test_run_forward_gate_pass_from_paper_promotes_and_consumes(conn, repo, monkeypatch):
    _pin_identity(monkeypatch)
    _seed_passing_window(conn)
    out = _run(repo, conn)
    assert out.decision.passed is True
    assert out.promoted is True
    assert repo.get("s").stage is Stage.FORWARD_TESTED
    row = conn.execute("SELECT * FROM forward_gate_evaluations").fetchone()
    assert row["passed"] == 1
    assert row["consumed"] == 1  # born-and-spent: recorded atomically with the stage change
    assert row["actor"] == "agent"
    for col in ("realized_sharpe", "holdout_sharpe", "first_tick_id", "last_tick_id",
                "n_concurrent_forward", "account_id"):
        assert row[col] is not None, col
    assert row["account_id"] == "acct"
    assert row["n_concurrent_forward"] == 1
    assert row["first_tick_id"] == out.assembled.first_tick_id
    assert row["last_tick_id"] == out.assembled.last_tick_id
    assert json.loads(row["decision_json"]) == out.decision.to_dict()


def test_run_forward_gate_fail_records_unconsumable_row_no_transition(conn, repo, monkeypatch):
    _pin_identity(monkeypatch)
    _two_admissible(conn)  # 1 return observation: window floor fails
    seed_gate_row(conn, value=1.0)
    out = _run(repo, conn)
    assert out.decision.passed is False
    assert out.promoted is False
    assert repo.get("s").stage is Stage.PAPER
    row = conn.execute("SELECT passed, consumed FROM forward_gate_evaluations").fetchone()
    assert row["passed"] == 0 and row["consumed"] == 0
    assert repo.find_consumable_forward_gate_evaluation(
        1, "c", "g", "d", now=datetime.now(UTC).isoformat(),
        ttl_days=FORWARD_TOKEN_TTL_DAYS) is None


def test_run_forward_gate_reevaluation_at_forward_tested_records_new_row(
        conn, repo, monkeypatch):
    _pin_identity(monkeypatch)
    _seed_passing_window(conn)
    first = _run(repo, conn)
    assert first.promoted is True
    second = _run(repo, conn)  # now at forward_tested: certificate refresh, no stage change
    assert second.decision.passed is True
    assert second.promoted is False
    assert repo.get("s").stage is Stage.FORWARD_TESTED
    rows = conn.execute(
        "SELECT passed, consumed FROM forward_gate_evaluations ORDER BY id").fetchall()
    assert len(rows) == 2
    assert rows[0]["passed"] == 1 and rows[0]["consumed"] == 1  # the promotion token
    # The refresh row is a CERTIFICATE, not a token: born consumed, so a later demotion can
    # never bank it for re-entry (#124 GATE-2)...
    assert rows[1]["passed"] == 1 and rows[1]["consumed"] == 1
    assert repo.find_consumable_forward_gate_evaluation(
        1, "c", "g", "d", now=datetime.now(UTC).isoformat(),
        ttl_days=FORWARD_TOKEN_TTL_DAYS) is None
    # ...while the live wall's certificate selection (which ignores consumed) still sees it.
    latest = repo.latest_forward_gate_row(1, "c", "g", "d")
    assert latest is not None and latest["passed"] == 1 and latest["consumed"] == 1


def test_run_forward_gate_refresh_row_is_not_bankable_for_repromotion(
        conn, repo, monkeypatch):
    """The GATE-2 banking attack: pass from paper, refresh at forward_tested, demote, then try
    to re-promote by consuming the banked refresh row. Re-entry to forward_tested from below
    must always require a fresh full gate pass — the refresh row never satisfies it."""
    _pin_identity(monkeypatch)
    _seed_passing_window(conn)
    assert _run(repo, conn).promoted is True
    assert _run(repo, conn).decision.passed is True  # refresh at forward_tested
    transition_strategy(repo, "s", Stage.PAPER, Actor.AGENT, "demote")  # back-step is free
    with pytest.raises(TransitionError, match="algua paper promote"):
        transition_strategy(repo, "s", Stage.FORWARD_TESTED, Actor.AGENT, "bank the refresh")


def test_run_forward_gate_failing_reevaluation_at_forward_tested_no_demotion(
        conn, repo, monkeypatch):
    """A FAILING certificate refresh records passed=0 and demotes nothing — the stale-certificate
    consequence lives in the live wall, not here. Tightened criteria are agent-legal, so an agent
    can produce this row without any relaxation."""
    _pin_identity(monkeypatch)
    _seed_passing_window(conn)
    assert _run(repo, conn).promoted is True
    out = _run(repo, conn, criteria=ForwardGateCriteria(sharpe_floor=20.0))  # stricter: fails
    assert out.decision.passed is False
    assert out.promoted is False
    assert repo.get("s").stage is Stage.FORWARD_TESTED  # no demotion
    rows = conn.execute(
        "SELECT passed, consumed FROM forward_gate_evaluations ORDER BY id").fetchall()
    assert len(rows) == 2
    # Re-evaluation rows at forward_tested are certificates: born consumed, pass OR fail.
    assert rows[1]["passed"] == 0 and rows[1]["consumed"] == 1
    # Nothing bankable either: the promotion token is spent, the failing row never consumable.
    assert repo.find_consumable_forward_gate_evaluation(
        1, "c", "g", "d", now=datetime.now(UTC).isoformat(),
        ttl_days=FORWARD_TOKEN_TTL_DAYS) is None


def test_run_forward_gate_human_pass_from_paper(conn, repo, monkeypatch):
    _pin_identity(monkeypatch)
    _seed_passing_window(conn)
    out = _run(repo, conn, actor=Actor.HUMAN)
    assert out.promoted is True
    assert repo.get("s").stage is Stage.FORWARD_TESTED
    row = conn.execute("SELECT actor, passed, consumed FROM forward_gate_evaluations").fetchone()
    assert row["actor"] == "human"
    assert row["passed"] == 1
    # Born consumed like the agent's: the atomic record+promote spends the row at birth for
    # EVERY actor (a human row was never consumable anyway — the actor='agent' token filter).
    assert row["consumed"] == 1
