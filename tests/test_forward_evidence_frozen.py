"""Story 1.3d §4 (CAP-3): forward evidence for a FROZEN deployment counts only linked ticks.

A frozen tick is admissible only when it links a phase ``b`` ``frozen_invocations`` row of the same
deployment and snapshot whose result is a ``decision`` or a ``late_no_decision``; the filter,
``invocation_unlinked``, runs immediately after ``deployment_mismatch`` and only for a frozen
deployment. Every other filter then applies unchanged, so a linked late no-decision counts exactly
like the equivalent working-tree tick.

The v48 ``tick_snapshots_frozen_link`` trigger makes an unlinked frozen tick unwritable today, so
the unlinked shapes here are what the filter exists for: 1.3c-era rows written before the trigger,
and raw-write fabrications. They are inserted with that trigger dropped and then restored, the way
tests/test_frozen_runtime.py fabricates a corrupt descriptor.
"""
from __future__ import annotations

import uuid
from datetime import UTC, date, datetime

import pytest

from algua.execution.tick_snapshots import record_tick_snapshot
from algua.registry.db import connect, migrate
from algua.registry.db.frozen_evidence import TICK_FROZEN_LINK_TRIGGER
from algua.registry.forward_evidence import (
    _EXCLUSION_FILTERS,
    AssembledEvidence,
    assemble_forward_evidence,
)
from algua.registry.repository import ArtifactIdentity
from algua.registry.store.frozen_evidence import record_frozen_invocation
from tests._frozen_evidence_helpers import (
    IDENTITY,
    attempt,
    failed_attempt,
    record_final_invocation,
    seed_deployment,
)
from tests.test_forward_promotion import FakeCalendar, _prev_weekday, _ts

NOW = datetime(2026, 9, 30, 21, 0, tzinfo=UTC)  # a Wednesday, after the 2026-09-01 activation
IDENT = ArtifactIdentity(*IDENTITY)
SNAP = "snap-1"
NAME = "frozen_s"


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "r.db")
    migrate(c)
    return c


@pytest.fixture
def frozen(conn) -> tuple[int, int]:
    return seed_deployment(conn, NAME, source_kind="frozen")


def _pin_recorded_at(conn, day: date) -> int:
    rid = conn.execute("SELECT max(id) FROM tick_snapshots").fetchone()[0]
    conn.execute("UPDATE tick_snapshots SET recorded_at=? WHERE id=?", (_ts(day), rid))
    conn.commit()
    return rid


def linked_tick(conn, ids: tuple[int, int], day: date, equity: float, *, name: str = NAME,
                result_kind: str = "decision", decision_ts: str | None = "AUTO",
                clock_source: str = "broker") -> int:
    """A frozen tick through the REAL writer, linking a fresh successful final invocation."""
    strategy_id, deployment_id = ids
    link = record_final_invocation(
        conn, deployment_id=deployment_id, snapshot_id=SNAP, result_kind=result_kind)
    record_tick_snapshot(
        conn, name, tick_ts=_ts(day),
        decision_ts=_ts(_prev_weekday(day)) if decision_ts == "AUTO" else decision_ts,
        equity=equity, peak_equity=None, positions={}, n_submitted=0, reconcile_ok=True,
        lane="paper", strategy_id=strategy_id, code_hash=IDENT.code_hash,
        config_hash=IDENT.config_hash, dependency_hash=IDENT.dependency_hash,
        account_id="acct", cash=0.0, clock_source=clock_source, snapshot_id=SNAP,
        deployment_id=deployment_id, frozen_invocation_id=link)
    return _pin_recorded_at(conn, day)


def unlinked_tick(conn, ids: tuple[int, int], day: date, equity: float, *,
                  link: int | None = None, snapshot_id: str = SNAP, reconcile_ok: int = 1,
                  clock_source: str = "broker", code_hash: str = IDENT.code_hash) -> int:
    """A frozen tick the v48 trigger would refuse: no link (the 1.3c-era shape) or a link to an
    invocation that is not this tick's successful final one (a raw-write fabrication)."""
    strategy_id, deployment_id = ids
    conn.execute("DROP TRIGGER tick_snapshots_frozen_link")
    try:
        conn.execute(
            "INSERT INTO tick_snapshots(strategy, tick_ts, decision_ts, equity, positions,"
            " n_submitted, reconcile_ok, lane, strategy_id, code_hash, config_hash,"
            " dependency_hash, account_id, cash, clock_source, recorded_at, snapshot_id,"
            " deployment_id, frozen_invocation_id)"
            " VALUES (?,?,?,?,'{}',0,?,'paper',?,?,?,?,'acct',0.0,?,?,?,?,?)",
            (NAME, _ts(day), _ts(_prev_weekday(day)), equity, reconcile_ok, strategy_id,
             code_hash, IDENT.config_hash, IDENT.dependency_hash, clock_source, _ts(day),
             snapshot_id, deployment_id, link))
    finally:
        conn.execute(TICK_FROZEN_LINK_TRIGGER)
        conn.commit()
    return int(conn.execute("SELECT max(id) FROM tick_snapshots").fetchone()[0])


def assemble(conn, ids: tuple[int, int], *, name: str = NAME) -> AssembledEvidence:
    strategy_id, deployment_id = ids
    return assemble_forward_evidence(
        conn, strategy_id=strategy_id, name=name, deployment_id=deployment_id, identity=IDENT,
        calendar=FakeCalendar(), now=NOW, activities_fetch=lambda after, until: [])


def _two_linked(conn, ids) -> None:
    linked_tick(conn, ids, date(2026, 9, 28), 100.0)
    linked_tick(conn, ids, date(2026, 9, 30), 101.0)


# --- the filter ---------------------------------------------------------------------------------


def test_invocation_unlinked_runs_immediately_after_deployment_mismatch():
    assert _EXCLUSION_FILTERS[:2] == ("deployment_mismatch", "invocation_unlinked")
    assert _EXCLUSION_FILTERS[2:] == (
        "local_clock", "identity_drift", "legacy_null", "bad_tick_ts", "no_decision",
        "bad_decision_ts", "stale_decision")


def test_linked_frozen_ticks_are_admissible(conn, frozen):
    days = [date(2026, 9, d) for d in (23, 24, 25, 28, 29, 30)]
    for i, day in enumerate(days):
        linked_tick(conn, frozen, day, 100.0 + i)

    res = assemble(conn, frozen)

    assert res.evidence.n_return_observations == 5
    assert res.evidence.session_coverage == pytest.approx(1.0)
    assert set(res.excluded) == set(_EXCLUSION_FILTERS)
    assert all(count == 0 for count in res.excluded.values())


def test_a_1_3c_era_unlinked_frozen_tick_never_counts(conn, frozen):
    unlinked_tick(conn, frozen, date(2026, 9, 29), 250.0)  # would be a huge return if counted
    _two_linked(conn, frozen)

    res = assemble(conn, frozen)

    assert res.excluded["invocation_unlinked"] == 1
    assert sum(res.excluded.values()) == 1
    assert res.evidence.n_return_observations == 1  # the two linked sessions only
    assert res.evidence.realized_max_drawdown == pytest.approx(0.0)


def _phase_a_only(conn, deployment_id: int, **overrides) -> int:
    return record_frozen_invocation(conn, attempt(
        deployment_id=deployment_id, request_id=uuid.uuid4().hex, snapshot_id=SNAP, **overrides))


def _phase_b(conn, deployment_id: int, *, failed: bool = False, **overrides) -> int:
    rid = uuid.uuid4().hex
    phase_a = record_frozen_invocation(conn, attempt(
        deployment_id=deployment_id, request_id=rid, snapshot_id=overrides.get(
            "snapshot_id", SNAP)))
    make = failed_attempt if failed else attempt
    fields = {"deployment_id": deployment_id, "request_id": rid, "phase": "b",
              "phase_a_invocation_id": phase_a, "snapshot_id": SNAP}
    if not failed:
        fields["result_kind"] = "decision"
    fields.update(overrides)
    return record_frozen_invocation(conn, make(**fields))


@pytest.mark.parametrize("shape", [
    "phase_a", "phase_a_claiming_decision", "failed_phase_b", "risk_failure_phase_b",
    "other_snapshot", "other_deployment",
])
def test_a_tick_linking_anything_but_its_successful_final_invocation_is_unlinked(
    conn, frozen, shape,
):
    _strategy_id, deployment_id = frozen
    if shape == "other_deployment":
        _sibling_id, sibling_deployment = seed_deployment(conn, "sibling", source_kind="frozen")
        link = _phase_b(conn, sibling_deployment)
    else:
        link = {
            "phase_a": lambda: _phase_a_only(conn, deployment_id),
            # Schema-legal but never produced: only a phase b row is a FINAL invocation.
            "phase_a_claiming_decision": lambda: _phase_a_only(
                conn, deployment_id, result_kind="decision"),
            "failed_phase_b": lambda: _phase_b(conn, deployment_id, failed=True),
            "risk_failure_phase_b": lambda: _phase_b(
                conn, deployment_id, result_kind="risk_failure"),
            "other_snapshot": lambda: _phase_b(conn, deployment_id, snapshot_id="snap-2"),
        }[shape]()
    unlinked_tick(conn, frozen, date(2026, 9, 29), 250.0, link=link)
    _two_linked(conn, frozen)

    res = assemble(conn, frozen)

    assert res.excluded["invocation_unlinked"] == 1
    assert res.evidence.n_return_observations == 1


def test_a_linked_late_no_decision_counts_exactly_like_a_decision(conn, frozen):
    """M2: a late warming no-decision carries Phase A's decision timestamp, so it is admissible
    like the equivalent working-tree tick; one without a decision timestamp is ``no_decision``
    through the unchanged filter, never ``invocation_unlinked``."""
    linked_tick(conn, frozen, date(2026, 9, 28), 100.0)
    linked_tick(conn, frozen, date(2026, 9, 29), 102.0, result_kind="late_no_decision")
    linked_tick(conn, frozen, date(2026, 9, 30), 101.0, result_kind="late_no_decision",
                decision_ts=None)

    res = assemble(conn, frozen)

    assert res.evidence.n_return_observations == 1  # the decision and the dated late no-decision
    assert res.excluded["no_decision"] == 1
    assert res.excluded["invocation_unlinked"] == 0


def test_the_link_is_checked_before_every_later_filter(conn, frozen):
    """First match wins: an unlinked tick that ALSO has a local clock and drifted identity is
    counted once, under ``invocation_unlinked``."""
    unlinked_tick(conn, frozen, date(2026, 9, 29), 99.0, clock_source="local",
                  code_hash="e" * 32)
    _two_linked(conn, frozen)

    res = assemble(conn, frozen)

    assert res.excluded["invocation_unlinked"] == 1
    assert res.excluded["local_clock"] == res.excluded["identity_drift"] == 0
    assert sum(res.excluded.values()) == 1


def test_an_unlinked_tick_stays_in_the_integrity_universe(conn, frozen):
    """Exclusion only removes a return observation; the tighten-only hygiene universe still sees
    every row of the epoch, so a failed reconcile cannot hide by being unlinked."""
    _two_linked(conn, frozen)
    unlinked_tick(conn, frozen, date(2026, 9, 29), 99.0, reconcile_ok=0)

    res = assemble(conn, frozen)

    assert res.excluded["invocation_unlinked"] == 1
    assert res.evidence.n_reconcile_failures == 1


def test_working_tree_ticks_never_need_a_link(conn):
    ids = seed_deployment(conn, "wt", source_kind="working_tree")
    for day, equity in ((date(2026, 9, 28), 100.0), (date(2026, 9, 30), 101.0)):
        record_tick_snapshot(
            conn, "wt", tick_ts=_ts(day), decision_ts=_ts(_prev_weekday(day)), equity=equity,
            peak_equity=None, positions={}, n_submitted=0, reconcile_ok=True, lane="paper",
            strategy_id=ids[0], code_hash=IDENT.code_hash, config_hash=IDENT.config_hash,
            dependency_hash=IDENT.dependency_hash, account_id="acct", cash=0.0,
            clock_source="broker", snapshot_id=SNAP, deployment_id=ids[1])
        _pin_recorded_at(conn, day)

    res = assemble(conn, ids, name="wt")

    assert res.excluded["invocation_unlinked"] == 0
    assert all(count == 0 for count in res.excluded.values())
    assert res.evidence.n_return_observations == 1
