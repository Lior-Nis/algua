"""Story 2.2 (#685): the paper-lane exit drain (contract §3, §4; tests T1-T13, T19).

Every paper-source book exit runs ``PaperExitGuard`` through the default selector: sync the venue,
skip the cancel while a material position is held, cancel only the strategy's own resting orders
(each request audited before its DELETE), re-list briefly while a cancel is pending, sync again,
settle every drain-cancelled order against the ledger, and re-list under the write lock. The venue
is ``tests._exit_drain.FakePaperVenue`` (or a real ``AlpacaPaperBroker`` over a fake HTTP layer for
the broker's own refusals); nothing here reaches the network.
"""
from __future__ import annotations

import json
import sqlite3
from typing import Any

import pytest
from typer.testing import CliRunner

from algua.audit import log as audit_log
from algua.cli.main import app
from algua.contracts.lifecycle import Actor, Stage, TransitionError
from algua.execution import alpaca_broker as ab
from algua.execution import lane_exit, venue_sync
from algua.execution.alpaca_broker import AlpacaPaperBroker
from algua.execution.errors import BrokerError
from algua.execution.live_ledger import (
    LedgerKind,
    backfill_paper_venue_broker_order_id,
    believed_positions,
    record_paper_venue_order,
)
from algua.execution.paper_exit_drain import CANCEL_REQUESTED, PaperExitDrainBroker, PaperExitGuard
from algua.registry import allocations, transitions
from algua.registry.db import connect, migrate
from algua.registry.repository import ArtifactIdentity
from algua.registry.store import SqliteStrategyRepository
from algua.registry.transitions import transition_strategy
from algua.risk import kill_switch
from tests._deployment_helpers import force_legacy_strategy
from tests._exit_drain import FakePaperVenue

#: A stored paper cursor in the production (nanosecond, offset-bearing) form. Every drain must
#: leave it byte-identical (T3e).
CURSOR = "2026-03-02T14:00:00.123456789+00:00"
PAPER_EXITS = [
    (Stage.PAPER, Stage.DORMANT), (Stage.PAPER, Stage.RETIRED), (Stage.PAPER, Stage.CANDIDATE),
    (Stage.FORWARD_TESTED, Stage.RETIRED), (Stage.FORWARD_TESTED, Stage.LIVE),
]
POSITIONS = "s1 is not flat (open paper positions ['AAPL']); flatten before this transition"
UNPUBLISHED = ("s1 is not flat (the paper venue reports fills on order(s) ['boid-1'] that the "
               "paper ledger does not hold yet); retry this transition once the venue publishes "
               "them")
OPEN_ORDER = "s1 is not flat (1 open paper order(s) ['boid-1']); flatten before this transition"


class World:
    """One strategy ``s1`` at ``source`` with a paper-book allocation and a stored cursor."""

    def __init__(self, tmp_path, monkeypatch, source: Stage, cursor: str | None) -> None:
        self.db = tmp_path / "reg.db"
        self.conn = connect(self.db)
        migrate(self.conn)
        self.repo = SqliteStrategyRepository(self.conn)
        self.sid = self.repo.add(name="s1").id
        force_legacy_strategy(self.conn, self.sid, stage=source.value)
        with self.conn:
            allocations.allocate_locked(self.conn, self.sid, 10_000.0, "human", 50_000.0)
            if cursor is not None:
                self.conn.execute("INSERT INTO paper_venue_fill_cursor(name, cursor)"
                                  " VALUES ('activities', ?)", (cursor,))
        self.source = source
        # go-live's identity recompute imports a strategy module; s1 is synthetic.
        monkeypatch.setattr(transitions, "_compute_hashes",
                            lambda name: ArtifactIdentity("c", "cfg", "d"))

    def exit(self, target: Stage, **kw: Any):
        extra: dict[str, Any] = {}
        if target is Stage.LIVE:
            extra = {"approval_verifier": lambda *a: True,
                     "forward_certificate_verifier": lambda *a: {}}
        actor = Actor.HUMAN if target is Stage.LIVE else Actor.AGENT
        reason = "bench" if target is Stage.DORMANT else None
        return transition_strategy(self.repo, "s1", target, actor, reason=reason, **extra, **kw)

    def own_order(self, oid="boid-1", coid="coid-1", name="s1", sid=None, side="buy") -> None:
        record_paper_venue_order(self.conn, name, "AAPL", side, 100.0, coid,
                                 strategy_id=sid or self.sid)
        backfill_paper_venue_broker_order_id(self.conn, coid, oid)

    def fill(self, qty: float, *, strategy: str | None = "s1", boid: str | None = None,
             aid: str = "seed-1", price: float = 100.0, symbol: str = "AAPL") -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO paper_venue_fills(activity_id, broker_order_id, strategy, symbol,"
                " qty, price, fill_ts) VALUES (?,?,?,?,?,?,?)",
                (aid, boid, strategy, symbol, qty, price, "2026-03-01T15:00:00+00:00"))

    def cursor(self) -> str | None:
        row = self.conn.execute(
            "SELECT cursor FROM paper_venue_fill_cursor WHERE name='activities'").fetchone()
        return None if row is None else row["cursor"]

    def actions(self, action: str) -> list[str]:
        return [r["reason"] for r in audit_log.read(self.conn, strategy="s1", action=action)]

    def assert_unchanged(self) -> None:
        assert self.repo.get("s1").stage is self.source
        assert allocations.active_allocation(self.conn, self.sid) is not None
        assert not kill_switch.is_tripped(self.conn, "s1")
        assert not self.conn.in_transaction


@pytest.fixture
def world(tmp_path, monkeypatch):
    def make(source: Stage = Stage.PAPER, cursor: str | None = CURSOR) -> World:
        return World(tmp_path, monkeypatch, source, cursor)
    return make


@pytest.fixture
def guards(monkeypatch) -> list[PaperExitGuard]:
    """Spy on the default selector (the module attribute ``transition_strategy`` resolves)."""
    seen: list[Any] = []
    real = lane_exit.select_exit_guard

    def spy(*args):
        guard = real(*args)
        seen.append(guard)
        return guard

    monkeypatch.setattr(lane_exit, "select_exit_guard", spy)
    return seen


def _injected(world: World, venue, sleeps: list[float]):
    guard = PaperExitGuard(world.conn, venue, "s1", sleep=sleeps.append)
    return guard, (lambda repo, name, source, target: guard)


# --- T1 / T2: each paper-source edge through the default selector ---------------------------------

@pytest.mark.parametrize(("source", "target"), PAPER_EXITS)
def test_t1_exit_without_resting_order_commits(world, empty_exit_venues, source, target):
    w = world(source)
    rec = w.exit(target)
    assert rec.stage is target
    assert allocations.active_allocation(w.conn, w.sid) is None
    assert "by_coid" not in empty_exit_venues.paper.names()
    assert "list_open_orders" in empty_exit_venues.paper.names()  # the drain did run
    assert w.cursor() == CURSOR


@pytest.mark.parametrize(("source", "target"), PAPER_EXITS)
def test_t1_own_resting_order_is_cancelled_and_the_exit_commits(
        world, empty_exit_venues, source, target):
    w = world(source)
    venue = empty_exit_venues.paper
    w.own_order()
    venue.add_order("boid-1", "coid-1")
    rows_at_delete: list[list[str]] = []
    venue.on_cancel = lambda oid: rows_at_delete.append(w.actions(CANCEL_REQUESTED))

    rec = w.exit(target)

    assert ("cancel", "boid-1") in venue.calls
    assert rows_at_delete == [["boid-1"]]  # committed before its DELETE reached the venue
    assert venue.answers == [{"id": "boid-1", "client_order_id": "coid-1", "symbol": "AAPL",
                              "filled_qty": "0", "status": "canceled"}]
    assert rec.stage is target
    assert allocations.active_allocation(w.conn, w.sid) is None
    assert w.cursor() == CURSOR


def test_t2_a_siblings_resting_order_survives(world, empty_exit_venues):
    w = world()
    venue = empty_exit_venues.paper
    sib = w.repo.add(name="s2").id
    force_legacy_strategy(w.conn, sib, stage="paper")
    w.own_order("boid-2", "coid-2", name="s2", sid=sib)
    venue.add_order("boid-2", "coid-2")
    w.own_order()
    venue.add_order("boid-1", "coid-1")

    rec = w.exit(Stage.RETIRED)

    assert rec.stage is Stage.RETIRED
    assert [c for c in venue.calls if c[0] == "cancel"] == [("cancel", "boid-1")]
    assert venue.orders["boid-2"].status == "accepted"
    assert "coid-2" not in [c[1] for c in venue.calls if c[0] == "by_coid"]


# --- T3: a fill racing the cancel -----------------------------------------------------------------

def test_t3_fill_published_at_once_is_refused_on_positions(world, empty_exit_venues, guards):
    w = world()
    w.own_order()
    empty_exit_venues.paper.add_order("boid-1", "coid-1", fill_on_cancel=True, qty=2.0)

    with pytest.raises(TransitionError) as exc:
        w.exit(Stage.RETIRED)

    assert str(exc.value) == POSITIONS
    assert guards[-1].state == "drained" and guards[-1].unpublished == set()  # settle passed
    assert believed_positions(w.conn, "s1", LedgerKind.PAPER) == {"AAPL": 2.0}
    w.assert_unchanged()
    assert w.cursor() == CURSOR


def _late(w: World, venue: FakePaperVenue, qty: float) -> None:
    w.own_order()
    venue.add_order("boid-1", "coid-1", fill_on_cancel=True, withhold=True, qty=qty)


def test_t3b_late_published_fill_is_refused(world, empty_exit_venues):
    w = world()
    venue = empty_exit_venues.paper
    _late(w, venue, 2.0)

    with pytest.raises(TransitionError) as exc:
        w.exit(Stage.RETIRED)

    assert str(exc.value) == UNPUBLISHED
    assert venue.answers[-1]["filled_qty"] == "2"
    assert believed_positions(w.conn, "s1", LedgerKind.PAPER) == {}
    w.assert_unchanged()
    assert w.cursor() == CURSOR


def test_t3c_an_immediate_retry_is_still_refused(world, empty_exit_venues):
    w = world()
    venue = empty_exit_venues.paper
    _late(w, venue, 2.0)
    with pytest.raises(TransitionError):
        w.exit(Stage.RETIRED)
    venue.calls.clear()

    with pytest.raises(TransitionError) as exc:
        w.exit(Stage.RETIRED)

    assert str(exc.value) == UNPUBLISHED
    assert "cancel" not in venue.names()  # nothing owned: the order is no longer open
    assert ("by_coid", "coid-1") in venue.calls  # the durable audit set settled it again
    w.assert_unchanged()
    assert w.cursor() == CURSOR


def _first_attempt_until(venue: FakePaperVenue) -> str:
    return [c for c in venue.calls if c[0] == "activities"][-1][2]


def test_t3d_released_dust_fill_is_ingested_and_the_exit_commits(world, empty_exit_venues):
    w = world()
    venue = empty_exit_venues.paper
    _late(w, venue, 0.0005)  # above DUST_SHARES, worth $0.05 < the $1 venue minimum
    with pytest.raises(TransitionError, match="does not hold yet"):
        w.exit(Stage.RETIRED)
    until = _first_attempt_until(venue)
    venue.release()
    assert venue.feed[0]["transaction_time"] < until

    rec = w.exit(Stage.RETIRED)

    assert rec.stage is Stage.RETIRED
    assert believed_positions(w.conn, "s1", LedgerKind.PAPER) == {"AAPL": 0.0005}
    assert w.cursor() == CURSOR


def test_t3d_released_material_fill_skips_and_is_refused_on_positions(
        world, empty_exit_venues, guards):
    w = world()
    venue = empty_exit_venues.paper
    _late(w, venue, 2.0)
    with pytest.raises(TransitionError, match="does not hold yet"):
        w.exit(Stage.RETIRED)
    venue.release()
    venue.calls.clear()

    with pytest.raises(TransitionError) as exc:
        w.exit(Stage.RETIRED)

    assert str(exc.value) == POSITIONS
    assert guards[-1].state == "skipped"
    assert "by_coid" not in venue.names()
    w.assert_unchanged()
    assert w.cursor() == CURSOR


# --- T3e: the drain never moves the shared paper cursor -------------------------------------------

def test_t3e_skipped_drain_keeps_the_cursor(world, empty_exit_venues):
    w = world()
    w.fill(5.0)
    empty_exit_venues.paper.feed.append(
        {"id": "act-x", "activity_type": "FILL", "side": "buy", "qty": "1", "price": "100",
         "symbol": "MSFT", "order_id": "boid-x", "transaction_time": "2026-03-02T14:30:00+00:00"})
    with pytest.raises(TransitionError, match="open paper positions"):
        w.exit(Stage.RETIRED)
    assert w.cursor() == CURSOR
    assert w.conn.execute("SELECT COUNT(*) FROM paper_venue_fills WHERE activity_id='act-x'"
                          ).fetchone()[0] == 1  # fetched and ingested, cursor untouched


def test_t3e_absent_cursor_row_reads_far_past_after_the_drain(world, empty_exit_venues):
    w = world(cursor=None)
    w.exit(Stage.RETIRED)
    assert w.cursor() == venue_sync._PAPER_CURSOR_FAR_PAST


def test_t3e_the_operators_ingest_still_advances_the_cursor(world):
    w = world()
    venue_sync.ingest_paper_venue(w.conn, FakePaperVenue(), "2026-03-02T16:00:00+00:00")
    assert w.cursor() == "2026-03-02T16:00:00+00:00"


# --- T4: orders the cancel does not remove at once ------------------------------------------------

def test_t4_non_cancelable_order_is_refused_under_the_lock(world):
    w = world()
    venue = FakePaperVenue()
    w.own_order()
    venue.add_order("boid-1", "coid-1", cancelable=False)
    sleeps: list[float] = []
    _guard, selector = _injected(w, venue, sleeps)

    with pytest.raises(TransitionError) as exc:
        w.exit(Stage.RETIRED, exit_guard_selector=selector)

    assert str(exc.value) == OPEN_ORDER
    assert venue.names().count("list_open_orders") == 1 + 1 + 3  # owned, recheck, 3 re-lists
    assert sleeps == [1.0, 1.0, 1.0]
    w.assert_unchanged()


def test_t4b_an_order_open_at_the_recheck_but_gone_under_the_lock_is_refused(world):
    w = world()
    venue = FakePaperVenue()
    w.own_order()
    venue.add_order("boid-1", "coid-1", listed_after_cancel=4)  # recheck + 3 re-lists
    _guard, selector = _injected(w, venue, [])

    with pytest.raises(TransitionError) as exc:
        w.exit(Stage.RETIRED, exit_guard_selector=selector)

    assert str(exc.value) == OPEN_ORDER
    assert venue.list_open_orders() == []  # a fresh re-list would have shown nothing
    w.assert_unchanged()


def test_t4c_a_pending_cancel_that_clears_commits_after_one_sleep(world):
    w = world()
    venue = FakePaperVenue()
    w.own_order()
    venue.add_order("boid-1", "coid-1", listed_after_cancel=1)
    sleeps: list[float] = []
    _guard, selector = _injected(w, venue, sleeps)

    rec = w.exit(Stage.RETIRED, exit_guard_selector=selector)

    assert rec.stage is Stage.RETIRED
    assert sleeps == [1.0]


# --- T5: the skip rule ----------------------------------------------------------------------------

def test_t5_material_position_keeps_the_resting_offset(world, empty_exit_venues, guards):
    w = world()
    venue = empty_exit_venues.paper
    w.fill(5.0)
    w.own_order(side="sell")
    venue.add_order("boid-1", "coid-1", side="sell")

    with pytest.raises(TransitionError) as exc:
        w.exit(Stage.RETIRED)

    assert str(exc.value) == POSITIONS
    assert guards[-1].state == "skipped"
    assert "cancel" not in venue.names()
    assert venue.orders["boid-1"].status == "accepted"
    assert w.actions(CANCEL_REQUESTED) == []
    w.assert_unchanged()


def test_t5b_dust_residual_with_a_resting_order_drains_and_commits(world, empty_exit_venues):
    w = world()
    venue = empty_exit_venues.paper
    w.fill(0.0005)  # $0.05 at $100: dust by the Story 1.4 rule
    w.own_order()
    venue.add_order("boid-1", "coid-1")

    rec = w.exit(Stage.RETIRED)

    assert rec.stage is Stage.RETIRED
    assert ("cancel", "boid-1") in venue.calls


# --- T6: credentials absent -----------------------------------------------------------------------

UNAVAILABLE = ("cannot exit paper-lane strategy 's1': Alpaca paper credentials are not "
               "configured, so its resting paper orders cannot be drained; set "
               "ALGUA_ALPACA_API_KEY and ALGUA_ALPACA_API_SECRET, then retry")


@pytest.fixture
def no_paper_credentials(monkeypatch, tmp_path):
    for var in ("ALGUA_ALPACA_API_KEY", "ALGUA_ALPACA_API_SECRET"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("algua.operator.deployment_lock._operator_lock_path",
                        lambda: tmp_path / "operator.lock")


@pytest.mark.parametrize(("source", "target"), PAPER_EXITS)
def test_t6_missing_credentials_refuse_every_paper_exit(world, no_paper_credentials, source,
                                                        target):
    w = world(source)
    with pytest.raises(TransitionError) as exc:
        w.exit(target)
    assert str(exc.value) == UNAVAILABLE
    assert w.actions("paper_exit_drain_unavailable") == [
        "Alpaca paper credentials not configured; cannot drain resting paper orders"]
    w.assert_unchanged()
    assert w.cursor() == CURSOR


def test_t6_missing_credentials_through_the_cli_envelope(world, no_paper_credentials,
                                                         monkeypatch):
    w = world()
    monkeypatch.setenv("ALGUA_DB_PATH", str(w.db))
    result = CliRunner().invoke(app, ["registry", "transition", "s1", "--to", "dormant",
                                      "--actor", "agent", "--reason", "bench"])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert (payload["ok"], payload["code"], payload["error"]) == (False, "wrong_stage",
                                                                 UNAVAILABLE)
    w.assert_unchanged()


# --- T7: a failure at each drain step -------------------------------------------------------------

def _failed(w: World, exc: pytest.ExceptionInfo, step: str, detail: str) -> None:
    assert str(exc.value) == (
        f"cannot exit paper-lane strategy 's1': its exit drain failed at {step} ({detail}); its "
        "stage and allocation are unchanged; retry when the paper venue is reachable")
    assert w.actions("paper_exit_drain_failed") == [f"{step}: {detail}"]
    w.assert_unchanged()
    assert w.cursor() == CURSOR


_LOCAL = ("BrokerError: the paper venue clock is unusable; the exit drain refuses the "
          "local-clock fallback")


def _raising_clock(v: FakePaperVenue) -> None:
    v.fail["clock"] = BrokerError("alpaca GET /v2/clock failed after 3 attempts: 503")


def _naive_clock(v: FakePaperVenue) -> None:
    v.clock_value = "2026-03-02T15:00:00"


def _ingest_fails(v: FakePaperVenue) -> None:
    v.fail["account_activities_window"] = BrokerError("activities 500")


def _stranded_lookup_fails(v: FakePaperVenue) -> None:
    v.fail["get_order_by_client_order_id"] = BrokerError("by-coid 503")


def _open_orders_fail(v: FakePaperVenue) -> None:
    v.fail["list_open_orders"] = BrokerError("orders 500")


def _cancel_fails(v: FakePaperVenue) -> None:
    v.fail["cancel_order"] = BrokerError("alpaca 500 canceling order boid-1: boom")


def _recheck_fails(v: FakePaperVenue) -> None:
    v.fail_at["list_open_orders"] = {2: BrokerError("orders 503")}


@pytest.mark.parametrize(("arrange", "step", "detail"), [
    (_raising_clock, "clock", _LOCAL),
    (_naive_clock, "clock", _LOCAL),
    (_ingest_fails, "ingest", "BrokerError: activities 500"),
    (_open_orders_fail, "open_orders", "BrokerError: orders 500"),
    (_cancel_fails, "cancel", "BrokerError: alpaca 500 canceling order boid-1: boom"),
    (_recheck_fails, "recheck", "BrokerError: orders 503"),
])
def test_t7_a_venue_step_failure_is_audited_and_refused(world, empty_exit_venues, arrange, step,
                                                        detail):
    w = world()
    w.own_order()
    empty_exit_venues.paper.add_order("boid-1", "coid-1")
    arrange(empty_exit_venues.paper)
    with pytest.raises(BrokerError) as exc:
        w.exit(Stage.RETIRED)
    _failed(w, exc, step, detail)


def test_t7_a_stranded_row_lookup_failure_is_an_ingest_failure(world, empty_exit_venues):
    w = world()
    record_paper_venue_order(w.conn, "s2", "MSFT", "buy", 1.0, "coid-stranded", strategy_id=w.sid)
    _stranded_lookup_fails(empty_exit_venues.paper)
    with pytest.raises(BrokerError) as exc:
        w.exit(Stage.RETIRED)
    _failed(w, exc, "ingest", "BrokerError: by-coid 503")


def _payload(**fields: Any) -> dict[str, Any]:
    return {"id": "boid-1", "client_order_id": "coid-1", "symbol": "AAPL", **fields}


@pytest.mark.parametrize(("answer", "detail"), [
    (None, "ValueError: the paper venue has no order with client_order_id 'coid-1'"),
    (_payload(), "ValueError: order 'boid-1': bad filled_qty None"),
    (_payload(filled_qty="two"), "ValueError: order 'boid-1': bad filled_qty 'two'"),
    (_payload(filled_qty="-1"), "ValueError: order 'boid-1': bad filled_qty '-1'"),
    (_payload(filled_qty="nan"), "ValueError: order 'boid-1': bad filled_qty 'nan'"),
    ({**_payload(filled_qty="0"), "id": "other"},
     "ValueError: the paper venue's order for client_order_id 'coid-1' has id 'other', expected "
     "'boid-1'"),
])
def test_t7_a_settle_failure_is_audited_and_refused(world, empty_exit_venues, answer, detail):
    w = world()
    venue = empty_exit_venues.paper
    w.own_order()
    venue.add_order("boid-1", "coid-1")
    venue.by_coid_override["coid-1"] = answer
    with pytest.raises(BrokerError) as exc:
        w.exit(Stage.RETIRED)
    _failed(w, exc, "settle", detail)


def test_t7_settle_fails_on_a_requested_order_no_row_maps(world, empty_exit_venues):
    w = world()
    audit_log.append(w.conn, actor="system", action=CANCEL_REQUESTED, reason="ghost",
                     strategy="s1")
    with pytest.raises(BrokerError, match="its exit drain failed at settle"):
        w.exit(Stage.RETIRED)
    assert w.actions("paper_exit_drain_failed") == [
        "settle: ValueError: no paper order row maps broker order 'ghost' to 's1'"]
    w.assert_unchanged()


def test_t7_settle_fails_when_the_ledger_holds_more_than_the_venue(world, empty_exit_venues):
    w = world()
    w.own_order()
    empty_exit_venues.paper.add_order("boid-1", "coid-1")
    w.fill(2.0, strategy=None, boid="boid-1")  # unattributed: s1 stays flat, the order does not
    with pytest.raises(BrokerError) as exc:
        w.exit(Stage.RETIRED)
    _failed(w, exc, "settle", "ValueError: order 'boid-1': the paper ledger holds 2.0 shares but "
                               "the venue reports 0.0 filled")


def test_t7_a_step_failure_through_the_cli_envelope(world, empty_exit_venues, monkeypatch):
    w = world()
    monkeypatch.setenv("ALGUA_DB_PATH", str(w.db))
    _raising_clock(empty_exit_venues.paper)
    result = CliRunner().invoke(app, ["registry", "transition", "s1", "--to", "retired",
                                      "--actor", "agent"])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert (payload["code"], payload["retryable"]) == ("broker_error", False)
    assert "its exit drain failed at clock" in payload["error"]
    w.assert_unchanged()


# --- T7 against the real broker's own refusals (fake HTTP layer) ----------------------------------

class _Resp:
    def __init__(self, status: int, payload: Any = None) -> None:
        self.status_code, self._payload, self.text = status, payload, f"status {status}"

    def json(self) -> Any:
        return self._payload


class _AlpacaHTTP:
    """The v2 paper endpoints the drain reaches, answered from canned state."""

    def __init__(self, open_orders: list[dict], delete_status: int = 204,
                 by_coid: _Resp | None = None) -> None:
        self.open_orders, self.delete_status, self.by_coid = open_orders, delete_status, by_coid
        self.seen: list[str] = []

    def get(self, url: str, **_kw: Any) -> _Resp:
        self.seen.append(f"GET {url}")
        if url.endswith("/v2/clock"):
            return _Resp(200, {"timestamp": "2026-03-02T15:00:00.5-05:00"})
        if "/v2/account/activities" in url:
            return _Resp(200, [])
        if "/v2/orders?status=open" in url:
            return _Resp(200, self.open_orders)
        if "by_client_order_id" in url:
            return self.by_coid or _Resp(404)
        raise AssertionError(url)

    def delete(self, url: str, **_kw: Any) -> _Resp:
        self.seen.append(f"DELETE {url}")
        return _Resp(self.delete_status)

    def post(self, url: str, **_kw: Any) -> _Resp:
        raise AssertionError(f"the drain never POSTs: {url}")


def _real_broker(monkeypatch, http: _AlpacaHTTP) -> AlpacaPaperBroker:
    real = ab.call_with_backoff
    monkeypatch.setattr(ab, "call_with_backoff",
                        lambda *a, **kw: real(*a, **{**kw, "sleep": lambda _s: None}))
    monkeypatch.setattr(ab, "requests", http)
    return AlpacaPaperBroker(api_key="k", api_secret="s")


def test_the_alpaca_paper_broker_satisfies_the_drain_protocol():
    assert isinstance(AlpacaPaperBroker(api_key="k", api_secret="s"), PaperExitDrainBroker)


def test_t7_real_broker_full_open_order_page_fails_at_open_orders(
        world, empty_exit_venues, monkeypatch):
    w = world()
    page = [{"id": f"o{i}", "client_order_id": f"c{i}", "symbol": "X"} for i in range(500)]
    empty_exit_venues.paper = _real_broker(monkeypatch, _AlpacaHTTP(page))
    with pytest.raises(BrokerError) as exc:
        w.exit(Stage.RETIRED)
    _failed(w, exc, "open_orders", "BrokerError: alpaca /v2/orders: malformed, or would fill "
                                   "the page: [{'id': '")


def test_t7_real_broker_open_order_without_client_order_id_fails_at_open_orders(
        world, empty_exit_venues, monkeypatch):
    """Codex review: an open order the venue lists without a client_order_id would be dropped by the
    ownership filter as "not ours", so a resting order could survive the exit. It fails closed."""
    w = world()
    w.own_order()
    page = [{"id": "o1", "client_order_id": None, "symbol": "X"}]
    empty_exit_venues.paper = _real_broker(monkeypatch, _AlpacaHTTP(page))
    with pytest.raises(BrokerError) as exc:
        w.exit(Stage.RETIRED)
    _failed(w, exc, "open_orders", "BrokerError: alpaca /v2/orders: malformed, or would fill "
                                   "the page: [{'id': '")


def test_t7_real_broker_cancel_with_a_server_error_fails_at_cancel(
        world, empty_exit_venues, monkeypatch):
    w = world()
    w.own_order()
    http = _AlpacaHTTP([{"id": "boid-1", "client_order_id": "coid-1", "symbol": "AAPL"}],
                       delete_status=500)
    empty_exit_venues.paper = _real_broker(monkeypatch, http)
    with pytest.raises(BrokerError) as exc:
        w.exit(Stage.RETIRED)
    _failed(w, exc, "cancel", "BrokerError: alpaca 500 canceling order boid-1: status 500")
    assert not any(s.endswith("/v2/orders") for s in http.seen if s.startswith("DELETE"))


def test_t7_real_broker_by_coid_server_error_fails_at_settle(
        world, empty_exit_venues, monkeypatch):
    w = world()
    w.own_order()
    http = _AlpacaHTTP([], by_coid=_Resp(503))
    empty_exit_venues.paper = _real_broker(monkeypatch, http)
    audit_log.append(w.conn, actor="system", action=CANCEL_REQUESTED, reason="boid-1",
                     strategy="s1")  # an earlier drain asked to cancel it
    with pytest.raises(BrokerError) as exc:
        w.exit(Stage.RETIRED)
    assert str(exc.value).startswith(
        "cannot exit paper-lane strategy 's1': its exit drain failed at settle (BrokerError: "
        "alpaca 503")
    assert w.actions("paper_exit_drain_failed")[0].startswith("settle: BrokerError: alpaca 503")
    w.assert_unchanged()


# --- T8-T12: database faults, the under-lock re-list, state and transaction hygiene ---------------

def test_t8_a_database_fault_during_ingest_propagates_unaudited(world, empty_exit_venues,
                                                                monkeypatch):
    w = world()
    fault = sqlite3.OperationalError("database is locked")

    def boom(*_a: Any, **_k: Any) -> None:
        raise fault

    monkeypatch.setattr(venue_sync, "ingest_activities", boom)
    with pytest.raises(sqlite3.OperationalError) as exc:
        w.exit(Stage.RETIRED)
    assert exc.value is fault
    assert w.actions("paper_exit_drain_failed") == []
    w.assert_unchanged()


def test_t9_an_under_lock_relist_failure_rolls_back_unaudited(world, empty_exit_venues):
    w = world()
    venue = empty_exit_venues.paper
    venue.fail_at["list_open_orders"] = {2: BrokerError("orders 502")}  # 1 = drain, 2 = under lock
    rows = w.conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]

    with pytest.raises(BrokerError) as exc:
        w.exit(Stage.RETIRED)

    assert str(exc.value) == (
        "cannot exit paper-lane strategy 's1': re-listing its open paper orders under the "
        "registry lock failed (BrokerError: orders 502); the transition was rolled back; retry")
    assert w.conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0] == rows
    w.assert_unchanged()


def test_t10_a_recheck_before_the_drain_is_a_caller_bug(world):
    w = world()
    with pytest.raises(RuntimeError, match="re-check ran before its drain"):
        PaperExitGuard(w.conn, FakePaperVenue(), "s1").owned_open_order_ids()


def test_t11_a_skipped_drain_refuses_even_once_the_ledger_is_flat(world):
    w = world()
    w.fill(5.0)
    guard = PaperExitGuard(w.conn, FakePaperVenue(), "s1")
    guard.cancel_and_ingest()
    assert guard.state == "skipped"
    with w.conn:
        w.conn.execute("DELETE FROM paper_venue_fills")  # a concurrent ingest flattened it
    with pytest.raises(TransitionError) as exc:
        guard.owned_open_order_ids()
    assert str(exc.value) == ("s1 is not flat (open paper positions when the exit drain ran); "
                              "flatten before this transition")


def test_t12_transaction_hygiene(world):
    w = world()
    w.own_order()
    venue = FakePaperVenue()
    venue.add_order("boid-1", "coid-1")
    guard = PaperExitGuard(w.conn, venue, "s1")
    guard.cancel_and_ingest()
    assert not w.conn.in_transaction
    changes = w.conn.total_changes
    assert guard.owned_open_order_ids() == []
    assert w.conn.total_changes == changes


# --- T13: call order ------------------------------------------------------------------------------

def test_t13_call_order_and_broker_clock_untils(world):
    w = world()
    w.own_order()
    venue = FakePaperVenue()
    venue.add_order("boid-1", "coid-1", listed_after_cancel=1)
    guard = PaperExitGuard(w.conn, venue, "s1", sleep=lambda _s: None)
    guard.cancel_and_ingest()

    names = venue.names()
    assert names == ["clock", "activities", "list_open_orders", "cancel", "list_open_orders",
                     "list_open_orders", "clock", "activities", "by_coid"]
    clocks = [c[1] for c in venue.calls if c[0] == "clock"]
    untils = [c[2] for c in venue.calls if c[0] == "activities"]
    assert untils == clocks  # both ingests end at the venue's own clock reading
    last_observation = max(i for i, n in enumerate(names) if n == "list_open_orders")
    assert names.index("clock", 1) > last_observation
    assert names.index("by_coid") > max(i for i, n in enumerate(names) if n == "activities")


# --- T19: the skip rule is the store's positions rule ---------------------------------------------

@pytest.mark.parametrize("ledger", [
    [],                                              # none
    [("AAPL", 5e-7, 100.0)],                         # dust by shares
    [("AAPL", 0.0005, 100.0)],                       # dust by notional
    [("AAPL", 5.0, 100.0)],                          # material
    [("AAPL", 0.0005, 100.0), ("MSFT", 3.0, 300.0)],  # mixed
    [("AAPL", 0.009, 5_000.0)],                      # material at the latest price
])
def test_t19_the_drain_skips_iff_the_bench_check_refuses_on_positions(world, ledger):
    w = world()
    for i, (symbol, qty, price) in enumerate(ledger):
        w.fill(qty, aid=f"seed-{i}", price=price, symbol=symbol)
    guard = PaperExitGuard(w.conn, FakePaperVenue(), "s1")
    guard.cancel_and_ingest()
    try:
        w.repo._assert_flat_for_bench("s1", Stage.PAPER)
        refuses = False
    except TransitionError:
        refuses = True
    assert (guard.state == "skipped") is refuses
