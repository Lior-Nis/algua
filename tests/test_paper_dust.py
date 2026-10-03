"""Paper-ledger residual rules (Story 1.4): the dust rule the flatten loop and the bench flatness
check share, and the cross-tenant ledger net that caps a paper long offset.

Production 2026-10-01: notional-sized fills left `liquidity_stable_quality_momentum` believing
+3.552855 UNH and `distributed_gains_quality_momentum` believing -0.000123 UNH while the account
held their sum, 3.55273207. The full sale was refused (insufficient qty), the buy-back was refused
(under the venue's $1 minimum), and neither tenant could ever be retired."""
from __future__ import annotations

import sqlite3
from contextlib import closing

import pytest

from algua.audit.log import read as audit_read
from algua.contracts.lifecycle import Actor, Stage, TransitionError
from algua.execution import paper_reconcile
from algua.execution.dust import paper_dust, paper_ledger_net
from algua.execution.errors import BrokerError
from algua.execution.flatten import flatten_strategy
from algua.execution.live_ledger import LedgerKind, paper_believed_positions
from algua.registry import allocations
from algua.registry.db import connect, migrate
from algua.registry.store import SqliteStrategyRepository
from algua.registry.transitions import transition_strategy
from tests._deployment_helpers import force_legacy_strategy

UNH_PRICE = 500.0
_seq = iter(range(1, 10_000))


@pytest.fixture
def conn(tmp_path):
    with closing(connect(tmp_path / "d.db")) as c:
        migrate(c)
        yield c


def _fill(conn: sqlite3.Connection, strategy: str | None, symbol: str, qty: float,
          price: float = UNH_PRICE, ts: str = "2026-10-01T14:00:00Z") -> None:
    n = next(_seq)
    with conn:
        conn.execute(
            "INSERT INTO paper_venue_fills(activity_id, broker_order_id, strategy, symbol, qty,"
            " price, fill_ts) VALUES (?,?,?,?,?,?,?)",
            (f"act-{n}", f"boid-{n}", strategy, symbol, qty, price, ts))


# --- the dust rule ----------------------------------------------------------------------------

def test_a_residual_within_the_reconcile_tolerance_is_dust_even_without_a_price(conn):
    assert paper_dust(conn, "NOPRICE", 1e-16)
    assert paper_dust(conn, "NOPRICE", -1e-7)


def test_a_residual_worth_less_than_the_venue_minimum_is_dust(conn):
    _fill(conn, "a", "UNH", 1.0)
    assert paper_dust(conn, "UNH", 0.000123)       # $0.06
    assert paper_dust(conn, "UNH", -0.000123)
    assert paper_dust(conn, "UNH", 0.0019)          # $0.95


def test_a_residual_worth_the_venue_minimum_or_more_is_not_dust(conn):
    _fill(conn, "a", "UNH", 1.0)
    assert not paper_dust(conn, "UNH", 0.002)       # exactly $1
    assert not paper_dust(conn, "UNH", -3.55)


def test_without_a_recorded_price_a_material_residual_is_not_dust(conn):
    assert not paper_dust(conn, "NOPRICE", 0.000123)


def test_the_latest_recorded_fill_prices_the_residual(conn):
    _fill(conn, "a", "UNH", 1.0, price=5_000.0, ts="2026-09-01T14:00:00Z")
    _fill(conn, "b", "UNH", 1.0, price=100.0, ts="2026-10-01T14:00:00Z")
    assert paper_dust(conn, "UNH", 0.009)            # $0.90 at the latest price, $45 at the older


def test_the_ledger_net_sums_every_tenant_and_unattributed_fill(conn):
    _fill(conn, "liquidity", "UNH", 3.552855)
    _fill(conn, "gains", "UNH", -0.000123)
    _fill(conn, None, "UNH", 0.5)
    assert paper_ledger_net(conn, "UNH") == pytest.approx(4.052732)
    assert paper_ledger_net(conn, "NONE") == 0.0


# --- the flatten loop -------------------------------------------------------------------------

class _Venue:
    """Records offsets; refuses any order under the $1 minimum, as Alpaca does."""

    def __init__(self) -> None:
        self.offsets: list[tuple[str, float]] = []

    def submit_offset(self, symbol: str, qty: float, coid: str) -> str:
        if abs(qty) * UNH_PRICE < 1.0:
            raise AssertionError(f"an untradeable {symbol} offset reached the venue: {qty}")
        self.offsets.append((symbol, qty))
        return f"o-{len(self.offsets)}"


def _flatten(conn, venue, name):
    repo = SqliteStrategyRepository(conn)
    known = {r.name for r in repo.list_strategies()}
    sid = repo.get(name).id if name in known else repo.add(name=name).id
    return flatten_strategy(conn, venue, name, LedgerKind.PAPER, lane="paper", strategy_id=sid,
                            cancel=lambda: None, ingest=lambda: None)


def test_flatten_skips_an_untradeable_residual_and_offsets_the_rest(conn):
    _fill(conn, "gains", "UNH", -0.000123)
    _fill(conn, "gains", "AAPL", 2.0)
    venue = _Venue()
    res = _flatten(conn, venue, "gains")
    assert res.flatten_error is None
    assert venue.offsets == [("AAPL", 2.0)]
    assert res.n_offsets == 1


def test_flatten_sells_only_what_the_ledger_says_the_account_holds(conn):
    _fill(conn, "liquidity", "UNH", 3.552855)
    _fill(conn, "gains", "UNH", -0.000123)
    venue = _Venue()
    res = _flatten(conn, venue, "liquidity")
    assert res.flatten_error is None
    [(symbol, qty)] = venue.offsets
    assert symbol == "UNH" and qty == pytest.approx(3.552732)


def test_flatten_sells_nothing_when_the_ledger_net_is_not_long(conn):
    _fill(conn, "liquidity", "UNH", 2.0)
    _fill(conn, "other", "UNH", -2.5)
    venue = _Venue()
    res = _flatten(conn, venue, "liquidity")
    assert res.flatten_error is None and venue.offsets == []
    assert res.unsold == {"UNH": 2.0}                # reported, never hidden


def test_flatten_reports_a_material_unsold_remainder_but_not_dust(conn):
    _fill(conn, "liquidity", "UNH", 3.552855)
    _fill(conn, "gains", "UNH", -0.000123)            # $0.06 short: the remainder is dust
    _fill(conn, "big", "KO", 5.0, price=60.0)
    _fill(conn, "other", "KO", -2.0, price=60.0)      # account holds 3: 2 KO ($120) unsold
    res = _flatten(conn, _Venue(), "liquidity")
    assert res.unsold == {}
    res = _flatten(conn, _Venue(), "big")
    assert res.unsold == {"KO": pytest.approx(2.0)}


class _RefusingVenue(_Venue):
    def __init__(self, refuse: str) -> None:
        super().__init__()
        self.refuse = refuse

    def submit_offset(self, symbol: str, qty: float, coid: str) -> str:
        if symbol == self.refuse:
            raise BrokerError("alpaca 403 on /v2/orders (offset): insufficient qty")
        return super().submit_offset(symbol, qty, coid)


def test_a_venue_refusal_of_one_symbol_does_not_strand_the_rest(conn):
    for symbol in ("AAA", "BBB", "CCC"):
        _fill(conn, "s", symbol, 2.0)
    venue = _RefusingVenue("BBB")
    res = _flatten(conn, venue, "s")
    assert sorted(s for s, _ in venue.offsets) == ["AAA", "CCC"]
    assert res.n_offsets == 2
    assert res.flatten_error is not None and res.flatten_error.startswith("BBB: alpaca 403")
    [row] = [r for r in audit_read(conn) if r["action"] == "flatten_failed"]
    assert "BBB" in row["reason"]


def test_a_systemic_error_still_stops_the_loop(conn):
    for symbol in ("AAA", "BBB"):
        _fill(conn, "s", symbol, 2.0)

    class _Locked(_Venue):
        def submit_offset(self, symbol, qty, coid):
            raise sqlite3.OperationalError("database is locked")

    res = _flatten(conn, _Locked(), "s")
    assert res.n_offsets == 0 and res.flatten_error == "database is locked"


def test_flatten_buys_back_a_material_short_in_full(conn):
    _fill(conn, "short", "UNH", -2.0)
    _fill(conn, "long", "UNH", 5.0)
    venue = _Venue()
    _flatten(conn, venue, "short")
    assert venue.offsets == [("UNH", -2.0)]


def test_flatten_leaves_the_uncapped_long_untouched_when_the_ledger_covers_it(conn):
    _fill(conn, "a", "UNH", 3.0)
    _fill(conn, "b", "UNH", 1.0)
    venue = _Venue()
    _flatten(conn, venue, "a")
    assert venue.offsets == [("UNH", 3.0)]


def test_the_production_cohort_residual_flattens_and_retires(conn):
    """The 2026-10-01 UNH pair end to end: one capped sale, then both tenants are flat enough to
    retire, and their remaining beliefs still sum to what the account holds (zero)."""
    _fill(conn, "liquidity", "UNH", 3.552855)
    _fill(conn, "gains", "UNH", -0.000123)
    venue = _Venue()
    _flatten(conn, venue, "liquidity")
    [(_, sold)] = venue.offsets
    _fill(conn, "liquidity", "UNH", -sold)          # the attributed fill of the capped sale
    _flatten(conn, venue, "gains")
    assert len(venue.offsets) == 1                   # the buy-back is dust, never sent
    assert paper_ledger_net(conn, "UNH") == pytest.approx(0.0, abs=1e-9)
    repo = SqliteStrategyRepository(conn)
    for name in ("liquidity", "gains"):
        force_legacy_strategy(conn, repo.get(name).id, stage=Stage.PAPER.value)
        assert transition_strategy(repo, name, Stage.RETIRED, Actor.AGENT).stage is Stage.RETIRED
    assert not [r for r in audit_read(conn) if r["action"] == "flatten_failed"]


# --- the bench flatness check -----------------------------------------------------------------

def _paper_tenant(conn, name: str) -> int:
    repo = SqliteStrategyRepository(conn)
    sid = repo.add(name=name).id
    force_legacy_strategy(conn, sid, stage=Stage.PAPER.value)
    with conn:
        allocations.allocate_locked(conn, sid, 10_000.0, "human", 50_000.0)
    return sid


def test_a_paper_tenant_holding_only_dust_retires(conn):
    sid = _paper_tenant(conn, "s1")
    _fill(conn, "s1", "UNH", 1.0)
    _fill(conn, "s1", "UNH", -0.999877)              # +0.000123 left, $0.06
    _fill(conn, "s1", "KO", 1e-16)                   # float residual
    assert paper_believed_positions(conn, "s1")      # the raw ledger is not exactly zero
    repo = SqliteStrategyRepository(conn)
    assert transition_strategy(repo, "s1", Stage.RETIRED, Actor.AGENT).stage is Stage.RETIRED
    assert allocations.active_allocation(conn, sid) is None


def test_a_paper_tenant_holding_a_material_position_cannot_retire(conn):
    sid = _paper_tenant(conn, "s1")
    _fill(conn, "s1", "UNH", 0.002)                  # $1.00 — tradeable
    _fill(conn, "s1", "KO", 1e-16)
    repo = SqliteStrategyRepository(conn)
    with pytest.raises(TransitionError, match=r"not flat .*\['UNH'\]"):
        transition_strategy(repo, "s1", Stage.RETIRED, Actor.AGENT)
    assert repo.get("s1").stage is Stage.PAPER
    assert allocations.active_allocation(conn, sid) is not None


def test_the_live_lane_bench_check_keeps_its_exact_rule(conn):
    repo = SqliteStrategyRepository(conn)
    sid = repo.add(name="s1").id
    force_legacy_strategy(conn, sid, stage=Stage.LIVE.value)
    _fill(conn, "other", "UNH", 1.0)                 # a paper price exists for UNH
    with conn:
        conn.execute(
            "INSERT INTO live_fills(activity_id, strategy, symbol, qty, price, fill_ts) "
            "VALUES ('l1','s1','UNH',0.000123,500.0,'2026-10-01T14:00:00Z')")
    with pytest.raises(TransitionError, match="not flat"):
        transition_strategy(repo, "s1", Stage.DORMANT, Actor.AGENT, reason="bench")


# --- the paper account reconcile --------------------------------------------------------------

def _stage(conn, name: str, stage: Stage) -> None:
    repo = SqliteStrategyRepository(conn)
    known = {r.name for r in repo.list_strategies()}
    sid = repo.get(name).id if name in known else repo.add(name=name).id
    force_legacy_strategy(conn, sid, stage=stage.value)


def test_a_retired_tenants_dust_still_explains_its_half_of_the_account(conn):
    """The review's halting sequence: retire the dust-short tenant first while its long sibling is
    still on the lane. The broker holds the pair's sum, so the account must stay clean."""
    _fill(conn, "liquidity", "UNH", 3.552855)
    _fill(conn, "gains", "UNH", -0.000123)
    _stage(conn, "liquidity", Stage.PAPER)
    _stage(conn, "gains", Stage.RETIRED)
    for _ in range(4):
        r = paper_reconcile.reconcile(conn, {"UNH": 3.55273207}, paper_reconcile.next_cycle(conn))
        assert r.clean and not r.halt


def test_a_retired_tenants_material_belief_never_explains_a_position(conn):
    _fill(conn, "pap", "UNH", 1.0)
    _fill(conn, "gone", "UNH", 2.0)                    # $1,000 left on a retired strategy
    _stage(conn, "pap", Stage.PAPER)
    _stage(conn, "gone", Stage.RETIRED)
    assert paper_reconcile.attributed_paper_net(conn) == {"UNH": pytest.approx(1.0)}
    r = paper_reconcile.reconcile(conn, {"UNH": 3.0}, paper_reconcile.next_cycle(conn))
    assert not r.clean


def test_the_cohort_exit_leaves_a_clean_empty_account(conn):
    _fill(conn, "liquidity", "UNH", 3.552855)
    _fill(conn, "gains", "UNH", -0.000123)
    _fill(conn, "liquidity", "UNH", -3.55273207)       # the capped sale's fill
    _stage(conn, "liquidity", Stage.RETIRED)
    _stage(conn, "gains", Stage.RETIRED)
    r = paper_reconcile.reconcile(conn, {}, paper_reconcile.next_cycle(conn))
    assert r.clean and not r.halt
