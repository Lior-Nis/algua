"""Book-exit / lane-crossing allocation shed (#497).

Every edge that takes a strategy OUT of its operating book (or crosses lanes) must revoke its
capital reservation ATOMICALLY with the stage CAS, else run-all (which iterates only the source
lane) would orphan the reservation. This file exercises the `_REVOKE_ON_EXIT` wiring in
`transitions.transition_strategy`, the SOURCE-lane flatness re-check in
`store._assert_flat_for_bench`, and the narrowed go-live guards in `store.apply_transition`.
"""

import sqlite3

import pytest

from algua.contracts.lifecycle import Actor, Stage, TransitionError
from algua.contracts.types import PendingLiveAuthorization
from algua.registry import allocations
from algua.registry.db import connect, migrate
from algua.registry.store import SqliteStrategyRepository
from algua.registry.transitions import transition_strategy


def _repo(tmp_path) -> tuple[SqliteStrategyRepository, sqlite3.Connection]:
    conn = connect(tmp_path / "reg.db")
    migrate(conn)
    return SqliteStrategyRepository(conn), conn


def _set_stage(conn: sqlite3.Connection, sid: int, stage: Stage) -> None:
    # Set the source stage DIRECTLY rather than driving the full lifecycle — we are unit-testing the
    # exit edge in isolation, not the gates that guard reaching that stage.
    with conn:
        conn.execute("UPDATE strategies SET stage=? WHERE id=?", (stage.value, sid))


def _seed_alloc(conn: sqlite3.Connection, sid: int, capital: float = 10_000.0) -> None:
    with conn:
        allocations.allocate_locked(conn, sid, capital, "human", 50_000.0)


def _seed_live_fill(conn: sqlite3.Connection, name: str) -> None:
    with conn:
        conn.execute(
            "INSERT INTO live_fills(activity_id, strategy, symbol, qty, price, fill_ts) "
            "VALUES (?,?,?,?,?,?)",
            ("live-a1", name, "AAPL", 5.0, 100.0, "2026-01-01T00:00:00Z"))


def _seed_paper_fill(conn: sqlite3.Connection, name: str) -> None:
    with conn:
        conn.execute(
            "INSERT INTO paper_venue_fills(activity_id, strategy, symbol, qty, price, fill_ts) "
            "VALUES (?,?,?,?,?,?)",
            ("paper-a1", name, "AAPL", 5.0, 100.0, "2026-01-01T00:00:00Z"))


# --- flat exits from the PAPER lane (paper-venue ledger governs flatness) -------------------------

@pytest.mark.parametrize(
    ("source", "target", "actor", "reason"),
    [
        (Stage.PAPER, Stage.DORMANT, Actor.AGENT, "bench"),
        (Stage.PAPER, Stage.RETIRED, Actor.AGENT, None),
        (Stage.PAPER, Stage.CANDIDATE, Actor.AGENT, None),
        (Stage.FORWARD_TESTED, Stage.RETIRED, Actor.AGENT, None),
    ],
)
def test_paper_lane_exit_revokes_allocation(tmp_path, source, target, actor, reason):
    repo, conn = _repo(tmp_path)
    sid = repo.add(name="s1").id
    _set_stage(conn, sid, source)
    _seed_alloc(conn, sid)
    assert allocations.active_allocation(conn, sid) is not None

    rec = transition_strategy(repo, "s1", target, actor, reason=reason)

    assert rec.stage is target
    assert allocations.active_allocation(conn, sid) is None


def test_paper_to_dormant_flat_check_uses_paper_ledger(tmp_path):
    # A paper-lane exit must consult the PAPER (paper-venue) ledger for flatness, NOT the live one.
    repo, conn = _repo(tmp_path)
    sid = repo.add(name="s1").id
    _set_stage(conn, sid, Stage.PAPER)
    _seed_alloc(conn, sid)
    _seed_paper_fill(conn, "s1")

    with pytest.raises(TransitionError, match="flat"):
        transition_strategy(repo, "s1", Stage.DORMANT, Actor.AGENT, reason="bench")

    # Atomic: neither the stage nor the allocation moved.
    assert repo.get("s1").stage is Stage.PAPER
    assert allocations.active_allocation(conn, sid) is not None


def test_paper_exit_flatness_ignores_live_ledger(tmp_path):
    # A stray LIVE fill for a paper-stage strategy must NOT block a paper-lane exit — the check is
    # scoped to the SOURCE lane's ledger.
    repo, conn = _repo(tmp_path)
    sid = repo.add(name="s1").id
    _set_stage(conn, sid, Stage.PAPER)
    _seed_alloc(conn, sid)
    _seed_live_fill(conn, "s1")  # wrong lane — must be ignored

    rec = transition_strategy(repo, "s1", Stage.DORMANT, Actor.AGENT, reason="bench")
    assert rec.stage is Stage.DORMANT
    assert allocations.active_allocation(conn, sid) is None


# --- exits from the LIVE lane (live ledger governs flatness) --------------------------------------

@pytest.mark.parametrize(
    ("target", "reason"),
    [(Stage.PAPER, None), (Stage.RETIRED, None), (Stage.DORMANT, "bench")],
)
def test_live_lane_exit_revokes_when_flat(tmp_path, target, reason):
    repo, conn = _repo(tmp_path)
    sid = repo.add(name="s1").id
    _set_stage(conn, sid, Stage.LIVE)
    _seed_alloc(conn, sid)

    rec = transition_strategy(repo, "s1", target, Actor.HUMAN, reason=reason)

    assert rec.stage is target
    assert allocations.active_allocation(conn, sid) is None


@pytest.mark.parametrize(
    ("target", "reason"),
    [(Stage.PAPER, None), (Stage.RETIRED, None), (Stage.DORMANT, "bench")],
)
def test_live_lane_exit_blocked_when_not_flat(tmp_path, target, reason):
    repo, conn = _repo(tmp_path)
    sid = repo.add(name="s1").id
    _set_stage(conn, sid, Stage.LIVE)
    _seed_alloc(conn, sid)
    _seed_live_fill(conn, "s1")

    with pytest.raises(TransitionError, match="flat"):
        transition_strategy(repo, "s1", target, Actor.HUMAN, reason=reason)

    # Atomic: the open-position strategy stayed live AND keeps its allocation.
    assert repo.get("s1").stage is Stage.LIVE
    assert allocations.active_allocation(conn, sid) is not None


# --- source-lane open-order drain (exit_guard, #497 F2/H1) ----------------------------------------

class _FakeExitGuard:
    """A stand-in ExitLaneGuard: records that cancel_and_ingest ran and reports a canned set of
    still-open order ids under the lock (what the CLI's LiveExitGuard returns from the broker)."""

    def __init__(self, open_ids: list[str], on_ingest=None) -> None:
        self._open_ids = open_ids
        self.cancel_and_ingest_calls = 0
        self._on_ingest = on_ingest

    def cancel_and_ingest(self) -> None:
        self.cancel_and_ingest_calls += 1
        if self._on_ingest is not None:
            self._on_ingest()

    def owned_open_order_ids(self) -> list[str]:
        return list(self._open_ids)


def test_exit_guard_blocks_on_residual_open_order(tmp_path):
    # A resting order the cancel failed to remove (reported under the lock) blocks the revoke+CAS,
    # even though the strategy holds no FILLED position — closing the open-order orphan gap.
    repo, conn = _repo(tmp_path)
    sid = repo.add(name="s1").id
    _set_stage(conn, sid, Stage.LIVE)
    _seed_alloc(conn, sid)
    guard = _FakeExitGuard(open_ids=["oid-1"])

    with pytest.raises(TransitionError, match="open live order"):
        transition_strategy(repo, "s1", Stage.DORMANT, Actor.HUMAN, reason="bench",
                            exit_guard=guard)

    assert guard.cancel_and_ingest_calls == 1  # the pre-lock cancel/ingest ceremony ran
    # Atomic: stage + allocation both survive the blocked exit.
    assert repo.get("s1").stage is Stage.LIVE
    assert allocations.active_allocation(conn, sid) is not None


def test_exit_guard_permits_when_drained_flat(tmp_path):
    # cancel_and_ingest ran, no open order survives -> the exit proceeds and the allocation is shed.
    repo, conn = _repo(tmp_path)
    sid = repo.add(name="s1").id
    _set_stage(conn, sid, Stage.LIVE)
    _seed_alloc(conn, sid)
    guard = _FakeExitGuard(open_ids=[])

    rec = transition_strategy(repo, "s1", Stage.RETIRED, Actor.HUMAN, exit_guard=guard)

    assert guard.cancel_and_ingest_calls == 1
    assert rec.stage is Stage.RETIRED
    assert allocations.active_allocation(conn, sid) is None


def test_exit_guard_ingest_captured_fill_blocks_via_positions(tmp_path):
    # The pre-lock cancel_and_ingest can turn a resting order into a FILLED position; the under-lock
    # believed_positions check then catches it (guard reports no still-open order). This is the fill
    # that lands during the drain — it must block the exit, not slip through.
    repo, conn = _repo(tmp_path)
    sid = repo.add(name="s1").id
    _set_stage(conn, sid, Stage.LIVE)
    _seed_alloc(conn, sid)
    guard = _FakeExitGuard(open_ids=[], on_ingest=lambda: _seed_live_fill(conn, "s1"))

    with pytest.raises(TransitionError, match="open live positions"):
        transition_strategy(repo, "s1", Stage.DORMANT, Actor.HUMAN, reason="bench",
                            exit_guard=guard)

    assert repo.get("s1").stage is Stage.LIVE
    assert allocations.active_allocation(conn, sid) is not None


def test_exit_guard_rejected_on_non_revoke_transition(tmp_path):
    # exit_guard only applies to a book-exit (revoke) edge; passing it on a non-revoke transition is
    # a caller bug the store layer fails closed on.
    repo, conn = _repo(tmp_path)
    sid = repo.add(name="s1").id
    _set_stage(conn, sid, Stage.PAPER)
    rec = repo.get("s1")

    with pytest.raises(ValueError, match="exit_guard is only valid"):
        repo.apply_transition(rec, Stage.FORWARD_TESTED, Actor.AGENT,
                              revoke_allocation=False, exit_guard=_FakeExitGuard([]))


# --- store-layer guard narrowing ------------------------------------------------------------------

def _pending() -> PendingLiveAuthorization:
    return PendingLiveAuthorization(
        nonce="n0", expires_at="2999-01-01T00:00:00+00:00", principal="lior",
        signature_b64="c2ln")


def test_go_live_permits_authorization_with_revoke(tmp_path):
    # forward_tested->live is the ONE edge that carries BOTH a live_authorization AND
    # revoke_allocation (#497). The incompatibility guard must NOT fire here; the call instead fails
    # LATER on the challenge-consume (no matching live_challenges row) — a TransitionError, not the
    # incompatibility ValueError.
    repo, conn = _repo(tmp_path)
    sid = repo.add(name="s1").id
    _set_stage(conn, sid, Stage.FORWARD_TESTED)
    rec = repo.get("s1")

    with pytest.raises(TransitionError, match="go-live challenge"):
        repo.apply_transition(
            rec, Stage.LIVE, Actor.HUMAN,
            code_hash="c", config_hash="cfg", dependency_hash="d",
            revoke_allocation=True, live_authorization=_pending())


def test_authorization_with_revoke_off_go_live_edge_raises_incompatible(tmp_path):
    # Any OTHER revoke edge carrying a live_authorization is a caller bug: a live->dormant-style rec
    # with both set must raise the incompatibility ValueError.
    repo, conn = _repo(tmp_path)
    sid = repo.add(name="s1").id
    _set_stage(conn, sid, Stage.LIVE)
    rec = repo.get("s1")

    with pytest.raises(ValueError, match="incompatible with revoke_allocation"):
        repo.apply_transition(
            rec, Stage.DORMANT, Actor.HUMAN, reason="bench",
            revoke_allocation=True, live_authorization=_pending())


def test_authorization_off_live_edge_rejected(tmp_path):
    # The acceptance predicate is narrowed to the human forward_tested->live edge: an authorization
    # on any other edge (here paper->forward_tested, no revoke) is rejected.
    repo, conn = _repo(tmp_path)
    sid = repo.add(name="s1").id
    _set_stage(conn, sid, Stage.PAPER)
    rec = repo.get("s1")

    with pytest.raises(ValueError, match="forward_tested->live"):
        repo.apply_transition(
            rec, Stage.FORWARD_TESTED, Actor.HUMAN,
            revoke_allocation=False, live_authorization=_pending())


# --- go-live end-to-end paper-slice shed (full signed ceremony) -----------------------------------

def test_go_live_sheds_paper_slice_end_to_end(tmp_path, monkeypatch):
    """Full two-step signed go-live: a strategy allocated in the paper book while forward_tested is
    SHED (allocation revoked) when it reaches live (#497)."""
    import json

    from typer.testing import CliRunner

    from algua.cli.main import app
    from tests.test_cli_registry import (
        _advance_to_forward_tested,
        _allowed_signers_file,
        _make_key,
        _sign_file,
        _stub_passing_certificate,
    )

    runner = CliRunner()
    db_path = tmp_path / "r.db"
    monkeypatch.setenv("ALGUA_DB_PATH", str(db_path))

    strategy = "cross_sectional_momentum"
    _advance_to_forward_tested(strategy)
    _stub_passing_certificate(monkeypatch)

    key, pub = _make_key(tmp_path)
    signers = _allowed_signers_file(tmp_path, "lior", pub)
    monkeypatch.setattr("algua.cli.registry_cmd.ALLOWED_SIGNERS_PATH", signers)

    # Seed a paper-book allocation while the strategy sits at forward_tested.
    seed_conn = connect(db_path)
    sid = SqliteStrategyRepository(seed_conn).get(strategy).id
    _seed_alloc(seed_conn, sid)
    assert allocations.active_allocation(seed_conn, sid) is not None
    seed_conn.close()

    # Step 1: challenge.
    out1 = json.loads(runner.invoke(
        app, ["registry", "transition", strategy, "--to", "live", "--actor", "human"]).stdout)
    assert out1["action"] == "go_live_challenge"
    challenge_file = tmp_path / "challenge.txt"
    challenge_file.write_text(out1["challenge"])
    sig_file = _sign_file(key, challenge_file)

    # Step 2: signature -> live.
    result2 = runner.invoke(app, ["registry", "transition", strategy, "--to", "live",
                                  "--actor", "human", "--signature", str(sig_file)])
    assert result2.exit_code == 0, result2.stdout
    out2 = json.loads(result2.stdout)
    assert out2["ok"] is True
    assert out2["stage"] == "live"

    # The paper slice was shed as part of go-live.
    check_conn = connect(db_path)
    assert allocations.active_allocation(check_conn, sid) is None
    check_conn.close()
