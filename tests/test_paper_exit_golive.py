"""T18 (Story 2.2 §2.2, §2.5): go-live is a paper-lane exit and drains the paper book first.

The signed two-step ceremony runs through the CLI against ``tests._exit_drain.FakePaperVenue``. The
live walls (actor, certificate, signature) run before the operator lock and before any venue call;
the challenge is consumed under the write lock only after the drain's re-check, so a refused drain
or a held lock leaves it unconsumed and the same signature completes later.
"""
from __future__ import annotations

import functools
import json
import os
import socket

from typer.testing import CliRunner

from algua.cli.main import app
from algua.contracts.lifecycle import TransitionError
from algua.execution import paper_exit_drain
from algua.execution.live_ledger import (
    backfill_paper_venue_broker_order_id,
    record_paper_venue_order,
)
from algua.operator.schedule import operator_run_lock
from algua.registry import allocations
from algua.registry.db import connect
from algua.registry.store import SqliteStrategyRepository
from tests.test_cli_registry import (
    _advance_to_forward_tested,
    _allowed_signers_file,
    _make_key,
    _sign_file,
    _stub_passing_certificate,
)

STRATEGY = "cross_sectional_momentum"
runner = CliRunner()


class GoLive:
    def __init__(self, tmp_path, monkeypatch) -> None:
        self.db = tmp_path / "r.db"
        monkeypatch.setenv("ALGUA_DB_PATH", str(self.db))
        _advance_to_forward_tested(STRATEGY)
        _stub_passing_certificate(monkeypatch)
        self.key, pub = _make_key(tmp_path)
        monkeypatch.setattr("algua.cli.registry_cmd.ALLOWED_SIGNERS_PATH",
                            _allowed_signers_file(tmp_path, "lior", pub))
        self.conn = connect(self.db)
        self.sid = SqliteStrategyRepository(self.conn).get(STRATEGY).id
        with self.conn:
            allocations.allocate_locked(self.conn, self.sid, 10_000.0, "human", 50_000.0)
        out = json.loads(runner.invoke(
            app, ["registry", "transition", STRATEGY, "--to", "live", "--actor", "human"]).stdout)
        assert out["action"] == "go_live_challenge"
        self.nonce = out["nonce"]
        challenge = tmp_path / "challenge.txt"
        challenge.write_text(out["challenge"])
        self.sig = _sign_file(self.key, challenge)

    def resting_order(self, venue, **kw) -> None:
        record_paper_venue_order(self.conn, STRATEGY, "AAPL", "buy", 100.0, "coid-1",
                                 strategy_id=self.sid)
        backfill_paper_venue_broker_order_id(self.conn, "coid-1", "boid-1")
        venue.add_order("boid-1", "coid-1", **kw)

    def complete(self, sig=None):
        return runner.invoke(app, ["registry", "transition", STRATEGY, "--to", "live",
                                   "--actor", "human", "--signature", str(sig or self.sig)])

    def stage(self) -> str:
        return SqliteStrategyRepository(self.conn).get(STRATEGY).stage.value

    def consumed(self) -> bool:
        row = self.conn.execute("SELECT consumed_at FROM live_challenges WHERE nonce=?",
                                (self.nonce,)).fetchone()
        return row["consumed_at"] is not None


def test_t18_a_failing_certificate_makes_no_venue_call(tmp_path, monkeypatch, empty_exit_venues):
    g = GoLive(tmp_path, monkeypatch)

    def refuse(*_a):
        raise TransitionError("forward certificate is stale")

    monkeypatch.setattr("algua.registry.transitions._default_forward_certificate_verifier",
                        lambda: refuse)
    result = g.complete()

    assert result.exit_code == 1
    assert "forward certificate is stale" in json.loads(result.stdout)["error"]
    assert empty_exit_venues.builds == [] and empty_exit_venues.paper.calls == []
    assert g.stage() == "forward_tested" and not g.consumed()


def test_t18_a_failing_signature_makes_no_venue_call(tmp_path, monkeypatch, empty_exit_venues):
    g = GoLive(tmp_path, monkeypatch)
    other, _pub = _make_key(tmp_path, name="intruder")  # not enrolled
    copy = tmp_path / "forged.txt"
    copy.write_text((tmp_path / "challenge.txt").read_text())
    forged = _sign_file(other, copy)

    result = g.complete(forged)

    assert result.exit_code == 1
    assert json.loads(result.stdout)["error"] == (
        "no matching human approval for this code+config+dependency")
    assert empty_exit_venues.builds == [] and empty_exit_venues.paper.calls == []
    assert g.stage() == "forward_tested" and not g.consumed()


def test_t18_a_resting_paper_order_is_cancelled_and_go_live_completes(
        tmp_path, monkeypatch, empty_exit_venues):
    g = GoLive(tmp_path, monkeypatch)
    venue = empty_exit_venues.paper
    g.resting_order(venue)

    result = g.complete()

    assert result.exit_code == 0, result.stdout
    assert json.loads(result.stdout)["stage"] == "live"
    assert ("cancel", "boid-1") in venue.calls
    assert g.consumed()
    assert allocations.active_allocation(g.conn, g.sid) is None


def test_t18_a_non_cancelable_order_refuses_and_the_same_signature_completes_later(
        tmp_path, monkeypatch, empty_exit_venues):
    monkeypatch.setattr(paper_exit_drain, "PaperExitGuard",
                        functools.partial(paper_exit_drain.PaperExitGuard, sleep=lambda _s: None))
    g = GoLive(tmp_path, monkeypatch)
    venue = empty_exit_venues.paper
    g.resting_order(venue, cancelable=False)

    refused = g.complete()

    assert refused.exit_code == 1
    payload = json.loads(refused.stdout)
    assert (payload["code"], payload["error"]) == (
        "wrong_stage", f"{STRATEGY} is not flat (1 open paper order(s) ['boid-1']); flatten "
                       "before this transition")
    assert g.stage() == "forward_tested" and not g.consumed()
    assert allocations.active_allocation(g.conn, g.sid) is not None

    venue.orders["boid-1"].status = "canceled"  # the order is gone at the venue
    done = g.complete()

    assert done.exit_code == 0, done.stdout
    assert g.stage() == "live" and g.consumed()


def test_t18_a_held_operator_lock_leaves_the_challenge_unconsumed(
        tmp_path, monkeypatch, empty_exit_venues):
    g = GoLive(tmp_path, monkeypatch)

    with operator_run_lock(tmp_path / "operator.lock", job="paper", host=socket.gethostname(),
                           pid=os.getpid()):
        refused = g.complete()

    assert refused.exit_code == 1
    assert json.loads(refused.stdout)["error"] == (
        "operator.lock is held; a paper-lane exit cannot interleave with a paper tick")
    assert empty_exit_venues.builds == []
    assert g.stage() == "forward_tested" and not g.consumed()
    assert g.complete().exit_code == 0  # once the lock is free, the same signature completes
    assert g.consumed()
