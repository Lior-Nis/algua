"""Task 11 (#124): the live wall demands a fresh forward certificate.

``verify_forward_certificate`` is the EVIDENCE precondition in front of the go-live signature:
newest forward-gate row (pass OR fail) for this strategy + current identity, fresh, with a clean
record since. Fakes everywhere (weekday calendar, injected activities fetch); only the
default-wiring tests in this file exercise ``_default_forward_certificate_verifier`` shallowly.
"""
import json
from datetime import UTC, date, datetime, timedelta

import pytest

from algua.contracts.lifecycle import Actor, Stage, TransitionError
from algua.registry.db import connect, migrate
from algua.registry.forward_promotion import verify_forward_certificate
from algua.registry.repository import ArtifactIdentity
from algua.registry.store import SqliteStrategyRepository
from algua.registry.transitions import transition_strategy
from algua.research.forward_gates import CERTIFICATE_FRESH_SESSIONS
from algua.risk import global_halt, kill_switch

# Friday 2026-06-12 (June 2026 weekdays: Jun 1-5 and Jun 8-12).
NOW = datetime(2026, 6, 12, 21, 0, tzinfo=UTC)
IDENT = ArtifactIdentity(code_hash="c", config_hash="g", dependency_hash="d")


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
NO_ACTS = lambda after, until: []  # noqa: E731


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "r.db")
    migrate(c)
    c.execute("INSERT INTO strategies(name, stage, created_at, updated_at) "
              "VALUES ('s', 'forward_tested', 't', 't')")
    c.commit()
    return c


@pytest.fixture
def repo(conn):
    return SqliteStrategyRepository(conn)


def seed_cert(repo, conn, strategy_id=1, *, passed=True, created_at="2026-06-10T20:00:00+00:00",
              last_tick_id=None, account_id="acct", code_hash="c", config_hash="g",
              dependency_hash="d") -> int:
    """Seed one forward_gate_evaluations row via the real writer, then pin created_at."""
    rid = repo.record_forward_gate_evaluation(
        strategy_id, passed=passed, n_forward_observations=80, min_forward_observations=63,
        session_coverage=0.95, realized_sharpe=1.2, holdout_sharpe=1.5, degradation_factor=0.5,
        sharpe_floor=0.3, realized_vol=0.1, min_forward_vol=0.02, realized_max_drawdown=0.1,
        max_forward_drawdown=0.25, first_tick_id=1, last_tick_id=last_tick_id,
        first_tick_ts=None, last_tick_ts=None, max_staleness_sessions=5, n_reconcile_failures=0,
        n_concurrent_forward=2, account_id=account_id, code_hash=code_hash,
        config_hash=config_hash, dependency_hash=dependency_hash, actor="agent",
        decision_json="{}", consumable=False)
    conn.execute("UPDATE forward_gate_evaluations SET created_at=? WHERE id=?",
                 (created_at, rid))
    conn.commit()
    return rid


def raw_tick(conn, *, strategy_id=1, reconcile_ok=1, lane="paper",
             tick_ts="2026-06-11T20:00:00+00:00") -> int:
    cur = conn.execute(
        "INSERT INTO tick_snapshots(strategy, tick_ts, decision_ts, equity, positions,"
        " n_submitted, reconcile_ok, lane, strategy_id)"
        " VALUES ('s', ?, NULL, 100.0, '{}', 0, ?, ?, ?)",
        (tick_ts, reconcile_ok, lane, strategy_id))
    conn.commit()
    rowid = cur.lastrowid
    assert rowid is not None
    return rowid


def verify(repo, conn, **kw):
    args = dict(name="s", strategy_id=1, identity=IDENT, calendar=CAL, now=NOW,
                activities_fetch=NO_ACTS, account_id_fetch=lambda: "acct")
    args.update(kw)
    return verify_forward_certificate(repo, conn, **args)


# ---------------------------------------------------------------------------
# verify_forward_certificate
# ---------------------------------------------------------------------------

def test_happy_path_returns_certificate_summary(repo, conn):
    rid = seed_cert(repo, conn)
    out = verify(repo, conn)
    assert out == {
        "id": rid, "created_at": "2026-06-10T20:00:00+00:00",
        "realized_sharpe": 1.2, "holdout_sharpe": 1.5,
        "n_forward_observations": 80, "n_concurrent_forward": 2,
    }


def test_no_certificate_row_fails(repo, conn):
    with pytest.raises(TransitionError, match="forward-test certificate"):
        verify(repo, conn)


def test_newest_failed_row_invalidates_older_pass(repo, conn):
    seed_cert(repo, conn, passed=True, created_at="2026-06-09T20:00:00+00:00")
    failed_id = seed_cert(repo, conn, passed=False, created_at="2026-06-10T20:00:00+00:00")
    with pytest.raises(TransitionError, match=rf"(?s)FAILED.*{failed_id}|{failed_id}.*FAILED"):
        verify(repo, conn)


def test_stale_certificate_fails_and_boundary_passes(repo, conn):
    # 2026-05-28 (Thu) -> 2026-06-12 (Fri) is 11 sessions: one past the max of 10.
    seed_cert(repo, conn, created_at="2026-05-28T20:00:00+00:00")
    with pytest.raises(TransitionError, match=f"11.*{CERTIFICATE_FRESH_SESSIONS}"):
        verify(repo, conn)
    # Exactly CERTIFICATE_FRESH_SESSIONS old (2026-05-29, Fri) still passes.
    conn.execute("DELETE FROM forward_gate_evaluations")
    seed_cert(repo, conn, created_at="2026-05-29T20:00:00+00:00")
    assert verify(repo, conn)["created_at"] == "2026-05-29T20:00:00+00:00"


def test_sibling_strategy_certificate_is_not_mine(repo, conn):
    # An identity-identical row under ANOTHER strategy_id must not satisfy MY wall.
    conn.execute("INSERT INTO strategies(name, stage, created_at, updated_at) "
                 "VALUES ('sib', 'forward_tested', 't', 't')")
    conn.commit()
    seed_cert(repo, conn, strategy_id=2)
    with pytest.raises(TransitionError, match="forward-test certificate"):
        verify(repo, conn)


def test_reconcile_failed_tick_after_certification_fails(repo, conn):
    tick_before = raw_tick(conn)
    seed_cert(repo, conn, last_tick_id=tick_before)
    raw_tick(conn, reconcile_ok=0)
    with pytest.raises(TransitionError, match="1 reconcile-failed paper tick"):
        verify(repo, conn)


def test_reconcile_failed_tick_inside_certified_window_does_not_double_count(repo, conn):
    # A bad tick AT OR BEFORE last_tick_id was already judged by the gate run itself.
    bad = raw_tick(conn, reconcile_ok=0)
    seed_cert(repo, conn, last_tick_id=bad)
    assert verify(repo, conn)["n_forward_observations"] == 80


def test_null_last_tick_id_means_every_bad_tick_counts(repo, conn):
    seed_cert(repo, conn, last_tick_id=None)
    raw_tick(conn, reconcile_ok=0)
    with pytest.raises(TransitionError, match="reconcile-failed paper tick"):
        verify(repo, conn)


def test_malformed_or_future_tick_after_certification_fails(repo, conn):
    # Same defect rule as the gate's integrity sweep: unparseable/naive ts, or a future stamp.
    seed_cert(repo, conn)
    raw_tick(conn, tick_ts="2026-06-11T20:00:00")  # tz-naive == raw-write fabrication
    raw_tick(conn, tick_ts="2026-06-13T20:00:00+00:00")  # after NOW
    with pytest.raises(TransitionError, match="2 malformed or future-stamped paper tick"):
        verify(repo, conn)


def test_malformed_tick_inside_certified_window_does_not_double_count(repo, conn):
    # A defective tick AT OR BEFORE last_tick_id was already judged by the gate run itself.
    bad = raw_tick(conn, tick_ts="garbage")
    seed_cert(repo, conn, last_tick_id=bad)
    assert verify(repo, conn)["n_forward_observations"] == 80


def test_kill_trip_audit_event_after_certification_fails(repo, conn):
    seed_cert(repo, conn, created_at="2026-06-10T20:00:00+00:00")
    conn.execute("INSERT INTO audit_log(ts, actor, action, reason, strategy)"
                 " VALUES ('2026-06-11T10:00:00+00:00', 'system', 'kill_switch_trip', 'dd', 's')")
    conn.commit()
    with pytest.raises(TransitionError, match="kill-switch trip"):
        verify(repo, conn)


def test_kill_trip_before_certification_is_fine(repo, conn):
    seed_cert(repo, conn, created_at="2026-06-10T20:00:00+00:00")
    conn.execute("INSERT INTO audit_log(ts, actor, action, reason, strategy)"
                 " VALUES ('2026-06-09T10:00:00+00:00', 'system', 'kill_switch_trip', 'dd', 's')")
    conn.commit()
    assert verify(repo, conn)["id"] > 0


def test_kill_switch_tripped_fails(repo, conn):
    seed_cert(repo, conn)
    kill_switch.trip(conn, "s", reason="x", actor="human")
    conn.execute("DELETE FROM audit_log")  # isolate: only the live switch state
    conn.commit()
    with pytest.raises(TransitionError, match="kill switch / global halt"):
        verify(repo, conn)


def test_global_halt_engaged_fails(repo, conn):
    seed_cert(repo, conn)
    global_halt.engage(conn, reason="x", actor="human")
    with pytest.raises(TransitionError, match="kill switch / global halt"):
        verify(repo, conn)


def test_external_capital_flow_since_certification_fails(repo, conn):
    seed_cert(repo, conn)
    fetch = lambda after, until: [{"activity_type": "CSD", "net_amount": "1000"}]  # noqa: E731
    with pytest.raises(TransitionError, match="1 external"):
        verify(repo, conn, activities_fetch=fetch)


def test_unattributable_fill_since_certification_fails(repo, conn):
    seed_cert(repo, conn)
    fetch = lambda after, until: [{"activity_type": "FILL", "order_id": "ghost"}]  # noqa: E731
    with pytest.raises(TransitionError, match="1 unattributable"):
        verify(repo, conn, activities_fetch=fetch)


def test_activities_fetch_window_is_created_at_to_now(repo, conn):
    seed_cert(repo, conn, created_at="2026-06-10T20:00:00+00:00")
    seen = {}

    def fetch(after, until):
        seen["window"] = (after, until)
        return []

    verify(repo, conn, activities_fetch=fetch)
    assert seen["window"] == ("2026-06-10T20:00:00+00:00", NOW.isoformat())


def test_account_drift_since_certification_fails(repo, conn):
    # Certificate earned on account A, credentials now resolve to B: deposits on A are
    # invisible to the hygiene re-check, so continuity is unverifiable — fail closed.
    seed_cert(repo, conn, account_id="acct-A")
    with pytest.raises(TransitionError, match="(?s)'acct-B'.*earned on account 'acct-A'"):
        verify(repo, conn, account_id_fetch=lambda: "acct-B")


def test_certificate_without_account_id_fails(repo, conn):
    seed_cert(repo, conn, account_id=None)
    with pytest.raises(TransitionError, match="records no account_id"):
        verify(repo, conn)


def test_account_id_fetch_failure_fails_closed(repo, conn):
    seed_cert(repo, conn)

    def fetch():
        raise RuntimeError("alpaca down")

    with pytest.raises(TransitionError, match="(?s)could not verify the broker account id.*"
                                              "failing closed"):
        verify(repo, conn, account_id_fetch=fetch)


def test_activities_fetch_failure_fails_closed(repo, conn):
    seed_cert(repo, conn)

    def fetch(after, until):
        raise RuntimeError("alpaca down")

    with pytest.raises(TransitionError, match="(?s)could not verify account activities.*"
                                              "failing closed"):
        verify(repo, conn, activities_fetch=fetch)


# ---------------------------------------------------------------------------
# transition_strategy: the wall ordering (actor -> certificate -> approval)
# ---------------------------------------------------------------------------

def _registered(repo, conn, stage="forward_tested"):
    conn.execute("UPDATE strategies SET stage=? WHERE name='s'", (stage,))
    conn.commit()
    return repo.get("s")


def test_live_with_approval_but_no_certificate_fails(repo, conn, monkeypatch):
    rec = _registered(repo, conn)
    monkeypatch.setattr("algua.registry.transitions._compute_hashes", lambda n: IDENT)
    repo.record_approval(rec.id, "c", "g", "d", "lior")
    # The REAL verify against the seeded DB (no forward_gate_evaluations row).
    verifier = lambda r, name, sid, ident: verify_forward_certificate(  # noqa: E731
        r, conn, name=name, strategy_id=sid, identity=ident, calendar=CAL, now=NOW,
        activities_fetch=NO_ACTS, account_id_fetch=lambda: "acct")
    with pytest.raises(TransitionError, match="forward-test certificate"):
        transition_strategy(repo, "s", Stage.LIVE, Actor.HUMAN,
                            forward_certificate_verifier=verifier)


def test_live_with_certificate_passes_through_to_approval(repo, conn, monkeypatch):
    rec = _registered(repo, conn)
    monkeypatch.setattr("algua.registry.transitions._compute_hashes", lambda n: IDENT)
    seed_cert(repo, conn, rec.id)
    verifier = lambda r, name, sid, ident: verify_forward_certificate(  # noqa: E731
        r, conn, name=name, strategy_id=sid, identity=ident, calendar=CAL, now=NOW,
        activities_fetch=NO_ACTS, account_id_fetch=lambda: "acct")
    # Certificate passes, so the NEXT wall (approval) is what refuses.
    with pytest.raises(TransitionError, match="no matching human approval"):
        transition_strategy(repo, "s", Stage.LIVE, Actor.HUMAN,
                            forward_certificate_verifier=verifier)
    # With the approval recorded too, the transition completes.
    repo.record_approval(rec.id, "c", "g", "d", "lior")
    out = transition_strategy(repo, "s", Stage.LIVE, Actor.HUMAN,
                              forward_certificate_verifier=verifier)
    assert out.stage is Stage.LIVE


def test_non_human_actor_fails_before_the_certificate_check(repo, conn):
    _registered(repo, conn)
    calls = []

    def verifier(r, name, sid, ident):
        calls.append(name)
        return {}

    with pytest.raises(TransitionError, match="human actor"):
        transition_strategy(repo, "s", Stage.LIVE, Actor.AGENT,
                            forward_certificate_verifier=verifier)
    assert calls == []  # the actor wall fires first; the verifier is never consulted


def test_certificate_check_runs_before_approval(repo, conn, monkeypatch):
    _registered(repo, conn)
    # Identity is computed (once) before either wall; pin it so the real loader never runs.
    monkeypatch.setattr("algua.registry.transitions._compute_hashes", lambda n: IDENT)

    # No approval exists either — the CERTIFICATE error proves its check runs first.
    def verifier(r, name, sid, ident):
        raise TransitionError("certificate says no")

    with pytest.raises(TransitionError, match="certificate says no"):
        transition_strategy(repo, "s", Stage.LIVE, Actor.HUMAN,
                            forward_certificate_verifier=verifier)


def test_live_gate_computes_identity_once_and_shares_it(repo, conn, monkeypatch):
    """The TOCTOU fix (#124 GATE-2): ONE identity computation feeds both the certificate
    verifier and the approval check — an injected verifier gets the same identity the approval
    is judged against, never a second (driftable) recompute."""
    rec = _registered(repo, conn)
    computes = []

    def _hashes(name):
        computes.append(name)
        return IDENT

    monkeypatch.setattr("algua.registry.transitions._compute_hashes", _hashes)
    repo.record_approval(rec.id, "c", "g", "d", "lior")
    seen = {}

    def verifier(r, name, sid, ident):
        seen["identity"] = ident
        return {"id": 1}

    out = transition_strategy(repo, "s", Stage.LIVE, Actor.HUMAN,
                              forward_certificate_verifier=verifier)
    assert out.stage is Stage.LIVE
    assert computes == ["s"]  # exactly one compute for the whole live gate
    assert seen["identity"] is IDENT  # the verifier judged THAT identity, not its own


# ---------------------------------------------------------------------------
# Default wiring (shallow): fail-closed refusals, no network, no real broker
# ---------------------------------------------------------------------------

def test_default_verifier_requires_sqlite_repo_or_injection():
    from algua.registry.transitions import _default_forward_certificate_verifier

    class NotSqlite:  # no `connection` attribute
        pass

    with pytest.raises(TransitionError, match="sqlite-backed repository or an injected"):
        _default_forward_certificate_verifier()(NotSqlite(), "s", 1, IDENT)


def test_default_verifier_missing_paper_creds_refuses(repo, conn, monkeypatch):
    from algua.config import settings as settings_mod
    from algua.registry.transitions import _default_forward_certificate_verifier

    stripped = settings_mod.get_settings().model_copy(
        update={"alpaca_api_key": None, "alpaca_api_secret": None})
    monkeypatch.setattr(settings_mod, "get_settings", lambda: stripped)
    with pytest.raises(TransitionError, match="(?s)account hygiene.*ALGUA_ALPACA_API_KEY"):
        _default_forward_certificate_verifier()(repo, "s", 1, IDENT)


# ---------------------------------------------------------------------------
# CLI ceremony: the challenge carries the evidence; no certificate, no challenge
# ---------------------------------------------------------------------------

CERT_SUMMARY = {"id": 7, "created_at": "2026-06-10T20:00:00+00:00", "realized_sharpe": 1.2,
                "holdout_sharpe": 1.5, "n_forward_observations": 80, "n_concurrent_forward": 2}


def _cli_to_forward_tested(runner, app, name):
    runner.invoke(app, ["registry", "add", name])
    for stage, actor in (("backtested", "agent"), ("candidate", "human"), ("paper", "agent"),
                         ("forward_tested", "human")):
        r = runner.invoke(app, ["registry", "transition", name, "--to", stage, "--actor", actor])
        assert r.exit_code == 0, r.stdout


@pytest.fixture
def cli(monkeypatch, tmp_path):
    from typer.testing import CliRunner

    from algua.cli.main import app
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "cli.db"))
    monkeypatch.setattr("algua.cli.live_cmd._live_account_equity", lambda: 100_000.0)
    return CliRunner(), app


def test_challenge_payload_carries_certificate_summary(cli, monkeypatch):
    runner, app = cli
    name = "cross_sectional_momentum"
    _cli_to_forward_tested(runner, app, name)
    assert runner.invoke(app, ["live", "allocate", name, "--capital", "1000"]).exit_code == 0
    monkeypatch.setattr(
        "algua.registry.transitions._default_forward_certificate_verifier",
        lambda: (lambda repo, n, sid, ident: dict(CERT_SUMMARY)))
    r = runner.invoke(app, ["registry", "transition", name, "--to", "live", "--actor", "human"])
    assert r.exit_code == 0, r.stdout
    out = json.loads(r.stdout)
    assert out["action"] == "go_live_challenge"
    assert out["forward_certificate"] == CERT_SUMMARY


def test_no_certificate_refuses_before_issuing_a_challenge(cli, monkeypatch):
    runner, app = cli
    name = "cross_sectional_momentum"
    _cli_to_forward_tested(runner, app, name)
    assert runner.invoke(app, ["live", "allocate", name, "--capital", "1000"]).exit_code == 0

    def raising():
        def v(repo, n, sid, ident):
            raise TransitionError("go-live requires a forward-test certificate")
        return v

    monkeypatch.setattr(
        "algua.registry.transitions._default_forward_certificate_verifier", raising)
    r = runner.invoke(app, ["registry", "transition", name, "--to", "live", "--actor", "human"])
    assert r.exit_code == 1
    out = json.loads(r.stdout)
    assert out["ok"] is False and "certificate" in out["error"]
    # No challenge row was persisted — the refusal happened BEFORE issuance.
    from contextlib import closing

    from algua.config.settings import get_settings
    with closing(connect(get_settings().db_path)) as c:
        assert c.execute("SELECT COUNT(*) FROM live_challenges").fetchone()[0] == 0


def test_store_exposes_readonly_connection(repo, conn):
    assert repo.connection is conn
