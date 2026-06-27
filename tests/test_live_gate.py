import subprocess
from datetime import UTC, datetime, timedelta

from algua.registry import live_gate
from algua.registry.db import connect, migrate


def _conn(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    conn.execute("INSERT INTO strategies(id, name, stage, created_at, updated_at) "
                 "VALUES (1, 's', 'paper', '2026-01-01', '2026-01-01')")
    conn.commit()
    return conn


def test_build_challenge_is_deterministic():
    exp = "2026-06-05T00:10:00+00:00"
    a = live_gate.build_challenge("s", 1, "ch", "cfg", "dep", "nonce123", exp)
    b = live_gate.build_challenge("s", 1, "ch", "cfg", "dep", "nonce123", exp)
    assert a == b and "nonce=nonce123" in a and "dependency_hash=dep" in a
    assert a.startswith("algua-go-live")


def test_issue_then_find_then_consume(tmp_path):
    conn = _conn(tmp_path)
    now = datetime(2026, 6, 5, tzinfo=UTC)
    issued = live_gate.issue_challenge(conn, 1, "s", "ch", "cfg", "dep", now=now)
    assert "nonce" in issued and "challenge" in issued and "expires_at" in issued
    row = live_gate.find_pending_challenge(conn, 1, "ch", "cfg", "dep", now=now)
    assert row is not None and row["nonce"] == issued["nonce"]
    assert live_gate.consume_challenge(conn, issued["nonce"], now=now) is True
    assert live_gate.consume_challenge(conn, issued["nonce"], now=now) is False  # single-use
    assert live_gate.find_pending_challenge(conn, 1, "ch", "cfg", "dep", now=now) is None  # used


def test_find_pending_rejects_expired_and_wrong_hash(tmp_path):
    conn = _conn(tmp_path)
    now = datetime(2026, 6, 5, tzinfo=UTC)
    live_gate.issue_challenge(conn, 1, "s", "ch", "cfg", "dep", now=now)
    later = now + timedelta(hours=1)
    assert live_gate.find_pending_challenge(conn, 1, "ch", "cfg", "dep", now=later) is None  # exp
    assert live_gate.find_pending_challenge(conn, 1, "DIFFERENT", "cfg", "dep", now=now) is None
    assert live_gate.find_pending_challenge(conn, 1, "ch", "cfg", "OTHER", now=now) is None  # dep


def _make_key(tmp_path, name="id"):
    key = tmp_path / name
    subprocess.run(["ssh-keygen", "-t", "ed25519", "-N", "", "-C", "test", "-f", str(key)],
                   check=True, capture_output=True)
    return key, (tmp_path / f"{name}.pub").read_text().strip()


def _allowed_signers(tmp_path, principal, pub):
    keytype, keyblob = pub.split()[0], pub.split()[1]
    path = tmp_path / "allowed_signers"
    path.write_text(f'{principal} namespaces="algua-go-live" {keytype} {keyblob}\n')
    return path


def _sign(key, payload, tmp_path):
    data = tmp_path / "payload.txt"
    data.write_text(payload)
    subprocess.run(["ssh-keygen", "-Y", "sign", "-n", "algua-go-live", "-f", str(key), str(data)],
                   check=True, capture_output=True)
    return (tmp_path / "payload.txt.sig").read_bytes()


def test_verify_signature_happy_path(tmp_path):
    key, pub = _make_key(tmp_path)
    signers = _allowed_signers(tmp_path, "lior", pub)
    payload = "algua-go-live\nstrategy=s\nnonce=abc"
    sig = _sign(key, payload, tmp_path)
    assert live_gate.verify_signature(signers, payload, sig) == "lior"


def test_verify_signature_rejects_unenrolled_key(tmp_path):
    key, _pub = _make_key(tmp_path, "mine")
    _other_key, other_pub = _make_key(tmp_path, "other")
    signers = _allowed_signers(tmp_path, "lior", other_pub)  # enroll a DIFFERENT key
    payload = "algua-go-live\nx=1"
    sig = _sign(key, payload, tmp_path)
    assert live_gate.verify_signature(signers, payload, sig) is None


def test_verify_signature_rejects_tampered_payload(tmp_path):
    key, pub = _make_key(tmp_path)
    signers = _allowed_signers(tmp_path, "lior", pub)
    sig = _sign(key, "original-payload", tmp_path)
    assert live_gate.verify_signature(signers, "DIFFERENT-payload", sig) is None


def test_verify_signature_missing_anchor_raises(tmp_path):
    import pytest

    from algua.registry.live_gate import SignatureError
    with pytest.raises(SignatureError):
        live_gate.verify_signature(tmp_path / "nope", "payload", b"sig")


def _record_authorization(conn, pending, *, code="ch", cfg="cfg", dep="dep",
                          now=datetime(2026, 6, 5, tzinfo=UTC)):
    """Mimic the consume + live_authorizations insert that apply_transition now performs atomically
    with the stage CAS (#254), so the trade-time tests can seed an authorization from a
    verify_pending result without driving a full transition."""
    conn.execute("UPDATE live_challenges SET consumed_at=? WHERE nonce=? AND consumed_at IS NULL",
                 (now.isoformat(), pending.nonce))
    conn.execute(
        "INSERT INTO live_authorizations(strategy_id, code_hash, config_hash, dependency_hash, "
        "nonce, expires_at, signature, principal, authorized_at) VALUES (1,?,?,?,?,?,?,?,?)",
        (code, cfg, dep, pending.nonce, pending.expires_at, pending.signature_b64,
         pending.principal, now.isoformat()),
    )
    conn.commit()


def test_verify_pending_then_record_writes_live_authorization(tmp_path):
    import base64

    conn = _conn(tmp_path)
    now = datetime(2026, 6, 5, tzinfo=UTC)
    issued = live_gate.issue_challenge(conn, 1, "s", "ch", "cfg", "dep", now=now)
    key, pub = _make_key(tmp_path)
    signers = _allowed_signers(tmp_path, "lior", pub)
    sig = _sign(key, issued["challenge"], tmp_path)
    pending = live_gate.verify_pending(conn, "s", 1, "ch", "cfg", "dep", sig, signers, now=now)
    assert pending is not None and pending.principal == "lior"
    # verify_pending performs NO writes (#254): nothing is consumed/recorded until apply_transition.
    assert conn.execute("SELECT COUNT(*) FROM live_authorizations").fetchone()[0] == 0
    assert live_gate.find_pending_challenge(conn, 1, "ch", "cfg", "dep", now=now) is not None
    _record_authorization(conn, pending, now=now)
    row = conn.execute("SELECT * FROM live_authorizations WHERE strategy_id=1").fetchone()
    assert row is not None
    assert row["code_hash"] == "ch" and row["config_hash"] == "cfg"
    assert row["dependency_hash"] == "dep" and row["principal"] == "lior"
    # nonce+expires_at are stored (so the payload can be REBUILT), not the challenge text
    assert row["nonce"] == issued["nonce"] and row["expires_at"] == issued["expires_at"]
    rebuilt = live_gate.build_challenge("s", 1, "ch", "cfg", "dep", row["nonce"], row["expires_at"])
    assert rebuilt == issued["challenge"]
    sig_bytes = base64.b64decode(row["signature"])
    assert live_gate.verify_signature(signers, rebuilt, sig_bytes) == "lior"


def test_verify_pending_returns_none_on_bad_signature(tmp_path):
    conn = _conn(tmp_path)
    now = datetime(2026, 6, 5, tzinfo=UTC)
    issued = live_gate.issue_challenge(conn, 1, "s", "ch", "cfg", "dep", now=now)
    key, _pub = _make_key(tmp_path, "mine")
    _ok, other_pub = _make_key(tmp_path, "other")
    signers = _allowed_signers(tmp_path, "lior", other_pub)  # different key enrolled
    sig = _sign(key, issued["challenge"], tmp_path)
    assert live_gate.verify_pending(
        conn, "s", 1, "ch", "cfg", "dep", sig, signers, now=now
    ) is None
    assert conn.execute("SELECT COUNT(*) FROM live_authorizations").fetchone()[0] == 0


def test_apply_transition_rolls_back_authorization_when_cas_fails(tmp_path):
    """#254: the challenge consume + authorization insert are atomic with the stage CAS. If the
    CAS misses (a concurrent stage change), the whole transition rolls back — the nonce is NOT
    burned and no orphan authorization row remains."""
    import dataclasses

    import pytest

    from algua.contracts.lifecycle import Actor, Stage, TransitionError
    from algua.registry.store import SqliteStrategyRepository

    conn = _conn(tmp_path)  # strategy id=1, stage 'paper'
    repo = SqliteStrategyRepository(conn)
    now = datetime(2026, 6, 5, tzinfo=UTC)
    issued = live_gate.issue_challenge(conn, 1, "s", "ch", "cfg", "dep", now=now)
    key, pub = _make_key(tmp_path)
    signers = _allowed_signers(tmp_path, "lior", pub)
    sig = _sign(key, issued["challenge"], tmp_path)
    pending = live_gate.verify_pending(conn, "s", 1, "ch", "cfg", "dep", sig, signers, now=now)
    assert pending is not None
    # Hand apply_transition a STALE rec (claims FORWARD_TESTED) so the CAS WHERE stage=... matches
    # 0 rows (the DB row is 'paper') and the transaction raises + rolls back.
    stale = dataclasses.replace(repo.get("s"), stage=Stage.FORWARD_TESTED)
    with pytest.raises(TransitionError):
        repo.apply_transition(stale, Stage.LIVE, Actor.HUMAN, "go",
                              code_hash="ch", config_hash="cfg", dependency_hash="dep",
                              live_authorization=pending)
    # Rolled back: the challenge is still consumable and no authorization row was written.
    assert live_gate.find_pending_challenge(conn, 1, "ch", "cfg", "dep", now=now) is not None
    assert conn.execute("SELECT COUNT(*) FROM live_authorizations").fetchone()[0] == 0


def test_apply_transition_consume_rechecks_identity(tmp_path):
    """#254: the atomic consume re-asserts the full identity predicate, so a pending authorization
    can't be applied against a drifted identity (consume matches 0 rows -> rollback)."""
    import dataclasses

    import pytest

    from algua.contracts.lifecycle import Actor, Stage, TransitionError
    from algua.registry.store import SqliteStrategyRepository

    conn = _conn(tmp_path)
    repo = SqliteStrategyRepository(conn)
    now = datetime(2026, 6, 5, tzinfo=UTC)
    issued = live_gate.issue_challenge(conn, 1, "s", "ch", "cfg", "dep", now=now)
    key, pub = _make_key(tmp_path)
    signers = _allowed_signers(tmp_path, "lior", pub)
    sig = _sign(key, issued["challenge"], tmp_path)
    pending = live_gate.verify_pending(conn, "s", 1, "ch", "cfg", "dep", sig, signers, now=now)
    assert pending is not None
    rec = dataclasses.replace(repo.get("s"), stage=Stage.FORWARD_TESTED)
    # The DB stage is 'paper' (so the CAS would also miss), but the consume's identity recheck
    # fires FIRST: code_hash 'DRIFTED' != the challenge's 'ch' -> 0 rows -> rollback.
    with pytest.raises(TransitionError):
        repo.apply_transition(rec, Stage.LIVE, Actor.HUMAN, "go",
                              code_hash="DRIFTED", config_hash="cfg", dependency_hash="dep",
                              live_authorization=pending)
    assert live_gate.find_pending_challenge(conn, 1, "ch", "cfg", "dep", now=now) is not None
    assert conn.execute("SELECT COUNT(*) FROM live_authorizations").fetchone()[0] == 0


def test_apply_transition_rejects_live_authorization_on_non_live_edge(tmp_path):
    """#254 defense in depth: the repo refuses a live_authorization on any edge that isn't a
    human transition to LIVE, so the invariant doesn't rest on transition_strategy alone."""
    import pytest

    from algua.contracts.lifecycle import Actor, Stage
    from algua.contracts.types import PendingLiveAuthorization
    from algua.registry.store import SqliteStrategyRepository

    conn = _conn(tmp_path)
    repo = SqliteStrategyRepository(conn)
    bogus = PendingLiveAuthorization(nonce="n", expires_at="2099-01-01T00:00:00+00:00",
                                     principal="lior", signature_b64="x")
    rec = repo.get("s")  # stage 'paper'
    with pytest.raises(ValueError, match="only valid for a human transition to live"):
        repo.apply_transition(rec, Stage.CANDIDATE, Actor.HUMAN, "x", live_authorization=bogus)


def _live_strategy(conn, stage="live"):
    conn.execute("UPDATE strategies SET stage=? WHERE id=1", (stage,))
    conn.commit()


def _seed_authorization(conn, tmp_path, *, code="ch", cfg="cfg", dep="dep", principal="lior"):
    key, pub = _make_key(tmp_path)
    signers = _allowed_signers(tmp_path, principal, pub)
    now = datetime(2026, 6, 5, tzinfo=UTC)
    issued = live_gate.issue_challenge(conn, 1, "s", code, cfg, dep, now=now)
    sig = _sign(key, issued["challenge"], tmp_path)
    pending = live_gate.verify_pending(conn, "s", 1, code, cfg, dep, sig, signers, now=now)
    assert pending is not None
    _record_authorization(conn, pending, code=code, cfg=cfg, dep=dep, now=now)
    return signers


def _identity(monkeypatch, code="ch", cfg="cfg", dep="dep"):
    from algua.registry.repository import ArtifactIdentity
    monkeypatch.setattr("algua.registry.approvals.compute_artifact_hashes",
                        lambda name: ArtifactIdentity(code, cfg, dep))


def test_verify_live_authorization_happy_path(tmp_path, monkeypatch):
    from algua.registry.store import SqliteStrategyRepository
    conn = _conn(tmp_path)
    signers = _seed_authorization(conn, tmp_path)
    _live_strategy(conn)
    _identity(monkeypatch)
    auth = live_gate.verify_live_authorization(conn, SqliteStrategyRepository(conn), "s", signers)
    from algua.contracts.types import LiveAuthorization
    assert isinstance(auth, LiveAuthorization)
    assert auth.principal == "lior" and auth.strategy_id == 1


def test_verify_live_authorization_rejects_non_live(tmp_path, monkeypatch):
    import pytest

    from algua.registry.live_gate import LiveAuthorizationError
    from algua.registry.store import SqliteStrategyRepository
    conn = _conn(tmp_path)
    signers = _seed_authorization(conn, tmp_path)  # stage stays 'paper'
    _identity(monkeypatch)
    with pytest.raises(LiveAuthorizationError):
        live_gate.verify_live_authorization(conn, SqliteStrategyRepository(conn), "s", signers)


def test_verify_live_authorization_rejects_changed_code(tmp_path, monkeypatch):
    import pytest

    from algua.registry.live_gate import LiveAuthorizationError
    from algua.registry.store import SqliteStrategyRepository
    conn = _conn(tmp_path)
    signers = _seed_authorization(conn, tmp_path, code="ch")
    _live_strategy(conn)
    _identity(monkeypatch, code="CHANGED")
    with pytest.raises(LiveAuthorizationError):
        live_gate.verify_live_authorization(conn, SqliteStrategyRepository(conn), "s", signers)


def test_verify_live_authorization_rejects_revoked_or_unenrolled(tmp_path, monkeypatch):
    import pytest

    from algua.registry.live_gate import LiveAuthorizationError
    from algua.registry.store import SqliteStrategyRepository
    conn = _conn(tmp_path)
    signers = _seed_authorization(conn, tmp_path)
    _live_strategy(conn)
    _identity(monkeypatch)
    conn.execute("UPDATE live_authorizations SET revoked_at='2026-06-05' WHERE strategy_id=1")
    conn.commit()
    with pytest.raises(LiveAuthorizationError):
        live_gate.verify_live_authorization(conn, SqliteStrategyRepository(conn), "s", signers)
    conn.execute("UPDATE live_authorizations SET revoked_at=NULL WHERE strategy_id=1")
    conn.commit()
    signers.write_text("# no keys enrolled\n")
    with pytest.raises(LiveAuthorizationError):
        live_gate.verify_live_authorization(conn, SqliteStrategyRepository(conn), "s", signers)


def test_verify_live_authorization_rejects_forged_identity_columns(tmp_path, monkeypatch):
    # CRITICAL regression: an agent inserts a row whose identity COLUMNS match the current vetted
    # code, but whose signature is a real human signature over DIFFERENT code. Trade-time verify
    # must reject it — it rebuilds the payload from the recomputed identity, not the stored bytes.
    import pytest

    from algua.registry.live_gate import LiveAuthorizationError
    from algua.registry.store import SqliteStrategyRepository
    conn = _conn(tmp_path)
    signers = _seed_authorization(conn, tmp_path, code="ORIG")  # real signature over code=ORIG
    conn.execute("UPDATE live_authorizations SET code_hash='CURRENT' WHERE strategy_id=1")
    conn.commit()
    _live_strategy(conn)
    _identity(monkeypatch, code="CURRENT")
    with pytest.raises(LiveAuthorizationError):
        live_gate.verify_live_authorization(conn, SqliteStrategyRepository(conn), "s", signers)


def test_verify_live_authorization_revoked_newest_blocks_older(tmp_path, monkeypatch):
    import pytest

    from algua.registry.live_gate import LiveAuthorizationError
    from algua.registry.store import SqliteStrategyRepository
    conn = _conn(tmp_path)
    signers = _seed_authorization(conn, tmp_path)  # valid unrevoked (older) row
    conn.execute(
        "INSERT INTO live_authorizations(strategy_id, code_hash, config_hash, dependency_hash, "
        "nonce, expires_at, signature, principal, authorized_at, revoked_at) "
        "SELECT strategy_id, code_hash, config_hash, dependency_hash, nonce, expires_at, "
        "signature, principal, authorized_at, '2026-06-05' FROM live_authorizations "
        "WHERE id=(SELECT MAX(id) FROM live_authorizations)")
    conn.commit()
    _live_strategy(conn)
    _identity(monkeypatch)
    with pytest.raises(LiveAuthorizationError):
        live_gate.verify_live_authorization(conn, SqliteStrategyRepository(conn), "s", signers)


def test_authorization_active_true_until_revoked(tmp_path):
    from algua.contracts.types import LiveAuthorization
    conn = _conn(tmp_path)
    _seed_authorization(conn, tmp_path, code="ch", cfg="cfg", dep="dep")
    auth = LiveAuthorization(strategy_id=1, code_hash="ch", config_hash="cfg",
                             dependency_hash="dep", principal="lior", authorized_at="t")
    assert live_gate.authorization_active(conn, auth) is True
    conn.execute("UPDATE live_authorizations SET revoked_at='2026-06-05' WHERE strategy_id=1")
    conn.commit()
    assert live_gate.authorization_active(conn, auth) is False
    other = LiveAuthorization(1, "OTHER", "cfg", "dep", "lior", "t")
    assert live_gate.authorization_active(conn, other) is False
