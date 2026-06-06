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


def test_verify_and_consume_writes_live_authorization(tmp_path):
    import base64

    conn = _conn(tmp_path)
    now = datetime(2026, 6, 5, tzinfo=UTC)
    issued = live_gate.issue_challenge(conn, 1, "s", "ch", "cfg", "dep", now=now)
    key, pub = _make_key(tmp_path)
    signers = _allowed_signers(tmp_path, "lior", pub)
    sig = _sign(key, issued["challenge"], tmp_path)
    principal = live_gate.verify_and_consume(
        conn, "s", 1, "ch", "cfg", "dep", sig, signers, now=now
    )
    assert principal == "lior"
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


def test_verify_and_consume_no_authorization_on_bad_signature(tmp_path):
    conn = _conn(tmp_path)
    now = datetime(2026, 6, 5, tzinfo=UTC)
    issued = live_gate.issue_challenge(conn, 1, "s", "ch", "cfg", "dep", now=now)
    key, _pub = _make_key(tmp_path, "mine")
    _ok, other_pub = _make_key(tmp_path, "other")
    signers = _allowed_signers(tmp_path, "lior", other_pub)  # different key enrolled
    sig = _sign(key, issued["challenge"], tmp_path)
    assert live_gate.verify_and_consume(
        conn, "s", 1, "ch", "cfg", "dep", sig, signers, now=now
    ) is None
    assert conn.execute("SELECT COUNT(*) FROM live_authorizations").fetchone()[0] == 0


def _live_strategy(conn, stage="live"):
    conn.execute("UPDATE strategies SET stage=? WHERE id=1", (stage,))
    conn.commit()


def _seed_authorization(conn, tmp_path, *, code="ch", cfg="cfg", dep="dep", principal="lior"):
    key, pub = _make_key(tmp_path)
    signers = _allowed_signers(tmp_path, principal, pub)
    issued = live_gate.issue_challenge(conn, 1, "s", code, cfg, dep,
                                       now=datetime(2026, 6, 5, tzinfo=UTC))
    sig = _sign(key, issued["challenge"], tmp_path)
    live_gate.verify_and_consume(conn, "s", 1, code, cfg, dep, sig, signers,
                                 now=datetime(2026, 6, 5, tzinfo=UTC))
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
