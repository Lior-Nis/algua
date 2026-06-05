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
    assert row["dependency_hash"] == "dep"
    assert row["principal"] == "lior" and row["challenge"] == issued["challenge"]
    assert live_gate.verify_signature(signers, row["challenge"],
                                      base64.b64decode(row["signature"])) == "lior"


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
