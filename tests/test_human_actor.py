"""Human-actor authentication (#329): forge/replay defenses on the `--actor human` gate.

Adversary model: an autonomous agent driving the identical CLI that tries to FORGE an authenticated
`--actor human` or REPLAY a captured signature to unlock human-only gate relaxations. It cannot
write the tree/DB (the accepted deferred residual). Every test here is a concrete forge/replay
attempt that MUST fail closed, plus the happy path and the agent-unchanged path.
"""
import subprocess
from datetime import UTC, datetime, timedelta

import pytest

from algua.contracts.lifecycle import Actor
from algua.registry import human_actor, live_gate
from algua.registry.db import connect, migrate
from algua.registry.human_actor import (
    HumanActorChallengeRequired,
    canonical_run_context,
    resolve_effective_actor,
    verify_actor_assertion,
)

_NS = "algua-human-actor"


def _conn(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    conn.execute("INSERT INTO strategies(id, name, stage, created_at, updated_at) "
                 "VALUES (1, 's', 'backtested', '2026-01-01', '2026-01-01')")
    conn.commit()
    return conn


def _make_key(tmp_path, name="id"):
    key = tmp_path / name
    subprocess.run(["ssh-keygen", "-t", "ed25519", "-N", "", "-C", "t", "-f", str(key)],
                   check=True, capture_output=True)
    return key, (tmp_path / f"{name}.pub").read_text().strip()


def _anchor(tmp_path, principal, pub, namespaces=_NS):
    keytype, keyblob = pub.split()[0], pub.split()[1]
    path = tmp_path / "allowed_signers"
    path.write_text(f'{principal} namespaces="{namespaces}" {keytype} {keyblob}\n')
    return path


def _sign(key, payload, tmp_path, namespace=_NS):
    data = tmp_path / "payload.txt"
    data.write_text(payload)
    subprocess.run(["ssh-keygen", "-Y", "sign", "-n", namespace, "-f", str(key), str(data)],
                   check=True, capture_output=True)
    return (tmp_path / "payload.txt.sig").read_bytes()


# ---- pure canonicalization: injective, order-independent, delimiter-safe -----------------------

def test_canonical_run_context_is_injective_and_sorted():
    a = canonical_run_context({"start": "2023-01-01", "allow_non_pit": True, "n_combos": None})
    b = canonical_run_context({"n_combos": None, "allow_non_pit": True, "start": "2023-01-01"})
    assert a == b  # key order does not matter
    assert '"n_combos"' not in a  # None dropped
    assert '"allow_non_pit":true' in a and '"start":"2023-01-01"' in a


def test_canonical_run_context_binds_resolved_provenance():
    # Binding only the mutable NAME is insufficient (codex GATE-2): the same `--universe csm`
    # can resolve to a different snapshot timeline after a `data ingest-universe`. The signed
    # run_context includes the RESOLVED provenance (snapshot ids + effective dates), so a timeline
    # shift produces a different canonical string -> the old signature no longer matches.
    prov_a = [{"snapshot_id": "snapA", "effective_date": "2022-01-01"}]
    prov_b = [{"snapshot_id": "snapB", "effective_date": "2022-06-01"},
              {"snapshot_id": "snapA", "effective_date": "2022-01-01"}]
    a = canonical_run_context({"universe": "csm", "universe_prov": prov_a})
    b = canonical_run_context({"universe": "csm", "universe_prov": prov_b})
    assert a != b  # a resolved-provenance shift breaks the binding even with an identical name


def test_canonical_run_context_not_delimiter_injectable():
    # A value containing what would be a naive `key=value;` delimiter cannot collide with a
    # different invocation — JSON escapes it.
    x = canonical_run_context({"a": "1;b=2", "b": "3"})
    y = canonical_run_context({"a": "1", "b": "2;...", "extra": "3"})
    assert x != y


# ---- issue / find / consume: single-use, expiry, full-field binding ----------------------------

def test_issue_find_consume_single_use(tmp_path):
    conn = _conn(tmp_path)
    now = datetime(2026, 7, 2, tzinfo=UTC)
    rc = canonical_run_context({"allow_non_pit": True})
    issued = human_actor.issue_actor_challenge(
        conn, "research promote", 1, "s", "backtested", "candidate", "ch", "cfg", "dep", rc,
        now=now)
    assert issued["challenge"].startswith(_NS)
    row = human_actor.find_pending_actor_challenge(
        conn, "research promote", 1, "backtested", "candidate", "ch", "cfg", "dep", rc, now=now)
    assert row is not None and row["nonce"] == issued["nonce"]
    assert human_actor.consume_actor_challenge(conn, issued["nonce"], now=now) is True
    assert human_actor.consume_actor_challenge(conn, issued["nonce"], now=now) is False


def test_find_rejects_expired_and_any_field_mismatch(tmp_path):
    conn = _conn(tmp_path)
    now = datetime(2026, 7, 2, tzinfo=UTC)
    rc = canonical_run_context({"allow_non_pit": True})
    human_actor.issue_actor_challenge(
        conn, "research promote", 1, "s", "backtested", "candidate", "ch", "cfg", "dep", rc,
        now=now)
    later = now + timedelta(hours=1)
    f = human_actor.find_pending_actor_challenge
    assert f(conn, "research promote", 1, "backtested", "candidate", "ch", "cfg", "dep", rc,
             now=later) is None  # expired
    assert f(conn, "paper promote", 1, "backtested", "candidate", "ch", "cfg", "dep", rc,
             now=now) is None  # command
    assert f(conn, "research promote", 1, "backtested", "live", "ch", "cfg", "dep", rc,
             now=now) is None  # stage_to
    assert f(conn, "research promote", 1, "backtested", "candidate", "X", "cfg", "dep", rc,
             now=now) is None  # code_hash
    assert f(conn, "research promote", 1, "backtested", "candidate", "ch", "cfg", "dep",
             canonical_run_context({"allow_non_pit": False}), now=now) is None  # run_context


# ---- signature verification: happy path + every replay/forge attempt ---------------------------

def _identity():
    return ("code1", "cfg1", "dep1")


def _issue_and_sign(conn, key, tmp_path, *, command="research promote", strategy="s",
                    strategy_id=1, stage_from="backtested", stage_to="candidate",
                    identity=None, run_context=None, now):
    ch, cfg, dep = identity or _identity()
    rc = run_context if run_context is not None else canonical_run_context({"allow_non_pit": True})
    issued = human_actor.issue_actor_challenge(
        conn, command, strategy_id, strategy, stage_from, stage_to, ch, cfg, dep, rc, now=now)
    sig = _sign(key, issued["challenge"], tmp_path)
    return sig, rc, (ch, cfg, dep)


def test_verify_happy_path_then_consumed(tmp_path):
    conn = _conn(tmp_path)
    now = datetime(2026, 7, 2, tzinfo=UTC)
    key, pub = _make_key(tmp_path)
    anchor = _anchor(tmp_path, "lior", pub)
    sig, rc, (ch, cfg, dep) = _issue_and_sign(conn, key, tmp_path, now=now)
    principal = verify_actor_assertion(
        conn, "research promote", "s", 1, "backtested", "candidate", ch, cfg, dep, rc, sig,
        anchor, now=now)
    assert principal == "lior"
    # nonce is now consumed -> the SAME signature cannot be replayed a second time (replay defense)
    assert verify_actor_assertion(
        conn, "research promote", "s", 1, "backtested", "candidate", ch, cfg, dep, rc, sig,
        anchor, now=now) is None


def test_replay_across_artifact_identity_fails(tmp_path):
    conn = _conn(tmp_path)
    now = datetime(2026, 7, 2, tzinfo=UTC)
    key, pub = _make_key(tmp_path)
    anchor = _anchor(tmp_path, "lior", pub)
    sig, rc, _ = _issue_and_sign(conn, key, tmp_path, identity=("codeA", "cfgA", "depA"), now=now)
    # Re-verify with a DIFFERENT recomputed identity (artifact changed) -> no matching challenge.
    assert verify_actor_assertion(
        conn, "research promote", "s", 1, "backtested", "candidate", "codeB", "cfgA", "depA", rc,
        sig, anchor, now=now) is None


def test_replay_across_run_context_fails(tmp_path):
    conn = _conn(tmp_path)
    now = datetime(2026, 7, 2, tzinfo=UTC)
    key, pub = _make_key(tmp_path)
    anchor = _anchor(tmp_path, "lior", pub)
    # Signed for start=X; resubmit for start=Y with the SAME relaxation set -> fails closed.
    rc_signed = canonical_run_context({"start": "2023-01-01", "allow_non_pit": True})
    sig, _, (ch, cfg, dep) = _issue_and_sign(conn, key, tmp_path, run_context=rc_signed, now=now)
    rc_other = canonical_run_context({"start": "2020-01-01", "allow_non_pit": True})
    assert verify_actor_assertion(
        conn, "research promote", "s", 1, "backtested", "candidate", ch, cfg, dep, rc_other, sig,
        anchor, now=now) is None


def test_replay_across_relaxation_set_fails(tmp_path):
    conn = _conn(tmp_path)
    now = datetime(2026, 7, 2, tzinfo=UTC)
    key, pub = _make_key(tmp_path)
    anchor = _anchor(tmp_path, "lior", pub)
    rc_signed = canonical_run_context({"start": "2023-01-01"})  # no relaxation
    sig, _, (ch, cfg, dep) = _issue_and_sign(conn, key, tmp_path, run_context=rc_signed, now=now)
    rc_escalated = canonical_run_context({"start": "2023-01-01", "allow_non_pit": True})
    assert verify_actor_assertion(
        conn, "research promote", "s", 1, "backtested", "candidate", ch, cfg, dep, rc_escalated,
        sig, anchor, now=now) is None


def test_replay_across_command_stage_fails(tmp_path):
    conn = _conn(tmp_path)
    now = datetime(2026, 7, 2, tzinfo=UTC)
    key, pub = _make_key(tmp_path)
    anchor = _anchor(tmp_path, "lior", pub)
    sig, rc, (ch, cfg, dep) = _issue_and_sign(conn, key, tmp_path, now=now)
    assert verify_actor_assertion(
        conn, "paper promote", "s", 1, "backtested", "candidate", ch, cfg, dep, rc, sig, anchor,
        now=now) is None  # different command


def test_expired_challenge_signature_fails(tmp_path):
    conn = _conn(tmp_path)
    now = datetime(2026, 7, 2, tzinfo=UTC)
    key, pub = _make_key(tmp_path)
    anchor = _anchor(tmp_path, "lior", pub)
    sig, rc, (ch, cfg, dep) = _issue_and_sign(conn, key, tmp_path, now=now)
    later = now + timedelta(hours=1)
    assert verify_actor_assertion(
        conn, "research promote", "s", 1, "backtested", "candidate", ch, cfg, dep, rc, sig, anchor,
        now=later) is None


# ---- namespace confusion + unscoped-line defense (live_gate primitive) -------------------------

def test_go_live_namespace_key_cannot_authenticate_actor(tmp_path):
    conn = _conn(tmp_path)
    now = datetime(2026, 7, 2, tzinfo=UTC)
    key, pub = _make_key(tmp_path)
    # Enroll the key ONLY for algua-go-live; sign the actor challenge under algua-go-live too.
    anchor = _anchor(tmp_path, "lior", pub, namespaces="algua-go-live")
    ch, cfg, dep = _identity()
    rc = canonical_run_context({"allow_non_pit": True})
    issued = human_actor.issue_actor_challenge(
        conn, "research promote", 1, "s", "backtested", "candidate", ch, cfg, dep, rc, now=now)
    sig = _sign(key, issued["challenge"], tmp_path, namespace="algua-go-live")
    # verify_actor_assertion verifies under algua-human-actor -> the go-live-only key fails closed.
    assert verify_actor_assertion(
        conn, "research promote", "s", 1, "backtested", "candidate", ch, cfg, dep, rc, sig, anchor,
        now=now) is None


def test_unscoped_anchor_line_fails_closed(tmp_path):
    key, pub = _make_key(tmp_path)
    keytype, keyblob = pub.split()[0], pub.split()[1]
    path = tmp_path / "allowed_signers"
    path.write_text(f'lior {keytype} {keyblob}\n')  # NO namespaces= restriction (unscoped)
    with pytest.raises(live_gate.SignatureError):
        live_gate.verify_signature(path, "payload", b"sig", namespace=_NS)


def test_dual_namespace_key_authenticates_actor(tmp_path):
    conn = _conn(tmp_path)
    now = datetime(2026, 7, 2, tzinfo=UTC)
    key, pub = _make_key(tmp_path)
    anchor = _anchor(tmp_path, "lior", pub, namespaces="algua-go-live,algua-human-actor")
    sig, rc, (ch, cfg, dep) = _issue_and_sign(conn, key, tmp_path, now=now)
    assert verify_actor_assertion(
        conn, "research promote", "s", 1, "backtested", "candidate", ch, cfg, dep, rc, sig, anchor,
        now=now) == "lior"


# ---- resolve_effective_actor chokepoint --------------------------------------------------------

def test_resolve_agent_unchanged_no_challenge(tmp_path):
    conn = _conn(tmp_path)
    ch, cfg, dep = _identity()
    rc = canonical_run_context({"allow_non_pit": True})
    # An agent never issues a challenge and stays an agent (downstream refuses its relaxations).
    result = resolve_effective_actor(
        conn, "research promote", "s", 1, "backtested", "candidate", ch, cfg, dep, Actor.AGENT, rc,
        None)
    assert result is Actor.AGENT
    assert conn.execute("SELECT COUNT(*) FROM actor_challenges").fetchone()[0] == 0


def test_resolve_human_no_signature_issues_challenge_and_runs_nothing(tmp_path):
    conn = _conn(tmp_path)
    ch, cfg, dep = _identity()
    rc = canonical_run_context({"allow_non_pit": True})
    with pytest.raises(HumanActorChallengeRequired) as exc:
        resolve_effective_actor(
            conn, "research promote", "s", 1, "backtested", "candidate", ch, cfg, dep, Actor.HUMAN,
            rc, None)
    assert exc.value.challenge["challenge"].startswith(_NS)
    # A bare `--actor human` unlocked nothing — only a pending challenge exists.
    assert conn.execute("SELECT COUNT(*) FROM actor_challenges").fetchone()[0] == 1


def test_resolve_human_bad_signature_raises_value_error(tmp_path):
    conn = _conn(tmp_path)
    now = datetime(2026, 7, 2, tzinfo=UTC)
    key, pub = _make_key(tmp_path)
    _other, other_pub = _make_key(tmp_path, "other")
    anchor = _anchor(tmp_path, "lior", other_pub)  # enroll a DIFFERENT key
    ch, cfg, dep = _identity()
    rc = canonical_run_context({"allow_non_pit": True})
    issued = human_actor.issue_actor_challenge(
        conn, "research promote", 1, "s", "backtested", "candidate", ch, cfg, dep, rc, now=now)
    sig = _sign(key, issued["challenge"], tmp_path)  # signed by the NON-enrolled key
    with pytest.raises(ValueError, match="human actor authentication failed"):
        resolve_effective_actor(
            conn, "research promote", "s", 1, "backtested", "candidate", ch, cfg, dep, Actor.HUMAN,
            rc, sig, anchor, now=now)


def test_resolve_human_valid_signature_returns_human(tmp_path):
    conn = _conn(tmp_path)
    now = datetime(2026, 7, 2, tzinfo=UTC)
    key, pub = _make_key(tmp_path)
    anchor = _anchor(tmp_path, "lior", pub)
    sig, rc, (ch, cfg, dep) = _issue_and_sign(conn, key, tmp_path, now=now)
    result = resolve_effective_actor(
        conn, "research promote", "s", 1, "backtested", "candidate", ch, cfg, dep, Actor.HUMAN, rc,
        sig, anchor, now=now)
    assert result is Actor.HUMAN
