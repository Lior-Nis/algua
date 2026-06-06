from __future__ import annotations

import base64
import binascii
import secrets
import sqlite3
import subprocess
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

from algua.contracts.types import LiveAuthorization
from algua.registry.repository import StrategyRepository

_NAMESPACE = "algua-go-live"
_TTL = timedelta(minutes=10)

# The go-live trust anchor: enrolled approver PUBLIC keys, resolved from the INSTALLED source tree
# (not CWD) so the gate always reads the vetted, CODEOWNERS-reviewed copy. Shared by the CLI and
# the trade-time verifier so there is exactly one anchor.
ALLOWED_SIGNERS_PATH = Path(__file__).resolve().parents[2] / "approvers" / "allowed_signers"


def _now() -> datetime:
    return datetime.now(UTC)


def build_challenge(strategy: str, strategy_id: int, code_hash: str, config_hash: str,
                    dependency_hash: str | None, nonce: str, expires_at: str) -> str:
    """The exact bytes the operator signs. One definition, used to both issue and verify so the
    two can never drift. Binds the go-live to a specific strategy + full artifact identity
    (code + config + locked dependencies) + single-use nonce."""
    return (
        f"{_NAMESPACE}\nstrategy={strategy}\nstrategy_id={strategy_id}\n"
        f"code_hash={code_hash}\nconfig_hash={config_hash}\ndependency_hash={dependency_hash}\n"
        f"nonce={nonce}\nexpires_at={expires_at}"
    )


def issue_challenge(conn: sqlite3.Connection, strategy_id: int, strategy: str, code_hash: str,
                    config_hash: str, dependency_hash: str | None, *,
                    now: datetime | None = None) -> dict[str, str]:
    """Create + persist a pending go-live challenge; return {nonce, challenge, expires_at}."""
    now = now or _now()
    nonce = secrets.token_hex(32)
    expires_at = (now + _TTL).isoformat()
    conn.execute(
        "INSERT INTO live_challenges(nonce, strategy_id, code_hash, config_hash, dependency_hash, "
        "issued_at, expires_at, consumed_at) VALUES (?,?,?,?,?,?,?,NULL)",
        (nonce, strategy_id, code_hash, config_hash, dependency_hash, now.isoformat(), expires_at),
    )
    conn.commit()
    return {"nonce": nonce, "expires_at": expires_at,
            "challenge": build_challenge(strategy, strategy_id, code_hash, config_hash,
                                         dependency_hash, nonce, expires_at)}


def find_pending_challenge(conn: sqlite3.Connection, strategy_id: int, code_hash: str,
                           config_hash: str, dependency_hash: str | None, *,
                           now: datetime | None = None) -> sqlite3.Row | None:
    """The newest unconsumed, unexpired challenge matching the strategy + recomputed identity."""
    now = now or _now()
    return conn.execute(
        "SELECT * FROM live_challenges WHERE strategy_id=? AND code_hash=? AND config_hash=? "
        "AND dependency_hash IS ? AND consumed_at IS NULL AND expires_at > ? "
        "ORDER BY issued_at DESC LIMIT 1",
        (strategy_id, code_hash, config_hash, dependency_hash, now.isoformat()),
    ).fetchone()


def consume_challenge(conn: sqlite3.Connection, nonce: str, *,
                      now: datetime | None = None) -> bool:
    """Mark a challenge consumed (single-use). Returns False if already consumed/missing."""
    now = now or _now()
    cur = conn.execute(
        "UPDATE live_challenges SET consumed_at=? WHERE nonce=? AND consumed_at IS NULL",
        (now.isoformat(), nonce),
    )
    conn.commit()
    return cur.rowcount > 0


class SignatureError(RuntimeError):
    """ssh-keygen is unavailable or failed in an unexpected way (not a plain bad signature)."""


def verify_signature(allowed_signers_path: Path, payload: str, signature: bytes) -> str | None:
    """Verify an SSH signature over `payload` against the enrolled keys in `allowed_signers_path`.
    Returns the matched principal on success, or None if the signature is invalid / the signer is
    not enrolled. Raises SignatureError when ssh-keygen can't run OR the trust anchor file is
    missing — a missing anchor is a configuration error, never a silent pass (codex review)."""
    if not allowed_signers_path.is_file():
        raise SignatureError(f"allowed_signers trust anchor not found: {allowed_signers_path}")
    with tempfile.NamedTemporaryFile("wb", suffix=".sig", delete=True) as sigf:
        sigf.write(signature)
        sigf.flush()
        data = payload.encode()
        try:
            found = subprocess.run(
                ["ssh-keygen", "-Y", "find-principals", "-f", str(allowed_signers_path),
                 "-s", sigf.name],
                input=data, capture_output=True,
            )
        except FileNotFoundError as exc:  # ssh-keygen not installed
            raise SignatureError("ssh-keygen not found on PATH") from exc
        if found.returncode != 0:
            return None  # signer not enrolled (or malformed signature)
        out = found.stdout.decode().splitlines()
        principal = out[0].strip() if out and out[0].strip() else ""
        if not principal:
            return None
        verified = subprocess.run(
            ["ssh-keygen", "-Y", "verify", "-f", str(allowed_signers_path), "-I", principal,
             "-n", _NAMESPACE, "-s", sigf.name],
            input=data, capture_output=True,
        )
        return principal if verified.returncode == 0 else None


def verify_and_consume(conn: sqlite3.Connection, strategy: str, strategy_id: int, code_hash: str,
                       config_hash: str, dependency_hash: str | None, signature: bytes,
                       allowed_signers_path: Path, *, now: datetime | None = None) -> str | None:
    """Find the pending challenge for this artifact, verify the signature over its exact payload,
    and atomically consume it. Returns the approver principal on success, else None."""
    now = now or _now()
    row = find_pending_challenge(conn, strategy_id, code_hash, config_hash, dependency_hash,
                                 now=now)
    if row is None:
        return None
    payload = build_challenge(strategy, strategy_id, code_hash, config_hash, dependency_hash,
                              row["nonce"], row["expires_at"])
    principal = verify_signature(allowed_signers_path, payload, signature)
    if principal is None:
        return None
    if not consume_challenge(conn, row["nonce"], now=now):
        return None
    # Persist the durable proof of this go-live: the identity, the nonce+expires_at (so the exact
    # signed payload can be REBUILT from a recomputed identity at trade time), the signature, and
    # the approver. We deliberately do NOT store the challenge text — trade-time verification must
    # rebuild it from the recomputed identity, never trust agent-writable bytes (codex CRITICAL).
    conn.execute(
        "INSERT INTO live_authorizations(strategy_id, code_hash, config_hash, dependency_hash, "
        "nonce, expires_at, signature, principal, authorized_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (strategy_id, code_hash, config_hash, dependency_hash, row["nonce"], row["expires_at"],
         base64.b64encode(signature).decode(), principal, now.isoformat()),
    )
    conn.commit()
    return principal


class LiveAuthorizationError(RuntimeError):
    """The current live artifact is NOT covered by a re-verifiable human go-live signature."""


def verify_live_authorization(conn: sqlite3.Connection, repo: StrategyRepository, name: str,
                              allowed_signers_path: Path) -> LiveAuthorization:
    """Trade-time wall: prove the strategy's CURRENT artifact is human-authorized for live by
    re-verifying a stored signature against the CURRENT trust anchor. Trusts nothing forgeable —
    not the `stage` column, not an `approvals` row. Raises LiveAuthorizationError on any failure;
    returns the authorization row on success. The future live loop calls this before every order."""
    from algua.contracts.lifecycle import Stage
    from algua.registry.approvals import compute_artifact_hashes

    rec = repo.get(name)
    if rec.stage is not Stage.LIVE:
        raise LiveAuthorizationError(f"{name} is not live (stage={rec.stage.value})")
    identity = compute_artifact_hashes(name)
    # Newest authorization for this strategy + recomputed identity REGARDLESS of revocation, so a
    # revoked newest row BLOCKS even if an older unrevoked one for the same artifact exists.
    row = conn.execute(
        "SELECT * FROM live_authorizations WHERE strategy_id=? AND code_hash=? AND config_hash=? "
        "AND dependency_hash IS ? ORDER BY id DESC LIMIT 1",
        (rec.id, identity.code_hash, identity.config_hash, identity.dependency_hash),
    ).fetchone()
    if row is None:
        raise LiveAuthorizationError(
            f"no live authorization matching the current artifact of {name}")
    if row["revoked_at"] is not None:
        raise LiveAuthorizationError(f"the live authorization for the current artifact of {name} "
                                     "is revoked")
    # REBUILD the canonical payload from the RECOMPUTED identity (+ strategy + the stored
    # nonce/expires_at) and re-verify the signature over THAT — never over agent-writable stored
    # bytes. A signature validates only if the human signed this exact strategy + current identity,
    # so forged identity columns paired with a foreign signature can't pass (codex CRITICAL).
    payload = build_challenge(name, rec.id, identity.code_hash, identity.config_hash,
                              identity.dependency_hash, row["nonce"], row["expires_at"])
    try:
        principal = verify_signature(allowed_signers_path, payload,
                                     base64.b64decode(row["signature"]))
    except (SignatureError, binascii.Error) as exc:
        raise LiveAuthorizationError(
            f"could not re-verify the live authorization for {name}: {exc}") from exc
    if principal is None or principal != row["principal"]:
        raise LiveAuthorizationError(
            f"live authorization signature for {name} failed re-verification against the anchor")
    return LiveAuthorization(
        strategy_id=rec.id,
        code_hash=row["code_hash"],
        config_hash=row["config_hash"],
        dependency_hash=row["dependency_hash"],
        principal=row["principal"],
        authorized_at=row["authorized_at"],
    )


def authorization_active(conn: sqlite3.Connection, authorization: LiveAuthorization) -> bool:
    """Cheap mid-tick check: does an UNREVOKED live_authorizations row matching the (already
    re-verified) authorization's identity still exist? No ssh-keygen, no hash recompute — for the
    per-order should_halt hook so a legitimate operator revocation aborts the rest of a tick.

    ADVISORY, NOT A SECURITY BOUNDARY: it matches identity columns only and does NOT re-verify the
    signature, so it trusts that `verify_live_authorization` signature-verified the row at the START
    of THIS invocation (the real wall, run every tick before any order). It is an in-tick liveness
    check for mid-tick revocation, not protection against a forged row — that is the full
    re-verify's job. (Slice 4 may tie this to the verified row id for defense in depth.)"""
    row = conn.execute(
        "SELECT 1 FROM live_authorizations WHERE strategy_id=? AND code_hash=? AND config_hash=? "
        "AND dependency_hash IS ? AND revoked_at IS NULL LIMIT 1",
        (authorization.strategy_id, authorization.code_hash, authorization.config_hash,
         authorization.dependency_hash),
    ).fetchone()
    return row is not None
