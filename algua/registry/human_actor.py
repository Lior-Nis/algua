from __future__ import annotations

import json
import secrets
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from algua.contracts.lifecycle import Actor
from algua.registry.live_gate import ALLOWED_SIGNERS_PATH, verify_signature

# Human-actor authentication (#329). A bare `--actor human` CLI string is forgeable: any agent
# driving the identical CLI can pass it to unlock every human-only relaxation below the live wall
# (declared breadth `--n-combos`, `--allow-holdout-reuse`, `--allow-non-pit`,
# `--assume-terminal-last-close`, the NOVEL/PARENTAGE `--new-family` mint, and the paper-promote
# threshold relaxations). This module authenticates the human actor by REUSING the go-live signature
# trust anchor (approvers/allowed_signers) under a DISTINCT signing namespace, so a key must be
# enrolled *for the human-actor namespace* to authenticate a human — a go-live-only key cannot.
#
# The signed challenge binds the command + strategy + RECOMPUTED artifact identity + the FULL
# canonical run_context (every gate-relevant invocation input, incl. the exact relaxation set) + a
# single-use nonce + expiry, so a captured signature cannot be replayed onto a new artifact, a
# different concrete run (different start/end/snapshot/universe/threshold), a different relaxation
# set, a different command/stage edge, another strategy, or a second run.
#
# Trust boundary: same as go-live — the installed CODEOWNERS-reviewed anchor + gate code + registry
# DB. A process that can write the local tree/DB defeats this exactly as it defeats go-live; that
# residual is #329's DEFERRED deploy-time-anchor-immutability half, filed separately.

_NAMESPACE = "algua-human-actor"
_TTL = timedelta(minutes=10)


class HumanActorChallengeRequired(RuntimeError):
    """`--actor human` was asserted on a gated command without a signature. Carries the freshly
    issued challenge dict so the CLI can print it (mirrors the go-live challenge print)."""

    def __init__(self, challenge: dict[str, str]) -> None:
        super().__init__("human actor assertion requires a signature")
        self.challenge = challenge


def _now() -> datetime:
    return datetime.now(UTC)


def canonical_run_context(opts: dict[str, object]) -> str:
    """INJECTIVE canonical string of the FULL gate-relevant invocation input set (NOT only the
    human-only relaxations). Canonical JSON — sorted keys, compact separators, None-valued keys
    dropped — so a value that contains a delimiter cannot forge a different invocation into the same
    canonical string (a `key=value;` join would not be injective; codex GATE-1). This is signed, so
    a signature authorizes EXACTLY this concrete run: asking for a different or additional input at
    completion re-canonicalizes to different bytes and fails verification (no
    escalation-by-substitution and no cross-run replay)."""
    clean = {k: v for k, v in opts.items() if v is not None}
    return json.dumps(clean, sort_keys=True, separators=(",", ":"))


def build_actor_challenge(
    command: str, strategy: str, strategy_id: int, stage_from: str, stage_to: str,
    code_hash: str, config_hash: str, dependency_hash: str | None, run_context: str,
    nonce: str, expires_at: str,
) -> str:
    """The exact bytes the human signs. ONE definition, used to both issue and verify so the two can
    never drift. Binds the human-actor assertion to a specific command + strategy + full artifact
    identity + the canonical run_context + single-use nonce + expiry."""
    return (
        f"{_NAMESPACE}\ncommand={command}\nstrategy={strategy}\nstrategy_id={strategy_id}\n"
        f"stage_from={stage_from}\nstage_to={stage_to}\n"
        f"code_hash={code_hash}\nconfig_hash={config_hash}\ndependency_hash={dependency_hash}\n"
        f"run_context={run_context}\nnonce={nonce}\nexpires_at={expires_at}"
    )


def issue_actor_challenge(
    conn: sqlite3.Connection, command: str, strategy_id: int, strategy: str, stage_from: str,
    stage_to: str, code_hash: str, config_hash: str, dependency_hash: str | None, run_context: str,
    *, now: datetime | None = None,
) -> dict[str, str]:
    """Create + persist a pending human-actor challenge; return {nonce, expires_at, challenge}."""
    now = now or _now()
    nonce = secrets.token_hex(32)
    expires_at = (now + _TTL).isoformat()
    conn.execute(
        "INSERT INTO actor_challenges(nonce, command, strategy_id, stage_from, stage_to, "
        "code_hash, config_hash, dependency_hash, run_context, issued_at, expires_at, consumed_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,NULL)",
        (nonce, command, strategy_id, stage_from, stage_to, code_hash, config_hash,
         dependency_hash, run_context, now.isoformat(), expires_at),
    )
    conn.commit()
    return {
        "nonce": nonce, "expires_at": expires_at,
        "challenge": build_actor_challenge(
            command, strategy, strategy_id, stage_from, stage_to, code_hash, config_hash,
            dependency_hash, run_context, nonce, expires_at),
    }


def find_pending_actor_challenge(
    conn: sqlite3.Connection, command: str, strategy_id: int, stage_from: str, stage_to: str,
    code_hash: str, config_hash: str, dependency_hash: str | None, run_context: str,
    *, now: datetime | None = None,
) -> sqlite3.Row | None:
    """Newest unconsumed, unexpired challenge matching EVERY bound field (command + strategy +
    recomputed identity + re-canonicalized run_context + stage edge)."""
    now = now or _now()
    return conn.execute(
        "SELECT * FROM actor_challenges WHERE command=? AND strategy_id=? AND stage_from=? "
        "AND stage_to=? AND code_hash=? AND config_hash=? AND dependency_hash IS ? "
        "AND run_context=? AND consumed_at IS NULL AND expires_at > ? "
        "ORDER BY issued_at DESC LIMIT 1",
        (command, strategy_id, stage_from, stage_to, code_hash, config_hash, dependency_hash,
         run_context, now.isoformat()),
    ).fetchone()


def consume_actor_challenge(conn: sqlite3.Connection, nonce: str, *,
                            now: datetime | None = None) -> bool:
    """Mark a challenge consumed (single-use). Returns False if already consumed / missing."""
    now = now or _now()
    cur = conn.execute(
        "UPDATE actor_challenges SET consumed_at=? WHERE nonce=? AND consumed_at IS NULL",
        (now.isoformat(), nonce),
    )
    conn.commit()
    return cur.rowcount > 0


def verify_actor_assertion(
    conn: sqlite3.Connection, command: str, strategy: str, strategy_id: int, stage_from: str,
    stage_to: str, code_hash: str, config_hash: str, dependency_hash: str | None, run_context: str,
    signature: bytes, allowed_signers_path: Path | None = None, *, now: datetime | None = None,
) -> str | None:
    """Verify a human-actor signature over the REBUILT payload (the caller supplies the RECOMPUTED
    identity + re-canonicalized run_context, never agent-writable stored bytes) against the enrolled
    keys for the ``algua-human-actor`` namespace, then consume the matching single-use nonce.

    Returns the matched principal on success, or None on any failure (no enrolled signer for this
    namespace, bad signature, no matching/expired/consumed challenge, lost consume race) — fail
    closed. Raises SignatureError (via verify_signature) only when ssh-keygen can't run, the anchor
    is missing, or an anchor line is unscoped — a config error, never a silent pass."""
    anchor = allowed_signers_path or ALLOWED_SIGNERS_PATH
    now = now or _now()
    row = find_pending_actor_challenge(
        conn, command, strategy_id, stage_from, stage_to, code_hash, config_hash, dependency_hash,
        run_context, now=now)
    if row is None:
        return None
    payload = build_actor_challenge(
        command, strategy, strategy_id, stage_from, stage_to, code_hash, config_hash,
        dependency_hash, run_context, row["nonce"], row["expires_at"])
    principal = verify_signature(anchor, payload, signature, namespace=_NAMESPACE)
    if principal is None:
        return None
    # Consume ONLY after a valid signature, single-use. A lost race (already consumed) fails closed.
    if not consume_actor_challenge(conn, row["nonce"], now=now):
        return None
    return principal


def resolve_effective_actor(
    conn: sqlite3.Connection, command: str, strategy: str, strategy_id: int, stage_from: str,
    stage_to: str, code_hash: str, config_hash: str, dependency_hash: str | None,
    declared_actor: Actor, run_context: str, signature: bytes | None,
    allowed_signers_path: Path | None = None, *, now: datetime | None = None,
) -> Actor:
    """The ONE chokepoint that turns a declared ``--actor`` + optional signature into the effective
    Actor the downstream human-only guards trust. Fail closed:

    - declared AGENT / SYSTEM  -> returned unchanged (agents never sign; SYSTEM is refused later by
      the preflight actor-legality check).
    - declared HUMAN, no signature -> issue+persist a fresh challenge and raise
      HumanActorChallengeRequired (the CLI prints it). A bare `--actor human` string thus unlocks
      NOTHING on a gated command.
    - declared HUMAN + signature -> verify_actor_assertion; return HUMAN iff it authenticates, else
      raise ValueError (a forged / replayed / expired / cross-run signature is refused)."""
    if declared_actor is not Actor.HUMAN:
        return declared_actor
    if signature is None:
        issued = issue_actor_challenge(
            conn, command, strategy_id, strategy, stage_from, stage_to, code_hash, config_hash,
            dependency_hash, run_context, now=now)
        raise HumanActorChallengeRequired(issued)
    principal = verify_actor_assertion(
        conn, command, strategy, strategy_id, stage_from, stage_to, code_hash, config_hash,
        dependency_hash, run_context, signature, allowed_signers_path, now=now)
    if principal is None:
        raise ValueError(
            "human actor authentication failed: --actor-signature does not match an enrolled "
            "algua-human-actor key over a fresh challenge bound to this exact strategy, artifact "
            "identity, and run context. Re-run without --actor-signature to get a new challenge, "
            "sign it with your enrolled key (ssh-keygen -Y sign -n algua-human-actor), and retry. "
            "A bare --actor human does not unlock human-only paths."
        )
    return Actor.HUMAN
