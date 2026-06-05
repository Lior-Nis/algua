from __future__ import annotations

import secrets
import sqlite3
from datetime import UTC, datetime, timedelta

_NAMESPACE = "algua-go-live"
_TTL = timedelta(minutes=10)


def _now() -> datetime:
    return datetime.now(UTC)


def build_challenge(strategy: str, strategy_id: int, code_hash: str, config_hash: str,
                    nonce: str, expires_at: str) -> str:
    """The exact bytes the operator signs. One definition, used to both issue and verify so the
    two can never drift. Binds the go-live to a specific strategy + artifact + single-use nonce."""
    return (
        f"{_NAMESPACE}\nstrategy={strategy}\nstrategy_id={strategy_id}\n"
        f"code_hash={code_hash}\nconfig_hash={config_hash}\nnonce={nonce}\nexpires_at={expires_at}"
    )


def issue_challenge(conn: sqlite3.Connection, strategy_id: int, strategy: str, code_hash: str,
                    config_hash: str, *, now: datetime | None = None) -> dict[str, str]:
    """Create + persist a pending go-live challenge; return {nonce, challenge, expires_at}."""
    now = now or _now()
    nonce = secrets.token_hex(32)
    expires_at = (now + _TTL).isoformat()
    conn.execute(
        "INSERT INTO live_challenges(nonce, strategy_id, code_hash, config_hash, issued_at, "
        "expires_at, consumed_at) VALUES (?,?,?,?,?,?,NULL)",
        (nonce, strategy_id, code_hash, config_hash, now.isoformat(), expires_at),
    )
    conn.commit()
    return {"nonce": nonce, "expires_at": expires_at,
            "challenge": build_challenge(strategy, strategy_id, code_hash, config_hash, nonce,
                                         expires_at)}


def find_pending_challenge(conn: sqlite3.Connection, strategy_id: int, code_hash: str,
                           config_hash: str, *, now: datetime | None = None) -> sqlite3.Row | None:
    """The newest unconsumed, unexpired challenge matching the strategy + recomputed hashes."""
    now = now or _now()
    return conn.execute(
        "SELECT * FROM live_challenges WHERE strategy_id=? AND code_hash=? AND config_hash=? "
        "AND consumed_at IS NULL AND expires_at > ? ORDER BY issued_at DESC LIMIT 1",
        (strategy_id, code_hash, config_hash, now.isoformat()),
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
