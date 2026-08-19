"""One single-use signed-challenge lifecycle (spec §4.4) behind both signing namespaces.

`live_gate` (algua-go-live) and `human_actor` (algua-human-actor, #329) were line-for-line
parallel issue/find/consume stacks over their own tables; each is now a `ChallengeSpec` +
thin wrappers. The payload format (`namespace\\nk=v…\\nnonce=…\\nexpires_at=…`) is
byte-identical to the previous per-module builders, so existing enrolled keys and any
in-flight signed challenges verify unchanged. Signature verification itself stays in
`live_gate.verify_signature` — this module owns only nonce issuance, matching, and
single-use consumption. Column matching uses SQLite `IS` uniformly: identical to `=` for
non-NULL values, and NULL-correct for nullable identity columns (dependency_hash)."""
from __future__ import annotations

import secrets
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class ChallengeSpec:
    """One namespace's challenge shape: its table, the ordered payload lines the human
    signs, and the stored/matched DB columns (a subset of the payload keys)."""

    table: str
    namespace: str
    payload_fields: tuple[str, ...]
    column_fields: tuple[str, ...]
    ttl: timedelta = timedelta(minutes=10)


def build_payload(
    spec: ChallengeSpec, values: dict[str, object], nonce: str, expires_at: str
) -> str:
    """The exact bytes the operator signs. ONE definition used to both issue and verify so
    the two can never drift (each wrapper module passes the same spec+values to both)."""
    lines = [spec.namespace] + [f"{k}={values[k]}" for k in spec.payload_fields]
    lines += [f"nonce={nonce}", f"expires_at={expires_at}"]
    return "\n".join(lines)


def issue(
    conn: sqlite3.Connection,
    spec: ChallengeSpec,
    values: dict[str, object],
    *,
    now: datetime | None = None,
) -> dict[str, str]:
    """Create + persist a pending challenge; return {nonce, expires_at, challenge}."""
    now = now or _now()
    nonce = secrets.token_hex(32)
    expires_at = (now + spec.ttl).isoformat()
    cols = ", ".join(spec.column_fields)
    marks = ", ".join("?" for _ in spec.column_fields)
    conn.execute(
        f"INSERT INTO {spec.table}(nonce, {cols}, issued_at, expires_at, consumed_at)"
        f" VALUES (?, {marks}, ?, ?, NULL)",
        (nonce, *[values[k] for k in spec.column_fields], now.isoformat(), expires_at),
    )
    conn.commit()
    return {
        "nonce": nonce,
        "expires_at": expires_at,
        "challenge": build_payload(spec, values, nonce, expires_at),
    }


def find_pending(
    conn: sqlite3.Connection,
    spec: ChallengeSpec,
    values: dict[str, object],
    *,
    now: datetime | None = None,
) -> sqlite3.Row | None:
    """Newest unconsumed, unexpired challenge matching EVERY bound column."""
    now = now or _now()
    where = " AND ".join(f"{c} IS ?" for c in spec.column_fields)
    return conn.execute(
        f"SELECT * FROM {spec.table} WHERE {where} AND consumed_at IS NULL"
        f" AND expires_at > ? ORDER BY issued_at DESC LIMIT 1",
        (*[values[k] for k in spec.column_fields], now.isoformat()),
    ).fetchone()


def consume(
    conn: sqlite3.Connection, spec: ChallengeSpec, nonce: str, *, now: datetime | None = None
) -> bool:
    """Mark a challenge consumed (single-use). False if already consumed / missing — a lost
    consume race fails closed at the caller."""
    now = now or _now()
    cur = conn.execute(
        f"UPDATE {spec.table} SET consumed_at=? WHERE nonce=? AND consumed_at IS NULL",
        (now.isoformat(), nonce),
    )
    conn.commit()
    return cur.rowcount > 0
