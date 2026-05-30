from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime

from algua.contracts.lifecycle import Actor, Stage


def _now() -> str:
    return datetime.now(UTC).isoformat()


class StrategyExists(ValueError):
    pass


class StrategyNotFound(LookupError):
    pass


@dataclass
class StrategyRecord:
    id: int
    name: str
    stage: Stage
    created_at: str
    updated_at: str


def _row_to_record(row: sqlite3.Row) -> StrategyRecord:
    return StrategyRecord(
        id=row["id"], name=row["name"], stage=Stage(row["stage"]),
        created_at=row["created_at"], updated_at=row["updated_at"],
    )


def add_strategy(conn: sqlite3.Connection, name: str) -> StrategyRecord:
    now = _now()
    try:
        cur = conn.execute(
            "INSERT INTO strategies(name, stage, created_at, updated_at) VALUES (?,?,?,?)",
            (name, Stage.IDEA.value, now, now),
        )
    except sqlite3.IntegrityError as exc:
        raise StrategyExists(name) from exc
    conn.execute(
        "INSERT INTO stage_transitions"
        "(strategy_id, from_stage, to_stage, actor, reason, created_at) VALUES (?,?,?,?,?,?)",
        (cur.lastrowid, None, Stage.IDEA.value, Actor.SYSTEM.value, "created", now),
    )
    conn.commit()
    return get_strategy(conn, name)


def get_strategy(conn: sqlite3.Connection, name: str) -> StrategyRecord:
    row = conn.execute("SELECT * FROM strategies WHERE name = ?", (name,)).fetchone()
    if row is None:
        raise StrategyNotFound(name)
    return _row_to_record(row)


def list_strategies(conn: sqlite3.Connection, stage: Stage | None = None) -> list[StrategyRecord]:
    if stage is None:
        rows = conn.execute("SELECT * FROM strategies ORDER BY id").fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM strategies WHERE stage = ? ORDER BY id", (stage.value,)
        ).fetchall()
    return [_row_to_record(r) for r in rows]


def list_transitions(conn: sqlite3.Connection, name: str) -> list[dict]:
    rec = get_strategy(conn, name)
    rows = conn.execute(
        "SELECT * FROM stage_transitions WHERE strategy_id = ? ORDER BY id", (rec.id,)
    ).fetchall()
    return [dict(r) for r in rows]


def transition(
    conn: sqlite3.Connection,
    name: str,
    to: Stage | str,
    actor: Actor | str,
    reason: str | None = None,
    code_hash: str | None = None,
    config_hash: str | None = None,
) -> StrategyRecord:
    from algua.registry.transitions import transition_strategy

    return transition_strategy(conn, name, to, actor, reason, code_hash, config_hash)


def apply_transition(
    conn: sqlite3.Connection,
    rec: StrategyRecord,
    to: Stage,
    actor: Actor,
    reason: str | None = None,
    code_hash: str | None = None,
    config_hash: str | None = None,
) -> StrategyRecord:
    now = _now()
    conn.execute(
        "UPDATE strategies SET stage = ?, updated_at = ? WHERE id = ?",
        (to.value, now, rec.id),
    )
    conn.execute(
        "INSERT INTO stage_transitions"
        "(strategy_id, from_stage, to_stage, actor, reason, code_hash, config_hash, created_at)"
        " VALUES (?,?,?,?,?,?,?,?)",
        (rec.id, rec.stage.value, to.value, actor.value, reason, code_hash, config_hash, now),
    )
    conn.commit()
    return get_strategy(conn, rec.name)
