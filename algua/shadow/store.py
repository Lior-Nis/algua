from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime

from algua.shadow.evaluate import ShadowResult

# The shadow_evaluations table is created by algua.registry.db.migrate (CREATE TABLE IF NOT EXISTS
# in _SCHEMA, schema v34). This module owns ONLY its read/write — it is deliberately separate from
# algua.execution.order_state (real/paper execution state) so a forward/live gate can never read an
# advisory shadow row by accident. It imports no broker, ledger, allocation, or cli module.


def record_shadow_evaluation(
    conn: sqlite3.Connection,
    *,
    challenger: str,
    champion: str | None,
    snapshot_id: str | None,
    timeframe: str,
    start: str,
    end: str,
    cash: float,
    universe: list[str],
    result: ShadowResult,
    code_hash: str,
    config_hash: str,
) -> int:
    """Append one advisory shadow-evaluation row and return its id. Records the full evaluation
    surface ({snapshot, timeframe, start, end, cash, universe} + code/config identity) so a later
    comparison can prove champion and challenger were scored on the SAME point-in-time inputs."""
    cur = conn.execute(
        "INSERT INTO shadow_evaluations("
        "challenger, champion, snapshot_id, timeframe, start, end, cash, universe, "
        "code_hash, config_hash, final_equity, total_return, ann_return, ann_volatility, "
        "sharpe, max_drawdown, n_bars, final_positions, equity_curve, recorded_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            challenger, champion, snapshot_id, timeframe, start, end, float(cash),
            json.dumps(universe), code_hash, config_hash,
            result.final_equity, result.total_return, result.ann_return, result.ann_volatility,
            result.sharpe, result.max_drawdown, result.n_bars,
            json.dumps(result.final_positions), json.dumps(result.equity_curve),
            datetime.now(UTC).isoformat(),
        ),
    )
    conn.commit()
    row_id = cur.lastrowid
    assert row_id is not None  # a successful INSERT always yields a rowid
    return int(row_id)


_COLUMNS = (
    "id", "challenger", "champion", "snapshot_id", "timeframe", "start", "end", "cash",
    "universe", "code_hash", "config_hash", "final_equity", "total_return", "ann_return",
    "ann_volatility", "sharpe", "max_drawdown", "n_bars", "final_positions", "equity_curve",
    "recorded_at",
)

_JSON_COLUMNS = ("universe", "final_positions", "equity_curve")


def latest_shadow_evaluation(conn: sqlite3.Connection, strategy: str) -> dict | None:
    """The most recent shadow evaluation where `strategy` was the CHALLENGER (the evaluated side),
    or None. JSON columns are parsed back to Python objects."""
    row = conn.execute(
        f"SELECT {', '.join(_COLUMNS)} FROM shadow_evaluations "
        "WHERE challenger = ? ORDER BY id DESC LIMIT 1",
        (strategy,),
    ).fetchone()
    if row is None:
        return None
    out = {c: row[c] for c in _COLUMNS}
    for c in _JSON_COLUMNS:
        out[c] = json.loads(out[c]) if out[c] is not None else None
    return out
