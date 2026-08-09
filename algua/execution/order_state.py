from __future__ import annotations

import json
import re
import sqlite3
from datetime import UTC, datetime

from algua.live.paper_loop import PaperRunResult

# Alpaca client_order_id allows up to 128 chars; keep ours under that and strip anything outside
# [A-Za-z0-9_-] so a symbol or strategy name with odd characters can't produce an invalid id.
_COID_SANITIZE = re.compile(r"[^A-Za-z0-9_-]")


def client_order_id(strategy: str, decision_ts: datetime, symbol: str) -> str:
    """Deterministic Alpaca client_order_id for one (strategy, decision_ts, symbol). Identical
    inputs always produce the same id, so a retried submit (after a transient failure) or a re-run
    of the same tick reuses the id and Alpaca de-duplicates rather than double-filling (#18, #24).
    The decision timestamp is normalised to UTC so the id does not depend on the caller's tzinfo."""
    ts = decision_ts.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    raw = f"{strategy}-{ts}-{symbol}"
    return _COID_SANITIZE.sub("_", raw)[:128]


def persist_run(conn: sqlite3.Connection, result: PaperRunResult) -> None:
    """Persist a run's orders and fills. Fills link to their order by broker_order_id.

    A paper run is a full from-scratch replay, so it REPLACES this strategy's paper book
    (orders + fills) rather than appending — otherwise re-running a replay would double the
    persisted positions. Cross-run history lives in audit_log, not here. (Incremental /
    session semantics arrive with the real wall-clock paper adapter.)
    """
    conn.execute(
        "DELETE FROM paper_fills WHERE order_id IN "
        "(SELECT id FROM paper_orders WHERE strategy = ?)",
        (result.strategy,),
    )
    conn.execute("DELETE FROM paper_orders WHERE strategy = ?", (result.strategy,))

    now = datetime.now(UTC).isoformat()
    fills_by_order: dict[str, list] = {}
    for f in result.fills:
        fills_by_order.setdefault(f.broker_order_id, []).append(f)

    for record in result.orders:
        intent = record.intent
        # Read the broker id submit() returned rather than reconstructing sim-{seq} positionally,
        # so a skipped/"noop" submit can't shift the mapping (#30).
        broker_order_id = record.broker_order_id
        # A rejected fill (zero-qty) carries no shares; the order's status reflects whether any
        # shares actually executed, and only executing fills are persisted as fills.
        executed = [f for f in fills_by_order.get(broker_order_id, []) if f.qty != 0.0]
        # Derive status from the fills' own status rather than hardcoding "filled": a buy clamped
        # to available cash produces Fill.status="partial", which must be preserved here so callers
        # can distinguish fully-filled orders from cash-constrained ones.
        if any(f.status == "partial" for f in executed):
            status = "partial"
        elif executed:
            status = "filled"
        else:
            status = "rejected"
        cols = (
            "(strategy, symbol, side, target_weight,"
            " decision_ts, submitted_ts, status, broker_order_id)"
        )
        cur = conn.execute(
            f"INSERT INTO paper_orders{cols} VALUES (?,?,?,?,?,?,?,?)",
            (result.strategy, intent.symbol, intent.side.value, intent.target_weight,
             intent.decision_ts.isoformat(), now, status, broker_order_id),
        )
        order_row_id = cur.lastrowid
        for f in executed:
            conn.execute(
                "INSERT INTO paper_fills(order_id, symbol, qty, price, fill_ts) VALUES (?,?,?,?,?)",
                (order_row_id, f.symbol, f.qty, f.price, f.fill_ts.isoformat()),
            )
    conn.commit()


def count_orders(conn: sqlite3.Connection, strategy: str) -> int:
    """Number of persisted paper orders for a strategy (the `paper show` order count)."""
    return int(
        conn.execute(
            "SELECT COUNT(*) FROM paper_orders WHERE strategy = ?", (strategy,)
        ).fetchone()[0]
    )


def derive_positions(conn: sqlite3.Connection, strategy: str) -> dict[str, float]:
    rows = conn.execute(
        "SELECT f.symbol AS symbol, SUM(f.qty) AS qty FROM paper_fills f "
        "JOIN paper_orders o ON o.id = f.order_id WHERE o.strategy = ? GROUP BY f.symbol",
        (strategy,),
    ).fetchall()
    return {r["symbol"]: float(r["qty"]) for r in rows if float(r["qty"]) != 0.0}


def record_submitted_order(
    conn: sqlite3.Connection, strategy: str, symbol: str, side: str,
    target_weight: float, decision_ts: str | None, broker_order_id: str,
    *, strategy_id: int,
) -> None:
    """Persist ONE accepted live order IMMEDIATELY after the broker accepts it, so a mid-tick death
    can never leave Alpaca holding an order the DB never recorded (#18). Each row commits on its own
    rather than being batched after the whole loop.

    Idempotent on (strategy, broker_order_id): a crash/retry or a duplicate Alpaca client_order_id
    path that re-returns the SAME broker order leaves the existing row untouched instead of writing
    a duplicate (the unique index enforces this; INSERT OR IGNORE makes it a no-op)."""
    conn.execute(
        "INSERT OR IGNORE INTO paper_orders"
        "(strategy, symbol, side, target_weight, decision_ts, submitted_ts,"
        " status, broker_order_id, strategy_id) VALUES (?,?,?,?,?,?,?,?,?)",
        (strategy, symbol, side, target_weight, decision_ts,
         datetime.now(UTC).isoformat(), "submitted", broker_order_id, strategy_id),
    )
    conn.commit()


def get_peak_equity(conn: sqlite3.Connection, strategy: str) -> float | None:
    row = conn.execute(
        "SELECT peak_equity FROM strategy_peaks WHERE strategy = ?", (strategy,)
    ).fetchone()
    return float(row["peak_equity"]) if row is not None else None


def update_peak_equity(conn: sqlite3.Connection, strategy: str, equity: float) -> float:
    """Persist the running peak equity for a strategy (the drawdown denominator across ticks) and
    return the new peak. The peak only ever ratchets up; a tick's equity below it is a drawdown the
    breaker can act on (#27)."""
    prior = get_peak_equity(conn, strategy)
    peak = equity if prior is None else max(prior, equity)
    conn.execute(
        "INSERT INTO strategy_peaks(strategy, peak_equity, updated_at) VALUES (?,?,?) "
        "ON CONFLICT(strategy) DO UPDATE SET peak_equity=excluded.peak_equity, "
        "updated_at=excluded.updated_at",
        (strategy, peak, datetime.now(UTC).isoformat()),
    )
    conn.commit()
    return peak


def clear_peak_equity(conn: sqlite3.Connection, strategy: str) -> None:
    """Drop a strategy's persisted peak so the next tick re-bases the high-water mark to current
    equity. Called on resume after a trip: without it, a strategy halted by the drawdown breaker
    and flattened to cash would re-trip every tick against its stale pre-loss peak (#27).

    Semantics: the peak is thus the high-water mark *since the last tripped resume*, not lifetime.
    A manual `paper kill` then `resume` also re-bases it, so a manual halt/resume can lower the
    drawdown denominator to current equity — intentional (the operator is re-baselining)."""
    conn.execute("DELETE FROM strategy_peaks WHERE strategy = ?", (strategy,))
    conn.commit()


def clear_all_peaks(conn: sqlite3.Connection) -> None:
    """Wipe every strategy's persisted peak — used by the global resume-all after the whole account
    is flattened, so each strategy re-bases its drawdown high-water mark on its next tick (#27)."""
    conn.execute("DELETE FROM strategy_peaks")
    conn.commit()


def get_nav_peak(conn: sqlite3.Connection, strategy: str) -> float | None:
    row = conn.execute(
        "SELECT peak FROM live_nav_peaks WHERE strategy = ?", (strategy,)
    ).fetchone()
    return float(row["peak"]) if row is not None else None


def update_nav_peak(conn: sqlite3.Connection, strategy: str, nav: float) -> float:
    """Persist the running per-strategy NAV peak (the live drawdown denominator) and return it.
    Ratchets up only; a tick's NAV below it is the drawdown the breaker acts on."""
    prior = get_nav_peak(conn, strategy)
    peak = nav if prior is None else max(prior, nav)
    conn.execute(
        "INSERT INTO live_nav_peaks(strategy, peak, updated_ts) VALUES (?,?,?) "
        "ON CONFLICT(strategy) DO UPDATE SET peak=excluded.peak, updated_ts=excluded.updated_ts",
        (strategy, peak, datetime.now(UTC).isoformat()),
    )
    conn.commit()
    return peak


def clear_nav_peak(conn: sqlite3.Connection, strategy: str) -> None:
    conn.execute("DELETE FROM live_nav_peaks WHERE strategy = ?", (strategy,))
    conn.commit()


def clear_all_nav_peaks(conn: sqlite3.Connection) -> None:
    """Wipe every strategy's NAV peak — the live counterpart of clear_all_peaks, for resume-all."""
    conn.execute("DELETE FROM live_nav_peaks")
    conn.commit()


_VALID_LANES = frozenset({"paper", "live"})
_VALID_CLOCK_SOURCES = frozenset({"broker", "local"})


def record_tick_snapshot(
    conn: sqlite3.Connection, strategy: str, *, tick_ts: str, decision_ts: str | None,
    equity: float, peak_equity: float | None, positions: dict[str, float], n_submitted: int,
    reconcile_ok: bool,
    lane: str, strategy_id: int, code_hash: str, config_hash: str,
    dependency_hash: str | None, account_id: str, cash: float, clock_source: str,
) -> None:
    """Append one completed-tick snapshot (equity + positions) for a strategy — the per-tick
    operability/equity-curve record read by `paper show`.

    ``lane`` must be one of ``("paper", "live")`` and ``clock_source`` must be one of
    ``("broker", "local")``. These are enforced here rather than via a DB CHECK constraint
    because SQLite ALTER TABLE cannot add CHECK constraints to existing tables — writer
    discipline is the enforcement layer; the forward gate rejects NULL/invalid values
    fail-closed. Legacy rows (NULL) are inadmissible by design."""
    if lane not in _VALID_LANES:
        raise ValueError(f"lane must be one of {sorted(_VALID_LANES)!r}, got {lane!r}")
    if clock_source not in _VALID_CLOCK_SOURCES:
        raise ValueError(
            f"clock_source must be one of {sorted(_VALID_CLOCK_SOURCES)!r}, got {clock_source!r}"
        )
    conn.execute(
        "INSERT INTO tick_snapshots(strategy, tick_ts, decision_ts, equity, peak_equity, "
        "positions, n_submitted, reconcile_ok, lane, strategy_id, code_hash, config_hash, "
        "dependency_hash, account_id, cash, clock_source, recorded_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (strategy, tick_ts, decision_ts, equity, peak_equity, json.dumps(positions),
         n_submitted, 1 if reconcile_ok else 0,
         lane, strategy_id, code_hash, config_hash, dependency_hash,
         account_id, cash, clock_source, datetime.now(UTC).isoformat()),
    )
    conn.commit()


def latest_tick_snapshot(conn: sqlite3.Connection, strategy: str) -> dict | None:
    """The most recent tick snapshot for a strategy (positions parsed back to a dict), or None."""
    row = conn.execute(
        "SELECT tick_ts, decision_ts, equity, peak_equity, positions, n_submitted, reconcile_ok, "
        "lane, strategy_id, code_hash, config_hash, dependency_hash, account_id, cash, "
        "clock_source, recorded_at "
        "FROM tick_snapshots WHERE strategy = ? ORDER BY id DESC LIMIT 1", (strategy,)
    ).fetchone()
    if row is None:
        return None
    return {
        "tick_ts": row["tick_ts"], "decision_ts": row["decision_ts"], "equity": row["equity"],
        "peak_equity": row["peak_equity"], "positions": json.loads(row["positions"]),
        "n_submitted": row["n_submitted"], "reconcile_ok": bool(row["reconcile_ok"]),
        "lane": row["lane"], "strategy_id": row["strategy_id"],
        "code_hash": row["code_hash"], "config_hash": row["config_hash"],
        "dependency_hash": row["dependency_hash"], "account_id": row["account_id"],
        "cash": row["cash"], "clock_source": row["clock_source"],
        "recorded_at": row["recorded_at"],
    }


def _parse_snapshot_ts(value: str) -> datetime | None:
    """Parse a persisted tick_ts to an aware-UTC datetime; naive is assumed UTC. Returns None on
    unparseable input — tick_ts writers accept arbitrary strings, so readers must never trust the
    column to sort/compare as SQL text."""
    try:
        dt = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def tick_snapshot_series(
    conn: sqlite3.Connection,
    strategy_id: int,
    strategy_name: str,
    *,
    lane: str | None = None,
    since: datetime | None = None,
    limit: int = 500,
) -> dict:
    """Per-lane tick-snapshot time series for one strategy (the equity-curve read behind
    ``fleet series``), event-time-ordered ascending, newest-``limit`` per lane.

    Data rows are keyed by ``strategy_id`` (never by name — names can be reused); legacy pre-v21
    rows predate the ``strategy_id``/``lane`` columns, so they are only findable by name and are
    surfaced as an exclusion COUNT, never as series rows. All ordering/filtering happens in Python
    on parsed timestamps — no SQL text comparison anywhere (see ``_parse_snapshot_ts``). ``since``
    filtering happens BEFORE the newest-N cut: the result is the newest N of the filtered interval.

    A lane not requested (when ``lane`` is given) is ``None`` in ``series``/``truncated``; a
    requested lane with no rows is ``[]``/``False``.
    """
    # Explicit BEGIN DEFERRED: a plain `with conn:` does not pin a read snapshot, so the series
    # SELECT and the legacy COUNT could otherwise straddle a concurrent writer's commit. Never
    # IMMEDIATE — this is a reader and must not take the write lock.
    conn.execute("BEGIN DEFERRED")
    try:
        raw = conn.execute(
            "SELECT id, tick_ts, recorded_at, equity, peak_equity, reconcile_ok, lane "
            "FROM tick_snapshots WHERE strategy_id = ?",
            (strategy_id,),
        ).fetchall()
        n_legacy_excluded = int(
            conn.execute(
                "SELECT COUNT(*) FROM tick_snapshots WHERE strategy = ? "
                "AND (strategy_id IS NULL OR lane IS NULL)",
                (strategy_name,),
            ).fetchone()[0]
        )
        conn.commit()
    except BaseException:
        conn.rollback()
        raise

    lanes = tuple(sorted(_VALID_LANES)) if lane is None else (lane,)
    per_lane: dict[str, list[tuple[datetime, int, dict]]] = {ln: [] for ln in lanes}
    n_unparseable = 0
    n_invalid_lane = 0
    for row in raw:
        row_lane = row["lane"]
        if row_lane is None:
            # A strategy_id-bearing row with NULL lane is in the legacy bucket by definition
            # (already counted in n_legacy_excluded above) — skip without double-counting.
            continue
        parsed = _parse_snapshot_ts(row["tick_ts"])
        if parsed is None:
            n_unparseable += 1
            continue
        if row_lane not in _VALID_LANES:
            # Lane discipline is writer-enforced only (no DB CHECK constraint) — a raw-write
            # fabrication must be surfaced as a count, never silently mixed into a lane.
            n_invalid_lane += 1
            continue
        if since is not None and parsed < since:
            continue
        if row_lane not in per_lane:
            continue  # valid lane, just not requested by the lane filter
        per_lane[row_lane].append((parsed, row["id"], {
            "id": row["id"], "tick_ts": row["tick_ts"], "recorded_at": row["recorded_at"],
            "equity": row["equity"], "peak_equity": row["peak_equity"],
            "reconcile_ok": bool(row["reconcile_ok"]),
        }))

    series: dict[str, list[dict] | None] = {"paper": None, "live": None}
    truncated: dict[str, bool | None] = {"paper": None, "live": None}
    for ln in lanes:
        entries = sorted(per_lane[ln], key=lambda e: (e[0], e[1]))
        truncated[ln] = len(entries) > limit
        series[ln] = [e[2] for e in entries[-limit:]]
    return {
        "series": series,
        "truncated": truncated,
        "n_legacy_excluded": n_legacy_excluded,
        "n_unparseable": n_unparseable,
        "n_invalid_lane": n_invalid_lane,
    }


def recent_orders(conn: sqlite3.Connection, strategy: str, limit: int = 10) -> list[dict]:
    """The most recent paper_orders rows for a strategy, newest first."""
    rows = conn.execute(
        "SELECT symbol, side, status, broker_order_id, submitted_ts FROM paper_orders "
        "WHERE strategy = ? ORDER BY id DESC LIMIT ?", (strategy, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def count_venue_orders(conn: sqlite3.Connection, strategy: str) -> int:
    """Number of paper_venue_orders rows for a strategy (the venue-lane order count)."""
    return int(
        conn.execute(
            "SELECT COUNT(*) FROM paper_venue_orders WHERE strategy = ?", (strategy,)
        ).fetchone()[0]
    )


def recent_venue_orders(conn: sqlite3.Connection, strategy: str, limit: int = 10) -> list[dict]:
    """The most recent paper_venue_orders rows for a strategy, newest first."""
    rows = conn.execute(
        "SELECT symbol, side, status, broker_order_id, submitted_ts FROM paper_venue_orders "
        "WHERE strategy = ? ORDER BY id DESC LIMIT ?", (strategy, limit),
    ).fetchall()
    return [dict(r) for r in rows]
