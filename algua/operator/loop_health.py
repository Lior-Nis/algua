"""Machine-liveness rollup for the autonomous loops (`algua ops status`).

The fleet rollup (``execution/fleet_health.py``) answers "is this STRATEGY being ticked". Nothing
answered "is the MACHINE still running" — and on 2026-08-15 that gap cost five days: the hourly
``algua-research`` loop had been exiting non-zero on a Codex usage limit since 2026-08-14 and the
monitor was green throughout, because a dead research loop produces no strategy rows to be unhealthy
ABOUT. The top of the funnel can stop entirely without a single strategy looking wrong.

Pure reads of artifacts the loops ALREADY write — no systemd, no journalctl, no subprocess. Coupling
the monitor to the host's init system was considered and rejected: it would tie a read-only domain
view to a specific service manager, and the durable digests are both richer (they see INSIDE a run)
and portable.

Cadence is measured per loop against how that loop is actually scheduled — WALL-CLOCK for the
research and merge-back loops (plain systemd timers, hourly / half-hourly, market-independent) and
COMPLETED MARKET SESSIONS for the paper operator (session-gated, so a weekend gap is not a fault).
Using one clock for all three would either false-alarm every weekend or hide a day-long research
outage.

Fail-closed throughout: a missing, unparseable, or wrong-shaped artifact is never ``ok``. ``ok``
requires positive, fresh, parseable evidence — the same rule ``strategy_health`` applies to ticks.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from algua.contracts.types import SessionSpanCalendar

__all__ = [
    "LOOPS",
    "MERGEBACK_STALE_AFTER_S",
    "PAPER_STALE_AFTER_SESSIONS",
    "RESEARCH_STALE_AFTER_S",
    "loop_status",
]

# Wall-clock staleness bounds, each ~3x the loop's configured timer period so an ordinary skipped
# firing (a reboot, a long run) is not an alert but a stopped timer is.
RESEARCH_STALE_AFTER_S = 3 * 60 * 60.0  # algua-research.timer: hourly
MERGEBACK_STALE_AFTER_S = 90 * 60.0  # algua-mergeback-drain.timer: half-hourly

# The paper operator is session-gated (one cycle per COMPLETED session), so its cadence is counted
# in sessions — never wall-clock, or every weekend reads as a dead loop.
PAPER_STALE_AFTER_SESSIONS = 2

LOOPS = ("research", "paper", "mergeback")

# Worst-first, mirroring fleet_health's severity convention (LOWER int = WORSE).
_SEVERITY = {"failing": 0, "rate_limited": 1, "stale": 2, "unknown": 3, "idle": 4, "ok": 5}
_ALERTING = frozenset({"failing", "rate_limited", "stale", "unknown"})


def _parse_utc(value: Any) -> datetime | None:
    """ISO-8601 -> aware UTC, or None on anything unparseable INCLUDING a tz-naive string.

    Mirrors ``fleet_health._parse_utc``: every legitimate writer stamps an explicit offset, so a
    naive value can only be a fabrication and is rejected rather than guessed at.
    """
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed.astimezone(UTC) if parsed.tzinfo is not None else None


def _read_json(path: Path) -> tuple[Any, str | None]:
    """``(payload, None)`` or ``(None, reason)`` — a read/parse fault is data, never an exception.

    One corrupt artifact must degrade its own loop to ``unknown`` and leave the other two
    readable, exactly as ``_safe_latest_tick`` protects the fleet view from one bad row.
    """
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except FileNotFoundError:
        return None, "missing"
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _read_digest_tail(path: Path, limit: int) -> tuple[list[dict], str | None]:
    """The newest ``limit`` parseable records of the JSONL run digest, oldest-first.

    Unparseable lines are SKIPPED rather than fatal: the digest is appended by a shell driver, so a
    torn final line (killed mid-append) must not blind the whole rollup.
    """
    try:
        raw = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return [], "missing"
    except (OSError, UnicodeDecodeError) as exc:
        return [], f"{type(exc).__name__}: {exc}"
    records: list[dict] = []
    for line in reversed(raw):
        if len(records) >= limit:
            break
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    records.reverse()
    return records, None


def _run_failed(record: dict) -> bool:
    """A digest record that did not complete cleanly.

    ``exit_code`` is authoritative; ``timed_out`` is carried separately because the driver records
    a timeout with its own flag and the exit code of the killed process.
    """
    if bool(record.get("timed_out")):
        return True
    code = record.get("exit_code")
    return not (isinstance(code, int) and code == 0)


def _digest_stamp(record: dict) -> datetime | None:
    """The run's instant. Prefers an explicit ISO field, falling back to the ``YYYYmmdd-HHMMSS``
    ``stamp`` the driver names each run branch with (its only guaranteed time carrier).

    The stamp is LOCAL time — ``run-research-loop.sh`` builds it with a bare ``date
    +%Y%m%d-%H%M%S``, no ``-u``. Reading it as UTC shifts every run by the host's offset and, east
    of Greenwich, lands the newest run in the FUTURE (which would then read as perfectly fresh
    forever). ``astimezone()`` on a naive datetime interprets it in local time, which is exactly
    the driver's convention.
    """
    for key in ("finished_at", "started_at", "recorded_at"):
        parsed = _parse_utc(record.get(key))
        if parsed is not None:
            return parsed
    stamp = record.get("stamp")
    if isinstance(stamp, str):
        try:
            return datetime.strptime(stamp, "%Y%m%d-%H%M%S").astimezone(UTC)
        except ValueError:
            return None
    return None


def _research_health(data_dir: Path, now: datetime, digest_limit: int) -> dict[str, Any]:
    """The Codex-driven research loop, from its durable per-run digest.

    ``rate_limited`` is reported as its OWN health rather than collapsed into ``failing``: an
    exhausted account quota is not a broken loop, it is a loop that cannot run until the quota
    resets, and the two want completely different operator responses (wait/buy credits vs. debug).
    """
    records, error = _read_digest_tail(data_dir / "research-runs.jsonl", digest_limit)
    if not records:
        return {
            "health": "unknown",
            "detail": error or "no parseable run digest",
            "last_run_at": None,
            "last_ok_at": None,
            "consecutive_failures": None,
        }

    newest = records[-1]
    last_run_at = _digest_stamp(newest)
    last_ok_at = None
    for record in reversed(records):
        if not _run_failed(record):
            last_ok_at = _digest_stamp(record)
            break
    consecutive_failures = 0
    for record in reversed(records):
        if not _run_failed(record):
            break
        consecutive_failures += 1

    if bool(newest.get("rate_limited")):
        health = "rate_limited"
        detail = "provider usage limit reached"
    elif _run_failed(newest):
        health = "failing"
        detail = "timed out" if newest.get("timed_out") else f"exit {newest.get('exit_code')!r}"
    elif last_run_at is None:
        # Ran clean but the record carries no usable instant: we cannot prove freshness.
        health, detail = "unknown", "run digest has no parseable timestamp"
    elif last_run_at > now:
        # A future instant cannot be evidence of freshness — a clock skew or a mis-stamped record
        # would otherwise read as permanently fresh. Fail closed, exactly like a future tick_ts.
        health, detail = "unknown", "newest run is stamped in the future"
    elif (now - last_run_at).total_seconds() > RESEARCH_STALE_AFTER_S:
        health, detail = "stale", "no run within the expected cadence"
    else:
        health, detail = "ok", None

    return {
        "health": health,
        "detail": detail,
        "last_run_at": last_run_at.isoformat() if last_run_at else None,
        "last_ok_at": last_ok_at.isoformat() if last_ok_at else None,
        "consecutive_failures": consecutive_failures,
        "window": len(records),
        "last_outcome": newest.get("outcome"),
        "last_branch": newest.get("branch"),
    }


def _paper_health(data_dir: Path, now: datetime, calendar: SessionSpanCalendar) -> dict[str, Any]:
    """The session-gated paper operator, from the session marker it writes after each cycle.

    Staleness is COMPLETED SESSIONS, not wall-clock: the loop is supposed to be silent on a
    weekend or holiday, and a wall-clock bound would page every Saturday.
    """
    payload, error = _read_json(data_dir / "operator_sessions.json")
    entry = payload.get("paper") if isinstance(payload, dict) else None
    if not isinstance(entry, dict):
        return {
            "health": "unknown" if error != "missing" else "idle",
            "detail": error or "no paper entry in the session marker",
            "last_run_at": None,
            "last_rc": None,
            "session": None,
        }

    recorded_at = _parse_utc(entry.get("recorded_at"))
    rc = entry.get("last_rc", entry.get("rc"))
    staleness_sessions: int | None = None
    if recorded_at is not None and recorded_at <= now:
        try:
            staleness_sessions = calendar.sessions_between_instants(recorded_at, now)
        except Exception:  # noqa: BLE001 — any calendar mapping failure fails closed to stale
            staleness_sessions = PAPER_STALE_AFTER_SESSIONS + 1

    if not (isinstance(rc, int) and rc == 0):
        health, detail = "failing", f"last cycle exited {rc!r}"
    elif recorded_at is None:
        health, detail = "unknown", "session marker has no parseable timestamp"
    elif staleness_sessions is not None and staleness_sessions > PAPER_STALE_AFTER_SESSIONS:
        health, detail = "stale", f"{staleness_sessions} completed sessions since the last cycle"
    else:
        health, detail = "ok", None

    return {
        "health": health,
        "detail": detail,
        "last_run_at": recorded_at.isoformat() if recorded_at else None,
        "last_rc": rc if isinstance(rc, int) else None,
        "session": entry.get("session"),
        "staleness_sessions": staleness_sessions,
    }


def _mergeback_health(data_dir: Path, now: datetime) -> dict[str, Any]:
    """The merge-back queue: depth, age of the oldest item, and repeated-attempt wedging.

    A queue that is merely deep is fine — the drainer is half-hourly. A queue whose OLDEST item has
    outlived several drain cycles is wedged, which is the failure this exists to surface.
    """
    payload, error = _read_json(data_dir / "mergeback-queue.json")
    if payload is None:
        # A queue file is only created once something is enqueued: absent == nothing queued.
        if error == "missing":
            return {"health": "idle", "detail": "queue empty", "queue_depth": 0,
                    "max_attempts": 0, "oldest_enqueued_at": None}
        return {"health": "unknown", "detail": error, "queue_depth": None,
                "max_attempts": None, "oldest_enqueued_at": None}

    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, dict):
        return {"health": "unknown", "detail": "queue file has no items mapping",
                "queue_depth": None, "max_attempts": None, "oldest_enqueued_at": None}
    rows = [row for row in items.values() if isinstance(row, dict)]
    if not rows:
        return {"health": "idle", "detail": "queue empty", "queue_depth": 0,
                "max_attempts": 0, "oldest_enqueued_at": None}

    attempts = [int(r["attempts"]) for r in rows if isinstance(r.get("attempts"), int)]
    stamps = [dt for dt in (_parse_utc(r.get("enqueued_at")) for r in rows) if dt is not None]
    oldest = min(stamps) if stamps else None

    if oldest is not None and (now - oldest).total_seconds() > MERGEBACK_STALE_AFTER_S:
        health = "stale"
        detail = "oldest item has outlived several drain cycles"
    else:
        health, detail = "ok", None

    return {
        "health": health,
        "detail": detail,
        "queue_depth": len(rows),
        "max_attempts": max(attempts) if attempts else 0,
        "oldest_enqueued_at": oldest.isoformat() if oldest else None,
    }


def loop_status(
    data_dir: Path, calendar: SessionSpanCalendar, *, now: datetime, digest_limit: int = 50
) -> dict[str, Any]:
    """Every autonomous loop's liveness, worst-first, with an overall ``ok`` verdict.

    ``ok`` is false iff ANY loop is alerting (``failing``/``rate_limited``/``stale``/``unknown``).
    ``idle`` is explicitly NOT alerting — an empty merge-back queue or a never-run operator is a
    quiet system, not a broken one.
    """
    loops = {
        "research": _research_health(data_dir, now, digest_limit),
        "paper": _paper_health(data_dir, now, calendar),
        "mergeback": _mergeback_health(data_dir, now),
    }
    alerting = sorted(
        (name for name, row in loops.items() if row["health"] in _ALERTING),
        key=lambda name: (_SEVERITY.get(loops[name]["health"], 99), name),
    )
    return {
        "ok": not alerting,
        "checked_at": now.isoformat(),
        "alerting": alerting,
        "loops": loops,
    }
