"""Testable scheduling core for the always-on paper operator (#486).

Pure logic over an injected :class:`~algua.calendar.market_calendar.MarketCalendar` + a file marker
+ a couple of ``flock`` helpers; imports ``algua.calendar`` + stdlib only (the git dir for the run
lock is resolved by the CLI and passed in, so this module never shells out).

Three concerns:

* :func:`target_session` — which COMPLETED XNYS session an operator run should act on as of ``now``.
  A past-the-horizon instant (the calendar has RUN OUT) raises :class:`CalendarOutOfBounds` — a real
  operational anomaly the operator must be told about, NOT a silent no-op; a genuine
  before-first-session instant still returns ``None`` (benign bring-up).
* :class:`SessionMarker` + :func:`session_gate` — once-per-session idempotency. The marker records
  the last session each job ran (an enriched audit entry binding the FULL canonical argv). Reads
  FAIL CLOSED: a present-but-corrupt marker raises :class:`MarkerCorrupt` so the operator refuses to
  run rather than silently re-trading; only an ABSENT marker is treated as "no record → run".
* :func:`operator_run_lock` — a non-blocking, git-dir-anchored, process-level run lock that
  serializes wrapped operator runs, writing holder metadata into its body so a wedged holder is
  surfaced (:class:`OperatorLockHeld` carries the parsed holder) rather than silently no-op'd.
"""

from __future__ import annotations

import fcntl
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from algua.calendar.market_calendar import MarketCalendar

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = [
    "CalendarOutOfBounds",
    "MarkerCorrupt",
    "OperatorLockHeld",
    "SessionGateDecision",
    "SessionMarker",
    "operator_run_lock",
    "session_gate",
    "target_session",
]

_MARKER_NAME = "operator_sessions.json"
_MARKER_LOCK_NAME = "operator_sessions.lock"


class CalendarOutOfBounds(Exception):
    """``now`` is past the calendar's precomputed horizon — the calendar needs a refresh. Distinct
    from an ordinary weekend/holiday gap (a benign no-op) or a before-first-session bring-up."""


class MarkerCorrupt(Exception):
    """The session-marker file is present but unparseable / wrong-shaped / carries a non-ISO
    session value — the operator can no longer trust its idempotency state and must fail closed."""


class OperatorLockHeld(Exception):
    """The git-dir-anchored ``operator.lock`` is already held by a live operator run.

    ``holder`` carries the parsed lock-file body (``{"pid","job","started_at","host"}``) or ``None``
    if the metadata is missing/garbled (a holder that died mid-write)."""

    def __init__(self, holder: dict | None) -> None:
        self.holder = holder
        super().__init__("operator.lock is held")


def target_session(now: datetime, calendar: MarketCalendar) -> date | None:
    """The most-recent COMPLETED XNYS session as of ``now``.

    A naive ``now`` is treated as UTC. If ``now`` falls before the close of its own session, that
    session has not completed yet, so the prior session is the target.

    A ``MinuteOutOfBounds`` from the calendar is disambiguated against the calendar's horizon:
    ``now`` before the first minute is a benign before-first-session bring-up (returns ``None``);
    ``now`` past the last minute means the calendar has RUN OUT — an operational anomaly — so
    :class:`CalendarOutOfBounds` is raised for the caller to alert on.
    """
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    try:
        sess = calendar.session_of_instant(now)
    except Exception as exc:  # noqa: BLE001 - disambiguate out-of-range below; re-raise the rest
        if type(exc).__name__ == "MinuteOutOfBounds" or isinstance(exc, ValueError):
            first_minute, _last_minute = calendar.bounds()
            if now < first_minute:
                return None  # before the calendar's first session — benign bring-up
            raise CalendarOutOfBounds(
                f"now={now.isoformat()} is past the calendar horizon; refresh the exchange calendar"
            ) from exc
        raise
    if now < calendar.session_close(sess):
        return calendar.previous_session(sess)
    return sess


def _read_lock_holder(lock_path: Path) -> dict | None:
    """Recover the holder metadata from the lock-file body without taking the lock (flock is
    advisory, so a read needs no lock). Returns ``None`` on a missing/empty/garbled body."""
    try:
        raw = lock_path.read_text().strip()
    except OSError:
        return None
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


@contextmanager
def operator_run_lock(
    lock_path: Path, *, job: str, host: str, pid: int
) -> Iterator[None]:
    """Non-blocking process-level run lock, anchored at the caller-supplied ``lock_path`` (the
    per-worktree git dir's ``operator.lock``, §D2/round-3 fix #2).

    On acquisition the holder's identity (``{"pid","job","started_at","host"}``) is written into the
    lock-file body and fsync'd, so a wedged holder can be recovered on contention. On
    ``BlockingIOError`` the body is read back (read-only, no lock) and :class:`OperatorLockHeld` is
    raised carrying the parsed holder. The body is truncated and the lock released in a ``finally``.
    The kernel releases the flock on holder death, so a hard kill never wedges the next fire.
    """
    handle = open(lock_path, "a+")  # noqa: SIM115 — released in the finally below
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError) as exc:
        handle.close()
        raise OperatorLockHeld(_read_lock_holder(lock_path)) from exc
    try:
        body = json.dumps(
            {"pid": pid, "job": job, "started_at": datetime.now(UTC).isoformat(), "host": host}
        )
        handle.seek(0)
        handle.truncate()
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())
        yield
    finally:
        try:
            handle.seek(0)
            handle.truncate()
            handle.flush()
        except OSError:
            pass
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


class SessionMarker:
    """A crash-safe, per-job once-per-session marker persisted as a single JSON map.

    The file at ``<dir>/operator_sessions.json`` maps ``job -> {"session","recorded_at","command",
    "rc","host","pid"}`` — ``command`` is the FULL canonical argv actually run (round-3 fix #4), so
    the marker records exactly what marked the session done. Reads FAIL CLOSED: an absent file/job
    is ``None`` (benign — run), but a present-but-corrupt file raises :class:`MarkerCorrupt`. Writes
    are atomic + fsync-durable (tmp -> fsync(tmp_fd) -> os.replace -> fsync(dir_fd)) under a
    blocking ``flock`` on a dedicated ``operator_sessions.lock``, preserving other jobs' entries.
    """

    def __init__(self, directory: Path) -> None:
        self._dir = Path(directory)
        self._path = self._dir / _MARKER_NAME
        self._lock_path = self._dir / _MARKER_LOCK_NAME

    def _read_map(self) -> dict:
        """The whole marker map. Absent file → ``{}``; present-but-unparseable → MarkerCorrupt."""
        try:
            raw = self._path.read_text()
        except FileNotFoundError:
            return {}
        except OSError as exc:
            raise MarkerCorrupt(f"marker unreadable: {exc}") from exc
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            raise MarkerCorrupt(f"marker is not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise MarkerCorrupt(f"marker root is {type(data).__name__}, expected object")
        return data

    @staticmethod
    def _parse_session(value: str) -> date:
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise MarkerCorrupt(f"marker session {value!r} is not an ISO date") from exc

    def last_session(self, job: str) -> date | None:
        """The recorded session for ``job``: ``None`` for an absent file/job (benign), the recorded
        date for a valid entry (object or legacy bare-string), else :class:`MarkerCorrupt`."""
        entry = self._read_map().get(job)
        if entry is None:
            return None
        if isinstance(entry, str):  # legacy bare-string value — forward tolerance
            return self._parse_session(entry)
        if isinstance(entry, dict):
            sess = entry.get("session")
            if not isinstance(sess, str):
                raise MarkerCorrupt(f"marker entry for {job!r} has no ISO session")
            return self._parse_session(sess)
        raise MarkerCorrupt(f"marker entry for {job!r} is {type(entry).__name__}")

    @contextmanager
    def _marker_lock(self) -> Iterator[None]:
        self._dir.mkdir(parents=True, exist_ok=True)
        handle = open(self._lock_path, "a")  # noqa: SIM115 — released in the finally below
        try:
            fcntl.flock(handle, fcntl.LOCK_EX)  # blocking — inner, on a distinct file/fd
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()

    def record(
        self,
        job: str,
        session: date,
        *,
        command: list[str],
        rc: int,
        host: str,
        pid: int,
    ) -> None:
        """Write ``job``'s enriched entry (``command`` = full argv), preserving other jobs' entries.

        Atomic + fsync-durable under the marker lock (§D3): read-modify-write the whole map, write a
        sibling temp file, ``fsync`` it, ``os.replace``, then ``fsync`` the dir so the rename's
        directory entry is durable too.
        """
        self._dir.mkdir(parents=True, exist_ok=True)
        with self._marker_lock():
            data = self._read_map()
            data[job] = {
                "session": session.isoformat(),
                "recorded_at": datetime.now(UTC).isoformat(),
                "command": list(command),
                "rc": rc,
                "host": host,
                "pid": pid,
            }
            payload = json.dumps(data, indent=2, sort_keys=True).encode("utf-8")
            tmp = self._dir / f".{_MARKER_NAME}.{os.getpid()}.tmp"
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            try:
                os.write(fd, payload)
                os.fsync(fd)
            finally:
                os.close(fd)
            os.replace(tmp, self._path)
            dir_fd = os.open(self._dir, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)


@dataclass(frozen=True)
class SessionGateDecision:
    due: bool
    session: date | None
    reason: str
    skipped_sessions: int = 0


def session_gate(
    job: str, now: datetime, calendar: MarketCalendar, marker: SessionMarker
) -> SessionGateDecision:
    """Decide whether ``job`` is due to run as of ``now``.

    ``reason`` ∈ ``{"due","already_ran","no_session","marker_corrupt","calendar_out_of_bounds"}``.
    A past-the-horizon calendar (``CalendarOutOfBounds``) and a corrupt marker (``MarkerCorrupt``)
    both fail closed (``due=False``) with a distinct reason so the CLI can alert + exit non-zero. On
    a ``due`` decision, ``skipped_sessions`` counts the completed sessions strictly between the last
    recorded session and the target (0 in the ordinary daily cadence).

    A recorded session STRICTLY AFTER the current target (``last > sess``) is a SEMANTIC anomaly,
    not an ordinary re-fire: a legitimately-written marker only ever records the exact session it
    was gated `due` for, which can never outrun `target_session(now, ...)` as of the SAME `now` — so
    a future-dated entry means the file was corrupted by an external actor (manual edit, bad
    restore, clock-skewed writer). Silently treating it as `already_ran` (GATE-2 finding, #486)
    would suppress trading on every real subsequent session with zero signal — the exact "fleet
    quietly stops trading" hazard the corrupt-marker/stuck-lock alerts exist to prevent elsewhere in
    this module. It fails closed as `marker_corrupt` (alerted by the caller), same as unparseable
    JSON; only an exact match (``last == sess``) is the benign, expected re-fire case.
    """
    try:
        sess = target_session(now, calendar)
    except CalendarOutOfBounds:
        return SessionGateDecision(due=False, session=None, reason="calendar_out_of_bounds")
    if sess is None:
        return SessionGateDecision(due=False, session=None, reason="no_session")
    try:
        last = marker.last_session(job)
    except MarkerCorrupt:
        return SessionGateDecision(due=False, session=sess, reason="marker_corrupt")
    if last is not None and last > sess:
        return SessionGateDecision(due=False, session=sess, reason="marker_corrupt")
    if last is not None and last == sess:
        return SessionGateDecision(due=False, session=sess, reason="already_ran")
    skipped = 0
    if last is not None:
        # sessions_between(last, sess) counts session-steps from last to target (1 for adjacent
        # sessions); one fewer is the count strictly BETWEEN them.
        skipped = max(0, calendar.sessions_between(last, sess) - 1)
    return SessionGateDecision(due=True, session=sess, reason="due", skipped_sessions=skipped)
