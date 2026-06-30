"""Structured JSON logging for the always-on loop (issue #346).

A stdlib-only pure leaf: it imports no other ``algua`` module. Records are emitted as ONE
physical line of JSON to **stderr** so the stdout one-JSON-document machine contract
(``cli.app.emit``) is never corrupted. The systemd journal captures stderr.

Usage::

    from algua.observability import configure_logging, correlation_context, get_logger

    configure_logging()
    log = get_logger(__name__)
    with correlation_context():               # one id per loop cycle
        log.info("cycle_start", extra={"fields": {"lane": "paper"}})
        log.error("breach", extra={"fields": {"strategy": "alpha"}}, exc_info=True)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from uuid import uuid4

_ROOT = "algua"
_CID: ContextVar[str | None] = ContextVar("algua_correlation_id", default=None)


def current_correlation_id() -> str | None:
    """The correlation id bound to the current context, or ``None`` outside a cycle."""
    return _CID.get()


@contextmanager
def correlation_context(cid: str | None = None) -> Iterator[str]:
    """Bind a correlation id for the block; reset on exit, even on exception.

    A uuid4 hex is generated when ``cid`` is ``None``. The yielded value is the active id.
    """
    value = cid or uuid4().hex
    token = _CID.set(value)
    try:
        yield value
    finally:
        _CID.reset(token)


class JsonFormatter(logging.Formatter):
    """Render a ``LogRecord`` as one physical line of JSON.

    Caller structured data arrives via ``logger.info(msg, extra={"fields": {...}})``. Core keys
    (``ts``/``level``/``logger``/``msg``/``correlation_id`` and the ``exc_*`` triple) always win
    over caller fields. A non-serializable field never drops the record: serialization uses
    ``default=str`` and the whole body has a minimal-record fallback so it never raises into the
    logging machinery.
    """

    def format(self, record: logging.LogRecord) -> str:
        try:
            fields = getattr(record, "fields", None)
            payload: dict[str, object] = dict(fields) if isinstance(fields, dict) else {}
            core: dict[str, object] = {
                "ts": datetime.now(UTC).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "msg": record.getMessage(),
            }
            cid = _CID.get()
            if cid is not None:
                core["correlation_id"] = cid
            if record.exc_info and record.exc_info[0] is not None:
                core["exc_type"] = record.exc_info[0].__name__
                core["exc_message"] = str(record.exc_info[1])
                core["stacktrace"] = self.formatException(record.exc_info)
            payload.update(core)  # core overwrites any caller-field collision
            return json.dumps(payload, default=str)
        except Exception as exc:  # noqa: BLE001 - never raise / never drop a record
            return json.dumps(
                {
                    "ts": datetime.now(UTC).isoformat(),
                    "level": getattr(record, "levelname", "ERROR"),
                    "msg": "log_format_error",
                    "format_error": str(exc),
                }
            )


class _StderrHandler(logging.StreamHandler):
    """A StreamHandler that re-resolves ``sys.stderr`` on every emit.

    A plain ``StreamHandler(sys.stderr)`` binds the stream object at construction. In a long-lived
    process where ``sys.stderr`` is swapped or closed (pytest capture / typer CliRunner / any
    repeated in-process invocation), that stale stream wedges the logger ("Logging error"). Mirrors
    the stdlib ``logging._StderrHandler`` used by ``lastResort`` to always target the live stream.
    """

    def __init__(self) -> None:
        logging.Handler.__init__(self)

    @property
    def stream(self):
        return sys.stderr

    @stream.setter
    def stream(self, value: object) -> None:  # StreamHandler.__init__ assigns; ignore it
        pass


def get_logger(name: str = _ROOT) -> logging.Logger:
    """Return a logger under the ``algua`` root (the one carrying the stderr handler)."""
    if name != _ROOT and not name.startswith(_ROOT + "."):
        name = f"{_ROOT}.{name}"
    return logging.getLogger(name)


def _resolve_level(raw: str | None) -> int:
    if raw:
        level = logging.getLevelName(raw.strip().upper())
        if isinstance(level, int):
            return level
    return logging.INFO


def configure_logging() -> None:
    """Idempotently wire the ``algua`` logger to a single JSON stderr handler.

    Re-reads ``ALGUA_LOG_LEVEL`` (default/unknown -> INFO) on EVERY call so env changes take
    effect; adds the marked handler only once; never removes foreign handlers; ``propagate=False``
    keeps records off the root ``lastResort`` handler (and off stdout).
    """
    logger = logging.getLogger(_ROOT)
    logger.setLevel(_resolve_level(os.environ.get("ALGUA_LOG_LEVEL")))
    logger.propagate = False
    if not any(getattr(h, "_algua_observability", False) for h in logger.handlers):
        handler = _StderrHandler()
        handler.setFormatter(JsonFormatter())
        handler._algua_observability = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
