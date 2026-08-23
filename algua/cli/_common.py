"""Shared CLI helpers — the one place command modules reach for connection and time boilerplate.

This exists so command modules stop importing each other's private helpers (a cross-module
private-import smell): the public names here are the sanctioned shared surface.
"""

from __future__ import annotations

import math
import re
import sqlite3
from collections.abc import Collection
from datetime import UTC, datetime, timedelta

from algua.config.settings import get_settings

# Exception types that signal a SYSTEMIC / book-wide condition rather than one tenant's setup fault
# (#374 GATE-2 fix): a locked/unavailable shared SQLite connection during one strategy's setup read
# means every sibling's read is equally suspect. Isolating it per-tenant would silently burn a whole
# run-all cycle marking every strategy ``setup_error`` while masking the retryable condition that
# the top-level ``json_errors`` envelope exists to surface (docs/contracts/cli-error-envelope.md,
# code ``db_unavailable``, ``retryable: true``). Callers at the setup-boundary try/except re-raise
# these RAW (never wrap in :class:`StrategySetupError`) so run-all aborts the whole cycle and the
# top-level envelope's retry signal survives, instead of isolating a systemic DB fault as if it were
# one tenant's config problem.
SYSTEMIC_SETUP_EXCEPTIONS: tuple[type[BaseException], ...] = (sqlite3.Error,)

# A defensively narrow allowlist for StrategySetupError.code: it is derived from an exception CLASS
# NAME, which is normally a safe fixed identifier, but a dynamically-constructed class (e.g. built
# at runtime from strategy-supplied or otherwise untrusted data) could in principle produce an
# arbitrary string. Since ``code`` is surfaced verbatim in the JSON envelope and the audit-log
# ``reason`` column, anything that doesn't look like a plain identifier is replaced with a fixed
# fallback rather than trusted as-is (#374 GATE-2).
_SAFE_SETUP_CODE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


class StrategySetupError(Exception):
    """A per-tenant setup failure raised BEFORE any broker/ledger side effect began this cycle.

    Fault isolation boundary (#374 GATE-2): ``run-all`` (live and paper) may safely demote ONE
    tenant's *setup* failure — module/strategy load, missing allocation, identity/config error —
    to a benign ``{"ok": False, "kind": "setup_error"}`` marker and keep ticking the rest of the
    book. It must NOT swallow a failure that escapes the tick helper's own breach/halt handling
    (``trip_for_breach`` / ``flatten_strategy`` / audit) or that fires from an ``on_submitted`` /
    ledger-persist hook AFTER a real order has hit the venue: those are book-integrity-critical and
    must fail closed (abort the cycle). It must ALSO not swallow a :data:`SYSTEMIC_SETUP_EXCEPTIONS`
    member (e.g. ``sqlite3.Error``) — a shared-infrastructure fault, not a single tenant's problem —
    which setup-boundary callers re-raise raw instead of wrapping here. Only the code that runs
    strictly before the first side effect, and that is genuinely tenant-local, wraps its exceptions
    in this type; everything else propagates raw.

    ``code`` is a stable, redacted classifier (the raising exception's class name, allowlist-
    sanitized via :data:`_SAFE_SETUP_CODE_RE`) suitable for the JSON envelope and audit trail — the
    raw ``str(exc)`` (which can carry credentials/paths) is NEVER surfaced there; it survives only
    in the ``exc_info=True`` structured log.
    """

    def __init__(self, strategy: str, cause: BaseException) -> None:
        self.strategy = strategy
        # Keep the wrapped cause reachable (not only via ``__cause__``) so a SINGLE-strategy command
        # path — which has no siblings to isolate and wants the ORIGINAL fault's actionable message
        # and specific error code, not this redacted wrapper — can unwrap and re-raise it (#374).
        self.cause: BaseException = cause
        raw_code = type(cause).__name__
        self.code = raw_code if _SAFE_SETUP_CODE_RE.match(raw_code) else "SetupError"
        super().__init__(f"{strategy}: {self.code}")


def ok(data: dict) -> dict:
    """Stamp a success payload with the ``ok: true`` discriminator.

    CLI JSON-envelope convention: every object-shaped *success* payload carries ``"ok": true`` as
    its first key, mirroring the ``{"ok": false, "error": ...}`` failure envelope (see
    ``cli.errors.json_errors`` and ``cli.main.main``). Callers that emit a JSON *array*
    (``registry list``, ``data inspect``) are the explicit exception: they stay bare arrays.
    """
    return {"ok": True, **data}


def project(payload: dict, keep: Collection[str]) -> dict:
    """Project a success payload to its decision-relevant subset for ``--summary``.

    Context-rot defense (#349): the heavy agent-facing commands emit large payloads (per-window
    or per-combo lists, deep gate diagnostics) that degrade an unattended operator's finite
    context. ``--summary`` projects to the decision-relevant scalars instead.

    Always preserves the ``ok`` discriminator and stamps ``summary: True`` so a consumer can tell
    a projected payload from a full one; keeps only the listed keys that are present. Each command
    passes its own curated keep-list (keep-lists, not drop-lists, so any future diagnostic field
    is excluded-by-default). SUCCESS PAYLOADS ONLY — the ``@json_errors`` failure envelope is
    produced by the decorator and never reaches this, so ``--summary`` can never strip ``error``.
    """
    return {k: v for k, v in payload.items() if k == "ok" or k in keep} | {"summary": True}


def breach_payload(error: str, **extra: object) -> dict:
    """A failure envelope for a tripped kill-switch: ``{"ok": false, "kill_switch": "tripped"...}``.

    The shared skeleton of every paper/live-command halt/breach emit; callers pass the
    human-readable ``error`` plus whatever variant keys (``kind``, ``strategy``, ``halted``, ...)
    that path adds. Pure presentation — lives beside ``ok`` in the CLI infrastructure, not in a
    command module (so paper and live share it without a cli→cli import).
    """
    return {"ok": False, "kill_switch": "tripped", "error": error, **extra}


def resolve_drawdown_breaker(max_drawdown: float | None, disabled: bool) -> float | None:
    """Resolve the effective per-strategy drawdown breaker for a trading loop (#390).

    The breaker is default-ON: an omitted ``--max-drawdown`` (None) resolves to the conservative
    ``settings.strategy_max_drawdown_default`` rather than leaving the breaker OFF. An explicit
    ``--max-drawdown`` value is honored as-is. The breaker is turned OFF (returns None, the
    ``DRAWDOWN_DISABLED`` sentinel ``check_drawdown`` recognizes) ONLY via the explicit human-only
    ``--disable-drawdown-breaker`` flag; the caller audits that disable. Shared by paper and live
    so the default-ON policy can never drift between the lanes.
    """
    if disabled:
        return None
    if max_drawdown is None:
        default = float(get_settings().strategy_max_drawdown_default)
        # Fail closed on a misconfigured default (env override): a non-finite or out-of-(0,1] value
        # would silently make the default-ON breaker ineffective without the audited disable path.
        if not math.isfinite(default) or not 0.0 < default <= 1.0:
            raise ValueError(
                "strategy_max_drawdown_default (ALGUA_STRATEGY_MAX_DRAWDOWN_DEFAULT) must be a "
                f"finite fraction in (0, 1]; got {default!r}"
            )
        return default
    return max_drawdown


def resolve_wall_clock_window(start: str | None, end: str | None) -> tuple[str, str]:
    """Fill an unspecified live/paper wall-clock window with a recent rolling window (#452).

    Default wall-clock runs without explicit --start/--end should NOT size/risk-check against a
    frozen stale window (e.g., 2023-01-01 to 2023-12-31); instead they should default to a recent
    rolling window (end=today UTC, start=today-LIVE_WINDOW_LOOKBACK_DAYS).

    Explicit values pass through unchanged. Returns (start_iso, end_iso) both as ISO date strings.
    """
    LIVE_WINDOW_LOOKBACK_DAYS = 400  # ~275 sessions; covers typical warmups with slack
    today = datetime.now(UTC).date()
    end_iso = end or today.isoformat()
    start_iso = start or (today - timedelta(days=LIVE_WINDOW_LOOKBACK_DAYS)).isoformat()
    return start_iso, end_iso


def now_iso() -> str:
    """Current UTC instant as an ISO-8601 string — the shared 'now' for persisted timestamps."""
    return datetime.now(UTC).isoformat()
