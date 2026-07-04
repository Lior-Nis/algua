from __future__ import annotations

import functools
from collections.abc import Callable

import typer

from algua.cli.app import emit


def _registry() -> list[tuple[type[BaseException], str]]:
    """The stable exception-type -> machine code registry (the documented source of truth for
    ``docs/contracts/cli-error-envelope.md``).

    Built lazily inside this function (only ever called while rendering an error) so the CLI happy
    path never imports the domain-exception modules for their own sake: any exception that reaches
    here has, by construction, already had its defining module imported (that is where it was
    raised), so every import below is a cached ``sys.modules`` hit — zero extra load cost and no
    import-cycle risk. Keyed by exception *type* (identity), not class name, so an unrelated future
    class that merely shares a name can never mis-map.

    Order matters: subclasses precede their bases so :func:`error_code`'s first-match walk returns
    the most specific code, falling through to the generic ``ValueError``/``LookupError`` buckets
    last. Every failure still resolves — anything unmatched is ``internal`` (see the resolver).
    """
    import sqlite3

    from algua.backtest.engine import BacktestError
    from algua.contracts.lifecycle import TransitionError
    from algua.data.manifest import ManifestLockReplacedError
    from algua.data.providers.errors import ProviderError
    from algua.data.store import SnapshotNotFound
    from algua.execution.alpaca_broker import BrokerError
    from algua.execution.live_sizing import LiveSizingError
    from algua.live.live_loop import TickHalted
    from algua.portfolio.construction import ConstructionError
    from algua.registry.allocations import AllocationError
    from algua.registry.live_gate import LiveAuthorizationError, SignatureError
    from algua.risk.limits import RiskBreach

    # Most-specific first. ValueError/LookupError subclasses precede the generic buckets so a
    # specific code wins; the two generic buckets and stdlib types come after.
    return [
        # --- ValueError family (specific -> generic) ---
        (AllocationError, "allocation_error"),
        (TransitionError, "wrong_stage"),
        (ProviderError, "provider_error"),
        (ConstructionError, "construction_error"),
        (LiveSizingError, "sizing_error"),
        (RiskBreach, "risk_breach"),
        # --- LookupError family ---
        (SnapshotNotFound, "not_found"),  # explicit; other *NotFound inherit the generic below
        # --- RuntimeError family (distinct domain types kept out of the `internal` bucket) ---
        (SignatureError, "bad_signature"),
        (LiveAuthorizationError, "live_unauthorized"),
        (BrokerError, "broker_error"),
        (BacktestError, "backtest_error"),
        (TickHalted, "tick_halted"),
        (ManifestLockReplacedError, "manifest_lock_replaced"),
        # --- stdlib ---
        (FileNotFoundError, "file_not_found"),
        (sqlite3.OperationalError, "db_unavailable"),
        # --- generic buckets (last) ---
        (ValueError, "invalid_input"),
        (LookupError, "not_found"),
    ]


def error_code(exc: BaseException) -> str:
    """Resolve a stable, machine-readable code for a failure envelope.

    Walks the type-keyed :func:`_registry` most-specific-first and returns the first matching code;
    anything unmatched (a genuinely unexpected/bug-class exception — ``KeyError``, a pandas error,
    ``AttributeError``, ...) resolves to ``"internal"``. Total function: every exception is coded.
    """
    for typ, code in _registry():
        if isinstance(exc, typ):
            return code
    return "internal"


# The set of codes an operator (human or agent) MAY safely retry with backoff — the failure is a
# transient environmental condition (a busy/locked SQLite DB), NOT a deterministic input/logic error
# that would fail identically on replay. Deliberately conservative: retry defaults to FALSE and a
# code is opt-in here. Single shared definition — every envelope surface derives `retryable` from
# this set (see ``docs/contracts/cli-error-envelope.md``), never duplicating the policy per command.
RETRYABLE_CODES: frozenset[str] = frozenset({"db_unavailable"})


def is_retryable(code: str) -> bool:
    """Whether a resolved error ``code`` denotes a transient, safe-to-retry-with-backoff failure."""
    return code in RETRYABLE_CODES


def json_errors(fn: Callable[..., None]) -> Callable[..., None]:
    """Render ANY command-body failure as the JSON error envelope
    ``{"ok": false, "error", "code", "retryable"}`` and exit non-zero — so an unexpected exception
    can never leak a raw traceback and break the JSON contract mid-run (issue #337).

    Catch-all by design: unlike the old per-command exception-tuple, every exception type renders as
    JSON. ``typer.Exit``/``typer.Abort`` are re-raised first — they are control flow (a command that
    emitted its own envelope and asked to exit), never an error to re-wrap. ``SystemExit``/
    ``KeyboardInterrupt`` are ``BaseException`` and pass straight through untouched.

    The ``error`` field carries ``str(exc)`` (the message, NEVER a traceback); ``code`` comes from
    :func:`error_code`; ``retryable`` is derived from that code via :func:`is_retryable` so an agent
    can branch retry-with-backoff vs abort. See ``docs/contracts/cli-error-envelope.md``.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (typer.Exit, typer.Abort):
            raise
        except Exception as exc:
            code = error_code(exc)
            emit({"ok": False, "error": str(exc), "code": code, "retryable": is_retryable(code)})
            raise typer.Exit(code=1) from exc

    return wrapper
