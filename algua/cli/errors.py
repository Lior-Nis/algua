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

    from algua.backtest.errors import BacktestError
    from algua.contracts.lifecycle import TransitionError
    from algua.data.manifest import ManifestLockReplacedError
    from algua.data.providers.errors import ProviderError
    from algua.data.refresh import RefreshError
    from algua.data.store import SnapshotNotFound
    from algua.execution.errors import BrokerError
    from algua.execution.live_sizing import LiveSizingError
    from algua.live.live_loop import TickHalted
    from algua.portfolio.construction import ConstructionError
    from algua.registry.allocations import AllocationError
    from algua.registry.artifact_errors import (
        ArtifactNotFound,
        FrozenAssetsUnsupported,
        FrozenBundleCorrupt,
        FrozenDescriptorConflict,
        FrozenEnvironmentCorrupt,
        FrozenEnvironmentIncompatible,
        FrozenEnvironmentUnavailable,
        FrozenQualificationPending,
        FrozenSourceDrift,
        FrozenSourceInvalid,
    )
    from algua.registry.idea_attempts import ClaimTokenMismatch
    from algua.registry.live_gate import LiveAuthorizationError, SignatureError
    from algua.risk.limits import RiskBreach

    # Most-specific first. ValueError/LookupError subclasses precede the generic buckets so a
    # specific code wins; the two generic buckets and stdlib types come after.
    return [
        # --- ValueError family (specific -> generic) ---
        (FrozenSourceInvalid, "frozen_source_invalid"),
        (FrozenSourceDrift, "frozen_source_drift"),
        (FrozenAssetsUnsupported, "frozen_assets_unsupported"),
        (FrozenBundleCorrupt, "frozen_bundle_corrupt"),
        (FrozenEnvironmentIncompatible, "frozen_environment_incompatible"),
        (FrozenEnvironmentCorrupt, "frozen_environment_corrupt"),
        (FrozenDescriptorConflict, "frozen_descriptor_conflict"),
        (FrozenQualificationPending, "frozen_qualification_pending"),
        (AllocationError, "allocation_error"),
        (ClaimTokenMismatch, "claim_token_mismatch"),
        (TransitionError, "wrong_stage"),
        (ProviderError, "provider_error"),
        (ConstructionError, "construction_error"),
        (LiveSizingError, "sizing_error"),
        (RiskBreach, "risk_breach"),
        # --- LookupError family ---
        (ArtifactNotFound, "artifact_not_found"),
        (SnapshotNotFound, "not_found"),  # explicit; other *NotFound inherit the generic below
        # --- RuntimeError family (distinct domain types kept out of the `internal` bucket) ---
        (FrozenEnvironmentUnavailable, "frozen_environment_unavailable"),
        (RefreshError, "refresh_failed"),
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

    A frozen tenant's fault (Story 1.3c §8) carries its own stable ``code``: the dispatcher's
    ``FrozenTenantFailure`` (one of the ten §8 codes) and the registry's frozen refusals
    (``FrozenContentUnavailable``, ``FrozenTenantUnsupported``, ``FrozenLiveUnsupported``) resolve
    to it. Everything else walks the type-keyed :func:`_registry` most-specific-first and returns
    the first matching code; anything unmatched (a genuinely unexpected/bug-class exception —
    ``KeyError``, a pandas error, ``AttributeError``, ...) resolves to ``"internal"``. Total
    function: every exception is coded.
    """
    frozen = _frozen_tenant_fault(exc)
    if frozen is not None:
        return frozen[0]
    for typ, code in _registry():
        if isinstance(exc, typ):
            return code
    return "internal"


def _frozen_tenant_fault(exc: BaseException) -> tuple[str, int | None] | None:
    """A frozen tenant fault's own ``(code, deployment_id)`` (Story 1.3c §8), else ``None``."""
    # Imported lazily, like _registry's types: only ever needed while rendering an error.
    from algua.live.frozen_dispatch import FrozenTenantFailure
    from algua.registry.frozen_tenant_errors import FrozenTenantError

    if isinstance(exc, (FrozenTenantFailure, FrozenTenantError)):
        return exc.code, exc.deployment_id
    return None


def error_envelope(exc: BaseException) -> dict[str, object]:
    """The standard failure envelope ``{"ok": false, "error", "code", "retryable"}`` for ``exc``.

    A frozen tenant fault also carries ``deployment_id``, the deployment it is bound to, so a
    registry-side refusal (whose fixed message names no deployment) and a dispatcher failure are
    bound the same way (Story 1.3c AC8). Every other envelope keeps exactly the four keys.
    """
    code = error_code(exc)
    envelope: dict[str, object] = {
        "ok": False, "error": str(exc), "code": code, "retryable": is_retryable(code)}
    frozen = _frozen_tenant_fault(exc)
    if frozen is not None:
        envelope["deployment_id"] = frozen[1]
    return envelope


# The set of codes an operator (human or agent) MAY safely retry with backoff — the failure is a
# transient environmental condition (a busy/locked SQLite DB), NOT a deterministic input/logic error
# that would fail identically on replay. Deliberately conservative: retry defaults to FALSE and a
# code is opt-in here. Single shared definition — every envelope surface derives `retryable` from
# this set (see ``docs/contracts/cli-error-envelope.md``), never duplicating the policy per command.
RETRYABLE_CODES: frozenset[str] = frozenset({
    "db_unavailable", "frozen_environment_unavailable",
})


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
    can branch retry-with-backoff vs abort; a frozen tenant fault adds its ``deployment_id``
    (:func:`error_envelope`). See ``docs/contracts/cli-error-envelope.md``.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (typer.Exit, typer.Abort):
            raise
        except Exception as exc:
            emit(error_envelope(exc))
            raise typer.Exit(code=1) from exc

    return wrapper
