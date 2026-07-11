"""Shared CLI helpers — the one place command modules reach for connection, time, and
evaluation-input boilerplate.

This exists so command modules stop importing each other's private helpers (a cross-module
private-import smell): the public names here are the sanctioned shared surface.
"""

from __future__ import annotations

import math
import sqlite3
import sys
from collections.abc import Collection, Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from algua.backtest._sample import SyntheticProvider
from algua.config.settings import get_settings
from algua.contracts.lifecycle import Actor
from algua.contracts.types import DataProvider
from algua.data.serve import StoreBackedProvider
from algua.data.store import DataStore
from algua.registry.db import connect, migrate
from algua.strategies.base import LoadedStrategy
from algua.strategies.loader import load_strategy


class StrategySetupError(Exception):
    """A per-tenant setup failure raised BEFORE any broker/ledger side effect began this cycle.

    Fault isolation boundary (#374 GATE-2): ``run-all`` (live and paper) may safely demote ONE
    tenant's *setup* failure — module/strategy load, missing allocation, identity/config error —
    to a benign ``{"ok": False, "kind": "setup_error"}`` marker and keep ticking the rest of the
    book. It must NOT swallow a failure that escapes the tick helper's own breach/halt handling
    (``trip_for_breach`` / ``flatten_strategy`` / audit) or that fires from an ``on_submitted`` /
    ledger-persist hook AFTER a real order has hit the venue: those are book-integrity-critical and
    must fail closed (abort the cycle). Only the code that runs strictly before the first side
    effect wraps its exceptions in this type; everything else propagates raw.

    ``code`` is a stable, redacted classifier (the raising exception's class name) suitable for the
    JSON envelope and audit trail — the raw ``str(exc)`` (which can carry credentials/paths) is
    NEVER surfaced there; it survives only in the ``exc_info=True`` structured log.
    """

    def __init__(self, strategy: str, cause: BaseException) -> None:
        self.strategy = strategy
        self.code = type(cause).__name__
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


@contextmanager
def registry_conn() -> Iterator[sqlite3.Connection]:
    """Yield a migrated registry connection, closed on exit.

    The single idiom for opening the registry DB: connect + migrate + auto-close. Replaces the
    two competing forms (a bare ``_conn()`` and inline ``closing(connect(...))`` + ``migrate``).
    """
    conn = connect(get_settings().db_path)
    try:
        migrate(conn)
        yield conn
    finally:
        conn.close()


def authenticate_actor(
    conn: sqlite3.Connection, *, command: str, name: str, rec: object, stage_to: str,
    declared_actor: Actor, actor_signature: str | None, run_context: str,
) -> Actor:
    """Turn a declared ``--actor`` + optional ``--actor-signature`` into the EFFECTIVE actor the
    downstream human-only guards may trust (#329). The single shared chokepoint for the gated
    promote commands (research + paper), so the authentication is wired identically in one place.

    - declared agent/system -> returned unchanged (agents never sign).
    - declared human, NO signature -> a fresh single-use challenge is issued+persisted and PRINTED
      as JSON (mirrors the go-live challenge print), then the command EXITS 0 having run nothing.
    - declared human + signature -> the SSH signature is verified (namespace algua-human-actor) over
      the REBUILT payload bound to this command + strategy + RECOMPUTED artifact identity + the full
      ``run_context``; on success the effective actor is HUMAN, else a ValueError is raised
      (fail closed — a forged/replayed/expired/cross-run signature is refused).

    ``rec`` is the strategy record (used for ``rec.id`` + ``rec.stage``). Imports the gate module +
    ``emit`` lazily so ``_common`` stays free of a cli->cli / heavy-registry import at module load.
    """
    import typer

    from algua.cli.app import emit
    from algua.registry.approvals import compute_artifact_hashes
    from algua.registry.human_actor import (
        HumanActorChallengeRequired,
        resolve_effective_actor,
    )

    if declared_actor is not Actor.HUMAN:
        return declared_actor
    identity = compute_artifact_hashes(name)
    signature = Path(actor_signature).read_bytes() if actor_signature else None
    try:
        return resolve_effective_actor(
            conn, command=command, strategy=name, strategy_id=rec.id,  # type: ignore[attr-defined]
            stage_from=rec.stage.value, stage_to=stage_to,  # type: ignore[attr-defined]
            code_hash=identity.code_hash, config_hash=identity.config_hash,
            dependency_hash=identity.dependency_hash, declared_actor=declared_actor,
            run_context=run_context, signature=signature)
    except HumanActorChallengeRequired as exc:
        emit(ok({
            "action": "human_actor_challenge", "strategy": name, "command": command,
            **exc.challenge,
            "instructions": (
                "sign the 'challenge' value with your enrolled algua-human-actor key: "
                "ssh-keygen -Y sign -n algua-human-actor -f <key> <file>; "
                "then re-run this command with --actor-signature <file>.sig"),
        }))
        raise typer.Exit() from None


def now_iso() -> str:
    """Current UTC instant as an ISO-8601 string — the shared 'now' for persisted timestamps."""
    return datetime.now(UTC).isoformat()


def utc(date_str: str) -> datetime:
    """Parse an ISO date/datetime string and stamp it UTC."""
    return datetime.fromisoformat(date_str).replace(tzinfo=UTC)


def sync_kb_doc(name: str) -> None:
    """Best-effort: re-sync ``name``'s kb doc + family roster + indexes to current registry truth.

    The transactional side effect (#331) wired into every stage-mutating gate command (research
    promote, paper promote, backtest --register, registry transition) so the Obsidian vault is
    never stale-by-default. Two invariants make it safe:

    * OUT OF TRANSACTION — it opens its OWN short registry connection, which callers invoke only
      AFTER their write transaction has committed and closed, so vault file I/O never runs while a
      registry write lock is held.
    * NON-FATAL — any failure is swallowed (with a one-line stderr warning); a stale vault is a
      curation gap, never a reason to break or roll back a real promotion. The binding audit (the
      ``gate_evaluations`` row + result JSON) is always-on and reproducible-by-hash regardless.

    Lazy imports keep the mlflow-importing knowledge layer off the hot path of unrelated commands.
    """
    try:
        from algua.knowledge.sync import sync_strategy_and_dependents
        from algua.registry.repository import kb_metadata
        from algua.registry.store import SqliteStrategyRepository

        with registry_conn() as conn:
            rec = SqliteStrategyRepository(conn).get(name)
        sync_strategy_and_dependents(
            get_settings(), name, stage=rec.stage.value, metadata=kb_metadata(rec)
        )
    except Exception as exc:  # noqa: BLE001 — vault curation must never break a promotion
        print(f"warning: kb doc sync for {name!r} failed: {exc}", file=sys.stderr)


def select_provider(demo: bool, snapshot: str | None) -> DataProvider:
    """Pick the data provider from the mutually-exclusive --demo / --snapshot flags."""
    if demo and snapshot:
        raise ValueError("pass only one of --demo or --snapshot")
    if demo:
        return SyntheticProvider(seed=0)
    if snapshot:
        return StoreBackedProvider(DataStore(get_settings().data_dir), snapshot)
    raise ValueError("pass one of --demo (synthetic) or --snapshot <id> (real data)")


def resolve_eval_inputs(
    name: str, demo: bool, snapshot: str | None, start: str, end: str, *, reload: bool = False
) -> tuple[LoadedStrategy, DataProvider, datetime, datetime]:
    """Resolve the shared backtest-family preamble: load the strategy, pick the provider, and
    parse the period. Returns ``(strategy, provider, start_dt, end_dt)``.

    ``reload=True`` force-reloads the strategy module (see ``load_strategy``) — used by the
    long-lived ``research run-all`` batch worker (#326) so a warm process does not carry a
    strategy's own module-level state from one task into the next."""
    strategy = load_strategy(name, reload=reload)
    provider = select_provider(demo, snapshot)
    return strategy, provider, utc(start), utc(end)


def resolve_delisting_inputs(
    delistings_name: str | None, end_dt: datetime
) -> tuple[Mapping[str, list] | None, str | None]:
    """Resolve opt-in delisting records as-of end_dt (mirror of resolve_universe_inputs).

    ``delistings_name is None`` (no ``--delistings``) => ``(None, None)``.
    Returns ``(records, snapshot_id)`` where ``snapshot_id`` is the ACTUAL snapshot selected
    (not the user-supplied name label) for truthful provenance stamping.
    Raises ``ValueError`` if no delistings snapshot is effective on or before ``end_dt``.
    """
    if delistings_name is None:
        return None, None
    store = DataStore(get_settings().data_dir)
    # Single manifest read: records and snapshot_id come from the SAME selected snapshot, so a
    # concurrent ingest can never make the stamped provenance disagree with the loaded records.
    records, snapshot_id = store.read_delistings_with_snapshot(as_of=end_dt.isoformat())
    if not records:
        raise ValueError(
            f"--delistings {delistings_name!r}: no delistings snapshot effective on or before "
            f"{end_dt.date().isoformat()}"
        )
    return records, snapshot_id


def resolve_universe_inputs(
    universe_name: str | None, start_dt: datetime, end_dt: datetime
) -> tuple[Mapping[date, Collection[str]] | None, list[dict[str, str]] | None]:
    """Resolve the opt-in point-in-time universe for a backtest-family command.

    `universe_name is None` (no `--universe`) => static mode: returns ``(None, None)`` and the
    engine fetches/shows the strategy's declared universe unchanged.

    Otherwise reads the named universe's membership timeline from the `DataStore`, restricts it to
    snapshots effective on or before `end_dt` (so the union fetched for bars never includes a
    member that only becomes effective after the backtest window — and the as-of resolution at any
    `t <= end_dt` is unaffected), and returns:
      * a sparse ``{effective_date: symbols}`` map the engine resolves as-of-t (greatest
        effective_date <= t; empty before the earliest), and
      * the provenance list ``[{"snapshot_id", "effective_date"}, ...]`` for the result JSON.
    Raises ``ValueError`` if the universe has no membership effective by `end_dt`.
    """
    if universe_name is None:
        return None, None
    timeline = DataStore(get_settings().data_dir).read_universe(universe_name)
    end_date = end_dt.date()
    in_window = [snap for snap in timeline if snap.effective_date <= end_date]
    if not in_window:
        raise ValueError(
            f"universe {universe_name!r} has no membership effective on or before "
            f"{end_date.isoformat()}; ingest a snapshot with --effective-date <= end"
        )
    universe_by_date: dict[date, Collection[str]] = {
        snap.effective_date: snap.symbols for snap in in_window
    }
    provenance = [
        {"snapshot_id": snap.snapshot_id, "effective_date": snap.effective_date.isoformat()}
        for snap in in_window
    ]
    return universe_by_date, provenance
