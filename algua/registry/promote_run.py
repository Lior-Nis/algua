"""``promote_task`` -- the ``research promote`` task body: the ``backtested -> candidate`` gate,
shared with the ``research run-all`` batch worker (#326) and the merge-back authoritative-promote
seam (#485) — see ``algua.cli.paper_cmd``'s merge-back saga.

Moved out of ``algua.cli.research_cmd`` so ``paper_cmd``'s merge-back saga can reach the REAL
promote body via a legal static import instead of a dynamic ``importlib`` dodge around the
cli-independence contract (issue #165) — the last cli->cli escape on this seam. This package is
importable by both ``cli`` and ``registry`` without ``algua.registry`` importing
``algua.cli`` itself. ``promote_task`` opens+closes its own registry connection, re-syncs the kb
doc, and records the advisory negative-result capture exactly as it did in ``research_cmd``, via the
shared ``algua.registry.db.registry_conn``, ``algua.registry.kb_sync.sync_kb_doc``, and
``algua.registry.human_actor.authenticate_actor`` — the owning leaves — so none of those idioms is
duplicated here.
"""

from __future__ import annotations

from typing import Any

from algua.backtest.engine import holdout_window
from algua.backtest.walkforward import walk_forward
from algua.config.settings import get_settings
from algua.contracts.lifecycle import Actor, Stage
from algua.data.models import Dataset
from algua.data.serve import StoreBackedFundamentalsProvider, StoreBackedNewsProvider
from algua.data.store import DataStore
from algua.evaluation.inputs import (
    resolve_delisting_inputs,
    resolve_eval_inputs,
    resolve_universe_inputs,
)
from algua.knowledge.experience import write_experience_note
from algua.observability.log import get_logger
from algua.primitives.timeparse import now_iso
from algua.registry.db import registry_conn
from algua.registry.human_actor import authenticate_actor, canonical_run_context
from algua.registry.kb_sync import sync_kb_doc
from algua.registry.negative_results import (
    build_gate_fail_record,
    record_negative_result,
    sanitize_record,
)
from algua.registry.promotion import (
    _revalidate_pending_novel,
    promotion_preflight,
    run_gate,
)
from algua.registry.store import SqliteStrategyRepository
from algua.research.gates import GateCriteria

_HOLDOUT_REUSE_OVERRIDE = "override"


def capture_gate_fail_experience(
    conn: Any,
    *,
    name: str,
    decision: Any,
    actor: Actor,
    config_hash: str | None,
    strategy_id: int,
    period_start: str,
    period_end: str,
    holdout: dict[str, Any] | None,
    stability: dict[str, Any] | None,
) -> dict[str, Any]:
    """Best-effort advisory capture of a FAILED gate into the negative-result log (#332).

    Writes the queryable ledger row (primary) and a graph-linked vault note (secondary), reporting
    each independently. Every failure mode is caught: this is knowledge-capture, so it must NEVER
    propagate and break the promote it is describing. ``gate_evaluation_id`` is an advisory
    back-link resolved by a read-only lookup on the just-written gate_evaluations row.
    """
    ledger: dict[str, Any] = {"status": "skipped", "id": None, "error": None}
    note: dict[str, Any] = {"status": "skipped", "path": None, "error": None}
    created_at = now_iso()
    record = build_gate_fail_record(
        name, decision.to_dict(), actor=actor.value,
        period_start=period_start, period_end=period_end, holdout=holdout, stability=stability)
    try:
        gate_eval_id: int | None = None
        if config_hash is not None:
            row = conn.execute(
                "SELECT id FROM gate_evaluations WHERE strategy_id=? AND config_hash=? "
                "ORDER BY id DESC LIMIT 1", (strategy_id, config_hash)).fetchone()
            gate_eval_id = int(row[0]) if row else None
        rid = record_negative_result(
            conn, gate_evaluation_id=gate_eval_id, created_at=created_at, **record)
        ledger = {"status": "recorded", "id": rid, "error": None}
    except Exception as e:  # noqa: BLE001 - advisory capture must never break the promote
        return {"ledger": {"status": "error", "id": None, "error": f"{type(e).__name__}: {e}"},
                "note": note}
    try:
        note_record = sanitize_record(
            {**record, "created_at": created_at, "gate_evaluation_id": gate_eval_id})
        path = write_experience_note(get_settings(), note_record, record_id=rid)
        note = {"status": "written", "path": str(path), "error": None}
    except Exception as e:  # noqa: BLE001 - the vault note is a best-effort secondary surface
        note = {"status": "error", "path": None, "error": f"{type(e).__name__}: {e}"}
    return {"ledger": ledger, "note": note}


def promote_task(  # noqa: PLR0913, PLR0915
    name: str, *, start: str = "2023-01-01", end: str = "2023-12-31", demo: bool = False,
    snapshot: str | None = None, fundamentals_snapshot: str | None = None,
    news_snapshot: str | None = None, universe: str | None = None, windows: int = 4,
    holdout_frac: float = 0.2, min_holdout_sharpe: float = 0.5, min_holdout_return: float = 0.0,
    min_pct_positive: float = 0.6, min_window_sharpe: float = 0.0, n_combos: int | None = None,
    allow_holdout_reuse: bool = False, allow_non_pit: bool = False, delistings: str | None = None,
    assume_terminal_last_close: bool = False, actor: str = "agent",
    actor_signature: str | None = None,
    new_family: str | None = None, reload: bool = False,
    attempt_token: str | None = None,
) -> dict:
    """Run the backtested->candidate gate and return the (pre-``--summary``) payload dict — the
    body of ``research promote``, shared with the ``research run-all`` batch worker (#326).

    Opens+closes its own ``registry_conn()`` per call (NO caller-owned connection): the holdout
    single-use guard is a DB row reserved under BEGIN IMMEDIATE, so reusing ONE warm process across
    many promote tasks reuses NOTHING — a second task on an already-burned window hits the same
    committed-burn overlap and fails closed, identical to two separate cold processes. ``reload``
    force-reloads the strategy module (warm-worker state hygiene)."""
    actor_enum = Actor(actor)  # fail fast on a bad actor before running the walk-forward
    if n_combos is not None and n_combos < 1:
        raise ValueError("--n-combos must be >= 1 when provided")
    if not 0.0 <= min_pct_positive <= 1.0:
        raise ValueError("--min-pct-positive must be in [0, 1]")
    # HUMAN-ONLY guard (same mechanism as guard_agent_relaxations in promotion_preflight):
    # --assume-terminal-last-close is a data-integrity relaxation that must never be granted to
    # an agent. An agent must supply explicit delisting records; a held-into-gap name with no
    # record fails closed on the agent path. Humans may pass the flag (and accept the cost).
    if assume_terminal_last_close and actor_enum is not Actor.HUMAN:
        raise ValueError(
            "--assume-terminal-last-close is human-only (an agent must supply delisting records "
            "via --delistings; a held-into-gap name without a record fails closed for the agent "
            "path). Pass --actor human to accept the cost."
        )
    # 1. Resolve inputs. The PIT universe is resolved up front alongside the other inputs (a bad
    # --universe refuses here, before any holdout is peeked at). The universe is intentionally NOT
    # part of the holdout-burn identity below (conservative: the same OOS data window is burned
    # regardless of universe).
    strategy, provider, start_dt, end_dt = resolve_eval_inputs(
        name, demo, snapshot, start, end, reload=reload)
    # PIT sidecar guards (misuse + early fail-closed) BEFORE any holdout reservation/peek: a
    # needs_X strategy without its snapshot must refuse before reserve_holdout touches the window.
    if fundamentals_snapshot and not strategy.config.needs_fundamentals:
        raise ValueError("--fundamentals-snapshot was given but the strategy does not declare "
                         "needs_fundamentals")
    if news_snapshot and not strategy.config.needs_news:
        raise ValueError("--news-snapshot was given but the strategy does not declare needs_news")
    if strategy.config.needs_fundamentals and not fundamentals_snapshot:
        raise ValueError("strategy declares needs_fundamentals; pass --fundamentals-snapshot")
    if strategy.config.needs_news and not news_snapshot:
        raise ValueError("strategy declares needs_news; pass --news-snapshot")
    fundamentals_provider = (
        StoreBackedFundamentalsProvider(DataStore(get_settings().data_dir), fundamentals_snapshot)
        if fundamentals_snapshot else None)
    news_provider = (
        StoreBackedNewsProvider(DataStore(get_settings().data_dir), news_snapshot)
        if news_snapshot else None)
    # Fail fast on a missing/wrong-kind PIT snapshot BEFORE any holdout reservation, so a typo'd
    # snapshot id can never strand a pending reservation (#132 GATE-2). get_snapshot raises
    # SnapshotNotFound (LookupError) on a missing id; the dataset-kind check adds the wrong-kind
    # case. Both surface as JSON via @json_errors and both precede reserve_holdout.
    if news_provider is not None:
        rec = news_provider.store.get_snapshot(news_provider.snapshot_id)
        if rec.dataset != Dataset.NEWS.value:
            raise ValueError(f"--news-snapshot {news_provider.snapshot_id!r} is dataset "
                             f"{rec.dataset!r}, not {Dataset.NEWS.value!r}")
    if fundamentals_provider is not None:
        rec = fundamentals_provider.store.get_snapshot(fundamentals_provider.snapshot_id)
        if rec.dataset != Dataset.FUNDAMENTALS.value:
            raise ValueError(f"--fundamentals-snapshot {fundamentals_provider.snapshot_id!r} is "
                             f"dataset {rec.dataset!r}, not {Dataset.FUNDAMENTALS.value!r}")
    universe_by_date, universe_prov = resolve_universe_inputs(universe, start_dt, end_dt)
    delisting_records, delisting_prov = resolve_delisting_inputs(delistings, end_dt)
    data_source = type(provider).__name__
    snapshot_id = getattr(provider, "snapshot_id", None)
    period_start = start_dt.date().isoformat()
    period_end = end_dt.date().isoformat()
    criteria = GateCriteria(
        min_holdout_sharpe=min_holdout_sharpe, min_holdout_return=min_holdout_return,
        min_pct_positive_windows=min_pct_positive, min_window_sharpe=min_window_sharpe,
    )

    experience_log: dict[str, Any] | None = None
    with registry_conn() as conn:
        repo = SqliteStrategyRepository(conn)
        rec0 = repo.get(name)  # StrategyNotFound -> JSON error before any work
        # AUTHENTICATE the human actor (#329) BEFORE any relaxation is honored or the holdout is
        # touched. A bare `--actor human` is forgeable, so asserting a human actor here requires an
        # SSH signature (namespace algua-human-actor) over a fresh single-use challenge that binds
        # this command + strategy + RECOMPUTED artifact identity + the FULL canonical run_context
        # (every gate-relevant input, incl. the exact relaxation set). No signature => a challenge
        # is issued+printed and NOTHING runs. A declared agent/system is returned unchanged (the
        # downstream guards refuse its relaxations exactly as before).
        actor_enum = authenticate_actor(
            conn, command="research promote", name=name, rec=rec0, stage_to=Stage.CANDIDATE.value,
            declared_actor=actor_enum, actor_signature=actor_signature,
            run_context=canonical_run_context({
                "start": start, "end": end, "demo": demo, "snapshot": snapshot,
                "fundamentals_snapshot": fundamentals_snapshot, "news_snapshot": news_snapshot,
                "universe": universe, "windows": windows, "holdout_frac": holdout_frac,
                "min_holdout_sharpe": min_holdout_sharpe, "min_holdout_return": min_holdout_return,
                "min_pct_positive": min_pct_positive, "min_window_sharpe": min_window_sharpe,
                "n_combos": n_combos, "allow_holdout_reuse": allow_holdout_reuse,
                "allow_non_pit": allow_non_pit, "delistings": delistings,
                "assume_terminal_last_close": assume_terminal_last_close,
                "new_family": new_family,
                # Bind the RESOLVED immutable data provenance, not just the mutable name: an agent
                # can `data ingest-universe`/`import-delistings` a new effective-date between the
                # challenge and the signature to change what the SAME name resolves to. Binding the
                # resolved snapshot ids/effective-dates makes a captured signature fail if the
                # named universe/delistings timeline shifts under it (codex GATE-2 CRITICAL).
                "universe_prov": universe_prov, "delistings_prov": delisting_prov,
            }),
        )
        # PREFLIGHT (pre-peek): relaxations-need-human + stage legality + breadth. Refuses here,
        # before walk_forward touches the holdout.
        breadth = promotion_preflight(
            repo, name, actor=actor_enum, declared_combos=n_combos,
            allow_holdout_reuse=allow_holdout_reuse, allow_non_pit=allow_non_pit,
            provider=provider, start=start_dt, end=end_dt,
            new_family_slug=new_family,
            universe_name=universe, universe_by_date=universe_by_date)
        # Atomic holdout reservation (#161): claim the window under the write lock (fast SELECT +
        # INSERT a pending row), run walk_forward with NO lock held, then finalize on success /
        # release on a clean failure. The match identity is the data window and deliberately
        # EXCLUDES the universe (the same OOS window is burned regardless of universe). A pending
        # reservation blocks a concurrent run exactly like a committed burn (fail closed).
        # Compute the EXACT OOS interval walk_forward will burn (from the bar date-index, without
        # running the strategy) so the single-use guard matches on the actual bars, not the full
        # period + holdout_frac (#192).
        holdout_start, holdout_end = holdout_window(
            strategy, provider, start_dt, end_dt,
            holdout_frac=holdout_frac, universe_by_date=universe_by_date)
        sid = repo.get(name).id
        reservation_id, reused = repo.reserve_holdout(
            sid, data_source=data_source, snapshot_id=snapshot_id,
            period_start=period_start, period_end=period_end, holdout_frac=holdout_frac,
            holdout_start=holdout_start, holdout_end=holdout_end,
            allow_reuse=allow_holdout_reuse)  # raises here = fail closed (overlap, no reuse)
        # #524 (R9-H4): the holdout-burn-on-drift window is NARROWED, not fully closed. on_peek
        # commits the reservation the instant BEFORE the holdout metric is read; wrap it so it FIRST
        # re-runs the pending-NOVEL revalidation (still-unassigned + graph fingerprint + per-window
        # rate cap — all pure DB reads) and raises BEFORE finalize on drift/bound. Drift
        # caught here → the reservation stays pending → the except-release below frees the window →
        # no holdout burned. RESIDUAL (documented + monitored, NOT silently "closed"): run_gate
        # below runs its OWN pre-lock pending-NOVEL re-check AFTER walk_forward has returned — AFTER
        # on_peek already committed the burn. Drift that first becomes visible in that post-peek
        # pre-lock window burns the holdout and then fails the gate. That path is given the same
        # release-on-failure discipline as walk_forward (release_holdout_reservation, a post-burn
        # no-op) PLUS an explicit WARNING audit record below, so the race is observable and cannot
        # masquerade as fully closed. Fully closing it would require folding finalize_holdout into
        # the same atomic tx as the mint (record_gate_with_fdr_and_maybe_promote) — deferred.
        def _on_peek(cfg: str) -> None:
            if breadth.pending_novel_family is not None:
                _revalidate_pending_novel(repo, name, breadth.pending_novel_family)
            repo.finalize_holdout_reservation(
                reservation_id, config_hash=cfg, strategy_id=sid)

        try:
            wf = walk_forward(
                strategy, provider, start_dt, end_dt, windows=windows,
                holdout_frac=holdout_frac, universe_by_date=universe_by_date,
                universe_name=universe, universe_snapshots=universe_prov,
                fundamentals_provider=fundamentals_provider, news_provider=news_provider,
                delisting_records=delisting_records,
                assume_terminal_last_close=assume_terminal_last_close,
                # Burn-on-peek: commit the reservation into a burn the instant BEFORE walk_forward
                # evaluates the holdout metric. Because release_holdout_reservation no-ops on a
                # committed row, the except-release below is then correct for EVERY post-peek
                # failure (incl. KeyboardInterrupt) — a computed holdout can never be released.
                on_peek=_on_peek,
            )
        except BaseException:
            # Pre-peek failure: the row is still pending, so release frees the window. Post-peek
            # failure: on_peek already committed, so this DELETE matches 0 rows (harmless no-op) and
            # the burn survives. Swallow a release error so it never masks the original failure.
            try:
                repo.release_holdout_reservation(reservation_id)
            except Exception:
                pass
            raise
        # walk_forward returned, so on_peek ran and the holdout burn is ALREADY committed. Give
        # run_gate the same release-on-failure discipline (#524 R9-H4 residual): a raise here —
        # including run_gate's own pre-lock pending-NOVEL drift re-check — leaves a burned holdout,
        # so release_holdout_reservation is a post-burn no-op. Emit an explicit WARNING audit record
        # for exactly this post-peek/pre-commit failure so the narrowed race is monitored, never
        # silently masquerading as fully closed. Then re-raise unchanged.
        try:
            outcome = run_gate(
                repo, wf, name=name, actor=actor_enum, criteria=criteria, breadth=breadth,
                universe_name=universe, universe_snapshots=universe_prov,
                period_start=start_dt.date(), period_end=end_dt.date(), holdout_frac=holdout_frac,
                data_source=data_source, snapshot_id=snapshot_id, allow_non_pit=allow_non_pit,
                holdout_evaluation_id=reservation_id, attempt_token=attempt_token,
                reason_suffix=("; holdout_reuse=" + _HOLDOUT_REUSE_OVERRIDE) if reused else "")
        except BaseException as exc:
            try:
                repo.release_holdout_reservation(reservation_id)  # post-burn no-op
            except Exception:
                pass
            get_logger(__name__).warning(
                "holdout_burned_post_peek_gate_failed",
                extra={"fields": {
                    "strategy": name, "reservation_id": reservation_id,
                    "exc_type": type(exc).__name__, "exc_message": str(exc),
                    "note": "#524 R9-H4 narrowed residual: holdout was burned at on_peek before "
                            "run_gate raised; window documented + monitored, not fully closed",
                }},
            )
            raise
        decision, promoted = outcome.decision, outcome.promoted
        # Advisory negative-result capture (#332): on a gate FAIL only, record the refuted
        # hypothesis into the experience log so it is not lost with the branch. BEST-EFFORT — a
        # capture failure NEVER breaks the promote (it is knowledge-capture, not a gate). Pre-gate
        # refusals / post-peek crashes are operator errors / operational burns, not refuted
        # hypotheses, and are intentionally out of scope (a manual `research log record` covers
        # arbitrary discards).
        if not promoted:
            experience_log = capture_gate_fail_experience(
                conn, name=name, decision=decision, actor=actor_enum,
                config_hash=wf.config_hash, strategy_id=sid,
                period_start=start_dt.date().isoformat(), period_end=end_dt.date().isoformat(),
                holdout=wf.holdout_metrics, stability=wf.stability)

    payload: dict[str, Any] = {
        **decision.to_dict(),
        "n_funnel": decision.n_combos,
        "strategy": name,
        "promoted": promoted,
        "config_hash": wf.config_hash,
        "snapshot_id": wf.snapshot_id,
        "holdout": wf.holdout_metrics,
        "stability": wf.stability,
        "universe_name": wf.universe_name,
        "universe_snapshots": wf.universe_snapshots,
        "fundamentals_snapshot": wf.fundamentals_snapshot,
        "news_snapshot": wf.news_snapshot,
    }
    if reused:
        payload["holdout_reuse"] = _HOLDOUT_REUSE_OVERRIDE
    if experience_log is not None:
        payload["experience_log"] = experience_log
    # Re-sync the kb doc to the (possibly) new stage (#331): best-effort, out-of-transaction —
    # the `with registry_conn() as conn:` block above has already committed and closed.
    sync_kb_doc(name)
    return payload
