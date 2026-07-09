from __future__ import annotations

import hashlib
import json
import shutil
import time
from collections import Counter
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import typer

import algua.strategies
from algua.backtest.engine import holdout_window
from algua.backtest.sweep import parse_grid, sweep_with_matrix
from algua.backtest.walkforward import walk_forward
from algua.cli._common import (
    authenticate_actor,
    now_iso,
    ok,
    project,
    registry_conn,
    resolve_delisting_inputs,
    resolve_eval_inputs,
    resolve_universe_inputs,
    select_provider,
    sync_kb_doc,
    utc,
)
from algua.cli.app import app, emit
from algua.cli.errors import json_errors
from algua.config.settings import get_settings
from algua.contracts.lifecycle import Actor, Stage
from algua.data.models import Dataset
from algua.data.serve import StoreBackedFundamentalsProvider, StoreBackedNewsProvider
from algua.data.store import DataStore
from algua.knowledge.experience import write_experience_note
from algua.registry.human_actor import canonical_run_context
from algua.registry.negative_results import (
    build_gate_fail_record,
    record_negative_result,
    sanitize_record,
)
from algua.registry.promotion import promotion_preflight, run_gate
from algua.registry.search_breadth import record_search_breadth
from algua.registry.store import SqliteStrategyRepository
from algua.research import family_audit, lifecycle_gc
from algua.research.cscv import CSCV_MIN_WINDOWS
from algua.research.cscv import pbo as compute_pbo
from algua.research.gates import GateCriteria
from algua.strategies.base import config_hash
from algua.strategies.loader import load_strategy


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

research_app = typer.Typer(help="Research workflow: gates and promotion", no_args_is_help=True)
app.add_typer(research_app, name="research")

_HOLDOUT_REUSE_OVERRIDE = "override"

# --summary keep-list (#349): the decision essence of a promote — the pass/fail verdict, the
# per-check breakdown (`checks` carries each gate's name/value/threshold/pass), the breadth and
# the binding flags, the holdout/stability scalars, and provenance. Keep-list (not drop-list) so
# the ~25 deep dsr_*/fdr_* internals, per-regime sharpes, and shadow-audit fields are
# excluded-by-default from the operator-facing summary (context-rot defense).
_PROMOTE_SUMMARY_KEYS = (
    "promoted", "strategy", "passed", "checks", "n_combos", "n_funnel", "breadth_provenance",
    "base_min_holdout_sharpe", "effective_min_holdout_sharpe", "pit_ok", "pit_override",
    "dsr_binding", "dsr_bootstrap_binding", "fdr_binding", "regime_robustness_binding",
    "returns_available", "holdout", "stability", "config_hash", "snapshot_id", "universe_name",
    "universe_snapshots", "fundamentals_snapshot", "news_snapshot", "holdout_reuse",
)


@research_app.command("promote")
# sqlite3.OperationalError keeps lock-contention ("database is locked") from reserve_holdout's
# BEGIN IMMEDIATE inside the JSON envelope, not a leaked traceback (CLI JSON-output contract).
@json_errors
def promote(
    name: str,
    start: str = typer.Option("2023-01-01", "--start"),
    end: str = typer.Option("2023-12-31", "--end"),
    demo: bool = typer.Option(False, "--demo", help="use the synthetic data provider"),
    snapshot: str = typer.Option(None, "--snapshot", help="backtest an ingested bars snapshot id"),
    fundamentals_snapshot: str = typer.Option(
        None, "--fundamentals-snapshot",
        help="ingested fundamentals snapshot id (required for a needs_fundamentals strategy)"),
    news_snapshot: str = typer.Option(
        None, "--news-snapshot",
        help="ingested news snapshot id (required for a needs_news strategy)"),
    universe: str = typer.Option(
        None, "--universe",
        help="point-in-time universe name (opt into survivorship-bias-free membership)"),
    windows: int = typer.Option(4, "--windows", help="walk-forward windows"),
    holdout_frac: float = typer.Option(0.2, "--holdout-frac", help="fraction reserved as holdout"),
    min_holdout_sharpe: float = typer.Option(0.5, "--min-holdout-sharpe"),
    min_holdout_return: float = typer.Option(0.0, "--min-holdout-return"),
    min_pct_positive: float = typer.Option(0.6, "--min-pct-positive"),
    min_window_sharpe: float = typer.Option(0.0, "--min-window-sharpe"),
    n_combos: int = typer.Option(
        None, "--n-combos",
        help="OPERATOR DECLARATION of search breadth, used ONLY when no measured sweep trials "
             "exist; the measured sum from `backtest sweep` is preferred and always wins",
    ),
    allow_holdout_reuse: bool = typer.Option(
        False, "--allow-holdout-reuse",
        help="OVERRIDE the single-use holdout guard: re-evaluate a holdout window already burned "
             "by a prior promote. Records the reuse (reused=1) and marks it in the audit trail. "
             "Statistically costly — only with fresh justification.",
    ),
    allow_non_pit: bool = typer.Option(
        False, "--allow-non-pit",
        help="HUMAN-ONLY override: promote a non-PIT (survivorship-biased) backtest. Audited. "
             "Agents may not pass this.",
    ),
    delistings: str = typer.Option(
        None, "--delistings",
        help="delistings snapshot handle (survivorship-free: realize held delisted names)"),
    assume_terminal_last_close: bool = typer.Option(
        False, "--assume-terminal-last-close",
        help="HUMAN-ONLY: realize a held-into-gap name at its last close when no delisting record "
             "exists. An agent must supply explicit delisting records; no-record-gap fails closed.",
    ),
    actor: str = typer.Option("agent", "--actor", help="human | agent | system"),
    actor_signature: str = typer.Option(
        None, "--actor-signature",
        help="path to the SSH signature over the printed human-actor challenge (#329). Required to "
             "authenticate --actor human: a bare --actor human unlocks NO human-only path — run "
             "once without this to print a challenge, sign it with your enrolled algua-human-actor "
             "key (ssh-keygen -Y sign -n algua-human-actor), then re-run with --actor-signature."),
    new_family: str = typer.Option(
        None, "--new-family",
        help="HUMAN-ONLY: slug for a new family when clustering verdict is NOVEL or PARENTAGE. "
             "Ignored when the strategy is already assigned to a family. "
             "Required for a human actor facing a NOVEL verdict.",
    ),
    summary: bool = typer.Option(
        False, "--summary",
        help="emit only decision-relevant scalars (drops deep dsr_*/fdr_*/regime diagnostics; "
             "context-rot defense)"),
) -> None:
    """Gate backtested->candidate on walk-forward holdout + stability; promote only on pass.

    The holdout-Sharpe bar is DEFLATED by FUNNEL-WIDE search breadth (the multiple-testing defense):
    the max of this strategy's lifetime recorded breadth and the funnel-wide breadth in the rolling
    window. Breadth is MEASURED as the sum of recorded `search_trials` (from `backtest sweep`); an
    agent must have measured breadth (no measured trials => refused). Declaring breadth via
    --n-combos is HUMAN-ONLY and recorded with provenance="declared" (auditably less trustworthy).
    For an agent the universe must be PIT (`--universe`); non-PIT fails closed unless a human passes
    --allow-non-pit. A minimum holdout-observations floor (63) also fails closed (underpowered
    holdouts). On pass for an agent this mints the single-use gate token the BACKTESTED->CANDIDATE
    transition consumes.
    """
    payload = promote_task(
        name, start=start, end=end, demo=demo, snapshot=snapshot,
        fundamentals_snapshot=fundamentals_snapshot, news_snapshot=news_snapshot,
        universe=universe, windows=windows, holdout_frac=holdout_frac,
        min_holdout_sharpe=min_holdout_sharpe, min_holdout_return=min_holdout_return,
        min_pct_positive=min_pct_positive, min_window_sharpe=min_window_sharpe,
        n_combos=n_combos, allow_holdout_reuse=allow_holdout_reuse, allow_non_pit=allow_non_pit,
        delistings=delistings, assume_terminal_last_close=assume_terminal_last_close,
        actor=actor, actor_signature=actor_signature, new_family=new_family,
    )
    out = ok(payload)
    emit(project(out, _PROMOTE_SUMMARY_KEYS) if summary else out)


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
            new_family_slug=new_family)
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
                on_peek=lambda cfg: repo.finalize_holdout_reservation(
                    reservation_id, config_hash=cfg, strategy_id=sid),
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
        outcome = run_gate(
            repo, wf, name=name, actor=actor_enum, criteria=criteria, breadth=breadth,
            universe_name=universe, universe_snapshots=universe_prov,
            period_start=start_dt.date(), period_end=end_dt.date(), holdout_frac=holdout_frac,
            data_source=data_source, snapshot_id=snapshot_id, allow_non_pit=allow_non_pit,
            holdout_evaluation_id=reservation_id, attempt_token=attempt_token,
            reason_suffix=("; holdout_reuse=" + _HOLDOUT_REUSE_OVERRIDE) if reused else "")
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
    # the `with registry_conn()` block above has already committed and closed.
    sync_kb_doc(name)
    return payload


@research_app.command("dormant-sweep")
@json_errors
def dormant_sweep(
    start: str = typer.Option("2023-01-01", "--start"),
    end: str = typer.Option("2023-12-31", "--end"),
    demo: bool = typer.Option(False, "--demo", help="use the synthetic data provider"),
    snapshot: str = typer.Option(None, "--snapshot", help="screen an ingested bars snapshot id"),
    universe: str = typer.Option(
        None, "--universe",
        help="optional single point-in-time universe applied to ALL dormant strategies; "
             "omit to use each strategy's own static universe"),
    windows: int = typer.Option(4, "--windows", help="walk-forward windows"),
    holdout_frac: float = typer.Option(
        0.2, "--holdout-frac",
        help="walk-forward holdout fraction (shapes the windows; the holdout is NOT revealed)"),
    min_window_sharpe: float = typer.Option(
        0.0, "--min-window-sharpe", help="screen threshold on MEAN walk-forward window Sharpe"),
    min_pct_positive: float = typer.Option(
        0.6, "--min-pct-positive",
        help="screen threshold on the fraction of positive walk-forward windows"),
    top: int = typer.Option(
        None, "--top", help="cap passed/failed lists to the top N by mean window Sharpe"),
) -> None:
    """Advisory STABILITY screen over the dormant pool. For each dormant strategy, re-run
    walk-forward on a common window and report whether its WINDOW/stability metrics look healthy
    again. This is NOT a gate: it never reads, reveals, or burns the single-use holdout, writes no
    ledger rows, and transitions nothing. A pass is a prioritization signal (re-audition via
    `registry transition --to paper`), not a guarantee of re-promotion or forward-gate clearance."""
    if not 0.0 <= min_pct_positive <= 1.0:
        raise ValueError("--min-pct-positive must be in [0, 1]")
    if top is not None and top < 1:
        raise ValueError("--top must be >= 1 when provided")
    start_dt, end_dt = utc(start), utc(end)
    provider = select_provider(demo, snapshot)
    data_source = type(provider).__name__
    snapshot_id = getattr(provider, "snapshot_id", None)
    universe_by_date, universe_prov = (
        resolve_universe_inputs(universe, start_dt, end_dt) if universe else (None, None))

    with registry_conn() as conn:
        dormant = SqliteStrategyRepository(conn).list_strategies(Stage.DORMANT)

    passed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for rec in dormant:
        try:
            strategy = load_strategy(rec.name)
            # walk_forward CAN thread PIT sidecars now (#132), but dormant-sweep takes a single
            # --snapshot and cannot carry a per-strategy fundamentals/news snapshot across a
            # heterogeneous pool — so skip PIT strategies here with an accurate reason rather than
            # letting them land in errors[]. Re-audition them individually instead.
            sidecar = next(
                (flag for flag in ("needs_fundamentals", "needs_news")
                 if getattr(strategy.config, flag, False)), None)
            if sidecar is not None:
                snap_flag = "fundamentals" if sidecar == "needs_fundamentals" else "news"
                skipped.append({
                    "strategy": rec.name,
                    "reason": f"{sidecar}: dormant-sweep takes no per-strategy PIT snapshot — "
                              f"re-audition individually via "
                              f"backtest walk-forward/research promote --{snap_flag}-snapshot"})
                continue
            wf = walk_forward(
                strategy, provider, start_dt, end_dt, windows=windows,
                holdout_frac=holdout_frac, universe_by_date=universe_by_date,
                universe_name=universe, universe_snapshots=universe_prov)
            stability = wf.stability
            screen_passed = (stability["mean_sharpe"] >= min_window_sharpe
                             and stability["pct_positive_windows"] >= min_pct_positive)
            result = {
                "strategy": rec.name, "screen_passed": screen_passed,
                "stability": stability, "windows": wf.window_metrics,
                "config_hash": wf.config_hash, "universe_name": wf.universe_name,
                "universe_snapshots": wf.universe_snapshots,
                "pit": universe_prov is not None,
            }
            (passed if screen_passed else failed).append(result)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:  # noqa: BLE001 - per-strategy isolation: one bad strategy must not abort the sweep
            errors.append({"strategy": rec.name, "error": f"{type(e).__name__}: {e}"})

    evaluated = len(passed) + len(failed)
    passed.sort(key=lambda r: r["stability"]["mean_sharpe"], reverse=True)
    failed.sort(key=lambda r: r["stability"]["mean_sharpe"], reverse=True)
    if top is not None:
        passed, failed = passed[:top], failed[:top]
    emit(ok({
        "note": ("advisory stability screen over walk-forward windows; NOT the holdout gate. A "
                 "pass means the strategy's windows look healthy again - worth re-auditioning via "
                 "`registry transition --to paper` - it does NOT guarantee it will clear "
                 "re-promotion (which burns a fresh holdout) or the #124 forward gate. Residual "
                 "multiple-testing risk: acting on top-ranked names is a human judgement."),
        "period": {"start": start_dt.date().isoformat(), "end": end_dt.date().isoformat()},
        "data_source": data_source, "snapshot_id": snapshot_id,
        "thresholds": {"min_window_sharpe": min_window_sharpe,
                       "min_pct_positive": min_pct_positive},
        "total_dormant": len(dormant), "evaluated": evaluated,
        "passed": passed, "failed": failed, "skipped": skipped, "errors": errors,
    }))


_PBO_NOTE = (
    "ADVISORY overfitting diagnostic (PBO/CSCV). PBO = the probability that the "
    "in-sample-best combo lands below the OOS median across combinatorially-"
    "symmetric splits; a high PBO (>~0.5) means the SELECTION RULE does not "
    "generalize OOS. RECORDS search breadth (metered — repeated runs self-penalize "
    "at promotion); burns NO holdout statistic, writes NO gate ledger row, "
    "transitions nothing. Aggregate-only (no matrix/ranked). Orthogonal to DSR/FDR."
)


@research_app.command("pbo")
@json_errors
def pbo_cmd(  # noqa: PLR0913
    name: str,
    start: str = typer.Option("2023-01-01", "--start"),
    end: str = typer.Option("2023-12-31", "--end"),
    demo: bool = typer.Option(False, "--demo", help="use the synthetic data provider"),
    snapshot: str = typer.Option(None, "--snapshot", help="screen an ingested bars snapshot id"),
    universe: str = typer.Option(
        None, "--universe",
        help="point-in-time universe name (opt into survivorship-bias-free membership)"),
    windows: int = typer.Option(
        4, "--windows",
        help="walk-forward windows per combo; must be >= 4 (CSCV_MIN_WINDOWS) or PBO fails closed. "
             "Odd counts are fine — CSCV bounds the sub-period count to an even S internally"),
    holdout_frac: float = typer.Option(0.2, "--holdout-frac", help="fraction reserved as holdout"),
    param: list[str] = typer.Option(None, "--param", help="KEY=v1,v2,... (repeatable)"),
    rank_by: str = typer.Option("mean_sharpe", "--rank-by", help="mean_sharpe | min_sharpe"),
    fundamentals_snapshot: str = typer.Option(
        None, "--fundamentals-snapshot",
        help="ingested fundamentals snapshot id (required for a needs_fundamentals strategy)"),
    news_snapshot: str = typer.Option(
        None, "--news-snapshot",
        help="ingested news snapshot id (required for a needs_news strategy)"),
    delistings: str = typer.Option(
        None, "--delistings",
        help="delistings snapshot handle (survivorship-free: realize held delisted names)"),
    assume_terminal_last_close: bool = typer.Option(
        False, "--assume-terminal-last-close",
        help="realize a held-into-gap name at its last close when no delisting record exists"),
) -> None:
    """ADVISORY overfitting diagnostic — Probability of Backtest Overfitting via CSCV (#467).

    Runs the SAME sweep as `backtest sweep`, then measures, across combinatorially-symmetric
    in-sample/out-of-sample window splits, how often the IS-best combo lands below the OOS median.
    That fraction is the PBO. High PBO => the sweep is fitting noise, not signal. This is a REAL
    grid search, so — exactly like `backtest sweep` — it RECORDS its measured breadth (repeated
    `pbo` runs self-penalize at promotion via funnel-wide breadth). It burns NO holdout STATISTIC,
    writes NO gate/FDR ledger row, and transitions nothing. Output is AGGREGATE-ONLY (no matrix, no
    ranked combos, no per-split detail). It is orthogonal to the DSR/FDR promotion machinery — a
    winner can look great on DSR yet have a high PBO (the SELECTION RULE doesn't generalize)."""
    strategy, provider, start_dt, end_dt = resolve_eval_inputs(name, demo, snapshot, start, end)
    universe_by_date, universe_prov = resolve_universe_inputs(universe, start_dt, end_dt)
    if fundamentals_snapshot and not strategy.config.needs_fundamentals:
        raise ValueError(
            "--fundamentals-snapshot was given but the strategy does not declare needs_fundamentals"
        )
    if news_snapshot and not strategy.config.needs_news:
        raise ValueError(
            "--news-snapshot was given but the strategy does not declare needs_news"
        )
    fundamentals_provider = (
        StoreBackedFundamentalsProvider(DataStore(get_settings().data_dir), fundamentals_snapshot)
        if fundamentals_snapshot
        else None
    )
    news_provider = (
        StoreBackedNewsProvider(DataStore(get_settings().data_dir), news_snapshot)
        if news_snapshot
        else None
    )
    # Capture the RESOLVED delisting snapshot id at the CLI layer for provenance — SweepResult
    # carries neither the resolved id, the raw handle, nor the assume-last-close flag (#467 R2-4).
    delisting_records, delisting_snapshot = resolve_delisting_inputs(delistings, end_dt)
    grid = parse_grid(param or [])
    # FULL untruncated grid hash (64 hex chars) so a recorded PBO is reconstructable to the exact
    # grid without emitting the raw grid dict (aggregate-only surface) — #467 R2-4.
    grid_hash = hashlib.sha256(json.dumps(grid, sort_keys=True).encode()).hexdigest()
    # Fail closed BEFORE the sweep on a sub-2 --windows. cscv.pbo is the fail-closed authority for
    # 2 <= windows < CSCV_MIN_WINDOWS (the sweep runs, its matrix has < 4 columns, cscv returns
    # pbo=None), but walk_forward's _segment_bounds REQUIRES windows >= 2 and would raise a
    # ValueError first, turning the advisory "pbo: null + warning, exit 0" contract into an exit-1
    # error envelope. A < 2 windows sweep is definitionally uncarveable, so short-circuit to the
    # SAME aggregate fail-closed payload here — no sweep runs (no real search, so no breadth
    # recorded), and the sweep-runtime provenance fields (code/dependency hash, data_source,
    # snapshot, seed, timeframe) are null because nothing ran to stamp them.
    if windows < 2:
        emit(ok({
            "note": _PBO_NOTE,
            "pbo": None,
            "split_count": 0,
            "trial_count": 0,
            "window_count": windows,
            "subperiod_count": 0,
            "rank_by": rank_by,
            "warnings": [
                f"PBO needs >= {CSCV_MIN_WINDOWS} windows; got {windows} (fail closed)"
            ],
            "provenance": {
                "config_hash": config_hash(strategy),
                "code_hash": None,
                "dependency_hash": None,
                "data_source": None,
                "snapshot_id": None,
                "timeframe": None,
                "seed": None,
                "period": {
                    "start": start_dt.date().isoformat(),
                    "end": end_dt.date().isoformat(),
                },
                "universe_name": universe,
                "universe_snapshots": universe_prov,
                "fundamentals_snapshot": fundamentals_snapshot or None,
                "news_snapshot": news_snapshot or None,
                "windows": windows,
                "holdout_frac": holdout_frac,
                "rank_by": rank_by,
                "grid_hash": grid_hash,
                "delisting_snapshot": delisting_snapshot,
                "delistings_name": delistings,
                "assume_terminal_last_close": assume_terminal_last_close,
            },
        }))
        return
    # sweep_with_matrix returns BOTH the SweepResult (for breadth recording) and the trials x
    # windows OOS-Sharpe matrix (for CSCV) as SEPARATE values — the matrix never rides on the
    # result (#467 R2-2). compute_holdout=False makes each combo's walk_forward skip the holdout
    # STATISTIC/burn entirely; the matrix is bit-identical to a normal sweep.
    result, matrix = sweep_with_matrix(
        strategy, provider, start_dt, end_dt,
        grid=grid, windows=windows, holdout_frac=holdout_frac, rank_by=rank_by,
        universe_by_date=universe_by_date,
        universe_name=universe, universe_snapshots=universe_prov,
        fundamentals_provider=fundamentals_provider,
        news_provider=news_provider,
        delisting_records=delisting_records,
        assume_terminal_last_close=assume_terminal_last_close,
        compute_holdout=False)
    # METER this search (#467 R2-3 / HIGH-1): record the sweep's measured breadth via the SAME path
    # `backtest sweep`/`sweep_task` use, keyed by strategy NAME, so repeated `pbo` runs inflate
    # funnel-wide breadth (over-counting only ever TIGHTENS downstream gates — fail-safe). This
    # records breadth ONLY; burns no holdout, writes no gate/FDR/holdout-burn row, mints no token.
    with registry_conn() as conn:
        record_search_breadth(SqliteStrategyRepository(conn), name, result)
    # A 1-combo grid (< 2 rows) or a `< 4` --windows makes cscv.pbo FAIL CLOSED — pbo=None + a
    # warning, exit 0 (advisory), never a raise.
    diag = compute_pbo(matrix, rank_by=rank_by)
    emit(ok({
        "note": _PBO_NOTE,
        "pbo": diag.pbo,
        "split_count": diag.split_count,
        "trial_count": diag.trial_count,
        "window_count": diag.window_count,
        "subperiod_count": diag.subperiod_count,
        "rank_by": diag.rank_by,
        "warnings": diag.warnings,
        "provenance": {
            "config_hash": config_hash(strategy),
            "code_hash": result.code_hash,
            "dependency_hash": result.dependency_hash,
            "data_source": result.data_source,
            "snapshot_id": result.snapshot_id,
            "timeframe": result.timeframe,
            "seed": result.seed,
            "period": result.period,
            "universe_name": result.universe_name,
            "universe_snapshots": result.universe_snapshots,
            "fundamentals_snapshot": result.fundamentals_snapshot,
            "news_snapshot": result.news_snapshot,
            "windows": result.windows,
            "holdout_frac": result.holdout_frac,
            "rank_by": result.rank_by,
            "grid_hash": grid_hash,
            "delisting_snapshot": delisting_snapshot,
            "delistings_name": delistings,
            "assume_terminal_last_close": assume_terminal_last_close,
        },
    }))


@research_app.command("family-audit")
@json_errors
def family_audit_cmd() -> None:
    """ADVISORY cross-family gaming detector. Scans the family DAG for separate families that
    empirically behave as one thesis (deliberate-split breadth evasion that #222's assignment-time
    clustering can't see), ranks them by family-term breadth dodged, and recommends a human-governed
    consolidation. READ-ONLY: no holdout, no ledger writes, no transitions, no graph mutation."""
    started = time.monotonic()
    with registry_conn() as conn:
        # One connection = one consistent read snapshot for the whole scan (no writes).
        repo = SqliteStrategyRepository(conn)
        conn.execute("BEGIN")
        profile_list = repo.all_families_with_member_profiles()
        profiles = {fid: members for fid, members in profile_list}
        fam_names = repo.family_names()

        # Batch-load each distinct strategy's returns ONCE (O(M), not O(M^2)).
        all_names = {m["name"] for members in profiles.values() for m in members}
        returns = {name: repo.load_backtest_returns(name) for name in all_names}
        returns = {k: v for k, v in returns.items() if v is not None}

        # Pipeline step 1: similarity-only candidate edges.
        candidate_edges = family_audit.flag_edges(profiles, returns)

        # Step 2 (CLI I/O): pairwise union breadth for candidate pairs only + per-family breadth.
        individual_breadth = {fid: repo.family_lifetime_combos(fid) for fid in profiles}
        pair_breadth = {
            frozenset({e.family_a, e.family_b}):
                repo.lifetime_combos_for_families([e.family_a, e.family_b])
            for e in candidate_edges
        }

        # Step 3: evasion skip + components.
        components, kept = family_audit.build_components(
            candidate_edges, pair_breadth=pair_breadth, individual_breadth=individual_breadth)

        # Step 4 (CLI I/O): unified breadth per component.
        component_breadth = {
            comp: repo.lifetime_combos_for_families(list(comp)) for comp in components}

        # Step 5: rank + assemble.
        active_counts = {fid: len(members) for fid, members in profiles.items()}
        clusters = family_audit.rank_clusters(
            components, kept, candidate_edges, component_breadth=component_breadth,
            individual_breadth=individual_breadth, family_names=fam_names,
            active_counts=active_counts)
        conn.rollback()

    n_pairs = len(list(combinations(profiles, 2)))
    emit(ok({
        "note": ("ADVISORY cross-family gaming screen; NOT a gate. Flags separate families that "
                 "behave as one thesis (return-correlation authoritative). Acting on a cluster "
                 "is a human judgement: consolidate via member reassignment (--actor human). "
                 "Mutates nothing."),
        "clusters": clusters,
        "n_families_scanned": len(profiles),
        "n_pairs_total": n_pairs,
        "n_pairs_flagged_or_inconclusive": len(candidate_edges),
        "n_pairs_skipped_zero_evasion": len([e for e in candidate_edges if e.flagged]) - len(kept),
        "wall_time_seconds": round(time.monotonic() - started, 3),
        "config": {
            "audit_flag_threshold": family_audit.AUDIT_FLAG_THRESHOLD,
            "return_independent_threshold": family_audit.RETURN_INDEPENDENT_THRESHOLD,
            "return_correlation_min_overlap": family_audit.RETURN_CORRELATION_MIN_OVERLAP,
        },
    }))


_GC_NOTE = (
    "advisory lifecycle GC: read-only by default; a listing is a prioritization signal, not a "
    "transition. Only --archive --actor human MOVES files; the immutable registry DB row is NEVER "
    "touched."
)


def _gc_inventory(settings: Any) -> list[lifecycle_gc.FileItem]:
    """Scan the two on-disk strategy surfaces into FileItems (the I/O the pure module refuses).

    Surface 1 — strategy modules: every ``*.py`` under a public family dir
    (``algua/strategies/<family>/`` with an ``__init__.py`` and a non-underscore name), skipping
    ``__init__.py`` and any private ``_*.py`` helper. Example families are scanned too; with no
    registry row they classify as ``untracked_module`` and are never reaped, so no special-casing.
    Surface 2 — reports: the TOP-LEVEL ``*.md`` files under ``<knowledge_dir>/strategies/`` (never
    recursing into ``families/`` subdirs), skipping ``README.md``.
    """
    items: list[lifecycle_gc.FileItem] = []
    root = Path(algua.strategies.__file__).parent
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name.startswith("_") or not (d / "__init__.py").exists():
            continue
        for p in sorted(d.glob("*.py")):
            if p.name == "__init__.py" or p.stem.startswith("_"):
                continue
            items.append(lifecycle_gc.FileItem(
                path=str(p), strategy=p.stem, surface=lifecycle_gc.SURFACE_MODULE,
                size_bytes=p.stat().st_size))
    reports_dir = settings.knowledge_dir / "strategies"
    if reports_dir.exists():
        for p in sorted(reports_dir.glob("*.md")):
            if p.name == "README.md":
                continue
            items.append(lifecycle_gc.FileItem(
                path=str(p), strategy=p.stem, surface=lifecycle_gc.SURFACE_REPORT,
                size_bytes=p.stat().st_size))
    return items


def _gc_archive(
    reap: list[lifecycle_gc.Classified], archive_dir: Path,
) -> tuple[str, list[dict[str, Any]]]:
    """MOVE each reapable file into a timestamped run dir under ``archive_dir``, mirroring its path.

    Idempotent/race-safe: a source that has already vanished (a concurrent GC, a manual delete) is
    skipped rather than raising, so a re-run of the same command moves nothing left to move.
    """
    run_dir = archive_dir / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    moved: list[dict[str, Any]] = []
    for c in reap:
        src = Path(c.path)
        if not src.exists():
            continue
        # Mirror the source's directory structure under run_dir. Strip the anchor first: scanned
        # paths are ABSOLUTE (both algua.strategies.__file__ and an absolute knowledge_dir), and
        # `run_dir / <absolute>` would collapse to the absolute path itself (pathlib join rule),
        # moving the file onto itself. Stripping the root ("/") yields a unique, nested dest.
        rel = src.relative_to(src.anchor) if src.is_absolute() else src
        dest = run_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))
        moved.append({
            "src": c.path, "dest": str(dest), "strategy": c.strategy, "surface": c.surface,
            "reason": c.reason, "size_bytes": c.size_bytes,
        })
    return (str(run_dir), moved)


@research_app.command("gc")
@json_errors
def gc(
    retention_days: float = typer.Option(
        90.0, "--retention-days",
        help="reap retired strategies only after this many days retired (conservative default)"),
    archive: bool = typer.Option(
        False, "--archive",
        help="GOVERNED cleanup: MOVE reapable files to the archive tree (human-only). Default is "
             "read-only advisory."),
    actor: str = typer.Option("agent", "--actor"),
    archive_dir: Path = typer.Option(
        Path("archive"), "--archive-dir", help="root of the archive tree for --archive"),
    top: int = typer.Option(
        None, "--top", help="cap the reapable list to the top N by reclaimable size"),
) -> None:
    """ADVISORY reaper of dead strategy artifacts — retired-strategy modules/reports & orphaned
    reports. READ-ONLY by default: it classifies the on-disk strategy modules and top-level reports
    against the registry and lists what is safely reapable (a retired strategy older than
    --retention-days, or a report with no registry row). A listing is a prioritization signal, NOT
    a transition. Fail-safe throughout: a module with no row (untracked), a non-terminal strategy,
    or a retired-without-timestamp is ALWAYS kept. Only `--archive --actor human` MOVES the reapable
    files into a timestamped archive tree; it NEVER deletes and NEVER touches the registry row."""
    if retention_days < 0:
        raise ValueError("--retention-days must be >= 0")
    if top is not None and top < 1:
        raise ValueError("--top must be >= 1 when provided")
    if archive and actor != "human":
        raise ValueError("research gc --archive is a governed cleanup: pass --actor human")

    settings = get_settings()
    items = _gc_inventory(settings)

    registry: dict[str, lifecycle_gc.RegistryEntry] = {}
    with registry_conn() as conn:
        repo = SqliteStrategyRepository(conn)
        for rec in repo.list_strategies():
            retired_at: str | None = None
            if rec.stage == Stage.RETIRED:
                trans = repo.list_transitions(rec.name)
                rets = [t["created_at"] for t in trans if t["to_stage"] == Stage.RETIRED.value]
                retired_at = max(rets) if rets else None
            registry[rec.name] = lifecycle_gc.RegistryEntry(
                stage=rec.stage.value, retired_at=retired_at)

    now = datetime.now(UTC)
    classified = lifecycle_gc.classify(
        items, registry, now=now, retention_days=retention_days)
    reap = lifecycle_gc.reapable(classified)
    by_reason: Counter[str] = Counter(c.reason for c in classified)

    archive_run_dir: str | None = None
    archived: list[dict[str, Any]] = []
    if archive and reap:
        archive_run_dir, archived = _gc_archive(reap, archive_dir)

    shown = reap[:top] if top else reap
    reapable_dicts = [{
        "path": c.path, "strategy": c.strategy, "surface": c.surface, "reason": c.reason,
        "size_bytes": c.size_bytes,
        "age_days": round(c.age_days, 3) if c.age_days is not None else None,
        "stage": c.stage,
    } for c in shown]

    emit(ok({
        "note": _GC_NOTE,
        "dry_run": not archive,
        "retention_days": retention_days,
        "total_files_scanned": len(classified),
        "reapable_count": len(reap),
        "reclaimable_bytes": sum(c.size_bytes for c in reap),
        "by_reason": dict(by_reason),
        "reapable": reapable_dicts,
        "archived": archived,
        "archive_run_dir": archive_run_dir,
    }))
