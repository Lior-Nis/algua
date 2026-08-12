"""Merge-back authoritative intake: register a factory survivor + reproduce its promote evidence.

The factory research loop explores on a per-run SCRATCH registry inside a sandboxed worktree; the
producer is forbidden from writing the authoritative DB. A survivor therefore reaches the trusted
drainer's ``paper merge-back`` saga with (1) NO authoritative registry row and (2) NO authoritative
promote evidence (search breadth from ``backtest sweep``; a persisted ``backtest_returns`` series
for the family classifier). This module is the saga's post-merge chokepoint that closes both gaps
AUTHORITATIVELY — scratch evidence is NEVER imported (a sandboxed agent could under-report breadth);
the queue transports a validated *recipe* (data-context ids + the exact sweep grid) and the trusted
drainer re-runs it here against the just-merged module on ``main``.

Two helpers, invoked by ``run_merge_back`` between the durable gate-green merge and the metered
promote (see the design doc ``docs/superpowers/specs/2026-08-12-mergeback-authoritative-intake-
design.md``):

* :func:`ensure_backtested` — ONE ``BEGIN IMMEDIATE`` transaction: create-if-absent + CAS
  transition to ``backtested``. Idempotent; fails closed on any stage other than absent/idea/
  backtested. Returns ``"created" | "existed"`` (decided inside the same tx — race-free input to
  the produce_evidence skip predicate).
* :func:`produce_evidence` — reproduce the promote evidence against the transported pinned context
  via the injected sweep/backtest task callables (the REAL ``backtest sweep`` / ``backtest run``
  bodies, so breadth is truly measured and metered). Skipped for a pre-existing strategy that
  already carries authoritative breadth (the direct-authoritative-funnel no-op — a #534-style
  strategy must not be re-swept or double-counted).

Idempotency mechanism (design §C.2): ``record_search_trial`` and ``persist_backtest_returns`` are
single-row AUTOCOMMIT inserts running INSIDE the reused task functions, so trial + returns + marker
cannot land in one caller transaction. The fallback the design mandates is used instead:
a ``mergeback_evidence`` marker row (UNIQUE on ``strategy_id + branch_tip``) written ``'started'``
— with a ``search_trials`` MAX(id) watermark and the canonical grid hash — BEFORE the compute and
flipped ``'completed'`` AFTER both evidence rows landed, plus keyed dedup of the trial layer on
``(strategy_name, grid_json, id > watermark)`` at resume. Duplicate trials are NOT harmless (they
permanently inflate funnel/window breadth and the agent-NOVEL lifetime seed for later siblings),
so the dedup is REQUIRED, not cosmetic; a duplicate *returns* row is benign (the classifier reads
only the newest series for a name).

Lives in ``algua.registry`` (unprotected) so it may compose the repository, the contracts cost
floor, the strategy loader, and the shared sweep-grid validator — but it must NEVER import
``algua.cli``: the sweep/backtest task callables are INJECTED by the CLI wiring (the same seam
architecture as ``run_merge_back`` itself).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from typing import Any

from algua.backtest.sweep import parse_grid, validate_sweep_grid
from algua.contracts.lifecycle import Actor, Stage, validate_transition
from algua.contracts.registry_metadata import Author, HypothesisStatus
from algua.contracts.types import assert_gated_costs
from algua.registry.metadata import dump_tags
from algua.registry.store import SqliteStrategyRepository
from algua.strategies.loader import load_strategy

__all__ = [
    "INTAKE_TAG",
    "MergeBackIntakeError",
    "ensure_backtested",
    "evidence_marker",
    "produce_evidence",
]

# The provenance tag stamped on every strategy row this chokepoint CREATES, so a promote-failed
# orphan row (merge reverted, registry row deliberately kept — reverting registry state is the
# dangerous direction, #485 philosophy) is classifiable by gc/clustering/dashboards.
INTAKE_TAG = "mergeback:intake"


class MergeBackIntakeError(RuntimeError):
    """A fail-closed refusal from the merge-back intake chokepoint.

    Raised by :func:`ensure_backtested` when the strategy sits at a stage the intake must not
    touch, and by :func:`produce_evidence` on an inconsistent resume (grid drift vs the started
    marker, a resume without the grid that started it). The saga routes it through the same
    proven-failure revert machinery as a promote exception.
    """


def _now() -> str:
    return datetime.now(UTC).isoformat()


def ensure_backtested(
    conn: sqlite3.Connection,
    *,
    strategy: str,
    branch: str,
    branch_tip: str,
    merge_sha: str,
    base_sha: str,
) -> str:
    """Create-if-absent + CAS-transition ``strategy`` to ``backtested`` in ONE ``BEGIN IMMEDIATE``.

    The factory survivor's registration step: a missing row is created (stage ``idea``, actor
    provenance below) and advanced to ``backtested``; a row already at ``idea`` is advanced only; a
    row already at ``backtested`` is a no-op. ANY other stage fails closed — a candidate/paper/
    live/retired strategy must never be silently re-based by an autonomous merge-back.

    Returns ``"created"`` iff THIS call inserted the row, else ``"existed"`` — decided inside the
    same write transaction, so the produce_evidence skip predicate can never race a concurrent
    writer. Actor is ``Actor.AGENT``; the transition reason + (on create) the row tags bind the
    full merge provenance (:data:`INTAKE_TAG`, branch, branch_tip, merge_sha, base_sha).

    Deliberately inline SQL under one explicit transaction: the repository's ``add``/transition
    APIs each commit via ``with conn:`` (autocommit-per-call), so composing them could never yield
    the single-commit create+transition this chokepoint requires; ``store.py`` is CODEOWNERS-
    protected, so no commit-less variant can be added there. TOP-LEVEL ONLY (mirrors
    ``reserve_holdout``): a manual BEGIN inside an open transaction raises.
    """
    if conn.in_transaction:
        raise RuntimeError(
            "ensure_backtested must run at top level, not inside an open transaction")
    validate_transition(Stage.IDEA, Stage.BACKTESTED)  # the one edge this helper drives
    now = _now()
    reason = (
        f"merge-back intake ({INTAKE_TAG}): branch={branch} branch_tip={branch_tip} "
        f"merge_sha={merge_sha} base_sha={base_sha}"
    )
    tags = [INTAKE_TAG, f"mergeback:branch-tip:{branch_tip}"]
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT id, stage FROM strategies WHERE name = ?", (strategy,)
        ).fetchone()
        if row is None:
            cur = conn.execute(
                "INSERT INTO strategies"
                "(name, stage, created_at, updated_at, tags, author, hypothesis_status)"
                " VALUES (?,?,?,?,?,?,?)",
                (strategy, Stage.IDEA.value, now, now, dump_tags(tags),
                 Author.AGENT.value, HypothesisStatus.UNTESTED.value),
            )
            strategy_id = cur.lastrowid
            assert strategy_id is not None
            conn.execute(
                "INSERT INTO stage_transitions"
                "(strategy_id, from_stage, to_stage, actor, reason, created_at)"
                " VALUES (?,?,?,?,?,?)",
                (strategy_id, None, Stage.IDEA.value, Actor.SYSTEM.value, "created", now),
            )
            _advance_idea_to_backtested(conn, int(strategy_id), reason, now)
            status = "created"
        else:
            stage = str(row["stage"])
            if stage == Stage.BACKTESTED.value:
                status = "existed"  # idempotent no-op (incl. the crash-resume of THIS chokepoint)
            elif stage == Stage.IDEA.value:
                _advance_idea_to_backtested(conn, int(row["id"]), reason, now)
                status = "existed"
            else:
                raise MergeBackIntakeError(
                    f"merge-back intake refuses to touch {strategy!r} at stage {stage!r}; only "
                    f"absent/idea/backtested strategies may be (re)based by an autonomous "
                    f"merge-back (fail closed)")
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    return status


def _advance_idea_to_backtested(
    conn: sqlite3.Connection, strategy_id: int, reason: str, now: str
) -> None:
    """Commit-less ``idea -> backtested`` CAS + audit transition row; caller owns the txn."""
    cur = conn.execute(
        "UPDATE strategies SET stage = ?, updated_at = ? WHERE id = ? AND stage = ?",
        (Stage.BACKTESTED.value, now, strategy_id, Stage.IDEA.value),
    )
    if cur.rowcount != 1:  # unreachable under BEGIN IMMEDIATE; guards a caller bug
        raise MergeBackIntakeError(
            f"idea->backtested CAS failed for strategy_id={strategy_id} (stage moved)")
    conn.execute(
        "INSERT INTO stage_transitions"
        "(strategy_id, from_stage, to_stage, actor, reason, created_at)"
        " VALUES (?,?,?,?,?,?)",
        (strategy_id, Stage.IDEA.value, Stage.BACKTESTED.value, Actor.AGENT.value, reason, now),
    )


# ---------------------------------------------------------------------------- evidence marker

def evidence_marker(
    conn: sqlite3.Connection, strategy_id: int, branch_tip: str
) -> sqlite3.Row | None:
    """The ``mergeback_evidence`` marker row for ``(strategy_id, branch_tip)``, or None."""
    return conn.execute(
        "SELECT * FROM mergeback_evidence WHERE strategy_id = ? AND branch_tip = ?",
        (strategy_id, branch_tip),
    ).fetchone()


def _create_started_marker(
    conn: sqlite3.Connection, strategy_id: int, branch_tip: str, grid_hash: str,
) -> None:
    """Insert the ``'started'`` marker with the current ``search_trials`` MAX(id) watermark."""
    watermark = conn.execute(
        "SELECT COALESCE(MAX(id), 0) FROM search_trials").fetchone()[0]
    with conn:
        conn.execute(
            "INSERT INTO mergeback_evidence"
            "(strategy_id, branch_tip, grid_hash, search_trials_watermark, status, created_at)"
            " VALUES (?,?,?,?,'started',?)",
            (strategy_id, branch_tip, grid_hash, int(watermark), _now()),
        )


def _complete_marker(conn: sqlite3.Connection, strategy_id: int, branch_tip: str) -> None:
    with conn:
        cur = conn.execute(
            "UPDATE mergeback_evidence SET status='completed', completed_at=?"
            " WHERE strategy_id = ? AND branch_tip = ? AND status='started'",
            (_now(), strategy_id, branch_tip),
        )
        if cur.rowcount != 1:
            raise MergeBackIntakeError(
                f"evidence marker for strategy_id={strategy_id} branch_tip={branch_tip} "
                f"vanished/flipped mid-produce; refusing to claim completion")


def _canonical_grid(sweep_params: list[str]) -> tuple[dict[str, list[Any]], str, str]:
    """Parse the transported ``KEY=v1,v2`` params into ``(grid, canonical_json, grid_hash)``.

    The canonical JSON is byte-identical to what ``record_search_breadth`` stores as
    ``grid_json`` (``json.dumps(result.grid, sort_keys=True)`` over the SAME ``parse_grid``
    output), so the trial-layer dedup can key on exact equality.
    """
    grid = parse_grid(sweep_params)
    canonical = json.dumps(grid, sort_keys=True)
    return grid, canonical, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def produce_evidence(
    *,
    strategy: str,
    branch_tip: str,
    ensure_status: str,
    sweep_params: list[str] | None,
    conn_factory: Callable[[], AbstractContextManager[sqlite3.Connection]],
    sweep_fn: Callable[[], dict],
    backtest_fn: Callable[[], dict],
    load_strategy_fn: Callable[[str], Any] = load_strategy,
) -> str:
    """Reproduce ``strategy``'s promote evidence authoritatively (or prove it may be skipped).

    Runs at the saga chokepoint AFTER the gate-green merge is durably on ``main`` (so the module
    the injected tasks load IS the merged code) and immediately before the metered promote.
    Returns the ``evidence_status`` the saga mirrors into its journal:

    * ``"already_produced"`` — a ``'completed'`` marker exists for this ``(strategy, branch_tip)``:
      a prior attempt finished recording; nothing is re-recorded (re-recording would permanently
      inflate breadth).
    * ``"authoritative_breadth"`` — the row pre-existed this attempt (``ensure_status ==
      "existed"``), no marker was ever started for this attempt, and the strategy already carries
      authoritative measured breadth: the direct-authoritative-funnel no-op (a #534-style strategy
      with fresh authoritative evidence must not be re-swept or double-counted).
    * ``"no_context"`` — no sweep grid was transported. Nothing is produced; the promote gate's own
      binding breadth floor fails closed downstream if evidence is genuinely missing, so this
      cannot manufacture a promotable state.
    * ``"produced"`` — the full evidence recipe ran: grid validated against the now-on-``main``
      module (:func:`validate_sweep_grid` — the same checks the real sweep raises), the agent cost
      floor asserted BEFORE anything persists (``assert_gated_costs`` — the classifier's returns
      must be the same cost-realistic stream promote evaluates), the injected sweep task recorded
      the measured breadth, the injected full-period backtest task persisted ``backtest_returns``,
      and the marker flipped ``'completed'``.

    Crash idempotency (design §C.2, marker-last + keyed dedup): the ``'started'`` marker (with a
    ``search_trials`` watermark) lands BEFORE any compute; on resume the trial layer is deduped on
    ``(strategy_name, canonical grid_json, id > watermark)`` so a crash between the sweep task's
    autocommitted trial row and the marker flip can never double-count breadth. A resume that
    transports a DIFFERENT grid than the one that started the marker fails closed.
    """
    grid_inputs = _canonical_grid(sweep_params) if sweep_params else None
    with conn_factory() as conn:
        rec = SqliteStrategyRepository(conn).get(strategy)
        marker = evidence_marker(conn, rec.id, branch_tip)
        if marker is not None and marker["status"] == "completed":
            return "already_produced"
        if marker is None and ensure_status == "existed" and (
                SqliteStrategyRepository(conn).total_search_combos(strategy) > 0):
            return "authoritative_breadth"
        if grid_inputs is None:
            if marker is not None:
                raise MergeBackIntakeError(
                    f"evidence production for {strategy!r} @ {branch_tip} was started with a "
                    f"sweep grid but this resume transported none; refusing an inconsistent "
                    f"resume (fail closed)")
            return "no_context"
        grid, canonical_json, grid_hash = grid_inputs
        if marker is not None and marker["grid_hash"] != grid_hash:
            raise MergeBackIntakeError(
                f"evidence production for {strategy!r} @ {branch_tip} was started with grid_hash "
                f"{marker['grid_hash']} but this resume transported {grid_hash}; refusing an "
                f"inconsistent resume (fail closed)")

    # Validate BEFORE the marker exists (an invalid recipe must leave no 'started' residue) and
    # BEFORE anything persists (cost floor: classifier returns == promote's cost-realistic stream).
    loaded = load_strategy_fn(strategy)
    validate_sweep_grid(loaded, grid)
    assert_gated_costs(loaded.execution)

    if marker is None:
        with conn_factory() as conn:
            _create_started_marker(conn, rec.id, branch_tip, grid_hash)
            created = evidence_marker(conn, rec.id, branch_tip)
            assert created is not None  # just inserted under the same connection
            watermark = created["search_trials_watermark"]
    else:
        watermark = marker["search_trials_watermark"]

    with conn_factory() as conn:
        trial_recorded = conn.execute(
            "SELECT 1 FROM search_trials"
            " WHERE strategy_name = ? AND grid_json = ? AND id > ? LIMIT 1",
            (strategy, canonical_json, int(watermark)),
        ).fetchone() is not None
    if not trial_recorded:
        sweep_fn()  # records the single search_trials breadth row (autocommit, inside the task)
    backtest_fn()  # persists backtest_returns for the registered strategy (autocommit, inside)

    with conn_factory() as conn:
        _complete_marker(conn, rec.id, branch_tip)
    return "produced"
