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
— with a ``search_trials`` MAX(id) watermark and the canonical RECIPE hash over the full eval
context (grid + snapshot|demo + sidecars + universe + window + delistings + rank_by) — BEFORE the
compute and flipped ``'completed'`` AFTER both evidence rows landed, plus keyed dedup of the trial
layer on ``(strategy_name, grid_json, id > watermark)`` at resume (resume ONLY — a fresh marker
always sweeps). Duplicate trials are NOT harmless (they
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

from algua.backtest.sweep import _RANK_KEYS, parse_grid, validate_sweep_grid
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
    "validate_transport_inputs",
]

# The provenance tag stamped on every strategy row this chokepoint CREATES, so a promote-failed
# orphan row (merge reverted, registry row deliberately kept — reverting registry state is the
# dangerous direction, #485 philosophy) is classifiable by gc/clustering/dashboards.
INTAKE_TAG = "mergeback:intake"
# The per-attempt creation-provenance tag (`mergeback:branch-tip:<sha>`), committed ATOMICALLY
# with the row create. It is the durable created-by-THIS-attempt witness the journal cannot be
# (a crash can always beat a journal append that follows the registry commit): on resume,
# ensure_backtested reads it back inside the same tx to classify created|existed (GATE-2 #1).
_BRANCH_TIP_TAG_PREFIX = "mergeback:branch-tip:"


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

    Returns ``"created"`` iff THIS ATTEMPT created the row — either this call inserted it, or a
    backtested row already carries THIS attempt's branch-tip creation tag (a crashed prior
    invocation of the SAME attempt created it and the crash beat the saga's journal write; the
    provenance tag is committed ATOMICALLY with the create, so it survives every crash window the
    journal cannot cover — GATE-2 #1 residual, closed). ``"existed"`` iff the row genuinely
    pre-existed this attempt (no merge-back creation provenance at all — the direct-authoritative-
    funnel case, or an ``idea`` row being advanced). A backtested row carrying FOREIGN or
    unreadable merge-back provenance fails closed (:class:`MergeBackIntakeError`) — never guess
    whose evidence a same-name orphan's breadth belongs to. All decided inside the same write
    transaction, so the produce_evidence skip predicate can never race a concurrent writer. Actor
    is ``Actor.AGENT``; the transition reason + (on create) the row tags bind the full merge
    provenance (:data:`INTAKE_TAG`, branch, branch_tip, merge_sha, base_sha).

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
    tags = [INTAKE_TAG, f"{_BRANCH_TIP_TAG_PREFIX}{branch_tip}"]
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT id, stage, tags FROM strategies WHERE name = ?", (strategy,)
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
                # No-op — but classify WHO created the row from its creation-provenance tags
                # (committed atomically with the create), inside this same tx. A crashed prior
                # invocation of THIS attempt must resume as "created" (its stale same-name breadth
                # must not satisfy the direct-funnel skip); foreign/unreadable merge-back
                # provenance fails closed (GATE-2 #1 residual, closed).
                status = _classify_existing_backtested(row["tags"], strategy, branch_tip)
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


def _classify_existing_backtested(raw_tags: str | None, strategy: str, branch_tip: str) -> str:
    """Classify a pre-found ``backtested`` row as ``created`` | ``existed`` from its provenance
    tags — MUST run inside ensure_backtested's ``BEGIN IMMEDIATE`` (the classification is part of
    the same race-free created|existed decision).

    * THIS attempt's branch-tip tag present → ``"created"``: a crashed prior invocation of the
      SAME attempt created the row; treating it as pre-existing would let stale SAME-NAME
      search_trials satisfy the direct-funnel skip and carry promote without this attempt's sweep.
    * NO merge-back provenance at all (NULL tags, or a tag list without any ``mergeback:*``
      creation tag) → ``"existed"``: the genuine pre-existing case.
    * FOREIGN merge-back provenance (another attempt's branch-tip tag / a bare intake tag) or an
      unreadable/malformed tags column → fail closed. Whose evidence a same-name merge-back
      orphan's breadth belongs to cannot be guessed.
    """
    if raw_tags is None:
        return "existed"
    try:
        parsed = json.loads(raw_tags)
    except (TypeError, ValueError) as exc:
        raise MergeBackIntakeError(
            f"strategy {strategy!r} is at backtested but its tags column is unreadable "
            f"({raw_tags!r}); cannot classify merge-back creation provenance (fail closed)"
        ) from exc
    if not isinstance(parsed, list) or not all(isinstance(t, str) for t in parsed):
        raise MergeBackIntakeError(
            f"strategy {strategy!r} is at backtested but its tags column is not a tag list "
            f"({raw_tags!r}); cannot classify merge-back creation provenance (fail closed)")
    tags = {t.strip().lower() for t in parsed}
    if f"{_BRANCH_TIP_TAG_PREFIX}{branch_tip}".lower() in tags:
        return "created"
    if INTAKE_TAG in tags or any(t.startswith(_BRANCH_TIP_TAG_PREFIX) for t in tags):
        raise MergeBackIntakeError(
            f"strategy {strategy!r} is at backtested with FOREIGN merge-back creation provenance "
            f"(tags {sorted(tags)!r}, this attempt's branch_tip {branch_tip}); refusing to treat "
            f"another attempt's orphan as pre-existing evidence (fail closed)")
    return "existed"


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
    conn: sqlite3.Connection, strategy_id: int, branch_tip: str, recipe_hash: str,
) -> int:
    """Insert the ``'started'`` marker and return its ``search_trials`` MAX(id) watermark.

    The watermark read and the marker insert happen under ONE ``BEGIN IMMEDIATE`` (GATE-2 #3): a
    separate autocommit read would let a trial row land between the read and the insert, making
    the recorded watermark lie about what predated this attempt. TOP-LEVEL ONLY.
    """
    if conn.in_transaction:
        raise RuntimeError(
            "_create_started_marker must run at top level, not inside an open transaction")
    try:
        conn.execute("BEGIN IMMEDIATE")
        watermark = int(conn.execute(
            "SELECT COALESCE(MAX(id), 0) FROM search_trials").fetchone()[0])
        conn.execute(
            "INSERT INTO mergeback_evidence"
            "(strategy_id, branch_tip, recipe_hash, search_trials_watermark, status, created_at)"
            " VALUES (?,?,?,?,'started',?)",
            (strategy_id, branch_tip, recipe_hash, watermark, _now()),
        )
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    return watermark


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


def _canonical_grid(sweep_params: list[str]) -> tuple[dict[str, list[Any]], str]:
    """Parse the transported ``KEY=v1,v2`` params into ``(grid, canonical_json)``.

    The canonical JSON is byte-identical to what ``record_search_breadth`` stores as
    ``grid_json`` (``json.dumps(result.grid, sort_keys=True)`` over the SAME ``parse_grid``
    output), so the trial-layer dedup can key on exact equality.
    """
    grid = parse_grid(sweep_params)
    return grid, json.dumps(grid, sort_keys=True)


def _recipe_hash(canonical_grid_json: str, eval_context: dict[str, Any]) -> str:
    """SHA-256 over the FULL evidence recipe: the canonical grid AND the whole eval context
    (snapshot|demo, fundamentals/news snapshots, universe, start/end, delistings, rank_by).

    GATE-2 #4: a marker binding only the grid would let a resumed attempt with a DIFFERENT data
    context silently reuse a prior attempt's marker — claiming evidence that was produced against
    other data. The recipe hash makes any context drift a fail-closed mismatch on resume.
    """
    payload = json.dumps(
        {"grid_json": canonical_grid_json, "context": eval_context}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_transport_inputs(
    *, demo: bool, snapshot: str | None, rank_by: str, sweep_params: list[str] | None,
) -> None:
    """Fail-fast CLI preflight for the merge-back eval-context transport flags (GATE-2 #5).

    A typo'd operator invocation must die BEFORE the saga takes the repo lock or touches
    git/journal state — not preview-merge, run the ~9-minute quality gate, and then fail/revert
    deep inside promote. Raises ValueError on: ``--demo``/``--snapshot`` not EXACTLY one; a
    ``--rank-by`` outside the sweep engine's rank keys; an unparseable ``--sweep-param`` grid.
    (Strategy-dependent grid-KEY validation still runs post-merge in :func:`produce_evidence` —
    the merged module isn't on ``main`` yet at invocation time.)
    """
    if demo == (snapshot is not None):
        raise ValueError(
            "pass exactly one of --demo or --snapshot: the eval context the authoritative "
            "evidence + promote run against (neither/both can never promote and would waste a "
            "full merge + quality-gate cycle)")
    if rank_by not in _RANK_KEYS:
        raise ValueError(f"--rank-by must be one of {sorted(_RANK_KEYS)}, got {rank_by!r}")
    if sweep_params:
        parse_grid(sweep_params)  # raises ValueError on a malformed KEY=v1,v2 flag


def produce_evidence(
    *,
    strategy: str,
    branch_tip: str,
    ensure_status: str,
    sweep_params: list[str] | None,
    eval_context: dict[str, Any],
    conn_factory: Callable[[], AbstractContextManager[sqlite3.Connection]],
    sweep_fn: Callable[[], dict],
    backtest_fn: Callable[[], dict],
    load_strategy_fn: Callable[[str], Any] = load_strategy,
) -> str:
    """Reproduce ``strategy``'s promote evidence authoritatively (or prove it may be skipped).

    Runs at the saga chokepoint AFTER the gate-green merge is durably on ``main`` (so the module
    the injected tasks load IS the merged code) and immediately before the metered promote.
    ``ensure_status`` must be the saga's JOURNALED first-attempt value, never a fresh re-read
    (GATE-2 #1); ``eval_context`` is the full transported data context (snapshot|demo,
    fundamentals/news snapshots, universe, start/end, delistings, rank_by) — it binds the marker
    via the recipe hash (GATE-2 #4). Returns the ``evidence_status`` the saga mirrors into its
    journal:

    * ``"already_produced"`` — a ``'completed'`` marker exists for this ``(strategy, branch_tip)``
      AND its recipe hash matches this invocation: a prior attempt finished recording; nothing is
      re-recorded (re-recording would permanently inflate breadth).
    * ``"authoritative_breadth"`` — the row pre-existed this attempt (``ensure_status ==
      "existed"``), no marker was ever started for this attempt, the strategy already carries
      authoritative measured breadth, AND a persisted ``backtest_returns`` series exists: the
      direct-authoritative-funnel no-op (a #534-style strategy with fresh authoritative evidence
      must not be re-swept or double-counted).
    * ``"authoritative_breadth_returns_backfilled"`` — same as above but the classifier's
      ``backtest_returns`` series was MISSING (GATE-2 #2: breadth alone does not prove the
      correlation axis the intake must guarantee) — only the full-period backtest ran (cost floor
      asserted first); the sweep was still skipped.
    * ``"no_context"`` — no sweep grid was transported. Nothing is produced; the promote gate's own
      binding breadth floor fails closed downstream if evidence is genuinely missing, so this
      cannot manufacture a promotable state.
    * ``"produced"`` — the full evidence recipe ran: grid validated against the now-on-``main``
      module (:func:`validate_sweep_grid` — the same checks the real sweep raises), the agent cost
      floor asserted BEFORE anything persists (``assert_gated_costs`` — the classifier's returns
      must be the same cost-realistic stream promote evaluates), the injected sweep task recorded
      the measured breadth, the injected full-period backtest task persisted ``backtest_returns``,
      and the marker flipped ``'completed'``.

    Crash idempotency (design §C.2, marker-last + keyed dedup): the ``'started'`` marker (recipe
    hash + a ``search_trials`` watermark, read+inserted under ONE ``BEGIN IMMEDIATE``) lands
    BEFORE any compute. The sweep runs UNCONDITIONALLY when the marker was newly created by THIS
    call; the ``(strategy_name, canonical grid_json, id > watermark)`` dedup applies ONLY when
    resuming a pre-existing ``'started'`` marker (GATE-2 #3 — otherwise a concurrent/manual
    same-grid sweep landing after the watermark would be misattributed to this attempt and the
    authoritative sweep silently skipped). A resume whose grid OR data context differs from the
    recipe that started (or completed) the marker fails closed — never a silent re-produce.
    """
    grid: dict[str, list[Any]] | None = None
    canonical_json = ""
    recipe_hash = ""
    if sweep_params:
        grid, canonical_json = _canonical_grid(sweep_params)
        recipe_hash = _recipe_hash(canonical_json, eval_context)
    needs_returns_backfill = False
    with conn_factory() as conn:
        repo = SqliteStrategyRepository(conn)
        rec = repo.get(strategy)
        marker = evidence_marker(conn, rec.id, branch_tip)
        if (marker is None and ensure_status == "existed"
                and repo.total_search_combos(strategy) > 0):
            if repo.load_backtest_returns(strategy) is not None:
                return "authoritative_breadth"
            # GATE-2 #2: authoritative breadth proves search_trials, NOT the classifier's
            # return-correlation axis. Backfill ONLY the returns below (never re-sweep).
            needs_returns_backfill = True
        if grid is None and not needs_returns_backfill:
            if marker is not None:
                raise MergeBackIntakeError(
                    f"evidence production for {strategy!r} @ {branch_tip} was started with a "
                    f"sweep grid but this resume transported none; refusing an inconsistent "
                    f"resume (fail closed)")
            return "no_context"
        if grid is not None and marker is not None and marker["recipe_hash"] != recipe_hash:
            raise MergeBackIntakeError(
                f"evidence production for {strategy!r} @ {branch_tip} was started with recipe "
                f"{marker['recipe_hash']} but this resume transported {recipe_hash} (grid or "
                f"data context drifted); refusing an inconsistent resume (fail closed)")
        if marker is not None and marker["status"] == "completed":
            return "already_produced"

    # Validate BEFORE the marker exists (an invalid recipe must leave no 'started' residue) and
    # BEFORE anything persists (cost floor: classifier returns == promote's cost-realistic stream).
    loaded = load_strategy_fn(strategy)
    assert_gated_costs(loaded.execution)
    if needs_returns_backfill:
        backtest_fn()  # persists backtest_returns (autocommit, inside the task); sweep skipped
        return "authoritative_breadth_returns_backfilled"
    assert grid is not None  # every no-grid path returned or raised above
    validate_sweep_grid(loaded, grid)

    if marker is None:
        with conn_factory() as conn:
            watermark = _create_started_marker(conn, rec.id, branch_tip, recipe_hash)
        # GATE-2 #3: a marker newly created by THIS call means THIS attempt has recorded nothing
        # yet — sweep unconditionally. The watermark dedup below is a RESUME-only device; applying
        # it here would let a concurrent/manual same-grid sweep (landing after the watermark) be
        # misattributed to this attempt, silently skipping the authoritative sweep.
        sweep_fn()  # records the single search_trials breadth row (autocommit, inside the task)
    else:
        watermark = int(marker["search_trials_watermark"])
        with conn_factory() as conn:
            trial_recorded = conn.execute(
                "SELECT 1 FROM search_trials"
                " WHERE strategy_name = ? AND grid_json = ? AND id > ? LIMIT 1",
                (strategy, canonical_json, watermark),
            ).fetchone() is not None
        if not trial_recorded:
            sweep_fn()
    backtest_fn()  # persists backtest_returns for the registered strategy (autocommit, inside)

    with conn_factory() as conn:
        _complete_marker(conn, rec.id, branch_tip)
    return "produced"
