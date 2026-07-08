"""Journal-driven merge-back cycle orchestration (#485, Task 4).

Turns a research cycle's strategy branch into an on-``main`` strategy without a human merging the
branch, driven as a **saga across git and the registry** whose durable state is a per-strategy
journal keyed on the immutable **branch-tip SHA** (NOT the branch name, NOT the lifecycle stage).

``run_merge_back`` is a pure state machine over injected seams (``GitOps``, ``Journal``, and the
registry callables), so every branch is unit-testable with a ``FakeGit``, a fake journal, and stubs.
The hard correctness properties it encodes (each closing a specific GATE-2 finding):

* **Branch-tip identity, not ancestry (finding #1).** ``git merge-base --is-ancestor`` is abandoned:
  a *reverted* merge leaves the branch tip an ancestor of HEAD forever, so ancestry cannot tell
  "present on main" from "merged then reverted". Whether THIS driver's merge is still live is read
  from the journal (``merge_sha``) and then *verified*: second-parent match + ancestor of the
  freshly-fetched ``origin/main`` + an **effective-presence content check** (the allowlisted blobs
  byte-match ``origin/main`` right now) — before PROMOTE and again before INTAKE.
* **Real remote CAS push (finding #2).** After a green gate the merge SHA is pushed to
  ``refs/heads/main`` under a compare-and-swap (``push_cas``), so nothing the driver does stays
  local and a concurrent remote mutation fails closed instead of being promoted over.
* **Diff-policy gate before merging (finding #3).** The branch's change set is gated by
  :func:`algua.operator.diff_policy.evaluate_diff` (allowlist + CODEOWNERS denylist + mode/rename
  hardening) BEFORE any merge — the security boundary between an autonomous agent and the files that
  enforce the paper->live wall.
* **Stage-drift fail-closed (finding #4).** A ``candidate``/``paper`` stage that the journal does
  NOT prove THIS attempt produced is treated as external drift and **fails closed** — never intaked.
* **Token-bound promote attribution (finding #5).** Promotion success is read from the registry by
  the per-attempt ``attempt_token`` stamped on the gate row, never from the ambient stage nor from
  "did the call raise".

Imports only from within ``algua.operator`` (a package leaf), so no import-linter contract touches
it. ``RealGitOps``/``merge_back_lock`` re-export from :mod:`algua.operator.gitops` for the CLI.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

from algua.operator.diff_policy import evaluate_diff
from algua.operator.gitops import GitOps, RealGitOps, RemoteMovedError, merge_back_lock
from algua.operator.journal import (
    Journal,
    MergeBackRecord,
    derive_attempt_token,
    strict_relaxation_fingerprint,
)

__all__ = [
    "CycleResult",
    "GitOps",
    "LocalMainDriftError",
    "MergeContentAbsentError",
    "RealGitOps",
    "RemoteMovedError",
    "StageDriftError",
    "merge_back_lock",
    "run_merge_back",
]

_MAIN = "main"


class LocalMainDriftError(RuntimeError):
    """Local ``main`` HEAD diverges from ``origin/main`` in a way the journal cannot explain (#1).

    Raised as a hard precondition BEFORE any merge: the diff-policy/CODEOWNERS gate and the CAS push
    are anchored on ``origin/main``, so a locally-drifted ``main`` (an EXTERNAL unpushed commit, a
    stale local ref) would let the driver merge onto — and PROMOTE from — a tree that is not what
    the shared remote holds, bypassing the gate. A local ``main`` that is ahead of ``origin/main``
    solely because of THIS driver's own journaled merge/revert (a crash between commit and push) is
    NOT drift and is adopted for resume; only an unaccounted-for local commit fails closed here.
    """


class StageDriftError(RuntimeError):
    """The lifecycle stage was advanced by something other than THIS attempt (finding #1/#4).

    Raised when the journal cannot prove this driver produced the observed ``candidate``/``paper``
    stage. Reverting a possibly-committed external promotion is the dangerous direction, so the
    driver fails closed for human triage rather than acting on drifted state.
    """


class MergeContentAbsentError(RuntimeError):
    """The allowlisted code this attempt merged is NOT byte-live on ``origin/main`` (C4/R1).

    Raised before PROMOTE or INTAKE when a blob-equality check against the freshly-fetched
    ``origin/main`` tip fails — a revert/rewrite (of any shape) removed the merged code. The driver
    fails closed so capital is never allocated to code no longer on the shared remote.
    """


@dataclass(frozen=True)
class CycleResult:
    """Outcome of one merge-back cycle.

    ``status`` is the terminal outcome: ``already_done``, ``diff_policy_rejected``, ``gate_failed``,
    ``promote_failed`` (gate green but promote rejected — merge reverted), ``promoted_allocated``,
    or ``promoted_queued`` (promoted, but intake left it queued behind capacity). The flags record
    which side effects ran; ``intake`` carries the intake result on the promoted paths (else None).
    """

    ok: bool
    status: str
    merged: bool
    reverted: bool
    promoted: bool
    intake: dict | None
    branch_tip: str | None = None
    attempt_token: str | None = None
    gate_id: int | None = None


def run_merge_back(  # noqa: PLR0912, PLR0913, PLR0915 — a saga state machine; each branch is a step
    *,
    git: GitOps,
    journal: Journal,
    strategy: str,
    branch: str,
    codeowners_text: str,
    stage_of: Callable[[str], str],
    run_gate: Callable[[], bool],
    promote: Callable[[str], object],
    passing_gate_by_token: Callable[[str], int | None],
    gate_exists_by_token: Callable[[str], bool],
    intake: Callable[[], dict],
    target_allocated: Callable[[str], bool],
    audit_log: Callable[[dict], None] | None = None,
    base_ref: str = _MAIN,
) -> CycleResult:
    """Drive one merge-back cycle for ``strategy`` on ``branch`` as a journal-driven state machine.

    The concrete effects are injected: ``git`` (:class:`~algua.operator.gitops.GitOps`), ``journal``
    (:class:`~algua.operator.journal.Journal`), ``codeowners_text`` (for the diff-policy denylist),
    ``stage_of`` (read the lifecycle stage), ``run_gate`` (full quality gate on the staged tree),
    ``promote`` (the strict-agent promote seam — it takes the ``attempt_token`` and stamps it on the
    gate row), ``passing_gate_by_token`` (authoritative read of THIS attempt's passing gate row id,
    or None), ``gate_exists_by_token`` (does ANY gate row — passing OR failing — already bear this
    attempt's token; the crash-idempotency read that stops a resume from re-invoking a metered
    promote whose prior attempt already burned the holdout), ``intake`` (FIFO candidate->paper
    intake), ``target_allocated`` (did intake allocate
    THIS strategy?), and an optional ``audit_log`` sink for push/revert records.

    Returns a :class:`CycleResult` on a completed cycle (including the non-error ``gate_failed`` /
    ``promote_failed`` / ``diff_policy_rejected`` terminals); raises :class:`StageDriftError`,
    :class:`MergeContentAbsentError`, or :class:`~algua.operator.gitops.RemoteMovedError` when it
    must fail closed, and re-raises a genuine promote exception after reverting.
    """
    audit = audit_log if audit_log is not None else (lambda _e: None)

    # ------------------------------------------------------------------ 1. recover + preconditions
    if git.merge_in_progress():
        git.abort_merge()
    if git.current_branch() != _MAIN or not git.working_tree_clean():
        raise ValueError(
            f"merge-back requires a clean checkout on 'main' (branch={git.current_branch()!r}, "
            f"clean={git.working_tree_clean()})"
        )
    git.fetch_remote(base_ref)
    branch_tip = git.resolve(branch)
    base_sha = git.remote_tip(base_ref)
    local_main = git.resolve(base_ref)
    # Load the journal BEFORE the drift check (finding #1): a local `main` HEAD AHEAD of the
    # freshly-fetched `origin/main` is EXPECTED recovery state when THIS driver committed a merge
    # (or revert) whose push crashed before it confirmed — git advances HEAD on commit, and that
    # commit is durably recorded here as merge_sha/revert_sha. So drift is evaluated against the
    # journal, not against ``origin/main`` unconditionally.
    rec = journal.latest(strategy, branch_tip)

    def _write(**changes: object) -> MergeBackRecord:
        nonlocal rec
        base = rec if rec is not None else MergeBackRecord(
            strategy=strategy, branch=branch, branch_tip=branch_tip)
        rec = replace(base, **changes)  # type: ignore[arg-type]
        journal.append(rec)
        return rec

    # Hard precondition (finding #1): local `main` must be reconciled with the freshly-fetched
    # `origin/main`. Equal is the common case. If it differs, the ONLY benign explanation is a prior
    # crash of THIS driver that left its own committed merge/revert on local `main` (recorded in the
    # journal) before the push confirmed — recognized by matching ``local_main`` to a recorded
    # commit. Anything the journal cannot account for is external/unexplained drift → fail closed
    # rather than merge onto / promote from a tree the shared remote does not hold (a gate bypass).
    if local_main != base_sha:
        explained_by: set[str] = set()
        if rec is not None:
            explained_by = {sha for sha in (rec.merge_sha, rec.revert_sha) if sha is not None}
        if local_main not in explained_by:
            # MEDIUM-1: a crash BETWEEN ``commit_merge()`` and journaling ``merge_sha`` leaves local
            # `main` at an unjournaled merge commit (the journal only has diff_policy=passed). That
            # is NOT external drift — adopt it by SECOND-PARENT verification: if we were mid-merge
            # (diff policy passed, no merge_sha/revert_sha yet) AND local `main` is a merge commit
            # whose 2nd parent is our branch tip, it is THIS attempt's own merge. Journal it and
            # resume forward (the push retry below) rather than failing closed forever.
            adoptable = (
                rec is not None and rec.diff_policy == "passed"
                and rec.merge_sha is None and rec.revert_sha is None
                and git.is_merge_of(local_main, branch_tip))
            if adoptable:
                assert rec is not None
                _write(base_sha=rec.base_sha if rec.base_sha is not None else base_sha,
                       diff_policy="passed", gate_status="green", merge_sha=local_main)
            else:
                raise LocalMainDriftError(
                    f"local {base_ref!r} is {local_main} but freshly-fetched origin/{base_ref} is "
                    f"{base_sha} and no journal record for branch_tip {branch_tip} accounts for "
                    f"it; refusing to merge onto a drifted local main (would bypass the "
                    f"diff-policy/CODEOWNERS gate)")
    stage = stage_of(strategy)

    def _allow_paths() -> list[str]:
        """The allowlisted paths the branch introduces vs merge-base(base, tip) — re-derivable on
        recovery, so the content check is reproducible without persisting the path set."""
        anchor = rec.base_sha if rec is not None and rec.base_sha is not None else base_sha
        entries = git.changed_entries(git.merge_base(anchor, branch_tip), branch_tip)
        return [e.new_path for e in entries if not e.change_type.startswith("D")]

    def _content_present(merge_sha: str) -> bool:
        """C4/R1 effective-presence check: the allowlisted blobs the merge introduced are
        byte-identical at the freshly-fetched ``origin/main`` tip RIGHT NOW."""
        paths = _allow_paths()
        captured = git.tree_blobs(merge_sha, paths)
        git.fetch_remote(base_ref)
        for path in paths:
            if git.blob_at(base_ref, path) != captured.get(path):
                return False
        return True

    def _merge_verified(merge_sha: str) -> bool:
        """Branch-tip identity: second-parent match + ancestor of the fetched ``origin/main`` +
        content still present — NOT a bare ancestry heuristic (finding #1)."""
        if git.commit_second_parent(merge_sha) != branch_tip:
            return False
        if not git.is_ancestor(merge_sha, f"refs/remotes/origin/{base_ref}"):
            return False
        return _content_present(merge_sha)

    def _terminal_result(record: MergeBackRecord) -> CycleResult:
        status = record.terminal or "already_done"
        return CycleResult(
            ok=status in {"already_done", "promoted_allocated", "promoted_queued"},
            status=status,
            # A merge DID happen iff a merge commit was recorded; a subsequent revert does NOT unset
            # it (LOW-1: match the fresh promote_failed path — merged=True/reverted=True — so
            # idempotent replay output equals the original result).
            merged=record.merge_sha is not None,
            reverted=record.revert_sha is not None,
            promoted=record.promote_status == "passed",
            intake=None,
            branch_tip=branch_tip,
            attempt_token=record.attempt_token,
            gate_id=record.promote_gate_id,
        )

    def _already_done() -> CycleResult:
        return CycleResult(ok=True, status="already_done", merged=False, reverted=False,
                           promoted=False, intake=None, branch_tip=branch_tip)

    def _do_intake(merge_sha: str) -> CycleResult:
        """INTAKE (shared by the durable-promote resume path and the fresh promote path). Re-checks
        effective presence against ``origin/main`` before allocating, then emits a target-verified
        outcome (allocated vs still-queued behind capacity)."""
        if not _content_present(merge_sha):
            raise MergeContentAbsentError(
                f"merged code for {strategy!r} absent from origin/{base_ref} at intake")
        result = intake()
        tok = rec.attempt_token if rec is not None else None
        gid = rec.promote_gate_id if rec is not None else None
        if target_allocated(strategy):
            _write(intake_status="allocated", terminal="promoted_allocated")
            return CycleResult(ok=True, status="promoted_allocated", merged=True, reverted=False,
                               promoted=True, intake=result, branch_tip=branch_tip,
                               attempt_token=tok, gate_id=gid)
        _write(intake_status="queued", terminal="promoted_queued")
        return CycleResult(ok=True, status="promoted_queued", merged=True, reverted=False,
                           promoted=True, intake=result, branch_tip=branch_tip,
                           attempt_token=tok, gate_id=gid)

    def _complete_revert(record: MergeBackRecord) -> CycleResult:
        """Resume a crash BETWEEN ``revert_merge`` and the revert push (finding #3).

        ``revert_sha`` is durably journaled but ``terminal`` is not, so the revert commit exists
        locally yet may not be on ``origin``. If the revert push ALREADY LANDED before the crash
        (``origin/main == revert_sha``, finding #3) it is accepted as done and converged forward —
        NOT re-pushed (a re-push would ``RemoteMovedError`` against the moved origin). Otherwise
        re-run the revert push under its compare-and-swap (idempotent — a concurrent remote advance
        fails closed as ``RemoteMovedError``). Either way reach the ``promote_failed`` terminal —
        never fall through to re-``promote()`` or leave a stray revert unpushed."""
        assert record.merge_sha is not None and record.revert_sha is not None
        git.fetch_remote(base_ref)
        if git.remote_tip(base_ref) == record.revert_sha:
            # The revert push landed remotely before the crash; only the terminal journal write was
            # lost. Converge without re-pushing.
            audit({"event": "autonomous_revert", "strategy": strategy,
                   "merge_sha": record.merge_sha, "revert_sha": record.revert_sha,
                   "phase": "after", "result": "already_landed"})
        else:
            audit({"event": "autonomous_revert", "strategy": strategy,
                   "merge_sha": record.merge_sha, "revert_sha": record.revert_sha,
                   "phase": "resume_before"})
            git.push_revert(record.revert_sha, record.merge_sha)
            audit({"event": "autonomous_revert", "strategy": strategy,
                   "merge_sha": record.merge_sha, "revert_sha": record.revert_sha,
                   "phase": "after", "result": "ok"})
        _write(promote_status="failed", revert_sha=record.revert_sha, terminal="promote_failed")
        return CycleResult(ok=False, status="promote_failed", merged=True, reverted=True,
                           promoted=False, intake=None, branch_tip=branch_tip,
                           attempt_token=record.attempt_token, gate_id=record.promote_gate_id)

    # ------------------------------------------------------------------ 2. journal-first dispatch
    if rec is not None and rec.terminal is not None:
        return _terminal_result(rec)
    if rec is not None and rec.intake_status == "allocated":
        return _terminal_result(_write(terminal="promoted_allocated"))

    # Resume a crash between revert_merge and the revert push (finding #3): a durable ``revert_sha``
    # with NO ``terminal`` means the revert is committed locally but its push is unconfirmed. Finish
    # the revert-push instead of falling through to ``have_merge`` -> a second ``promote()`` (and a
    # stray local revert commit contaminating the next cycle).
    if rec is not None and rec.revert_sha is not None:
        return _complete_revert(rec)

    # Stage cross-check (finding #4): a candidate/paper stage the journal does NOT prove we produced
    # is external drift → fail closed, never intake.
    proven_promoted = rec is not None and rec.promote_status == "passed"
    if stage in {"candidate", "paper"} and not proven_promoted:
        if stage == "paper" and rec is None:
            # No attempt on record and already at paper — an idempotent no-op, safe (no mutation).
            return _already_done()
        raise StageDriftError(
            f"strategy {strategy!r} is at stage {stage!r} but no journal record proves THIS "
            f"attempt (branch_tip {branch_tip}) promoted it; refusing to intake drifted state")

    # Resume into INTAKE if we durably promoted (verify the merge is still live first).
    if proven_promoted:
        assert rec is not None and rec.merge_sha is not None
        if not _merge_verified(rec.merge_sha):
            raise MergeContentAbsentError(
                f"promoted merge {rec.merge_sha} for {strategy!r} is no longer live on "
                f"origin/{base_ref}")
        return _do_intake(rec.merge_sha)

    # Fresh / early-resume paths require a backtested source stage.
    if stage != "backtested":
        raise ValueError(f"unexpected stage for merge-back: {stage!r}")

    # ------------------------------------------------------------------ 3. resume: merge recorded?
    have_merge = rec is not None and rec.merge_sha is not None and rec.revert_sha is None
    if have_merge:
        assert rec is not None and rec.merge_sha is not None
        merge_sha = rec.merge_sha
        if rec.push_status == "pushed":
            if not _merge_verified(merge_sha):
                raise MergeContentAbsentError(
                    f"recorded merge {merge_sha} for {strategy!r} not live on origin/{base_ref}")
        else:
            # Committed locally but the crash landed BEFORE/DURING the push (finding #2): the merge
            # is NOT yet on origin, so `_merge_verified` (which requires origin presence) would
            # wrongly fail closed and re-drive nothing. Instead confirm the local merge commit still
            # corresponds to our branch tip and origin still sits at the recorded base, then retry
            # the CAS push.
            anchor_base = rec.base_sha if rec.base_sha is not None else base_sha
            if git.commit_second_parent(merge_sha) != branch_tip:
                raise MergeContentAbsentError(
                    f"recorded local merge {merge_sha} for {strategy!r} is not a merge of branch "
                    f"tip {branch_tip}; cannot resume push")
            git.fetch_remote(base_ref)
            remote_now = git.remote_tip(base_ref)
            if remote_now == merge_sha:
                # HIGH-1: the push ACTUALLY LANDED before the crash — only the journal flip to
                # "pushed" was lost. origin/main already carries our merge, so accepting
                # ``origin/main == merge_sha`` as done and converging forward is correct; re-driving
                # push_cas (which expects origin still at ``anchor_base``) would wrongly
                # RemoteMovedError. Verify branch-tip identity is still live before converging.
                if not _merge_verified(merge_sha):
                    raise MergeContentAbsentError(
                        f"resume: origin/{base_ref} is at recorded merge {merge_sha} for "
                        f"{strategy!r} but its branch-tip identity/content no longer verifies")
                audit({"event": "autonomous_push", "strategy": strategy, "merge_sha": merge_sha,
                       "phase": "after", "result": "already_landed", "resume": True})
                _write(base_sha=anchor_base, diff_policy="passed", gate_status="green",
                       merge_sha=merge_sha, push_status="pushed")
            elif remote_now != anchor_base:
                raise RemoteMovedError(
                    f"origin/{base_ref} moved to {remote_now} (expected recorded base "
                    f"{anchor_base}); local merge {merge_sha} is stale, refusing to resume push")
            else:
                audit({"event": "autonomous_push", "strategy": strategy, "branch": branch,
                       "branch_tip": branch_tip, "base_sha": anchor_base, "merge_sha": merge_sha,
                       "refspec": f"{merge_sha}:refs/heads/{base_ref}", "phase": "before",
                       "resume": True})
                git.push_cas(merge_sha, anchor_base)  # raises RemoteMovedError -> fail closed
                audit({"event": "autonomous_push", "strategy": strategy, "merge_sha": merge_sha,
                       "phase": "after", "result": "ok", "resume": True})
                _write(base_sha=anchor_base, diff_policy="passed", gate_status="green",
                       merge_sha=merge_sha, push_status="pushed")
    else:
        # -------------------------------------------------------------- DIFF POLICY (pre-merge)
        if rec is None or rec.diff_policy != "passed":
            entries = git.changed_entries(git.merge_base(base_sha, branch_tip), branch_tip)
            policy = evaluate_diff(entries, codeowners_text)
            if not policy.ok:
                _write(base_sha=base_sha, diff_policy="rejected", terminal="diff_policy_rejected")
                return CycleResult(ok=False, status="diff_policy_rejected", merged=False,
                                   reverted=False, promoted=False, intake=None,
                                   branch_tip=branch_tip)
            _write(base_sha=base_sha, diff_policy="passed")

        # -------------------------------------------------------------- MERGE (gate before commit)
        git.begin_merge(branch_tip)
        # MEDIUM-2: an EXCEPTION out of the gate seam (a crashed subprocess, an OSError launching
        # it) is a RED gate, exactly as the CLI doc promises ("each check is a separate subprocess
        # so a crash in one is a red gate"). A raised gate must NOT propagate with the ``--no-ff
        # --no-commit`` preview merge left staged in the working tree — abort it and take the
        # gate_failed path, same as a clean ``False`` return.
        try:
            gate_green = run_gate()
        except Exception:  # noqa: BLE001 — any gate-seam failure is a red gate, never a raised saga
            gate_green = False
        if not gate_green:
            git.abort_merge()
            _write(base_sha=base_sha, diff_policy="passed", gate_status="failed",
                   terminal="gate_failed")
            return CycleResult(ok=False, status="gate_failed", merged=False, reverted=False,
                               promoted=False, intake=None, branch_tip=branch_tip)
        git.commit_merge()
        merge_sha = git.merge_commit_of(branch_tip)
        _write(base_sha=base_sha, diff_policy="passed", gate_status="green", merge_sha=merge_sha)

        # -------------------------------------------------------------- PUSH (remote CAS)
        audit({"event": "autonomous_push", "strategy": strategy, "branch": branch,
               "branch_tip": branch_tip, "base_sha": base_sha, "merge_sha": merge_sha,
               "refspec": f"{merge_sha}:refs/heads/{base_ref}", "phase": "before"})
        git.push_cas(merge_sha, base_sha)  # raises RemoteMovedError -> fail closed
        audit({"event": "autonomous_push", "strategy": strategy, "merge_sha": merge_sha,
               "phase": "after", "result": "ok"})
        _write(base_sha=base_sha, diff_policy="passed", gate_status="green", merge_sha=merge_sha,
               push_status="pushed")

    # ------------------------------------------------------------------ 4. PROMOTE (token-bound)
    # The record's base_sha is authoritative from here: a resume may have re-read a top-level
    # ``base_sha`` that already advanced to ``merge_sha`` (origin fast-forwarded to our merge), so
    # keep the ORIGINAL base for the journal + the ``_allow_paths`` content-check anchor.
    if rec is not None and rec.base_sha is not None:
        base_sha = rec.base_sha
    if not _content_present(merge_sha):
        raise MergeContentAbsentError(
            f"merged code for {strategy!r} absent from origin/{base_ref} before promote")
    token = derive_attempt_token(strategy, branch_tip, merge_sha, strict_relaxation_fingerprint())
    _write(base_sha=base_sha, diff_policy="passed", gate_status="green", merge_sha=merge_sha,
           push_status="pushed", attempt_token=token)

    # HIGH-2 crash-idempotency: a prior attempt under THIS (deterministic) token may already have
    # run the metered promote and crashed BEFORE journaling the outcome. The token is unique-indexed
    # on the gate row, so blindly re-invoking promote() would double-burn the holdout and late-fail
    # on the unique index AFTER side effects. Read the AUTHORITATIVE registry state by token FIRST
    # and branch on it; invoke promote() only when NO row under this token exists yet.
    raised: BaseException | None = None
    gate_id = passing_gate_by_token(token)
    if gate_id is None and not gate_exists_by_token(token):
        try:
            promote(token)
        except BaseException as exc:  # noqa: BLE001 — outcome read authoritatively below, not raise
            raised = exc
        gate_id = passing_gate_by_token(token)
    # else: a gate row already bears this token (a crashed prior attempt). Never re-invoke — a
    # passing row converges to intake just below; a failing-only row (gate_id stays None) falls
    # through to the revert path.

    # Success is read from AUTHORITATIVE registry state by attempt_token, never from "did it raise".
    if gate_id is not None:
        rec = _write(base_sha=base_sha, diff_policy="passed", gate_status="green",
                     merge_sha=merge_sha, push_status="pushed", attempt_token=token,
                     promote_status="passed", promote_gate_id=gate_id)
        return _do_intake(merge_sha)

    # No gate row bearing our token. If the stage also drifted off backtested, we cannot prove state
    # → fail closed (reverting a possibly-committed promotion is the dangerous direction).
    stage_now = stage_of(strategy)
    if stage_now != "backtested":
        raise StageDriftError(
            f"promote left no gate row for token {token[:12]}… yet {strategy!r} is at "
            f"{stage_now!r}; cannot prove promotion state, failing closed without revert")

    # Proven non-committed promote failure: git-only revert (research-integrity ledgers are NOT
    # rolled back by design — a burned holdout survives).
    audit({"event": "autonomous_revert", "strategy": strategy, "merge_sha": merge_sha,
           "phase": "before"})
    revert_sha = git.revert_merge(merge_sha)
    # Durable marker BEFORE the push (finding #3): journal ``revert_sha`` with NO ``terminal`` so a
    # crash between the local revert commit and its push resumes as "revert committed, push
    # unconfirmed" (-> _complete_revert) instead of re-entering ``have_merge`` and re-``promote()``.
    _write(base_sha=base_sha, diff_policy="passed", gate_status="green", merge_sha=merge_sha,
           push_status="pushed", attempt_token=token, promote_status="failed",
           revert_sha=revert_sha)
    git.push_revert(revert_sha, merge_sha)  # remote CAS: RemoteMovedError -> stale, fail closed
    audit({"event": "autonomous_revert", "strategy": strategy, "merge_sha": merge_sha,
           "revert_sha": revert_sha, "phase": "after", "result": "ok"})
    _write(base_sha=base_sha, diff_policy="passed", gate_status="green", merge_sha=merge_sha,
           push_status="pushed", attempt_token=token, promote_status="failed",
           revert_sha=revert_sha, terminal="promote_failed")
    if raised is not None:
        raise raised
    return CycleResult(ok=False, status="promote_failed", merged=True, reverted=True,
                       promoted=False, intake=None, branch_tip=branch_tip, attempt_token=token)
