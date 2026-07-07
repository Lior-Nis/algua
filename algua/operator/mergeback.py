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
    "MergeContentAbsentError",
    "RealGitOps",
    "RemoteMovedError",
    "StageDriftError",
    "merge_back_lock",
    "run_merge_back",
]

_MAIN = "main"


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
    or None), ``intake`` (FIFO candidate->paper intake), ``target_allocated`` (did intake allocate
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
    rec = journal.latest(strategy, branch_tip)
    stage = stage_of(strategy)

    def _write(**changes: object) -> MergeBackRecord:
        nonlocal rec
        base = rec if rec is not None else MergeBackRecord(
            strategy=strategy, branch=branch, branch_tip=branch_tip)
        rec = replace(base, **changes)  # type: ignore[arg-type]
        journal.append(rec)
        return rec

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
            merged=record.merge_sha is not None and record.revert_sha is None,
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

    # ------------------------------------------------------------------ 2. journal-first dispatch
    if rec is not None and rec.terminal is not None:
        return _terminal_result(rec)
    if rec is not None and rec.intake_status == "allocated":
        return _terminal_result(_write(terminal="promoted_allocated"))

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
        if not _merge_verified(rec.merge_sha):
            raise MergeContentAbsentError(
                f"recorded merge {rec.merge_sha} for {strategy!r} not live on origin/{base_ref}")
        merge_sha = rec.merge_sha
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
        if not run_gate():
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
    if not _content_present(merge_sha):
        raise MergeContentAbsentError(
            f"merged code for {strategy!r} absent from origin/{base_ref} before promote")
    token = derive_attempt_token(strategy, branch_tip, merge_sha, strict_relaxation_fingerprint())
    _write(base_sha=base_sha, diff_policy="passed", gate_status="green", merge_sha=merge_sha,
           push_status="pushed", attempt_token=token)

    raised: BaseException | None = None
    try:
        promote(token)
    except BaseException as exc:  # noqa: BLE001 — outcome is read authoritatively below, not from raise
        raised = exc

    # Success is read from AUTHORITATIVE registry state by attempt_token, never from "did it raise".
    gate_id = passing_gate_by_token(token)
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
    git.push_current(base_ref)
    audit({"event": "autonomous_revert", "strategy": strategy, "merge_sha": merge_sha,
           "revert_sha": revert_sha, "phase": "after", "result": "ok"})
    _write(base_sha=base_sha, diff_policy="passed", gate_status="green", merge_sha=merge_sha,
           push_status="pushed", attempt_token=token, promote_status="failed",
           revert_sha=revert_sha, terminal="promote_failed")
    if raised is not None:
        raise raised
    return CycleResult(ok=False, status="promote_failed", merged=True, reverted=True,
                       promoted=False, intake=None, branch_tip=branch_tip, attempt_token=token)
