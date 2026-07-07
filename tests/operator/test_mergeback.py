"""Unit tests for the journal-driven :func:`run_merge_back` state machine (#485, Task 4).

Every git/registry effect is a fake or stub, so each resume branch and crash point is exercised with
no subprocess, no DB, and no real merge. ``FakeGit`` models one shared tree + an authoritative
``origin/main`` (blobs, tip SHA), recording mutating calls so tests assert which side effects ran —
and, crucially, which did NOT (no revert on a committed promote; no intake on drift).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import pytest

from algua.operator.diff_policy import DiffEntry
from algua.operator.gitops import RemoteMovedError
from algua.operator.journal import (
    MergeBackRecord,
    derive_attempt_token,
    strict_relaxation_fingerprint,
)
from algua.operator.mergeback import (
    LocalMainDriftError,
    MergeContentAbsentError,
    StageDriftError,
    run_merge_back,
)

_CODEOWNERS = "/algua/registry/store.py @x\n/algua/registry/promotion.py @x\n"
_TIP = "BRANCHTIP"
_BASE = "BASE0"
_MERGE = "MERGE0"
_STRAT_ENTRY = DiffEntry("100644", "A", None, "algua/strategies/foo/bar.py")


@dataclass
class FakeGit:
    branch: str = "main"
    clean: bool = True
    merge_flag: bool = False
    branch_tip: str = _TIP
    origin_main: str = _BASE
    local_main: str = _BASE
    merge_sha: str = _MERGE
    entries: list[DiffEntry] = field(default_factory=lambda: [_STRAT_ENTRY])
    second_parent: str | None = None
    ancestor: bool = True
    origin_blobs: dict[str, str] | None = None
    cas_fails: bool = False
    push_current_fail_once: bool = False
    calls: list[object] = field(default_factory=list)

    def merge_in_progress(self) -> bool:
        return self.merge_flag

    def abort_merge(self) -> None:
        self.calls.append("abort")
        self.merge_flag = False
        self.clean = True

    def current_branch(self) -> str:
        return self.branch

    def working_tree_clean(self) -> bool:
        return self.clean

    def fetch_remote(self, ref: str) -> None:
        self.calls.append(("fetch", ref))

    def resolve(self, ref: str) -> str:
        # Local `main` HEAD resolves independently of the branch tip, so a drifted local main
        # (local_main != origin_main) is modelable for the finding #1 precondition.
        if ref == "main":
            return self.local_main
        return self.branch_tip

    def remote_tip(self, ref: str) -> str:
        return self.origin_main

    def merge_base(self, a: str, b: str) -> str:
        return "MB"

    def changed_entries(self, base: str, tip: str) -> list[DiffEntry]:
        return self.entries

    def begin_merge(self, tip: str) -> None:
        self.calls.append(("begin", tip))
        self.merge_flag = True

    def commit_merge(self) -> None:
        self.calls.append("commit")
        self.merge_flag = False

    def merge_commit_of(self, tip: str) -> str:
        return self.merge_sha

    def commit_second_parent(self, sha: str) -> str:
        return self.second_parent if self.second_parent is not None else self.branch_tip

    def is_ancestor(self, sha: str, ref: str) -> bool:
        return self.ancestor

    def push_cas(self, merge_sha: str, expected_base: str) -> None:
        if self.cas_fails or expected_base != self.origin_main:
            raise RemoteMovedError("origin/main moved")
        self.calls.append(("push", merge_sha))
        self.origin_main = merge_sha

    def tree_blobs(self, sha: str, paths: list[str]) -> dict[str, str]:
        return {p: f"blob:{p}" for p in paths}

    def blob_at(self, ref: str, path: str) -> str | None:
        if self.origin_blobs is not None:
            return self.origin_blobs.get(path)
        return f"blob:{path}"

    def revert_merge(self, sha: str) -> str:
        self.calls.append(("revert", sha))
        return "REVERT0"

    def push_current(self, ref: str) -> None:
        if self.push_current_fail_once:
            self.push_current_fail_once = False
            raise RuntimeError("crash after revert before push confirmed")
        self.calls.append(("push_current", ref))


class FakeJournal:
    def __init__(self) -> None:
        self.records: dict[tuple[str, str], list[MergeBackRecord]] = {}

    def latest(self, strategy: str, branch_tip: str) -> MergeBackRecord | None:
        recs = self.records.get((strategy, branch_tip))
        return recs[-1] if recs else None

    def append(self, record: MergeBackRecord) -> None:
        self.records.setdefault((record.strategy, record.branch_tip), []).append(record)


class Promoter:
    """A promote seam that optionally 'commits' (a token-stamped passing row exists afterwards)."""

    def __init__(self, *, commit: bool = True, raises: BaseException | None = None,
                 gate_id: int = 42) -> None:
        self.commit = commit
        self.raises = raises
        self.gate_id = gate_id
        self.tokens: list[str] = []
        self._committed: dict[str, int] = {}

    def __call__(self, token: str) -> object:
        self.tokens.append(token)
        if self.commit:
            self._committed[token] = self.gate_id
        if self.raises is not None:
            raise self.raises
        return {"promoted": self.commit}

    def by_token(self, token: str) -> int | None:
        return self._committed.get(token)


def _stage(mapping):
    return lambda name: mapping[name]


def _run(git, journal, *, stage, promoter=None, run_gate=True, intake=None,
         allocated=True, audit=None):
    promoter = promoter if promoter is not None else Promoter()
    return run_merge_back(
        git=git, journal=journal, strategy="s", branch="feat/s",
        codeowners_text=_CODEOWNERS,
        stage_of=_stage(stage),
        run_gate=(lambda: run_gate) if isinstance(run_gate, bool) else run_gate,
        promote=promoter,
        passing_gate_by_token=promoter.by_token,
        intake=(intake if intake is not None else (lambda: {"admitted": ["s"]})),
        target_allocated=lambda name: allocated,
        audit_log=audit,
    )


# (a) paper + no journal record -> already_done, no merge/promote.
def test_paper_no_record_already_done() -> None:
    git = FakeGit()
    p = Promoter()
    result = _run(git, FakeJournal(), stage={"s": "paper"}, promoter=p)
    assert result.status == "already_done"
    assert p.tokens == []
    assert not any(c in ("commit", "abort") or (isinstance(c, tuple) and c[0] in ("begin", "push"))
                   for c in git.calls)


# (b) candidate + no journal proof -> stage_drift fail closed (finding #4).
def test_candidate_without_journal_proof_fails_closed() -> None:
    with pytest.raises(StageDriftError, match="no journal record proves"):
        _run(FakeGit(), FakeJournal(), stage={"s": "candidate"})


# (c) backtested + gate red -> gate_failed, begin then abort, no promote, no push.
def test_backtested_gate_red_aborts() -> None:
    git = FakeGit()
    p = Promoter()
    result = _run(git, FakeJournal(), stage={"s": "backtested"}, promoter=p, run_gate=False)
    assert result.status == "gate_failed"
    assert ("begin", _TIP) in git.calls and "abort" in git.calls
    assert "commit" not in git.calls
    assert p.tokens == []
    assert not any(isinstance(c, tuple) and c[0] == "push" for c in git.calls)


# (d) backtested + green + promote commits -> promoted_allocated, pushed, token-stamped, intake ran.
def test_backtested_green_promote_commits_allocates() -> None:
    git = FakeGit()
    p = Promoter(commit=True)
    journal = FakeJournal()
    result = _run(git, journal, stage={"s": "backtested"}, promoter=p)
    assert result.status == "promoted_allocated"
    assert result.merged and result.promoted and not result.reverted
    assert ("push", _MERGE) in git.calls          # remote CAS push happened (finding #2)
    assert result.intake == {"admitted": ["s"]}
    # Promote was driven with the re-derivable attempt token (finding #5).
    expected = derive_attempt_token("s", _TIP, _MERGE, strict_relaxation_fingerprint())
    assert p.tokens == [expected]
    assert result.attempt_token == expected and result.gate_id == 42
    # No revert on a committed promote.
    assert not any(isinstance(c, tuple) and c[0] == "revert" for c in git.calls)
    # Journal reached the promoted_allocated terminal.
    assert journal.latest("s", _TIP).terminal == "promoted_allocated"


# (d2) green + promote commits but intake leaves it queued -> promoted_queued.
def test_backtested_green_promote_commits_queued() -> None:
    git = FakeGit()
    result = _run(git, FakeJournal(), stage={"s": "backtested"}, allocated=False)
    assert result.status == "promoted_queued"
    assert result.promoted and result.merged


# (e) green + promote does NOT commit, stage stays backtested -> revert + promote_failed.
def test_backtested_green_promote_not_committed_reverts() -> None:
    git = FakeGit()
    p = Promoter(commit=False)
    result = _run(git, FakeJournal(), stage={"s": "backtested"}, promoter=p)
    assert result.status == "promote_failed"
    assert result.reverted and not result.promoted
    assert ("revert", _MERGE) in git.calls
    assert ("push_current", "main") in git.calls


# (f) promote RAISES and did not commit -> revert then the exception propagates.
def test_promote_raises_not_committed_reverts_then_propagates() -> None:
    git = FakeGit()
    boom = RuntimeError("promote blew up")
    p = Promoter(commit=False, raises=boom)
    with pytest.raises(RuntimeError, match="promote blew up"):
        _run(git, FakeJournal(), stage={"s": "backtested"}, promoter=p)
    assert ("revert", _MERGE) in git.calls


# (g) promote RAISES post-commit (token row exists) -> treated committed, NO revert, allocated.
def test_promote_raises_post_commit_is_treated_committed() -> None:
    git = FakeGit()
    p = Promoter(commit=True, raises=RuntimeError("post-commit sync blew up"))
    result = _run(git, FakeJournal(), stage={"s": "backtested"}, promoter=p)
    assert result.status == "promoted_allocated"
    assert not any(isinstance(c, tuple) and c[0] == "revert" for c in git.calls)


# (h) diff policy rejects a denylisted entry -> diff_policy_rejected, no merge.
def test_diff_policy_reject_no_merge() -> None:
    git = FakeGit(entries=[DiffEntry("100644", "M", None, "algua/registry/store.py")])
    p = Promoter()
    result = _run(git, FakeJournal(), stage={"s": "backtested"}, promoter=p)
    assert result.status == "diff_policy_rejected"
    assert not any(isinstance(c, tuple) and c[0] == "begin" for c in git.calls)
    assert p.tokens == []


# (i) not on main -> ValueError.
def test_not_on_main_raises() -> None:
    with pytest.raises(ValueError, match="main"):
        _run(FakeGit(branch="feat/s"), FakeJournal(), stage={"s": "backtested"})


# (j) push CAS fails (origin/main moved) -> RemoteMovedError, no promote.
def test_push_cas_stale_base_fails_closed() -> None:
    git = FakeGit(cas_fails=True)
    p = Promoter()
    with pytest.raises(RemoteMovedError):
        _run(git, FakeJournal(), stage={"s": "backtested"}, promoter=p)
    assert p.tokens == []


# (k) content mismatch on origin/main before promote -> MergeContentAbsentError.
def test_content_absent_before_promote_fails_closed() -> None:
    git = FakeGit(origin_blobs={})  # blob_at returns None -> mismatch vs captured
    p = Promoter()
    with pytest.raises(MergeContentAbsentError):
        _run(git, FakeJournal(), stage={"s": "backtested"}, promoter=p)
    assert p.tokens == []


# (l) stage drifted to candidate but our token row absent -> fail closed, NO revert.
def test_promote_stage_drift_without_token_fails_closed_no_revert() -> None:
    # stage_of returns backtested at start, then candidate after promote (external advance).
    seq = iter(["backtested", "candidate"])
    git = FakeGit()
    p = Promoter(commit=False)
    with pytest.raises(StageDriftError):
        run_merge_back(
            git=git, journal=FakeJournal(), strategy="s", branch="feat/s",
            codeowners_text=_CODEOWNERS,
            stage_of=lambda name: next(seq),
            run_gate=lambda: True, promote=p, passing_gate_by_token=p.by_token,
            intake=lambda: {}, target_allocated=lambda n: True)
    assert not any(isinstance(c, tuple) and c[0] == "revert" for c in git.calls)


# (m) resume: journal proves promote passed -> straight to INTAKE (verify merge first).
def test_resume_promoted_goes_to_intake() -> None:
    git = FakeGit()
    journal = FakeJournal()
    token = derive_attempt_token("s", _TIP, _MERGE, strict_relaxation_fingerprint())
    journal.append(MergeBackRecord(
        strategy="s", branch="feat/s", branch_tip=_TIP, base_sha=_BASE, diff_policy="passed",
        gate_status="green", merge_sha=_MERGE, push_status="pushed", attempt_token=token,
        promote_status="passed", promote_gate_id=42))
    p = Promoter()
    result = _run(git, journal, stage={"s": "candidate"}, promoter=p)
    assert result.status == "promoted_allocated"
    # Resume never re-merged or re-promoted.
    assert not any(isinstance(c, tuple) and c[0] in ("begin", "push") for c in git.calls)
    assert p.tokens == []


# (n) resume: a recorded merge no longer present on origin/main -> fail closed at verify.
def test_resume_promoted_but_content_gone_fails_closed() -> None:
    git = FakeGit(origin_blobs={})  # merge blobs absent from origin/main now
    journal = FakeJournal()
    token = derive_attempt_token("s", _TIP, _MERGE, strict_relaxation_fingerprint())
    journal.append(MergeBackRecord(
        strategy="s", branch="feat/s", branch_tip=_TIP, base_sha=_BASE, merge_sha=_MERGE,
        push_status="pushed", attempt_token=token, promote_status="passed", promote_gate_id=42))
    with pytest.raises(MergeContentAbsentError):
        _run(git, journal, stage={"s": "candidate"})


# (o) terminal record replays idempotently without touching git/promote.
def test_terminal_record_replays() -> None:
    git = FakeGit()
    journal = FakeJournal()
    journal.append(MergeBackRecord(
        strategy="s", branch="feat/s", branch_tip=_TIP, merge_sha=_MERGE,
        promote_status="passed", terminal="promoted_allocated"))
    p = Promoter()
    result = _run(git, journal, stage={"s": "paper"}, promoter=p)
    assert result.status == "promoted_allocated"
    assert p.tokens == []
    assert not any(isinstance(c, tuple) and c[0] in ("begin", "push", "revert") for c in git.calls)


# (p) audit_log receives a push record on the successful path.
def test_audit_log_emits_push_event() -> None:
    events: list[dict] = []
    _run(FakeGit(), FakeJournal(), stage={"s": "backtested"}, audit=events.append)
    kinds = [e.get("event") for e in events]
    assert "autonomous_push" in kinds


# (q) recovery resume after a recorded merge (promote pending) -> skips merge, re-derives token.
def test_resume_merge_recorded_promote_pending() -> None:
    git = FakeGit()
    journal = FakeJournal()
    journal.append(MergeBackRecord(
        strategy="s", branch="feat/s", branch_tip=_TIP, base_sha=_BASE, diff_policy="passed",
        gate_status="green", merge_sha=_MERGE, push_status="pushed"))
    p = Promoter(commit=True)
    result = _run(git, journal, stage={"s": "backtested"}, promoter=p)
    assert result.status == "promoted_allocated"
    # No new merge (resume path).
    assert not any(isinstance(c, tuple) and c[0] == "begin" for c in git.calls)
    assert p.tokens == [derive_attempt_token("s", _TIP, _MERGE, strict_relaxation_fingerprint())]


# (r) finding #1: local main HEAD != freshly-fetched origin/main tip -> fail closed BEFORE merge.
def test_local_main_drift_fails_closed_before_merge() -> None:
    git = FakeGit(local_main="DRIFTED")  # local main diverged from origin/main (_BASE)
    p = Promoter()
    with pytest.raises(LocalMainDriftError, match="drifted local main"):
        _run(git, FakeJournal(), stage={"s": "backtested"}, promoter=p)
    # The merge (and everything after it) never began — the gate bypass is closed.
    assert not any(isinstance(c, tuple) and c[0] in ("begin", "push") for c in git.calls)
    assert "commit" not in git.calls
    assert p.tokens == []


# (s) finding #2: crash committed-locally-but-not-pushed -> resume retries push_cas, no re-merge.
def test_resume_merge_committed_not_pushed_retries_push() -> None:
    git = FakeGit()
    journal = FakeJournal()
    # A crash landed after commit_merge but before push_status flipped to 'pushed'.
    journal.append(MergeBackRecord(
        strategy="s", branch="feat/s", branch_tip=_TIP, base_sha=_BASE, diff_policy="passed",
        gate_status="green", merge_sha=_MERGE, push_status="pending"))
    p = Promoter(commit=True)
    result = _run(git, journal, stage={"s": "backtested"}, promoter=p)
    assert result.status == "promoted_allocated"
    # No re-merge; the CAS push was RETRIED (not skipped, not fail-closed on absent-from-origin).
    assert not any(isinstance(c, tuple) and c[0] == "begin" for c in git.calls)
    assert ("push", _MERGE) in git.calls
    assert p.tokens == [derive_attempt_token("s", _TIP, _MERGE, strict_relaxation_fingerprint())]
    assert journal.latest("s", _TIP).push_status == "pushed"


# (s2) finding #2: committed-not-pushed but origin/main moved under us -> fail closed, no promote.
def test_resume_merge_committed_not_pushed_origin_moved_fails_closed() -> None:
    # origin (and the local main that tracks it) drifted off the recorded base since the crash, so
    # the finding #1 precondition passes but the recorded-base CAS check must fail closed.
    git = FakeGit(origin_main="MOVED", local_main="MOVED")
    journal = FakeJournal()
    journal.append(MergeBackRecord(
        strategy="s", branch="feat/s", branch_tip=_TIP, base_sha=_BASE, diff_policy="passed",
        gate_status="green", merge_sha=_MERGE, push_status="pending"))
    p = Promoter(commit=True)
    with pytest.raises(RemoteMovedError):
        _run(git, journal, stage={"s": "backtested"}, promoter=p)
    assert p.tokens == []
    assert not any(isinstance(c, tuple) and c[0] == "push" for c in git.calls)


# (t) finding #3: crash BETWEEN revert_merge and push_current -> resume completes the revert-push,
# does NOT re-promote, and leaves no stray commit to contaminate a later cycle.
def test_crash_between_revert_and_push_resumes_revert_push() -> None:
    journal = FakeJournal()
    # First cycle: promote does not commit, stage stays backtested -> revert path; push crashes.
    git = FakeGit(push_current_fail_once=True)
    p = Promoter(commit=False)
    with pytest.raises(RuntimeError, match="crash after revert"):
        _run(git, journal, stage={"s": "backtested"}, promoter=p)
    # Durable marker: revert_sha journaled, terminal NOT yet set (push unconfirmed).
    rec = journal.latest("s", _TIP)
    assert rec.revert_sha == "REVERT0"
    assert rec.terminal is None
    assert ("revert", _MERGE) in git.calls

    # Resume with a fresh git whose push_current now succeeds.
    git2 = FakeGit()
    p2 = Promoter(commit=False)
    result = _run(git2, journal, stage={"s": "backtested"}, promoter=p2)
    assert result.status == "promote_failed" and result.reverted
    # The revert-push completed; there was NO second promote and NO re-merge / second revert.
    assert p2.tokens == []
    assert ("push_current", "main") in git2.calls
    assert not any(isinstance(c, tuple) and c[0] in ("begin", "revert", "push")
                   for c in git2.calls)
    assert journal.latest("s", _TIP).terminal == "promote_failed"


def test_replace_helper_sanity() -> None:
    # Guard the record dataclass is frozen + replace-friendly (used by the orchestrator's _write).
    rec = MergeBackRecord(strategy="s", branch="b", branch_tip="t")
    assert replace(rec, terminal="x").terminal == "x"
    assert rec.terminal is None
