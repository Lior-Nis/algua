"""Tests for the durable per-strategy merge-back journal (#485, Task 1)."""

from __future__ import annotations

from pathlib import Path

from algua.operator.journal import (
    JsonlJournal,
    MergeBackRecord,
    derive_attempt_token,
    strict_relaxation_fingerprint,
)


def test_append_and_read_back(tmp_path: Path) -> None:
    j = JsonlJournal(tmp_path)
    rec = MergeBackRecord(strategy="s", branch="feat/s", branch_tip="TIP", merge_sha="M")
    j.append(rec)
    assert j.latest("s", "TIP") == rec


def test_latest_wins_on_duplicate_key(tmp_path: Path) -> None:
    j = JsonlJournal(tmp_path)
    j.append(MergeBackRecord(strategy="s", branch="b", branch_tip="TIP", diff_policy="pending"))
    j.append(MergeBackRecord(strategy="s", branch="b", branch_tip="TIP", diff_policy="passed",
                             terminal="promoted_allocated"))
    latest = j.latest("s", "TIP")
    assert latest is not None
    assert latest.diff_policy == "passed"
    assert latest.terminal == "promoted_allocated"


def test_missing_returns_none(tmp_path: Path) -> None:
    assert JsonlJournal(tmp_path).latest("nobody", "TIP") is None


def test_different_branch_tip_is_a_new_attempt(tmp_path: Path) -> None:
    j = JsonlJournal(tmp_path)
    j.append(MergeBackRecord(strategy="s", branch="b", branch_tip="TIP1", terminal="gate_failed"))
    assert j.latest("s", "TIP2") is None
    assert j.latest("s", "TIP1").terminal == "gate_failed"


def test_crash_truncated_last_line_is_ignored(tmp_path: Path) -> None:
    j = JsonlJournal(tmp_path)
    good = MergeBackRecord(strategy="s", branch="b", branch_tip="TIP", gate_status="green")
    j.append(good)
    # Simulate a torn final append (a half-written JSON line from a crash mid-write).
    path = tmp_path / "merge_back.s.journal"
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"strategy": "s", "branch_tip": "TIP", "gate_stat')
    assert j.latest("s", "TIP") == good  # reader stops at the last well-formed record


def test_two_strategies_write_disjoint_files(tmp_path: Path) -> None:
    j = JsonlJournal(tmp_path)
    j.append(MergeBackRecord(strategy="alpha", branch="b", branch_tip="T", terminal="gate_failed"))
    j.append(MergeBackRecord(strategy="beta", branch="b", branch_tip="T",
                             terminal="promoted_queued"))
    assert (tmp_path / "merge_back.alpha.journal").exists()
    assert (tmp_path / "merge_back.beta.journal").exists()
    assert j.latest("alpha", "T").terminal == "gate_failed"
    assert j.latest("beta", "T").terminal == "promoted_queued"


def test_strategy_name_is_sanitized(tmp_path: Path) -> None:
    j = JsonlJournal(tmp_path)
    # A name with path-ish characters must not escape the journal dir.
    j.append(MergeBackRecord(strategy="a/../b", branch="x", branch_tip="T", terminal="gate_failed"))
    files = list(tmp_path.glob("merge_back.*.journal"))
    assert len(files) == 1
    assert files[0].parent == tmp_path
    assert j.latest("a/../b", "T").terminal == "gate_failed"


def test_attempt_token_is_deterministic_and_context_sensitive() -> None:
    fp = strict_relaxation_fingerprint()
    t1 = derive_attempt_token("s", "TIP", "M", fp)
    t2 = derive_attempt_token("s", "TIP", "M", fp)
    assert t1 == t2  # re-derivable on every recovery pass
    assert derive_attempt_token("s", "TIP", "M2", fp) != t1  # different merge -> different token
    assert derive_attempt_token("s", "TIP", "M", "other") != t1  # relaxed context -> new token
    assert len(t1) == 64  # sha256 hex
