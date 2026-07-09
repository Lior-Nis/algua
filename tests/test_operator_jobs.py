"""Tests for the per-job operator manifest (#486, round-3 fix #4 / #1)."""

from __future__ import annotations

import pytest

from algua.operator.jobs import OPERATOR_JOBS, CommandMismatch, OperatorJob

_PAPER_ARGV = ("algua", "paper", "run-all", "--snapshot", "SNAP")


def _paper() -> OperatorJob:
    return OPERATOR_JOBS["paper"]


# --- manifest shape ---------------------------------------------------------------------------


def test_manifest_ships_only_the_paper_job() -> None:
    # The research job is deferred (round-3 #1): exactly one job ships today.
    assert set(OPERATOR_JOBS) == {"paper"}
    assert _paper().expected_duration_seconds == 900.0


def test_unknown_key_lookup_returns_none() -> None:
    assert OPERATOR_JOBS.get("frobnicate") is None


# --- is_completed truth table -----------------------------------------------------------------


def test_is_completed_rc0_no_deferred_is_true() -> None:
    assert _paper().is_completed(0, {"ok": True}) is True


def test_is_completed_rc0_deferred_is_false() -> None:
    assert _paper().is_completed(0, {"ok": True, "deferred": True}) is False


def test_is_completed_nonzero_is_false() -> None:
    assert _paper().is_completed(1, {"ok": False}) is False
    assert _paper().is_completed(2, None) is False


def test_is_completed_rc0_without_explicit_ok_true_is_false() -> None:
    # GATE-2 (#486): rc==0 alone is not proof of success — the driver must say `ok: true` itself.
    # A missing/absent-payload/`ok:false`-at-rc0 outcome must never be trusted as a completion (the
    # in-repo `paper run-all` always exits non-zero on `ok:false` today, but the predicate is the
    # only backstop against a future/altered driver silently marking a broken session complete).
    assert _paper().is_completed(0, None) is False
    assert _paper().is_completed(0, {"ok": False}) is False
    assert _paper().is_completed(0, {}) is False


# --- bind: exact-arity full-argv match --------------------------------------------------------


def test_bind_accepts_canonical_argv_and_captures_snapshot() -> None:
    assert _paper().bind(_PAPER_ARGV) == {"snapshot": "SNAP"}


def test_bind_rejects_trailing_extra_token() -> None:
    with pytest.raises(CommandMismatch):
        _paper().bind((*_PAPER_ARGV, "--evil"))


def test_bind_rejects_wrong_head() -> None:
    with pytest.raises(CommandMismatch):
        _paper().bind(("algua", "data", "inspect", "--snapshot", "SNAP"))


def test_bind_rejects_missing_snapshot_value() -> None:
    with pytest.raises(CommandMismatch):
        _paper().bind(("algua", "paper", "run-all", "--snapshot"))


def test_bind_rejects_empty_placeholder_value() -> None:
    with pytest.raises(CommandMismatch):
        _paper().bind(("algua", "paper", "run-all", "--snapshot", ""))


def test_bind_rejects_swapped_flag() -> None:
    with pytest.raises(CommandMismatch):
        _paper().bind(("algua", "paper", "run-all", "--dataset", "SNAP"))


def test_bind_rejects_short_arity() -> None:
    with pytest.raises(CommandMismatch):
        _paper().bind(("algua", "paper", "run-all"))
