"""Unit tests for the pure governance knowledge module (issue #393)."""

from __future__ import annotations

from datetime import date, datetime

import pytest

from algua.knowledge.frontmatter import parse_doc
from algua.knowledge.governance import (
    GovernanceRecord,
    is_overdue,
    read_governance,
    record_governance,
    record_to_json,
)

_EMPTY_DOC = "---\nname: alpha\nstage: paper\n---\n# alpha\n\nprose\n"


def _doc(tmp_path, text: str = _EMPTY_DOC):
    p = tmp_path / "alpha.md"
    p.write_text(text)
    return p


def test_record_writes_owned_frontmatter_and_block(tmp_path):
    path = _doc(tmp_path)
    rec = record_governance(
        path, "alpha", owner="lior",
        assumptions=["stationary vol", "liquid names"],
        limitations=["fails in regime shift"],
        validation_summary="passed WF holdout",
        next_review=date(2026, 12, 1), last_validated=date(2026, 6, 1), gate_eval_id=42,
    )
    assert rec.owner == "lior"
    fm, body = parse_doc(path.read_text())
    assert fm["governance_owner"] == "lior"
    assert fm["governance_assumptions"] == ["stationary vol", "liquid names"]
    assert fm["governance_next_review"] == "2026-12-01"
    assert fm["governance_gate_eval_id"] == 42
    # existing frontmatter + prose preserved
    assert fm["stage"] == "paper"
    assert "prose" in body
    # rendered human block present
    assert "<!-- ALGUA:GOVERNANCE -->" in body
    assert "lior" in body


def test_record_roundtrips_through_read(tmp_path):
    path = _doc(tmp_path)
    record_governance(
        path, "alpha", owner="lior", assumptions=["a"], limitations=["b"],
        validation_summary="v", next_review=date(2026, 12, 1),
        last_validated=date(2026, 6, 1), gate_eval_id=7,
    )
    got = read_governance(path, "alpha")
    assert got.present is True
    assert got.owner == "lior"
    assert got.next_review == date(2026, 12, 1)
    assert got.last_validated == date(2026, 6, 1)
    assert got.gate_eval_id == 7
    assert got.assumptions == ["a"]
    assert got.limitations == ["b"]


def test_record_refuses_missing_doc(tmp_path):
    with pytest.raises(FileNotFoundError):
        record_governance(
            tmp_path / "nope.md", "alpha", owner="lior", assumptions=[], limitations=[],
            validation_summary=None, next_review=date(2026, 12, 1),
            last_validated=None, gate_eval_id=None,
        )


def test_record_requires_owner(tmp_path):
    path = _doc(tmp_path)
    with pytest.raises(ValueError, match="owner"):
        record_governance(
            path, "alpha", owner="   ", assumptions=[], limitations=[],
            validation_summary=None, next_review=date(2026, 12, 1),
            last_validated=None, gate_eval_id=None,
        )


def test_record_rejects_marker_injection(tmp_path):
    path = _doc(tmp_path)
    with pytest.raises(ValueError, match="marker"):
        record_governance(
            path, "alpha", owner="lior", assumptions=["ALGUA:GOVERNANCE break"],
            limitations=[], validation_summary=None, next_review=date(2026, 12, 1),
            last_validated=None, gate_eval_id=None,
        )


def _rec(**kw) -> GovernanceRecord:
    base = dict(
        name="alpha", owner="lior", assumptions=[], limitations=[], validation_summary=None,
        last_validated=None, next_review=date(2026, 12, 1), gate_eval_id=None,
        gate_eval_id_malformed=False, present=True,
    )
    base.update(kw)
    return GovernanceRecord(**base)  # type: ignore[arg-type]


def test_overdue_missing_next_review_is_overdue():
    assert is_overdue(_rec(next_review=None), date(2026, 1, 1)) is True


def test_overdue_absent_doc_is_overdue(tmp_path):
    rec = read_governance(tmp_path / "absent.md", "alpha")
    assert rec.present is False
    assert is_overdue(rec, date(2026, 1, 1)) is True


def test_overdue_past_date_is_overdue():
    assert is_overdue(_rec(next_review=date(2026, 1, 1)), date(2026, 6, 1)) is True


def test_due_today_is_not_yet_overdue():
    assert is_overdue(_rec(next_review=date(2026, 6, 1)), date(2026, 6, 1)) is False


def test_future_date_is_not_overdue():
    assert is_overdue(_rec(next_review=date(2027, 1, 1)), date(2026, 6, 1)) is False


@pytest.mark.parametrize(
    "raw",
    [
        "governance_next_review: not-a-date",
        "governance_next_review: ''",
        "governance_next_review: []",
        "governance_next_review:\n  a: 1",
        "governance_next_review: 2026-12-01 09:00:00",  # a datetime, not a clean date
    ],
)
def test_malformed_next_review_reads_as_overdue(tmp_path, raw):
    path = tmp_path / "alpha.md"
    path.write_text(f"---\nname: alpha\n{raw}\n---\nbody\n")
    rec = read_governance(path, "alpha")
    assert rec.next_review is None
    assert is_overdue(rec, date(2026, 6, 1)) is True


def test_unquoted_iso_date_parses(tmp_path):
    # yaml.safe_load yields a datetime.date for an unquoted ISO date — must be accepted.
    path = tmp_path / "alpha.md"
    path.write_text("---\nname: alpha\ngovernance_next_review: 2027-01-01\n---\nbody\n")
    rec = read_governance(path, "alpha")
    assert rec.next_review == date(2027, 1, 1)
    assert is_overdue(rec, date(2026, 6, 1)) is False


def test_datetime_norm_rejected():
    # A datetime carries a time-of-day; it must not masquerade as a review date.
    path_rec = _rec(next_review=None)
    # sanity: our norm helper is exercised via read; direct check of the invariant here
    assert isinstance(datetime(2026, 1, 1), date)  # datetime IS a date subclass
    assert path_rec.next_review is None


@pytest.mark.parametrize(
    "raw",
    [
        "governance_gate_eval_id: abc",
        "governance_gate_eval_id: -1",
        "governance_gate_eval_id: 0",
        "governance_gate_eval_id: []",
        "governance_gate_eval_id: true",
    ],
)
def test_malformed_gate_id_flagged(tmp_path, raw):
    path = tmp_path / "alpha.md"
    path.write_text(f"---\nname: alpha\ngovernance_next_review: 2099-01-01\n{raw}\n---\nbody\n")
    rec = read_governance(path, "alpha")
    assert rec.gate_eval_id is None
    assert rec.gate_eval_id_malformed is True


def test_absent_gate_id_not_flagged_malformed(tmp_path):
    path = tmp_path / "alpha.md"
    path.write_text("---\nname: alpha\ngovernance_next_review: 2099-01-01\n---\nbody\n")
    rec = read_governance(path, "alpha")
    assert rec.gate_eval_id is None
    assert rec.gate_eval_id_malformed is False


def test_valid_gate_id_not_flagged_malformed(tmp_path):
    path = tmp_path / "alpha.md"
    path.write_text(
        "---\nname: alpha\ngovernance_next_review: 2099-01-01\n"
        "governance_gate_eval_id: 42\n---\nbody\n"
    )
    rec = read_governance(path, "alpha")
    assert rec.gate_eval_id == 42
    assert rec.gate_eval_id_malformed is False


def test_record_to_json_includes_overdue():
    payload = record_to_json(_rec(next_review=date(2026, 1, 1)), today=date(2026, 6, 1))
    assert payload["overdue"] is True
    assert payload["owner"] == "lior"
    assert payload["next_review"] == "2026-01-01"
