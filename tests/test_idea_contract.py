# tests/test_idea_contract.py
from algua.contracts.idea import (
    REFUTING_OUTCOMES,
    AttemptOutcome,
    Horizon,
    IdeaStatus,
    Market,
    Obscurity,
    SourceType,
    can_change_status,
)


def test_new_enums_have_exact_values():
    assert [m.value for m in Market] == ["us_equities", "crypto", "forex", "prediction", "any"]
    assert [h.value for h in Horizon] == ["intraday", "daily", "weekly", "monthly", "event"]
    assert [o.value for o in Obscurity] == ["canon", "common", "niche", "rare"]
    assert [a.value for a in AttemptOutcome] == [
        "integrity_fail", "holdout_negative", "walkforward_refuted", "sweep_unstable",
        "candidate_preview_pass", "promoted_candidate", "abandoned", "run_error",
    ]
    assert SourceType.INSPIRATION.value == "inspiration"


def test_refuting_outcomes_are_the_four_research_failures():
    assert REFUTING_OUTCOMES == frozenset({
        AttemptOutcome.INTEGRITY_FAIL, AttemptOutcome.HOLDOUT_NEGATIVE,
        AttemptOutcome.WALKFORWARD_REFUTED, AttemptOutcome.SWEEP_UNSTABLE,
    })


def test_open_to_refuted_is_now_legal_but_authored_to_open_is_not():
    assert can_change_status(IdeaStatus.OPEN, IdeaStatus.REFUTED)
    assert not can_change_status(IdeaStatus.AUTHORED, IdeaStatus.OPEN)
    assert not can_change_status(IdeaStatus.REFUTED, IdeaStatus.OPEN)
