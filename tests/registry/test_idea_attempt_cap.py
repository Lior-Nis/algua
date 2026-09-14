"""An idea that never reaches a verdict must not be retried forever.

Only `REFUTING_OUTCOMES` move an idea to `refuted`. `run_error` and `abandoned` release the claim
and leave it OPEN, so the next research cycle claims it again. That is correct for a transient
failure and catastrophic for a structural one: an idea whose construction the platform cannot
express fails identically every pass.

It happened. On 2026-09-14 ideas 9, 13 and 14 each carried FOUR `run_error` attempts, six
consecutive research runs authored zero strategies, and the loop was cycling the same handful of
ideas with no new work reachable -- a livelock that consumed a research cycle (and the model quota
behind it) every two hours while producing nothing.
"""

from __future__ import annotations

import pytest

from algua.contracts.idea import (
    AttemptOutcome,
    DataCapability,
    Horizon,
    IdeaStatus,
    Market,
    SourceType,
)
from algua.registry.db import connect, migrate
from algua.registry.idea_attempts import MAX_IDEA_ATTEMPTS, IdeaAttemptsRepository
from algua.registry.ideas import IdeaRepository

TTL = 60


@pytest.fixture
def repos(tmp_path):
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    return IdeaRepository(conn), IdeaAttemptsRepository(conn)


def _add(ideas: IdeaRepository, title: str) -> int:
    return ideas.add(
        title=title, hypothesis=f"hypothesis for {title}", family=None, tags=[],
        source_type=SourceType.MANUAL, source_ref=None, source_date=None, source_note=None,
        required_data=[DataCapability.OHLCV], status=IdeaStatus.OPEN, category="momentum",
        market=Market.US_EQUITIES, horizon=Horizon.DAILY, falsification="refuted if x",
        created_by_run="t",
    ).id


def _work(attempts: IdeaAttemptsRepository, idea_id: int, outcome: AttemptOutcome, run: str):
    (claimed,) = attempts.claim(run_stamp=run, limit=1, ttl_minutes=TTL)
    assert claimed.id == idea_id
    return attempts.record_outcome(
        idea_id, token=claimed.claim_token, outcome=outcome, reason="structurally unbuildable")


def test_repeated_run_errors_eventually_discard_the_idea(repos):
    ideas, attempts = repos
    idea_id = _add(ideas, "an idea the construction policies cannot express")

    for i in range(MAX_IDEA_ATTEMPTS - 1):
        after = _work(attempts, idea_id, AttemptOutcome.RUN_ERROR, f"r{i}")
        assert after.status is IdeaStatus.OPEN, "a transient failure must stay retryable"

    final = _work(attempts, idea_id, AttemptOutcome.RUN_ERROR, "r-last")
    assert final.status is IdeaStatus.DISCARDED


def test_a_discarded_idea_is_no_longer_claimable(repos):
    """The point of the cap: the cycle stops paying for it."""
    ideas, attempts = repos
    idea_id = _add(ideas, "unbuildable")
    for i in range(MAX_IDEA_ATTEMPTS):
        _work(attempts, idea_id, AttemptOutcome.RUN_ERROR, f"r{i}")

    assert attempts.claim(run_stamp="next", limit=5, ttl_minutes=TTL) == []


def test_the_cap_does_not_refute_what_was_never_tested(repos):
    """DISCARDED, not REFUTED.

    Nothing was learned about whether the edge is real -- only that this pool entry cannot be
    worked. Marking it refuted would poison the scorecard's refutation rate and tell the leap step
    a venue produces WRONG ideas when it is producing UNBUILDABLE ones.
    """
    ideas, attempts = repos
    idea_id = _add(ideas, "unbuildable")
    for i in range(MAX_IDEA_ATTEMPTS):
        final = _work(attempts, idea_id, AttemptOutcome.RUN_ERROR, f"r{i}")
    assert final.status is IdeaStatus.DISCARDED
    assert final.status is not IdeaStatus.REFUTED


def test_a_real_verdict_still_refutes_on_the_first_attempt(repos):
    """The cap must not delay or dilute a genuine refutation."""
    ideas, attempts = repos
    idea_id = _add(ideas, "an idea that gets tested and fails")
    after = _work(attempts, idea_id, AttemptOutcome.HOLDOUT_NEGATIVE, "r0")
    assert after.status is IdeaStatus.REFUTED


def test_an_explicit_abandon_retires_on_the_first_attempt_too(repos):
    """An agent that KNOWS an idea is impossible should not have to fail three times to say so.

    The cap is a safety net that works regardless of which outcome the agent picks; `abandoned` is
    the precise signal, and after it the idea is equally gone.
    """
    ideas, attempts = repos
    idea_id = _add(ideas, "impossible, and the agent knows it")
    _work(attempts, idea_id, AttemptOutcome.ABANDONED, "r0")
    _work(attempts, idea_id, AttemptOutcome.ABANDONED, "r1")
    final = _work(attempts, idea_id, AttemptOutcome.ABANDONED, "r2")
    assert final.status is IdeaStatus.DISCARDED
    assert attempts.claim(run_stamp="next", limit=5, ttl_minutes=TTL) == []
