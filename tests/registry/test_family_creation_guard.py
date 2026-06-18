"""TDD tests for Task 4 of #222: governed family creation guard in promotion_preflight.

Tests cover the clustering verdict paths inserted into promotion_preflight BEFORE the
breadth-resolution step.  All heavy I/O (load_strategy, compute_artifact_hashes,
verify_signal_panel_parity, factors_used_by) is patched out so tests are fast and
deterministic.

Strategy under test: ``promotion_preflight`` from ``algua.registry.promotion``.
"""
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from algua.backtest._sample import SyntheticProvider
from algua.contracts.lifecycle import Actor, Stage
from algua.registry import db
from algua.registry.approvals import ArtifactIdentity
from algua.registry.promotion import BreadthContext, promotion_preflight
from algua.registry.store import SqliteStrategyRepository
from algua.research.clustering import (
    SimVerdict,
    clustering_version,
)

_START = datetime(2024, 1, 1, tzinfo=UTC)
_END = datetime(2024, 6, 1, tzinfo=UTC)

_FAKE_IDENTITY = ArtifactIdentity(
    code_hash="deadbeef" * 8,   # 64-char hex
    config_hash="cafebabe" * 8,
    dependency_hash="f00dcafe" * 8,
)

_MEMBER_CODE_HASH_IDENTICAL = "deadbeef" * 8   # same as _FAKE_IDENTITY.code_hash → MERGE score
_MEMBER_CODE_HASH_DIFFERENT = "11112222" * 8


def _make_repo() -> SqliteStrategyRepository:
    """Fresh in-memory repo with the current schema applied."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON;")
    db.migrate(conn)
    return SqliteStrategyRepository(conn)


def _add_backtested_strategy(repo: SqliteStrategyRepository, name: str) -> None:
    """Add a strategy and advance it to BACKTESTED (the required source stage)."""
    rec = repo.add(name)
    repo.apply_transition(rec, Stage.BACKTESTED, Actor.HUMAN, "setup")
    # Record measured breadth so the breadth gate doesn't fire.
    repo.record_search_trial(name, 4, "{}")


def _assign_to_family(
    repo: SqliteStrategyRepository,
    strategy_name: str,
    family_id: int,
    actor: str = "agent",
) -> None:
    """Helper to assign a strategy to a family with minimal dummy metadata."""
    repo.assign_strategy_to_family(
        strategy_name,
        family_id,
        actor,
        verdict="NOVEL",
        similarity_score=0.0,
        clustering_version=clustering_version(),
        clustering_config_json="{}",
        axis_json="{}",
        matched_family_id=None,
    )


def _call_preflight(
    repo: SqliteStrategyRepository,
    name: str,
    *,
    actor: Actor,
    new_family_slug: str | None = None,
) -> BreadthContext:
    """Call promotion_preflight with all non-clustering I/O patched out."""
    provider = SyntheticProvider(seed=0)
    with (
        patch("algua.registry.promotion.load_strategy", return_value=None),
        patch(
            "algua.registry.promotion.compute_artifact_hashes",
            return_value=_FAKE_IDENTITY,
        ),
        patch("algua.registry.promotion.verify_signal_panel_parity"),
        patch(
            "algua.registry.promotion.factors_used_by",
            return_value=[],
        ),
    ):
        return promotion_preflight(
            repo,
            name,
            actor=actor,
            declared_combos=None,
            allow_holdout_reuse=False,
            allow_non_pit=False,
            provider=provider,
            start=_START,
            end=_END,
            new_family_slug=new_family_slug,
        )


# ---------------------------------------------------------------------------
# BreadthContext dataclass extension
# ---------------------------------------------------------------------------

def test_breadth_context_has_family_id_field() -> None:
    """BreadthContext must expose family_id and expected_family_id with None defaults."""
    ctx = BreadthContext(n_funnel=10, own=5, windowed_total=10, provenance="measured")
    assert ctx.family_id is None
    assert ctx.expected_family_id is None


# ---------------------------------------------------------------------------
# NOVEL verdict — agent path
# ---------------------------------------------------------------------------

def test_agent_novel_verdict_raises(tmp_path) -> None:
    """No existing families; actor=agent → ValueError (empty-registry message)."""
    repo = _make_repo()
    _add_backtested_strategy(repo, "strat_novel_agent")

    with pytest.raises(ValueError, match="family registry is empty"):
        _call_preflight(repo, "strat_novel_agent", actor=Actor.AGENT)


def test_agent_novel_message_mentions_family_assignment(tmp_path) -> None:
    """The ValueError for an agent NOVEL should mention family."""
    repo = _make_repo()
    _add_backtested_strategy(repo, "strat_novel_msg")

    with pytest.raises(ValueError, match="family"):
        _call_preflight(repo, "strat_novel_msg", actor=Actor.AGENT)


# ---------------------------------------------------------------------------
# NOVEL verdict — human path
# ---------------------------------------------------------------------------

def test_human_novel_without_slug_raises() -> None:
    """No existing families; actor=human, new_family_slug=None → ValueError."""
    repo = _make_repo()
    _add_backtested_strategy(repo, "strat_human_novel")

    with pytest.raises(ValueError, match="--new-family"):
        _call_preflight(repo, "strat_human_novel", actor=Actor.HUMAN, new_family_slug=None)


def test_human_novel_creates_family() -> None:
    """No existing families; actor=human, new_family_slug provided → new family created."""
    repo = _make_repo()
    _add_backtested_strategy(repo, "strat_novel_new_fam")

    ctx = _call_preflight(
        repo, "strat_novel_new_fam", actor=Actor.HUMAN, new_family_slug="momentum_v2"
    )

    fam_id = repo.strategy_family("strat_novel_new_fam")
    assert fam_id is not None
    assert ctx.family_id == fam_id

    # The family name should be "momentum_v2"
    row = repo.connection.execute(
        "SELECT name FROM families WHERE id=?", (fam_id,)
    ).fetchone()
    assert row["name"] == "momentum_v2"


def test_human_novel_family_event_row_written() -> None:
    """After human NOVEL create, a family_events row with event_type=family_created exists."""
    repo = _make_repo()
    _add_backtested_strategy(repo, "strat_event_check")

    _call_preflight(
        repo, "strat_event_check", actor=Actor.HUMAN, new_family_slug="my_family"
    )

    rows = repo.connection.execute(
        "SELECT event_type FROM family_events WHERE event_type='family_created'"
    ).fetchall()
    assert len(rows) >= 1


# ---------------------------------------------------------------------------
# MERGE verdict — agent and human
# ---------------------------------------------------------------------------

def _setup_family_for_clustering(
    repo: SqliteStrategyRepository, family_name: str, member_strategy: str
) -> int:
    """Create a family and assign an existing strategy as its member."""
    try:
        repo.add(member_strategy)
    except Exception:
        pass  # already exists
    fam_id = repo.create_family(family_name, actor="human")
    _assign_to_family(repo, member_strategy, fam_id, actor="human")
    return fam_id


def test_agent_merge_assigns_to_matched_family() -> None:
    """family_similarity returns MERGE → strategy assigned to existing family F."""
    repo = _make_repo()
    fam_id = _setup_family_for_clustering(repo, "fam_merge", "existing_member")
    _add_backtested_strategy(repo, "strat_merge_agent")

    # Patch family_similarity to return MERGE directly (MERGE_THRESHOLD is 0.85, unreachable
    # without the return-correlation axis which is stubbed to 0 until Task 7).
    with patch(
        "algua.registry.promotion.family_similarity",
        return_value=(SimVerdict.MERGE, 0.9),
    ), patch(
        "algua.registry.promotion._get_all_family_members_for_clustering",
        return_value=[(fam_id, [{"code_hash": _FAKE_IDENTITY.code_hash, "factors": set()}])],
    ):
        ctx = _call_preflight(repo, "strat_merge_agent", actor=Actor.AGENT)

    assert repo.strategy_family("strat_merge_agent") == fam_id
    assert ctx.family_id == fam_id


def test_human_merge_assigns_to_matched_family() -> None:
    """MERGE verdict with actor=human → assigned to existing family (no child created)."""
    repo = _make_repo()
    fam_id = _setup_family_for_clustering(repo, "fam_merge_h", "existing_h")
    _add_backtested_strategy(repo, "strat_merge_human")

    with patch(
        "algua.registry.promotion.family_similarity",
        return_value=(SimVerdict.MERGE, 0.9),
    ), patch(
        "algua.registry.promotion._get_all_family_members_for_clustering",
        return_value=[(fam_id, [{"code_hash": _FAKE_IDENTITY.code_hash, "factors": set()}])],
    ):
        ctx = _call_preflight(repo, "strat_merge_human", actor=Actor.HUMAN)

    assert repo.strategy_family("strat_merge_human") == fam_id
    assert ctx.family_id == fam_id


def test_merge_family_events_has_clustering_version() -> None:
    """After MERGE assignment, family_events row has a non-None clustering_version."""
    repo = _make_repo()
    fam_id = _setup_family_for_clustering(repo, "fam_cv", "existing_cv")
    _add_backtested_strategy(repo, "strat_cv_check")

    with patch(
        "algua.registry.promotion.family_similarity",
        return_value=(SimVerdict.MERGE, 0.9),
    ), patch(
        "algua.registry.promotion._get_all_family_members_for_clustering",
        return_value=[(fam_id, [{"code_hash": _FAKE_IDENTITY.code_hash, "factors": set()}])],
    ):
        _call_preflight(repo, "strat_cv_check", actor=Actor.AGENT)

    row = repo.connection.execute(
        "SELECT clustering_version FROM family_events"
        " WHERE strategy_name='strat_cv_check' AND clustering_version IS NOT NULL"
        " LIMIT 1"
    ).fetchone()
    assert row is not None
    assert row["clustering_version"] == clustering_version()


# ---------------------------------------------------------------------------
# PARENTAGE verdict
# ---------------------------------------------------------------------------

def _parentage_member_profile() -> dict:
    """Return a member profile that yields PARENTAGE (not MERGE) against _FAKE_IDENTITY.

    PARENTAGE: score in [PARENTAGE_THRESHOLD, MERGE_THRESHOLD).
    A different code_hash gives code_score=0.0 (weight 0.50).
    We need total score ≥ 0.50 but < 0.85.
    Factor Jaccard of 1.0 gives factor_score=0.30, total=0.30 → NOVEL.
    We need a partial factor overlap to hit PARENTAGE range.
    PARENTAGE_THRESHOLD = 0.50 → impossible with only code+factor axes summing to 0.80 at max
    without matching code.

    Actually: if code matches (code_score=1.0) but it's a DIFFERENT hash, code_score=0.0.
    Factor-only can't reach PARENTAGE_THRESHOLD=0.50 (max factor contribution = 0.30).

    So we need code_hash identical for MERGE. For PARENTAGE we need a partial code match,
    which clustering doesn't support (it's binary). We can only hit PARENTAGE by:
    - same code hash (code_score=1.0): 0.50*1.0 + 0.30*jaccard + 0 = 0.50 + 0.30*j
      which is ≥ 0.85 iff j ≥ 1.167... → impossible (j ≤ 1). So same code = MERGE (j=0).
      Wait: 0.50*1.0 = 0.50, plus 0.30*j. For j=0: total=0.50, which equals PARENTAGE_THRESHOLD.
      So code_match + no_factors = score 0.50 → PARENTAGE (score >= 0.50 but < 0.85).

    Let's verify: WEIGHT_CODE_ANCESTRY=0.50, WEIGHT_FACTOR_LINEAGE=0.30, WEIGHT_RETURN=0.20.
    code_score=1.0, factor_score=0.0, return_score=0.0 → score=0.50.
    0.50 >= PARENTAGE_THRESHOLD (0.50) AND 0.50 < MERGE_THRESHOLD (0.85) → PARENTAGE. ✓
    """
    return {"code_hash": _FAKE_IDENTITY.code_hash, "factors": {"other_factor_not_used_by_strat"}}


def test_agent_parentage_resolves_to_parent_family() -> None:
    """PARENTAGE verdict + actor=agent → assigned to existing family (not a new child)."""
    repo = _make_repo()
    fam_id = repo.create_family("fam_parentage", actor="human")
    _add_backtested_strategy(repo, "strat_parentage_agent")

    member_profile = _parentage_member_profile()
    with patch(
        "algua.registry.promotion._get_all_family_members_for_clustering",
        return_value=[(fam_id, [member_profile])],
    ):
        ctx = _call_preflight(repo, "strat_parentage_agent", actor=Actor.AGENT)

    # Agent: PARENTAGE → assigned to parent family, no child created
    assert repo.strategy_family("strat_parentage_agent") == fam_id
    assert ctx.family_id == fam_id

    # Confirm no new families were created (only the one we set up)
    count = repo.connection.execute("SELECT COUNT(*) c FROM families").fetchone()["c"]
    assert count == 1


def test_human_parentage_creates_child_with_parent_edge() -> None:
    """PARENTAGE verdict + actor=human → new child family created with parent edge."""
    repo = _make_repo()
    fam_id = repo.create_family("fam_parent_h", actor="human")
    _add_backtested_strategy(repo, "strat_parentage_human")

    member_profile = _parentage_member_profile()
    with patch(
        "algua.registry.promotion._get_all_family_members_for_clustering",
        return_value=[(fam_id, [member_profile])],
    ):
        ctx = _call_preflight(
            repo,
            "strat_parentage_human",
            actor=Actor.HUMAN,
            new_family_slug="child_family",
        )

    child_fam_id = repo.strategy_family("strat_parentage_human")
    assert child_fam_id is not None
    assert child_fam_id != fam_id  # new child family
    assert ctx.family_id == child_fam_id

    # Parent edge must exist: child's ancestry includes fam_id
    ancestors = repo.family_ancestry(child_fam_id)
    assert fam_id in ancestors


def test_human_parentage_child_family_uses_slug() -> None:
    """Human PARENTAGE with new_family_slug → child family name matches slug."""
    repo = _make_repo()
    fam_id = repo.create_family("fam_parent_slug", actor="human")
    _add_backtested_strategy(repo, "strat_slug_check")

    member_profile = _parentage_member_profile()
    with patch(
        "algua.registry.promotion._get_all_family_members_for_clustering",
        return_value=[(fam_id, [member_profile])],
    ):
        _call_preflight(
            repo,
            "strat_slug_check",
            actor=Actor.HUMAN,
            new_family_slug="custom_child_slug",
        )

    child_fam_id = repo.strategy_family("strat_slug_check")
    row = repo.connection.execute(
        "SELECT name FROM families WHERE id=?", (child_fam_id,)
    ).fetchone()
    assert row["name"] == "custom_child_slug"


def test_human_parentage_no_slug_uses_default_name() -> None:
    """Human PARENTAGE with no slug → child family name is f'{strategy_name}_family'."""
    repo = _make_repo()
    fam_id = repo.create_family("fam_default_slug", actor="human")
    _add_backtested_strategy(repo, "strat_default_name")

    member_profile = _parentage_member_profile()
    with patch(
        "algua.registry.promotion._get_all_family_members_for_clustering",
        return_value=[(fam_id, [member_profile])],
    ):
        _call_preflight(
            repo,
            "strat_default_name",
            actor=Actor.HUMAN,
            new_family_slug=None,
        )

    child_fam_id = repo.strategy_family("strat_default_name")
    row = repo.connection.execute(
        "SELECT name FROM families WHERE id=?", (child_fam_id,)
    ).fetchone()
    assert row["name"] == "strat_default_name_family"


# ---------------------------------------------------------------------------
# Already-assigned strategy — skip reclassification
# ---------------------------------------------------------------------------

def test_already_assigned_strategy_skips_reclassification() -> None:
    """Strategy already has a family assignment → no new clustering call, family_id unchanged."""
    repo = _make_repo()
    fam_id = repo.create_family("fam_existing", actor="agent")
    _add_backtested_strategy(repo, "strat_preassigned")
    _assign_to_family(repo, "strat_preassigned", fam_id)

    cluster_mock = MagicMock(return_value=(SimVerdict.NOVEL, 0.0))
    with patch("algua.registry.promotion.family_similarity", cluster_mock):
        ctx = _call_preflight(repo, "strat_preassigned", actor=Actor.AGENT)

    # family_similarity must NOT have been called — already assigned
    cluster_mock.assert_not_called()
    assert ctx.family_id == fam_id
    assert ctx.expected_family_id == fam_id


# ---------------------------------------------------------------------------
# BreadthContext carries family_id
# ---------------------------------------------------------------------------

def test_breadth_context_carries_family_id() -> None:
    """preflight returns BreadthContext with family_id set (not None) after classification."""
    repo = _make_repo()
    fam_id = repo.create_family("fam_ctx_check", actor="agent")
    _add_backtested_strategy(repo, "strat_ctx")
    _assign_to_family(repo, "strat_ctx", fam_id)

    ctx = _call_preflight(repo, "strat_ctx", actor=Actor.AGENT)

    assert ctx.family_id is not None
    assert ctx.family_id == fam_id


def test_breadth_context_expected_family_id_matches_family_id() -> None:
    """BreadthContext.expected_family_id must equal family_id after successful classification."""
    repo = _make_repo()
    fam_id = repo.create_family("fam_expected", actor="agent")
    _add_backtested_strategy(repo, "strat_expected")
    _assign_to_family(repo, "strat_expected", fam_id)

    ctx = _call_preflight(repo, "strat_expected", actor=Actor.AGENT)

    assert ctx.expected_family_id == ctx.family_id


# ---------------------------------------------------------------------------
# family_events clustering_version tracking
# ---------------------------------------------------------------------------

def test_family_events_row_has_clustering_version_after_novel_create() -> None:
    """After human NOVEL create + assignment, family_events has a clustering_version."""
    repo = _make_repo()
    _add_backtested_strategy(repo, "strat_cv_novel")

    _call_preflight(
        repo, "strat_cv_novel", actor=Actor.HUMAN, new_family_slug="novel_fam_cv"
    )

    row = repo.connection.execute(
        "SELECT clustering_version FROM family_events"
        " WHERE strategy_name='strat_cv_novel' AND clustering_version IS NOT NULL LIMIT 1"
    ).fetchone()
    assert row is not None
    assert row["clustering_version"] == clustering_version()
