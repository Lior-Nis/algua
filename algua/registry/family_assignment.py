"""Family classification (#222/#524): the MERGE / PARENTAGE / NOVEL clustering verdict, and
assignment of a strategy into a family.

The NOVEL + agent path is deliberately DEFERRED: no family is created here. A repeated-founder
attack defeats the naive "mint on NOVEL" design (the founder passes and escapes the family-lifetime
breadth tax before its family exists), so classification instead returns a ``PendingNovelFamily``
spec and the seeded family is minted only at the pass moment, inside the atomic promote transaction
(see ``algua.registry.promotion.run_gate`` / ``record_gate_with_fdr_and_maybe_promote``). Minting is
bounded by the ``AGENT_NOVEL_MINT_CAP`` per-window rate cap and re-checks a family-graph fingerprint
CAS under the write lock.
"""
from __future__ import annotations

import json

from algua.contracts.lifecycle import Actor
from algua.registry.approvals import compute_artifact_hashes
from algua.registry.lineage import factors_used_by
from algua.registry.repository import (
    ClassifyResult,
    FamilyGraphDriftError,
    PendingNovelFamily,
    StrategyRepository,
)
from algua.research.clustering import (
    _RETURN_STANDALONE_ESCALATION,
    MERGE_THRESHOLD,
    PARENTAGE_THRESHOLD,
    WEIGHT_CODE_ANCESTRY,
    WEIGHT_FACTOR_LINEAGE,
    WEIGHT_RETURN_CORRELATION,
    SimVerdict,
    clustering_version,
    family_similarity,
)
from algua.strategies.loader import StrategyNotFound


def get_all_family_members_for_clustering(
    repo: StrategyRepository,
) -> list[tuple[int, list[dict]]]:
    """Return [(family_id, members_list)] for all families with active members.

    Each member dict: {"code_hash": str, "factors": set[str]}.
    Delegates to the repository so the Protocol seam is respected.
    """
    return repo.all_families_with_member_profiles()


def classify_and_assign_family(
    repo: StrategyRepository,
    name: str,
    *,
    actor: Actor,
    new_family_slug: str | None,
) -> ClassifyResult:
    """Classify ``name`` against all known families and assign it if needed (#524, R9).

    Returns a ``ClassifyResult``: a resolved ``family_id`` (MERGE / PARENTAGE / human-NOVEL /
    already-assigned) OR a deferred ``pending_novel_family`` spec (agent NOVEL — NO family created
    here; it is minted at the pass moment inside the atomic promote tx).

    Decision tree:
    - Already assigned: return current family_id, no re-classification.
    - MERGE: assign to best-matching family.
    - PARENTAGE + agent: fold into best parent family (agents cannot mint child families).
    - PARENTAGE + human: create a child family with a parent edge, assign there.
    - NOVEL + agent: return a PendingNovelFamily spec (deferred create) with a graph-fingerprint
      snapshot captured before==after classification (fail closed on mid-classification drift).
    - NOVEL + human: create a new root family using new_family_slug (required, fresh 0 prior);
      assign.
    """
    # #524 (R9-H3): one-time-materialise any legacy family_members row still carrying a NULL
    # profile BEFORE the classifier reads member profiles, so the profile axis is DB state covered
    # by the graph fingerprint. Idempotent + cheap (steady state selects zero rows, early-returns);
    # a NULL→value UPDATE leaves family_members COUNT/MAX(id) unchanged, so the fingerprint captured
    # below is unaffected. Runs top-level (no open tx here) so the brief write lock is safe.
    repo.materialise_legacy_member_profiles()
    # #524 (R6-CRITICAL): capture the classifier read-set fingerprint at the TOP, before the graph
    # read below, so the stored snapshot provably equals the graph the NOVEL verdict is computed on.
    fp_before = repo.family_graph_fingerprint()
    current_family_id = repo.strategy_family(name)
    if current_family_id is not None:
        # Already assigned — skip reclassification, keep existing assignment.
        return ClassifyResult(family_id=current_family_id)

    # Get the strategy's identity for clustering comparison. A strategy whose module cannot be
    # loaded (e.g. test-only names) gets code_hash="" and factors=set(): it will never match an
    # existing member's real code_hash, so these strategies get NOVEL verdict (fail-closed).
    try:
        strategy_code_hash = compute_artifact_hashes(name).code_hash
    except StrategyNotFound:
        strategy_code_hash = ""
    try:
        factor_specs = factors_used_by(name)
        # factors_used_by returns list[FactorSpec]; get names.
        strategy_factors: set[str] = {
            f.name if hasattr(f, "name") else str(f) for f in factor_specs
        }
    except Exception:  # noqa: BLE001 — unregistered/test strategies silently get no factors
        strategy_factors = set()

    # Load family data once; also collect member names for return-correlation axis
    all_family_data = get_all_family_members_for_clustering(repo)
    all_member_names: list[str] = [
        m["name"]
        for _fid, members in all_family_data
        for m in members
        if "name" in m
    ]

    # Build returns_lookup from stored backtest returns so the correlation axis is live
    # whenever prior run() or sweep results have been persisted (#222, Task 7)
    returns_lookup: dict[str, object] = {}
    strategy_stored_returns = repo.load_backtest_returns(name)
    if strategy_stored_returns is not None:
        returns_lookup["__strategy__"] = strategy_stored_returns
    for member_name in all_member_names:
        if member_name not in returns_lookup:
            member_returns = repo.load_backtest_returns(member_name)
            if member_returns is not None:
                returns_lookup[member_name] = member_returns

    def _rank(escalate: bool) -> tuple[int | None, SimVerdict, float, bool]:
        """Select the best-matching family. ``escalate`` toggles the #338 return-correlation
        NOVEL-rescue. Returns (family_id, verdict, score, has_any_family)."""
        b_id: int | None = None
        b_verdict = SimVerdict.NOVEL
        b_score = 0.0
        any_family = False
        for fam_id, members in all_family_data:
            any_family = True
            verdict, score = family_similarity(
                strategy_code_hash, strategy_factors, members,
                returns_lookup=returns_lookup or None,
                escalate=escalate,
            )
            if score > b_score or (score == b_score and b_verdict == SimVerdict.NOVEL):
                b_score = score
                b_verdict = verdict
                b_id = fam_id
        return b_id, b_verdict, b_score, any_family

    # Forward-only family selection (#338): rank first on the PURE blend (escalate=False),
    # identical to the pre-#338 behaviour. Only if NO family matches on the blend (best is
    # NOVEL) do we re-rank WITH the standalone return-correlation escalation to rescue an
    # identical-trading clone out of NOVEL. This guarantees a return-only match can never
    # DISPLACE a code/factor (blend) match into a narrower-breadth family — escalation can
    # only pull a would-be-NOVEL strategy INTO a family (strictly tightening).
    best_family_id, best_verdict, best_score, _has_any_family = _rank(escalate=False)
    if best_verdict == SimVerdict.NOVEL:
        best_family_id, best_verdict, best_score, _has_any_family = _rank(escalate=True)

    cv = clustering_version()
    clustering_config_json = json.dumps({
        "version": cv,
        "merge_threshold": MERGE_THRESHOLD,
        "parentage_threshold": PARENTAGE_THRESHOLD,
        "weights": {
            "code_ancestry": WEIGHT_CODE_ANCESTRY,
            "factor_lineage": WEIGHT_FACTOR_LINEAGE,
            "return_correlation": WEIGHT_RETURN_CORRELATION,
        },
        "return_standalone_escalation": _RETURN_STANDALONE_ESCALATION,
    }, sort_keys=True)
    axis_json = json.dumps({
        "verdict": best_verdict.value,
        "score": best_score,
        "has_returns_data": bool(returns_lookup),
    }, sort_keys=True)

    # #524 (R9-H3): every new member's classified profile is persisted onto its family_members row.
    member_factors_json = json.dumps(sorted(strategy_factors))

    def _do_assign(target_family_id: int, *, matched_family_id: int | None = None) -> None:
        repo.assign_strategy_to_family(
            name, target_family_id, actor=actor.value,
            verdict=best_verdict.value, similarity_score=best_score,
            clustering_version=cv, clustering_config_json=clustering_config_json,
            axis_json=axis_json,
            matched_family_id=matched_family_id if matched_family_id is not None
            else target_family_id,
            member_code_hash=strategy_code_hash,
            member_factors_json=member_factors_json,
        )

    if best_verdict == SimVerdict.MERGE:
        assert best_family_id is not None
        _do_assign(best_family_id)
        return ClassifyResult(family_id=best_family_id)

    if best_verdict == SimVerdict.PARENTAGE:
        assert best_family_id is not None
        if actor is Actor.AGENT:
            # Agent cannot mint a child family. Fold into the best parent.
            _do_assign(best_family_id)
            return ClassifyResult(family_id=best_family_id)
        else:
            # Human: create a child family, add a parent edge, assign.
            child_name = new_family_slug or f"{name}_family"
            child_fam_id = repo.create_family(child_name, actor=actor.value,
                                               created_by_strategy=name)
            repo.add_parent_edge(child_fam_id, best_family_id)
            _do_assign(child_fam_id, matched_family_id=best_family_id)
            return ClassifyResult(family_id=child_fam_id)

    # NOVEL verdict
    if actor is Actor.AGENT:
        # #532 (Finding 1): `_has_any_family` (from the clustering rank over ACTIVE members) is
        # False in TWO distinct states: (a) a true cold-start — the `families` table is genuinely
        # empty; and (b) families exist but every one has zero active members (all removed).
        # Only (a) is the "nothing to game" case the issue permits an agent to found #0 in. In
        # state (b) real (if dead) sibling families are on the graph, so letting an agent silently
        # found a fresh near-zero-prior family alongside them is exactly the multiplicity evasion
        # #524 defends against — fail closed, human-only. `family_count()` is fingerprint component
        # 0 and is read here inside the fp_before/fp_after window, so this branch decision is
        # covered by the same CAS that guards the NOVEL verdict (a mint bumps the fingerprint).
        if not _has_any_family and repo.family_count() > 0:
            raise ValueError(
                f"strategy {name!r}: the family registry has {repo.family_count()} family(ies) but "
                "none with an active member. An agent cannot found a new family alongside existing "
                "(if dormant) families; a human must intervene via "
                "`research promote --actor human --new-family <slug>`."
            )
        # Cold-start (family_count()==0) OR the normal non-empty case both fall through to the #524
        # deferred PendingNovelFamily mint — no branch on emptiness in the seed or the mint.
        # #524 (R9): do NOT create here. Defer to the atomic pass-moment. The classification
        # snapshot is a single graph_fingerprint (member profiles are DB-persisted, so the whole
        # classifier read-set is one DB digest). It MUST equal the state the NOVEL verdict was
        # computed on: re-read fp_after and require fp_before == fp_after (R6-CRITICAL) — a mismatch
        # means the graph mutated DURING classification, so the verdict is already stale.
        fp_after = repo.family_graph_fingerprint()
        if fp_after != fp_before:
            raise FamilyGraphDriftError(
                f"family graph changed while {name!r} was being classified NOVEL; re-run promote",
                axis="graph_fingerprint")
        pending = PendingNovelFamily(
            slug_base=f"{name}_family", actor=actor.value, verdict=best_verdict.value,
            similarity_score=best_score, clustering_version=cv,
            clustering_config_json=clustering_config_json, axis_json=axis_json,
            graph_fingerprint=fp_after,
            founder_code_hash=strategy_code_hash,
            founder_factors_json=member_factors_json,
        )
        return ClassifyResult(family_id=None, pending_novel_family=pending)
    else:
        # Human: create a new root family using the provided slug (fresh zero-prior).
        if new_family_slug is None:
            raise ValueError(
                f"strategy {name!r}: clustering verdict is NOVEL (no matching family). "
                "Provide --new-family <slug> to create a new family."
            )
        new_fam_id = repo.create_family(new_family_slug, actor=actor.value,
                                         created_by_strategy=name)
        _do_assign(new_fam_id, matched_family_id=None)
        return ClassifyResult(family_id=new_fam_id)
