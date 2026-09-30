"""Tests for `paper intake` (#317): deterministic candidate → paper book admission.

Three required behaviours:
  1. Empty book, headroom for all: every candidate is admitted, equal slice, `queued` empty; each
     admitted strategy is now Stage.PAPER with a non-None active allocation.
  2. The --max-concurrent count cap binds: exactly one candidate is admitted (FIFO, lower sid), the
     other stays queued and remains Stage.CANDIDATE with no allocation.
  3. An already-occupied slot counts against the cap: with the sole slot already taken by an
     allocated paper-lane strategy, the queued candidate is not admitted.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

import algua
import algua.registry.intake as intake_module
import algua.strategies.momentum as _momentum_pkg
from algua.cli.errors import error_code, is_retryable
from algua.cli.main import app
from algua.config.settings import get_settings
from algua.contracts.lifecycle import Actor, Stage
from algua.execution.alpaca_broker import AccountState
from algua.registry.allocations import active_allocation
from algua.registry.approvals import compute_artifact_hashes
from algua.registry.artifact_errors import (
    ArtifactNotFound,
    FrozenAssetsUnsupported,
    FrozenBundleCorrupt,
    FrozenDescriptorConflict,
    FrozenEnvironmentCorrupt,
    FrozenEnvironmentIncompatible,
    FrozenEnvironmentUnavailable,
    FrozenSourceDrift,
    FrozenSourceInvalid,
)
from algua.registry.artifact_preparation import FrozenPreparationResult
from algua.registry.artifact_recording import (
    frozen_deployment_manifest,
    parse_frozen_deployment_manifest,
)
from algua.registry.artifact_verification import FrozenVerificationResult
from algua.registry.db import connect, migrate
from algua.registry.deployment import DeploymentError
from algua.registry.intake import prepare_and_verify_frozen, run_intake
from algua.registry.store import SqliteStrategyRepository
from algua.strategies.loader import load_tradable_strategy
from tests._deployment_helpers import frozen_manifest

runner = CliRunner()

# _S1 is registered first → lower DB id → FIFO tie-break admits it before _S2.
_S1 = "cross_sectional_momentum"
_S2 = "liquid10_momentum"


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "p.db"))
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")


@pytest.fixture(autouse=True)
def frozen_calls(monkeypatch) -> list[tuple]:
    """Story 1.3b preparation and verification without Git, uv or published content.

    The fake preparer records exactly what the real one records -- a canonical ``frozen``
    descriptor for the strategy's current identity, bound to its newest qualifying gate through
    the REAL ledger (``record_frozen_artifact`` re-qualifies under its own short write
    transaction) -- and maps qualification drift to ``FrozenSourceDrift`` as the real one does.
    The fake verifier reads the descriptor back from the ledger by digest and parses it with the
    real 1.3b parser. Each call is logged so tests can pin order and the roots passed in."""
    calls: list[tuple] = []

    def prepare(repo, name, *, repo_root, store_root):
        calls.append(("prepare", name, repo_root, store_root))
        identity = compute_artifact_hashes(name)
        try:
            qualification = repo.qualify_frozen_candidate(
                name, code_hash=identity.code_hash, config_hash=identity.config_hash,
                dependency_hash=str(identity.dependency_hash))
        except DeploymentError as exc:
            raise FrozenSourceDrift() from exc
        frozen = frozen_manifest(
            code_hash=identity.code_hash, config_hash=identity.config_hash,
            dependency_hash=str(identity.dependency_hash),
            resolved_config=load_tradable_strategy(name).config.model_dump(mode="json"),
            universe_name=qualification.universe_name)
        artifact_id = repo.record_frozen_artifact(
            name, frozen_deployment_manifest(frozen),
            research_gate_id=qualification.research_gate_id)
        return FrozenPreparationResult(name, artifact_id, qualification.research_gate_id, frozen)

    def verify(repo, manifest_digest, *, store_root):
        calls.append(("verify", manifest_digest, store_root))
        record = repo.deployment_artifact_by_digest(manifest_digest)
        manifest = record.frozen_manifest()
        return FrozenVerificationResult(manifest.resolved_config["name"], record.id, manifest)

    monkeypatch.setattr("algua.registry.intake.prepare_frozen_artifact", prepare)
    monkeypatch.setattr("algua.registry.intake.verify_frozen_artifact", verify)
    return calls


@pytest.fixture(autouse=True)
def _second_strategy():
    """`_S2` is a second REAL, loadable, demo-backtestable strategy in the momentum family so
    `_index()` discovers it (mirrors tests/test_paper_run_all._second_strategy)."""
    p = Path(_momentum_pkg.__path__[0]) / f"{_S2}.py"
    p.write_text(
        '"""Second demo strategy for paper intake tests: trailing-return momentum, top-k."""\n'
        "from __future__ import annotations\n"
        "from typing import Any\n"
        "import pandas as pd\n"
        "from algua.contracts.types import ExecutionContract\n"
        "from algua.features.alphas import xs_trailing_return\n"
        "from algua.strategies.base import StrategyConfig\n"
        f"CONFIG = StrategyConfig(name={_S2!r},\n"
        "    universe=['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL'],\n"
        "    execution=ExecutionContract(rebalance_frequency='1d', decision_lag_bars=1),\n"
        "    params={'lookback': 60}, construction='top_k_equal_weight',\n"
        "    construction_params={'top_k': 3}, feature_lookback=60)\n"
        "def signal(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:\n"
        "    return xs_trailing_return(view, params)\n"
    )
    try:
        yield
    finally:
        p.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeBroker:
    """Minimal paper broker: `intake` reads only `.account().equity` (READ-ONLY, no trading)."""

    def __init__(self, equity: float) -> None:
        self._equity = equity

    def account(self) -> AccountState:
        return AccountState(equity=self._equity, cash=self._equity,
                            buying_power=self._equity, account_id="t")


def _to_candidate(name: str) -> None:
    """Register, seed a human gate, then bind the candidate episode to its exact identity."""
    assert runner.invoke(app, ["backtest", "run", name, "--demo", "--register",
                               "--start", "2022-01-01", "--end", "2023-12-31"]).exit_code == 0
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        repo = SqliteStrategyRepository(conn)
        rec = repo.get(name)
        identity = compute_artifact_hashes(name)
        repo.record_gate_evaluation(
            rec.id, passed=True, n_funnel=1, own_lifetime_combos=1,
            windowed_total_combos=1, funnel_window_days=90, breadth_provenance="measured",
            pit_ok=True, pit_override=False, holdout_n_bars=63,
            min_holdout_observations=63, code_hash=identity.code_hash,
            config_hash=identity.config_hash, dependency_hash=identity.dependency_hash,
            data_source="test", snapshot_id="snap", period_start="2022-01-01",
            period_end="2023-12-31", holdout_frac=0.2, actor="human",
            decision_json="{}", universe_name=None)
        repo.apply_transition(
            rec, Stage.CANDIDATE, Actor.HUMAN, "fixture gate promotion",
            code_hash=identity.code_hash, config_hash=identity.config_hash,
            dependency_hash=identity.dependency_hash)


def _prepared(conn, name: str):
    return prepare_and_verify_frozen(
        SqliteStrategyRepository(conn), name, repo_root=Path("unused"),
        store_root=Path("unused"))


def _force_stage(name: str, stage_value: str) -> None:
    """Force a strategy's lifecycle stage directly (bypasses the promote gate for test setup)."""
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        rec = SqliteStrategyRepository(conn).get(name)
        conn.execute("UPDATE strategies SET stage = ? WHERE id = ?", (stage_value, rec.id))
        conn.commit()


def _seed_allocation(name: str, capital: float = 10_000.0) -> None:
    """Insert a strategy_allocations row directly (no paper-allocate CLI dependency)."""
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        rec = SqliteStrategyRepository(conn).get(name)
        conn.execute(
            "INSERT INTO strategy_allocations(strategy_id, capital, effective_ts, actor) "
            "VALUES (?,?,?,?)",
            (rec.id, capital, datetime.now(UTC).isoformat(), "agent"),
        )
        conn.commit()


def _stage_of(name: str):
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        return SqliteStrategyRepository(conn).get(name).stage


def _has_allocation(name: str) -> bool:
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        rec = SqliteStrategyRepository(conn).get(name)
        return active_allocation(conn, rec.id) is not None


# ---------------------------------------------------------------------------
# Test 1: empty book, headroom for all candidates
# ---------------------------------------------------------------------------

def test_intake_admits_all_candidates_when_book_empty(monkeypatch):
    """Two candidates, empty book, cap 5, equity 100k → BOTH admitted with an equal 20k slice,
    `queued` empty; both are now Stage.PAPER and each carries an active allocation."""
    _to_candidate(_S1)
    _to_candidate(_S2)

    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings",
                        lambda: _FakeBroker(100_000.0))

    result = runner.invoke(app, ["paper", "intake", "--max-concurrent", "5"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload.get("ok") is True

    admitted = {a["strategy"]: a["capital"] for a in payload["admitted"]}
    assert admitted == {_S1: 20_000.0, _S2: 20_000.0}
    assert payload["queued"] == []
    assert payload["slice"] == 20_000.0
    assert payload["occupied_before"] == 0
    assert payload["equity"] == 100_000.0

    from algua.contracts.lifecycle import Stage
    for name in (_S1, _S2):
        assert _stage_of(name) is Stage.PAPER
        assert _has_allocation(name)


# ---------------------------------------------------------------------------
# Test 2: the concurrency cap binds — only the FIFO-first candidate is admitted
# ---------------------------------------------------------------------------

def test_intake_cap_admits_only_first_candidate(monkeypatch):
    """cap 1, two candidates → exactly ONE admitted (the earlier-registered / lower-sid _S1); the
    other stays queued AND remains Stage.CANDIDATE with no allocation."""
    _to_candidate(_S1)
    _to_candidate(_S2)

    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings",
                        lambda: _FakeBroker(100_000.0))

    result = runner.invoke(app, ["paper", "intake", "--max-concurrent", "1"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)

    assert [a["strategy"] for a in payload["admitted"]] == [_S1]
    assert payload["queued"] == [_S2]

    from algua.contracts.lifecycle import Stage
    assert _stage_of(_S1) is Stage.PAPER
    assert _has_allocation(_S1)
    # The un-admitted candidate is untouched: still candidate, still unallocated.
    assert _stage_of(_S2) is Stage.CANDIDATE
    assert not _has_allocation(_S2)


# ---------------------------------------------------------------------------
# Test 3: an already-occupied slot counts against the cap
# ---------------------------------------------------------------------------

def test_intake_occupied_slot_blocks_admission(monkeypatch):
    """The sole slot (cap 1) is already taken by an allocated paper-lane strategy → the queued
    candidate is NOT admitted; it stays Stage.CANDIDATE with no allocation."""
    # _S2 is an already-admitted paper-lane tenant (forced stage + a seeded allocation row).
    _to_candidate(_S2)
    _force_stage(_S2, "paper")
    _seed_allocation(_S2)
    # _S1 is the queued candidate.
    _to_candidate(_S1)

    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings",
                        lambda: _FakeBroker(100_000.0))

    result = runner.invoke(app, ["paper", "intake", "--max-concurrent", "1"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)

    assert payload["admitted"] == []
    assert payload["queued"] == [_S1]
    assert payload["occupied_before"] == 1

    from algua.contracts.lifecycle import Stage
    assert _stage_of(_S1) is Stage.CANDIDATE
    assert not _has_allocation(_S1)


# ---------------------------------------------------------------------------
# The atomic admit primitive directly (findings #1/#3/#4)
# ---------------------------------------------------------------------------

def test_intake_reports_empty_stale_bucket_on_clean_run(monkeypatch):
    """`skipped_stale` is always present in the envelope (empty on a race-free run)."""
    _to_candidate(_S1)
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings",
                        lambda: _FakeBroker(100_000.0))
    payload = json.loads(runner.invoke(app, ["paper", "intake"]).output)
    assert payload["skipped_stale"] == []
    assert [a["strategy"] for a in payload["admitted"]] == [_S1]


def test_primitive_rejects_non_candidate():
    """`intake_candidate_to_paper` fails closed (TransitionError) on a non-candidate stage — the
    'stale selection' signal the intake loop treats as skipped_stale."""
    from algua.contracts.lifecycle import Actor, TransitionError
    _to_candidate(_S1)
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        prepared = _prepared(conn, _S1)  # verified while still a candidate
    _force_stage(_S1, "paper")  # then raced out of candidate before the admit
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        repo = SqliteStrategyRepository(conn)
        with pytest.raises(TransitionError):
            repo.intake_candidate_to_paper(
                repo.get(_S1), capital=10_000.0, actor=Actor.AGENT,
                account_equity=100_000.0, max_concurrent=5,
                deployment_manifest=prepared.manifest,
                research_gate_id=prepared.research_gate_id)


def test_primitive_count_cap_is_atomic_and_rolls_back():
    """At the count cap the primitive raises CountCapReached and leaves the candidate exactly
    candidate with NO allocation (the allocation insert is rolled back with the failed txn)."""
    from algua.contracts.lifecycle import Actor, Stage
    from algua.registry.allocations import CountCapReached
    # _S2 occupies the sole slot (allocated paper tenant); _S1 is the queued candidate.
    _to_candidate(_S2)
    _force_stage(_S2, "paper")
    _seed_allocation(_S2)
    _to_candidate(_S1)
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        repo = SqliteStrategyRepository(conn)
        prepared = _prepared(conn, _S1)
        with pytest.raises(CountCapReached):
            repo.intake_candidate_to_paper(
                repo.get(_S1), capital=10_000.0, actor=Actor.AGENT,
                account_equity=100_000.0, max_concurrent=1,
                deployment_manifest=prepared.manifest,
                research_gate_id=prepared.research_gate_id)
        assert repo.get(_S1).stage is Stage.CANDIDATE
        assert active_allocation(conn, repo.get(_S1).id) is None


def test_primitive_capital_bound_rolls_back():
    """When the slice would breach Σ ≤ equity the primitive raises AllocationError and leaves the
    strategy candidate + unallocated (atomic rollback of the whole admit)."""
    from algua.contracts.lifecycle import Actor, Stage
    from algua.registry.allocations import AllocationError
    _to_candidate(_S1)
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        repo = SqliteStrategyRepository(conn)
        prepared = _prepared(conn, _S1)
        with pytest.raises(AllocationError):
            repo.intake_candidate_to_paper(
                repo.get(_S1), capital=200_000.0, actor=Actor.AGENT,
                account_equity=100_000.0, max_concurrent=5,
                deployment_manifest=prepared.manifest,
                research_gate_id=prepared.research_gate_id)
        assert repo.get(_S1).stage is Stage.CANDIDATE
        assert active_allocation(conn, repo.get(_S1).id) is None


# ---------------------------------------------------------------------------
# `paper allocate` (#497): the lane-scoped re-admission path for recovery/demotion
# re-entrants (dormant→paper, live→paper land UNALLOCATED) and manual paper-book resizes.
# ---------------------------------------------------------------------------

def _seed_paper_allocation(name: str, capital: float = 10_000.0) -> None:
    """Force `name` to stage paper and give it an active allocation via the shared
    ``allocate_locked`` body (caller-owns-txn) wrapped in a `with conn:` commit — so it counts as
    an active paper-lane tenant against the concurrency cap."""
    from algua.registry.allocations import allocate_locked
    _force_stage(name, "paper")
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        rec = SqliteStrategyRepository(conn).get(name)
        with conn:
            allocate_locked(conn, rec.id, capital, "agent", 100_000.0)


def _capital_of(name: str) -> float:
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        rec = SqliteStrategyRepository(conn).get(name)
        row = active_allocation(conn, rec.id)
        assert row is not None
        return float(row["capital"])


def test_paper_allocate_sets_then_resizes_paper_stage(monkeypatch):
    """`paper allocate` on a paper-stage, unallocated strategy sets the capital base and emits
    prior_capital == 0.0; a second allocate RESIZES it and emits the prior capital."""
    _to_candidate(_S1)
    _force_stage(_S1, "paper")
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings",
                        lambda: _FakeBroker(100_000.0))

    r1 = runner.invoke(app, ["paper", "allocate", _S1, "--capital", "10000"])
    assert r1.exit_code == 0, r1.output
    p1 = json.loads(r1.output)
    assert p1["ok"] is True
    assert p1["strategy"] == _S1
    assert p1["capital"] == 10_000.0
    assert p1["prior_capital"] == 0.0
    assert _has_allocation(_S1)
    assert _capital_of(_S1) == 10_000.0

    r2 = runner.invoke(app, ["paper", "allocate", _S1, "--capital", "20000"])
    assert r2.exit_code == 0, r2.output
    p2 = json.loads(r2.output)
    assert p2["capital"] == 20_000.0
    assert p2["prior_capital"] == 10_000.0
    assert _capital_of(_S1) == 20_000.0


def test_paper_allocate_rejected_on_candidate_stage(monkeypatch):
    """A candidate-stage strategy has no paper re-admission path here — it enters only via
    `paper intake`. `paper allocate` fails closed (exit != 0), message names the stage."""
    _to_candidate(_S1)  # stays candidate
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings",
                        lambda: _FakeBroker(100_000.0))
    result = runner.invoke(app, ["paper", "allocate", _S1, "--capital", "10000"])
    assert result.exit_code != 0
    payload = json.loads(result.output)
    assert payload["ok"] is False
    assert "candidate" in payload["error"]
    assert not _has_allocation(_S1)


def test_paper_allocate_rejected_on_live_stage(monkeypatch):
    """A live-stage strategy is out of the paper lane — `live allocate` owns it. `paper allocate`
    fails closed (exit != 0), message names the live stage."""
    _to_candidate(_S1)
    _force_stage(_S1, "live")
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings",
                        lambda: _FakeBroker(100_000.0))
    result = runner.invoke(app, ["paper", "allocate", _S1, "--capital", "10000"])
    assert result.exit_code != 0
    payload = json.loads(result.output)
    assert payload["ok"] is False
    assert "live" in payload["error"]
    assert not _has_allocation(_S1)


def test_paper_allocate_count_cap_blocks_new_tenant_but_allows_resize(monkeypatch):
    """At the max-concurrent cap, a count-INCREASING allocation (a currently-unallocated paper
    strategy) is refused (CountCapReached surfaced, exit != 0), while RESIZING an already-allocated
    tenant at the cap succeeds (it admits no new tenant)."""
    # _S2 occupies the sole slot as an allocated paper tenant; _S1 is a paper strategy with no
    # active allocation yet.
    _to_candidate(_S2)
    _seed_paper_allocation(_S2, capital=10_000.0)
    _to_candidate(_S1)
    _force_stage(_S1, "paper")
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings",
                        lambda: _FakeBroker(100_000.0))

    blocked = runner.invoke(app, ["paper", "allocate", _S1, "--capital", "10000",
                                  "--max-concurrent", "1"])
    assert blocked.exit_code != 0
    blocked_payload = json.loads(blocked.output)
    assert blocked_payload["ok"] is False
    assert "capacity" in blocked_payload["error"]
    assert not _has_allocation(_S1)

    # Resizing the already-allocated tenant at the same cap succeeds (no new tenant admitted).
    resized = runner.invoke(app, ["paper", "allocate", _S2, "--capital", "20000",
                                  "--max-concurrent", "1"])
    assert resized.exit_code == 0, resized.output
    assert json.loads(resized.output)["prior_capital"] == 10_000.0
    assert _capital_of(_S2) == 20_000.0


# ---------------------------------------------------------------------------
# Re-admission: an unallocated BOOK-STAGE tenant is a broken invariant, not a queue member.
#
# `paper -> dormant` (and `live -> paper`) atomically REVOKES the allocation, but coming back
# `dormant -> paper` restores only the stage. Before this, the strategy sat at stage `paper`
# holding no book slice: `paper run-all` skipped it as unallocated, it never ticked, and
# `fleet health` alerted on it forever with nothing in the autonomous loop able to fix it.
# Observed on main 2026-08-17: liquidity_stable_quality_momentum, benched 08-13, returned 08-14,
# never ticked once.
# ---------------------------------------------------------------------------

def _bench_and_return(name: str) -> None:
    """Take an allocated paper tenant out to `dormant` (revoking its slice) and back to `paper`,
    the exact round-trip that leaves a book-stage strategy holding no allocation."""
    out = runner.invoke(app, ["registry", "transition", name, "--to", "dormant",
                              "--actor", "agent", "--reason", "benched for the test"])
    assert out.exit_code == 0, out.output
    back = runner.invoke(app, ["registry", "transition", name, "--to", "paper",
                               "--actor", "agent", "--reason", "returning from bench"])
    assert back.exit_code == 0, back.output


def test_intake_readmits_an_unallocated_paper_tenant(monkeypatch):
    """A strategy returned from `dormant` sits at stage paper with no slice. The next intake must
    re-admit it — otherwise it can never trade again without a human running `paper allocate`."""
    _to_candidate(_S1)
    _seed_paper_allocation(_S1, capital=10_000.0)
    _bench_and_return(_S1)
    assert _stage_of(_S1).value == "paper"
    assert not _has_allocation(_S1)

    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings",
                        lambda: _FakeBroker(100_000.0))
    result = runner.invoke(app, ["paper", "intake", "--max-concurrent", "5"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)

    # Reported separately from `admitted`: no stage changed, a book slice was restored.
    assert [r["strategy"] for r in payload["readmitted"]] == [_S1]
    assert payload["readmitted"][0]["capital"] == 20_000.0
    assert payload["admitted"] == []
    assert _has_allocation(_S1) and _capital_of(_S1) == 20_000.0


def test_readmission_precedes_a_fresh_candidate_when_the_cap_binds(monkeypatch):
    """At a cap of 1, the returning book tenant wins the slot: it was admitted to the book before
    the candidate existed, and a strategy the operator explicitly returned to `paper` must not be
    starved by a newcomer."""
    _to_candidate(_S1)
    _seed_paper_allocation(_S1, capital=10_000.0)
    _bench_and_return(_S1)
    _to_candidate(_S2)

    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings",
                        lambda: _FakeBroker(100_000.0))
    result = runner.invoke(app, ["paper", "intake", "--max-concurrent", "1"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)

    assert [r["strategy"] for r in payload["readmitted"]] == [_S1]
    assert payload["admitted"] == []
    assert payload["queued"] == [_S2]
    assert _has_allocation(_S1)
    assert _stage_of(_S2).value == "candidate" and not _has_allocation(_S2)


def test_readmission_is_refused_when_the_book_is_already_full(monkeypatch):
    """A full book binds re-admission exactly as it binds admission — the returning tenant is
    reported queued, never funded past the count cap."""
    _to_candidate(_S2)
    _seed_paper_allocation(_S2, capital=10_000.0)      # occupies the sole slot
    _to_candidate(_S1)
    _seed_paper_allocation(_S1, capital=10_000.0)
    _bench_and_return(_S1)

    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings",
                        lambda: _FakeBroker(100_000.0))
    result = runner.invoke(app, ["paper", "intake", "--max-concurrent", "1"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)

    assert payload["readmitted"] == []
    assert payload["queued"] == [_S1]
    assert not _has_allocation(_S1)


def test_intake_never_resizes_an_already_allocated_tenant(monkeypatch):
    """Re-admission targets ONLY the unallocated. An existing tenant's capital base is left exactly
    as the operator set it — intake is not a rebalancer."""
    _to_candidate(_S1)
    _seed_paper_allocation(_S1, capital=10_000.0)

    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings",
                        lambda: _FakeBroker(100_000.0))
    result = runner.invoke(app, ["paper", "intake", "--max-concurrent", "5"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["readmitted"] == []
    assert _capital_of(_S1) == 10_000.0


# ---------------------------------------------------------------------------
# Story 1.3c: every new admission is FROZEN — Story 1.3b preparation for the current clean HEAD,
# offline verification of the recorded descriptor, then the atomic admit bound to that exact row.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(algua.__file__).resolve().parents[1]


def _snapshot(name: str) -> dict:
    """Everything a refused admission must leave untouched for ``name``."""
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        repo = SqliteStrategyRepository(conn)
        rec = repo.get(name)
        return {
            "stage": rec.stage,
            "allocation": active_allocation(conn, rec.id) is not None,
            "deployment": repo.active_deployment(rec.id),
            "epochs": conn.execute(
                "SELECT COUNT(*) FROM strategy_deployments WHERE strategy_id=?",
                (rec.id,)).fetchone()[0],
            "transitions": len(repo.list_transitions(name)),
        }


def _assert_untouched(name: str, before: dict) -> None:
    after = _snapshot(name)
    assert after == before
    assert after["stage"] is Stage.CANDIDATE
    assert after["allocation"] is False and after["deployment"] is None and after["epochs"] == 0


def _run(step=None, **roots) -> dict:
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        return run_intake(conn, equity=100_000.0, max_concurrent=5, actor=Actor.AGENT,
                          prepare_and_verify=step, **roots)


class _FailingFor:
    """An injected prepare-and-verify step that raises ``exc`` for ``target`` and otherwise runs
    the real 1.3b composition (over the Git/uv-free fakes), logging every candidate it sees."""

    def __init__(self, target: str, exc: BaseException) -> None:
        self.target, self.exc, self.seen = target, exc, []

    def __call__(self, repo, name, *, repo_root, store_root):
        self.seen.append(name)
        if name == self.target:
            raise self.exc
        return prepare_and_verify_frozen(repo, name, repo_root=repo_root, store_root=store_root)


def test_intake_admits_each_candidate_against_its_verified_frozen_descriptor(
    monkeypatch, frozen_calls,
):
    """The CLI path wires the production step and roots: prepare (clean HEAD of THIS checkout,
    trusted store = settings.data_dir), then verify THAT recorded digest, then admit. The epoch is
    bound to the exact 1.3b row (no second artifact row), is ``source_kind='frozen'`` and parses
    with the 1.3b parser."""
    _to_candidate(_S1)
    _to_candidate(_S2)
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings",
                        lambda: _FakeBroker(100_000.0))

    result = runner.invoke(app, ["paper", "intake", "--max-concurrent", "5"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert [a["strategy"] for a in payload["admitted"]] == [_S1, _S2]
    assert payload["refused"] == []

    store = get_settings().data_dir
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        repo = SqliteStrategyRepository(conn)
        digests = []
        for name in (_S1, _S2):
            rec = repo.get(name)
            deployment = repo.active_deployment(rec.id)
            assert deployment is not None and rec.stage is Stage.PAPER
            assert active_allocation(conn, rec.id) is not None
            assert deployment.source_kind == "frozen"
            recorded = repo.deployment_artifact_by_digest(deployment.manifest_digest)
            assert recorded.id == deployment.artifact_id
            assert recorded.manifest() == deployment.manifest()
            frozen = parse_frozen_deployment_manifest(deployment.manifest())
            assert frozen.resolved_config["name"] == name
            identity = compute_artifact_hashes(name)
            assert (deployment.code_hash, deployment.config_hash, deployment.dependency_hash) == (
                identity.code_hash, identity.config_hash, identity.dependency_hash)
            digests.append(deployment.manifest_digest)
        assert conn.execute("SELECT COUNT(*) FROM deployment_artifacts").fetchone()[0] == 2
    assert frozen_calls == [
        ("prepare", _S1, _REPO_ROOT, store), ("verify", digests[0], store),
        ("prepare", _S2, _REPO_ROOT, store), ("verify", digests[1], store),
    ]


def test_run_intake_passes_the_injected_roots_to_the_injected_step(tmp_path):
    _to_candidate(_S1)
    seen: list[tuple] = []

    def step(repo, name, *, repo_root, store_root):
        seen.append((name, repo_root, store_root))
        return prepare_and_verify_frozen(repo, name, repo_root=repo_root, store_root=store_root)

    payload = _run(step, repo_root=tmp_path / "checkout", store_root=tmp_path / "store")
    assert [a["strategy"] for a in payload["admitted"]] == [_S1]
    assert seen == [(_S1, tmp_path / "checkout", tmp_path / "store")]


def test_environment_unavailable_refuses_stops_and_leaves_the_rest_queued():
    """The one retryable 1.3b refusal: nothing is admitted for this candidate, admission STOPS, the
    rest stay queued (never prepared) — and the next intake re-prepares and admits it."""
    _to_candidate(_S1)
    _to_candidate(_S2)
    before = {name: _snapshot(name) for name in (_S1, _S2)}
    step = _FailingFor(_S1, FrozenEnvironmentUnavailable())

    payload = _run(step)

    assert payload["refused"] == [{"strategy": _S1, "code": "frozen_environment_unavailable"}]
    assert is_retryable(payload["refused"][0]["code"])
    assert payload["admitted"] == [] and payload["queued"] == [_S2]
    assert step.seen == [_S1]  # the queue behind it was never prepared
    for name in (_S1, _S2):
        _assert_untouched(name, before[name])

    retry = _run()
    assert [a["strategy"] for a in retry["admitted"]] == [_S1, _S2]
    assert retry["refused"] == []


@pytest.mark.parametrize("exc", [
    FrozenSourceInvalid(), FrozenSourceDrift(), FrozenAssetsUnsupported(),
    FrozenBundleCorrupt(), FrozenEnvironmentIncompatible(), FrozenEnvironmentCorrupt(),
    FrozenDescriptorConflict(), ArtifactNotFound(),
], ids=lambda exc: type(exc).__name__)
def test_non_retryable_refusal_skips_only_that_candidate(exc):
    """Every other 1.3b preparation/verification refusal: this candidate is not admitted and keeps
    its exact state; the loop CONTINUES and the next candidate is admitted. The reported code is
    the stable 1.3b code the CLI error registry assigns."""
    _to_candidate(_S1)
    _to_candidate(_S2)
    before = _snapshot(_S1)

    payload = _run(_FailingFor(_S1, exc))

    assert payload["refused"] == [{"strategy": _S1, "code": error_code(exc)}]
    assert not is_retryable(error_code(exc))
    assert [a["strategy"] for a in payload["admitted"]] == [_S2]
    assert payload["queued"] == []
    _assert_untouched(_S1, before)
    assert _stage_of(_S2) is Stage.PAPER and _has_allocation(_S2)


def test_verification_failure_refuses_a_recorded_but_unverified_descriptor(
    monkeypatch, frozen_calls,
):
    """Preparation may record the descriptor row (append-only, 1.3b), but a failed OFFLINE
    verification refuses admission: no epoch, allocation or stage change for that candidate."""
    _to_candidate(_S1)
    _to_candidate(_S2)
    before = _snapshot(_S1)
    real_verify = intake_module.verify_frozen_artifact  # the fixture's ledger-backed fake

    def verify(repo, manifest_digest, *, store_root):
        result = real_verify(repo, manifest_digest, store_root=store_root)
        if result.strategy == _S1:
            raise FrozenBundleCorrupt()
        return result

    monkeypatch.setattr("algua.registry.intake.verify_frozen_artifact", verify)
    payload = _run()

    assert payload["refused"] == [{"strategy": _S1, "code": "frozen_bundle_corrupt"}]
    assert [a["strategy"] for a in payload["admitted"]] == [_S2]
    _assert_untouched(_S1, before)
    with closing(connect(get_settings().db_path)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM deployment_artifacts").fetchone()[0] == 2
    assert [c[0] for c in frozen_calls] == ["prepare", "verify", "prepare", "verify"]


@pytest.mark.parametrize("field", ["strategy", "artifact_id", "manifest"])
def test_verification_must_confirm_exactly_the_prepared_descriptor(monkeypatch, field):
    """The verifier's answer must name the same strategy, artifact row and descriptor preparation
    recorded; anything else is a descriptor conflict, never an admission."""
    _to_candidate(_S1)
    before = _snapshot(_S1)
    real_verify = intake_module.verify_frozen_artifact  # the fixture's ledger-backed fake

    def verify(repo, manifest_digest, *, store_root):
        result = real_verify(repo, manifest_digest, store_root=store_root)
        other = {
            "strategy": "someone_else", "artifact_id": result.artifact_id + 1,
            "manifest": frozen_manifest(
                code_hash="b" * 32, config_hash="c" * 32,
                dependency_hash=result.manifest.dependency_hash,
                resolved_config={"name": _S1}, universe_name=None),
        }
        return FrozenVerificationResult(**{
            "strategy": result.strategy, "artifact_id": result.artifact_id,
            "manifest": result.manifest, field: other[field]})

    monkeypatch.setattr("algua.registry.intake.verify_frozen_artifact", verify)
    payload = _run()

    assert payload["refused"] == [{"strategy": _S1, "code": "frozen_descriptor_conflict"}]
    assert payload["admitted"] == []
    _assert_untouched(_S1, before)


@pytest.mark.parametrize("exc", [
    sqlite3.OperationalError("database is locked"), KeyboardInterrupt(), SystemExit(3),
    RuntimeError("unexpected preparation bug"),
], ids=lambda exc: type(exc).__name__)
def test_systemic_and_unexpected_errors_propagate_without_admitting(exc):
    """Only 1.3b's typed refusals are per-candidate outcomes. SQLite faults, interrupts, exits and
    untyped bugs abort the intake — never recorded as a refusal, never admitting anything."""
    _to_candidate(_S1)
    _to_candidate(_S2)
    before = {name: _snapshot(name) for name in (_S1, _S2)}

    with pytest.raises(type(exc)):
        _run(_FailingFor(_S1, exc))
    for name in (_S1, _S2):
        _assert_untouched(name, before[name])
