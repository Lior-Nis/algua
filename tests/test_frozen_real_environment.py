"""Story 1.3c contract §10: one opt-in test runs REAL prepare -> verify -> dispatch.

Everything below the planner inputs is production machinery, nothing is faked:

- Story 1.3b preparation through intake's own composition (`prepare_and_verify_frozen`): the
  repository's clean `HEAD` is qualified, `algua/` is exported from Git objects, the bundle is
  published, a REAL uv environment is provisioned from the committed lock with the production argv
  (its private HOME has no uv cache, so every locked wheel is downloaded), published and recorded,
  then the recorded descriptor is verified offline with the Story 1.3b verifier;
- admission (`intake_candidate_to_paper`) and tick-time resolution (`resolve_paper_tenant`, whose
  `FrozenContentVerifier` re-verifies the published bundle and environment and whose strict decoder
  reads the REAL strategy's recorded config);
- dispatch: `FrozenPlanner` launches the child with the provisioned environment's interpreter over
  the published bundle, for Phase A and for Phase B with a resolved (disabled) venue belief.

The frozen results must equal the in-process planner's on the checkout strategy, as canonical
encodings, and no invocation directory may survive.

The strategy is `cross_sectional_momentum`, committed at `HEAD` (the bundle comes from Git objects,
so an uncommitted fixture would not be in it). The checkout's tracked files must match `HEAD`: the
qualified identity and the in-process reference are computed from this checkout. Preparation reads
a private `git clone --shared` of this checkout at the same commit (the same Git objects), because
a checkout that has run the regular suite holds orphaned `__pycache__` bytecode of the suite's
temporary strategy modules, which the Story 1.3b clean-`HEAD` check refuses as untracked content
below `algua/` (surfaced as `frozen_source_drift`).

Opt in with `ALGUA_REAL_FROZEN_TESTS=1`: it needs uv, network access for the wheel download, about
2 GB of temporary disk (the uv download cache plus the ~1 GB environment) and a few minutes.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, date, datetime, time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from algua.calendar.market_calendar import MarketCalendar
from algua.contracts.lifecycle import Actor
from algua.data.store import DataStore
from algua.live.frozen_dispatch import FrozenPlanner, FrozenTarget
from algua.live.frozen_wire import WireIdentity
from algua.live.frozen_wire_result import encode_result
from algua.live.live_loop import planner_context_for_deployment
from algua.live.planner import InProcessPlanner
from algua.live.planner_contract import (
    BOUNDARY_VERSION,
    CapturedStrategyState,
    Decision,
    EarlyPlannerInput,
    LatePlannerInput,
    SnapshotRequired,
    VenueBeliefDisabled,
    VenueBeliefPending,
    VenueBeliefRequired,
)
from algua.registry.approvals import compute_artifact_hashes
from algua.registry.artifact_recording import parse_frozen_deployment_manifest
from algua.registry.artifact_store import resolve_locator
from algua.registry.db import connect, migrate
from algua.registry.frozen_runtime import (
    FrozenContentVerifier,
    FrozenTenant,
    resolve_paper_tenant,
)
from algua.registry.intake import prepare_and_verify_frozen
from algua.registry.store import SqliteStrategyRepository
from algua.strategies.loader import load_tradable_strategy
from tests._frozen_harness import CHECKOUT_ROOT, unseal

pytestmark = pytest.mark.skipif(
    os.environ.get("ALGUA_REAL_FROZEN_TESTS") != "1",
    reason="opt-in (ALGUA_REAL_FROZEN_TESTS=1): provisions a real ~1 GB uv environment from the "
    "committed lock, downloading every locked wheel; needs uv, network and a few minutes",
)

STRATEGY = "cross_sectional_momentum"  # CONFIG universe AAPL, MSFT, NVDA, AMZN, GOOGL; top 3
GATE_NAME = "real-frozen-gate"
GATE = ["AAPL", "AMZN", "MSFT", "NVDA"]  # the gate universe: a real overlay onto CONFIG's five
HELD = "GOOGL"  # held outside the gate universe: its bars are fetched and it is exited
DRIFT = {"AAPL": 0.001, "MSFT": 0.002, "NVDA": 0.004, "AMZN": -0.001, "GOOGL": 0.0005}
REQUEST_ID = "5eed" * 8
CALENDAR = "XNYS"
SESSIONS = MarketCalendar(CALENDAR).sessions_in_range(date(2024, 1, 2), date(2024, 6, 28))
NOW = datetime.combine(MarketCalendar(CALENDAR).next_session(SESSIONS[-1]), time(15), UTC)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True,
                          text=True).stdout.strip()


@pytest.fixture
def head_clone(tmp_path: Path) -> tuple[Path, str]:
    """A clean clone of this checkout at its exact ``HEAD``, and that commit's full OID."""
    assert _git(CHECKOUT_ROOT, "status", "--porcelain=v1", "--untracked-files=no") == "", (
        "tracked files must match HEAD: the qualified identity is computed from this checkout")
    head = _git(CHECKOUT_ROOT, "rev-parse", "--verify", "HEAD^{commit}")
    clone = tmp_path / "repo"
    subprocess.run(["git", "clone", "--quiet", "--shared", str(CHECKOUT_ROOT), str(clone)],
                   check=True, capture_output=True)
    assert _git(clone, "rev-parse", "--verify", "HEAD^{commit}") == head
    return clone, head


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Path]:
    """The trusted store root (the data dir); its sealed objects are unsealed and removed after."""
    root = tmp_path / "data"
    root.mkdir()
    try:
        yield root
    finally:
        unseal(root)  # published objects are sealed 0555/0444: restore directory write first
        shutil.rmtree(root)


def _qualified_candidate(repo: SqliteStrategyRepository) -> None:
    """A candidate whose newest passing, unanchored human gate binds the checkout's REAL identity
    and the gate universe, entered into ``candidate`` after that gate (1.3b qualification)."""
    identity = compute_artifact_hashes(STRATEGY)
    rec = repo.add(STRATEGY)
    gate_id = repo.record_gate_evaluation(
        rec.id, passed=True, n_funnel=1, own_lifetime_combos=1, windowed_total_combos=1,
        funnel_window_days=90, breadth_provenance="measured", pit_ok=True, pit_override=False,
        holdout_n_bars=63, min_holdout_observations=63, code_hash=identity.code_hash,
        config_hash=identity.config_hash, dependency_hash=identity.dependency_hash,
        data_source="test", snapshot_id="snap", period_start="2024-01-01",
        period_end="2024-12-31", holdout_frac=0.2, actor="human", decision_json="{}",
        universe_name=GATE_NAME)
    conn = repo.connection
    created_at = conn.execute(
        "SELECT created_at FROM gate_evaluations WHERE id=?", (gate_id,)).fetchone()[0]
    conn.execute("UPDATE strategies SET stage='candidate' WHERE id=?", (rec.id,))
    conn.execute(
        "INSERT INTO stage_transitions(strategy_id, from_stage, to_stage, actor, reason,"
        " code_hash, config_hash, dependency_hash, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (rec.id, "backtested", "candidate", Actor.HUMAN.value, "fixture", identity.code_hash,
         identity.config_hash, identity.dependency_hash, created_at))
    conn.commit()


def _bars() -> pd.DataFrame:
    """Deterministic daily bars at UTC midnight on real XNYS sessions: a distinct drift per
    symbol (NVDA > MSFT > AAPL > AMZN over any 60-bar window) plus a small bounded wobble."""
    steps = np.arange(len(SESSIONS), dtype="float64")
    rows = []
    for offset, (symbol, drift) in enumerate(DRIFT.items()):
        closes = 100.0 * (1.0 + drift) ** steps * (1.0 + 0.005 * np.sin(steps + offset))
        rows += [
            {"timestamp": datetime.combine(session, time(), UTC), "symbol": symbol,
             "open": close * 0.998, "high": close * 1.01, "low": close * 0.99, "close": close,
             "adj_close": close, "volume": 1_000_000.0 + offset}
            for session, close in zip(SESSIONS, closes.tolist(), strict=True)
        ]
    return pd.DataFrame(rows).sort_values(["timestamp", "symbol"]).set_index("timestamp")


def _canonical(phase: str, result: object) -> bytes:
    return encode_result(phase, REQUEST_ID, result)


def test_real_prepare_verify_dispatch_equals_the_in_process_planner(
    head_clone, store, tmp_path, monkeypatch,
):
    repo_root, head = head_clone
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "registry.db"))
    monkeypatch.setenv("ALGUA_DATA_DIR", str(store))
    conn = connect(tmp_path / "registry.db")
    migrate(conn)
    repo = SqliteStrategyRepository(conn)
    _qualified_candidate(repo)

    # --- prepare -> verify: Story 1.3b for real, exactly as paper intake composes it ------------
    admission = prepare_and_verify_frozen(repo, STRATEGY, repo_root=repo_root, store_root=store)
    manifest = parse_frozen_deployment_manifest(admission.manifest)
    assert manifest.source_ref == head
    bundle_root = resolve_locator(store.resolve(), manifest.bundle.locator,
                                  expected_digest=manifest.bundle.digest, kind="bundles")
    environment_root = resolve_locator(store.resolve(), manifest.environment.locator,
                                       expected_digest=manifest.environment.digest,
                                       kind="environments")
    assert (bundle_root / "algua/live/frozen_child.py").is_file()
    assert (environment_root / "bin/python").exists()
    assert not list(environment_root.glob("lib/python*/site-packages/algua*"))

    # --- admit and resolve the tenant the way a paper tick does ---------------------------------
    DataStore(store).ingest_universe(universe=GATE_NAME, symbols=GATE,
                                     effective_date="2020-01-01",
                                     as_of="2020-01-01T00:00:00Z", source="test")
    repo.intake_candidate_to_paper(
        repo.get(STRATEGY), 10_000.0, Actor.AGENT, 100_000.0, 5,
        deployment_manifest=admission.manifest, research_gate_id=admission.research_gate_id)
    tenant = resolve_paper_tenant(
        conn, STRATEGY, command="paper trade-tick", verifier=FrozenContentVerifier(store),
        data_dir=store, identity_loader=compute_artifact_hashes)
    assert isinstance(tenant, FrozenTenant)
    assert (tenant.bundle_root, tenant.environment_root) == (bundle_root, environment_root)
    assert list(tenant.view.universe) == GATE

    # --- the target and inputs, as the paper tick builds them -----------------------------------
    target = FrozenTarget(
        WireIdentity(tenant.name, tenant.deployment_id, tenant.artifact_id,
                     tenant.deployment.manifest_digest, manifest.bundle.digest,
                     manifest.environment.digest),
        tenant.bundle_root, tenant.environment_root, tenant.interpreter, tenant.view.execution,
        tuple(tenant.view.universe))
    context = planner_context_for_deployment(tenant.deployment, CALENDAR)
    assert context is not None
    frame = _bars()
    held_qty = 10.0
    early = EarlyPlannerInput(
        boundary_version=BOUNDARY_VERSION, request_id=REQUEST_ID, strategy_name=tenant.name,
        deployment_id=context.deployment_id, artifact_id=context.artifact_id,
        manifest_digest=context.manifest_digest, config_hash=context.config_hash,
        resolved_config_json=context.resolved_config_json, now=NOW, timeframe="1d",
        calendar_code=context.calendar_code, raw_bars=frame, early_positions={HELD: held_qty},
        gate_universe=tuple(tenant.view.universe), max_drawdown=0.1)
    checkout = load_tradable_strategy(STRATEGY)
    reference = InProcessPlanner(
        replace(checkout, config=checkout.config.model_copy(update={"universe": GATE})))
    invocations = store / "frozen/invocations"
    port = FrozenPlanner(target, invocations_root=invocations)

    # --- dispatch: Phase A, closed bars, Phase B through the real provisioned interpreter -------
    first = port.phase_a(early)
    assert _canonical("a", first) == _canonical("a", reference.phase_a(early))
    assert isinstance(first, SnapshotRequired) and not first.warming, first
    pd.testing.assert_frame_equal(port.closed_bars(early), reference.closed_bars(early))

    held_close = float(frame[frame.symbol == HELD]["close"].iloc[-1])
    captured = CapturedStrategyState(
        request_id=REQUEST_ID, sizing_equity=10_000.0, drawdown_equity=10_000.0,
        quantities={HELD: held_qty}, market_values={HELD: held_qty * held_close},
        persisted_peak_equity=10_000.0, venue_belief=VenueBeliefDisabled())
    late = LatePlannerInput(early, first.phase_a_binding, captured)
    pending = replace(late, captured=replace(captured, venue_belief=VenueBeliefPending()))
    assert port.phase_b(pending) == VenueBeliefRequired()

    second = port.phase_b(late)
    assert _canonical("b", second) == _canonical("b", reference.phase_b(late))
    assert isinstance(second, Decision), second
    sides = {intent.symbol: intent.side.value for intent in second.ordered_intents}
    assert sides[HELD] == "sell" and "buy" in sides.values(), sides
    assert list(invocations.iterdir()) == []
