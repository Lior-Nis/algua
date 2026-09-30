"""Story 1.3d §6 (AC7): a recorded frozen planner attempt replays to its recorded result bytes.

The attempts are recorded once per module, as production records them: a candidate admitted as a
frozen deployment through ``run_intake`` against a REALLY published bundle (this checkout's
``algua/`` plus the harness fixture strategy in its own family and 1.3b's generated ``_algua/``),
resolved by ``resolve_paper_tenant``, and ticked by the real ``run_tick`` through a real
``FrozenPlanner`` whose ``record`` is ``record_frozen_invocation`` on a migrated v48 registry, over
a real bars snapshot served by ``StoreBackedProvider``. The tick decides, so it records one Phase A
(``snapshot_required``) and one Phase B (``decision``) attempt, each from a real child.

Injected, exactly as in tests/_frozen_paper_world.py: the candidate's 1.3b preparation (Git/uv
export; the recorded descriptor is real) and the environment's offline verification, replaced by
the locator check (a real inventory of the development site-packages is not a unit-test cost).
The bundle is verified for real.

The snapshot holds rows the tick must not fetch, so the replay's symbol set and half-open window
matter: CCC (the strategy's CONFIG universe, outside the gate universe), ZZZ (held at quantity
zero), a bar before the window's start and one at its exclusive end.
"""

from __future__ import annotations

import errno
import hashlib
import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from algua.calendar.factory import get_calendar
from algua.contracts.lifecycle import Actor
from algua.data.serve import StoreBackedProvider
from algua.data.store import DataStore
from algua.execution.alpaca_broker import TickSnapshot
from algua.live.frozen_dispatch import FrozenPlanner, FrozenTarget
from algua.live.frozen_wire import BOOTSTRAP, WireIdentity
from algua.live.frozen_wire_arrow import decode_bars, encode_bars
from algua.live.live_loop import TickHooks, planner_context_for_deployment, run_tick
from algua.live.planner_binding import bars_digest
from algua.registry.artifact_contract import BundleDescriptor
from algua.registry.artifact_preparation import _bundle_files
from algua.registry.artifact_recording import frozen_deployment_manifest
from algua.registry.artifact_store import publish_bundle
from algua.registry.db import connect, migrate
from algua.registry.frozen_manifest_contract import FrozenManifest
from algua.registry.frozen_runtime import (
    FrozenContentVerifier,
    FrozenTenant,
    resolve_paper_tenant,
)
from algua.registry.frozen_source import FrozenFile
from algua.registry.frozen_tenant_errors import FrozenContentUnavailable
from algua.registry.intake import FrozenAdmission, run_intake
from algua.registry.store import SqliteStrategyRepository
from algua.registry.store.frozen_evidence import frozen_invocation, record_frozen_invocation
from algua.strategies.base import config_hash
from tests._frozen_harness import (
    CHECKOUT_ALGUA,
    CHECKOUT_ROOT,
    NOW,
    PROTOCOL,
    STAMP,
    build_environment,
    in_process,
    recorded,
    seal,
    strategy_source,
    unseal,
)
from tests._frozen_paper_world import (
    CODE_HASH,
    DEP,
    environment_descriptor,
    verify_environment_locator,
)
from tests._frozen_replay import ReplayMismatch, prepare_replay, replay_attempt

TENANT = "frozen_replay_tenant"  # its own name: the 1.3c e2e world owns frozen_paper_tenant
FAMILY = "frozen_replay_family"
GATE_NAME = "frozen-replay-gate"
GATE = ["AAA", "BBB"]  # the gate universe; the strategy's CONFIG universe adds CCC
RECORDED = recorded(in_process(TENANT))
CONFIG_HASH = config_hash(in_process(TENANT))
CHECKOUT_MODULE = CHECKOUT_ALGUA / "strategies" / "momentum" / f"{TENANT}.py"
CHECKOUT_DOTTED = f"algua.strategies.momentum.{TENANT}"
RAISING_MODULE = 'raise RuntimeError("a replay must never import the checkout module")\n'
BUNDLE_MODULE = f"algua/strategies/{FAMILY}/{TENANT}.py"
DAY = timedelta(days=1)
START, END = NOW - 10 * DAY, NOW  # the half-open [start, end) the tick fetches
HELD = {"OLD": 2.0, "ZZZ": 0.0}  # ZZZ is held at quantity zero: never fetched
SNAPSHOT = TickSnapshot(equity=100.0, market_values={"AAA": 0.0, "BBB": 0.0, "OLD": 10.0},
                        qtys={"AAA": 0.0, "BBB": 0.0, "OLD": 2.0})


# --- content and admission ---------------------------------------------------------------------


def _tenant_bundle_files() -> tuple[FrozenFile, ...]:
    """This checkout's ``algua/`` with the tenant in its own family, plus 1.3b's ``_algua/``."""
    source = [
        FrozenFile(f"algua/{path.relative_to(CHECKOUT_ALGUA).as_posix()}", "100644",
                   path.read_bytes())
        for path in sorted(CHECKOUT_ALGUA.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts and path.name != f"{TENANT}.py"
    ]
    source += [
        FrozenFile(f"algua/strategies/{FAMILY}/__init__.py", "100644", b""),
        FrozenFile(BUNDLE_MODULE, "100644", strategy_source(TENANT).encode()),
    ]
    files = [item for item in _bundle_files(tuple(source), RECORDED) if item.path != PROTOCOL]
    files.append(FrozenFile(PROTOCOL, "100644", STAMP))
    return tuple(sorted(files, key=lambda item: item.path.encode()))


def _add_frozen_candidate(conn) -> None:
    """A qualified candidate whose newest passing gate binds the tenant's identity and gate
    universe, entered into ``candidate`` after that gate (the 1.3b qualification rule)."""
    repo = SqliteStrategyRepository(conn)
    rec = repo.add(TENANT)
    gate_id = repo.record_gate_evaluation(
        rec.id, passed=True, n_funnel=1, own_lifetime_combos=1, windowed_total_combos=1,
        funnel_window_days=90, breadth_provenance="measured", pit_ok=True, pit_override=False,
        holdout_n_bars=63, min_holdout_observations=63, code_hash=CODE_HASH,
        config_hash=CONFIG_HASH, dependency_hash=DEP, data_source="test", snapshot_id="snap",
        period_start="2024-01-01", period_end="2024-12-31", holdout_frac=0.2, actor="human",
        decision_json="{}", universe_name=GATE_NAME)
    created_at = conn.execute(
        "SELECT created_at FROM gate_evaluations WHERE id=?", (gate_id,)).fetchone()[0]
    conn.execute("UPDATE strategies SET stage='candidate' WHERE id=?", (rec.id,))
    conn.execute(
        "INSERT INTO stage_transitions(strategy_id, from_stage, to_stage, actor, reason,"
        " code_hash, config_hash, dependency_hash, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (rec.id, "backtested", "candidate", Actor.HUMAN.value, "fixture", CODE_HASH,
         CONFIG_HASH, DEP, created_at))
    conn.commit()


def _snapshot_frame(bump: float = 0.0) -> pd.DataFrame:
    """Daily bars: AAA/BBB (gate), CCC (CONFIG only), OLD (held) and ZZZ (held at zero), from a
    session before the window's start through the window's exclusive end."""
    closes = {"AAA": 10.0, "BBB": 12.0, "CCC": 20.0, "OLD": 5.0, "ZZZ": 7.0}
    days = [datetime(2022, 12, 22, tzinfo=UTC), *(datetime(2023, 1, d, tzinfo=UTC)
                                                  for d in (3, 4, 5, 6))]
    return pd.DataFrame([
        {"ts": day, "symbol": symbol, "open": close, "high": close, "low": close,
         "close": close + (bump if symbol == "BBB" else 0.0), "adj_close": close,
         "volume": 100.0}
        for day in days for symbol, close in closes.items()
    ])


def _ingest(store: DataStore, *, bump: float = 0.0) -> str:
    return store.ingest_bars(
        provider="test", symbols=["AAA", "BBB", "CCC", "OLD", "ZZZ"], start="2022-12-22",
        end="2023-01-06", as_of="2023-01-07", source=f"replay-test-{bump}",
        frame=_snapshot_frame(bump), timeframe="1d", adjustment="none").snapshot_id


# --- the recorded world ------------------------------------------------------------------------


@dataclass(frozen=True)
class Recorded:
    data_dir: Path
    db_path: Path
    snapshot_id: str
    interpreter: str
    bundle_root: Path
    phase_a: int
    phase_b: int

    @contextmanager
    def restarted(self) -> Iterator:
        """A supervisor restart: a brand-new registry connection, migrated as at startup."""
        with closing(connect(self.db_path)) as conn:
            migrate(conn)
            yield conn


def _target(tenant: FrozenTenant) -> FrozenTarget:
    """The tick's frozen target, from plain tenant values as the paper CLI builds it."""
    identity = WireIdentity(tenant.name, tenant.deployment_id, tenant.artifact_id,
                            tenant.deployment.manifest_digest, tenant.manifest.bundle.digest,
                            tenant.manifest.environment.digest)
    return FrozenTarget(identity, tenant.bundle_root, tenant.environment_root,
                        tenant.interpreter, tenant.view.execution, tuple(tenant.view.universe))


def _no_checkout_identity(name: str):
    raise AssertionError(f"a frozen tenant must never hash its checkout identity ({name})")


def _record_decision_tick(data_dir: Path, db_path: Path, snapshot_id: str) -> list[int]:
    """One real frozen tick; returns the ids ``record_frozen_invocation`` gave its attempts."""
    ids: list[int] = []
    with closing(connect(db_path)) as conn:
        migrate(conn)
        tenant = resolve_paper_tenant(
            conn, TENANT, command="paper trade-tick", verifier=FrozenContentVerifier(data_dir),
            data_dir=data_dir, identity_loader=_no_checkout_identity)
        assert isinstance(tenant, FrozenTenant)

        def record(attempt):
            ids.append(record_frozen_invocation(conn, attempt))
            return ids[-1]

        port = FrozenPlanner(
            _target(tenant), invocations_root=data_dir / "frozen" / "invocations", record=record,
            snapshot_id=snapshot_id, bars_start=START.isoformat(), bars_end=END.isoformat())
        broker = SimpleNamespace(
            get_positions=lambda: dict(HELD), snapshot=lambda universe: SNAPSHOT,
            cancel_open_orders=lambda: None,
            submit_sized=lambda intent, snap, coid, reserve=None: f"order-{intent.symbol}")
        hooks = TickHooks(
            planner=port,
            planner_context=planner_context_for_deployment(tenant.deployment,
                                                           get_calendar().code),
            venue_belief=lambda: {"OLD": 2.0}, peak_equity=100.0)
        result = run_tick(tenant.view, broker,
                          StoreBackedProvider(DataStore(data_dir), snapshot_id), START, END,
                          now=NOW, hooks=hooks, max_drawdown=0.1)
    assert set(result.target_weights) == {"BBB"} and result.submitted  # a real decision
    assert port.final_invocation_id == ids[-1]
    return ids


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Recorded]:
    root = tmp_path_factory.mktemp("replay")
    data_dir, db_path = root / "data", root / "registry.db"
    data_dir.mkdir()
    store = DataStore(data_dir)
    store.ingest_universe(universe=GATE_NAME, symbols=GATE, effective_date="2020-01-01",
                          as_of="2020-01-01T00:00:00Z", source="test")
    snapshot_id = _ingest(store)
    files = _tenant_bundle_files()
    bundle = BundleDescriptor.from_files(tuple(item.contract_entry for item in files))
    publish_bundle(data_dir, files, bundle)
    environment = environment_descriptor()
    build_environment(data_dir, environment.digest)
    CHECKOUT_MODULE.write_text(strategy_source(TENANT))  # present at admission, as in production

    def prepare(repo, name, *, repo_root, store_root):
        assert Path(store_root) == data_dir
        qualification = repo.qualify_frozen_candidate(
            name, code_hash=CODE_HASH, config_hash=CONFIG_HASH, dependency_hash=DEP)
        frozen = FrozenManifest(
            source_ref="a" * 40, code_hash=CODE_HASH, config_hash=CONFIG_HASH,
            dependency_hash=DEP, resolved_config=RECORDED,
            universe_name=qualification.universe_name, bundle=bundle, environment=environment)
        manifest = frozen_deployment_manifest(frozen)
        repo.record_frozen_artifact(
            name, manifest, research_gate_id=qualification.research_gate_id)
        return FrozenAdmission(manifest, qualification.research_gate_id)

    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr("algua.registry.frozen_runtime.verify_published_environment",
                          verify_environment_locator)
            with closing(connect(db_path)) as conn:
                migrate(conn)
                _add_frozen_candidate(conn)
                intake = run_intake(conn, equity=100_000.0, max_concurrent=5, actor=Actor.AGENT,
                                    prepare_and_verify=prepare, repo_root=CHECKOUT_ROOT,
                                    store_root=data_dir)
            assert [entry["strategy"] for entry in intake["admitted"]] == [TENANT], intake
            phase_a, phase_b = _record_decision_tick(data_dir, db_path, snapshot_id)
        yield Recorded(
            data_dir=data_dir, db_path=db_path, snapshot_id=snapshot_id,
            interpreter=str(data_dir.resolve() / environment.locator / "bin" / "python"),
            bundle_root=data_dir.resolve() / bundle.locator, phase_a=phase_a, phase_b=phase_b)
    finally:
        CHECKOUT_MODULE.unlink(missing_ok=True)
        if (data_dir / "frozen").exists():
            unseal(data_dir / "frozen")


@pytest.fixture(autouse=True)
def _offline_environment_check(monkeypatch):
    """The injected environment verification (see the module docstring); the bundle's is real."""
    monkeypatch.setattr("algua.registry.frozen_runtime.verify_published_environment",
                        verify_environment_locator)


def _never(*args, **kwargs):
    raise AssertionError("no child may be launched for a replay refused before launch")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _only_the_frozen_interpreter(monkeypatch, tmp_path: Path, interpreter: str) -> list[list[str]]:
    """Observe every process started through ``subprocess`` beneath every helper; any launch but
    the frozen interpreter fails as if its program were not installed, and Git is out of reach
    (nothing on ``PATH``, ``GIT_DIR`` naming no repository, the working directory outside any)."""
    launched: list[list[str]] = []
    execute_child = subprocess.Popen._execute_child  # type: ignore[attr-defined]

    def execute(self, args, executable, *rest):
        argv = ([os.fsdecode(args)] if isinstance(args, (str, bytes, os.PathLike))
                else [os.fsdecode(arg) for arg in args])
        launched.append(argv)
        if argv[0] != interpreter:
            raise FileNotFoundError(errno.ENOENT, "not installed", argv[0])
        return execute_child(self, args, executable, *rest)

    empty_path, outside = tmp_path / "empty-path", tmp_path / "outside-any-repository"
    empty_path.mkdir()
    outside.mkdir()
    monkeypatch.setattr(subprocess.Popen, "_execute_child", execute)
    monkeypatch.setenv("PATH", str(empty_path))
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "no-repository"))
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    monkeypatch.chdir(outside)
    return launched


# --- replay determinism ------------------------------------------------------------------------


def test_the_tick_recorded_one_phase_a_and_one_phase_b_attempt_over_its_bars(world):
    with world.restarted() as conn:
        first = frozen_invocation(conn, world.phase_a)
        final = frozen_invocation(conn, world.phase_b)
    assert first is not None and final is not None
    assert (first.phase, first.result_kind, final.phase, final.result_kind) == (
        "a", "snapshot_required", "b", "decision")
    assert final.phase_a_invocation_id == world.phase_a
    for row in (first, final):
        assert (row.snapshot_id, row.bars_start, row.bars_end) == (
            world.snapshot_id, START.isoformat(), END.isoformat())


@pytest.mark.parametrize("checkout", ["edited", "deleted"])
def test_each_recorded_attempt_replays_to_its_result_digest_after_a_restart(
    world, monkeypatch, tmp_path, checkout
):
    """After a restart, with the tenant's checkout module armed to raise or gone, replaying the
    Phase A and the Phase B attempt reproduces each recorded result byte for byte, with no Git and
    no uv: the only processes started are the two frozen children, launched exactly as §5 does."""
    if checkout == "edited":
        CHECKOUT_MODULE.write_text(RAISING_MODULE)
    else:
        CHECKOUT_MODULE.unlink(missing_ok=True)
    launched = _only_the_frozen_interpreter(monkeypatch, tmp_path, world.interpreter)

    for invocation_id in (world.phase_a, world.phase_b):
        with world.restarted() as conn:
            row = frozen_invocation(conn, invocation_id)
            assert row is not None and row.result_sha256 is not None
            stdout = replay_attempt(conn, world.data_dir, invocation_id)
        assert _sha256(stdout) == row.result_sha256, row.phase

    assert len(launched) == 2
    invocations = (world.data_dir / "frozen" / "invocations").resolve()
    for argv in launched:
        assert argv[:-1] == [world.interpreter, "-I", "-B", "-c", BOOTSTRAP,
                             str(world.bundle_root)]
        assert Path(argv[-1]).parent.resolve() == invocations
    assert list(invocations.iterdir()) == []  # each sealed directory removed after its child
    assert CHECKOUT_DOTTED not in sys.modules


def test_the_re_encoded_bars_are_the_recorded_bytes_within_one_supervisor_environment(world):
    """``bars_sha256`` is comparable only under the supervisor's own pyarrow (pinned below by the
    golden-bytes test); within it, the re-read, re-encoded bars are the bytes the child was sent."""
    with world.restarted() as conn:
        for invocation_id in (world.phase_a, world.phase_b):
            row = frozen_invocation(conn, invocation_id)
            assert row is not None
            inputs = prepare_replay(conn, world.data_dir, row)
            assert _sha256(inputs.bars_arrow) == row.bars_sha256
            assert _sha256(inputs.request_json) == row.request_sha256


# --- a replay over other bars or content is refused before any child ---------------------------


@pytest.mark.parametrize("change", [
    {"bars_start": (START - 10 * DAY).isoformat()},  # takes in the 2022-12-22 session
    {"bars_end": (END + DAY).isoformat()},  # takes in the bar at the exclusive end
    {"bars_end": (END - DAY).isoformat()},  # drops the 2023-01-05 session
])
def test_a_replay_over_another_bars_window_fails_the_logical_digest_check(world, change):
    with world.restarted() as conn:
        row = frozen_invocation(conn, world.phase_b)
        assert row is not None
        with pytest.raises(ReplayMismatch) as refused:
            prepare_replay(conn, world.data_dir, replace(row, **change))
    assert refused.value.reason == "bars_digest"


def test_a_replay_over_a_replaced_snapshot_fails_the_logical_digest_check(world):
    """The recorded snapshot's files replaced by another snapshot's (same rows, BBB's closes
    moved): the re-read frame no longer reproduces the recorded ``bars_digest``; no child
    runs."""
    store = DataStore(world.data_dir)
    target = world.data_dir / store.get_snapshot(world.snapshot_id).data_path
    other = world.data_dir / store.get_snapshot(_ingest(store, bump=0.5)).data_path
    kept = target.with_name(target.name + ".kept")
    os.rename(target, kept)
    try:
        shutil.copytree(other, target)
        with world.restarted() as conn, pytest.raises(ReplayMismatch) as refused:
            replay_attempt(conn, world.data_dir, world.phase_a, run=_never)
    finally:
        shutil.rmtree(target, ignore_errors=True)
        os.rename(kept, target)
    assert refused.value.reason == "bars_digest"


@contextmanager
def _corrupted(world: Recorded, part: str) -> Iterator[None]:
    if part == "bundle":  # the tenant's own strategy module, changed inside the sealed bundle
        module = world.bundle_root / BUNDLE_MODULE
        original = module.read_bytes()
        unseal(world.bundle_root)
        module.chmod(0o644)
        module.write_bytes(original + b"# changed after publication\n")
        try:
            yield
        finally:
            module.write_bytes(original)
            seal(world.bundle_root)
    else:  # the environment's interpreter removed
        python = Path(world.interpreter)
        moved = python.with_name("python.moved")
        os.rename(python, moved)
        try:
            yield
        finally:
            os.rename(moved, python)


@pytest.mark.parametrize("part", ["bundle", "environment"])
def test_a_replay_against_corrupted_content_fails_verification(world, part):
    with _corrupted(world, part), world.restarted() as conn:
        with pytest.raises(FrozenContentUnavailable):
            replay_attempt(conn, world.data_dir, world.phase_a, run=_never)
    with world.restarted() as conn:  # restored content verifies again
        assert prepare_replay(conn, world.data_dir, frozen_invocation(conn, world.phase_a))


# --- the Arrow encoding the recorded bars_sha256 follows ----------------------------------------


def _golden_frame() -> pd.DataFrame:
    index = pd.DatetimeIndex(
        [datetime(2023, 1, 3, tzinfo=UTC), datetime(2023, 1, 3, tzinfo=UTC),
         datetime(2023, 1, 4, 21, 0, 0, 123456, tzinfo=UTC)], name="timestamp")
    return pd.DataFrame({
        "symbol": ["AAA", "BBB", "AAA"],
        "open": [10.0, -0.0, np.nan],
        "high": [10.5, np.inf, 11.25],
        "low": [9.5, -np.inf, 10.0],
        "close": [10.25, 12.0, 1.0e-300],
        "adj_close": [10.25, 12.0, 1.0e300],
        "volume": [100.0, 0.0, 5e6],
    }, index=index)


#: Length and sha256 of ``encode_bars(_golden_frame())`` under the locked pyarrow (23.0.1). A
#: pyarrow upgrade that changes these bytes changes every new ``bars_sha256`` (contract §6):
#: recorded digests stay comparable only within one supervisor environment.
GOLDEN_BARS = (1730, "c6576abb6bfe69c904b90d525fbe7d52b5c64a331e22ac2e324ba5052e19218d")


def test_encode_bars_pins_the_arrow_bytes_of_a_fixed_frame():
    frame = _golden_frame()
    data = encode_bars(frame)

    assert (len(data), _sha256(data)) == GOLDEN_BARS
    assert encode_bars(frame) == data  # deterministic within one process
    assert bars_digest(decode_bars(data)) == bars_digest(frame)
