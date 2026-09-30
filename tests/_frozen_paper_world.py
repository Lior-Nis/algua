"""A paper book holding one FROZEN tenant and one legacy working-tree sibling (Story 1.3c e2e).

The frozen tenant is admitted through the real ``paper intake`` (Story 1.3b preparation and
verification injected, as in tests/test_paper_intake.py) against a REALLY published bundle: this
checkout's ``algua/`` plus the harness fixture strategy (tests/_frozen_harness.py) and 1.3b's own
generated ``_algua/`` files. The environment is the harness's venv-shaped tree at its descriptor's
locator. At tick time the bundle is verified offline for real; only the environment's offline
verification is injected (a real inventory of the development site-packages is not a unit-test
cost), and it is still a locator check. The checkout also holds the tenant's module at admission,
exactly as a real admission would, so a test can edit or delete it afterwards.

The sibling is a legacy (fixed-cohort) working-tree tenant with a legacy gate row, so it ticks on
its CONFIG universe through today's in-process path. Both tenants tick through the real
``run_tick`` over one recording fake broker and the deterministic ``SyntheticProvider``.
"""
from __future__ import annotations

import hashlib
import json
from contextlib import closing
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from typer.testing import CliRunner

from algua.backtest._sample import SyntheticProvider
from algua.cli.main import app
from algua.config.settings import get_settings
from algua.contracts.lifecycle import Actor
from algua.data.store import DataStore
from algua.execution.alpaca_broker import AccountState
from algua.registry.artifact_contract import BundleDescriptor
from algua.registry.artifact_preparation import FrozenPreparationResult, _bundle_files
from algua.registry.artifact_recording import frozen_deployment_manifest
from algua.registry.artifact_store import publish_bundle, resolve_locator
from algua.registry.artifact_verification import FrozenVerificationResult
from algua.registry.db import connect, migrate
from algua.registry.environment_contract import EnvironmentDescriptor, EnvironmentKey
from algua.registry.environment_store import EnvironmentStoreError
from algua.registry.frozen_manifest_contract import FrozenManifest
from algua.registry.frozen_source import FrozenFile
from algua.registry.planner_environment import current_interpreter_identity
from algua.registry.store import SqliteStrategyRepository
from algua.strategies.base import config_hash
from tests._deployment_helpers import force_legacy_strategy
from tests._frozen_harness import (
    CHECKOUT_ALGUA,
    PROTOCOL,
    STAMP,
    build_environment,
    in_process,
    protocol_bytes,
    recorded,
    strategy_source,
    unseal,
)
from tests._gate_row_helpers import seed_passing_gate

runner = CliRunner()

TENANT = "frozen_paper_tenant"
SIBLING = "cross_sectional_momentum"
FAMILY = "frozen_paper_family"
GATE_NAME = "frozen-gate"
GATE = ["AAA", "BBB"]  # the frozen tenant's gate universe; its CONFIG universe adds CCC
SNAP = "snap1"
DEP = hashlib.sha256(b"frozen-paper-dependencies").hexdigest()
CODE_HASH = hashlib.sha256(b"frozen-paper-code").hexdigest()[:32]
RECORDED = recorded(in_process(TENANT))
CONFIG_HASH = config_hash(in_process(TENANT))
CHECKOUT_MODULE = CHECKOUT_ALGUA / "strategies" / "momentum" / f"{TENANT}.py"
WIRE_V2 = protocol_bytes(frozen_wire={"name": "frozen-planner", "version": 2})


def window() -> tuple[str, str]:
    """A rolling window ending today: the #452 wall refuses marks older than two sessions."""
    end = datetime.now(UTC).date()
    return (end - timedelta(days=90)).isoformat(), end.isoformat()


# --- content -----------------------------------------------------------------------------------


def bundle_files(protocol: bytes | None) -> tuple[FrozenFile, ...]:
    """This checkout's ``algua/`` with the tenant in its own family, plus 1.3b's ``_algua/``."""
    source = [
        FrozenFile(f"algua/{path.relative_to(CHECKOUT_ALGUA).as_posix()}", "100644",
                   path.read_bytes())
        for path in sorted(CHECKOUT_ALGUA.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts and path.name != f"{TENANT}.py"
    ]
    source += [
        FrozenFile(f"algua/strategies/{FAMILY}/__init__.py", "100644", b""),
        FrozenFile(f"algua/strategies/{FAMILY}/{TENANT}.py", "100644",
                   strategy_source(TENANT).encode()),
    ]
    files = [item for item in _bundle_files(tuple(source), RECORDED) if item.path != PROTOCOL]
    if protocol is not None:
        files.append(FrozenFile(PROTOCOL, "100644", protocol))
    return tuple(sorted(files, key=lambda item: item.path.encode()))


def environment_descriptor() -> EnvironmentDescriptor:
    identity = current_interpreter_identity()
    key = EnvironmentKey(
        build_inputs_digest="1" * 64, dependency_hash=DEP, interpreter=identity,
        uv_version="uv 0.9.26", create_argv=("uv", "venv"), sync_argv=("uv", "sync"),
    )
    return EnvironmentDescriptor(key, "2" * 64, identity)


def verify_environment_locator(store_root: Path, descriptor: EnvironmentDescriptor) -> Path:
    """The injected offline environment check: the canonical locator must hold an interpreter."""
    target = resolve_locator(Path(store_root).resolve(), descriptor.locator,
                             expected_digest=descriptor.digest, kind="environments")
    if not (target / "bin" / "python").is_file():
        raise EnvironmentStoreError("frozen environment is missing or corrupt")
    return target


# --- the fake venue ----------------------------------------------------------------------------


class RecordingBroker:
    """A clean paper account that records every venue effect (cancel/submit/offset)."""

    def __init__(self, equity: float = 100_000.0) -> None:
        self.equity = equity
        self.effects: list[tuple[Any, ...]] = []

    def account(self) -> AccountState:
        return AccountState(equity=self.equity, cash=self.equity, buying_power=self.equity,
                            account_id="frozen-acct")

    def clock(self) -> str:
        return "2026-01-02T14:00:00+00:00"

    def account_activities_window(self, after: str, until: str) -> list:
        return []

    def get_positions(self) -> pd.Series:
        return pd.Series({}, dtype="float64")

    def list_open_orders(self) -> list:
        return []

    def cancel_order(self, order_id: str) -> None:
        self.effects.append(("cancel_order", order_id))

    def cancel_open_orders(self) -> None:
        self.effects.append(("cancel_open_orders",))

    def submit_sized(self, intent, snap, coid=None, reserve=None) -> str:
        self.effects.append(("submit", intent.symbol, intent.side.value, coid))
        return f"o-{coid}"

    def submit_offset(self, sym: str, qty: float, coid: str) -> str:
        self.effects.append(("offset", sym, qty, coid))
        return f"o-offset-{sym}"

    def submitted_for(self, name: str) -> list[tuple[Any, ...]]:
        return [e for e in self.effects if e[0] == "submit" and str(e[3]).startswith(f"{name}-")]


# --- the world ---------------------------------------------------------------------------------


@dataclass
class World:
    data_dir: Path
    broker: RecordingBroker
    bundle: BundleDescriptor
    environment: EnvironmentDescriptor
    launches: list[Path] = field(default_factory=list)
    prepared: list[str] = field(default_factory=list)

    @property
    def bundle_root(self) -> Path:
        return self.data_dir.resolve() / self.bundle.locator

    def conn(self):
        conn = connect(get_settings().db_path)
        migrate(conn)
        return closing(conn)

    def deployment(self) -> Any:
        with self.conn() as conn:
            repo = SqliteStrategyRepository(conn)
            return repo.active_deployment(repo.get(TENANT).id)

    def rows(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        with self.conn() as conn:
            return [dict(row) for row in conn.execute(sql, args).fetchall()]

    def ticks(self, name: str) -> list[dict[str, Any]]:
        return self.rows("SELECT * FROM tick_snapshots WHERE strategy=? ORDER BY id", name)

    def orders(self, name: str) -> list[dict[str, Any]]:
        return self.rows("SELECT * FROM paper_venue_orders WHERE strategy=? ORDER BY id", name)

    def audit(self, name: str) -> list[tuple[str, str]]:
        return [(row["action"], row["reason"]) for row in self.rows(
            "SELECT action, reason FROM audit_log WHERE strategy=? ORDER BY id", name)]

    def trade_tick(self, name: str = TENANT) -> tuple[int, dict[str, Any]]:
        start, end = window()
        result = runner.invoke(app, ["paper", "trade-tick", name, "--snapshot", SNAP,
                                     "--start", start, "--end", end])
        return result.exit_code, json.loads(result.stdout)

    def run_all(self) -> tuple[int, dict[str, Any]]:
        start, end = window()
        result = runner.invoke(app, ["paper", "run-all", "--snapshot", SNAP,
                                     "--start", start, "--end", end])
        return result.exit_code, json.loads(result.stdout)


def _add_sibling(conn) -> None:
    repo = SqliteStrategyRepository(conn)
    rec = repo.add(SIBLING)
    force_legacy_strategy(conn, rec.id)
    seed_passing_gate(SIBLING)
    _allocate(conn, rec.id)


def _allocate(conn, strategy_id: int, capital: float = 10_000.0) -> None:
    conn.execute(
        "INSERT INTO strategy_allocations(strategy_id, capital, effective_ts, actor)"
        " VALUES (?,?,?,?)", (strategy_id, capital, datetime.now(UTC).isoformat(), "agent"))
    conn.commit()


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


def build_world(monkeypatch, tmp_path: Path, *, protocol: bytes | None = STAMP) -> World:
    """Publish content, register the sibling and the candidate, and admit the candidate through
    the real ``paper intake``. The CLI's broker, provider and child launch are wired to fakes/spies;
    the child itself is real."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "registry.db"))
    monkeypatch.setenv("ALGUA_DATA_DIR", str(data_dir))
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    DataStore(data_dir).ingest_universe(
        universe=GATE_NAME, symbols=GATE, effective_date="2020-01-01",
        as_of="2020-01-01T00:00:00Z", source="test")
    files = bundle_files(protocol)
    bundle = BundleDescriptor.from_files(tuple(item.contract_entry for item in files))
    publish_bundle(data_dir, files, bundle)
    environment = environment_descriptor()
    build_environment(data_dir, environment.digest)
    world = World(data_dir, RecordingBroker(), bundle, environment)
    CHECKOUT_MODULE.write_text(strategy_source(TENANT))

    def prepare(repo, name, *, repo_root, store_root):
        world.prepared.append(name)
        assert Path(store_root) == data_dir
        qualification = repo.qualify_frozen_candidate(
            name, code_hash=CODE_HASH, config_hash=CONFIG_HASH, dependency_hash=DEP)
        frozen = FrozenManifest(
            source_ref="a" * 40, code_hash=CODE_HASH, config_hash=CONFIG_HASH,
            dependency_hash=DEP, resolved_config=RECORDED,
            universe_name=qualification.universe_name, bundle=bundle, environment=environment)
        artifact_id = repo.record_frozen_artifact(
            name, frozen_deployment_manifest(frozen),
            research_gate_id=qualification.research_gate_id)
        return FrozenPreparationResult(name, artifact_id, qualification.research_gate_id, frozen)

    def verify(repo, manifest_digest, *, store_root):
        record = repo.deployment_artifact_by_digest(manifest_digest)
        manifest = record.frozen_manifest()
        return FrozenVerificationResult(manifest.resolved_config["name"], record.id, manifest)

    from algua.live import frozen_dispatch

    real_launch = frozen_dispatch.launch_child

    def spy_launch(**kwargs):
        world.launches.append(Path(kwargs["bundle_root"]))
        return real_launch(**kwargs)

    monkeypatch.setattr("algua.registry.intake.prepare_frozen_artifact", prepare)
    monkeypatch.setattr("algua.registry.intake.verify_frozen_artifact", verify)
    monkeypatch.setattr("algua.registry.frozen_runtime.verify_published_environment",
                        verify_environment_locator)
    monkeypatch.setattr("algua.live.frozen_dispatch.launch_child", spy_launch)
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings", lambda: world.broker)
    monkeypatch.setattr("algua.cli.paper_cmd._select_provider",
                        lambda demo, snapshot: SyntheticProvider())
    with world.conn() as conn:
        # The frozen tenant registers first (lower id), so run-all ticks it BEFORE the sibling:
        # an isolated frozen failure must let the cycle continue on to the sibling.
        _add_frozen_candidate(conn)
        _add_sibling(conn)
    result = runner.invoke(app, ["paper", "intake", "--max-concurrent", "5"])
    assert result.exit_code == 0, result.output
    assert [a["strategy"] for a in json.loads(result.stdout)["admitted"]] == [TENANT]
    return world


def teardown_world(data_dir: Path) -> None:
    """Remove the checkout module and unseal published content so the tmp tree can be removed."""
    CHECKOUT_MODULE.unlink(missing_ok=True)
    if (data_dir / "frozen").exists():
        unseal(data_dir / "frozen")
