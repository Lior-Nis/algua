"""Story 1.3c end to end through the paper CLI (CAP-2, CAP-4): a frozen tenant beside a sibling.

A tenant admitted by the real ``paper intake`` as ``frozen`` ticks in ``paper trade-tick`` and
``paper run-all`` from its published bundle: each planner phase runs in a fresh child (observed
through a spy that still launches the real child), the supervisor reads only the recorded config
plus the gate universe, and orders and the tick row are attributed to the tenant and stamped with
its deployment and descriptor identity. The legacy working-tree sibling keeps today's path exactly:
its checkout load, identity recomputation and in-process planner are pinned here. The frozen tenant
never imports its checkout module, its results do not change when that module is edited or deleted,
a new CLI invocation resolves the same stored artifact, and ``paper run`` refuses to replay it.
"""
from __future__ import annotations

import errno
import json
import os
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

from algua.cli import paper_cmd
from algua.cli.main import app
from algua.live.frozen_dispatch import FrozenPlanner
from algua.live.frozen_wire import BOOTSTRAP
from algua.registry.frozen_view import FrozenStrategyView
from algua.strategies.base import LoadedStrategy
from tests._frozen_paper_world import (
    CHECKOUT_MODULE,
    CODE_HASH,
    CONFIG_HASH,
    DEP,
    GATE,
    SIBLING,
    SNAP,
    TENANT,
    build_world,
    runner,
    teardown_world,
)

CHECKOUT_DOTTED = f"algua.strategies.momentum.{TENANT}"
RAISING_MODULE = "raise RuntimeError('the frozen tenant was imported from the checkout')\n"


@pytest.fixture
def world(monkeypatch, tmp_path):
    sys.modules.pop(CHECKOUT_DOTTED, None)
    try:
        yield build_world(monkeypatch, tmp_path)
    finally:
        teardown_world(tmp_path / "data")
        sys.modules.pop(CHECKOUT_DOTTED, None)


def _stamped(world, tick: dict) -> None:
    deployment = world.deployment()
    assert tick["deployment_id"] == deployment.id
    assert (tick["code_hash"], tick["config_hash"], tick["dependency_hash"]) == (
        CODE_HASH, CONFIG_HASH, DEP)
    assert tick["snapshot_id"] == SNAP


def test_trade_tick_plans_a_frozen_tenant_in_two_fresh_children(world):
    deployment = world.deployment()

    code, payload = world.trade_tick()

    assert code == 0, payload
    assert payload["ok"] is True and payload["strategy"] == TENANT
    assert world.launches == [world.bundle_root, world.bundle_root]  # Phase A, then Phase B
    assert payload["target_weights"] and set(payload["target_weights"]) <= set(GATE)
    submitted = world.broker.submitted_for(TENANT)
    assert submitted and [order["symbol"] for order in payload["submitted"]] == [
        effect[1] for effect in submitted]
    orders = world.orders(TENANT)
    assert [order["client_order_id"] for order in orders] == [effect[3] for effect in submitted]
    assert {order["strategy_id"] for order in orders} == {deployment.strategy_id}
    [tick] = world.ticks(TENANT)
    _stamped(world, tick)
    assert list((world.data_dir / "frozen" / "invocations").iterdir()) == []
    assert world.prepared == [TENANT]  # prepared once, at intake: a tick never rebuilds content
    assert CHECKOUT_DOTTED not in sys.modules


def test_run_all_ticks_the_frozen_tenant_from_children_and_the_sibling_in_process(world):
    code, alone = world.trade_tick(SIBLING)
    assert code == 0, alone
    assert world.launches == []  # a working-tree tenant never launches a child

    code, payload = world.run_all()

    assert code == 0, payload
    by_name = {entry["strategy"]: entry for entry in payload["strategies"]}
    assert set(by_name) == {TENANT, SIBLING} and payload["setup_errors"] == []
    assert world.launches == [world.bundle_root, world.bundle_root]
    assert by_name[TENANT]["ok"] is True and by_name[TENANT]["target_weights"]
    assert by_name[SIBLING] == alone  # the sibling decides exactly as it does on its own
    [frozen_tick] = world.ticks(TENANT)
    _stamped(world, frozen_tick)
    sibling_ticks = world.ticks(SIBLING)
    assert len(sibling_ticks) == 2 and {t["deployment_id"] for t in sibling_ticks} == {None}
    assert world.broker.submitted_for(TENANT)
    assert CHECKOUT_DOTTED not in sys.modules


def test_run_all_hands_run_tick_the_view_and_port_and_keeps_the_sibling_path(world, monkeypatch):
    """The plumbing pin: the frozen tenant reaches ``run_tick`` as its supervisor view with a fresh
    frozen port and its descriptor's planner context; the sibling reaches it as a checkout-loaded
    ``LoadedStrategy`` on the in-process planner, loaded exactly as before (the preflight load and
    the pre-tick re-gate) with one identity recomputation. The checkout loader and the identity
    recomputation never see the frozen tenant."""
    seen: dict[str, tuple] = {}
    loads: list[tuple[str, str]] = []
    identities: list[str] = []
    real_tick, real_load = paper_cmd.run_tick, paper_cmd.load_gated_strategy
    real_identity = paper_cmd.compute_artifact_hashes

    def tick(strategy, broker, provider, start, end, hooks=None, max_drawdown=None):
        seen[strategy.name] = (strategy, hooks)
        return real_tick(strategy, broker, provider, start, end, hooks=hooks,
                         max_drawdown=max_drawdown)

    def load(conn, name, command):
        loads.append((name, command))
        return real_load(conn, name, command)

    def identity(name):
        identities.append(name)
        return real_identity(name)

    monkeypatch.setattr(paper_cmd, "run_tick", tick)
    monkeypatch.setattr(paper_cmd, "load_gated_strategy", load)
    monkeypatch.setattr(paper_cmd, "compute_artifact_hashes", identity)

    code, payload = world.run_all()

    assert code == 0, payload
    view, frozen_hooks = seen[TENANT]
    deployment = world.deployment()
    assert isinstance(view, FrozenStrategyView) and list(view.universe) == GATE
    assert isinstance(frozen_hooks.planner, FrozenPlanner)
    context = frozen_hooks.planner_context
    assert (context.deployment_id, context.artifact_id, context.manifest_digest,
            context.config_hash) == (deployment.id, deployment.artifact_id,
                                     deployment.manifest_digest, CONFIG_HASH)
    strategy, sibling_hooks = seen[SIBLING]
    assert isinstance(strategy, LoadedStrategy) and sibling_hooks.planner is None
    assert loads == [(SIBLING, "paper run-all"), (SIBLING, "paper run-all")]
    assert identities == [SIBLING]


def test_trade_tick_builds_a_fresh_port_for_every_tick(world, monkeypatch):
    ports: list[object] = []
    real_tick = paper_cmd.run_tick

    def tick(strategy, broker, provider, start, end, hooks=None, max_drawdown=None):
        ports.append(hooks.planner)
        return real_tick(strategy, broker, provider, start, end, hooks=hooks,
                         max_drawdown=max_drawdown)

    monkeypatch.setattr(paper_cmd, "run_tick", tick)
    assert world.trade_tick()[0] == 0
    assert world.run_all()[0] == 0
    frozen_ports = [port for port in ports if port is not None]
    assert len(frozen_ports) == 2 and frozen_ports[0] is not frozen_ports[1]


def test_frozen_results_are_independent_of_the_mutable_checkout_across_restarts(world):
    """CAP-4: edit the checkout module into one that raises on import, then delete it; every tick
    is a NEW CLI invocation (a restart: a new verifier and resolution of the same stored artifact),
    and each decides exactly as the first. Nothing is re-prepared from the checkout."""
    code, first = world.trade_tick()
    assert code == 0, first
    CHECKOUT_MODULE.write_text(RAISING_MODULE)
    code_edited, edited = world.trade_tick()
    CHECKOUT_MODULE.unlink()
    code_deleted, deleted = world.trade_tick()

    assert (code_edited, code_deleted) == (0, 0), (edited, deleted)
    assert edited == first and deleted == first
    ticks = world.ticks(TENANT)
    assert len(ticks) == 3
    for tick in ticks:
        _stamped(world, tick)
    assert world.rows("SELECT COUNT(*) AS n FROM deployment_artifacts") == [{"n": 1}]
    assert world.prepared == [TENANT]
    assert len(world.launches) == 6 and set(world.launches) == {world.bundle_root}
    assert CHECKOUT_DOTTED not in sys.modules


def test_a_frozen_tick_makes_no_git_or_uv_call_and_needs_no_git_checkout(
    world, monkeypatch, tmp_path
):
    """AC5/AC10: a frozen tenant resolves, verifies and plans with no ``uv`` or Git call and no Git
    checkout. Every process started through ``subprocess`` is observed where it executes, beneath
    every helper: across a frozen ``trade-tick`` and a ``run-all`` the only ones are the frozen
    children (the verified interpreter with the exact §5 bootstrap argv), and any other launch fails
    as if its program were not installed. Git is out of reach as well: nothing on ``PATH``,
    ``GIT_DIR`` naming no repository and the working directory outside any. The tenant still
    ticks in both commands."""
    interpreter = str(world.data_dir.resolve() / world.environment.locator / "bin" / "python")
    launched: list[list[str]] = []
    execute_child = subprocess.Popen._execute_child

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

    code, tick = world.trade_tick()
    cycle_code, cycle = world.run_all()

    assert (code, cycle_code) == (0, 0), (tick, cycle)
    assert tick["ok"] is True and tick["target_weights"]
    frozen = next(entry for entry in cycle["strategies"] if entry["strategy"] == TENANT)
    assert frozen["ok"] is True and frozen["target_weights"]
    assert len(world.ticks(TENANT)) == 2
    invocations = (world.data_dir / "frozen" / "invocations").resolve()
    assert len(launched) == 4  # Phase A and Phase B, in each command
    for argv in launched:
        assert argv[:-1] == [interpreter, "-I", "-B", "-c", BOOTSTRAP, str(world.bundle_root)]
        assert Path(argv[-1]).parent.resolve() == invocations


def test_run_all_never_imports_the_frozen_tenants_checkout_module(world):
    CHECKOUT_MODULE.write_text(RAISING_MODULE)

    code, payload = world.run_all()

    assert code == 0, payload
    assert {entry["strategy"] for entry in payload["strategies"] if entry["ok"]} == {
        TENANT, SIBLING}
    assert CHECKOUT_DOTTED not in sys.modules


def test_run_all_refresh_plans_the_frozen_tenant_from_its_view(world, monkeypatch):
    """``--refresh`` sizes the lane's bars from each tenant's plan: the frozen tenant's comes from
    its supervisor view (gate universe AAA/BBB, recorded lookback), never its checkout module (armed
    to raise), and never its wider CONFIG universe (CCC)."""
    CHECKOUT_MODULE.write_text(RAISING_MODULE)
    requested: dict = {}

    def refresh(symbols, *, end, min_rows, kind):
        requested.update(symbols=list(symbols), min_rows=dict(min_rows))
        start = (date.fromisoformat(end) - timedelta(days=200)).isoformat()
        return {"id": SNAP, "refreshed": True, "start": start, "end": end}

    monkeypatch.setattr(paper_cmd, "refresh_lane_snapshot", refresh)

    result = runner.invoke(app, ["paper", "run-all", "--refresh"])

    assert result.exit_code == 0, result.stdout
    by_name = {entry["strategy"]: entry for entry in json.loads(result.stdout)["strategies"]}
    assert by_name[TENANT]["ok"] is True and by_name[SIBLING]["ok"] is True
    assert set(GATE) <= set(requested["symbols"]) and "CCC" not in requested["symbols"]
    assert requested["min_rows"]["AAA"] == requested["min_rows"]["BBB"] == 2  # lookback 1 + 1
    assert len(world.launches) == 2
    assert CHECKOUT_DOTTED not in sys.modules


def test_paper_run_refuses_a_frozen_deployment_before_replaying_anything(world):
    CHECKOUT_MODULE.write_text(RAISING_MODULE)

    result = runner.invoke(app, ["paper", "run", TENANT, "--demo",
                                 "--start", "2023-01-01", "--end", "2023-03-31"])

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False and payload["code"] == "frozen_live_unsupported"
    assert payload["retryable"] is False
    assert world.rows("SELECT * FROM paper_orders WHERE strategy=?", TENANT) == []
    assert world.audit(TENANT) == [("paper_intake", "slice 20000.0")]
    assert CHECKOUT_DOTTED not in sys.modules



def test_raw_forward_and_go_live_transitions_never_import_the_checkout_module(world):
    """``registry transition`` refuses the frozen tenant on the forward edge and the go-live
    challenge before hashing the checkout (armed to raise): no module import, no pinned
    checkout code_hash and no stage change."""
    CHECKOUT_MODULE.write_text(RAISING_MODULE)
    history = "SELECT from_stage, to_stage, code_hash FROM stage_transitions WHERE strategy_id=?"
    tenant_id = world.deployment().strategy_id
    before = world.rows(history, tenant_id)

    forward = runner.invoke(app, ["registry", "transition", TENANT, "--to", "forward_tested",
                                  "--actor", "human", "--reason", "raw forward edge"])
    with world.conn() as conn:  # the shape a pre-fix raw forward transition left behind
        conn.execute("UPDATE strategies SET stage='forward_tested' WHERE id=?", (tenant_id,))
        conn.commit()
    live = runner.invoke(app, ["registry", "transition", TENANT, "--to", "live",
                               "--actor", "human"])

    assert json.loads(forward.stdout)["code"] == "frozen_qualification_pending"
    assert json.loads(live.stdout)["code"] == "frozen_live_unsupported"
    assert (forward.exit_code, live.exit_code) == (1, 1)
    assert world.rows(history, tenant_id) == before
    assert world.rows("SELECT * FROM live_challenges") == []
    assert CHECKOUT_DOTTED not in sys.modules
