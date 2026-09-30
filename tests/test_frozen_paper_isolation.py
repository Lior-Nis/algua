"""Story 1.3c §8 (CAP-3): every frozen-tenant fault is isolated to that tenant before any effect.

Each failure class — content that is missing or corrupt, unsupported content, an oversized request,
a failed launch, a timeout (the REAL child killed at a tiny timeout), an abnormal exit, oversized
output, an invalid result and a planner refusal (the REAL child's, and the supervisor's own
strategy-free check) — gives zero cancel/submit/offset/ledger/tick effects for the frozen tenant, a
nonzero ``paper trade-tick`` carrying the stable code and the tenant's ``deployment_id``, and a
``run-all`` that records ``{"ok": false, "strategy", "kind": "setup_error", "error": code,
"deployment_id"}``, audits it and goes on to tick the sibling (the frozen tenant ticks FIRST, so the
sibling ticking proves the cycle continued). Systemic faults (SQLite, a global halt, an interrupt)
are never demoted to a tenant failure. The CLI error registry and ``StrategySetupError`` read the
frozen exceptions' own codes.

Story 1.3d (CAP-1): every attempt the port decided to run a child for leaves exactly one failure
row carrying its code (a pre-launch refusal records no request bytes); a fault the supervisor
settles itself before any attempt, and every systemic fault, records nothing. A failed attempt
never yields a tick row.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from algua.cli import paper_cmd
from algua.cli._common import StrategySetupError
from algua.cli.errors import error_code, is_retryable
from algua.cli.main import app
from algua.live import frozen_dispatch
from algua.live.frozen_dispatch import FROZEN_FAILURE_CODES, FrozenTenantFailure
from algua.live.frozen_invocation import LaunchFailure
from algua.live.live_loop import run_tick
from algua.primitives.contained_process import ContainedResult
from algua.registry.frozen_tenant_errors import (
    FrozenContentUnavailable,
    FrozenLiveUnsupported,
    FrozenTenantUnsupported,
)
from algua.registry.frozen_view import FrozenStrategyView
from algua.risk import global_halt, kill_switch
from tests._frozen_harness import NOW, STAMP, STRATEGY, in_process, seal, unseal
from tests._frozen_paper_world import (
    SIBLING,
    SNAP,
    TENANT,
    WIRE_V2,
    World,
    build_world,
    runner,
    teardown_world,
)

REPO = Path(__file__).resolve().parents[1]


def _corrupt_bundle(world: World, monkeypatch) -> None:
    root = world.bundle_root
    unseal(root)
    target = root / "algua" / "__init__.py"
    target.chmod(0o644)
    target.write_bytes(target.read_bytes() + b"# tampered after publication\n")
    seal(root)


def _answer(result: ContainedResult):
    def fake_launch(world: World):
        def launch(**kwargs):
            world.launches.append(Path(kwargs["bundle_root"]))
            return result
        return launch
    return fake_launch


def _raise(exc: BaseException):
    def fake_launch(world: World):
        def launch(**kwargs):
            world.launches.append(Path(kwargs["bundle_root"]))
            raise exc
        return launch
    return fake_launch


def _ended(returncode=0, *, stdout=b"", stdout_exceeded=False) -> ContainedResult:
    return ContainedResult(returncode, None, False, stdout, stdout_exceeded, b"boom", False)


def _config_hash(value: str):
    """The frozen tenant's planner requests carry ``value`` as its config hash (only the frozen
    tenant's: the sibling's context is left alone)."""
    def setup(world: World, monkeypatch) -> None:
        frozen_id = world.deployment().id
        real = paper_cmd.planner_context_for_deployment

        def context(deployment, calendar_code):
            built = real(deployment, calendar_code)
            if deployment is None or deployment.id != frozen_id:
                return built
            return replace(built, config_hash=value)

        monkeypatch.setattr(paper_cmd, "planner_context_for_deployment", context)
    return setup


# case -> (child launches expected, attempt recorded, setup); a case is its code, or
# ``code/variant`` when one code has two causes. ``setup(world, monkeypatch)`` injects the fault
# after admission. Launches are counted at the dispatcher's launch seam (the spy or the fake
# answering). An attempt is recorded for every phase dispatch the port decided to run a child for,
# including one refused before launch; not for content refused at resolution or an input the
# supervisor itself refuses.
FAILURES = {
    "frozen_content_unavailable": (0, False, _corrupt_bundle),
    "frozen_content_unsupported": (0, True, None),  # the world is built on a wire-v2 bundle
    "frozen_request_too_large": (0, True, lambda world, mp: mp.setattr(
        "algua.live.frozen_wire.MAX_REQUEST_BYTES", 16)),
    "frozen_launch_failed": (1, True, lambda world, mp: mp.setattr(
        "algua.live.frozen_dispatch.launch_child", _raise(LaunchFailure("no interpreter"))(world))),
    "frozen_timeout": (1, True, lambda world, mp: mp.setattr(
        "algua.live.frozen_invocation.TIMEOUT_SECONDS", 0.05)),
    "frozen_exit_abnormal": (1, True, lambda world, mp: mp.setattr(
        "algua.live.frozen_dispatch.launch_child", _answer(_ended(1))(world))),
    "frozen_output_exceeded": (1, True, lambda world, mp: mp.setattr(
        "algua.live.frozen_dispatch.launch_child",
        _answer(_ended(stdout=b"{", stdout_exceeded=True))(world))),
    "frozen_result_invalid": (1, True, lambda world, mp: mp.setattr(
        "algua.live.frozen_dispatch.launch_child",
        _answer(_ended(stdout=b'{"not": "a result"}'))(world))),
    # A well-formed config hash that is not the bundle strategy's passes the supervisor's
    # strategy-free checks; the REAL child re-hashes its CONFIG and refuses the request
    # (``strategy_identity_mismatch``), answering ``planner_rejected``.
    "frozen_planner_rejected": (1, True, _config_hash("0" * 32)),
    # A malformed one is refused by the supervisor's own run of those checks (§7), before any child.
    "frozen_planner_rejected/supervisor": (0, False, _config_hash("not-a-config-hash")),
}
#: Refused before the request could be encoded: the row carries no request bytes.
_NO_REQUEST = frozenset({"frozen_content_unsupported", "frozen_request_too_large"})


@pytest.fixture(params=sorted(FAILURES))
def failing(request, monkeypatch, tmp_path):
    launches, recorded, setup = FAILURES[request.param]
    code = request.param.partition("/")[0]
    try:
        world = build_world(monkeypatch, tmp_path,
                            protocol=WIRE_V2 if code == "frozen_content_unsupported" else STAMP)
        if setup is not None:
            setup(world, monkeypatch)
        yield code, launches, recorded, world
    finally:
        teardown_world(tmp_path / "data")


def _invocations(world: World) -> list[dict]:
    return world.rows("SELECT * FROM frozen_invocations ORDER BY id")


def _assert_failure_recorded(world: World, code: str, recorded: bool) -> None:
    """Exactly one failed Phase A row with ``code`` when the port ran an attempt, else none."""
    rows = _invocations(world)
    if not recorded:
        assert rows == []
        return
    [row] = rows
    assert (row["phase"], row["failure_code"], row["deployment_id"], row["snapshot_id"]) == (
        "a", code, world.deployment().id, SNAP)
    assert row["result_kind"] is None and row["result_sha256"] is None
    assert row["phase_a_invocation_id"] is None and row["diagnostic"]
    assert row["bars_start"] and row["bars_end"]
    assert (row["request_json"] is None) is (code in _NO_REQUEST)
    assert (row["bars_sha256"] is None) is (code in _NO_REQUEST)
    assert bool(row["timed_out"]) is (code == "frozen_timeout")
    assert bool(row["stdout_exceeded"]) is (code == "frozen_output_exceeded")


def _assert_no_effects(world: World) -> None:
    assert world.broker.submitted_for(TENANT) == []
    assert world.orders(TENANT) == [] and world.ticks(TENANT) == []
    assert all(action != "trade_tick" for action, _ in world.audit(TENANT))
    assert world.rows("SELECT * FROM strategy_peaks WHERE strategy=?", TENANT) == []
    with world.conn() as conn:
        assert not kill_switch.is_tripped(conn, TENANT)
        assert not global_halt.is_engaged(conn)


def test_trade_tick_exits_nonzero_with_the_code_and_no_effect(failing):
    code, launches, recorded, world = failing

    exit_code, payload = world.trade_tick()

    assert exit_code == 1, payload
    assert payload["ok"] is False and payload["code"] == code and payload["retryable"] is False
    assert payload["deployment_id"] == world.deployment().id  # deployment-bound (AC8)
    assert len(world.launches) == launches
    assert world.broker.effects == []  # not even the account-wide cancel
    _assert_no_effects(world)
    _assert_failure_recorded(world, code, recorded)


def test_run_all_records_the_bound_entry_and_continues_the_sibling(failing):
    code, launches, recorded, world = failing
    deployment = world.deployment()

    exit_code, payload = world.run_all()

    assert exit_code == 0, payload
    entry = {"ok": False, "strategy": TENANT, "kind": "setup_error", "error": code,
             "deployment_id": deployment.id}
    assert payload["strategies"][0] == entry and payload["setup_errors"] == [entry]
    sibling = payload["strategies"][1]
    assert sibling["strategy"] == SIBLING and sibling["ok"] is True
    [sibling_tick] = world.ticks(SIBLING)
    assert sibling_tick["frozen_invocation_id"] is None
    assert ("strategy_setup_error", code) in world.audit(TENANT)
    assert len(world.launches) == launches
    assert not [e for e in world.broker.effects
                if e[0] != "submit" or str(e[3]).startswith(f"{TENANT}-")]
    _assert_no_effects(world)
    _assert_failure_recorded(world, code, recorded)


@pytest.fixture
def world(monkeypatch, tmp_path):
    try:
        yield build_world(monkeypatch, tmp_path)
    finally:
        teardown_world(tmp_path / "data")


def test_a_failed_phase_b_attempt_is_recorded_after_its_phase_a_and_yields_no_tick(
    world, monkeypatch
):
    """Phase A's real child succeeds and is recorded; Phase B's child exits abnormally: its row
    follows the Phase A row it names and carries the code, and no tick links either."""
    spy_launch = frozen_dispatch.launch_child  # the world's spy over the real child launch

    def launch(**kwargs):
        if not world.launches:
            return spy_launch(**kwargs)  # Phase A: the real child
        world.launches.append(Path(kwargs["bundle_root"]))
        return _ended(1)  # Phase B: an abnormal exit

    monkeypatch.setattr("algua.live.frozen_dispatch.launch_child", launch)

    exit_code, payload = world.run_all()

    assert exit_code == 0, payload
    entry = payload["strategies"][0]
    assert (entry["strategy"], entry["error"]) == (TENANT, "frozen_exit_abnormal")
    assert payload["strategies"][1]["ok"] is True and len(world.ticks(SIBLING)) == 1
    phase_a, phase_b = _invocations(world)
    assert (phase_a["phase"], phase_a["result_kind"], phase_a["failure_code"]) == (
        "a", "snapshot_required", None)
    assert (phase_b["phase"], phase_b["failure_code"], phase_b["returncode"]) == (
        "b", "frozen_exit_abnormal", 1)
    assert phase_b["phase_a_invocation_id"] == phase_a["id"]
    assert phase_b["request_id"] == phase_a["request_id"] and phase_b["result_sha256"] is None
    assert phase_b["phase_a_binding"] == phase_a["phase_a_binding"]
    _assert_no_effects(world)


@pytest.mark.parametrize("exc, code", [
    (sqlite3.OperationalError("database is locked"), "db_unavailable"),
    # the data volume failing under the invocation directory affects every tenant
    (OSError(28, "No space left on device"), "internal"),
    (global_halt.GlobalHaltActive("global halt active"), "invalid_input"),
])
def test_a_systemic_fault_inside_a_frozen_tick_aborts_the_cycle(world, monkeypatch, exc, code):
    monkeypatch.setattr("algua.live.frozen_dispatch.launch_child", _raise(exc)(world))

    exit_code, payload = world.run_all()

    assert exit_code == 1
    assert payload["ok"] is False and payload["code"] == code
    assert "setup_error" not in json.dumps(payload)
    assert world.ticks(SIBLING) == [] and world.broker.effects == []  # the cycle stopped there
    assert _invocations(world) == []  # a systemic fault inside an attempt records nothing


def test_an_interrupt_inside_a_frozen_tick_is_never_a_tenant_failure(world, monkeypatch):
    monkeypatch.setattr("algua.live.frozen_dispatch.launch_child",
                        _raise(KeyboardInterrupt())(world))

    result = runner.invoke(app, ["paper", "run-all", "--snapshot", "snap1"])

    assert result.exit_code != 0
    assert "setup_error" not in result.stdout
    assert world.ticks(SIBLING) == [] and world.broker.effects == []
    assert all(action != "strategy_setup_error" for action, _ in world.audit(TENANT))
    assert _invocations(world) == []


# --- the code registry ---------------------------------------------------------------------------


@pytest.mark.parametrize("code", sorted(FROZEN_FAILURE_CODES))
def test_every_frozen_failure_code_is_stable_non_retryable_and_deployment_bound(code):
    failure = FrozenTenantFailure(code, 5, "diagnostic")

    assert error_code(failure) == code and is_retryable(code) is False
    setup = StrategySetupError("s", failure)
    assert setup.code == code
    assert setup.entry() == {"ok": False, "strategy": "s", "kind": "setup_error",
                             "error": code, "deployment_id": 5}


@pytest.mark.parametrize("exc", [
    FrozenContentUnavailable(deployment_id=4), FrozenTenantUnsupported("x", deployment_id=4),
    FrozenLiveUnsupported(4),
], ids=lambda exc: type(exc).__name__)
def test_registry_frozen_refusals_resolve_to_their_own_codes(exc):
    assert error_code(exc) == exc.code and is_retryable(exc.code) is False
    assert StrategySetupError("s", exc).entry() == {
        "ok": False, "strategy": "s", "kind": "setup_error", "error": exc.code,
        "deployment_id": 4}


def test_any_other_setup_error_keeps_todays_redacted_entry():
    setup = StrategySetupError("s", ValueError("/secret/path"))

    assert setup.entry() == {"ok": False, "strategy": "s", "kind": "setup_error",
                             "error": "ValueError"}


def test_every_frozen_code_is_documented_in_the_error_envelope_contract():
    text = (REPO / "docs/contracts/cli-error-envelope.md").read_text()

    assert [code for code in sorted(FROZEN_FAILURE_CODES) if f"`{code}`" not in text] == []
    for name in ("FrozenTenantFailure", "FrozenContentUnavailable", "FrozenTenantUnsupported",
                 "FrozenLiveUnsupported"):
        assert f"`{name}`" in text


# --- run_tick never plans a supervisor view in process ------------------------------------------


def test_run_tick_refuses_to_plan_a_strategy_without_planner_code_in_process():
    """A frozen tenant's supervisor view carries no planner code. Without a planner port,
    ``run_tick`` refuses it before any position read, bar fetch or venue call, rather than routing
    it through the in-process planner."""
    view = FrozenStrategyView(in_process(STRATEGY).config)
    calls: list[str] = []
    broker = SimpleNamespace(get_positions=lambda: calls.append("positions") or {})
    provider = SimpleNamespace(get_bars=lambda *args: calls.append("bars"))

    with pytest.raises(TypeError, match="planner"):
        run_tick(view, broker, provider, NOW, NOW, now=NOW)
    assert calls == []
