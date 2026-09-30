"""Story 1.3d §5 (CAP-4): `paper promote` qualifies a FROZEN deployment without the checkout.

The frozen tenant of tests/_frozen_paper_world.py is admitted by the real ``paper intake`` against a
really published bundle (verified offline for real at promotion; only the environment's offline
check is the world's injected locator check). Its forward evidence is a window of ticks written
through the REAL tick writer, each linking a successful final invocation recorded through the real
recorder, exactly the shape the v48 trigger admits. ``paper promote`` then takes the identity from
the recorded descriptor, verifies the bundle and environment with a FRESH verifier once per
promotion, and runs the unchanged forward gate. The checkout module is armed to raise and every
checkout identity hash is a tripwire, so a pass here is a pass from recorded evidence alone.

A second fixture publishes a small bundle AND environment that both verify for real (the Story
1.3b stores, as tests/test_frozen_runtime.py does), for the environment's permission-drift and
replacement refusals the world's injected locator check cannot see.
"""
from __future__ import annotations

import dataclasses
import json
import os
import shutil
import sys
from contextlib import closing
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from algua.cli.main import app
from algua.config.settings import get_settings
from algua.execution.tick_snapshots import record_tick_snapshot
from algua.registry.artifact_recording import (
    frozen_deployment_manifest,
    parse_frozen_deployment_manifest,
)
from algua.registry.db import connect, migrate
from algua.registry.db.frozen_evidence import TICK_FROZEN_LINK_TRIGGER
from algua.registry.store import SqliteStrategyRepository
from tests._frozen_evidence_helpers import record_final_invocation
from tests._frozen_harness import unseal
from tests._frozen_paper_world import (
    CHECKOUT_MODULE,
    CODE_HASH,
    CONFIG_HASH,
    DEP,
    GATE_NAME,
    SNAP,
    TENANT,
    build_world,
    runner,
    teardown_world,
)
from tests._human_actor_helpers import _sign as sign_challenge
from tests._human_actor_helpers import install_human_actor_anchor, promote_signed
from tests._venv_fixture import SITE_PACKAGES
from tests.test_forward_promotion import FakeCalendar

CHECKOUT_DOTTED = f"algua.strategies.momentum.{TENANT}"
RAISING_MODULE = "raise RuntimeError('the frozen tenant was imported from the checkout')\n"
DESCRIPTOR = (CODE_HASH, CONFIG_HASH, DEP)


@pytest.fixture
def world(monkeypatch, tmp_path):
    sys.modules.pop(CHECKOUT_DOTTED, None)
    try:
        yield build_world(monkeypatch, tmp_path)
    finally:
        teardown_world(tmp_path / "data")
        sys.modules.pop(CHECKOUT_DOTTED, None)


# --- evidence ----------------------------------------------------------------------------------


def _past_weekdays(n: int) -> list[date]:
    """The n weekdays strictly before today (UTC), oldest first (as tests/test_cli_paper.py)."""
    out, day = [], datetime.now(UTC).date() - timedelta(days=1)
    while len(out) < n:
        if day.weekday() < 5:
            out.append(day)
        day -= timedelta(days=1)
    return list(reversed(out))


def _at(day: date) -> str:
    return datetime(day.year, day.month, day.day, 20, tzinfo=UTC).isoformat()


def _previous_weekday(day: date) -> date:
    day -= timedelta(days=1)
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day


def _tick_values(strategy_id: int, deployment_id: int, day: date, equity: float) -> dict:
    return {
        "tick_ts": _at(day), "decision_ts": _at(_previous_weekday(day)), "equity": equity,
        "peak_equity": None, "positions": {}, "n_submitted": 0, "reconcile_ok": True,
        "lane": "paper", "strategy_id": strategy_id, "code_hash": CODE_HASH,
        "config_hash": CONFIG_HASH, "dependency_hash": DEP, "account_id": "frozen-acct",
        "cash": 0.0, "clock_source": "broker", "snapshot_id": SNAP,
        "deployment_id": deployment_id,
    }


def seed_linked_window(world, *, n: int = 64, unlinked: int = 0) -> None:
    """``n`` sessions of linked frozen ticks (n-1 returns, a gently rising non-degenerate path)
    plus a qualified holdout row for the DESCRIPTOR identity (bar = max(.5 * 1.0, .3)).
    ``unlinked`` 1.3c-era ticks (no link, written with the trigger dropped) carry an equity that
    would wreck the path if they were ever counted."""
    deployment = world.deployment()
    with world.conn() as conn:
        equity = 100.0
        for i, day in enumerate(_past_weekdays(n)):
            link = record_final_invocation(conn, deployment_id=deployment.id, snapshot_id=SNAP)
            record_tick_snapshot(
                conn, TENANT, **_tick_values(deployment.strategy_id, deployment.id, day, equity),
                frozen_invocation_id=link)
            equity *= 1.004 if i % 2 == 0 else 0.999
        conn.execute("DROP TRIGGER tick_snapshots_frozen_link")
        for day in _past_weekdays(n)[-unlinked:] if unlinked else []:
            conn.execute(
                "INSERT INTO tick_snapshots(strategy, tick_ts, decision_ts, equity, positions,"
                " n_submitted, reconcile_ok, lane, strategy_id, code_hash, config_hash,"
                " dependency_hash, account_id, cash, clock_source, recorded_at, snapshot_id,"
                " deployment_id) VALUES (?,?,?,1.0,'{}',0,1,'paper',?,?,?,?,'frozen-acct',0.0,"
                " 'broker',?,?,?)",
                (TENANT, _at(day), _at(_previous_weekday(day)), deployment.strategy_id,
                 *DESCRIPTOR, datetime.now(UTC).isoformat(), SNAP, deployment.id))
        conn.execute(TICK_FROZEN_LINK_TRIGGER)
        conn.commit()
        SqliteStrategyRepository(conn).record_gate_evaluation(
            deployment.strategy_id, passed=True, n_funnel=1, own_lifetime_combos=1,
            windowed_total_combos=1, funnel_window_days=90, breadth_provenance="measured",
            pit_ok=True, pit_override=False, holdout_n_bars=63, min_holdout_observations=63,
            code_hash=CODE_HASH, config_hash=CONFIG_HASH, dependency_hash=DEP,
            data_source="test", snapshot_id="snap", period_start="2024-01-01",
            period_end="2024-12-31", holdout_frac=0.2, actor="human",
            decision_json=json.dumps({"checks": [{"name": "holdout_sharpe", "value": 1.0}]}),
            universe_name=GATE_NAME)


# --- the promotion invocation ------------------------------------------------------------------


def promote(*args: str, human_key: Path | None = None, tmp_path: Path | None = None):
    """``paper promote TENANT`` on weekday session arithmetic (the gate's calendar only), agent
    by default; with ``human_key`` the full sign-and-resubmit ceremony."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("algua.cli.paper_cmd.get_calendar", lambda: FakeCalendar())
        base = ["paper", "promote", TENANT, *args]
        if human_key is not None:
            assert tmp_path is not None
            return promote_signed(runner, app, [*base, "--actor", "human"], human_key, tmp_path)
        return runner.invoke(app, base)


class Tripwires:
    """Every checkout identity hash and checkout strategy import a promotion could reach."""

    TARGETS = (
        "algua.registry.approvals.compute_artifact_hashes",
        "algua.registry.forward_promotion.compute_artifact_hashes",
        "algua.cli.paper_cmd.compute_artifact_hashes",
        "algua.registry.transitions._compute_hashes",
        "algua.registry.approvals.load_strategy",
    )

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.calls: list[str] = []
        for target in self.TARGETS:
            monkeypatch.setattr(target, self._tripwire(target))
        CHECKOUT_MODULE.write_text(RAISING_MODULE)

    def _tripwire(self, label: str):
        def _fn(*_args, **_kwargs):
            self.calls.append(label)
            raise AssertionError(f"{label} touched the checkout for a frozen deployment")
        return _fn


class CountingEnvironmentCheck:
    """Counts the offline environment verifications (the expensive step) behind the world's
    injected check, which stays in place."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from algua.registry import frozen_runtime

        self.calls = 0
        real = frozen_runtime.verify_published_environment

        def counted(store_root, descriptor):
            self.calls += 1
            return real(store_root, descriptor)

        monkeypatch.setattr(frozen_runtime, "verify_published_environment", counted)


def _state(world) -> dict[str, Any]:
    """Every row a promotion attempt could leave behind."""
    tenant = world.deployment().strategy_id
    return {
        "stage": world.rows("SELECT stage FROM strategies WHERE id=?", tenant)[0]["stage"],
        "transitions": world.rows(
            "SELECT from_stage, to_stage, code_hash FROM stage_transitions WHERE strategy_id=?"
            " ORDER BY id", tenant),
        "forward_rows": world.rows("SELECT * FROM forward_gate_evaluations ORDER BY id"),
        "challenges": world.rows("SELECT * FROM actor_challenges"),
        "promote_audits": world.rows("SELECT * FROM audit_log WHERE action='paper_promote'"),
    }


# --- success from linked evidence --------------------------------------------------------------


def test_paper_promote_qualifies_a_frozen_deployment_from_linked_evidence_alone(
    world, monkeypatch,
):
    seed_linked_window(world, unlinked=2)
    deployment = world.deployment()
    tripwires = Tripwires(monkeypatch)
    environments = CountingEnvironmentCheck(monkeypatch)

    result = promote()

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["ok"] is True and payload["passed"] is True and payload["promoted"] is True
    # The 1.3c-era unlinked ticks are excluded and counted; nothing else is.
    assert payload["excluded_ticks"]["invocation_unlinked"] == 2
    assert sum(payload["excluded_ticks"].values()) == 2
    state = _state(world)
    assert state["stage"] == "forward_tested"
    [row] = state["forward_rows"]
    assert (row["code_hash"], row["config_hash"], row["dependency_hash"]) == DESCRIPTOR
    assert row["deployment_id"] == deployment.id
    assert (row["passed"], row["consumed"], row["actor"]) == (1, 1, "agent")
    assert (row["n_forward_observations"], row["account_id"]) == (63, "frozen-acct")
    assert state["transitions"][-1] == {
        "from_stage": "paper", "to_stage": "forward_tested", "code_hash": CODE_HASH}
    # Identity came from the descriptor; content was verified once for this promotion.
    assert tripwires.calls == [] and CHECKOUT_DOTTED not in sys.modules
    assert environments.calls == 1


def test_a_human_challenge_binds_the_descriptor_hashes(world, monkeypatch, tmp_path):
    seed_linked_window(world)
    key = install_human_actor_anchor(monkeypatch, tmp_path)
    tripwires = Tripwires(monkeypatch)

    challenge = promote("--actor", "human")

    assert challenge.exit_code == 0, challenge.stdout
    issued = json.loads(challenge.stdout)
    assert issued["action"] == "human_actor_challenge"
    assert all(value in issued["challenge"] for value in DESCRIPTOR)
    # A frozen challenge names the exact epoch and the content its descriptor verified.
    deployment = world.deployment()
    assert f'"deployment_id":{deployment.id}' in issued["challenge"]
    assert f'"manifest_digest":"{deployment.manifest_digest}"' in issued["challenge"]
    [pending] = world.rows("SELECT * FROM actor_challenges")
    assert (pending["code_hash"], pending["config_hash"], pending["dependency_hash"]) == DESCRIPTOR
    assert world.rows("SELECT * FROM forward_gate_evaluations") == []

    signed = promote(human_key=key, tmp_path=tmp_path)

    assert signed.exit_code == 0, signed.stdout
    payload = json.loads(signed.stdout)
    assert payload["passed"] is True and payload["promoted"] is True
    [row] = world.rows("SELECT actor, code_hash, config_hash, dependency_hash"
                       " FROM forward_gate_evaluations")
    assert row == {"actor": "human", "code_hash": CODE_HASH, "config_hash": CONFIG_HASH,
                   "dependency_hash": DEP}
    assert tripwires.calls == [] and CHECKOUT_DOTTED not in sys.modules


def _successor_epoch(world, *, same_artifact: bool) -> int:
    """Retire the frozen epoch and activate a successor with IDENTICAL hashes and verifying
    content: on the same artifact, or on a new descriptor (another source ref, so another manifest
    digest) of the same bundle and environment. Returns the successor's id."""
    old = world.deployment()
    with world.conn() as conn:
        artifact_id = old.artifact_id
        if not same_artifact:
            frozen = dataclasses.replace(
                parse_frozen_deployment_manifest(old.manifest()), source_ref="c" * 40)
            fields = dataclasses.asdict(frozen_deployment_manifest(frozen))
            artifact_id = conn.execute(
                f"INSERT INTO deployment_artifacts({', '.join(fields)}, created_at)"
                f" VALUES ({', '.join('?' * len(fields))}, 't')",
                tuple(fields.values())).lastrowid
        gate_id = SqliteStrategyRepository(conn).record_gate_evaluation(
            old.strategy_id, passed=True, n_funnel=1, own_lifetime_combos=1,
            windowed_total_combos=1, funnel_window_days=90, breadth_provenance="measured",
            pit_ok=True, pit_override=False, holdout_n_bars=63, min_holdout_observations=63,
            code_hash=CODE_HASH, config_hash=CONFIG_HASH, dependency_hash=DEP,
            data_source="test", snapshot_id="snap-2", period_start="2024-01-01",
            period_end="2024-12-31", holdout_frac=0.2, actor="human", decision_json="{}",
            universe_name=GATE_NAME)
        now = datetime.now(UTC).isoformat()
        conn.execute("UPDATE strategy_deployments SET retired_at=? WHERE id=?", (now, old.id))
        successor = conn.execute(
            "INSERT INTO strategy_deployments(strategy_id, artifact_id, research_gate_id,"
            " activated_at) VALUES (?,?,?,?)",
            (old.strategy_id, artifact_id, gate_id, now)).lastrowid
        conn.commit()
    assert successor is not None
    return int(successor)


@pytest.mark.parametrize("same_artifact", [True, False])
def test_a_human_signature_for_one_frozen_epoch_is_refused_on_another(
    world, monkeypatch, tmp_path, same_artifact,
):
    """Two frozen epochs can share the three hashes yet be different evidence epochs or run
    different content, so a frozen challenge also binds the deployment id and manifest digest: a
    signature over epoch F1 never authenticates a promotion of F2."""
    key = install_human_actor_anchor(monkeypatch, tmp_path)
    first = world.deployment()
    issued = json.loads(promote("--actor", "human").stdout)
    assert issued["action"] == "human_actor_challenge"
    signature = sign_challenge(key, issued["challenge"], tmp_path)
    second = _successor_epoch(world, same_artifact=same_artifact)
    successor = world.deployment()
    assert successor.id == second != first.id
    assert (successor.code_hash, successor.config_hash, successor.dependency_hash) == DESCRIPTOR
    assert (successor.manifest_digest == first.manifest_digest) is same_artifact

    result = promote("--actor", "human", "--actor-signature", str(signature))

    assert result.exit_code == 1, result.stdout
    payload = json.loads(result.stdout)
    assert payload["code"] == "invalid_input"
    assert "human actor authentication failed" in payload["error"]
    assert world.rows("SELECT * FROM forward_gate_evaluations") == []
    assert [row["consumed_at"] for row in world.rows("SELECT * FROM actor_challenges")] == [None]
    assert _state(world)["stage"] == "paper"


def test_a_promoted_frozen_deployment_keeps_ticking_and_refreshes_its_certificate(
    world, monkeypatch,
):
    seed_linked_window(world)
    CHECKOUT_MODULE.write_text(RAISING_MODULE)
    assert promote().exit_code == 0

    code, tick = world.trade_tick()  # forward_tested still ticks in the paper book

    assert code == 0, tick
    assert tick["ok"] is True
    newest = world.ticks(TENANT)[-1]
    assert newest["frozen_invocation_id"] is not None  # linked by construction

    refresh = promote()

    assert refresh.exit_code == 0, refresh.stdout
    payload = json.loads(refresh.stdout)
    assert payload["passed"] is True and payload["promoted"] is False
    assert payload["excluded_ticks"]["invocation_unlinked"] == 0
    rows = world.rows("SELECT passed, consumed, code_hash FROM forward_gate_evaluations"
                      " ORDER BY id")
    assert len(rows) == 2 and rows[1] == {"passed": 1, "consumed": 1, "code_hash": CODE_HASH}
    assert _state(world)["stage"] == "forward_tested"
    assert CHECKOUT_DOTTED not in sys.modules


def test_a_promoted_frozen_deployment_still_cannot_go_live(world, monkeypatch):
    seed_linked_window(world)
    CHECKOUT_MODULE.write_text(RAISING_MODULE)
    assert promote().exit_code == 0
    tripwires = Tripwires(monkeypatch)

    result = runner.invoke(app, ["registry", "transition", TENANT, "--to", "live",
                                 "--actor", "human"])

    assert result.exit_code == 1, result.stdout
    payload = json.loads(result.stdout)
    assert payload["code"] == "frozen_live_unsupported"
    assert payload["deployment_id"] == world.deployment().id
    assert world.rows("SELECT * FROM live_challenges") == []
    assert _state(world)["stage"] == "forward_tested"
    assert tripwires.calls == [] and CHECKOUT_DOTTED not in sys.modules


# --- content that does not verify refuses first ------------------------------------------------


def _corrupt(world) -> None:
    """Rewrite one bundle file's bytes, restoring every permission so only the content drifts."""
    target = world.bundle_root / "algua" / "__init__.py"
    target.parent.chmod(0o755)
    target.chmod(0o644)
    target.write_bytes(target.read_bytes() + b"\n# tampered\n")
    target.chmod(0o444)
    target.parent.chmod(0o555)


def _replace(world) -> None:
    """Swap the whole bundle directory for a different sealed tree at the same locator."""
    root = world.bundle_root
    replacement = root.with_name(root.name + ".replacement")
    shutil.copytree(root, replacement, symlinks=True)
    unseal(replacement)
    swapped = replacement / "algua" / "__init__.py"
    swapped.chmod(0o644)
    swapped.write_text("# replaced\n")
    for directory, _dirs, files in os.walk(replacement):
        for name in files:
            (Path(directory) / name).chmod(0o444)
    for directory, _dirs, _files in sorted(os.walk(replacement), reverse=True):
        Path(directory).chmod(0o555)
    root.parent.chmod(0o755)
    root.rename(root.with_name(root.name + ".original"))
    replacement.rename(root)


def _drift_permissions(world) -> None:
    world.bundle_root.chmod(0o755)


def _lose_environment(world) -> None:
    (world.data_dir.resolve() / world.environment.locator / "bin" / "python").unlink()


class RefusalTripwires:
    """Every step a content refusal must precede, in the CLI and the forward gate."""

    TARGETS = {
        "authenticate": "algua.cli.paper_cmd.authenticate_actor",
        "preflight": "algua.cli.paper_cmd.forward_promotion_preflight",
        "broker": "algua.cli.paper_cmd._alpaca_broker_from_settings",
        "assemble": "algua.registry.forward_promotion.assemble_forward_evidence",
        "evaluate": "algua.registry.forward_promotion.evaluate_forward_gate",
    }

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.calls: list[str] = []
        for label, target in self.TARGETS.items():
            monkeypatch.setattr(target, self._tripwire(label))
        for method in ("record_forward_gate_evaluation", "record_forward_pass_and_promote"):
            monkeypatch.setattr(SqliteStrategyRepository, method, self._tripwire(method))

    def _tripwire(self, label: str):
        def _fn(*_args, **_kwargs):
            self.calls.append(label)
            raise AssertionError(f"{label} ran before the content refusal")
        return _fn


@pytest.mark.parametrize("damage", [_corrupt, _replace, _drift_permissions, _lose_environment])
@pytest.mark.parametrize("actor", ["agent", "human"])
def test_unverifiable_content_refuses_before_authentication_or_any_row(
    world, monkeypatch, tmp_path, damage, actor,
):
    seed_linked_window(world)
    install_human_actor_anchor(monkeypatch, tmp_path)
    before = _state(world)
    damage(world)
    Tripwires(monkeypatch)
    refusals = RefusalTripwires(monkeypatch)

    result = promote("--actor", actor)

    assert result.exit_code == 1, result.stdout
    payload = json.loads(result.stdout)
    assert payload["code"] == "frozen_content_unavailable"
    assert payload["deployment_id"] == world.deployment().id
    assert payload["retryable"] is False
    assert refusals.calls == []
    assert _state(world) == before  # no evaluation row, challenge, audit, token or stage change


# --- a bundle AND environment that verify for real ---------------------------------------------


REAL = "s"


@pytest.fixture
def published(monkeypatch, tmp_path):
    """One frozen strategy on really published content (bundle and environment both verified by
    the Story 1.3b verifiers), with the CLI's data dir as the store root."""
    from tests.test_frozen_runtime import _admit, _config, _frozen, _publish

    content = _publish(tmp_path, (REAL,))
    monkeypatch.setenv("ALGUA_DATA_DIR", str(content.store))
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")

    class _Broker:
        def account_activities_window(self, after, until):
            return []

    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings", _Broker)
    conn = connect(get_settings().db_path)
    migrate(conn)
    with closing(conn):
        _admit(SqliteStrategyRepository(conn), REAL, _frozen(content, REAL, _config(REAL)))
    try:
        yield content
    finally:
        unseal(content.store / "frozen")


def _real_promote():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("algua.cli.paper_cmd.get_calendar", lambda: FakeCalendar())
        return runner.invoke(app, ["paper", "promote", REAL])


def _forward_rows() -> int:
    with closing(connect(get_settings().db_path)) as conn:
        return int(conn.execute("SELECT COUNT(*) FROM forward_gate_evaluations").fetchone()[0])


def test_each_promotion_verifies_content_fresh(published):
    """Verified content reaches the gate (no evidence yet, so it fails and records its row); the
    SAME process then refuses once the environment's permissions drift: no verdict is reused."""
    first = _real_promote()
    assert first.exit_code == 1, first.stdout
    assert json.loads(first.stdout)["passed"] is False and _forward_rows() == 1

    site = published.store / published.environment.locator / SITE_PACKAGES
    site.chmod(0o755)
    second = _real_promote()

    assert second.exit_code == 1, second.stdout
    assert json.loads(second.stdout)["code"] == "frozen_content_unavailable"
    assert _forward_rows() == 1


def test_a_corrupt_environment_is_refused(published):
    environment = published.store / published.environment.locator
    target = environment / SITE_PACKAGES / "six.py"
    unseal(environment)
    target.chmod(0o644)
    target.write_text("VERSION = '9.9.9'\n")
    target.chmod(0o444)
    for directory, _dirs, _files in sorted(os.walk(environment), reverse=True):
        Path(directory).chmod(0o555)

    result = _real_promote()

    assert result.exit_code == 1, result.stdout
    assert json.loads(result.stdout)["code"] == "frozen_content_unavailable"
    assert _forward_rows() == 0


def test_unsupported_recorded_content_is_refused(monkeypatch, tmp_path):
    """A descriptor whose recorded config names another strategy is content this supervisor
    cannot run: ``frozen_content_unsupported``, before any row."""
    from tests.test_frozen_runtime import _admit, _config, _frozen, _publish

    content = _publish(tmp_path, (REAL,))
    monkeypatch.setenv("ALGUA_DATA_DIR", str(content.store))
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        _admit(SqliteStrategyRepository(conn), REAL, _frozen(content, REAL, _config("impostor")))
    try:
        result = _real_promote()
    finally:
        unseal(content.store / "frozen")

    assert result.exit_code == 1, result.stdout
    payload = json.loads(result.stdout)
    assert payload["code"] == "frozen_content_unsupported" and payload["deployment_id"] is not None
    assert _forward_rows() == 0
