"""CLI integration tests for --delistings / --assume-terminal-last-close flags.

Task 9 of issue #212: survivorship-free CLI wiring.

* backtest run/walk-forward/sweep accept --delistings NAME
* research promote accepts --delistings but rejects --assume-terminal-last-close (human-only)
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from algua.cli.main import app
from algua.data.store import DataStore
from tests._human_actor_helpers import install_human_actor_anchor, promote_signed

runner = CliRunner()

# --- #329 authenticated --actor human plumbing ---------------------------------------------------
_HUMAN_KEY = None
_TMP_PATH = None


def _promote(args):
    """Drop-in for `runner.invoke(app, args)` at gated promote call sites: when `args` requests
    `--actor human`, run the signed challenge dance; otherwise a plain invoke."""
    if "human" in args:
        return promote_signed(runner, app, args, _HUMAN_KEY, _TMP_PATH)
    return runner.invoke(app, args)

STRATEGY = "cross_sectional_momentum"


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    global _HUMAN_KEY, _TMP_PATH
    _HUMAN_KEY = install_human_actor_anchor(monkeypatch, tmp_path)
    _TMP_PATH = tmp_path


def _seed_delistings(tmp_path):
    """Import a minimal delistings CSV into the DataStore (as-of well before backtest end dates)."""
    csv = tmp_path / "d.csv"
    csv.write_text("symbol,delisting_date,delisting_value\nB,2020-01-02,5.0\n")
    r = runner.invoke(app, [
        "data", "import-delistings", "--file", str(csv), "--source", "v",
        "--as-of", "2019-12-31T00:00:00+00:00",
    ])
    assert r.exit_code == 0, r.output


def test_backtest_run_accepts_delistings_flag(tmp_path):
    _seed_delistings(tmp_path)
    res = runner.invoke(app, [
        "backtest", "run", STRATEGY, "--demo",
        "--start", "2020-01-01", "--end", "2020-03-01", "--delistings", "vendor",
    ])
    assert res.exit_code == 0, res.output
    assert json.loads(res.output)["ok"] is True


def test_backtest_run_delisting_snapshot_is_actual_snapshot_id(tmp_path):
    """delisting_snapshot in the result JSON must be the real snapshot_id, not the user label."""
    _seed_delistings(tmp_path)
    # Discover the actual snapshot_id from the store
    store = DataStore(tmp_path)
    real_snapshot_id = store.latest_delistings_snapshot_id()
    assert real_snapshot_id is not None, "seed should have written a delistings snapshot"

    res = runner.invoke(app, [
        "backtest", "run", STRATEGY, "--demo",
        "--start", "2020-01-01", "--end", "2020-03-01", "--delistings", "ANY",
    ])
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    # Must be the actual snapshot_id, not the user-supplied label "ANY"
    assert payload.get("delisting_snapshot") == real_snapshot_id
    assert payload.get("delisting_snapshot") != "ANY"


def test_backtest_run_accepts_assume_terminal_last_close(tmp_path):
    _seed_delistings(tmp_path)
    res = runner.invoke(app, [
        "backtest", "run", STRATEGY, "--demo",
        "--start", "2020-01-01", "--end", "2020-03-01",
        "--delistings", "vendor", "--assume-terminal-last-close",
    ])
    assert res.exit_code == 0, res.output
    assert json.loads(res.output)["ok"] is True


def test_backtest_walk_forward_accepts_delistings_flag(tmp_path):
    _seed_delistings(tmp_path)
    res = runner.invoke(app, [
        "backtest", "walk-forward", STRATEGY, "--demo",
        "--start", "2020-01-01", "--end", "2022-01-01", "--delistings", "vendor",
    ])
    assert res.exit_code == 0, res.output
    assert json.loads(res.output)["ok"] is True


def test_backtest_sweep_accepts_delistings_flag(tmp_path):
    _seed_delistings(tmp_path)
    res = runner.invoke(app, [
        "backtest", "sweep", STRATEGY, "--demo",
        "--start", "2020-01-01", "--end", "2022-01-01",
        "--delistings", "vendor",
        "--param", "lookback=20,40",
    ])
    assert res.exit_code == 0, res.output
    assert json.loads(res.output)["ok"] is True


def test_backtest_run_no_delistings_snapshot_is_json_error(tmp_path):
    """--delistings with no ingested snapshot fails closed (ValueError -> JSON error)."""
    # no seed — nothing in the store
    res = runner.invoke(app, [
        "backtest", "run", STRATEGY, "--demo",
        "--start", "2020-01-01", "--end", "2020-03-01", "--delistings", "vendor",
    ])
    assert res.exit_code == 1
    assert json.loads(res.output)["ok"] is False


def test_research_promote_rejects_assume_terminal_last_close_for_agent():
    """--assume-terminal-last-close is human-only: agent path must fail closed early."""
    res = runner.invoke(app, [
        "research", "promote", STRATEGY,
        "--universe", "U",
        "--start", "2020-01-01", "--end", "2020-12-31",
        "--assume-terminal-last-close",
        # default --actor is "agent"
    ])
    assert res.exit_code != 0
    out = res.output.lower()
    assert "human" in out or "human-only" in out or "not allowed" in out


def test_research_promote_fdr_throttle_override_flag_removed():
    """Factory soft gate: --fdr-throttle-override was REMOVED with the FDR freeze — the CLI must
    reject it as an UNKNOWN option (usage error), for every actor, not route it anywhere."""
    for actor_args in ([], ["--actor", "human"]):
        res = runner.invoke(app, [
            "research", "promote", STRATEGY,
            "--universe", "U",
            "--start", "2020-01-01", "--end", "2020-12-31",
            "--fdr-throttle-override",
            *actor_args,
        ])
        assert res.exit_code == 2  # usage error: unknown option
        assert "no such option" in res.output.lower() or "unexpected" in res.output.lower()


def test_promote_signed_run_context_no_longer_contains_fdr_throttle_override(monkeypatch):
    """The #329 canonical run_context (the exact input set a human signature binds) no longer
    carries the removed fdr_throttle_override key — old signatures are nonce-bound and die
    naturally; no compat key is kept."""
    import algua.cli.research_cmd as rc

    captured: dict = {}
    orig = rc.canonical_run_context

    def _capture(d):
        captured.update(d)
        return orig(d)

    monkeypatch.setattr(rc, "canonical_run_context", _capture)
    add = runner.invoke(app, ["registry", "add", STRATEGY])
    assert add.exit_code == 0, add.output
    # Fails later (stage is idea, not backtested) — but AFTER the run_context was built.
    res = runner.invoke(app, [
        "research", "promote", STRATEGY, "--demo",
        "--start", "2020-01-01", "--end", "2020-12-31",
    ])
    assert res.exit_code != 0
    assert captured, "canonical_run_context was not reached"
    assert "fdr_throttle_override" not in captured
    assert "allow_non_pit" in captured  # the rest of the signed input set is intact


def test_research_promote_human_actor_can_pass_assume_terminal_last_close(tmp_path, monkeypatch):
    """With --actor human the flag is accepted (not rejected at the human-only guard).

    This test only verifies the guard does NOT fire for a human — the subsequent
    promote will fail for other reasons (no registered strategy, no breadth, etc.),
    which is fine as long as the failure is NOT the human-only guard message.
    """
    res = _promote([
        "research", "promote", STRATEGY,
        "--start", "2020-01-01", "--end", "2020-12-31",
        "--demo",
        "--assume-terminal-last-close",
        "--actor", "human",
        "--n-combos", "1",
        "--allow-non-pit",
    ])
    # The command may fail (no strategy registered, etc.) but the failure must NOT
    # be the human-only guard (i.e. must not contain "human-only" from our new guard).
    # If it DID fail with our new guard message, that would be wrong.
    # (Other failures — StrategyNotFound, etc. — are fine and expected here.)
    if res.exit_code != 0:
        payload = json.loads(res.output)
        # The error should NOT be our assume-terminal-last-close rejection
        assert "assume-terminal-last-close is human-only" not in payload.get("error", "").lower()
