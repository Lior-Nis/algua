from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

import algua.strategies.momentum as momentum_pkg
from algua.backtest._sample import SyntheticProvider
from algua.backtest.engine import BacktestError
from algua.contracts.lifecycle import Actor, Stage, TransitionError
from algua.registry.db import connect, migrate
from algua.registry.promotion import (
    BreadthContext,
    guard_agent_relaxations,
    promotion_preflight,
    resolve_pit_ok,
    run_gate,
)
from algua.registry.store import SqliteStrategyRepository
from algua.research.dsr import effective_funnel_breadth
from algua.research.gates import FUNNEL_WINDOW_DAYS, GateCriteria

_START = datetime(2024, 1, 1, tzinfo=UTC)
_END = datetime(2024, 6, 1, tzinfo=UTC)


def _repo(tmp_path):
    conn = connect(tmp_path / "p.db")
    migrate(conn)
    return SqliteStrategyRepository(conn)


def test_guard_agent_relaxations_refuses_agent():
    for kw in (dict(declared_combos=9, allow_holdout_reuse=False, allow_non_pit=False),
               dict(declared_combos=None, allow_holdout_reuse=True, allow_non_pit=False),
               dict(declared_combos=None, allow_holdout_reuse=False, allow_non_pit=True)):
        with pytest.raises(ValueError, match="human"):
            guard_agent_relaxations(Actor.AGENT, **kw)


def test_guard_allows_clean_agent_and_any_human():
    guard_agent_relaxations(Actor.AGENT, declared_combos=None, allow_holdout_reuse=False,
                            allow_non_pit=False)
    guard_agent_relaxations(Actor.HUMAN, declared_combos=9, allow_holdout_reuse=True,
                            allow_non_pit=True)


def test_resolve_pit_ok_requires_coverage():
    cover = [{"snapshot_id": "u1", "effective_date": "2021-06-01"}]
    late = [{"snapshot_id": "u1", "effective_date": "2022-06-01"}]
    assert resolve_pit_ok("sp", cover, date(2022, 1, 1)) is True
    assert resolve_pit_ok("sp", late, date(2022, 1, 1)) is False
    assert resolve_pit_ok(None, None, date(2022, 1, 1)) is False


def test_resolve_pit_ok_fails_closed_on_malformed_snapshot():
    # A malformed/missing effective_date must NOT raise (the holdout was already recorded) — it
    # fails closed to non-PIT (not promotable).
    missing = [{"snapshot_id": "u1"}]  # KeyError
    bad_format = [{"snapshot_id": "u1", "effective_date": "not-a-date"}]  # ValueError
    wrong_type = [{"snapshot_id": "u1", "effective_date": None}]  # TypeError
    assert resolve_pit_ok("sp", missing, date(2022, 1, 1)) is False
    assert resolve_pit_ok("sp", bad_format, date(2022, 1, 1)) is False
    assert resolve_pit_ok("sp", wrong_type, date(2022, 1, 1)) is False


@pytest.mark.parametrize("stages", [
    (),                                              # idea
    (Stage.BACKTESTED, Stage.CANDIDATE),           # candidate
    (Stage.BACKTESTED, Stage.CANDIDATE, Stage.PAPER),  # paper (PAPER->CANDIDATE is legal!)
])
def test_preflight_refuses_non_backtested_source(tmp_path, stages):
    repo = _repo(tmp_path)
    rec = repo.add("alpha")
    repo.record_search_trial("alpha", 4, "{}")  # measured breadth present (so stage is the refusal)
    for s in stages:
        rec = repo.apply_transition(rec, s, Actor.HUMAN, "setup")
    with pytest.raises(TransitionError, match="backtested"):
        promotion_preflight(repo, "alpha", actor=Actor.AGENT, declared_combos=None,
                            allow_holdout_reuse=False, allow_non_pit=False,
                            provider=SyntheticProvider(seed=0), start=_START, end=_END)
    # Refused in preflight => no gate row, no holdout burn.
    assert repo._conn.execute("SELECT COUNT(*) c FROM gate_evaluations").fetchone()["c"] == 0
    assert repo._conn.execute("SELECT COUNT(*) c FROM holdout_evaluations").fetchone()["c"] == 0


def test_preflight_refuses_system_actor_before_any_holdout_or_gate_row(tmp_path):
    # SYSTEM is not a valid promote actor. The refusal is the FIRST check in preflight (before the
    # relaxation guard and before walk_forward), so no holdout is burned and no gate row is minted.
    repo = _repo(tmp_path)
    rec = repo.add("alpha")
    repo.apply_transition(rec, Stage.BACKTESTED, Actor.AGENT, "bt")
    repo.record_search_trial("alpha", 4, "{}")  # measured breadth present (isolate the actor check)
    with pytest.raises(ValueError, match="agent or human"):
        promotion_preflight(repo, "alpha", actor=Actor.SYSTEM, declared_combos=None,
                            allow_holdout_reuse=False, allow_non_pit=False,
                            provider=SyntheticProvider(seed=0), start=_START, end=_END)
    assert repo._conn.execute("SELECT COUNT(*) c FROM gate_evaluations").fetchone()["c"] == 0
    assert repo._conn.execute("SELECT COUNT(*) c FROM holdout_evaluations").fetchone()["c"] == 0


class _NonReproducibleProvider:
    """Neither snapshot_id nor a reproducible marker; get_bars must NOT be reached (the guard fires
    before any provider read)."""

    def get_bars(self, symbols, start, end, timeframe):
        raise AssertionError("provider must not be read: reproducible guard should fire first")


def test_agent_refused_non_reproducible_source(tmp_path):
    repo = _repo(tmp_path)
    repo.add("alpha")
    with pytest.raises(ValueError, match="reproducible data source"):
        promotion_preflight(repo, "alpha", actor=Actor.AGENT, declared_combos=None,
                            allow_holdout_reuse=False, allow_non_pit=False,
                            provider=_NonReproducibleProvider(), start=_START, end=_END)


def test_reproducible_guard_skipped_for_synthetic_agent_and_any_human(tmp_path):
    # SyntheticProvider is reproducible -> the guard does NOT fire for an agent (a later stage check
    # raises instead, since "alpha" is at stage idea). A human is exempt even for a non-reproducible
    # provider (get_bars is never reached because the stage check short-circuits first).
    repo = _repo(tmp_path)
    repo.add("alpha")
    with pytest.raises(Exception) as agent_ei:
        promotion_preflight(repo, "alpha", actor=Actor.AGENT, declared_combos=None,
                            allow_holdout_reuse=False, allow_non_pit=False,
                            provider=SyntheticProvider(seed=0), start=_START, end=_END)
    assert "reproducible data source" not in str(agent_ei.value)
    with pytest.raises(Exception) as human_ei:
        promotion_preflight(repo, "alpha", actor=Actor.HUMAN, declared_combos=None,
                            allow_holdout_reuse=False, allow_non_pit=False,
                            provider=_NonReproducibleProvider(), start=_START, end=_END)
    assert "reproducible data source" not in str(human_ei.value)


class _NonReproducibleWorkingProvider(SyntheticProvider):
    """A working (deterministic) provider that deliberately does NOT advertise reproducibility —
    stands in for a future mutable/live provider for the human-exemption test."""

    reproducible = False


def test_human_exempt_from_reproducible_guard_through_preflight(tmp_path):
    # A human may promote off a non-reproducible source: the guard is agent-only, so preflight runs
    # to COMPLETION (breadth resolved) rather than refusing — proving the exemption holds all the
    # way through, not just that it doesn't raise at the guard. ("alpha" is not a bundled module, so
    # signal-panel parity step is skipped and the provider is never read.)
    # Task 4 (#222): pass new_family_slug so the NOVEL verdict creates a family for this human.
    repo = _repo(tmp_path)
    rec = repo.add("alpha")
    repo.apply_transition(rec, Stage.BACKTESTED, Actor.HUMAN, "bt")
    repo.record_search_trial("alpha", 4, "{}")
    ctx = promotion_preflight(repo, "alpha", actor=Actor.HUMAN, declared_combos=None,
                              allow_holdout_reuse=False, allow_non_pit=False,
                              provider=_NonReproducibleWorkingProvider(), start=_START, end=_END,
                              new_family_slug="alpha_family")
    assert ctx.own == 4 and ctx.provenance == "measured"


def test_preflight_refuses_agent_without_measured_breadth(tmp_path):
    # Task 4 (#222): pre-assign strategy to a family so the clustering guard passes and the test
    # isolates the breadth check (the ORIGINAL intent of this test).
    repo = _repo(tmp_path)
    rec = repo.add("alpha")
    repo.apply_transition(rec, Stage.BACKTESTED, Actor.AGENT, "bt")
    fam_id = repo.create_family("alpha_family", actor="agent")
    repo.assign_strategy_to_family(
        "alpha", fam_id, "agent", verdict="NOVEL", similarity_score=0.0,
        clustering_version="v0", clustering_config_json="{}", axis_json="{}",
    )
    with pytest.raises(ValueError, match="search breadth"):
        promotion_preflight(repo, "alpha", actor=Actor.AGENT, declared_combos=None,
                            allow_holdout_reuse=False, allow_non_pit=False,
                            provider=SyntheticProvider(seed=0), start=_START, end=_END)


def test_preflight_resolves_measured_funnel_breadth(tmp_path):
    # Task 4 (#222): pre-assign strategy to a family so the clustering guard passes.
    repo = _repo(tmp_path)
    rec = repo.add("alpha")
    repo.apply_transition(rec, Stage.BACKTESTED, Actor.AGENT, "bt")
    repo.record_search_trial("alpha", 4, "{}")
    repo.record_search_trial("beta", 10, "{}")  # sibling effort raises the funnel bar
    fam_id = repo.create_family("alpha_family", actor="agent")
    repo.assign_strategy_to_family(
        "alpha", fam_id, "agent", verdict="NOVEL", similarity_score=0.0,
        clustering_version="v0", clustering_config_json="{}", axis_json="{}",
    )
    ctx = promotion_preflight(repo, "alpha", actor=Actor.AGENT, declared_combos=None,
                              allow_holdout_reuse=False, allow_non_pit=False,
                              provider=SyntheticProvider(seed=0), start=_START, end=_END)
    assert ctx.own == 4 and ctx.windowed_total == 14 and ctx.n_funnel == 14
    assert ctx.provenance == "measured"


def _write_tmp_strategy(filename: str, body: str) -> Path:
    """Write a temp strategy module into the momentum family dir so load_strategy can find it.
    Caller must unlink it in a finally block (mirrors tests/test_fast_path.py loader tests)."""
    p = Path(momentum_pkg.__path__[0]) / filename
    p.write_text(body)
    return p


def test_preflight_refuses_divergent_signal_panel_before_holdout(tmp_path):
    """A strategy whose signal_panel diverges from its per-bar signal is refused in preflight --
    before any holdout or gate row -- by the exhaustive parity gate (#178)."""
    mod = _write_tmp_strategy(
        "tmp_divergent_panel.py",
        "from typing import Any\n"
        "import pandas as pd\n"
        "from algua.contracts.types import ExecutionContract\n"
        "from algua.strategies.base import StrategyConfig\n"
        "CONFIG = StrategyConfig(name='tmp_divergent_panel', universe=['AAA', 'BBB'],\n"
        "    execution=ExecutionContract("
        "rebalance_frequency='1d', decision_lag_bars=1, warmup_bars=0),\n"
        "    construction='equal_weight_positive')\n"
        "def signal(view, params):\n"
        "    syms = sorted(view['symbol'].unique())\n"
        "    return pd.Series(1.0, index=syms)\n"
        "def signal_panel(bars, params):\n"
        "    adj = bars.reset_index().pivot("
        "index='timestamp', columns='symbol', values='adj_close')\n"
        "    out = pd.DataFrame(0.0, index=adj.index, columns=adj.columns)\n"
        "    out['AAA'] = 1.0\n"
        "    return out\n",
    )
    try:
        repo = _repo(tmp_path)
        rec = repo.add("tmp_divergent_panel")
        repo.apply_transition(rec, Stage.BACKTESTED, Actor.AGENT, "bt")
        repo.record_search_trial("tmp_divergent_panel", 4, "{}")
        with pytest.raises(BacktestError, match="parity"):
            promotion_preflight(
                repo, "tmp_divergent_panel", actor=Actor.AGENT, declared_combos=None,
                allow_holdout_reuse=False, allow_non_pit=False,
                provider=SyntheticProvider(seed=2), start=_START, end=_END)
        assert repo._conn.execute("SELECT COUNT(*) c FROM gate_evaluations").fetchone()["c"] == 0
        assert repo._conn.execute("SELECT COUNT(*) c FROM holdout_evaluations").fetchone()["c"] == 0
    finally:
        mod.unlink()


def test_preflight_passes_parity_for_faithful_bundled_strategy(tmp_path):
    """A real bundled strategy with a faithful signal_panel passes the parity gate and preflight
    resolves breadth as usual (no false positive)."""
    # Task 4 (#222): pre-assign strategy to a family so the clustering guard passes.
    repo = _repo(tmp_path)
    rec = repo.add("cross_sectional_momentum")
    repo.apply_transition(rec, Stage.BACKTESTED, Actor.AGENT, "bt")
    repo.record_search_trial("cross_sectional_momentum", 4, "{}")
    fam_id = repo.create_family("csm_family", actor="agent")
    repo.assign_strategy_to_family(
        "cross_sectional_momentum", fam_id, "agent", verdict="NOVEL", similarity_score=0.0,
        clustering_version="v0", clustering_config_json="{}", axis_json="{}",
    )
    ctx = promotion_preflight(
        repo, "cross_sectional_momentum", actor=Actor.AGENT, declared_combos=None,
        allow_holdout_reuse=False, allow_non_pit=False,
        provider=SyntheticProvider(seed=7), start=_START, end=_END)
    assert ctx.own == 4 and ctx.provenance == "measured"


# --- gated-universe subset guard (#559) --------------------------------------------------------
# cross_sectional_momentum's CONFIG.universe is the 5-symbol template incl. NVDA — exactly the
# divergence class that shipped: a module template wider than the universe the gate validated.

def test_preflight_refuses_config_universe_outside_gated_universe(tmp_path):
    """An AGENT promote on a PIT universe whose full membership timeline never contained a CONFIG
    symbol fails closed BEFORE the holdout is touched, naming the offending symbols (#559)."""
    repo = _repo(tmp_path)
    rec = repo.add("cross_sectional_momentum")
    repo.apply_transition(rec, Stage.BACKTESTED, Actor.AGENT, "bt")
    repo.record_search_trial("cross_sectional_momentum", 4, "{}")
    gated = {date(2020, 1, 1): ["AAPL", "MSFT", "AMZN", "GOOGL"]}  # no NVDA, ever
    with pytest.raises(ValueError, match=r"NVDA") as ei:
        promotion_preflight(
            repo, "cross_sectional_momentum", actor=Actor.AGENT, declared_combos=None,
            allow_holdout_reuse=False, allow_non_pit=False,
            provider=SyntheticProvider(seed=7), start=_START, end=_END,
            universe_name="liquid4", universe_by_date=gated)
    assert "liquid4" in str(ei.value)
    # Refused in preflight => no gate row, no holdout burn.
    assert repo._conn.execute("SELECT COUNT(*) c FROM gate_evaluations").fetchone()["c"] == 0
    assert repo._conn.execute("SELECT COUNT(*) c FROM holdout_evaluations").fetchone()["c"] == 0


def test_preflight_accepts_config_universe_subset_of_gated_timeline_union(tmp_path):
    """The guard checks membership against the UNION of the gated universe's timeline — a symbol
    that was a member at ANY effective date passes; preflight completes for a covered CONFIG."""
    repo = _repo(tmp_path)
    rec = repo.add("cross_sectional_momentum")
    repo.apply_transition(rec, Stage.BACKTESTED, Actor.AGENT, "bt")
    repo.record_search_trial("cross_sectional_momentum", 4, "{}")
    fam_id = repo.create_family("csm_family", actor="agent")
    repo.assign_strategy_to_family(
        "cross_sectional_momentum", fam_id, "agent", verdict="NOVEL", similarity_score=0.0,
        clustering_version="v0", clustering_config_json="{}", axis_json="{}",
    )
    gated = {
        date(2020, 1, 1): ["AAPL", "MSFT", "NVDA"],
        date(2021, 1, 1): ["AAPL", "MSFT", "AMZN", "GOOGL", "TSLA"],  # NVDA dropped; union covers
    }
    ctx = promotion_preflight(
        repo, "cross_sectional_momentum", actor=Actor.AGENT, declared_combos=None,
        allow_holdout_reuse=False, allow_non_pit=False,
        provider=SyntheticProvider(seed=7), start=_START, end=_END,
        universe_name="liquid5", universe_by_date=gated)
    assert ctx.own == 4 and ctx.provenance == "measured"


def test_human_exempt_from_gated_universe_subset_guard(tmp_path):
    """The subset guard is agent-only: a HUMAN promote with an uncovered CONFIG symbol still runs
    preflight to completion (mirrors the other human-accepts-the-cost exemptions)."""
    repo = _repo(tmp_path)
    rec = repo.add("cross_sectional_momentum")
    repo.apply_transition(rec, Stage.BACKTESTED, Actor.HUMAN, "bt")
    repo.record_search_trial("cross_sectional_momentum", 4, "{}")
    gated = {date(2020, 1, 1): ["AAPL", "MSFT", "AMZN", "GOOGL"]}  # no NVDA
    ctx = promotion_preflight(
        repo, "cross_sectional_momentum", actor=Actor.HUMAN, declared_combos=None,
        allow_holdout_reuse=False, allow_non_pit=False,
        provider=SyntheticProvider(seed=7), start=_START, end=_END,
        new_family_slug="csm_family",
        universe_name="liquid4", universe_by_date=gated)
    assert ctx.own == 4 and ctx.provenance == "measured"


# --- run_gate DSR binding (#211) ---------------------------------------------------------------
# run_gate calls compute_artifact_hashes(name) -> load_strategy(name), so these tests use a real
# bundled strategy ("cross_sectional_momentum"). The holdout Sharpe is set far above the deflated
# bar at the n_combos used, so the DSR check under test is the only thing that can flip `passed`.

_GATE_START = date(2024, 1, 1)
_GATE_END = date(2024, 6, 1)
_GATE_NAME = "cross_sectional_momentum"

# a walk-forward that passes everything-but-DSR; high Sharpe clears the breadth-deflated bar.
_GATE_HOLDOUT = {
    "sharpe": 7.0, "total_return": 0.2, "n_bars": 252, "skewness": 0.0, "kurtosis": 3.0}
_GATE_STAB = {"pct_positive_windows": 0.8, "min_sharpe": 0.1}


def _gate_wf(holdout=None, stability=None):
    from algua.backtest.walkforward import WalkForwardResult

    return WalkForwardResult(
        strategy=_GATE_NAME, config_hash="c", data_source="synthetic", snapshot_id=None,
        timeframe="1d", seed=None,
        period={"start": "2024-01-01", "end": "2024-06-01"}, windows=4, holdout_frac=0.2,
        window_metrics=[], holdout_metrics=holdout or dict(_GATE_HOLDOUT),
        stability=stability or dict(_GATE_STAB))


def _gate_repo(tmp_path):
    repo = _repo(tmp_path)
    rec = repo.add(_GATE_NAME)
    repo.apply_transition(rec, Stage.BACKTESTED, Actor.AGENT, "bt")
    return repo


def _breadth(repo, provenance: str, *, n: int = 5) -> BreadthContext:
    windowed_total = repo.windowed_search_combos(FUNNEL_WINDOW_DAYS)
    n_funnel = effective_funnel_breadth(n, windowed_total)
    return BreadthContext(n_funnel, n, windowed_total, provenance)


def _run(repo, breadth, wf=None):
    return run_gate(
        repo, wf or _gate_wf(), name=_GATE_NAME, actor=Actor.AGENT, criteria=GateCriteria(),
        breadth=breadth, universe_name=None, universe_snapshots=None,
        period_start=_GATE_START, period_end=_GATE_END, holdout_frac=0.2,
        data_source="synthetic", snapshot_id=None, allow_non_pit=True, reason_suffix="")


def test_run_gate_measured_binds_dsr(tmp_path):
    repo = _gate_repo(tmp_path)
    repo.record_search_trial(_GATE_NAME, 5, "{}", trial_sharpe_count=5,
                             trial_sharpe_mean=0.5, trial_sharpe_var_ann=0.04)
    breadth = _breadth(repo, "measured")
    outcome = _run(repo, breadth)
    d = outcome.decision
    assert d.dsr_binding is True
    assert any(c["name"] == "dsr_evidence" for c in d.checks)


def test_run_gate_declared_breadth_omits_dsr(tmp_path):
    repo = _gate_repo(tmp_path)
    breadth = _breadth(repo, "declared")
    outcome = _run(repo, breadth)
    d = outcome.decision
    assert d.dsr_binding is False
    assert all(c["name"] != "dsr_evidence" for c in d.checks)
    assert d.dsr_skip_reason == "no_measured_dispersion"


def test_run_gate_measured_but_missing_stats_records_failed_advisory(tmp_path):
    # Soft gate: a measured run with NULL trial stats still records a FAILED dsr_evidence check
    # (the gap is visible), marked advisory — it no longer vetoes the integrity-floor pass.
    repo = _gate_repo(tmp_path)
    repo.record_search_trial(_GATE_NAME, 5, "{}")  # measured row, NULL stats (pre-migration)
    breadth = _breadth(repo, "measured")
    outcome = _run(repo, breadth)
    d = outcome.decision
    assert d.dsr_binding is True
    dsr = next(c for c in d.checks if c["name"] == "dsr_evidence")
    assert dsr["passed"] is False and dsr["advisory"] is True
    assert d.passed is True
    assert outcome.promoted is True


# --- run_gate FDR freeze (factory soft gate, 2026-08-10 spec) ----------------------------
# The statistical stack is advisory, so the promote path NEVER writes a LORD++ binding row:
# every gate_evaluations row lands with fdr_binding NULL + fdr_skip_reason="stats_advisory",
# the stream reader and the windowed throttle count see nothing, and final_passed collapses to
# the provisional integrity-floor verdict. The ledger machinery itself is pinned alive by the
# direct-store tests in tests/test_registry_store.py.


def _wf_sharpe(sharpe: float):
    return _gate_wf(holdout={**_GATE_HOLDOUT, "sharpe": sharpe})


def _run_measured(repo, *, sharpe: float = 7.0):
    """run_gate with measured breadth and trial stats, variable holdout Sharpe."""
    breadth = _breadth(repo, "measured")
    return run_gate(
        repo, _wf_sharpe(sharpe), name=_GATE_NAME, actor=Actor.AGENT, criteria=GateCriteria(),
        breadth=breadth, universe_name=None, universe_snapshots=None,
        period_start=_GATE_START, period_end=_GATE_END, holdout_frac=0.2,
        data_source="synthetic", snapshot_id=None, allow_non_pit=True, reason_suffix="")


def test_run_gate_measured_promote_writes_no_binding_row(tmp_path):
    """A measured promote with full trial stats — the OLD maximal-FDR case — now records
    fdr_binding NULL + stats_advisory and never consults the stream/throttle."""
    import json as _json
    repo = _gate_repo(tmp_path)
    repo.record_search_trial(_GATE_NAME, 5, "{}", trial_sharpe_count=5,
                             trial_sharpe_mean=0.5, trial_sharpe_var_ann=0.04)
    outcome = _run_measured(repo, sharpe=7.0)
    d = outcome.decision
    assert d.dsr_binding is True                       # DSR still armed + recorded
    assert d.dsr_confidence is not None
    assert d.fdr_binding is False
    assert d.fdr_skip_reason == "stats_advisory"
    assert d.fdr_p_value is None and d.fdr_alpha_level is None and d.fdr_test_index is None
    assert outcome.promoted is True
    # No fdr_evidence / fdr_throttle checks anywhere — returned decision or stored row.
    assert all(c["name"] not in ("fdr_evidence", "fdr_throttle") for c in d.checks)
    row = repo._conn.execute(
        "SELECT fdr_binding, decision_json FROM gate_evaluations ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert row["fdr_binding"] is None                  # NULL, not 0 — invisible to the stream
    stored = _json.loads(row["decision_json"])
    assert stored["fdr_binding"] is False
    assert stored["fdr_skip_reason"] == "stats_advisory"
    assert all(c["name"] not in ("fdr_evidence", "fdr_throttle")
               for c in stored.get("checks", []))


def test_run_gate_stream_sees_zero_binding_rows_across_promotes(tmp_path):
    """N promotes (pass and fail) advance NEITHER the LORD++ stream nor the (now-deleted)
    windowed promotion-throttle count — the ledger stays frozen while stats are advisory."""
    repo = _gate_repo(tmp_path)
    repo.record_search_trial(_GATE_NAME, 5, "{}", trial_sharpe_count=5,
                             trial_sharpe_mean=0.5, trial_sharpe_var_ann=0.04)
    out1 = _run_measured(repo, sharpe=7.0)             # floor pass → promoted
    assert out1.promoted is True
    # Back-step to backtested and promote again (a legal candidate→backtested back-step).
    repo.apply_transition(repo.get(_GATE_NAME), Stage.BACKTESTED, Actor.HUMAN, "re-run")
    out2 = _run_measured(repo, sharpe=7.0)
    assert out2.promoted is True
    # A floor FAILURE row too (short holdout): commits a failing row, still non-binding.
    repo.apply_transition(repo.get(_GATE_NAME), Stage.BACKTESTED, Actor.HUMAN, "re-run")
    out3 = _run_measured(repo, sharpe=-1.0)            # holdout_sharpe_floor fails
    assert out3.promoted is False
    n_rows = repo._conn.execute("SELECT COUNT(*) c FROM gate_evaluations").fetchone()["c"]
    assert n_rows == 3                                 # every attempt is still audited
    # The stream reader sees an untouched (fresh) stream.
    stream = repo.fdr_stream_state()
    assert stream is not None
    assert stream.t == 1 and stream.binding_tests == 0 and stream.discoveries == 0


def test_run_gate_final_passed_equals_provisional(tmp_path):
    """final_passed == provisional integrity-floor verdict: no FDR term can flip it either way."""
    repo = _gate_repo(tmp_path)
    repo.record_search_trial(_GATE_NAME, 5, "{}", trial_sharpe_count=5,
                             trial_sharpe_mean=0.5, trial_sharpe_var_ann=0.04)
    outcome = _run_measured(repo, sharpe=7.0)
    d = outcome.decision
    assert d.passed is True
    assert d.passed == all(c["passed"] for c in d.checks if not c.get("advisory"))
    assert outcome.promoted is d.passed


def test_run_gate_weak_evidence_stats_fail_advisory_still_promotes(tmp_path):
    """Sharpe=2.3 (weak evidence the OLD LORD++ gate rejected): every floor check passes, the
    stats record whatever they record, and the strategy PROMOTES — market-first selection."""
    repo = _gate_repo(tmp_path)
    repo.record_search_trial(_GATE_NAME, 5, "{}", trial_sharpe_count=5,
                             trial_sharpe_mean=0.5, trial_sharpe_var_ann=0.04)
    outcome = _run_measured(repo, sharpe=2.3)
    d = outcome.decision
    assert d.fdr_binding is False and d.fdr_skip_reason == "stats_advisory"
    assert outcome.promoted is True
    assert repo.get(_GATE_NAME).stage is Stage.CANDIDATE


def test_run_gate_declared_breadth_also_stats_advisory(tmp_path):
    """Declared breadth (no DSR armed) records the SAME stats_advisory skip reason — the FDR
    freeze is unconditional, not a dsr_binding branch."""
    repo = _gate_repo(tmp_path)
    breadth = _breadth(repo, "declared")
    outcome = _run(repo, breadth)
    d = outcome.decision
    assert d.dsr_binding is False
    assert d.dsr_skip_reason == "no_measured_dispersion"   # the DSR reason is unchanged
    assert d.fdr_binding is False
    assert d.fdr_skip_reason == "stats_advisory"
    assert all(c["name"] != "fdr_evidence" for c in d.checks)


def test_run_gate_fdr_freeze_in_to_dict(tmp_path):
    """to_dict() carries the frozen-FDR audit state for downstream CLI serialisation."""
    repo = _gate_repo(tmp_path)
    repo.record_search_trial(_GATE_NAME, 5, "{}", trial_sharpe_count=5,
                             trial_sharpe_mean=0.5, trial_sharpe_var_ann=0.04)
    outcome = _run_measured(repo, sharpe=7.0)
    d_dict = outcome.decision.to_dict()
    assert d_dict["fdr_binding"] is False
    assert d_dict["fdr_skip_reason"] == "stats_advisory"
    assert d_dict["fdr_rejected"] is None
    assert d_dict["fdr_test_index"] is None
    assert d_dict["fdr_p_value"] is None


def test_dsr_below_threshold_records_failed_advisory_and_promotes(tmp_path):
    """A run whose dsr_confidence < 0.95 records the FAILED dsr_evidence check (advisory) and
    still promotes on the floor — the DSR verdict is telemetry for the audit trail now."""
    repo = _gate_repo(tmp_path)
    # Larger trial dispersion (var_ann=0.8) pushes dsr_confidence below 0.95 while holdout Sharpe
    # 2.5 still clears the (advisory) breadth-deflated bar (~2.29).
    repo.record_search_trial(_GATE_NAME, 5, "{}", trial_sharpe_count=5,
                             trial_sharpe_mean=0.5, trial_sharpe_var_ann=0.8)
    outcome = _run_measured(repo, sharpe=2.5)
    d = outcome.decision
    assert d.dsr_binding is True
    assert d.dsr_confidence is not None and d.dsr_confidence < 0.95
    hs_checks = [c for c in d.checks if c["name"] == "holdout_sharpe"]
    assert len(hs_checks) == 1 and hs_checks[0]["passed"] is True  # breadth bar still clears
    dsr_checks = [c for c in d.checks if c["name"] == "dsr_evidence"]
    assert len(dsr_checks) == 1 and dsr_checks[0]["passed"] is False
    assert dsr_checks[0]["advisory"] is True
    assert outcome.promoted is True
    assert d.passed == all(c["passed"] for c in d.checks if not c.get("advisory"))


def test_run_gate_non_binding_decision_json_has_fdr_skip_reason(tmp_path):
    """Every row must store fdr_binding=False and fdr_skip_reason='stats_advisory' in
    decision_json so the DB audit record matches the returned GateDecision (C3 guard)."""
    import json as _json
    repo = _gate_repo(tmp_path)
    breadth = _breadth(repo, "declared")
    outcome = _run(repo, breadth)
    assert outcome.decision.fdr_binding is False
    assert outcome.decision.fdr_skip_reason == "stats_advisory"
    row = repo._conn.execute(
        "SELECT decision_json FROM gate_evaluations ORDER BY id DESC LIMIT 1"
    ).fetchone()
    stored = _json.loads(row["decision_json"])
    assert stored.get("fdr_binding") is False
    assert stored.get("fdr_skip_reason") == "stats_advisory"


# --- the soft gate end-to-end: floor-only promotion + born-consumed token ----------------------


def test_e2e_all_stats_fail_floor_pass_promotes_with_born_consumed_token(tmp_path):
    """A strategy failing EVERY statistical check but clearing the integrity floor PROMOTES to
    candidate, minting a born-consumed agent gate token (the stage advanced in the same tx)."""
    repo = _gate_repo(tmp_path)
    repo.record_search_trial(_GATE_NAME, 5, "{}", trial_sharpe_count=5,
                             trial_sharpe_mean=0.5, trial_sharpe_var_ann=400.0)  # DSR will fail
    wf = _gate_wf(
        holdout={**_GATE_HOLDOUT, "sharpe": 0.05, "total_return": -0.01},  # stats fail, floor ok
        stability={"pct_positive_windows": 0.0, "min_sharpe": -2.0})
    outcome = run_gate(
        repo, wf, name=_GATE_NAME, actor=Actor.AGENT, criteria=GateCriteria(),
        breadth=_breadth(repo, "measured"), universe_name=None, universe_snapshots=None,
        period_start=_GATE_START, period_end=_GATE_END, holdout_frac=0.2,
        data_source="synthetic", snapshot_id=None, allow_non_pit=True, reason_suffix="")
    d = outcome.decision
    failed = [c for c in d.checks if not c["passed"]]
    assert {c["name"] for c in failed} >= {
        "holdout_sharpe", "holdout_return", "pct_positive_windows", "min_window_sharpe",
        "dsr_evidence"}
    assert all(c.get("advisory") is True for c in failed)
    assert outcome.promoted is True
    assert repo.get(_GATE_NAME).stage is Stage.CANDIDATE
    row = repo._conn.execute(
        "SELECT passed, consumed, actor FROM gate_evaluations ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert row["passed"] == 1 and row["consumed"] == 1 and row["actor"] == "agent"


@pytest.mark.parametrize("holdout_patch, stability_patch", [
    ({"sharpe": 0.0}, None),                        # raw Sharpe <= 0 (strict >)
    ({"sharpe": -0.5}, None),                       # losing holdout
    ({"n_bars": 62}, None),                         # under the observations floor
])
def test_e2e_floor_failures_still_fail_closed(tmp_path, holdout_patch, stability_patch):
    repo = _gate_repo(tmp_path)
    repo.record_search_trial(_GATE_NAME, 5, "{}", trial_sharpe_count=5,
                             trial_sharpe_mean=0.5, trial_sharpe_var_ann=0.04)
    wf = _gate_wf(holdout={**_GATE_HOLDOUT, **holdout_patch},
                  stability={**_GATE_STAB, **(stability_patch or {})})
    outcome = run_gate(
        repo, wf, name=_GATE_NAME, actor=Actor.AGENT, criteria=GateCriteria(),
        breadth=_breadth(repo, "measured"), universe_name=None, universe_snapshots=None,
        period_start=_GATE_START, period_end=_GATE_END, holdout_frac=0.2,
        data_source="synthetic", snapshot_id=None, allow_non_pit=True, reason_suffix="")
    assert outcome.promoted is False
    assert repo.get(_GATE_NAME).stage is Stage.BACKTESTED


def test_e2e_pit_failure_still_fails_closed(tmp_path):
    """No universe + no allow_non_pit → pit_required (binding) vetoes even a stellar holdout."""
    repo = _gate_repo(tmp_path)
    repo.record_search_trial(_GATE_NAME, 5, "{}", trial_sharpe_count=5,
                             trial_sharpe_mean=0.5, trial_sharpe_var_ann=0.04)
    outcome = run_gate(
        repo, _gate_wf(), name=_GATE_NAME, actor=Actor.AGENT, criteria=GateCriteria(),
        breadth=_breadth(repo, "measured"), universe_name=None, universe_snapshots=None,
        period_start=_GATE_START, period_end=_GATE_END, holdout_frac=0.2,
        data_source="synthetic", snapshot_id=None, allow_non_pit=False, reason_suffix="")
    assert outcome.promoted is False
    pit = next(c for c in outcome.decision.checks if c["name"] == "pit_required")
    assert pit["passed"] is False and not pit.get("advisory")
    assert repo.get(_GATE_NAME).stage is Stage.BACKTESTED


def test_forward_gate_reads_holdout_sharpe_value_from_soft_pass_row(tmp_path):
    """Forward coupling: qualified_holdout_sharpe resolves the (advisory) holdout_sharpe check's
    VALUE by name from a soft-pass row written via the REAL run_gate path — the 0.5×holdout
    forward-bar parameterization survives the advisory flip byte-identically."""
    from algua.registry.approvals import compute_artifact_hashes
    from algua.registry.forward_evidence import qualified_holdout_sharpe

    repo = _gate_repo(tmp_path)
    repo.record_search_trial(_GATE_NAME, 5, "{}", trial_sharpe_count=5,
                             trial_sharpe_mean=0.5, trial_sharpe_var_ann=0.04)
    # PIT-clean row (pit_ok=1, no override): a real universe snapshot covering the period.
    snapshots = [{"snapshot_id": "u1", "effective_date": "2020-01-01"}]
    # sharpe=0.3 FAILS the (advisory) 0.5 statistical bar but clears the floor → soft pass.
    outcome = run_gate(
        repo, _wf_sharpe(0.3), name=_GATE_NAME, actor=Actor.AGENT, criteria=GateCriteria(),
        breadth=_breadth(repo, "measured"), universe_name="u", universe_snapshots=snapshots,
        period_start=_GATE_START, period_end=_GATE_END, holdout_frac=0.2,
        data_source="synthetic", snapshot_id=None, allow_non_pit=False, reason_suffix="")
    assert outcome.promoted is True
    hs = next(c for c in outcome.decision.checks if c["name"] == "holdout_sharpe")
    assert hs["passed"] is False and hs["advisory"] is True   # a genuine soft pass
    sid = repo.get(_GATE_NAME).id
    identity = compute_artifact_hashes(_GATE_NAME)
    assert qualified_holdout_sharpe(repo._conn, sid, identity) == pytest.approx(0.3)
    # #559: run_gate stamps the gated-universe identity onto the gate row it mints.
    row = repo._conn.execute(
        "SELECT universe_name FROM gate_evaluations ORDER BY id DESC LIMIT 1").fetchone()
    assert row["universe_name"] == "u"


# --- run_gate returns_available audit (#221 Slice 1) ------------------------------------------


def test_run_gate_no_holdout_id_is_returns_unavailable(tmp_path):
    """_run/_run_measured (no holdout_evaluation_id) must yield returns_available=False
    and write NO holdout_returns row."""
    repo = _gate_repo(tmp_path)
    repo.record_search_trial(_GATE_NAME, 5, "{}", trial_sharpe_count=5,
                             trial_sharpe_mean=0.5, trial_sharpe_var_ann=0.04)
    outcome = _run(repo, _breadth(repo, "measured"))
    assert outcome.decision.returns_available is False
    assert repo._conn.execute("SELECT COUNT(*) c FROM holdout_returns").fetchone()["c"] == 0


def test_run_gate_with_holdout_id_and_returns_writes_row(tmp_path):
    """When holdout_evaluation_id is provided and wf.holdout_returns is populated,
    run_gate writes a holdout_returns row and sets returns_available=True."""
    import json as _json  # noqa: PLC0415

    from algua.backtest.walkforward import WalkForwardResult

    repo = _gate_repo(tmp_path)
    repo.record_search_trial(_GATE_NAME, 5, "{}", trial_sharpe_count=5,
                             trial_sharpe_mean=0.5, trial_sharpe_var_ann=0.04)

    # Reserve and finalize a holdout burn to get a real holdout_evaluation_id.
    sid = repo.get(_GATE_NAME).id
    h_start, h_end = "2024-01-01", "2024-06-01"
    rid, _ = repo.reserve_holdout(
        sid, data_source="synthetic", snapshot_id=None,
        period_start="2024-01-01", period_end="2024-06-01",
        holdout_frac=0.2, holdout_start=h_start, holdout_end=h_end, allow_reuse=False)
    repo.finalize_holdout_reservation(rid, config_hash="c", strategy_id=sid)

    rets = [0.01, -0.02, 0.005]
    bar_dates = ["2024-01-02", "2024-03-15", "2024-05-30"]
    wf = WalkForwardResult(
        strategy=_GATE_NAME, config_hash="c", data_source="synthetic", snapshot_id=None,
        timeframe="1d", seed=None,
        period={"start": "2024-01-01", "end": "2024-06-01"}, windows=4, holdout_frac=0.2,
        window_metrics=[],
        holdout_metrics={**_GATE_HOLDOUT, "start": h_start, "end": h_end, "n_bars": len(rets)},
        stability=dict(_GATE_STAB),
        holdout_returns=(rets, bar_dates),
    )

    outcome = run_gate(
        repo, wf, name=_GATE_NAME, actor=Actor.AGENT, criteria=GateCriteria(),
        breadth=_breadth(repo, "measured"), universe_name=None, universe_snapshots=None,
        period_start=_GATE_START, period_end=_GATE_END, holdout_frac=0.2,
        data_source="synthetic", snapshot_id=None, allow_non_pit=True, reason_suffix="",
        holdout_evaluation_id=rid,
    )

    assert outcome.decision.returns_available is True
    assert "returns_available" in outcome.decision.to_dict()
    assert outcome.decision.to_dict()["returns_available"] is True

    # holdout_returns row must exist for this holdout_evaluation_id
    count = repo._conn.execute(
        "SELECT COUNT(*) c FROM holdout_returns WHERE holdout_evaluation_id=?", (rid,)
    ).fetchone()["c"]
    assert count == 1

    # The persisted decision_json must carry returns_available=True (written before serializing)
    row = repo._conn.execute(
        "SELECT decision_json FROM gate_evaluations ORDER BY id DESC LIMIT 1"
    ).fetchone()
    stored = _json.loads(row["decision_json"])
    assert stored.get("returns_available") is True


def test_run_gate_write_persists_even_on_failed_gate(tmp_path):
    """The returns vector is written even when the gate FAILS. The burn already committed
    at on_peek, so the vector must persist regardless of pass/fail."""
    from algua.backtest.walkforward import WalkForwardResult

    repo = _gate_repo(tmp_path)
    # Use measured breadth but no trial stats → DSR fails closed → gate fails.
    repo.record_search_trial(_GATE_NAME, 5, "{}")  # no stats → fails DSR

    sid = repo.get(_GATE_NAME).id
    h_start, h_end = "2024-01-01", "2024-06-01"
    rid, _ = repo.reserve_holdout(
        sid, data_source="synthetic", snapshot_id=None,
        period_start="2024-01-01", period_end="2024-06-01",
        holdout_frac=0.2, holdout_start=h_start, holdout_end=h_end, allow_reuse=False)
    repo.finalize_holdout_reservation(rid, config_hash="c", strategy_id=sid)

    rets = [0.01, -0.02]
    bar_dates = ["2024-01-02", "2024-03-15"]
    wf = WalkForwardResult(
        strategy=_GATE_NAME, config_hash="c", data_source="synthetic", snapshot_id=None,
        timeframe="1d", seed=None,
        period={"start": "2024-01-01", "end": "2024-06-01"}, windows=4, holdout_frac=0.2,
        window_metrics=[],
        holdout_metrics={**_GATE_HOLDOUT, "start": h_start, "end": h_end, "n_bars": len(rets)},
        stability=dict(_GATE_STAB),
        holdout_returns=(rets, bar_dates),
    )

    outcome = run_gate(
        repo, wf, name=_GATE_NAME, actor=Actor.AGENT, criteria=GateCriteria(),
        breadth=_breadth(repo, "measured"), universe_name=None, universe_snapshots=None,
        period_start=_GATE_START, period_end=_GATE_END, holdout_frac=0.2,
        data_source="synthetic", snapshot_id=None, allow_non_pit=True, reason_suffix="",
        holdout_evaluation_id=rid,
    )

    # Gate must have failed (DSR fails closed with NULL stats)
    assert outcome.promoted is False
    # But the returns row must still exist
    assert outcome.decision.returns_available is True
    count = repo._conn.execute(
        "SELECT COUNT(*) c FROM holdout_returns WHERE holdout_evaluation_id=?", (rid,)
    ).fetchone()["c"]
    assert count == 1

    # The persisted decision_json must carry returns_available=True on the failed-gate path too.
    import json as _json  # noqa: PLC0415
    row = repo._conn.execute(
        "SELECT decision_json FROM gate_evaluations ORDER BY id DESC LIMIT 1"
    ).fetchone()
    stored = _json.loads(row["decision_json"])
    assert stored.get("returns_available") is True


# --- run_gate bootstrap AND-check (#221 Slice 2) -----------------------------------------------
# Calibration notes:
#   Bootstrap binds iff dsr_binding=True AND wf.holdout_returns is not None.
#   - wf WITHOUT holdout_returns (all _run/_run_measured calls) → bootstrap NOT binding.
#   - wf WITH holdout_returns AND measured breadth → bootstrap binds, adds dsr_bootstrap check.
#   The benign fixture uses 63 bars of alternating [0.01, 0.005] (mean ~0.0076, low autocorr)
#   so the bootstrap lower-confidence is well above the DSR_ALPHA threshold.

# 63-bar benign return series: alternating 0.01 / 0.005, low autocorrelation, positive Sharpe.
_BOOTSTRAP_RETS = [0.01 if i % 2 == 0 else 0.005 for i in range(63)]
_BOOTSTRAP_DATES = [f"2024-{(i // 20) + 1:02d}-{(i % 20) + 1:02d}" for i in range(63)]
_BOOTSTRAP_N = len(_BOOTSTRAP_RETS)
_BOOTSTRAP_H_START = "2024-01-01"
_BOOTSTRAP_H_END = "2024-03-31"


def _gate_repo_with_stats(tmp_path):
    """Repo ready for bootstrap tests: measured breadth with trial stats."""
    repo = _gate_repo(tmp_path)
    repo.record_search_trial(_GATE_NAME, 5, "{}", trial_sharpe_count=5,
                             trial_sharpe_mean=0.5, trial_sharpe_var_ann=0.04)
    return repo


def _wf_with_holdout_returns(rets=None, bar_dates=None, h_start=None, h_end=None):
    """Build a WalkForwardResult carrying holdout_returns (bootstrap-binding)."""
    from algua.backtest.walkforward import WalkForwardResult

    rets = rets if rets is not None else _BOOTSTRAP_RETS
    bar_dates = bar_dates if bar_dates is not None else _BOOTSTRAP_DATES
    h_start = h_start or _BOOTSTRAP_H_START
    h_end = h_end or _BOOTSTRAP_H_END
    return WalkForwardResult(
        strategy=_GATE_NAME, config_hash="c", data_source="synthetic", snapshot_id=None,
        timeframe="1d", seed=None,
        period={"start": "2024-01-01", "end": "2024-06-01"}, windows=4, holdout_frac=0.2,
        window_metrics=[],
        holdout_metrics={**_GATE_HOLDOUT, "start": h_start, "end": h_end, "n_bars": len(rets)},
        stability=dict(_GATE_STAB),
        holdout_returns=(rets, bar_dates),
    )


def _run_bootstrap(repo, wf, rid=None):
    """run_gate with measured breadth and holdout_returns wf."""
    return run_gate(
        repo, wf, name=_GATE_NAME, actor=Actor.AGENT, criteria=GateCriteria(),
        breadth=_breadth(repo, "measured"), universe_name=None, universe_snapshots=None,
        period_start=_GATE_START, period_end=_GATE_END, holdout_frac=0.2,
        data_source="synthetic", snapshot_id=None, allow_non_pit=True, reason_suffix="",
        holdout_evaluation_id=rid,
    )


def _reserve_holdout(repo):
    """Reserve and finalize a holdout burn; returns rid."""
    sid = repo.get(_GATE_NAME).id
    rid, _ = repo.reserve_holdout(
        sid, data_source="synthetic", snapshot_id=None,
        period_start="2024-01-01", period_end="2024-06-01",
        holdout_frac=0.2, holdout_start=_BOOTSTRAP_H_START, holdout_end=_BOOTSTRAP_H_END,
        allow_reuse=False)
    repo.finalize_holdout_reservation(rid, config_hash="c", strategy_id=sid)
    return rid


def test_bootstrap_binds_when_holdout_returns_present(tmp_path):
    """Measured promote with holdout_returns → dsr_bootstrap_binding=True, check in decision,
    audit fields populated, dsr_bootstrap_lower in persisted decision_json."""
    import json as _json  # noqa: PLC0415

    repo = _gate_repo_with_stats(tmp_path)
    rid = _reserve_holdout(repo)
    wf = _wf_with_holdout_returns()
    outcome = _run_bootstrap(repo, wf, rid)
    d = outcome.decision

    # Bootstrap must bind
    assert d.dsr_bootstrap_binding is True
    # A dsr_bootstrap check must appear in the checks list
    assert any(c["name"] == "dsr_bootstrap" for c in d.checks)
    # Audit scalars must be populated
    assert d.dsr_bootstrap_seed is not None
    assert d.dsr_bootstrap_b is not None
    assert d.dsr_bootstrap_block_len is not None
    # The lower confidence must be a finite float (benign series → high value)
    assert d.dsr_bootstrap_lower is not None
    assert isinstance(d.dsr_bootstrap_lower, float)

    # decision_json in DB must carry the bootstrap lower confidence
    row = repo._conn.execute(
        "SELECT decision_json FROM gate_evaluations ORDER BY id DESC LIMIT 1"
    ).fetchone()
    stored = _json.loads(row["decision_json"])
    assert "dsr_bootstrap_lower" in stored
    assert stored["dsr_bootstrap_lower"] is not None


def test_bootstrap_not_binding_without_holdout_returns(tmp_path):
    """Existing _run helper (wf has no holdout_returns) → bootstrap NOT binding, no check,
    and existing pass/fail outcome is UNCHANGED (measured passing case still passes)."""
    repo = _gate_repo(tmp_path)
    repo.record_search_trial(_GATE_NAME, 5, "{}", trial_sharpe_count=5,
                             trial_sharpe_mean=0.5, trial_sharpe_var_ann=0.04)
    breadth = _breadth(repo, "measured")
    # _gate_wf() has no holdout_returns — bootstrap must NOT bind
    outcome = _run(repo, breadth)
    d = outcome.decision

    assert d.dsr_bootstrap_binding is False
    assert all(c["name"] != "dsr_bootstrap" for c in d.checks)
    # The existing DSR+FDR outcome is unchanged: dsr Sharpe=7.0 → passes, FDR accepts → promoted
    assert d.dsr_binding is True
    assert outcome.promoted is True


def test_bootstrap_determinism(tmp_path):
    """Two identical run_gate calls on fresh repos produce the same dsr_bootstrap_lower."""
    wf = _wf_with_holdout_returns()

    def _run_fresh(path):
        repo = _gate_repo_with_stats(path)
        rid = _reserve_holdout(repo)
        return _run_bootstrap(repo, wf, rid).decision.dsr_bootstrap_lower

    lower1 = _run_fresh(tmp_path / "db1")
    lower2 = _run_fresh(tmp_path / "db2")
    assert lower1 is not None
    assert lower1 == lower2


# --- run_gate N_eff shadow wiring (#221 Slice 3) -----------------------------------------------
# N_eff is SHADOW-ONLY: recorded in dsr_n_eff/dsr_rho_bar/dsr_n_siblings, NEVER fed into
# evaluate_gate. The binding DSR continues to use raw n_funnel (dsr_n_trials == n_funnel).
#
# Sibling seeding: we create extra registered strategies, reserve+finalize a holdout burn for
# each, and call record_holdout_returns directly (bypassing run_gate) to plant sibling vectors
# without touching the strategy-under-promotion's gate path.
#
# Calibration: _BOOTSTRAP_H_START/_BOOTSTRAP_H_END overlap the sibling window; each sibling
# vector uses the same date range so overlap is guaranteed.

def _seed_sibling_returns(repo, n_siblings: int, *, h_start: str, h_end: str,
                          n_bars: int = 63) -> None:
    """Register n_siblings extra strategies and plant overlapping holdout_returns rows.

    Each sibling has its own strategy row, a committed holdout burn (reserve+finalize), and a
    holdout_returns row with a synthetic return vector that overlaps [h_start, h_end].
    The vectors use distinct but valid return series so correlations are well-defined.

    NOTE: siblings do NOT get a search_trial row — we must not inflate windowed_search_combos
    or n_funnel for the strategy-under-promotion (shadow-only invariant: seeding siblings changes
    no gate inputs for the focal strategy).
    """
    from datetime import timedelta  # noqa: PLC0415
    bar_dates = [(date(2024, 1, 1) + timedelta(days=i)).isoformat() for i in range(n_bars)]
    for k in range(n_siblings):
        sib_name = f"_sibling_{k}"
        sib_rec = repo.add(sib_name)
        repo.apply_transition(sib_rec, Stage.BACKTESTED, Actor.AGENT, "bt")
        sib_id = repo.get(sib_name).id
        rid, _ = repo.reserve_holdout(
            sib_id, data_source="synthetic", snapshot_id=None,
            period_start="2024-01-01", period_end="2024-06-01",
            holdout_frac=0.2, holdout_start=h_start, holdout_end=h_end,
            allow_reuse=False)
        repo.finalize_holdout_reservation(rid, config_hash=f"c{k}", strategy_id=sib_id)
        # Small variation per sibling so all pairwise correlations are well-defined (not identical).
        rets = [0.01 * (1 + 0.01 * k) if i % 2 == 0 else -0.005 * (1 + 0.01 * k)
                for i in range(n_bars)]
        repo.record_holdout_returns(
            rid, sib_id, holdout_start=h_start, holdout_end=h_end,
            returns=rets, bar_dates=bar_dates)


def _run_bootstrap_neff(repo, rid=None):
    """run_gate with measured breadth, holdout_returns, using the bootstrap window for N_eff."""
    return run_gate(
        repo, _wf_with_holdout_returns(), name=_GATE_NAME, actor=Actor.AGENT,
        criteria=GateCriteria(),
        breadth=_breadth(repo, "measured"), universe_name=None, universe_snapshots=None,
        period_start=_GATE_START, period_end=_GATE_END, holdout_frac=0.2,
        data_source="synthetic", snapshot_id=None, allow_non_pit=True, reason_suffix="",
        holdout_evaluation_id=rid,
    )


def test_n_eff_populated_with_sufficient_siblings(tmp_path):
    """Measured promote with >=5 sibling holdout_returns rows overlapping the OOS interval:
    dsr_n_eff is populated, 1 <= n_eff <= n_funnel, n_siblings >= 5, and dsr_n_eff lands
    in the persisted decision_json."""
    import json as _json  # noqa: PLC0415

    from algua.research.dsr import MIN_N_EFF_SIBLINGS

    repo = _gate_repo_with_stats(tmp_path)
    rid = _reserve_holdout(repo)

    # Seed MIN_N_EFF_SIBLINGS sibling return vectors overlapping the bootstrap OOS window.
    _seed_sibling_returns(repo, MIN_N_EFF_SIBLINGS,
                          h_start=_BOOTSTRAP_H_START, h_end=_BOOTSTRAP_H_END,
                          n_bars=_BOOTSTRAP_N)

    outcome = _run_bootstrap_neff(repo, rid)
    d = outcome.decision

    # N_eff fields must be populated
    assert d.dsr_n_eff is not None, "dsr_n_eff should be populated with >=5 siblings"
    assert d.dsr_n_siblings is not None
    assert d.dsr_n_siblings >= MIN_N_EFF_SIBLINGS

    # N_eff is bounded: 1 <= n_eff <= n_funnel (raw N)
    n_funnel = d.dsr_n_trials
    assert n_funnel is not None
    assert 1 <= d.dsr_n_eff <= n_funnel

    # dsr_n_eff must appear in the persisted decision_json
    row = repo._conn.execute(
        "SELECT decision_json FROM gate_evaluations ORDER BY id DESC LIMIT 1"
    ).fetchone()
    stored = _json.loads(row["decision_json"])
    assert "dsr_n_eff" in stored
    assert stored["dsr_n_eff"] is not None


def test_n_eff_shadow_invariant(tmp_path):
    """SHADOW invariant: N_eff must NEVER affect the binding gate outcome.

    Two identical promotes — one with >=5 siblings seeded, one with none — must produce:
    - the SAME decision.passed
    - the SAME dsr_evidence check value
    - dsr_n_trials == raw n_funnel in both cases (binding DSR used raw N, not N_eff)
    """
    from algua.research.dsr import MIN_N_EFF_SIBLINGS

    # --- Run WITH siblings ---
    repo_with = _gate_repo_with_stats(tmp_path / "with")
    rid_with = _reserve_holdout(repo_with)
    _seed_sibling_returns(repo_with, MIN_N_EFF_SIBLINGS,
                          h_start=_BOOTSTRAP_H_START, h_end=_BOOTSTRAP_H_END,
                          n_bars=_BOOTSTRAP_N)
    out_with = _run_bootstrap_neff(repo_with, rid_with)
    d_with = out_with.decision

    # --- Run WITHOUT siblings ---
    repo_without = _gate_repo_with_stats(tmp_path / "without")
    rid_without = _reserve_holdout(repo_without)
    out_without = _run_bootstrap_neff(repo_without, rid_without)
    d_without = out_without.decision

    # Binding DSR used raw N in both cases
    assert d_with.dsr_n_trials == d_without.dsr_n_trials, (
        "dsr_n_trials must be identical (raw N) regardless of sibling count")

    # passed must be identical
    assert d_with.passed == d_without.passed, (
        "N_eff shadow recording must not change gate outcome")

    # dsr_evidence check value must be identical
    def _dsr_evidence_value(d):
        for c in d.checks:
            if c["name"] == "dsr_evidence":
                return c["value"]
        return None

    val_with = _dsr_evidence_value(d_with)
    val_without = _dsr_evidence_value(d_without)
    assert val_with == val_without, (
        f"dsr_evidence value changed with siblings: {val_with} vs {val_without}")

    # With siblings: n_eff should be populated; without: None
    assert d_with.dsr_n_eff is not None
    assert d_without.dsr_n_eff is None


def test_n_eff_none_with_insufficient_siblings(tmp_path):
    """Measured promote with FEWER than MIN_N_EFF_SIBLINGS overlapping siblings:
    dsr_n_eff is None, dsr_n_siblings < MIN_N_EFF_SIBLINGS, gate outcome unchanged."""
    from algua.research.dsr import MIN_N_EFF_SIBLINGS

    repo = _gate_repo_with_stats(tmp_path)
    rid = _reserve_holdout(repo)

    # Seed fewer than the minimum
    n_too_few = max(0, MIN_N_EFF_SIBLINGS - 1)
    if n_too_few > 0:
        _seed_sibling_returns(repo, n_too_few,
                              h_start=_BOOTSTRAP_H_START, h_end=_BOOTSTRAP_H_END,
                              n_bars=_BOOTSTRAP_N)

    outcome = _run_bootstrap_neff(repo, rid)
    d = outcome.decision

    # N_eff must be None (not enough siblings)
    assert d.dsr_n_eff is None, "dsr_n_eff should be None with insufficient siblings"
    # n_siblings reflects the actual count (may be 0 or less than min)
    assert d.dsr_n_siblings is not None
    assert d.dsr_n_siblings < MIN_N_EFF_SIBLINGS

    # Gate outcome still valid (binding DSR still runs)
    assert d.dsr_binding is True
    assert any(c["name"] == "dsr_evidence" for c in d.checks)


# --- run_gate regime robustness wiring (#221 Slice 4) ------------------------------------------
# Task 5 integration tests: verify that passing wf.market_returns into run_gate causes the
# regime_robustness check to bind (via evaluate_gate receiving market_returns=wf.market_returns).
#
# Vector construction:
#   _REGIME_MKT_DATES: 84 real ISO dates starting 2023-01-02 (step=1 calendar day).
#   Market series: 84 returns whose trailing-21-bar vol clearly varies by tertile.
#     - dates 0..20  (warmup, no vol label)
#     - dates 21..41 (low-vol label)   → market returns near-zero (~0.001)
#     - dates 42..62 (mid-vol label)   → market returns small (~0.005)
#     - dates 63..83 (high-vol label)  → market returns large (~0.02)
#   Holdout covers dates 21..83 = 63 labeled dates (overlap == 63 >= MIN_REGIME_OVERLAP_BARS).
#   The vol-tertile split of those 63 dates:
#     - low-vol tertile  (21 dates): 2023-01-23 .. 2023-02-12
#     - mid-vol tertile  (21 dates): 2023-02-13 .. 2023-03-05
#     - high-vol tertile (21 dates): 2023-03-06 .. 2023-03-26
#
# Regime-passing fixture: holdout returns are +0.01 on all 63 dates → all regimes pass.
# Regime-failing fixture: holdout returns are -0.05 on the high-vol-tertile dates → high-vol
#   regime fails (mean<0, negative Sharpe) while aggregate Sharpe stays ≥ gate bar.

# 84 real ISO dates from 2023-01-02 (inclusive) — real calendar days for reliable sorting
_REGIME_MKT_N = 84
_REGIME_MKT_DATES = [
    (date(2023, 1, 2) + timedelta(days=i)).isoformat() for i in range(_REGIME_MKT_N)
]

# Market returns: low vol for first third (after warmup), stepping up.
# Window vol is computed over trailing-21 bars.  The first labeled date is index 20.
# Dates 20..40 → windows of ~0.001 amplitude → low vol
# Dates 41..61 → windows of ~0.005 amplitude → mid vol
# Dates 62..83 → windows of ~0.02  amplitude → high vol
_REGIME_MKT_RETS: list[float] = []
for _i in range(_REGIME_MKT_N):
    if _i < 42:          # first two thirds of the full period (includes warmup)
        _REGIME_MKT_RETS.append(0.001 if _i % 2 == 0 else -0.001)
    elif _i < 63:
        _REGIME_MKT_RETS.append(0.005 if _i % 2 == 0 else -0.005)
    else:
        _REGIME_MKT_RETS.append(0.02 if _i % 2 == 0 else -0.02)

# Holdout spans dates 21..83 (63 dates): the 63 vol-labeled dates
_REGIME_HOLDOUT_DATES = _REGIME_MKT_DATES[21:]   # 63 dates, all with vol labels
_REGIME_HOLDOUT_N = len(_REGIME_HOLDOUT_DATES)    # == 63
assert _REGIME_HOLDOUT_N == 63

# Regime-passing holdout: alternating +0.01 / +0.005 on every date so all three regimes pass
# MIN_REGIME_SHARPE (positive mean, non-zero vol so bootstrap doesn't fail due to zero variance).
_REGIME_PASS_RETS = [0.01 if i % 2 == 0 else 0.005 for i in range(_REGIME_HOLDOUT_N)]

# For the blocking test we need to identify the 21 high-vol-tertile dates.
# regime_splits ranks by (trailing_vol, date_string) asc; the highest-vol tertile is the last 21.
# The trailing-21-bar vols over our market series:
#   - dates 20..41: vol of windows mostly drawn from ±0.001 amplitude → ~annualized ~0.022
#   - dates 42..62: vol of windows mostly drawn from ±0.005 amplitude → ~annualized ~0.11
#   - dates 63..83: vol of windows mostly drawn from ±0.02  amplitude → ~annualized ~0.45
# The 21 highest-vol dates are indices 63..83 of _REGIME_MKT_DATES, i.e. the last 21 holdout dates.
_REGIME_HIGH_VOL_DATES = set(_REGIME_MKT_DATES[63:])  # the 21 high-vol holdout dates

# Regime-failing holdout: -0.05 on high-vol dates (forces negative regime Sharpe there),
# +0.015 on low/mid dates (ensures aggregate Sharpe well above the gate bar).
_REGIME_FAIL_RETS = [
    -0.05 if d in _REGIME_HIGH_VOL_DATES else 0.015
    for d in _REGIME_HOLDOUT_DATES
]


def _wf_with_regime(holdout_rets, holdout_dates, market_rets, market_dates,
                    h_start=None, h_end=None):
    """Build a WalkForwardResult carrying both holdout_returns and market_returns."""
    from algua.backtest.walkforward import WalkForwardResult

    h_start = h_start or holdout_dates[0]
    h_end = h_end or holdout_dates[-1]
    return WalkForwardResult(
        strategy=_GATE_NAME, config_hash="c", data_source="synthetic", snapshot_id=None,
        timeframe="1d", seed=None,
        period={"start": "2024-01-01", "end": "2024-06-01"}, windows=4, holdout_frac=0.2,
        window_metrics=[],
        holdout_metrics={**_GATE_HOLDOUT, "start": h_start, "end": h_end,
                         "n_bars": len(holdout_rets)},
        stability=dict(_GATE_STAB),
        holdout_returns=(holdout_rets, holdout_dates),
        market_returns=(market_rets, market_dates),
    )


def _run_regime(repo, wf):
    """run_gate call for regime robustness tests (measured breadth, allow_non_pit)."""
    return run_gate(
        repo, wf, name=_GATE_NAME, actor=Actor.AGENT, criteria=GateCriteria(),
        breadth=_breadth(repo, "measured"), universe_name=None, universe_snapshots=None,
        period_start=_GATE_START, period_end=_GATE_END, holdout_frac=0.2,
        data_source="synthetic", snapshot_id=None, allow_non_pit=True, reason_suffix="",
    )


def test_regime_robustness_binds_when_market_returns_present(tmp_path):
    """Measured promote with wf carrying holdout_returns + market_returns with real vol structure
    and >=63 overlap: regime_robustness binds, vol_tertile method, audit fields populated."""
    repo = _gate_repo_with_stats(tmp_path)
    wf = _wf_with_regime(
        _REGIME_PASS_RETS, _REGIME_HOLDOUT_DATES,
        _REGIME_MKT_RETS, _REGIME_MKT_DATES,
    )
    outcome = _run_regime(repo, wf)
    d = outcome.decision

    # Regime check must bind
    assert d.regime_robustness_binding is True, (
        "regime_robustness_binding should be True when market_returns present with >=63 overlap")
    # A regime_robustness check must appear in the checks list
    assert any(c["name"] == "regime_robustness" for c in d.checks), (
        "regime_robustness check must be in decision.checks when binding")
    # Method must be vol_tertile
    assert d.regime_method == "vol_tertile", f"expected vol_tertile, got {d.regime_method}"
    # Audit counts must be populated
    assert d.n_regimes_attempted is not None and d.n_regimes_attempted == 3
    assert d.n_regimes_surviving is not None and d.n_regimes_surviving >= 2
    assert d.per_regime_sharpes is not None and len(d.per_regime_sharpes) == 3

    # Regime fields must appear in to_dict / decision_json
    d_dict = d.to_dict()
    assert "regime_robustness_binding" in d_dict
    assert d_dict["regime_robustness_binding"] is True
    assert "regime_method" in d_dict and d_dict["regime_method"] == "vol_tertile"

    # Shadow fields must also flow through a real run_gate (Task 4)
    # 0b11111 = bits 0-4 = Phase 3 slices 0-4 all active (dispersion-floor, persistence,
    # bootstrap, n_eff, multi-regime)
    assert d.phase3_component_mask == 0b11111
    assert "haircut_would_have_blocked" in d_dict

    # Promoted: regime passes, DSR+FDR pass at sharpe=7.0
    assert outcome.promoted is True


def test_regime_robustness_omits_when_no_market_returns(tmp_path):
    """Promote with no market_returns: regime check omits, regime_method=unavailable,
    existing pass/fail outcome UNCHANGED."""
    repo = _gate_repo_with_stats(tmp_path)
    # _gate_wf() has no holdout_returns nor market_returns → regime check must omit
    outcome = _run(repo, _breadth(repo, "measured"))
    d = outcome.decision

    assert d.regime_robustness_binding is False
    assert d.regime_method == "unavailable"
    assert all(c["name"] != "regime_robustness" for c in d.checks)
    # Existing outcome must be unchanged: measured breadth + sharpe=7.0 → promotes
    assert outcome.promoted is True


def test_regime_robustness_failure_recorded_advisory_still_promotes(tmp_path):
    """Strategy that passes aggregate holdout Sharpe but fails the high-vol regime records the
    FAILED regime_robustness check as an advisory flag — the soft gate still promotes on the
    integrity floor (the forward gate is the blocker)."""
    repo = _gate_repo_with_stats(tmp_path)
    wf = _wf_with_regime(
        _REGIME_FAIL_RETS, _REGIME_HOLDOUT_DATES,
        _REGIME_MKT_RETS, _REGIME_MKT_DATES,
    )
    outcome = _run_regime(repo, wf)
    d = outcome.decision

    # Regime check must be armed and present
    assert d.regime_robustness_binding is True
    regime_check = next(c for c in d.checks if c["name"] == "regime_robustness")
    # High-vol regime returns are negative → regime_robustness check FAILS (advisory)
    assert regime_check["passed"] is False, (
        "high-vol regime has negative returns; regime_robustness check must fail")
    assert regime_check["advisory"] is True
    assert d.passed is True
    assert outcome.promoted is True

    # decision_json must carry regime fields
    d_dict = d.to_dict()
    assert d_dict.get("regime_robustness_binding") is True
    assert d_dict.get("regime_method") == "vol_tertile"
