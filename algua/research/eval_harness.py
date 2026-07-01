"""Gate-decision eval harness (#347): pass@k / pass^k + trace error-analysis.

WHAT THIS IS — and is NOT.

algua evaluates *strategies* rigorously (the promotion gates) but has never measured the reliability
of the *promote/discard judgment itself*. This harness does that: it runs the REAL gate decision
layer (``walk_forward -> evaluate_gate``) over a bank of seeded synthetic scenarios with known true
edge, and reports the catastrophic-failure rate (``false_promote_rate``), the KB's named reliability
metrics (``pass@k`` / ``pass^k``, defined precisely below), and an error-analysis failure taxonomy.

It is honestly a GATE-DECISION eval, NOT a full operator-loop eval. The autonomous loop
(``run-the-research-loop`` skill) is ideate -> author -> backtest -> walk-forward -> gate -> record;
it is LLM/skill-driven with no programmatic entrypoint, and its only reproducible-offline
catastrophic surface is the promote/discard decision. This harness measures exactly that surface and
NOTHING upstream. It also does NOT exercise the stateful multiple-testing controls (funnel-breadth
deflation, DSR binding, FDR alpha-wealth, PIT enforcement) — those are single-use/global-stateful,
cannot be re-run k times reproducibly, so the harness pins ``n_combos=1`` and ``pit_ok=True`` and
measures the core predicate only. The full operator-loop eval (replaying real operator traces) is a
deferred follow-up needing the #332 negative-result trace substrate. See the design doc
``docs/superpowers/specs/2026-07-01-operator-eval-harness-347-design.md``.

METRIC HONESTY. ``pass@k``/``pass^k`` here mean ``any_seed_correct``/``all_seed_correct`` over
resampled synthetic price paths — Monte-Carlo path-robustness of the gate decision, NOT operator
attempt-to-attempt consistency. The headline SAFETY metric is ``false_promote_rate`` (a promoted
no-edge/losing book), never ``pass@k`` (which can hide a false promote behind one correct discard).
"""
from __future__ import annotations

import math
import subprocess
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Literal

import pandas as pd

from algua.backtest._sample import SyntheticProvider
from algua.backtest.walkforward import walk_forward
from algua.contracts.types import ExecutionContract
from algua.portfolio.construction import get_construction_policy
from algua.research.gates import GateCriteria, evaluate_gate
from algua.strategies.base import LoadedStrategy, StrategyConfig

# --- harness constants (exposed for tests) ---------------------------------------------------
BANK_VERSION = 1
# A wide window so the holdout (last 20%) clears the 63-observation power floor with margin:
# ~6 years of XNYS sessions -> ~1500 bars -> ~300 holdout bars.
START = datetime(2016, 1, 1, tzinfo=UTC)
END = datetime(2022, 12, 31, tzinfo=UTC)
DEFAULT_K = 8
# Deterministic CI regression guards over the fixed, well-SEPARATED scenario bank. These are NOT
# statistical confidence bounds — they assert "no decision flips in this fixture". A future
# walk_forward change that flips one seed is a deliberate fixture update; a systemic gate-softening
# or path-dependent leak makes a rate jump and fails the eval.
MAX_FALSE_PROMOTE_RATE = 0.0
MAX_CRASH_RATE = 0.0

ScenarioKind = Literal["promote", "discard", "ambiguous"]
VALID_KINDS: frozenset[str] = frozenset(("promote", "discard", "ambiguous"))


@dataclass(frozen=True)
class Scenario:
    """One seeded synthetic regime with a known true edge. ``kind`` decides ground truth:
    ``promote``/``discard`` carry a label (scored against safety rates); ``ambiguous`` straddles the
    bar and is scored for decision CONSISTENCY only (no label, never in the safety rates)."""

    name: str
    kind: ScenarioKind
    drift: float
    vol: float
    symbols: tuple[str, ...] = ("AAA", "BBB")


# Scenarios chosen FAR from the 0.5 Sharpe bar so the label is not in dispute; `marginal_edge`
# deliberately straddles it (consistency-only). drift/vol are per-bar GBM parameters.
SCENARIO_BANK: tuple[Scenario, ...] = (
    Scenario("obvious_edge", "promote", drift=0.0030, vol=0.010),
    Scenario("marginal_edge", "ambiguous", drift=0.0009, vol=0.020),
    Scenario("no_edge", "discard", drift=0.0, vol=0.020),
    Scenario("negative_edge", "discard", drift=-0.0015, vol=0.015),
)

STRATEGY_ID = "equal_weight_positive_long"

# Outcome taxonomy (the error-analysis classes).
CORRECT_PROMOTE = "correct_promote"
CORRECT_DISCARD = "correct_discard"
WRONGLY_PROMOTED = "wrongly_promoted"  # the catastrophic class -> drives false_promote_rate
WRONGLY_DISCARDED = "wrongly_discarded"
AMBIGUOUS_PROMOTE = "ambiguous_promote"
AMBIGUOUS_DISCARD = "ambiguous_discard"
CRASHED = "crashed"


@dataclass(frozen=True)
class RunTrace:
    """One (scenario, seed) gate decision + its error-analysis classification."""

    scenario: str
    kind: str
    seed: int
    actual: str  # "promote" | "discard" | "crashed"
    outcome: str
    failed_checks: list[str]  # FACTUAL: gate checks that did not pass (no causal claim)
    stats: dict[str, float]  # measured numbers (holdout_sharpe/return/n_obs, window stats)
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "kind": self.kind,
            "seed": self.seed,
            "actual": self.actual,
            "outcome": self.outcome,
            "failed_checks": list(self.failed_checks),
            "stats": {k: _clean(v) for k, v in self.stats.items()},
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ScenarioResult:
    scenario: str
    kind: str
    k: int
    n_promote: int
    n_discard: int
    n_crashed: int
    n_correct: int | None  # None for ambiguous (no label)
    pass_at_k: bool | None  # any_seed_correct (None for ambiguous)
    pass_pow_k: bool | None  # all_seed_correct (None for ambiguous)
    promote_rate: float  # over non-crashed runs; always defined (consistency figure)
    traces: list[RunTrace]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "kind": self.kind,
            "k": self.k,
            "n_promote": self.n_promote,
            "n_discard": self.n_discard,
            "n_crashed": self.n_crashed,
            "n_correct": self.n_correct,
            "pass_at_k": self.pass_at_k,
            "pass_pow_k": self.pass_pow_k,
            "any_seed_correct": self.pass_at_k,  # explicit honest alias
            "all_seed_correct": self.pass_pow_k,
            "promote_rate": _clean(self.promote_rate),
            "traces": [t.to_dict() for t in self.traces],
        }


@dataclass(frozen=True)
class EvalProvenance:
    """Everything needed to reproduce the report (review HIGH: determinism record)."""

    start: str
    end: str
    seeds: list[int]
    criteria: dict[str, Any]
    provider: str
    strategy_id: str
    bank_version: int
    algua_version: str
    git_sha: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "seeds": list(self.seeds),
            "criteria": self.criteria,
            "provider": self.provider,
            "strategy_id": self.strategy_id,
            "bank_version": self.bank_version,
            "algua_version": self.algua_version,
            "git_sha": self.git_sha,
        }


@dataclass(frozen=True)
class EvalReport:
    k: int
    scenarios: list[ScenarioResult]
    false_promote_rate: float  # HEADLINE safety metric (over non-crashed discard runs)
    false_discard_rate: float
    crash_rate: float
    accuracy: float  # over labelled (non-crashed) runs
    pass_at_k: float  # fraction of LABELLED scenarios that are any_seed_correct
    pass_pow_k: float  # fraction of LABELLED scenarios that are all_seed_correct
    confusion: dict[str, dict[str, int]]
    failure_histogram: dict[str, int]
    provenance: EvalProvenance

    def to_dict(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "scenarios": [s.to_dict() for s in self.scenarios],
            "false_promote_rate": _clean(self.false_promote_rate),
            "false_discard_rate": _clean(self.false_discard_rate),
            "crash_rate": _clean(self.crash_rate),
            "accuracy": _clean(self.accuracy),
            "pass_at_k": _clean(self.pass_at_k),
            "pass_pow_k": _clean(self.pass_pow_k),
            "any_seed_correct": _clean(self.pass_at_k),  # explicit honest aliases
            "all_seed_correct": _clean(self.pass_pow_k),
            "confusion": self.confusion,
            "failure_histogram": self.failure_histogram,
            "provenance": self.provenance.to_dict(),
        }


def _clean(x: float | None) -> float | None:
    """Null any non-finite float so the JSON payload stays clean."""
    if x is None:
        return None
    return x if math.isfinite(x) else None


def build_strategy(symbols: tuple[str, ...]) -> LoadedStrategy:
    """The canonical equal-weight-long decision-maker. FIXED by design: the harness varies the
    DATA (scenario edge), not the strategy — strategy-authoring diversity is a deferred surface."""
    cfg = StrategyConfig(
        name="eval_ew",
        universe=list(symbols),
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
        params={},
        construction="equal_weight_positive",
    )
    return LoadedStrategy(
        config=cfg,
        signal_fn=lambda v, p: pd.Series(1.0, index=sorted(v["symbol"].unique())),
        construct_fn=get_construction_policy(cfg.construction),
    )


def _classify(kind: str, actual: str) -> str:
    if actual == CRASHED:
        return CRASHED
    if kind == "promote":
        return CORRECT_PROMOTE if actual == "promote" else WRONGLY_DISCARDED
    if kind == "discard":
        return WRONGLY_PROMOTED if actual == "promote" else CORRECT_DISCARD
    # ambiguous: no label, not a failure either way
    return AMBIGUOUS_PROMOTE if actual == "promote" else AMBIGUOUS_DISCARD


def run_one(
    scenario: Scenario, seed: int, *, strategy: LoadedStrategy | None = None
) -> RunTrace:
    """Run the REAL gate decision once on a seeded synthetic path. Never raises: any exception is
    captured as a ``crashed`` trace so one bad run cannot wedge the sweep (per-item fail-closed)."""
    try:
        strat = strategy if strategy is not None else build_strategy(scenario.symbols)
        provider = SyntheticProvider(seed=seed, drift=scenario.drift, vol=scenario.vol)
        wf = walk_forward(strat, provider, START, END, seed=seed)
        decision = evaluate_gate(wf, GateCriteria(), n_combos=1, pit_ok=True)
        actual = "promote" if decision.passed else "discard"
        failed = [c["name"] for c in decision.checks if not c["passed"]]
        hm = wf.holdout_metrics
        stab = wf.stability
        stats = {
            "holdout_sharpe": float(hm.get("sharpe", 0.0)),
            "holdout_return": float(hm.get("total_return", 0.0)),
            "holdout_n_obs": float(hm.get("n_bars", 0)),
            "pct_positive_windows": float(stab.get("pct_positive_windows", 0.0)),
            "min_window_sharpe": float(stab.get("min_sharpe", 0.0)),
        }
        outcome = _classify(scenario.kind, actual)
        # Format the reason from the CLEANED sharpe so a non-finite metric cannot leak a
        # "nan"/"inf" token into the (otherwise JSON-clean) report string.
        sharpe_clean = _clean(stats["holdout_sharpe"])
        sharpe_str = f"{sharpe_clean:.3f}" if sharpe_clean is not None else "n/a"
        reason = f"{outcome}: holdout_sharpe={sharpe_str} vs bar; failed={failed or 'none'}"
        return RunTrace(
            scenario=scenario.name, kind=scenario.kind, seed=seed, actual=actual,
            outcome=outcome, failed_checks=failed, stats=stats, reason=reason,
        )
    except Exception as exc:  # noqa: BLE001 - a crash is a recorded outcome, never a sweep abort
        return RunTrace(
            scenario=scenario.name, kind=scenario.kind, seed=seed, actual=CRASHED,
            outcome=CRASHED, failed_checks=[], stats={},
            reason=f"crashed: {type(exc).__name__}: {exc}",
        )


def run_scenario(
    scenario: Scenario, k: int = DEFAULT_K, *, strategy: LoadedStrategy | None = None
) -> ScenarioResult:
    # Fail closed on an unknown kind: a mistyped kind would otherwise be silently treated as
    # ``ambiguous`` (see ``_classify``) and vanish from the ``discard``/``promote`` safety
    # denominators in ``run_eval`` — hiding a catastrophic false-promote. Reject it loudly.
    if scenario.kind not in VALID_KINDS:
        raise ValueError(
            f"scenario {scenario.name!r} has unknown kind {scenario.kind!r}; "
            f"expected one of {sorted(VALID_KINDS)}"
        )
    traces = [run_one(scenario, seed, strategy=strategy) for seed in range(k)]
    n_promote = sum(1 for t in traces if t.actual == "promote")
    n_discard = sum(1 for t in traces if t.actual == "discard")
    n_crashed = sum(1 for t in traces if t.actual == CRASHED)
    non_crashed = n_promote + n_discard
    promote_rate = (n_promote / non_crashed) if non_crashed else 0.0

    if scenario.kind == "ambiguous":
        n_correct: int | None = None
        pass_at_k: bool | None = None
        pass_pow_k: bool | None = None
    else:
        want = "promote" if scenario.kind == "promote" else "discard"
        n_correct = sum(1 for t in traces if t.actual == want)
        # A crash is never "correct". pass^k requires ALL k seeds correct (so a crash fails it).
        pass_at_k = n_correct >= 1
        pass_pow_k = n_correct == k
    return ScenarioResult(
        scenario=scenario.name, kind=scenario.kind, k=k,
        n_promote=n_promote, n_discard=n_discard, n_crashed=n_crashed,
        n_correct=n_correct, pass_at_k=pass_at_k, pass_pow_k=pass_pow_k,
        promote_rate=promote_rate, traces=traces,
    )


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:  # noqa: BLE001 - provenance is best-effort, never fatal
        return None


def _algua_version() -> str:
    try:
        return version("algua")
    except PackageNotFoundError:
        return "unknown"


def run_eval(
    scenarios: tuple[Scenario, ...] = SCENARIO_BANK, k: int = DEFAULT_K
) -> EvalReport:
    results = [run_scenario(s, k) for s in scenarios]
    traces = [t for r in results for t in r.traces]

    # Headline rates over RUNS (not scenarios). Crashes excluded from rate denominators so a
    # broken provider/gate cannot deflate false_promote_rate to a false 0.0 — crash_rate guards it.
    discard_runs = [t for t in traces if t.kind == "discard" and t.actual != CRASHED]
    promote_runs = [t for t in traces if t.kind == "promote" and t.actual != CRASHED]
    labelled_runs = discard_runs + promote_runs

    n_wrong_promote = sum(1 for t in discard_runs if t.actual == "promote")
    n_wrong_discard = sum(1 for t in promote_runs if t.actual == "discard")
    false_promote_rate = (n_wrong_promote / len(discard_runs)) if discard_runs else 0.0
    false_discard_rate = (n_wrong_discard / len(promote_runs)) if promote_runs else 0.0
    crash_rate = (sum(1 for t in traces if t.actual == CRASHED) / len(traces)) if traces else 0.0
    n_correct_labelled = (len(discard_runs) - n_wrong_promote) + (
        len(promote_runs) - n_wrong_discard
    )
    accuracy = (n_correct_labelled / len(labelled_runs)) if labelled_runs else 0.0

    labelled = [r for r in results if r.kind != "ambiguous"]
    pass_at_k = (
        sum(1 for r in labelled if r.pass_at_k) / len(labelled) if labelled else 0.0
    )
    pass_pow_k = (
        sum(1 for r in labelled if r.pass_pow_k) / len(labelled) if labelled else 0.0
    )

    # Confusion matrix by scenario band x actual decision (crashes shown as their own column).
    confusion: dict[str, dict[str, int]] = {}
    for r in results:
        confusion[r.scenario] = {
            "promote": r.n_promote, "discard": r.n_discard, "crashed": r.n_crashed,
        }
    failure_histogram = dict(Counter(t.outcome for t in traces))

    provenance = EvalProvenance(
        start=START.date().isoformat(),
        end=END.date().isoformat(),
        seeds=list(range(k)),
        criteria=_criteria_dict(GateCriteria()),
        provider="SyntheticProvider",
        strategy_id=STRATEGY_ID,
        bank_version=BANK_VERSION,
        algua_version=_algua_version(),
        git_sha=_git_sha(),
    )
    return EvalReport(
        k=k, scenarios=results,
        false_promote_rate=false_promote_rate, false_discard_rate=false_discard_rate,
        crash_rate=crash_rate, accuracy=accuracy,
        pass_at_k=pass_at_k, pass_pow_k=pass_pow_k,
        confusion=confusion, failure_histogram=failure_histogram, provenance=provenance,
    )


def _criteria_dict(c: GateCriteria) -> dict[str, Any]:
    return {
        "min_holdout_sharpe": c.min_holdout_sharpe,
        "min_holdout_return": c.min_holdout_return,
        "min_pct_positive_windows": c.min_pct_positive_windows,
        "min_window_sharpe": c.min_window_sharpe,
        "min_holdout_observations": c.min_holdout_observations,
        "n_combos": 1,
        "pit_ok": True,
    }
