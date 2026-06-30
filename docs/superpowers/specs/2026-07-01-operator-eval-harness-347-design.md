# Gate-Decision Eval Harness — false-promote rate + pass@k / pass^k + trace error-analysis (#347)

## Problem

algua evaluates **strategies** with great rigor (the promotion gates), but it **never evaluates the
agent's promote/discard judgment under resampling**. There is no measured false-promote rate, no
`pass@k` / `pass^k`, no seeded scenarios, and no error-analysis failure taxonomy. Lior's agentic KB
names **error-analysis-first** "the single most important activity" and **pass^k** the reliability
metric that matters. This is a prerequisite to trusting the "hundreds/day" north star: without it
there is no *measured* reliability for the decision the agent leans on.

## Honest scope — this is a GATE-DECISION eval, not a full operator-loop eval

Issue #347 asks (in the title) to evaluate "the autonomous OPERATOR loop." Be precise about what is
and is not built here. The autonomous loop (`run-the-research-loop` skill) is
ideate → author → backtest → walk-forward → **gate (`research promote`)** → record. The full loop is
**LLM/skill-driven with no programmatic entrypoint** (a prior `algua operator run` was removed), and
its only reproducible-offline, catastrophic-failure surface is the **promote/discard decision**:
promoting a no-edge / losing strategy to `candidate` is the failure the north star cannot tolerate.

This harness evaluates **exactly that decision layer** — the real `walk_forward → evaluate_gate`
predicate — over seeded synthetic scenarios. It does **NOT** evaluate ideation, authoring, strategy
diversity, CLI orchestration, or the stateful multiple-testing controls. The framing, the module
name, and the CLI (`algua eval gate`) all say "gate", not "operator". The **full** operator-loop eval
(replaying real operator traces) remains an explicit deferred follow-up — it needs the operator-trace
substrate from **#332** (negative-result capture, not yet built) and/or a programmatic loop
entrypoint. This is stated in the spec, the module docstring, and the issue-closing comment so no one
mistakes a gate-regression guard for full operator coverage.

## Methodology — scenarios as distributions, rates not deterministic labels

The deepest correctness point: the gate evaluates **finite-sample** walk-forward statistics, so a
zero-drift path can clear the bar by chance and a strong-drift path can fail by drawdown noise. A
single chance-promote of a no-edge path is **statistically expected**, not necessarily a bug.
Therefore the harness treats each scenario as a **distribution** and reports **empirical rates over a
seed set**, never demanding a deterministic correct label on every seed.

**Scenario bank** (a `Scenario` is `name, kind, drift, vol`; `kind` decides ground truth):

| name | kind | drift/bar | vol/bar | expected behaviour |
|---|---|---|---|---|
| `obvious_edge`   | `promote` | +0.0030 | 0.010 | true Sharpe ≫ bar ⇒ promote nearly always |
| `marginal_edge`  | `ambiguous` | +0.0009 | 0.020 | true Sharpe ≈ bar ⇒ **consistency-only**, no label |
| `no_edge`        | `discard` | 0.0     | 0.020 | true Sharpe ≈ 0 ⇒ discard nearly always |
| `negative_edge`  | `discard` | −0.0015 | 0.015 | losing book ⇒ discard nearly always |

- `promote` / `discard` scenarios are chosen **far from the 0.5 bar** so ground truth is not in
  dispute; they are scored against labels via empirical **rates**.
- `marginal_edge` deliberately straddles the bar: it has **no label** (`kind="ambiguous"`). It is
  scored for **decision consistency only** — its `promote_rate` and `pass^k`-style agreement measure
  how reproducibly the gate decides a borderline case. This is where pass^k earns its keep and where
  a gate-softening regression shows up first.

## Metrics — false-promote rate is the headline; pass@k/pass^k precisely defined

Per the review, `pass@k` ("≥1 of k seeds correct") **must not** be headlined as a safety metric: for
a discard scenario, one correct discard makes `pass@k`=True while hiding catastrophic false promotes.
So:

- **Headline safety metric: `false_promote_rate`** = (# runs that promoted a `discard`-labelled
  scenario) / (# **non-crashed** `discard` runs). This is the catastrophic-failure rate the north
  star cares about. Crashes are excluded from the denominator (a crash is not a promote) so they
  cannot deflate the rate — see `crash_rate` below.
- **`crash_rate`** = (# `crashed` runs) / (total runs). A broken gate/provider that crashes every
  discard run would otherwise leave `false_promote_rate=0.0` and look perfectly safe; `crash_rate`
  closes that hole and is asserted `<= MAX_CRASH_RATE` (default 0.0) in the default eval.
- **`false_discard_rate`** = (# runs that discarded a `promote`-labelled scenario) / (# non-crashed
  `promote` runs) — the opportunity-cost rate.
- **`accuracy`** = correct / total over labelled runs.
- **`pass_at_k`** (computed, clearly defined as `any_seed_correct` per scenario, **not** headlined as
  safety) and **`pass_pow_k`** (`all_seed_correct` per scenario) — defined narrowly as **Monte-Carlo
  path-robustness of the gate decision** (all/any resampled synthetic paths classified per label),
  NOT as "operator consistency". The headline `pass_pow_k` is the fraction of **labelled** scenarios
  whose every seed was correct; `marginal_edge` is reported separately as a consistency figure
  (`promote_rate` + agreement), never folded into the safety rates.
- **`failure_histogram`** — the error-analysis taxonomy counts (below).
- A **confusion matrix by band** (`obvious/marginal/no/negative` × `promote/discard`).

Seed-to-seed variance is **honestly labelled** as synthetic-path Monte-Carlo variation, not operator
attempt-to-attempt variance. The value: a gate that softens (or a path-dependent leak) raises
`false_promote_rate` / drops `pass_pow_k` — making this a **statistical gate-regression guard**.

## Architecture

Three new files + one additive, behavior-preserving change + a CLI mount. **No CODEOWNERS-protected
file is edited** (`gates.py` is only *called*).

### 1. `algua/research/eval_harness.py` (new, pure)

```python
@dataclass(frozen=True)
class Scenario:
    name: str
    kind: Literal["promote", "discard", "ambiguous"]
    drift: float
    vol: float
    symbols: tuple[str, ...] = ("AAA", "BBB")

SCENARIO_BANK: tuple[Scenario, ...] = (...4 scenarios above...)

@dataclass(frozen=True)
class RunTrace:
    scenario: str; kind: str; seed: int
    actual: str                 # "promote" | "discard" | "crashed"
    outcome: str                # taxonomy class
    failed_checks: list[str]    # FACTUAL: gate checks that failed (no causal claim)
    stats: dict[str, float]     # measured: holdout_sharpe, holdout_return, pct_positive_windows, n_obs
    reason: str

@dataclass(frozen=True)
class ScenarioResult:
    scenario: str; kind: str; k: int
    n_promote: int; n_discard: int; n_crashed: int
    n_correct: int | None       # None for ambiguous (no label)
    pass_at_k: bool | None      # any_seed_correct (None for ambiguous)
    pass_pow_k: bool | None     # all_seed_correct (None for ambiguous)
    promote_rate: float         # always defined (consistency figure)
    traces: list[RunTrace]

@dataclass(frozen=True)
class EvalProvenance:           # determinism/repro record (review HIGH)
    start: str; end: str; seeds: list[int]
    criteria: dict; provider: str; strategy_id: str
    bank_version: int; algua_version: str; git_sha: str | None

@dataclass(frozen=True)
class EvalReport:
    k: int
    scenarios: list[ScenarioResult]
    false_promote_rate: float           # HEADLINE safety metric (non-crashed discard runs)
    false_discard_rate: float
    crash_rate: float                   # crashes cannot masquerade as safety
    accuracy: float                     # over labelled runs
    # pass_at_k/pass_pow_k are emitted WITH explicit aliases any_seed_correct/all_seed_correct
    # in the JSON, to disarm the LLM-eval baggage of the pass@k/pass^k names.
    pass_at_k: float                    # = any_seed_correct: fraction of LABELLED scenarios, >=1 seed correct
    pass_pow_k: float                   # = all_seed_correct: fraction of LABELLED scenarios, all seeds correct
    confusion: dict[str, dict[str, int]]
    failure_histogram: dict[str, int]
    provenance: EvalProvenance
    def to_dict(self) -> dict: ...

def run_one(scenario, seed) -> RunTrace
def run_scenario(scenario, k) -> ScenarioResult
def run_eval(scenarios=SCENARIO_BANK, k=8) -> EvalReport
```

**Failure taxonomy** (`outcome`):

| label kind | actual | outcome |
|---|---|---|
| promote | promote | `correct_promote` |
| discard | discard | `correct_discard` |
| discard | promote | `wrongly_promoted` (**the catastrophic class — drives false_promote_rate**) |
| promote | discard | `wrongly_discarded` |
| ambiguous | promote/discard | `ambiguous_promote` / `ambiguous_discard` (consistency only, not a failure) |
| any | crashed | `crashed` |

`run_one` builds the canonical **equal-weight-long** `LoadedStrategy` (`equal_weight_positive`,
constant signal — mirrors the existing test helper; the decision-strategy is **fixed by design**, the
DATA varies), runs `walk_forward(strategy, provider, START, END, seed=seed)` on a seeded scenario
provider, then `evaluate_gate(wf, GateCriteria(), n_combos=1, pit_ok=True)`.
`actual = "promote" if decision.passed else "discard"`. Any exception → `crashed` (the sweep never
aborts on one bad run — per-item fail-closed, mirroring the platform). `stats` are pulled from
`wf.holdout_metrics`/`wf.stability` so each run reports the **measured** numbers (not just check
names), keeping `failed_checks` factual rather than causal.

**What this harness deliberately does NOT cover** (documented in the module + spec, deferred):
- The stateful multiple-testing controls — funnel-breadth deflation, DSR binding, FDR alpha-wealth,
  PIT enforcement (`n_combos=1`, `pit_ok=True` bypass them). Those need a populated registry + are
  single-use/global-stateful, so they cannot be re-run k times reproducibly; they are separately
  unit-tested. A future "stateful mode" driving `run_gate` against a throwaway registry is a deferred
  follow-up. The harness therefore does NOT claim to catch leakage/overfit/data-snooping; it measures
  the **core promote/discard predicate's** false-promote rate on no-edge / losing books.
- Adversarial scenarios (leaky feature, intentionally-overfit, regime break, missing/dup bars, PIT
  violation) and strategy-authoring diversity — deferred (need the stateful path + an LLM/author layer).

### 2. `algua/backtest/_sample.py` (additive, behavior-preserving)

Add keyword-only `drift` / `vol` to `SyntheticProvider.__init__`, **defaulting to the current
`0.0005` / `0.02`** so `SyntheticProvider(seed=0)` output is byte-identical (zero behavior change; a
regression test pins this). The harness realizes each scenario's true edge via
`SyntheticProvider(seed=k, drift=s.drift, vol=s.vol)` — the DRY choice (one GBM generator,
parameterized) over a duplicated eval-only provider.

### 3. `algua/cli/eval_cmd.py` (new) + mount in `algua/cli/main.py` (additive)

`eval_app = typer.Typer(...)`; `app.add_typer(eval_app, name="eval")`. One command:
`uv run algua eval gate [--k 8] [--scenario NAME]...` → `run_eval` → `emit(ok(report.to_dict()))`.
JSON to stdout (data-contract rule). `--scenario` (repeatable) subsets the bank. `main.py` gains
`eval_cmd` in its import tuple (composition root; not protected). Named `eval gate` (not `operator`)
so the CLI does not overclaim.

## Testing

`tests/research/test_eval_harness.py` (pure, fast, no DB/network):
- `obvious_edge` ⇒ `correct_promote`, `false_discard_rate <= ALPHA` at k≥4.
- `no_edge` and `negative_edge` ⇒ `correct_discard`; **`false_promote_rate <=
  MAX_FALSE_PROMOTE_RATE`** over a fixed seed set (the load-bearing safety assertion).
  **`MAX_FALSE_PROMOTE_RATE` is a deterministic CI regression guard over known-separated seeds (a
  rate threshold, default 0.0), NOT a statistical confidence bound** — it asserts "no flips in this
  fixture of well-separated scenarios", and is documented as such so it is never mistaken for
  evidence of a calibrated false-positive rate. A future `walk_forward` change that flips one seed
  surfaces as a deliberate fixture update; a systemic leak (rate jumps across seeds) fails the test.
- `crash_rate == 0` in the default eval (`<= MAX_CRASH_RATE`) — crashes cannot masquerade as safety.
- `marginal_edge` ⇒ no label; assert its `promote_rate` is reported and it is excluded from the
  safety rates / `pass_pow_k` numerator-of-labelled.
- A deliberately mislabeled fixture (label `promote` on a no-edge provider) ⇒ `wrongly_discarded`
  recorded and `pass_pow_k` reacts — proves the taxonomy + metrics are wired correctly.
- `crashed` path: inject a strategy that raises ⇒ `crashed` outcome, sweep still completes.
- `to_dict()` JSON shape, `provenance` populated, and determinism (same k ⇒ identical report modulo
  `git_sha`).
- `tests/test_cli_eval.py`: `algua eval gate --k 4` emits `ok:true` with the headline fields;
  `--scenario` subsetting works; mounted in `main.py`.
- `_sample.py` additive params: `SyntheticProvider(seed=0)` output unchanged (regression assert vs the
  existing default), and a non-default `drift` shifts the mean return.

## Quality gate

`uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`. Import
boundaries: `eval_harness.py` is in `algua/research/` and imports `algua/backtest` +
`algua/strategies` + `algua/portfolio` (same direction `gates.py`/`promotion.py` already use); the
CLI imports the harness. No new cross-boundary edge.

## Why this is the simplest useful thing

It measures the agent's one reproducible-offline catastrophic surface (`false_promote_rate`) with the
KB's named metrics (`pass@k`/`pass^k`, precisely and honestly defined), reuses the existing seeded
provider + pure gate, adds zero state and zero hot-path risk (new files + one additive,
behavior-preserving param), ships a permanent statistical gate-regression guard, and **names itself
honestly** (`eval gate`, not `eval operator`) with the full operator-loop eval, stateful
multiple-testing coverage, and adversarial/authoring scenarios written down as deferred follow-ups
rather than half-built.
