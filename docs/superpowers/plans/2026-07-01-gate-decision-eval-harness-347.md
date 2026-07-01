# Plan — Gate-Decision Eval Harness (#347)

Spec: `docs/superpowers/specs/2026-07-01-operator-eval-harness-347-design.md`. Quality gate after
each task: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`.

## Task 1 — `algua/backtest/_sample.py` additive `drift`/`vol` params
- Add keyword-only `drift: float = 0.0005`, `vol: float = 0.02` to `SyntheticProvider.__init__`.
- Use them in `sub.normal(loc=self.drift, scale=self.vol, ...)`. Defaults preserve byte-identical
  output for `SyntheticProvider(seed=0)`.
- Test (`tests/test_synthetic_provider_params.py`): seed=0 default output unchanged vs a pinned
  expected close value; non-default drift shifts mean return upward.

## Task 2 — `algua/research/eval_harness.py` (the pure harness)
- Dataclasses per spec: `Scenario`, `RunTrace`, `ScenarioResult`, `EvalProvenance`, `EvalReport`.
- `SCENARIO_BANK` = the 4 scenarios (obvious_edge/marginal_edge/no_edge/negative_edge).
- `_build_strategy()` → canonical equal-weight-long `LoadedStrategy` (mirror the test helper:
  `equal_weight_positive`, constant `signal_fn`).
- `run_one(scenario, seed)`: `SyntheticProvider(seed, drift, vol)` → `walk_forward(..., seed=seed)`
  → `evaluate_gate(wf, GateCriteria(), n_combos=1, pit_ok=True)`. `actual = "promote" if passed
  else "discard"`. Classify `outcome` via taxonomy. `failed_checks` = `[c["name"] for c in
  decision.checks if not c["passed"]]`. `stats` from `wf.holdout_metrics`/`wf.stability`. Wrap the
  whole body in try/except → `crashed` trace (records the exception message in `reason`).
- `run_scenario(scenario, k)`: seeds `range(k)`, aggregate counts + pass_at_k/pass_pow_k (None for
  ambiguous), promote_rate over non-crashed runs.
- `run_eval(scenarios, k)`: aggregate `false_promote_rate` (over non-crashed discard runs),
  `false_discard_rate`, `crash_rate`, `accuracy`, headline `pass_at_k`/`pass_pow_k` over labelled
  scenarios, `confusion`, `failure_histogram`, `provenance` (git SHA via a cheap helper, algua
  version via `importlib.metadata`). Constants `MAX_FALSE_PROMOTE_RATE=0.0`, `MAX_CRASH_RATE=0.0`,
  `BANK_VERSION=1` exposed for tests.
- `EvalReport.to_dict()` emits all fields incl. `any_seed_correct`/`all_seed_correct` aliases and
  nested scenario/trace dicts (JSON-clean: no NaN/inf — null them).
- Tests (`tests/research/test_eval_harness.py`) per spec: safety rate bound, crash_rate==0,
  marginal no-label, mislabel→wrongly_discarded, crashed path, determinism, to_dict shape.

## Task 3 — `algua/cli/eval_cmd.py` + mount in `main.py`
- `eval_app = typer.Typer(help="Operator/gate evaluation harness (#347)", no_args_is_help=True)`;
  `app.add_typer(eval_app, name="eval")`.
- `@eval_app.command("gate")` `gate(k: int = 8, scenario: list[str] = typer.Option(None))` decorated
  with `@json_errors(...)`; subset the bank by name (unknown name → error), `run_eval`,
  `emit(ok(report.to_dict()))`.
- Add `eval_cmd` to the `from algua.cli import (...)` tuple in `main.py` (alphabetical).
- Tests (`tests/test_cli_eval.py`): `main(["eval","gate","--k","4"])` → JSON with `ok:true` +
  headline fields; `--scenario no_edge` subsets; unknown scenario → `ok:false`.

## Task 4 — full quality gate + commit, then GATE 2 review.
