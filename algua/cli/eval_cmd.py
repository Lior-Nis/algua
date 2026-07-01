"""`algua eval` — the operator gate-decision eval harness (#347).

Named `eval gate` (not `eval operator`): it measures the promote/discard DECISION layer of the
autonomous loop, not the full LLM-driven loop. See algua/research/eval_harness.py for the honest
scope statement. Emits JSON on stdout per the data-contract rule.
"""
from __future__ import annotations

import typer

from algua.cli._common import ok
from algua.cli.app import app, emit
from algua.cli.errors import json_errors
from algua.research.eval_harness import SCENARIO_BANK, run_eval

eval_app = typer.Typer(
    help="Operator gate-decision eval: pass@k/pass^k + error-analysis (#347)",
    no_args_is_help=True,
)
app.add_typer(eval_app, name="eval")


@eval_app.command("gate")
@json_errors
def gate(
    k: int = typer.Option(8, "--k", min=1, help="seeds (Monte-Carlo paths) per scenario"),
    scenario: list[str] = typer.Option(
        None, "--scenario", help="restrict to these scenario name(s); repeatable"
    ),
) -> None:
    """Run the gate-decision eval over the seeded scenario bank and emit the report as JSON.

    Headline safety metric is `false_promote_rate` (a promoted no-edge/losing book). `pass@k` and
    `pass^k` (aliased `any_seed_correct`/`all_seed_correct`) are Monte-Carlo path-robustness of the
    gate decision, NOT operator-attempt consistency — see the module docstring for the honest scope.
    """
    bank = SCENARIO_BANK
    if scenario:
        by_name = {s.name: s for s in SCENARIO_BANK}
        unknown = [n for n in scenario if n not in by_name]
        if unknown:
            known = ", ".join(s.name for s in SCENARIO_BANK)
            raise ValueError(f"unknown scenario(s): {unknown}; known: {known}")
        # Preserve bank order; dedup.
        wanted = set(scenario)
        bank = tuple(s for s in SCENARIO_BANK if s.name in wanted)

    report = run_eval(scenarios=bank, k=k)
    emit(ok(report.to_dict()))
