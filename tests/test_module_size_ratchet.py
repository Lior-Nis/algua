"""Structural ratchet: carved files cannot regrow, and new god-files cannot appear.

Nine stages of a strangler refactor cut `engine.py` 837->322, `gates.py` 668->544,
`operator_cmd.py` 611->334 and moved ~4,000 lines out of the CLI into domain modules. **None of
that is durable without this test.** Line budgets rot silently: nobody notices a module growing 40
lines a quarter, and in a year the god-file is back with a different name.

Pattern precedent: the #277 AST data-wall test (`tests/test_data_wall.py`) — a pytest-visible
re-assertion of a property that would otherwise only hold by convention.

HOW IT WORKS (shrink-only ratchet)
    Every module at or above 300 lines is pinned at the size it had when this test was written.
    * A pinned module may SHRINK freely — that is the direction the program is pushing.
    * A pinned module may NOT grow past its pin.
    * A NEW module may not appear at or above 300 lines.

WHEN THIS TEST FAILS, the fix is almost never "raise the number". It is one of:
    1. The change belongs in a different module -> put it there.
    2. The module is doing two jobs -> carve it, and DELETE its pin.
    3. The growth is genuinely essential -> lower the pin to the new size in the SAME commit,
       with the reason in the commit message. Raising a pin is a design decision, not a chore;
       it should be visible in review, which is exactly what editing this file forces.

Ratcheting DOWN is free and encouraged: shrink a module, lower its pin, and the floor moves with
you. `test_pins_are_not_stale` nags when a pin drifts far above reality so the ratchet keeps biting.
"""

from __future__ import annotations

import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]

#: Modules at or above this size are pinned. Below it, grow freely.
FLOOR = 300

#: How far a pin may drift above the real size before `test_pins_are_not_stale` asks you to lower
#: it. Generous enough that ordinary churn does not nag; tight enough that a module which halved
#: cannot hide behind its old ceiling.
STALE_SLACK = 80

#: path -> max lines. Generated from reality at the end of the simplification program; every entry
#: is a ceiling, never a target.
BUDGET: dict[str, int] = {
    # Crossed the floor adding the attempt cap (#649 follow-up): an idea that never reaches a
    # verdict now retires as `discarded` instead of being re-claimed forever. The module stays one
    # cohesive thing -- claims, attempts, outcomes for the idea pool -- and splitting the cap away
    # from the outcome transition it guards would put two halves of one rule in two files.
    "algua/registry/idea_attempts.py": 332,
    "algua/backtest/bootstrap.py": 365,
    "algua/backtest/decision_path.py": 382,
    "algua/backtest/engine.py": 322,
    "algua/backtest/sweep.py": 461,
    "algua/backtest/walkforward.py": 347,
    "algua/cli/data_cmd.py": 410,
    # +26 for #560: paper/live PARITY -- the live lane now surfaces and audits a venue-blocked
    # leg (it previously dropped it silently, which is worse in the money lane than in the
    # rehearsal one) and refunds the reservation the venue refused.
    "algua/cli/live_cmd.py": 775,
    "algua/cli/operator_cmd.py": 334,
    # +32 for #560: auditing a venue-BLOCKED leg (wash trade) and refunding a reservation the
    # venue refused. (An earlier contained-vs-uncontained breach split was REVERTED -- flatten
    # reports success once offsets are submitted, which is not proof of a flat book.)
    "algua/cli/paper_cmd.py": 1441,
    "algua/cli/registry_cmd.py": 446,
    # +29 for #560: ExecutionContract.target_gross_utilization. The field is 2 lines; the rest is
    # the comment deriving the 0.95 default from an explicit between-tick move policy (u <= 1/(1+r))
    # and saying what it does NOT solve. It belongs on the contract it constrains.
    "algua/contracts/types.py": 455,
    # +12 for #560: submits now recover from Alpaca's duplicate-client_order_id 422 instead of
    # aborting the cycle. The concern itself was carved to algua/execution/alpaca_rejections.py;
    # what stayed here is the minimum wiring both submit paths share.
    "algua/execution/alpaca_broker.py": 534,
    # +17 for #560: delete_live_order, the LIVE mirror of delete_paper_venue_order. flatten now
    # records an intent before a submit that may end with no order existing, so both lanes need to
    # retract the phantom row.
    # Crossed the 300-line floor with #560's `blocked` verdict: a strategy whose newest tick had a
    # leg refused by the venue is not `ok`. Six lines in the module that already owns every other
    # health verdict -- splitting the precedence chain across two files would be worse.
    "algua/execution/fleet_health.py": 306,
    "algua/execution/live_ledger.py": 637,
    # +26 for #560: client_order_id fails closed instead of truncating (truncation let distinct
    # decisions collide on one id), plus the venue_blocked column on the tick writer and reader.
    # Mostly the comment recording why truncation was unsafe and why a digest was tried and
    # reverted.
    "algua/execution/order_state.py": 416,
    "algua/knowledge/sync.py": 475,
    # +12 for #560: TickResult.blocked plus the wash-trade branch in the submit loop. The
    # classification lives in algua/execution/alpaca_rejections.py; this is the plumbing that
    # carries it out to the lane.
    "algua/live/live_loop.py": 435,
    "algua/models/registry.py": 329,
    "algua/operator/gitops.py": 315,
    "algua/operator/loop_health.py": 321,
    "algua/operator/mergeback.py": 668,
    # +9 for #560: the venue_blocked exclusion filter. A tick whose leg the venue refused did not
    # execute the strategy's decision and must never count as forward evidence.
    "algua/registry/forward_evidence.py": 410,
    "algua/registry/mergeback_intake.py": 466,
    "algua/registry/promote_run.py": 348,
    "algua/registry/promotion.py": 542,
    "algua/registry/repository.py": 965,
    "algua/registry/store/crud.py": 386,
    "algua/registry/store/family.py": 470,
    "algua/registry/store/gate.py": 594,
    "algua/research/eval_harness.py": 423,
    "algua/research/forward_gates.py": 392,
    "algua/research/gates.py": 544,
    "algua/research/regime.py": 365,
    # +14 for #560: the gross-utilization step at the shared construction chokepoint. The logic
    # itself lives in portfolio/construction.py; this is the wiring plus why it runs last.
    "algua/strategies/base.py": 394,
    "algua/tracking/mlflow_tracker.py": 316,
}


def _size(path: str) -> int:
    return len((REPO / path).read_text().splitlines())


def _live_modules() -> dict[str, int]:
    out = {}
    for p in sorted((REPO / "algua").rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        out[str(p.relative_to(REPO))] = len(p.read_text().splitlines())
    return out


def test_no_pinned_module_grew_past_its_budget():
    """A carved file may not regrow."""
    over = []
    for path, cap in sorted(BUDGET.items()):
        if not (REPO / path).exists():
            continue  # deletion/rename is handled by test_budget_has_no_dead_entries
        n = _size(path)
        if n > cap:
            over.append(f"{path}: {n} lines > pin {cap} (+{n - cap})")
    assert not over, (
        "these modules grew past their pin:\n  " + "\n  ".join(over) +
        "\n\nRaising the pin is the LAST option. Prefer: put the change in the module that owns "
        "it, or carve this one. If the growth is genuinely essential, lower the pin in the same "
        "commit and say why — this file is edited in review on purpose."
    )


def test_no_new_god_file_appeared():
    """A module that did not exist (or was small) at ratchet time may not show up oversized."""
    newcomers = [
        f"{path}: {n} lines"
        for path, n in sorted(_live_modules().items())
        if n >= FLOOR and path not in BUDGET
    ]
    assert not newcomers, (
        "new modules at or above the floor:\n  " + "\n  ".join(newcomers) +
        f"\n\nA new module >= {FLOOR} lines is a god-file forming. Split it, or add it to BUDGET "
        "in this commit with the reason — the point is that it cannot happen silently."
    )


def test_pins_are_not_stale():
    """A module that shrank well below its pin should have the pin lowered, or the ratchet stops
    biting: the gap becomes free headroom for the next person to grow back into."""
    slack = [
        f"{path}: pinned {cap}, actually {_size(path)} (lower the pin by {cap - _size(path)})"
        for path, cap in sorted(BUDGET.items())
        if (REPO / path).exists() and cap - _size(path) > STALE_SLACK
    ]
    assert not slack, (
        "these pins drifted above reality:\n  " + "\n  ".join(slack) +
        "\n\nLower them — a pin far above the real size is headroom to regrow into, which is the "
        "opposite of a ratchet."
    )


def test_budget_has_no_dead_entries():
    """A pin for a module that no longer exists is noise that hides a real miss."""
    dead = [path for path in sorted(BUDGET) if not (REPO / path).exists()]
    assert not dead, (
        "BUDGET pins modules that no longer exist:\n  " + "\n  ".join(dead) +
        "\n\nIf they were renamed, re-pin under the new path so the ceiling travels with the code."
    )
