"""The planner-visible failure vocabularies are closed, named sets (Story 1.3c contract §7).

A frozen planner child reports a risk breach by `risk_kind` and a planner refusal by `code`; the
supervisor accepts only members of `RISK_BREACH_KINDS` / `PLANNER_FAILURE_CODES`. Those constants
are only safe if they are EXHAUSTIVE: a kind the in-process planner can raise but the constant
omits would turn a real breach into a `frozen_result_invalid` tenant failure (no kill switch, no
dark-feed halt), and a stale entry would widen what a child may claim.

So these scans read every construction site under `algua/` and require:
  * every literal kind/code is in its constant, and every constant member is still constructed
    somewhere (equality, so the constant cannot rot in either direction);
  * every site whose kind/code the scan cannot read (a forwarded variable, `*args`, `**kwargs`,
    an f-string) is one of the reviewed forwarding sites below;
  * no class subclasses a carrier (a subclass could hard-code a kind the call scan never sees).

`test_scan_agrees_with_real_execution` fuzzes the scanner itself against `exec`: every shape a
kind can actually be constructed through must be reported, literally or as a flagged site.
Known blind spot: a carrier reached through `getattr(module, "RiskBreach")` is not seen.
"""

from __future__ import annotations

import ast
import pathlib
import textwrap
from dataclasses import dataclass, field

import pytest

from algua.live.planner_contract import PLANNER_FAILURE_CODES
from algua.risk.limits import DARK_FEED_KINDS, RISK_BREACH_KINDS, RiskBreach

REPO = pathlib.Path(__file__).resolve().parents[1]

#: carrier name -> the parameter holding its kind.
RISK_CARRIERS = {"RiskBreach": "kind", "PlannerRiskFailure": "kind"}
#: carrier name -> the parameter holding its code (`input_failure` is planner_early's helper).
FAILURE_CARRIERS = {"PlannerInputFailure": "code", "PhaseBindingFailure": "code",
                    "input_failure": "code"}

#: Reviewed sites that forward a kind/code which was itself literal at its origin. A new dynamic
#: site must be added here in the same change, after checking that what it forwards is validated.
RISK_FORWARDING_SITES = {
    # Re-raises a planner result's kind; the planner built it from a RiskBreach via risk_failure.
    ("algua/live/live_loop.py", "_raise_planner_failure", "result.kind"),
    # Copies a caught RiskBreach into the planner's result value.
    ("algua/live/planner_early.py", "risk_failure", "exc.kind"),
    # Decodes a frozen child's risk_kind, refused unless it is in RISK_BREACH_KINDS.
    ("algua/live/frozen_wire_result.py", "_decode_body", "risk_kind"),
    # Re-wraps a decoded child risk_failure or a caught supervisor RiskBreach with a sanitized
    # detail; the helper itself refuses any kind outside RISK_BREACH_KINDS before constructing.
    ("algua/live/frozen_dispatch.py", "FrozenPlanner._risk", "kind"),
}
FAILURE_FORWARDING_SITES = {
    # The helper itself: every caller passes a literal, which the scan checks at the call.
    ("algua/live/planner_early.py", "input_failure", "code"),
}


@dataclass
class Scan:
    literals: dict[str, list[str]] = field(default_factory=dict)
    dynamic: set[tuple[str, str, str]] = field(default_factory=set)
    subclasses: list[str] = field(default_factory=list)


class _Visitor(ast.NodeVisitor):
    def __init__(self, path: str, carriers: dict[str, str], scan: Scan) -> None:
        self.path, self.carriers, self.scan = path, carriers, scan
        self.local: dict[str, str] = {name: name for name in carriers}
        self.stack: list[str] = []

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name in self.carriers:
                self.local[alias.asname or alias.name] = alias.name

    def _carrier(self, func: ast.expr) -> str | None:
        if isinstance(func, ast.Name):
            return self.local.get(func.id)
        if isinstance(func, ast.Attribute) and func.attr in self.carriers:
            return func.attr
        return None

    def _scoped(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> None:
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    visit_FunctionDef = visit_AsyncFunctionDef = _scoped

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if any(self._carrier(base) for base in node.bases):
            self.scan.subclasses.append(f"{self.path}:{node.lineno} class {node.name}")
        self._scoped(node)

    def visit_Call(self, node: ast.Call) -> None:
        carrier = self._carrier(node.func)
        if carrier is not None:
            param = self.carriers[carrier]
            arg: ast.expr | None = None
            if node.args and not isinstance(node.args[0], ast.Starred):
                arg = node.args[0]
            else:
                arg = next((kw.value for kw in node.keywords if kw.arg == param), None)
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                site = f"{self.path}:{node.lineno}"
                self.scan.literals.setdefault(arg.value, []).append(site)
            else:
                shown = "<missing>" if arg is None else ast.unparse(arg)
                self.scan.dynamic.add((self.path, ".".join(self.stack), shown))
        self.generic_visit(node)


def scan_sources(sources: dict[str, str], carriers: dict[str, str]) -> Scan:
    scan = Scan()
    for path, source in sorted(sources.items()):
        _Visitor(path, carriers, scan).visit(ast.parse(source, filename=path))
    return scan


def _algua_sources() -> dict[str, str]:
    return {
        str(p.relative_to(REPO)): p.read_text()
        for p in sorted((REPO / "algua").rglob("*.py"))
        if "__pycache__" not in p.parts
    }


def _assert_closed(scan: Scan, constant: frozenset[str], sites: set, label: str) -> None:
    unknown = {value: where for value, where in scan.literals.items() if value not in constant}
    assert not unknown, (
        f"literal {label}(s) not in the named constant: {unknown}. Add them to the constant in "
        "the same change — a frozen child reporting one would otherwise be rejected as invalid."
    )
    stale = sorted(constant - set(scan.literals))
    assert not stale, f"constant names {label}(s) nothing constructs any more: {stale}"
    assert scan.dynamic == sites, (
        f"unreviewed dynamic {label} site(s): {sorted(scan.dynamic - sites)}; "
        f"vanished reviewed site(s): {sorted(sites - scan.dynamic)}"
    )
    assert not scan.subclasses, f"carrier subclasses hide their {label}: {scan.subclasses}"


def test_risk_breach_kinds_is_exactly_the_contract_vocabulary():
    assert isinstance(RISK_BREACH_KINDS, frozenset)
    assert RISK_BREACH_KINDS == {
        "drawdown", "gross_exposure", "gross_exposure_realized", "long_only",
        "max_weight_per_symbol", "non_finite_weight", "non_positive_equity", "out_of_universe",
        "reconcile", "stale_marks", "unvaluable_marks",
    }
    assert DARK_FEED_KINDS < RISK_BREACH_KINDS


def test_planner_failure_codes_is_exactly_the_story_1_3a_vocabulary():
    assert isinstance(PLANNER_FAILURE_CODES, frozenset)
    assert PLANNER_FAILURE_CODES == {
        # Phase A input validation.
        "unsupported_boundary_version", "invalid_request_id", "strategy_identity_mismatch",
        "invalid_deployment_identity", "invalid_config_hash", "invalid_resolved_config",
        "invalid_now", "unsupported_timeframe", "invalid_calendar", "invalid_raw_bars",
        "invalid_early_input", "invalid_max_drawdown", "invalid_gate_universe",
        # Phase B binding and captured-state validation.
        "phase_a_outcome_mismatch", "invalid_phase_a_binding", "phase_a_binding_mismatch",
        "request_id_mismatch", "invalid_venue_belief", "missing_decision_time",
        "invalid_captured_state",
    }


def test_every_risk_breach_kind_constructed_under_algua_is_named():
    scan = scan_sources(_algua_sources(), RISK_CARRIERS)
    _assert_closed(scan, RISK_BREACH_KINDS, RISK_FORWARDING_SITES, "risk kind")


def test_every_planner_failure_code_constructed_under_algua_is_named():
    scan = scan_sources(_algua_sources(), FAILURE_CARRIERS)
    _assert_closed(scan, PLANNER_FAILURE_CODES, FAILURE_FORWARDING_SITES, "failure code")


_SHAPES = {
    "positional": 'from algua.risk.limits import RiskBreach\n'
                  'def f():\n    raise RiskBreach("k_positional", "d")',
    "keyword": 'from algua.risk.limits import RiskBreach\n'
               'def f():\n    raise RiskBreach(detail="d", kind="k_keyword")',
    "alias": 'from algua.risk.limits import RiskBreach as RB\n'
             'def f():\n    raise RB("k_alias", "d")',
    "attribute": 'import algua.risk.limits as limits\n'
                 'def f():\n    raise limits.RiskBreach("k_attribute", "d")',
    "nested": 'from algua.risk.limits import RiskBreach\n'
              'def f():\n    def g():\n        return RiskBreach("k_nested", "d")\n    raise g()',
    "variable": 'from algua.risk.limits import RiskBreach\nKIND = "k_variable"\n'
                'def f():\n    raise RiskBreach(KIND, "d")',
    "fstring": 'from algua.risk.limits import RiskBreach\n'
               'def f():\n    raise RiskBreach(f"k_{1}", "d")',
    "starred": 'from algua.risk.limits import RiskBreach\nARGS = ("k_starred", "d")\n'
               'def f():\n    raise RiskBreach(*ARGS)',
    "kwargs": 'from algua.risk.limits import RiskBreach\n'
              'KW = {"kind": "k_kwargs", "detail": "d"}\n'
              'def f():\n    raise RiskBreach(**KW)',
    "subclass": 'from algua.risk.limits import RiskBreach\n'
                'class Mine(RiskBreach):\n    def __init__(self):\n'
                '        super().__init__("k_subclass", "d")\n'
                'def f():\n    raise Mine()',
}


@pytest.mark.parametrize("shape", sorted(_SHAPES))
def test_scan_agrees_with_real_execution(shape):
    source = textwrap.dedent(_SHAPES[shape])
    namespace: dict[str, object] = {}
    exec(compile(source, f"<{shape}>", "exec"), namespace)
    with pytest.raises(RiskBreach) as raised:
        namespace["f"]()
    scan = scan_sources({f"algua/{shape}.py": source}, RISK_CARRIERS)
    reported = raised.value.kind in scan.literals or scan.dynamic or scan.subclasses
    assert reported, f"{shape}: executed kind {raised.value.kind!r} escaped the scan: {scan}"
    if raised.value.kind in scan.literals:
        assert not scan.dynamic and not scan.subclasses


def test_risk_breach_messages_are_plain_ascii():
    # A frozen child's breach detail is sanitized to printable ASCII (Story 1.3c §7); planner and
    # risk messages that are already ASCII keep frozen and in-process kill-switch text identical.
    offenders = []
    for path in sorted((REPO / "algua").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if name != "RiskBreach":
                continue
            for part in ast.walk(node):
                if isinstance(part, ast.Constant) and isinstance(part.value, str) and not (
                        part.value.isascii()):
                    offenders.append(f"{path.relative_to(REPO)}:{node.lineno}")
    assert offenders == []
