"""A consumer's own error stays primary when closing its walk also fails."""
from __future__ import annotations

import ast
import errno
import importlib.util
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

import pytest

from algua.primitives import bounded_walk as walk_module
from algua.primitives.bounded_walk import TraversalLimitExceeded
from tests._walk_faults import track_closes

REPO = Path(__file__).resolve().parents[2]


class TypedRefusal(ValueError):
    """Stands in for a consumer's typed validation error."""


def _scoped(root: Path, *, files: int = 100):
    return walk_module.scoped_walk(
        root, max_files=files, max_directories=100, max_path_bytes=1024)


def _chain(root: Path) -> None:
    (root / "d1/d2/d3").mkdir(parents=True)
    for index in range(5):
        (root / f"d1/d2/f{index}").touch()


@pytest.mark.parametrize("fault", [RuntimeError, ValueError, errno.EIO])
def test_a_consumer_error_stays_primary_when_closing_the_walk_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: object,
) -> None:
    _chain(tmp_path)
    closed = track_closes(monkeypatch, faulty=tmp_path / "d1", fault=fault)

    with pytest.raises(TypedRefusal), _scoped(tmp_path) as tree:
        for entry in tree:
            if entry.relative == "d1/d2/d3":
                raise TypedRefusal("consumer refused this entry")

    assert len(closed) == 3 and all(closed.values()), closed


def test_a_generator_exit_from_a_close_is_visible_after_an_early_scope_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _chain(tmp_path)
    closed = track_closes(monkeypatch, faulty=tmp_path / "d1", fault=GeneratorExit)

    with pytest.raises(RuntimeError) as caught, _scoped(tmp_path) as tree:
        for entry in tree:
            if entry.relative == "d1/d2/d3":
                break

    assert type(caught.value) is getattr(walk_module, "WalkCleanupError", None)
    assert isinstance(caught.value.__cause__, GeneratorExit)
    assert all(closed.values()), closed


def test_a_generator_exit_from_a_close_never_displaces_a_consumer_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _chain(tmp_path)
    closed = track_closes(monkeypatch, faulty=tmp_path / "d1", fault=GeneratorExit)

    with pytest.raises(TypedRefusal), _scoped(tmp_path) as tree:
        for entry in tree:
            if entry.relative == "d1/d2/d3":
                raise TypedRefusal("consumer refused this entry")

    assert all(closed.values()), closed


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
def test_a_cleanup_interrupt_is_never_swallowed_by_a_consumer_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interrupt: type[BaseException],
) -> None:
    _chain(tmp_path)
    closed = track_closes(monkeypatch, faulty=tmp_path / "d1", fault=interrupt)

    with pytest.raises(interrupt) as caught, _scoped(tmp_path) as tree:
        for entry in tree:
            if entry.relative == "d1/d2/d3":
                raise TypedRefusal("consumer refused this entry")

    assert isinstance(caught.value.__cause__, TypedRefusal)
    assert all(closed.values()), closed


def test_an_early_exit_still_reports_the_cleanup_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _chain(tmp_path)
    closed = track_closes(monkeypatch, faulty=tmp_path / "d1", fault=RuntimeError)

    with pytest.raises(walk_module.WalkCleanupError) as caught, _scoped(tmp_path) as tree:
        for entry in tree:
            if entry.relative == "d1/d2/d3":
                break

    assert type(caught.value.__cause__) is RuntimeError
    assert all(closed.values()), closed


def test_a_traversal_failure_stays_primary_inside_the_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _chain(tmp_path)
    closed = track_closes(monkeypatch, faulty=tmp_path / "d1", fault=RuntimeError)

    with pytest.raises(TraversalLimitExceeded), _scoped(tmp_path, files=1) as tree:
        for _entry in tree:
            pass

    assert all(closed.values()), closed


def test_a_completed_scope_closes_cleanly(tmp_path: Path) -> None:
    _chain(tmp_path)

    with _scoped(tmp_path) as tree:
        relatives = [entry.relative for entry in tree]

    assert "d1/d2/f4" in relatives


class TypedCleanupFailure(RuntimeError):
    """Stands in for a consumer's typed cleanup error."""


def _translating(root: Path):
    return walk_module.scoped_walk(
        root, max_files=100, max_directories=100, max_path_bytes=1024,
        cleanup_error=lambda: TypedCleanupFailure("listing could not be closed"))


def test_the_walks_own_exhausted_close_failure_is_translated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _chain(tmp_path)
    closed = track_closes(monkeypatch, faulty=tmp_path / "d1/d2/d3", fault=RuntimeError)

    with pytest.raises(TypedCleanupFailure) as caught, _translating(tmp_path) as tree:
        for _entry in tree:
            pass

    assert type(caught.value.__cause__) is walk_module.WalkCleanupError
    assert all(closed.values()), closed


def test_the_walks_own_close_failure_after_an_early_exit_is_translated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _chain(tmp_path)
    closed = track_closes(monkeypatch, faulty=tmp_path / "d1", fault=RuntimeError)

    with pytest.raises(TypedCleanupFailure), _translating(tmp_path) as tree:
        for entry in tree:
            if entry.relative == "d1/d2/d3":
                break

    assert all(closed.values()), closed


def test_a_walk_cleanup_error_raised_by_the_body_is_never_translated(tmp_path: Path) -> None:
    _chain(tmp_path)
    body_error = walk_module.WalkCleanupError("raised by the consumer body itself")

    with pytest.raises(walk_module.WalkCleanupError) as caught, _translating(tmp_path) as tree:
        for entry in tree:
            if entry.relative == "d1/d2/d3":
                raise body_error

    assert caught.value is body_error


def test_an_early_scope_exit_retries_a_listing_that_failed_before_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _chain(tmp_path)
    closed = track_closes(monkeypatch, faulty=tmp_path / "d1", release=False, times=1)

    with pytest.raises(walk_module.WalkCleanupError), _scoped(tmp_path) as tree:
        for entry in tree:
            if entry.relative == "d1/d2/d3":
                break

    assert len(closed) == 3 and all(closed.values()), closed


def test_a_consumer_error_retries_a_listing_that_failed_before_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _chain(tmp_path)
    closed = track_closes(monkeypatch, faulty=tmp_path / "d1", release=False, times=1)

    with pytest.raises(TypedRefusal), _scoped(tmp_path) as tree:
        for entry in tree:
            if entry.relative == "d1/d2/d3":
                raise TypedRefusal("consumer refused this entry")

    assert len(closed) == 3 and all(closed.values()), closed


WALK_MODULE = "algua.primitives.bounded_walk"
RAW_WALK = f"{WALK_MODULE}.bounded_walk"


BUILTIN_GETATTR = "builtins.getattr"
# narrow, explicit exceptions to "any call may retain its argument": built-ins that only
# construct, test or advance what they are handed, never store or return it
NON_RETAINING_CALLS = frozenset({"bool", "next", "any", "all"})
# an explicit bound on how many `if`/`else` levels of one ternary chain are walked (iteratively,
# not recursed into): comfortably past any chain a human would write, but finite, so a truly
# pathological input still terminates rather than growing the walk without limit
MAX_IFEXP_CHAIN = 10_000
Function = ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda
Deferred = Function | ast.GeneratorExp  # analyzed after its scope's statements
Comprehension = ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp
ALIVE = "<alive>"  # the key naming the lazy objects that exist on some path to a point


class _Alive:
    """A persistent, immutable set of live-object ids. Adding one, or joining what two paths each
    have alive, is O(1) and shares every node already built rather than copying it, so a
    straight-line run of N creations costs O(N) in total, not the O(N^2) that unioning a growing
    `frozenset` on every one of them would. Iterating it, or comparing it for equality (needed for
    a loop's fixpoint check, since two differently-built sets of the same ids must still compare
    equal), instead flattens it: computed once per distinct node and cached, and walked with an
    explicit stack rather than recursed into, since a long straight-line chain would otherwise
    recurse one Python frame per id and exceed the recursion limit.

    Deliberately not a `frozenset` subclass: `set(x)`, `frozenset(x)` and `set.update(x)` all
    special-case an actual `frozenset` (or subclass) instance and copy its underlying hash table
    directly in C, bypassing any Python-level `__iter__` override -- confirmed empirically before
    this design was chosen -- so a lazily-flattened node would read as empty through those. Being
    a genuinely different type, checked with `isinstance` at the few places `Bindings` reads or
    writes ALIVE, is what keeps this sound."""

    __slots__ = ("_oid", "_left", "_right", "_flat")

    def __init__(
        self, oid: str | None = None, left: frozenset[str] | _Alive | None = None,
        right: frozenset[str] | _Alive | None = None,
    ) -> None:
        self._oid = oid
        self._left = left
        self._right = right
        self._flat: frozenset[str] | None = None

    def added(self, oid: str) -> _Alive:
        return _Alive(oid=oid, left=self)

    def flattened(self) -> frozenset[str]:
        """``self``'s set, computed and cached once per distinct node: a straight run of N
        `added` calls chains N nodes deep, so this walks that chain with an explicit stack (each
        node visited once, children before their parent) rather than recursing, which would
        raise ``RecursionError`` well before N reaches the default recursion limit. A child link
        already holding an ordinary, already-materialized `frozenset` (rather than another
        `_Alive` node still to flatten) is used as-is."""
        if self._flat is not None:
            return self._flat
        stack: list[tuple[_Alive, bool]] = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if node._flat is not None:
                continue
            if expanded:
                own = frozenset((node._oid,)) if node._oid is not None else frozenset()
                node._flat = _flat_of(node._left) | _flat_of(node._right) | own
            else:
                stack.append((node, True))
                for child in (node._left, node._right):
                    if isinstance(child, _Alive) and child._flat is None:
                        stack.append((child, False))
        return _flat_of(self)

    def __iter__(self) -> Iterator[str]:
        return iter(self.flattened())

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _Alive):
            return self.flattened() == other.flattened()
        if isinstance(other, frozenset):
            return self.flattened() == other
        return NotImplemented


def _flat_of(node: frozenset[str] | _Alive | None) -> frozenset[str]:
    """A child link's already-materialized set: `_Alive`'s own cached flatten if it is one
    (guaranteed populated -- `flattened` visits every child before its parent), the plain
    `frozenset` itself if it is already one, or nothing at all for `None`."""
    if node is None:
        return frozenset()
    if isinstance(node, _Alive):
        flat = node._flat
        assert flat is not None
        return flat
    return node


def _alive_add(alive: frozenset[str] | _Alive | None, oid: str) -> _Alive:
    """``alive`` with ``oid`` added, in O(1): everything already alive -- whether it is another
    `_Alive` node or an ordinary, already-materialized `frozenset` -- stays reachable, untouched,
    through the new node."""
    return alive.added(oid) if isinstance(alive, _Alive) else _Alive(oid=oid, left=alive)


def _alive_union(
    a: frozenset[str] | _Alive | None, b: frozenset[str] | _Alive | None,
) -> frozenset[str] | _Alive | None:
    """Everything alive in ``a`` or ``b``, in O(1): a join over both, flattened only if and when
    the result actually needs walking or comparing."""
    if a is None:
        return b
    if b is None:
        return a
    return _Alive(left=a, right=b)


# each name mapped to every walk-relevant target it may hold on some path to this point, except
# ALIVE, whose value is a `frozenset[str] | _Alive | None`, not a `frozenset[str]` alone (see
# `_Alive` for why it is a genuinely different type rather than a `frozenset` subclass)
Bindings = dict[str, frozenset[str] | _Alive]
# a lazy object a call hands back: its function, and the names it reads from the closure of the
# function that made it (whose own flow resolves them)
Returned = tuple[Deferred, frozenset[str]]


def _dotted(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return None if base is None else f"{base}.{node.attr}"
    return None


def _parameters(function: Function) -> set[str]:
    arguments = function.args
    names = {arg.arg for arg in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs)}
    names.update(arg.arg for arg in (arguments.vararg, arguments.kwarg) if arg is not None)
    return names


def _bound_names(target: ast.AST) -> set[str]:
    """The plain names a binding target (or match pattern) binds or deletes."""
    names = {
        node.id for node in ast.walk(target)
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del))
    }
    names.update(
        node.name for node in ast.walk(target)
        if isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name
    )
    names.update(
        node.rest for node in ast.walk(target) if isinstance(node, ast.MatchMapping) and node.rest
    )
    return names


def _relevant(target: str) -> bool:
    """Only a target on the way to the raw walk or `getattr` can ever resolve to either."""
    return any(
        goal == target or goal.startswith(f"{target}.") for goal in (RAW_WALK, BUILTIN_GETATTR))


def _merge(into: Bindings, state: Bindings) -> None:
    for name, targets in state.items():
        if name == ALIVE:
            merged = _alive_union(into.get(ALIVE), targets)
            if merged is not None:
                into[ALIVE] = merged
        else:
            into[name] = into.get(name, frozenset()) | targets


def _join(*states: Bindings) -> Bindings:
    joined: Bindings = {}
    for state in states:
        _merge(joined, state)
    return joined


def _irrefutable(pattern: ast.pattern) -> bool:
    """A wildcard or capture pattern, which matches every subject."""
    return isinstance(pattern, ast.MatchAs) and pattern.pattern is None


def _static_number(node: ast.expr) -> complex | None:
    """The value of a numeric or boolean literal under any unary `+`, `-` or integer `~`."""
    operators: list[ast.unaryop] = []
    while isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub, ast.Invert)):
        operators.append(node.op)  # iteratively, since a chain may be deeper than the stack
        node = node.operand
    if not isinstance(node, ast.Constant) or not isinstance(node.value, (int, float, complex)):
        return None
    # a bool as its integer, so `~True` is computed without the deprecated bool inversion
    value: complex = int(node.value) if isinstance(node.value, bool) else node.value
    for operator in reversed(operators):
        if isinstance(operator, ast.USub):
            value = -value
        elif isinstance(operator, ast.UAdd):
            value = +value
        elif isinstance(value, int):
            value = ~value
        else:
            return None
    return value


def _hashable_literal(node: ast.expr) -> bool:
    """A literal that can be a set item or dict key: no list, set or dict display within it."""
    if isinstance(node, ast.Tuple):
        return all(_hashable_literal(item) for item in node.elts)
    return not isinstance(node, (ast.List, ast.Set, ast.Dict)) and _static_truth(node) is not None


def _static_truth(node: ast.expr) -> bool | None:
    """The truth of a side-effect-free literal, decided without evaluating code, or None.

    A constant, `not` of a literal, a numeric or boolean literal under `+`, `-` or `~` and an
    f-string of constant parts qualify, and so do tuple, list, set and dict displays of literals
    without unpacking (set items and dict keys must be hashable, since anything else raises). A
    call such as `set()` never qualifies, since its name can be rebound, and neither does a
    formatted value, which can run code.
    """
    negated = False
    while isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        node, negated = node.operand, not negated  # iteratively, since a chain may be deep
    truth = _literal_truth(node)
    return None if truth is None else truth != negated


def _literal_truth(node: ast.expr) -> bool | None:
    """`_static_truth` of a literal that is not itself a `not`."""
    if isinstance(node, ast.Constant):
        return bool(node.value)
    number = _static_number(node)
    if number is not None:
        return bool(number)
    if isinstance(node, ast.JoinedStr):
        if all(isinstance(part, ast.Constant) for part in node.values):
            return any(part.value for part in node.values if isinstance(part, ast.Constant))
    elif isinstance(node, (ast.Tuple, ast.List)):
        if all(_static_truth(item) is not None for item in node.elts):
            return bool(node.elts)
    elif isinstance(node, ast.Set):
        if all(_hashable_literal(item) for item in node.elts):
            return bool(node.elts)
    elif isinstance(node, ast.Dict):
        keys = all(key is not None and _hashable_literal(key) for key in node.keys)
        if keys and all(_static_truth(value) is not None for value in node.values):
            return bool(node.keys)
    return None


def _either(states: Iterable[Bindings | None]) -> Bindings | None:
    """The join of the outcomes some path can have, or None when none can."""
    reached = [state for state in states if state is not None]
    return _join(*reached) if reached else None


def _lexical(state: Bindings) -> Bindings:
    """The names alone, without the lazy objects alive."""
    return {key: targets for key, targets in state.items() if key != ALIVE}


def _marker(function: Function) -> str:
    """The target a name holds while it refers to ``function``."""
    return f"<def {id(function)}>"


def _own_nodes(nodes: Iterable[ast.AST]) -> Iterator[ast.AST]:
    """Every node of one scope, not descending into the functions and classes nested in it."""
    stack = list(nodes)[::-1]
    while stack:  # iteratively, since an expression may be nested deeper than the stack
        node = stack.pop()
        yield node
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            stack.extend(reversed(list(ast.iter_child_nodes(node))))


def _own_body(function: Function) -> list[ast.AST]:
    """``function``'s own statements, or its one expression for a lambda, as a list."""
    return [function.body] if isinstance(function, ast.Lambda) else list(function.body)


def _declared(function: Function, kind: type | tuple[type, ...] = (ast.Global, ast.Nonlocal),
              ) -> frozenset[str]:
    """The names ``function``'s own code declares `global` and/or `nonlocal`, per ``kind``: a
    lambda can declare neither, since its body is one expression, not a statement."""
    if isinstance(function, ast.Lambda):
        return frozenset()
    return frozenset(
        name for node in _own_nodes(_own_body(function)) if isinstance(node, kind)
        for name in node.names)


def _global(function: Deferred) -> frozenset[str]:
    """The names ``function``'s own code declares `global`: only a `def` can, so a generator
    expression (which cannot contain a `global` statement at all) never has any."""
    return _declared(function, ast.Global) if isinstance(function, Function) else frozenset()


def _locals(function: Function) -> frozenset[str]:
    """The names local to ``function``: its parameters and the names its own code binds, less
    those it declares `global` or `nonlocal` and a comprehension's own targets. A lambda's body
    is one expression, not a list of statements, but a walrus in it (unlike a comprehension's)
    binds in the lambda's own scope, so it is walked exactly as a function's statements are."""
    names = _parameters(function)
    nodes = list(_own_nodes(_own_body(function)))
    targets = {
        id(name) for node in nodes if isinstance(node, ast.comprehension)
        for name in ast.walk(node.target)}
    for node in nodes:
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            continue  # accounted for by `_declared`, subtracted below
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(
                (alias.asname or alias.name).partition(".")[0]
                for alias in node.names if alias.name != "*")
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, (ast.Name, ast.pattern)) and id(node) not in targets:
            names.update(_bound_names(node))
    return frozenset(names) - _declared(function)


def _values(node: ast.expr | None) -> set[ast.AST]:
    """The expressions making the objects ``node``'s value may be, or hold as a display does.

    Walked iteratively, with an explicit worklist bounded by MAX_IFEXP_CHAIN nodes visited,
    rather than recursed into: a deeply right-nested ternary chain -- `a if c else b if c else
    ...`, exactly the shape a comprehension result may hold -- would otherwise recurse one Python
    frame per `else` and exceed the recursion limit well under a chain of hundreds of levels.
    """
    made: set[ast.AST] = set()
    stack: list[ast.expr | None] = [node]
    visited = 0
    while stack and visited < MAX_IFEXP_CHAIN:
        visited += 1
        current = stack.pop()
        if isinstance(current, (ast.Call, ast.GeneratorExp)):
            made.add(current)
        elif isinstance(current, ast.IfExp):
            stack.append(current.body)
            stack.append(current.orelse)
        elif isinstance(current, ast.BoolOp):
            stack.extend(current.values)
        elif isinstance(current, ast.NamedExpr):
            stack.append(current.value)
        elif isinstance(current, (ast.Tuple, ast.List, ast.Set)):
            stack.extend(current.elts)
        elif isinstance(current, ast.Dict):
            stack.extend(current.keys)
            stack.extend(current.values)
        elif isinstance(current, (ast.ListComp, ast.SetComp)):
            stack.append(current.elt)
        elif isinstance(current, ast.DictComp):
            stack.append(current.key)
            stack.append(current.value)
    return made


def _escaping(node: ast.ListComp | ast.SetComp | ast.DictComp) -> set[ast.AST]:
    """The expressions whose objects outlive comprehension ``node``: those its result holds, a
    walrus binds (in the enclosing scope), or a call is handed, since an unknown callee (a method
    call or a plain call of a name that is not statically known) may keep it reachable. The
    narrow, explicit exceptions in ``NON_RETAINING_CALLS`` are calls known to only construct and
    test their argument, never store or return it; passing an object to one of them, and nothing
    else, leaves it a temporary of the comprehension."""
    kept = _values(node)
    for inner in _own_nodes([node]):
        if isinstance(inner, ast.NamedExpr):
            kept |= _values(inner.value)
        elif isinstance(inner, ast.Call) and not (
            isinstance(inner.func, ast.Name) and inner.func.id in NON_RETAINING_CALLS
        ):
            for argument in (*inner.args, *(keyword.value for keyword in inner.keywords)):
                kept |= _values(argument)
    return kept


def _runs_later(function: Deferred) -> bool:
    """A generator or coroutine, whose body runs when its object is advanced or awaited."""
    if isinstance(function, (ast.AsyncFunctionDef, ast.GeneratorExp)):
        return True
    body: list[ast.AST] = [function.body] if isinstance(function, ast.Lambda) else [*function.body]
    return any(isinstance(node, (ast.Yield, ast.YieldFrom)) for node in _own_nodes(body))


def _continue_from(bindings: Bindings, ends: list[tuple[Bindings, bool]]) -> bool:
    """Continue from the join of the ends a normal path leaves; report whether there is one.

    With none, nothing after the construct runs, but a function it calls still reads the names
    it reached, so the join of every end is kept for late-bound function bodies.
    """
    leaving = [state for state, falls in ends if falls]
    joined = _join(*(leaving or [state for state, _ in ends]))
    bindings.clear()
    bindings.update(joined)
    return bool(leaving)


class _WalkReferences:
    """Statement-ordered, scope-aware resolution of every target a name may hold on some path.

    Followed statically: imports (absolute or relative, aliased or not, star), single-name plain or
    annotated assignments whose value resolves to an imported module, package, function or the
    `getattr` builtin, attribute access, and `getattr(module, "bounded_walk")`. Any other binding
    of a name (a parameter, an unresolved assignment, a loop, `with`, `except` or match target, a
    `def` or `class`, `del`) clears it, so unrelated locals are never flagged. Dynamic access
    (`__import__`, `importlib.import_module`, `vars`, `globals`, `exec`, computed attribute names)
    is out of scope: this is a guard against accidental or casual bypass, not arbitrary dynamic
    execution.

    A function or lambda body is resolved against its enclosing scope's final bindings (Python's
    late binding) minus its own parameters, joined with the bindings at each plain call of it (by
    its name, a simple alias or the lambda itself) made in the scope that defines it; a call in a
    class body reads the enclosing scope's bindings, never the class's own names. A generator or
    coroutine body (and the lazy part of a generator expression) runs whenever its object is
    advanced or awaited, so it is resolved against every state of that scope from the call that
    made the object on, along the paths the object exists on, and only those: the object is
    marked alive in the bindings, each join keeps it only where some path has it, and each later
    rebinding on a path it is alive on adds to what it saw, so a sibling path it does not exist on
    never reaches it. A lazy function no object of which is made in its scope is made elsewhere,
    and is resolved against the final bindings like a plain function. An object made inside a
    comprehension outlives it only when the result holds it, a walrus binds it or a method call is
    handed it; one a followed call hands back exists in the caller's flow from that call on. What
    a call hands back comes from the callee's feasible `return` states (or a lambda's body): an
    object made there directly (a generator expression, or a call of a generator or coroutine
    function by any definition its name may hold), through a simple alias, or as either result of
    a conditional expression or `and`/`or`. Such an object reads the callee's own names from its
    closure, which the callee's own flow resolves, and every other name late-bound in the
    caller's flow.
    Calls from another scope, or through attributes, containers, arguments or other functions,
    and objects handed back by a call inside the callee, are not followed: this is no call graph.
    Class-level names do not leak into methods, a class body, nested or not, runs from the
    lexical names, never an outer class's, and a class comprehension runs its first iterable in
    the class and the rest from the lexical names.

    Paths are joined conservatively, and only where a normal path continues. `break` and `continue`
    end their path (later statements of the block are unreachable and not visited) and carry its
    state to the loop's exit or next iteration, first through every enclosing `finally`; a `finally`
    that does not complete replaces the transfer. A condition (of an `if`, a `while` or a `match`
    guard) yields the states where it is true and where it is false: an `and`/`or` operand runs only
    while the earlier ones have not decided, a conditional expression runs one branch, `not` swaps
    the two, and a side-effect-free literal is decided statically, leaving the impossible operand,
    branch or body unvisited. An `if` or loop body starts from the true state and its `else` from
    the false one; a comprehension's later filters and element see only a filter's true state, and a
    false literal filter ends it. A later `match` case starts after an earlier one failed: its
    pattern did not match (the state before that case; a wildcard or capture always matches) or its
    guard was false. Control falls past the last case unless it cannot fail. A loop body runs zero
    or more times, each iteration starting from the entry state or any state that reaches the next
    one (the body's end or a `continue`); the loop leaves through its `else` once its test is false
    or its iterable is exhausted, or through a `break`, and a test that is never false leaves only
    by `break`. A `try` handler may start from any point of the body (an `except*` handler also
    after an earlier one), `else` follows the body alone, and `finally` may start from any point of
    the whole statement. `raise` is not modeled, nor is `return` outside the flow a call hands an
    object back from, so a join may include a path that cannot run, which only ever adds a flag.
    Only targets on the way to the raw walk or `getattr`, and the functions defined in the module,
    are kept, which keeps loop fixpoints finite. A lazy object's view lives beside the bindings,
    not in them, so the work per statement does not grow with the objects alive.
    """

    def __init__(self, package: str) -> None:
        self._package = package
        self.uses: list[str] = []
        self._watchers: list[Bindings] = []  # every state reached inside an enclosing `try`
        # the break/continue states carried to each open loop, or first to an enclosing `finally`
        self._loops: list[tuple[list[Bindings], list[Bindings]]] = []
        # the states and objects each feasible `return` hands back, or first to a `finally`
        self._exits: list[list[tuple[Bindings, frozenset[str]]]] = []
        self._defined: dict[str, Function] = {}  # each function by the target naming it
        self._calls: dict[Deferred, list[Bindings]] = {}  # the bindings each call in scope sees
        # each lazy object, by id: its function, the names it reads from a closure, and the
        # expression that made it
        self._lazy: dict[str, tuple[Deferred, frozenset[str], ast.AST]] = {}
        self._seen: dict[str, Bindings] = {}  # in the current scope, every state each object saw
        # in a class body: the lexical names it runs from, and the lazy objects it makes
        self._enclosing: tuple[Bindings, list[str]] | None = None
        # the locals of every function scope currently being resolved, innermost last: a call
        # from within one of them to a function defined elsewhere never sees a name it shadows
        self._shadow: list[frozenset[str]] = []
        # syntax summaries, computed once per node
        self._later: dict[Deferred, bool] = {}
        self._escapes: dict[ast.AST, set[ast.AST]] = {}
        self._reads: dict[Function, frozenset[str]] = {}
        self._returns: dict[tuple[Function, frozenset[tuple[str, frozenset[str]]]], list[Returned]]
        self._returns = {}
        # the targets a call of a function may leave a global or nonlocal name of its own
        # holding, from its own code alone; a function already being summarized (a direct or
        # mutual recursive helper) summarizes as having none, closing the recursion
        self._effects: dict[tuple[Function, frozenset[tuple[str, frozenset[str]]]], Bindings] = {}
        self._summarizing: set[Function] = set()

    def module(self, tree: ast.Module) -> list[str]:
        self._scope(tree.body, {"getattr": frozenset({BUILTIN_GETATTR})})
        return list(dict.fromkeys(self.uses))  # loop and `finally` bodies may be visited twice

    def _scope(self, body: list[ast.stmt], bindings: Bindings) -> None:
        seen, self._seen = self._seen, {}
        deferred: list[Deferred] = []
        self._block(body, bindings, deferred)
        # a sibling defined earlier may be called from another sibling's body, discovered only
        # once that sibling is itself resolved; a second round sees every such call recorded by
        # the first (duplicate self.uses lines from the retry are deduplicated on return)
        for _ in range(2):
            for function in deferred:
                calls = self._calls.get(function, [])
                # a lazy body runs only while one of its objects exists, so from what those
                # objects saw on their own paths; one never made here is made later, from the
                # final names
                reached = calls if calls and self._runs_later(function) else [bindings, *calls]
                entry = _lexical(_join(*reached))
                if isinstance(function, ast.GeneratorExp):
                    self._comprehension(function, entry, deferred)
                    continue
                local = {k: v for k, v in entry.items() if k not in _parameters(function)}
                if isinstance(function, ast.Lambda):
                    self._expression(function.body, local, deferred)
                else:
                    self._shadow.append(_locals(function))
                    try:
                        self._scope(function.body, local)
                    finally:
                        self._shadow.pop()
        self._seen = seen

    def _block(
        self, body: list[ast.stmt], bindings: Bindings, deferred: list[Deferred],
    ) -> bool:
        """Visit ``body`` in order; report whether a normal path reaches its end."""
        for statement in body:
            falls = self._statement(statement, bindings, deferred)
            for reached in self._watchers:
                _merge(reached, bindings)
            if not falls:
                return False  # the rest of the block is unreachable
        return True

    def _runs_later(self, function: Deferred) -> bool:
        later = self._later.get(function)
        if later is None:
            later = self._later[function] = _runs_later(function)
        return later

    def _resolved(self, node: ast.expr | None, bindings: Bindings) -> frozenset[str]:
        if isinstance(node, ast.Lambda):
            return frozenset({_marker(node)})
        dotted = None if node is None else _dotted(node)
        if dotted is None:
            return frozenset()
        head, _, rest = dotted.partition(".")
        # a name not bound here is a local or builtin name, not something imported
        targets = bindings.get(head, frozenset())
        return frozenset(f"{target}.{rest}" if rest else target for target in targets)

    def _followed(self, target: str) -> bool:
        """Whether a name holding ``target`` is kept in the bindings."""
        return _relevant(target) or target in self._defined

    def _bind(self, name: str, targets: Iterable[str], bindings: Bindings) -> None:
        self._set(name, frozenset(t for t in targets if self._followed(t)), bindings)

    def _set(self, name: str, targets: frozenset[str], bindings: Bindings) -> None:
        """Rebind ``name`` (to nothing followed when empty) on this path; every lazy object alive
        on it that reads ``name`` late-bound, not from a closure, sees the new targets."""
        if not targets:
            bindings.pop(name, None)
            return
        bindings[name] = targets
        for oid in bindings.get(ALIVE, frozenset()):
            if name not in self._lazy[oid][1]:
                seen = self._seen[oid]
                seen[name] = seen.get(name, frozenset()) | targets

    def _propagate(self, name: str, value: ast.expr, bindings: Bindings) -> None:
        """Bind ``name`` to every target ``value`` may resolve to."""
        self._bind(name, self._resolved(value, bindings), bindings)

    def _unbind(self, target: ast.AST, bindings: Bindings) -> None:
        for name in _bound_names(target):
            self._set(name, frozenset(), bindings)

    def _start(self, oid: str, bindings: Bindings) -> None:
        """The object ``oid`` exists on this path from here on, seeing the names as they are, but
        for those it reads from a closure."""
        function, closure, _ = self._lazy[oid]
        seen = self._seen.get(oid)
        if seen is None:
            seen = self._seen[oid] = {}
            self._calls.setdefault(function, []).append(seen)
        _merge(seen, {k: v for k, v in _lexical(bindings).items() if k not in closure})
        bindings[ALIVE] = _alive_add(bindings.get(ALIVE), oid)

    def _made(self, oid: str, bindings: Bindings) -> None:
        if self._enclosing:
            self._enclosing[1].append(oid)  # it exists once the class statement has run
        else:
            self._start(oid, bindings)  # it may run at any later point of the scope

    def _read_names(self, function: Function) -> frozenset[str]:
        """The plain names ``function``'s own top-level code may read (a `Load`-context `Name`),
        not descending into a nested function or lambda -- which resolves its own free names
        separately when its own turn comes. A return-summary or effect cache key is projected
        onto this, not onto every name tracked in scope so far, so an unrelated function defined
        earlier or later in the same module can never inflate it."""
        if function not in self._reads:
            self._reads[function] = frozenset(
                node.id for node in _own_nodes(_own_body(function))
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load))
        return self._reads[function]

    def _entry(self, function: Function, state: Bindings) -> Bindings:
        """``state`` projected to the names ``function`` may actually read: its parameters are
        its own, freshly bound at the call, and any name it never reads cannot affect what it
        does, so neither belongs in what a cache keys its summary or effect on."""
        reads = self._read_names(function)
        return {k: v for k, v in state.items() if k in reads and k not in _parameters(function)}

    def _returned(self, function: Function, state: Bindings) -> list[Returned]:
        """The lazy objects a call of ``function`` from ``state`` may hand back."""
        entry = self._entry(function, state)
        key = (function, frozenset(entry.items()))
        if key not in self._returns:
            self._returns[key] = self._summarize(function, entry)
        return self._returns[key]

    def _summarize(self, function: Function, entry: Bindings) -> list[Returned]:
        body = [ast.Return(function.body)] if isinstance(function, ast.Lambda) else function.body
        own = set(_own_nodes(body))
        if not any(isinstance(node, ast.Return) and node.value for node in own):
            return []
        flow = _ReturnFlow(self)
        flow._block(body, entry, [])
        made = dict.fromkeys(flow.objects[t] for _, objects in flow.returns for t in objects)
        # an object made from ``function``'s own code reads its names from the closure; one made
        # from a function of the caller's scope reads that scope's names. A name the object
        # itself declares `global` is never read from that closure, no matter what ``function``
        # binds it to: `global` skips every enclosing function scope, straight to the module's.
        closure = _locals(function)
        return [
            (lazy, (closure - _global(lazy)) if lazy in own else frozenset())
            for lazy in made
        ]

    def _effect(self, function: Function, state: Bindings) -> Bindings:
        """The targets a call of ``function`` from ``state`` may leave one of its own declared
        global or nonlocal names holding, from every feasible exit of its own code alone (falling
        through the end, or any `return`). A call ``function`` makes is never followed -- this
        bounds the work to ``function``'s own body regardless of how deep a call graph it sits in
        -- so a directly or mutually recursive helper summarizes its own in-flight call as having
        no effect, closing the recursion; the real answer, from its own code, is still cached
        once computed."""
        entry = self._entry(function, state)
        key = (function, frozenset(entry.items()))
        if key in self._effects:
            return self._effects[key]
        if function in self._summarizing:
            return {}
        declared = _declared(function)
        if not declared:
            effect: Bindings = {}
        else:
            self._summarizing.add(function)
            try:
                flow = _ReturnFlow(self)
                work = dict(entry)
                falls = flow._block(_own_body(function), work, [])
                exits = ([work] if falls else []) + [s for s, _ in flow.returns]
                merged = _lexical(_join(*exits)) if exits else {}
                effect = {name: targets for name, targets in merged.items() if name in declared}
            finally:
                self._summarizing.discard(function)
        self._effects[key] = effect
        return effect

    def _called(
        self, function: Deferred, site: ast.AST, bindings: Bindings, deferred: list[Deferred],
    ) -> None:
        """Record a call of ``function`` at ``site``: a call in the scope that defines it, or one
        made from another function's body, is followed; one from a class body is not (its own
        names never leak to what it calls, and neither does the reverse). A call from another
        function only ever forwards the names that function does not shadow with locals of its
        own, since a local of that name is never the one the callee's free reference resolves to.
        """
        if function not in deferred:
            if self._enclosing or not self._shadow:
                return
            shadow = self._shadow[-1]
            bindings = {k: v for k, v in bindings.items() if k == ALIVE or k not in shadow}
        # a class body runs at once, but what it calls reads the lexical names
        state = self._enclosing[0] if self._enclosing else _lexical(bindings)
        made: list[Returned]
        if self._runs_later(function):
            made = [(function, frozenset())]
        else:
            self._calls.setdefault(function, []).append(dict(state))
            made = self._returned(function, state)  # the lazy objects the call hands back
            if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for name, targets in self._effect(function, state).items():
                    self._set(name, targets, bindings)
        for lazy, closure in made:
            oid = f"{id(lazy)}@{id(site)}"  # one object per function and expression making it
            self._lazy[oid] = (lazy, closure, site)
            self._made(oid, bindings)

    def _comprehension(
        self, node: Comprehension, bindings: Bindings, deferred: list[Deferred],
    ) -> set[str]:
        """Analyze a comprehension in its own scope (a generator's first iterable ran when made);
        return the lazy objects alive in it."""
        local = dict(bindings)
        alive: set[str] = set()
        enclosing = self._enclosing
        try:
            for index, generator in enumerate(node.generators):
                if index or not isinstance(node, ast.GeneratorExp):
                    self._expression(generator.iter, local, deferred)
                if self._enclosing:
                    # only the first iterable runs in the class; the rest runs in the
                    # comprehension's own scope, which never sees the class's names
                    local, self._enclosing = dict(self._enclosing[0]), None
                self._unbind(generator.target, local)
                for condition in generator.ifs:
                    truthy, falsy = self._condition(condition, local, deferred)
                    alive.update(*(
                        s.get(ALIVE, frozenset()) for s in (truthy, falsy) if s is not None))
                    if truthy is None:
                        return alive  # the filter is never true: the rest never runs
                    local = truthy  # only an item the filter keeps reaches what follows
            parts = (node.key, node.value) if isinstance(node, ast.DictComp) else (node.elt,)
            for part in parts:
                self._expression(part, local, deferred)
            return alive | set(local.get(ALIVE, frozenset()))
        finally:
            self._enclosing = enclosing

    def _condition(
        self, node: ast.expr, bindings: Bindings, deferred: list[Deferred],
    ) -> tuple[Bindings | None, Bindings | None]:
        """The states in which ``node`` is true and false; None for an outcome no path can have.

        ``bindings`` is the state before ``node`` is evaluated, and is consumed.
        """
        negated = False
        while isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            node, negated = node.operand, not negated  # iteratively, since a chain may be deep
        truthy, falsy = self._outcomes(node, bindings, deferred)
        return (falsy, truthy) if negated else (truthy, falsy)  # `not` swaps the two

    def _outcomes(
        self, node: ast.expr, bindings: Bindings, deferred: list[Deferred],
    ) -> tuple[Bindings | None, Bindings | None]:
        """`_condition` of an expression that is not itself a `not`."""
        truth = _static_truth(node)
        if truth is not None:  # a literal: nothing to visit, one outcome
            return (bindings, None) if truth else (None, bindings)
        if isinstance(node, ast.BoolOp):
            # `and` goes on while its operands are true and `or` while they are false; an
            # operand with the other outcome decides the result
            conjunction = isinstance(node.op, ast.And)
            decided: list[Bindings | None] = []
            going: Bindings | None = bindings
            for value in node.values:
                if going is None:
                    break  # the remaining operands never run
                truthy, falsy = self._condition(value, going, deferred)
                going, done = (truthy, falsy) if conjunction else (falsy, truthy)
                decided.append(done)
            settled = _either(decided)
            return (going, settled) if conjunction else (settled, going)
        if isinstance(node, ast.IfExp):
            # a ternary chain nests only through `orelse`, so it is flattened and walked
            # iteratively (bounded by MAX_IFEXP_CHAIN AST nodes) rather than recursed into one
            # level per `else`, which would exceed the recursion limit well under a chain a
            # comprehension result can realistically hold (hundreds of levels)
            branches: list[tuple[ast.expr, ast.expr]] = []
            tail: ast.expr = node
            while isinstance(tail, ast.IfExp) and len(branches) < MAX_IFEXP_CHAIN:
                branches.append((tail.test, tail.body))
                tail = tail.orelse
            outcomes: list[tuple[Bindings | None, Bindings | None]] = []
            state: Bindings | None = bindings
            for test, body in branches:
                if state is None:
                    break  # every earlier test decided true: this level never runs
                truthy, falsy = self._condition(test, state, deferred)
                if truthy is not None:
                    outcomes.append(self._condition(body, truthy, deferred))
                state = falsy
            if state is not None:
                outcomes.append(self._condition(tail, state, deferred))
            return _either(t for t, _ in outcomes), _either(f for _, f in outcomes)
        self._expression(node, bindings, deferred)
        return bindings, dict(bindings)

    def _expression(
        self, node: ast.AST | None, bindings: Bindings, deferred: list[Deferred],
    ) -> None:
        while isinstance(node, ast.UnaryOp):
            node = node.operand  # iteratively, since a chain of unary operators may be deep
        if node is None:
            return
        if isinstance(node, ast.Lambda):
            for default in (*node.args.defaults, *node.args.kw_defaults):
                self._expression(default, bindings, deferred)
            self._defined[_marker(node)] = node
            deferred.append(node)
            return
        if isinstance(node, ast.GeneratorExp):
            # only the first iterable runs now; the rest runs as the generator is advanced
            self._expression(node.generators[0].iter, bindings, deferred)
            deferred.append(node)
            self._called(node, node, bindings, deferred)
            return
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp)):
            already = set(bindings.get(ALIVE, frozenset()))
            made = self._comprehension(node, bindings, deferred) - already
            escaping = self._escapes.get(node)
            if escaping is None:
                escaping = self._escapes[node] = _escaping(node)
            for oid in made:
                if self._lazy[oid][2] in escaping:
                    self._made(oid, bindings)  # made inside, it outlives the comprehension
            return
        if isinstance(node, ast.NamedExpr):
            self._expression(node.value, bindings, deferred)
            self._unbind(node.target, bindings)
            return
        if isinstance(node, (ast.BoolOp, ast.IfExp)):
            outcome = _either(self._condition(node, dict(bindings), deferred))
            bindings.clear()
            bindings.update(outcome or {})
            return
        if isinstance(node, ast.Attribute) and RAW_WALK in self._resolved(node, bindings):
            self.uses.append(f"line {node.lineno}: references {RAW_WALK}")
        if (
            isinstance(node, ast.Call) and BUILTIN_GETATTR in self._resolved(node.func, bindings)
            and len(node.args) >= 2 and WALK_MODULE in self._resolved(node.args[0], bindings)
            and isinstance(node.args[1], ast.Constant) and node.args[1].value == "bounded_walk"
        ):
            self.uses.append(f"line {node.lineno}: getattr of {RAW_WALK}")
        if isinstance(node, ast.Call):
            self._expression(node.func, bindings, deferred)  # the callee is evaluated first
            callees = [
                self._defined[t] for t in self._resolved(node.func, bindings) if t in self._defined]
            for argument in (*node.args, *node.keywords):
                self._expression(argument, bindings, deferred)
            for function in callees:
                self._called(function, node, bindings, deferred)
            return
        for child in ast.iter_child_nodes(node):
            self._expression(child, bindings, deferred)

    def _loop(
        self, node: ast.For | ast.AsyncFor | ast.While, bindings: Bindings,
        deferred: list[Deferred],
    ) -> bool:
        if not isinstance(node, ast.While):
            self._expression(node.iter, bindings, deferred)
        breaks: list[Bindings] = []
        continues: list[Bindings] = []
        head = dict(bindings)  # every state an iteration may start from
        while True:
            if isinstance(node, ast.While):
                entered, left = self._condition(node.test, dict(head), deferred)
            else:
                entered, left = dict(head), dict(head)  # a next item, or the iterable is exhausted
            following = head
            if entered is not None:
                if not isinstance(node, ast.While):
                    self._unbind(node.target, entered)
                self._loops.append((breaks, continues))
                falls = self._block(node.body, entered, deferred)
                self._loops.pop()
                following = _join(head, *([entered] if falls else []), *continues)
            if following == head:
                break
            head = following
        ends = [(state, True) for state in breaks]
        if left is None:
            ends.append((head, False))  # the test is never false: only a break leaves
        else:
            ends.append((left, self._block(node.orelse, left, deferred)))
        return _continue_from(bindings, ends)
    def _try(
        self, node: ast.Try | ast.TryStar, bindings: Bindings, deferred: list[Deferred],
    ) -> bool:
        anywhere, raised = dict(bindings), dict(bindings)
        transfers: tuple[list[Bindings], list[Bindings]] = ([], [])
        returns: list[tuple[Bindings, frozenset[str]]] = []
        if node.finalbody:
            self._loops.append(transfers)  # a break or continue leaves through `finally` first
            self._exits.append(returns)  # and so does a `return`
        self._watchers += [anywhere, raised]
        body_falls = self._block(node.body, bindings, deferred)
        self._watchers.pop()
        ends: list[tuple[Bindings, bool]] = []
        for handler in node.handlers:
            # the `except*` handlers of one exception group can each run, in order
            state = (
                _join(raised, *(end for end, _ in ends)) if isinstance(node, ast.TryStar)
                else dict(raised)
            )
            self._expression(handler.type, state, deferred)
            if handler.name:
                self._set(handler.name, frozenset(), state)
            falls = self._block(handler.body, state, deferred)
            if handler.name:
                self._set(handler.name, frozenset(), state)  # Python deletes it as it ends
            ends.append((state, falls))
        ends.append((bindings, body_falls and self._block(node.orelse, bindings, deferred)))
        self._watchers.pop()
        falls = _continue_from(bindings, ends)
        if not node.finalbody:
            return falls
        self._loops.pop()
        self._exits.pop()
        self._block(node.finalbody, anywhere, deferred)  # leaving by an exception or return
        outer = self._loops[-1] if self._loops else ([], [])
        for captured, target in zip(transfers, outer, strict=True):
            if captured:
                state = _join(*captured)
                if self._block(node.finalbody, state, deferred):
                    target.append(state)  # otherwise the transfer in `finally` replaces it
        if returns:  # only where returns are followed
            state = _join(*(reached for reached, _ in returns))
            if self._block(node.finalbody, state, deferred):
                self._exits[-1].append((state, frozenset().union(*(o for _, o in returns))))
        return falls and self._block(node.finalbody, bindings, deferred)

    def _match(self, node: ast.Match, bindings: Bindings, deferred: list[Deferred]) -> bool:
        self._expression(node.subject, bindings, deferred)
        unmatched = dict(bindings)
        ends: list[tuple[Bindings, bool]] = []
        for case in node.cases:
            state = dict(unmatched)
            self._unbind(case.pattern, state)
            # the guard runs only once the pattern matched; the body sees it true
            success, failure = (
                (state, None) if case.guard is None
                else self._condition(case.guard, state, deferred))
            # a failed pattern leaves the state before it (a partial capture only removes names)
            failed = [] if _irrefutable(case.pattern) else [unmatched]
            if failure is not None:
                failed.append(failure)  # a false guard, after its named expressions ran
            unmatched = _join(*failed)
            if success is not None:
                ends.append((success, self._block(case.body, success, deferred)))
        if failed:
            ends.append((unmatched, True))
        return _continue_from(bindings, ends)

    def _statement(
        self, node: ast.stmt, bindings: Bindings, deferred: list[Deferred],
    ) -> bool:
        """Analyze ``node``; report whether a normal path leaves it."""
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    self._bind(alias.asname, [alias.name], bindings)
                else:
                    head = alias.name.partition(".")[0]
                    self._bind(head, [head], bindings)
        elif isinstance(node, ast.ImportFrom):
            level = "." * node.level
            base = importlib.util.resolve_name(level + (node.module or ""), self._package)
            for alias in node.names:
                if alias.name == "*":
                    if base == WALK_MODULE:
                        self.uses.append(f"line {node.lineno}: star import of {WALK_MODULE}")
                    continue
                target = f"{base}.{alias.name}"
                self._bind(alias.asname or alias.name, [target], bindings)
                if target == RAW_WALK:
                    self.uses.append(f"line {node.lineno}: imports {RAW_WALK}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            arguments = node.args
            for part in (*node.decorator_list, *arguments.defaults, *arguments.kw_defaults):
                self._expression(part, bindings, deferred)
            self._defined[_marker(node)] = node  # the name now refers to this function
            self._set(node.name, frozenset({_marker(node)}), bindings)
            deferred.append(node)
        elif isinstance(node, ast.ClassDef):
            for part in (*node.decorator_list, *node.bases, *(k.value for k in node.keywords)):
                self._expression(part, bindings, deferred)
            saved = self._watchers, self._enclosing
            # the body runs now, from the lexical names: a class body never closes over an outer
            # class's. Its own names stay in it, and what it calls reads the lexical names.
            self._enclosing = self._enclosing or (_lexical(bindings), [])
            self._watchers = []
            self._block(node.body, dict(self._enclosing[0]), deferred)  # methods resolve later
            made = self._enclosing[1]
            self._watchers, self._enclosing = saved
            if not self._enclosing:
                for oid in made:
                    self._start(oid, bindings)  # made in the body, it exists once the class does
            self._set(node.name, frozenset(), bindings)
        elif isinstance(node, ast.Assign):
            self._expression(node.value, bindings, deferred)
            for target in node.targets:
                self._expression(target, bindings, deferred)
            single = node.targets[0] if len(node.targets) == 1 else None
            for target in node.targets:
                self._unbind(target, bindings)
            if isinstance(single, ast.Name):
                self._propagate(single.id, node.value, bindings)
        elif isinstance(node, ast.AnnAssign):
            self._expression(node.value, bindings, deferred)
            self._expression(node.target, bindings, deferred)
            self._unbind(node.target, bindings)
            if isinstance(node.target, ast.Name) and node.value is not None:
                self._propagate(node.target.id, node.value, bindings)
        elif isinstance(node, ast.AugAssign):
            self._expression(node.value, bindings, deferred)
            self._expression(node.target, bindings, deferred)
            self._unbind(node.target, bindings)
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            return self._loop(node, bindings, deferred)
        elif isinstance(node, (ast.Break, ast.Continue)):
            if self._loops:
                breaks, continues = self._loops[-1]
                (breaks if isinstance(node, ast.Break) else continues).append(dict(bindings))
            return False
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                self._expression(item.context_expr, bindings, deferred)
                if item.optional_vars is not None:
                    self._unbind(item.optional_vars, bindings)
            return self._block(node.body, bindings, deferred)
        elif isinstance(node, ast.If):
            ends = []
            tested = self._condition(node.test, dict(bindings), deferred)
            for branch, state in zip((node.body, node.orelse), tested, strict=True):
                if state is not None:  # a literal test leaves the other branch unreachable
                    ends.append((state, self._block(branch, state, deferred)))
            return _continue_from(bindings, ends)
        elif isinstance(node, (ast.Try, ast.TryStar)):
            return self._try(node, bindings, deferred)
        elif isinstance(node, ast.Match):
            return self._match(node, bindings, deferred)
        elif isinstance(node, ast.Delete):
            for target in node.targets:
                self._unbind(target, bindings)
        else:
            self._expression(node, bindings, deferred)
        return True


class _ReturnFlow(_WalkReferences):
    """The flow of one callee body, for the objects its feasible `return` statements hand back:
    nothing is reported and no call in it is followed, and a name may also hold an object made
    in it (by the expression that made it and its function)."""

    def __init__(self, caller: _WalkReferences) -> None:
        super().__init__(caller._package)
        self._defined = dict(caller._defined)
        self._later = caller._later
        self.returns: list[tuple[Bindings, frozenset[str]]] = []
        self._exits = [self.returns]
        self.objects: dict[str, Deferred] = {}  # each object target by its function

    def _called(
        self, function: Deferred, site: ast.AST, bindings: Bindings, deferred: list[Deferred],
    ) -> None:
        """A call in the callee is not followed."""

    def _followed(self, target: str) -> bool:
        return target in self.objects or super()._followed(target)

    def _object(self, function: Deferred, site: ast.AST) -> frozenset[str]:
        target = f"<object {id(function)}@{id(site)}>"
        self.objects[target] = function
        return frozenset({target})

    def _callees(self, node: ast.expr, bindings: Bindings) -> list[Deferred]:
        return [
            self._defined[t] for t in super()._resolved(node, bindings) if t in self._defined]

    def _truth(self, node: ast.expr, bindings: Bindings) -> bool | None:
        """`_static_truth`, extended: a call whose every feasible callee only ever hands back a
        lazy object (a generator or coroutine, which has no `__bool__` or `__len__`) is always
        truthy, and so is a generator expression written in place."""
        truth = _static_truth(node)
        if truth is not None:
            return truth
        if isinstance(node, ast.GeneratorExp):
            return True
        if isinstance(node, ast.Call):
            callees = self._callees(node.func, bindings)
            if callees and all(self._runs_later(callee) for callee in callees):
                return True
        return None

    def _resolved(self, node: ast.expr | None, bindings: Bindings) -> frozenset[str]:
        if isinstance(node, ast.GeneratorExp):
            return self._object(node, node)
        if isinstance(node, ast.Call):
            callees = self._callees(node.func, bindings)
            return frozenset().union(*(
                self._object(callee, node) for callee in callees if self._runs_later(callee)))
        parts: list[ast.expr] = []
        if isinstance(node, ast.NamedExpr):
            parts = [node.value]
        elif isinstance(node, ast.IfExp):
            truth = _static_truth(node.test)
            parts = [branch for branch, taken in ((node.body, True), (node.orelse, False))
                     if truth in (None, taken)]
        elif isinstance(node, ast.BoolOp):
            conjunction = isinstance(node.op, ast.And)
            last = len(node.values) - 1
            for index, value in enumerate(node.values):
                truth = self._truth(value, bindings)
                # an operand that definitely keeps the chain going is never itself the result,
                # unless nothing follows it to become the result instead
                if truth != conjunction or index == last:
                    parts.append(value)
                if truth is (not conjunction):
                    break  # it decides the result: the rest never runs
        else:
            return super()._resolved(node, bindings)
        return frozenset().union(*(self._resolved(part, bindings) for part in parts))

    def _statement(self, node: ast.stmt, bindings: Bindings, deferred: list[Deferred]) -> bool:
        if not isinstance(node, ast.Return):
            return super()._statement(node, bindings, deferred)
        self._expression(node.value, bindings, deferred)
        objects = self._resolved(node.value, bindings) & self.objects.keys()
        self._exits[-1].append((_lexical(bindings), frozenset(objects)))
        return False  # nothing after it runs


def _direct_walk_uses(source: str, module: str) -> list[str]:
    """Every static way ``source`` (the module ``module``) reaches `bounded_walk` directly."""
    return _WalkReferences(module.rpartition(".")[0]).module(ast.parse(source))


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(REPO).with_suffix("").parts)
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def test_every_production_consumer_walks_through_the_scoped_seam() -> None:
    direct = {}
    for path in sorted((REPO / "algua").rglob("*.py")):
        module = _module_name(path)
        if module == WALK_MODULE:
            continue  # the defining module is the only place the raw walk may be used
        uses = _direct_walk_uses(path.read_text(), module)
        if uses:
            direct[module] = uses
    assert direct == {}


DIRECT_WALK_SPELLINGS = {
    "direct-import": (
        "from algua.primitives.bounded_walk import bounded_walk\nbounded_walk(root)\n"),
    "aliased-import": (
        "from algua.primitives.bounded_walk import bounded_walk as bw\nbw(root)\n"),
    "aliased-import-unused-call": (
        "from algua.primitives.bounded_walk import bounded_walk as bw\n"),
    "star-import": "from algua.primitives.bounded_walk import *\n",
    "module-alias-call": (
        "from algua.primitives import bounded_walk as walks\nwalks.bounded_walk(root)\n"),
    "module-alias-reference": (
        "from algua.primitives import bounded_walk as w\nfn = w.bounded_walk\nfn(root)\n"),
    "import-as": "import algua.primitives.bounded_walk as w\nw.bounded_walk(root)\n",
    "fully-qualified": (
        "import algua.primitives.bounded_walk\n"
        "algua.primitives.bounded_walk.bounded_walk(root)\n"),
    "package-alias": "import algua.primitives as p\np.bounded_walk.bounded_walk(root)\n",
    "from-package": "from algua import primitives\nprimitives.bounded_walk.bounded_walk(root)\n",
    "relative-import": "from ..primitives.bounded_walk import bounded_walk as bw\nbw(root)\n",
    "relative-module": "from ..primitives import bounded_walk as w\nw.bounded_walk(root)\n",
    "getattr": (
        "import algua.primitives.bounded_walk as w\ngetattr(w, 'bounded_walk')(root)\n"),
}


@pytest.mark.parametrize(
    "source", DIRECT_WALK_SPELLINGS.values(), ids=DIRECT_WALK_SPELLINGS.keys())
def test_the_guard_catches_every_direct_walk_spelling(source: str) -> None:
    assert _direct_walk_uses(source, "algua.registry.consumer")


SCOPED_OR_UNRELATED = {
    "scoped-import": (
        "from algua.primitives.bounded_walk import WalkCleanupError, scoped_walk\n"
        "with scoped_walk(root, max_files=1, max_directories=1, max_path_bytes=1) as tree:\n"
        "    pass\n"),
    "module-scoped-call": (
        "from algua.primitives import bounded_walk as w\nw.scoped_walk(root)\n"),
    "mention-in-text": '"""Never call bounded_walk(root) directly."""\n# bounded_walk(root)\n',
    "unrelated-local-name": "def bounded_walk(root):\n    return root\nbounded_walk(1)\n",
}


@pytest.mark.parametrize(
    "source", SCOPED_OR_UNRELATED.values(), ids=SCOPED_OR_UNRELATED.keys())
def test_the_guard_allows_scoped_use_and_unrelated_mentions(source: str) -> None:
    assert _direct_walk_uses(source, "algua.registry.consumer") == []


ASSIGNMENT_ALIASES = {
    "module-alias": (
        "from algua.primitives import bounded_walk as w\nm = w\nm.bounded_walk(root)\n"),
    "chained-module-alias": (
        "import algua.primitives.bounded_walk as w\na = w\nb = a\n"
        "fn = b.bounded_walk\nfn(root)\n"),
    "package-alias": (
        "import algua.primitives\np = algua.primitives\np.bounded_walk.bounded_walk(root)\n"),
    "module-attribute-alias": (
        "from algua import primitives\nwalks = primitives.bounded_walk\n"
        "walks.bounded_walk(root)\n"),
    "annotated-module-alias": (
        "import algua.primitives.bounded_walk as w\nm: object = w\nm.bounded_walk(root)\n"),
    "getattr-on-module-alias": (
        "import algua.primitives.bounded_walk as w\nm = w\ngetattr(m, 'bounded_walk')(root)\n"),
    "getattr-callable-alias": (
        "import algua.primitives.bounded_walk as w\ng = getattr\ng(w, 'bounded_walk')(root)\n"),
    "builtins-getattr": (
        "import builtins\nimport algua.primitives.bounded_walk as w\n"
        "builtins.getattr(w, 'bounded_walk')(root)\n"),
    "function-reads-module-alias-bound-later": (
        "def walk(root):\n    return m.bounded_walk(root)\n"
        "import algua.primitives.bounded_walk as w\nm = w\n"),
    "method-reads-module-alias": (
        "import algua.primitives.bounded_walk as w\nm = w\n"
        "class Store:\n    def walk(self, root):\n        return m.bounded_walk(root)\n"),
    "class-name-does-not-shadow-module-alias-in-method": (
        "import algua.primitives.bounded_walk as m\n"
        "class Store:\n    m = None\n    def walk(self, root):\n"
        "        return m.bounded_walk(root)\n"),
    "function-local-import": (
        "def walk(root):\n    import algua.primitives.bounded_walk as w\n"
        "    return w.bounded_walk(root)\n"),
    "lambda-reads-module-alias": (
        "import algua.primitives.bounded_walk as w\nm = w\n"
        "walk = lambda root: m.bounded_walk(root)\n"),
    "comprehension-reads-module-alias": (
        "import algua.primitives.bounded_walk as w\n[w.bounded_walk(root) for root in roots]\n"),
}


@pytest.mark.parametrize("source", ASSIGNMENT_ALIASES.values(), ids=ASSIGNMENT_ALIASES.keys())
def test_the_guard_follows_simple_assignment_aliases(source: str) -> None:
    assert _direct_walk_uses(source, "algua.registry.consumer")


SHADOWED_OR_REBOUND = {
    "module-rebound-to-local": (
        "import algua.primitives.bounded_walk as w\nw = object()\nw.bounded_walk(root)\n"),
    "alias-rebound-to-local": (
        "import algua.primitives.bounded_walk as w\nm = w\nm = make()\nm.bounded_walk(root)\n"),
    "parameter-shadows-module": (
        "import algua.primitives.bounded_walk as w\n"
        "def walk(w, root):\n    return w.bounded_walk(root)\n"),
    "local-shadows-module": (
        "import algua.primitives.bounded_walk as w\n"
        "def walk(root):\n    w = make()\n    return w.bounded_walk(root)\n"),
    "for-target-shadows-module": (
        "import algua.primitives.bounded_walk as w\nfor w in items:\n    w.bounded_walk(root)\n"),
    "with-target-shadows-module": (
        "import algua.primitives.bounded_walk as w\n"
        "with open(path) as w:\n    w.bounded_walk(root)\n"),
    "except-name-shadows-module": (
        "import algua.primitives.bounded_walk as w\n"
        "try:\n    pass\nexcept Exception as w:\n    w.bounded_walk(root)\n"),
    "def-shadows-module": (
        "import algua.primitives.bounded_walk as w\ndef w():\n    pass\nw.bounded_walk(root)\n"),
    "class-shadows-module": (
        "import algua.primitives.bounded_walk as w\nclass w:\n    pass\nw.bounded_walk(root)\n"),
    "deleted-module-alias": (
        "import algua.primitives.bounded_walk as w\ndel w\nw.bounded_walk(root)\n"),
    "getattr-shadowed-by-local-def": (
        "import algua.primitives.bounded_walk as w\n"
        "def getattr(obj, name):\n    return None\ngetattr(w, 'bounded_walk')\n"),
    "lambda-parameter-shadows-module": (
        "import algua.primitives.bounded_walk as w\nwalk = lambda w: w.bounded_walk(root)\n"),
    "comprehension-target-shadows-module": (
        "import algua.primitives.bounded_walk as w\n[w.bounded_walk(root) for w in items]\n"),
    "walrus-rebinds-module": (
        "import algua.primitives.bounded_walk as w\nif (w := make()):\n    w.bounded_walk(root)\n"),
    "match-capture-shadows-module": (
        "import algua.primitives.bounded_walk as w\n"
        "match item:\n    case w:\n        w.bounded_walk(root)\n"),
    "augmented-assignment-rebinds-alias": (
        "import algua.primitives.bounded_walk as w\nm = w\nm += 1\nm.bounded_walk(root)\n"),
    "unrelated-attribute": "helper.bounded_walk(root)\n",
}


@pytest.mark.parametrize("source", SHADOWED_OR_REBOUND.values(), ids=SHADOWED_OR_REBOUND.keys())
def test_the_guard_does_not_flag_shadowed_or_rebound_names(source: str) -> None:
    assert _direct_walk_uses(source, "algua.registry.consumer") == []


IMPORT_W = "import algua.primitives.bounded_walk as w\n"
NOTS = "not " * 1100  # deeper than the default recursion limit
IFEXP_CHAIN_DEPTH = 520  # at least 500, still comfortably under the default recursion limit


def _ternary_chain_comprehension(leaf: str, depth: int = IFEXP_CHAIN_DEPTH) -> str:
    """A comprehension whose result is one right-nested `a if c else b if c else ...` chain,
    ``depth`` levels deep: too deep to classify by recursing one Python frame per `else`."""
    chain = " if c else ".join([leaf] * (depth + 1))
    return f"[{chain} for r in roots]\n"
REACHED_ON_SOME_PATH = {
    "if-without-else-may-keep-module": "if c:\n    w = make()\nw.bounded_walk(root)\n",
    "else-branch-keeps-module": "if c:\n    w = make()\nelse:\n    w.bounded_walk(root)\n",
    "alias-from-one-branch": "if c:\n    m = w\nelse:\n    m = make()\nm.bounded_walk(root)\n",
    "while-runs-zero-times": "while c:\n    w = make()\nw.bounded_walk(root)\n",
    "while-next-iteration": "m = make()\nwhile c:\n    m.bounded_walk(root)\n    m = w\n",
    "for-runs-zero-times": "for item in items:\n    w = make()\nw.bounded_walk(root)\n",
    "for-next-iteration": "m = make()\nfor item in items:\n    m.bounded_walk(root)\n    m = w\n",
    "continue-reaches-next-iteration": (
        "m = make()\nfor item in items:\n    m.bounded_walk(root)\n    m = w\n"
        "    if c:\n        continue\n    m = make()\n"),
    "for-break-skips-else": (
        "for item in items:\n    break\nelse:\n    w = make()\nw.bounded_walk(root)\n"),
    "while-break-skips-else": "while c:\n    break\nelse:\n    w = make()\nw.bounded_walk(root)\n",
    "break-through-finally-carries-its-alias": (
        "m = make()\nfor item in items:\n    try:\n        break\n    finally:\n        m = w\n"
        "    m = make()\nm.bounded_walk(root)\n"),
    "continue-through-finally-carries-its-alias": (
        "m = make()\nfor item in items:\n    m.bounded_walk(root)\n    try:\n        continue\n"
        "    finally:\n        m = w\n    m = make()\n"),
    "function-called-inside-an-endless-loop": (
        "m = make()\ndef walk(root):\n    return m.bounded_walk(root)\n"
        "while True:\n    m = w\n    walk(root)\n"),
    "uncalled-function-reads-names-an-endless-loop-retains": (
        "m = make()\ndef walk(root):\n    return m.bounded_walk(root)\n"
        "while True:\n    m = w\n    serve()\n"),
    "continue-only-endless-loop-iterates": (
        "m = make()\nwhile True:\n    m.bounded_walk(root)\n    m = w\n    continue\n"),
    "nested-finally-forwards-a-break-alias": (
        "m = make()\nfor item in items:\n    try:\n        try:\n            break\n"
        "        finally:\n            m = w\n    finally:\n        pass\n    m = make()\n"
        "m.bounded_walk(root)\n"),
    "conditional-finally-keeps-a-break": (
        "m = make()\nwhile True:\n    try:\n        m = w\n        break\n    finally:\n"
        "        if c:\n            continue\nm.bounded_walk(root)\n"),
    "false-constant-loop-skips-its-body-rebind": (
        "while 0:\n    w = make()\nw.bounded_walk(root)\n"),
    "call-before-a-later-rebind": (
        "m = make()\ndef walk(root):\n    return m.bounded_walk(root)\n"
        "m = w\nwalk(root)\nm = make()\n"),
    "aliased-call-before-a-later-rebind": (
        "m = make()\ndef walk(root):\n    return m.bounded_walk(root)\n"
        "run = walk\nm = w\nrun(root)\nm = make()\n"),
    "nested-call-before-a-later-rebind": (
        "def outer(root):\n    import algua.primitives.bounded_walk as v\n"
        "    def inner():\n        return v.bounded_walk(root)\n    inner()\n    v = make()\n"),
    "short-circuited-if-test-keeps-module": (
        "if c and (w := make()):\n    pass\nw.bounded_walk(root)\n"),
    "break-in-a-branch-carries-its-alias-out": (
        "m = make()\nfor item in items:\n    if c:\n        m = w\n        break\n"
        "m.bounded_walk(root)\n"),
    "async-for-runs-zero-times": (
        "async def walk(items, root):\n    import algua.primitives.bounded_walk as v\n"
        "    async for item in items:\n        v = make()\n    v.bounded_walk(root)\n"),
    "alias-extended-in-a-loop": (
        "m = w\nwhile c:\n    n = m.child\n    m = n\nm.bounded_walk(root)\n"),
    "body-may-raise-before-rebind": (
        "try:\n    w = make()\nexcept Exception:\n    pass\nw.bounded_walk(root)\n"),
    "handler-sees-state-before-body": (
        "try:\n    w = make()\nexcept Exception:\n    w.bounded_walk(root)\n"),
    "handler-sees-mid-body-alias": (
        "m = make()\ntry:\n    m = w\n    m = make()\nexcept Exception:\n"
        "    m.bounded_walk(root)\n"),
    "else-follows-body-not-handler": (
        "try:\n    pass\nexcept Exception:\n    w = make()\nelse:\n    w.bounded_walk(root)\n"),
    "finally-after-raise-before-rebind": (
        "try:\n    w = make()\nfinally:\n    w.bounded_walk(root)\n"),
    "except-star-handlers-may-both-run": (
        "m = make()\ntry:\n    pass\nexcept* ValueError:\n    m = w\nexcept* TypeError:\n"
        "    m.bounded_walk(root)\n"),
    "no-match-case-matches": (
        "match value:\n    case 1:\n        w = make()\nw.bounded_walk(root)\n"),
    "later-match-case-keeps-module": (
        "match value:\n    case 1:\n        w = make()\n    case _:\n"
        "        w.bounded_walk(root)\n"),
    "guard-failure-after-a-refutable-pattern-keeps-module": (
        "match value:\n    case 1 if (w := make()):\n        pass\n    case _:\n"
        "        w.bounded_walk(root)\n"),
    "short-circuited-guard-failure-keeps-module": (
        "match value:\n    case _ if c and (w := make()):\n        pass\n    case _:\n"
        "        w.bounded_walk(root)\n"),
    "short-circuited-guard-success-keeps-module": (
        "match value:\n    case _ if c or (w := make()):\n        w.bounded_walk(root)\n"),
    "conditional-expression-guard-failure-keeps-module": (
        "match value:\n    case _ if ((w := make()) if c else False):\n        pass\n"
        "    case _:\n        w.bounded_walk(root)\n"),
    # 6hfPWRXC8MMwMrcG: only side-effect-free literals are decided
    "while-unpacked-list-may-run-zero-times": (
        "while [*items]:\n    w = make()\n    break\nw.bounded_walk(root)\n"),
    "while-unpacked-dict-may-run-zero-times": (
        "while {**options}:\n    w = make()\n    break\nw.bounded_walk(root)\n"),
    "while-set-call-is-not-a-literal": "while set():\n    w.bounded_walk(root)\n",
    # 6hfPfM2QXfxpvjvp: separate true and false states
    "if-else-sees-the-skipped-walrus": (
        "if c and (w := make()):\n    pass\nelse:\n    w.bounded_walk(root)\n"),
    "not-and-body-keeps-module": "if not (c and (w := make())):\n    w.bounded_walk(root)\n",
    "while-exit-keeps-the-skipped-walrus": (
        "while c and (w := make()):\n    pass\nw.bounded_walk(root)\n"),
    "statically-true-or-skips-the-walrus": "if 1 or (w := make()):\n    w.bounded_walk(root)\n",
    # 6hfPfM48RRh6Mc2G: lambdas called in their own scope
    "lambda-called-before-a-later-rebind": (
        "m = make()\nwalk = lambda root: m.bounded_walk(root)\nm = w\nwalk(root)\nm = make()\n"),
    "lambda-called-where-it-is-made": "m = w\n(lambda: m.bounded_walk(root))()\nm = make()\n",
    # 6hfPfM3X26Qxj9FG: class bodies run at import time
    "class-body-calls-a-module-function-while-the-alias-is-raw": (
        "m = make()\ndef walk(root):\n    return m.bounded_walk(root)\n"
        "m = w\nclass Store:\n    walk(root)\nm = make()\n"),
    "class-body-calls-its-own-function-with-module-names": (
        "m = w\nclass Store:\n    def helper():\n        return m.bounded_walk(root)\n"
        "    helper()\nm = make()\n"),
    # 6hfPfM2HWPqcrXCG: generator and coroutine bodies run after they are made
    "generator-advanced-after-a-later-rebind": (
        "m = make()\ndef gen():\n    yield m.bounded_walk(root)\n"
        "items = gen()\nm = w\nnext(items)\nm = make()\n"),
    "generator-iterated-while-the-alias-is-raw": (
        "m = make()\ndef gen():\n    while True:\n        yield m.bounded_walk(root)\n"
        "for item in gen():\n    m = w\nm = make()\n"),
    "coroutine-awaited-after-a-later-rebind": (
        "async def main(root):\n    import algua.primitives.bounded_walk as v\n    m = make()\n"
        "    async def fetch():\n        return m.bounded_walk(root)\n"
        "    pending = fetch()\n    m = v\n    await pending\n    m = make()\n"),
    "async-generator-made-before-a-later-rebind": (
        "m = make()\nasync def stream():\n    yield m.bounded_walk(root)\n"
        "items = stream()\nm = w\nm = make()\n"),
    "generator-expression-advanced-after-a-later-rebind": (
        "m = make()\nitems = (m.bounded_walk(r) for r in roots)\nm = w\nnext(items)\nm = make()\n"),
    "generator-made-in-a-class-body-advanced-later": (
        "m = make()\ndef gen():\n    yield m.bounded_walk(root)\n"
        "class Store:\n    items = gen()\nm = w\nnext(Store.items)\nm = make()\n"),
    # 6hfPWRXC8MMwMrcG: formatted or raising displays stay undecided
    "while-formatted-fstring-is-undecided": "while f'{flag}':\n    w.bounded_walk(root)\n",
    "while-set-of-an-unhashable-tuple-is-undecided": (
        "while {(1, [2])}:\n    w = make()\n    break\nw.bounded_walk(root)\n"),
    # 6hfPmMP337jqHpGG: a lazy object sees every later state of the paths it exists on
    "generator-made-in-a-branch-sees-a-later-alias": (
        "m = make()\ndef gen():\n    yield m.bounded_walk(root)\n"
        "if c:\n    items = gen()\nm = w\nm = make()\n"),
    "generator-from-an-earlier-iteration-sees-a-later-alias": (
        "m = make()\ndef gen():\n    yield m.bounded_walk(root)\n"
        "for item in items:\n    m = w\n    m = make()\n    made = gen()\n"),
    "generator-invoked-from-another-function-sees-a-feasible-sibling-alias": (
        "m = make()\ndef gen():\n    yield m.bounded_walk(root)\n"
        "def runner():\n    next(gen())\n"
        "if c:\n    m = w\nelse:\n    pass\nrunner()\nm = make()\n"),
    # 6hfQ9Rqc6X7wf35G: a followed helper's declared global writes reach the caller's state
    "helper-declaring-global-writes-into-the-caller-state": (
        "m = make()\ndef mutate():\n    global m\n    m = w\n"
        "mutate()\nm.bounded_walk(root)\n"),
    "self-recursive-global-writing-helper-still-reports-its-own-write": (
        "m = make()\ndef mutate():\n    global m\n    m = w\n    mutate()\n"
        "mutate()\nm.bounded_walk(root)\n"),
    "generator-made-in-a-comprehension-advanced-later": (
        "m = make()\ndef gen():\n    yield m.bounded_walk(root)\n"
        "items = [gen() for _ in range(2)]\nm = w\nnext(items[0])\nm = make()\n"),
    "generator-made-in-a-false-filter-escapes": (
        "m = make()\ndef gen():\n    yield m.bounded_walk(root)\nitems = []\n"
        "[x for x in xs if items.append(gen()) and 0]\nm = w\nnext(items[0])\nm = make()\n"),
    # 6hfPmMMPmhrfqvjG: a lazy object a followed factory returns sees the caller's later states
    "factory-returned-generator-advanced-after-a-later-rebind": (
        "m = make()\ndef gen():\n    yield m.bounded_walk(root)\ndef factory():\n    return gen()\n"
        "items = factory()\nm = w\nnext(items)\nm = make()\n"),
    "factory-local-generator-advanced-after-a-later-rebind": (
        "m = make()\ndef factory():\n    def gen():\n        yield m.bounded_walk(root)\n"
        "    return gen()\nitems = factory()\nm = w\nnext(items)\nm = make()\n"),
    "factory-returned-generator-expression-advanced-after-a-later-rebind": (
        "m = make()\ndef factory():\n    return (m.bounded_walk(r) for r in roots)\n"
        "items = factory()\nm = w\nnext(items)\nm = make()\n"),
    "factory-returned-coroutine-awaited-after-a-later-rebind": (
        "async def main(root):\n    import algua.primitives.bounded_walk as v\n    m = make()\n"
        "    def factory():\n        async def fetch():\n            return m.bounded_walk(root)\n"
        "        return fetch()\n    pending = factory()\n    m = v\n    await pending\n"
        "    m = make()\n"),
    "lambda-factory-returned-generator-advanced-after-a-later-rebind": (
        "m = make()\ndef gen():\n    yield m.bounded_walk(root)\nfactory = lambda: gen()\n"
        "items = factory()\nm = w\nnext(items)\nm = make()\n"),
    "lambda-factory-returned-generator-expression-sees-a-later-caller-alias": (
        "factory = lambda: (m.bounded_walk(r) for r in roots)\n"
        "items = factory()\nm = w\nnext(items)\nm = make()\n"),
    # 6hfPmMQ6h2xXcJ8G: a nested class body runs from the lexical names, not the outer class's
    "nested-class-calls-a-module-helper-an-outer-class-shadows": (
        "m = make()\ndef helper(root):\n    return m.bounded_walk(root)\nm = w\n"
        "class Outer:\n    helper = other\n    class Inner:\n        helper(root)\nm = make()\n"),
    "nested-class-reads-a-module-alias-an-outer-class-shadows": (
        "class Outer:\n    w = make()\n    class Inner:\n        w.bounded_walk(root)\n"),
    "nested-class-bases-evaluate-in-the-outer-class": (
        "class Outer:\n    base = w\n    class Inner(base.bounded_walk):\n        pass\n"),
    # 6hfPmMMpfmMvppxG: a filter's true state reaches the element
    "comprehension-element-keeps-an-or-filter-alias": (
        "[w.bounded_walk(r) for r in roots if c or (w := make())]\n"),
    "comprehension-element-after-a-true-literal-filter": (
        "[w.bounded_walk(r) for r in roots if 1]\n"),
    # 6hfPmMP337jqHpGG: a lazy function no object of which is made here reads the final names
    "uncalled-generator-function-reads-the-final-names": (
        "m = make()\ndef gen():\n    yield m.bounded_walk(root)\nm = w\n"),
    # 6hfPmMMPmhrfqvjG: a returned object's free names that are not the factory's stay late-bound
    "factory-global-generator-sees-a-later-caller-alias": (
        "m = make()\ndef factory():\n    global m\n    m = make()\n    def gen():\n"
        "        yield m.bounded_walk(root)\n    return gen()\n"
        "items = factory()\nm = w\nnext(items)\nm = make()\n"),
    "generator-declaring-global-bypasses-a-safe-factory-local": (
        "m = make()\ndef factory():\n    m = make()\n    def gen():\n"
        "        global m\n        yield m.bounded_walk(root)\n    return gen()\n"
        "items = factory()\nm = w\nnext(items)\nm = make()\n"),
    "factory-closure-captures-a-raw-local": (
        "def factory():\n    v = w\n    def gen():\n        yield v.bounded_walk(root)\n"
        "    return gen()\nitems = factory()\nnext(items)\n"),
    # 6hfPvjHrFwhGvmxp: every feasible returned object reaches the caller
    "factory-returning-a-generator-alias": (
        "m = make()\ndef gen():\n    yield m.bounded_walk(root)\n"
        "def factory():\n    made = gen()\n    return made\n"
        "items = factory()\nm = w\nnext(items)\nm = make()\n"),
    "factory-returning-a-call-of-a-callee-alias": (
        "m = make()\ndef gen():\n    yield m.bounded_walk(root)\n"
        "def factory():\n    build = gen\n    return build()\n"
        "items = factory()\nm = w\nnext(items)\nm = make()\n"),
    "factory-returning-a-conditional-expression": (
        "m = make()\ndef gen():\n    yield m.bounded_walk(root)\n"
        "def factory():\n    return gen() if c else None\n"
        "items = factory()\nm = w\nnext(items)\nm = make()\n"),
    "factory-returning-a-boolean-operation": (
        "m = make()\ndef gen():\n    yield m.bounded_walk(root)\n"
        "def factory():\n    return c and gen()\n"
        "items = factory()\nm = w\nnext(items)\nm = make()\n"),
    "factory-returning-the-last-operand-after-a-decided-boolean": (
        "m = make()\ndef gen():\n    yield m.bounded_walk(root)\n"
        "def factory():\n    return None or gen()\n"
        "items = factory()\nm = w\nnext(items)\nm = make()\n"),
    "factory-returning-either-same-name-definition": (
        "m = make()\ndef factory():\n    if c:\n        def gen():\n"
        "            yield m.bounded_walk(root)\n    else:\n        def gen():\n"
        "            yield None\n    return gen()\n"
        "items = factory()\nm = w\nnext(items)\nm = make()\n"),
    "factory-returning-a-walrus": (
        "m = make()\ndef gen():\n    yield m.bounded_walk(root)\n"
        "def factory():\n    return (made := gen())\n"
        "items = factory()\nm = w\nnext(items)\nm = make()\n"),
    "factory-comprehension-target-is-not-a-closure-name": (
        "m = make()\ndef factory():\n    names = [m for m in ms]\n    def gen():\n"
        "        yield m.bounded_walk(root)\n    return gen()\n"
        "items = factory()\nm = w\nnext(items)\nm = make()\n"),
    "factory-local-shadow-leaves-a-caller-generator-late-bound": (
        "m = make()\ndef gen():\n    yield m.bounded_walk(root)\n"
        "def factory():\n    m = make()\n    return gen()\n"
        "items = factory()\nm = w\nnext(items)\nm = make()\n"),
    "factory-returning-through-a-finally": (
        "m = make()\ndef gen():\n    yield m.bounded_walk(root)\n"
        "def factory():\n    try:\n        return gen()\n    finally:\n        pass\n"
        "items = factory()\nm = w\nnext(items)\nm = make()\n"),
    # 6hfPvjP4PqJ74J2G: an object kept in a comprehension's result or stored by it escapes
    "generator-bound-by-a-filter-walrus-escapes": (
        "m = make()\ndef gen():\n    yield m.bounded_walk(root)\n"
        "[x for x in xs if (made := gen())]\nm = w\nnext(made)\nm = make()\n"),
    "generator-stored-by-a-method-call-escapes": (
        "m = make()\ndef gen():\n    yield m.bounded_walk(root)\n"
        "[store.add(gen()) for _ in xs]\nm = w\nm = make()\n"),
    "generator-stashed-by-an-unknown-named-callee-escapes": (
        "m = make()\ndef gen():\n    yield m.bounded_walk(root)\n"
        "[stash(gen()) for _ in xs]\nm = w\nm = make()\n"),
    "generator-in-a-nested-comprehension-result-escapes": (
        "m = make()\ndef gen():\n    yield m.bounded_walk(root)\n"
        "items = [[gen() for _ in ys] for _ in xs]\nm = w\nm = make()\n"),
    "generator-in-a-dict-comprehension-value-escapes": (
        "m = make()\ndef gen():\n    yield m.bounded_walk(root)\n"
        "items = {k: gen() for k in keys}\nm = w\nm = make()\n"),
    "generator-in-a-tuple-result-escapes": (
        "m = make()\ndef gen():\n    yield m.bounded_walk(root)\n"
        "items = [(k, gen()) for k in keys]\nm = w\nm = make()\n"),
    "generator-in-a-list-display-result-escapes": (
        "m = make()\ndef gen():\n    yield m.bounded_walk(root)\n"
        "items = [[gen()] for _ in xs]\nm = w\nm = make()\n"),
    "generator-in-a-set-display-result-escapes": (
        "m = make()\ndef gen():\n    yield m.bounded_walk(root)\n"
        "items = [{gen()} for _ in xs]\nm = w\nm = make()\n"),
    "generator-in-a-dict-display-result-escapes": (
        "m = make()\ndef gen():\n    yield m.bounded_walk(root)\n"
        "items = [{k: gen()} for k in keys]\nm = w\nm = make()\n"),
    "generator-in-a-conditional-result-escapes": (
        "m = make()\ndef gen():\n    yield m.bounded_walk(root)\n"
        "items = [gen() if c else None for _ in xs]\nm = w\nm = make()\n"),
    "generator-in-a-boolean-result-escapes": (
        "m = make()\ndef gen():\n    yield m.bounded_walk(root)\n"
        "items = [c and gen() for _ in xs]\nm = w\nm = make()\n"),
    "generator-stored-after-a-temporary-from-the-same-function-escapes": (
        "m = make()\ndef gen():\n    yield m.bounded_walk(root)\n"
        "[(store.add(gen()), bool(gen())) for _ in xs]\nm = w\nm = make()\n"),
    # 6hfPvjR73hhFw8fG: a class comprehension's body never sees the class's names
    "class-comprehension-element-reads-a-module-alias-the-class-shadows": (
        "m = w\nclass Store:\n    m = make()\n    items = [m.bounded_walk(r) for r in roots]\n"
        "m = make()\n"),
    "class-comprehension-filter-reads-a-module-alias-the-class-shadows": (
        "m = w\nclass Store:\n    m = make()\n    items = {r for r in roots if m.bounded_walk(r)}\n"
        "m = make()\n"),
    "class-comprehension-later-iterable-reads-a-module-alias-the-class-shadows": (
        "m = w\nclass Store:\n    m = make()\n"
        "    items = {r: s for r in roots for s in m.bounded_walk(r)}\nm = make()\n"),
    "class-nested-comprehension-reads-a-module-alias-the-class-shadows": (
        "m = w\nclass Store:\n    m = make()\n"
        "    items = [[m.bounded_walk(r) for r in group] for group in groups]\nm = make()\n"),
    # 6hfPWRXC8MMwMrcG: a long `not` chain is classified without recursion
    "odd-not-chain-while-body-is-reachable": (
        f"while not {NOTS}0:\n    w.bounded_walk(root)\n    break\n"),
    "even-not-chain-of-a-name-keeps-the-if-body": f"if {NOTS}c:\n    w.bounded_walk(root)\n",
    "not-chain-around-a-walk-in-an-assignment": f"flag = {NOTS}w.bounded_walk(root)\n",
    "not-chain-around-a-walk-in-a-function": (
        f"def check(root):\n    return {NOTS}w.bounded_walk(root)\n"),
    # 6hfQ9Rq8R3RWMr3G: a deep ternary chain in a comprehension result is classified iteratively
    "deep-ternary-chain-in-a-comprehension-result-is-reached": (
        _ternary_chain_comprehension("w.bounded_walk(r)")),
}


@pytest.fixture
def default_recursion_limit() -> Iterator[None]:
    """Python's default limit, whatever the runner raised it to, so a deep chain really recurses."""
    raised = sys.getrecursionlimit()
    sys.setrecursionlimit(1000)
    yield
    sys.setrecursionlimit(raised)


@pytest.mark.usefixtures("default_recursion_limit")
@pytest.mark.parametrize(
    "source", REACHED_ON_SOME_PATH.values(), ids=REACHED_ON_SOME_PATH.keys())
def test_the_guard_flags_a_walk_reached_on_any_feasible_path(source: str) -> None:
    assert _direct_walk_uses(IMPORT_W + source, "algua.registry.consumer")


NOT_REACHED_ON_ANY_PATH = {
    "both-branches-rebind": (
        "if c:\n    w = make()\nelse:\n    w = other()\nw.bounded_walk(root)\n"),
    "else-does-not-see-if-alias": (
        "m = make()\nif c:\n    m = w\nelse:\n    m.bounded_walk(root)\n"),
    "rebound-before-use-in-while": "while c:\n    w = make()\n    w.bounded_walk(root)\n",
    "rebound-before-use-in-for": "for item in items:\n    w = make()\n    w.bounded_walk(root)\n",
    "loop-else-without-break-rebinds": (
        "for item in items:\n    pass\nelse:\n    w = make()\nw.bounded_walk(root)\n"),
    "use-after-break-is-unreachable": "for item in items:\n    break\n    w.bounded_walk(root)\n",
    "alias-after-break-is-unreachable": (
        "m = make()\nfor item in items:\n    break\n    m = w\nm.bounded_walk(root)\n"),
    "use-after-continue-is-unreachable": (
        "for item in items:\n    continue\n    w.bounded_walk(root)\n"),
    "alias-after-continue-is-unreachable": (
        "m = make()\nfor item in items:\n    continue\n    m = w\nm.bounded_walk(root)\n"),
    "break-through-finally-rebinds": (
        "m = make()\nfor item in items:\n    try:\n        m = w\n        break\n"
        "    finally:\n        m = make()\nm.bounded_walk(root)\n"),
    "continue-through-finally-rebinds": (
        "m = make()\nfor item in items:\n    try:\n        m = w\n        continue\n"
        "    finally:\n        m = make()\nm.bounded_walk(root)\n"),
    "while-true-rebinds-before-break": (
        "while True:\n    w = make()\n    break\nw.bounded_walk(root)\n"),
    "while-true-else-is-unreachable": (
        "m = make()\nwhile True:\n    break\nelse:\n    m = w\nm.bounded_walk(root)\n"),
    "use-after-an-endless-loop-is-unreachable": "while True:\n    serve()\nw.bounded_walk(root)\n",
    "use-after-branches-that-both-transfer-is-unreachable": (
        "for item in items:\n    if c:\n        break\n    else:\n        continue\n"
        "    w.bounded_walk(root)\n"),
    "branch-that-breaks-is-not-merged": (
        "m = make()\nfor item in items:\n    if c:\n        m = w\n        break\n"
        "    m.bounded_walk(root)\n"),
    "handler-that-breaks-is-not-merged": (
        "m = make()\nfor item in items:\n    try:\n        pass\n    except Exception:\n"
        "        m = w\n        break\n    m.bounded_walk(root)\n"),
    "case-that-breaks-is-not-merged": (
        "m = make()\nfor item in items:\n    match value:\n        case 1:\n            m = w\n"
        "            break\n    m.bounded_walk(root)\n"),
    "transfer-in-finally-overrides-a-break": (
        "m = make()\nwhile True:\n    try:\n        m = w\n        break\n    finally:\n"
        "        continue\nm.bounded_walk(root)\n"),
    "break-does-not-reach-the-next-iteration": (
        "m = make()\nfor item in items:\n    m.bounded_walk(root)\n    m = w\n    break\n"),
    "use-after-a-loop-whose-else-breaks-is-unreachable": (
        "for group in groups:\n    for item in group:\n        pass\n    else:\n        break\n"
        "    w.bounded_walk(root)\n"),
    "use-after-a-finally-that-breaks-is-unreachable": (
        "for item in items:\n    try:\n        pass\n    finally:\n        break\n"
        "    w.bounded_walk(root)\n"),
    "else-after-a-body-that-breaks-is-unreachable": (
        "for item in items:\n    try:\n        break\n    except Exception:\n        pass\n"
        "    else:\n        w.bounded_walk(root)\n"),
    "use-after-a-with-that-breaks-is-unreachable": (
        "for item in items:\n    with lock:\n        break\n    w.bounded_walk(root)\n"),
    "use-after-a-continue-only-endless-loop-is-unreachable": (
        "while True:\n    continue\nw.bounded_walk(root)\n"),
    "nested-finally-replaces-a-break-alias": (
        "m = make()\nfor item in items:\n    try:\n        try:\n            m = w\n"
        "            break\n        finally:\n            pass\n    finally:\n"
        "        m = make()\nm.bounded_walk(root)\n"),
    "while-one-rebinds-before-break": "while 1:\n    w = make()\n    break\nw.bounded_walk(root)\n",
    "while-nonempty-string-rebinds-before-break": (
        "while 'forever':\n    w = make()\n    break\nw.bounded_walk(root)\n"),
    "while-one-else-is-unreachable": (
        "m = make()\nwhile 1:\n    break\nelse:\n    m = w\nm.bounded_walk(root)\n"),
    "while-zero-body-is-unreachable": "while 0:\n    w.bounded_walk(root)\n",
    "while-none-body-alias-is-unreachable": (
        "m = make()\nwhile None:\n    m = w\nm.bounded_walk(root)\n"),
    "call-before-the-alias-is-bound": (
        "m = make()\ndef walk(root):\n    return m.bounded_walk(root)\n"
        "walk(root)\nm = w\nm = make()\n"),
    "function-rebound-before-the-call": (
        "m = make()\ndef walk(root):\n    return m.bounded_walk(root)\n"
        "walk = other\nm = w\nwalk(root)\nm = make()\n"),
    "call-from-a-class-body-sees-module-names": (
        "m = make()\ndef walk(root):\n    return m.bounded_walk(root)\n"
        "class Store:\n    m = w\n    walk(root)\n"),
    "call-from-another-function-sees-module-names": (
        "m = make()\ndef other(root):\n    m = w\n    walk(root)\n"
        "def walk(root):\n    return m.bounded_walk(root)\n"),
    # 6hfQ9Rqc6X7wf35G: a helper's plain (non-`global`/`nonlocal`) write never reaches the caller
    "helper-writing-a-plain-local-leaves-the-caller-state-safe": (
        "m = make()\ndef helper():\n    m = w\n"
        "helper()\nm.bounded_walk(root)\n"),
    "helper-writing-a-shadowing-parameter-leaves-the-caller-state-safe": (
        "m = make()\ndef helper(m):\n    m = w\n"
        "helper(make())\nm.bounded_walk(root)\n"),
    "body-and-handler-both-rebind": (
        "try:\n    w = make()\nexcept Exception:\n    w = other()\nw.bounded_walk(root)\n"),
    "handler-does-not-see-else-alias": (
        "m = make()\ntry:\n    pass\nexcept Exception:\n    m.bounded_walk(root)\n"
        "else:\n    m = w\n"),
    "else-does-not-see-handler-alias": (
        "m = make()\ntry:\n    pass\nexcept Exception:\n    m = w\nelse:\n"
        "    m.bounded_walk(root)\n"),
    "except-handlers-are-exclusive": (
        "m = make()\ntry:\n    pass\nexcept ValueError:\n    m = w\nexcept TypeError:\n"
        "    m.bounded_walk(root)\n"),
    "handler-name-is-deleted-as-the-handler-ends": (
        "try:\n    pass\nexcept Exception as m:\n    m = w\nm.bounded_walk(root)\n"),
    "finally-rebinds": "try:\n    pass\nfinally:\n    w = make()\nw.bounded_walk(root)\n",
    "normal-exit-continues-from-body-end": (
        "m = make()\ntry:\n    m = w\n    m = make()\nfinally:\n    pass\nm.bounded_walk(root)\n"),
    "match-cases-are-exclusive": (
        "m = make()\nmatch value:\n    case 1:\n        m = w\n    case _:\n"
        "        m.bounded_walk(root)\n"),
    "irrefutable-match-rebinds-on-every-path": (
        "match value:\n    case 1:\n        w = make()\n    case _:\n        w = other()\n"
        "w.bounded_walk(root)\n"),
    "guard-rebinds-before-a-later-case": (
        "match value:\n    case _ if (w := make()):\n        pass\n    case _:\n"
        "        w.bounded_walk(root)\n"),
    "guard-rebinds-on-every-path": (
        "match value:\n    case _ if (w := make()):\n        pass\nw.bounded_walk(root)\n"),
    "guard-success-sees-the-rebind": (
        "match value:\n    case _ if (w := make()):\n        w.bounded_walk(root)\n"),
    # 6hfPWRXC8MMwMrcG: side-effect-free literals and unary literals are decided
    "while-list-literal-rebinds-before-break": (
        "while [1]:\n    w = make()\n    break\nw.bounded_walk(root)\n"),
    "while-tuple-literal-else-is-unreachable": (
        "m = make()\nwhile (1,):\n    break\nelse:\n    m = w\nm.bounded_walk(root)\n"),
    "while-dict-literal-rebinds-before-break": (
        "while {'k': 1}:\n    w = make()\n    break\nw.bounded_walk(root)\n"),
    "while-set-literal-rebinds-before-break": (
        "while {1}:\n    w = make()\n    break\nw.bounded_walk(root)\n"),
    "while-not-zero-rebinds-before-break": (
        "while not 0:\n    w = make()\n    break\nw.bounded_walk(root)\n"),
    "while-negative-one-rebinds-before-break": (
        "while -1:\n    w = make()\n    break\nw.bounded_walk(root)\n"),
    "while-inverted-zero-rebinds-before-break": (
        "while ~0:\n    w = make()\n    break\nw.bounded_walk(root)\n"),
    "while-empty-list-body-is-unreachable": "while []:\n    w.bounded_walk(root)\n",
    "while-empty-dict-body-is-unreachable": "while {}:\n    w.bounded_walk(root)\n",
    "while-not-one-body-is-unreachable": "while not 1:\n    w.bounded_walk(root)\n",
    "while-negative-zero-body-is-unreachable": "while -0.0:\n    w.bounded_walk(root)\n",
    # 6hfPfM2QXfxpvjvp: a true outcome never inherits a false-only alias, nor the reverse
    "guard-success-does-not-inherit-a-failure-alias": (
        "match value:\n    case _ if c and (w := make()):\n        w.bounded_walk(root)\n"),
    "false-or-guard-rebinds-before-a-later-case": (
        "match value:\n    case _ if c or (w := make()):\n        pass\n    case _:\n"
        "        w.bounded_walk(root)\n"),
    "conditional-expression-guard-success-rebinds": (
        "match value:\n    case _ if ((w := make()) if c else False):\n"
        "        w.bounded_walk(root)\n"),
    "if-body-sees-the-and-walrus": "if c and (w := make()):\n    w.bounded_walk(root)\n",
    "not-or-body-sees-the-walrus": "if not (c or (w := make())):\n    w.bounded_walk(root)\n",
    "while-body-sees-the-and-walrus": "while c and (w := make()):\n    w.bounded_walk(root)\n",
    "unreachable-and-operand-is-not-visited": "flag = 0 and w.bounded_walk(root)\n",
    "unreachable-or-operand-is-not-visited": "flag = 1 or w.bounded_walk(root)\n",
    "unreachable-conditional-branch-is-not-visited": (
        "flag = w.bounded_walk(root) if 0 else None\n"),
    "statically-false-if-body-is-unreachable": "if 0:\n    w.bounded_walk(root)\n",
    # 6hfPfM48RRh6Mc2G: the bounded same-scope call model, for lambdas
    "lambda-parameter-shadows-at-the-call": (
        "m = make()\nwalk = lambda m: m.bounded_walk(root)\nm = w\nwalk(other)\nm = make()\n"),
    "lambda-called-before-the-alias-is-bound": (
        "m = make()\nwalk = lambda root: m.bounded_walk(root)\nwalk(root)\nm = w\nm = make()\n"),
    "lambda-called-from-another-function-sees-module-names": (
        "m = make()\ndef other(root):\n    m = w\n    walk(root)\n"
        "walk = lambda root: m.bounded_walk(root)\n"),
    # 6hfPfM3X26Qxj9FG: class locals stay isolated from what a called function reads
    "class-body-call-ignores-class-locals": (
        "m = make()\nclass Store:\n    m = w\n    def helper():\n"
        "        return m.bounded_walk(root)\n    helper()\n"),
    "class-local-name-shadows-the-module-function": (
        "m = make()\ndef walk(root):\n    return m.bounded_walk(root)\n"
        "m = w\nclass Store:\n    walk = other\n    walk(root)\nm = make()\n"),
    # 6hfPfM2HWPqcrXCG: only states from the object's creation on
    "generator-made-after-the-alias-was-raw": (
        "m = make()\ndef gen():\n    yield m.bounded_walk(root)\n"
        "m = w\nm = make()\nitems = gen()\nnext(items)\n"),
    "generator-expression-first-iterable-runs-only-at-creation": (
        "m = make()\nitems = (r for r in m.bounded_walk(root))\nm = w\nnext(items)\n"),
    "class-local-alias-does-not-reach-a-running-generator": (
        "m = make()\ndef gen():\n    yield m.bounded_walk(root)\n"
        "items = gen()\nclass Store:\n    m = w\nnext(items)\n"),
    "class-body-binding-does-not-reach-a-handler": (
        "try:\n    class Store:\n        m = w\nexcept Exception:\n    m.bounded_walk(root)\n"),
    # 6hfPWRXC8MMwMrcG: unary booleans, nested safe literals and constant f-strings are decided
    "while-plus-true-rebinds-before-break": (
        "while +True:\n    w = make()\n    break\nw.bounded_walk(root)\n"),
    "while-inverted-true-rebinds-before-break": (
        "while ~True:\n    w = make()\n    break\nw.bounded_walk(root)\n"),
    "while-plus-false-body-is-unreachable": "while +False:\n    w.bounded_walk(root)\n",
    "while-inverted-minus-one-body-is-unreachable": "while ~-1:\n    w.bounded_walk(root)\n",
    "while-list-of-not-zero-rebinds-before-break": (
        "while [not 0]:\n    w = make()\n    break\nw.bounded_walk(root)\n"),
    "while-set-of-tuples-rebinds-before-break": (
        "while {(1, 2)}:\n    w = make()\n    break\nw.bounded_walk(root)\n"),
    "while-dict-with-a-tuple-key-rebinds-before-break": (
        "while {(1,): 'x'}:\n    w = make()\n    break\nw.bounded_walk(root)\n"),
    "while-constant-fstring-rebinds-before-break": (
        "while f'x':\n    w = make()\n    break\nw.bounded_walk(root)\n"),
    "while-empty-fstring-body-is-unreachable": "while f'':\n    w.bounded_walk(root)\n",
    # 6hfPmMP337jqHpGG: a lazy object never sees a sibling path it does not exist on
    "generator-made-in-one-branch-ignores-a-sibling-alias": (
        "m = make()\ndef gen():\n    yield m.bounded_walk(root)\n"
        "if c:\n    items = gen()\nelse:\n    m = w\nm = make()\n"),
    "generator-made-in-a-handler-ignores-the-else-alias": (
        "m = make()\ndef gen():\n    yield m.bounded_walk(root)\n"
        "try:\n    risky()\nexcept Exception:\n    items = gen()\nelse:\n    m = w\nm = make()\n"),
    "generator-made-in-one-case-ignores-a-sibling-case-alias": (
        "m = make()\ndef gen():\n    yield m.bounded_walk(root)\n"
        "match value:\n    case 1:\n        items = gen()\n    case _:\n        m = w\n"
        "m = make()\n"),
    # 6hfPmMMPmhrfqvjG: only caller states from the factory call on
    "factory-returned-generator-made-after-the-alias-was-raw": (
        "m = make()\ndef gen():\n    yield m.bounded_walk(root)\ndef factory():\n    return gen()\n"
        "m = w\nm = make()\nitems = factory()\nnext(items)\n"),
    "factory-returned-generator-expression-made-after-the-alias-was-raw": (
        "m = make()\ndef factory():\n    return (m.bounded_walk(r) for r in roots)\n"
        "m = w\nm = make()\nitems = factory()\nnext(items)\n"),
    "factory-returned-coroutine-made-after-the-alias-was-raw": (
        "async def main(root):\n    import algua.primitives.bounded_walk as v\n    m = make()\n"
        "    def factory():\n        async def fetch():\n            return m.bounded_walk(root)\n"
        "        return fetch()\n    m = v\n    m = make()\n    pending = factory()\n"
        "    await pending\n"),
    "factory-returning-a-list-is-eager": (
        "m = make()\ndef factory():\n    return [m.bounded_walk(r) for r in roots]\n"
        "items = factory()\nm = w\nm = make()\n"),
    "factory-returning-a-plain-call-is-eager": (
        "m = make()\ndef helper():\n    return m.bounded_walk(root)\n"
        "def factory():\n    return helper()\nitems = factory()\nm = w\nm = make()\n"),
    "function-local-alias-does-not-reach-a-module-generator": (
        "m = make()\ndef other():\n    m = w\ndef gen():\n    yield m.bounded_walk(root)\n"
        "items = gen()\nm = make()\n"),
    "factory-called-in-one-branch-ignores-a-sibling-alias": (
        "m = make()\ndef factory():\n    return (m.bounded_walk(r) for r in roots)\n"
        "if c:\n    items = factory()\nelse:\n    m = w\nm = make()\n"),
    # 6hfPmMQ6h2xXcJ8G: class locals stay isolated at every level
    "nested-class-ignores-an-outer-class-alias": (
        "m = make()\nclass Outer:\n    m = w\n    class Inner:\n        m.bounded_walk(root)\n"),
    "nested-class-local-shadows-the-module-alias": (
        "class Outer:\n    class Inner:\n        w = make()\n        w.bounded_walk(root)\n"),
    "nested-class-decorator-evaluates-in-the-outer-class": (
        "class Outer:\n    w = make()\n    @w.bounded_walk\n    class Inner:\n        pass\n"),
    # 6hfPmMMpfmMvppxG: only a filter's true state reaches later filters and the element
    "list-comprehension-element-sees-the-true-filter": (
        "[w.bounded_walk(r) for r in roots if c and (w := make())]\n"),
    "set-comprehension-element-sees-the-true-filter": (
        "{w.bounded_walk(r) for r in roots if c and (w := make())}\n"),
    "dict-comprehension-element-sees-the-true-filter": (
        "{r: w.bounded_walk(r) for r in roots if c and (w := make())}\n"),
    "generator-expression-element-sees-the-true-filter": (
        "items = (w.bounded_walk(r) for r in roots if c and (w := make()))\n"),
    "later-filter-sees-the-true-filter": (
        "[r for r in roots if c and (w := make()) if w.bounded_walk(r)]\n"),
    "filter-after-a-false-literal-filter-is-unreachable": (
        "[r for r in roots if 0 if w.bounded_walk(r)]\n"),
    "element-after-a-false-literal-filter-is-unreachable": (
        "[w.bounded_walk(r) for r in roots if 0]\n"),
    "generator-after-a-false-literal-filter-is-unreachable": (
        "[s for r in roots if 0 for s in w.bounded_walk(r)]\n"),
    # 6hfPmMP337jqHpGG: a lazy body starts only from the states of the paths its object is on
    "generator-made-in-one-branch-ignores-a-final-sibling-alias": (
        "m = make()\ndef gen():\n    yield m.bounded_walk(root)\n"
        "if c:\n    items = gen()\nelse:\n    m = w\nnext(items)\n"),
    "generator-made-in-a-handler-ignores-a-final-else-alias": (
        "m = make()\ndef gen():\n    yield m.bounded_walk(root)\n"
        "try:\n    risky()\nexcept Exception:\n    items = gen()\nelse:\n    m = w\nnext(items)\n"),
    "generator-made-in-one-case-ignores-a-final-sibling-alias": (
        "m = make()\ndef gen():\n    yield m.bounded_walk(root)\n"
        "match value:\n    case 1:\n        items = gen()\n    case _:\n        m = w\n"
        "next(items)\n"),
    "generator-expression-made-in-one-branch-ignores-a-final-sibling-alias": (
        "m = make()\nif c:\n    items = (m.bounded_walk(r) for r in roots)\nelse:\n    m = w\n"
        "next(items)\n"),
    # 6hfPmMMPmhrfqvjG: a returned object reads the factory's own names from its closure
    "factory-local-shadow-hides-a-later-caller-alias": (
        "m = make()\ndef factory():\n    m = make()\n    def gen():\n"
        "        yield m.bounded_walk(root)\n    return gen()\n"
        "items = factory()\nm = w\nnext(items)\nm = make()\n"),
    "generator-declaring-nonlocal-resolves-to-the-factory-local": (
        "m = make()\ndef factory():\n    m = make()\n    def gen():\n"
        "        nonlocal m\n        yield m.bounded_walk(root)\n    return gen()\n"
        "items = factory()\nm = w\nnext(items)\nm = make()\n"),
    "factory-parameter-hides-a-later-caller-alias": (
        "m = make()\ndef factory(m):\n    def gen():\n        yield m.bounded_walk(root)\n"
        "    return gen()\nitems = factory(make())\nm = w\nnext(items)\nm = make()\n"),
    "factory-local-shadow-hides-a-caller-alias-from-a-generator-expression": (
        "m = make()\ndef factory():\n    m = make()\n"
        "    return (m.bounded_walk(r) for r in roots)\n"
        "items = factory()\nm = w\nnext(items)\nm = make()\n"),
    "lambda-walrus-hides-a-caller-alias-from-a-generator-expression": (
        "factory = lambda: (m := make()) and (m.bounded_walk(r) for r in roots)\n"
        "items = factory()\nm = w\nnext(items)\nm = make()\n"),
    "factory-local-import-hides-a-later-caller-alias": (
        "m = make()\ndef factory():\n    from os import path as m\n    def gen():\n"
        "        yield m.bounded_walk(root)\n    return gen()\n"
        "items = factory()\nm = w\nnext(items)\nm = make()\n"),
    "factory-local-match-capture-hides-a-later-caller-alias": (
        "m = make()\ndef factory():\n    match value:\n        case m:\n            pass\n"
        "    def gen():\n        yield m.bounded_walk(root)\n    return gen()\n"
        "items = factory()\nm = w\nnext(items)\nm = make()\n"),
    "factory-local-definition-hides-a-later-caller-alias": (
        "m = make()\ndef factory():\n    def m():\n        pass\n    def gen():\n"
        "        yield m.bounded_walk(root)\n    return gen()\n"
        "items = factory()\nm = w\nnext(items)\nm = make()\n"),
    "factory-local-handler-name-hides-a-later-caller-alias": (
        "m = make()\ndef factory():\n    try:\n        pass\n    except Exception as m:\n"
        "        pass\n    def gen():\n        yield m.bounded_walk(root)\n    return gen()\n"
        "items = factory()\nm = w\nnext(items)\nm = make()\n"),
    "generator-with-a-starred-display-in-a-comprehension-stays-inside": (
        "m = make()\ndef gen():\n    yield m.bounded_walk(root)\n"
        "items = [[*gen()] for _ in xs]\nm = w\nm = make()\n"),
    "factory-local-shadow-hides-a-caller-alias-raw-at-the-call": (
        "m = make()\ndef factory():\n    m = make()\n    def gen():\n"
        "        yield m.bounded_walk(root)\n    return gen()\n"
        "m = w\nitems = factory()\nm = make()\nnext(items)\n"),
    # 6hfPvjHrFwhGvmxp: a return no path reaches, or a finally overrides, hands back nothing
    "factory-return-after-a-return-is-unreachable": (
        "m = make()\ndef gen():\n    yield m.bounded_walk(root)\n"
        "def factory():\n    return None\n    return gen()\n"
        "items = factory()\nm = w\nm = make()\n"),
    "factory-return-under-a-false-literal-is-unreachable": (
        "m = make()\ndef gen():\n    yield m.bounded_walk(root)\n"
        "def factory():\n    if 0:\n        return gen()\n    return None\n"
        "items = factory()\nm = w\nm = make()\n"),
    "factory-return-replaced-by-a-finally-return": (
        "m = make()\ndef gen():\n    yield m.bounded_walk(root)\n"
        "def factory():\n    try:\n        return gen()\n    finally:\n        return None\n"
        "items = factory()\nm = w\nm = make()\n"),
    "factory-conditional-branch-under-a-false-literal-hands-back-nothing": (
        "m = make()\ndef gen():\n    yield m.bounded_walk(root)\n"
        "def factory():\n    return gen() if 0 else None\n"
        "items = factory()\nm = w\nm = make()\n"),
    "factory-and-operand-after-a-false-literal-hands-back-nothing": (
        "m = make()\ndef gen():\n    yield m.bounded_walk(root)\n"
        "def factory():\n    return 0 and gen()\n"
        "items = factory()\nm = w\nm = make()\n"),
    "factory-and-discards-an-always-truthy-generator": (
        "m = make()\ndef gen():\n    yield m.bounded_walk(root)\n"
        "def factory():\n    return gen() and None\n"
        "items = factory()\nm = w\nnext(items)\nm = make()\n"),
    "factory-returning-a-value-its-generator-yields": (
        "m = make()\ndef gen():\n    yield m.bounded_walk(root)\n"
        "def factory():\n    made = gen()\n    return next(made)\n"
        "items = factory()\nm = w\nm = make()\n"),
    # 6hfPvjP4PqJ74J2G: an object a comprehension only uses is not alive after it
    "generator-consumed-in-a-comprehension-element-stays-inside": (
        "m = make()\ndef gen():\n    yield m.bounded_walk(root)\n"
        "flags = [bool(gen()) for _ in xs]\nm = w\nm = make()\n"),
    "generator-tested-by-a-comprehension-filter-stays-inside": (
        "m = make()\ndef gen():\n    yield m.bounded_walk(root)\n"
        "kept = [x for x in xs if gen()]\nm = w\nm = make()\n"),
    "generator-consumed-in-a-set-comprehension-stays-inside": (
        "m = make()\ndef gen():\n    yield m.bounded_walk(root)\n"
        "firsts = {next(gen(), None) for _ in xs}\nm = w\nm = make()\n"),
    "generator-consumed-in-a-dict-comprehension-stays-inside": (
        "m = make()\ndef gen():\n    yield m.bounded_walk(root)\n"
        "flags = {x: bool(gen()) for x in xs}\nm = w\nm = make()\n"),
    "generator-only-bool-tested-in-a-comprehension-stays-inside": (
        "m = make()\ndef gen():\n    yield m.bounded_walk(root)\n"
        "[bool(gen()) for _ in xs]\nm = w\nm = make()\n"),
    "generator-expression-consumed-in-a-comprehension-stays-inside": (
        "m = make()\nflags = [any(m.bounded_walk(r) for r in roots) for _ in xs]\n"
        "m = w\nm = make()\n"),
    # 6hfPvjR73hhFw8fG: a class comprehension's first iterable runs in the class
    "class-comprehension-first-iterable-reads-the-class-shadow": (
        "m = w\nclass Store:\n    m = make()\n    items = [r for r in m.bounded_walk(root)]\n"
        "m = make()\n"),
    "generator-consumed-in-a-class-comprehension-stays-inside": (
        "m = make()\ndef gen():\n    yield m.bounded_walk(root)\n"
        "class Store:\n    flags = [bool(gen()) for _ in xs]\nm = w\nm = make()\n"),
    "class-comprehension-body-never-sees-a-class-alias": (
        "class Store:\n    v = w\n    items = [v.bounded_walk(r) for r in roots]\n"),
    # 6hfPWRXC8MMwMrcG: a long `not` chain is classified without recursion
    "even-not-chain-while-body-is-unreachable": f"while {NOTS}0:\n    w.bounded_walk(root)\n",
    "while-list-of-a-not-chain-rebinds-before-break": (
        f"while [{NOTS}0]:\n    w = make()\n    break\nw.bounded_walk(root)\n"),
    "even-not-chain-filter-ends-the-comprehension": (
        f"[r for r in roots if {NOTS}0 if w.bounded_walk(r)]\n"),
    # 6hfQ9Rq8R3RWMr3G: a deep ternary chain in a comprehension result is classified iteratively
    "deep-ternary-chain-in-a-comprehension-result-stays-clean": (
        _ternary_chain_comprehension("safe.bounded_walk(r)")),
}


@pytest.mark.usefixtures("default_recursion_limit")
@pytest.mark.parametrize(
    "source", NOT_REACHED_ON_ANY_PATH.values(), ids=NOT_REACHED_ON_ANY_PATH.keys())
def test_the_guard_does_not_flag_a_walk_no_feasible_path_reaches(source: str) -> None:
    assert _direct_walk_uses(IMPORT_W + source, "algua.registry.consumer") == []


def _live_generators(count: int) -> str:
    """A module making ``count`` distinct lazy objects that are all alive when the alias is raw."""
    made = "".join(
        f"items{index} = (m.bounded_walk(r) for r in roots)\n" for index in range(count))
    return f"{IMPORT_W}m = make()\n{made}m = w\nm = make()\n"


def test_lazy_observation_stays_near_linear_in_the_live_objects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 6hfQ949RRWWvrqvp: `len(bindings)` stays near-constant (a handful of names) no matter how
    # many objects are alive, so it cannot see a quadratic alive-set regression: instrument the
    # ACTUAL live-object work instead -- every `_Alive` node built (`_alive_add`, one per object
    # made, and `_alive_union`, one per branch join) and every oid an actual iteration
    # (`_Alive.__iter__`, e.g. a rebind's late-binding update) walks. A `frozenset | {oid}`
    # rebuild on every object made -- the regression this replaces -- would show as O(n^2)
    # node-creation work (each of n creations copying the n-1 already alive); the real,
    # `_Alive`-based path is O(1) per creation, and the whole run stays linear in the number of
    # objects made.
    created = 0
    real_add, real_union = _alive_add, _alive_union

    def counting_add(alive: frozenset[str] | None, oid: str) -> _Alive:
        nonlocal created
        created += 1
        return real_add(alive, oid)

    def counting_union(a: frozenset[str] | None, b: frozenset[str] | None) -> frozenset[str] | None:
        nonlocal created
        created += 1
        return real_union(a, b)

    touched = 0
    real_iter = _Alive.__iter__

    def counting_iter(self: _Alive) -> Iterator[str]:
        nonlocal touched
        for oid in real_iter(self):
            touched += 1
            yield oid

    module = sys.modules[__name__]
    monkeypatch.setattr(module, "_alive_add", counting_add)
    monkeypatch.setattr(module, "_alive_union", counting_union)
    monkeypatch.setattr(_Alive, "__iter__", counting_iter)
    totals = {}
    for count in (100, 200, 400, 800):
        created = touched = 0
        assert _direct_walk_uses(_live_generators(count), "algua.registry.consumer")
        totals[count] = created + touched
    assert totals[200] <= 2.1 * totals[100], totals
    assert totals[400] <= 2.1 * totals[200], totals
    assert totals[800] <= 2.1 * totals[400], totals


def _accumulating_factories(count: int) -> str:
    """``count`` independent generator+factory pairs, each called once. Every pair adds two more
    defined names the module tracks (its generator and its factory), so a later factory's
    return-summary cache key would grow with how many unrelated names came before it -- not with
    what that factory itself reads -- unless the key is projected onto the latter."""
    pairs = "".join(
        f"def gen{i}():\n    yield m.bounded_walk(root)\n"
        f"def factory{i}():\n    return gen{i}()\n"
        f"items{i} = factory{i}()\n"
        for i in range(count)
    )
    calls = "".join(f"next(items{i})\n" for i in range(count))
    return f"{IMPORT_W}m = make()\n{pairs}m = w\n{calls}m = make()\n"


def test_return_summary_cache_keys_stay_near_linear_in_the_factories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 6hfQ9RmmQ7HwXJ3p: a return-summary cache key built from every name tracked in scope so far
    # (rather than projected onto what the summarized function can actually read) grows with every
    # unrelated function definition accumulated before it; over `count` independent factories that
    # shows as O(n^2) total key-construction work (the i-th factory's key carrying ~2i unrelated
    # names). Instrument the deterministic SIZE of the entry `_summarize` is actually called with
    # (the cache key's real size) rather than wall clock, which a slow CI runner could mask either
    # way.
    touched = 0
    real_summarize = _WalkReferences._summarize

    def counting_summarize(
        self: _WalkReferences, function: Function, entry: Bindings,
    ) -> list[Returned]:
        nonlocal touched
        touched += len(entry)
        return real_summarize(self, function, entry)

    monkeypatch.setattr(_WalkReferences, "_summarize", counting_summarize)
    totals = {}
    for count in (50, 100, 200, 400):
        touched = 0
        assert _direct_walk_uses(_accumulating_factories(count), "algua.registry.consumer")
        totals[count] = touched
    assert totals[100] <= 2.1 * totals[50], totals
    assert totals[200] <= 2.1 * totals[100], totals
    assert totals[400] <= 2.1 * totals[200], totals
