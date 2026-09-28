"""A consumer's own error stays primary when closing its walk also fails."""
from __future__ import annotations

import ast
import errno
import gc
import importlib.util
import sys
import tracemalloc
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import NamedTuple

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
# the key naming which of NON_RETAINING_CALLS the source binds, in any way and whatever the value,
# on some path to a point. A name's own entry cannot say so: `Bindings` keeps only the targets on
# the way to the raw walk, so a rebinding to anything else simply drops the entry
SHADOWED = "<shadowed>"


class _Alive:
    """A persistent, immutable set of live-object ids. Adding one, or joining what two paths each
    have alive, is O(1) and shares every node already built rather than copying it, so a
    straight-line run of N creations costs O(N) in total, not the O(N^2) that unioning a growing
    `frozenset` on every one of them would. Iterating it, or comparing it for equality (needed for
    a loop's fixpoint check, since two differently-built sets of the same ids must still compare
    equal), flattens it: walked with an explicit stack rather than recursed into (a long
    straight-line chain would otherwise recurse one Python frame per id and exceed the recursion
    limit), and deduplicated by node identity within one walk (so a DAG reachable through more
    than one join is still visited once). Deliberately UNCACHED: caching the flattened set on
    every node a walk visits, not only the one asked for, once made a single flatten of an
    n-object chain retain O(n) separate, ever-larger frozenset copies (sizes 1..n) forever, not
    the O(n) a persistent set actually needs -- about 1.38 GB at 8,000 objects. Materializing
    costs memory only for the flatten actually requested, freed once that call returns.

    Nothing that runs once per rebind, comprehension or statement materializes it any more: a
    rebind is recorded on the node it happens in and handed down to the objects that node
    includes in one pass per scope (`_WalkReferences._observe` and `_flush`), and a comprehension
    learns which objects it made as they start. What still flattens is a loop's fixpoint check,
    unless both sides are the very same node (a loop that makes no lazy object, since joining a
    set with itself returns it): a loop that itself makes a lazy object compares its alive set
    with the one it started from by materializing both, once per iteration.

    Deliberately not a `frozenset` subclass either: `set(x)`, `frozenset(x)` and `set.update(x)`
    all special-case an actual `frozenset` (or subclass) instance and copy its underlying hash
    table directly in C, bypassing any Python-level `__iter__` override -- confirmed empirically
    before this design was chosen -- so a lazily-flattened node would read as empty through
    those. Being a genuinely different type, checked with `isinstance` at the few places
    `Bindings` reads or writes ALIVE, is what keeps this sound."""

    __slots__ = ("_oid", "_left", "_right")

    def __init__(
        self, oid: str | None = None, left: frozenset[str] | _Alive | None = None,
        right: frozenset[str] | _Alive | None = None,
    ) -> None:
        self._oid = oid
        self._left = left
        self._right = right

    def added(self, oid: str) -> _Alive:
        return _Alive(oid=oid, left=self)

    def flattened(self) -> frozenset[str]:
        """``self``'s set, recomputed on every call (never cached -- see the class docstring),
        with an explicit stack rather than recursion, and node-identity deduplication within
        this one walk."""
        result: set[str] = set()
        stack: list[frozenset[str] | _Alive | None] = [self]
        visited: set[int] = set()
        while stack:
            node = stack.pop()
            if node is None:
                continue
            if isinstance(node, _Alive):
                if id(node) in visited:
                    continue
                visited.add(id(node))
                if node._oid is not None:
                    result.add(node._oid)
                stack.append(node._left)
                stack.append(node._right)
            else:
                result |= node
        return frozenset(result)

    def __iter__(self) -> Iterator[str]:
        return iter(self.flattened())

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _Alive):
            return self is other or self.flattened() == other.flattened()
        if isinstance(other, frozenset):
            return self.flattened() == other
        return NotImplemented


def _alive_order(roots: Iterable[_Alive]) -> list[_Alive]:
    """Every node reachable from ``roots``, each one AFTER every reachable node that includes it
    (an ancestor), so what is recorded on a node can be handed down to those it includes in one
    pass. Walked with an explicit stack, not recursed into (a chain is as deep as it is long),
    and each node once by identity however many joins reach it."""
    order: list[_Alive] = []
    started: set[int] = set()
    stack: list[tuple[_Alive, bool]] = [(root, False) for root in roots]
    while stack:
        node, finished = stack.pop()
        if finished:
            order.append(node)  # every node it includes is already in `order`
            continue
        if id(node) in started:
            continue
        started.add(id(node))
        stack.append((node, True))
        stack.extend(
            (child, False) for child in (node._left, node._right)
            if isinstance(child, _Alive) and id(child) not in started)
    order.reverse()
    return order


def _union_views(a: Views, b: Views) -> Views:
    """What ``a`` and ``b`` record together, as a new mapping: neither is ever changed, so one can
    be handed to several nodes at once."""
    joined = dict(a)
    for name, targets in b.items():
        joined[name] = joined.get(name, frozenset()) | targets
    return joined


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
    if a is b:
        # the same set on both paths (nothing made on either): no new node, so a loop's fixpoint
        # check finds the very same object instead of flattening two equal sets
        return a
    return _Alive(left=a, right=b)


# each name mapped to every walk-relevant target it may hold on some path to this point, except
# ALIVE, whose value is a `frozenset[str] | _Alive | None`, not a `frozenset[str]` alone (see
# `_Alive` for why it is a genuinely different type rather than a `frozenset` subclass)
Bindings = dict[str, frozenset[str] | _Alive]
# names mapped to targets alone, never the alive set: what rebinds recorded or defaults held
Views = dict[str, frozenset[str]]
# a lazy object a call hands back: its function, and the names it reads from the closure of the
# function that made it (whose own flow resolves them)
Returned = tuple[Deferred, frozenset[str]]


class _Write(NamedTuple):
    """What a call may leave a variable it does not own holding."""

    targets: frozenset[str]
    exact: bool  # the variable's whole value after the call; otherwise only what it may have gained


# the variables a call of a function may write, each by the scope that owns it (None: the module)
# and the name it goes by
Effect = dict[tuple[Function | None, str], _Write]


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
    """The names, and which builtins they shadow, without the lazy objects alive."""
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


def _bound_by(body: list[ast.AST], names: set[str]) -> set[str]:
    """``names`` and every name the scope whose own statements are ``body`` binds, whatever it is
    bound to and however (an assignment, import, loop, `with`, `except` or match target, a nested
    `def` or `class`, a walrus): everything but a comprehension's own targets, which live in the
    comprehension's scope, and what a `global` or `nonlocal` statement merely declares."""
    nodes = list(_own_nodes(body))
    targets = {
        id(name) for node in nodes if isinstance(node, ast.comprehension)
        for name in ast.walk(node.target)}
    for node in nodes:
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            continue  # what it declares is not bound by it
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
    return names


def _locals(function: Function) -> frozenset[str]:
    """The names local to ``function``: its parameters and the names its own code binds, less
    those it declares `global` or `nonlocal` and a comprehension's own targets. A lambda's body
    is one expression, not a list of statements, but a walrus in it (unlike a comprehension's)
    binds in the lambda's own scope, so it is walked exactly as a function's statements are."""
    names = _bound_by(_own_body(function), _parameters(function))
    return frozenset(names) - _declared(function)


def _scope_parents(tree: ast.Module) -> dict[Function, Function | None]:
    """Each function and lambda of ``tree`` mapped to the function or lambda whose scope encloses
    it, or None for the module's: a class body between them is skipped, since a name a class binds
    is never what a function nested in it resolves to. A decorator or default runs in the scope
    that defines the function, only its body in its own. Walked with an explicit stack."""
    parents: dict[Function, Function | None] = {}
    stack: list[tuple[ast.AST, Function | None]] = [(tree, None)]
    while stack:
        node, scope = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            parents[node] = scope
            arguments = node.args
            stack.extend(
                (part, scope) for part in (
                    *getattr(node, "decorator_list", []), *arguments.defaults,
                    *(default for default in arguments.kw_defaults if default is not None)))
            stack.extend((part, node) for part in _own_body(node))
        else:
            stack.extend((child, scope) for child in ast.iter_child_nodes(node))
    return parents


def _global_writes(tree: ast.Module) -> frozenset[str]:
    """The names some function or class body of ``tree`` declares `global` and also binds: a name
    bound in the module's own scope by code that runs whenever that function is called, which no
    statement of the module's own flow shows."""
    written: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body: list[ast.AST] = list(node.body)
            declared = {
                name for inner in _own_nodes(body) if isinstance(inner, ast.Global)
                for name in inner.names}
            if declared:
                written |= declared & _bound_by(body, set())
    return frozenset(written)


def _values(node: ast.expr | None) -> set[ast.AST] | None:
    """The expressions making the objects ``node``'s value may be, or hold as a display does, or
    ``None`` if MAX_IFEXP_CHAIN is exhausted before the whole structure is visited.

    Walked iteratively, with an explicit worklist bounded by MAX_IFEXP_CHAIN nodes visited,
    rather than recursed into: a deeply right-nested ternary chain -- `a if c else b if c else
    ...`, exactly the shape a comprehension result may hold -- would otherwise recurse one Python
    frame per `else` and exceed the recursion limit well under a chain of hundreds of levels.
    Exhausting the bound with more of the structure left unvisited returns ``None`` instead of
    the incomplete set visited so far, so a caller fails closed (treats everything as possibly
    escaping) rather than silently acting on an unsound, truncated result.
    """
    made: set[ast.AST] = set()
    stack: list[ast.expr | None] = [node]
    visited = 0
    while stack:
        if visited >= MAX_IFEXP_CHAIN:
            return None
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


def _shadowed_non_retaining(bindings: Bindings) -> frozenset[str]:
    """Which of ``NON_RETAINING_CALLS`` are shadowed at this point: bound, on some path, to
    something other than the true builtin (a `def`, an assignment, an import, a parameter, a loop
    or `with`/`except`/match target -- anything that rebinds the name at all, whatever it is bound
    to). Read from `SHADOWED`, which every such rebinding records: a name's own entry in
    ``bindings`` cannot say, since a rebinding to anything the walk does not follow only drops
    it."""
    shadowed = bindings.get(SHADOWED, frozenset())
    return frozenset(name for name in NON_RETAINING_CALLS if name in shadowed)


def _comprehension_binders(node: Comprehension) -> frozenset[str]:
    """Which of ``NON_RETAINING_CALLS`` comprehension ``node`` itself rebinds: as the target of
    any generator of it or of a comprehension nested in it, or as a walrus target anywhere in it
    (which binds in the enclosing scope, but for every later iteration). A call anywhere within
    may then be calling the rebound name, so all of it is treated as shadowed (`_escaping` but
    for the outermost first iterable, which runs before any of it exists)."""
    bound: set[str] = set()
    for inner in _own_nodes([node]):
        if isinstance(inner, (ast.comprehension, ast.NamedExpr)):
            bound |= _bound_names(inner.target)
    return frozenset(bound & NON_RETAINING_CALLS)


def _non_retaining_argument_values(call: ast.Call, shadowed: frozenset[str]) -> list[ast.expr]:
    """The arguments of ``call`` that a narrow, unshadowed ``NON_RETAINING_CALLS`` exemption
    still leaves retaining: none, for `bool`/`any`/`all` (their one argument is only constructed
    and tested), but `next`'s own second (default) argument, since `next` may itself return that
    value verbatim, un-advanced, if the iterable is exhausted -- only its first (iterable)
    argument is genuinely just advanced."""
    if not isinstance(call.func, ast.Name) or call.func.id in shadowed:
        return [*call.args, *(keyword.value for keyword in call.keywords)]
    if call.func.id not in NON_RETAINING_CALLS:
        return [*call.args, *(keyword.value for keyword in call.keywords)]
    if call.func.id == "next":
        # `next` takes its default positionally only; any keyword argument here is already
        # invalid Python, so it is passed through as retaining rather than guessed at
        return [*call.args[1:], *(keyword.value for keyword in call.keywords)]
    return []


def _escaping(
    node: ast.ListComp | ast.SetComp | ast.DictComp, bindings: Bindings,
) -> set[ast.AST] | None:
    """The expressions whose objects outlive comprehension ``node``: those its result holds, a
    walrus binds (in the enclosing scope), or a call is handed, since an unknown callee (a method
    call, or a plain call of a name that is not statically known to be an unshadowed builtin) may
    keep it reachable. The narrow, explicit, binding- and argument-position-aware exceptions in
    ``NON_RETAINING_CALLS`` (via `_non_retaining_argument_values`) are calls known to only
    construct and test their argument, never store or return it; passing an object to one of
    them, and nothing else, leaves it a temporary of the comprehension. ``None`` (MAX_IFEXP_CHAIN
    exhausted for some sub-expression) propagates out, since a caller must then fail closed."""
    kept = _values(node)
    if kept is None:
        return None
    outside = _shadowed_non_retaining(bindings)
    inside = outside | _comprehension_binders(node)
    # the outermost first iterable runs before the comprehension's own targets exist, so what it
    # calls sees only the scope around it, and any comprehension nested within it
    first = node.generators[0].iter
    ahead = outside | _comprehension_binders(first)
    in_first = {id(inner) for inner in _own_nodes([first])}
    for inner in _own_nodes([node]):
        if isinstance(inner, ast.NamedExpr):
            more = _values(inner.value)
            if more is None:
                return None
            kept |= more
        elif isinstance(inner, ast.Call):
            shadowed = ahead if id(inner) in in_first else inside
            for argument in _non_retaining_argument_values(inner, shadowed):
                more = _values(argument)
                if more is None:
                    return None
                kept |= more
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

    A followed call of a `def` also carries what it writes to a variable it does not own: its
    `global` and `nonlocal` names, and what a function it calls writes that it can see. Each
    write belongs to the scope that OWNS the variable -- the module for `global`, the nearest
    enclosing function binding the name for `nonlocal` -- and reaches only a scope whose own name
    for it is that variable, never a local that merely shares the name; a write a shadowing caller
    cannot show still outlives it and reaches the module, or the function around it, from there.
    A name is left holding what every feasible exit leaves it, each exit starting from what the
    caller held (so one exit that leaves it alone keeps the caller's alias, and it is cleared only
    if every exit clears it). A parameter holds what the call passes: a conditional, `and`/`or` or
    walrus argument is any of its feasible values, one before a star binds exactly while one after
    it may land in any later parameter, a parameter the call certainly supplies never falls back
    to its default, and one it may not supply holds the default it had when the `def` ran.

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
    not in them, and a rebind is recorded once, on the set of objects alive where it happens, then
    handed to those objects in one pass per scope: the work per statement does not grow with the
    objects alive.

    A builtin `NON_RETAINING_CALLS` exempts (`bool`, `next`, `any`, `all`) keeps that exemption
    only where the source has not bound the name: every rebinding of it is recorded on the path
    (`SHADOWED`) whatever it is rebound to, a function's own locals shadow it throughout its
    body, a comprehension's targets and walrus targets shadow it throughout the comprehension
    (its outermost first iterable runs before them, in the scope around it), and a function that
    declares it `global` and binds it shadows it in the module. What a class body rebinds shadows
    only the class body itself: a comprehension in it runs its element in the enclosing scope.
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
        # in the current scope, the rebinds recorded but not yet handed to the objects that saw
        # them: on each alive-set node, by identity (with the node kept), the targets a name took
        # while that set was alive -- see `_observe` and `_flush`
        self._events: dict[int, tuple[_Alive, Views]] = {}
        # the objects started so far inside each comprehension being analyzed, outermost first
        self._started: list[list[str]] = []
        # in a class body: the lexical names it runs from, and the lazy objects it makes
        self._enclosing: tuple[Bindings, list[str]] | None = None
        # the locals of every function (or lambda) scope currently being resolved, innermost
        # last: a cross-scope call forwards a name only from at or above the callee's OWN
        # nesting depth (`_defined_depth`), since anything deeper is some intermediate caller's
        # own local, never what the callee's free reference resolves to -- the callee's own
        # ancestry, however deep, must still reach it unfiltered
        self._shadow: list[frozenset[str]] = []
        # the `len(self._shadow)` at the moment each function/lambda was itself defined: how many
        # enclosing function scopes lexically surround it
        self._defined_depth: dict[Deferred, int] = {}
        # syntax summaries, computed once per node
        self._later: dict[Deferred, bool] = {}
        self._escapes: dict[tuple[ast.AST, frozenset[str]], set[ast.AST] | None] = {}
        self._reads: dict[Function, frozenset[str]] = {}
        # the names a function's code may read and those it declares `global` or `nonlocal`,
        # widened with the same of every function it calls or nests, transitively
        self._reach_cache: dict[Function, tuple[frozenset[str], frozenset[str]]] = {}
        self._locals_cache: dict[Function, frozenset[str]] = {}
        self._globals_cache: dict[Function, frozenset[str]] = {}
        # each function and lambda by the function or lambda whose scope encloses it, from the
        # source alone: which scope owns a name a `global` or `nonlocal` declaration reaches
        self._parent: dict[Function, Function | None] = {}
        # every `def` of the module by name, wherever it is nested
        self._by_name: dict[str, list[Function]] = {}
        # the function scopes currently being resolved, innermost last (`_shadow`'s owners)
        self._functions: list[Function] = []
        # what each parameter's default may have held whenever the `def` ran: a default is
        # evaluated once, at definition, not at each call
        self._defaults: dict[Function, Views] = {}
        self._returns: dict[tuple[Function, frozenset[tuple[str, frozenset[str]]]], list[Returned]]
        self._returns = {}
        # the variables a call of a function may write and what it may leave each holding, by the
        # function and what it is seeded with (`_effect`); a function already being summarized (a
        # direct or mutual recursive helper) summarizes as having none, closing the recursion
        self._effects: dict[tuple[Function, frozenset[tuple[str, frozenset[str]]]], Effect] = {}
        self._summarizing: set[Function] = set()

    def module(self, tree: ast.Module) -> list[str]:
        state: Bindings = {"getattr": frozenset({BUILTIN_GETATTR})}
        # a function that declares one of these `global` and binds it rebinds it in the module
        # whenever it is called, which no statement of the module's own flow shows
        written = _global_writes(tree) & NON_RETAINING_CALLS
        if written:
            state[SHADOWED] = written
        self._parent = _scope_parents(tree)
        for function in self._parent:
            if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._by_name.setdefault(function.name, []).append(function)
        self._scope(tree.body, state)
        return list(dict.fromkeys(self.uses))  # loop and `finally` bodies may be visited twice

    def _scope(self, body: list[ast.stmt], bindings: Bindings) -> None:
        seen, self._seen = self._seen, {}
        events, self._events = self._events, {}
        deferred: list[Deferred] = []
        self._block(body, bindings, deferred)
        # a sibling defined earlier may be called from another sibling's body, possibly
        # transitively through several more (each hop discovered only once its own caller is
        # itself resolved): repeat until a full pass changes no sibling's own resolved entry, a
        # true fixpoint, bounded by one round per sibling -- enough for any acyclic chain among
        # them -- so a call cycle still terminates instead of looping forever. Re-walking a
        # sibling whose entry has not actually changed is skipped: re-walking it unconditionally
        # would re-discover the very same call sites every round (each nested `_scope` call
        # starts its own object bookkeeping fresh), making every round look like progress and
        # defeating the fixpoint check (duplicate self.uses lines from a genuine retry are
        # deduplicated on return regardless).
        resolved: dict[Deferred, frozenset[tuple[str, frozenset[str]]]] = {}
        for _ in range(len(deferred) + 1):
            changed = False
            for function in deferred:
                self._flush()  # every view read below must already include the rebinds so far
                calls = self._calls.get(function, [])
                # a lazy body runs only while one of its objects exists, so from what those
                # objects saw on their own paths; one never made here is made later, from the
                # final names
                reached = calls if calls and self._runs_later(function) else [bindings, *calls]
                entry = _lexical(_join(*reached))
                snapshot = frozenset(entry.items())
                if resolved.get(function) == snapshot:
                    continue
                resolved[function] = snapshot
                changed = True
                if isinstance(function, ast.GeneratorExp):
                    self._comprehension(function, entry, deferred)
                    continue
                local = {k: v for k, v in entry.items() if k not in _parameters(function)}
                # whatever it binds anywhere, a parameter included, is local to it throughout
                # its body (Python scopes a name to the whole function), so it shadows a builtin
                # from its first statement on
                owned = self._locals_of(function)
                own = owned & NON_RETAINING_CALLS
                if own:
                    local[SHADOWED] = frozenset(local.get(SHADOWED, frozenset())) | own
                # a lambda shares its enclosing scope's deferred list, but still owns its own
                # locals: it must push them too, so a cross-scope call made from inside it
                # excludes exactly what a `def` in the same position would
                self._shadow.append(owned)
                self._functions.append(function)
                try:
                    if isinstance(function, ast.Lambda):
                        self._expression(function.body, local, deferred)
                    else:
                        self._scope(function.body, local)
                finally:
                    self._shadow.pop()
                    self._functions.pop()
            if not changed:
                break
        self._seen, self._events = seen, events

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
        on it that reads ``name`` late-bound, not from a closure, sees the new targets. Rebinding
        a builtin `NON_RETAINING_CALLS` exempts is recorded whatever it is rebound to, so the
        exemption is never granted for a name the source has bound."""
        if name in NON_RETAINING_CALLS:
            shadowed = frozenset(bindings.get(SHADOWED, frozenset())) | {name}
            bindings[SHADOWED] = shadowed
            self._observe(SHADOWED, shadowed, bindings)
        if not targets:
            bindings.pop(name, None)
            return
        bindings[name] = targets
        self._observe(name, targets, bindings)

    def _observe(self, name: str, targets: frozenset[str], bindings: Bindings) -> None:
        """Every lazy object alive on this path that reads ``name`` late-bound, not from a
        closure, sees ``targets``. Recorded once, in O(1), on the alive set the rebind happens in
        (an immutable node, so what it includes never changes); `_flush` hands it to each object
        that set includes, all rebinds at once, instead of visiting every live object per rebind."""
        alive = bindings.get(ALIVE)
        if alive is None:
            return
        node = alive if isinstance(alive, _Alive) else _Alive(left=alive)
        seen = self._events.setdefault(id(node), (node, {}))[1]
        seen[name] = seen.get(name, frozenset()) | targets

    def _flush(self) -> None:
        """Hand every object the rebinds recorded on the alive sets it is in: what is recorded on a
        node reaches every object that node includes (an id joined in later, or made on a sibling
        path, is in a different node and does not see it), in one pass over those nodes, each
        after every node that includes it. Applied at most once, and only where a view is read."""
        events, self._events = self._events, {}
        if not events:
            return
        reaching: dict[int, Views] = {}  # per node, what it and every node including it saw
        for node in _alive_order([node for node, _ in events.values()]):
            view = reaching.pop(id(node), {})
            if id(node) in events:
                view = _union_views(view, events[id(node)][1])
            if not view:
                continue
            if node._oid is not None:
                self._deliver(node._oid, view)
            for child in (node._left, node._right):
                if isinstance(child, _Alive):
                    reaching[id(child)] = _union_views(reaching.get(id(child), {}), view)
                elif child is not None:  # an already-materialized set of ids
                    for oid in child:
                        self._deliver(oid, view)

    def _deliver(self, oid: str, view: Views) -> None:
        """Object ``oid`` sees ``view``, but for the names it reads from a closure."""
        closure = self._lazy[oid][1]
        seen = self._seen[oid]
        for name, targets in view.items():
            if name not in closure:
                seen[name] = frozenset(seen.get(name, frozenset())) | targets

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
        for started in self._started:
            started.append(oid)

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
            own = list(_own_nodes(_own_body(function)))
            # a nested function's decorators and defaults run HERE, when its `def` does
            evaluated = [
                part for node in own if isinstance(node, Function)
                for part in (
                    *getattr(node, "decorator_list", []), *node.args.defaults,
                    *(default for default in node.args.kw_defaults if default is not None))]
            self._reads[function] = frozenset(
                node.id for node in (*own, *_own_nodes(evaluated))
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load))
        return self._reads[function]

    def _reach(
        self, function: Function, visiting: frozenset[Function] = frozenset(),
    ) -> tuple[frozenset[str], frozenset[str]]:
        """The plain names ``function``'s own top-level code may read and those it declares
        `global` or `nonlocal`, widened with the same of every function of the module it calls by
        a plain name, transitively -- whether it is defined beside it, around it, or inside it: a
        helper that only forwards to another (reading nothing by name itself) must still see
        everything the forwarded-to function needs seeded, or a composed effect (see
        `_ReturnFlow._called`) would starve for names ``_entry`` projected away one level too
        early. A name may be several functions' (any that shares it is included: seeding more
        than is read only widens a key). Bounded against a call cycle by ``visiting`` (the
        functions already being expanded on this path); the top-level result (an empty
        ``visiting``) is cached, since it does not depend on it."""
        if not visiting and function in self._reach_cache:
            return self._reach_cache[function]
        if function in visiting:
            return frozenset(), frozenset()
        reads, declared = self._read_names(function), _declared(function)
        if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            called = frozenset(
                node.func.id for node in _own_nodes(_own_body(function))
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name))
            deeper = visiting | {function}
            for name in called:
                for callee in self._by_name.get(name, ()):
                    more_reads, more_declared = self._reach(callee, deeper)
                    reads, declared = reads | more_reads, declared | more_declared
        if not visiting:
            self._reach_cache[function] = (reads, declared)
        return reads, declared

    def _locals_of(self, function: Function) -> frozenset[str]:
        if function not in self._locals_cache:
            self._locals_cache[function] = _locals(function)
        return self._locals_cache[function]

    def _globals_of(self, function: Function) -> frozenset[str]:
        if function not in self._globals_cache:
            self._globals_cache[function] = _declared(function, ast.Global)
        return self._globals_cache[function]

    def _owner(self, scope: Function | None, name: str) -> Function | None:
        """The scope whose variable ``name`` is when read or written from ``scope`` (None for the
        module's): the nearest enclosing function that binds it, `global` skipping them all, and
        `nonlocal` or a name it merely reads passing on to the scope around it."""
        while scope is not None:
            if name in self._globals_of(scope):
                return None
            if name in self._locals_of(scope):
                return scope
            scope = self._parent.get(scope)
        return None

    def _extend(self, name: str, targets: frozenset[str], bindings: Bindings) -> None:
        """``name`` may also hold ``targets`` from here on: a write that may not have happened."""
        if targets:
            self._set(name, frozenset(bindings.get(name, frozenset())) | targets, bindings)

    def _values_of(self, node: ast.expr | None, bindings: Bindings) -> frozenset[str]:
        """Every target the value of ``node`` may be on some feasible path: what a name or
        attribute chain holds, a lambda, either result of a conditional expression that a literal
        test does not decide, each operand of `and`/`or` up to the first a literal decides, and a
        walrus's value. Walked iteratively: a chain of them may be deeper than the stack."""
        found: set[str] = set()
        stack: list[ast.expr | None] = [node]
        while stack:
            current = stack.pop()
            if isinstance(current, ast.IfExp):
                truth = _static_truth(current.test)
                stack.extend(
                    branch for branch, taken in ((current.body, True), (current.orelse, False))
                    if truth in (None, taken))
            elif isinstance(current, ast.BoolOp):
                conjunction = isinstance(current.op, ast.And)
                for value in current.values:
                    stack.append(value)
                    if _static_truth(value) is (not conjunction):
                        break  # a literal decides the result: the rest never runs
            elif isinstance(current, ast.NamedExpr):
                stack.append(current.value)
            else:
                found |= _WalkReferences._resolved(self, current, bindings)
        return frozenset(found)

    def _snapshot_defaults(self, function: Function, bindings: Bindings) -> None:
        """What each of ``function``'s defaults holds as of NOW, when its `def` runs: a default is
        evaluated once, at definition, never again at a call, whatever its names hold by then. A
        `def` that runs more than once (in a loop, or on several paths) may have held different
        values each time, so they accumulate."""
        arguments = function.args
        positional = [*arguments.posonlyargs, *arguments.args]
        pairs = [
            *zip(positional[len(positional) - len(arguments.defaults):], arguments.defaults,
                 strict=True),
            *((parameter, default) for parameter, default
              in zip(arguments.kwonlyargs, arguments.kw_defaults, strict=True)
              if default is not None),
        ]
        held = self._defaults.setdefault(function, {})
        for parameter, default in pairs:
            targets = frozenset(t for t in self._values_of(default, bindings) if self._followed(t))
            if targets:
                held[parameter.arg] = frozenset(held.get(parameter.arg, frozenset())) | targets

    def _bind_arguments(
        self, function: Function, site: ast.Call | None, bindings: Bindings,
    ) -> Bindings:
        """The targets each of ``function``'s parameters may hold when called at ``site``, with
        the arguments resolved from ``bindings`` (the caller's own state). A parameter the call
        certainly supplies -- by a positional argument before any star, or by name -- holds only
        what was passed, however little of it the walk follows: it never falls back to its
        default. One it may or may not supply holds the default too, and that is what it was at
        definition (`_snapshot_defaults`). An argument after a star may land in any parameter
        from the earliest position the expansions leave it (each may be empty); a `**` fills no
        parameter the walk can name. A conditional, Boolean or walrus argument is any of its
        feasible values (`_values_of`)."""
        bound: Bindings = {}
        if isinstance(function, ast.Lambda):
            return bound
        arguments = function.args
        positional = [*arguments.posonlyargs, *arguments.args]
        keywordable = {a.arg for a in (*arguments.args, *arguments.kwonlyargs)}
        given: dict[str, set[str]] = {}
        supplied: set[str] = set()
        if site is not None:
            star = next((i for i, a in enumerate(site.args) if isinstance(a, ast.Starred)), None)
            for parameter, argument in zip(positional, site.args[:star], strict=False):
                supplied.add(parameter.arg)
                given.setdefault(parameter.arg, set()).update(self._values_of(argument, bindings))
            if star is not None:
                expansions = 0
                for index, argument in enumerate(site.args[star:], start=star):
                    if isinstance(argument, ast.Starred):
                        expansions += 1
                        continue
                    for parameter in positional[index - expansions:]:
                        given.setdefault(parameter.arg, set()).update(
                            self._values_of(argument, bindings))
            for keyword in site.keywords:
                if keyword.arg in keywordable:
                    supplied.add(keyword.arg)
                    given.setdefault(keyword.arg, set()).update(
                        self._values_of(keyword.value, bindings))
        for name, held in self._defaults.get(function, {}).items():
            if name not in supplied:
                given.setdefault(name, set()).update(held)
        for name, passed in given.items():
            self._bind(name, passed, bound)
        return bound

    def _entry(self, function: Function, state: Bindings) -> Bindings:
        """``state`` projected to the names ``function`` (or a function it calls by a plain name
        or nests, transitively -- `_reach`, so a forwarding helper's own call still seeds what the
        forwarded-to function needs) may actually read: every name it owns (not only its
        parameters, but any other name it locally binds -- an import, a loop or
        `with`/`except`/match target, a nested `def`/`class`, anywhere in its own top-level code)
        is its own, never the caller's, and any name nothing in this chain reads cannot affect
        what it does, so neither belongs in what a cache keys its summary or effect on. A name it
        declares `global` or `nonlocal` is excluded from its own locals (`_locals`), so a genuine
        read of one still comes through."""
        reads, _ = self._reach(function)
        owned = self._locals_of(function)
        return {k: v for k, v in state.items() if k in reads and k not in owned}

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
        flow = _ReturnFlow(self, function)
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

    def _effect(
        self, function: Function, state: Bindings, scope: Function | None,
        site: ast.Call | None = None, given: Bindings | None = None,
    ) -> Effect:
        """The variables a call of ``function`` may write, and what it may leave each holding.

        A variable is one a call may write if the function declares it `global` or `nonlocal`, or a
        function it calls writes it where it can see it. Each write is keyed by the scope that
        OWNS the variable (the module for `global`, the nearest enclosing function binding the
        name for `nonlocal`), so it reaches only a scope whose own name resolves to that variable
        (`_owner`), never a local that merely shares the name. A variable ``function`` itself owns
        is not an effect: it dies with the call. A write a called function makes to a variable
        this one shadows with a local of its own cannot show in this one's state, but the variable
        outlives the call, so the write is passed on as one that may have happened (not exact),
        since what the variable held before is unknown here.

        ``scope`` is the function (or lambda) making the call, None for the module: a declared
        name is seeded from ``state`` only where the caller's own name for it is the very variable
        the callee's is, so the caller's local is never mistaken for the module's. ``given`` is
        the caller's own state, where the arguments of ``site`` are resolved (they are evaluated
        in the caller, whatever ``state`` hides from the callee). The value a declared name is
        left holding is what every feasible exit -- falling through the end, or any `return` --
        leaves it, each starting from what it held on the way in; so a name that some exit leaves
        untouched keeps what the caller held, and it is cleared only if EVERY exit rebinds it to
        something the walk does not follow (or deletes it). A function no exit of which is
        feasible writes nothing. A same-scope call it makes composes that function's own effect in
        (see `_ReturnFlow._called`); a directly or mutually recursive helper's in-flight call
        summarizes as having none (the shared ``_summarizing`` guard), closing the recursion, and
        the real answer, from its own code, is still cached once computed."""
        _, declared_chain = self._reach(function)
        seed = self._entry(function, state)
        for name in declared_chain:
            if name in state and self._owner(scope, name) is self._owner(function, name):
                seed[name] = state[name]
            else:
                seed.pop(name, None)
        seed.update(self._bind_arguments(function, site, state if given is None else given))
        key = (function, frozenset(seed.items()))
        if key in self._effects:
            return self._effects[key]
        if function in self._summarizing:
            return {}
        declared = _declared(function)
        # a function with no declared name of its own and no call of any kind has nothing to
        # compose either, so the flow below is skipped entirely for the common case; one with a
        # call but no declared name of its own may still COMPOSE one transitively (see
        # `_ReturnFlow._called`), so it is not skipped just for that
        has_calls = any(isinstance(n, ast.Call) for n in _own_nodes(_own_body(function)))
        effect: Effect = {}
        if declared or has_calls:
            self._summarizing.add(function)
            try:
                flow = _ReturnFlow(self, function, declared)
                work = dict(seed)
                falls = flow._block(_own_body(function), work, [])
                exits = ([work] if falls else []) + [s for s, _ in flow.returns]
                if exits:
                    for name in flow.relevant:
                        owner = self._owner(function, name)
                        if owner is not function:  # a local of the function itself dies with it
                            after = frozenset().union(
                                *(state_.get(name, frozenset()) for state_ in exits))
                            if after != seed.get(name, frozenset()):  # only a change is a write
                                effect[(owner, name)] = _Write(after, True)
                    for variable, targets in flow.bypass.items():
                        effect[variable] = _Write(targets, False)
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
        function resolves ``function`` from ITS OWN defining lexical environment, not the
        caller's: only the names shadowed by some scope strictly deeper than where ``function``
        was itself defined are excluded (a caller's own local, or one belonging to any
        intermediate scope between the call site and ``function``'s ancestry, is never what its
        free reference resolves to); anything at or above that depth is part of ``function``'s
        own lexical ancestry and must reach it unfiltered, however many scopes it climbs through.
        What the call writes to a variable the caller's own scope sees (`_effect`) reaches the
        caller's state itself, not the copy that hides its locals from the callee.
        """
        real = bindings
        if function not in deferred:
            if self._enclosing or not self._shadow:
                return
            depth = self._defined_depth.get(function, 0)
            exclude: frozenset[str] = frozenset().union(*self._shadow[depth:])
            bindings = {
                k: v for k, v in bindings.items() if k in (ALIVE, SHADOWED) or k not in exclude}
            if SHADOWED in bindings:  # a deeper scope's own shadow is never the callee's
                bindings[SHADOWED] = frozenset(bindings[SHADOWED]) - exclude
        # a class body runs at once, but what it calls reads the lexical names
        state = self._enclosing[0] if self._enclosing else _lexical(bindings)
        made: list[Returned]
        if self._runs_later(function):
            made = [(function, frozenset())]
        else:
            self._calls.setdefault(function, []).append(dict(state))
            made = self._returned(function, state)  # the lazy objects the call hands back
            if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                call_site = site if isinstance(site, ast.Call) else None
                scope = self._functions[-1] if self._functions else None
                effect = self._effect(function, state, scope, call_site, _lexical(real))
                for (owner, name), write in effect.items():
                    if self._owner(scope, name) is owner:
                        if write.exact:
                            self._set(name, write.targets, real)
                        else:
                            self._extend(name, write.targets, real)
        for lazy, closure in made:
            oid = f"{id(lazy)}@{id(site)}"  # one object per function and expression making it
            self._lazy[oid] = (lazy, closure, site)
            self._made(oid, bindings)

    def _comprehension(
        self, node: Comprehension, bindings: Bindings, deferred: list[Deferred],
    ) -> None:
        """Analyze a comprehension in its own scope (a generator's first iterable ran when made)."""
        local = dict(bindings)
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
                    truthy, _ = self._condition(condition, local, deferred)
                    if truthy is None:
                        return  # the filter is never true: the rest never runs
                    local = truthy  # only an item the filter keeps reaches what follows
            parts = (node.key, node.value) if isinstance(node, ast.DictComp) else (node.elt,)
            for part in parts:
                self._expression(part, local, deferred)
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
            self._defined_depth[node] = len(self._shadow)
            deferred.append(node)
            return
        if isinstance(node, ast.GeneratorExp):
            # only the first iterable runs now; the rest runs as the generator is advanced
            self._expression(node.generators[0].iter, bindings, deferred)
            deferred.append(node)
            self._called(node, node, bindings, deferred)
            return
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp)):
            # the objects it made, recorded as they start: none is flattened out of the alive set
            started: list[str] = []
            self._started.append(started)
            try:
                self._comprehension(node, bindings, deferred)
            finally:
                self._started.pop()
            # a class comprehension's element runs in the enclosing scope, never in the class, so
            # what the class itself rebinds does not shadow what it calls
            scope = self._enclosing[0] if self._enclosing else bindings
            shadowed = _shadowed_non_retaining(scope)
            key = (node, shadowed)
            if key in self._escapes:
                escaping = self._escapes[key]
            else:
                escaping = self._escapes[key] = _escaping(node, scope)
            for oid in dict.fromkeys(started):
                # MAX_IFEXP_CHAIN exhausted (`escaping is None`): fail closed, nothing here is a
                # temporary rather than silently trusting an incomplete, unsound result
                if escaping is None or self._lazy[oid][2] in escaping:
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
            self._snapshot_defaults(node, bindings)  # evaluated now, once, not at each call
            self._defined[_marker(node)] = node  # the name now refers to this function
            self._defined_depth[node] = len(self._shadow)
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

    def __init__(
        self, caller: _WalkReferences, function: Function,
        declared: frozenset[str] = frozenset(),
    ) -> None:
        super().__init__(caller._package)
        self._caller = caller
        self._function = function  # the scope this flow runs in
        # every function seen so far, whichever flow saw it: a `def` run here must be known to a
        # call composed later from another flow (a marker names one node, so nothing collides)
        self._defined = caller._defined
        self._later = caller._later
        self._defined_depth = caller._defined_depth
        self._parent = caller._parent  # source facts, shared
        self._by_name = caller._by_name
        self._defaults = caller._defaults  # a nested `def` run here is defined for later calls too
        self._locals_cache = caller._locals_cache
        self._globals_cache = caller._globals_cache
        self.returns: list[tuple[Bindings, frozenset[str]]] = []
        self._exits = [self.returns]
        self.objects: dict[str, Deferred] = {}  # each object target by its function
        # the names an effect actually reports: the callee's own declared names, seeded up
        # front, plus whatever a composed call (below) writes to a variable this function sees
        self.relevant: set[str] = set(declared)
        # what a composed call writes to a variable this function's own local hides, which
        # outlives the call: not in this flow's state, but still a write the caller must see
        self.bypass: dict[tuple[Function | None, str], frozenset[str]] = {}

    def _called(
        self, function: Deferred, site: ast.AST, bindings: Bindings, deferred: list[Deferred],
    ) -> None:
        """A call in the callee is not followed for its own lazy-object returns -- this is no
        call graph -- but a call it makes to ANOTHER function that writes a variable composes that
        function's own effect into this flow (delegating back to the caller's own `_effect`, which
        shares its `_summarizing` guard, so a direct or mutual recursive cycle still closes rather
        than recursing forever), so a helper's helper's write is still seen by the time this
        callee's own effect is computed, even when this callee declares no global/nonlocal name of
        its own. A write to the variable this function's own name for it denotes lands in this
        flow's state, exactly or as an addition as the callee's effect says. One to a variable
        this function's own local hides never touches that local, but the variable belongs to a
        scope around this function (or the module) and outlives the call, so the write is kept
        aside as one that may have happened."""
        if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)) and not self._runs_later(
            function,
        ):
            call_site = site if isinstance(site, ast.Call) else None
            state = _lexical(bindings)
            effect = self._caller._effect(function, state, self._function, call_site)
            for (owner, name), write in effect.items():
                if self._caller._owner(self._function, name) is owner:
                    if write.exact:
                        self._set(name, write.targets, bindings)
                    else:
                        self._extend(name, write.targets, bindings)
                    self.relevant.add(name)
                else:  # a scope around this function owns it, or the module: it outlives the call
                    variable = (owner, name)
                    self.bypass[variable] = self.bypass.get(variable, frozenset()) | write.targets

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


# a generator that reads `m`, made before the alias goes raw: if whatever it is handed retains it,
# it sees `m = w`. The shadow family below builds on these two halves.
LAZY_READS_M = "m = make()\ndef gen():\n    yield m.bounded_walk(root)\n"
RAW_THEN_SAFE = "m = w\nm = make()\n"
# a helper that leaves the module's `m` holding whatever it is passed (or, by default, the raw
# module), then the use of `m` that a raw value would make a direct walk
MUTATE_GLOBAL = "m = make()\ndef mutate(x):\n    global m\n    m = x\n"
MUTATE_GLOBAL_RAW_DEFAULT = "m = make()\ndef mutate(x=w):\n    global m\n    m = x\n"
WALK_M = "m.bounded_walk(root)\n"
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
    # 6hfQqGxWrc4QpJqp: a shadowed bool/next/any/all is not the true builtin, so it retains
    "generator-passed-to-a-shadowed-bool-retains": (
        "m = make()\ndef gen():\n    yield m.bounded_walk(root)\n"
        "def bool(x):\n    stash.append(x)\n    return True\n"
        "[bool(gen()) for _ in xs]\nm = w\nm = make()\n"),
    # 6hfQqGxWrc4QpJqp: `next`'s own second (default) argument may itself be the un-advanced
    # result, so it retains even though the true, unshadowed builtin is used
    "generator-as-next-default-argument-retains": (
        "m = make()\ndef gen():\n    yield m.bounded_walk(root)\n"
        "items = [next(gen(), gen()) for _ in xs]\nm = w\nm = make()\n"),
    # 6hfQqH5p7CXJJCvG: deferred cross-scope discovery reaches a chain longer than two hops
    "cross-scope-discovery-reaches-a-three-hop-reverse-ordered-chain": (
        "m = make()\ndef gen():\n    yield m.bounded_walk(root)\n"
        "def c1():\n    next(gen())\n"
        "def c2():\n    c1()\n"
        "def c3():\n    c2()\n"
        "if x:\n    m = w\nc3()\nm = make()\n"),
    "cross-scope-discovery-reaches-a-five-hop-reverse-ordered-chain": (
        "m = make()\ndef gen():\n    yield m.bounded_walk(root)\n"
        "def c1():\n    next(gen())\n"
        "def c2():\n    c1()\n"
        "def c3():\n    c2()\n"
        "def c4():\n    c3()\n"
        "def c5():\n    c4()\n"
        "if x:\n    m = w\nc5()\nm = make()\n"),
    # 6hfQqH2qxwxhVWHG: a lambda nested in a function forwards a feasible sibling alias to a
    # cross-scope call it makes, the same as a `def` in the same position would
    "lambda-nested-in-a-function-forwards-a-cross-scope-call": (
        "m = make()\ndef gen():\n    yield m.bounded_walk(root)\n"
        "def runner():\n    call = lambda: next(gen())\n    call()\n"
        "if c:\n    m = w\nrunner()\nm = make()\n"),
    # 6hfQqH4P4QC6r8Rp: a helper's effect binds the actual call-site argument to its parameter
    "helper-effect-binds-a-positional-argument": (
        "m = make()\ndef mutate(x):\n    global m\n    m = x\n"
        "mutate(w)\nm.bounded_walk(root)\n"),
    "helper-effect-binds-a-keyword-argument": (
        "m = make()\ndef mutate(x):\n    global m\n    m = x\n"
        "mutate(x=w)\nm.bounded_walk(root)\n"),
    "helper-effect-binds-an-unspecified-parameters-default": (
        "m = make()\ndef mutate(x=w):\n    global m\n    m = x\n"
        "mutate()\nm.bounded_walk(root)\n"),
    # 6hfQqH53pjrMWgxp: a helper's own effect composes a same-scope helper it calls that declares
    # no global/nonlocal name of its own
    "helper-effect-composes-a-transitively-called-helpers-effect": (
        "m = make()\ndef deeper():\n    global m\n    m = w\n"
        "def mutate():\n    deeper()\n"
        "mutate()\nm.bounded_walk(root)\n"),
    "helper-effect-composes-through-two-hops": (
        "m = make()\ndef deepest():\n    global m\n    m = w\n"
        "def deeper():\n    deepest()\n"
        "def mutate():\n    deeper()\n"
        "mutate()\nm.bounded_walk(root)\n"),
    # 6hfQqH53pjrMWgxp: a directly or mutually recursive effect-composing helper still reports
    # its own direct write and terminates rather than recursing forever
    "self-recursive-composing-helper-still-reports-its-own-write": (
        "m = make()\ndef mutate():\n    global m\n    m = w\n    mutate()\n"
        "mutate()\nm.bounded_walk(root)\n"),
    "mutually-recursive-composing-helpers-terminate-and-report": (
        "m = make()\ndef a():\n    global m\n    m = w\n    b()\n"
        "def b():\n    global m\n    a()\n"
        "a()\nm.bounded_walk(root)\n"),
    # 6hfQqH5HcRMg38jp: MAX_IFEXP_CHAIN exhaustion fails closed instead of silently dropping the
    # unvisited remainder of a wide (not merely deep) structure
    "max-ifexp-chain-exhaustion-fails-closed-on-a-wide-tuple": (
        "m = make()\ndef gen():\n    yield m.bounded_walk(root)\n"
        + "[(store.add((gen()," + ",".join(["safe()"] * MAX_IFEXP_CHAIN) + ")), 0) for _ in xs]\n"
        + "m = w\nm = make()\n"),
    # 6hfRG3rqrGgFGWfp: a bool/next/any/all the source binds in ANY way is not the builtin, so it
    # may retain what it is handed; `Bindings` only ever holds targets on the way to the raw walk,
    # so it cannot be what says a name is shadowed
    "assignment-shadows-bool": (
        LAZY_READS_M + "bool = stash\n[bool(gen()) for _ in xs]\n" + RAW_THEN_SAFE),
    "assignment-shadows-next": (
        LAZY_READS_M + "next = stash\n[next(gen()) for _ in xs]\n" + RAW_THEN_SAFE),
    "assignment-shadows-any": (
        LAZY_READS_M + "any = stash\n[any(gen()) for _ in xs]\n" + RAW_THEN_SAFE),
    "assignment-shadows-all": (
        LAZY_READS_M + "all = stash\n[all(gen()) for _ in xs]\n" + RAW_THEN_SAFE),
    "annotated-assignment-shadows-bool": (
        LAZY_READS_M + "bool: object = stash\n[bool(gen()) for _ in xs]\n" + RAW_THEN_SAFE),
    "import-from-shadows-bool": (
        LAZY_READS_M + "from helpers import bool\n[bool(gen()) for _ in xs]\n" + RAW_THEN_SAFE),
    "import-as-shadows-bool": (
        LAZY_READS_M + "import stash as bool\n[bool(gen()) for _ in xs]\n" + RAW_THEN_SAFE),
    "for-target-shadows-bool": (
        LAZY_READS_M + "for bool in fns:\n    [bool(gen()) for _ in xs]\n" + RAW_THEN_SAFE),
    "with-target-shadows-bool": (
        LAZY_READS_M + "with opener() as bool:\n    [bool(gen()) for _ in xs]\n" + RAW_THEN_SAFE),
    "except-name-shadows-bool": (
        LAZY_READS_M + "try:\n    pass\nexcept Exception as bool:\n"
        "    [bool(gen()) for _ in xs]\n" + RAW_THEN_SAFE),
    "match-capture-shadows-bool": (
        LAZY_READS_M + "match value:\n    case bool:\n        [bool(gen()) for _ in xs]\n"
        + RAW_THEN_SAFE),
    "class-definition-shadows-bool": (
        LAZY_READS_M + "class bool:\n    pass\n[bool(gen()) for _ in xs]\n" + RAW_THEN_SAFE),
    "shadow-on-one-branch-may-retain": (
        LAZY_READS_M + "if c:\n    bool = stash\n[bool(gen()) for _ in xs]\n" + RAW_THEN_SAFE),
    "parameter-shadows-bool": (
        "def outer(bool):\n    m = make()\n    def gen():\n        yield m.bounded_walk(root)\n"
        "    [bool(gen()) for _ in xs]\n    m = w\n    m = make()\n"),
    "keyword-only-parameter-shadows-next": (
        "def outer(*, next):\n    m = make()\n    def gen():\n        yield m.bounded_walk(root)\n"
        "    [next(gen()) for _ in xs]\n    m = w\n    m = make()\n"),
    "variadic-parameter-shadows-any": (
        "def outer(*any):\n    m = make()\n    def gen():\n        yield m.bounded_walk(root)\n"
        "    [any(gen()) for _ in xs]\n    m = w\n    m = make()\n"),
    "function-local-import-shadows-bool": (
        "def outer():\n    from helpers import bool\n    m = make()\n    def gen():\n"
        "        yield m.bounded_walk(root)\n    [bool(gen()) for _ in xs]\n"
        "    m = w\n    m = make()\n"),
    "nested-function-sees-an-enclosing-parameter-shadow": (
        "def outer(bool):\n    def inner():\n        m = make()\n        def gen():\n"
        "            yield m.bounded_walk(root)\n        [bool(gen()) for _ in xs]\n"
        "        m = w\n        m = make()\n    inner()\n"),
    "comprehension-target-shadows-bool": (
        LAZY_READS_M + "[bool(gen()) for bool in fns]\n" + RAW_THEN_SAFE),
    "set-comprehension-target-shadows-next": (
        LAZY_READS_M + "{next(gen()) for next in fns}\n" + RAW_THEN_SAFE),
    "dict-comprehension-target-shadows-all": (
        LAZY_READS_M + "{k: all(gen()) for k, all in fns}\n" + RAW_THEN_SAFE),
    "later-generator-target-shadows-bool": (
        LAZY_READS_M + "[bool(gen()) for _ in xs for bool in fns]\n" + RAW_THEN_SAFE),
    "nested-comprehension-target-shadows-bool": (
        LAZY_READS_M + "[[bool(gen()) for bool in fns] for _ in xs]\n" + RAW_THEN_SAFE),
    "walrus-in-a-filter-shadows-bool": (
        LAZY_READS_M + "[bool(gen()) for _ in xs if (bool := stash)]\n" + RAW_THEN_SAFE),
    "helper-declaring-global-and-binding-bool-shadows-it": (
        LAZY_READS_M + "def install():\n    global bool\n    bool = stash\ninstall()\n"
        "[bool(gen()) for _ in xs]\n" + RAW_THEN_SAFE),
    # 6hfRG3rqrGgFGWfp: a comprehension's own targets exist only after its outermost first
    # iterable has run, but a comprehension nested in that iterable has its own
    "a-nested-comprehension-in-the-first-iterable-keeps-its-own-target-shadow": (
        "m = make()\ndef gen():\n    yield m.bounded_walk(root)\n"
        "[x for x in [bool(gen()) for bool in fns]]\nm = w\nm = make()\n"),
    # a class comprehension's element runs in the enclosing scope, so a parameter of the function
    # around the class still shadows it
    "a-class-comprehension-element-sees-the-enclosing-functions-parameter-shadow": (
        "def outer(bool):\n    m = make()\n    def gen():\n"
        "        yield m.bounded_walk(root)\n    class K:\n"
        "        flags = [bool(gen()) for _ in xs]\n    m = w\n    m = make()\n"),
    # nothing in the module's own flow calls `install`, but whoever does rebinds the module's `bool`
    # for every function that reads it
    "uncalled-helper-declaring-global-and-binding-bool-shadows-it-for-a-function": (
        "def install():\n    global bool\n    bool = stash\n"
        "def worker():\n    n = make()\n    def gen():\n        yield n.bounded_walk(root)\n"
        "    [bool(gen()) for _ in xs]\n    n = w\n    n = make()\n"),
    # 6hfRG42g8WcVmw8G: a rebind is recorded on the alive set it happens in and reaches every
    # object that set includes, whichever side of a join holds it
    "objects-made-on-both-branches-see-a-rebind-after-the-join-first-branch-reads": (
        "m = make()\ndef gen1():\n    yield m.bounded_walk(root)\ndef gen2():\n    yield None\n"
        "if c:\n    a = gen1()\nelse:\n    b = gen2()\nm = w\nm = make()\n"),
    "objects-made-on-both-branches-see-a-rebind-after-the-join-second-branch-reads": (
        "m = make()\ndef gen1():\n    yield None\ndef gen2():\n    yield m.bounded_walk(root)\n"
        "if c:\n    a = gen1()\nelse:\n    b = gen2()\nm = w\nm = make()\n"),
    "an-earlier-rebind-survives-the-name-being-rebound-again-on-the-same-set": (
        "import algua.primitives as p\nm = make()\ndef gen():\n    yield m.bounded_walk(root)\n"
        "items = gen()\nm = w\nm = p\nm = make()\n"),
    "an-object-reaching-a-join-through-both-sides-sees-a-later-rebind": (
        "m = make()\ndef gen():\n    yield m.bounded_walk(root)\n"
        "if c:\n    a = gen()\nif d:\n    pass\nelse:\n    pass\nm = w\nm = make()\n"),
    # a lazy body reads `bool` late-bound: one bound after the object was made is what it sees
    "a-builtin-shadowed-after-a-lazy-object-is-made-is-seen-when-it-runs": (
        "def lazy():\n    n = make()\n    def gen():\n        yield n.bounded_walk(root)\n"
        "    [bool(gen()) for _ in xs]\n    n = w\n    n = make()\n    yield 1\n"
        "items = lazy()\nbool = stash\n"),
    # 6hfRG3rM9F8mQjmp: a call's arguments bind conservatively -- a conditional, Boolean or walrus
    # argument may be any feasible operand, a default is what it was when the function was DEFINED,
    # what precedes a star binds exactly while what follows may land in any later parameter, and a
    # keyword beside `**` still binds
    "conditional-argument-may-be-raw": (MUTATE_GLOBAL + "mutate(w if c else make())\n" + WALK_M),
    "conditional-argument-else-may-be-raw": (
        MUTATE_GLOBAL
        + "mutate(make() if c else w)\n"
        + WALK_M),
    "and-argument-may-be-raw": (MUTATE_GLOBAL + "mutate(c and w)\n" + WALK_M),
    "or-argument-may-be-raw": (MUTATE_GLOBAL + "mutate(c or w)\n" + WALK_M),
    "walrus-argument-value-may-be-raw": (MUTATE_GLOBAL + "mutate((y := w))\n" + WALK_M),
    "nested-conditional-argument-may-be-raw": (
        MUTATE_GLOBAL
        + "mutate(make() if c else (other() if d else w))\n"
        + WALK_M),
    "default-is-snapshotted-when-the-function-is-defined-raw": (
        "a = w\ndef mutate(x=a):\n    global m\n    m = x\na = make()\nmutate()\n"
        + WALK_M),
    "keyword-only-default-is-snapshotted-when-defined-raw": (
        "a = w\ndef mutate(*, x=a):\n    global m\n    m = x\na = make()\nmutate()\n"
        + WALK_M),
    "default-snapshotted-after-one-branch-joins-both-values": (
        "a = make()\nif c:\n    a = w\ndef mutate(x=a):\n    global m\n    m = x\n"
        "a = make()\nmutate()\n"
        + WALK_M),
    "starred-positional-may-land-in-a-parameter": (
        "m = make()\ndef mutate(x, y):\n    global m\n    m = y\nmutate(*rest, w)\n"
        + WALK_M),
    "starred-positional-may-land-in-the-only-parameter": (
        MUTATE_GLOBAL + "mutate(*rest, w)\n" + WALK_M),
    "positional-before-a-star-binds-exactly": (
        "m = make()\ndef mutate(x, y):\n    global m\n    m = x\nmutate(w, *rest)\n"
        + WALK_M),
    "double-star-keeps-an-explicit-keyword": (MUTATE_GLOBAL + "mutate(x=w, **options)\n" + WALK_M),
    "double-star-leaves-a-raw-default-feasible": (
        MUTATE_GLOBAL_RAW_DEFAULT
        + "mutate(**options)\n"
        + WALK_M),
    "star-leaves-a-raw-default-feasible": (MUTATE_GLOBAL_RAW_DEFAULT + "mutate(*rest)\n" + WALK_M),
    "argument-is-resolved-in-the-callers-scope-for-a-cross-scope-call": (
        MUTATE_GLOBAL
        + "def outer():\n    v = w\n    mutate(v)\n    m.bounded_walk(root)\n"),
    # 6hfRG3xrpw3Pm5gG: a helper's effect starts from the state it was called in, and only what
    # every feasible exit does to a declared name is a clear
    "conditional-assignment-keeps-the-raw-alias-on-the-false-path": (
        "m = w\ndef maybe_clear():\n    global m\n    if c:\n        m = make()\n"
        "maybe_clear()\n"
        + WALK_M),
    "early-return-keeps-the-raw-alias": (
        "m = w\ndef helper():\n    global m\n    if c:\n        return\n    m = make()\n"
        "helper()\n"
        + WALK_M),
    "loop-that-may-skip-its-assignment-keeps-the-raw-alias": (
        "m = w\ndef helper():\n    global m\n    for item in items:\n        m = make()\n"
        "helper()\n"
        + WALK_M),
    "composed-conditional-clear-keeps-the-raw-alias": (
        "m = w\ndef maybe_clear():\n    global m\n    if c:\n        m = make()\n"
        "def forward():\n    maybe_clear()\nforward()\n"
        + WALK_M),
    "declared-but-unwritten-name-keeps-the-raw-alias": (
        "m = w\ndef peek():\n    global m\n    return 1\npeek()\n"
        + WALK_M),
    "nested-helper-write-reaches-the-module-through-its-caller": (
        "m = make()\ndef outer():\n    def mutate():\n        global m\n        m = w\n"
        "    mutate()\nouter()\n"
        + WALK_M),
    # 6hfRG3vGHf7Jp22G: a `global` write belongs to the module and a `nonlocal` one to the nearest
    # enclosing function that binds the name; each reaches only the scopes that see that variable
    "cross-scope-global-write-reaches-a-free-reader": (
        "m = make()\ndef mutate():\n    global m\n    m = w\ndef outer():\n    mutate()\n"
        "    m.bounded_walk(root)\n"),
    "cross-scope-global-write-reaches-a-global-declaring-caller": (
        "m = make()\ndef mutate():\n    global m\n    m = w\ndef outer():\n    global m\n"
        "    mutate()\n    m.bounded_walk(root)\n"),
    "global-write-survives-a-callers-later-local-rebind": (
        "m = make()\ndef outer():\n    m = make()\n    def mutate(x):\n        global m\n"
        "        m = x\n    mutate(w)\n    m = make()\nouter()\n"
        + WALK_M),
    "global-write-through-a-shadowing-caller-reaches-the-module": (
        "m = make()\ndef outer():\n    m = make()\n    def mutate(x):\n        global m\n"
        "        m = x\n    mutate(w)\nouter()\n"
        + WALK_M),
    "global-write-through-a-shadowing-cross-scope-caller-reaches-the-module": (
        "m = make()\ndef mutate():\n    global m\n    m = w\ndef outer():\n    m = make()\n"
        "    mutate()\nouter()\n"
        + WALK_M),
    "nonlocal-write-reaches-the-owners-later-read": (
        "def outer():\n    m = make()\n    def mutate():\n        nonlocal m\n"
        "        m = w\n    mutate()\n    m.bounded_walk(root)\n"),
    "nonlocal-write-through-an-intermediate-reaches-the-owner": (
        "def outer():\n    m = make()\n    def middle():\n        def inner():\n"
        "            nonlocal m\n            m = w\n        inner()\n    middle()\n"
        "    m.bounded_walk(root)\nouter()\n"),
    "cross-scope-nonlocal-write-reaches-a-deeper-free-reader": (
        "def outer():\n    m = make()\n    def mutate():\n        nonlocal m\n"
        "        m = w\n    def middle():\n        mutate()\n        m.bounded_walk(root)\n"
        "    middle()\nouter()\n"),
    "global-write-in-a-nested-helper-reaches-a-free-reader-below-it": (
        "m = make()\ndef outer():\n    def mutate():\n        global m\n        m = w\n"
        "    mutate()\n    m.bounded_walk(root)\nouter()\n"),
    # 6hfRG3rM9F8mQjmp: a `def` run again (here in a loop) keeps what its default held on an
    # earlier run, and a star never supplies a parameter for certain, so its default stays feasible
    "default-defined-in-a-loop-keeps-what-it-held-on-an-earlier-run": (
        "import algua.primitives as p\na = w\nfor item in items:\n    def mutate(x=a):\n"
        "        global m\n        m = x\n    a = p\nmutate()\n"
        + WALK_M),
    "a-star-does-not-supply-a-defaulted-parameter": (
        "m = make()\ndef mutate(x, y=w):\n    global m\n    m = y\nmutate(make(), *rest)\n"
        + WALK_M),
    # 6hfRG3xrpw3Pm5gG, 6hfRG3vGHf7Jp22G: a helper nested in a function is seeded with what it and
    # the helpers beside it read (defaults included), and what a shadowing caller cannot show
    # still outlives it as a write that may have happened, without clearing what was there
    "nested-helper-default-reads-a-name-the-caller-must-seed": (
        "m = make()\ndef outer():\n    def h1(*, x=w):\n        global m\n        m = x\n"
        "    h1()\nouter()\n"
        + WALK_M),
    "nested-forwarder-composes-a-sibling-helper-of-the-same-function": (
        "m = make()\ndef outer():\n    def h1():\n        global m\n        m = w\n"
        "    def middle():\n        h1()\n    middle()\nouter()\n"
        + WALK_M),
    "a-write-through-a-shadowing-caller-cannot-clear-what-the-module-held": (
        "import algua.primitives as p\nm = w\ndef outer():\n    m = make()\n"
        "    def mutate():\n        global m\n        if c:\n            m = p\n"
        "    mutate()\nouter()\n"
        + WALK_M),
    # 6hfRG3rM9F8mQjmp: a `def` visited by several paths that all reach the call (here, a
    # `finally` a `break` leaves through, and one it falls out of) keeps the default of each
    "a-def-in-a-finally-keeps-the-default-of-every-path-that-reaches-the-call": (
        "import algua.primitives as p\nm = make()\na = w\nfor item in items:\n    try:\n"
        "        if c:\n            break\n        a = p\n    finally:\n"
        "        def mutate(x=a):\n            global m\n            m = x\nmutate()\n"
        + WALK_M),
    "global-write-through-two-shadowing-levels-reaches-the-module": (
        "m = make()\ndef outer():\n    m = make()\n    def middle():\n        m = make()\n"
        "        def inner():\n            global m\n            m = w\n        inner()\n"
        "    middle()\nouter()\n"
        + WALK_M),
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
    # 6hfQqGxWrc4QpJqp: the true, unshadowed `next` builtin's own single (iterable) argument is
    # genuinely only advanced, never retained
    "generator-as-next-sole-argument-stays-non-retaining": (
        "m = make()\ndef gen():\n    yield m.bounded_walk(root)\n"
        "[next(gen()) for _ in xs]\nm = w\nm = make()\n"),
    # 6hfQqH3pfGW6W7Vp / 6hfQqGxfgHcvPFXp: a cross-scope call resolves the callee from its own
    # defining lexical environment; an intermediate scope's own local (here `outer`'s, forwarded
    # transparently through `inner`, which has no shadow of its own) is excluded, since it is
    # never what the callee's free reference -- module-level here -- resolves to
    "cross-scope-call-does-not-leak-an-intermediate-scopes-shadow": (
        "m = make()\ndef gen():\n    yield m.bounded_walk(root)\n"
        "def outer():\n    m = w\n    def inner():\n        next(gen())\n    inner()\n"
        "outer()\nm = make()\n"),
    # 6hfQqH4P4QC6r8Rp: an argument bound to an unrelated parameter leaves the caller state safe
    "helper-effect-argument-binding-leaves-an-unrelated-value-safe": (
        "m = make()\ndef mutate(x):\n    global m\n    m = x\n"
        "mutate(make())\nm.bounded_walk(root)\n"),
    # 6hfQqGxXJRvpH62G: a helper that only calls a safe, non-effect-having function has nothing
    # to compose and leaves the caller state untouched
    "helper-composing-a-safe-callee-leaves-the-caller-state-untouched": (
        "m = make()\ndef safe_helper():\n    return make()\n"
        "def caller():\n    safe_helper()\n"
        "caller()\nm.bounded_walk(root)\n"),
    # 6hfQqGxXJRvpH62G: a helper that definitely reassigns a declared global to something not
    # followed clears the stale raw alias, rather than retaining it unexamined
    "helper-effect-propagates-a-definite-clear-of-a-stale-alias": (
        "m = w\ndef clear_it():\n    global m\n    m = make()\n"
        "clear_it()\nm.bounded_walk(root)\n"),
    # 6hfRG3rqrGgFGWfp: a name the source does not bind, or binds where the call cannot see it, or
    # only binds after the comprehension has run, leaves the true builtin exempt
    "an-unrelated-assignment-does-not-shadow-bool": (
        LAZY_READS_M + "flag = stash\n[bool(gen()) for _ in xs]\n" + RAW_THEN_SAFE),
    "shadowing-any-leaves-bool-exempt": (
        LAZY_READS_M + "any = stash\n[bool(gen()) for _ in xs]\n" + RAW_THEN_SAFE),
    "a-sibling-functions-parameter-does-not-shadow-bool": (
        LAZY_READS_M + "def other(bool):\n    return bool\n[bool(gen()) for _ in xs]\n"
        + RAW_THEN_SAFE),
    "a-shadow-bound-after-the-comprehension-is-not-retroactive": (
        LAZY_READS_M + "[bool(gen()) for _ in xs]\nbool = stash\n" + RAW_THEN_SAFE),
    "a-differently-named-comprehension-target-keeps-bool-exempt": (
        LAZY_READS_M + "[bool(gen()) for item in fns]\n" + RAW_THEN_SAFE),
    "a-sibling-comprehension-target-does-not-shadow-bool": (
        LAZY_READS_M + "[x for bool in fns]\n[bool(gen()) for _ in xs]\n" + RAW_THEN_SAFE),
    "a-helper-local-binding-does-not-shadow-module-bool": (
        LAZY_READS_M + "def helper():\n    bool = stash\n[bool(gen()) for _ in xs]\n"
        + RAW_THEN_SAFE),
    "a-helper-declaring-global-without-binding-does-not-shadow-bool": (
        LAZY_READS_M + "def peek():\n    global bool\n    return 1\npeek()\n"
        "[bool(gen()) for _ in xs]\n" + RAW_THEN_SAFE),
    # 6hfRG42g8WcVmw8G: the rebind happens on a path with an object of its own; the object made on
    # the other branch is not alive there and never sees it
    "a-rebind-on-a-sibling-path-with-its-own-object-does-not-reach-this-branchs-object": (
        "m = make()\ndef gen1():\n    yield m.bounded_walk(root)\ndef gen2():\n    yield None\n"
        "if c:\n    a = gen1()\nelse:\n    b = gen2()\n    m = w\nm = make()\n"),
    "a-comprehension-target-does-not-shadow-a-call-in-its-own-first-iterable": (
        "m = make()\ndef gen():\n    yield m.bounded_walk(root)\n"
        "[bool for bool in [bool(gen())]]\nm = w\nm = make()\n"),
    # 6hfRG3rqrGgFGWfp: a class comprehension's element runs in the enclosing scope, so a
    # class-level rebinding does not shadow it (Python resolves the element there, not in the class)
    "a-class-level-rebinding-of-bool-does-not-shadow-a-comprehension-elements-bool": (
        "m = make()\ndef gen():\n    yield m.bounded_walk(root)\nclass K:\n"
        "    bool = stash\n    flags = [bool(gen()) for _ in xs]\nm = w\nm = make()\n"),
    # the callee is defined outside the caller, so its `bool` is not the caller's parameter
    "a-callers-parameter-shadow-does-not-reach-a-module-level-callee": (
        "def worker():\n    n = make()\n    def gen():\n        yield n.bounded_walk(root)\n"
        "    [bool(gen()) for _ in xs]\n    n = w\n    n = make()\n"
        "def outer(bool):\n    worker()\n"),
    # 6hfRG3rM9F8mQjmp: only feasible operands, definition-time defaults and unsupplied parameters
    # can supply a raw value
    "statically-false-conditional-branch-is-not-a-feasible-argument": (
        MUTATE_GLOBAL
        + "mutate(w if 0 else make())\n"
        + WALK_M),
    "statically-true-conditional-hides-the-other-branch": (
        MUTATE_GLOBAL
        + "mutate(make() if 1 else w)\n"
        + WALK_M),
    "conditional-argument-of-two-safe-values": (
        MUTATE_GLOBAL
        + "mutate(make() if c else other())\n"
        + WALK_M),
    "default-is-snapshotted-when-the-function-is-defined-safe": (
        "a = make()\ndef mutate(x=a):\n    global m\n    m = x\na = w\nmutate()\n"
        + WALK_M),
    "keyword-only-default-is-snapshotted-when-defined-safe": (
        "a = make()\ndef mutate(*, x=a):\n    global m\n    m = x\na = w\nmutate()\n"
        + WALK_M),
    "supplied-safe-positional-does-not-fall-back-to-a-raw-default": (
        MUTATE_GLOBAL_RAW_DEFAULT
        + "mutate(make())\n"
        + WALK_M),
    "supplied-safe-keyword-does-not-fall-back-to-a-raw-default": (
        MUTATE_GLOBAL_RAW_DEFAULT
        + "mutate(x=make())\n"
        + WALK_M),
    "supplied-safe-keyword-only-does-not-fall-back-to-a-raw-default": (
        "m = make()\ndef mutate(*, x=w):\n    global m\n    m = x\nmutate(x=make())\n"
        + WALK_M),
    "supplied-unresolved-name-does-not-fall-back-to-a-raw-default": (
        MUTATE_GLOBAL_RAW_DEFAULT
        + "mutate(unknown)\n"
        + WALK_M),
    "positional-before-a-star-does-not-bind-a-later-parameter": (
        "m = make()\ndef mutate(x, y):\n    global m\n    m = y\nmutate(w, *rest)\n"
        + WALK_M),
    "supplied-positional-before-a-star-blocks-the-raw-default": (
        MUTATE_GLOBAL_RAW_DEFAULT
        + "mutate(make(), *rest)\n"
        + WALK_M),
    "supplied-keyword-beside-a-double-star-blocks-the-raw-default": (
        MUTATE_GLOBAL_RAW_DEFAULT
        + "mutate(x=make(), **options)\n"
        + WALK_M),
    # 6hfRG3xrpw3Pm5gG: a clear needs every feasible exit to clear
    "infeasible-early-return-does-not-keep-the-raw-alias": (
        "m = w\ndef helper():\n    global m\n    if 0:\n        return\n    m = make()\n"
        "helper()\n"
        + WALK_M),
    "clear-in-both-branches-clears-the-raw-alias": (
        "m = w\ndef clear_it():\n    global m\n    if c:\n        m = make()\n    else:\n"
        "        m = other()\nclear_it()\n"
        + WALK_M),
    "composed-definite-clear-clears-the-raw-alias": (
        "m = w\ndef clear_it():\n    global m\n    m = make()\ndef forward():\n"
        "    clear_it()\nforward()\n"
        + WALK_M),
    "delete-of-a-declared-name-clears-the-raw-alias": (
        "m = w\ndef drop():\n    global m\n    del m\ndrop()\n"
        + WALK_M),
    # 6hfRG3vGHf7Jp22G: a write to a variable a scope does not see never reaches that scope
    "global-write-leaves-the-callers-own-local-alone": (
        "m = make()\ndef outer():\n    m = make()\n    def mutate():\n        global m\n"
        "        m = w\n    mutate()\n    m.bounded_walk(root)\n"),
    "nonlocal-write-stays-in-its-owning-function": (
        "m = make()\ndef outer():\n    m = make()\n    def mutate(x):\n        nonlocal m\n"
        "        m = x\n    mutate(w)\nouter()\n"
        + WALK_M),
    "nonlocal-write-owned-by-an-intermediate-scope-does-not-escape-it": (
        "m = make()\ndef outer():\n    def middle():\n        m = make()\n"
        "        def inner(x):\n            nonlocal m\n            m = x\n"
        "        inner(w)\n    middle()\n    m.bounded_walk(root)\nouter()\n"),
    "caller-local-shadows-a-cross-scope-global-write": (
        "m = make()\ndef mutate():\n    global m\n    m = w\ndef outer():\n    m = make()\n"
        "    mutate()\n    m.bounded_walk(root)\nouter()\n"),
    "cross-scope-nonlocal-write-leaves-a-deeper-local-alone": (
        "def outer():\n    m = make()\n    def mutate():\n        nonlocal m\n"
        "        m = w\n    def middle():\n        m = make()\n        mutate()\n"
        "        m.bounded_walk(root)\n    middle()\nouter()\n"),
    # 6hfRG3vGHf7Jp22G: a helper's declared name is seeded from the caller only where the
    # caller's own name for it is the same variable, and a variable a function owns is not the
    # one a recursive call of it (a new frame) writes for the frame that called it
    "a-helper-reads-the-modules-variable-not-the-callers-local-of-the-same-name": (
        "m = make()\nk = make()\ndef outer():\n    m = w\n    def copy():\n"
        "        global m, k\n        k = m\n    copy()\nouter()\nk.bounded_walk(root)\n"),
    "a-recursive-call-does-not-write-the-enclosing-frames-local": (
        "def outer(x, depth):\n    m = make()\n    def mutate():\n        nonlocal m\n"
        "        m = x\n    def middle():\n        if depth:\n"
        "            outer(w, depth - 1)\n        m.bounded_walk(root)\n    middle()\n"
        "    mutate()\n"),
    # 6hfRG3rM9F8mQjmp: a literal operand that decides a Boolean argument hides the ones after it
    "statically-false-and-operand-hides-the-raw-alias": (
        MUTATE_GLOBAL
        + "mutate(0 and w)\n"
        + WALK_M),
    "statically-true-or-operand-hides-the-raw-alias": (MUTATE_GLOBAL + "mutate(1 or w)\n" + WALK_M),
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


def _interleaved_live_generators(count: int) -> str:
    """Like `_live_generators`, but rebinds the tracked alias right after every single creation,
    forcing a late-binding update -- and so a flatten of the alive set -- at every growing chain
    length, not once at the end: the adversarial pattern that exposes per-prefix retention."""
    made = "".join(
        f"items{index} = (m.bounded_walk(r) for r in roots)\nm = w\nm = make()\n"
        for index in range(count))
    return f"{IMPORT_W}m = make()\n{made}"


def test_alive_materialization_retains_no_per_prefix_cache() -> None:
    # 6hfQqH2m97ch7g8p: `_Alive.flattened()` used to cache the flattened frozenset on EVERY node
    # visited during one walk, not only the node actually asked for; flattening at every growing
    # chain length (as a rebind interleaved with every creation forces) then retained one
    # separate, ever-larger frozenset copy per length -- O(n) copies summing to O(n^2) bytes,
    # measured about 1.38 GB at 8,000 objects -- not the O(n) a persistent set actually needs.
    # Instrument actual traced memory (not a proxy or wall clock): near-linear growth (~4x for a
    # 4x object count, measured) passes; the O(n^2) caching this replaces measured ~10.6x at this
    # same (smaller, kept fast here) scale, clearly over the threshold below.
    def peak_bytes(count: int) -> int:
        gc.collect()
        tracemalloc.start()
        try:
            assert _direct_walk_uses(
                _interleaved_live_generators(count), "algua.registry.consumer")
        finally:
            _current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        return peak

    small = peak_bytes(500)
    large = peak_bytes(2_000)  # 4x the objects
    assert large <= 6 * small, (small, large)  # near-linear; O(n^2) caching measured ~10.6x here


def test_a_recorded_rebind_reaches_the_objects_of_an_alive_set_and_no_others() -> None:
    # 6hfRG42g8WcVmw8G: what is recorded on an alive set reaches every object it includes, held as
    # nodes or as an ordinary set, except a name that object reads from a closure -- and never an
    # object made after it, whose own node merely includes the set
    walk = _WalkReferences("algua.registry.consumer")
    function = ast.parse("def gen():\n    yield 1\n").body[0]
    assert isinstance(function, ast.FunctionDef)
    closures = {"plain": frozenset(), "node": frozenset(), "closed": frozenset({"m"}),
                "later": frozenset()}
    for oid, closure in closures.items():
        walk._lazy[oid] = (function, closure, function)
        walk._seen[oid] = {}
    alive = _alive_add(_alive_add(frozenset({"plain"}), "node"), "closed")
    walk._observe("m", frozenset({"raw"}), {ALIVE: alive})
    _alive_add(alive, "later")  # made after the rebind: its node includes `alive`, not the reverse
    walk._flush()
    raw = {"m": frozenset({"raw"})}
    assert walk._seen == {"plain": raw, "node": raw, "closed": {}, "later": {}}
    assert walk._events == {}  # applied once, then forgotten


def test_the_alive_order_lists_a_shared_node_once_after_everything_that_includes_it() -> None:
    shared = _alive_add(None, "shared")
    left, right = _alive_add(shared, "left"), _alive_add(shared, "right")
    joined = _alive_union(left, right)
    assert isinstance(joined, _Alive)
    ids = [id(node) for node in _alive_order([joined, left])]  # `left` again, through a second root
    assert sorted(ids) == sorted({id(joined), id(left), id(right), id(shared)})
    assert ids.index(id(joined)) < ids.index(id(left)) < ids.index(id(shared))
    assert ids.index(id(joined)) < ids.index(id(right)) < ids.index(id(shared))


@pytest.mark.usefixtures("default_recursion_limit")
def test_a_rebind_reaches_a_chain_of_live_objects_longer_than_the_recursion_limit() -> None:
    # the pass that hands a recorded rebind down an alive set walks it with an explicit stack: a
    # chain is as deep as the objects made in a row, so recursing would fail well before 1,500
    assert _direct_walk_uses(_live_generators(1_500), "algua.registry.consumer")


def _live_generators_beside_comprehensions(count: int) -> str:
    """``count`` lazy objects, each followed by a comprehension that makes none of its own, while
    every earlier object is still alive."""
    made = "".join(
        f"items{index} = (m.bounded_walk(r) for r in roots)\n[y for y in ys if y]\n"
        for index in range(count))
    return f"{IMPORT_W}m = make()\n{made}m = w\nm = make()\n"


def _live_generators_made_by_comprehensions(count: int) -> str:
    """``count`` comprehensions, each keeping a lazy object in its result, while every earlier
    object is still alive."""
    made = "".join(f"items{index} = [gen() for _ in ys]\n" for index in range(count))
    return (
        f"{IMPORT_W}m = make()\ndef gen():\n    yield m.bounded_walk(root)\n{made}"
        "m = w\nm = make()\n")


def _loops_beside_live_generators(count: int) -> str:
    """``count`` lazy objects alive at once, then ``count`` loops that make none: every loop's
    fixpoint check compares the alive set with itself."""
    made = "".join(
        f"items{index} = (m.bounded_walk(r) for r in roots)\n" for index in range(count))
    loops = "for item in items:\n    pass\n" * count
    return f"{IMPORT_W}m = make()\n{made}{loops}m = w\nm = make()\n"


ALIVE_WORK_SHAPES = {
    "a-rebind-after-every-creation": _interleaved_live_generators,
    "a-comprehension-after-every-creation": _live_generators_beside_comprehensions,
    "a-comprehension-keeping-every-creation": _live_generators_made_by_comprehensions,
    "a-loop-that-makes-nothing-after-every-creation": _loops_beside_live_generators,
}


@pytest.mark.parametrize(
    "shape", ALIVE_WORK_SHAPES.values(), ids=ALIVE_WORK_SHAPES.keys())
def test_alive_state_work_stays_near_linear_in_the_objects_alive(
    monkeypatch: pytest.MonkeyPatch, shape: Callable[[int], str],
) -> None:
    # 6hfRG42g8WcVmw8G: the straight-line shape above (every object made, then ONE rebind) cannot
    # see this. With a rebind after EVERY creation, each rebind used to visit every object alive so
    # far -- a flatten of the alive set plus one view update per object -- so n creations cost
    # n(n+1)/2 object visits: 31,375 / 125,250 / 500,500 / 2,001,000 at 250 / 500 / 1,000 / 2,000
    # (exactly x4 per doubling), against 250 / 500 / 1,000 / 2,000 for the straight-line shape.
    # Every comprehension likewise flattened the whole alive set on entry and on each filter.
    # Counted here, not timed: each alive node built, each id an iteration yields or a flatten
    # returns, and the work of the pass that hands recorded rebinds to the objects that saw them.
    work = {"steps": 0}
    real_add, real_union = _alive_add, _alive_union
    real_iter, real_flatten = _Alive.__iter__, _Alive.flattened
    real_order, real_views, real_deliver = _alive_order, _union_views, _WalkReferences._deliver

    def counting_add(alive: frozenset[str] | _Alive | None, oid: str) -> _Alive:
        work["steps"] += 1
        return real_add(alive, oid)

    def counting_union(
        a: frozenset[str] | _Alive | None, b: frozenset[str] | _Alive | None,
    ) -> frozenset[str] | _Alive | None:
        work["steps"] += 1
        return real_union(a, b)

    def counting_iter(self: _Alive) -> Iterator[str]:
        for oid in real_iter(self):
            work["steps"] += 1
            yield oid

    def counting_flatten(self: _Alive) -> frozenset[str]:
        flat = real_flatten(self)
        work["steps"] += len(flat)
        return flat

    def counting_order(roots: Iterable[_Alive]) -> list[_Alive]:
        order = real_order(roots)
        work["steps"] += len(order)  # each node the pass visits
        return order

    def counting_views(a: Views, b: Views) -> Views:
        work["steps"] += 1 + len(a) + len(b)  # each name a merge of recorded rebinds walks
        return real_views(a, b)

    def counting_deliver(self: _WalkReferences, oid: str, view: Views) -> None:
        work["steps"] += 1 + len(view)  # each object handed what it saw, and each name of it
        real_deliver(self, oid, view)

    module = sys.modules[__name__]
    monkeypatch.setattr(module, "_alive_add", counting_add)
    monkeypatch.setattr(module, "_alive_union", counting_union)
    monkeypatch.setattr(module, "_alive_order", counting_order)
    monkeypatch.setattr(module, "_union_views", counting_views)
    monkeypatch.setattr(_Alive, "__iter__", counting_iter)
    monkeypatch.setattr(_Alive, "flattened", counting_flatten)
    monkeypatch.setattr(_WalkReferences, "_deliver", counting_deliver)
    totals = {}
    for count in (100, 200, 400, 800):
        work["steps"] = 0
        assert _direct_walk_uses(shape(count), "algua.registry.consumer")
        totals[count] = work["steps"]
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


def _same_name_local_accumulation(count: int) -> str:
    """One shared generator-returning factory `g`, called ``count`` times, each after a
    module-level `x` -- also `g`'s own read-after-write local, never a parameter -- is set to a
    distinct target (a different dummy function's own marker, so each is genuinely different, not
    an alias of the same module). `g`'s own first statement always overwrites `x` before it is
    read, so its actual behavior never depends on what the caller provided; only a cache key
    projected onto every name `g` owns, not only its parameters, sees that and reuses the one
    answer."""
    dummies = "".join(f"def dummy{i}():\n    pass\n" for i in range(count))
    calls = "".join(f"x = dummy{i}\nitems{i} = g()\n" for i in range(count))
    return (
        f"{IMPORT_W}m = make()\n{dummies}"
        "def gen():\n    yield m.bounded_walk(root)\n"
        "def g():\n    x = w\n    return gen() if x else None\n"
        f"{calls}"
    )


def test_summary_cache_excludes_a_functions_own_local_not_only_its_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 6hfQqH4rMMvxq9JG: a cache key projected only onto a function's parameters still varies with
    # whatever a caller happens to provide for a same-named local the function immediately
    # overwrites before using -- `g`'s own `x`, read-after-write, never actually depends on the
    # caller. Instrument the deterministic NUMBER of cache misses (`_summarize` calls, keyed by
    # (function, entry)): before this fix each of `count` differently-preceded calls to the SAME
    # `g` missed the cache (misses == count); projecting the key onto every name `g` owns, not
    # only its parameters, collapses them to a single, reused answer (misses == 1), regardless of
    # `count` -- not merely near-linear in it.
    misses = 0
    real_summarize = _WalkReferences._summarize

    def counting_summarize(
        self: _WalkReferences, function: Function, entry: Bindings,
    ) -> list[Returned]:
        nonlocal misses
        misses += 1
        return real_summarize(self, function, entry)

    monkeypatch.setattr(_WalkReferences, "_summarize", counting_summarize)
    for count in (10, 50, 200):
        misses = 0
        assert _direct_walk_uses(
            _same_name_local_accumulation(count), "algua.registry.consumer") == []
        assert misses == 1, (count, misses)
