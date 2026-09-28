"""A consumer's own error stays primary when closing its walk also fails."""
from __future__ import annotations

import ast
import errno
import importlib.util
from collections.abc import Iterable
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
Function = ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda
# each name mapped to every walk-relevant target it may hold on some path to this point
Bindings = dict[str, frozenset[str]]


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
        into[name] = into.get(name, frozenset()) | targets


def _join(*states: Bindings) -> Bindings:
    joined: Bindings = {}
    for state in states:
        _merge(joined, state)
    return joined


def _irrefutable(case: ast.match_case) -> bool:
    return (
        isinstance(case.pattern, ast.MatchAs) and case.pattern.pattern is None
        and case.guard is None
    )


class _WalkReferences:
    """Statement-ordered, scope-aware resolution of every target a name may hold on some path.

    Followed statically: imports (absolute or relative, aliased or not, star), single-name plain or
    annotated assignments whose value resolves to an imported module, package, function or the
    `getattr` builtin, attribute access, and `getattr(module, "bounded_walk")`. Any other binding
    of a name (a parameter, an unresolved assignment, a loop, `with`, `except` or match target, a
    `def` or `class`, `del`) clears it, so unrelated locals are never flagged. A function body is
    resolved against its enclosing scope's final bindings (Python's late binding) minus its own
    parameters; class-level names do not leak into methods. Dynamic access (`__import__`,
    `importlib.import_module`, `vars`, `globals`, `exec`, computed attribute names) is out of
    scope: this is a guard against accidental or casual bypass, not arbitrary dynamic execution.

    Paths are joined conservatively. `if` branches start from the state before them; a `match`
    case also starts after earlier cases failed to match, and control falls through them unless
    the last is irrefutable. A loop body runs zero or more times, each iteration starting from
    the entry state or any state that reaches the next one (the body's end or a `continue`), and
    the loop leaves through its `else` or a `break`. A `try` handler may start from any point of
    the body (an `except*` handler also after an earlier one), `else` follows the body alone, and
    `finally` may start from any point of the whole statement. `return` and `raise` are not
    modeled, so a join may include a path that cannot run, which only ever adds a flag. Only
    targets on the way to the raw walk or `getattr` are kept, which keeps loop fixpoints finite.
    """

    def __init__(self, package: str) -> None:
        self._package = package
        self.uses: list[str] = []
        self._watchers: list[Bindings] = []  # every state reached inside an enclosing `try`
        self._loops: list[tuple[Bindings, Bindings]] = []  # each open loop's break/continue states

    def module(self, tree: ast.Module) -> list[str]:
        self._scope(tree.body, {"getattr": frozenset({BUILTIN_GETATTR})})
        return list(dict.fromkeys(self.uses))  # loop and `finally` bodies may be visited twice

    def _scope(self, body: list[ast.stmt], bindings: Bindings) -> None:
        deferred: list[Function] = []
        self._block(body, bindings, deferred)
        for function in deferred:
            local = {k: v for k, v in bindings.items() if k not in _parameters(function)}
            if isinstance(function, ast.Lambda):
                self._expression(function.body, local, deferred)
            else:
                self._scope(function.body, local)

    def _block(
        self, body: list[ast.stmt], bindings: Bindings, deferred: list[Function],
    ) -> None:
        for statement in body:
            self._statement(statement, bindings, deferred)
            for reached in self._watchers:
                _merge(reached, bindings)

    def _resolved(self, node: ast.expr | None, bindings: Bindings) -> frozenset[str]:
        dotted = None if node is None else _dotted(node)
        if dotted is None:
            return frozenset()
        head, _, rest = dotted.partition(".")
        # a name not bound here is a local or builtin name, not something imported
        targets = bindings.get(head, frozenset())
        return frozenset(f"{target}.{rest}" if rest else target for target in targets)

    def _bind(self, name: str, targets: Iterable[str], bindings: Bindings) -> None:
        kept = frozenset(target for target in targets if _relevant(target))
        if kept:
            bindings[name] = kept
        else:
            bindings.pop(name, None)

    def _propagate(self, name: str, value: ast.expr, bindings: Bindings) -> None:
        """Bind ``name`` to every target ``value`` may resolve to."""
        self._bind(name, self._resolved(value, bindings), bindings)

    def _unbind(self, target: ast.AST, bindings: Bindings) -> None:
        for name in _bound_names(target):
            bindings.pop(name, None)

    def _expression(
        self, node: ast.AST | None, bindings: Bindings, deferred: list[Function],
    ) -> None:
        if node is None:
            return
        if isinstance(node, ast.Lambda):
            for default in (*node.args.defaults, *node.args.kw_defaults):
                self._expression(default, bindings, deferred)
            deferred.append(node)
            return
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            local = dict(bindings)
            for generator in node.generators:
                self._expression(generator.iter, local, deferred)
                self._unbind(generator.target, local)
                for condition in generator.ifs:
                    self._expression(condition, local, deferred)
            parts = (node.key, node.value) if isinstance(node, ast.DictComp) else (node.elt,)
            for part in parts:
                self._expression(part, local, deferred)
            return
        if isinstance(node, ast.NamedExpr):
            self._expression(node.value, bindings, deferred)
            self._unbind(node.target, bindings)
            return
        if isinstance(node, ast.Attribute) and RAW_WALK in self._resolved(node, bindings):
            self.uses.append(f"line {node.lineno}: references {RAW_WALK}")
        if (
            isinstance(node, ast.Call) and BUILTIN_GETATTR in self._resolved(node.func, bindings)
            and len(node.args) >= 2 and WALK_MODULE in self._resolved(node.args[0], bindings)
            and isinstance(node.args[1], ast.Constant) and node.args[1].value == "bounded_walk"
        ):
            self.uses.append(f"line {node.lineno}: getattr of {RAW_WALK}")
        for child in ast.iter_child_nodes(node):
            self._expression(child, bindings, deferred)

    def _loop(
        self, node: ast.For | ast.AsyncFor | ast.While, bindings: Bindings,
        deferred: list[Function],
    ) -> None:
        if not isinstance(node, ast.While):
            self._expression(node.iter, bindings, deferred)
        breaks: Bindings = {}
        continues: Bindings = {}
        head = dict(bindings)  # every state an iteration may start from
        while True:
            tested = dict(head)
            if isinstance(node, ast.While):
                self._expression(node.test, tested, deferred)
            body = dict(tested)
            if not isinstance(node, ast.While):
                self._unbind(node.target, body)
            self._loops.append((breaks, continues))
            self._block(node.body, body, deferred)
            self._loops.pop()
            following = _join(head, body, continues)
            if following == head:
                break
            head = following
        self._block(node.orelse, tested, deferred)
        bindings.clear()
        bindings.update(_join(tested, breaks))

    def _try(
        self, node: ast.Try | ast.TryStar, bindings: Bindings, deferred: list[Function],
    ) -> None:
        anywhere, raised = dict(bindings), dict(bindings)
        self._watchers += [anywhere, raised]
        self._block(node.body, bindings, deferred)
        self._watchers.pop()
        ends: list[Bindings] = []
        for handler in node.handlers:
            # the `except*` handlers of one exception group can each run, in order
            state = _join(raised, *ends) if isinstance(node, ast.TryStar) else dict(raised)
            self._expression(handler.type, state, deferred)
            if handler.name:
                state.pop(handler.name, None)
            self._block(handler.body, state, deferred)
            if handler.name:
                state.pop(handler.name, None)  # Python deletes it as the handler ends
            ends.append(state)
        self._block(node.orelse, bindings, deferred)
        self._watchers.pop()
        for state in ends:
            _merge(bindings, state)
        if node.finalbody:
            self._block(node.finalbody, anywhere, deferred)  # leaving by an exception or return
            self._block(node.finalbody, bindings, deferred)

    def _match(self, node: ast.Match, bindings: Bindings, deferred: list[Function]) -> None:
        self._expression(node.subject, bindings, deferred)
        unmatched = dict(bindings)
        ends: list[Bindings] = []
        for case in node.cases:
            state = dict(unmatched)
            self._unbind(case.pattern, state)
            self._expression(case.guard, state, deferred)
            _merge(unmatched, state)  # a pattern or guard may fail after binding captures
            self._block(case.body, state, deferred)
            ends.append(state)
        if not _irrefutable(node.cases[-1]):
            ends.append(unmatched)
        bindings.clear()
        bindings.update(_join(*ends))

    def _statement(
        self, node: ast.stmt, bindings: Bindings, deferred: list[Function],
    ) -> None:
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
            bindings.pop(node.name, None)
            deferred.append(node)
        elif isinstance(node, ast.ClassDef):
            for part in (*node.decorator_list, *node.bases, *(k.value for k in node.keywords)):
                self._expression(part, bindings, deferred)
            watchers, self._watchers = self._watchers, []  # class-level names stay in the class
            self._block(node.body, dict(bindings), deferred)  # methods resolve late, outside
            self._watchers = watchers
            bindings.pop(node.name, None)
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
            self._loop(node, bindings, deferred)
        elif isinstance(node, (ast.Break, ast.Continue)) and self._loops:
            breaks, continues = self._loops[-1]
            _merge(breaks if isinstance(node, ast.Break) else continues, bindings)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                self._expression(item.context_expr, bindings, deferred)
                if item.optional_vars is not None:
                    self._unbind(item.optional_vars, bindings)
            self._block(node.body, bindings, deferred)
        elif isinstance(node, ast.If):
            self._expression(node.test, bindings, deferred)
            orelse = dict(bindings)
            self._block(node.body, bindings, deferred)
            self._block(node.orelse, orelse, deferred)
            _merge(bindings, orelse)
        elif isinstance(node, (ast.Try, ast.TryStar)):
            self._try(node, bindings, deferred)
        elif isinstance(node, ast.Match):
            self._match(node, bindings, deferred)
        elif isinstance(node, ast.Delete):
            for target in node.targets:
                self._unbind(target, bindings)
        else:
            self._expression(node, bindings, deferred)


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
}


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
    "class-body-binding-does-not-reach-a-handler": (
        "try:\n    class Store:\n        m = w\nexcept Exception:\n    m.bounded_walk(root)\n"),
}


@pytest.mark.parametrize(
    "source", NOT_REACHED_ON_ANY_PATH.values(), ids=NOT_REACHED_ON_ANY_PATH.keys())
def test_the_guard_does_not_flag_a_walk_no_feasible_path_reaches(source: str) -> None:
    assert _direct_walk_uses(IMPORT_W + source, "algua.registry.consumer") == []
