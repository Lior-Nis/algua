"""AST defense for the decision/execution data wall (issue #277).

The import-linter contracts in pyproject.toml forbid every decision/execution lane from
importing algua.data (the hindsight/look-ahead lane) — but those only run under
`lint-imports`, not `pytest`. This module re-verifies the SAME property with a static AST
walk so a developer running pytest alone still catches a fresh `from algua.data import ...`
(or a disguised relative/dynamic variant) added to any walled lane.

Consolidates the previously duplicated per-lane scanners (test_fundamentals_wall.py,
test_news_wall.py) into ONE canonical scanner over all six lanes, so the guards cannot drift.
"""

import ast
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]

# Every lane the data wall protects. Package dirs are relative to the repo root.
WALLED_LANES = [
    "algua/strategies",
    "algua/contracts",
    "algua/live",
    "algua/execution",
    "algua/backtest",
    "algua/features",
]

# Dynamic-import call targets whose FIRST positional literal string arg we inspect.
_DYNAMIC_IMPORT_FUNCS = {"import_module", "__import__"}


def _is_data_target(module: str) -> bool:
    """True iff a fully-qualified module name IS algua.data or lives under it.

    Precise on purpose: `algua.database` (a hypothetical sibling) must NOT match.
    """
    return module == "algua.data" or module.startswith("algua.data.")


def _file_package(path: pathlib.Path) -> str:
    """Dotted package that a relative import in `path` resolves against.

    e.g. algua/backtest/engine.py -> 'algua.backtest';
         algua/backtest/__init__.py -> 'algua.backtest'.
    """
    rel = path.relative_to(REPO).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] != "__init__":
        parts = parts[:-1]  # a module resolves relative to its containing package
    else:
        parts = parts[:-1]  # __init__ IS the package's module; parent is the package
    return ".".join(parts)


def _resolve_relative(package: str, level: int, module: str | None) -> str | None:
    """Resolve a (possibly relative) `from ... import` base to an absolute module.

    Returns None if the level walks above the top package (an unresolvable/invalid import
    we leave for the interpreter to reject).
    """
    if level == 0:
        return module or ""
    parts = package.split(".") if package else []
    # level 1 == the package itself; each extra level strips one more component.
    keep = len(parts) - (level - 1)
    if keep <= 0:  # walked above the top-level package -> unresolvable
        return None
    base = ".".join(parts[:keep])
    if module:
        return f"{base}.{module}" if base else module
    return base


def _dynamic_func_name(node: ast.Call) -> str | None:
    """The called name if this is an importlib.import_module / __import__ call, else None."""
    func = node.func
    name = func.attr if isinstance(func, ast.Attribute) else (
        func.id if isinstance(func, ast.Name) else None
    )
    return name if name in _DYNAMIC_IMPORT_FUNCS else None


def _import_module_anchor(node: ast.Call, file_package: str) -> str | None:
    """The package `importlib.import_module(name, package=...)` resolves a RELATIVE name
    against: a literal package arg verbatim, or the file's package when the arg is the
    `__package__` builtin. None when absent/non-literal (a relative name is then unresolvable
    and raises at runtime — nothing to flag)."""
    arg = node.args[1] if len(node.args) >= 2 else next(
        (k.value for k in node.keywords if k.arg == "package"), None
    )
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return arg.value
    if isinstance(arg, ast.Name) and arg.id == "__package__":
        return file_package
    return None


def _call_reaches_data(node: ast.Call, package: str) -> bool:
    """True iff a literal-arg importlib.import_module / __import__ call targets algua.data.

    Absolute literal names are checked directly; a RELATIVE name is resolved per importlib
    semantics (import_module only — against its `package` arg / `__package__`)."""
    fname = _dynamic_func_name(node)
    if fname is None or not node.args:
        return False
    first = node.args[0]
    if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
        return False
    name = first.value
    if not name.startswith("."):
        return _is_data_target(name)
    if fname != "import_module":  # __import__ uses a numeric `level`, not a dotted-name prefix
        return False
    anchor = _import_module_anchor(node, package)
    if anchor is None:
        return False
    level = len(name) - len(name.lstrip("."))
    resolved = _resolve_relative(anchor, level, name.lstrip(".") or None)
    return resolved is not None and _is_data_target(resolved)


def _node_reaches_data(node: ast.AST, package: str) -> bool:
    """True iff a single AST node reaches algua.data: an absolute/relative `from ... import`,
    a plain `import`, or a literal-arg importlib.import_module / __import__ call.

    `package` is the dotted package the file lives in, used to resolve relative imports.
    """
    if isinstance(node, ast.ImportFrom):
        base = _resolve_relative(package, node.level, node.module)
        return base is not None and (
            _is_data_target(base)
            # covers `from algua import data` / `from .. import data`
            or any(_is_data_target(f"{base}.{a.name}" if base else a.name)
                   for a in node.names)
        )
    if isinstance(node, ast.Import):
        return any(_is_data_target(a.name) for a in node.names)
    if isinstance(node, ast.Call):
        return _call_reaches_data(node, package)
    return False


def _tree_reaches_data(tree: ast.AST, package: str) -> bool:
    return any(_node_reaches_data(n, package) for n in ast.walk(tree))


def _offenders_in(pkg: str) -> list[str]:
    """Files under `pkg` that statically reach algua.data (absolute, relative, or a
    literal-arg dynamic import)."""
    offenders: list[str] = []
    for path in sorted((REPO / pkg).rglob("*.py")):
        if _tree_reaches_data(ast.parse(path.read_text()), _file_package(path)):
            offenders.append(str(path.relative_to(REPO)))
    return offenders


def test_no_data_lane_reference_in_any_walled_lane():
    """Fail-closed: no module in any walled lane may statically reach algua.data."""
    offenders = {pkg: _offenders_in(pkg) for pkg in WALLED_LANES}
    bad = {pkg: files for pkg, files in offenders.items() if files}
    assert not bad, f"walled lanes reach algua.data: {bad}"


def test_scanner_flags_each_data_reference_shape():
    """Guard the guard: the scanner must catch every import shape it claims to (absolute,
    `from algua import data`, relative, and literal dynamic) and must NOT false-positive on
    a sibling like algua.database or a non-literal importlib call."""
    should_flag = [
        "from algua.data import hindsight",
        "from algua.data.store import read_news",
        "import algua.data",
        "import algua.data.store as s",
        "from algua import data",
        "from ..data import store",          # relative, resolves to algua.data
        "from .. import data",               # relative bare-name
        "import importlib; importlib.import_module('algua.data.store')",
        "__import__('algua.data')",
        # relative dynamic import resolves against its package anchor
        "import importlib; importlib.import_module('..data.store', __package__)",
        "import importlib; importlib.import_module('.store', 'algua.data')",
    ]
    should_not_flag = [
        "import algua.database",             # sibling, not the data lane
        "from algua.contracts import types",
        "import importlib; importlib.import_module(f'{pkg}.{name}')",  # non-literal
        "x = 'algua.data'  # a bare string, e.g. in a docstring or list",
        # relative dynamic name with NO package anchor raises at runtime -> unresolvable
        "import importlib; importlib.import_module('..data.store')",
    ]
    # Anchor synthetic modules in a two-deep package so `from ..data ...` resolves to algua.data.
    package = "algua.backtest"

    def flags(src: str) -> bool:
        return _tree_reaches_data(ast.parse(src), package)

    for src in should_flag:
        assert flags(src), f"scanner missed a forbidden shape: {src!r}"
    for src in should_not_flag:
        assert not flags(src), f"scanner false-positived on: {src!r}"
