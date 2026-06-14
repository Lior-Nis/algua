# Factor Catalogue + Derived Lineage Implementation Plan (#140, slices A+C)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a discoverable in-code catalogue of pure factors plus derived strategy→factor lineage, so the agent reuses factors instead of reinventing and can answer "which strategies use this factor?".

**Architecture:** A pure `@factor` annotation in `algua/features/catalogue.py` stamps an immutable `FactorSpec` on each factor function; `load_all_factors()` scans feature modules for the stamp (defining-module only) and builds the registry transactionally. Lineage is derived on demand by intersecting each registered strategy's existing live-gate import closure with the catalogue (a thin `closure_module_names` wrapper over the closure `compute_artifact_hashes` already hashes). A `factor` CLI group exposes list/show/dependents/uses as JSON.

**Tech Stack:** Python 3.12, Typer (CLI), pytest, pandas. Pure-layer purity enforced by import-linter.

**Design spec:** `docs/superpowers/specs/2026-06-13-factor-catalogue-issue-140-design.md`

**Quality gate (run after every task):** `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`

---

## File Structure

- Create `algua/features/catalogue.py` — `FactorKind`, `FactorSpec`, `@factor`, registry + discovery + read API.
- Modify `algua/features/indicators.py` — decorate `momentum` + `zscore` (the seed).
- Modify `algua/registry/approvals.py` — add `closure_module_names(loaded)`.
- Create `algua/registry/lineage.py` — `factors_used_by`, `dependents_of`, `Dependents`.
- Create `algua/cli/factor_cmd.py` — the `factor` Typer group.
- Modify `algua/cli/main.py` — import `factor_cmd` to register the group.
- Modify `.claude/skills/author-a-strategy/SKILL.md` (+ `.codex` copy) — authoring guidance.
- Tests: `tests/test_features_catalogue.py`, `tests/test_registry_lineage.py`, `tests/test_cli_factor.py`, and additions to `tests/test_registry_approvals.py`.

---

## Task 1: FactorKind + FactorSpec + the `@factor` annotation

**Files:**
- Create: `algua/features/catalogue.py`
- Test: `tests/test_features_catalogue.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_features_catalogue.py
import inspect

import pytest

from algua.contracts.idea import DataCapability
from algua.features.catalogue import FactorKind, FactorSpec, factor


def test_factor_stamps_spec_and_returns_same_object():
    def raw(x):
        """One-line summary here. More detail ignored."""
        return x

    decorated = factor(kind=FactorKind.MOMENTUM, tags=["a", "b"])(raw)
    assert decorated is raw  # pure annotation, no wrapper
    spec = decorated.__factor_spec__
    assert isinstance(spec, FactorSpec)
    assert spec.name == "raw"
    assert spec.summary == "One-line summary here. More detail ignored."
    assert spec.kind is FactorKind.MOMENTUM
    assert spec.tags == ("a", "b")  # tuple, not list
    assert spec.data_needs == (DataCapability.OHLCV,)  # default
    assert spec.module == "tests.test_features_catalogue"
    assert spec.import_path == "tests.test_features_catalogue:test_factor_stamps_spec_and_returns_same_object.<locals>.raw"
    assert spec.signature == str(inspect.signature(raw))
    assert decorated(7) == 7  # behaviour unchanged


def test_explicit_overrides():
    @factor(name="z", summary="explicit", kind=FactorKind.VALUE,
            data_needs=[DataCapability.FUNDAMENTALS])
    def f(a, b):
        return a

    spec = f.__factor_spec__
    assert spec.name == "z"
    assert spec.summary == "explicit"
    assert spec.data_needs == (DataCapability.FUNDAMENTALS,)


def test_missing_summary_and_docstring_fails_closed():
    def nodoc(x):
        return x

    with pytest.raises(ValueError, match="summary"):
        factor()(nodoc)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_features_catalogue.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'algua.features.catalogue'`.

- [ ] **Step 3: Write minimal implementation**

```python
# algua/features/catalogue.py
from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from algua.contracts.idea import DataCapability


class FactorKind(StrEnum):
    """Controlled, deliberately minimal factor categories (OTHER is the escape hatch). Extend as
    real factors demand, not speculatively."""

    MOMENTUM = "momentum"
    REVERSION = "reversion"
    VALUE = "value"
    SENTIMENT = "sentiment"
    VOLATILITY = "volatility"
    QUALITY = "quality"
    OTHER = "other"


class FactorNotFound(LookupError):
    pass


@dataclass(frozen=True)
class FactorSpec:
    """Catalogued metadata for one pure factor. Collection fields are tuples so a registered spec
    cannot be mutated in place (a frozen dataclass does not stop list mutation)."""

    name: str
    summary: str
    kind: FactorKind
    tags: tuple[str, ...]
    data_needs: tuple[DataCapability, ...]
    import_path: str
    module: str
    signature: str
    doc: str | None


_SPEC_ATTR = "__factor_spec__"


def _first_nonempty_line(doc: str | None) -> str | None:
    if not doc:
        return None
    for line in doc.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return None


def factor(
    *,
    name: str | None = None,
    summary: str | None = None,
    tags: Iterable[str] = (),
    kind: FactorKind = FactorKind.OTHER,
    data_needs: Iterable[DataCapability] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Annotate a pure factor function with discoverability metadata. PURE: it stamps a FactorSpec
    on the function as ``__factor_spec__`` and returns the function UNCHANGED (no wrapper) so
    call semantics, ``inspect.getsource`` and the live-gate ``code_hash`` see the real function.
    It mutates no module global at import time — discovery (``load_all_factors``) scans for the
    stamp. ``data_needs`` states the factor's INPUT requirement, not current platform availability.
    """

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        resolved_name = name or fn.__name__
        doc = inspect.getdoc(fn)
        resolved_summary = summary or _first_nonempty_line(doc)
        if not resolved_summary:
            raise ValueError(
                f"factor {resolved_name!r} needs a summary (pass summary= or add a docstring)"
            )
        spec = FactorSpec(
            name=resolved_name,
            summary=resolved_summary,
            kind=kind,
            tags=tuple(tags),
            data_needs=tuple(data_needs) if data_needs is not None else (DataCapability.OHLCV,),
            import_path=f"{fn.__module__}:{fn.__qualname__}",
            module=fn.__module__,
            signature=str(inspect.signature(fn)),
            doc=doc,
        )
        setattr(fn, _SPEC_ATTR, spec)
        return fn

    return decorate
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_features_catalogue.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add algua/features/catalogue.py tests/test_features_catalogue.py
git commit -m "feat(140): @factor annotation + FactorSpec/FactorKind"
```

---

## Task 2: Transactional discovery + read API

**Files:**
- Modify: `algua/features/catalogue.py`
- Test: `tests/test_features_catalogue.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_features_catalogue.py`. These tests write temporary modules into the real `algua/features/` package directory (mirroring how `tests/test_strategy_loader.py` writes temp strategy modules) and always clean up.

```python
import sys
from pathlib import Path

import algua.features as _featpkg
from algua.features.catalogue import (
    FactorNotFound, all_factors, filter_factors, get_factor, load_all_factors,
)
from algua.features import catalogue as _cat


@pytest.fixture(autouse=True)
def _clean_registry():
    _cat._reset_registry()
    yield
    _cat._reset_registry()


def _write_feature_module(stem: str, body: str) -> Path:
    path = Path(_featpkg.__path__[0]) / f"{stem}.py"
    path.write_text(body)
    return path


def _drop_module(stem: str, path: Path) -> None:
    path.unlink(missing_ok=True)
    sys.modules.pop(f"algua.features.{stem}", None)


def test_discovers_decorated_factor_in_a_feature_module():
    path = _write_feature_module(
        "tmp_disc_mod",
        "from algua.features.catalogue import factor, FactorKind\n"
        "@factor(summary='tmp', kind=FactorKind.MOMENTUM, tags=['t'])\n"
        "def tmp_fac(x):\n"
        "    return x\n",
    )
    try:
        reg = load_all_factors()
        assert "tmp_fac" in reg
        assert reg["tmp_fac"].kind is FactorKind.MOMENTUM
    finally:
        _drop_module("tmp_disc_mod", path)


def test_reexport_is_not_double_registered():
    # A module that imports another module's catalogued factor must NOT re-register it.
    defn = _write_feature_module(
        "tmp_defn_mod",
        "from algua.features.catalogue import factor\n"
        "@factor(summary='defined here')\n"
        "def shared_fac(x):\n"
        "    return x\n",
    )
    reexp = _write_feature_module(
        "tmp_reexp_mod",
        "from algua.features.tmp_defn_mod import shared_fac  # re-export\n"
        "from algua.features.catalogue import factor\n"
        "@factor(summary='own')\n"
        "def own_fac(x):\n"
        "    return x\n",
    )
    try:
        reg = load_all_factors()  # must NOT raise duplicate for shared_fac
        assert reg["shared_fac"].module == "algua.features.tmp_defn_mod"
        assert "own_fac" in reg
    finally:
        _drop_module("tmp_reexp_mod", reexp)
        _drop_module("tmp_defn_mod", defn)


def test_duplicate_name_fails_closed():
    a = _write_feature_module(
        "tmp_dup_a",
        "from algua.features.catalogue import factor\n"
        "@factor(summary='a')\n"
        "def clashing(x):\n"
        "    return x\n",
    )
    b = _write_feature_module(
        "tmp_dup_b",
        "from algua.features.catalogue import factor\n"
        "@factor(summary='b')\n"
        "def clashing(x):\n"  # same bare name, different module
        "    return x\n",
    )
    try:
        with pytest.raises(ValueError, match="duplicate factor name"):
            load_all_factors()
    finally:
        _drop_module("tmp_dup_a", a)
        _drop_module("tmp_dup_b", b)


def test_transactional_failed_import_preserves_prior_registry():
    good = _write_feature_module(
        "tmp_good_mod",
        "from algua.features.catalogue import factor\n"
        "@factor(summary='good')\n"
        "def good_fac(x):\n"
        "    return x\n",
    )
    try:
        load_all_factors()  # registry now has good_fac
        bad = _write_feature_module("tmp_bad_mod", "raise RuntimeError('boom')\n")
        try:
            with pytest.raises(RuntimeError, match="boom"):
                load_all_factors()
            # prior registry intact, never half-populated
            assert "good_fac" in all_factors_names()
        finally:
            _drop_module("tmp_bad_mod", bad)
    finally:
        _drop_module("tmp_good_mod", good)


def all_factors_names():
    return {f.name for f in all_factors()}


def test_get_factor_unknown_raises():
    with pytest.raises(FactorNotFound):
        get_factor("does_not_exist_factor")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_features_catalogue.py -q`
Expected: FAIL with `ImportError: cannot import name 'load_all_factors'` (and `_reset_registry`, `all_factors`, etc.).

- [ ] **Step 3: Write minimal implementation**

Append to `algua/features/catalogue.py`:

```python
import importlib
import pkgutil

import algua.features as _features_pkg

_REGISTRY: dict[str, FactorSpec] = {}
_loaded = False


def load_all_factors() -> dict[str, FactorSpec]:
    """Discover every catalogued factor by scanning ``algua.features`` modules for the
    ``__factor_spec__`` stamp. Transactional: builds a fresh dict and binds it to ``_REGISTRY`` in
    ONE assignment only after every module imports cleanly (a failing import raises before the
    bind, leaving the prior registry intact — no half-populated global is ever observable). Skips
    ``_``-prefixed modules. Accepts a stamped function ONLY at its defining module
    (``fn.__module__ == module.__name__``) so a re-export does not double-register. Idempotent:
    re-scans the import-cached modules each call. Fails closed on a duplicate ``name``."""
    global _REGISTRY, _loaded
    fresh: dict[str, FactorSpec] = {}
    for mod_info in pkgutil.iter_modules(_features_pkg.__path__):
        if mod_info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{_features_pkg.__name__}.{mod_info.name}")
        for value in vars(module).values():
            spec = getattr(value, _SPEC_ATTR, None)
            if spec is None or getattr(value, "__module__", None) != module.__name__:
                continue  # not a factor, or a re-export not defined here
            if spec.name in fresh:
                raise ValueError(
                    f"duplicate factor name {spec.name!r}: {fresh[spec.name].import_path} "
                    f"and {spec.import_path} (pass name= to disambiguate)"
                )
            fresh[spec.name] = spec
    _REGISTRY = fresh
    _loaded = True
    return _REGISTRY


def _reset_registry() -> None:
    """Test hook: clear discovered state so the next read re-discovers."""
    global _REGISTRY, _loaded
    _REGISTRY = {}
    _loaded = False


def _ensure_loaded() -> None:
    if not _loaded:
        load_all_factors()


def get_factor(name: str) -> FactorSpec:
    _ensure_loaded()
    try:
        return _REGISTRY[name]
    except KeyError:
        raise FactorNotFound(name) from None


def all_factors() -> list[FactorSpec]:
    _ensure_loaded()
    return [_REGISTRY[k] for k in sorted(_REGISTRY)]


def filter_factors(
    *, tag: str | None = None, kind: FactorKind | None = None
) -> list[FactorSpec]:
    """Catalogue factors filtered by tag and/or kind (AND-combined)."""
    out = all_factors()
    if tag is not None:
        out = [f for f in out if tag in f.tags]
    if kind is not None:
        out = [f for f in out if f.kind is kind]
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_features_catalogue.py -q`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```bash
git add algua/features/catalogue.py tests/test_features_catalogue.py
git commit -m "feat(140): transactional factor discovery + read API"
```

---

## Task 3: Seed the catalogue — decorate `momentum` and `zscore`

**Files:**
- Modify: `algua/features/indicators.py`
- Test: `tests/test_features_catalogue.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_features_catalogue.py`:

```python
def test_seeded_factors_present_and_filterable():
    reg = load_all_factors()
    assert {"momentum", "zscore"} <= set(reg)

    mom = reg["momentum"]
    assert mom.kind is FactorKind.MOMENTUM
    assert "momentum" in mom.tags
    assert mom.import_path == "algua.features.indicators:momentum"
    assert mom.data_needs == (DataCapability.OHLCV,)

    only_momentum = filter_factors(kind=FactorKind.MOMENTUM)
    assert "momentum" in {f.name for f in only_momentum}
    assert "zscore" not in {f.name for f in only_momentum}

    tagged = filter_factors(tag="cross-sectional")
    assert {"momentum", "zscore"} <= {f.name for f in tagged}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_features_catalogue.py::test_seeded_factors_present_and_filterable -q`
Expected: FAIL — `momentum`/`zscore` not in the registry (not yet decorated).

- [ ] **Step 3: Write minimal implementation**

Edit `algua/features/indicators.py`. Add the import and decorate both functions (do not change their bodies):

```python
from algua.features.catalogue import FactorKind, factor
```

```python
@factor(
    summary="Trailing simple return per symbol over `lookback` periods.",
    kind=FactorKind.MOMENTUM,
    tags=["momentum", "cross-sectional"],
)
def momentum[PandasObj: (pd.Series, pd.DataFrame)](prices: PandasObj, lookback: int) -> PandasObj:
    ...  # body unchanged
```

```python
@factor(
    summary="Cross-sectional z-score (population std; all-NaN on a degenerate cross-section).",
    kind=FactorKind.OTHER,
    tags=["normalization", "cross-sectional"],
)
def zscore(values: pd.Series) -> pd.Series:
    ...  # body unchanged
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_features_catalogue.py tests/test_features_indicators.py -q`
Expected: PASS (catalogue seeded; existing indicator tests still green — the decorator returns the same function).

- [ ] **Step 5: Run the full gate** (lint-imports is the key check — `catalogue` must stay pure)

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add algua/features/indicators.py tests/test_features_catalogue.py
git commit -m "feat(140): catalogue momentum + zscore (seed)"
```

---

## Task 4: `closure_module_names` in approvals (shared closure for lineage)

**Files:**
- Modify: `algua/registry/approvals.py`
- Test: `tests/test_registry_approvals.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_registry_approvals.py`:

```python
from pathlib import Path

import algua.strategies.momentum as _momfam
from algua.registry.approvals import (
    _merged_closure_for,
    closure_module_names,
    compute_artifact_hashes,
)
from algua.strategies.loader import load_strategy


def _write_factor_using_strategy(stem: str) -> Path:
    path = Path(_momfam.__path__[0]) / f"{stem}.py"
    path.write_text(
        "from typing import Any\n"
        "import pandas as pd\n"
        "from algua.contracts.types import ExecutionContract\n"
        "from algua.strategies.base import StrategyConfig\n"
        "from algua.features.indicators import momentum\n"
        f"CONFIG = StrategyConfig(name='{stem}', universe=['AAPL'],\n"
        "    execution=ExecutionContract(rebalance_frequency='1d'),\n"
        "    construction='equal_weight_positive')\n"
        "def signal(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:\n"
        "    wide = view.reset_index().pivot(index='timestamp', columns='symbol',\n"
        "        values='adj_close')\n"
        "    return momentum(wide.iloc[-1], 1).dropna()\n"
    )
    return path


def test_closure_module_names_equals_source_closure_keys():
    path = _write_factor_using_strategy("tmp_closure_strat")
    try:
        loaded = load_strategy("tmp_closure_strat")
        names = closure_module_names(loaded)
        assert names == frozenset(_merged_closure_for(loaded))
        # a top-level-imported factor's module is reached by the closure
        assert "algua.features.indicators" in names
        # compute_artifact_hashes still works (smoke)
        ident = compute_artifact_hashes("tmp_closure_strat")
        assert isinstance(ident.code_hash, str) and ident.code_hash
    finally:
        path.unlink(missing_ok=True)
        import sys
        sys.modules.pop("algua.strategies.momentum.tmp_closure_strat", None)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_registry_approvals.py::test_closure_module_names_equals_source_closure_keys -q`
Expected: FAIL with `ImportError: cannot import name 'closure_module_names'`.

- [ ] **Step 3: Write minimal implementation**

In `algua/registry/approvals.py`, add (right after `_merged_closure_for`):

```python
def closure_module_names(loaded: LoadedStrategy) -> frozenset[str]:
    """The first-party module names in a strategy's identity closure (signal + construction). This
    is exactly the key set of the source closure ``compute_artifact_hashes`` hashes, so lineage
    (issue #140) and ``code_hash`` invalidation share ONE definition of a strategy's dependencies
    at the same module granularity. Adding this consumer does NOT change the hash payload."""
    return frozenset(_merged_closure_for(loaded))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_registry_approvals.py -q`
Expected: PASS (new test + all existing approvals tests — `compute_artifact_hashes` is unchanged).

- [ ] **Step 5: Commit**

```bash
git add algua/registry/approvals.py tests/test_registry_approvals.py
git commit -m "feat(140): closure_module_names (shared closure for lineage)"
```

---

## Task 5: Derived lineage — `factors_used_by` + `dependents_of`

**Files:**
- Create: `algua/registry/lineage.py`
- Test: `tests/test_registry_lineage.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_registry_lineage.py
import sys
from pathlib import Path

import pytest

import algua.strategies.momentum as _momfam
from algua.features.catalogue import FactorNotFound
from algua.registry.db import connect, migrate
from algua.registry.lineage import dependents_of, factors_used_by
from algua.registry.store import SqliteStrategyRepository


@pytest.fixture()
def repo(tmp_path):
    c = connect(tmp_path / "r.db")
    migrate(c)
    return SqliteStrategyRepository(c)


def _write_strategy_using_momentum(stem: str) -> Path:
    path = Path(_momfam.__path__[0]) / f"{stem}.py"
    path.write_text(
        "from typing import Any\n"
        "import pandas as pd\n"
        "from algua.contracts.types import ExecutionContract\n"
        "from algua.strategies.base import StrategyConfig\n"
        "from algua.features.indicators import momentum\n"
        f"CONFIG = StrategyConfig(name='{stem}', universe=['AAPL'],\n"
        "    execution=ExecutionContract(rebalance_frequency='1d'),\n"
        "    construction='equal_weight_positive')\n"
        "def signal(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:\n"
        "    wide = view.reset_index().pivot(index='timestamp', columns='symbol',\n"
        "        values='adj_close')\n"
        "    return momentum(wide.iloc[-1], 1).dropna()\n"
    )
    return path


def _drop(stem: str, path: Path) -> None:
    path.unlink(missing_ok=True)
    sys.modules.pop(f"algua.strategies.momentum.{stem}", None)


def test_factors_used_by_reports_module_granular():
    path = _write_strategy_using_momentum("tmp_uses_strat")
    try:
        used = {f.name for f in factors_used_by("tmp_uses_strat")}
        # module-granular: importing `momentum` pulls in indicators.py -> both factors reported
        assert {"momentum", "zscore"} <= used
    finally:
        _drop("tmp_uses_strat", path)


def test_cross_sectional_momentum_uses_no_catalogued_factor():
    # The bundled strategy inlines its math and imports no catalogued factor.
    assert factors_used_by("cross_sectional_momentum") == []


def test_dependents_of_lists_registered_importer(repo):
    path = _write_strategy_using_momentum("tmp_dep_strat")
    try:
        repo.add("tmp_dep_strat")
        result = dependents_of(repo, "momentum")
        assert "tmp_dep_strat" in result.dependents
        assert result.unloadable == []
    finally:
        _drop("tmp_dep_strat", path)


def test_dependents_of_buckets_unloadable_registered_strategy(repo):
    repo.add("ghost_strategy")  # registered but no module on disk
    result = dependents_of(repo, "momentum")
    assert any(u["name"] == "ghost_strategy" for u in result.unloadable)
    assert "ghost_strategy" not in result.dependents


def test_dependents_of_unknown_factor_raises(repo):
    with pytest.raises(FactorNotFound):
        dependents_of(repo, "no_such_factor")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_registry_lineage.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'algua.registry.lineage'`.

- [ ] **Step 3: Write minimal implementation**

```python
# algua/registry/lineage.py
from __future__ import annotations

from dataclasses import dataclass

from algua.features.catalogue import FactorSpec, all_factors, get_factor, load_all_factors
from algua.registry.approvals import closure_module_names
from algua.registry.repository import StrategyRepository
from algua.strategies.loader import load_strategy


@dataclass
class Dependents:
    """Result of a blast-radius query. ``unloadable`` lists registered strategies whose module
    failed to load (reported, never silently dropped)."""

    factor: str
    dependents: list[str]
    unloadable: list[dict[str, str]]


def factors_used_by(strategy_name: str) -> list[FactorSpec]:
    """Catalogue factors whose defining module is in this strategy's identity closure. Module
    granular (matches code_hash); best-effort for top-level imports (lazy/dynamic imports escape
    the closure). Raises ``StrategyNotFound`` if the strategy module cannot be loaded."""
    load_all_factors()
    loaded = load_strategy(strategy_name)
    modules = closure_module_names(loaded)
    return [f for f in all_factors() if f.module in modules]


def dependents_of(repo: StrategyRepository, factor_name: str) -> Dependents:
    """Registered strategies whose identity closure reaches ``factor_name``'s module. Iterates the
    registry (not just filesystem-discoverable modules) so a registered strategy cannot silently
    vanish from blast radius; a strategy that fails to load lands in ``unloadable`` rather than
    being dropped. Raises ``FactorNotFound`` for an unknown factor."""
    spec = get_factor(factor_name)
    dependents: list[str] = []
    unloadable: list[dict[str, str]] = []
    for rec in repo.list_strategies():
        try:
            loaded = load_strategy(rec.name)
        except Exception as exc:  # noqa: BLE001 - any load failure is reported, not raised
            unloadable.append({"name": rec.name, "error": str(exc)})
            continue
        if spec.module in closure_module_names(loaded):
            dependents.append(rec.name)
    return Dependents(factor=factor_name, dependents=sorted(dependents), unloadable=unloadable)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_registry_lineage.py -q`
Expected: PASS (all 5 tests).

- [ ] **Step 5: Run the full gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all green (note: `registry → features` is allowed; no contract forbids it).

- [ ] **Step 6: Commit**

```bash
git add algua/registry/lineage.py tests/test_registry_lineage.py
git commit -m "feat(140): derived strategy->factor lineage"
```

---

## Task 6: `factor` CLI group (list / show / dependents / uses)

**Files:**
- Create: `algua/cli/factor_cmd.py`
- Modify: `algua/cli/main.py`
- Test: `tests/test_cli_factor.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli_factor.py
import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

import algua.strategies.momentum as _momfam
from algua.cli.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _tmp_db(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))


def _json(result):
    assert result.exit_code == 0, result.stdout
    return json.loads(result.stdout)


def test_list_includes_seeded_factors():
    rows = _json(runner.invoke(app, ["factor", "list"]))
    names = {r["name"] for r in rows}
    assert {"momentum", "zscore"} <= names
    mom = next(r for r in rows if r["name"] == "momentum")
    assert mom["import_path"] == "algua.features.indicators:momentum"
    assert mom["platform_supported"] is True
    assert mom["data_needs"] == ["ohlcv"]


def test_list_filters_by_kind():
    rows = _json(runner.invoke(app, ["factor", "list", "--kind", "momentum"]))
    names = {r["name"] for r in rows}
    assert "momentum" in names
    assert "zscore" not in names


def test_invalid_kind_uses_error_envelope():
    result = runner.invoke(app, ["factor", "list", "--kind", "bogus"])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert "bogus" in payload["error"]


def test_show_full_spec():
    out = _json(runner.invoke(app, ["factor", "show", "momentum"]))
    assert out["ok"] is True
    assert out["module"] == "algua.features.indicators"
    assert out["doc"]


def test_show_unknown_uses_error_envelope():
    result = runner.invoke(app, ["factor", "show", "nope"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False


def test_uses_reports_no_catalogued_factor_for_bundled():
    out = _json(runner.invoke(app, ["factor", "uses", "cross_sectional_momentum"]))
    assert out["factors"] == []


def _write_dep_strategy(stem: str) -> Path:
    path = Path(_momfam.__path__[0]) / f"{stem}.py"
    path.write_text(
        "from typing import Any\n"
        "import pandas as pd\n"
        "from algua.contracts.types import ExecutionContract\n"
        "from algua.strategies.base import StrategyConfig\n"
        "from algua.features.indicators import momentum\n"
        f"CONFIG = StrategyConfig(name='{stem}', universe=['AAPL'],\n"
        "    execution=ExecutionContract(rebalance_frequency='1d'),\n"
        "    construction='equal_weight_positive')\n"
        "def signal(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:\n"
        "    wide = view.reset_index().pivot(index='timestamp', columns='symbol',\n"
        "        values='adj_close')\n"
        "    return momentum(wide.iloc[-1], 1).dropna()\n"
    )
    return path


def test_dependents_lists_registered_importer():
    path = _write_dep_strategy("tmp_cli_dep")
    try:
        assert runner.invoke(app, ["registry", "add", "tmp_cli_dep"]).exit_code == 0
        out = _json(runner.invoke(app, ["factor", "dependents", "momentum"]))
        assert "tmp_cli_dep" in out["dependents"]
        assert out["unloadable"] == []
    finally:
        path.unlink(missing_ok=True)
        sys.modules.pop("algua.strategies.momentum.tmp_cli_dep", None)


def test_dependents_nonzero_exit_on_unloadable_without_allow_partial():
    assert runner.invoke(app, ["registry", "add", "ghost_cli"]).exit_code == 0
    result = runner.invoke(app, ["factor", "dependents", "momentum"])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert any(u["name"] == "ghost_cli" for u in payload["unloadable"])
    # with --allow-partial it exits 0 but still reports
    ok_result = runner.invoke(app, ["factor", "dependents", "momentum", "--allow-partial"])
    assert ok_result.exit_code == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli_factor.py -q`
Expected: FAIL — no `factor` command group registered (Typer usage error / non-zero exit).

- [ ] **Step 3: Write the implementation**

```python
# algua/cli/factor_cmd.py
from __future__ import annotations

import typer

from algua.cli._common import ok, registry_conn
from algua.cli.app import app, emit
from algua.cli.errors import json_errors
from algua.data.capabilities import supported_capabilities
from algua.features.catalogue import (
    FactorKind,
    FactorSpec,
    filter_factors,
    get_factor,
)
from algua.registry.lineage import dependents_of, factors_used_by
from algua.registry.store import SqliteStrategyRepository

factor_app = typer.Typer(
    help="Factor catalogue: discover and trace composable factors", no_args_is_help=True
)
app.add_typer(factor_app, name="factor")


def _spec_json(spec: FactorSpec, *, full: bool) -> dict:
    supported = supported_capabilities()
    data = {
        "name": spec.name,
        "summary": spec.summary,
        "kind": spec.kind.value,
        "tags": list(spec.tags),
        "data_needs": [c.value for c in spec.data_needs],
        "import_path": spec.import_path,
        "signature": spec.signature,
        "platform_supported": all(c in supported for c in spec.data_needs),
    }
    if full:
        data["module"] = spec.module
        data["doc"] = spec.doc
    return data


def _coerce_kind(raw: str | None) -> FactorKind | None:
    if raw is None:
        return None
    try:
        return FactorKind(raw)
    except ValueError as exc:
        allowed = ", ".join(k.value for k in FactorKind)
        raise ValueError(f"unknown kind {raw!r}; allowed: {allowed}") from exc


@factor_app.command("list")
@json_errors()
def list_factors(
    tag: str = typer.Option(None, "--tag", help="filter by tag"),
    kind: str = typer.Option(None, "--kind", help="filter by FactorKind"),
) -> None:
    """List catalogued factors as a JSON array (filters AND-combined)."""
    specs = filter_factors(tag=tag, kind=_coerce_kind(kind))
    emit([_spec_json(s, full=False) for s in specs])


@factor_app.command("show")
@json_errors()
def show_factor(name: str = typer.Argument(..., help="factor name")) -> None:
    """Show one factor's full spec as JSON."""
    emit(ok(_spec_json(get_factor(name), full=True)))


@factor_app.command("dependents")
@json_errors()
def factor_dependents(
    name: str = typer.Argument(..., help="factor name"),
    allow_partial: bool = typer.Option(False, "--allow-partial",
                                       help="exit 0 even if some strategies are unloadable"),
) -> None:
    """Registered strategies whose closure reaches this factor (blast radius)."""
    with registry_conn() as conn:
        result = dependents_of(SqliteStrategyRepository(conn), name)
    emit(ok({"factor": result.factor, "dependents": result.dependents,
             "unloadable": result.unloadable}))
    if result.unloadable and not allow_partial:
        raise typer.Exit(code=1)


@factor_app.command("uses")
@json_errors()
def factor_uses(strategy: str = typer.Argument(..., help="strategy name")) -> None:
    """Catalogued factors a strategy composes."""
    specs = factors_used_by(strategy)
    emit(ok({"strategy": strategy, "factors": [s.name for s in specs]}))
```

Then register the group by adding `factor_cmd` to the import block in `algua/cli/main.py`:

```python
from algua.cli import (  # noqa: F401 - imports register subcommands
    backtest_cmd,
    data_cmd,
    factor_cmd,
    idea_cmd,
    live_cmd,
    paper_cmd,
    registry_cmd,
    research_cmd,
    strategy_cmd,
)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli_factor.py -q`
Expected: PASS (all tests).

- [ ] **Step 5: Run the full gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add algua/cli/factor_cmd.py algua/cli/main.py tests/test_cli_factor.py
git commit -m "feat(140): factor CLI group (list/show/dependents/uses)"
```

---

## Task 7: Authoring guidance — consult the catalogue, import at top level

**Files:**
- Modify: `.claude/skills/author-a-strategy/SKILL.md` (and the `.codex/skills/author-a-strategy/SKILL.md` copy — check whether one is a symlink to the other with `ls -l`; edit the real file(s) so both surfaces show it).

- [ ] **Step 1: Add a "Discover existing factors first" section**

Insert near the top of the "how to author" body (before the strategy is written), text equivalent to:

```markdown
## Reuse before you reinvent: the factor catalogue

Before writing a `signal`, check what already exists:

- `uv run algua factor list [--tag T] [--kind K]` — discover catalogued factors (JSON).
- `uv run algua factor show <name>` — one factor's summary, signature, `import_path`, and data needs.

Prefer composing a catalogued factor over re-deriving it. Import it **at module top level via its
`import_path`** — e.g. `from algua.features.indicators import momentum`. Do NOT import factors lazily
inside `signal` (a function-body or dynamic import escapes the live-gate `code_hash` closure AND the
`algua factor dependents` lineage, so a factor change would silently fail to invalidate your
strategy). A bespoke factor is fine when nothing in the catalogue fits — and consider cataloguing it
with `@factor(...)` so the next strategy can reuse it.

After authoring, `uv run algua factor uses <strategy>` shows which catalogued factors your strategy
pulled in.
```

- [ ] **Step 2: Verify the docs reference real commands**

Run: `uv run algua factor list` and confirm it returns JSON including `momentum`/`zscore` (sanity check that the guidance matches the built CLI).
Expected: a JSON array containing the seeded factors.

- [ ] **Step 3: Commit**

```bash
git add .claude/skills/author-a-strategy/SKILL.md .codex/skills/author-a-strategy/SKILL.md
git commit -m "docs(140): author-a-strategy — discover + compose catalogued factors"
```

---

## Final verification (before opening the PR)

- [ ] Run the full gate one last time: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports` — all green.
- [ ] Confirm `lint-imports` still passes (the catalogue stays pure; only `cli→{features,registry,data}` and `registry→features` were added; the approval path imports no catalogue).
- [ ] Smoke the CLI end-to-end: `uv run algua factor list`, `uv run algua factor show momentum`, `uv run algua factor uses cross_sectional_momentum`.
