# Strategy pool `strategies/<family>/` + recursive loader (#121) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Move strategies from the flat `algua/strategies/examples/` into `algua/strategies/<family>/<name>.py`, with a recursive, filesystem-only (zero-import) loader that preserves the bare-`name`→strategy contract.

**Architecture:** The loader builds a `name → dotted-path` index by walking the family subpackages of `algua.strategies` on the **filesystem** (`pkgutil.iter_modules([family_dir])`, no imports), rebuilt per call (cheap; so freshly-written temp modules in tests are seen), fails closed on duplicate bare names, and imports exactly the one requested module. `strategy new --family` becomes required and scaffolds into the family dir via one canonical `family_package_dir()` helper. The two existing strategies move to `momentum/` and `fundamentals/`; `examples/` is deleted.

**Tech Stack:** Python 3.12, pkgutil/importlib, pytest, typer.

Spec: `docs/superpowers/specs/2026-06-09-strategy-pool-family-loader-issue-121-design.md`

**Order rationale:** The new loader treats EVERY subpackage of `algua.strategies` as a family — including `examples/` — so Task 1 (loader rewrite) stays green while the strategies still live in `examples/`. Task 3 then relocates them and deletes `examples/`.

---

## Task 1: Recursive, filesystem-only loader (TDD)

**Files:**
- Modify: `algua/strategies/loader.py`
- Test: `tests/test_strategy_loader.py` (add cases)

- [ ] **Step 1: Add failing tests** to `tests/test_strategy_loader.py`:

```python
def test_load_by_bare_name_across_families(tmp_path):
    # Both bundled strategies resolve by bare name regardless of which family dir they live in.
    assert load_strategy("cross_sectional_momentum").name == "cross_sectional_momentum"
    assert load_strategy("fundamentals_earnings_tilt").name == "fundamentals_earnings_tilt"


def test_duplicate_bare_name_across_families_raises():
    """A bare name appearing in two family dirs is a hard, fail-closed error."""
    import algua.strategies as sp
    root = Path(sp.__file__).parent
    fam_a, fam_b = root / "_dupfam_a", root / "_dupfam_b"
    body = (
        "from algua.contracts.types import ExecutionContract\n"
        "from algua.strategies.base import StrategyConfig\n"
        "import pandas as pd\n"
        "CONFIG = StrategyConfig(name='dupe', universe=['AAA'],\n"
        "    execution=ExecutionContract(rebalance_frequency='1d'))\n"
        "def compute_weights(view, params):\n    return pd.Series(dtype='float64')\n"
    )
    try:
        for fam in (fam_a, fam_b):
            fam.mkdir()
            (fam / "__init__.py").write_text("")
            (fam / "dupe.py").write_text(body)
        with pytest.raises(StrategyNotFound, match="duplicate"):
            load_strategy("dupe")
    finally:
        import shutil
        shutil.rmtree(fam_a, ignore_errors=True)
        shutil.rmtree(fam_b, ignore_errors=True)


def test_loading_one_strategy_does_not_import_siblings():
    """The single-import contract: loading one strategy must not pull a sibling into sys.modules."""
    import sys
    # cross_sectional_momentum lives with no imported siblings; assert the fundamentals module
    # (a different family) is not imported as a side effect of loading momentum.
    sys.modules.pop("algua.strategies.fundamentals.fundamentals_earnings_tilt", None)
    load_strategy("cross_sectional_momentum")
    assert "algua.strategies.fundamentals.fundamentals_earnings_tilt" not in sys.modules
```

NOTE: `test_load_by_bare_name_across_families` and the sibling test reference the POST-MOVE module path; they pass after Task 3. To keep Task 1 green, in Task 1 write the sibling test against the CURRENT path `algua.strategies.examples.fundamentals_earnings_tilt` and the bare-name test will already pass (examples is discovered as a family). Update the path in Task 3. (The duplicate test is path-independent and passes in Task 1.)

- [ ] **Step 2: Run — expect FAIL** (`duplicate` match: current loader has no duplicate detection).

Run: `uv run pytest tests/test_strategy_loader.py -q`

- [ ] **Step 3: Rewrite `algua/strategies/loader.py`:**

```python
from __future__ import annotations

import importlib
import inspect
import pkgutil
from pathlib import Path

import algua.strategies as _strategies_pkg
from algua.strategies.base import LoadedStrategy


class StrategyNotFound(LookupError):
    pass


def _family_dirs() -> list[Path]:
    """Family subpackages directly under algua/strategies/: a dir with an __init__.py that is not
    a private/temp dir. Top-level infra modules (loader.py, base.py, __init__.py) are FILES, not
    dirs, so they are excluded structurally."""
    root = Path(_strategies_pkg.__file__).parent
    return [
        p for p in sorted(root.iterdir())
        if p.is_dir() and (p / "__init__.py").exists() and not p.name.startswith("_")
    ]


def _index() -> dict[str, str]:
    """Map bare strategy name -> dotted module path by walking family dirs on the FILESYSTEM —
    imports nothing. Rebuilt per call (a directory listing is cheap, and tests write temp modules
    after import). Fails closed (raises) on a duplicate bare name across families."""
    index: dict[str, str] = {}
    for fam in _family_dirs():
        for mod in pkgutil.iter_modules([str(fam)]):
            if mod.ispkg:
                continue  # sub-subpackages are not strategies
            dotted = f"algua.strategies.{fam.name}.{mod.name}"
            if mod.name in index:
                raise StrategyNotFound(
                    f"duplicate strategy name {mod.name!r}: {index[mod.name]} and {dotted}"
                )
            index[mod.name] = dotted
    return index


def load_strategy(name: str) -> LoadedStrategy:
    """Load a bundled strategy by bare name; it must expose CONFIG + compute_weights. Resolves the
    name via the filesystem index, then imports EXACTLY ONE module."""
    dotted = _index().get(name)
    if dotted is None:
        raise StrategyNotFound(name)
    module = importlib.import_module(dotted)
    if not hasattr(module, "CONFIG") or not hasattr(module, "compute_weights"):
        raise StrategyNotFound(f"{name} is missing CONFIG or compute_weights")

    panel_fn = getattr(module, "compute_weights_panel", None)
    if panel_fn is not None and not callable(panel_fn):
        raise StrategyNotFound(
            f"{name}.compute_weights_panel is not callable (got {type(panel_fn).__name__})"
        )

    needs_fundamentals = bool(getattr(module.CONFIG, "needs_fundamentals", False))
    n_params = len(inspect.signature(module.compute_weights).parameters)
    if needs_fundamentals:
        if panel_fn is not None:
            raise StrategyNotFound(
                f"{name}: compute_weights_panel is not supported with needs_fundamentals "
                f"(no vectorized fundamentals fast path yet)"
            )
        if n_params != 3:
            raise StrategyNotFound(
                f"{name}: needs_fundamentals=True requires compute_weights(view, params, "
                f"fundamentals); got {n_params} params"
            )
        return LoadedStrategy(config=module.CONFIG, fundamentals_fn=module.compute_weights)

    if n_params != 2:
        raise StrategyNotFound(
            f"{name}: compute_weights must take (view, params); got {n_params} params"
        )
    return LoadedStrategy(config=module.CONFIG, fn=module.compute_weights, panel_fn=panel_fn)


def list_strategies() -> list[str]:
    """All discoverable strategy names (excludes private/temp `_`-prefixed modules)."""
    return sorted(name for name in _index() if not name.startswith("_"))
```

- [ ] **Step 4: Run loader tests — expect PASS** (`uv run pytest tests/test_strategy_loader.py -q`).
- [ ] **Step 5: Run the full suite** (`uv run pytest -q`) — expect the baseline count (examples/ still discovered as a family, so every existing load/list test still passes), plus the new tests.
- [ ] **Step 6: Commit**

```bash
git add algua/strategies/loader.py tests/test_strategy_loader.py
git commit -m "feat(loader): recursive filesystem-only family discovery, fail-closed on dup name (#121)"
```

---

## Task 2: `family_package_dir` helper + required `--family` scaffolding into family dir

**Files:**
- Modify: `algua/cli/strategy_cmd.py`
- Test: `tests/test_cli_strategy.py`

- [ ] **Step 1: Update failing tests first.** In `tests/test_cli_strategy.py`:
  - `test_strategy_new_path_is_package_relative`: add `--family momentum` to the invoke; change `expected_dir = Path(algua.strategies.__file__).parent / "examples"` → `/ "momentum"`.
  - Add a test that `strategy new` WITHOUT `--family` fails with a clear message:

```python
def test_strategy_new_requires_family(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["strategy", "new", "needs_family"])
    assert result.exit_code == 1, result.stdout
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert "family is required" in payload["error"].lower()
```

  - Add a hyphen-slug mapping test:

```python
def test_strategy_new_hyphen_family_maps_to_underscore_dir(tmp_path, monkeypatch, _cleanup_scaffolded):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["strategy", "new", "hyx", "--family", "mean-reversion"])
    assert result.exit_code == 0, result.stdout
    p = Path(json.loads(result.stdout)["path"])
    _cleanup_scaffolded.append(p)
    assert p.parent.name == "mean_reversion"  # dir uses underscores
```

  - Any OTHER `strategy new <name>` invocation in this file that expects SUCCESS but omits `--family` must gain a `--family <slug>` (grep the file for `"new"`).
  - `test_cli_strategy.py:~202` `module_path = ... / "examples" / "rollbackme.py"` → `/ "<family used in that test>" / "rollbackme.py"` (use whatever `--family` that test's invoke passes; add `--family momentum` if absent).

- [ ] **Step 2: Run — expect FAIL** (`uv run pytest tests/test_cli_strategy.py -q`).

- [ ] **Step 3: Edit `algua/cli/strategy_cmd.py`:**
  - Add the helper (near `_FAMILY_RE`):

```python
def family_package_dir(family: str) -> str:
    """The on-disk package dir name for a family slug. `_FAMILY_RE` allows hyphens but Python
    packages can't, so hyphens map to underscores. This is the ONE place the slug->dir transform
    lives (CLI scaffolding + any future doctor check). The registry/kb keep the hyphen slug form."""
    return family.replace("-", "_")
```

  - Make `--family` required and validated, and scaffold into the family dir. Replace the option + the preflight + the `path =` line:

```python
    family: str = typer.Option(..., "--family", help="thesis family this belongs to (required)"),
```
  Keep the existing `if family is not None and not _FAMILY_RE.match(family)` validation but simplify to required:
```python
    if not _FAMILY_RE.match(family or ""):
        raise ValueError(
            "--family is required; use a lowercase slug such as 'momentum' "
            "(a-z, 0-9, hyphen)"
        )
```
  (typer's `...` makes it required at the CLI layer; the explicit check gives the precise message and guards the slug. If typer raises its own "Missing option" first, that's also a non-zero exit — the new `test_strategy_new_requires_family` asserts on the JSON error path, so ensure the command body validates and emits JSON; if typer short-circuits, adjust the option to `default=None` and validate in-body to keep the JSON-error contract. Prefer the in-body `default=None` + explicit check so the error is JSON, matching `_json_errors`.)

  - Change the scaffold path:
```python
    path = (
        Path(__file__).parent.parent / "strategies" / family_package_dir(family) / f"{name}.py"
    )
```
  The existing rollback block already does `path.parent.mkdir(parents=True, exist_ok=True)`; ensure the family `__init__.py` exists (create an EMPTY one if absent, inside the same try/rollback):
```python
        path.parent.mkdir(parents=True, exist_ok=True)
        init_py = path.parent / "__init__.py"
        if not init_py.exists():
            init_py.write_text("")
        path.write_text(_TEMPLATE.format(name=name))
```

- [ ] **Step 4: Run `tests/test_cli_strategy.py` — expect PASS**, then the full suite (`uv run pytest -q`).

- [ ] **Step 5: Commit**

```bash
git add algua/cli/strategy_cmd.py tests/test_cli_strategy.py
git commit -m "feat(cli): strategy new --family required, scaffolds into strategies/<family>/ (#121)"
```

---

## Task 3: Relocate the two strategies; delete `examples/`

**Files:**
- Move: `algua/strategies/examples/cross_sectional_momentum.py` → `algua/strategies/momentum/cross_sectional_momentum.py`
- Move: `algua/strategies/examples/fundamentals_earnings_tilt.py` → `algua/strategies/fundamentals/fundamentals_earnings_tilt.py`
- Create: `algua/strategies/momentum/__init__.py` (empty), `algua/strategies/fundamentals/__init__.py` (empty)
- Delete: `algua/strategies/examples/` (both files moved out + its `__init__.py`)
- Modify tests: `tests/test_strategy_momentum.py`, `tests/test_strategy_loader.py`, `tests/test_fast_path.py`

- [ ] **Step 1: Move files with git** (preserve history):

```bash
cd /home/liornisimov/Projects/algua/.claude/worktrees/121-strategy-family-pool
mkdir -p algua/strategies/momentum algua/strategies/fundamentals
: > algua/strategies/momentum/__init__.py
: > algua/strategies/fundamentals/__init__.py
git mv algua/strategies/examples/cross_sectional_momentum.py algua/strategies/momentum/cross_sectional_momentum.py
git mv algua/strategies/examples/fundamentals_earnings_tilt.py algua/strategies/fundamentals/fundamentals_earnings_tilt.py
git rm algua/strategies/examples/__init__.py
rmdir algua/strategies/examples 2>/dev/null || true
```

- [ ] **Step 2: Repoint the 3 examples-path tests:**
  - `tests/test_strategy_momentum.py`: `from algua.strategies.examples import cross_sectional_momentum as csm` → `from algua.strategies.momentum import cross_sectional_momentum as csm`.
  - `tests/test_strategy_loader.py`: the temp-injection (`_tmp_no_weights_fn`) writes into `algua.strategies.examples.__path__[0]`; retarget to the `momentum` family dir:
    ```python
    import algua.strategies.momentum as fam
    mod_path = Path(fam.__path__[0]) / "_tmp_no_weights_fn.py"
    ```
    Also update the Task-1 sibling test's module path to `algua.strategies.fundamentals.fundamentals_earnings_tilt` (now correct).
  - `tests/test_fast_path.py`: `import algua.strategies.examples as examples` → `import algua.strategies.momentum as examples` (keep the local alias name `examples` to minimize churn, OR rename to `fam`); both `_tmp_with_panel`/`_tmp_no_panel` then land in `momentum/`.

- [ ] **Step 3: Run the full suite** (`uv run pytest -q`). The 9 fundamentals tests + all `load_strategy("...")` callers need NO change (bare-name contract). Expect green.

- [ ] **Step 4: Confirm `examples/` is gone and no references remain**

```bash
test ! -d algua/strategies/examples && echo "examples/ removed"
grep -rn "strategies.examples\|strategies/examples" algua/ tests/ docs/ .claude .codex CLAUDE.md AGENTS.md | grep -v "docs/superpowers/"
```
Expected: the grep returns nothing outside historical `docs/superpowers/`.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor(strategies): relocate strategies into family packages; drop examples/ (#121)"
```

---

## Task 4: Docs + author-a-strategy guidance

**Files:** `CLAUDE.md` (command surface mentions of strategies/examples, if any), `.claude/skills/author-a-strategy/SKILL.md`, `.codex/skills/author-a-strategy/SKILL.md`, any `operating-algua` mention.

- [ ] **Step 1: Update the strategy-location guidance.** Anywhere docs say strategies live in `strategies/examples/`, change to `strategies/<family>/<name>.py` and note: created via `strategy new --family <slug>`; family packages must keep an **empty, side-effect-free `__init__.py`** (a family `__init__` must never import its member strategies — the loader relies on that for the single-import contract). Mention the slug→dir mapping (`mean-reversion` slug → `mean_reversion/` dir) once.

- [ ] **Step 2: Grep**

```bash
grep -rn "strategies/examples\|strategies\.examples\|examples/" CLAUDE.md AGENTS.md .claude/skills .codex/skills docs/agent | grep -v "docs/superpowers/"
```
Expected: no non-historical matches.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "docs(121): strategy pool now strategies/<family>/; author-a-strategy + empty-__init__ convention"
```

---

## Task 5: Final gate + identity assertion

- [ ] **Step 1: Assert nothing is live before relying on the move** (identity note: moving a module resets its code_hash). On the local DB this should be empty:

Run: `uv run algua registry list --stage live`
Expected: empty list (no live strategy whose module moved). If non-empty, STOP and escalate.

- [ ] **Step 2: Full quality gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all green.

---

## Self-review notes
- **Spec coverage:** filesystem-only zero-import index (T1), fail-closed dup (T1), single-import + empty-`__init__` convention (T1+T4), `family_package_dir` helper + required `--family` (T2), both strategies moved + examples deleted (T3), bare-name contract keeps 9 fundamentals tests unchanged (T3), docs (T4), identity/nothing-live assertion (T5). ✓
- **Placeholder scan:** loader + helper given in full; test bodies given in full. ✓
- **Type consistency:** `family_package_dir`, `_index`, `_family_dirs`, `load_strategy`, `list_strategies` names consistent across tasks. ✓
- **Ordering:** loader rewrite first (examples still discovered as a family → green), then CLI, then the move + examples deletion. ✓
