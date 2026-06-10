# Strategy pool: `strategies/<family>/` + recursive loader (#121)

**Status:** design · **Date:** 2026-06-09 · **Issue:** #121

## Problem

The production strategy pool — meant to scale to **hundreds** of strategies
authored by human and agent — currently lives in a single flat directory literally
named `algua/strategies/examples/`, with a loader hardcoded to it:
`importlib.import_module(f"algua.strategies.examples.{name}")` and
`list_strategies()` iterating only `examples.__path__`.

Two problems at the target scale: (1) the real catalog is misnamed `examples`, and
(2) one flat dir with no sub-organization, and a loader that can only ever see that
one package.

## Decision

Strategies live at **`algua/strategies/<family>/<name>.py`**. There is no
`examples/` and no intermediate `catalog/` package — the `examples/` indirection is
removed entirely. Each thesis family is a subpackage directly under
`algua/strategies/`. The loader **discovers strategies recursively** across family
subpackages.

### Loader contract (`algua/strategies/loader.py`)

- `load_strategy(name)` and `list_strategies()` keep their **bare-name → strategy**
  contract: a strategy is still addressed by its globally-unique `name`, never by
  `family/name`. The registry, gates, CLI, and `config_hash` are unchanged.
- **The index is filename-only and imports NOTHING (GATE-1 HIGH, all three
  reviewers).** Enumerate family directories from the *already-imported*
  `algua.strategies.__path__` (filesystem), then for each family dir build a
  `name → dotted-module-path` map (strings) via `pkgutil.iter_modules([family_dir])`
  — passing the **filesystem path**, NOT importing `algua.strategies.<family>` to
  obtain its `__path__` (GATE-1 round 2: that import would run the family
  `__init__`). This reads module *names from the filesystem* — it `import`s nothing.
  Modules starting with `_` are skipped (private/temp fixtures).
- **`load_strategy(name)` imports exactly ONE strategy module** — it resolves `name`
  to its dotted path via the index, then `importlib.import_module(path)`. The dotted
  import also executes the parent **family `__init__.py`**, so family packages MUST
  be side-effect-free: the scaffold writes an **empty** `__init__.py` and the
  convention (documented in the loader + `author-a-strategy`) is that a family
  `__init__` never imports its member strategies. With that, loading one strategy
  pulls in only an empty family init + that one module — preserving the lazy
  single-import contract the live gate's `code_hash` closure depends on (building
  the index, and loading one strategy, must never pull *other* strategy modules into
  `sys.modules`, or the first-party closure could shift and every strategy's
  `code_hash` would move). A loader test asserts `load_strategy("a")` in a two-member
  family leaves the sibling **absent from `sys.modules`** (the "poison sibling"
  guard).
- **Fail-closed on duplicate names, detected by filename (no import):** if two
  family dirs contain the same bare module name, index construction raises
  (ambiguous identity is a hard error, not silent last-wins). The check is purely
  on the discovered names — it does not import either module.
- **Validation stays on the requested module only:** a module is a strategy iff it
  exposes `CONFIG` + `compute_weights` (unchanged); the optional
  `compute_weights_panel` callable-check is preserved verbatim. This runs in
  `load_strategy` for the one module imported — **not** across the whole catalog.
  (Whole-catalog validation, if ever wanted, is a separate `doctor`/CI concern, out
  of scope here.)
- **Infra exclusion:** `loader.py`, `base.py`, `__init__.py` are top-level *modules*
  in `algua/strategies/`, not subpackages — iterating *subpackages* then their
  modules naturally excludes them; we additionally hard-exclude those known infra
  names as a guard.

### `strategy new` (`algua/cli/strategy_cmd.py`)

- `--family` becomes **required** (currently optional, defaulting to no family).
  With the flat `examples/` gone there is no default target dir, and requiring it
  makes on-disk family == registry family by construction — the agreement the
  issue asks for ("folders and the queryable registry should agree"). A **default
  "uncategorized" family was considered and rejected** (GATE-1): it would
  reintroduce exactly the unsorted-pool problem this issue removes. The active
  `run-the-research-loop` skill already passes `--family`, so the required flag
  breaks no live automation; the omission case raises a precise error:
  `--family is required; use a lowercase slug such as 'momentum'`. Any remaining
  docs/tests that call `strategy new` without it are updated in this PR.
- The scaffold target changes from `strategies/examples/<name>.py` to
  `strategies/<family-dir>/<name>.py`. The family directory (and its `__init__.py`)
  is created if absent, inside the same transactional/rollback block that already
  guards the strategy file + kb docs.
- **One canonical mapping helper (GATE-1, all three reviewers).** `_FAMILY_RE`
  (lowercase a-z/0-9/hyphen) allows hyphens; Python package dirs can't. A single
  helper `family_package_dir(slug) -> str` (= `slug.replace("-", "_")`) is the
  *only* place the slug→dir transform lives, used by `strategy new` (and any future
  doctor check). The registry/kb keep the hyphen slug form; the on-disk dir uses
  underscores. Because `_FAMILY_RE` forbids underscores, `mean-reversion` and a
  hypothetical `mean_reversion` slug can't both exist, so the map is collision-free
  one-way. The **reverse** map (dir→slug) and a folders-vs-registry reconciliation
  check are deferred with the doctor check (Out of scope). The loader itself never
  needs the mapping — it discovers strategies by walking dirs and keys on the bare
  module `name`, not the family.

### Existing strategies (TWO — main advanced mid-design)

`examples/` now holds **two** strategies (`#132` added `fundamentals_earnings_tilt`
when it merged into main, `d2bb494`). Both move:
- `cross_sectional_momentum` → `strategies/momentum/cross_sectional_momentum.py`
  (family `momentum`).
- `fundamentals_earnings_tilt` → `strategies/fundamentals/fundamentals_earnings_tilt.py`
  (family `fundamentals`). It uses the `needs_fundamentals` 3-arg `compute_weights`
  from #132 — the move is a pure relocation; its contract is unchanged.

Each new family dir gets an `__init__.py`. `examples/__init__.py` and the now-empty
`examples/` dir are deleted. The corresponding `registry set <name> --family ...`
(or a kb/registry family assignment) is applied so the registry family matches the
new folder for both.

### Tests

Repoint every test that reaches into `examples/` (grep `strategies.examples` /
`strategies/examples` across `tests/` — both strategy modules are referenced):
- `test_strategy_momentum.py` and any `fundamentals_earnings_tilt` importers —
  import path → `algua.strategies.<family>...`.
- `test_strategy_loader.py`, `test_fast_path.py` — they write temp `_tmp_*.py`
  modules into `examples.__path__`; retarget to a family package dir (e.g. a
  dedicated `strategies/_fixtures/`-style family or the `momentum` dir). Choose one
  family-dir target and centralize it so the temp-module injection has a single
  home. Cover the **duplicate-name → raises** path with a new test (same bare name
  in two family dirs).
- `test_cli_strategy.py` — the path assertions expecting `strategies/examples`
  become `strategies/<family>`; `strategy new` calls gain a `--family`.

## Identity note (reviewed, accepted — strengthened per GATE-1)

`code_hash` is computed from the **source text of the strategy's first-party module
closure, keyed by dotted module name** (`# module: {mod_name}\n{source}`). Moving a
strategy's file changes its dotted module name, so its `code_hash` changes **once**.
`config_hash` (name + universe + params + execution) is **unaffected**.

This invalidates any **existing** identity-keyed record for the moved strategies —
`approvals`, `live_authorizations`, pending `live_challenges`, unconsumed
`gate_evaluations` tokens, and prior backtest stamps keyed on the old `code_hash` no
longer match the recomputed hash.
That is correct fail-closed behavior (a renamed artifact must re-earn its gate), but
it is operationally real, so the PR makes it explicit and safe:
- **Pre-move assertion:** the build asserts nothing is live before the move
  (`registry list --stage live` empty) — moving a live strategy's module would
  strand its live authorization. Locally this holds (nothing live).
- **Documented:** moving a strategy intentionally invalidates its prior
  approvals/gate tokens; it must be re-promoted/re-approved after the move.

We do **not** change hash semantics in this issue (out of scope; a separate
decision). The dotted-name dependence of `code_hash` is the reason the move resets
it; making `code_hash` path-independent is explicitly deferred.

## Out of scope

- Enforcing on-disk-family == registry-family for *already-registered* strategies
  (a `doctor` check). `strategy new` establishes the agreement going forward; a
  reconciliation check is a possible follow-up.
- Moving/renaming families after creation (no `strategy move` command).
- Changing `code_hash` to be path-independent.

## Gate

`uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports` green.
