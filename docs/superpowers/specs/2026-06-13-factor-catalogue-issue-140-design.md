# Composable factor catalogue + derived lineage (#140, slices A+C)

**Status:** design (GATE-1 round 1 folded) · **Date:** 2026-06-13 · **Issue:** #140

## Problem

The platform is built for an agent that authors *many* strategies. Every strategy's `signal`
re-derives its alpha building blocks from scratch. The pure composable layer that *should* be the
shared toolbox — `algua/features/indicators.py` — is import-enforced pure (good) but barely
populated (two functions: `momentum`, `zscore`) and is a **plain module, not a catalogue**: there
is no way for the agent to *discover* what factors exist, what each computes, how to call it, or
what data it needs. So the agent reinvents instead of reusing — the opposite of the leverage the
issue wants. And when a factor turns out to be flawed, there is no way to ask **"which strategies
use it?"** without hand-introspecting every strategy.

Issue #140 as filed bundles five pieces: (A) a factor catalogue/discoverability layer, (B)
individual factor evaluation (IC/IR or a 1-factor→weights adapter backtest), (C) factor
versioning + a strategy→factor lineage graph, (D) folding factor versions into `config_hash`, and
(E) the #137 DS-integrity multiple-testing multiplier. **This spec is slices A + C** — the
discoverability catalogue plus a *derived* lineage graph for blast-radius queries. B and E are
deferred to their own slices; D is **dropped** (see below).

### Why D is dropped, not built

The issue's central "two non-trivial consequences" claim is that `config_hash` must grow to
include the versions of composed factors, "so a factor change invalidates dependent strategies'
backtest results AND live authorizations." Reading the live gate, **the invalidation property
already holds** for in-repo factors composed by a normal top-level import — but it is enforced by
`code_hash`, not `config_hash`, and the two `code_hash` notions must not be conflated:

- **Live-gate artifact `code_hash`** (`algua.registry.approvals.compute_artifact_hashes`) — a
  *first-party import closure*: starting from the strategy's signal module (+ construction module)
  it transitively walks every imported `algua.*` module and hashes their sorted source. A factor in
  `algua/features/indicators.py` imported via `from algua.features.indicators import momentum` lands
  in the strategy signal module's globals, so `_imported_first_party_modules` resolves
  `algua.features.indicators` from `sys.modules` and hashes it. **Editing the factor changes this
  artifact `code_hash`, invalidating the prior live authorization and any approval bound to it.**
  This is the fine-grained, per-strategy identity the live gate checks.
- **Backtest-stamp `code_hash`** (`algua.backtest.stamps`) — coarse git-HEAD-plus-dirty-workspace
  provenance stamped onto a `BacktestResult`. It is *not* the import closure: it changes on any repo
  edit and does not tell you which strategy/factor closure changed. It is provenance, not identity.

So the safety property the issue wants from D is delivered by the live-gate artifact `code_hash`
closure. Adding a parallel per-factor version into `config_hash` would duplicate git/source hashing
and introduce a second source of truth that can drift from the actual functions. It earns its keep
*only* if commit-churn over-invalidation becomes a real usability problem at agentic speed (a
fine-grained, per-factor dependency hash) — a distinct optimization, not a safety necessity, and
out of scope here.

**Closure limitation (carried, not introduced).** The closure only reaches factors referenced from
a module's globals — i.e. **top-level imports**. A factor imported *lazily inside a function body*,
imported dynamically by string, or rebound with a misleading `__module__` escapes both the
`code_hash` closure and the derived lineage below. This is a pre-existing property of
`compute_artifact_hashes`, not new to factors. This slice does not add a runtime lazy-import
detector (a separate live-gate hardening — noted as a follow-up); instead it **makes top-level
imports of catalogue factors the documented authoring contract** (see *Authoring guidance*) and
**adds a gate test proving the closure reaches a top-level-imported factor**.

## Decision

Add an **in-code factor catalogue** plus a **derived lineage** read layer:

1. A decorator that annotates an existing pure factor function with discoverability metadata and
   self-registers it, behind a transactional discovery pass, with a pure read API and a CLI.
2. A derived strategy→factor lineage: compute edges on demand by intersecting each strategy's
   first-party import closure (the *same* walker the live gate runs) with the catalogue. No DB, no
   migration, can't drift — consistent with why D is dropped.

This is the **in-code decorator + derived lineage** shape (confirmed with the operator). Rejected
alternatives: a **DB-backed factor registry** (schema migration + a second source of truth that
duplicates git versioning) and a **persisted lineage table** (a sync/staleness problem and a second
source of truth — the very anti-pattern avoided with `code_hash`). A `uses_factors` declaration in
`StrategyConfig` was also rejected: it is manual, drift-prone, and would change the strategy
contract and identity.

## Architecture

### New pure module: `algua/features/catalogue.py`

Lives inside `algua.features`, which the import-linter forbids from importing `cli`, `registry`,
`data`, `backtest`, `strategies`. It may import `algua.contracts`. `lint-imports` continues to
enforce this — the catalogue must reach nothing forbidden. Contents:

- `FactorKind(StrEnum)` — controlled, deliberately minimal with an escape hatch:
  `MOMENTUM, REVERSION, VALUE, SENTIMENT, VOLATILITY, QUALITY, OTHER`. Co-located here (not in
  `contracts`) so a catalogue-only UX taxonomy does not freeze a premature platform-wide vocabulary;
  promote to `contracts` later only if ideas/strategies/gates need it.
- `FactorSpec` — a frozen dataclass. Collection fields are **tuples** (frozen dataclasses don't stop
  list mutation), so a registered spec can't be mutated in place:
  - author-supplied: `summary: str`, `tags: tuple[str, ...]`, `kind: FactorKind`,
    `data_needs: tuple[DataCapability, ...]`.
  - auto-derived: `name: str`, `signature: str`, `import_path: str`
    (`algua.features.indicators:momentum` — the stable machine identity), `module: str`,
    `doc: str | None`.
- `@factor(*, name=None, summary=None, tags=(), kind=FactorKind.OTHER, data_needs=None)` — the
  decorator. It builds a `FactorSpec` and registers it, then **returns the function unchanged** (a
  pure annotation — no wrapper, so call semantics, `inspect.getsource`, and `code_hash` see the real
  function). Defaults: `name = fn.__name__`; `summary` ← first non-empty docstring line (raise if
  neither given — a catalogued factor must be describable); `data_needs = (DataCapability.OHLCV,)`
  (prices-only is the common case). An explicit `name=` override exists to disambiguate cross-module
  collisions. `data_needs` states the factor's **input requirement**, NOT current platform
  availability (a `needs:fundamentals` factor declares `FUNDAMENTALS` even though only OHLCV is
  served today).
- Module-level `_REGISTRY: dict[str, FactorSpec]` plus the read API: `get_factor(name)` (raises
  `FactorNotFound`), `all_factors()` (sorted by name), `filter_factors(*, tag=None, kind=None)`
  (AND-combined).
- **Transactional discovery** — `load_all_factors()` `pkgutil`-walks `algua.features.*` (skipping
  `_`-prefixed modules, mirroring the loader), importing into a *temporary* registry, and swaps it
  into `_REGISTRY` only after every module imports cleanly. A failing module leaves the previous
  registry intact rather than a half-populated one. Idempotent. Duplicate `name` across modules
  **fails closed** (raises) during the pass — `import_path` is always unique, so the optional `name=`
  override resolves any genuine collision. A `_reset_registry()` test hook clears state for isolation.
  (Feature modules are purity-walled and hold only cheap pure definitions, so importing all of them
  is safe; the `_`-skip + transactional swap bound the failure surface.)

### Derived lineage: `algua/registry/lineage.py`

No contract restricts `registry → features` (registry already imports `strategies`, which may import
`features`), so this module may import the catalogue and the loader. It reuses the live gate's
closure walker so lineage and `code_hash` invalidation share **one** definition of "what a strategy
depends on" and therefore the same granularity.

- Refactor `algua.registry.approvals`: extract the existing closure computation into a public
  `closure_module_names(loaded) -> frozenset[str]` (the set of first-party module names reachable
  from the signal + construction modules). `compute_artifact_hashes` keeps hashing the same closure;
  lineage consumes the name set. One walker, two consumers.
- `factors_used_by(strategy_name) -> list[FactorSpec]` — `load_all_factors()`, load the strategy,
  and return catalogue factors whose `.module` is in `closure_module_names(loaded)`.
- `dependents_of(factor_name) -> list[str]` — for each `list_strategies()` name, include it if the
  factor's module is in its closure. Validates the factor exists first (`FactorNotFound` otherwise).

**Granularity = module, deliberately.** A strategy is reported as using a factor if the factor's
*module* is in the strategy's closure — the exact granularity `code_hash` already uses. This
over-reports (a strategy importing one factor from `indicators.py` is reported against every
catalogued factor in that module), which is the **safe** direction for blast-radius incident
response: it never misses an affected strategy. The spec states this explicitly so it isn't read as
a bug.

### Seed: decorate the existing factors

`momentum` and `zscore` in `indicators.py` get `@factor(...)` annotations (importing the decorator
from the sibling `catalogue` module — features→features is allowed). Additive; the only edit to
existing factor code.

### CLI: `algua/cli/factor_cmd.py`

`cli` may import both `features` and `registry`. Every subcommand emits JSON on stdout and uses the
existing platform JSON error envelope (`algua/cli/errors.py`) for failures — including unknown-name
and invalid-`--kind` — rather than a raw Typer traceback. Calls `load_all_factors()` first.

- `algua factor list [--tag T] [--kind K]` → JSON array of
  `{name, summary, kind, tags, data_needs, import_path, signature}`, sorted by name; filters
  AND-combined.
- `algua factor show <name>` → the full `FactorSpec` as JSON (adds `module`, `doc`). Unknown name →
  error envelope + non-zero exit.
- `algua factor dependents <name>` → JSON `{factor, dependents: [strategy, ...]}`. The blast-radius
  query. Unknown factor → error envelope + non-zero exit.
- `algua factor uses <strategy>` → JSON `{strategy, factors: [name, ...]}`. The authoring-time dual.
  Unknown strategy → error envelope + non-zero exit.

Wire the sub-app into `algua/cli/app.py` / `main.py` alongside the existing command groups.

### Authoring guidance

Update the `author-a-strategy` skill/doc (`kb/` + the skill): before authoring, consult
`algua factor list` / `show` and prefer composing catalogued factors; import them **at module top
level via their `import_path`** (lazy/function-body imports escape both the live-gate `code_hash`
closure and lineage); a bespoke factor is acceptable when nothing in the catalogue fits. This closes
the "discoverability-only attractor" gap (a catalogue nobody must use won't stop reinvention) and is
the human-readable companion to the closure's top-level-import requirement.

## Data flow

1. A `factor` CLI command runs → `load_all_factors()` imports all `algua.features.*` modules; each
   `@factor`-decorated function self-registers a `FactorSpec` at import time (transactional swap).
2. `list`/`show` serve the registry as JSON. `dependents`/`uses` additionally load strategies and
   intersect each strategy's closure (`closure_module_names`) with the catalogue.
3. The agent reads the JSON, discovers a factor + its `import_path`, and composes it into a strategy
   `signal` via a top-level import — automatically pulling the factor's module into that strategy's
   `code_hash` closure (existing behavior) and into derived lineage (no new wiring).

## Error handling

- `@factor` with neither `summary` nor a docstring → `ValueError` at import time (fail closed).
- Duplicate `name` across modules → raise during the discovery pass (fail closed); resolve with
  `name=`.
- A module that fails to import during `load_all_factors()` → the whole pass fails and the prior
  registry is preserved (transactional); the CLI surfaces it via the error envelope.
- `get_factor` / `factor show` / `dependents` / `uses` on an unknown name → `FactorNotFound` /
  `StrategyNotFound` → error envelope + non-zero exit.
- Invalid `--kind` → error envelope + non-zero exit (not a raw Typer error).

## Testing

- **catalogue.py:** decorator registers the correct spec; defaults resolve (summary←docstring,
  `data_needs`←`(OHLCV,)`, `name`←`__name__`); missing-summary-and-docstring fails closed;
  duplicate-name fails closed and `name=` resolves it; `filter_factors` AND-semantics; auto-derived
  `signature`/`import_path` correct; the decorated function is the **same object** and behaves
  identically (no accidental wrapping); `FactorSpec` collection fields are tuples.
- **load_all_factors():** discovers the seeded factors; skips `_`-prefixed modules; idempotent;
  **transactional** — a deliberately-failing temp module leaves the prior registry intact;
  `_reset_registry()` isolates tests.
- **closure invalidation gate test:** a strategy with `from algua.features.indicators import momentum`
  has `algua.features.indicators` in `closure_module_names`, and editing that module changes
  `compute_artifact_hashes().code_hash` (proves D's claim and guards the closure regression).
- **lineage.py:** `factors_used_by` / `dependents_of` correct for a strategy importing a seeded
  factor; module-granular over-inclusion behaves as specified; unknown factor/strategy raise.
- **CLI:** `list` / `show` / `dependents` / `uses` JSON shape + exit codes; filter combinations;
  unknown-name and invalid-kind go through the error envelope.
- **Walls:** `lint-imports` green (catalogue pure; `cli→features`, `cli→registry`, `registry→features`
  only).

## Scope boundary — explicitly deferred / dropped

- **Slice B — individual factor evaluation** (factor IC/IR, 1-factor→weights adapter backtest).
  Ties into #137 multiple-testing. A persisted store may first earn its keep here (eval results).
- **Slice E — #137 DS-integrity multiplier** (each factor hypothesis feeds funnel-FDR; leaky-factor
  blast radius now *answerable* via `factor dependents`). Only bites once evaluation (B) exists.
- **Dropped — D, folding factor versions into `config_hash`.** Already covered by the live-gate
  artifact `code_hash` closure. Revisit only as a commit-churn usability optimization (a
  fine-grained per-factor dependency hash).
- **Deferred hardening:** a runtime detector that rejects lazy `algua.*` imports inside `signal`
  (closes the closure-escape hole structurally rather than by authoring guidance); richer
  structured factor input specs (frequency / PIT-sensitivity / lookback) beyond coarse
  `DataCapability`.

## Gate

`uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports` green.
