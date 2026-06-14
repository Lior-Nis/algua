# Composable factor catalogue + derived lineage (#140, slices A+C)

**Status:** design (GATE-1 approved — 3 rounds folded) · **Date:** 2026-06-13 · **Issue:** #140

## Problem

The platform is built for an agent that authors *many* strategies. Every strategy's `signal`
re-derives its alpha building blocks from scratch. The pure composable layer that *should* be the
shared toolbox — `algua/features/indicators.py` — is import-enforced pure (good) but barely
populated (two functions: `momentum`, `zscore`) and is a **plain module, not a catalogue**: there
is no way for the agent to *discover* what factors exist, what each computes, how to call it, or
what data it needs. So the agent reinvents instead of reusing. And when a factor turns out to be
flawed, there is no quick way to ask **"which strategies use it?"**

Issue #140 as filed bundles five pieces: (A) a factor catalogue/discoverability layer, (B)
individual factor evaluation (IC/IR or a 1-factor→weights adapter backtest), (C) factor
versioning + a strategy→factor lineage graph, (D) folding factor versions into `config_hash`, and
(E) the #137 DS-integrity multiple-testing multiplier. **This spec is slices A + C** — the
discoverability catalogue plus a *derived* lineage graph for best-effort blast-radius queries. B and
E are deferred to their own slices; D is **dropped** (see below).

### Why D is dropped, not built

The issue's central claim is that `config_hash` must grow to include the versions of composed
factors, "so a factor change invalidates dependent strategies' backtest results AND live
authorizations." Reading the live gate, the invalidation property already holds for in-repo factors
composed by a normal **top-level** import — but it is enforced by `code_hash`, not `config_hash`,
and the two `code_hash` notions must not be conflated:

- **Live-gate artifact `code_hash`** (`algua.registry.approvals.compute_artifact_hashes`) — a
  *first-party import closure*: from the strategy's signal module (+ construction module) it
  transitively walks every imported `algua.*` module and hashes their sorted `module → source` map.
  A factor in `algua/features/indicators.py` imported via `from algua.features.indicators import
  momentum` lands in the strategy module's globals, so `_imported_first_party_modules` resolves
  `algua.features.indicators` from `sys.modules` and hashes it. **Editing the factor changes this
  artifact `code_hash`, invalidating the prior live authorization and any approval bound to it.**
- **Backtest-stamp `code_hash`** (`algua.backtest.stamps`) — coarse git-HEAD-plus-dirty provenance
  stamped onto a `BacktestResult`. It is *not* the import closure: it changes on any repo edit and
  does not say which closure changed. It is provenance, not identity.

So D's safety goal is delivered by the live-gate artifact `code_hash`. A parallel per-factor version
in `config_hash` would duplicate git/source hashing and add a second source of truth that can drift.
It earns its keep only if commit-churn over-invalidation becomes a real usability problem (a
fine-grained per-factor dependency hash) — a distinct optimization, out of scope here.

**Closure limitation (carried, not introduced).** The closure reaches factors referenced from a
module's **globals** — i.e. top-level imports. A factor imported *lazily inside a function body*,
imported dynamically by string, or rebound with a misleading `__module__` escapes both the
`code_hash` closure and the derived lineage below. This is a pre-existing property of
`compute_artifact_hashes`. This slice does **not** add a runtime lazy-import detector (a separate
live-gate hardening — noted as a follow-up); instead it makes top-level imports of catalogue factors
the **documented authoring contract** and adds a gate test proving the closure reaches a
top-level-imported factor. Consequently the lineage in slice C is **best-effort for the
top-level-import contract, not a blast-radius safety guarantee** (see *Lineage*).

## Decision

Add an **in-code factor catalogue** plus a **derived lineage** read layer:

1. A decorator that *annotates* an existing pure factor function with discoverability metadata by
   stamping an immutable `FactorSpec` onto the function object (no wrapper, no import-time global
   mutation). Discovery scans feature modules for the stamp and builds the registry transactionally.
2. A derived strategy→factor lineage: compute edges on demand by intersecting each **registered**
   strategy's first-party import closure (the *same* closure the live gate hashes) with the
   catalogue, at module granularity. No DB, no migration, can't drift.

Rejected alternatives: a **DB-backed factor registry** and a **persisted lineage table** (both add a
second source of truth that duplicates git versioning / the import graph — the anti-pattern avoided
with `code_hash`); a `uses_factors` declaration in `StrategyConfig` (manual, drift-prone, changes
the strategy contract and identity).

## Architecture

### New pure module: `algua/features/catalogue.py`

Inside `algua.features` (import-linter forbids it from `cli`/`registry`/`data`/`backtest`/
`strategies`; it may import `algua.contracts`). Contents:

- `FactorKind(StrEnum)`: `MOMENTUM, REVERSION, VALUE, SENTIMENT, VOLATILITY, QUALITY, OTHER` —
  minimal, with `OTHER` as the escape hatch. Co-located here (not `contracts`) so a catalogue-only
  taxonomy doesn't freeze a premature platform-wide vocabulary.
- `FactorSpec` — a frozen dataclass; collection fields are **tuples** (a frozen dataclass doesn't
  stop list mutation): author-supplied `summary: str`, `tags: tuple[str, ...]`, `kind: FactorKind`,
  `data_needs: tuple[DataCapability, ...]`; auto-derived `name: str`, `signature: str`,
  `import_path: str` (`algua.features.indicators:momentum` — the stable machine identity),
  `module: str`, `doc: str | None`.
- `@factor(*, name=None, summary=None, tags=(), kind=FactorKind.OTHER, data_needs=None)` — a **pure
  annotation**: it builds the `FactorSpec` and stamps it on the function as `fn.__factor_spec__`,
  then **returns the function unchanged** (same object — `inspect.getsource`/`code_hash` see the real
  function; the stamp is a function attribute, not source, so it does not change the source hash).
  It does **not** touch any module global at import time. Defaults: `name = fn.__name__`; `summary` ←
  first non-empty docstring line (raise if neither given); `data_needs = (DataCapability.OHLCV,)`.
  `data_needs` states the factor's **input requirement**, NOT current platform availability.
- **Transactional discovery** — `load_all_factors() -> dict[str, FactorSpec]` `pkgutil`-walks
  `algua.features.*` (skipping `_`-prefixed modules, mirroring the loader), imports each, then scans
  each module's members for callables carrying `__factor_spec__`, collecting into a **fresh local
  dict**. The scan accepts a stamped function **only when it is defined in the module being scanned**
  (`fn.__module__ == module.__name__`); this filters re-exports — a module doing `from
  algua.features.indicators import momentum` re-exposes `momentum`'s stamp, which must NOT count as a
  second registration (otherwise the duplicate-name guard would falsely break the whole catalogue).
  With that filter each factor is discovered exactly once at its defining module, so `import_path` is
  unique and a genuine cross-module `name` clash → fail closed (`name=` resolves it). It then assigns
  the dict to the module global `_REGISTRY` in **one reference binding** (atomic in CPython — no
  half-populated global is ever observable; a failing module import raises before the swap, leaving
  the prior `_REGISTRY` intact). Idempotent: every call rebuilds from a fresh scan of the
  (import-cached) modules, so it never goes stale and re-runs cleanly after `_reset_registry()`. This
  scan-the-stamp model is what makes "transactional" actually hold — the decorator can't write a temp
  registry, but the *scan* can.
- Read API: `get_factor(name)` (raises `FactorNotFound`), `all_factors()` (sorted), `filter_factors
  (*, tag=None, kind=None)` (AND-combined). Each ensures the registry is loaded.

Feature modules are purity-walled and hold only cheap pure definitions, so importing all of them is
safe; the `_`-skip + transactional rebuild bound the failure surface.

### Live-authorization invalidation note (one-time, expected)

Decorating `indicators.py` requires it to `import` the `catalogue` module, which imports
`contracts.idea` (for `DataCapability`). The existing closure walker therefore now pulls
`algua.features.catalogue` (+ `algua.contracts.idea`) into the `code_hash` of **every strategy that
imports `indicators`**. This is a deliberate, one-time **live-authorization / backtest-identity
invalidation event**, not a silent change. It is harmless today (nothing is live; no strategy is
past `forward_tested`), but the change note records it. Tests separate the two hash effects: (a) a
**refactor-only no-op** test that the `closure_module_names` extraction (below) does NOT change
`compute_artifact_hashes` output, and (b) the **intentional** decoration change.

### Derived lineage: `algua/registry/lineage.py`

No contract restricts `registry → features` (registry already imports `strategies`, which may import
`features`). This module imports the catalogue, the loader, and the registry repository. It reuses
the live gate's closure so lineage and `code_hash` share **one** definition of a strategy's
dependencies, at the same granularity. To keep the approval path catalogue-free, lineage lives in
its **own** module — `compute_artifact_hashes` does not import the catalogue.

- In `approvals.py`, keep the existing source closure as the primitive (the sorted `module → source`
  map that `compute_artifact_hashes` hashes — unchanged). **Add** a thin
  `closure_module_names(loaded) -> frozenset[str]` defined as `frozenset(_merged_closure_for
  (loaded))` (the keys). This adds a consumer; it does not alter the hash payload.
- `factors_used_by(strategy_name) -> list[FactorSpec]` — load the catalogue + the strategy, return
  catalogue factors whose `.module` is in `closure_module_names(loaded)`.
- `dependents_of(repo: StrategyRepository, factor_name) -> Dependents` — the repository is an
  **explicit injected seam** (no hidden default-DB access). Validate the factor exists, then iterate
  the **registry's** strategies (`repo`, not just filesystem-discoverable modules, so a
  registered strategy can't silently vanish from blast radius). For each, attempt to load + compute
  the closure; include it if the factor's module is present. Returns
  `{factor, dependents: [name…], unloadable: [{name, error}…]}` — strategies that fail to load are
  **reported, never dropped**.

**Granularity = module, deliberately, and best-effort.** A strategy counts as using a factor if the
factor's *module* is in the strategy's closure — the granularity `code_hash` already uses. This
**over-reports within the supported path** (a strategy importing one factor from `indicators.py` is
reported against every catalogued factor in that module), which is the safe direction. But it is
**not** an absolute blast-radius guarantee: factors hidden by lazy/dynamic/rebound imports escape
the closure (see *Closure limitation*), so `dependents_of` can under-report for strategies that
violate the top-level-import authoring contract. The spec states this plainly: lineage is an
operational aid scoped to the authoring contract, plus an explicit `unloadable` bucket — not a wall.

### CLI: `algua/cli/factor_cmd.py`

`cli` may import `features`, `registry`, and `data`. Every subcommand emits JSON on stdout and wraps
its body in `json_errors` (`algua/cli/errors.py`) so all failures use the platform envelope
(`{"ok": false, "error": …}`, exit 1). `--kind` is accepted as `str | None` and coerced to
`FactorKind` **inside** the wrapped body (Typer validates enum params before the body runs, which
would bypass the envelope — matching the existing `_parse_required_data` idiom). `list`/`show`
enrich each factor at the CLI seam with `platform_supported: bool` by comparing `data_needs` to
`algua.data.capabilities.supported_capabilities()` (cli→data is allowed; keeps features pure) — so
the agent distinguishes "needs data we can't serve yet" from runnable.

- `algua factor list [--tag T] [--kind K]` → JSON array of `{name, summary, kind, tags, data_needs,
  import_path, signature, platform_supported}`, sorted by name; filters AND-combined.
- `algua factor show <name>` → the full `FactorSpec` JSON + `module`, `doc`, `platform_supported`.
- `algua factor dependents <name> [--allow-partial]` → `{factor, dependents:[…], unloadable:[…]}`.
  Exits non-zero if `unloadable` is non-empty unless `--allow-partial` (fail-closed: a broken
  strategy must not silently shrink blast radius).
- `algua factor uses <strategy>` → `{strategy, factors:[name…]}`.

Unknown factor/strategy and invalid `--kind` all surface via the error envelope + non-zero exit.

### Authoring guidance

Update the `author-a-strategy` skill/doc: before authoring, consult `algua factor list`/`show` and
prefer composing catalogued factors; import them **at module top level via their `import_path`**
(lazy/function-body imports escape both the live-gate `code_hash` closure and lineage); a bespoke
factor is acceptable when nothing fits. This closes the "discoverability-only attractor" gap and is
the human-readable companion to the closure's top-level-import requirement.

## Error handling

- `@factor` with neither `summary` nor docstring → `ValueError` at import (fail closed).
- Duplicate `name` during discovery → raise (fail closed); resolve with `name=`.
- A feature module failing to import during discovery → the rebuild raises before the swap; the
  prior `_REGISTRY` is preserved; the CLI surfaces it via the envelope.
- A registered strategy failing to load during `dependents_of` → reported in `unloadable`, not
  dropped; CLI exits non-zero unless `--allow-partial`.
- Unknown name (`get_factor`/`show`/`dependents`/`uses`) → `FactorNotFound`/`StrategyNotFound` →
  envelope + exit 1. Invalid `--kind` → envelope + exit 1.

## Testing

- **catalogue.py:** decorator stamps the correct spec and returns the **same function object**
  (identity) behaving identically; defaults resolve (summary←docstring, `data_needs`←`(OHLCV,)`,
  `name`←`__name__`); missing-summary-and-docstring fails closed; duplicate-name fails closed and
  `name=` resolves it; `filter_factors` AND-semantics; `signature`/`import_path` correct; collection
  fields are tuples.
- **load_all_factors():** discovers seeded factors; skips `_`-prefixed; idempotent; **transactional**
  — a deliberately-failing temp module leaves the prior `_REGISTRY` intact and observable; re-runs
  after `_reset_registry()`; **re-export filter** — a feature module that does `from
  algua.features.indicators import momentum` while defining its own factor registers only its own
  (no false duplicate from the re-exported stamp).
- **closure / hash gates:** (a) refactor-only no-op — adding `closure_module_names` and computing it
  from the source closure does NOT change `compute_artifact_hashes` output; (b) a strategy with
  `from algua.features.indicators import momentum` has `algua.features.indicators` in
  `closure_module_names`, and editing that module changes the artifact `code_hash`.
- **lineage.py:** `factors_used_by`/`dependents_of` correct for a registered strategy importing a
  seeded factor; module-granular over-inclusion behaves as specified; an unloadable registered
  strategy lands in the `unloadable` bucket; unknown factor/strategy raise.
- **CLI:** `list`/`show`/`dependents`/`uses` JSON shape + exit codes; filter combinations;
  `platform_supported` derivation; `--allow-partial` toggles the dependents exit code;
  unknown-name and invalid-kind go through the envelope.
- **Walls:** `lint-imports` green (catalogue pure; `cli→{features,registry,data}`,
  `registry→features` only; the approval path stays catalogue-free).

## Scope boundary — deferred / dropped

- **Slice B — individual factor evaluation** (IC/IR, 1-factor→weights adapter). A persisted store may
  first earn its keep here.
- **Slice E — #137 DS-integrity multiplier** (factor hypotheses feed funnel-FDR; leaky-factor blast
  radius now *queryable* via `factor dependents`). Bites once B exists.
- **Dropped — D** (factor versions into `config_hash`): already covered by the live-gate artifact
  `code_hash` closure.
- **Deferred hardening:** a runtime detector rejecting lazy `algua.*` imports inside `signal` (closes
  the closure-escape hole structurally, upgrading lineage from best-effort to guaranteed); richer
  structured factor input specs (frequency / PIT / lookback) beyond coarse `DataCapability`.

## Gate

`uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports` green.
