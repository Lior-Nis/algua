# Composable factor catalogue: discoverable, reusable pure factors (#140, slice A)

**Status:** design (pre-GATE-1) · **Date:** 2026-06-13 · **Issue:** #140

## Problem

The platform is built for an agent that authors *many* strategies. Every strategy's `signal`
re-derives its alpha building blocks from scratch. The pure composable layer that *should* be the
shared toolbox — `algua/features/indicators.py` — is import-enforced pure (good) but barely
populated (two functions: `momentum`, `zscore`) and is a **plain module, not a catalogue**: there
is no way for the agent to *discover* what factors exist, what each computes, how to call it, or
what data it needs. So the agent reinvents instead of reusing — the opposite of the leverage the
issue wants.

Issue #140 as filed bundles five pieces: (A) a factor catalogue/discoverability layer, (B)
individual factor evaluation (IC/IR or a 1-factor→weights adapter backtest), (C) factor
versioning + a strategy→factor lineage graph, (D) folding factor versions into `config_hash`, and
(E) the #137 DS-integrity multiple-testing multiplier. **This spec is slice A only** — the
discoverability spine. B, C, E are deferred to their own slices; D is **dropped** (see below).

### Why D is dropped, not built

The issue's central "two non-trivial consequences" claim is that `config_hash` must grow to
include the versions of composed factors, "so a factor change invalidates dependent strategies'
backtest results AND live authorizations." Reading the live gate, **this is already true** for
in-repo factors composed by a normal import. The live-gate identity (`algua.registry.approvals
.compute_artifact_hashes`) computes `code_hash` as a *first-party import closure*: starting from
the strategy's signal module it transitively walks every imported `algua.*` module and hashes
their sorted source. A factor in `algua/features/indicators.py` imported via
`from algua.features.indicators import momentum` is therefore already in the closure — editing it
changes `code_hash`, which invalidates that strategy's backtest stamps and live authorization.
Blast-radius invalidation exists today.

Adding a parallel per-factor version into `config_hash` would duplicate git/source hashing and
introduce a second source of truth that can drift from the actual functions. It earns its keep
*only* if commit-churn over-invalidation (every commit changes whole-repo provenance) becomes a
real usability problem at agentic speed — a fine-grained, per-factor dependency hash. That is a
distinct optimization, not a safety necessity, and is explicitly out of scope here.

(One pre-existing closure limitation is unchanged by this slice and called out for the reviewer: a
factor imported *lazily inside a function body* does not land in the strategy module's globals, so
it can escape the import closure. That is a general property of `compute_artifact_hashes`, not
specific to factors, and authoring guidance is to import at module top level.)

## Decision

Add an **in-code factor catalogue**: a decorator that annotates an existing pure factor function
with discoverability metadata and self-registers it, plus a read API and a CLI surface. No DB, no
migration, no parallel versioning — **the source is the version**, and the existing code_hash
closure already governs invalidation. The catalogue rides the existing purity walls entirely
inside `algua.features`.

This is the **in-code decorator** shape (confirmed with the operator). The two alternatives were
rejected: a **DB-backed registry** (mirroring the strategy/idea tables) adds a schema migration and
a second source of truth that can drift from the real functions and duplicates git versioning — too
heavy for pure in-repo functions; a **hybrid** (decorator + a thin DB table) defers the DB to the
slices that genuinely need persistence (B's eval results, C's lineage edges) rather than building
it speculatively now.

## Architecture

### New pure module: `algua/features/catalogue.py`

Lives inside `algua.features`, which the import-linter forbids from importing `cli`, `registry`,
`data`, `backtest`, `strategies`. It may import `algua.contracts` (universally-importable pure
base). Contents:

- `FactorSpec` — a frozen dataclass holding the catalogued metadata:
  - **author-supplied:** `summary: str`, `tags: list[str]`, `kind: FactorKind`,
    `data_needs: list[DataCapability]`.
  - **auto-derived:** `name: str`, `signature: str`, `import_path: str`
    (`algua.features.indicators:momentum`), `module: str`, `doc: str | None`.
- `@factor(*, summary=None, tags=None, kind=FactorKind.OTHER, data_needs=None)` — the decorator.
  It builds a `FactorSpec` and registers it in the module-level `_REGISTRY: dict[str, FactorSpec]`,
  then **returns the function unchanged** (a pure annotation — no wrapper, so call semantics,
  `inspect.getsource`, and `code_hash` see the real function). Defaults: `name = fn.__name__`,
  `summary` ← first non-empty line of the docstring (error if neither given), `data_needs =
  [DataCapability.OHLCV]` (the only platform-supported capability today, per
  `algua.data.capabilities`), `tags = []`.
- Duplicate-name registration **fails closed** (raises) — mirrors the strategy loader's duplicate
  guard so the catalogue can never silently shadow one factor with another.
- Read API: `get_factor(name) -> FactorSpec` (raises `FactorNotFound` on miss),
  `all_factors() -> list[FactorSpec]` (sorted by name), `filter_factors(*, tag=None, kind=None)`
  (AND-combined).
- `load_all_factors() -> None` — `pkgutil`-walks the `algua.features` package and imports each
  module so decorators self-register regardless of import order (mirrors the loader's
  filesystem-walk discovery). Idempotent.

### New enum: `FactorKind` in `algua/contracts/`

Controlled vocabularies live in `contracts` (cf. `DataCapability`, `Author`, `HypothesisStatus`).
`FactorKind(StrEnum)`: `MOMENTUM, REVERSION, VALUE, SENTIMENT, VOLATILITY, QUALITY, OTHER`.
Deliberately minimal with `OTHER` as the escape hatch — extend as real factors demand, not
speculatively. `data_needs` reuses the existing `algua.contracts.idea.DataCapability` vocabulary
(no fork — the #126/#132 tie-in for free).

### Seed: decorate the existing factors

`momentum` and `zscore` in `indicators.py` get `@factor(...)` annotations (importing the decorator
from the sibling `catalogue` module — features→features is allowed). This is the only edit to
existing factor code, and it is additive.

### New CLI: `algua/cli/factor_cmd.py`

`cli`→`features` is allowed. Calls `load_all_factors()` first, then:

- `algua factor list [--tag T] [--kind K]` → JSON array of
  `{name, summary, kind, tags, data_needs, import_path, signature}`, sorted by name; filters
  AND-combined. JSON on stdout per the platform contract.
- `algua factor show <name>` → the full `FactorSpec` as JSON (adds `module`, `doc`). Unknown name
  → non-zero exit + error JSON, matching existing CLI error conventions (`algua/cli/errors.py`).

Wire the sub-app into `algua/cli/app.py` / `main.py` alongside the existing command groups.

## Data flow

1. CLI command runs → `load_all_factors()` imports all `algua.features.*` modules.
2. Each `@factor`-decorated function self-registers a `FactorSpec` into `_REGISTRY` at import time.
3. The read API (`all_factors`/`filter_factors`/`get_factor`) serves the registry to the CLI, which
   emits JSON on stdout.
4. The agent reads the JSON to discover factors and their `import_path`, then composes them into a
   strategy `signal` via a plain top-level import — which automatically pulls the factor's module
   into that strategy's code_hash closure (existing behavior, no new wiring).

## Error handling

- `@factor` with neither `summary` nor a docstring → `ValueError` at import time (fail closed; a
  catalogued factor must be describable).
- Duplicate `name` across modules → raise at registration (fail closed).
- `get_factor`/`factor show` on an unknown name → `FactorNotFound` → CLI non-zero exit + error JSON.
- Invalid `--kind` value → CLI validation error (Typer enum binding) + non-zero exit.

## Testing

- **catalogue.py:** decorator registers the correct spec; defaults resolve (summary←docstring,
  `data_needs`←`[OHLCV]`, `name`←`__name__`); missing-summary-and-docstring fails closed;
  duplicate-name fails closed; `filter_factors` AND-semantics; auto-derived `signature`/`import_path`
  are correct; the decorated function is the **same object** (identity) and behaves identically (no
  accidental wrapping).
- **load_all_factors():** discovers the seeded factors via the pkgutil walk; idempotent across
  repeated calls.
- **CLI:** `factor list` / `factor show` JSON shape + exit codes; filter combinations (tag, kind,
  both); unknown-name error path; invalid-kind error path.
- **Walls:** `lint-imports` green (catalogue is pure; only `cli`→`features` added).

## Scope boundary — explicitly deferred / dropped

- **Slice B — individual factor evaluation** (factor IC/IR, 1-factor→weights adapter backtest).
  Ties directly into #137 multiple-testing.
- **Slice C — strategy→factor lineage graph** + blast-radius queries ("factor X is leaky → which
  strategies use it?"). This is where a thin DB table may first earn its keep.
- **Slice E — #137 DS-integrity multiplier** (each factor hypothesis feeds funnel-FDR; leaky-factor
  blast radius). Only bites once evaluation (B) exists.
- **Dropped — D, folding factor versions into `config_hash`.** Already covered by the code_hash
  import closure (see *Why D is dropped*). Revisit only as a commit-churn usability optimization.

## Gate

`uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports` green.
