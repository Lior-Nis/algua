# Non-tabular data seam — fundamentals (foundation slice)

**Issue:** #132 — Non-tabular data (news / fundamentals): typed seam, point-in-time discipline, storage substrate.
**Status:** design (foundation slice of a decomposed #132). Revised after GATE-1 multi-model design review (see §12).
**Date:** 2026-06-08.

## 1. Problem & framing

Strategies can only consume the bar-schema `get_bars` seam. There is no path for
non-tabular data (news, fundamentals) into the research layer: storage exists
(`ingest_file` holds content-hashed, `as_of`-provenanced blobs) but there is no typed
consumption seam, no validator, and — most dangerously — no enforced point-in-time
discipline for data where look-ahead is subtle (an as-reported figure later restated, a
headline later revised).

Issue #132 also names **two co-equal consumers with opposite time semantics**:

- **Strategy-signal use** — fundamentals as inputs to `compute_weights`. Strict as-of: a
  record is visible at decision `t` only if its *knowable-at* timestamp ≤ `t`. A leak here
  is a false edge.
- **Agent analysis use** (post-mortem, idea sourcing) — wants full hindsight, to explain
  the past.

The platform's worst possible leak is these two sharing an accessor: the agent reasons
with hindsight, forms a hypothesis, and that hindsight silently becomes a backtest
feature. Hindsight (analysis-mode) data must be **structurally unable** to reach a
backtest's `compute_weights`.

### Decomposition (why this is a slice, not all of #132)

#132 spans two data types (news *and* fundamentals), each needing typed contract +
validator + ingest + serve + PIT enforcement + sanitization, plus two access modes, plus
a storage-substrate decision. That is several vertical seams. This slice builds the
**architecture spine** — the typed point-in-time record, the dual access-mode separation,
the structural wall, and the storage substrate — proven end-to-end against **one** concrete
type: **fundamentals**. News (with its entity→symbol mapping and dedup-heavy sanitization)
becomes a fast follow-up that reuses the established pattern.

Fundamentals is chosen first because the as-of/hindsight wall is hardest exactly where the
same underlying fact has two values at two times — and fundamentals give that cleanly: a
fiscal period's figure has an *as-reported* value (knowable at filing) and a later
*restated* value (knowable only after the restatement). News has an analogous but fuzzier
"headline revised" story and drags in a sanitization rabbit hole orthogonal to the
architecture.

## 2. Architecture — two structurally-separated lanes

### 2.1 As-of signal lane (feeds `compute_weights`)

A typed consumption seam in `algua/contracts/types.py`:

```python
@runtime_checkable
class FundamentalsProvider(Protocol):
    snapshot_id: str  # for provenance stamping, like StoreBackedProvider
    def get_fundamentals(self, symbols: list[str], end: datetime) -> pd.DataFrame: ...
```

The provider returns the **full bitemporal history** for `symbols` with
`knowable_at < end` — **no lower time bound**, because the first decision bar needs the
latest *prior* report, whose `fiscal_period_end` may predate the backtest start. (The
earlier `start` parameter was **dropped**: it invited an implementer to lower-bound on
`knowable_at` and silently starve the first bars — GATE-1 finding.) `symbols` is pushed
down to storage so the materialized frame is bounded by the strategy's universe, mirroring
`StoreBackedProvider.get_bars`.

**The engine owns `t`.** `simulate()/run()` calls `get_fundamentals` **once**, then applies
a pure per-decision-bar mask. Full `validate_fundamentals` runs at **ingest** and inside the
**provider's read** (both in `algua.data`); the engine — which the import wall forbids from
importing `algua.data` — does a **light shape assertion** (`_assert_fundamentals_shape`:
contract columns present + `knowable_at` tz-aware, using the `algua.contracts` constants) so
a buggy provider can't poison the loop (GATE-1), without crossing the wall. The mask:

```python
def _fundamentals_as_of(frame: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    """As-of-t fundamentals: of the rows with knowable_at <= t, keep for each
    (symbol, fiscal_period_end, metric) the one with the greatest knowable_at
    (the latest revision knowable by t). knowable_at is unique per
    (symbol, fiscal_period_end, metric) within a snapshot (validator-enforced), so the
    pick is unambiguous and deterministic. Uses only knowable_at <= t -> no look-ahead.
    Returns a NEW frame (never a view); empty in => empty out (empty_fundamentals())."""
```

This is the shape of the existing universe PIT helper `_members_as_of`
(`algua/backtest/engine.py:31`). `t` is the bar timestamp — a **tz-aware UTC**
`pd.Timestamp` (daily bars are session-date midnight UTC per the bar-schema doc); the mask
asserts UTC-awareness defensively. Because daily `t` is midnight UTC, an *intraday* filing
on day D (knowable_at = D 14:00 UTC) is first visible at the decision for **day D+1** — a
deliberately conservative cut that can never leak. The strategy receives only the masked
frame, so future rows are structurally absent — the guarantee the bar `t→t+1` rule gives.

**Point-in-time universe interaction (GATE-1 CRITICAL).** When `universe_by_date` is
active the engine fetches bars (and fundamentals) for the *union* of ever-members. At each
bar `t`, the fundamentals frame must be masked to the **as-of-`t` members** *as well as*
`knowable_at ≤ t`, before `target_weights` runs — otherwise a strategy could rank a current
member using a *future* constituent's (legitimately knowable) fundamentals. The masking is
applied to both `view` and the fundamentals frame at the same point in `_decision_weights`.

**Restatement semantics fall out for free.** A restatement is just a *new row* for the
same `(symbol, fiscal_period_end, metric)` with a later `knowable_at`. Before that
`knowable_at`, the as-of mask yields the originally-reported value; at/after it, the
restated value. No reconciliation math — we time-travel over revisions.

**Defense-in-depth (in lieu of branded types).** Per the agreed wall-depth decision there
are no branded return types; instead `_fundamentals_as_of` asserts the input frame carries
the expected columns (constants from `algua.contracts`, see §4) — catching an accidental
wrong-frame swap at the call site cheaply.

### 2.2 Hindsight analysis lane (agent post-mortems / idea sourcing)

A separate accessor in `algua/data/hindsight.py`:

```python
def query_fundamentals(
    store: DataStore, snapshot_id: str, symbols: list[str] | None = None,
) -> pd.DataFrame: ...
```

Returns **all** rows (full hindsight). It is never wired into the engine. Exposed only via
CLI (`algua data query-fundamentals`, JSON on stdout) — the agent's queryable surface.
There is deliberately **no `as_of` parameter** in this slice (GATE-1: a footgun + a
mask-parity duplication question; hindsight is full-history by definition, and the agent
can filter `knowable_at` itself). If as-of *inspection* is ever wanted, it is added at the
CLI seam by composing the engine's `_fundamentals_as_of` over the hindsight frame — keeping
one masking definition.

### 2.3 The wall — and exactly what it does/doesn't guarantee

The wall is the **static** import graph, enforced in the quality gate by `lint-imports`.
It makes a hindsight leak a **build failure**, which is the realistic threat model here: an
agent (or human) *accidentally* authoring a strategy/feature/engine module that pulls in
the hindsight accessor. It is **not** a runtime sandbox: a strategy is executable Python
and could, in principle, `importlib.import_module(...)`, shell out to the CLI, or read the
data dir directly. Defending against *adversarial dynamic exfiltration* is explicitly
**out of scope** for this slice (documented limitation; possible future hardening = a
loader-side AST/purity scan of bundled strategies). What this slice guarantees: **no static
import path** carries hindsight into the signal lane, and the gate enforces it.

Current state has a real hole: existing contracts bar **`backtest`** and **`features`**
from `algua.data`, but **nothing constrains `algua.strategies`, and `algua.contracts`
isn't barred from `algua.data`** (its forbidden list is `cli`/`registry`/`config`/
`calendar` only). A strategy could therefore `import algua.data.hindsight` straight into
`compute_weights`. (Nothing under `algua/strategies` or `algua/contracts` imports
`algua.data` today, so the new bans break nothing.) This slice closes it:

1. **Strategies + contracts off the data lane.** A new import-linter contract forbids
   `algua.strategies` from importing `algua.data` (strategy modules are pure authored
   functions — they receive *all* data, bars and fundamentals, via the engine), and
   `algua.data` is added to `algua.contracts`'s forbidden list (contracts is the base
   layer). This is the actual structural wall for the strategy lane.
2. **A dedicated forbidden contract for the hindsight module** — `algua.data.hindsight`
   forbidden for `algua.backtest`, `algua.features`, `algua.contracts`, `algua.strategies`,
   `algua.live`, `algua.execution`. States the platform's most dangerous wall directly and
   survives any future relocation of the accessor.
3. **A wall test** — not a runtime `import` test (which can't prove a static-graph property —
   GATE-1), and not a must-fail fixture (which would break the real `lint-imports` gate).
   Instead: a test that parses `pyproject.toml` and asserts the three forbidden contracts are
   present with the right modules, plus an **AST scan** asserting no module under
   `algua/strategies` or `algua/contracts` imports `algua.data`. The real `lint-imports` in the
   quality gate enforces the property on every run.

## 3. Data model — tidy/long, bitemporal

One fundamentals record is one row:

| column | type | meaning |
|---|---|---|
| `symbol` | str (non-null) | issuer ticker (given by the vendor; no entity resolution needed) |
| `fiscal_period_end` | date (non-null) | the period the figure describes (e.g. 2025-03-31) |
| `metric` | str (non-null) | metric name (e.g. `revenue`, `eps_diluted`) |
| `value` | float64 (**NaN allowed**) | the figure |
| `knowable_at` | tz-aware UTC datetime (non-null) | report-availability time (filing + lag) — the PIT key |
| `source` | str (non-null) | provenance label (vendor / dataset) |

**Long/tidy, not wide.** New metrics are new rows, never schema changes; the validator
checks structure, not a fixed metric set. Strategies pivot to wide on demand, exactly as
they already pivot bars (`cross_sectional_momentum.py:32`).

**NaN semantics (GATE-1).** `value` may be `NaN`, meaning "the metric was reported but the
figure is unavailable" — distinct from an *absent row*, meaning "not reported for this
period." The as-of mask passes `NaN` through unchanged; strategies must handle it (as they
already handle insufficient bar history). Key columns (`symbol`, `fiscal_period_end`,
`metric`, `knowable_at`, `source`) are non-null.

`knowable_at` (not `fiscal_period_end`) is the point-in-time axis. The engine masks on
`knowable_at ≤ t`.

## 4. Schema, validator, storage

**Column-name constants live in `algua/contracts`** (GATE-1) — the base layer the engine
*can* import — so `_fundamentals_as_of` and `validate_fundamentals` share one source of
truth without the engine reaching into `algua.data`. Concretely, `algua/contracts/types.py`
gains string constants for the key/columns (no pandas needed — just names) alongside the
`FundamentalsProvider` protocol.

New module `algua/data/fundamentals_schema.py`, parallel to `algua/data/schema.py`, imports
those names and provides:

- `FUNDAMENTALS_COLUMNS` — canonical column list/order (built from the contracts constants).
- `validate_fundamentals(frame)` — enforces:
  - columns/order/dtypes; `symbol`/`metric`/`source` are non-null strings; `value` is
    float64 (NaN permitted); `fiscal_period_end` is a date; `knowable_at` is **tz-aware UTC
    and non-null**;
  - **PIT floor:** `knowable_at >= pd.Timestamp(fiscal_period_end, tz="UTC")` (start-of-day
    UTC; a report filed on the last day of the period is valid). This is a *sanity floor*,
    not a precise availability model — the true availability time is whatever the ingester
    put in `knowable_at` (filing + lag);
  - **bitemporal-key uniqueness:** no two rows share
    `(symbol, fiscal_period_end, metric, knowable_at)` (GATE-1 — guarantees the as-of mask
    has a unique winner, so it is deterministic with no tie-break heuristic; `source` is
    *not* part of this key, so a single snapshot is effectively one coherent source —
    multi-source reconciliation is deferred);
  - canonical sort + no exact-duplicate rows.
- `to_fundamentals_schema(frame)` — reshape/normalize incoming data into canonical form
  (dtype coercion, tz-normalization to UTC) then validate.
- `empty_fundamentals()` — contract-shaped empty frame.
- `logical_fundamentals_hash(frame)` — deterministic content hash over sorted logical rows,
  using the same float/NaN canonicalization discipline as `logical_bars_hash`
  (`algua/data/files.py:138`) so snapshot identity + dedup are reproducible across
  environments.

**Storage substrate.** Reuse the snapshot/manifest/content-hash machinery wholesale:

- `algua/data/models.py`: add `Dataset.FUNDAMENTALS = "fundamentals"` and
  `Kind.FUNDAMENTALS = "fundamentals"`. `SnapshotMetadata`/`SnapshotRecord` are reused as-is;
  `ingest_fundamentals` **derives** `start`/`end` from the data (`min`/`max` of
  `knowable_at`), never trusting user-supplied coverage (GATE-1).
- `algua/data/store.py`: `ingest_fundamentals(...)` (validate → **assert every
  `knowable_at <= as_of`**, i.e. you cannot have fetched a record that becomes knowable
  after you fetched it (GATE-1) → tidy parquet under
  `snapshots/fundamentals/<id>/fundamentals.parquet` → append manifest, dedup on snapshot id)
  and `read_fundamentals(...)` (read back, pushdown symbol filter).
- Local `data_dir` + manifest now; **cloud lift deferred per architecture spec §8**.

`algua/data/serve.py`: `StoreBackedFundamentalsProvider` implements `FundamentalsProvider`
over a snapshot (reads via `store.read_fundamentals`, returns the bitemporal frame; exposes
`.snapshot_id`).

## 5. Strategy integration (opt-in optional param)

- `StrategyConfig` (`algua/strategies/base.py:28`) gains `needs_fundamentals: bool = False`.
- **A typed second authored signature, not signature-overloading** (GATE-1): a declaring
  strategy authors `compute_weights(view, params, fundamentals) -> pd.Series`. `base.py`
  defines `ComputeFundamentalsWeightsFn = Callable[[pd.DataFrame, dict, pd.DataFrame],
  pd.Series]`; `LoadedStrategy` gains a `fundamentals_fn` field (analogous to `panel_fn`).
  The adapter's `target_weights` dispatches on `needs_fundamentals`: `True` →
  `fundamentals_fn(view, params, fundamentals)`, `False` → `fn(view, params)`.
- **Loader validation** (GATE-1): the loader inspects signatures and **fails closed** on
  mismatch — `needs_fundamentals=True` requires a 3-arg `compute_weights` and forbids
  `compute_weights_panel` (no vectorized fundamentals path yet); `needs_fundamentals=False`
  requires the 2-arg form. No silent contract drift.
- `simulate()`/`run()` (`algua/backtest/engine.py:270`) gain an optional
  `fundamentals_provider: FundamentalsProvider | None`. When `needs_fundamentals` and the
  provider is absent → **fail closed** (raise). When present: materialize once (with symbol
  pushdown) → `validate_fundamentals` → mask per bar.
- **The materialized frame forces the per-bar loop.** The loop-forcing condition is the
  *presence of a fundamentals frame*, not merely the config flag (GATE-1): the vectorized
  `panel_fn` fast path cannot express per-`t` as-of masking, exactly as PIT universe already
  forces the loop (`_decision_weights_fast_or_loop`, `engine.py:247`). A vectorized
  fundamentals fast path is deferred.
- **`config_hash` includes the dependency.** `config_hash` (`base.py:68`) is extended to
  include `needs_fundamentals`. Intentional one-time change to all `config_hash` values
  (documented migration note); safe because nothing is live.
- **`BacktestResult` records the fundamentals snapshot.** A new optional
  `fundamentals_snapshot: str | None = None` field, stamped from the provider's
  `.snapshot_id` (via the existing `provenance()` `getattr` path) — reproducibility parity
  with the bars `snapshot_id`/`universe_snapshots`.

## 6. CLI

Both emit JSON on stdout (the agent/operator seam):

- `algua data ingest-fundamentals --from-file PATH --provider P --symbols S --as-of TS
  [--source SRC]` — reads a local tidy file (CSV/parquet), normalizes + validates via
  `to_fundamentals_schema`, asserts `knowable_at <= as_of`, writes one fundamentals
  snapshot. Output `{"ok": true, "snapshot": {...}}`. (A vendor-API `FundamentalsProvider`
  ingest, like `ingest-bars`, is a later addition.)
- `algua data query-fundamentals --snapshot-id ID [--symbols S]` — the **hindsight** read
  (full history). **Stable JSON** (GATE-1): canonical row order
  `(symbol, fiscal_period_end, metric, knowable_at, source)`, ISO-8601 UTC timestamps, ISO
  dates, so agent post-mortems are reproducible/diffable. This is the agent's
  analysis/error-diagnosis surface.

`algua data inspect` already lists snapshots generically; only the new dataset routing is
needed.

## 7. Promotion / paper-live guard

A `needs_fundamentals` strategy must never trade paper/live blind (no fundamentals lane
there yet). Two fail-closed checks (GATE-1 — the human raw-transition path must not bypass):

1. **Paper/live run entry — the real wall.** The paper/live decision path already loads the
   strategy; it refuses to run any `needs_fundamentals` strategy, raising a clear
   "fundamentals strategies cannot run paper/live yet (#132 follow-up)". This is
   actor-agnostic: however a strategy reached `paper`/`live` (agent promote *or* human raw
   transition), it cannot execute blind.
2. **`research promote` early block — friendly stop.** The agent's only path to
   `shortlisted` (it loads + backtests the strategy) refuses a `needs_fundamentals`
   strategy up front with a pointer to the deferred lane, so the agent fails fast rather
   than at the paper boundary.

## 8. Deliberate scope cuts

- **Backtest decision path only.** Paper/live fundamentals wiring deferred; the §7 guards
  make the gap fail-closed, not silent.
- **No vectorized fundamentals fast path** — the loop is forced (consistent with PIT
  universe).
- **Sanitization is light** — dtype normalization + exact-duplicate-row dedup +
  bitemporal-key uniqueness only. **No entity→symbol mapping** (the news rabbit hole).
- **Single coherent source per snapshot** — multi-source reconciliation (same key, two
  vendors, differing values) deferred; the validator rejects key collisions rather than
  silently picking a winner.
- **No corporate-action / restatement math** — restatements are later-`knowable_at` rows we
  time-travel over.
- **No runtime strategy sandbox / branded types** — the wall is the static import graph
  (§2.3); adversarial dynamic exfiltration is an explicit out-of-scope limitation.

## 9. Module / file plan

| Area | File | Change |
|---|---|---|
| Contract seam + column names | `algua/contracts/types.py` | add `FundamentalsProvider` protocol + fundamentals column-name constants |
| Schema + validator | `algua/data/fundamentals_schema.py` | new: `FUNDAMENTALS_COLUMNS`, `validate_fundamentals`, `to_fundamentals_schema`, `empty_fundamentals`, `logical_fundamentals_hash` |
| Models | `algua/data/models.py` | add `Dataset.FUNDAMENTALS`, `Kind.FUNDAMENTALS` |
| Store | `algua/data/store.py` | `ingest_fundamentals` (derive start/end, assert `knowable_at<=as_of`), `read_fundamentals` |
| As-of serving | `algua/data/serve.py` | `StoreBackedFundamentalsProvider` |
| Hindsight | `algua/data/hindsight.py` | new: `query_fundamentals` (full history, no `as_of`) |
| Engine | `algua/backtest/engine.py` | `_fundamentals_as_of` (UTC assert, empty-safe, column assert); thread+validate provider; mask per bar incl. PIT members; force loop on frame presence; fail-closed when missing |
| Strategy contract | `algua/strategies/base.py` | `needs_fundamentals`; `ComputeFundamentalsWeightsFn`; `LoadedStrategy.fundamentals_fn`; adapter dispatch; `config_hash` includes it |
| Loader | strategy loader | signature ↔ `needs_fundamentals` validation; reject `panel_fn`+`needs_fundamentals` |
| Result | `algua/backtest/result.py` | `BacktestResult.fundamentals_snapshot: str \| None = None` |
| Paper/live guard | live/paper run entry | fail-closed refuse `needs_fundamentals` |
| Promote guard | `algua/registry/promotion.py` (research promote) | early friendly block |
| CLI | `algua/cli/data_cmd.py` | `ingest-fundamentals`, `query-fundamentals` (stable JSON) |
| Lint wall | `pyproject.toml` | strategies-off-data; `algua.data` added to contracts' forbidden list; dedicated `algua.data.hindsight` forbidden contract |
| Contract doc | `docs/contracts/fundamentals-schema.md` | new, parallel to `bar-schema.md` |

## 10. Testing

Following existing patterns (`tests/test_data_schema.py`, `tests/test_data_ingest_streamed.py`,
`tests/test_cli_data.py`, `tests/test_contracts.py`):

- **Validator** — conformant passes; each violation rejected separately (missing column,
  wrong dtype, tz-naive `knowable_at`, null in a key column, `knowable_at <
  fiscal_period_end` incl. the same-day-filing *pass* case, bitemporal-key collision,
  exact-duplicate rows); NaN in `value` is allowed; `to_fundamentals_schema` reshapes valid
  inputs; `empty_fundamentals` is contract-shaped.
- **PIT as-of mask (centerpiece)** — `_fundamentals_as_of`:
  - before a report's `knowable_at` → invisible; at/after → visible;
  - **restatement leak test** — at `t` between original filing and restatement the as-of
    value is the *originally reported* one; after the restatement's `knowable_at` it flips;
  - **intraday-filing conservatism** — a filing at `D 14:00Z` is invisible at the `D 00:00Z`
    daily bar, visible at `D+1`;
  - **empty frame** → empty out;
  - **UTC assertion** — a tz-naive `t` raises.
- **PIT universe × fundamentals (GATE-1 CRITICAL)** — with `universe_by_date` active, a
  future constituent's (knowable) fundamentals do **not** influence a current member's
  weight; the fundamentals frame is masked to as-of members.
- **Structural wall** — the new import-linter contracts hold; a must-fail fixture
  (`from algua.data import hindsight` under a forbidden package) makes `lint-imports` fail;
  a test asserts the engine only ever receives the as-of provider.
- **Store roundtrip** — `ingest_fundamentals` → `read_fundamentals` passes
  `validate_fundamentals`; identical input → identical `snapshot_id` (dedup);
  `knowable_at > as_of` is rejected at ingest.
- **Engine integration** — a `needs_fundamentals` example strategy backtests end-to-end;
  fundamentals seen at each bar honor `knowable_at ≤ t`; missing provider fails closed;
  loader rejects a 2-arg `compute_weights` declaring `needs_fundamentals=True` (and the
  `panel_fn`+`needs_fundamentals` combo); `BacktestResult.fundamentals_snapshot` is stamped.
- **Guards** — `research promote` refuses a `needs_fundamentals` strategy; the paper/live
  run entry refuses it regardless of how it was transitioned.
- **CLI** — `ingest-fundamentals` then `query-fundamentals` roundtrip; hindsight returns
  full history in stable JSON order.

## 11. Gate

`uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports` green.

## 12. GATE-1 review log (multi-model design review)

Panel: **Codex** (deep correctness), **Gemini 2.5 Flash** (broad sweep), **OpenCode/GLM**
(diverse lineage). Findings triaged on merit; the spec above already incorporates the
accepted ones. Summary:

**Folded in (accepted):** PIT-universe × fundamentals masking leak (CRITICAL, Codex);
`knowable_at >= fiscal_period_end` start-of-day-UTC semantics (all 3); typed second
signature + `fundamentals_fn` + loader signature validation (all 3); guard moved to the
paper/live run entry, fail-closed for all actors (Codex + Gemini); drop `start` from
`get_fundamentals` (OpenCode); column constants in `contracts` so the engine doesn't cross
the wall (OpenCode); reframe the wall as a static/lint-enforced guarantee + lint-imports
wall test instead of a runtime import test (Codex + OpenCode); validate provider output at
the seam (Codex); bitemporal-key uniqueness ⇒ deterministic mask (Codex + OpenCode); NaN
allowed in `value`, not in keys (OpenCode); ingest caps `knowable_at <= as_of` (Gemini);
empty as-of frame passes `empty_fundamentals()` (OpenCode + Gemini); symbol pushdown
(OpenCode); stable hindsight JSON (Codex); UTC assertion + defensive copy in the mask
(Gemini); single coherent source per snapshot (resolves the multi-source/tie-break
ambiguity).

**Declined / deferred (with rationale):** branded return types — superseded by the agreed
no-brand decision; replaced with a cheap column-assertion in the mask. AST/runtime sandbox
of strategies against dynamic-import exfiltration — over-scoped for a foundation slice;
documented as an explicit trust-boundary limit + future hardening. `--as-of` on the
hindsight CLI — dropped as a footgun + mask-parity duplication; full hindsight is the
defined behavior, as-of inspection (if ever needed) composes the engine mask at the CLI
seam.

## 13. Deferred follow-ups (out of this slice)

- **News** type (entity→symbol mapping, near-duplicate dedup, encoding) — second instance
  of this architecture.
- **Paper/live fundamentals lane** — wire as-of fundamentals into the paper/live decision
  core; lift the §7 guards once parity holds.
- **Vectorized fundamentals fast path** — a parity-guarded panel form.
- **Multi-source fundamentals reconciliation** — cross-vendor merge with a conflict policy.
- **Vendor-API `FundamentalsProvider` ingest** (analogous to `ingest-bars`).
- **Strategy purity hardening** — optional loader-side AST scan for dynamic
  imports/process/file access, closing the adversarial-exfiltration gap §2.3 leaves open.
- **Cloud storage substrate** — per architecture spec §8.
