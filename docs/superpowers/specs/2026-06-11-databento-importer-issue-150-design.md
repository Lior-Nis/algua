# Databento importer — raw OHLC + separate CA dataset → bar-schema (#150)

**Status:** design APPROVED — GATE-1 passed (panel: Codex + Gemini Flash + OpenCode/GLM, 3 rounds;
1 CRITICAL [same-date-event rejection] + assorted HIGH/MED folded in; round-3 approved by Codex +
OpenCode)
**Issue:** #150 (slice 4 of #129). Depends on #149 (CA back-adjustment engine, MERGED).
**Date:** 2026-06-11

## Summary

A `DatabentoImporter` under the existing `BarImporter` seam that normalizes a **canonical local
parquet** form (raw OHLC per symbol + a single separate corporate-action events file) into the
bar-schema as one consolidated snapshot. Because there is no vendor adjusted column, it computes
`adj_close` via the #149 engine (`algua.data.corpactions.back_adjust`), keeping raw OHLC auditable
(the bar-schema already stores raw `open/high/low/close` alongside `adj_close`).

This slice also does the **adapter hygiene** the #149 design deferred to the consumer: mapping CA
rows to the typed `Split`/`Dividend` events, UTC-midnight ex-date normalization, row-level
validation, and source-identified (`event_id`-keyed) de-duplication — letting the #149 engine
correctly *aggregate* legitimately-distinct same-date events rather than rejecting them.

## Decisions (settled in brainstorming)

1. **Canonical, vendor-neutral local parquet schema** — we own the import contract. The importer
   parses a documented schema, not Databento's evolving binary wire format (int-scaled prices,
   `instrument_id` symbology, ns timestamps). That native-format parsing is **out of scope**; an
   operator conforms a Databento export to the canonical schema (a thin pre-step they own).
2. **Per-vendor request types** — no bag of optionals. `ImportRequest` becomes the common base;
   `FirstRateImportRequest` carries `adjusted_dir`, `DatabentoImportRequest` carries
   `corp_actions_path`.
3. **Defer `adj_factor` persistence** — `adj_close` lands in the bar-schema snapshot; raw OHLC is
   already auditable there; the factor is recomputable from raw + events. `to_bar_schema` /
   `ingest_bars_streamed` reject extra columns, so a stored factor needs a new sidecar artifact +
   store plumbing — filed as a follow-up, built only if a consumer needs it.

## Canonical input schemas

- **`--raw-dir`** — a directory of **per-symbol** parquet files named `<SYMBOL>.parquet`. Symbol is
  the file stem, canonicalized via `normalize_symbols` (matching FirstRate's filename→symbol model).
  Required columns: `ts`, `open`, `high`, `low`, `close`, `volume`. **`ts` policy (1d session date,
  must be UTC midnight):** tz-aware **UTC** → pass; tz-**naive** → localize to UTC (mirrors
  FirstRate); tz-aware **non-UTC** → **reject** (converting e.g. Eastern-midnight would shift the
  session *date*, not just the clock — never silently `tz_convert` a non-UTC daily column). After
  localization the `ts` **must be UTC midnight** (`ts == ts.normalize()`) — a non-midnight value
  (naive `2024-01-02 16:00`, or a tz-aware UTC intraday stamp) is **rejected**, not accepted, because
  this is the `1d` importer and intraday/session mapping is deferred to #151. **Raw OHLCV values:**
  `open/high/low/close/volume` must be **finite** (no NaN/±inf — `to_bar_schema` rejects NaN but not
  `inf`, and `back_adjust` only checks `close`), prices `> 0`, `volume ≥ 0`; else raise. One file =
  one symbol = one streamed chunk → bounded RAM, satisfying `ingest_bars_streamed`'s
  one-chunk-per-symbol rule.
- **`--corp-actions`** — a single parquet of all symbols' events. Required columns: `symbol`,
  `ex_date`, `kind`, `value`. Optional column: `event_id` (string). `kind` ∈ {`split`, `dividend`}
  (case-insensitive, whitespace-trimmed; anything else → raise). `value` is the split ratio (new/old)
  for a split, or cash-per-share **in the same units as raw `close`** (dollars, no scaling) for a
  dividend. Events are small; the file is read once and grouped per symbol. A symbol with **no** rows
  here is legitimate (→ identity adjustment, `adj_close == close`). **Only ordinary splits and cash
  dividends are modeled** — the operator must NOT encode spin-offs / rights / returns-of-capital as a
  `dividend` (out of scope; #149 engine guards reject economically implausible ones, but a mislabeled
  one is the operator's responsibility).

Both parsers fail closed (clear `ValueError`) on missing columns, unparseable dates/numbers, blank
symbols, NaN/±inf or non-positive `value`, or an unknown `kind`. No NaN-fill.

## `ImportRequest` refactor (per-vendor request types)

```python
@dataclass(frozen=True, kw_only=True)
class ImportRequest:                       # common fields only
    raw_dir: Path
    timeframe: str = "1d"
    as_of: str | None = None
    adjustment: str = "split_div"
    symbols: tuple[str, ...] | None = None

@dataclass(frozen=True, kw_only=True)
class FirstRateImportRequest(ImportRequest):
    adjusted_dir: Path                     # required, no default

@dataclass(frozen=True, kw_only=True)
class DatabentoImportRequest(ImportRequest):
    corp_actions_path: Path                # required, no default
```

**`kw_only=True`** (pinned mechanism): with keyword-only fields a required subclass field after the
base's defaulted fields is legal — no ordering `TypeError`, no `= ...` sentinel (a sentinel would make
omission *constructible* at runtime, the opposite of fail-closed). It is a deliberate, clean break to
`ImportRequest`'s constructor (now keyword-only); the only in-repo constructor is the CLI (already
keyword) and the FirstRate tests (updated). No compat shim.

`BarImporter.import_bars(request: ImportRequest)` keeps its seam signature **unchanged** in both the
Protocol *and* every implementation — narrowing the *method parameter* to a subtype would violate
the Protocol (parameters are contravariant) and mislead mypy. Instead each importer takes the base
`ImportRequest` and `isinstance`-narrows *inside the body*, failing closed on the wrong type
(`if not isinstance(request, DatabentoImportRequest): raise ValueError(...)`). The Protocol docstring
documents the contract: a caller pairs an importer with its matching request subtype; the registry
erases the subtype, and the importer raises on a mismatch. The FirstRate importer narrows to
`FirstRateImportRequest` the same way; its existing tests must stay green (regression guard).

## `DatabentoImporter`

`name = "databento"`, `vendor_label = "databento"`, registered via
`register_importer("databento", _build_databento)` in `algua/data/importers/__init__.py` (no CLI
if/elif for *construction*).

`import_bars(request)`:
1. Narrow to `DatabentoImportRequest`; reject `timeframe != "1d"`.
2. Discover per-symbol raw files in `raw_dir` (stem→symbol via `normalize_symbols`, dup-symbol
   collision → raise, mirroring FirstRate's `_discover` intent). Apply the optional `symbols` subset
   filter (missing → raise).
3. Load the CA-events parquet once; canonicalize each CA `symbol` the **same way** raw stems are —
   `normalize_symbols([s])[0]` (strip + uppercase), matching FirstRate's per-value normalization — so
   both sides match on one canonical form (kills case/whitespace drift). Note `normalize_symbols` does
   *not* collapse punctuation (`BRK.B` ≠ `BRKB`); punctuation-invariant matching is the operator's
   responsibility when conforming data to the canonical schema. Build `dict[symbol,
   list[CorporateAction]]` (see hygiene below). Events for symbols absent from `raw_dir` are
   **ignored** (a superset CA file covering more symbols than this import is a normal workflow —
   documented, not an error).
4. Per symbol (one chunk): parse raw → `result = back_adjust(raw[["ts","close"]],
   events.get(symbol, []))` → **explicit runtime check** (a real `if … raise ValueError`, **not** a
   bare `assert` — asserts are stripped under `python -O` and this is data-integrity): `len(result)
   == len(raw)` and `result["ts"]` equals the raw `ts` element-wise (length + order). back_adjust
   documents "one row per input bar in input order"; the check fails loud if that postcondition ever
   drifts, instead of silently misaligning `adj_close`. Then assemble `[ts, symbol, open, high, low,
   close, adj_close, volume]` → `ProviderBars(frame, source_metadata={vendor, symbol, raw_file})`.
   `adj_factor` is dropped.

The engine's own guards (raw `ts` strictly ascending/unique/UTC, `close` finite/positive, UTC-midnight
ex-dates, dividend < prior close, etc.) are reused — the importer does **not** re-implement them. An
empty (0-row) per-symbol raw file **fails closed** (`parse_databento_raw` raises) — silently dropping
a present/requested symbol from the consolidated snapshot would be a data-integrity hole (GATE-2).

### CA-events → typed events (the #149-deferred adapter hygiene)

The adapter validates **at the row level** (actionable messages naming symbol / ex_date / kind), then
constructs typed events; the `Split`/`Dividend` `__post_init__` guards remain as defense-in-depth:

- **`kind`**: trim + lowercase, then require ∈ {`split`, `dividend`} (else raise naming the bad value).
- **`value`**: require finite and `> 0` (NaN/±inf/≤0 → raise naming the row), *before* constructing
  the event — so the operator gets a source-file diagnostic, not a bare dataclass error.
- **`ex_date`** → tz-aware **UTC-midnight** `pd.Timestamp` (back_adjust requires it; a bare date
  localizes to UTC midnight; a tz-aware non-UTC or non-midnight value fails closed).
- `kind == "split"` → `Split(ex_date, ratio=value)`; `kind == "dividend"` → `Dividend(ex_date,
  cash=value)`.

**De-duplication (the #149-mandated source dedup), then let the engine aggregate.** The #149 engine
already *sums* same-ex-date dividends and *multiplies* same-ex-date split ratios — so legitimately
distinct same-date events (a regular **and** a special dividend on one ex-date; a split **and** a
dividend; Codex/OpenCode CRITICAL) are correct and **must not be rejected**. The earlier
"raise-on-same-key-different-value" rule was wrong and is dropped. Instead:

- **`event_id` present → all-or-nothing, keyed by `(symbol, event_id)`.** If the column exists, every
  row must carry a non-blank (trimmed) string `event_id` (else raise — a half-populated id column is
  ambiguous). The dedup key is **`(normalized_symbol, event_id)`** (an id is only unique within a
  symbol; a bare-`event_id` key could collide across symbols and wrongly drop a real event). For rows
  sharing a key: if their economics `(ex_date, kind, value)` are **identical**, keep one (true
  duplicate); if they **differ**, **raise** naming the rows — same id with different economics is
  corrupt source data, not something to silently pick from.
- **`event_id` absent → an exact full-row duplicate** `(symbol, ex_date, kind, value)` **raises**
  (#264). Two *genuinely distinct* same-`(ex_date, kind, value)` events are indistinguishable from a
  torn/double-listed feed without `event_id`, and silently dropping one would under-count the
  distribution and under-adjust `adj_close` (`back_adjust` SUMS distinct same-date events). Failing
  closed surfaces the ambiguity loudly — supply `event_id` to disambiguate that rare case.
- All surviving distinct events flow to `back_adjust`, which composes same-date ones correctly.

## CLI (`data import-bars`)

`--corp-actions PATH` is added; `--adjusted-dir` becomes optional. A small **explicit per-vendor
arm** builds the right request subtype and the provenance `source_metadata`, failing closed when a
vendor's required flag is absent or the vendor is unknown:

```
if vendor == "firstrate":   require --adjusted-dir  → FirstRateImportRequest(...)
elif vendor == "databento": require --corp-actions  → DatabentoImportRequest(...)
else: raise ValueError(unsupported)
```

This arm is honest — the *flags* are inherently vendor-specific — and is separate from the registry,
which keeps importer *construction* extension-only. The rest of the command (streaming into
`ingest_bars_streamed`, snapshot record emit) is unchanged.

**Provenance (run-level `source_metadata`):** for databento, record `vendor="databento"`,
`raw_dir`, `corp_actions_file`, and a **`corp_actions_sha256`** content hash of the CA file plus a
`ca_schema_version` marker (hardcoded `"1"` for this canonical layout). Because `adj_close` is
*computed* (not stored as a factor), the CA file
is what makes a re-adjustment reproducible — the hash fingerprints exactly which events produced this
snapshot. (Recomputing adjusted OHLC — `adj_open/high/low` — later means re-providing the same CA
file and re-running `back_adjust`; the hash lets an operator confirm they have the right one.)

## Boundaries

- New `algua/data/importers/databento.py` imports `algua.data.corpactions` (same package — no
  boundary crossing) + `algua.data.contracts` + `algua.data.store.normalize_symbols`. `lint-imports`
  stays green; no new contract needed. No protected files.
- `algua/data/contracts.py` (request types), `algua/data/importers/__init__.py` (registry entry),
  `algua/data/importers/firstrate.py` (request narrowing), `algua/cli/data_cmd.py` (flags + arm).

## Out of scope (deferred)

- `adj_factor` sidecar persistence (follow-up; raw OHLC already auditable in the bar-schema).
- Parsing Databento's **native** binary format (int-scaled prices, `instrument_id` symbology, ns
  `ts_event`) — the canonical schema decouples us from it.
- Intraday timeframes / session mapping (#151).
- A single consolidated raw parquet (vs per-symbol files) — would need in-engine grouping to stay
  RAM-bounded; per-symbol files match the existing streaming model.

## Testing (TDD)

`tests/test_databento_importer.py` (canonical fixtures written to tmp parquet):

- **Engine wiring:** 2:1 split, single dividend, reverse split — `adj_close` matches a hand-computed
  expectation (and `back_adjust` directly, as a cross-check).
- **Multi-symbol:** two symbols, events filtered per symbol; one symbol with **no** events →
  `adj_close == close`. Each symbol is exactly one chunk (satisfies `ingest_bars_streamed`).
- **Same-date composition (the GATE-1 CRITICAL):** a regular **and** a special dividend on one
  ex-date → both summed by the engine (`adj_close` matches the summed-cash expectation, NOT rejected);
  a split **and** a dividend on one ex-date → composed correctly.
- **De-dup:** with `event_id`, a duplicate-`(symbol, event_id)` row is dropped (factor unchanged vs
  the de-duped set); a blank/missing `event_id` when the column exists → raises; same `(symbol,
  event_id)` with **differing** economics → raises; the **same `event_id` across two different
  symbols** keeps both (no cross-symbol drop); without `event_id`, an exact full-row duplicate
  **raises** (#264), and two distinct same-value same-date dividends **with distinct `event_id`** are
  both kept (summed).
- **Adapter validation (row-level):** `kind` case/whitespace normalized (`"Split "`→split); unknown
  `kind` → raises naming the value; `value` NaN/±inf/≤0 → raises naming the row; non-midnight or
  tz-aware non-UTC `ex_date` → raises.
- **Raw `ts` policy:** tz-naive midnight → localized to UTC (accepted); tz-aware UTC midnight →
  passes; tz-aware non-UTC → raises; **non-midnight ts (naive `16:00` or tz-aware UTC intraday) →
  raises** (1d importer).
- **Raw OHLCV:** an `inf` in `open/high/low`, a non-positive price, or a negative `volume` → raises.
- **Guards:** wrong request type → raises; `timeframe != "1d"` → raises; raw file missing a required
  column → raises; dup-symbol raw files → raises.
- **Provenance:** the snapshot's `source_metadata` carries `corp_actions_sha256` matching the CA file.
- **CLI:** `data import-bars --vendor databento --raw-dir … --corp-actions … --as-of …` yields a
  servable snapshot (read it back through `read_bars`, assert `adj_close`); a databento run with
  `--adjusted-dir` and no `--corp-actions` → fails closed.
- **FirstRate regression:** existing `tests/test_firstrate_importer.py` + CLI firstrate path stay
  green after the request-type refactor.

Gate: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`.
