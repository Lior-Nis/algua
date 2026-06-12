# Non-tabular data seam — news (hindsight slice)

**Issue:** #132 — Non-tabular data (news / fundamentals): typed seam, point-in-time discipline, storage substrate.
**Status:** design (second concrete instance of the #132 spine; follows the merged fundamentals slice PR #154 / `d2bb494`). Revised after GATE-1 multi-model design review (see §11).
**Date:** 2026-06-10.

## 1. Problem & framing

The fundamentals slice (PR #154) built the #132 architecture **spine**: a typed, bitemporal,
point-in-time record; a snapshot/manifest/content-hash storage substrate; two
structurally-separated access modes (an as-of **signal** lane that the engine masks into
`compute_weights`, and a full-hindsight **analysis** lane for the agent); and the static
import wall that makes the hindsight lane unreachable from any decision lane. **News** is the
second concrete type named in #132. This slice adds it.

### What is genuinely new for news (vs fundamentals)

Fundamentals arrives already tidy and numeric — the vendor hands a `(symbol, period, metric,
value)` tuple. News is different in three ways the design must address:

1. **It is text, not a number.** There is no obvious numeric payload to feed `compute_weights`.
   "What does a strategy receive from a headline — raw text? a sentiment score? an event
   flag?" is genuinely unsettled and best answered by a concrete consumer.
2. **One article maps to many symbols.** A single article mentions zero or more issuers, so
   the record needs an **entity→symbol** step that fundamentals never needed.
3. **Duplication is endemic, across sources.** The same story arrives from multiple wires, and
   headlines get revised. Revision discipline (a later-revised headline is look-ahead),
   multi-source coexistence, and dedup are first-class, not afterthoughts.

### Why this slice is *hindsight-only* (the key scoping decision)

The agent-analysis use of news is **immediately valuable and well-understood**: the agent
cross-references news to ideate strategies (feeding #126's idea pool and #134's web-research
sourcing) and to diagnose degradation ("why did this strategy fall off — an earnings miss, a
news-regime shift?"). The issue calls the agent a *co-equal first-class consumer*.

The as-of **signal** use, by contrast, hinges on item #1 above — the unsettled "what does
`compute_weights` get from a headline" question. This is **not** a drop in rigor versus
fundamentals; it is the correct response to a real asymmetry. Fundamentals could build its
signal lane in the same slice precisely because its payload was obvious (a `value` float that
drops straight into a tidy frame). News has no such obvious payload — a sound signal API
(raw rows? a scored feature? an event table?) is a design problem a *concrete strategy* must
pull, and building one speculatively now is the YAGNI trap. So this slice ships the valuable,
settled half (hindsight) and **defers the as-of signal lane** to a follow-up — the mirror
image of how the fundamentals slice shipped its signal lane and deferred paper/live. (A
placeholder/dummy signal lane was considered and rejected: dead passthrough code plus a fake
wall surface is strictly worse than its clean absence.)

Crucially, deferring the signal lane costs **no safety**: news is structurally unreachable
from `compute_weights` (see §2.2). And the news-specific hard parts — the tidy multi-symbol
record, the explode step, multi-source identity, dedup, the bitemporal text schema — are
**fully exercised by the hindsight lane alone**; the as-of mask itself is already proven by
fundamentals and would add nothing new here.

## 2. Architecture — one lane (hindsight), and the wall that makes it safe

### 2.1 Hindsight analysis lane

A single new accessor in `algua/data/hindsight.py`, parallel to `query_fundamentals`:

```python
def query_news(
    store: DataStore, snapshot_id: str, symbols: list[str] | None = None,
) -> pd.DataFrame: ...
```

It wraps `store.read_news` (mirroring how `query_fundamentals` wraps `read_fundamentals`),
returning **all** rows for the snapshot (full hindsight), in canonical order, with an optional
symbol pushdown. It is never wired into the engine, and there is deliberately **no `as_of`
parameter** (same GATE-1 reasoning as fundamentals: hindsight is full-history by definition;
the agent filters `knowable_at` itself for an as-of *inspection*). Exposed only via the CLI
(`algua data query-news`, JSON on stdout) — the agent's queryable surface.

### 2.2 The wall — one new contract completes it; no engine, no guards

Because there is **no** `NewsProvider`, **no** `needs_news` config flag, **no** engine
threading, and **no** strategy-facing API, there is no signal path to guard. The remaining
question is purely the structural wall: can any decision lane reach an **unmasked full-history
news read** via a static import?

The full-history read lives in `store.read_news` (in `algua.data`), wrapped by
`hindsight.query_news` (in `algua.data.hindsight`). The decision lanes split into two groups:

- **Research/backtest lanes** — `algua.backtest`, `algua.features`, `algua.strategies`,
  `algua.contracts` — are **already** forbidden from `algua.data` wholesale (and
  `algua.data.hindsight` specifically). They cannot reach `read_news` or `query_news`. ✓
- **Live/execution lanes** — `algua.live`, `algua.execution` — are **already** forbidden from
  `algua.data.hindsight`, **but NOT from `algua.data` wholesale**. They could therefore
  statically `import algua.data.store` and call `read_news` (full, unmasked history) directly,
  bypassing the hindsight accessor. (GATE-1 CRITICAL, Codex.) This is a real gap — and it
  exists today for `read_fundamentals` too.

**Fix — one new import-linter contract:** forbid `algua.live` and `algua.execution` from
`algua.data`. This **completes the data wall** so that *every* decision/execution lane is off
`algua.data`, leaving only the CLI composition root and `algua.data`-internal modules touching
it. It is verified safe: `algua.live`/`algua.execution` import only `algua.contracts`,
`algua.risk.limits`, `algua.strategies.base`, and their own packages — none of which reach
`algua.data` (all already walled or pure) — so they have **no current static path** to
`algua.data`, direct or transitive, and the contract breaks nothing. As a bonus it
**retroactively closes the same `read_fundamentals` bypass** for fundamentals. (If the build
surfaces an unexpected transitive dependency, that is a real hidden coupling to fix, not to
work around — `lint-imports` must stay green.)

With that contract in place, news reaches a backtest or a live decision by exactly **zero**
static routes: nothing threads it into the engine/`compute_weights`, and no decision lane can
import any full-history news read.

The wall's nature and limits are unchanged from #154: a **static import-graph** guarantee
enforced by `lint-imports`, not a runtime sandbox. Adversarial dynamic-import exfiltration (a
strategy calling `importlib.import_module("algua.data.store")`) remains the same documented,
out-of-scope trust-boundary limit, to be closed by the shared strategy-purity-hardening
follow-up — nothing news-specific.

This slice therefore touches `pyproject.toml` only to **tighten** the wall (the safe
direction); it adds no `NewsProvider`, no `needs_news`, no engine code, no strategy-contract
change, and no `promotion.py`/paper/live guard (nothing to guard).

## 3. Data model — tidy/long, bitemporal, per-(source, article, symbol)

One news record is one row, one symbol:

| column | type | meaning |
|---|---|---|
| `source` | str (non-null) | publisher / wire (e.g. `reuters`, `ap`); part of the identity (article ids are only unique *within* a source) |
| `article_id` | str (non-null) | the source's stable article identity (vendor id or URL); the revision-identity axis, scoped by `source` |
| `symbol` | str (upper-cased, non-null) | one mentioned issuer; an article with N symbols explodes to N rows |
| `published_at` | tz-aware UTC datetime (non-null) | original publication time (the economic-event axis; ~ `fiscal_period_end`) |
| `knowable_at` | tz-aware UTC datetime (non-null) | when *this row* became knowable (= `published_at` for an original, later for a correction); **the PIT key**. Always ingester-supplied — never defaulted |
| `headline` | str (non-null) | the headline text |
| `url` | str (**nullable**) | article link |
| `body` | str (**nullable**) | full article text when the source provides it (heavy but optional) |

**Long/tidy, per-symbol.** An article mentioning `AAPL` and `MSFT` is two rows sharing
`source`, `article_id`, `published_at`, `knowable_at`, `headline`, `url`, `body`. This mirrors
the fundamentals tidy shape, so the symbol pushdown in `read_news`/`query_news` is the same
in-memory filter, and snapshot identity/dedup reuse the same machinery.

**Multi-source identity (GATE-1, Codex).** `article_id` is the *source's* id and is only
unique within that source; two wires can legitimately reuse the same numeric id for unrelated
stories. So `source` is part of the identity. The **as-of identity key** is
`(source, article_id, symbol)` and the **unique row key** is
`(source, article_id, symbol, knowable_at)`. This is a deliberate, justified deviation from
fundamentals (which is single-source per snapshot): news *aggregates* multiple wires into one
snapshot, and the schema models that directly rather than forbidding it.

**Two time axes (bitemporal).** `published_at` is the original publication instant — the event
time, analogous to fundamentals' `fiscal_period_end`. `knowable_at` is when this row's content
became knowable to us; for an original it equals `published_at`, for a correction it is
strictly later. `knowable_at` (not `published_at`) is the point-in-time axis a future signal
lane would mask on, and it is what the hindsight contract documents as the PIT key.
**`knowable_at` is always supplied by the ingester, never defaulted** — see §4.

**Revisions (the news analogue of restatements).** A corrected headline/body is a **new row**
sharing the as-of identity key `(source, article_id, symbol)` with a later `knowable_at`. A
future as-of mask would time-travel over revisions exactly as `_fundamentals_as_of` does (keep,
per `(source, article_id, symbol)`, the greatest `knowable_at ≤ t`). Scope limit (GATE-1,
Codex): this models **content** revisions (headline/body/url corrections) cleanly and with no
data migration. It does **not** model **symbol-set** revisions — a correction that *adds* or
*removes* which tickers an article tags — because a per-symbol-row schema cannot express a
removal without a tombstone. Symbol-set revision semantics (tombstones, or article-level
full-replacement) are a signal-lane design question, deferred with that lane (§7); they are not
a hindsight concern (hindsight shows every row).

**Nullable text.** `url` and `body` may be null (the source may omit them); the six
key/required columns (`source`, `article_id`, `symbol`, `published_at`, `knowable_at`,
`headline`) are non-null. Null text is canonicalized with a dedicated sentinel in the logical
hash (§4) so snapshot identity is stable.

## 4. Schema, validator, storage

**Column-name constants live in `algua/contracts/types.py`** (same rationale as fundamentals —
the base layer a future engine mask *can* import without crossing the wall), beside
`FUNDAMENTALS_COLUMNS`: `NEWS_COLUMNS`, `NEWS_AS_OF_KEY = ("source", "article_id", "symbol")`,
`NEWS_KNOWABLE_AT = "knowable_at"`.

New module `algua/data/news_schema.py`, parallel to `algua/data/fundamentals_schema.py`:

- `NEWS_COLUMNS` — canonical column list/order (built from the contracts constants).
- `validate_news(frame)` — enforces:
  - exact columns/order/dtypes; `source`/`article_id`/`symbol`/`headline` non-null strings;
    `url`/`body` nullable strings; `published_at` and `knowable_at` **tz-aware UTC and
    non-null**;
  - **PIT floor:** `knowable_at >= published_at` (a correction cannot become knowable before
    the thing it corrects was published; a sanity floor, not a precise model);
  - **unique row key** `(source, article_id, symbol, knowable_at)` — no two rows collide, so a
    future as-of mask has a unique winner per `(source, article_id, symbol)`;
  - **two-level article consistency** (GATE-1, Codex + OpenCode): `published_at` is invariant
    per `(source, article_id)` (the publication instant is an article-identity attribute, not a
    per-revision one); `headline`/`url`/`body` are invariant per `(source, article_id,
    knowable_at)` (one article revision, exploded across symbols). Catches an inconsistent
    hand-built input;
  - **canonical sort** by `(symbol, source, article_id, knowable_at)` (a total order over the
    unique key; the same order the hindsight CLI emits, §5) + no exact-duplicate rows.

  Two normalization entry points keep ingest and read consistent (GATE-1, Codex — a single
  exploding normalizer would be non-idempotent and break readback):

- `explode_news_symbols(raw_frame)` — **ingest-only pre-step.** Takes raw input carrying a
  `symbols` field (a real list in parquet, or a comma-delimited string like `"AAPL, MSFT"` in
  CSV); splits on comma, strips whitespace, upper-cases, drops blanks, de-dupes symbols within
  an article, and emits one row per `(article-row, symbol)` with a canonical `symbol` column.
  **Rejects an article with zero resulting symbols** (fail closed — symbol-less / macro news is
  an explicit scope cut, §7, not a silent drop).
- `to_news_schema(frame)` — **idempotent canonical normalizer**, run by **both** `ingest_news`
  (after the explode) **and** `read_news`. Expects an already-exploded per-`symbol` frame.
  Canonicalizes `symbol` (strip + upper) and **`source` (strip + lower)** — since `source` is
  part of the identity key, an un-canonicalized `Reuters`/`reuters` would fragment identity
  (GATE-1, OpenCode); coerces dtypes; parses `published_at`/`knowable_at` and **normalizes to
  UTC** (rejects tz-naive, matching fundamentals); **requires `knowable_at`** — a missing/empty
  value is rejected, never defaulted to `published_at` (GATE-1, Codex: defaulting back-dates a
  revised archive row and corrupts the bitemporal record); **canonicalizes nullable `url`/`body`
  to a single null representation** (`pd.NA`) distinct from the empty string (GATE-1, OpenCode);
  drops byte-identical duplicate rows; canonical-sorts; then `validate_news`. Idempotent on a
  canonical frame, so `read_news` can re-run it safely.
- `empty_news()` — contract-shaped empty frame (returned when a read yields nothing).
- `logical_news_hash(frame)` — deterministic content hash over sorted logical rows, using the
  same length-prefixed-string / int64-ns-UTC discipline as `logical_fundamentals_hash` (and
  `logical_bars_hash`). Nullable `url`/`body` use a **dedicated null sentinel byte** chosen so
  that `null`, the empty string `""`, and the literal string `"None"` all hash **distinctly**
  (GATE-1, Gemini: never `str(None)`); tested explicitly.

**Storage substrate.** Reuse the snapshot/manifest/content-hash machinery wholesale, as
fundamentals did:

- `algua/data/models.py`: add `Dataset.NEWS = "news"` and `Kind.NEWS = "news"`.
  `SnapshotMetadata`/`SnapshotRecord` reused as-is.
- `algua/data/store.py`: `ingest_news(...)` (`explode_news_symbols` → `to_news_schema` → assert
  non-empty → **assert every `knowable_at <= as_of`** → **derive** `start`/`end` from the data
  (`min`/`max` of `knowable_at`) and the covered symbol/source sets from the data, never trusting
  declared coverage → write tidy parquet under `snapshots/news/<id>/news.parquet` → append
  manifest, dedup on snapshot id) and `read_news(snapshot_id, *, symbols=None)` (read back →
  `to_news_schema` (idempotent) → symbol pushdown → `empty_news()` when empty). Both are
  unreachable from every decision lane via the §2.2 wall.
- **Snapshot metadata for a multi-source dataset (GATE-1, Codex).** `SnapshotMetadata` is reused
  unchanged: `metadata.source` is set to the **`--provider`** ingest label (who ran the ingest —
  a single deterministic value), and the **derived sorted distinct row-`source` set and symbol
  set** are recorded in `source_metadata` (the existing free-form provenance dict). This removes
  the provenance ambiguity (one well-defined home for the per-item sources) without a schema
  migration.
- Local `data_dir` + manifest now; **cloud lift deferred** (especially relevant for `body`
  text at scale — see §7 / §12).

There is **no** `serve.py` provider for news this slice (no signal lane).

## 5. CLI

Both emit JSON on stdout (the agent/operator seam), mirroring the fundamentals commands:

- `algua data ingest-news --from-file PATH --provider P --as-of TS` — reads a local file
  (CSV/parquet) whose rows carry `source`, `article_id`, `symbols`, `published_at`,
  `knowable_at`, `headline`, and optionally `url`/`body`; normalizes (incl. the explode) +
  validates via `to_news_schema`; asserts `knowable_at <= as_of`; writes one news snapshot.
  Output `{"ok": true, "snapshot": {...}}`. **`source` is a required per-row column**, so there
  is **no `--source` flag** (GATE-1, Codex: a CLI `--source` plus a row `source` plus snapshot
  metadata is three ambiguous provenance sources); likewise **no `--symbols` flag** — the
  covered symbol/source sets are *derived* from the exploded data. `--provider` remains the
  snapshot-level ingest label (who ran the ingest), distinct from per-row `source` (which wire
  published the item). (A vendor-API news ingest, like `ingest-bars`, is a later addition.)
- `algua data query-news --snapshot-id ID [--symbols S]` — the **hindsight** read (full
  history). **Stable JSON:** canonical row order `(symbol, source, article_id, knowable_at)`,
  ISO-8601 UTC timestamps, `null` for absent `url`/`body`, so agent post-mortems are
  reproducible/diffable. This is the agent's analysis/error-diagnosis surface.

`algua data inspect` already lists snapshots generically; only the new dataset routing is
needed.

## 6. The wall — one contract + a test

§2.2 adds one `lint-imports` contract (forbid `algua.live`/`algua.execution` from
`algua.data`); the standing contracts (`algua.data.hindsight` unreachable from decision lanes;
strategies/contracts off `algua.data`) are unchanged. To make the property a **guarded
invariant**, add a test (parallel to `test_fundamentals_wall.py`):

- assert the **new** contract is present in `pyproject.toml` with `algua.live`/`algua.execution`
  forbidden from `algua.data`, alongside the standing decision-lane bans — so the complete set
  of decision/execution lanes is provably off the data lane;
- assert `query_news` is defined in the walled `algua.data.hindsight` module, and that the only
  full-history news read surfaces (`store.read_news`, `hindsight.query_news`) sit behind the
  wall (a regression guard against a future news-read accessor landing in a non-walled module
  or an un-walled lane) (GATE-1, Codex LOW).

The real `lint-imports` in the quality gate enforces the static property on every run.

## 7. Deliberate scope cuts

- **Hindsight lane only.** No as-of signal lane: no `NewsProvider`, no `needs_news`, no engine
  masking, no `compute_weights` news argument, no paper/live/promote guards (nothing to guard).
  News is structurally unreachable from a backtest or live decision (§2.2).
- **Entity→symbol = explode given symbols.** The ingest input supplies each article's symbols;
  the pipeline explodes + normalizes them. **No NER** (resolving company names from free text)
  and **no PIT-versioned ticker remapping** (handling ticker reassignments over time) — both
  deferred. The schema stores the normalized trading `symbol`; we do **not** claim it is
  forward-compatible with PIT remapping (GATE-1, Codex) — that follow-up may require re-ingest
  with a richer source/entity-id schema.
- **Content revisions only.** A later-`knowable_at` row revises an existing
  `(source, article_id, symbol)`'s headline/body/url. **Symbol-set revisions** (a correction
  adding/removing tagged tickers) need tombstones / full-replacement modeling — deferred with
  the signal lane (§3).
- **Dedup is structural.** Exact-duplicate-row drop + `(source, article_id, symbol, knowable_at)`
  uniqueness only. **No fuzzy / semantic cross-source dedup** (clustering an AP reprint with the
  Reuters original) — deferred; cross-source items legitimately coexist as distinct rows.
- **At least one symbol per article.** Symbol-less / macro news is out of scope (the record is
  symbol-keyed); such an article is rejected at ingest, not silently dropped.
- **Multi-source supported.** `source` is part of the identity; multiple wires coexist in one
  snapshot. Cross-source *reconciliation* (deciding two items are the same story) is the
  deferred fuzzy-dedup follow-up.
- **File-based ingest only** — vendor-API news ingest deferred.
- **`body` size unbounded this slice.** Large `body` text inflates local parquet and
  full-hindsight reads; a body-length cap or a body-excluding `query-news` projection is a
  deferred scale concern tied to the cloud substrate (§12) (GATE-1, Gemini), not built now.
- **Wall limits unchanged** — static import graph; adversarial dynamic exfiltration remains the
  shared, out-of-scope trust-boundary limit.

## 8. Module / file plan

| Area | File | Change |
|---|---|---|
| Column-name constants | `algua/contracts/types.py` | add `NEWS_COLUMNS`, `NEWS_AS_OF_KEY=("source","article_id","symbol")`, `NEWS_KNOWABLE_AT` (names only, no pandas) |
| Schema + validator | `algua/data/news_schema.py` | new: `NEWS_COLUMNS`, `validate_news` (PIT floor, unique key, two-level article consistency), `explode_news_symbols` (ingest-only `symbols`→rows), `to_news_schema` (idempotent canonical normalize: source/symbol canonicalization, required-knowable_at, null-canonicalization, dedup, UTC, sort, validate), `empty_news`, `logical_news_hash` (dedicated null sentinel) |
| Models | `algua/data/models.py` | add `Dataset.NEWS`, `Kind.NEWS` |
| Store | `algua/data/store.py` | `ingest_news` (explode→normalize; require knowable_at; assert `knowable_at<=as_of`; `metadata.source`=provider, derived sources/symbols in `source_metadata`), `read_news` (idempotent re-normalize) |
| Hindsight | `algua/data/hindsight.py` | add `query_news` (wraps `read_news`; full history, no `as_of`, stable order) |
| CLI | `algua/cli/data_cmd.py` | `ingest-news` (no `--source`/`--symbols`), `query-news` (stable JSON) |
| **Wall** | `pyproject.toml` | **new contract: forbid `algua.live`/`algua.execution` from `algua.data`** (completes the data wall; also closes the `read_fundamentals` bypass) |
| Wall test | `tests/test_news_wall.py` | assert the new contract + standing contracts; `query_news` in `algua.data.hindsight`; full-history read surfaces are walled |
| Contract doc | `docs/contracts/news-schema.md` | new, parallel to `fundamentals-schema.md` |

No changes to `algua/backtest/`, `algua/strategies/`, `algua/registry/`, `algua/live/`,
`algua/execution/`, or `serve.py`. (The only `pyproject.toml` change is wall-tightening.)

## 9. Testing

Following the fundamentals test patterns (`tests/test_fundamentals_schema.py`,
`tests/test_data_fundamentals_store.py`, `tests/test_fundamentals_serve_hindsight.py`,
`tests/test_cli_data_fundamentals.py`, `tests/test_fundamentals_wall.py`):

- **Validator** — conformant passes; each violation rejected separately: missing/extra column,
  wrong dtype, tz-naive `published_at`/`knowable_at`, null in a required column,
  `knowable_at < published_at` (incl. the equal-instant *pass* case), bitemporal-key collision,
  article-revision inconsistency (same `(source, article_id, knowable_at)`, differing
  headline/url/body), exact-duplicate rows. Nullable `url`/`body` pass as null and as text.
  `empty_news()` is contract-shaped.
- **Multi-source identity** — two sources reusing the same `article_id` for different stories
  **coexist** (distinct `(source, article_id, symbol)`); they do not collide or masquerade as
  revisions of each other.
- **Explode (`explode_news_symbols`)** — an article with `symbols="AAPL, MSFT"` (and a list
  form) yields two rows sharing article fields; symbols upper-cased/stripped, blanks dropped,
  within-article duplicates collapsed; an article with **zero symbols is rejected**.
- **Normalizer (`to_news_schema`)** — `source` canonicalized (`Reuters`→`reuters`), `symbol`
  upper-cased; a **missing `knowable_at` is rejected** (not defaulted); tz-naive timestamps
  rejected; null vs empty-string `url`/`body` preserved distinctly (`pd.NA` vs `""`);
  byte-identical rows dropped; **idempotent** — `to_news_schema(to_news_schema(x))` equals
  `to_news_schema(x)` (so `read_news`'s re-normalization is safe).
- **Content revision** — two rows sharing `(source, article_id, symbol)` with different
  `knowable_at` (original + correction) both validate and coexist, distinguished by
  `knowable_at`.
- **Logical hash** — identical logical content → identical hash regardless of input row order;
  `null` `body`, empty-string `body`, and literal `"None"` `body` all hash **differently**
  (dedicated sentinel); a changed headline changes the hash.
- **Store roundtrip** — `ingest_news` → `read_news` passes `validate_news`; identical input →
  identical `snapshot_id` (dedup); `knowable_at > as_of` is **rejected** at ingest; `start`/`end`
  + covered symbols/sources are derived from the data; symbol pushdown returns only requested
  symbols.
- **Hindsight** — `query_news` returns full history (no masking) in stable canonical order;
  symbol filter honored; empty result is `empty_news()`-shaped.
- **CLI** — `ingest-news` then `query-news` roundtrip; hindsight JSON is stable-ordered with
  ISO-8601 UTC timestamps and `null` for absent `url`/`body`; ingest rejects a file missing
  `source`/`knowable_at`.
- **Wall** — `test_news_wall.py`: the new `live`/`execution`-off-`algua.data` contract is
  present alongside the standing decision-lane bans; `query_news` resides in
  `algua.data.hindsight`; no full-history news read surface is reachable from a decision lane.
  (The real `lint-imports` gate enforces it.)

## 10. Gate

`uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports` green.

## 11. GATE-1 review log (multi-model design review)

Panel: **Codex** (GPT-5-codex, deep correctness/wall lens) + **Gemini 2.5 Flash** (broad
sweep). **OpenCode/GLM** produced no usable output this run (first attempt crashed on a
tool-permission error before emitting findings; a re-run hit a model-quota wall) — Codex +
Gemini are the default panel and gave strong coverage. Both verdicts: **REVISE**. Findings
triaged once on merit (`superpowers:receiving-code-review`); accepted ones are folded into the
sections above.

**Accepted / folded in:**
- **[CRITICAL, Codex]** Full-history bypass: `algua.live`/`algua.execution` were not barred from
  `algua.data`, so they could `import algua.data.store` and call `read_news` (unmasked) without
  touching the walled `query_news`. → §2.2/§6/§8 add a contract forbidding `live`/`execution`
  from `algua.data` (verified to break nothing; also closes the same `read_fundamentals` gap).
- **[HIGH, Codex]** `source` excluded from the key contradicts multi-source coexistence
  (`article_id` is source-scoped). → §3/§4 add `source` to `NEWS_AS_OF_KEY` and the unique row
  key.
- **[HIGH, Codex]** Per-symbol explode + future keep-latest can't represent symbol *removals*;
  "drops in without migration" over-claimed. → §3/§7 narrow the claim to **content** revisions;
  defer symbol-set-revision semantics (tombstones) with the signal lane.
- **[HIGH, Codex]** Defaulting `knowable_at = published_at` back-dates revised archive content.
  → §3/§4 **require explicit `knowable_at`**; reject missing.
- **[HIGH, Gemini]** Null-text determinism in `logical_news_hash`. → §4 mandate a dedicated null
  sentinel distinct from `""`/`"None"`; §9 tests it.
- **[MEDIUM, Codex]** Article-level consistency unvalidated. → §4 add the
  `(source, article_id, knowable_at)` group-consistency check.
- **[MEDIUM, Codex]** CLI source provenance ambiguity (`--source` vs row vs metadata). → §5
  `source` is a required row column; **drop `--source`**; keep `--provider` as the ingest label.
- **[MEDIUM, Codex]** PIT-remap forward-compat. → §7 narrow the claim (no zero-migration promise
  for ticker remapping).
- **[LOW, Codex]** Wall test must cover the complete read surface, not just `query_news`'s
  location. → §6/§9 assert the new contract + that no decision lane reaches any full-history
  news read.

**Declined (with rationale):**
- **[MEDIUM, Gemini]** "Add a placeholder/dummy news signal lane for architectural symmetry." —
  Declined: that is exactly the speculative build §1 deliberately avoids; a dead passthrough
  signal lane plus a fake wall surface is strictly worse than its clean absence. The §1
  rationale was strengthened to address the precedent-divergence concern head-on instead.

**Deferred (with rationale):**
- **[MEDIUM, Gemini]** Large-`body` performance in local parquet / full reads. — Real but tied
  to the already-deferred cloud substrate; noted as a scope cut (§7) + follow-up (§12) with a
  body-excluding projection as the likely lever; not built in a file-ingest foundation slice.

### Round 2 (Codex deep re-review of the revised spec; OpenCode round-1 re-run also landed)

The revised spec was re-reviewed. Codex confirmed all round-1 fixes are coherent end-to-end
(`source` consistently in key/sort/query/dedup; `knowable_at` consistently required; content
revisions sound; null-hash intent coherent; CLI `--source`/`--symbols` removal reflected) and
raised three more; OpenCode's late round-1 output added four precision items. All accepted ones
are now folded above.

**Accepted / folded in (round 2):**
- **[HIGH, Codex]** A single exploding `to_news_schema` is non-idempotent and breaks readback
  (canonical parquet has `symbol`, not `symbols`). → §4 splits `explode_news_symbols`
  (ingest-only) from an **idempotent** `to_news_schema` (run by ingest *and* read), mirroring
  `to_fundamentals_schema`.
- **[HIGH, Codex]** `SnapshotMetadata.source` left undefined for a multi-source dataset after
  `--source` was dropped. → §4 sets `metadata.source = --provider` and records the derived
  sorted row-source + symbol sets in `source_metadata`.
- **[MEDIUM, OpenCode]** Nullable `url`/`body` canonicalization in the normalizer (not just the
  hash). → §4 `to_news_schema` canonicalizes nulls to a single `pd.NA` distinct from `""`.
- **[MEDIUM, OpenCode]** Symbol-list delimiter underspecified. → §4 `explode_news_symbols`
  specifies list-or-comma-delimited, strip/upper/drop-blank/de-dupe.
- **[MEDIUM, OpenCode]** `published_at` not validated invariant within an article. → §4 makes
  the consistency check two-level (`published_at` per `(source, article_id)`; content per
  `(source, article_id, knowable_at)`).
- **[LOW, OpenCode]** `source` not canonicalized — now key-relevant. → §4 `to_news_schema`
  canonicalizes `source` (strip + lower).

**Confirmed, not a spec change:**
- **[CRITICAL, Codex]** "The `live`/`execution`-off-`algua.data` contract isn't in
  `pyproject.toml`." — Correct: this is a *design* spec; the contract does not yet exist in the
  repo. Codex verified current imports are clean (so the planned contract breaks nothing) and
  that the gap is real until implemented. The contract is specified (§2.2/§6/§8); adding it is a
  **load-bearing implementation task** (the plan makes it task #1 of the wall work), and the
  `lint-imports` gate + the wall test enforce it.

**Convergence:** the wall gap was independently flagged by **both Codex (CRITICAL) and OpenCode
(HIGH)** — high confidence — and is resolved in the design. After round 2 no *accepted,
unapplied* design finding remains; the residue is the implementation tasks the spec already
enumerates. GATE-1 is **APPROVED** (design level); correctness of the wall contract and the
idempotent read path will be re-checked on the diff at GATE-2.

## 12. Deferred follow-ups (out of this slice)

- **As-of news signal lane** — the symmetric build to fundamentals' deferred paper/live:
  `NewsProvider` + engine `_news_as_of` masking + `needs_news` opt-in + fail-closed
  paper/live/promote guards + loop-forcing. Where the unsettled "what does `compute_weights`
  receive from a headline" question (raw rows? a scored feature? an event table?) gets answered
  by a concrete strategy. Must also settle **symbol-set revision** semantics (tombstones /
  article-level full-replacement).
- **Fuzzy / semantic cross-source dedup** — cluster near-duplicate stories across wires
  (threshold-sensitive; better motivated once a consumer exists).
- **NER entity resolution** — resolve issuer mentions from free text to symbols, and
  **PIT-versioned ticker mapping** for reassignments over time (likely needs a richer
  source/entity-id schema — a re-ingest, not an in-place migration).
- **Symbol-less / macro news** — a non-symbol-keyed channel (market-wide headlines).
- **`body`-size handling** — a length cap and/or a body-excluding `query-news` projection for
  cheap reads.
- **Vendor-API news ingest** (analogous to `ingest-bars`).
- **Cloud storage substrate** — per architecture spec §8; sharper for news `body` text at scale
  than for fundamentals.
