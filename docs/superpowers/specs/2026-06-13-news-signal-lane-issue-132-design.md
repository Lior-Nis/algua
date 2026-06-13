# News signal lane — issue #132 (slice 3)

**Status:** design
**Date:** 2026-06-13
**Issue:** #132 — non-tabular data (news / fundamentals): typed seam, point-in-time discipline.

## Context

Issue #132 has two merged slices:

- **Fundamentals seam** (PR#154) — the full vertical: a `FundamentalsProvider` as-of
  protocol, a `StoreBackedFundamentalsProvider`, the `needs_fundamentals` signal lane
  threaded through the backtest engine with per-bar `knowable_at <= t` masking, a schema
  validator, and a hindsight `query_fundamentals` accessor walled off from decision lanes.
  Paper/live **fail closed**.
- **News seam** (PR#176) — **hindsight-only**: a tidy bitemporal news schema + validator,
  `store.read_news`, and the hindsight `query_news` accessor. It **completed the data wall**
  (every decision/execution lane is now barred from importing `algua.data`). It has **no**
  as-of consumption seam: no `NewsProvider`, no `needs_news`, no engine consumer.

This slice builds the **news signal lane** — the as-of consumption path that lets a strategy
read point-in-time news inside `compute_weights`, mirroring the merged fundamentals lane.

## Goals

1. A strategy may declare `needs_news=True` and author a 3-arg `signal(view, params, news)`;
   the engine injects the PIT-correct news frame per bar.
2. News visible to a decision at bar `t` is exactly the news **knowable at `t`** — the same
   structural look-ahead defense the fundamentals lane has.
3. Close the **symbol-set-revision (retraction) gap**: a symbol dropped by a later article
   revision must stop being visible from that revision's `knowable_at` onward.
4. Paper/live remain **fail-closed** for `needs_news` (the as-of lane is wired only into the
   backtest engine this slice — exactly as fundamentals shipped).

## Non-goals (deferred, documented)

- **News in paper/live** — fail-closed now; lifting the guard is a follow-up (parallel to the
  fundamentals paper/live deferral).
- **walk_forward / sweep threading + a vectorized `signal_panel` fast path for news** — both
  deferred. `signal_panel` is rejected at load for a `needs_news` strategy. **But** WF/sweep
  must **fail closed with a clear error** for a `needs_news` strategy rather than the confusing
  deep `BacktestError` they raise today — and this slice adds the *same* clear guard for
  `needs_fundamentals` (which currently crashes opaquely in WF/sweep — see §11). Threading the
  providers through WF/sweep is the lift-the-guard follow-up.
- **A strategy needing BOTH news and fundamentals** — rejected at load this slice (the chosen
  "parallel 3-arg, mutually exclusive" contract shape). A unified PIT-context arg is a future
  refactor if alt-data multiplies.
- **Cross-snapshot retraction reconciliation** — tombstones are derived **within a single
  ingest frame's revision history** (see §4). A correction arriving in a separate later
  snapshot is not auto-linked; this matches the immutable, self-contained snapshot model and
  the backtest pinning to exactly one news snapshot.
- Fuzzy dedup, NER/PIT ticker remap, macro news, vendor ingest, cloud storage — already
  deferred from the hindsight slice; untouched here.

## Design

The lane is a near-exact mirror of the fundamentals lane. Each component below names its
fundamentals twin.

### 1. Contract layer — `algua/contracts/types.py`

- Add `NEWS_RETRACTED = "retracted"` and append `"retracted"` to `NEWS_COLUMNS` (the canonical
  source of truth the engine and the data validator share — neither imports the other).
- Add the as-of consumption protocol (twin of `FundamentalsProvider`):

  ```python
  @runtime_checkable
  class NewsProvider(Protocol):
      """As-of consumption seam for point-in-time news (issue #132). Returns the FULL
      bitemporal history for `symbols` with knowable_at < end — no lower bound. The engine
      owns decision `t` and masks knowable_at <= t per bar; the provider never sees `t`."""
      snapshot_id: str
      def get_news(self, symbols: list[str], end: datetime) -> pd.DataFrame: ...
  ```

### 2. Schema layer — `algua/data/news_schema.py`

Add `retracted` as a **non-null `bool`** column.

- `validate_news`: assert `retracted` is numpy `bool` dtype (NOT nullable `pd.BooleanDtype`),
  non-null. The existing per-revision invariants (`published_at` per `(source, article_id)`;
  `headline`/`url`/`body` per `(source, article_id, knowable_at)`) and the unique key
  `(source, article_id, symbol, knowable_at)` are unchanged — a tombstone is a normal-shaped
  row that happens to carry `retracted=True`.
- `empty_news`: add `retracted` as `pd.Series([], dtype="bool")` (numpy bool, not nullable).
- `logical_news_hash`: hash `retracted` as a `u1` byte column (a tombstone hashes distinctly,
  and the hash changes for all news data — see the clean-break note below).
- `to_news_schema`: **require** `retracted` (it is in `COLUMNS`, so the existing
  missing-column check rejects a frame without it) and coerce it to numpy `bool`. **No
  default-fill / back-compat path** — `explode_news_symbols` always produces `retracted`
  explicitly, and read re-normalizes a frame this slice wrote.

**Clean break (no migration shim):** adding `retracted` to `NEWS_COLUMNS` changes the schema
and `logical_news_hash`, so news snapshots written by the hindsight slice are no longer
readable and would re-hash to new ids. This is acceptable and deliberate: `data_dir` is
git-ignored and ephemeral, re-ingest is the path, and it mirrors the fundamentals slice's
`config_hash` clean break. We do **not** add a legacy default-fill read path (consistent with
the project's no-backwards-compat-cruft norm).

### 3. Tombstone generation — `explode_news_symbols`

**Input contract (stated explicitly + enforced):** each input row is a *full article
revision* — one `(source, article_id, knowable_at)` carrying that revision's **complete**
symbol set as-of that `knowable_at`. Multiple revisions of one article are multiple input rows.
`explode_news_symbols` **rejects** a frame with a duplicate `(source, article_id, knowable_at)`
(two rows claiming to be the same revision) — the revision walk would otherwise be ambiguous
and could invent or suppress tombstones. The duplicate check + the grouping key are computed on
the **canonicalized** identity (`source` stripped+lower, `article_id` stripped, `knowable_at`
UTC-normalized) — the same normalization `to_news_schema` applies — so `Reuters` vs ` reuters `
or two encodings of the same instant cannot bypass the guard. The walk operates on the **parsed
`_syms` lists BEFORE the explode** (the lists are the per-revision sets; after explode they're
gone).

Algorithm, per `(source, article_id)` ordered by `knowable_at`:

1. Parse each revision's `symbols` → a set; explode the **non-empty** sets into normal rows
   (`retracted=False`).
2. For each revision `K_i` after the first, compute `dropped = symbols(K_{i-1}) - symbols(K_i)`
   (full-restatement semantics — a revision republishes the authoritative set). `symbols(K_i)`
   may be **empty** for a non-first revision (a full retraction of the article — every prior
   symbol is dropped); only the **first/only** revision of an article must be non-empty
   (symbol-less news is out of scope — today's rule, kept for the first revision).
3. For each `x in dropped`, synthesize a **tombstone** row:
   `(source, article_id, symbol=x, published_at, knowable_at=K_i, headline_i, url_i, body_i,
   retracted=True)`. It carries revision `K_i`'s content, so the per-revision invariants hold,
   and `x` is absent from `K_i`'s normal rows, so the unique key never collides.
4. Reject a revision whose symbol set is empty **when the running set is already empty** (an
   empty revision after a full retraction retracts nothing and adds nothing — a malformed
   no-op); this keeps every persisted revision meaningful and validatable.

A symbol dropped then re-added produces `present(K_a) → tombstone(K_b) → present(K_c)`; the
as-of mask (§5) resolves each `t` correctly because it picks the **latest revision per key**.
A full retraction at `K_i` tombstones every symbol the article last carried.

### 4. Data serving — `algua/data/serve.py`

`StoreBackedNewsProvider` (twin of `StoreBackedFundamentalsProvider`): reads one news
snapshot via `store.read_news`, returns the full bitemporal history (**including tombstones**)
with `knowable_at < end`. Never sees `t`. `symbols` filter pushed to `read_news`.

### 5. Engine — `algua/backtest/engine.py`

- `_assert_news_shape(frame)` (twin of `_assert_fundamentals_shape`): contract columns present,
  `knowable_at` **and `published_at`** tz-aware UTC + non-null, the PIT floor
  `published_at <= knowable_at` (strategies window on `published_at`, so a foreign provider must
  not slip a future publication time past an old `knowable_at`), `retracted` numpy bool, unique
  `(source, article_id, symbol, knowable_at)`. Fails closed for a foreign provider.
- `_news_as_of(frame, t)` (twin of `_fundamentals_as_of`): of rows with `knowable_at <= t`,
  keep for each `(source, article_id, symbol)` the latest revision (greatest `knowable_at` —
  unique per key, so deterministic), then **drop `retracted=True` rows** from the as-of view.
  Uses only `knowable_at <= t` → no look-ahead. Empty-in/empty-out returns
  `frame.iloc[0:0].copy()` (preserves the `retracted` bool + tz-aware dtypes), never a view into
  future rows. The strategy windows on `published_at` itself.
- Thread `news_provider: NewsProvider | None` through `simulate` / `build_portfolio` / `run`
  (mirrors `fundamentals_provider`). `_decision_weights` gains a `news` param; the `needs_news`
  branch masks to the as-of-t members (same membership handling as fundamentals) and calls
  `strategy.target_weights(view, news=news_asof)`.
- `_decision_weights_fast_or_loop` forces the per-bar loop when **`news is not None`** (mirrors
  the `fundamentals is not None` guard) — defense-in-depth so a directly-constructed
  `needs_news` `LoadedStrategy` can never reach the news-blind vectorized path even if the
  loader guard is bypassed.
- `simulate` raises `BacktestError` if `needs_news` but no `news_provider` (fail closed).

### 6. Strategy adapter / loader — `algua/strategies/base.py`, `loader.py`

- `StrategyConfig.needs_news: bool = False`.
- `NewsSignalFn = Callable[[pd.DataFrame, dict, pd.DataFrame], pd.Series]`; add
  `LoadedStrategy.news_signal_fn`.
- `__post_init__`: enforce mutual exclusion **structurally**, not just via the flags — among the
  **three decision fns** `{signal_fn, fundamentals_signal_fn, news_signal_fn}` (NOT
  `signal_panel_fn`, which is an accelerator paired with an ordinary `signal_fn`) require
  **exactly one** non-None, and require that one to match the config (`needs_fundamentals` ⇒
  only `fundamentals_signal_fn`; `needs_news` ⇒ only `news_signal_fn`; neither ⇒ only
  `signal_fn`, with `signal_panel_fn` allowed). Reject `needs_fundamentals and needs_news`
  together. This rejects a stray sidecar fn that the flags alone would silently ignore.
- `target_weights(features, fundamentals=None, news=None)` and `signal(...)` route to the news
  fn when `needs_news`. The adapter **fails closed** if the active sidecar is missing OR a
  non-active sidecar is passed non-None (so a news frame can never reach a fundamentals strategy
  or vice-versa). At most one sidecar is ever non-None.
- The `Strategy` protocol's `target_weights` signature in `contracts/types.py` is extended to
  `(features, fundamentals=None, news=None)` so the protocol does not lag the adapter.
- `authored_signal` returns the active fn; `config_hash` folds in `needs_news`
  (`"needs_news": strategy.config.needs_news` in the payload dict).
- `assert_tradable_without_news(strategy)` (twin of `assert_tradable_without_fundamentals`):
  fail closed at every paper/live load point.
- `loader.load_strategy`: `needs_news` → require a 3-arg `signal`, reject `signal_panel`
  (no vectorized news fast path yet), reject co-declared `needs_fundamentals`.

### 7. Promotion guard — `algua/registry/promotion.py`

Parallel `needs_news` block: a `needs_news` strategy cannot be promoted past `backtested`
until the paper/live news lane exists (same message shape as the fundamentals guard).

### 8. CLI — `algua/cli/backtest_cmd.py`

Add `--news-snapshot` and wire `StoreBackedNewsProvider` through to `run_backtest`
(mirrors `--fundamentals-snapshot`).

### 9. Result — `algua/backtest/result.py`

Add `news_snapshot` field, stamped from the news provider's `snapshot_id` **only when the lane
is active** (`strategy.config.needs_news`) — so a `--news-snapshot` passed to a non-`needs_news`
strategy (which the engine ignores) does not produce a provenance record claiming it was used.
To make that misuse loud rather than silent, the CLI **errors** if `--news-snapshot` is given
for a strategy that does not declare `needs_news`.

### 10. Example strategy — `algua/strategies/news/`

A minimal `needs_news=True` example: a recent-headline-**count** tilt (count each symbol's
headlines in a trailing window from the as-of `news` frame; long the most-covered names). No
sentiment/NLP — it exercises the lane end-to-end and gives the agent a template, exactly as
`fundamentals_earnings_tilt` does for fundamentals.

### 11. walk_forward / sweep fail-closed guard — `algua/backtest/walkforward.py`, `sweep.py`

`walk_forward` and `sweep` call `build_portfolio` / `walk_forward` **without** a provider, so a
`needs_*` strategy currently dies with the deep `BacktestError` from `simulate`. Add an explicit
guard at the entry of `walk_forward` and `sweep`: if `strategy.config.needs_news` **or**
`needs_fundamentals`, raise a clear `BacktestError` ("`needs_news`/`needs_fundamentals`
strategies are not supported in walk-forward/sweep yet — #132 follow-up"). This converts an
opaque crash into an intelligible refusal for **both** lanes (fixing the pre-existing
fundamentals rough edge while we're here). Threading the providers through WF/sweep is the
deferred follow-up. Also: `sweep`'s `_override` rebuilds `LoadedStrategy` — it must copy
`news_signal_fn` alongside `fundamentals_signal_fn` so the field is faithfully preserved even
though the guard makes a `needs_news` sweep currently unreachable.

## Look-ahead / correctness argument

- **No look-ahead:** the provider returns history with `knowable_at < end` and never sees `t`;
  the engine's per-bar mask uses only `knowable_at <= t`. Identical to the fundamentals lane.
- **Headline corrections** are same-key later revisions → the latest-revision-per-key mask
  serves the corrected headline. **Symbol additions** appear as a new key at their revision
  time. **Symbol retractions** are closed by tombstones (§3/§5) — the dropped mention's latest
  revision-as-of-`t` is a tombstone, which the mask excludes.
- **Hindsight stays walled:** `query_news` lives in `algua.data.hindsight`, already
  unreachable from `backtest/features/contracts/strategies/live/execution` by import-linter.
  The as-of `StoreBackedNewsProvider` lives in `algua.data.serve`; the engine imports only the
  `NewsProvider` protocol from `contracts` — the CLI injects the concrete provider. No new
  import-linter contract is required; the existing walls cover the new code.

## Testing

- **Schema:** tombstone generation across revisions (drop, drop-then-re-add, multi-symbol,
  **full retraction** — empty later revision tombstones all prior symbols), rejection of a
  zero-symbol first/only revision, rejection of an empty-after-empty revision, rejection of
  duplicate `(source, article_id, knowable_at)` input revisions **including case/whitespace/tz
  variants that canonicalize equal**, tombstone `published_at` == article `published_at` (not
  `knowable_at`),
  `retracted` validation (numpy-bool/non-null), hash distinctness, `empty_news` shape,
  idempotent `to_news_schema` round-trip including `retracted`.
- **Engine:** `_news_as_of` keeps latest revision per key, excludes tombstones, respects
  `knowable_at <= t` (no leak of a record knowable only after `t`), empty-in/empty-out preserves
  dtypes; `_assert_news_shape` rejects a foreign provider with `published_at > knowable_at` /
  naive ts / non-bool `retracted`; the fast-path selector forces the loop when `news is not
  None`; end-to-end `simulate` with a `needs_news` strategy; fail-closed when provider absent.
- **Adapter/loader:** structural mutual exclusion (`needs_fundamentals and needs_news` rejected;
  a stray non-active sidecar fn rejected; wrong/missing sidecar at call time rejected), arity
  enforcement, `signal_panel` rejected, `config_hash` changes with `needs_news`.
- **Serve:** `StoreBackedNewsProvider` honors `knowable_at < end` and the `symbols` filter,
  returns tombstones.
- **Guards:** `assert_tradable_without_news` raises at paper/live load; promotion blocked past
  `backtested`; `walk_forward`/`sweep` raise the clear fail-closed guard for `needs_news` **and**
  `needs_fundamentals`.
- **CLI:** `--news-snapshot` wires the provider; the example strategy backtests green.

## Quality gate

`uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports` green.
