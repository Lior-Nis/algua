# News Signal Lane (issue #132) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the as-of NEWS consumption lane — a `needs_news` strategy reads point-in-time news inside its signal, with the same structural look-ahead defense the merged fundamentals lane has, plus symbol-set-revision tombstones.

**Architecture:** Mirror the merged fundamentals vertical exactly. The data layer gains a `retracted` column + tombstone generation; `contracts` gains a `NewsProvider` protocol; `algua.data.serve` gains a `StoreBackedNewsProvider`; the backtest engine gains a per-bar `knowable_at <= t` news mask; the strategy adapter/loader gain a `needs_news` 3-arg signal lane (mutually exclusive with `needs_fundamentals`). Paper/live and walk_forward/sweep fail closed.

**Tech Stack:** Python 3.12, pandas, pydantic, typer, pytest. The walls are enforced by import-linter (already covers the new code — see spec §5).

**Spec:** `docs/superpowers/specs/2026-06-13-news-signal-lane-issue-132-design.md`

**Quality gate (run between tasks):** `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`

---

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `algua/contracts/types.py` | modify | `NEWS_RETRACTED` const + `"retracted"` in `NEWS_COLUMNS`; `NewsProvider` protocol; `Strategy.target_weights` gains `news=None` |
| `algua/data/news_schema.py` | modify | `retracted` column in validate/empty/to_schema/hash; tombstone generation + revision guards in `explode_news_symbols` |
| `algua/data/serve.py` | modify | `StoreBackedNewsProvider` (as-of seam, `knowable_at < end`) |
| `algua/strategies/base.py` | modify | `needs_news` config, `NewsSignalFn`, `news_signal_fn`, structural mutual-exclusion, `news` routing, `config_hash`, `assert_tradable_without_news` |
| `algua/strategies/loader.py` | modify | `needs_news` load rules (3-arg, no panel, not with fundamentals) |
| `algua/backtest/engine.py` | modify | `_assert_news_shape`, `_news_as_of`, `news_provider` threading, fast-path-forces-loop on news |
| `algua/backtest/result.py` | modify | `news_snapshot` field, stamped only when lane active |
| `algua/registry/promotion.py` | modify | block `needs_news` past `backtested` |
| `algua/backtest/walkforward.py`, `sweep.py` | modify | fail-closed guard for `needs_news`+`needs_fundamentals`; `_override` copies `news_signal_fn` |
| `algua/cli/paper_cmd.py`, `live_cmd.py` | modify | call `assert_tradable_without_news` |
| `algua/cli/backtest_cmd.py` | modify | `--news-snapshot` option + misuse error |
| `algua/strategies/news/news_coverage_tilt.py` | create | example `needs_news` strategy |
| `tests/test_news_schema.py` + new test files | modify/create | mirror the `*_fundamentals*` test suite |

---

## Task 1: News `retracted` column (schema plumbing, no tombstones yet)

Introduce the non-null `retracted` bool column everywhere the schema is defined, so the column exists end-to-end. Tombstone *generation* is Task 2; here `explode_news_symbols` just emits `retracted=False`.

**Files:**
- Modify: `algua/contracts/types.py:85-95` (NEWS constants)
- Modify: `algua/data/news_schema.py` (validate_news, empty_news, to_news_schema, logical_news_hash, explode_news_symbols)
- Test: `tests/test_news_schema.py`

- [ ] **Step 1: Add the constant + column to contracts**

In `algua/contracts/types.py`, change the news constants block:

```python
NEWS_COLUMNS: tuple[str, ...] = (
    "source", "article_id", "symbol", "published_at", "knowable_at",
    "headline", "url", "body", "retracted",
)
NEWS_AS_OF_KEY: tuple[str, ...] = ("source", "article_id", "symbol")
NEWS_KNOWABLE_AT = "knowable_at"
NEWS_RETRACTED = "retracted"  # True = a symbol-mention retracted by a later article revision
```

- [ ] **Step 2: Write/extend failing tests for the column**

In `tests/test_news_schema.py`, add:

```python
import numpy as np
import pandas as pd
import pytest

from algua.data.news_schema import (
    empty_news, explode_news_symbols, logical_news_hash, to_news_schema, validate_news,
)


def _raw(source, article_id, symbols, ka, headline="h", pub=None):
    return {
        "source": source, "article_id": article_id, "symbols": symbols,
        "published_at": pub or ka, "knowable_at": ka, "headline": headline,
    }


def test_empty_news_has_retracted_bool_column():
    e = empty_news()
    assert "retracted" in e.columns
    assert e["retracted"].dtype == np.dtype("bool")


def test_explode_emits_retracted_false_for_normal_rows():
    raw = pd.DataFrame([_raw("Reuters", "a1", ["AAPL", "MSFT"], "2023-01-01T00:00:00Z")])
    out = to_news_schema(explode_news_symbols(raw))
    assert set(out["symbol"]) == {"AAPL", "MSFT"}
    assert out["retracted"].dtype == np.dtype("bool")
    assert not out["retracted"].any()


def test_validate_rejects_non_bool_retracted():
    out = to_news_schema(explode_news_symbols(
        pd.DataFrame([_raw("r", "a1", ["AAPL"], "2023-01-01T00:00:00Z")])))
    bad = out.copy()
    bad["retracted"] = bad["retracted"].astype("object")
    bad.loc[bad.index[0], "retracted"] = "false"
    with pytest.raises(ValueError, match="retracted"):
        validate_news(bad)


def test_hash_changes_with_retracted():
    out = to_news_schema(explode_news_symbols(
        pd.DataFrame([_raw("r", "a1", ["AAPL"], "2023-01-01T00:00:00Z")])))
    flipped = out.copy()
    flipped["retracted"] = True
    assert logical_news_hash(out) != logical_news_hash(flipped)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_news_schema.py -k "retracted or hash_changes" -q`
Expected: FAIL (column absent / hash unchanged).

- [ ] **Step 4: Implement the column in news_schema.py**

In `algua/data/news_schema.py`:

Add `NEWS_RETRACTED` to the import and a `BOOL_COLUMNS` list near the other column lists:

```python
from algua.contracts.types import (
    NEWS_AS_OF_KEY, NEWS_COLUMNS, NEWS_KNOWABLE_AT, NEWS_RETRACTED,
)
...
BOOL_COLUMNS = [NEWS_RETRACTED]  # non-nullable numpy bool
```

In `validate_news`, after the `NULLABLE_STRING_COLUMNS` loop, add:

```python
    for col in BOOL_COLUMNS:
        if df[col].dtype != np.dtype("bool"):
            raise ValueError(f"news {col!r} must be non-nullable bool dtype (got {df[col].dtype})")
```

In `empty_news`, add the column to the `data` dict (before `[COLUMNS]`):

```python
        "retracted": pd.Series([], dtype="bool"),
```

In `to_news_schema`, after the timestamp normalization loop and before `drop_duplicates`, coerce:

```python
    out[NEWS_RETRACTED] = out[NEWS_RETRACTED].astype("bool")
```

In `logical_news_hash`, after the `TS_COLUMNS` loop, add:

```python
    for col in BOOL_COLUMNS:
        flags = np.array([bool(v) for v in ordered[col]], dtype="u1")
        digest.update(flags.tobytes())
```

In `explode_news_symbols`, just before `return out[COLUMNS]`, add:

```python
    out[NEWS_RETRACTED] = False
```

- [ ] **Step 5: Run the news schema tests + gate**

Run: `uv run pytest tests/test_news_schema.py -q && uv run mypy algua`
Expected: PASS. Fix any pre-existing news-schema test that hand-builds a frame without `retracted` by routing it through `explode_news_symbols`/`to_news_schema` (which now supply the column) or adding `retracted=False`.

- [ ] **Step 6: Commit**

```bash
git add algua/contracts/types.py algua/data/news_schema.py tests/test_news_schema.py
git commit -m "feat(132): add non-null retracted column to the news schema"
```

---

## Task 2: Tombstone generation in `explode_news_symbols`

Derive retraction tombstones from an article's revision history within one ingest frame.

**Files:**
- Modify: `algua/data/news_schema.py` (`explode_news_symbols`)
- Test: `tests/test_news_schema.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_news_schema.py`:

```python
def _key(out, sym, ka):
    m = out[(out["symbol"] == sym) & (out["knowable_at"] == pd.Timestamp(ka, tz="UTC"))]
    return None if m.empty else bool(m["retracted"].iloc[0])


def test_dropped_symbol_gets_a_tombstone_at_the_dropping_revision():
    raw = pd.DataFrame([
        _raw("r", "a1", ["AAPL", "MSFT"], "2023-01-01T00:00:00Z", headline="h1"),
        _raw("r", "a1", ["AAPL"], "2023-01-02T00:00:00Z", headline="h2"),
    ])
    out = to_news_schema(explode_news_symbols(raw))
    assert _key(out, "MSFT", "2023-01-01T00:00:00Z") is False   # original mention
    assert _key(out, "MSFT", "2023-01-02T00:00:00Z") is True    # tombstone at the drop
    assert _key(out, "AAPL", "2023-01-02T00:00:00Z") is False   # still present


def test_tombstone_carries_article_published_at_not_knowable_at():
    raw = pd.DataFrame([
        _raw("r", "a1", ["AAPL", "MSFT"], "2023-01-05T00:00:00Z",
             pub="2023-01-01T00:00:00Z", headline="h1"),
        _raw("r", "a1", ["AAPL"], "2023-01-06T00:00:00Z",
             pub="2023-01-01T00:00:00Z", headline="h2"),
    ])
    out = to_news_schema(explode_news_symbols(raw))
    tomb = out[(out["symbol"] == "MSFT") & out["retracted"]]
    assert tomb["published_at"].iloc[0] == pd.Timestamp("2023-01-01T00:00:00Z")


def test_full_retraction_empty_later_revision_tombstones_all_prior():
    raw = pd.DataFrame([
        _raw("r", "a1", ["AAPL", "MSFT"], "2023-01-01T00:00:00Z", headline="h1"),
        _raw("r", "a1", [], "2023-01-02T00:00:00Z", headline="h2"),
    ])
    out = to_news_schema(explode_news_symbols(raw))
    assert _key(out, "AAPL", "2023-01-02T00:00:00Z") is True
    assert _key(out, "MSFT", "2023-01-02T00:00:00Z") is True


def test_drop_then_readd():
    raw = pd.DataFrame([
        _raw("r", "a1", ["AAPL"], "2023-01-01T00:00:00Z", headline="h1"),
        _raw("r", "a1", [], "2023-01-02T00:00:00Z", headline="h2"),
        _raw("r", "a1", ["AAPL"], "2023-01-03T00:00:00Z", headline="h3"),
    ])
    out = to_news_schema(explode_news_symbols(raw))
    assert _key(out, "AAPL", "2023-01-02T00:00:00Z") is True   # tombstone
    assert _key(out, "AAPL", "2023-01-03T00:00:00Z") is False  # re-added


def test_zero_symbol_first_revision_rejected():
    with pytest.raises(ValueError, match="empty revision is only valid"):
        explode_news_symbols(pd.DataFrame([_raw("r", "a1", [], "2023-01-01T00:00:00Z")]))


def test_empty_after_empty_rejected():
    raw = pd.DataFrame([
        _raw("r", "a1", ["AAPL"], "2023-01-01T00:00:00Z"),
        _raw("r", "a1", [], "2023-01-02T00:00:00Z"),
        _raw("r", "a1", [], "2023-01-03T00:00:00Z"),
    ])
    with pytest.raises(ValueError, match="empty revision is only valid"):
        explode_news_symbols(raw)


def test_duplicate_revision_rejected_on_canonical_identity():
    raw = pd.DataFrame([
        _raw("Reuters", "a1", ["AAPL"], "2023-01-01T00:00:00Z"),
        _raw(" reuters ", "a1", ["MSFT"], "2023-01-01T00:00:00Z"),  # canonicalizes equal
    ])
    with pytest.raises(ValueError, match="duplicate news revision"):
        explode_news_symbols(raw)
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_news_schema.py -k "tombstone or retraction or readd or revision or empty_after" -q`
Expected: FAIL (no tombstones generated; duplicates not rejected).

- [ ] **Step 3: Rewrite `explode_news_symbols` with the revision walk**

Replace the body of `explode_news_symbols` (keep the docstring; update it to describe the revision/tombstone contract) from the `out["_syms"] = ...` line through `return out[COLUMNS]`:

```python
    out["_syms"] = out["symbols"].apply(_parse)

    # Canonical revision identity (mirror to_news_schema) so case/whitespace/tz variants of the
    # same revision can't bypass the duplicate guard or split the per-article walk.
    out["_gsrc"] = out["source"].astype(str).str.strip().str.lower()
    out["_gart"] = out["article_id"].astype(str).str.strip()
    out["_gka"] = pd.to_datetime(out[NEWS_KNOWABLE_AT], errors="raise", utc=True)
    if out.duplicated(subset=["_gsrc", "_gart", "_gka"]).any():
        raise ValueError(
            "duplicate news revision: (source, article_id, knowable_at) must be unique "
            "(each input row is one full article revision)"
        )

    base_cols = [c for c in COLUMNS if c not in ("symbol", NEWS_RETRACTED)]
    rows: list[dict[str, object]] = []
    for _, grp in out.groupby(["_gsrc", "_gart"], sort=False):
        grp = grp.sort_values("_gka", kind="stable")
        prev: set[str] = set()
        first = True
        for _, row in grp.iterrows():
            cur = set(row["_syms"])
            if not cur and (first or not prev):
                raise ValueError(
                    "each news article must tag >= 1 symbol; an empty revision is only valid as "
                    "a later full retraction of a non-empty prior revision"
                )
            common = {c: row[c] for c in base_cols}
            for sym in row["_syms"]:
                rows.append({**common, "symbol": sym, NEWS_RETRACTED: False})
            if not first:
                for sym in sorted(prev - cur):
                    rows.append({**common, "symbol": sym, NEWS_RETRACTED: True})
            prev = cur
            first = False
    return pd.DataFrame(rows, columns=list(COLUMNS))
```

Remove the now-dead `out = (out.drop(columns=["symbols"]).explode(...).rename(...))` block and the old zero-symbol check (the walk now owns symbol-count validation). Keep `_parse` and the `_RAW_REQUIRED`/optional-column setup above.

- [ ] **Step 4: Run the tombstone tests + full schema file**

Run: `uv run pytest tests/test_news_schema.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add algua/data/news_schema.py tests/test_news_schema.py
git commit -m "feat(132): generate symbol-set-revision tombstones in explode_news_symbols"
```

---

## Task 3: `NewsProvider` protocol + `StoreBackedNewsProvider` + `Strategy` protocol news arg

**Files:**
- Modify: `algua/contracts/types.py` (NewsProvider protocol; Strategy.target_weights signature)
- Modify: `algua/data/serve.py` (StoreBackedNewsProvider)
- Test: `tests/test_news_serve_asof.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_news_serve_asof.py`:

```python
from datetime import datetime, timezone

import pandas as pd

from algua.data.news_schema import explode_news_symbols
from algua.data.serve import StoreBackedNewsProvider
from algua.data.store import DataStore


def _raw(article_id, symbols, ka):
    return {"source": "r", "article_id": article_id, "symbols": symbols,
            "published_at": ka, "knowable_at": ka, "headline": "h"}


def test_provider_returns_history_before_end_with_tombstones(tmp_path):
    store = DataStore(tmp_path)
    raw = pd.DataFrame([
        _raw("a1", ["AAPL", "MSFT"], "2023-01-01T00:00:00Z"),
        _raw("a1", ["AAPL"], "2023-01-10T00:00:00Z"),
        _raw("a2", ["AAPL"], "2023-02-01T00:00:00Z"),
    ])
    rec = store.ingest_news(provider="test", as_of="2023-03-01T00:00:00Z", frame=raw)
    prov = StoreBackedNewsProvider(store, rec.snapshot_id)
    end = datetime(2023, 1, 15, tzinfo=timezone.utc)
    out = prov.get_news(["AAPL", "MSFT"], end)
    # knowable_at < end -> the Feb a2 row is excluded; the MSFT tombstone (Jan 10) is included
    assert out["knowable_at"].max() < pd.Timestamp(end)
    assert ((out["symbol"] == "MSFT") & out["retracted"]).any()
    assert prov.snapshot_id == rec.snapshot_id
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_news_serve_asof.py -q`
Expected: FAIL (`StoreBackedNewsProvider` does not exist).

- [ ] **Step 3: Add the protocol + provider**

In `algua/contracts/types.py`, after the `FundamentalsProvider` protocol add:

```python
@runtime_checkable
class NewsProvider(Protocol):
    """As-of consumption seam for point-in-time news (issue #132). Returns the FULL bitemporal
    history (including retraction tombstones) for `symbols` with knowable_at < end — no lower
    bound. The engine owns decision `t` and masks knowable_at <= t per bar; the provider never
    sees `t`."""

    snapshot_id: str

    def get_news(self, symbols: list[str], end: datetime) -> pd.DataFrame: ...
```

Extend the `Strategy` protocol's `target_weights` signature:

```python
    def target_weights(
        self,
        features: pd.DataFrame,
        fundamentals: pd.DataFrame | None = None,
        news: pd.DataFrame | None = None,
    ) -> pd.Series: ...
```

In `algua/data/serve.py`, add after `StoreBackedFundamentalsProvider`:

```python
class StoreBackedNewsProvider:
    """Serves one news snapshot through the as-of `NewsProvider` seam. Returns the FULL bitemporal
    history (including tombstones) with knowable_at < end (no lower bound — the first decision bar
    needs prior news). The engine applies the per-bar knowable_at <= t mask; this provider never
    sees `t`."""

    def __init__(self, store: DataStore, snapshot_id: str) -> None:
        self.store = store
        self.snapshot_id = snapshot_id

    def get_news(self, symbols: list[str], end: datetime) -> pd.DataFrame:
        frame = self.store.read_news(self.snapshot_id, symbols=symbols)
        end_ts = pd.Timestamp(end)
        end_ts = end_ts.tz_localize("UTC") if end_ts.tzinfo is None else end_ts.tz_convert("UTC")
        return frame[frame["knowable_at"] < end_ts].reset_index(drop=True)
```

- [ ] **Step 4: Run the test + gate**

Run: `uv run pytest tests/test_news_serve_asof.py -q && uv run mypy algua`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add algua/contracts/types.py algua/data/serve.py tests/test_news_serve_asof.py
git commit -m "feat(132): NewsProvider protocol + StoreBackedNewsProvider as-of seam"
```

---

## Task 4: Strategy adapter — `needs_news` lane + guards

**Files:**
- Modify: `algua/strategies/base.py`
- Test: `tests/test_strategies_base_news.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_strategies_base_news.py`:

```python
import pandas as pd
import pytest

from algua.contracts.types import ExecutionContract
from algua.portfolio.construction import get_construction_policy
from algua.strategies.base import (
    LoadedStrategy, StrategyConfig, assert_tradable_without_news, config_hash,
)


def _cfg(**kw):
    base = dict(
        name="n", universe=["AAPL"],
        execution=ExecutionContract(rebalance_frequency="1d"),
        construction="equal_weight_positive",
    )
    base.update(kw)
    return StrategyConfig(**base)


def _news_fn(view, params, news):
    return pd.Series(dtype="float64")


def test_needs_news_requires_news_signal_fn():
    with pytest.raises(ValueError, match="needs_news"):
        LoadedStrategy(config=_cfg(needs_news=True),
                       construct_fn=get_construction_policy("equal_weight_positive"))


def test_needs_news_and_needs_fundamentals_together_rejected():
    with pytest.raises(ValueError, match="both"):
        LoadedStrategy(config=_cfg(needs_news=True, needs_fundamentals=True),
                       news_signal_fn=_news_fn,
                       construct_fn=get_construction_policy("equal_weight_positive"))


def test_stray_sidecar_fn_rejected():
    # needs_news=False but a news_signal_fn is set -> exactly-one violated
    with pytest.raises(ValueError):
        LoadedStrategy(config=_cfg(), news_signal_fn=_news_fn,
                       construct_fn=get_construction_policy("equal_weight_positive"))


def test_signal_routes_news_and_fails_closed_without_it():
    s = LoadedStrategy(config=_cfg(needs_news=True), news_signal_fn=_news_fn,
                       construct_fn=get_construction_policy("equal_weight_positive"))
    with pytest.raises(ValueError, match="news"):
        s.signal(pd.DataFrame())  # no news frame
    assert s.signal(pd.DataFrame(), news=pd.DataFrame()).empty


def test_config_hash_changes_with_needs_news():
    a = LoadedStrategy(config=_cfg(), signal_fn=lambda v, p: pd.Series(dtype="float64"),
                       construct_fn=get_construction_policy("equal_weight_positive"))
    b = LoadedStrategy(config=_cfg(needs_news=True), news_signal_fn=_news_fn,
                       construct_fn=get_construction_policy("equal_weight_positive"))
    assert config_hash(a) != config_hash(b)


def test_assert_tradable_without_news_raises():
    s = LoadedStrategy(config=_cfg(needs_news=True), news_signal_fn=_news_fn,
                       construct_fn=get_construction_policy("equal_weight_positive"))
    with pytest.raises(ValueError, match="needs_news"):
        assert_tradable_without_news(s)
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_strategies_base_news.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement in base.py**

Add the type alias after `FundamentalsSignalFn`:

```python
# OPT-IN news signal (issue #132): `signal(view, params, news)`. Distinct type so the news 3-arg
# form never silently overloads the 2-arg or fundamentals 3-arg forms.
NewsSignalFn = Callable[[pd.DataFrame, dict[str, Any], pd.DataFrame], pd.Series]
```

Add to `StrategyConfig`:

```python
    # Opt into the as-of news lane (issue #132). Mutually exclusive with needs_fundamentals.
    needs_news: bool = False
```

Add the field to `LoadedStrategy`:

```python
    news_signal_fn: NewsSignalFn | None = None
```

Replace `LoadedStrategy.__post_init__` with structural mutual exclusion:

```python
    def __post_init__(self) -> None:
        cfg = self.config
        if cfg.needs_fundamentals and cfg.needs_news:
            raise ValueError(
                "needs_fundamentals and needs_news cannot both be True "
                "(a strategy using both is not supported yet — #132 follow-up)"
            )
        decision_fns = {
            "signal_fn": self.signal_fn,
            "fundamentals_signal_fn": self.fundamentals_signal_fn,
            "news_signal_fn": self.news_signal_fn,
        }
        active = [k for k, v in decision_fns.items() if v is not None]
        expected = (
            "fundamentals_signal_fn" if cfg.needs_fundamentals
            else "news_signal_fn" if cfg.needs_news
            else "signal_fn"
        )
        if active != [expected]:
            raise ValueError(
                f"config requires exactly {expected!r} to be set "
                f"(needs_fundamentals={cfg.needs_fundamentals}, needs_news={cfg.needs_news}); "
                f"got {active}"
            )
```

Update `authored_signal`:

```python
    @property
    def authored_signal(self) -> SignalFn | FundamentalsSignalFn | NewsSignalFn:
        if self.config.needs_fundamentals:
            assert self.fundamentals_signal_fn is not None
            return self.fundamentals_signal_fn
        if self.config.needs_news:
            assert self.news_signal_fn is not None
            return self.news_signal_fn
        assert self.signal_fn is not None
        return self.signal_fn
```

Replace `signal` and `target_weights`:

```python
    def signal(
        self,
        view: pd.DataFrame,
        fundamentals: pd.DataFrame | None = None,
        news: pd.DataFrame | None = None,
    ) -> pd.Series:
        if self.config.needs_fundamentals:
            if fundamentals is None:
                raise ValueError(
                    f"strategy {self.name!r} needs fundamentals but signal was called without a "
                    f"fundamentals frame (fail closed)"
                )
            if news is not None:
                raise ValueError(f"strategy {self.name!r} was passed a news frame it does not use")
            assert self.fundamentals_signal_fn is not None
            return self.fundamentals_signal_fn(view, self.config.params, fundamentals)
        if self.config.needs_news:
            if news is None:
                raise ValueError(
                    f"strategy {self.name!r} needs news but signal was called without a news "
                    f"frame (fail closed)"
                )
            if fundamentals is not None:
                raise ValueError(
                    f"strategy {self.name!r} was passed a fundamentals frame it does not use"
                )
            assert self.news_signal_fn is not None
            return self.news_signal_fn(view, self.config.params, news)
        if fundamentals is not None or news is not None:
            raise ValueError(f"strategy {self.name!r} takes no PIT sidecar but one was passed")
        assert self.signal_fn is not None
        return self.signal_fn(view, self.config.params)

    def target_weights(
        self,
        features: pd.DataFrame,
        fundamentals: pd.DataFrame | None = None,
        news: pd.DataFrame | None = None,
    ) -> pd.Series:
        return self.construct(self.signal(features, fundamentals, news), features)
```

Add `assert_tradable_without_news` after `assert_tradable_without_fundamentals`:

```python
def assert_tradable_without_news(strategy: LoadedStrategy) -> None:
    """Fail closed: a needs_news strategy must NOT run paper/live yet — the as-of news lane is
    wired only into the backtest engine (issue #132). Called at every trading load point."""
    if strategy.config.needs_news:
        raise ValueError(
            f"strategy {strategy.name!r} declares needs_news; paper/live news wiring is not built "
            f"yet (#132 follow-up) — refusing to trade it blind"
        )
```

In `config_hash`, add to the payload dict (after `"needs_fundamentals": ...`):

```python
            "needs_news": strategy.config.needs_news,
```

- [ ] **Step 4: Run the tests + gate**

Run: `uv run pytest tests/test_strategies_base_news.py tests/test_strategies_base_fundamentals.py -q && uv run mypy algua`
Expected: PASS (the fundamentals tests confirm the refactor didn't regress its lane).

- [ ] **Step 5: Commit**

```bash
git add algua/strategies/base.py tests/test_strategies_base_news.py
git commit -m "feat(132): needs_news adapter lane with structural mutual exclusion + guards"
```

---

## Task 5: Loader — `needs_news` load rules

**Files:**
- Modify: `algua/strategies/loader.py`
- Test: `tests/test_loader_news.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_loader_news.py` (mirrors `test_loader_fundamentals.py`; writes temp strategy modules under a temp family dir — copy the temp-module fixture pattern from `test_loader_fundamentals.py`). Cover:

```python
# Using the same temp-family-module harness as test_loader_fundamentals.py:
# - a needs_news=True module with a 3-arg signal loads, binding news_signal_fn
# - a needs_news=True module with a 2-arg signal raises StrategyNotFound (arity)
# - a needs_news=True module exposing signal_panel raises StrategyNotFound (no fast path)
# - a module with CONFIG.needs_news=True AND needs_fundamentals=True raises (pydantic/loader)
```

(Reproduce the exact fixture mechanics from `test_loader_fundamentals.py`; assert
`loaded.news_signal_fn is module.signal` and `loaded.signal_fn is None` for the happy path.)

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_loader_news.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement the loader branch**

In `algua/strategies/loader.py`, after computing `needs_fundamentals` add:

```python
    needs_news = bool(getattr(config, "needs_news", False))
```

After the existing `if needs_fundamentals:` block (which already returns), add:

```python
    if needs_news:
        if panel_fn is not None:
            raise StrategyNotFound(
                f"{name}: signal_panel is not supported with needs_news "
                f"(no vectorized news fast path yet)"
            )
        if n_params != 3:
            raise StrategyNotFound(
                f"{name}: needs_news=True requires signal(view, params, news); "
                f"got {n_params} params"
            )
        return LoadedStrategy(
            config=config, news_signal_fn=module.signal, construct_fn=construct_fn
        )
```

(The `needs_fundamentals and needs_news` both-True case is rejected by `LoadedStrategy.__post_init__` from Task 4 when either branch constructs it; the fundamentals branch runs first, so add an explicit guard there only if a test shows it slips through — otherwise `__post_init__` is the wall.)

- [ ] **Step 4: Run + gate**

Run: `uv run pytest tests/test_loader_news.py tests/test_loader_fundamentals.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add algua/strategies/loader.py tests/test_loader_news.py
git commit -m "feat(132): loader binds needs_news 3-arg signal, rejects panel"
```

---

## Task 6: Engine — `_assert_news_shape`, `_news_as_of`, threading, fast-path guard

**Files:**
- Modify: `algua/backtest/engine.py`
- Test: `tests/test_engine_news.py`, `tests/test_engine_news_provider_assert.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_engine_news.py`:

```python
import pandas as pd

from algua.backtest.engine import _news_as_of
from algua.data.news_schema import explode_news_symbols, to_news_schema


def _news():
    raw = pd.DataFrame([
        {"source": "r", "article_id": "a1", "symbols": ["AAPL", "MSFT"],
         "published_at": "2023-01-01T00:00:00Z", "knowable_at": "2023-01-01T00:00:00Z",
         "headline": "h1"},
        {"source": "r", "article_id": "a1", "symbols": ["AAPL"],
         "published_at": "2023-01-01T00:00:00Z", "knowable_at": "2023-01-10T00:00:00Z",
         "headline": "h2"},
    ])
    return to_news_schema(explode_news_symbols(raw))


def test_as_of_before_drop_shows_both():
    out = _news_as_of(_news(), pd.Timestamp("2023-01-05T00:00:00Z"))
    assert set(out["symbol"]) == {"AAPL", "MSFT"}
    assert not out["retracted"].any()  # tombstones never surface in the as-of view


def test_as_of_after_drop_excludes_retracted_symbol():
    out = _news_as_of(_news(), pd.Timestamp("2023-01-15T00:00:00Z"))
    assert set(out["symbol"]) == {"AAPL"}  # MSFT's latest revision is a tombstone -> dropped


def test_as_of_excludes_future_knowable():
    out = _news_as_of(_news(), pd.Timestamp("2022-12-31T00:00:00Z"))
    assert out.empty


def test_as_of_empty_preserves_columns_and_dtypes():
    out = _news_as_of(_news(), pd.Timestamp("2022-01-01T00:00:00Z"))
    assert list(out.columns) == list(_news().columns)
    assert out["retracted"].dtype == _news()["retracted"].dtype
```

Create `tests/test_engine_news_provider_assert.py` (mirror `test_engine_fundamentals_provider_assert.py`): a foreign frame with `published_at > knowable_at`, a naive `knowable_at`, a non-bool `retracted`, and a duplicate key each raise `BacktestError` from `_assert_news_shape`.

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_engine_news.py tests/test_engine_news_provider_assert.py -q`
Expected: FAIL (`_news_as_of` / `_assert_news_shape` not defined).

- [ ] **Step 3: Implement in engine.py**

Extend the contracts import:

```python
from algua.contracts.types import (
    FUNDAMENTALS_AS_OF_KEY,
    FUNDAMENTALS_COLUMNS,
    FUNDAMENTALS_KNOWABLE_AT,
    NEWS_AS_OF_KEY,
    NEWS_COLUMNS,
    NEWS_KNOWABLE_AT,
    NEWS_RETRACTED,
    DataProvider,
    FundamentalsProvider,
    NewsProvider,
)
```

Add after `_fundamentals_as_of`:

```python
def _assert_news_shape(frame: pd.DataFrame) -> None:
    """Structural defense at the engine seam (no algua.data import): a foreign NewsProvider must
    hand back contract-shaped, UTC, unique-keyed data. Store-backed reads already validate; this
    fails closed for any other provider (spec §5)."""
    missing = [c for c in NEWS_COLUMNS if c not in frame.columns]
    if missing:
        raise BacktestError(f"news frame missing columns {missing}")
    for col in (NEWS_KNOWABLE_AT, "published_at"):
        ts = frame[col]
        if not isinstance(ts.dtype, pd.DatetimeTZDtype) or str(ts.dt.tz) != "UTC":
            raise BacktestError(f"news {col!r} must be tz-aware UTC")
        if ts.isna().any():
            raise BacktestError(f"news {col!r} must not be null")
    if (frame[NEWS_KNOWABLE_AT].to_numpy() < frame["published_at"].to_numpy()).any():
        raise BacktestError("news 'knowable_at' must be >= 'published_at'")
    if str(frame[NEWS_RETRACTED].dtype) != "bool":
        raise BacktestError("news 'retracted' must be non-nullable bool")
    key = [*NEWS_AS_OF_KEY, NEWS_KNOWABLE_AT]
    if frame[key].duplicated().any():
        raise BacktestError("news has duplicate (source, article_id, symbol, knowable_at) rows")


def _news_as_of(frame: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    """As-of-t news: of the rows with knowable_at <= t, keep for each (source, article_id, symbol)
    the latest revision (greatest knowable_at), then DROP retraction tombstones. knowable_at is
    unique per key within a snapshot, so the pick is deterministic. Uses only knowable_at <= t ->
    no look-ahead. Empty-in/empty-out returns a 0-row slice (preserves dtypes)."""
    if t.tz is None:
        raise BacktestError("news as-of mask requires a tz-aware (UTC) timestamp t")
    visible = frame[frame[NEWS_KNOWABLE_AT] <= t]
    if visible.empty:
        return frame.iloc[0:0].copy()
    ordered = visible.sort_values(NEWS_KNOWABLE_AT, kind="stable")
    latest = ordered.drop_duplicates(subset=list(NEWS_AS_OF_KEY), keep="last")
    live = latest[~latest[NEWS_RETRACTED]]
    return live.reset_index(drop=True)
```

In `_decision_weights`, add a `news` parameter (after `fundamentals`):

```python
    news: pd.DataFrame | None = None,
```

In the per-bar loop body, add a news branch parallel to the fundamentals branch (the fundamentals `if fundamentals is not None:` block already computes `allowed`/`members`; place the news branch so the active sidecar is masked and passed):

```python
        if fundamentals is not None:
            f_asof = _fundamentals_as_of(fundamentals, t)
            allowed = members if universe_by_date is not None else set(columns)
            f_asof = f_asof[f_asof["symbol"].isin(allowed)]
            w = strategy.target_weights(view, fundamentals=f_asof)
        elif news is not None:
            n_asof = _news_as_of(news, t)
            allowed = members if universe_by_date is not None else set(columns)
            n_asof = n_asof[n_asof["symbol"].isin(allowed)]
            w = strategy.target_weights(view, news=n_asof)
        else:
            w = strategy.target_weights(view)
```

In `_decision_weights_fast_or_loop`, add the `news` param and force the loop on it:

```python
    news: pd.DataFrame | None = None,
) -> pd.DataFrame:
    ...
    if (
        strategy.signal_panel_fn is None
        or universe_by_date is not None
        or fundamentals is not None
        or news is not None
    ):
        return _decision_weights(
            strategy, bars, adj,
            universe_by_date=universe_by_date, fundamentals=fundamentals, news=news,
        )
    return _decision_weights_fast(strategy, bars, adj)
```

In `simulate`, add `news_provider: NewsProvider | None = None` to the signature and, mirroring the fundamentals block:

```python
    news: pd.DataFrame | None = None
    if strategy.config.needs_news:
        if news_provider is None:
            raise BacktestError(
                f"strategy {strategy.name!r} declares needs_news but no news_provider was "
                f"supplied (fail closed)"
            )
        news = news_provider.get_news(_fetch_symbols(strategy, universe_by_date), end)
        _assert_news_shape(news)
```

Pass `news=news` into the `_decision_weights_fast_or_loop(...)` call. Add `news_provider` to the `run` signature and pass it to `simulate`.

- [ ] **Step 4: Run the tests + gate**

Run: `uv run pytest tests/test_engine_news.py tests/test_engine_news_provider_assert.py tests/test_engine_fundamentals.py -q && uv run mypy algua`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add algua/backtest/engine.py tests/test_engine_news.py tests/test_engine_news_provider_assert.py
git commit -m "feat(132): engine news as-of mask + provider threading + fast-path guard"
```

---

## Task 7: `BacktestResult.news_snapshot`

**Files:**
- Modify: `algua/backtest/engine.py` (`run` stamping), `algua/backtest/result.py`
- Test: `tests/test_engine_news.py` (extend)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_engine_news.py` an end-to-end assertion using the synthetic/demo provider + a `StoreBackedNewsProvider`, asserting `result.news_snapshot == snapshot_id` for a `needs_news` strategy, and `None` for a non-`needs_news` strategy even if a provider is passed. (Use the example strategy from Task 12 once available, or a locally-defined `LoadedStrategy` with a trivial news signal.)

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_engine_news.py -k news_snapshot -q`
Expected: FAIL (`news_snapshot` attribute missing).

- [ ] **Step 3: Implement**

In `algua/backtest/result.py`, add the field after `fundamentals_snapshot`:

```python
    # News snapshot used by a needs_news strategy (issue #132); None otherwise.
    news_snapshot: str | None = None
```

In `engine.py` `run`, stamp it only when the lane is active:

```python
        fundamentals_snapshot=getattr(fundamentals_provider, "snapshot_id", None),
        news_snapshot=(
            getattr(news_provider, "snapshot_id", None)
            if strategy.config.needs_news else None
        ),
```

- [ ] **Step 4: Run + gate**

Run: `uv run pytest tests/test_engine_news.py -q && uv run mypy algua`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add algua/backtest/result.py algua/backtest/engine.py tests/test_engine_news.py
git commit -m "feat(132): stamp news_snapshot provenance when the news lane is active"
```

---

## Task 8: Promotion guard — block `needs_news` past `backtested`

**Files:**
- Modify: `algua/registry/promotion.py:106-110`
- Test: `tests/test_promotion_needs_news.py` (create) or extend the fundamentals promotion-guard test.

- [ ] **Step 1: Write the failing test**

Mirror the existing `needs_fundamentals` promotion-guard test: a registered `needs_news` strategy cannot be research-promoted past `backtested`; assert the raised `ValueError` mentions `needs_news`.

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_promotion_needs_news.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement**

In `algua/registry/promotion.py`, extend the guard block:

```python
    if _loaded is not None and _loaded.config.needs_fundamentals:
        raise ValueError(
            f"strategy {name!r} declares needs_fundamentals; it cannot be promoted past "
            f"backtested until the paper/live fundamentals lane is built (#132)"
        )
    if _loaded is not None and _loaded.config.needs_news:
        raise ValueError(
            f"strategy {name!r} declares needs_news; it cannot be promoted past "
            f"backtested until the paper/live news lane is built (#132)"
        )
```

- [ ] **Step 4: Run + gate**

Run: `uv run pytest tests/test_promotion_needs_news.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add algua/registry/promotion.py tests/test_promotion_needs_news.py
git commit -m "feat(132): block needs_news strategies past backtested in promotion gate"
```

---

## Task 9: walk_forward / sweep fail-closed guard + `_override` copy

**Files:**
- Modify: `algua/backtest/walkforward.py` (`walk_forward`), `algua/backtest/sweep.py` (`sweep`, `_override`)
- Test: `tests/test_wf_sweep_pit_guard.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_wf_sweep_pit_guard.py`:

```python
import pytest

from algua.backtest.engine import BacktestError
from algua.backtest.sweep import _override, sweep
from algua.backtest.walkforward import walk_forward
# Build a needs_news and a needs_fundamentals LoadedStrategy via loader fixtures or _loaded helpers;
# a trivial demo provider is fine since the guard must fire BEFORE any data fetch.


def test_walk_forward_rejects_needs_news(needs_news_strategy, demo_provider, dates):
    with pytest.raises(BacktestError, match="not supported in walk-forward"):
        walk_forward(needs_news_strategy, demo_provider, *dates)


def test_walk_forward_rejects_needs_fundamentals(needs_fund_strategy, demo_provider, dates):
    with pytest.raises(BacktestError, match="not supported in walk-forward"):
        walk_forward(needs_fund_strategy, demo_provider, *dates)


def test_sweep_rejects_pit_sidecar(needs_news_strategy, demo_provider, dates):
    with pytest.raises(BacktestError, match="not supported in"):
        sweep(needs_news_strategy, demo_provider, *dates, grid={"x": [1]})


def test_override_preserves_news_signal_fn(needs_news_strategy):
    out = _override(needs_news_strategy, {})
    assert out.news_signal_fn is needs_news_strategy.news_signal_fn
```

(Define the fixtures with the loader/`_loaded` helpers used elsewhere; `dates` is a `(start, end)` datetime pair.)

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_wf_sweep_pit_guard.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement the guard + override copy**

In `algua/backtest/walkforward.py`, add a module-level helper and call it at the top of `walk_forward` (before `build_portfolio`):

```python
def _reject_pit_sidecar(strategy: LoadedStrategy, where: str) -> None:
    if strategy.config.needs_fundamentals or strategy.config.needs_news:
        kind = "needs_fundamentals" if strategy.config.needs_fundamentals else "needs_news"
        raise BacktestError(
            f"{kind} strategies are not supported in {where} yet (#132 follow-up): "
            f"provider threading through {where} is deferred"
        )
```

At the start of `walk_forward`:

```python
    _reject_pit_sidecar(strategy, "walk-forward")
```

In `algua/backtest/sweep.py`, at the start of `sweep` (after the `rank_by` check is fine):

```python
    from algua.backtest.walkforward import _reject_pit_sidecar
    _reject_pit_sidecar(strategy, "sweep")
```

In `_override`'s `LoadedStrategy(...)` call, add:

```python
        news_signal_fn=strategy.news_signal_fn,
```

(Ensure `LoadedStrategy` is imported in walkforward.py; it imports `build_portfolio` from engine already — add `from algua.strategies.base import LoadedStrategy` if not present.)

- [ ] **Step 4: Run + gate**

Run: `uv run pytest tests/test_wf_sweep_pit_guard.py tests/test_sweep_override_fundamentals.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add algua/backtest/walkforward.py algua/backtest/sweep.py tests/test_wf_sweep_pit_guard.py
git commit -m "feat(132): fail-closed WF/sweep guard for PIT-sidecar strategies + override copy"
```

---

## Task 10: paper/live trading guard wiring

**Files:**
- Modify: `algua/cli/paper_cmd.py:89-90`, `algua/cli/live_cmd.py:110-111`
- Test: `tests/test_news_guards.py` (create) — mirror `test_fundamentals_guards.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_news_guards.py` mirroring `test_fundamentals_guards.py`: loading a `needs_news` strategy into the paper and live trading preambles raises (the `assert_tradable_without_news` wall). Assert the message mentions `needs_news`.

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_news_guards.py -q`
Expected: FAIL.

- [ ] **Step 3: Wire the guard**

In `algua/cli/paper_cmd.py` (the `_load_for_trading` preamble, ~line 89):

```python
    from algua.strategies.base import (
        assert_tradable_without_fundamentals, assert_tradable_without_news,
    )
    assert_tradable_without_fundamentals(strategy)
    assert_tradable_without_news(strategy)
```

In `algua/cli/live_cmd.py` (~line 110), the same two-call pattern.

- [ ] **Step 4: Run + gate**

Run: `uv run pytest tests/test_news_guards.py tests/test_fundamentals_guards.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add algua/cli/paper_cmd.py algua/cli/live_cmd.py tests/test_news_guards.py
git commit -m "feat(132): fail closed on needs_news at paper/live trading load points"
```

---

## Task 11: CLI `--news-snapshot`

**Files:**
- Modify: `algua/cli/backtest_cmd.py`
- Test: `tests/test_cli_backtest_news.py` (create) — mirror `test_cli_backtest_fundamentals.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cli_backtest_news.py` mirroring the fundamentals CLI test: ingest a news snapshot, run `algua backtest run <news_strategy> --news-snapshot <id> --demo ...` via the typer `CliRunner`, assert exit 0 and `news_snapshot` in the JSON. Add a misuse test: `--news-snapshot` passed for a non-`needs_news` strategy exits non-zero with an error mentioning `needs_news`.

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_cli_backtest_news.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement the option + wiring + misuse error**

In `algua/cli/backtest_cmd.py`, import the provider:

```python
from algua.data.serve import StoreBackedFundamentalsProvider, StoreBackedNewsProvider
```

Add the option to `run`:

```python
    news_snapshot: str = typer.Option(
        None, "--news-snapshot",
        help="ingested news snapshot id (required for a needs_news strategy)"),
```

After resolving `strategy` and building `fundamentals_provider`:

```python
    if news_snapshot and not strategy.config.needs_news:
        raise ValueError(
            "--news-snapshot was given but the strategy does not declare needs_news"
        )
    news_provider = (
        StoreBackedNewsProvider(DataStore(get_settings().data_dir), news_snapshot)
        if news_snapshot
        else None
    )
```

Pass `news_provider=news_provider` into the `run_backtest(...)` call. (`run_backtest` is the imported alias of `engine.run`, which gained `news_provider` in Task 6.)

- [ ] **Step 4: Run + gate**

Run: `uv run pytest tests/test_cli_backtest_news.py tests/test_cli_backtest_fundamentals.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add algua/cli/backtest_cmd.py tests/test_cli_backtest_news.py
git commit -m "feat(132): backtest --news-snapshot wiring + misuse error"
```

---

## Task 12: Example `needs_news` strategy + end-to-end backtest

**Files:**
- Create: `algua/strategies/news/__init__.py`, `algua/strategies/news/news_coverage_tilt.py`
- Test: `tests/test_strategy_news_coverage_tilt.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_strategy_news_coverage_tilt.py`: load `news_coverage_tilt` via `load_strategy`, assert `config.needs_news` and a bound `news_signal_fn`; call the signal directly with a small as-of news frame + a bars view and assert it scores the more-covered symbol higher. Optionally a full `run` against the demo provider + an ingested news snapshot asserting a finite Sharpe.

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_strategy_news_coverage_tilt.py -q`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement the example strategy**

Create `algua/strategies/news/__init__.py` (empty). Create `algua/strategies/news/news_coverage_tilt.py`:

```python
"""Headline-coverage tilt: SIGNAL = count of an article-mention per symbol within a trailing
window of the as-of news; CONSTRUCTION = equal-weight the positively-covered names. A minimal
demonstration of the as-of news lane (issue #132) — no sentiment/NLP, just coverage counts."""
from __future__ import annotations

from typing import Any

import pandas as pd

from algua.contracts.types import ExecutionContract
from algua.strategies.base import StrategyConfig

CONFIG = StrategyConfig(
    name="news_coverage_tilt",
    universe=["AAPL", "MSFT", "NVDA"],
    execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
    params={"window_days": 5},
    construction="equal_weight_positive",
    needs_news=True,
)


def signal(view: pd.DataFrame, params: dict[str, Any], news: pd.DataFrame) -> pd.Series:
    """Score = number of distinct articles mentioning each symbol whose published_at is within the
    last `window_days` of the latest bar. The as-of mask already removed retracted mentions and any
    news not knowable by the decision bar; we window on published_at here."""
    if news.empty or view.empty:
        return pd.Series(dtype="float64")
    window = int(params["window_days"])
    asof = view.index.max() if isinstance(view.index, pd.DatetimeIndex) else view["timestamp"].max()
    cutoff = pd.Timestamp(asof).tz_convert("UTC") - pd.Timedelta(days=window)
    recent = news[news["published_at"] >= cutoff]
    if recent.empty:
        return pd.Series(dtype="float64")
    return recent.groupby("symbol")["article_id"].nunique().astype("float64")
```

(Confirm the bars `view` index/columns shape against `bar-schema.md` / how `fundamentals_earnings_tilt`'s view is shaped, and adjust the `asof` extraction to match the engine's `view`.)

- [ ] **Step 4: Run + gate**

Run: `uv run pytest tests/test_strategy_news_coverage_tilt.py -q && uv run lint-imports`
Expected: PASS (lint-imports confirms the new strategy stays off `algua.data`).

- [ ] **Step 5: Commit**

```bash
git add algua/strategies/news/ tests/test_strategy_news_coverage_tilt.py
git commit -m "feat(132): news_coverage_tilt example strategy exercising the as-of news lane"
```

---

## Final verification

- [ ] Run the full gate: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
- [ ] Confirm a `needs_news` strategy: backtests with `--news-snapshot`, is refused by paper/live, is refused by walk_forward/sweep, and cannot be promoted past `backtested`.
- [ ] Confirm tombstones: a dropped symbol disappears from the as-of view at/after the dropping revision and reappears if re-added.
