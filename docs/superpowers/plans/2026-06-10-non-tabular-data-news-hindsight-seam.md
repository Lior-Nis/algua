# News Hindsight Seam Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a point-in-time **news** non-tabular data type to algua — typed bitemporal record, entity→symbol explode, structural dedup, file ingest, and a full-hindsight `query-news` lane — mirroring the merged fundamentals seam, with the as-of signal lane deferred.

**Architecture:** News reuses the fundamentals "spine": a tidy/long bitemporal record validated by a `news_schema` module, persisted via the existing snapshot/manifest/content-hash store, and read back only through a hindsight accessor in the import-walled `algua.data.hindsight`. The slice adds **one** import-linter contract (forbid `live`/`execution` from `algua.data`) to complete the data wall; it adds **no** engine, strategy, provider, or promotion code.

**Tech Stack:** Python 3.12, pandas, pyarrow (parquet), typer (CLI), pytest, import-linter, ruff, mypy.

**Spec:** `docs/superpowers/specs/2026-06-10-non-tabular-data-news-hindsight-seam-design.md`

**Reference implementation to mirror (read these first):** `algua/data/fundamentals_schema.py`, `algua/contracts/types.py` (§ "Non-tabular: fundamentals seam"), `algua/data/store.py` (`ingest_fundamentals`/`read_fundamentals`), `algua/data/hindsight.py`, `algua/cli/data_cmd.py` (`ingest-fundamentals`/`query-fundamentals`), `tests/test_fundamentals_wall.py`, `docs/contracts/fundamentals-schema.md`.

**Quality gate (run after each task):** `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`

**Canonical schema (used throughout):**
- `NEWS_COLUMNS = ("source", "article_id", "symbol", "published_at", "knowable_at", "headline", "url", "body")`
- `NEWS_AS_OF_KEY = ("source", "article_id", "symbol")`; unique row key adds `knowable_at`.
- Canonical sort `_SORT = ["symbol", "source", "article_id", "knowable_at"]`.
- Non-null: `source`, `article_id`, `symbol`, `headline` (strings); `published_at`, `knowable_at` (tz-aware UTC). Nullable strings: `url`, `body`.

---

### Task 1: News column constants in contracts

**Files:**
- Modify: `algua/contracts/types.py` (append after the fundamentals constants block, ~line 80)
- Test: `tests/test_contracts.py` (add a test)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_contracts.py`:

```python
def test_news_column_constants():
    from algua.contracts.types import NEWS_AS_OF_KEY, NEWS_COLUMNS, NEWS_KNOWABLE_AT

    assert NEWS_COLUMNS == (
        "source", "article_id", "symbol", "published_at", "knowable_at", "headline", "url", "body",
    )
    assert NEWS_AS_OF_KEY == ("source", "article_id", "symbol")
    assert NEWS_KNOWABLE_AT == "knowable_at"
    # the as-of key columns are all real columns
    assert set(NEWS_AS_OF_KEY).issubset(set(NEWS_COLUMNS))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_contracts.py::test_news_column_constants -v`
Expected: FAIL with `ImportError: cannot import name 'NEWS_COLUMNS'`.

- [ ] **Step 3: Add the constants**

In `algua/contracts/types.py`, after the `FUNDAMENTALS_KNOWABLE_AT = "knowable_at"` line, add:

```python
# --- Non-tabular: news seam (issue #132, hindsight slice) --------------------------------------
# Tidy/long bitemporal news, one row per (article, mentioned symbol). `source` is part of the
# identity because an article id is only unique WITHIN a source. Hindsight-only this slice (no
# engine consumer), but the names live here beside the fundamentals constants for symmetry.
NEWS_COLUMNS: tuple[str, ...] = (
    "source", "article_id", "symbol", "published_at", "knowable_at", "headline", "url", "body",
)
# Identity of an article-mention across revisions: a correction is a new row sharing this key with
# a later knowable_at. `source` scopes the (source-local) article_id.
NEWS_AS_OF_KEY: tuple[str, ...] = ("source", "article_id", "symbol")
NEWS_KNOWABLE_AT = "knowable_at"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_contracts.py::test_news_column_constants -v`
Expected: PASS.

- [ ] **Step 5: Run the gate and commit**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`

```bash
git add algua/contracts/types.py tests/test_contracts.py
git commit -m "feat(contracts): news column constants (#132)"
```

---

### Task 2: Dataset.NEWS / Kind.NEWS enums

**Files:**
- Modify: `algua/data/models.py:12-27`
- Test: `tests/test_data_models.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_data_models.py`:

```python
def test_news_dataset_and_kind():
    from algua.data.models import Dataset, Kind

    assert Dataset.NEWS.value == "news"
    assert Kind.NEWS.value == "news"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_data_models.py::test_news_dataset_and_kind -v`
Expected: FAIL with `AttributeError: NEWS`.

- [ ] **Step 3: Add the enum members**

In `algua/data/models.py`, add `NEWS = "news"` to both enums:

```python
class Dataset(StrEnum):
    """Dataset routing key — the manifest `dataset` field and snapshot path component."""

    BARS = "bars"
    UNIVERSES = "universes"
    FUNDAMENTALS = "fundamentals"
    NEWS = "news"


class Kind(StrEnum):
    """Snapshot `kind` — the provenance of a snapshot's payload."""

    BARS = "bars"
    UNIVERSE = "universe"
    FILE = "file"
    FUNDAMENTALS = "fundamentals"
    NEWS = "news"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_data_models.py::test_news_dataset_and_kind -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add algua/data/models.py tests/test_data_models.py
git commit -m "feat(data): news Dataset/Kind enum values (#132)"
```

---

### Task 3: news_schema — validate_news + empty_news

**Files:**
- Create: `algua/data/news_schema.py`
- Test: `tests/test_news_schema.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_news_schema.py`:

```python
from datetime import datetime, timezone

import pandas as pd
import pytest

from algua.data.news_schema import NEWS_COLUMNS, empty_news, validate_news


def _row(**over):
    base = {
        "source": "reuters",
        "article_id": "a1",
        "symbol": "AAPL",
        "published_at": pd.Timestamp("2025-01-02T13:00:00Z"),
        "knowable_at": pd.Timestamp("2025-01-02T13:00:00Z"),
        "headline": "Apple ships",
        "url": "http://x/1",
        "body": "body text",
    }
    base.update(over)
    return base


def _frame(rows):
    return pd.DataFrame(rows, columns=list(NEWS_COLUMNS))


def test_valid_frame_passes_unchanged():
    df = _frame([_row()])
    assert validate_news(df) is df


def test_rejects_wrong_columns():
    df = _frame([_row()]).drop(columns=["body"])
    with pytest.raises(ValueError, match="columns"):
        validate_news(df)


def test_rejects_null_in_required_string():
    df = _frame([_row(headline=None)])
    with pytest.raises(ValueError, match="headline"):
        validate_news(df)


def test_allows_null_url_and_body():
    df = _frame([_row(url=pd.NA, body=pd.NA)])
    assert validate_news(df) is df


def test_rejects_tz_naive_timestamps():
    df = _frame([_row(knowable_at=pd.Timestamp("2025-01-02T13:00:00"))])
    with pytest.raises(ValueError, match="knowable_at"):
        validate_news(df)


def test_rejects_knowable_before_published():
    df = _frame([_row(
        published_at=pd.Timestamp("2025-01-02T13:00:00Z"),
        knowable_at=pd.Timestamp("2025-01-02T12:00:00Z"),
    )])
    with pytest.raises(ValueError, match="published_at"):
        validate_news(df)


def test_equal_published_and_knowable_passes():
    t = pd.Timestamp("2025-01-02T13:00:00Z")
    df = _frame([_row(published_at=t, knowable_at=t)])
    assert validate_news(df) is df


def test_rejects_duplicate_key():
    df = _frame([_row(), _row(headline="dup but same key")])
    # same (source, article_id, symbol, knowable_at) -> revision-content inconsistency or key dup
    with pytest.raises(ValueError):
        validate_news(df.sort_values(["symbol", "source", "article_id", "knowable_at"]).reset_index(drop=True))


def test_rejects_published_at_varying_within_article():
    df = _frame([
        _row(knowable_at=pd.Timestamp("2025-01-02T13:00:00Z"),
             published_at=pd.Timestamp("2025-01-02T13:00:00Z")),
        _row(knowable_at=pd.Timestamp("2025-01-03T13:00:00Z"),
             published_at=pd.Timestamp("2025-01-02T09:00:00Z"), headline="rev"),
    ])
    df = df.sort_values(["symbol", "source", "article_id", "knowable_at"]).reset_index(drop=True)
    with pytest.raises(ValueError, match="published_at"):
        validate_news(df)


def test_rejects_inconsistent_revision_content():
    t = pd.Timestamp("2025-01-02T13:00:00Z")
    df = _frame([
        _row(symbol="AAPL", knowable_at=t, headline="h1"),
        _row(symbol="MSFT", knowable_at=t, headline="h2"),  # same article revision, different headline
    ])
    df = df.sort_values(["symbol", "source", "article_id", "knowable_at"]).reset_index(drop=True)
    with pytest.raises(ValueError, match="headline"):
        validate_news(df)


def test_empty_news_is_contract_shaped():
    e = empty_news()
    assert list(e.columns) == list(NEWS_COLUMNS)
    assert len(e) == 0
    assert str(e["knowable_at"].dtype) == "datetime64[ns, UTC]"
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_news_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: algua.data.news_schema`.

- [ ] **Step 3: Create the module with constants, validate_news, empty_news**

Create `algua/data/news_schema.py`:

```python
from __future__ import annotations

import hashlib
import math
import struct

import numpy as np
import pandas as pd

from algua.contracts.types import NEWS_AS_OF_KEY, NEWS_COLUMNS, NEWS_KNOWABLE_AT

COLUMNS = list(NEWS_COLUMNS)
STRING_COLUMNS = ["source", "article_id", "symbol", "headline"]  # non-null strings
NULLABLE_STRING_COLUMNS = ["url", "body"]
TS_COLUMNS = ["published_at", NEWS_KNOWABLE_AT]
UNIQUE_KEY = [*NEWS_AS_OF_KEY, NEWS_KNOWABLE_AT]  # (source, article_id, symbol, knowable_at)
_SORT = ["symbol", "source", "article_id", NEWS_KNOWABLE_AT]


def _is_na(v: object) -> bool:
    return v is None or v is pd.NA or (isinstance(v, float) and math.isnan(v))


def validate_news(df: pd.DataFrame) -> pd.DataFrame:
    """Assert `df` matches the tidy/bitemporal news schema; return it unchanged on success.
    Raises ValueError describing the first violation."""
    if list(df.columns) != COLUMNS:
        raise ValueError(f"news columns must be {COLUMNS}, got {list(df.columns)}")
    for col in STRING_COLUMNS:
        if df[col].isna().any() or not all(isinstance(v, str) for v in df[col]):
            raise ValueError(f"news {col!r} must be non-null strings")
    for col in NULLABLE_STRING_COLUMNS:
        if not all(_is_na(v) or isinstance(v, str) for v in df[col]):
            raise ValueError(f"news {col!r} must be strings or null")
    for col in TS_COLUMNS:
        ts = df[col]
        if not isinstance(ts.dtype, pd.DatetimeTZDtype) or str(ts.dt.tz) != "UTC":
            raise ValueError(f"news {col!r} must be tz-aware UTC datetimes")
        if ts.isna().any():
            raise ValueError(f"news {col!r} must not be null")
    # PIT floor: a row cannot become knowable before it was published.
    if (df["knowable_at"].to_numpy() < df["published_at"].to_numpy()).any():
        raise ValueError("news 'knowable_at' must be >= 'published_at'")
    # Unique row key.
    if df[UNIQUE_KEY].duplicated().any():
        raise ValueError(f"news must have unique {tuple(UNIQUE_KEY)}")
    # published_at is invariant per (source, article_id) — it is an article-identity attribute.
    if (df.groupby(["source", "article_id"])["published_at"].nunique() > 1).any():
        raise ValueError("news 'published_at' must be invariant per (source, article_id)")
    # headline/url/body are invariant within one article revision (source, article_id, knowable_at).
    rev = df.groupby(["source", "article_id", "knowable_at"])
    for col in ["headline", *NULLABLE_STRING_COLUMNS]:
        if (rev[col].nunique(dropna=False) > 1).any():
            raise ValueError(f"news {col!r} must be identical within an article revision")
    if df.duplicated().any():
        raise ValueError("news must not contain exact-duplicate rows")
    expected = df.sort_values(_SORT).reset_index(drop=True)
    if not df.reset_index(drop=True).equals(expected):
        raise ValueError(f"news must be sorted by {_SORT}")
    return df


def empty_news() -> pd.DataFrame:
    """Contract-shaped empty news frame (exact columns + dtypes)."""
    data = {
        "source": pd.Series([], dtype="object"),
        "article_id": pd.Series([], dtype="object"),
        "symbol": pd.Series([], dtype="object"),
        "published_at": pd.Series([], dtype="datetime64[ns, UTC]"),
        "knowable_at": pd.Series([], dtype="datetime64[ns, UTC]"),
        "headline": pd.Series([], dtype="object"),
        "url": pd.Series([], dtype="object"),
        "body": pd.Series([], dtype="object"),
    }
    return validate_news(pd.DataFrame(data)[COLUMNS])
```

Re-export the constant so tests can import it from the module:

```python
NEWS_COLUMNS = tuple(COLUMNS)  # add at end of module for `from algua.data.news_schema import NEWS_COLUMNS`
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_news_schema.py -v`
Expected: PASS (all). If `test_rejects_duplicate_key` is ambiguous, it asserts only that *some* ValueError is raised — fine.

- [ ] **Step 5: Run the gate and commit**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`

```bash
git add algua/data/news_schema.py tests/test_news_schema.py
git commit -m "feat(data): news schema validator + empty (#132)"
```

---

### Task 4: news_schema — explode_news_symbols

**Files:**
- Modify: `algua/data/news_schema.py`
- Test: `tests/test_news_schema.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_news_schema.py`:

```python
from algua.data.news_schema import explode_news_symbols


def _raw_row(**over):
    base = {
        "source": "reuters",
        "article_id": "a1",
        "symbols": "AAPL,MSFT",
        "published_at": "2025-01-02T13:00:00Z",
        "knowable_at": "2025-01-02T13:00:00Z",
        "headline": "two names",
        "url": "http://x/1",
        "body": "b",
    }
    base.update(over)
    return base


def test_explode_comma_string():
    out = explode_news_symbols(pd.DataFrame([_raw_row()]))
    assert sorted(out["symbol"]) == ["AAPL", "MSFT"]
    assert set(out["headline"]) == {"two names"}
    assert list(out.columns) == list(NEWS_COLUMNS)


def test_explode_list_form_and_dedup_and_case():
    out = explode_news_symbols(pd.DataFrame([_raw_row(symbols=[" aapl ", "AAPL", "msft"])]))
    assert sorted(out["symbol"]) == ["AAPL", "MSFT"]  # stripped, upper, de-duped within article


def test_explode_adds_missing_optional_columns():
    row = _raw_row()
    del row["url"]
    del row["body"]
    out = explode_news_symbols(pd.DataFrame([row]))
    assert out["url"].isna().all() and out["body"].isna().all()


def test_explode_rejects_zero_symbols():
    with pytest.raises(ValueError, match="symbol"):
        explode_news_symbols(pd.DataFrame([_raw_row(symbols="  ,  ")]))


def test_explode_rejects_missing_required_input_column():
    row = _raw_row()
    del row["headline"]
    with pytest.raises(ValueError, match="missing"):
        explode_news_symbols(pd.DataFrame([row]))
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_news_schema.py -k explode -v`
Expected: FAIL with `ImportError: cannot import name 'explode_news_symbols'`.

- [ ] **Step 3: Add explode_news_symbols**

Add to `algua/data/news_schema.py` (before `to_news_schema`, which arrives in Task 5):

```python
_RAW_REQUIRED = ["source", "article_id", "symbols", "published_at", NEWS_KNOWABLE_AT, "headline"]


def explode_news_symbols(frame: pd.DataFrame) -> pd.DataFrame:
    """Ingest-only pre-step: turn a per-ARTICLE input (with a `symbols` field — a list, or a
    comma-delimited string) into one row per (article, symbol) with a canonical `symbol` column.
    Symbols are stripped/upper-cased, blanks dropped, de-duped within an article; an article with
    zero symbols is rejected. Optional `url`/`body` default to NA. Output carries NEWS_COLUMNS."""
    missing = [c for c in _RAW_REQUIRED if c not in frame.columns]
    if missing:
        raise ValueError(f"news input missing columns: {missing}")
    out = frame.copy()
    for opt in NULLABLE_STRING_COLUMNS:
        if opt not in out.columns:
            out[opt] = pd.NA

    def _parse(v: object) -> list[str]:
        items = list(v) if isinstance(v, (list, tuple, set, np.ndarray, pd.Series)) else str(v).split(",")
        seen: list[str] = []
        for s in items:
            s = str(s).strip().upper()
            if s and s not in seen:
                seen.append(s)
        return seen

    out["_syms"] = out["symbols"].apply(_parse)
    if (out["_syms"].apply(len) == 0).any():
        raise ValueError("each news article must tag >= 1 symbol (symbol-less news is out of scope)")
    out = (
        out.drop(columns=["symbols"])
        .explode("_syms", ignore_index=True)
        .rename(columns={"_syms": "symbol"})
    )
    return out[COLUMNS]
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_news_schema.py -k explode -v`
Expected: PASS.

- [ ] **Step 5: Run the gate and commit**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`

```bash
git add algua/data/news_schema.py tests/test_news_schema.py
git commit -m "feat(data): news entity->symbol explode pre-step (#132)"
```

---

### Task 5: news_schema — to_news_schema + logical_news_hash

**Files:**
- Modify: `algua/data/news_schema.py`
- Test: `tests/test_news_schema.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_news_schema.py`:

```python
from algua.data.news_schema import logical_news_hash, to_news_schema


def test_to_news_schema_normalizes_and_validates():
    raw = explode_news_symbols(pd.DataFrame([_raw_row(source="Reuters", symbols="aapl")]))
    canon = to_news_schema(raw)
    assert canon["source"].iloc[0] == "reuters"   # source canonicalized (strip+lower)
    assert canon["symbol"].iloc[0] == "AAPL"       # symbol upper
    assert str(canon["knowable_at"].dtype) == "datetime64[ns, UTC]"


def test_to_news_schema_is_idempotent():
    raw = explode_news_symbols(pd.DataFrame([_raw_row()]))
    once = to_news_schema(raw)
    twice = to_news_schema(once)
    assert once.equals(twice)


def test_to_news_schema_requires_knowable_at():
    raw = explode_news_symbols(pd.DataFrame([_raw_row()])).drop(columns=["knowable_at"])
    with pytest.raises(ValueError):
        to_news_schema(raw)


def test_to_news_schema_canonicalizes_null_distinct_from_empty():
    raw = explode_news_symbols(pd.DataFrame([_raw_row(body=None)]))
    canon = to_news_schema(raw)
    assert canon["body"].isna().all()


def test_hash_stable_under_row_order():
    a = to_news_schema(explode_news_symbols(pd.DataFrame([_raw_row(symbols="AAPL,MSFT")])))
    b = a.iloc[::-1].reset_index(drop=True)
    assert logical_news_hash(a) == logical_news_hash(to_news_schema(b))


def test_hash_distinguishes_null_empty_and_none_string():
    base = _raw_row()
    h_null = logical_news_hash(to_news_schema(explode_news_symbols(pd.DataFrame([{**base, "body": None}]))))
    h_empty = logical_news_hash(to_news_schema(explode_news_symbols(pd.DataFrame([{**base, "body": ""}]))))
    h_none = logical_news_hash(to_news_schema(explode_news_symbols(pd.DataFrame([{**base, "body": "None"}]))))
    assert len({h_null, h_empty, h_none}) == 3


def test_hash_changes_with_headline():
    h1 = logical_news_hash(to_news_schema(explode_news_symbols(pd.DataFrame([_raw_row(headline="a")]))))
    h2 = logical_news_hash(to_news_schema(explode_news_symbols(pd.DataFrame([_raw_row(headline="b")]))))
    assert h1 != h2
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_news_schema.py -k "to_news_schema or hash" -v`
Expected: FAIL with `ImportError: cannot import name 'to_news_schema'`.

- [ ] **Step 3: Add to_news_schema + logical_news_hash**

Add to `algua/data/news_schema.py`:

```python
def to_news_schema(frame: pd.DataFrame) -> pd.DataFrame:
    """Idempotent canonical normalizer for an already-exploded per-symbol frame (run by both
    ingest — after explode — and read). Canonicalizes source (strip+lower) and symbol (strip+upper),
    coerces dtypes, normalizes timestamps to UTC (naive rejected, knowable_at required), canonicalizes
    nullable url/body to pd.NA (distinct from ""), de-dups, sorts, validates."""
    missing = [c for c in COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"frame missing news columns: {missing}")
    out = frame[COLUMNS].copy()
    out["source"] = out["source"].astype(str).str.strip().str.lower()
    out["article_id"] = out["article_id"].astype(str)
    out["symbol"] = out["symbol"].astype(str).str.strip().str.upper()
    out["headline"] = out["headline"].astype(str)
    for col in NULLABLE_STRING_COLUMNS:
        out[col] = out[col].apply(lambda v: pd.NA if _is_na(v) else str(v))
    for col in TS_COLUMNS:
        ts = pd.to_datetime(out[col], errors="raise")
        if ts.dt.tz is None:
            raise ValueError(f"news {col!r} must be tz-aware (UTC); naive timestamps are rejected")
        out[col] = ts.dt.tz_convert("UTC")
    out = out.drop_duplicates().sort_values(_SORT).reset_index(drop=True)
    return validate_news(out)


def logical_news_hash(df: pd.DataFrame) -> str:
    """Deterministic content hash over the logical rows — the snapshot identity (mirrors
    logical_fundamentals_hash). Non-null strings length-prefixed UTF-8; nullable url/body carry a
    null-flag byte (so null, "", and "None" hash distinctly); timestamps as int64 ns UTC."""
    ordered = df.sort_values(_SORT, kind="stable").reset_index(drop=True)
    digest = hashlib.sha256()
    digest.update(struct.pack("<Q", len(ordered)))
    for col in STRING_COLUMNS:
        encoded = [s.encode("utf-8") for s in ordered[col].astype(str)]
        lengths = np.array([len(b) for b in encoded], dtype="<u8")
        digest.update(lengths.tobytes())
        digest.update(b"".join(encoded))
    for col in NULLABLE_STRING_COLUMNS:
        is_null = np.array([_is_na(v) for v in ordered[col]], dtype="u1")
        digest.update(is_null.tobytes())  # distinguishes null from "" and from "None"
        encoded = [("" if _is_na(v) else str(v)).encode("utf-8") for v in ordered[col]]
        lengths = np.array([len(b) for b in encoded], dtype="<u8")
        digest.update(lengths.tobytes())
        digest.update(b"".join(encoded))
    for col in TS_COLUMNS:
        naive = ordered[col].dt.tz_convert("UTC").dt.tz_localize(None)
        ns = naive.to_numpy(dtype="datetime64[ns]").view("int64").astype("<i8")
        digest.update(ns.tobytes())
    return digest.hexdigest()
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_news_schema.py -v`
Expected: PASS (whole file).

- [ ] **Step 5: Run the gate and commit**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`

```bash
git add algua/data/news_schema.py tests/test_news_schema.py
git commit -m "feat(data): news canonical normalizer + logical hash (#132)"
```

---

### Task 6: store — ingest_news + read_news

**Files:**
- Modify: `algua/data/store.py` (add two methods after `read_fundamentals`, ~line 502; add imports)
- Test: `tests/test_data_news_store.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_data_news_store.py`:

```python
import pandas as pd
import pytest

from algua.data.store import DataStore


def _raw(tmp_path):
    return pd.DataFrame([
        {"source": "Reuters", "article_id": "a1", "symbols": "AAPL,MSFT",
         "published_at": "2025-01-02T13:00:00Z", "knowable_at": "2025-01-02T13:00:00Z",
         "headline": "h", "url": "http://x/1", "body": "b"},
    ])


def test_ingest_then_read_roundtrip(tmp_path):
    store = DataStore(tmp_path)
    rec = store.ingest_news(provider="testfeed", as_of="2025-01-03T00:00:00Z", frame=_raw(tmp_path))
    assert rec.dataset == "news"
    out = store.read_news(rec.snapshot_id)
    assert sorted(out["symbol"]) == ["AAPL", "MSFT"]
    assert rec.metadata.source == "testfeed"             # metadata.source = provider label
    assert "reuters" in rec.metadata.source_metadata["row_sources"]


def test_ingest_is_deterministic_dedup(tmp_path):
    store = DataStore(tmp_path)
    r1 = store.ingest_news(provider="f", as_of="2025-01-03T00:00:00Z", frame=_raw(tmp_path))
    r2 = store.ingest_news(provider="f", as_of="2025-01-03T00:00:00Z", frame=_raw(tmp_path))
    assert r1.snapshot_id == r2.snapshot_id


def test_ingest_rejects_knowable_after_as_of(tmp_path):
    store = DataStore(tmp_path)
    with pytest.raises(ValueError, match="as_of"):
        store.ingest_news(provider="f", as_of="2025-01-01T00:00:00Z", frame=_raw(tmp_path))


def test_read_news_symbol_pushdown(tmp_path):
    store = DataStore(tmp_path)
    rec = store.ingest_news(provider="f", as_of="2025-01-03T00:00:00Z", frame=_raw(tmp_path))
    out = store.read_news(rec.snapshot_id, symbols=["AAPL"])
    assert list(out["symbol"].unique()) == ["AAPL"]
```

(If `DataStore`'s constructor signature differs, match the existing fundamentals store tests in `tests/test_data_fundamentals_store.py` — copy their fixture for building a store.)

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_data_news_store.py -v`
Expected: FAIL with `AttributeError: 'DataStore' object has no attribute 'ingest_news'`.

- [ ] **Step 3: Add the store methods**

In `algua/data/store.py`, add to the imports from `news_schema` (mirror the fundamentals import line):

```python
from algua.data.news_schema import (
    empty_news,
    explode_news_symbols,
    logical_news_hash,
    to_news_schema,
)
```

Add these two methods to the `DataStore` class, right after `read_fundamentals`:

```python
    def ingest_news(
        self,
        *,
        provider: str,
        as_of: str,
        frame: pd.DataFrame,
        source_metadata: dict[str, str] | None = None,
    ) -> SnapshotRecord:
        """Explode + normalize a per-article news frame and persist one immutable snapshot.
        `start`/`end` and the covered symbol/source sets are DERIVED from the data; every
        knowable_at must be <= `as_of`. `metadata.source` is the ingest `provider` label; the
        derived row-source/symbol sets live in `source_metadata` (multi-source dataset)."""
        canon = to_news_schema(explode_news_symbols(frame))
        if canon.empty:
            raise ValueError("cannot ingest an empty news frame")
        as_of_ts = pd.Timestamp(as_of)
        as_of_ts = (
            as_of_ts.tz_localize("UTC") if as_of_ts.tzinfo is None else as_of_ts.tz_convert("UTC")
        )
        if (canon["knowable_at"] > as_of_ts).any():
            raise ValueError(
                "news knowable_at must be <= as_of "
                "(cannot ingest a record knowable after the fetch time)"
            )
        start = canon["knowable_at"].min().date().isoformat()
        end = canon["knowable_at"].max().date().isoformat()
        symbols = sorted(canon["symbol"].unique())
        sources = sorted(canon["source"].unique())
        derived = {"row_sources": ",".join(sources), "row_symbols": ",".join(symbols)}
        if source_metadata:
            derived.update(source_metadata)
        metadata = _metadata(
            dataset=Dataset.NEWS.value,
            provider=provider,
            symbols=symbols,
            start=start,
            end=end,
            as_of=as_of,
            source=provider,
            kind=Kind.NEWS.value,
            source_metadata=derived,
        )
        content_hash = logical_news_hash(canon)
        snapshot_id = _snapshot_id(metadata, content_hash)
        existing = self.manifest.find(snapshot_id)
        if existing is not None:
            return existing
        relative_path = Path("snapshots") / metadata.dataset / snapshot_id / "news.parquet"
        write_bytes_snapshot(frame_to_parquet_bytes(canon), self.data_dir, relative_path)
        rec = SnapshotRecord(
            snapshot_id=snapshot_id,
            metadata=metadata,
            row_count=len(canon),
            content_hash=content_hash,
            data_path=relative_path,
            created_at=datetime.now(UTC).isoformat(),
            storage_format="parquet",
        )
        self.manifest.append(rec)
        return rec

    def read_news(self, snapshot_id: str, *, symbols: list[str] | None = None) -> pd.DataFrame:
        """Read a news snapshot as a validated tidy frame. `symbols` filters in-memory.
        Re-normalizes on read (idempotent) so parquet dtype drift cannot escape the schema.
        Empty => empty_news()."""
        rec = self.get_snapshot(snapshot_id)
        if rec.dataset != Dataset.NEWS.value:
            raise ValueError(
                f"snapshot {snapshot_id} is dataset {rec.dataset!r}, not {Dataset.NEWS.value!r}"
            )
        raw = pd.read_parquet(self.data_dir / rec.data_path)
        if symbols is not None:
            wanted = set(normalize_symbols(symbols))
            raw = raw[raw["symbol"].astype(str).str.upper().isin(wanted)]
        if raw.empty:
            return empty_news()
        return to_news_schema(raw)
```

(`_metadata`, `_snapshot_id`, `write_bytes_snapshot`, `frame_to_parquet_bytes`, `normalize_symbols`, `Dataset`, `Kind`, `Path`, `datetime`, `UTC`, `SnapshotRecord` are already imported/used by `ingest_fundamentals` in this file — reuse them.)

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_data_news_store.py -v`
Expected: PASS.

- [ ] **Step 5: Run the gate and commit**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`

```bash
git add algua/data/store.py tests/test_data_news_store.py
git commit -m "feat(data): ingest_news + read_news snapshots (#132)"
```

---

### Task 7: hindsight — query_news

**Files:**
- Modify: `algua/data/hindsight.py`
- Test: `tests/test_news_serve_hindsight.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_news_serve_hindsight.py`:

```python
import pandas as pd

from algua.data.hindsight import query_news
from algua.data.store import DataStore


def _raw():
    return pd.DataFrame([
        {"source": "reuters", "article_id": "a1", "symbols": "AAPL,MSFT",
         "published_at": "2025-01-02T13:00:00Z", "knowable_at": "2025-01-02T13:00:00Z",
         "headline": "h", "url": None, "body": None},
    ])


def test_query_news_returns_full_history(tmp_path):
    store = DataStore(tmp_path)
    rec = store.ingest_news(provider="f", as_of="2025-01-03T00:00:00Z", frame=_raw())
    out = query_news(store, rec.snapshot_id)
    assert sorted(out["symbol"]) == ["AAPL", "MSFT"]


def test_query_news_symbol_filter(tmp_path):
    store = DataStore(tmp_path)
    rec = store.ingest_news(provider="f", as_of="2025-01-03T00:00:00Z", frame=_raw())
    out = query_news(store, rec.snapshot_id, symbols=["MSFT"])
    assert list(out["symbol"].unique()) == ["MSFT"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_news_serve_hindsight.py -v`
Expected: FAIL with `ImportError: cannot import name 'query_news'`.

- [ ] **Step 3: Add query_news**

Append to `algua/data/hindsight.py`:

```python
def query_news(
    store: DataStore, snapshot_id: str, symbols: list[str] | None = None
) -> pd.DataFrame:
    """Full-hindsight news read (no as-of masking) — the agent's post-mortem/analysis surface.
    Wraps store.read_news, which returns the canonical sort for reproducible agent diffs."""
    return store.read_news(snapshot_id, symbols=symbols)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_news_serve_hindsight.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add algua/data/hindsight.py tests/test_news_serve_hindsight.py
git commit -m "feat(data): query_news hindsight accessor (#132)"
```

---

### Task 8: The wall — pyproject contract + wall test

**Files:**
- Modify: `pyproject.toml` (add one contract in `[tool.importlinter]`)
- Test: `tests/test_news_wall.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_news_wall.py`:

```python
import pathlib
import tomllib

REPO = pathlib.Path(__file__).resolve().parents[1]


def _contracts():
    data = tomllib.loads((REPO / "pyproject.toml").read_text())
    return data["tool"]["importlinter"]["contracts"]


def test_live_execution_barred_from_data_lane():
    cs = _contracts()
    assert any(
        set(c.get("source_modules", [])) == {"algua.live", "algua.execution"}
        and c.get("forbidden_modules") == ["algua.data"]
        for c in cs
    ), "missing: algua.live/algua.execution forbidden from algua.data"


def test_query_news_lives_in_walled_hindsight_module():
    import algua.data.hindsight as h

    assert h.query_news.__module__ == "algua.data.hindsight"


def test_news_full_history_read_not_on_a_decision_lane_module():
    # The only full-history news read surfaces are store.read_news (in algua.data, walled from all
    # decision/execution lanes) and hindsight.query_news (in the walled module). Assert no news read
    # leaked onto contracts.types (an importable base layer).
    import algua.contracts.types as t

    assert not hasattr(t, "read_news") and not hasattr(t, "query_news")
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_news_wall.py -v`
Expected: `test_live_execution_barred_from_data_lane` FAILS (contract not present yet); the other two PASS.

- [ ] **Step 3: Add the import-linter contract**

In `pyproject.toml`, after the existing "strategies layer stays off the data lane" / hindsight contracts block, add:

```toml
[[tool.importlinter.contracts]]
# Live/execution receive market data via the CLI composition seam, never by importing algua.data.
# Barring them completes the data wall so NO decision/execution lane can reach an unmasked
# full-history read (algua.data.store.read_news / read_fundamentals); only the walled hindsight
# accessors and the CLI may. (issue #132 — news hindsight slice.)
name = "live and execution stay off the data lane"
type = "forbidden"
source_modules = ["algua.live", "algua.execution"]
forbidden_modules = ["algua.data"]
```

- [ ] **Step 4: Run to verify the tests + lint-imports pass**

Run: `uv run pytest tests/test_news_wall.py -v`
Expected: PASS (all three).

Run: `uv run lint-imports`
Expected: all contracts KEPT (the new one included). **If the new contract is BROKEN**, a real hidden `live`/`execution` → `algua.data` dependency exists — STOP and surface it (do not weaken the contract to make it pass).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml tests/test_news_wall.py
git commit -m "feat(wall): bar live/execution from algua.data — complete the data wall (#132)"
```

---

### Task 9: CLI — ingest-news + query-news

**Files:**
- Modify: `algua/cli/data_cmd.py` (add two commands after `query-fundamentals`; extend the hindsight import)
- Test: `tests/test_cli_data_news.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cli_data_news.py`:

```python
import json

import pandas as pd
from typer.testing import CliRunner

from algua.cli.main import app

runner = CliRunner()


def _write_csv(tmp_path):
    p = tmp_path / "news.csv"
    pd.DataFrame([
        {"source": "reuters", "article_id": "a1", "symbols": "AAPL,MSFT",
         "published_at": "2025-01-02T13:00:00Z", "knowable_at": "2025-01-02T13:00:00Z",
         "headline": "h", "url": "http://x/1", "body": "b"},
    ]).to_csv(p, index=False)
    return p


def test_ingest_then_query_news(tmp_path, monkeypatch):
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path / "data"))  # match the fundamentals CLI test's env
    csv = _write_csv(tmp_path)
    r = runner.invoke(app, ["data", "ingest-news", "--provider", "f",
                            "--as-of", "2025-01-03T00:00:00Z", "--from-file", str(csv)])
    assert r.exit_code == 0, r.output
    snap = json.loads(r.output)["snapshot"]["snapshot_id"]
    q = runner.invoke(app, ["data", "query-news", "--snapshot-id", snap])
    assert q.exit_code == 0, q.output
    rows = json.loads(q.output)
    assert sorted(x["symbol"] for x in rows) == ["AAPL", "MSFT"]
    assert rows[0]["url"] in ("http://x/1", None)  # null-safe
```

(Match the store/env wiring used by `tests/test_cli_data_fundamentals.py` — copy its fixture/monkeypatch exactly if `ALGUA_DATA_DIR` is not the mechanism.)

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_cli_data_news.py -v`
Expected: FAIL (no `ingest-news` command).

- [ ] **Step 3: Add the CLI commands**

In `algua/cli/data_cmd.py`, extend the hindsight import to include `query_news`:

```python
from algua.data.hindsight import query_fundamentals, query_news
```

Add after `query_fundamentals_cmd`:

```python
@data_app.command("ingest-news")
@json_errors(ValueError, LookupError, FileNotFoundError)
def ingest_news(
    provider: str = typer.Option(..., "--provider"),
    as_of: str = typer.Option(..., "--as-of", help="point-in-time ISO datetime"),
    from_file: Path = FROM_FILE_OPTION,
) -> None:
    """Ingest a local news file (CSV/parquet) as one validated snapshot (hindsight lane).

    Rows carry: source, article_id, symbols, published_at, knowable_at, headline, [url], [body].
    `source` is a required per-row column and the covered symbol/source sets are derived, so there
    is no --source/--symbols flag; --provider is the ingest label."""
    path = from_file.expanduser()
    raw = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    rec = _store().ingest_news(provider=provider, as_of=as_of, frame=raw)
    emit(ok({"snapshot": rec.to_dict()}))


@data_app.command("query-news")
@json_errors(ValueError, LookupError, FileNotFoundError)
def query_news_cmd(
    snapshot_id: str = typer.Option(..., "--snapshot-id"),
    symbols: str = typer.Option(None, "--symbols", help="optional comma-separated subset"),
) -> None:
    """HINDSIGHT news read (full history) — the agent's post-mortem/analysis surface."""
    syms = normalize_symbols(symbols.split(",")) if symbols else None
    frame = query_news(_store(), snapshot_id, symbols=syms)
    records = [
        {
            "source": row.source,
            "article_id": row.article_id,
            "symbol": row.symbol,
            "published_at": row.published_at.isoformat(),
            "knowable_at": row.knowable_at.isoformat(),
            "headline": row.headline,
            "url": None if pd.isna(row.url) else str(row.url),
            "body": None if pd.isna(row.body) else str(row.body),
        }
        for row in frame.itertuples(index=False)
    ]
    emit(records)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_cli_data_news.py -v`
Expected: PASS.

- [ ] **Step 5: Run the gate and commit**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`

```bash
git add algua/cli/data_cmd.py tests/test_cli_data_news.py
git commit -m "feat(cli): data ingest-news + query-news (hindsight) (#132)"
```

---

### Task 10: Contract doc

**Files:**
- Create: `docs/contracts/news-schema.md`

- [ ] **Step 1: Write the doc**

Create `docs/contracts/news-schema.md` (parallel to `docs/contracts/fundamentals-schema.md`):

```markdown
# News schema (the non-tabular PIT seam — hindsight slice)

Tidy/long, bitemporal. One row = one mentioned symbol of one article revision, stamped with when
it became knowable.

| column | type | meaning |
|---|---|---|
| `source` | str (lower-cased, non-null) | publisher / wire; part of the identity (article ids are unique only within a source) |
| `article_id` | str (non-null) | the source's stable article id (or URL) |
| `symbol` | str (upper-cased, non-null) | one mentioned issuer (an article with N symbols → N rows) |
| `published_at` | tz-aware UTC datetime (non-null) | original publication time (invariant per article) |
| `knowable_at` | tz-aware UTC datetime (non-null) | when this row became knowable; the PIT key |
| `headline` | str (non-null) | the headline text |
| `url` | str or null | article link |
| `body` | str or null | full article text |

## Point-in-time rule
A record is knowable at `t` iff `knowable_at <= t`. **This slice is hindsight-only** — `query-news`
returns full history and is never wired into a decision. A future as-of signal lane would mask on
`knowable_at` exactly as fundamentals does.

## Identity & revisions
- As-of identity key: `(source, article_id, symbol)`; unique row key adds `knowable_at`.
- A **content** revision (corrected headline/body) is a new row sharing the identity key with a
  later `knowable_at`. Symbol-set revisions (adding/removing tagged tickers) need tombstones and are
  deferred with the signal lane.
- `published_at` is invariant per `(source, article_id)`; `headline`/`url`/`body` are invariant per
  `(source, article_id, knowable_at)` (one revision exploded across symbols).

## Validation floor
`knowable_at >= published_at`. Unique `(source, article_id, symbol, knowable_at)` within a snapshot.

## Two access modes
- **As-of (signal):** DEFERRED (no `NewsProvider`/`needs_news` this slice).
- **Hindsight (analysis):** `algua data query-news` (full history) — agent post-mortems / idea
  sourcing only. Structurally walled from every decision/execution lane by `lint-imports`
  (`algua.data` is forbidden to `backtest`/`features`/`strategies`/`contracts`/`live`/`execution`;
  `algua.data.hindsight` is forbidden directly). The wall is a STATIC import-graph guarantee, not a
  runtime sandbox.

## Ingest
`algua data ingest-news --from-file PATH --provider P --as-of TS`. Input rows carry `source`,
`article_id`, `symbols` (list or comma-string), `published_at`, `knowable_at`, `headline`, and
optionally `url`/`body`. `knowable_at` is required (never defaulted). `metadata.source` is the
`--provider` label; the derived row-source/symbol sets are in `source_metadata`.
```

- [ ] **Step 2: Verify gate (docs don't break anything)**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all green.

- [ ] **Step 3: Commit**

```bash
git add docs/contracts/news-schema.md
git commit -m "docs(contract): news schema doc (#132)"
```

---

## Final verification

- [ ] Run the full gate: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
- [ ] Confirm `lint-imports` reports the new "live and execution stay off the data lane" contract KEPT.
- [ ] Manual smoke (optional): write a small CSV, `uv run algua data ingest-news ...`, then `uv run algua data query-news --snapshot-id <id>` and confirm JSON.
- [ ] Spec coverage check: every §8 module-plan row maps to a task (Task 1=constants, 2=models, 3-5=news_schema, 6=store, 7=hindsight, 8=wall, 9=CLI, 10=doc). ✓
