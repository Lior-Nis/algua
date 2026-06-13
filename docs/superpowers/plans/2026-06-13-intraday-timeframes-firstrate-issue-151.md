# Intraday Timeframes (FirstRate) Implementation Plan — issue #151

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lift the `1d`-only restriction in the FirstRate importer so intraday bars (`1m`/`5m`/`30m`/`1h`) import to the bar-schema with timezone-aware, DST-correct, never-silently-shifted timestamps.

**Architecture:** A new pure `algua/data/timeframes.py` owns the canonical timeframe vocabulary and a daily-vs-intraday classifier. `parse_firstrate_file` becomes timeframe-aware: daily keeps UTC-midnight (now asserted); intraday localizes naive US/Eastern wall-clock to `America/New_York` and converts to UTC, failing closed on DST-ambiguous/nonexistent times and on tz-aware or local-midnight inputs. The vocabulary is enforced fail-closed at both bars-snapshot ingest chokepoints and at the serving seam. The partitioned write/read path is otherwise unchanged (already intraday-ready).

**Tech Stack:** Python 3.12, pandas, pytz (transitive via pandas — for DST exception types), typer CLI, pytest.

**Spec:** `docs/superpowers/specs/2026-06-13-intraday-timeframes-firstrate-issue-151-design.md`

**Quality gate (run between tasks):** `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`

---

## File Structure

- **Create** `algua/data/timeframes.py` — canonical vocabulary (`DAILY`, `INTRADAY`, `KNOWN`) + `validate_timeframe` + `is_intraday`. Pure stdlib, no imports from algua.
- **Create** `tests/test_timeframes.py` — unit tests for the vocab module.
- **Modify** `algua/data/importers/firstrate.py` — timeframe-aware parse + localization; lift the guard; full-timestamp diagnostic messages.
- **Modify** `tests/test_firstrate_importer.py` — intraday parse/import tests; update the now-stale `test_bad_timeframe_errors`.
- **Modify** `algua/data/store.py` — `validate_timeframe` guard in `ingest_bars` and `ingest_bars_streamed`.
- **Modify** `algua/data/serve.py` — `validate_timeframe` guard in `StoreBackedProvider.get_bars`.
- **Modify** `tests/test_data_store.py` + `tests/test_data_serve.py` — vocab-guard tests.
- **Modify** `tests/test_cli_import_bars.py` — end-to-end intraday import → read-back round-trip.
- **Modify** `docs/contracts/bar-schema.md` — reconcile timeframe vocabulary + fix the "UTC-aligned" wording.

---

## Task 1: Canonical timeframe vocabulary module

**Files:**
- Create: `algua/data/timeframes.py`
- Test: `tests/test_timeframes.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_timeframes.py
import pytest

from algua.data.timeframes import (
    DAILY,
    INTRADAY,
    KNOWN,
    is_intraday,
    validate_timeframe,
)


def test_known_set_is_daily_plus_intraday():
    assert DAILY == "1d"
    assert INTRADAY == frozenset({"1m", "5m", "30m", "1h"})
    assert KNOWN == frozenset({"1d", "1m", "5m", "30m", "1h"})


@pytest.mark.parametrize("tf", ["1d", "1m", "5m", "30m", "1h"])
def test_validate_timeframe_accepts_known(tf):
    assert validate_timeframe(tf) == tf


@pytest.mark.parametrize("tf", ["5min", "1hr", "15m", "2h", "", "1D"])
def test_validate_timeframe_rejects_unknown(tf):
    with pytest.raises(ValueError, match="unknown timeframe"):
        validate_timeframe(tf)


def test_is_intraday_classifies():
    assert is_intraday("1d") is False
    for tf in ["1m", "5m", "30m", "1h"]:
        assert is_intraday(tf) is True


def test_is_intraday_rejects_unknown():
    with pytest.raises(ValueError, match="unknown timeframe"):
        is_intraday("nope")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_timeframes.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'algua.data.timeframes'`

- [ ] **Step 3: Write the module**

```python
# algua/data/timeframes.py
"""Canonical timeframe vocabulary for the data layer (issue #151).

Single source of truth for the closed set of bar timeframes the system accepts, plus the
daily-vs-intraday classification the FirstRate importer keys its tz-localization off. Pure
(stdlib only) so any data-layer module can import it without a cycle.
"""
from __future__ import annotations

DAILY = "1d"
# FirstRate's actual intraday offerings. Extend deliberately if a new vendor needs another token.
INTRADAY: frozenset[str] = frozenset({"1m", "5m", "30m", "1h"})
KNOWN: frozenset[str] = frozenset({DAILY, *INTRADAY})


def validate_timeframe(timeframe: str) -> str:
    """Return `timeframe` if it is a known token; raise `ValueError` otherwise."""
    if timeframe not in KNOWN:
        raise ValueError(
            f"unknown timeframe {timeframe!r}; expected one of {sorted(KNOWN)}"
        )
    return timeframe


def is_intraday(timeframe: str) -> bool:
    """True for an intraday timeframe, False for daily. Validates first (raises on unknown)."""
    return validate_timeframe(timeframe) in INTRADAY
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_timeframes.py -q`
Expected: PASS (all tests green)

- [ ] **Step 5: Commit**

```bash
git add algua/data/timeframes.py tests/test_timeframes.py
git commit -m "feat(151): canonical timeframe vocabulary module"
```

---

## Task 2: Timeframe-aware FirstRate parse + tz localization

**Files:**
- Modify: `algua/data/importers/firstrate.py` (imports, new `_EXCHANGE_TZ` constant, new `_localize_timestamps` helper, `parse_firstrate_file` signature)
- Test: `tests/test_firstrate_importer.py`

- [ ] **Step 1: Write the failing tests**

Add these tests to `tests/test_firstrate_importer.py` (the existing `_write` helper writes raw rows; `parse_firstrate_file` is already imported there):

```python
def test_parse_intraday_summer_edt_to_utc(tmp_path):
    # 09:30 ET in July is EDT (UTC-4) -> 13:30 UTC.
    f = tmp_path / "AAPL_full_1min_UNADJUSTED.txt"
    _write(f, ["2024-07-01 09:30:00,10.0,11.0,9.5,10.5,1000\n"])
    out = parse_firstrate_file(f, timeframe="1m")
    assert out["ts"].iloc[0] == pd.Timestamp("2024-07-01 13:30:00", tz="UTC")
    assert str(out["ts"].dt.tz) == "UTC"


def test_parse_intraday_winter_est_to_utc(tmp_path):
    # 09:30 ET in January is EST (UTC-5) -> 14:30 UTC.
    f = tmp_path / "AAPL_full_1min_UNADJUSTED.txt"
    _write(f, ["2024-01-02 09:30:00,10.0,11.0,9.5,10.5,1000\n"])
    out = parse_firstrate_file(f, timeframe="1m")
    assert out["ts"].iloc[0] == pd.Timestamp("2024-01-02 14:30:00", tz="UTC")


def test_parse_intraday_dst_adjacent_valid(tmp_path):
    # 03:00 ET on spring-forward day (just after the gap) -> 07:00 UTC (EDT).
    # 00:30 ET on fall-back day (occurs once, unambiguous) -> 04:30 UTC (still EDT).
    f1 = tmp_path / "AAPL_full_1min_UNADJUSTED.txt"
    _write(f1, ["2024-03-10 03:00:00,1,1,1,1,1\n"])
    assert parse_firstrate_file(f1, timeframe="1m")["ts"].iloc[0] == pd.Timestamp(
        "2024-03-10 07:00:00", tz="UTC"
    )
    f2 = tmp_path / "MSFT_full_1min_UNADJUSTED.txt"
    _write(f2, ["2024-11-03 00:30:00,1,1,1,1,1\n"])
    assert parse_firstrate_file(f2, timeframe="1m")["ts"].iloc[0] == pd.Timestamp(
        "2024-11-03 04:30:00", tz="UTC"
    )


def test_parse_intraday_nonexistent_dst_time_raises(tmp_path):
    # 02:30 ET on 2024-03-10 does not exist (spring-forward gap).
    f = tmp_path / "AAPL_full_1min_UNADJUSTED.txt"
    _write(f, ["2024-03-10 02:30:00,1,1,1,1,1\n"])
    with pytest.raises(ValueError, match="DST-ambiguous or nonexistent"):
        parse_firstrate_file(f, timeframe="1m")


def test_parse_intraday_ambiguous_dst_time_raises(tmp_path):
    # 01:30 ET on 2024-11-03 occurs twice (fall-back) -> ambiguous.
    f = tmp_path / "AAPL_full_1min_UNADJUSTED.txt"
    _write(f, ["2024-11-03 01:30:00,1,1,1,1,1\n"])
    with pytest.raises(ValueError, match="DST-ambiguous or nonexistent"):
        parse_firstrate_file(f, timeframe="1m")


def test_parse_intraday_tz_aware_input_raises(tmp_path):
    # A wall-clock with an explicit offset is tz-aware -> rejected (wall-clock tz unknowable).
    f = tmp_path / "AAPL_full_1min_UNADJUSTED.txt"
    _write(f, ["2024-07-01 09:30:00-04:00,1,1,1,1,1\n"])
    with pytest.raises(ValueError, match="must be naive"):
        parse_firstrate_file(f, timeframe="1m")


def test_parse_intraday_local_midnight_rejected(tmp_path):
    # A date-only / local-midnight value under an intraday timeframe = daily file misfed.
    f = tmp_path / "AAPL_full_1min_UNADJUSTED.txt"
    _write(f, ["2024-07-01,1,1,1,1,1\n"])
    with pytest.raises(ValueError, match="local-midnight"):
        parse_firstrate_file(f, timeframe="1m")


def test_parse_daily_nonmidnight_rejected(tmp_path):
    # An intraday-shaped value under timeframe=1d -> non-midnight 1d bar = contract violation.
    f = tmp_path / "AAPL_full_1day_UNADJUSTED.txt"
    _write(f, ["2024-07-01 09:30:00,1,1,1,1,1\n"])
    with pytest.raises(ValueError, match="non-midnight"):
        parse_firstrate_file(f, timeframe="1d")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_firstrate_importer.py -q -k "intraday or nonmidnight"`
Expected: FAIL — `parse_firstrate_file()` takes no `timeframe` argument (TypeError) / behaviors absent.

- [ ] **Step 3: Update imports and add the `_EXCHANGE_TZ` constant + `_localize_timestamps` helper**

In `algua/data/importers/firstrate.py`, add `import pytz` to the imports block and the timeframes import:

```python
import pytz

from algua.data.timeframes import is_intraday
```

(Keep the existing `from algua.data.contracts import ...` and `from algua.data.store import normalize_symbols` lines.)

Add the constant near the other module constants (after `_UNADJUSTED_TOKEN_RE`):

```python
# FirstRate US-equity intraday timestamps are naive wall-clock in US/Eastern.
_EXCHANGE_TZ = "America/New_York"
```

Add the helper above `parse_firstrate_file`:

```python
def _localize_timestamps(parsed: pd.Series, timeframe: str, fname: str) -> pd.Series:
    """Localize a parsed `datetime` series to tz-aware UTC for the given timeframe.

    Daily: a bare date (naive) -> UTC midnight; a tz-aware value -> converted to UTC; then every
    value MUST be UTC midnight (a non-midnight `1d` timestamp means an intraday file was imported
    as daily — fail closed, parity with the Databento importer).

    Intraday: the input MUST be naive ET wall-clock. A tz-aware column is rejected (its wall-clock
    tz is unknowable). A local-midnight value is rejected (a 00:00 bar is never a valid FirstRate
    US-equity intraday bar -> a date-only/daily file was misfed). Otherwise localize to
    `America/New_York` (DST-ambiguous/nonexistent -> ValueError naming the time) and convert to UTC.
    """
    if not is_intraday(timeframe):
        utc = parsed.dt.tz_localize("UTC") if parsed.dt.tz is None else parsed.dt.tz_convert("UTC")
        if len(utc) and not bool((utc == utc.dt.normalize()).all()):
            bad = utc[utc != utc.dt.normalize()].iloc[0]
            raise ValueError(
                f"{fname}: 1d file has a non-midnight timestamp ({bad}); "
                "looks like an intraday file imported as daily"
            )
        return utc
    if parsed.dt.tz is not None:
        raise ValueError(
            f"{fname}: FirstRate intraday timestamps must be naive (wall-clock US/Eastern); "
            "found a tz-aware column — strip timezone offsets before importing"
        )
    if len(parsed) and bool((parsed == parsed.dt.normalize()).any()):
        bad = parsed[parsed == parsed.dt.normalize()].iloc[0]
        raise ValueError(
            f"{fname}: intraday file has a local-midnight timestamp ({bad}); "
            "looks like a date-only/daily file imported as intraday"
        )
    try:
        local = parsed.dt.tz_localize(_EXCHANGE_TZ, ambiguous="raise", nonexistent="raise")
    except (pytz.exceptions.AmbiguousTimeError, pytz.exceptions.NonExistentTimeError) as exc:
        raise ValueError(
            f"{fname}: DST-ambiguous or nonexistent local time in {_EXCHANGE_TZ}: {exc}"
        ) from exc
    return local.dt.tz_convert("UTC")
```

- [ ] **Step 4: Rewrite `parse_firstrate_file` to take `timeframe` and use the helper**

Change the signature and the timestamp-localization block. The full function becomes:

```python
def parse_firstrate_file(path: Path, timeframe: str = "1d") -> pd.DataFrame:
    """Parse one FirstRate file into a frame with columns `ts, open, high, low, close, volume`.

    `ts` is a tz-aware UTC timestamp. For `timeframe="1d"` the source is a bare date localized to
    UTC midnight; for an intraday `timeframe` the source is naive US/Eastern wall-clock localized to
    `America/New_York` and converted to UTC (see `_localize_timestamps`).

    Handles a present-or-absent header (sniffed from the first non-empty line). Raises ValueError on
    a malformed file (wrong column count, unparseable dates/numbers) or a timeframe/timestamp
    mismatch (non-midnight daily, tz-aware or local-midnight intraday, DST-invalid local time).
    """
    first_line = ""
    with path.open("r", encoding="utf-8-sig") as fh:
        for line in fh:
            if line.strip():
                first_line = line.strip().lower()
                break
    has_header = first_line.startswith("datetime")
    frame = pd.read_csv(
        path,
        header=0 if has_header else None,
        names=None if has_header else _FIRSTRATE_COLUMNS,
        encoding="utf-8-sig",
    )
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    missing = [c for c in _FIRSTRATE_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"{path.name}: FirstRate file missing columns {missing}")
    out = frame[_FIRSTRATE_COLUMNS].rename(columns={"datetime": "ts"})
    parsed = pd.to_datetime(out["ts"], errors="raise")
    out["ts"] = _localize_timestamps(parsed, timeframe, path.name)
    for col in [*_PRICE_COLUMNS, "volume"]:
        out[col] = pd.to_numeric(out[col], errors="raise").astype("float64")
    return out
```

- [ ] **Step 5: Run the new tests to verify they pass**

Run: `uv run pytest tests/test_firstrate_importer.py -q -k "intraday or nonmidnight"`
Expected: PASS

- [ ] **Step 6: Run the full importer test file (daily regression)**

Run: `uv run pytest tests/test_firstrate_importer.py -q`
Expected: PASS except `test_bad_timeframe_errors` (it still asserts the old "not yet supported" message for `1h`) — that test is updated in Task 3. If only that test fails here, that is expected.

- [ ] **Step 7: Commit**

```bash
git add algua/data/importers/firstrate.py tests/test_firstrate_importer.py
git commit -m "feat(151): timeframe-aware FirstRate parse with ET->UTC localization"
```

---

## Task 3: Lift the importer guard, thread timeframe, fix diagnostic messages

**Files:**
- Modify: `algua/data/importers/firstrate.py` (`FirstRateImporter.import_bars`, `_merge_symbol`)
- Test: `tests/test_firstrate_importer.py`

- [ ] **Step 1: Write/replace the failing tests**

In `tests/test_firstrate_importer.py`, REPLACE `test_bad_timeframe_errors` with the two tests below, and add the intraday import + duplicate-message tests. Also add the `30m` round-trip test:

```python
def test_unknown_timeframe_errors(tmp_path):
    raw, adj = _firstrate_dirs(tmp_path)
    _write_pair(raw, adj, "AAPL", ["2024-07-01,1,1,1,1,1\n"], ["2024-07-01,1,1,1,1,1\n"])
    with pytest.raises(ValueError, match="unknown timeframe"):
        list(FirstRateImporter().import_bars(
            FirstRateImportRequest(raw_dir=raw, adjusted_dir=adj, timeframe="5min")))


def _write_intraday_pair(raw, adj, symbol, raw_rows, adj_rows):
    (raw / f"{symbol}_full_1min_UNADJUSTED.txt").write_text("".join(raw_rows), encoding="utf-8")
    (adj / f"{symbol}_full_1min_adjsplitdiv.txt").write_text("".join(adj_rows), encoding="utf-8")


def test_intraday_import_merges_and_localizes(tmp_path):
    raw, adj = _firstrate_dirs(tmp_path)
    _write_intraday_pair(
        raw, adj, "AAPL",
        ["2024-07-01 09:30:00,100,110,95,105,1000\n", "2024-07-01 09:31:00,105,120,100,115,2000\n"],
        ["2024-07-01 09:30:00,50,55,47,52,1000\n", "2024-07-01 09:31:00,52,60,50,57,2000\n"],
    )
    req = FirstRateImportRequest(raw_dir=raw, adjusted_dir=adj, timeframe="1m")
    chunks = list(FirstRateImporter().import_bars(req))
    assert len(chunks) == 1
    frame = chunks[0].frame
    row = frame.set_index("ts").loc[pd.Timestamp("2024-07-01 13:30:00", tz="UTC")]
    assert row["close"] == 105.0
    assert row["adj_close"] == 52.0
    validate_bars(to_bar_schema(frame))


def test_intraday_import_30m(tmp_path):
    raw, adj = _firstrate_dirs(tmp_path)
    (raw / "AAPL_full_30min_UNADJUSTED.txt").write_text(
        "2024-07-01 09:30:00,1,1,1,1,1\n2024-07-01 10:00:00,1,1,1,1,1\n", encoding="utf-8")
    (adj / "AAPL_full_30min_adjsplitdiv.txt").write_text(
        "2024-07-01 09:30:00,1,1,1,1,1\n2024-07-01 10:00:00,1,1,1,1,1\n", encoding="utf-8")
    req = FirstRateImportRequest(raw_dir=raw, adjusted_dir=adj, timeframe="30m")
    chunks = list(FirstRateImporter().import_bars(req))
    frame = chunks[0].frame
    assert frame["ts"].tolist() == [
        pd.Timestamp("2024-07-01 13:30:00", tz="UTC"),
        pd.Timestamp("2024-07-01 14:00:00", tz="UTC"),
    ]
    validate_bars(to_bar_schema(frame))


def test_intraday_duplicate_message_shows_full_timestamp(tmp_path):
    raw, adj = _firstrate_dirs(tmp_path)
    _write_intraday_pair(
        raw, adj, "AAPL",
        ["2024-07-01 09:30:00,1,1,1,1,1\n", "2024-07-01 09:30:00,2,2,2,2,2\n"],  # dup intraday ts
        ["2024-07-01 09:30:00,1,1,1,1,1\n"],
    )
    with pytest.raises(ValueError, match="duplicate timestamps in raw file.*09:30:00"):
        list(FirstRateImporter().import_bars(
            FirstRateImportRequest(raw_dir=raw, adjusted_dir=adj, timeframe="1m")))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_firstrate_importer.py -q -k "unknown_timeframe or intraday_import or duplicate_message"`
Expected: FAIL — `import_bars` still raises "intraday import not yet supported" for non-`1d`, so intraday imports never reach the merge.

- [ ] **Step 3: Lift the guard and thread `timeframe` in `import_bars`**

In `algua/data/importers/firstrate.py`, add the timeframes import to the existing one from Task 2:

```python
from algua.data.timeframes import is_intraday, validate_timeframe
```

Replace the guard at the top of `FirstRateImporter.import_bars`:

```python
        if not isinstance(request, FirstRateImportRequest):
            raise ValueError("FirstRateImporter requires a FirstRateImportRequest")
        validate_timeframe(request.timeframe)
```

(Delete the `if request.timeframe != "1d": raise ValueError("intraday import not yet supported (1d only)")` lines.)

Change the final loop to pass the timeframe:

```python
        for symbol in symbols:
            yield self._merge_symbol(symbol, raw_map[symbol], adj_map[symbol], request.timeframe)
```

- [ ] **Step 4: Thread `timeframe` into `_merge_symbol` and switch messages to full timestamps**

Change the `_merge_symbol` signature and the two parse calls:

```python
    def _merge_symbol(
        self, symbol: str, raw_path: Path, adj_path: Path, timeframe: str
    ) -> ProviderBars:
        raw = parse_firstrate_file(raw_path, timeframe)
        adj = parse_firstrate_file(adj_path, timeframe)[["ts", "close"]].rename(
            columns={"close": "adj_close"}
        )
```

Replace the three diagnostic-message blocks (they currently render `.dt.date` / `.date()`, which collapses intraday timestamps) with full-timestamp renderings:

```python
        if raw["ts"].duplicated().any():
            dupes = sorted(
                str(ts) for ts in raw.loc[raw["ts"].duplicated(keep=False), "ts"].unique()
            )
            raise ValueError(f"{symbol}: duplicate timestamps in raw file: {dupes}")
        if adj["ts"].duplicated().any():
            dupes = sorted(
                str(ts) for ts in adj.loc[adj["ts"].duplicated(keep=False), "ts"].unique()
            )
            raise ValueError(f"{symbol}: duplicate timestamps in adjusted file: {dupes}")

        raw_keys = set(raw["ts"])
        adj_keys = set(adj["ts"])
        if raw_keys != adj_keys:
            unmatched = sorted(str(ts) for ts in raw_keys.symmetric_difference(adj_keys))
            raise ValueError(
                f"{symbol}: raw and adjusted key sets differ; refusing partial merge. "
                f"unmatched timestamps: {unmatched}"
            )
```

(The rest of `_merge_symbol` — the merge, price validation, column ordering, `ProviderBars` return — is unchanged.)

- [ ] **Step 5: Run the targeted tests to verify they pass**

Run: `uv run pytest tests/test_firstrate_importer.py -q -k "unknown_timeframe or intraday_import or duplicate_message"`
Expected: PASS

- [ ] **Step 6: Run the full importer test file**

Run: `uv run pytest tests/test_firstrate_importer.py -q`
Expected: PASS (all). The existing daily `test_key_disagreement_errors` still matches `"key sets differ"`.

- [ ] **Step 7: Commit**

```bash
git add algua/data/importers/firstrate.py tests/test_firstrate_importer.py
git commit -m "feat(151): lift FirstRate 1d guard; thread timeframe; full-ts diagnostics"
```

---

## Task 4: Enforce the vocabulary at the store + serve chokepoints

**Files:**
- Modify: `algua/data/store.py` (`ingest_bars`, `ingest_bars_streamed`)
- Modify: `algua/data/serve.py` (`StoreBackedProvider.get_bars`)
- Test: `tests/test_data_store.py`, `tests/test_data_serve.py`

- [ ] **Step 1: Write the failing tests**

First inspect how each test file builds a store/snapshot so the new tests match local fixtures:

Run: `uv run pytest tests/test_data_store.py tests/test_data_serve.py -q` (confirm they pass today), then `sed -n '1,40p' tests/test_data_serve.py` and `grep -n "ingest_bars_streamed\|ingest_bars\|DataStore(" tests/test_data_store.py | head`.

Add to `tests/test_data_store.py` (use the same store/chunk fixture the neighbouring streamed-ingest tests use — build a one-symbol UTC bars chunk frame `df` with columns `ts, symbol, open, high, low, close, adj_close, volume`):

```python
def test_ingest_bars_streamed_rejects_unknown_timeframe(tmp_path):
    store = DataStore(tmp_path / "data")
    df = _one_symbol_bars()  # local helper used by the other streamed tests
    with pytest.raises(ValueError, match="unknown timeframe"):
        store.ingest_bars_streamed(
            provider="firstrate", symbols=["AAPL"], as_of="2024-07-02T00:00:00+00:00",
            source="t", chunks=iter([df]), timeframe="15m",
        )


def test_ingest_bars_rejects_unknown_timeframe(tmp_path):
    store = DataStore(tmp_path / "data")
    df = _one_symbol_bars()
    with pytest.raises(ValueError, match="unknown timeframe"):
        store.ingest_bars(
            provider="firstrate", symbols=["AAPL"], start="2024-07-01", end="2024-07-01",
            as_of="2024-07-02T00:00:00+00:00", source="t", frame=df, timeframe="15m",
        )
```

> If `tests/test_data_store.py` has no reusable one-symbol-bars helper, add a small module-level `_one_symbol_bars()` that returns a valid bars frame (mirror the frame the existing streamed-ingest test constructs). Match the existing test's column names and dtypes exactly.

Add to `tests/test_data_serve.py` (it already builds a served snapshot via `ingest_bars`; reuse that fixture):

```python
def test_get_bars_rejects_unknown_timeframe(tmp_path):
    store, snapshot_id = _served_daily_snapshot(tmp_path)  # existing fixture/helper pattern
    provider = StoreBackedProvider(store, snapshot_id)
    from datetime import datetime
    with pytest.raises(ValueError, match="unknown timeframe"):
        provider.get_bars(["AAPL"], datetime(2024, 7, 1), datetime(2024, 7, 2), "15m")
```

> If `test_data_serve.py` builds the snapshot inline rather than via a helper, inline the same construction in this test. The assertion is only that an unknown *requested* token raises before the snapshot-mismatch check.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_data_store.py tests/test_data_serve.py -q -k "unknown_timeframe"`
Expected: FAIL — unknown tokens currently slip through (`15m` would be stored / produce a snapshot-mismatch message, not an "unknown timeframe" error).

- [ ] **Step 3: Add the guard in `algua/data/store.py`**

Add the import near the other `from algua.data.*` imports at the top of `store.py`:

```python
from algua.data.timeframes import validate_timeframe
```

In `ingest_bars` (the non-streamed method), add as the FIRST statement of the method body (before `metadata = _metadata(...)`):

```python
        validate_timeframe(timeframe)
```

In `ingest_bars_streamed`, add as the FIRST statement of the method body (before `staging_dir = ...`):

```python
        validate_timeframe(timeframe)
```

- [ ] **Step 4: Add the guard in `algua/data/serve.py`**

Add the import at the top of `serve.py`:

```python
from algua.data.timeframes import validate_timeframe
```

In `StoreBackedProvider.get_bars`, add as the FIRST statement (before `rec = self.store.get_snapshot(...)`):

```python
        validate_timeframe(timeframe)
```

- [ ] **Step 5: Run the targeted tests to verify they pass**

Run: `uv run pytest tests/test_data_store.py tests/test_data_serve.py -q -k "unknown_timeframe"`
Expected: PASS

- [ ] **Step 6: Run both full test files (regression — no daily path broke)**

Run: `uv run pytest tests/test_data_store.py tests/test_data_serve.py -q`
Expected: PASS (all). Existing callers pass `"1d"`, which is valid.

- [ ] **Step 7: Commit**

```bash
git add algua/data/store.py algua/data/serve.py tests/test_data_store.py tests/test_data_serve.py
git commit -m "feat(151): enforce closed timeframe vocab at ingest + serve chokepoints"
```

---

## Task 5: End-to-end intraday import → read-back round-trip

**Files:**
- Test: `tests/test_cli_import_bars.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_cli_import_bars.py` (the `_tmp_data_dir` autouse fixture sets `ALGUA_DATA_DIR`; read back through the same store the CLI wrote to):

```python
def test_import_bars_intraday_roundtrip(tmp_path, monkeypatch):
    raw = tmp_path / "raw"; adj = tmp_path / "adj"; raw.mkdir(); adj.mkdir()
    (raw / "AAPL_full_1min_UNADJUSTED.txt").write_text(
        "2024-07-01 09:30:00,100,110,95,105,10\n2024-07-01 09:31:00,105,120,100,115,20\n",
        encoding="utf-8")
    (adj / "AAPL_full_1min_adjsplitdiv.txt").write_text(
        "2024-07-01 09:30:00,50,55,47,52,10\n2024-07-01 09:31:00,52,60,50,57,20\n",
        encoding="utf-8")
    result = runner.invoke(app, [
        "data", "import-bars", "--vendor", "firstrate",
        "--raw-dir", str(raw), "--adjusted-dir", str(adj),
        "--timeframe", "1m", "--as-of", "2024-07-02T00:00:00+00:00",
        "--adjustment", "split_div",
    ])
    assert result.exit_code == 0, result.stdout
    snap = json.loads(result.stdout)["snapshot"]
    assert snap["timeframe"] == "1m"
    assert snap["row_count"] == 2

    # Read it back through the serving seam: time-of-day preserved, half-open [start, end).
    import os
    from datetime import datetime
    from algua.data.store import DataStore
    from algua.data.serve import StoreBackedProvider

    store = DataStore(os.environ["ALGUA_DATA_DIR"])
    provider = StoreBackedProvider(store, snap["snapshot_id"])
    bars = provider.get_bars(
        ["AAPL"], datetime(2024, 7, 1), datetime(2024, 7, 1, 13, 31), "1m"
    )
    # 09:30 ET -> 13:30 UTC is included; 09:31 ET -> 13:31 UTC is excluded by the half-open end.
    assert [str(ts) for ts in bars.index] == ["2024-07-01 13:30:00+00:00"]
    assert bars["close"].iloc[0] == 105.0
    assert bars["adj_close"].iloc[0] == 52.0
```

> Before running, confirm the snapshot JSON key for the id is `snapshot_id` (check `snap` keys in `test_import_bars_happy_path` output, or run `grep -n "snapshot_id\|to_dict" algua/data/models.py`). If the key differs, use the actual key.

- [ ] **Step 2: Run the test to verify it fails or passes**

Run: `uv run pytest tests/test_cli_import_bars.py::test_import_bars_intraday_roundtrip -q`
Expected: PASS if Tasks 2–4 are complete (this task is an integration test over already-implemented behavior). If it FAILS, read the failure: a real wiring gap to fix, not a test to weaken.

- [ ] **Step 3: Commit**

```bash
git add tests/test_cli_import_bars.py
git commit -m "test(151): end-to-end intraday import -> read-back round-trip"
```

---

## Task 6: Reconcile the bar-schema contract doc

**Files:**
- Modify: `docs/contracts/bar-schema.md` (the `### timeframe vocabulary` section, ~lines 70-73)

- [ ] **Step 1: Update the timeframe-vocabulary section**

Replace the existing block:

```markdown
### `timeframe` vocabulary
- `"1d"` — daily session bars. **Required first; the research lane targets `"1d"` initially.**
- `"1h"`, `"15m"`, `"1m"` — intraday, UTC-aligned bar boundaries (reserved; build later).
- Any other value → `ValueError`.
```

with:

```markdown
### `timeframe` vocabulary
- `"1d"` — daily session bars. The research lane targets `"1d"` initially.
- `"1m"`, `"5m"`, `"30m"`, `"1h"` — intraday bars. The timestamp preserves the **exchange/vendor bar
  label converted to a UTC instant**; it is NOT required to fall on a UTC-clock-aligned boundary
  (e.g. an 09:30 US/Eastern open is `13:30Z` in EDT, `14:30Z` in EST). Look-ahead safety comes from
  the half-open serving window plus the engine `t→t+1` shift, not from clock alignment.
- The vocabulary is closed: `algua/data/timeframes.py::validate_timeframe` is the single source of
  truth, enforced at both bars-ingest chokepoints (`DataStore.ingest_bars` /
  `ingest_bars_streamed`) and at the serving seam (`StoreBackedProvider.get_bars`). Any other value
  → `ValueError`.
- The declared timeframe is operator-asserted metadata, recorded as-is (like the `adjustment`
  flavor): the system does not infer bar granularity from the data, so importing `1m` bars under
  `--timeframe 5m` is operator error, not a detected condition. What IS guarded fail-closed is the
  daily↔intraday confusion that would corrupt timestamps (non-midnight `1d`, or a local-midnight
  intraday bar).
```

- [ ] **Step 2: Verify no other doc line contradicts (the "Intraday timeframes carry the bar's time-of-day" note earlier in the doc is already correct)**

Run: `grep -n "UTC-aligned\|15m\|intraday" docs/contracts/bar-schema.md`
Expected: no remaining `UTC-aligned` or `15m` references; the surviving "intraday" mentions are the (correct) index/time-of-day note and the new vocabulary block.

- [ ] **Step 3: Commit**

```bash
git add docs/contracts/bar-schema.md
git commit -m "docs(151): reconcile bar-schema timeframe vocabulary + UTC-instant wording"
```

---

## Task 7: Full quality gate

- [ ] **Step 1: Run the complete gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all green. In particular `lint-imports` confirms `algua/data/timeframes.py` and the new imports introduce no boundary violation (timeframes imports nothing from algua; `firstrate`/`store`/`serve` importing `algua.data.timeframes` is intra-package).

- [ ] **Step 2: If anything fails, fix it** (REQUIRED SUB-SKILL: `superpowers:systematic-debugging` for any non-obvious failure) and re-run the gate until green. Commit the fix.

---

## Self-Review notes (spec coverage)

- Timeframe-aware parse + ET→UTC localization → Task 2. ✓
- DST handled, fail-closed → Task 2 (`ambiguous="raise"`, `nonexistent="raise"`; wrapped). ✓
- Two midnight guards (daily UTC-midnight; intraday reject local-midnight) → Task 2. ✓
- tz-aware intraday actionable error → Task 2. ✓
- Lift the `1d` guard + thread timeframe → Task 3. ✓
- Full-timestamp diagnostic messages → Task 3. ✓
- Closed vocab module → Task 1; enforced at `ingest_bars` + `ingest_bars_streamed` + `get_bars` → Task 4. ✓
- Tests: ET↔UTC summer/winter, DST-adjacent valid, DST fail-closed, 1m/5m/30m/1h, ordering/uniqueness + dup message, unknown token, both midnight guards, end-to-end round-trip → Tasks 2/3/5. ✓
- Doc reconciliation (vocab + UTC-aligned wording) → Task 6. ✓
- Declined (spacing/filename granularity check) and deferred (`observed_start/end` time-of-day fidelity) → recorded in spec, intentionally NOT in this plan. ✓
