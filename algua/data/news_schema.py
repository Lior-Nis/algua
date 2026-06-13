from __future__ import annotations

import hashlib
import struct

import numpy as np
import pandas as pd

from algua.contracts.types import NEWS_AS_OF_KEY, NEWS_COLUMNS, NEWS_KNOWABLE_AT, NEWS_RETRACTED

COLUMNS = list(NEWS_COLUMNS)
STRING_COLUMNS = ["source", "article_id", "symbol", "headline"]  # non-null strings
NULLABLE_STRING_COLUMNS = ["url", "body"]
BOOL_COLUMNS = [NEWS_RETRACTED]  # non-nullable bool
TS_COLUMNS = ["published_at", NEWS_KNOWABLE_AT]
UNIQUE_KEY = [*NEWS_AS_OF_KEY, NEWS_KNOWABLE_AT]  # (source, article_id, symbol, knowable_at)
_SORT = ["symbol", "source", "article_id", NEWS_KNOWABLE_AT]


def _is_na(v: object) -> bool:
    """True for a scalar missing value (None, NaN, pd.NA, NaT, datetime64 NaT). A container is
    never NA itself — its items are checked individually (so `pd.isna` on an array can't raise)."""
    if isinstance(v, (list, tuple, set, dict, np.ndarray, pd.Series)):
        return False
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


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
    for col in BOOL_COLUMNS:
        if df[col].dtype != np.dtype("bool"):
            raise ValueError(
                f"news {col!r} must be non-nullable bool dtype (got {df[col].dtype})"
            )
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
        "retracted": pd.Series([], dtype="bool"),
    }
    return validate_news(pd.DataFrame(data)[COLUMNS])


NEWS_COLUMNS = tuple(COLUMNS)  # re-export for `from algua.data.news_schema import NEWS_COLUMNS`


_RAW_REQUIRED = ["source", "article_id", "symbols", "published_at", NEWS_KNOWABLE_AT, "headline"]


def explode_news_symbols(frame: pd.DataFrame) -> pd.DataFrame:
    """Ingest-only pre-step. Each input row is a FULL article REVISION — one
    (source, article_id, knowable_at) carrying that revision's COMPLETE symbol set as-of that
    knowable_at (full-restatement semantics: a revision republishes the authoritative set).
    Multiple revisions of one article = multiple input rows; (source, article_id, knowable_at)
    must be unique under the canonical identity (source strip+lower, article_id strip,
    knowable_at as UTC).

    Symbols (a list, or a comma-delimited string) are stripped/upper-cased, blanks dropped,
    de-duped within a revision. Emits one row per (revision, symbol) with retracted=False, PLUS a
    synthesized TOMBSTONE row (retracted=True) for each symbol present in the prior revision but
    absent from this one — marking that (source, article_id, symbol) mention retracted at this
    revision's knowable_at, so the as-of mask (latest revision per key) drops it. A tombstone
    carries the dropping revision's content (published_at/headline/url/body), so the per-revision
    invariants hold and the dropped symbol never collides with a live row in that revision.

    The first revision of an article must tag >= 1 symbol; a later empty revision is a valid full
    retraction of a non-empty prior revision (it tombstones every prior symbol). Optional
    `url`/`body` default to NA. Output carries NEWS_COLUMNS."""
    missing = [c for c in _RAW_REQUIRED if c not in frame.columns]
    if missing:
        raise ValueError(f"news input missing columns: {missing}")
    out = frame.copy()
    for opt in NULLABLE_STRING_COLUMNS:
        if opt not in out.columns:
            out[opt] = pd.NA

    def _parse(v: object) -> list[str]:
        if _is_na(v):
            return []  # null symbols -> zero symbols -> rejected below (never "NONE")
        if isinstance(v, (list, tuple, set, np.ndarray, pd.Series)):
            items = list(v)
        else:
            items = str(v).split(",")
        seen: list[str] = []
        for s in items:
            if _is_na(s):
                continue
            s = str(s).strip().upper()
            if s and s not in seen:
                seen.append(s)
        return seen

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


def to_news_schema(frame: pd.DataFrame) -> pd.DataFrame:
    """Idempotent canonical normalizer for an already-exploded per-symbol frame (run by both
    ingest — after explode — and read). Canonicalizes source (strip+lower) and symbol
    (strip+upper), coerces dtypes, normalizes timestamps to UTC (naive rejected, knowable_at
    required), canonicalizes nullable url/body to pd.NA (distinct from ""), de-dups, sorts,
    validates."""
    missing = [c for c in COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"frame missing news columns: {missing}")
    out = frame[COLUMNS].copy()
    # Reject nulls in required columns BEFORE str-coercion — otherwise None/NaN/<NA> would silently
    # become the literal strings "None"/"nan"/"<NA>" and pass the non-null validator (GATE-2).
    for col in STRING_COLUMNS:
        if out[col].isna().any():
            raise ValueError(f"news {col!r} must be non-null (got null in input)")
    out["source"] = out["source"].astype(str).str.strip().str.lower()
    out["article_id"] = out["article_id"].astype(str).str.strip()
    out["symbol"] = out["symbol"].astype(str).str.strip().str.upper()
    out["headline"] = out["headline"].astype(str)
    # Identity columns must be non-empty after stripping (a blank identity is meaningless).
    for col in ["source", "article_id", "symbol"]:
        if (out[col].str.len() == 0).any():
            raise ValueError(f"news {col!r} must be non-empty")
    for col in NULLABLE_STRING_COLUMNS:
        out[col] = out[col].apply(lambda v: pd.NA if _is_na(v) else str(v))
    for col in TS_COLUMNS:
        # Normalize any tz-aware input (a single offset OR a mix of offsets) to UTC, and reject
        # naive cleanly. `utc=True` handles mixed offsets robustly (a bare pd.to_datetime can
        # return object dtype or raise); naive is detected element-wise so the mixed-offset path
        # can't hide a naive value (GATE-2).
        col_vals = out[col]
        aware = col_vals.map(lambda x: _is_na(x) or pd.Timestamp(x).tzinfo is not None)
        if not bool(aware.all()):
            raise ValueError(f"news {col!r} must be tz-aware (UTC); naive timestamps are rejected")
        out[col] = pd.to_datetime(col_vals, errors="raise", utc=True)
    if out[NEWS_RETRACTED].isna().any():
        raise ValueError("news 'retracted' must be non-null (got null in input)")
    # Reject non-bool BEFORE .astype("bool") — otherwise "false"/any non-empty string -> True,
    # ints -> bool, silently flipping live<->tombstone past validate_news (whose dtype check runs
    # after the cast). Mirrors the STRING_COLUMNS "None"/"nan" coercion defense (GATE-2).
    if not all(isinstance(v, (bool, np.bool_)) for v in out[NEWS_RETRACTED]):
        raise ValueError("news 'retracted' must be bool values (no coercion of non-bool input)")
    out[NEWS_RETRACTED] = out[NEWS_RETRACTED].astype("bool")
    out = out.drop_duplicates().sort_values(_SORT).reset_index(drop=True)
    return validate_news(out)


def logical_news_hash(df: pd.DataFrame) -> str:
    """Deterministic content hash over the logical rows — the snapshot identity (mirrors
    logical_fundamentals_hash). Non-null strings length-prefixed UTF-8; nullable url/body carry a
    null-flag byte (so null, "", and "None" hash distinctly); timestamps as int64 ns UTC.

    Precondition: `df` is already validated (run after `validate_news`/`to_news_schema`); in
    particular STRING_COLUMNS are non-null, so their `.astype(str)` here cannot mask a null."""
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
    for col in BOOL_COLUMNS:
        flags = np.array([bool(v) for v in ordered[col]], dtype="u1")
        digest.update(flags.tobytes())
    return digest.hexdigest()
