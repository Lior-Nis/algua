# Stage 3 — `algua/data/store.py` Carve Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Carve the 970-line `algua/data/store.py` monolith into `algua/data/store/` — one module per dataset (bars, streamed-bars, universe, delistings, fundamentals, news) plus a shared `identity.py`, with `DataStore` reduced to a thin facade over its three existing collaborators (`SnapshotManifest`, `SnapshotStagingLease`, `SnapshotVerifier`). Zero behavior change; `from algua.data.store import DataStore` and every existing call site (`store.ingest_bars(...)`, `store.read_universe(...)`, etc.) keep working unchanged.

**Architecture:** Six dataset modules each define a mixin class holding that dataset's methods, referencing `self.data_dir`/`self.manifest`/`self._staging` exactly as today (declared as class-level type annotations for mypy, never assigned in the mixin — only `DataStore.__init__` assigns them). Two genuinely cross-dataset helpers (`_commit_bars_dir` shared by bars/streamed-bars, `_ingest_parquet` shared by universe/delistings) are inherited via a real base-class relationship (`BarsStreamedStoreMixin(BarsStoreMixin)`, `UniverseStoreMixin(IdentityMixin)`/`DelistingsStoreMixin(IdentityMixin)`) rather than duplicated. Everything dataset-agnostic (metadata building, snapshot-id hashing, symbol normalization, validation) becomes free functions in `identity.py`, promoted from file-private (`_metadata`) to package-internal-public (`build_metadata`) — the same promotion pattern Stage 2 used moving `_fsync_file` → `fsync_file` into `primitives/atomic_io.py`. `store/__init__.py` composes the mixins into the final `DataStore` class and keeps the handful of truly dataset-agnostic methods (`ingest_file`, `clear_staging`, `list_snapshots`, `get_snapshot`, `verify_snapshot`, `verify_snapshots`, `summary`) directly on itself.

**Tech Stack:** Python 3.12, uv, pytest, ruff, mypy, import-linter.

**Spec:** `docs/superpowers/specs/2026-08-18-system-simplification-design.md` §6 item 1.

## Global Constraints

- Quality gate on EVERY task before commit: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`. All four must pass.
- **Zero behavior change.** This is a pure structural carve of already-working, already-tested code — no new tests are written; the existing test suite (which constructs `DataStore` directly in ~35 files, no shared fixture) is the regression net. Every docstring, every safety-critical comment (power-loss durability rationale, fail-closed validation, immutability rules), and every line of logic moves **verbatim** — only its file location and (for the identity helpers only) its name changes.
- No compat shims, no dead re-exports beyond what's needed for `from algua.data.store import DataStore, SnapshotNotFound, normalize_symbols` to keep working (these three names have real external/test importers — see Task 2).
- `git add` is always scoped to the named files — never `git add -A`.
- Commits end with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Known pre-existing worktree hazard: some test writes a demo strategy file into the real `algua/strategies/momentum/` directory as a side effect. If `git status` shows an untracked file there after running tests, delete it before staging — don't commit it.
- **Process rule from Stage 0-2's execution — read this before dispatching or resuming any subagent**: if a background command (e.g. the full test suite, ~6-7 minutes on this repo) is started, the implementer/reviewer MUST actively poll/re-check its own output with real tool calls in a loop. There is no notification that wakes a dispatched subagent when a background command finishes — ending a turn to "wait" for one stalls the task indefinitely until manually resumed. This happened ~8 times across Stage 2's 13 tasks and cost real time every time.
- No import-linter contract needs to change for this carve — every existing `algua.data`-referencing contract matches by dotted-path prefix, so `algua.data.store`, `algua.data.store.bars`, etc. all still match `"algua.data"` for any contract that forbids it (verified against `pyproject.toml`).

---

### Task 1: Carve `algua/data/store.py` into `algua/data/store/`

**Files:**
- Create: `algua/data/store/__init__.py`
- Create: `algua/data/store/identity.py`
- Create: `algua/data/store/bars.py`
- Create: `algua/data/store/bars_streamed.py`
- Create: `algua/data/store/universe.py`
- Create: `algua/data/store/delistings.py`
- Create: `algua/data/store/fundamentals.py`
- Create: `algua/data/store/news.py`
- Delete: `algua/data/store.py`

**Interfaces:**
- Produces: `algua.data.store.DataStore` (same public API as today — every method listed below keeps its exact name, parameters, and return type), `algua.data.store.SnapshotNotFound`, `algua.data.store.normalize_symbols` — all importable exactly as `from algua.data.store import DataStore` / `SnapshotNotFound` / `normalize_symbols` (this is what external callers and tests already do; it must keep working unchanged since `algua/data/store/__init__.py` replaces `algua/data/store.py` at the same import path).

This is a single atomic task because Python cannot have both `algua/data/store.py` and `algua/data/store/__init__.py` present usefully at once (the package would shadow the module, leaving `store.py`'s content silently dead) — every new file must be created and `store.py` deleted in the same commit for the test suite to pass at any point you'd actually run it. Read the CURRENT `algua/data/store.py` in full before starting (970 lines) — the step-by-step file contents below tell you exactly which lines go where, but read the source directly rather than trusting line numbers alone (line numbers below were correct as of this plan's writing but re-verify against the file in front of you).

#### Design rules that apply to every file below

1. **Move code verbatim.** Every docstring and safety-critical comment moves with its function unchanged. Do not paraphrase, shorten, or "clean up" prose while moving it.
2. **Mixin attribute contract.** Every mixin class declares the collaborator attributes its OWN methods use as class-level type annotations with no assignment (e.g. `data_dir: Path`) — never define `__init__` on a mixin; only `DataStore.__init__` (in `store/__init__.py`) ever assigns these. This is the standard mypy-clean pattern for attribute-sharing mixins.
3. **Cross-mixin method calls that reach the facade** (`self.get_snapshot(...)`, called from `read_bars`/`read_fundamentals`/`read_news`): the calling mixin declares a **stub method** with the same signature and an `...` body, annotated as provided by the facade. This is purely for mypy — at runtime `self.get_snapshot` always resolves to the real implementation on `DataStore` itself (Python resolves attributes defined directly on the instantiated class before consulting any base class, regardless of base-class order), so the stub is never actually called.
4. **Cross-mixin method calls that reach a SIBLING mixin's real implementation** (`_commit_bars_dir` shared by bars/streamed-bars; `_ingest_parquet` shared by universe/delistings): resolved via REAL inheritance, not a stub — `BarsStreamedStoreMixin(BarsStoreMixin)`, `UniverseStoreMixin(IdentityMixin)`, `DelistingsStoreMixin(IdentityMixin)`. This gets both runtime correctness and full mypy checking for free, with no MRO ordering risk (see step 8's explanation of why `DataStore`'s own base list must NOT re-list `BarsStoreMixin`/`IdentityMixin`).
5. **Five module-level helper functions get promoted from file-private to package-internal-public** (matching Stage 2's `_fsync_file` → `fsync_file` precedent when those were promoted into `primitives/atomic_io.py`): `_metadata` → `build_metadata`, `_snapshot_id` → `compute_snapshot_id`, `_path_part` → `path_part`, `_validate_non_empty` → `validate_non_empty`, `_validate_date_bounds` → `validate_date_bounds`, `_validate_datetime` → `validate_datetime`. `normalize_symbols` keeps its name (already public). `_ingest_parquet` and `_commit_bars_dir` stay private (leading underscore) since they remain bound methods, not promoted top-level functions.

- [ ] **Step 1: Create `algua/data/store/identity.py`**

Move these from the current `algua/data/store.py` (module-level functions near the bottom of the file, and `_ingest_parquet` currently a bound method between `read_delistings_with_snapshot` and `clear_staging`):

- `normalize_symbols(symbols: list[str]) -> list[str]` — keep its exact name and body, keep its docstring verbatim.
- `_metadata(...) -> SnapshotMetadata` → rename to `build_metadata(...) -> SnapshotMetadata`, same parameters/body, update its internal call from `normalize_symbols(symbols)` (unchanged — same name) and its three `_validate_*` calls to the new public names (`validate_non_empty`, `validate_date_bounds`, `validate_datetime`).
- `_validate_non_empty(name: str, value: str) -> None` → rename to `validate_non_empty`.
- `_validate_date_bounds(start: str, end: str) -> None` → rename to `validate_date_bounds`.
- `_validate_datetime(name: str, value: str) -> None` → rename to `validate_datetime`.
- `_snapshot_id(metadata: SnapshotMetadata, content_hash: str) -> str` → rename to `compute_snapshot_id`, body unchanged.
- `_path_part(value: str) -> str` → rename to `path_part`, body unchanged.
- `_ingest_parquet(self, *, metadata, frame, filename, conflict_check=None) -> SnapshotRecord` — moves as a METHOD (keep `self`, keep the name with its leading underscore) onto a new class:

```python
class IdentityMixin:
    """Shared identity/parquet-publish plumbing for the dataset mixins that need it
    (universe, delistings). Declares the collaborator attributes it assumes the
    composing DataStore provides — never assigned here, only annotated."""

    data_dir: Path
    manifest: SnapshotManifest

    def _ingest_parquet(
        self,
        *,
        metadata: SnapshotMetadata,
        frame: pd.DataFrame,
        filename: str,
        conflict_check: Callable[[list[SnapshotRecord], SnapshotRecord], None] | None = None,
    ) -> SnapshotRecord:
        # move the exact current body verbatim — content_hash, snapshot_id via
        # compute_snapshot_id(...), self.manifest.find, write_bytes_durable to
        # self.data_dir / relative_path, self.manifest.append_if_absent. Keep the
        # full existing docstring verbatim.
        ...
```

Imports this file needs (derive from what actually gets used — this list is what the moved code requires): `hashlib`, `json`, `from collections.abc import Callable`, `from datetime import UTC, date, datetime`, `from pathlib import Path`, `from typing import Any`, `import pandas as pd`, `from algua.data.files import frame_to_parquet_bytes, sha256_bytes`, `from algua.data.manifest import SnapshotManifest`, `from algua.data.models import Dataset, Kind, SnapshotMetadata, SnapshotRecord`, `from algua.primitives.atomic_io import write_bytes_durable`.

- [ ] **Step 2: Create `algua/data/store/bars.py`**

Move `ingest_bars`, `_commit_bars_dir`, `read_bars` (their current bodies and docstrings verbatim) onto:

```python
class BarsStoreMixin:
    data_dir: Path
    manifest: SnapshotManifest
    _staging: SnapshotStagingLease

    def get_snapshot(self, snapshot_id: str) -> SnapshotRecord: ...  # provided by the DataStore facade (store/__init__.py); stub for mypy only

    def ingest_bars(self, *, ...) -> SnapshotRecord:
        ...  # moved verbatim from the current ingest_bars body; uses build_metadata,
        # compute_snapshot_id (imported from .identity), self.manifest, self._staging,
        # self._commit_bars_dir

    def _commit_bars_dir(self, rec: SnapshotRecord, staging_dir: Path, *, expected_symbols: set[str]) -> SnapshotRecord:
        ...  # moved verbatim

    def read_bars(self, snapshot_id: str, *, symbols=None, start=None, end=None) -> pd.DataFrame:
        ...  # moved verbatim; uses self.get_snapshot(...) — resolves to the real
        # facade method at runtime per design rule 3
```

Imports: `errno`, `os`, `from datetime import UTC, datetime`, `from pathlib import Path`, `import pandas as pd`, `from algua.data.files import logical_bars_hash, read_partitioned_bars, validate_partitioned_bars_dir, write_partitioned_bars`, `from algua.data.manifest import SnapshotManifest`, `from algua.data.models import Dataset, Kind, SnapshotRecord`, `from algua.data.schema import empty_bars, to_bar_schema`, `from algua.data.staging import SnapshotStagingLease`, `from algua.data.timeframes import validate_timeframe`, `from algua.primitives.atomic_io import fsync_parents, fsync_tree`, `from algua.data.store.identity import build_metadata, compute_snapshot_id`.

- [ ] **Step 3: Create `algua/data/store/bars_streamed.py`**

Move `ingest_bars_streamed` (body and docstring verbatim) onto:

```python
class BarsStreamedStoreMixin(BarsStoreMixin):
    def ingest_bars_streamed(self, *, ...) -> SnapshotRecord:
        ...  # moved verbatim; uses build_metadata, compute_snapshot_id (from .identity),
        # self._staging, self._commit_bars_dir (real impl inherited from BarsStoreMixin)
```

Imports: `from collections.abc import Iterable`, `from datetime import UTC, datetime`, `from pathlib import Path`, `import pandas as pd`, `from algua.data.files import BARS_STREAMED_HASH_ALGO, compose_bars_symbol_hash, logical_bars_hash, write_partitioned_bars`, `from algua.data.models import Dataset, Kind, SnapshotRecord`, `from algua.data.schema import to_bar_schema`, `from algua.data.timeframes import validate_timeframe`, `from algua.data.store.bars import BarsStoreMixin`, `from algua.data.store.identity import build_metadata, compute_snapshot_id`.

- [ ] **Step 4: Create `algua/data/store/universe.py`**

Move `ingest_universe` (including its nested `conflict_check` closure) and `read_universe` (bodies/docstrings verbatim) onto:

```python
class UniverseStoreMixin(IdentityMixin):
    def ingest_universe(self, *, ...) -> SnapshotRecord:
        ...  # moved verbatim; uses normalize_symbols, build_metadata (from .identity),
        # self._ingest_parquet (real impl inherited from IdentityMixin)

    def read_universe(self, universe: str) -> list[UniverseSnapshot]:
        ...  # moved verbatim; uses self.manifest, self.data_dir, normalize_symbols
```

Imports: `from datetime import date`, `import pandas as pd`, `from algua.data.models import Dataset, Kind, SnapshotRecord, UniverseSnapshot`, `from algua.data.store.identity import IdentityMixin, build_metadata, normalize_symbols`. (No `Path` import needed here — `read_universe`/`ingest_universe` only ever do `self.data_dir / ...` on an already-`Path` attribute; the one `Path("snapshots") / ...` construction for universe lives inside `_ingest_parquet` in `identity.py`, not here. `SnapshotRecord` is needed for `ingest_universe`'s return-type annotation.)

- [ ] **Step 5: Create `algua/data/store/delistings.py`**

Move `ingest_delistings`, `_latest_delistings_record`, `latest_delistings_snapshot_id`, `_parse_delistings`, `read_delistings`, `read_delistings_with_snapshot` (bodies/docstrings verbatim, including the lazy `from algua.backtest.delisting import DelistingRecord` inside `_parse_delistings` with its exact `# lazy: keep algua.data off algua.backtest` comment) onto:

```python
if TYPE_CHECKING:
    from algua.backtest.delisting import DelistingRecord


class DelistingsStoreMixin(IdentityMixin):
    def ingest_delistings(self, *, ...) -> SnapshotRecord:
        ...  # moved verbatim; uses normalize_symbols, build_metadata, self._ingest_parquet

    def _latest_delistings_record(self, as_of: str | None) -> SnapshotRecord | None:
        ...  # moved verbatim; uses self.manifest

    def latest_delistings_snapshot_id(self, as_of: str | None = None) -> str | None:
        ...  # moved verbatim

    def _parse_delistings(self, rec: SnapshotRecord) -> dict[str, list[DelistingRecord]]:
        ...  # moved verbatim, including the lazy import inside the method body

    def read_delistings(self, as_of: str | None = None) -> dict[str, list[DelistingRecord]]:
        ...  # moved verbatim

    def read_delistings_with_snapshot(self, as_of: str | None = None) -> tuple[dict[str, list[DelistingRecord]], str | None]:
        ...  # moved verbatim
```

Imports: `math`, `from datetime import date`, `from typing import TYPE_CHECKING`, `import pandas as pd`, `from algua.data.models import Dataset, Kind, SnapshotRecord`, `from algua.data.store.identity import IdentityMixin, build_metadata, normalize_symbols`. (`SnapshotRecord` is needed for `ingest_delistings`'s and `_latest_delistings_record`'s return-type annotations.)

- [ ] **Step 6: Create `algua/data/store/fundamentals.py`**

Move `ingest_fundamentals`, `read_fundamentals` (bodies/docstrings verbatim) onto:

```python
class FundamentalsStoreMixin:
    data_dir: Path
    manifest: SnapshotManifest

    def get_snapshot(self, snapshot_id: str) -> SnapshotRecord: ...  # provided by the DataStore facade; stub for mypy only

    def ingest_fundamentals(self, *, ...) -> SnapshotRecord:
        ...  # moved verbatim; uses build_metadata, compute_snapshot_id (from .identity)

    def read_fundamentals(self, snapshot_id: str, *, symbols=None) -> pd.DataFrame:
        ...  # moved verbatim; uses self.get_snapshot, normalize_symbols
```

Imports: `from datetime import UTC, datetime`, `from pathlib import Path`, `import pandas as pd`, `from algua.data.fundamentals_schema import empty_fundamentals, logical_fundamentals_hash, to_fundamentals_schema`, `from algua.data.manifest import SnapshotManifest`, `from algua.data.models import Dataset, Kind, SnapshotRecord`, `from algua.primitives.atomic_io import write_bytes_durable`, `from algua.data.store.identity import build_metadata, compute_snapshot_id, normalize_symbols`.

- [ ] **Step 7: Create `algua/data/store/news.py`**

Move `ingest_news`, `read_news` (bodies/docstrings verbatim) onto:

```python
class NewsStoreMixin:
    data_dir: Path
    manifest: SnapshotManifest

    def get_snapshot(self, snapshot_id: str) -> SnapshotRecord: ...  # provided by the DataStore facade; stub for mypy only

    def ingest_news(self, *, ...) -> SnapshotRecord:
        ...  # moved verbatim; uses build_metadata, compute_snapshot_id (from .identity)

    def read_news(self, snapshot_id: str, *, symbols=None) -> pd.DataFrame:
        ...  # moved verbatim; uses self.get_snapshot, normalize_symbols
```

Imports: `from datetime import UTC, datetime`, `from pathlib import Path`, `import pandas as pd`, `from algua.data.manifest import SnapshotManifest`, `from algua.data.models import Dataset, Kind, SnapshotRecord`, `from algua.data.news_schema import empty_news, explode_news_symbols, logical_news_hash, to_news_schema`, `from algua.primitives.atomic_io import write_bytes_durable`, `from algua.data.store.identity import build_metadata, compute_snapshot_id, normalize_symbols`.

- [ ] **Step 8: Create `algua/data/store/__init__.py`**

This is the facade. Move `SnapshotNotFound`, `DataStore.__init__`, `ingest_file`, `clear_staging`, `list_snapshots`, `get_snapshot`, `verify_snapshot`, `verify_snapshots`, `summary` (bodies/docstrings verbatim) here, and compose the six mixins:

```python
"""Filesystem-backed point-in-time data manifest, carved by dataset (spec §6 item 1).

Each dataset's ingest/read logic lives in its own module (bars.py, bars_streamed.py,
universe.py, delistings.py, fundamentals.py, news.py); DataStore composes them via
mixins and keeps only the truly dataset-agnostic methods (ingest_file, staging/
manifest/verifier delegation, summary) directly on itself.
"""
from __future__ import annotations

import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from algua.data.manifest import SnapshotManifest
from algua.data.models import Dataset, Kind, SnapshotRecord
from algua.data.staging import SnapshotStagingLease
from algua.data.store.bars_streamed import BarsStreamedStoreMixin
from algua.data.store.delistings import DelistingsStoreMixin
from algua.data.store.fundamentals import FundamentalsStoreMixin
from algua.data.store.identity import (
    build_metadata,
    compute_snapshot_id,
    normalize_symbols,
    path_part,
)
from algua.data.store.news import NewsStoreMixin
from algua.data.store.universe import UniverseStoreMixin
from algua.data.verify import SnapshotVerifier
from algua.primitives.atomic_io import fsync_file, fsync_parents


class SnapshotNotFound(LookupError):
    pass


class DataStore(
    BarsStreamedStoreMixin,
    UniverseStoreMixin,
    DelistingsStoreMixin,
    FundamentalsStoreMixin,
    NewsStoreMixin,
):
    """Filesystem-backed point-in-time data manifest.

    This first phase-2 slice records immutable local data snapshots. Provider-backed
    ingestion can build on the same manifest contract later.
    """
    # (docstring moved verbatim from the original class)

    def __init__(self, data_dir: Path) -> None:
        ...  # moved verbatim

    def ingest_file(self, *, ...) -> SnapshotRecord:
        ...  # moved verbatim — uses build_metadata, compute_snapshot_id, path_part
        # (from .identity), self._staging, self.manifest, os.replace, shutil.copy2,
        # fsync_file, fsync_parents

    def clear_staging(self, *, max_age_seconds: float = 3600.0) -> None:
        ...  # moved verbatim

    def list_snapshots(self, dataset: Dataset | None = None) -> list[SnapshotRecord]:
        ...  # moved verbatim

    def get_snapshot(self, snapshot_id: str) -> SnapshotRecord:
        ...  # moved verbatim — this is the REAL implementation; the stubs in bars.py/
        # fundamentals.py/news.py never fire at runtime because Python resolves
        # attributes defined directly on DataStore before any base class

    def verify_snapshot(self, rec: SnapshotRecord) -> None:
        ...  # moved verbatim

    def verify_snapshots(self, snapshot_id: str | None = None) -> list[dict[str, Any]]:
        ...  # moved verbatim

    def summary(self) -> dict[str, Any]:
        ...  # moved verbatim
```

**Do NOT list `BarsStoreMixin` or `IdentityMixin` in `DataStore`'s base classes** — they are already reachable transitively (`BarsStreamedStoreMixin` → `BarsStoreMixin`; `UniverseStoreMixin`/`DelistingsStoreMixin` → `IdentityMixin`). Explicitly re-listing them risks a Python `TypeError: Cannot create a consistent method resolution order` if the ordering relative to their subclasses is wrong — omitting them sidesteps the whole issue since C3 linearization resolves the diamond fine on its own.

Also add, at the bottom of `__init__.py` or wherever ruff/your judgment prefers within this file: nothing else — `normalize_symbols` is already re-exported by virtue of being imported at the top of this file (importing a name into a module's namespace makes it accessible as `algua.data.store.normalize_symbols`; no explicit `__all__` is required unless the existing codebase convention uses one — check whether the original `store.py` had an `__all__` and preserve that convention if so).

- [ ] **Step 9: Delete the old file**

```bash
git rm algua/data/store.py
```

- [ ] **Step 10: Full quality gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`

Expected: all four pass. If mypy complains about a mixin's stub method (`get_snapshot`, etc.) having an incompatible signature with the real one on `DataStore`, fix the stub's signature to match exactly (same parameter names, types, and return type). If ruff flags an unused import in any of the 8 new files, remove it — the import lists above are derived from the code's needs but double-check against what actually made it into each file.

Check `git status` for the known momentum-strategy test-fixture hazard; delete if present, don't stage it.

- [ ] **Step 11: Commit**

```bash
git add algua/data/store algua/data/store.py
```
(the `git rm` above already stages the deletion; this adds the 8 new files — `git status` first to confirm only these changes are staged, nothing else)

```bash
git commit -m "$(cat <<'EOF'
refactor: carve data/store.py into data/store/ — one module per dataset (spec §6.1)

DataStore becomes a thin facade composing six dataset mixins (bars, bars_streamed,
universe, delistings, fundamentals, news) plus a shared identity.py for
metadata/snapshot-id/validation helpers. Pure structural move — every method's
name, signature, and behavior is unchanged; external callers and
`from algua.data.store import DataStore` keep working identically.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Close-out verification

**Files:** none expected (verification only; fix anything found)

- [ ] **Step 1: Confirm every external non-test call site still resolves correctly**

```bash
grep -rn "from algua.data.store import\|from algua\.data import store\b\|DataStore(" algua/ --include='*.py' | grep -v 'algua/data/store/'
```

Expected: every hit is one of the known external call sites — `algua/cli/data_cmd.py`, `algua/cli/research_cmd.py`, `algua/cli/app.py`, `algua/cli/backtest_cmd.py`, `algua/cli/_common.py`, `algua/registry/universe_binding.py`, `algua/data/serve.py`, `algua/data/hindsight.py` — and none of them needed to change (they all import `DataStore` by name, unaffected by the internal carve).

- [ ] **Step 2: Confirm the aliased test import still resolves**

```bash
grep -n "from algua.data.store import" tests/test_concurrency.py
```

Expected: `from algua.data.store import DataStore as _DataStore` (or similar) still present and unedited — this proves the package's `__init__.py` re-export path works exactly like the old module did.

- [ ] **Step 3: Confirm `normalize_symbols`'s direct test import still works**

```bash
grep -n "normalize_symbols" tests/test_data_store.py
```

Read the actual import line and confirm it still says `from algua.data.store import normalize_symbols` (not `from algua.data.store.identity import ...`) — if the test imports it from the top-level package path, `store/__init__.py`'s re-export must keep satisfying it unedited.

- [ ] **Step 4: Grep for any stray reference to the removed private names**

```bash
grep -rn "_metadata\b\|_snapshot_id\b\|_validate_non_empty\|_validate_date_bounds\|_validate_datetime\b\|_path_part\b" algua/data/ tests/ --include='*.py'
```

Expected: no hits outside `algua/data/store/identity.py` itself (which now defines the renamed public versions) — any hit elsewhere means a stale reference to the old private name was missed during the carve.

- [ ] **Step 5: Full quality gate one more time**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`

Expected: all four pass (this re-confirms Task 1's gate on a clean tree, catching anything a `git status` oversight might have missed).

- [ ] **Step 6: CLI smoke test**

```bash
uv run algua data inspect --summary
uv run algua data verify
```

Expected: both exit 0 (or a clean, unrelated non-zero on an empty/no-data worktree — anything mentioning a missing `DataStore` attribute or import error is a real regression, not expected).

- [ ] **Step 7: Commit any fixes**

If steps 1-6 forced fixes, commit them (scoped `git add`, correct trailer). If nothing needed fixing, this task makes no commit — that's expected and consistent with how Stage 2's equivalent close-out task (Task 13) landed.
