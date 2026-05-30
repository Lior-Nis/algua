# Foundation & Command Surface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the Algua repo foundation — a runnable, JSON-emitting `algua` CLI backed by a typed contracts layer, a market-calendar wrapper, and the SQLite lifecycle registry with its live-gate rules — so both humans and agents can discover and drive the system.

**Architecture:** A single importable Python package (`algua/`) containing all submodules (`contracts`, `calendar`, `config`, `registry`, `cli`). Pure, well-bounded modules with typed interfaces; the CLI is the one command surface humans and agents share, and every data command emits JSON. The registry is the single source of truth for strategy lifecycle stage; transitions are validated, and the `→ live` transition is gated behind a verified human approval.

**Tech Stack:** Python 3.12, `uv` (packaging + venv), Pydantic v2 + `pydantic-settings` (config), Typer (CLI), `exchange-calendars` (market calendar), stdlib `sqlite3` (registry, WAL mode), pytest (tests), ruff + mypy + import-linter (quality gates).

**Implementation note on layout:** The spec's diagram drew `cli/` and `config/` at the repo top level for emphasis. For clean packaging they live *inside* the `algua/` package as `algua/cli` and `algua/config`. Single top-level importable package — standard and agent-legible.

---

### Task 0: Project scaffolding & tooling

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `algua/__init__.py`
- Create: `README.md`
- Create package dirs with `__init__.py`: `algua/contracts/`, `algua/calendar/`, `algua/config/`, `algua/registry/`, `algua/cli/`, `tests/`

- [ ] **Step 1: Initialize git and commit the existing spec**

Run:
```bash
cd /home/liornisimov/Projects/algua
git init
git add docs/superpowers/specs/2026-05-29-algua-platform-architecture-design.md docs/superpowers/plans/2026-05-29-foundation-command-surface.md
git commit -m "docs: add platform architecture spec and foundation plan"
```
Expected: a repo on branch `main` (or `master`) with one commit.

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "algua"
version = "0.0.1"
description = "Agent-first algotrading research and lifecycle platform"
requires-python = ">=3.12"
dependencies = [
    "pydantic>=2.7",
    "pydantic-settings>=2.3",
    "typer>=0.12",
    "exchange-calendars>=4.5",
    "pandas>=2.2",
]

[project.scripts]
algua = "algua.cli.main:app"

[dependency-groups]
dev = [
    "pytest>=8.2",
    "ruff>=0.5",
    "mypy>=1.10",
    "import-linter>=2.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["algua"]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]

[tool.mypy]
python_version = "3.12"
ignore_missing_imports = true
warn_unused_ignores = true
```

- [ ] **Step 3: Write `.gitignore`**

```gitignore
__pycache__/
*.py[cod]
.venv/
.mypy_cache/
.pytest_cache/
.ruff_cache/
data/
artifacts/
mlruns/
*.db
*.db-wal
*.db-shm
.env
```

- [ ] **Step 4: Write `.env.example`** (interface documentation only — real secrets handled later)

```dotenv
# Algua configuration (prefix ALGUA_). Copy to .env for local overrides.
ALGUA_DB_PATH=data/algua.db
ALGUA_DATA_DIR=data
ALGUA_EXCHANGE=XNYS
ALGUA_TIMEZONE=America/New_York
```

- [ ] **Step 5: Create package files**

`algua/__init__.py`:
```python
__version__ = "0.0.1"
```

Create empty `__init__.py` in each of: `algua/contracts/`, `algua/calendar/`, `algua/config/`, `algua/registry/`, `algua/cli/`. Create `tests/__init__.py`.

`README.md`:
```markdown
# Algua

Agent-first algorithmic-trading research and lifecycle platform.
See `docs/superpowers/specs/` for the architecture and `docs/agent/` for operating docs.

## Quickstart
```
uv sync
uv run algua doctor
```
```

- [ ] **Step 6: Sync the environment and verify the toolchain**

Run:
```bash
uv sync
uv run python -c "import algua; print(algua.__version__)"
```
Expected: prints `0.0.1` with no import errors.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .gitignore .env.example algua/ tests/ README.md uv.lock
git commit -m "chore: scaffold algua package and toolchain"
```

---

### Task 1: Configuration (Settings)

**Files:**
- Create: `algua/config/settings.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from pathlib import Path
from algua.config.settings import Settings, get_settings


def test_defaults():
    s = Settings()
    assert s.exchange == "XNYS"
    assert s.db_path == Path("data/algua.db")


def test_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "x.db"))
    monkeypatch.setenv("ALGUA_EXCHANGE", "XLON")
    s = get_settings()
    assert s.exchange == "XLON"
    assert s.db_path == tmp_path / "x.db"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'algua.config.settings'`.

- [ ] **Step 3: Write minimal implementation**

```python
# algua/config/settings.py
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ALGUA_", env_file=".env", extra="ignore")

    db_path: Path = Path("data/algua.db")
    data_dir: Path = Path("data")
    exchange: str = "XNYS"
    timezone: str = "America/New_York"


def get_settings() -> Settings:
    """Fresh Settings each call (re-reads env; keeps tests isolated)."""
    return Settings()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add algua/config/settings.py tests/test_config.py
git commit -m "feat: add pydantic settings"
```

---

### Task 2: Lifecycle stages & transition rules

**Files:**
- Create: `algua/contracts/lifecycle.py`
- Test: `tests/test_lifecycle.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_lifecycle.py
import pytest
from algua.contracts.lifecycle import (
    Stage, Actor, can_transition, validate_transition, TransitionError,
)


def test_legal_transition():
    assert can_transition(Stage.IDEA, Stage.BACKTESTED) is True
    assert can_transition(Stage.SHORTLISTED, Stage.PAPER) is True


def test_illegal_transition():
    assert can_transition(Stage.IDEA, Stage.LIVE) is False
    assert can_transition(Stage.RETIRED, Stage.IDEA) is False


def test_validate_raises_on_illegal():
    with pytest.raises(TransitionError):
        validate_transition(Stage.IDEA, Stage.LIVE)


def test_actor_values():
    assert {a.value for a in Actor} == {"human", "agent", "system"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_lifecycle.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'algua.contracts.lifecycle'`.

- [ ] **Step 3: Write minimal implementation**

```python
# algua/contracts/lifecycle.py
from __future__ import annotations

from enum import Enum


class Stage(str, Enum):
    IDEA = "idea"
    BACKTESTED = "backtested"
    SHORTLISTED = "shortlisted"
    PAPER = "paper"
    LIVE = "live"
    RETIRED = "retired"


class Actor(str, Enum):
    HUMAN = "human"
    AGENT = "agent"
    SYSTEM = "system"


ALLOWED_TRANSITIONS: dict[Stage, set[Stage]] = {
    Stage.IDEA: {Stage.BACKTESTED, Stage.RETIRED},
    Stage.BACKTESTED: {Stage.SHORTLISTED, Stage.IDEA, Stage.RETIRED},
    Stage.SHORTLISTED: {Stage.PAPER, Stage.BACKTESTED, Stage.RETIRED},
    Stage.PAPER: {Stage.LIVE, Stage.SHORTLISTED, Stage.RETIRED},
    Stage.LIVE: {Stage.PAPER, Stage.RETIRED},
    Stage.RETIRED: set(),
}


class TransitionError(ValueError):
    pass


def can_transition(frm: Stage, to: Stage) -> bool:
    return to in ALLOWED_TRANSITIONS[frm]


def validate_transition(frm: Stage, to: Stage) -> None:
    if not can_transition(frm, to):
        raise TransitionError(f"illegal transition {frm.value} -> {to.value}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_lifecycle.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add algua/contracts/lifecycle.py tests/test_lifecycle.py
git commit -m "feat: add lifecycle stages and transition rules"
```

---

### Task 3: Core contracts (types & protocols)

**Files:**
- Create: `algua/contracts/types.py`
- Test: `tests/test_contracts.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_contracts.py
import pytest
from datetime import datetime
from algua.contracts.types import ExecutionContract, OrderIntent, Side, Strategy


def test_execution_contract_rejects_same_bar_fill():
    with pytest.raises(ValueError):
        ExecutionContract(rebalance_frequency="1D", decision_lag_bars=0)


def test_execution_contract_defaults():
    c = ExecutionContract(rebalance_frequency="1D")
    assert c.decision_lag_bars == 1
    assert c.allow_fractional is True


def test_order_intent_fields():
    oi = OrderIntent(symbol="AAPL", side=Side.BUY, target_weight=0.1,
                     decision_ts=datetime(2025, 1, 2))
    assert oi.symbol == "AAPL"
    assert oi.side is Side.BUY


def test_strategy_protocol_runtime_check():
    class Dummy:
        name = "dummy"
        execution = ExecutionContract(rebalance_frequency="1D")

        def target_weights(self, features):  # noqa: ANN001
            return features

    assert isinstance(Dummy(), Strategy)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_contracts.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'algua.contracts.types'`.

- [ ] **Step 3: Write minimal implementation**

```python
# algua/contracts/types.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:  # keep contracts import-light; pandas only needed for typing
    import pandas as pd


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True)
class ExecutionContract:
    """How target weights become executable orders. Pinned per strategy.

    decision_lag_bars >= 1 enforces the t -> t+1 rule: features are computed on a
    fully closed bar t and orders may fill no earlier than t + lag. This forbids
    same-bar fills, the single most likely source of look-ahead bias.
    """

    rebalance_frequency: str
    decision_lag_bars: int = 1
    allow_fractional: bool = True
    max_gross_exposure: float = 1.0

    def __post_init__(self) -> None:
        if self.decision_lag_bars < 1:
            raise ValueError("decision_lag_bars must be >= 1 (no same-bar fills)")


@dataclass(frozen=True)
class OrderIntent:
    symbol: str
    side: Side
    target_weight: float
    decision_ts: datetime


@runtime_checkable
class Strategy(Protocol):
    name: str
    execution: ExecutionContract

    def target_weights(self, features: "pd.DataFrame") -> "pd.Series": ...


@runtime_checkable
class DataProvider(Protocol):
    def get_bars(
        self, symbols: list[str], start: datetime, end: datetime, timeframe: str
    ) -> "pd.DataFrame": ...


@runtime_checkable
class Broker(Protocol):
    def get_positions(self) -> "pd.Series": ...

    def submit(self, intent: OrderIntent) -> str: ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_contracts.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add algua/contracts/types.py tests/test_contracts.py
git commit -m "feat: add core contracts (ExecutionContract, OrderIntent, protocols)"
```

---

### Task 4: Market calendar wrapper

**Files:**
- Create: `algua/calendar/market_calendar.py`
- Test: `tests/test_calendar.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_calendar.py
from datetime import date
from algua.calendar.market_calendar import MarketCalendar


def test_holiday_is_not_a_session():
    cal = MarketCalendar("XNYS")
    assert cal.is_session(date(2025, 7, 4)) is False   # Independence Day
    assert cal.is_session(date(2025, 1, 1)) is False   # New Year's Day


def test_trading_day_is_a_session():
    cal = MarketCalendar("XNYS")
    assert cal.is_session(date(2025, 7, 3)) is True


def test_next_and_previous_session_skip_holiday():
    cal = MarketCalendar("XNYS")
    assert cal.next_session(date(2025, 7, 4)) == date(2025, 7, 7)      # Monday
    assert cal.previous_session(date(2025, 7, 4)) == date(2025, 7, 3)


def test_sessions_in_range_count():
    cal = MarketCalendar("XNYS")
    sessions = cal.sessions_in_range(date(2025, 7, 1), date(2025, 7, 7))
    # Jul 1,2,3 trading; Jul 4 holiday; Jul 5,6 weekend; Jul 7 trading
    assert sessions == [date(2025, 7, 1), date(2025, 7, 2),
                        date(2025, 7, 3), date(2025, 7, 7)]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_calendar.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'algua.calendar.market_calendar'`.

- [ ] **Step 3: Write minimal implementation**

```python
# algua/calendar/market_calendar.py
from __future__ import annotations

from datetime import date

import exchange_calendars as xcals
import pandas as pd


class MarketCalendar:
    """Thin wrapper over exchange_calendars. Both backtest and live depend on this."""

    def __init__(self, code: str = "XNYS") -> None:
        self.code = code
        self._cal = xcals.get_calendar(code)

    def is_session(self, day: date) -> bool:
        return bool(self._cal.is_session(pd.Timestamp(day)))

    def next_session(self, day: date) -> date:
        return self._cal.next_session(pd.Timestamp(day)).date()

    def previous_session(self, day: date) -> date:
        return self._cal.previous_session(pd.Timestamp(day)).date()

    def sessions_in_range(self, start: date, end: date) -> list[date]:
        idx = self._cal.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))
        return [ts.date() for ts in idx]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_calendar.py -v`
Expected: PASS (4 passed). If any `exchange_calendars` method name differs in the installed version, run `uv run python -c "import exchange_calendars as x; c=x.get_calendar('XNYS'); print([m for m in dir(c) if 'session' in m])"` and adjust the wrapper to the available names; keep the tests unchanged (they assert real NYSE facts).

- [ ] **Step 5: Commit**

```bash
git add algua/calendar/market_calendar.py tests/test_calendar.py
git commit -m "feat: add market calendar wrapper"
```

---

### Task 5: Registry — connection & migrations

**Files:**
- Create: `algua/registry/db.py`
- Test: `tests/test_registry_db.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_registry_db.py
from algua.registry.db import connect, migrate


def test_migrate_creates_tables_and_sets_version(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    tables = {r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert {"strategies", "stage_transitions", "approvals"} <= tables
    assert conn.execute("PRAGMA user_version;").fetchone()[0] == 1


def test_migrate_is_idempotent(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    migrate(conn)  # second run must not raise
    assert conn.execute("PRAGMA user_version;").fetchone()[0] == 1


def test_wal_mode_enabled(tmp_path):
    conn = connect(tmp_path / "r.db")
    mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
    assert mode.lower() == "wal"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_registry_db.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'algua.registry.db'`.

- [ ] **Step 3: Write minimal implementation**

```python
# algua/registry/db.py
from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS strategies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    stage TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS stage_transitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id INTEGER NOT NULL REFERENCES strategies(id),
    from_stage TEXT,
    to_stage TEXT NOT NULL,
    actor TEXT NOT NULL,
    reason TEXT,
    code_hash TEXT,
    config_hash TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS approvals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id INTEGER NOT NULL REFERENCES strategies(id),
    code_hash TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    approved_by TEXT NOT NULL,
    created_at TEXT NOT NULL,
    revoked_at TEXT
);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def migrate(conn: sqlite3.Connection) -> None:
    version = conn.execute("PRAGMA user_version;").fetchone()[0]
    if version < SCHEMA_VERSION:
        conn.executescript(_SCHEMA)
        conn.execute(f"PRAGMA user_version={SCHEMA_VERSION};")
        conn.commit()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_registry_db.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add algua/registry/db.py tests/test_registry_db.py
git commit -m "feat: add registry db connection and migrations"
```

---

### Task 6: Registry — strategy CRUD & transitions

**Files:**
- Create: `algua/registry/store.py`
- Test: `tests/test_registry_store.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_registry_store.py
import pytest
from algua.registry.db import connect, migrate
from algua.registry import store
from algua.contracts.lifecycle import Stage, Actor, TransitionError


@pytest.fixture()
def conn(tmp_path):
    c = connect(tmp_path / "r.db")
    migrate(c)
    return c


def test_add_creates_idea_with_initial_transition(conn):
    rec = store.add_strategy(conn, "alpha")
    assert rec.stage is Stage.IDEA
    transitions = store.list_transitions(conn, "alpha")
    assert len(transitions) == 1
    assert transitions[0]["to_stage"] == "idea"
    assert transitions[0]["actor"] == "system"


def test_duplicate_name_raises(conn):
    store.add_strategy(conn, "alpha")
    with pytest.raises(store.StrategyExists):
        store.add_strategy(conn, "alpha")


def test_legal_transition_updates_stage_and_history(conn):
    store.add_strategy(conn, "alpha")
    rec = store.transition(conn, "alpha", Stage.BACKTESTED, Actor.AGENT, "ran backtest")
    assert rec.stage is Stage.BACKTESTED
    assert len(store.list_transitions(conn, "alpha")) == 2


def test_illegal_transition_raises(conn):
    store.add_strategy(conn, "alpha")
    with pytest.raises(TransitionError):
        store.transition(conn, "alpha", Stage.LIVE, Actor.AGENT)


def test_list_filters_by_stage(conn):
    store.add_strategy(conn, "alpha")
    store.add_strategy(conn, "beta")
    store.transition(conn, "beta", Stage.BACKTESTED, Actor.AGENT)
    ideas = store.list_strategies(conn, Stage.IDEA)
    assert [r.name for r in ideas] == ["alpha"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_registry_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'algua.registry.store'`.

- [ ] **Step 3: Write minimal implementation**

```python
# algua/registry/store.py
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

from algua.contracts.lifecycle import Actor, Stage, validate_transition


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StrategyExists(ValueError):
    pass


class StrategyNotFound(LookupError):
    pass


@dataclass
class StrategyRecord:
    id: int
    name: str
    stage: Stage
    created_at: str
    updated_at: str


def _row_to_record(row: sqlite3.Row) -> StrategyRecord:
    return StrategyRecord(
        id=row["id"], name=row["name"], stage=Stage(row["stage"]),
        created_at=row["created_at"], updated_at=row["updated_at"],
    )


def add_strategy(conn: sqlite3.Connection, name: str) -> StrategyRecord:
    now = _now()
    try:
        cur = conn.execute(
            "INSERT INTO strategies(name, stage, created_at, updated_at) VALUES (?,?,?,?)",
            (name, Stage.IDEA.value, now, now),
        )
    except sqlite3.IntegrityError as exc:
        raise StrategyExists(name) from exc
    conn.execute(
        "INSERT INTO stage_transitions"
        "(strategy_id, from_stage, to_stage, actor, reason, created_at) VALUES (?,?,?,?,?,?)",
        (cur.lastrowid, None, Stage.IDEA.value, Actor.SYSTEM.value, "created", now),
    )
    conn.commit()
    return get_strategy(conn, name)


def get_strategy(conn: sqlite3.Connection, name: str) -> StrategyRecord:
    row = conn.execute("SELECT * FROM strategies WHERE name = ?", (name,)).fetchone()
    if row is None:
        raise StrategyNotFound(name)
    return _row_to_record(row)


def list_strategies(conn: sqlite3.Connection, stage: Stage | None = None) -> list[StrategyRecord]:
    if stage is None:
        rows = conn.execute("SELECT * FROM strategies ORDER BY id").fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM strategies WHERE stage = ? ORDER BY id", (stage.value,)
        ).fetchall()
    return [_row_to_record(r) for r in rows]


def list_transitions(conn: sqlite3.Connection, name: str) -> list[dict]:
    rec = get_strategy(conn, name)
    rows = conn.execute(
        "SELECT * FROM stage_transitions WHERE strategy_id = ? ORDER BY id", (rec.id,)
    ).fetchall()
    return [dict(r) for r in rows]


def transition(
    conn: sqlite3.Connection,
    name: str,
    to: Stage,
    actor: Actor,
    reason: str | None = None,
    code_hash: str | None = None,
    config_hash: str | None = None,
) -> StrategyRecord:
    rec = get_strategy(conn, name)
    validate_transition(rec.stage, to)
    # Live-gate enforcement is added in Task 7.
    now = _now()
    conn.execute(
        "UPDATE strategies SET stage = ?, updated_at = ? WHERE id = ?",
        (to.value, now, rec.id),
    )
    conn.execute(
        "INSERT INTO stage_transitions"
        "(strategy_id, from_stage, to_stage, actor, reason, code_hash, config_hash, created_at)"
        " VALUES (?,?,?,?,?,?,?,?)",
        (rec.id, rec.stage.value, to.value, actor.value, reason, code_hash, config_hash, now),
    )
    conn.commit()
    return get_strategy(conn, name)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_registry_store.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add algua/registry/store.py tests/test_registry_store.py
git commit -m "feat: add registry strategy CRUD and transitions"
```

---

### Task 7: Registry — approvals & live gate

**Files:**
- Create: `algua/registry/approvals.py`
- Modify: `algua/registry/store.py` (add live-gate enforcement to `transition`)
- Test: `tests/test_registry_approvals.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_registry_approvals.py
import pytest
from algua.registry.db import connect, migrate
from algua.registry import store
from algua.registry.approvals import record_approval, has_valid_approval
from algua.contracts.lifecycle import Stage, Actor, TransitionError


@pytest.fixture()
def conn(tmp_path):
    c = connect(tmp_path / "r.db")
    migrate(c)
    return c


def _advance_to_paper(conn, name):
    store.add_strategy(conn, name)
    store.transition(conn, name, Stage.BACKTESTED, Actor.AGENT)
    store.transition(conn, name, Stage.SHORTLISTED, Actor.AGENT)
    store.transition(conn, name, Stage.PAPER, Actor.AGENT)


def test_live_requires_approval(conn):
    _advance_to_paper(conn, "alpha")
    with pytest.raises(TransitionError):
        store.transition(conn, "alpha", Stage.LIVE, Actor.HUMAN,
                         code_hash="c1", config_hash="g1")


def test_live_requires_human_actor(conn):
    _advance_to_paper(conn, "alpha")
    record_approval(conn, "alpha", "c1", "g1", "lior")
    with pytest.raises(TransitionError):
        store.transition(conn, "alpha", Stage.LIVE, Actor.AGENT,
                         code_hash="c1", config_hash="g1")


def test_live_requires_matching_hash(conn):
    _advance_to_paper(conn, "alpha")
    record_approval(conn, "alpha", "c1", "g1", "lior")
    with pytest.raises(TransitionError):
        store.transition(conn, "alpha", Stage.LIVE, Actor.HUMAN,
                         code_hash="DIFFERENT", config_hash="g1")


def test_live_succeeds_with_human_and_matching_approval(conn):
    _advance_to_paper(conn, "alpha")
    record_approval(conn, "alpha", "c1", "g1", "lior")
    rec = store.transition(conn, "alpha", Stage.LIVE, Actor.HUMAN,
                           code_hash="c1", config_hash="g1")
    assert rec.stage is Stage.LIVE


def test_has_valid_approval(conn):
    store.add_strategy(conn, "alpha")
    s = store.get_strategy(conn, "alpha")
    assert has_valid_approval(conn, s.id, "c1", "g1") is False
    record_approval(conn, "alpha", "c1", "g1", "lior")
    assert has_valid_approval(conn, s.id, "c1", "g1") is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_registry_approvals.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'algua.registry.approvals'`.

- [ ] **Step 3a: Write the approvals module**

```python
# algua/registry/approvals.py
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from algua.registry.store import get_strategy


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def record_approval(
    conn: sqlite3.Connection, name: str, code_hash: str, config_hash: str, approved_by: str
) -> int:
    rec = get_strategy(conn, name)
    cur = conn.execute(
        "INSERT INTO approvals(strategy_id, code_hash, config_hash, approved_by, created_at)"
        " VALUES (?,?,?,?,?)",
        (rec.id, code_hash, config_hash, approved_by, _now()),
    )
    conn.commit()
    return int(cur.lastrowid)


def has_valid_approval(
    conn: sqlite3.Connection, strategy_id: int, code_hash: str, config_hash: str
) -> bool:
    row = conn.execute(
        "SELECT 1 FROM approvals WHERE strategy_id=? AND code_hash=? AND config_hash=?"
        " AND revoked_at IS NULL LIMIT 1",
        (strategy_id, code_hash, config_hash),
    ).fetchone()
    return row is not None
```

- [ ] **Step 3b: Add live-gate enforcement to `transition` in `algua/registry/store.py`**

Replace the comment line `# Live-gate enforcement is added in Task 7.` and the lines around it so the start of `transition` reads:

```python
    rec = get_strategy(conn, name)
    validate_transition(rec.stage, to)
    if to is Stage.LIVE:
        # Imported locally to avoid an import cycle (approvals imports store.get_strategy).
        from algua.registry.approvals import has_valid_approval

        if actor is not Actor.HUMAN:
            raise TransitionError("transition to live requires a human actor")
        if code_hash is None or config_hash is None:
            raise TransitionError("transition to live requires code_hash and config_hash")
        if not has_valid_approval(conn, rec.id, code_hash, config_hash):
            raise TransitionError("no matching human approval for this code+config")
    now = _now()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_registry_approvals.py tests/test_registry_store.py -v`
Expected: PASS (all green — earlier store tests still pass).

- [ ] **Step 5: Commit**

```bash
git add algua/registry/approvals.py algua/registry/store.py tests/test_registry_approvals.py
git commit -m "feat: add approvals and live-gate enforcement"
```

---

### Task 8: CLI — app skeleton, version & doctor

**Files:**
- Create: `algua/cli/app.py`
- Create: `algua/cli/main.py`
- Test: `tests/test_cli_core.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli_core.py
import json
from typer.testing import CliRunner
from algua.cli.main import app

runner = CliRunner()


def test_version_emits_json():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["name"] == "algua"


def test_doctor_passes_in_clean_env(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert {c["check"] for c in payload["checks"]} >= {"python", "registry_db", "calendar"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli_core.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'algua.cli.main'`.

- [ ] **Step 3: Write minimal implementation**

```python
# algua/cli/app.py
from __future__ import annotations

import json
import sys
from typing import Any

import typer

from algua import __version__
from algua.calendar.market_calendar import MarketCalendar
from algua.config.settings import get_settings
from algua.registry.db import connect, migrate

app = typer.Typer(
    help="Algua — agent-first algotrading platform", no_args_is_help=True
)


def emit(data: Any) -> None:
    """Print a value as indented JSON — the shared machine + human surface."""
    typer.echo(json.dumps(data, indent=2, default=str))


@app.command()
def version() -> None:
    """Print the package version as JSON."""
    emit({"name": "algua", "version": __version__})


@app.command()
def doctor() -> None:
    """Check environment readiness. Exits non-zero if any check fails."""
    settings = get_settings()
    checks: list[dict[str, Any]] = [
        {"check": "python", "ok": sys.version_info >= (3, 12),
         "detail": sys.version.split()[0]},
    ]
    try:
        conn = connect(settings.db_path)
        migrate(conn)
        conn.close()
        checks.append({"check": "registry_db", "ok": True, "detail": str(settings.db_path)})
    except Exception as exc:  # noqa: BLE001 - report any failure as a check result
        checks.append({"check": "registry_db", "ok": False, "detail": str(exc)})
    try:
        MarketCalendar(settings.exchange)
        checks.append({"check": "calendar", "ok": True, "detail": settings.exchange})
    except Exception as exc:  # noqa: BLE001
        checks.append({"check": "calendar", "ok": False, "detail": str(exc)})

    all_ok = all(c["ok"] for c in checks)
    emit({"ok": all_ok, "checks": checks})
    raise typer.Exit(code=0 if all_ok else 1)
```

```python
# algua/cli/main.py
from __future__ import annotations

from algua.cli.app import app
from algua.cli import registry_cmd  # noqa: F401 - import registers registry subcommands

__all__ = ["app"]
```

Note: `registry_cmd` is created in Task 9. Until then, temporarily comment out the
`from algua.cli import registry_cmd` line so `main.py` imports cleanly; re-enable it in Task 9.
To keep this task self-contained, write `main.py` now as:

```python
# algua/cli/main.py
from __future__ import annotations

from algua.cli.app import app

__all__ = ["app"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cli_core.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add algua/cli/app.py algua/cli/main.py tests/test_cli_core.py
git commit -m "feat: add CLI app with version and doctor commands"
```

---

### Task 9: CLI — registry commands

**Files:**
- Create: `algua/cli/registry_cmd.py`
- Modify: `algua/cli/main.py` (register the registry sub-app)
- Test: `tests/test_cli_registry.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli_registry.py
import json
import pytest
from typer.testing import CliRunner
from algua.cli.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _tmp_db(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))


def _json(result):
    assert result.exit_code == 0, result.stdout
    return json.loads(result.stdout)


def test_add_and_list():
    _json(runner.invoke(app, ["registry", "add", "alpha"]))
    listed = _json(runner.invoke(app, ["registry", "list"]))
    assert [s["name"] for s in listed] == ["alpha"]
    assert listed[0]["stage"] == "idea"


def test_transition_legal():
    runner.invoke(app, ["registry", "add", "alpha"])
    out = _json(runner.invoke(
        app, ["registry", "transition", "alpha", "--to", "backtested",
              "--actor", "agent", "--reason", "ran"]))
    assert out["stage"] == "backtested"


def test_transition_illegal_exits_nonzero():
    runner.invoke(app, ["registry", "add", "alpha"])
    result = runner.invoke(
        app, ["registry", "transition", "alpha", "--to", "live", "--actor", "agent"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False


def test_full_path_to_live_with_approval():
    runner.invoke(app, ["registry", "add", "alpha"])
    for stage in ("backtested", "shortlisted", "paper"):
        runner.invoke(app, ["registry", "transition", "alpha",
                            "--to", stage, "--actor", "agent"])
    runner.invoke(app, ["registry", "approve", "alpha",
                        "--code-hash", "c1", "--config-hash", "g1", "--by", "lior"])
    out = _json(runner.invoke(
        app, ["registry", "transition", "alpha", "--to", "live", "--actor", "human",
              "--code-hash", "c1", "--config-hash", "g1"]))
    assert out["stage"] == "live"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli_registry.py -v`
Expected: FAIL — the `registry` subcommand does not exist yet (non-zero exit / usage error).

- [ ] **Step 3: Write the registry command module**

```python
# algua/cli/registry_cmd.py
from __future__ import annotations

import sqlite3

import typer

from algua.cli.app import app, emit
from algua.config.settings import get_settings
from algua.contracts.lifecycle import Actor, Stage, TransitionError
from algua.registry import store
from algua.registry.approvals import record_approval
from algua.registry.db import connect, migrate

registry_app = typer.Typer(help="Strategy lifecycle registry", no_args_is_help=True)
app.add_typer(registry_app, name="registry")


def _conn() -> sqlite3.Connection:
    conn = connect(get_settings().db_path)
    migrate(conn)
    return conn


@registry_app.command("add")
def add(name: str) -> None:
    """Register a new strategy at stage 'idea'."""
    rec = store.add_strategy(_conn(), name)
    emit({"id": rec.id, "name": rec.name, "stage": rec.stage.value})


@registry_app.command("list")
def list_(stage: str = typer.Option(None, "--stage", help="filter by stage")) -> None:
    """List strategies, optionally filtered by stage."""
    st = Stage(stage) if stage else None
    recs = store.list_strategies(_conn(), st)
    emit([{"id": r.id, "name": r.name, "stage": r.stage.value} for r in recs])


@registry_app.command("show")
def show(name: str) -> None:
    """Show a strategy and its transition history."""
    conn = _conn()
    rec = store.get_strategy(conn, name)
    emit({"id": rec.id, "name": rec.name, "stage": rec.stage.value,
          "transitions": store.list_transitions(conn, name)})


@registry_app.command("transition")
def transition(
    name: str,
    to: str = typer.Option(..., "--to"),
    actor: str = typer.Option(..., "--actor"),
    reason: str = typer.Option(None, "--reason"),
    code_hash: str = typer.Option(None, "--code-hash"),
    config_hash: str = typer.Option(None, "--config-hash"),
) -> None:
    """Advance a strategy to a new lifecycle stage."""
    try:
        rec = store.transition(
            _conn(), name, Stage(to), Actor(actor), reason, code_hash, config_hash
        )
    except TransitionError as exc:
        emit({"ok": False, "error": str(exc)})
        raise typer.Exit(code=1) from exc
    emit({"ok": True, "name": rec.name, "stage": rec.stage.value})


@registry_app.command("approve")
def approve(
    name: str,
    code_hash: str = typer.Option(..., "--code-hash"),
    config_hash: str = typer.Option(..., "--config-hash"),
    by: str = typer.Option(..., "--by", help="human approver identity"),
) -> None:
    """Record a human approval binding code+config hashes (required for going live)."""
    aid = record_approval(_conn(), name, code_hash, config_hash, by)
    emit({"ok": True, "approval_id": aid})
```

- [ ] **Step 4: Register the sub-app in `algua/cli/main.py`**

Replace the contents of `algua/cli/main.py` with:

```python
# algua/cli/main.py
from __future__ import annotations

from algua.cli.app import app
from algua.cli import registry_cmd  # noqa: F401 - import registers registry subcommands

__all__ = ["app"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli_registry.py tests/test_cli_core.py -v`
Expected: PASS (all green).

- [ ] **Step 6: Commit**

```bash
git add algua/cli/registry_cmd.py algua/cli/main.py tests/test_cli_registry.py
git commit -m "feat: add registry CLI commands"
```

---

### Task 10: Architecture guardrail — import-linter contracts

**Files:**
- Modify: `pyproject.toml` (append import-linter config)
- Test: command-based (`lint-imports`)

- [ ] **Step 1: Append import-linter configuration to `pyproject.toml`**

```toml
[tool.importlinter]
root_package = "algua"

[[tool.importlinter.contracts]]
name = "contracts layer is pure (imports no other algua module)"
type = "forbidden"
source_modules = ["algua.contracts"]
forbidden_modules = [
    "algua.cli",
    "algua.registry",
    "algua.config",
    "algua.calendar",
]

[[tool.importlinter.contracts]]
name = "calendar stays independent of cli and registry"
type = "forbidden"
source_modules = ["algua.calendar"]
forbidden_modules = ["algua.cli", "algua.registry"]
```

- [ ] **Step 2: Run the contracts check**

Run: `uv run lint-imports`
Expected: `Contracts: 2 kept, 0 broken.` (output ends with all contracts KEPT).

If a contract is broken, the offending import violates a core design principle — fix the import, do not weaken the contract.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: enforce module boundaries with import-linter"
```

---

### Task 11: Agent operating docs

**Files:**
- Create: `CLAUDE.md`
- Create: `docs/agent/operating.md`

- [ ] **Step 1: Write `CLAUDE.md`**

```markdown
# Algua — Agent Operating Guide

Algua is an agent-first algotrading platform. You (an agent) and the human operator
drive the system through the **same** CLI. Every data command emits JSON on stdout.

## Golden rules
- Drive the system through `uv run algua ...`. Never reach into modules to bypass the CLI.
- You may operate the lifecycle autonomously **up to and including paper**.
- You may **never** put a strategy live. The `paper -> live` transition requires a
  verified human approval and a human actor; the system enforces this.
- Keep `algua/contracts` and `algua/features` pure (no I/O, no cross-module imports
  beyond contracts). Import-linter enforces boundaries; run `uv run lint-imports`.

## Command surface
- `uv run algua version` — version JSON.
- `uv run algua doctor` — environment readiness; non-zero exit means a failed check.
- `uv run algua registry add <name>` — register a strategy (stage `idea`).
- `uv run algua registry list [--stage S]` — list strategies.
- `uv run algua registry show <name>` — strategy + transition history.
- `uv run algua registry transition <name> --to S --actor agent --reason "..."` — advance stage.
- `uv run algua registry approve <name> --code-hash H --config-hash H --by NAME` — human-only.

## Lifecycle stages
`idea -> backtested -> shortlisted -> paper -> live -> retired`
(plus allowed back-steps and `-> retired`). See `algua/contracts/lifecycle.py`.

## Quality gates before committing
`uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
```

- [ ] **Step 2: Write `docs/agent/operating.md`**

```markdown
# Operating Playbook (detail)

This expands `CLAUDE.md` with the why behind the rules.

## The live gate
Lifecycle stage lives in the SQLite registry (`algua/registry`). State is a *record*,
not a wall: because agents can write the registry, a bare `stage='live'` flag is not a
security boundary. Going live therefore requires:
1. A human actor (`--actor human`), and
2. A matching, unrevoked approval row created by `registry approve`, binding the exact
   `code_hash` + `config_hash` being promoted.

The future live runner (sub-project 6) will additionally verify the approval against the
hash of the artifact it is about to run. Trust the approval, never the flag.

## Module boundaries
- `contracts/` — pure types/protocols. No I/O, no other algua imports.
- `calendar/` — market sessions; depended on by both backtest and live.
- `registry/` — lifecycle source of truth (`db`, `store`, `approvals`).
- `config/` — pydantic settings (env prefix `ALGUA_`).
- `cli/` — the shared command surface; `main.py` is the entry point.

## JSON everywhere
Commands print indented JSON so an agent can parse results and a human can read them.
Parse stdout; use exit codes for pass/fail (e.g. `doctor`, illegal `transition`).
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md docs/agent/operating.md
git commit -m "docs: add agent operating guide"
```

---

### Task 12: Full verification & quality gate

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest -q`
Expected: all tests pass (config, lifecycle, contracts, calendar, registry db/store/approvals, CLI core/registry).

- [ ] **Step 2: Run lint, type-check, and import contracts**

Run:
```bash
uv run ruff check .
uv run mypy algua
uv run lint-imports
```
Expected: ruff clean; mypy `Success: no issues found`; import-linter `2 kept, 0 broken`.

- [ ] **Step 3: Smoke-test the CLI end to end**

Run:
```bash
uv run algua doctor
uv run algua registry add demo
uv run algua registry show demo
```
Expected: `doctor` reports `"ok": true`; `add` returns the `demo` strategy at stage `idea`; `show` lists its initial `idea` transition. (Then remove the demo DB if undesired: `rm -f data/algua.db data/algua.db-wal data/algua.db-shm`.)

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "test: verify foundation end to end" --allow-empty
```

---

## Self-Review Notes

- **Spec coverage (sub-project 1 scope):** uv/pyproject (Task 0), config (Task 1), `contracts/`
  incl. lifecycle + execution contract / `t→t+1` invariant (Tasks 2–3), `calendar/` (Task 4),
  `registry/` schema + lifecycle + live-gate (Tasks 5–7), `cli/` structured-output surface
  (Tasks 8–9), import-boundary guardrail (Task 10), `CLAUDE.md` + agent docs (Task 11),
  verification (Task 12). Deferred items (data layer, backtest, paper/live execution) belong to
  later sub-projects per the spec and are intentionally absent.
- **Type consistency:** `Stage`/`Actor` enums, `StrategyRecord`, `transition(...)` signature,
  `has_valid_approval(conn, strategy_id, code_hash, config_hash)`, and `emit(...)` are used with
  identical names/signatures across tasks.
- **No placeholders:** every code step contains complete, runnable code and exact commands.
