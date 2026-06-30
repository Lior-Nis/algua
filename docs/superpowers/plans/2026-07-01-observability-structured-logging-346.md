# Observability — Structured Logging (#346) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans to implement task-by-task. Steps use `- [ ]` tracking.

**Goal:** Add a stdlib-only structured JSON logging facility (pure leaf `algua/observability/`) and wire it into the always-on `paper run-all` / `live run-all` loop with per-cycle correlation IDs, ERROR-level capture on breach/flatten/reconcile paths, and per-cycle golden-signal counters — all to **stderr** so the stdout JSON contract is untouched.

**Architecture:** A new pure stdlib leaf package `algua/observability/` (`log.py` = JsonFormatter + configure_logging + correlation_context + get_logger; `metrics.py` = CycleCounters). Import-linter `forbidden` contract keeps it a leaf and bars the pure layers from importing it. Wiring lives only in `cli/main.py`, `cli/paper_cmd.py`, `cli/live_cmd.py` — additive alongside existing `audit_append` + stdout `emit`.

**Tech Stack:** Python 3.12, stdlib `logging`/`json`/`contextvars`, Typer, pytest. No new dependency.

## Global Constraints
- Logs go to **stderr only**; stdout stays a one-JSON-document machine contract (`cli/app.py:emit`).
- `algua/observability/` imports **no** other `algua.*` module (pure stdlib leaf).
- `contracts`/`features`/`provenance`/`portfolio` must never import `algua.observability`.
- Quality gate per task: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`.
- `from __future__ import annotations` at top of every new module.

---

### Task 1: Pure observability leaf (`log.py` + `metrics.py`)

**Files:**
- Create: `algua/observability/__init__.py`
- Create: `algua/observability/log.py`
- Create: `algua/observability/metrics.py`
- Test: `tests/test_observability.py`

**Interfaces produced:**
- `JsonFormatter(logging.Formatter)` — `format(record) -> str` (one-line JSON).
- `get_logger(name: str) -> logging.Logger` — child of the `algua` logger.
- `configure_logging() -> None` — idempotent stderr handler + level from `ALGUA_LOG_LEVEL`.
- `correlation_context(cid: str | None = None)` — contextmanager yielding the active id.
- `current_correlation_id() -> str | None`.
- `CycleCounters` dataclass: int fields `ticks, breaches, flatten_failures, reconcile_deferred, reconcile_halted`; method `as_fields() -> dict[str, int]`.

**Behavioral spec (from the design):**
- `JsonFormatter.format`: build `fields` (merged first) then overwrite with CORE keys `ts` (UTC ISO-8601), `level`, `logger`, `msg`, and — when `record.exc_info` — `exc_type`/`exc_message`/`stacktrace`; include `correlation_id` only when set. `json.dumps(obj, default=str)`. Wrap the whole body: on ANY error, return minimal `{"ts","level","msg","format_error"}` JSON (never reuse unsafe fields, never raise).
- Caller passes structured data via `logger.info(msg, extra={"fields": {...}})`; formatter reads `getattr(record, "fields", {})`.
- `configure_logging`: get `logging.getLogger("algua")`; ALWAYS `setLevel(_resolve_level(os.environ.get("ALGUA_LOG_LEVEL")))` (unknown → INFO); add a `StreamHandler(sys.stderr)` with the formatter ONLY if no handler carrying `_algua_observability = True` is attached; set `logger.propagate = False`.
- `correlation_context`: `token = _CID.set(cid or uuid4().hex)`; `try: yield value finally: _CID.reset(token)`.

- [ ] **Step 1: Write failing tests** in `tests/test_observability.py`:

```python
from __future__ import annotations

import json
import logging

import pytest

from algua.observability import (
    CycleCounters,
    JsonFormatter,
    configure_logging,
    correlation_context,
    current_correlation_id,
    get_logger,
)


def _format(record: logging.LogRecord) -> dict:
    return json.loads(JsonFormatter().format(record))


def _record(**kw) -> logging.LogRecord:
    base = dict(name="algua.x", level=logging.INFO, pathname=__file__, lineno=1,
                msg="hello", args=(), exc_info=None)
    base.update(kw)
    return logging.LogRecord(**base)


def test_format_is_one_line_json_with_core_keys():
    out = JsonFormatter().format(_record())
    assert "\n" not in out
    d = json.loads(out)
    assert d["msg"] == "hello" and d["level"] == "INFO" and d["logger"] == "algua.x"
    assert "ts" in d


def test_fields_merged_but_core_keys_win():
    rec = _record()
    rec.fields = {"strategy": "alpha", "msg": "SPOOFED", "level": "SPOOFED"}
    d = json.loads(JsonFormatter().format(rec))
    assert d["strategy"] == "alpha"
    assert d["msg"] == "hello" and d["level"] == "INFO"  # core wins


def test_exc_info_rendered_to_strings_not_dropped():
    try:
        raise ValueError("boom")
    except ValueError:
        import sys
        rec = _record(exc_info=sys.exc_info())
    d = json.loads(JsonFormatter().format(rec))
    assert d["exc_type"] == "ValueError" and d["exc_message"] == "boom"
    assert "ValueError" in d["stacktrace"]


def test_nonserializable_field_does_not_drop_record():
    rec = _record()
    rec.fields = {"obj": object()}
    d = json.loads(JsonFormatter().format(rec))  # must not raise
    assert d["msg"] == "hello"


def test_correlation_id_present_only_inside_context():
    rec = _record()
    assert "correlation_id" not in json.loads(JsonFormatter().format(rec))
    with correlation_context("CID123"):
        rec2 = _record()
        assert json.loads(JsonFormatter().format(rec2))["correlation_id"] == "CID123"
    assert current_correlation_id() is None


def test_correlation_context_resets_on_exception():
    with pytest.raises(RuntimeError):
        with correlation_context("X"):
            raise RuntimeError("nope")
    assert current_correlation_id() is None


def test_configure_logging_idempotent_and_stderr(capfd, monkeypatch):
    monkeypatch.setenv("ALGUA_LOG_LEVEL", "DEBUG")
    configure_logging()
    configure_logging()
    logger = logging.getLogger("algua")
    marked = [h for h in logger.handlers if getattr(h, "_algua_observability", False)]
    assert len(marked) == 1
    assert logger.level == logging.DEBUG
    get_logger("algua.test").info("ping", extra={"fields": {"k": 1}})
    out, err = capfd.readouterr()
    assert out == ""           # nothing on stdout
    assert json.loads(err.strip().splitlines()[-1])["msg"] == "ping"


def test_configure_logging_unknown_level_falls_back_to_info(monkeypatch):
    monkeypatch.setenv("ALGUA_LOG_LEVEL", "NOPE")
    configure_logging()
    assert logging.getLogger("algua").level == logging.INFO


def test_configure_logging_preserves_foreign_handler():
    logger = logging.getLogger("algua")
    foreign = logging.NullHandler()
    logger.addHandler(foreign)
    configure_logging()
    assert foreign in logger.handlers
    logger.removeHandler(foreign)


def test_cycle_counters_as_fields():
    c = CycleCounters()
    c.ticks += 2
    c.breaches += 1
    f = c.as_fields()
    assert f["ticks"] == 2 and f["breaches"] == 1 and f["flatten_failures"] == 0
```

- [ ] **Step 2: Run, verify import failure**: `cd /tmp/algua-346 && uv run pytest tests/test_observability.py -q` → FAIL (module missing).

- [ ] **Step 3: Implement `algua/observability/metrics.py`:**

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CycleCounters:
    """Golden-signal counts for one always-on loop cycle. Pure, single-threaded."""

    ticks: int = 0
    breaches: int = 0
    flatten_failures: int = 0
    reconcile_deferred: int = 0
    reconcile_halted: int = 0

    def as_fields(self) -> dict[str, int]:
        return {
            "ticks": self.ticks,
            "breaches": self.breaches,
            "flatten_failures": self.flatten_failures,
            "reconcile_deferred": self.reconcile_deferred,
            "reconcile_halted": self.reconcile_halted,
        }
```

- [ ] **Step 4: Implement `algua/observability/log.py`:**

```python
from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from uuid import uuid4

_ROOT = "algua"
_CID: ContextVar[str | None] = ContextVar("algua_correlation_id", default=None)
_CORE_KEYS = frozenset(
    {"ts", "level", "logger", "msg", "correlation_id", "exc_type", "exc_message", "stacktrace"}
)


def current_correlation_id() -> str | None:
    return _CID.get()


@contextmanager
def correlation_context(cid: str | None = None) -> Iterator[str]:
    """Bind a correlation id for the duration of the block (reset on exit, even on error)."""
    value = cid or uuid4().hex
    token = _CID.set(value)
    try:
        yield value
    finally:
        _CID.reset(token)


class JsonFormatter(logging.Formatter):
    """Render a LogRecord as ONE physical line of JSON on stderr.

    Caller structured data arrives via ``logger.info(msg, extra={"fields": {...}})``.
    Core keys always win over caller fields; a non-serializable field never drops the record.
    """

    def format(self, record: logging.LogRecord) -> str:
        try:
            fields = getattr(record, "fields", None)
            payload: dict[str, object] = dict(fields) if isinstance(fields, dict) else {}
            core: dict[str, object] = {
                "ts": datetime.now(UTC).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "msg": record.getMessage(),
            }
            cid = _CID.get()
            if cid is not None:
                core["correlation_id"] = cid
            if record.exc_info and record.exc_info[0] is not None:
                core["exc_type"] = record.exc_info[0].__name__
                core["exc_message"] = str(record.exc_info[1])
                core["stacktrace"] = self.formatException(record.exc_info)
            payload.update(core)  # core overwrites any caller collision
            return json.dumps(payload, default=str)
        except Exception as exc:  # noqa: BLE001 - never raise into logging machinery / never drop
            return json.dumps(
                {
                    "ts": datetime.now(UTC).isoformat(),
                    "level": getattr(record, "levelname", "ERROR"),
                    "msg": "log_format_error",
                    "format_error": str(exc),
                }
            )


def get_logger(name: str = _ROOT) -> logging.Logger:
    """Return a logger under the ``algua`` root (the one carrying the stderr handler)."""
    if name != _ROOT and not name.startswith(_ROOT + "."):
        name = f"{_ROOT}.{name}"
    return logging.getLogger(name)


def _resolve_level(raw: str | None) -> int:
    if raw:
        level = logging.getLevelName(raw.strip().upper())
        if isinstance(level, int):
            return level
    return logging.INFO


def configure_logging() -> None:
    """Idempotently wire the ``algua`` logger to a single JSON stderr handler.

    Re-reads ``ALGUA_LOG_LEVEL`` on EVERY call (so tests/env changes take effect); adds our
    marked handler only once; never removes foreign handlers; ``propagate=False`` keeps records
    off the root ``lastResort`` handler.
    """
    logger = logging.getLogger(_ROOT)
    logger.setLevel(_resolve_level(os.environ.get("ALGUA_LOG_LEVEL")))
    logger.propagate = False
    if not any(getattr(h, "_algua_observability", False) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(JsonFormatter())
        handler._algua_observability = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
```

- [ ] **Step 5: Implement `algua/observability/__init__.py`:**

```python
from __future__ import annotations

from algua.observability.log import (
    JsonFormatter,
    configure_logging,
    correlation_context,
    current_correlation_id,
    get_logger,
)
from algua.observability.metrics import CycleCounters

__all__ = [
    "CycleCounters",
    "JsonFormatter",
    "configure_logging",
    "correlation_context",
    "current_correlation_id",
    "get_logger",
]
```

- [ ] **Step 6: Run tests** → `uv run pytest tests/test_observability.py -q` PASS.

- [ ] **Step 7: Commit** `git add algua/observability tests/test_observability.py && git commit -m "feat(observability): stdlib JSON logger + correlation context + cycle counters #346"`

---

### Task 2: Import-linter leaf contracts

**Files:**
- Modify: `pyproject.toml` (the `[tool.importlinter]` contracts block)

- [ ] **Step 1:** Add a new `forbidden` contract making `algua.observability` a pure leaf (mirror the `algua.provenance` contract — forbid `algua.cli, algua.registry, algua.config, algua.calendar, algua.contracts, algua.features, algua.data, algua.backtest, algua.strategies, algua.research, algua.live, algua.execution, algua.risk, algua.audit, algua.knowledge, algua.portfolio, algua.tracking`).

- [ ] **Step 2:** Add `algua.observability` to the `forbidden_modules` list of EACH pure-leaf contract: `contracts`, `features`, `provenance` (the "provenance layer is pure" one), and `portfolio`.

- [ ] **Step 3:** Run `uv run lint-imports` → all contracts kept.

- [ ] **Step 4: Commit** `git add pyproject.toml && git commit -m "build(importlinter): observability is a pure leaf; pure layers may not import it #346"`

---

### Task 3: Wire logging into the always-on loop

**Files:**
- Modify: `algua/cli/main.py` (call `configure_logging()` at top of `main()`)
- Modify: `algua/cli/paper_cmd.py` (`run_all`, `_run_paper_strategy_tick`)
- Modify: `algua/cli/live_cmd.py` (`run_all`, `_run_strategy_tick`)
- Test: `tests/test_observability_wiring.py`

**Interfaces consumed:** Task 1 public surface from `algua.observability`.

**Wiring spec:**
- `cli/main.py:main()` — first line of body: `from algua.observability import configure_logging; configure_logging()`.
- `paper_cmd.run_all` / `live_cmd.run_all`:
  - `log = get_logger(__name__)`; `configure_logging()` at entry; open `with correlation_context():`; `counters = CycleCounters()` created immediately; wrap the body in `try: ... finally: log.info("golden_signals", extra={"fields": counters.as_fields()})`.
  - INFO `cycle_start` with `{"snapshot": snapshot, "lane": "paper"|"live"}`.
  - On the existing `venue_ingest_failed` branch (paper): `log.error("venue_ingest_failed", extra={"fields": {...}}, exc_info=True)`.
  - On reconcile halt: `counters.reconcile_halted += 1; log.error("reconcile_halt", extra={"fields": {"mismatches": ...}})`.
  - On reconcile not-clean defer: `counters.reconcile_deferred += 1; log.info("reconcile_deferred", ...)`.
  - In the per-strategy loop after each tick: `counters.ticks += 1`; if `out.get("ok") is False`: `counters.breaches += 1`.
  - Catch an UNEXPECTED `Exception` around the body to `log.error("cycle_failed", exc_info=True)` then re-raise (let `typer.Exit` pass through untouched — do NOT count it as cycle_failed).
- `_run_paper_strategy_tick` / `_run_strategy_tick`: in the `except RiskBreach` block `log.error("breach", extra={"fields": {"strategy": name, "tick_ts": str(tick_ts), "kind": exc.kind}}, exc_info=True)`; in the flatten-failure `except` block `log.error("flatten_failed", extra={"fields": {"strategy": name}}, exc_info=True)`; in `except TickHalted` `log.info("tick_halted", extra={"fields": {"strategy": name}})`.

- [ ] **Step 1: Write failing wiring tests** `tests/test_observability_wiring.py` — attach a capture handler to `logging.getLogger("algua")` (NOT caplog), run a `paper run-all` over a seeded breaching + a clean strategy via the existing test harness/fixtures (reuse patterns from `tests/test_paper_run_all.py`), and assert: (a) a `golden_signals` record is emitted with `ticks>=1`; (b) a breach cycle emits an ERROR `breach` record AND still emits `golden_signals` with `breaches>=1`; (c) nothing is written to stdout by the logger (capfd `out == ""` apart from the command's own JSON envelope — assert no JSON log lines on stdout). Mirror existing run-all test setup; keep one clean-cycle and one breach-cycle case.

- [ ] **Step 2: Run** → FAIL (no log records yet).

- [ ] **Step 3: Implement** the wiring edits above in `main.py`, `paper_cmd.py`, `live_cmd.py`. Keep all logging additive — do not alter existing `emit`/`audit_append`/control flow.

- [ ] **Step 4: Run** `uv run pytest tests/test_observability_wiring.py -q` PASS.

- [ ] **Step 5: Full gate** `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`.

- [ ] **Step 6: Commit** `git add algua/cli/main.py algua/cli/paper_cmd.py algua/cli/live_cmd.py tests/test_observability_wiring.py && git commit -m "feat(observability): wire JSON logging into paper/live run-all loop #346"`

---

## Self-Review
- **Spec coverage:** module surface → Task 1; import-linter → Task 2; production wiring (main + both run-all + tick helpers) → Task 3; testing spread across Tasks 1 & 3. Golden-signals-in-finally, exc_info rendering, idempotency, bad-level fallback, fields-collision, format fallback all have explicit tests in Task 1.
- **Placeholders:** none — full code for the leaf; Task 3 wiring is per-call-site precise against named existing branches.
- **Type consistency:** `CycleCounters`, `configure_logging`, `correlation_context`, `get_logger`, `JsonFormatter` names identical across tasks and `__init__` re-exports.
