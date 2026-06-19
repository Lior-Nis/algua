# Issue #165 — Lift domain logic out of CLI command modules

**Status:** design approved
**Date:** 2026-06-19
**Issue:** #165 — "Domain logic embedded in CLI commands; live_cmd imports gating/breach helpers from paper_cmd" (P2, rank 6 of repo review 2026-06-10)

## Problem

Business rules live in `algua/cli/*` command modules, and command modules import each other:

- `algua/cli/paper_cmd.py` — `_load_gated_strategy` (stage validation + global-halt + kill-switch), `_breach_payload`/`_trip` (kill-switch mutation + breach payload), `_tick_clock` (evidence-tick clock stamp).
- `algua/cli/live_cmd.py:12` — imports `_breach_payload, _tick_clock, _trip` **from `paper_cmd`** (the cli→cli smell); `_run_strategy_tick` re-implements the load+tradability preamble inline.
- `algua/cli/backtest_cmd.py` — `_record_search_breadth` writes registry state from the CLI layer.

Two further pre-existing cli→cli imports surfaced during design:

- `algua/cli/strategy_cmd.py:12` — imports `_kb_metadata` **from `registry_cmd`** (same smell: a shared domain helper).
- `algua/cli/idea_cmd.py:10` — imports `research_app` **from `research_cmd`** to mount `idea_app` as a subcommand (legitimate Typer command-tree composition, NOT a domain-logic smell — but a blanket independence contract would still flag it).

### Why it matters

The CLI is meant to be a thin JSON seam. Any non-CLI consumer (the autonomous loop driving modules directly, a webhook, a second front-end) would have to duplicate these rules or import from `cli/*`. Paper/live gating drift is a live risk: a gate added to one lane can silently miss the other.

## Decisions (settled during brainstorming)

1. **Unification scope: minimal.** Live's gating genuinely differs from paper's (it gates on LIVE-stage via the `run-all` filter, checks kill-switch / global-halt / authorization inside `should_halt` rather than at load, and requires an allocation). The two lanes share only the *load + tradability* preamble; paper keeps its distinct stage+halt+kill gate, live keeps its distinct gate. No behavior change to live.
2. **Add the import-linter contract: yes**, broadest form — full cli→cli independence over the command modules, achieved by fixing all three violators so the contract passes with **zero exceptions**.

## Target homes

Each symbol moves to the layer that already owns its dependencies (verified: `registry` already imports `strategies.loader`, `strategies.base`, `risk.global_halt`, `risk.kill_switch`; `execution` imports none of those and `registry` already depends on `execution` lazily, so a loader in `execution` would create a cycle).

| Current (in `cli/`) | New home | New symbol |
|---|---|---|
| `paper_cmd._load_gated_strategy` | `algua/registry/gating.py` (NEW) | `load_gated_strategy(conn, name, command) -> tuple[LoadedStrategy, StrategyRecord]` |
| inline `load_strategy` + `assert_tradable_*` (paper gate + live `_run_strategy_tick`) | `algua/strategies/loader.py` | `load_tradable_strategy(name) -> LoadedStrategy` |
| `paper_cmd._trip` | `algua/risk/breach.py` (NEW) | `trip_for_breach(conn, name, exc: RiskBreach) -> None` |
| `paper_cmd._breach_payload` | `algua/cli/_common.py` (CLI infra) | `breach_payload(error: str, **extra: object) -> dict` |
| `paper_cmd._tick_clock` | `algua/execution/tick_clock.py` (NEW) | `tick_clock(clock: Callable[[], str]) -> tuple[str, str]` |
| `backtest_cmd._record_search_breadth` | `algua/registry/search_breadth.py` (NEW) | `record_search_breadth(repo, name, result) -> dict[str, int]` |
| `registry_cmd._kb_metadata` | `algua/registry/repository.py` (beside `StrategyRecord`) | `kb_metadata(rec: StrategyRecord) -> dict` |

### Notes on each move

- **`load_tradable_strategy`** = `load_strategy(name)` + `assert_tradable_without_fundamentals` + `assert_tradable_without_news`. The tradability asserts already live in `algua.strategies.base`, so the strategies layer already owns this concept; the combined helper belongs next to `load_strategy` in `algua.strategies.loader`. Both lanes import it (`live_cmd` already imports `load_strategy` from there).
- **`load_gated_strategy`** keeps paper's exact gate, hardcoded (NOT parameterized by stage — YAGNI, since live does not use it): load via `load_tradable_strategy`, then `rec = SqliteStrategyRepository(conn).get(name)`, stage ∈ {`PAPER`, `FORWARD_TESTED`} else `ValueError`, `global_halt.is_engaged` else `ValueError`, `kill_switch.is_tripped` else `ValueError`. Returns `(strategy, rec)`. Lives in `registry` because only `registry` already imports all of `strategies` + `risk` + the repo. The `command` parameter is a caller-supplied label that only colours the stage-error text (e.g. `"trade-tick requires 'paper'…"`) — document it as an error-message label, not a domain concept; keep it (a generic message would lose which command was refused).
- **`trip_for_breach` (in `risk/breach.py`)** preserves the current **mutate-then-audit** ordering: `kill_switch.trip()` first, then `audit_append()`. This is INTENTIONAL and must NOT be "improved" to audit-first — for a *trip*, fail-safe means the switch is persisted (halted) even if the audit write then fails; audit-first would leave the worse state (audited-as-tripped but switch unpersisted). (This is the opposite of `resume`, which audits-before-clearing because there fail-safe = stay-halted-with-trail.) `risk/breach.py` adds a new `risk → audit` edge; `algua.audit.log` is pure stdlib (sqlite3 + datetime), so no cycle, and no contract forbids `risk → audit`.
- **`breach_payload`** is a pure JSON-envelope constructor (`{"ok": false, "kill_switch": "tripped", "error": …, **extra}`) with ZERO `risk`/`audit` dependency — it is CLI presentation, not a risk operation. It goes to `algua/cli/_common.py` (beside `ok`), which is excluded from the independence contract, so both `paper_cmd` and `live_cmd` may import it.
- **`tick_clock`** catches `(BrokerError, ValueError, TypeError)`; `BrokerError` comes from `algua.execution.alpaca_broker`. A new leaf module `algua/execution/tick_clock.py` imports it + `pandas` + `datetime` + `from collections.abc import Callable` — no cycle (alpaca_broker does not import the new module). Its docstring must note the `alpaca_broker` coupling (it is not broker-agnostic today); extracting `BrokerError` to a shared exceptions leaf is deferred (only one broker exists — YAGNI).
- **`record_search_breadth`** takes a `repo` (and `result`) rather than opening its own `registry_conn` — the registry layer must not import `cli._common`. It also needs `import json` (for `json.dumps(result.grid, sort_keys=True)`). The caller (`backtest_cmd.sweep_cmd`) wraps it in `with registry_conn() as conn:`; its docstring should note the transaction scope is now caller-owned. The `SweepResult` annotation is imported under `TYPE_CHECKING` (`from __future__ import annotations` makes annotations strings), so no runtime `registry → backtest` import.
- **`kb_metadata`** takes a `StrategyRecord` and returns the registry-owned kb frontmatter dict. It cannot live in `algua.knowledge` (the knowledge layer is contractually barred from importing `registry`/`StrategyRecord`), so it lives beside `StrategyRecord` in `algua.registry.repository`. Both `registry_cmd` and `strategy_cmd` import it from there.

## Composition root (fix the idea→research mount)

`idea_cmd.py` currently does `from algua.cli.research_cmd import research_app` then `research_app.add_typer(idea_app, name="idea")`. Move the mount to the composition root:

- `idea_cmd.py`: define `idea_app` but do NOT import `research_app` and do NOT mount.
- `main.py` (already imports every `*_cmd` module to register subcommands): **after the `from algua.cli import (…)` block** (so `research_cmd` and `idea_cmd` are defined), add `research_cmd.research_app.add_typer(idea_cmd.idea_app, name="idea")` with a comment explaining it must follow the imports and precede `get_command(app)`.

Typer builds the command tree lazily at `get_command(app)` (called in `main()`), so mounting in `main.py` before that call preserves the exact `algua research idea …` command path. `research_cmd`'s own `app.add_typer(research_app, name="research")` self-mount is left as-is.

## Call-site rewrites

- **`paper_cmd.py`**: delete the moved defs; import `load_gated_strategy` from `registry.gating`, `trip_for_breach` from `risk.breach`, `breach_payload` from `cli._common` (beside the existing `ok` import), `tick_clock` from `execution.tick_clock`. Replace `_load_gated_strategy`→`load_gated_strategy`, `_trip`→`trip_for_breach`, `_breach_payload`→`breach_payload`, `_tick_clock`→`tick_clock` at every call site. `_live_strategy_flat`, `_strategy_held_symbols`, and the broker-construction helpers are paper-only and stay in `paper_cmd`. **`resume` and `flatten` intentionally keep their direct `load_strategy(name)` calls** (they don't need — and must not impose — the tradability gate); do NOT convert them to `load_tradable_strategy`.
- **`live_cmd.py`**: delete `from algua.cli.paper_cmd import _breach_payload, _tick_clock, _trip`; import `trip_for_breach` from `risk.breach`, `breach_payload` from `cli._common`, `tick_clock` from `execution.tick_clock`, and `load_tradable_strategy` from `strategies.loader`; replace the inline `load_strategy(name)` + two `assert_tradable_*` calls in `_run_strategy_tick` with `strategy = load_tradable_strategy(name)`.
- **`backtest_cmd.py`**: delete `_record_search_breadth`; at the call site wrap in `with registry_conn() as conn: recorded = record_search_breadth(SqliteStrategyRepository(conn), name, result)`.
- **`registry_cmd.py`**: delete `_kb_metadata`; import `kb_metadata` from `registry.repository`; update its one call site.
- **`strategy_cmd.py`**: replace `from algua.cli.registry_cmd import _kb_metadata` with `from algua.registry.repository import kb_metadata`; update its three call sites.
- **`idea_cmd.py`**: drop the `research_app` import and the mount line.

## Import-linter contract

Add an `independence` contract over the nine command modules:

```toml
[[tool.importlinter.contracts]]
name = "cli command modules are independent of one another (no cli->cli sibling imports)"
type = "independence"
modules = [
    "algua.cli.backtest_cmd",
    "algua.cli.data_cmd",
    "algua.cli.factor_cmd",
    "algua.cli.idea_cmd",
    "algua.cli.live_cmd",
    "algua.cli.paper_cmd",
    "algua.cli.registry_cmd",
    "algua.cli.research_cmd",
    "algua.cli.strategy_cmd",
]
```

Shared infrastructure (`algua.cli.app`, `algua.cli._common`, `algua.cli.errors`, `algua.cli.main`) is NOT in the list, so command modules may still import it. After the three fixes the contract passes with zero `ignore_imports`.

## Behavior preservation & verification

This is a pure relocation — no behavior changes. No production code or test imports the moved private symbols by name (only a comment reference in `tests/test_cli_paper.py:57`); the CLI tests drive everything through the JSON seam.

Quality gate at each commit: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`.

Targeted coverage to confirm green: `tests/test_cli_paper.py`, `tests/test_cli_live.py`, `tests/test_cli_registry.py`, `tests/test_cli_track.py`, `tests/test_cli_idea.py` (covers the `algua research idea …` path after the composition-root move), plus `lint-imports` for the new independence contract.

## Out of scope

- No change to live's gating semantics (per the minimal-unification decision).
- No broader composition-root refactor of the other `*_cmd` self-mounts onto `app` — only the idea→research edge is hoisted.
- No new tests beyond confirming the existing suite stays green (the change adds no new behavior to test).
