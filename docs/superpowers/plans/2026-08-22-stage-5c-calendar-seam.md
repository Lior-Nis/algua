# Stage 5c — Calendar Seam Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route every operational `MarketCalendar()` construction through one settings-honouring factory, and delete the one genuine Protocol duplicate. This closes a live defect: today `settings.exchange` is read by exactly one site — a `doctor` probe that does no work with it — while the go-live certificate, forward gate, fleet-health gate, operator session gate and live mark-staleness all silently run XNYS regardless of configuration.

**Architecture:** `algua/calendar/factory.py` reads `settings.exchange` and returns a `MarketCalendar`, mirroring Stage 5b's `execution/broker_factory.py` — a separate module in the same package as the concrete class, so `market_calendar.py` stays a config-free leaf (which matters: `backtest/_sample.py`, inside the pure backtest lane, imports it). Nine of the ten construction sites are already pure injection points that hand the calendar straight to a domain function, so this is a composition-root change, not an architecture change.

**Tech Stack:** Python 3.12, uv, pytest, ruff, mypy, import-linter.

**Spec:** `docs/superpowers/specs/2026-08-18-system-simplification-design.md` §5 item 2.

**Ground truth:** a research pass against `main`@`a336aaf` that regenerated every count untruncated and **measured the XNYS/XLON divergence by execution**. It corrected several prior claims and found two traps this plan encodes as hard constraints.

### What this stage actually fixes, quantified

`Settings.exchange` defaults to `"XNYS"`; `MarketCalendar.__init__` defaults to `code="XNYS"` — **the same string**. So at default configuration, converting the eight bare sites is a **provable behavioural no-op**. Divergence appears only when an operator sets `ALGUA_EXCHANGE`, and there today's behaviour is a silent lie: `doctor` reports the configured exchange while every operational path ignores it.

Measured, 2024: XNYS **252** sessions vs XLON **254**; symmetric difference **10 days**; `sessions_between_instants(Jul 3 → Jul 8)` XNYS **2** vs XLON **3**; `session_close(2024-07-05)` XNYS **20:00Z** vs XLON **15:30Z**. A one-session difference is exactly the granularity of `fleet health`'s staleness verdict, the forward certificate's *≤10 sessions old* bound, and `paper promote`'s *≤5 sessions stale* bound.

**Scope statement:** this stage makes the configured exchange *consistent*. It does **not** validate non-XNYS operation — the `1d = UTC-midnight` rail stays baked (spec §5 item 2's own non-goal), so running a non-XNYS exchange end-to-end remains unvalidated. Say this in the PR.

### Decisions (recorded for a reviewer to check, not re-derive)

1. **Two Protocols, not one union.** `execution/fleet_health.py:76` and `operator/loop_health.py:55` define **byte-identical** one-method `_Calendar` Protocols (`sessions_between_instants`) — that is the genuine duplicate, and it is deduped into `algua/contracts/types.py`. `registry/forward_promotion.py:59`'s `SessionCalendar` has a **4-method set with zero overlap** and is left alone. The spec asks for one `TradingCalendar`; a 5-method union would force every one-method fake in `tests/test_fleet_health.py` to grow four unused methods — a net complexity increase inside a simplification program — and would cut against `contracts/types.py`'s own convention, which already decomposes `Broker` into ten narrow role-Protocols for exactly this reason.
2. **Protocol home is `algua/contracts/types.py`.** The deduped pair has two consumers in two packages, matching the narrow-role precedent. Legal: a Protocol defined there imports nothing, so the "contracts imports no other algua module" contract is unaffected.
3. **Factory home is `algua/calendar/factory.py`,** not a function appended to `market_calendar.py`. Keeps the concrete class a config-free leaf. Legal: the import-linter contract *"calendar stays independent of cli and registry"* forbids `algua.cli`/`algua.registry`; `algua.config` is **not** forbidden.
4. **`backtest/_sample.py:44` stays hardcoded `MarketCalendar("XNYS")` — explicit non-goal.** It feeds `SyntheticProvider.reproducible = True`, whose comment pins the #205 guarantee: *"the OOS bars are identical on a re-run, so a burned holdout is reproducible w/o a snapshot."* Honouring `settings.exchange` there would make the generated bar index config-dependent — different sessions, different bars, a burned holdout that no longer reproduces — turning an environment variable into a silent breach of #205. **Task 2 adds an in-code comment saying so**, because otherwise a future tidy-sweep will "finish the job".

## Global Constraints

- Quality gate on EVERY task before commit: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`. All four must pass.
- **When running the full suite, pass `timeout: 600000` explicitly to the Bash tool.** Without that parameter the 120s default fires and the harness **auto-backgrounds** the command — this has stalled four agents on this program, every time for the same reason.
- **Regenerate every count untruncated at the moment you assert it** — no `| head`/`| tail` on an enumeration you are counting. Across the last two stages, nine separate figures in plans and specs were wrong; one would have caused a net regression. If a number here disagrees with what you measure, trust your measurement and say so.
- **Behaviour is unchanged at default configuration** (both defaults are `"XNYS"`), and deliberately *changed* for a configured non-default exchange. That is the point of the stage — but it means the "prove byte-identical" technique from prior stages does **not** apply. Prove the no-op at default, and prove the intended change under `ALGUA_EXCHANGE`.
- **`algua/calendar/factory.py` must not construct at import time** — no module-level `get_calendar()` call. `MarketCalendar._get_calendar` is already `@cache`d by exchange code, so the factory needs no caching of its own.
- **CODEOWNERS-protected files this stage touches: `algua/registry/transitions.py`, `algua/registry/forward_promotion.py`, `algua/cli/paper_cmd.py`.** Every one is on the paper→live wall. Expected; note it in the PR so human review is anticipated.
- `git add` scoped to named files — never `git add -A`.
- Commits end with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Known hazard: some test writes a demo strategy file into `algua/strategies/momentum/`. If `git status` shows an untracked file there, delete it before staging.

---

### Task 1: Add the calendar factory and the deduped Protocol

**Files:**
- Create: `algua/calendar/factory.py`, `tests/test_calendar_factory.py`
- Modify: `algua/contracts/types.py`

**No call site changes in this task** — the factory and Protocol land proven in isolation before anything depends on them.

- [ ] **Step 1: Read the pieces**

Read `algua/calendar/market_calendar.py` in full (102 lines, 11 public methods, `_get_calendar` `@cache`d by code). Read `algua/execution/broker_factory.py` — Stage 5b's factory is the shape to mirror, including its fail-closed discipline and its module docstring recording a deliberate spec deviation. Read the two `_Calendar` Protocols (`algua/execution/fleet_health.py:76`, `algua/operator/loop_health.py:55`) and confirm for yourself they are byte-identical; read `algua/registry/forward_promotion.py:59`'s `SessionCalendar` and confirm it shares **zero** methods with them.

- [ ] **Step 2: Add the shared Protocol to `algua/contracts/types.py`**

Add a one-method Protocol next to the existing narrow role-Protocols:

```python
class SessionSpanCalendar(Protocol):
    """Counts completed exchange sessions between two instants.

    The narrow slice of a trading calendar that liveness/staleness checks need — deliberately one
    method, matching this module's convention of narrow role Protocols (see the Broker split above)
    rather than one fat calendar interface. ``registry/forward_promotion.SessionCalendar`` is a
    different, non-overlapping slice and stays where it is.
    """

    def sessions_between_instants(self, a: datetime, b: datetime) -> int: ...
```

Match the surrounding style and the exact signature of the two definitions it replaces — verify against them rather than copying from this plan. No new import is needed — `types.py:5` already has `from datetime import datetime` (verified).

- [ ] **Step 3: Delete the two duplicates and re-point their consumers**

In `algua/execution/fleet_health.py` and `algua/operator/loop_health.py`, delete the local `_Calendar` definition and import `SessionSpanCalendar` from `algua.contracts.types`, updating the annotations that referenced `_Calendar`.

**Neither Protocol is `@runtime_checkable`, and there is no `isinstance(..., _Calendar)` anywhere** — verify that yourself before relying on it. It means this move cannot change runtime behaviour, and structural test fakes are unaffected by where the Protocol lives.

- [ ] **Step 4: Write `algua/calendar/factory.py`**

```python
"""Settings-honouring construction of the trading calendar (spec §5 item 2).

One place reads ``settings.exchange`` and turns it into a ``MarketCalendar``, so every operational
path agrees on which exchange it is running. Before this seam, ``settings.exchange`` was read by
exactly one site -- a ``doctor`` probe that does no work with it -- while the go-live certificate,
forward gate, fleet-health gate, operator session gate and live mark-staleness all ran XNYS
regardless of configuration.

Lives here rather than in ``market_calendar.py`` so that module stays a config-free leaf: the pure
backtest lane imports it (``backtest/_sample.py``) and must not acquire a settings dependency.

NOT a registry: unlike the broker and tracker seams there is one implementation and the selector is
a plain exchange code, so a name->factory table would add indirection without extensibility.
``MarketCalendar._get_calendar`` is already ``@cache``d per code, so this adds no caching either.
"""

from __future__ import annotations

from algua.calendar.market_calendar import MarketCalendar
from algua.config.settings import get_settings


def get_calendar(code: str | None = None) -> MarketCalendar:
    """The configured trading calendar. ``code`` overrides ``settings.exchange`` (for tests).

    Reads settings per call, never at import -- an import-time read would bind the value before a
    test's ``monkeypatch.setenv`` could take effect, which is exactly how Stage 5b silently disarmed
    a go-live guard.
    """
    return MarketCalendar(code if code is not None else get_settings().exchange)
```

That per-call read is a **hard requirement**, not a style preference — see the Task 3 hazard.

- [ ] **Step 5: Write the factory's tests**

In `tests/test_calendar_factory.py`:
- default config yields an XNYS calendar;
- `ALGUA_EXCHANGE=XLON` (via `monkeypatch.setenv`) yields an XLON calendar — proving the setting is honoured;
- an explicit `code` argument overrides the setting;
- **settings are read per call, not at import**: set `ALGUA_EXCHANGE` *after* importing the factory and confirm the next call reflects it. This is the regression guard for the Stage 5b failure mode;
- an invalid exchange code fails rather than silently falling back to XNYS (check what `MarketCalendar` does with a bad code and assert the actual behaviour — do not assume).

- [ ] **Step 6: Full quality gate, then commit** (`timeout: 600000`). Test count rises by the tests added; no call site has changed yet.

---

### Task 2: Convert the eight operational sites

**Files:**
- Modify: `algua/live/live_loop.py`, `algua/cli/fleet_cmd.py`, `algua/cli/ops_cmd.py`, `algua/cli/paper_cmd.py`, `algua/cli/operator_cmd.py`, `algua/registry/transitions.py`, `algua/cli/app.py`, `algua/backtest/_sample.py` (comment only), `tests/test_cli_paper.py`

**The hazard that defines this task — read before touching anything.**

`tests/test_cli_paper.py:1418` does `monkeypatch.setattr("algua.cli.paper_cmd.MarketCalendar", FakeCalendar)`, and `paper_cmd.py:14` imports `MarketCalendar` at module level for use at **two** sites (`:329`, `:1424`).

- Convert **only one** site → the module-level import survives → the patch still resolves, binds a now-unused name, and **silently stops covering the converted site**. Green, and wrong.
- Convert **both** and delete the module-level import → the patch raises `AttributeError` → **loud**, and you re-point it at the factory.

**This is the same failure the program shipped one stage ago** (`test_forward_certificate.py:422`, Stage 5b), so it is a plan-level constraint: **both `paper_cmd.py` sites are converted in this one task, the module-level import is deleted, and the `AttributeError` is the expected signal.** Do not split this across tasks.

- [ ] **Step 1: Convert the eight bare sites**

Replace `MarketCalendar()` with `get_calendar()` at: `live/live_loop.py:88`, `cli/fleet_cmd.py:47`, `cli/fleet_cmd.py:103`, `cli/ops_cmd.py:44`, `cli/paper_cmd.py:329`, `cli/paper_cmd.py:1424`, `cli/operator_cmd.py:412`, `registry/transitions.py:240`. Re-verify each line number against the live file first.

In `registry/transitions.py`, the `MarketCalendar` import is **lazy, inside a function** — keep it lazy, importing `get_calendar` in the same place. Stage 5b established why: it keeps heavy layers off the registry's import graph, and hoisting is legal but spends that property.

Delete each module-level `MarketCalendar` import that becomes unused. `ruff` will flag any you miss.

- [ ] **Step 2: Convert the `doctor` probe**

`cli/app.py:71`'s `_calendar_detail` already passes `settings.exchange` explicitly. Route it through `get_calendar()` so there is one path, and confirm the probe still reports the configured exchange. This site is where the old lie was visible; after this stage it tells the truth because everything else now agrees with it.

- [ ] **Step 3: Re-point the test patch**

`tests/test_cli_paper.py:1418` will fail with `AttributeError` once the import is gone. Re-point it at the factory — patch `algua.cli.paper_cmd.get_calendar` to return the existing `FakeCalendar`. Keep the fake and every assertion unchanged; only the patch target moves.

**If it does NOT fail after you delete the import, stop and investigate** — that means the name is still bound somewhere and the patch may be silently inert, which is the very thing this task exists to avoid.

- [ ] **Step 4: Leave `_sample.py` alone, and say why in the code**

Do **not** convert `backtest/_sample.py:44`. Add a comment at that line recording the reason:

```python
    # NOT settings-derived on purpose: these session dates feed SyntheticProvider.reproducible
    # (#205) -- a burned holdout must reproduce on a re-run. Honouring settings.exchange here would
    # make the generated bar index depend on an environment variable, so a holdout burned under one
    # ALGUA_EXCHANGE would not reproduce under another. Keep the literal.
```

- [ ] **Step 5: Prove both halves of the behaviour claim**

This stage cannot use "prove byte-identical", so prove the two halves separately:

- **No-op at default:** run the suites owning these paths (`test_fleet_health.py`, `test_cli_paper.py`, `test_cli_live.py`, `test_forward_certificate.py`, `test_operator_schedule.py`, `test_forward_promotion.py`) and confirm they pass with **no assertion changes** — only the one patch re-point from Step 3.
- **The intended change under configuration:** with `ALGUA_EXCHANGE=XLON`, show that a converted site now builds an XLON calendar where it previously built XNYS. Pick one operational site and demonstrate it directly. Capture the output — this is the evidence that the defect is actually closed, and the only place in the stage where behaviour is *supposed* to differ.

- [ ] **Step 6: Full quality gate, then commit** (`timeout: 600000`). Test count unchanged.

---

### Task 3: Close-out verification

**Files:** none expected (verification only; fix anything found)

- [ ] **Step 1: No bare construction survives outside the allowed sites**

```bash
grep -rn "MarketCalendar(" algua/ --include='*.py'
```
Expected hits: `calendar/factory.py` (the one construction), `backtest/_sample.py:44` (the deliberate literal), and the comment in `market_calendar.py`. **Anything else is a missed site** — regenerate this list untruncated rather than trusting the plan's.

- [ ] **Step 2: The setting is honoured everywhere it should be**

```bash
uv run python -c "
import os
os.environ['ALGUA_EXCHANGE'] = 'XLON'
from algua.calendar.factory import get_calendar
c = get_calendar()
assert c.code == 'XLON', c.code
print('factory honours setting:', c.code)
"
```
(`MarketCalendar.__init__` stores the code as `self.code` — verified at `market_calendar.py:21`.) Then confirm `algua doctor` still reports the configured exchange.

- [ ] **Step 3: The Protocol dedupe is complete and inert**

Confirm no `class _Calendar` remains, both consumers import `SessionSpanCalendar` from `algua.contracts.types`, `SessionCalendar` in `forward_promotion.py` is untouched, and `lint-imports` still passes (contracts must import nothing from algua).

- [ ] **Step 4: The `paper_cmd` patch is genuinely live**

The point of Task 2's constraint. Prove the re-pointed patch actually covers both sites: temporarily break `get_calendar` (e.g. make it raise), confirm the paper tests that rely on the fake **fail**, then restore. Clear `__pycache__` or set `PYTHONDONTWRITEBYTECODE=1` around it — a same-second restore can otherwise reuse mutated bytecode and make the result lie in either direction.

- [ ] **Step 5: Full quality gate + CLI smoke** (`timeout: 600000`): `uv run pytest -q`, ruff, mypy, lint-imports, then `uv run algua doctor` and `uv run algua fleet status`.

- [ ] **Step 6: Commit any fixes.** If nothing needed fixing, make no commit — expected, and consistent with every prior close-out in this program.
