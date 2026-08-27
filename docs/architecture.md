# Algua architecture — the one-page map

**Read this before adding anything.** It tells you which package owns what, and where the
extension points are. If you find yourself editing a core file to add a provider, broker, calendar,
tracker or command, stop — there is almost certainly a registration seam for it below.

The boundaries here are **enforced, not advisory**: 28 import-linter contracts
(`uv run lint-imports`) and a per-module size ratchet (`tests/test_module_size_ratchet.py`). If your
change fights a boundary, the boundary is usually right.

---

## The shape

Algua is a lifecycle: an idea becomes a backtest, a backtest becomes a gated candidate, a candidate
paper-trades, and only a human with a signed challenge puts it live. The packages follow that arc.

```
                 contracts ── the vocabulary (Protocols, lifecycle stages). Imports NOTHING.
                     │
    ┌────────────────┼──────────────────┬───────────────────┐
    │                │                  │                   │
  data          features/portfolio   backtest            calendar
  (bars,        (signals,            (simulate + score)  (sessions)
   universes,    construction)          │
   snapshots)         │                 │
    └────────────────┴─────────────────┘
                     │
                 research ── the statistical gate (does this edge survive scrutiny?)
                     │
                 registry ── the lifecycle authority: stages, gates, allocations, approvals
                     │
    ┌────────────────┴──────────────────┐
    │                                   │
  execution                          operator
  (brokers, orders, ledgers)         (the autonomous loop: schedule, merge-back)
    │                                   │
   live ── the tick engine both paper and live lanes run on
    │
   cli ── the ONLY way in. Every command emits JSON on stdout.
```

**The one-way rule:** `cli` composes everything; nothing composes `cli`. Domain packages never
import `algua.cli`. There is exactly one exception in the tree — `registry/human_actor.py` prints a
signing challenge — and it is tracked as debt in **#592**, deliberately left uncontracted so it
stays visible rather than being blessed by an exemption.

---

## What each package owns

| package | owns | do NOT put here |
|---|---|---|
| `contracts` | Protocols, lifecycle stages/transitions, the bar schema. Imports nothing from algua. | anything with I/O or a dependency |
| `primitives` | stdlib-only leaves: `flock`, `atomic_io`, `retry`, `timeparse`. Anything may import it. | anything importing algua |
| `config` | `Settings` (env-driven) and `get_settings()` | reading settings at import time — see the calendar seam |
| `data` | providers, importers, snapshots, PIT universes, the immutable store | decision logic; `data` is the **hindsight lane** and is walled off from decision lanes |
| `features` / `portfolio` | signal computation; weight construction | I/O, registry, cli |
| `backtest` | `simulate`/`run`, PIT masking (`pit_view`), the fast/canonical dual path + parity guard (`decision_path`), windowing (`walkforward`) | registry access, tracking, live/execution |
| `calendar` | `MarketCalendar` + the settings-honouring `get_calendar()` factory | cli or registry imports |
| `research` | the statistical gate: DSR, breadth deflation, regime robustness, forward gates | registry imports (`research` never imports `registry`) |
| `registry` | **the lifecycle authority**: stages, transitions, gate tokens, allocations, approvals, families, the SQLite schema | live-lane imports |
| `execution` | broker adapters, order state, ledgers, sizing, flatten/drain | cli imports |
| `live` | the shared tick engine (`run_tick`), book exposure | cli imports |
| `risk` | limits, breaches, kill-switch, global halt, peaks, book breaker | cli or live-lane imports |
| `operator` | the autonomous loop: session gating, merge-back saga, loop health | cli imports |
| `evaluation` | shared task bodies (`backtest_run`, `sweep_run`) + input resolution, importable by BOTH cli and registry | cli imports |
| `knowledge` | the Obsidian vault sync | imports of cli/registry/backtest/live/execution |
| `cli` | typer commands, the JSON envelope, flag resolution | domain logic — extract it |

---

## How to add things

### A data provider
`algua/data/providers/` — implement the provider, then register it:
```python
register_provider("myvendor", lambda settings: MyProvider(...))
```
`get_provider(name, settings)` resolves it. The built-ins register themselves the same way, so
your factory is not a second-class citizen — see the tracker section below for the one
unavoidable two-line cost (an import so your registration actually runs).

### A bulk importer
`algua/data/importers/` — same shape: `register_importer("myvendor", factory)`.

### A broker
`algua/execution/broker_factory.py` — add a `BrokerKind` member and register a `BrokerSpec`:
```python
register_broker(BrokerKind.MY_VENUE, BrokerSpec(
    key_field="my_api_key", secret_field="my_api_secret", url_field="my_url",
    construct=MyBroker, missing_credentials="...",
))
```
The registry is **closed and enum-keyed on purpose**: a config flag able to swap paper for live at
runtime would be a hazard, not a feature. Declare the host allowlist (`_ALLOWED_HOSTS`) on the
adapter — the base class enforces https + allowlisted host.

### An experiment tracker
`algua/tracking/` — implement the `ExperimentTracker` Protocol in **its own module**, then
`register_tracker("mybackend", MyTracker)`. `tracking_backend` in settings selects it.
Return `TRACKING_SKIPPED` from a no-op backend so the JSON reports "skipped", not a null run id.

**Two lines land in `tracking/factory.py`** — an import and the `register_tracker` call. That is the
whole core-file cost, and it is a *declaration that the backend exists*, not a change to any logic:
Python needs something to import your module before its registration can run. (Entry points would
remove even that, and would be over-engineering while every backend ships in-tree.) The same shape
applies to providers, importers and brokers.

**Expect the size ratchet to push you into a new file.** `tests/test_module_size_ratchet.py` pins
every module ≥300 lines at its current size, and several are pinned at *exactly* their length — so
appending a class to an existing module often fails the gate. That is the ratchet working: the new
thing belongs in its own module. Check `wc -l` before you plan to extend a large file.

### A trading calendar
There is one implementation and one selector (`ALGUA_EXCHANGE`), so there is **no registry** — just
`get_calendar()`, which reads settings **per call, never at import**. An import-time read binds the
value before a test's `monkeypatch.setenv` can take effect; that is how a go-live guard was once
silently disarmed.

### A strategy
`algua/strategies/<category>/<name>.py` — see the `author-a-strategy` skill. A strategy declares
`CONFIG`, a `signal()` and a named construction policy. It never touches `algua.data` directly.

### A CLI command
`algua/cli/<area>_cmd.py`. **Command modules may not import each other** — that contract is real and
has zero escapes. If two commands need the same body, the body belongs in a domain package
(`evaluation/`, `registry/`, `operator/`) and both import it from there. That is what
`evaluation/sweep_run.py` and `registry/promote_run.py` are.

---

## The walls (why a change might be refused)

- **PIT / anti-look-ahead** — `backtest/pit_view.py` masks universe, fundamentals and news as-of each
  decision instant; `backtest/engine.py` shifts decisions `t → t+1`. `decision_path.py`'s parity
  guard is what licenses the fast vectorised path: it proves the fast path agrees with the canonical
  loop, so weakening it silently licenses wrong numbers.
- **Single-use holdout (#192)** — `backtest/grid.py`'s index is the date-index truth that
  `holdout_window` reproduces; a holdout is burned on peek and cannot be re-read.
- **The paper→live wall** — an agent may drive the lifecycle up to `forward_tested` and **never**
  put a strategy live. `forward_tested → live` needs a verified human signature plus a fresh forward
  certificate.
- **Lane parity** — paper is the rehearsal for live, so both lanes must enforce the same invariants.
  `tests/test_lane_parity.py` asserts it structurally, because a fix reaching one lane and not the
  other is silent (that is exactly what happened in #559/#601).
- **CODEOWNERS is executable** — `operator/diff_policy.py` derives the autonomous merge-back's
  denylist from it, so a module missing from CODEOWNERS is one an agent can merge unreviewed.
  `tests/test_repo_hygiene.py` pins the integrity-critical set.

---

## Before you commit

```
uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports
```

Touching `web/`? Also `uv run --project web pytest web/backend/tests -q` and
`cd web/frontend && npm run check && npm run build`. The root `uv.lock` is `dependency_hash`
identity — never add web deps to it.
