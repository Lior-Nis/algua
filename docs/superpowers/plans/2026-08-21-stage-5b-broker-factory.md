# Stage 5b — Broker Factory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace five hand-rolled broker-construction helpers (each re-reading settings, re-checking credentials, and re-wiring URLs) with one `algua/execution/broker_factory.py` registry, so adding a second broker becomes one module plus one registry entry. Every helper **keeps its current module-level name as a thin delegate**; production call sites are unchanged; the paper/live safety boundary is strengthened, never routed around.

**Architecture:** The spec's seam pattern — *small Protocol + name→factory registry + config field*. The factory owns the parts that are identical across sites (read settings → check credentials → construct with the right creds/URL); each named delegate keeps the part that is genuinely per-site: **which venue, and which missing-credential policy**.

**Tech Stack:** Python 3.12, uv, pytest, ruff, mypy, import-linter.

**Spec:** `docs/superpowers/specs/2026-08-18-system-simplification-design.md` §5 item 1.

**Ground truth:** a research pass against `main`@`8de792e` (post-5a) that regenerated every count untruncated and **verified the paper/live boundary by execution**. It corrected three spec claims (see below) and produced the evidence behind every decision here.

### Spec claims that do not check out — do not repeat them

| Spec says | Reality |
|---|---|
| "~6 private `_alpaca_*_from_settings()` functions" | **2** are literally named that. There are **6 named helpers** across 3 files under 3 conventions, plus **2 inline** constructions = 8 sites. |
| "12 fine-grained Protocols in `contracts/types.py`" | **16** Protocols total, **9** broker-named. |
| "`registry/transitions.py` … construction is injected (as the merge-back saga already does)" | `algua/operator/mergeback.py` contains **zero** broker references. The real in-repo precedents are `transitions.py`'s own `approval_verifier` / `forward_certificate_verifier` parameters, and `ExitLaneGuard` — whose module docstring states the principle outright: *"injected as an `ExitLaneGuard` so the registry layer never imports a broker or the execution ledger directly."* Cite those. |
| §5 item 1: the factory is "name→factory, **config-selected**" | There is **no** settings field selecting a broker, and there should not be — a config flag able to swap paper for live would be a hazard, not a feature. The venue is selected by the CALL SITE (`BrokerKind` argument), never by config; `broker_factory.py`'s module docstring records this as a deliberate deviation. |

### Decisions (recorded for a reviewer to check, not re-derive)

1. **Every helper keeps its name as a delegate.** Not a shim to dodge test churn — **the codebase already does exactly this**: `cli/live_cmd.py:148`'s `_alpaca_live_broker` is a one-line delegate to `build_live_broker` with a comment explaining why (*"Single-sourced with the book-exit drain (#497) so the two never drift on how the real-money broker is built"*). This plan generalizes an existing local idiom. The evidence: **all 115 test pins are name-based** (106 dotted-string, 9 module-object — both resolve the attribute by name at patch time), so **115/115 keep working unchanged and zero break**. Deleting the names, per the spec's literal wording, would force ~115 pin edits across 9 files *and* make each fake dispatch on broker kind, because one shared factory serves paper, live, live-readonly and drain within the same test module. Stage 5a proved the shape at 1/20th scale: removing 3 names cost 6 retargets.
2. **Two entry points, one registry** (research Option 2): `build_broker(kind)` raises when credentials are absent; `maybe_broker(kind)` returns `None`. Rejected alternatives: a `required: bool` parameter (forces `Broker | None` on all 12 strict call sites — dishonest typing and mypy noise); and separate registry keys per policy (**worst for safety** — it multiplies near-identical keys, which is precisely the typo surface the boundary analysis warns about, and it encodes a *caller's* policy as venue identity). Test-pin impact is identical across all three options, so the choice turns purely on type honesty and boundary clarity.
3. **`SimBroker` stays inline** (`cli/paper_cmd.py:311`). It takes `cash=`, reads no settings, and has no credentials — forcing it through a credentials-shaped factory would be a poor fit. Explicit non-goal.
4. **`live/live_loop.py:13`'s private `_AlpacaBroker` import is flagged, not fixed.** It is a *type-annotation* import that constructs nothing. Annotating against a `contracts.types` Protocol instead would be an ISP cleanup orthogonal to this slice — do not bundle it.

## Global Constraints

- Quality gate on EVERY task before commit: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`. All four must pass.
- **When you run the full suite, pass `timeout: 600000` explicitly to the Bash tool.** Without it the 120s default fires and the harness **auto-backgrounds** the command — which is how the one Stage 5a stall happened *despite* a foreground instruction. This is the single most common process failure on this program.
- **Regenerate every count untruncated at the moment you assert it** — no `| head` or `| tail` on an enumeration you are counting. Six figures were wrong in Stage 5a alone, all from quoting instead of measuring. If a number here disagrees with what you measure, trust your measurement and say so.
- **Zero production behavior change.** Every delegate must preserve its current return type, its credential check, its error type and message text, and its policy (raise vs `None`). Production call sites are untouched.
- **The paper/live boundary is the thing that must not break.** Four separators exist today (different credential fields, different URL fields, different classes, and each concrete class's own `_ALLOWED_HOSTS` frozenset rejecting a crossed `base_url` at construction — the strongest of the four, and the most relevant one to a construction factory), and the load-bearing guard is **host pinning at config load** — verified by execution: assigning the live host to `alpaca_paper_url` (or vice versa) is rejected with a `ValidationError`. That guard is independent of construction, so a factory cannot by itself hand back a live broker for paper. Two holes remain, and this plan closes both: **(a) the registry lookup must have NO default** — a `.get(kind, something)` would silently substitute a broker; **(b) a test-registered fake must not leak** (Task 2).
- **`AlpacaLiveBroker`'s `LiveAuthorization` positional is a construction tollbooth and must survive.** It raises `BrokerError` if absent (`alpaca_broker.py:453-455`), and `AlpacaLiveDrainBroker` deliberately does *not* take one (documented at `lane_exit.py:38-43`). That asymmetry is intentional; preserve it.
- **Do not hoist `registry/transitions.py`'s lazy import to module level.** Verified: no import-linter contract forbids `algua.registry` → `algua.execution` (it already happens twice, both lazy inside functions), so injection is *legal* — the import is lazy for import-graph weight, not legality. Keeping it lazy is the point.
- **No import-time construction.** Verified by AST across all of `algua/`: zero module-level broker constructions exist today. Do not introduce one — a module-level `build_broker()` call would read settings at import.
- `git add` scoped to named files — never `git add -A`.
- Commits end with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Known hazard: some test writes a demo strategy file into `algua/strategies/momentum/`. If `git status` shows an untracked file there, delete it before staging.
- **CODEOWNERS-protected files this slice touches:** `cli/paper_cmd.py`, `registry/transitions.py`. Expected and unavoidable; note it in the PR so human review is anticipated.

---

### Task 1: Create the broker factory

**Files:**
- Create: `algua/execution/broker_factory.py`, `tests/test_broker_factory.py`

**Interfaces:**
- Produces: `BrokerKind` (enum), `build_broker(kind, *args)` (raises `ValueError` when credentials are absent), `maybe_broker(kind, *args)` (returns `None`), `register_broker(kind, spec)`, and the module-level `_REGISTRY` that Task 2's fixture snapshots.
- Consumes: `algua.config.settings.get_settings`, the four concrete Alpaca classes from `algua.execution.alpaca_broker`.

**No call site changes in this task.** The factory lands with its own tests first, so it is proven in isolation before anything depends on it.

- [ ] **Step 1: Read the five helpers you are consolidating**

Read them in full and note precisely what varies: `cli/paper_cmd.py:133` (`_alpaca_broker_from_settings`), `:235` (`_alpaca_live_readonly_from_settings`), `:246` (`_maybe_live_readonly`); `execution/lane_exit.py:23` (`build_live_broker`), `:36` (`build_live_drain_broker`).

What varies is exactly four things: the class, the credential field pair, the URL field, and the missing-credential policy. Everything else is duplicated. Note the constructor shapes differ — `AlpacaPaperBroker` takes keyword args; `AlpacaLiveReadOnlyBroker` and `AlpacaLiveDrainBroker` take positional key/secret plus `base_url=`; `AlpacaLiveBroker` takes the `LiveAuthorization` **first**, then key/secret, then `base_url=`.

Also read `algua/data/providers/__init__.py` — its `_REGISTRY` / `register_provider` / `get_provider` shape is the established in-repo pattern this mirrors.

- [ ] **Step 2: Write the factory**

```python
"""Name→factory registry for broker construction (spec §5 item 1).

Each entry owns the parts that are identical across construction sites -- read settings, check
credentials, construct with the right credential/URL fields. What stays at the call site is the
part that is genuinely per-site: WHICH venue, and WHICH missing-credential policy.

Two entry points rather than a policy flag, because the policy is load-bearing and differs by
CALLER, not by venue: ``build_broker`` raises when credentials are absent; ``maybe_broker`` returns
None so the caller can decide. Both resolve the same registry entry.

SAFETY: the lookup has NO default. A defaulted lookup is the one way this registry could hand back
a broker for a venue the caller did not ask for; the paper/live separation is otherwise enforced
independently of construction, by host pinning at config load (config/settings.py's field
validators reject assigning the live host to alpaca_paper_url and vice versa).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from algua.config.settings import get_settings


class BrokerKind(str, Enum):
    """The venues this factory can construct. Keyed as an enum, never a bare string, so a typo is
    an AttributeError at author time rather than a lookup miss at run time."""

    ALPACA_PAPER = "alpaca_paper"
    ALPACA_LIVE = "alpaca_live"
    ALPACA_LIVE_READONLY = "alpaca_live_readonly"
    ALPACA_LIVE_DRAIN = "alpaca_live_drain"


@dataclass(frozen=True)
class BrokerSpec:
    """How one venue is built: where its credentials and URL live in settings, how to construct it,
    and what to say when the credentials are missing."""

    key_field: str
    secret_field: str
    url_field: str
    construct: Callable[..., Any]
    missing_credentials: str


_REGISTRY: dict[BrokerKind, BrokerSpec] = {}


def register_broker(kind: BrokerKind, spec: BrokerSpec) -> None:
    """Register (or replace) the spec for ``kind``. Tests that register a fake MUST be isolated --
    see the autouse registry fixture in tests/conftest.py; a fake leaking into a later test would
    change what that test constructs across the paper/live boundary."""
    _REGISTRY[kind] = spec


def _resolve(kind: BrokerKind) -> tuple[BrokerSpec, str | None, str | None, str]:
    """The spec plus its configured credentials/URL. Fails closed on an unknown kind."""
    try:
        spec = _REGISTRY[kind]
    except KeyError:
        valid = ", ".join(sorted(k.value for k in _REGISTRY))
        raise ValueError(f"unknown broker kind {kind!r}; valid: {valid}") from None
    s = get_settings()
    return (spec, getattr(s, spec.key_field), getattr(s, spec.secret_field),
            getattr(s, spec.url_field))


def build_broker(kind: BrokerKind, *args: Any) -> Any:
    """Construct ``kind``, raising ``ValueError`` if its credentials are not configured.

    ``*args`` is passed to the constructor BEFORE the credentials -- it carries
    ``AlpacaLiveBroker``'s ``LiveAuthorization`` tollbooth, which no other venue takes. The return
    is ``Any`` because the four concrete classes share no public base; each named delegate re-asserts
    the concrete type, and mypy checks the delegate bodies against their declared returns.
    """
    spec, key, secret, url = _resolve(kind)
    if not key or not secret:
        raise ValueError(spec.missing_credentials)
    return spec.construct(*args, key, secret, base_url=url)


def maybe_broker(kind: BrokerKind, *args: Any) -> Any | None:
    """``build_broker``, but returns ``None`` when credentials are absent instead of raising.

    The two lenient call sites want this for OPPOSITE reasons -- one keeps going without a broker,
    the other refuses to proceed -- so the decision belongs to the caller and this returns None
    rather than encoding either.
    """
    spec, key, secret, url = _resolve(kind)
    if not key or not secret:
        return None
    return spec.construct(*args, key, secret, base_url=url)
```

Then register the four venues, moving each `missing_credentials` message **verbatim** from the helper it comes from (the messages name specific env vars and are user-facing). Note `AlpacaPaperBroker` currently takes keyword `api_key=`/`api_secret=` while the other three take positionals — normalize by wrapping it in a small lambda or `functools.partial`-style adapter in its spec's `construct`, rather than changing the class.

- [ ] **Step 3: Write the factory's tests**

In `tests/test_broker_factory.py`, cover at minimum:

- each of the four kinds constructs the expected concrete class when credentials are set (use `monkeypatch.setenv`, not a registered fake, so the real construction path is exercised);
- `build_broker` **raises** `ValueError` with the venue's exact message when credentials are absent;
- `maybe_broker` **returns `None`** in the same situation — same kind, same absent credentials, different outcome. This is the pair that proves the two-entry-point design does what the helpers did;
- an unknown kind **fails closed** with the valid set named, and — the safety-critical one — **there is no default**: assert that no lookup path returns a broker for an unregistered kind;
- `AlpacaLiveBroker` still requires its `LiveAuthorization`: `build_broker(BrokerKind.ALPACA_LIVE)` without the positional must fail rather than construct an unauthorized live broker.

- [ ] **Step 4: Full quality gate**

Run with `timeout: 600000`: `uv run pytest -q 2>&1 | tail -20`, then ruff, mypy, lint-imports. Test count rises by the tests you added; nothing else changes yet (no call site touches the factory).

- [ ] **Step 5: Commit**

```bash
git add algua/execution/broker_factory.py tests/test_broker_factory.py
```
Commit message: describe the registry, the two entry points and why (policy differs by caller, not venue), and the no-default safety rule.

---

### Task 2: Registry isolation fixture

**Files:**
- Modify: `tests/conftest.py`, and the two test files currently doing manual cleanup

**Why this is its own task, before any call site uses the factory:** isolation today is **manual** — `tests/test_data_providers.py:353-355` and `tests/test_data_ingest_streamed.py:28-30` do `del _REGISTRY["dummy"]` in a `finally`. There is **no autouse fixture anywhere**, and 5a's tracker registry has none either. That pattern is fragile in two ways: a test that errors before its `del` leaks, and a test that **overwrites an existing key** rather than adding a new one leaks even when it succeeds. For providers and trackers a leak is cosmetic. **For brokers it is a safety issue** — a fake registered under a live kind that survives into a later test changes what that test constructs across the paper/live boundary.

- [ ] **Step 1: Add an autouse snapshot/restore fixture**

In `tests/conftest.py`, alongside the existing isolation fixtures:

```python
@pytest.fixture(autouse=True)
def _isolated_registries():
    """Snapshot every name→factory registry and restore it wholesale after each test.

    Restore, not delete: the manual ``del _REGISTRY["dummy"]`` pattern this replaces reverts an
    ADDITION but silently keeps an OVERWRITE, and leaks entirely if the test errors first. For the
    broker registry that is a safety concern rather than tidiness -- a fake left registered under a
    live kind changes what a LATER test constructs across the paper/live boundary.
    """
    from algua.data.providers import _REGISTRY as providers
    from algua.execution.broker_factory import _REGISTRY as brokers
    from algua.tracking.factory import _REGISTRY as trackers

    snapshots = [(r, dict(r)) for r in (providers, brokers, trackers)]
    yield
    for registry, snapshot in snapshots:
        registry.clear()
        registry.update(snapshot)
```

Verify the import paths against the live modules before committing to them.

- [ ] **Step 2: Remove the now-redundant manual cleanup**

Delete the `del _REGISTRY[...]` / `finally` scaffolding in `tests/test_data_providers.py` and `tests/test_data_ingest_streamed.py` — the fixture supersedes it, and leaving both is the dual-path cruft this program removes. Keep each test's actual assertions unchanged.

- [ ] **Step 3: Prove the fixture actually isolates**

A fixture that silently does nothing is worse than none. Add a two-test pair to `tests/test_broker_factory.py` demonstrating it: the first registers a deliberately wrong spec under a kind, the second asserts that kind still resolves to the real class. Ordering-dependent by construction — that is the point, and pytest runs them in file order.

- [ ] **Step 4: Full quality gate, then commit** (`timeout: 600000`).

---

### Task 3: Reduce the five helpers to delegates

**Files:**
- Modify: `algua/cli/paper_cmd.py` (CODEOWNERS-protected), `algua/execution/lane_exit.py`

**The invariant:** each helper keeps its **exact current name, signature, return type, error type and message**. Only its body changes — from hand-rolled construction to a one-line factory call. Production call sites and all 115 test pins are untouched.

- [ ] **Step 1: Rewrite the five bodies**

Each becomes a delegate. For example `_alpaca_broker_from_settings`:

```python
def _alpaca_broker_from_settings() -> AlpacaPaperBroker:
    return build_broker(BrokerKind.ALPACA_PAPER)
```

and `_maybe_live_readonly` keeps its docstring verbatim (it explains *why* lenient) with the body reduced to `return maybe_broker(BrokerKind.ALPACA_LIVE_READONLY)`.

Apply the same to `_alpaca_live_readonly_from_settings` (`build_broker`), `build_live_broker` (`build_broker(BrokerKind.ALPACA_LIVE, authorization)` — note the tollbooth passes through as the leading positional), and `build_live_drain_broker` (`maybe_broker`).

**Keep every existing docstring and comment**, especially `build_live_drain_broker`'s explanation that it deliberately takes no `LiveAuthorization`, and `_maybe_live_readonly`'s "resume-all stays lenient". Those document *caller* semantics the factory does not encode.

Follow `cli/live_cmd.py:148`'s existing delegate as the house style.

- [ ] **Step 2: Prove the delegates are behaviour-identical**

For each of the five, assert the old and new produce the same outcome — same class with credentials set, same error type *and message* without. Compare against `git show HEAD~2:<file>` (or whatever the pre-Task-1 base is) rather than from memory.

The strongest cheap check: run the existing suites that own these paths — `tests/test_cli_paper.py`, `tests/test_cli_live.py`, `tests/test_lane_exit.py`, `tests/test_registry_live_exit_guard.py` — and confirm they pass **without edits**. That is the delegate design's whole thesis; if any pin needs touching, stop and report it, because the thesis is wrong.

- [ ] **Step 3: Full quality gate, then commit** (`timeout: 600000`). Test count unchanged.

---

### Task 4: Route the registry layer's default verifier through the factory

**Files:**
- Modify: `algua/registry/transitions.py` (CODEOWNERS-protected)

**Read this before editing — the spec misreads the current state.** `transition_strategy` **already** accepts `forward_certificate_verifier: ForwardCertificateVerifier | None = None` (`:43`, used at `:135`), and its docstring calls the default builder *"the single monkeypatch seam for tests"*. So the registry layer already accepts injected construction. What remains is that the *default* builder (`_default_forward_certificate_verifier`, `:203-239`) still hand-rolls a broker at `:228`.

So this task is narrower than "make it injectable": it is **make the default stop hand-rolling construction**, by having it call the factory.

- [ ] **Step 1: Replace the inline construction**

Inside `_default_forward_certificate_verifier`'s closure, replace the inline `AlpacaPaperBroker(...)` construction (and its credential check, which currently raises `TransitionError`) with a `build_broker(BrokerKind.ALPACA_PAPER)` call.

**Preserve the `TransitionError`.** The registry layer raises `TransitionError`, not `ValueError` — that is its error vocabulary and callers depend on it. Catch the factory's `ValueError` and re-raise as `TransitionError` with the existing message, or check credentials first; either way the exception type and message a caller sees must not change.

**Keep the import lazy.** The `from algua.execution... import ...` stays *inside* the function. Hoisting it would pull `execution` into the registry import graph — legal (no contract forbids it), but the laziness exists for import-graph weight and this task must not spend it.

- [ ] **Step 2: Confirm the injection seam still works**

The point of this file is that tests can inject a verifier. Confirm `tests/` still exercises that path and passes unchanged, and that the default path still constructs a real paper broker when credentials are configured.

- [ ] **Step 3: Full quality gate, then commit** (`timeout: 600000`).

---

### Task 5: Close-out verification

**Files:** none expected (verification only; fix anything found)

- [ ] **Step 1: No hand-rolled construction survives outside the factory**

```bash
grep -rn "AlpacaPaperBroker(\|AlpacaLiveBroker(\|AlpacaLiveReadOnlyBroker(\|AlpacaLiveDrainBroker(" algua/ --include='*.py'
```
Every hit should be inside `algua/execution/broker_factory.py` (the registry specs) or `algua/execution/alpaca_broker.py` (the definitions). A hit anywhere else is a missed site — read it and decide whether it was deliberately out of scope (`SimBroker` is, but it is a different class and will not match this grep).

- [ ] **Step 2: The registry has no default and fails closed**

```bash
uv run python -c "
from algua.execution.broker_factory import BrokerKind, build_broker, maybe_broker, _REGISTRY
print('registered:', sorted(k.value for k in _REGISTRY))
for fn in (build_broker, maybe_broker):
    try:
        fn('not-a-kind')  # type: ignore[arg-type]
        raise SystemExit(f'FAIL: {fn.__name__} returned for an unknown kind')
    except ValueError as e:
        print(f'{fn.__name__} fails closed:', e)
"
```
Both must raise `ValueError` naming the valid set. Neither may return a broker or `None` for an unknown kind — `maybe_broker` returning `None` there would conflate "unknown venue" with "credentials absent", which is the ambiguity Stage 5a's tracker seam had to design around.

- [ ] **Step 3: The paper/live boundary still holds**

Re-verify the guard the factory relies on, by execution:

```bash
uv run python -c "
from pydantic import ValidationError
from algua.config.settings import Settings
for field, host in (('alpaca_paper_url', 'https://api.alpaca.markets'),
                    ('alpaca_live_url', 'https://paper-api.alpaca.markets')):
    try:
        Settings(**{field: host})
        raise SystemExit(f'FAIL: {field} accepted the wrong host')
    except ValidationError:
        print(f'{field} correctly rejects the crossed host')
"
```

- [ ] **Step 4: All 115 pins still work — the delegate thesis, verified**

Regenerate the pin count untruncated and confirm no test file needed editing for the pins:

```bash
grep -rn "_alpaca_broker_from_settings\|_alpaca_live_broker\|_alpaca_live_readonly_from_settings\|build_live_drain_broker\|_maybe_live_readonly\|build_live_broker" tests/ --include='*.py' | grep -c "monkeypatch.setattr"
```
Report the number you get. Then confirm `git diff main --stat -- tests/` shows changes **only** in the files this plan named (the new factory tests, `conftest.py`, and the two files whose manual cleanup was removed) — no edits to the nine pin-bearing suites.

- [ ] **Step 5: Full quality gate + CLI smoke** (`timeout: 600000`)

`uv run pytest -q`, ruff, mypy, lint-imports. Then `uv run algua doctor` and `uv run algua registry list` — both should behave exactly as on `main`.

- [ ] **Step 6: Commit any fixes**

If steps 1-5 forced fixes, commit them (scoped `git add`, correct trailer). If nothing needed fixing, make no commit — expected, and consistent with every prior close-out in this program.
