# Live Alpaca Broker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A gated `AlpacaLiveBroker` (DRY with the paper broker via a shared base) that can reach `api.alpaca.markets` only with live keys AND a verified `LiveAuthorization` to construct it — without weakening the paper-only invariant. No live command/loop.

**Architecture:** Extract the venue-agnostic REST into a `_AlpacaBroker` base whose `__init__` checks `host in self._ALLOWED_HOSTS` (empty on the base). `AlpacaPaperBroker` and `AlpacaLiveBroker` are thin subclasses differing only in allowed hosts + default URL; `AlpacaLiveBroker.__init__` additionally requires a `LiveAuthorization` (a frozen dataclass minted only by `verify_live_authorization`). Add live settings. Nothing constructs the live broker yet.

**Tech Stack:** Python 3.12, requests, pydantic-settings, pytest, ruff, mypy, import-linter. Spec: `docs/superpowers/specs/2026-06-05-live-broker-design.md`.

---

## File structure

| File | Responsibility |
|---|---|
| `algua/contracts/types.py` (modify) | new frozen `LiveAuthorization` dataclass (pure data). |
| `algua/registry/live_gate.py` (modify) | `verify_live_authorization` returns `LiveAuthorization`. |
| `algua/execution/alpaca_broker.py` (modify) | `_AlpacaBroker` base; `AlpacaPaperBroker` subclass (unchanged behaviour); new `AlpacaLiveBroker` (tollbooth). |
| `algua/config/settings.py` (modify) | `alpaca_live_*` settings + host validator. |

---

### Task 1: `LiveAuthorization` token + `verify_live_authorization` returns it

**Files:** Modify `algua/contracts/types.py`, `algua/registry/live_gate.py`; Test `tests/test_live_gate.py`.

Context: `algua/contracts/types.py` is the pure contracts module (has `OrderIntent`, `Side`, a `Broker` Protocol; imports only stdlib). `algua/registry/live_gate.py::verify_live_authorization(conn, repo, name, allowed_signers_path) -> sqlite3.Row` currently returns the raw row; its happy-path test `test_verify_live_authorization_happy_path` asserts `row["principal"] == "lior"`. `rec` (the `StrategyRecord`) is in scope there with `.id`; `row` has columns `code_hash, config_hash, dependency_hash, principal, authorized_at`.

- [ ] **Step 1: Add the dataclass + update the test.** Append to `algua/contracts/types.py`:
```python
@dataclass(frozen=True)
class LiveAuthorization:
    """Proof that a strategy's CURRENT artifact is human-authorized for live trading. Minted ONLY by
    `algua.registry.live_gate.verify_live_authorization` after a successful signature re-verification;
    holding one is the proof. `AlpacaLiveBroker` requires one to construct, so a live broker cannot
    exist without a passed gate."""
    strategy_id: int
    code_hash: str
    config_hash: str
    dependency_hash: str | None
    principal: str
    authorized_at: str
```
In `tests/test_live_gate.py`, change `test_verify_live_authorization_happy_path`'s tail from:
```python
    row = live_gate.verify_live_authorization(conn, SqliteStrategyRepository(conn), "s", signers)
    assert row["principal"] == "lior"
```
to:
```python
    auth = live_gate.verify_live_authorization(conn, SqliteStrategyRepository(conn), "s", signers)
    from algua.contracts.types import LiveAuthorization
    assert isinstance(auth, LiveAuthorization)
    assert auth.principal == "lior" and auth.strategy_id == 1
```

- [ ] **Step 2: Run** `uv run pytest tests/test_live_gate.py -q -k live_authorization` → FAIL (returns a Row, no `.principal`).

- [ ] **Step 3: Return the token.** In `algua/registry/live_gate.py`: add `from algua.contracts.types import LiveAuthorization` to the imports; change `verify_live_authorization`'s return annotation `-> sqlite3.Row` to `-> LiveAuthorization`; replace its final `return row` with:
```python
    return LiveAuthorization(
        strategy_id=rec.id,
        code_hash=row["code_hash"],
        config_hash=row["config_hash"],
        dependency_hash=row["dependency_hash"],
        principal=row["principal"],
        authorized_at=row["authorized_at"],
    )
```

- [ ] **Step 4: Run** `uv run pytest tests/test_live_gate.py -q` → PASS (the rejection tests still raise; the happy path now returns a `LiveAuthorization`).

- [ ] **Step 5: Gate + commit** — `uv run ruff check . && uv run mypy algua && uv run pytest tests/test_live_gate.py -q && uv run lint-imports`
```bash
git add algua/contracts/types.py algua/registry/live_gate.py tests/test_live_gate.py
git commit -m "feat(contracts): LiveAuthorization token; verify_live_authorization returns it"
```

---

### Task 2: refactor `AlpacaPaperBroker` to a `_AlpacaBroker` base

**Files:** Modify `algua/execution/alpaca_broker.py`; Test `tests/test_alpaca_broker.py`.

Context: `alpaca_broker.py` has module constants `_DEFAULT_BASE_URL = "https://paper-api.alpaca.markets"` and `_PAPER_HOSTS = frozenset({"paper-api.alpaca.markets"})`, and `class AlpacaPaperBroker` whose `__init__(self, api_key, api_secret, base_url=_DEFAULT_BASE_URL)` raises `BrokerError` (message contains "non-paper") when `host not in _PAPER_HOSTS`, then sets `self.api_key/api_secret/base_url`. It has 8 public methods + `_headers`/`_request`/`_get`/`_post`/`_delete`/`_read`/`_num`. `tests/test_alpaca_broker.py` has `test_rejects_non_paper_base_url` (`pytest.raises(BrokerError, match="non-paper")` constructing against `https://api.alpaca.markets`) and `test_accepts_paper_base_url`, plus `_broker()` building `AlpacaPaperBroker(api_key="k", api_secret="s")` and a `Broker`-protocol conformance test.

- [ ] **Step 1: Add a failing test** — append to `tests/test_alpaca_broker.py`:
```python
def test_base_broker_alone_reaches_no_venue():
    # the base class has an EMPTY allowed-host set, so it can't be constructed against any host
    from algua.execution.alpaca_broker import _AlpacaBroker
    with pytest.raises(BrokerError):
        _AlpacaBroker("k", "s", "https://paper-api.alpaca.markets")
```
Also change `test_rejects_non_paper_base_url`'s matcher from `match="non-paper"` to `match="refusing"` (the generalized base message), keeping the rest (it still constructs `AlpacaPaperBroker(..., base_url="https://api.alpaca.markets")` and expects `BrokerError`).

- [ ] **Step 2: Run** `uv run pytest tests/test_alpaca_broker.py -q` → FAIL (`_AlpacaBroker` undefined; matcher).

- [ ] **Step 3: Refactor.** In `algua/execution/alpaca_broker.py`:
  - Rename the `_DEFAULT_BASE_URL` constant to `_PAPER_DEFAULT_URL` (same value) and add `_LIVE_DEFAULT_URL = "https://api.alpaca.markets"`. Delete the module-level `_PAPER_HOSTS` constant (it moves onto the subclass).
  - Rename `class AlpacaPaperBroker:` to `class _AlpacaBroker:` and give it `_ALLOWED_HOSTS: frozenset[str] = frozenset()` as the first class attribute. Change its `__init__` signature to `def __init__(self, api_key: str, api_secret: str, base_url: str) -> None:` (no default) and the guard to:
```python
        host = urlparse(base_url).hostname
        if host not in self._ALLOWED_HOSTS:
            raise BrokerError(
                f"refusing Alpaca endpoint {base_url!r} (host {host!r}); "
                f"permitted hosts: {sorted(self._ALLOWED_HOSTS)}"
            )
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url.rstrip("/")
```
  Leave ALL other methods (`_headers` … `submit`) unchanged inside `_AlpacaBroker`.
  - Immediately after the `_AlpacaBroker` class, add the paper subclass preserving the old public surface:
```python
class AlpacaPaperBroker(_AlpacaBroker):
    """The Alpaca PAPER venue. Hard-refuses any non-paper host (the platform invariant)."""

    _ALLOWED_HOSTS = frozenset({"paper-api.alpaca.markets"})

    def __init__(self, api_key: str, api_secret: str, base_url: str = _PAPER_DEFAULT_URL) -> None:
        super().__init__(api_key, api_secret, base_url)
```

- [ ] **Step 4: Run** `uv run pytest tests/test_alpaca_broker.py -q` → PASS (all existing paper tests + the new base test).

- [ ] **Step 5: Gate + commit** — `uv run ruff check . && uv run mypy algua && uv run pytest tests/test_alpaca_broker.py -q && uv run lint-imports`
```bash
git add algua/execution/alpaca_broker.py tests/test_alpaca_broker.py
git commit -m "refactor(broker): extract _AlpacaBroker base; AlpacaPaperBroker is a thin subclass"
```

---

### Task 3: `AlpacaLiveBroker` (tollbooth) + live settings

**Files:** Modify `algua/execution/alpaca_broker.py`, `algua/config/settings.py`; Test `tests/test_alpaca_broker.py`, `tests/test_settings.py` (or wherever settings are tested).

Context: from Task 2, `alpaca_broker.py` has `_AlpacaBroker`, `AlpacaPaperBroker`, `_LIVE_DEFAULT_URL`, `BrokerError`, and imports `from algua.contracts.types import OrderIntent` (add `LiveAuthorization`). `algua/config/settings.py` has `alpaca_api_key`/`alpaca_api_secret`/`alpaca_paper_url` + a `_paper_url_is_paper_endpoint` validator pinning the paper host; `_ALPACA_PAPER_HOST = "paper-api.alpaca.markets"` is a module constant. Find the existing settings test file (e.g. `tests/test_settings.py` or `tests/test_config.py`) by grepping for `alpaca_paper_url`.

- [ ] **Step 1: Failing tests** — append to `tests/test_alpaca_broker.py`:
```python
def _live_auth():
    from algua.contracts.types import LiveAuthorization
    return LiveAuthorization(strategy_id=1, code_hash="c", config_hash="cf",
                             dependency_hash="d", principal="lior", authorized_at="2026-06-05")


def test_live_broker_requires_authorization():
    from algua.execution.alpaca_broker import AlpacaLiveBroker
    with pytest.raises(BrokerError, match="LiveAuthorization"):
        AlpacaLiveBroker(None, "k", "s")  # no token
    with pytest.raises(BrokerError):
        AlpacaLiveBroker({"principal": "lior"}, "k", "s")  # a look-alike dict is not the token


def test_live_broker_constructs_with_authorization_and_live_host():
    from algua.execution.alpaca_broker import AlpacaLiveBroker
    b = AlpacaLiveBroker(_live_auth(), "k", "s")
    assert b.base_url == "https://api.alpaca.markets"
    assert b.authorization.principal == "lior"


def test_live_broker_rejects_non_live_host():
    from algua.execution.alpaca_broker import AlpacaLiveBroker
    with pytest.raises(BrokerError, match="refusing"):
        AlpacaLiveBroker(_live_auth(), "k", "s", base_url="https://paper-api.alpaca.markets")


def test_live_broker_account_uses_inherited_rest(monkeypatch):
    # the live subclass inherits the venue-agnostic REST: account() parses against the live host
    from algua.execution import alpaca_broker as ab
    from algua.execution.alpaca_broker import AlpacaLiveBroker
    fake = _FakeRequests({"/v2/account": _FakeResp(200, {"equity": "5", "cash": "5",
                                                         "buying_power": "5"})})
    monkeypatch.setattr(ab, "requests", fake)
    acct = AlpacaLiveBroker(_live_auth(), "k", "s").account()
    assert acct.equity == 5.0
```
(`_FakeRequests`/`_FakeResp` already exist in this test file from the paper tests — reuse them. `_FakeRequests` routes GET by URL suffix, so `account()` resolves regardless of host; the host targeting is already proven by `test_live_broker_constructs_with_authorization_and_live_host` + the host guard.)

Append to `tests/test_config.py`:
```python
def test_alpaca_live_url_validator(monkeypatch):
    import pytest

    from algua.config.settings import Settings
    monkeypatch.setenv("ALGUA_ALPACA_LIVE_URL", "https://paper-api.alpaca.markets")
    with pytest.raises(ValueError):
        Settings()
    monkeypatch.setenv("ALGUA_ALPACA_LIVE_URL", "https://api.alpaca.markets")
    assert Settings().alpaca_live_url == "https://api.alpaca.markets"
```

- [ ] **Step 2: Run** the two test files → FAIL.

- [ ] **Step 3: Add `AlpacaLiveBroker`** — in `algua/execution/alpaca_broker.py`, add `LiveAuthorization` to the `from algua.contracts.types import ...` line, and after `AlpacaPaperBroker`:
```python
class AlpacaLiveBroker(_AlpacaBroker):
    """The Alpaca LIVE (real-money) venue. Constructable ONLY with a verified LiveAuthorization
    (the construction tollbooth) AND against the live host — so a live broker cannot exist without
    a passed go-live gate. The authorization is kept for audit/the loop, never used for REST."""

    _ALLOWED_HOSTS = frozenset({"api.alpaca.markets"})

    def __init__(self, authorization: LiveAuthorization, api_key: str, api_secret: str,
                 base_url: str = _LIVE_DEFAULT_URL) -> None:
        if not isinstance(authorization, LiveAuthorization):
            raise BrokerError("AlpacaLiveBroker requires a verified LiveAuthorization")
        super().__init__(api_key, api_secret, base_url)
        self.authorization = authorization
```

- [ ] **Step 4: Add live settings** — in `algua/config/settings.py`, add a module constant `_ALPACA_LIVE_HOST = "api.alpaca.markets"`, the three fields:
```python
    alpaca_live_api_key: str | None = None
    alpaca_live_api_secret: str | None = None
    alpaca_live_url: str = "https://api.alpaca.markets"
```
and the validator (mirroring the paper one):
```python
    @field_validator("alpaca_live_url")
    @classmethod
    def _live_url_is_live_endpoint(cls, value: str) -> str:
        host = urlparse(value).hostname
        if host != _ALPACA_LIVE_HOST:
            raise ValueError(
                f"alpaca_live_url must point at the live endpoint ({_ALPACA_LIVE_HOST}); "
                f"got host {host!r}"
            )
        return value
```

- [ ] **Step 5: Run** `uv run pytest tests/test_alpaca_broker.py tests/test_config.py -q` → PASS.

- [ ] **Step 6: Full gate + commit** — `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports` (all pass; contracts kept, 0 broken).
```bash
git add algua/execution/alpaca_broker.py algua/config/settings.py tests/test_alpaca_broker.py tests/test_config.py
git commit -m "feat(broker): AlpacaLiveBroker (authorization tollbooth + live-host guard) + live settings"
```

---

## Self-review notes

- **Spec coverage:** `_AlpacaBroker` base with empty `_ALLOWED_HOSTS` (§3 → Task 2); `LiveAuthorization` token in contracts + returned by the gate (§4 → Task 1); `AlpacaLiveBroker` tollbooth + live-host guard (§5 → Task 3); live settings + validator (§6 → Task 3); tests for paper-unchanged, base-can't-construct, tollbooth (None + look-alike dict), non-live host, mocked live REST, the token return, and the settings validator (§8 → Tasks 1–3). No live command/loop (out of scope). Import-linter: `LiveAuthorization` in `algua.contracts` keeps it pure; execution + registry importing contracts is allowed.
- **Type consistency:** `LiveAuthorization(strategy_id, code_hash, config_hash, dependency_hash, principal, authorized_at)` is defined in Task 1 and constructed identically by `verify_live_authorization` (Task 1) and the test helper `_live_auth` (Task 3); `AlpacaLiveBroker.__init__(authorization, api_key, api_secret, base_url)` matches its tests. `_AlpacaBroker._ALLOWED_HOSTS` / `_PAPER_DEFAULT_URL` / `_LIVE_DEFAULT_URL` names are used consistently across Tasks 2–3.
- **No placeholders:** every code step is complete. The two "find the settings test file / mirror how the GET test asserts the host" notes are verifications against existing code, not gaps (the implementer greps for `alpaca_paper_url` tests and the existing `_FakeRequests` GET assertions).
- **Security invariant:** three independent walls (live keys in env, the `LiveAuthorization` tollbooth, the slice-3 re-verify); this slice delivers the tollbooth + the host guard, and the base class alone reaches no venue.
