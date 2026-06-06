# Live Alpaca Broker (Sub-project 6, live-execution slice 2)

**Date:** 2026-06-05
**Status:** Accepted (pending implementation)
**Scope:** A gated `AlpacaLiveBroker` that can reach `api.alpaca.markets` with separate live keys —
**impossible to construct without a verified `LiveAuthorization`** — without weakening the paper-only
invariant. **No live trading command or loop in this slice** (slice 3); nothing constructs the live
broker yet.

---

## 1. Context & non-goals

The paper broker (`AlpacaPaperBroker`) hard-refuses any non-paper host. Its 8 REST methods
(`account`/`get_positions`/`snapshot`/`submit_sized`/`submit`/`cancel_open_orders`/`close_positions`/
`close_all_positions`) are venue-agnostic — only the host differs. Slice 1 built
`verify_live_authorization`, the trade-time wall. This slice adds the *adapter* that can talk to the
live venue, behind a construction tollbooth so it cannot exist without proof of authorization.

**Non-goals (slice 3+):** the `live trade-tick` command, the live loop, calling
`verify_live_authorization` before each order, position reconciliation, capital caps. **Out of
scope:** any real live HTTP call (all tests mock the transport); changing `AlpacaPaperBroker`'s
behaviour or public signature.

---

## 2. Design decisions (settled in brainstorming)

- **Shared base + paper/live subclasses.** A `_AlpacaBroker` base holds the venue-agnostic REST; the
  base alone cannot construct against any host. `AlpacaPaperBroker` and `AlpacaLiveBroker` differ
  only in their allowed-host set + default URL.
- **Construction tollbooth.** `AlpacaLiveBroker.__init__` requires a `LiveAuthorization` and refuses
  without it — so a live broker object cannot exist unless the caller passed `verify_live_authorization`.
- **`LiveAuthorization` is a typed token** (frozen dataclass, only minted by `verify_live_authorization`),
  in `algua/contracts/` so both the registry and execution layers import it (no execution→registry dep).

---

## 3. Refactor — `_AlpacaBroker` base

In `algua/execution/alpaca_broker.py`, extract a base class holding everything venue-agnostic: the
`_request`/`_get`/`_post`/`_delete`/`_read`/`_num` helpers, `AccountState`, `TickSnapshot`, and the 8
public methods. Its `__init__(api_key, api_secret, base_url)` validates the host:
```python
class _AlpacaBroker:
    _ALLOWED_HOSTS: frozenset[str] = frozenset()   # empty: the base alone reaches NO venue
    def __init__(self, api_key, api_secret, base_url):
        host = urlparse(base_url).hostname
        if host not in self._ALLOWED_HOSTS:
            raise BrokerError(f"refusing endpoint {base_url!r} (host {host!r}); "
                              f"permitted: {sorted(self._ALLOWED_HOSTS)}")
        ...
```
`AlpacaPaperBroker(_AlpacaBroker)`: `_ALLOWED_HOSTS = frozenset({"paper-api.alpaca.markets"})`,
`__init__(api_key, api_secret, base_url=_PAPER_DEFAULT_URL)` — **identical public signature and
behaviour** to today (existing callers + tests unchanged). The old module-level `_PAPER_HOSTS` guard
folds into the subclass's `_ALLOWED_HOSTS`.

---

## 4. `LiveAuthorization` typed token

New frozen dataclass in `algua/contracts/` (pure data, no I/O):
```python
@dataclass(frozen=True)
class LiveAuthorization:
    strategy_id: int
    code_hash: str
    config_hash: str
    dependency_hash: str | None
    principal: str
    authorized_at: str
```
`live_gate.verify_live_authorization` returns a `LiveAuthorization` (built from the verified row)
instead of a raw `sqlite3.Row`. Only that function mints one after a successful re-verification, so
*holding* a `LiveAuthorization` is the proof. (Ripple: the one happy-path test that reads
`row["principal"]` becomes `.principal`; the rejection tests are unchanged.)

---

## 5. `AlpacaLiveBroker`

```python
class AlpacaLiveBroker(_AlpacaBroker):
    _ALLOWED_HOSTS = frozenset({"api.alpaca.markets"})
    def __init__(self, authorization: LiveAuthorization, api_key: str, api_secret: str,
                 base_url: str = _LIVE_DEFAULT_URL) -> None:
        if not isinstance(authorization, LiveAuthorization):
            raise BrokerError("AlpacaLiveBroker requires a verified LiveAuthorization")
        super().__init__(api_key, api_secret, base_url)
        self.authorization = authorization   # kept for audit/the loop; NOT used for REST
```
The tollbooth (`isinstance` check) + the live-host guard together mean a live broker exists only when
the caller holds a real authorization AND targets the live venue. It inherits all REST methods
unchanged. Nothing in this slice constructs it.

---

## 6. Live settings

In `algua/config/settings.py`, add:
```python
alpaca_live_api_key: str | None = None
alpaca_live_api_secret: str | None = None
alpaca_live_url: str = "https://api.alpaca.markets"
```
with a validator pinning `alpaca_live_url`'s host to `api.alpaca.markets` (mirror the paper-url
validator). The market-data URL is unchanged. **No `_alpaca_live_broker_from_settings` factory in
this slice** — wiring keys + an authorization into a live broker is the slice-3 loop's job. The live
keys being absent from the agent's sandbox env is an independent third wall.

---

## 7. The three independent walls

An agent must pass all three to place a live order: **(1)** live keys present in the env (they live
only in the trusted operator context); **(2)** a `LiveAuthorization` to construct the broker
(type/tollbooth); **(3)** the slice-3 loop re-verifying the signature via `verify_live_authorization`
before each order. This slice delivers walls 1 and 2.

---

## 8. Testing

- **Paper unchanged** — existing `AlpacaPaperBroker` tests pass as-is after the base extraction;
  `AlpacaPaperBroker` still refuses a live host.
- **`_AlpacaBroker` base** — constructing it directly against any host raises (empty `_ALLOWED_HOSTS`).
- **`AlpacaLiveBroker` tollbooth** — `__init__` without a `LiveAuthorization` (or with a non-token,
  e.g. a dict) raises `BrokerError`; with a `LiveAuthorization` + the live host it constructs.
- **Live host guard** — `AlpacaLiveBroker(auth, …, base_url=<paper or other host>)` raises.
- **REST works (mocked)** — with a fake `requests`, a constructed `AlpacaLiveBroker` hits
  `api.alpaca.markets` (assert the URL) for `account()` / a `submit_sized`.
- **`verify_live_authorization` returns `LiveAuthorization`** — its happy-path test reads `.principal`
  / `.code_hash`.
- **Settings** — `alpaca_live_url` validator rejects a non-`api.alpaca.markets` host; accepts the live host.
- **Gate** — `pytest · ruff · mypy · lint-imports` (contracts kept, 0 broken; the new
  execution→contracts and registry→contracts imports are allowed).

---

## 9. Consequences

- The platform now has a live-venue adapter that is DRY with the paper broker, keeps the paper
  hard-guard intact, and **cannot be instantiated without a verified authorization** — defense in
  depth around the money venue.
- `LiveAuthorization` becomes the typed currency of "this artifact may trade live," passed from the
  gate (slice 1) to the broker (slice 2) and, next, to the loop (slice 3) that re-verifies it per
  order and finally places real trades.
