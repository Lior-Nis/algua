"""Tests for the broker factory (algua/execution/broker_factory.py, spec §5 item 1).

Every "absent credentials" test explicitly ``delenv``s both fields first -- the developer's real
shell environment may have Alpaca credentials set, and ``tests/conftest.py`` only stops
pydantic-settings from reading the repo's ``.env`` file, not the process environment.
"""

from __future__ import annotations

import pytest

from algua.contracts.types import LiveAuthorization
from algua.execution.alpaca_broker import (
    AlpacaLiveBroker,
    AlpacaLiveDrainBroker,
    AlpacaLiveReadOnlyBroker,
    AlpacaPaperBroker,
)
from algua.execution.broker_factory import (
    _REGISTRY,
    BrokerKind,
    BrokerSpec,
    build_broker,
    maybe_broker,
    register_broker,
)
from algua.execution.errors import BrokerError

_PAPER_MSG = (
    "Alpaca paper credentials not configured; set ALGUA_ALPACA_API_KEY "
    "and ALGUA_ALPACA_API_SECRET"
)
_LIVE_MSG = (
    "Alpaca LIVE credentials not configured; set ALGUA_ALPACA_LIVE_API_KEY "
    "and ALGUA_ALPACA_LIVE_API_SECRET"
)
_LIVE_READONLY_MSG = (
    "Alpaca LIVE credentials not configured; the read-only live broker is unavailable — "
    "set ALGUA_ALPACA_LIVE_API_KEY and ALGUA_ALPACA_LIVE_API_SECRET"
)


def _live_auth() -> LiveAuthorization:
    return LiveAuthorization(strategy_id=1, code_hash="c", config_hash="cf",
                             dependency_hash="d", principal="lior", authorized_at="2026-06-05")


def _clear_paper(monkeypatch):
    monkeypatch.delenv("ALGUA_ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALGUA_ALPACA_API_SECRET", raising=False)


def _clear_live(monkeypatch):
    monkeypatch.delenv("ALGUA_ALPACA_LIVE_API_KEY", raising=False)
    monkeypatch.delenv("ALGUA_ALPACA_LIVE_API_SECRET", raising=False)


def _set_paper(monkeypatch):
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")


def _set_live(monkeypatch):
    monkeypatch.setenv("ALGUA_ALPACA_LIVE_API_KEY", "lk")
    monkeypatch.setenv("ALGUA_ALPACA_LIVE_API_SECRET", "ls")


# -- construction with credentials present: each kind builds its real concrete class -----------


def test_build_paper_broker(monkeypatch):
    _set_paper(monkeypatch)
    broker = build_broker(BrokerKind.ALPACA_PAPER)
    assert isinstance(broker, AlpacaPaperBroker)
    assert broker.base_url == "https://paper-api.alpaca.markets"


def test_build_live_broker(monkeypatch):
    _set_live(monkeypatch)
    auth = _live_auth()
    broker = build_broker(BrokerKind.ALPACA_LIVE, auth)
    assert isinstance(broker, AlpacaLiveBroker)
    assert broker.authorization is auth


def test_build_live_readonly_broker(monkeypatch):
    _set_live(monkeypatch)
    broker = build_broker(BrokerKind.ALPACA_LIVE_READONLY)
    assert isinstance(broker, AlpacaLiveReadOnlyBroker)


def test_build_live_drain_broker(monkeypatch):
    _set_live(monkeypatch)
    broker = build_broker(BrokerKind.ALPACA_LIVE_DRAIN)
    assert isinstance(broker, AlpacaLiveDrainBroker)


# -- build_broker raises with the venue's exact message when credentials are absent -------------


def test_build_paper_broker_missing_credentials_raises(monkeypatch):
    _clear_paper(monkeypatch)
    with pytest.raises(ValueError, match=r"^Alpaca paper credentials not configured"):
        build_broker(BrokerKind.ALPACA_PAPER)
    with pytest.raises(ValueError) as exc:
        build_broker(BrokerKind.ALPACA_PAPER)
    assert str(exc.value) == _PAPER_MSG


def test_build_live_broker_missing_credentials_raises(monkeypatch):
    _clear_live(monkeypatch)
    with pytest.raises(ValueError) as exc:
        build_broker(BrokerKind.ALPACA_LIVE, _live_auth())
    assert str(exc.value) == _LIVE_MSG


def test_build_live_readonly_missing_credentials_raises(monkeypatch):
    _clear_live(monkeypatch)
    with pytest.raises(ValueError) as exc:
        build_broker(BrokerKind.ALPACA_LIVE_READONLY)
    assert str(exc.value) == _LIVE_READONLY_MSG


def test_build_live_drain_missing_credentials_raises(monkeypatch):
    _clear_live(monkeypatch)
    with pytest.raises(ValueError) as exc:
        build_broker(BrokerKind.ALPACA_LIVE_DRAIN)
    assert str(exc.value) == _LIVE_MSG


# -- maybe_broker: same kind, same absent credentials, opposite outcome -------------------------


def test_maybe_paper_broker_returns_none_when_credentials_absent(monkeypatch):
    _clear_paper(monkeypatch)
    assert maybe_broker(BrokerKind.ALPACA_PAPER) is None


def test_maybe_live_readonly_returns_none_when_credentials_absent(monkeypatch):
    _clear_live(monkeypatch)
    assert maybe_broker(BrokerKind.ALPACA_LIVE_READONLY) is None


def test_maybe_live_drain_returns_none_when_credentials_absent(monkeypatch):
    _clear_live(monkeypatch)
    assert maybe_broker(BrokerKind.ALPACA_LIVE_DRAIN) is None


def test_maybe_broker_constructs_when_credentials_present(monkeypatch):
    _set_paper(monkeypatch)
    broker = maybe_broker(BrokerKind.ALPACA_PAPER)
    assert isinstance(broker, AlpacaPaperBroker)


# -- unknown kind fails closed; no default anywhere in the lookup path --------------------------


def test_build_broker_unknown_kind_raises_with_valid_set(monkeypatch):
    _set_paper(monkeypatch)
    with pytest.raises(ValueError, match="unknown broker kind") as exc:
        build_broker("totally_not_a_kind")
    for kind in BrokerKind:
        assert kind.value in str(exc.value)


def test_maybe_broker_unknown_kind_raises_never_returns_none(monkeypatch):
    # The safety-critical case: an UNKNOWN venue must raise, never silently resolve to None --
    # conflating "venue doesn't exist" with "credentials not configured" is exactly the ambiguity
    # this factory must not introduce.
    with pytest.raises(ValueError, match="unknown broker kind"):
        maybe_broker("totally_not_a_kind")


def test_no_default_for_unregistered_kind():
    # Prove there is no fallback branch: strip every entry out of the live registry and confirm
    # EVERY known kind now fails closed rather than resolving to some default spec.
    saved = dict(_REGISTRY)
    _REGISTRY.clear()
    try:
        for kind in BrokerKind:
            with pytest.raises(ValueError, match="unknown broker kind"):
                build_broker(kind)
            with pytest.raises(ValueError, match="unknown broker kind"):
                maybe_broker(kind)
    finally:
        _REGISTRY.clear()
        _REGISTRY.update(saved)


# -- AlpacaLiveBroker's LiveAuthorization tollbooth is not bypassable through the factory --------


def test_build_live_broker_without_authorization_fails(monkeypatch):
    _set_live(monkeypatch)
    # A non-LiveAuthorization value in the authorization slot must trip the tollbooth -- never
    # construct an unauthorized live (real-money) broker.
    with pytest.raises(BrokerError, match="LiveAuthorization"):
        build_broker(BrokerKind.ALPACA_LIVE, None)


def test_build_live_broker_omitted_authorization_fails(monkeypatch):
    _set_live(monkeypatch)
    # Omitting the positional entirely shifts every downstream arg (key lands where authorization
    # should be) and must still fail outright, never silently construct a broker.
    with pytest.raises(TypeError):
        build_broker(BrokerKind.ALPACA_LIVE)


# -- isolation proof: the autouse registry fixture (tests/conftest.py) must restore this registry
# between tests, not just for this file's own local save/restore in test_no_default_for_
# unregistered_kind above. This is an ORDERED PAIR by construction -- pytest runs tests within a
# file in source order, so the first test corrupts BrokerKind.ALPACA_PAPER's spec and the second
# proves it came back clean. A fixture that silently did nothing would let the corruption leak
# forward and the second test would build the FAKE class instead of the real one. -----------------


class _FakeSpecBroker:
    """A deliberately wrong broker class. If a spec built from this ever leaks into a later test,
    that test is constructing the wrong thing across the paper/live boundary."""


def test_isolation_proof_step1_corrupts_the_paper_registry_entry(monkeypatch):
    _set_paper(monkeypatch)
    fake_spec = BrokerSpec(
        key_field="alpaca_api_key",
        secret_field="alpaca_api_secret",
        url_field="alpaca_paper_url",
        construct=lambda *args, **kwargs: _FakeSpecBroker(),
        missing_credentials=_PAPER_MSG,
    )
    register_broker(BrokerKind.ALPACA_PAPER, fake_spec)
    # Confirm the corruption actually took: this test alone proves nothing about isolation.
    assert isinstance(build_broker(BrokerKind.ALPACA_PAPER), _FakeSpecBroker)


def test_isolation_proof_step2_next_test_sees_the_real_spec_restored(monkeypatch):
    _set_paper(monkeypatch)
    # If the autouse registry-isolation fixture did not restore _REGISTRY after the previous test,
    # this would build _FakeSpecBroker instead -- and this assertion would fail.
    broker = build_broker(BrokerKind.ALPACA_PAPER)
    assert isinstance(broker, AlpacaPaperBroker)
