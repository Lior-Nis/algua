"""push.py: state-dir hygiene, VAPID keygen, subscription store, send_to_all."""

from __future__ import annotations

import json
import os
import stat
from types import SimpleNamespace
from typing import Any

import pytest
from backend import push
from backend.params import InvalidParam
from py_vapid import b64urldecode
from pywebpush import WebPushException


def _sub(i: int = 0) -> dict[str, Any]:
    return {
        "endpoint": f"https://push.example.com/v1/{i}",
        "keys": {"p256dh": f"p256dh-{i}", "auth": f"auth-{i}"},
        "expirationTime": None,  # extra browser fields must not break validation
    }


def _mode(path: Any) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


# --- VAPID ---


def test_vapid_keygen_is_idempotent() -> None:
    first = push.vapid_public_key()
    second = push.vapid_public_key()
    assert first == second
    # A valid P-256 application-server key: 65 raw bytes, uncompressed-point marker.
    raw = b64urldecode(first.encode())
    assert len(raw) == 65
    assert raw[0] == 0x04


def test_state_dir_and_files_are_private() -> None:
    push.vapid_public_key()
    push.add_subscription(_sub())
    push.save_notify_state({"strategies": {}, "global_halt": {}, "unhealthy": False})
    directory = push.state_dir()
    assert _mode(directory) == 0o700
    for name in ("vapid_private.pem", "subscriptions.json", "notify_state.json"):
        assert _mode(directory / name) == 0o600


# --- subscription store ---


def test_add_and_remove_subscription() -> None:
    sub = _sub()
    push.add_subscription(sub)
    stored = push.load_subscriptions()
    assert stored[sub["endpoint"]]["keys"] == sub["keys"]
    push.remove_subscription(sub["endpoint"])
    assert push.load_subscriptions() == {}
    push.remove_subscription(sub["endpoint"])  # idempotent
    assert push.load_subscriptions() == {}


@pytest.mark.parametrize(
    "bad",
    [
        "not a dict",
        ["endpoint"],
        {"keys": {"p256dh": "p", "auth": "a"}},  # no endpoint
        {"endpoint": "http://insecure.example.com/x", "keys": {"p256dh": "p", "auth": "a"}},
        {"endpoint": "https://" + "e" * 2048, "keys": {"p256dh": "p", "auth": "a"}},  # oversize
        {"endpoint": "https://push.example.com/x"},  # no keys
        {"endpoint": "https://push.example.com/x", "keys": {"auth": "a"}},  # no p256dh
        {"endpoint": "https://push.example.com/x", "keys": {"p256dh": "p" * 513, "auth": "a"}},
        {"endpoint": "https://push.example.com/x", "keys": {"p256dh": "p", "auth": 7}},
    ],
)
def test_add_subscription_validation_rejects(bad: Any) -> None:
    with pytest.raises(InvalidParam):
        push.add_subscription(bad)
    assert push.load_subscriptions() == {}


def test_subscription_limit_rejects_the_17th_but_allows_resubscribe() -> None:
    for i in range(push.MAX_SUBSCRIPTIONS):
        push.add_subscription(_sub(i))
    with pytest.raises(push.SubscriptionLimit):
        push.add_subscription(_sub(99))
    # An ALREADY-KNOWN endpoint may refresh itself at the cap (no eviction).
    push.add_subscription(_sub(0))
    assert len(push.load_subscriptions()) == push.MAX_SUBSCRIPTIONS


# --- send_to_all ---


def _fake_webpush(
    monkeypatch: pytest.MonkeyPatch, failing: dict[str, Exception] | None = None
) -> list[dict[str, Any]]:
    """Fake pywebpush.webpush at the push.py boundary; records every call's kwargs."""
    calls: list[dict[str, Any]] = []
    failing = failing or {}

    def fake(**kwargs: Any) -> None:
        calls.append(kwargs)
        endpoint = kwargs["subscription_info"]["endpoint"]
        if endpoint in failing:
            raise failing[endpoint]

    monkeypatch.setattr(push, "webpush", fake)
    return calls


async def test_send_to_all_passes_ttl_timeout_urgency_and_counts_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    push.add_subscription(_sub(0))
    push.add_subscription(_sub(1))
    calls = _fake_webpush(monkeypatch)
    accepted = await push.send_to_all({"title": "x"}, urgency="high")
    assert accepted == 2
    for call in calls:
        assert call["ttl"] == 86400
        assert call["timeout"] == 10.0
        assert call["headers"] == {"Urgency": "high"}
        assert call["vapid_claims"] == {"sub": "mailto:lior3226@gmail.com"}
        assert json.loads(call["data"]) == {"title": "x"}


async def test_send_to_all_prunes_gone_subscription_on_410(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    push.add_subscription(_sub(0))
    push.add_subscription(_sub(1))
    gone = WebPushException("gone", response=SimpleNamespace(status_code=410))
    _fake_webpush(monkeypatch, failing={_sub(0)["endpoint"]: gone})
    accepted = await push.send_to_all({"title": "x"})
    assert accepted == 1
    remaining = push.load_subscriptions()
    assert list(remaining) == [_sub(1)["endpoint"]]


async def test_send_to_all_transient_failure_counts_zero_but_keeps_subscription(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    push.add_subscription(_sub(0))
    push.add_subscription(_sub(1))
    flaky = WebPushException("boom", response=SimpleNamespace(status_code=500))
    _fake_webpush(monkeypatch, failing={_sub(0)["endpoint"]: flaky})
    accepted = await push.send_to_all({"title": "x"})
    assert accepted == 1  # 2 subs, one fails transiently -> count 1
    assert len(push.load_subscriptions()) == 2  # transient failures never prune


async def test_send_to_all_without_subscriptions_is_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _fake_webpush(monkeypatch)
    assert await push.send_to_all({"title": "x"}) == 0
    assert calls == []
