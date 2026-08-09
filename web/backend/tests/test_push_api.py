"""/api/push endpoints: key shape, subscribe validation/limit, idempotent delete."""

from __future__ import annotations

from typing import Any

import backend.main as main_mod
import httpx
from backend import push
from py_vapid import b64urldecode


def _client() -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=main_mod.app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


def _sub(i: int = 0) -> dict[str, Any]:
    return {
        "endpoint": f"https://push.example.com/v1/{i}",
        "keys": {"p256dh": f"p256dh-{i}", "auth": f"auth-{i}"},
    }


async def test_push_key_shape() -> None:
    async with _client() as client:
        resp = await client.get("/api/push/key")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    raw = b64urldecode(body["key"].encode())
    assert len(raw) == 65 and raw[0] == 0x04  # uncompressed P-256 point


async def test_subscribe_persists() -> None:
    async with _client() as client:
        resp = await client.post("/api/push/subscribe", json=_sub())
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
    assert _sub()["endpoint"] in push.load_subscriptions()


async def test_subscribe_invalid_is_422_invalid_param() -> None:
    async with _client() as client:
        resp = await client.post(
            "/api/push/subscribe",
            json={
                "endpoint": "http://insecure.example.com/x",
                "keys": {"p256dh": "p", "auth": "a"},
            },
        )
    assert resp.status_code == 422
    body = resp.json()
    assert body["ok"] is False
    assert body["code"] == "invalid_param"
    assert push.load_subscriptions() == {}


async def test_subscribe_non_object_body_is_422() -> None:
    async with _client() as client:
        resp = await client.post("/api/push/subscribe", json=[1, 2, 3])
    assert resp.status_code == 422
    assert resp.json()["code"] == "invalid_param"


async def test_subscribe_limit_is_422_subscription_limit() -> None:
    for i in range(push.MAX_SUBSCRIPTIONS):
        push.add_subscription(_sub(i))
    async with _client() as client:
        resp = await client.post("/api/push/subscribe", json=_sub(99))
    assert resp.status_code == 422
    body = resp.json()
    assert body["ok"] is False
    assert body["code"] == "subscription_limit"


async def test_delete_is_idempotent() -> None:
    push.add_subscription(_sub())
    async with _client() as client:
        first = await client.delete(
            "/api/push/subscribe", params={"endpoint": _sub()["endpoint"]}
        )
        second = await client.delete(
            "/api/push/subscribe", params={"endpoint": _sub()["endpoint"]}
        )
    assert first.status_code == 200 and first.json() == {"ok": True}
    assert second.status_code == 200 and second.json() == {"ok": True}
    assert push.load_subscriptions() == {}
