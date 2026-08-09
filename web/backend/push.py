"""Web-push delivery + on-disk push state (slice E).

All push state lives in one 0700 directory (``ALGUA_WEB_STATE``, default
``web/.state``): ``vapid_private.pem``, ``subscriptions.json`` and
``notify_state.json``, each 0600 and written atomically (tmp file + fsync +
``os.replace``) so a crash mid-write never corrupts state (R9).

Delivery goes through pywebpush in a worker thread. A 404/410 push-service
response means the subscription is gone — it is pruned; any other failure is
logged and skipped. :func:`send_to_all` returns the count of ACCEPTED sends
(R5) so the caller can decide whether a notification actually landed.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from cryptography.hazmat.primitives import serialization
from py_vapid import Vapid02, b64urlencode
from pywebpush import WebPushException, webpush

from backend.params import InvalidParam

logger = logging.getLogger(__name__)

# web/backend/push.py -> web/backend -> web; state defaults to web/.state.
_DEFAULT_STATE = Path(__file__).resolve().parents[1] / ".state"

VAPID_CLAIMS_SUB = "mailto:lior3226@gmail.com"
MAX_SUBSCRIPTIONS = 16  # R12: one household of devices, not a public service
_MAX_ENDPOINT_CHARS = 2048
_MAX_KEY_CHARS = 512
_TTL_S = 86400  # R11: a day-old fleet alert is still worth delivering after a dead radio
_SEND_TIMEOUT_S = 10.0  # R10: a wedged push service must not stall the poller's thread


class SubscriptionLimit(Exception):
    """The subscription store is full (rendered as HTTP 422 ``subscription_limit``)."""


def state_dir() -> Path:
    """The push-state directory, created — and ALWAYS pinned — 0700 (R9)."""
    override = os.environ.get("ALGUA_WEB_STATE")
    directory = Path(override) if override else _DEFAULT_STATE
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    # Pin on EVERY use, not only creation: mkdir's mode is umask-filtered and a
    # pre-existing lax directory (0755) must not leave the key world-listable.
    directory.chmod(0o700)
    return directory


def _write_atomic(path: Path, data: bytes) -> None:
    """0600 tmp file + flush + fsync + os.replace + parent-dir fsync.

    The directory fsync makes the RENAME itself durable — without it a power
    loss right after os.replace can roll the directory entry back to the old
    (or no) file even though the data blocks were fsynced.
    """
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.")
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        dir_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


# --- VAPID ---

# Guards first-use keygen: two concurrent first calls must not each generate a
# keypair and race the os.replace (the loser's key would already be handed to a
# browser). Single-process uvicorn -> a process-local lock suffices.
_vapid_lock = threading.Lock()


def _vapid() -> Vapid02:
    """The server's VAPID keypair, generated on first use and persisted 0600."""
    path = state_dir() / "vapid_private.pem"
    if path.is_file():
        return Vapid02.from_file(str(path))
    with _vapid_lock:
        if path.is_file():  # re-check: another thread may have won the keygen
            return Vapid02.from_file(str(path))
        vapid = Vapid02()
        vapid.generate_keys()
        _write_atomic(path, vapid.private_pem())
        return vapid


def vapid_public_key() -> str:
    """The base64url application-server key browsers pass to ``pushManager.subscribe``."""
    raw = _vapid().public_key.public_bytes(
        serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint
    )
    return str(b64urlencode(raw))


# --- subscription store ---

# Guards EVERY subscription load-modify-write (add / remove / 410-prune): the
# store is one JSON file, so unsynchronized mutations lose writes. The server
# is single-process uvicorn, so a process-local lock suffices.
_subscriptions_lock = threading.Lock()


def _subscriptions_path() -> Path:
    return state_dir() / "subscriptions.json"


def load_subscriptions() -> dict[str, dict[str, Any]]:
    """endpoint -> subscription dict; unreadable/missing file is an empty store."""
    try:
        data = json.loads(_subscriptions_path().read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError):
        logger.warning("push: unreadable subscriptions.json; treating as empty")
        return {}
    return data if isinstance(data, dict) else {}


def _save_subscriptions(subs: dict[str, dict[str, Any]]) -> None:
    _write_atomic(_subscriptions_path(), json.dumps(subs, indent=2).encode())


def add_subscription(sub: Any) -> None:
    """Validate (R12) and persist one browser PushSubscription JSON.

    Raises :class:`backend.params.InvalidParam` on a malformed subscription and
    :class:`SubscriptionLimit` when a NEW endpoint would exceed the cap (an
    existing endpoint may always re-subscribe — no eviction, reject instead).
    """
    if not isinstance(sub, dict):
        raise InvalidParam("subscription must be a JSON object")
    endpoint = sub.get("endpoint")
    if not isinstance(endpoint, str) or not endpoint or len(endpoint) > _MAX_ENDPOINT_CHARS:
        raise InvalidParam(f"endpoint must be a string of 1..{_MAX_ENDPOINT_CHARS} chars")
    parsed = urlparse(endpoint)
    if parsed.scheme != "https" or not parsed.netloc:
        raise InvalidParam("endpoint must be an https URL")
    keys = sub.get("keys")
    if not isinstance(keys, dict):
        raise InvalidParam("subscription must carry a keys object")
    clean_keys: dict[str, str] = {}
    for name in ("p256dh", "auth"):
        value = keys.get(name)
        if not isinstance(value, str) or not value or len(value) > _MAX_KEY_CHARS:
            raise InvalidParam(f"keys.{name} must be a string of 1..{_MAX_KEY_CHARS} chars")
        clean_keys[name] = value
    with _subscriptions_lock:
        subs = load_subscriptions()
        if endpoint not in subs and len(subs) >= MAX_SUBSCRIPTIONS:
            raise SubscriptionLimit(f"subscription limit ({MAX_SUBSCRIPTIONS}) reached")
        subs[endpoint] = {"endpoint": endpoint, "keys": clean_keys}
        _save_subscriptions(subs)


def remove_subscription(endpoint: str) -> None:
    """Idempotent removal by endpoint."""
    with _subscriptions_lock:
        subs = load_subscriptions()
        if subs.pop(endpoint, None) is not None:
            _save_subscriptions(subs)


# --- notify state (the poller's diff memory; schema owned by backend.poller) ---


def _notify_state_path() -> Path:
    return state_dir() / "notify_state.json"


def load_notify_state() -> dict[str, Any] | None:
    """The persisted diff state, or ``None`` when never written — the baseline signal (R2).

    An unreadable file also re-baselines: observed state is rebuilt on the next
    fresh poll and nothing is notified off corrupt memory.
    """
    try:
        data = json.loads(_notify_state_path().read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError):
        logger.warning("push: unreadable notify_state.json; re-baselining")
        return None
    return data if isinstance(data, dict) else None


def save_notify_state(state: dict[str, Any]) -> None:
    _write_atomic(_notify_state_path(), json.dumps(state, indent=2).encode())


# --- delivery ---


def _send_one(sub: dict[str, Any], data: str, vapid: Vapid02, urgency: str) -> None:
    webpush(
        subscription_info=sub,
        data=data,
        vapid_private_key=vapid,
        # Fresh dict per send: pywebpush MUTATES vapid_claims (fills aud/exp).
        vapid_claims={"sub": VAPID_CLAIMS_SUB},
        ttl=_TTL_S,
        timeout=_SEND_TIMEOUT_S,
        headers={"Urgency": urgency},
    )


async def send_to_all(payload: dict[str, Any], *, urgency: str = "normal") -> int:
    """Send ``payload`` to every subscription; returns the ACCEPTED count (R5).

    A 404/410 response prunes that subscription (gone at the push service);
    any other failure logs a warning and moves on — one dead endpoint never
    blocks the rest.
    """
    subs = load_subscriptions()
    if not subs:
        return 0
    vapid = _vapid()
    data = json.dumps(payload)
    accepted = 0
    dead: list[str] = []
    for endpoint, sub in subs.items():
        try:
            await asyncio.to_thread(_send_one, sub, data, vapid, urgency)
        except WebPushException as exc:
            status = getattr(exc.response, "status_code", None)
            if status in (404, 410):
                dead.append(endpoint)
            else:
                logger.warning("push: send failed (status %s): %s", status, exc)
        except Exception:
            logger.warning("push: send failed", exc_info=True)
        else:
            accepted += 1
    for endpoint in dead:
        logger.info("push: pruning gone subscription %s", endpoint[:64])
        remove_subscription(endpoint)
    return accepted
