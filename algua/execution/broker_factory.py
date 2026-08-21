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
from enum import StrEnum
from typing import Any

from algua.config.settings import get_settings
from algua.execution.alpaca_broker import (
    AlpacaLiveBroker,
    AlpacaLiveDrainBroker,
    AlpacaLiveReadOnlyBroker,
    AlpacaPaperBroker,
)


class BrokerKind(StrEnum):
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
    is ``Any`` because the four concrete classes share no public base; each named delegate
    re-asserts the concrete type, and mypy checks the delegate bodies against their declared
    returns.
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


def _construct_paper(key: str, secret: str, base_url: str) -> AlpacaPaperBroker:
    # AlpacaPaperBroker takes keyword api_key=/api_secret=, unlike the three LIVE-family classes
    # below (positional key/secret) -- normalized here rather than changing the class.
    return AlpacaPaperBroker(api_key=key, api_secret=secret, base_url=base_url)


register_broker(BrokerKind.ALPACA_PAPER, BrokerSpec(
    key_field="alpaca_api_key",
    secret_field="alpaca_api_secret",
    url_field="alpaca_paper_url",
    construct=_construct_paper,
    missing_credentials=(
        "Alpaca paper credentials not configured; set ALGUA_ALPACA_API_KEY "
        "and ALGUA_ALPACA_API_SECRET"
    ),
))

register_broker(BrokerKind.ALPACA_LIVE, BrokerSpec(
    key_field="alpaca_live_api_key",
    secret_field="alpaca_live_api_secret",
    url_field="alpaca_live_url",
    construct=AlpacaLiveBroker,
    missing_credentials=(
        "Alpaca LIVE credentials not configured; set ALGUA_ALPACA_LIVE_API_KEY "
        "and ALGUA_ALPACA_LIVE_API_SECRET"
    ),
))

register_broker(BrokerKind.ALPACA_LIVE_READONLY, BrokerSpec(
    key_field="alpaca_live_api_key",
    secret_field="alpaca_live_api_secret",
    url_field="alpaca_live_url",
    construct=AlpacaLiveReadOnlyBroker,
    missing_credentials=(
        "Alpaca LIVE credentials not configured; cannot confirm the strategy is flat at the "
        "broker — set ALGUA_ALPACA_LIVE_API_KEY and ALGUA_ALPACA_LIVE_API_SECRET"
    ),
))

register_broker(BrokerKind.ALPACA_LIVE_DRAIN, BrokerSpec(
    key_field="alpaca_live_api_key",
    secret_field="alpaca_live_api_secret",
    url_field="alpaca_live_url",
    construct=AlpacaLiveDrainBroker,
    # build_live_drain_broker (lane_exit.py) never raises -- it only returns None on missing
    # credentials, so the caller can fail closed on the exit path -- but build_broker's contract is
    # to raise here, and the underlying credential requirement is identical to ALPACA_LIVE's (same
    # env vars), so this reuses that message verbatim rather than inventing new wording.
    missing_credentials=(
        "Alpaca LIVE credentials not configured; set ALGUA_ALPACA_LIVE_API_KEY "
        "and ALGUA_ALPACA_LIVE_API_SECRET"
    ),
))
