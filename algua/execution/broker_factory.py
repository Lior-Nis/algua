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

DEVIATION FROM SPEC: the spec describes this seam as "name→factory registry + config field" (a
settings value selecting the broker). There is deliberately NO such field. The venue is selected by
the CALL SITE (the ``BrokerKind`` argument), never by config -- a config flag able to swap paper for
live at runtime would be a hazard, not a feature, on this boundary.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal, overload

from algua.config.settings import get_settings
from algua.contracts.types import LiveAuthorization
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


class MissingBrokerCredentials(ValueError):
    """Raised by ``build_broker`` when the resolved venue's credentials are not configured.

    A narrow subclass of ``ValueError`` rather than bare ``ValueError`` -- ``get_settings()`` runs
    INSIDE this factory too, and pydantic's ``ValidationError`` (e.g. a malformed
    ``ALGUA_ALPACA_PAPER_URL``, including the paper/live boundary guard rejecting a crossed host)
    also subclasses ``ValueError``. A caller that wants to translate JUST the absent-credentials
    case (not every ``ValueError`` a settings load can raise) needs a type narrow enough to
    distinguish them; subclassing ``ValueError`` (not replacing it) keeps every existing
    ``pytest.raises(ValueError)`` pin on this path working unchanged."""


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
        # `getattr(kind, "value", kind)`, not `kind.value` -- callers deliberately probe this path
        # with a plain str that isn't even a BrokerKind member (there's no runtime enforcement of
        # the type hint), and a bare `.value` would raise AttributeError on that instead of the
        # intended ValueError. For an actual BrokerKind member this renders its str value
        # ("alpaca_paper"), not `!r`'s enum repr ("<BrokerKind.ALPACA_PAPER: 'alpaca_paper'>").
        label = getattr(kind, "value", kind)
        valid = ", ".join(sorted(k.value for k in _REGISTRY)) or "<none registered>"
        raise ValueError(f"unknown broker kind {label!r}; valid: {valid}") from None
    s = get_settings()
    return (spec, getattr(s, spec.key_field), getattr(s, spec.secret_field),
            getattr(s, spec.url_field))


@overload
def build_broker(kind: Literal[BrokerKind.ALPACA_PAPER]) -> AlpacaPaperBroker: ...
@overload
def build_broker(
    kind: Literal[BrokerKind.ALPACA_LIVE], authorization: LiveAuthorization, /,
) -> AlpacaLiveBroker: ...
@overload
def build_broker(
    kind: Literal[BrokerKind.ALPACA_LIVE_READONLY],
) -> AlpacaLiveReadOnlyBroker: ...
@overload
def build_broker(kind: Literal[BrokerKind.ALPACA_LIVE_DRAIN]) -> AlpacaLiveDrainBroker: ...
def build_broker(kind: BrokerKind, *args: Any) -> Any:
    """Construct ``kind``, raising ``MissingBrokerCredentials`` if its credentials are not
    configured. ``MissingBrokerCredentials`` subclasses ``ValueError``, so an existing
    ``except ValueError`` / ``pytest.raises(ValueError)`` still catches it -- but a caller that
    wants to distinguish absent credentials from any OTHER ``ValueError`` this call can raise
    (notably pydantic's ``ValidationError`` from ``get_settings()``, which also subclasses
    ``ValueError``) can now catch the narrower type instead of swallowing both.

    ``*args`` is passed to the constructor BEFORE the credentials -- it carries
    ``AlpacaLiveBroker``'s ``LiveAuthorization`` tollbooth, which no other venue takes. The
    implementation returns ``Any`` (the four concrete classes share no public base), but the
    ``@overload`` signatures above -- one per ``BrokerKind`` literal, keyed on the exact venue --
    are what a CALL SITE actually resolves against: passing the wrong ``BrokerKind`` into a
    delegate declared to return a different concrete class is a mypy error, not a silent ``Any``
    (``warn_return_any`` is NOT enabled in ``[tool.mypy]``, so the bare ``-> Any`` implementation
    alone would NOT catch this -- verified by mutation: rewiring a paper-declared delegate to call
    ``build_broker(BrokerKind.ALPACA_LIVE_READONLY)`` passes mypy without the overloads and fails
    with them).
    """
    spec, key, secret, url = _resolve(kind)
    if not key or not secret:
        raise MissingBrokerCredentials(spec.missing_credentials)
    return spec.construct(*args, key, secret, base_url=url)


@overload
def maybe_broker(kind: Literal[BrokerKind.ALPACA_PAPER]) -> AlpacaPaperBroker | None: ...
@overload
def maybe_broker(
    kind: Literal[BrokerKind.ALPACA_LIVE], authorization: LiveAuthorization, /,
) -> AlpacaLiveBroker | None: ...
@overload
def maybe_broker(
    kind: Literal[BrokerKind.ALPACA_LIVE_READONLY],
) -> AlpacaLiveReadOnlyBroker | None: ...
@overload
def maybe_broker(
    kind: Literal[BrokerKind.ALPACA_LIVE_DRAIN],
) -> AlpacaLiveDrainBroker | None: ...
def maybe_broker(kind: BrokerKind, *args: Any) -> Any | None:
    """``build_broker``, but returns ``None`` when credentials are absent instead of raising.

    The two lenient call sites want this for OPPOSITE reasons -- one keeps going without a broker,
    the other refuses to proceed -- so the decision belongs to the caller and this returns None
    rather than encoding either. Same per-venue ``@overload`` treatment as ``build_broker``, for
    the same reason.
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
