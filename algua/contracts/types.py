from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:  # keep contracts import-light; pandas only needed for typing
    import pandas as pd


class Side(StrEnum):
    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True)
class CapacityLimit:
    """An ADV / participation capacity budget for POSITION sizing (issue #344).

    Bar `volume` is in the schema but was never consulted for sizing, so a strategy could size a
    position into an untradeable fraction of a name's liquidity. This caps each held position at
    `max_participation_rate` of the name's trailing dollar-ADV, evaluated at a DECLARED
    `reference_aum` (the AUM-breakeven capital the capacity is assessed at — the backtest itself is
    unitless `targetpercent`, so a reference notional is required to turn a weight into dollars).

    Applied in `LoadedStrategy.construct` (see algua/portfolio/construction.apply_capacity_cap), so
    it composes with the existing weight->position pipeline and applies identically in the backtest
    loop, the vectorized fast path, and live/paper sizing. Folded into `config_hash` (part of
    strategy identity). This is a POSITION cap, not a per-trade order-delta participation cap.
    """

    reference_aum: float          # declared capital the capacity is evaluated at (dollars)
    max_participation_rate: float # max fraction of a name's ADV one position may occupy
    adv_window_bars: int          # trailing window length (bars) for the dollar-ADV estimate

    def __post_init__(self) -> None:
        # Fail closed on every neutering value, mirroring ExecutionContract's own guards: a
        # non-finite / non-positive reference_aum, an out-of-range rate, or a sub-1 window would
        # silently disable or corrupt the cap.
        if not math.isfinite(self.reference_aum) or self.reference_aum <= 0:
            raise ValueError("reference_aum must be a finite number > 0")
        if not math.isfinite(self.max_participation_rate):
            raise ValueError("max_participation_rate must be finite")
        if not 0.0 < self.max_participation_rate <= 1.0:
            raise ValueError("max_participation_rate must be in (0, 1]")
        # bool is an int subtype; reject it so True can't masquerade as a window of 1.
        if isinstance(self.adv_window_bars, bool) or not isinstance(self.adv_window_bars, int):
            raise ValueError("adv_window_bars must be an int")
        if self.adv_window_bars < 1:
            raise ValueError("adv_window_bars must be >= 1")


@dataclass(frozen=True)
class ExecutionContract:
    """How target weights become executable orders. Pinned per strategy.

    decision_lag_bars >= 1 enforces the t -> t+1 rule: features are computed on a
    fully closed bar t and orders may fill no earlier than t + lag. This forbids
    same-bar fills, the single most likely source of look-ahead bias.
    """

    rebalance_frequency: str
    decision_lag_bars: int = 1
    allow_fractional: bool = True
    max_gross_exposure: float = 1.0
    max_weight_per_symbol: float = 1.0  # cap on |weight| per symbol; 1.0 = no cap
    allow_short: bool = False           # False = long-only (today's behavior)
    warmup_bars: int = 0
    # Optional ADV / participation capacity budget (issue #344). None = no capacity cap (default),
    # so existing strategies are byte-unchanged. Last field: the whole codebase builds
    # ExecutionContract with keyword args, so appending here is safe.
    capacity: CapacityLimit | None = None

    def __post_init__(self) -> None:
        if self.decision_lag_bars < 1:
            raise ValueError("decision_lag_bars must be >= 1 (no same-bar fills)")
        if self.warmup_bars < 0:
            raise ValueError("warmup_bars must be >= 0")
        if not math.isfinite(self.max_weight_per_symbol):
            # A non-finite cap silently disables the rail: every `|w| > nan` / `> inf`
            # comparison is false, so a fallible/injectable agent could neuter the cap by
            # declaring nan. Fail closed (codex GATE 2 HIGH).
            raise ValueError("max_weight_per_symbol must be finite")
        if self.max_weight_per_symbol <= 0:
            raise ValueError("max_weight_per_symbol must be > 0")
        if not isinstance(self.allow_short, bool):
            # A truthy non-bool (e.g. the string "false") would silently enable shorts.
            raise ValueError("allow_short must be a bool")
        if self.capacity is not None and not isinstance(self.capacity, CapacityLimit):
            raise ValueError("capacity must be a CapacityLimit or None")


@dataclass(frozen=True)
class OrderIntent:
    symbol: str
    side: Side
    target_weight: float
    decision_ts: datetime


@runtime_checkable
class Strategy(Protocol):
    """`target_weights` is the composed pipeline `construct(signal(features), features)` — see
    `algua/strategies/base.py` and `algua/portfolio/construction.py` (issue #141)."""

    name: str
    execution: ExecutionContract

    def target_weights(
        self,
        features: pd.DataFrame,
        fundamentals: pd.DataFrame | None = None,
        news: pd.DataFrame | None = None,
    ) -> pd.Series: ...


# --- Non-tabular: fundamentals seam (issue #132) -----------------------------------------------
# Canonical column names live HERE (the base layer the engine may import) so the backtest engine's
# as-of mask and the data-layer validator share one source of truth without the engine importing
# algua.data (which the import wall forbids). Pure strings — no pandas needed.
FUNDAMENTALS_COLUMNS: tuple[str, ...] = (
    "symbol", "fiscal_period_end", "metric", "value", "knowable_at", "source",
)
# The identity of a fact across revisions: a restatement is a new row sharing this key with a later
# knowable_at. The as-of mask keeps, per key, the row with the greatest knowable_at <= t.
FUNDAMENTALS_AS_OF_KEY: tuple[str, ...] = ("symbol", "fiscal_period_end", "metric")
FUNDAMENTALS_KNOWABLE_AT = "knowable_at"

# --- Non-tabular: news seam (issue #132, hindsight slice) --------------------------------------
# Tidy/long bitemporal news, one row per (article, mentioned symbol). `source` is part of the
# identity because an article id is only unique WITHIN a source. Hindsight-only this slice (no
# engine consumer), but the names live here beside the fundamentals constants for symmetry.
NEWS_COLUMNS: tuple[str, ...] = (
    "source", "article_id", "symbol", "published_at", "knowable_at",
    "headline", "url", "body", "retracted",
)
# Identity of an article-mention across revisions: a correction is a new row sharing this key with
# a later knowable_at. `source` scopes the (source-local) article_id.
NEWS_AS_OF_KEY: tuple[str, ...] = ("source", "article_id", "symbol")
NEWS_KNOWABLE_AT = "knowable_at"
NEWS_RETRACTED = "retracted"  # True = a symbol-mention retracted by a later article revision


@runtime_checkable
class FundamentalsProvider(Protocol):
    """As-of consumption seam for point-in-time fundamentals (issue #132). Returns the FULL
    bitemporal history for `symbols` with knowable_at < end — no lower time bound, since the first
    decision bar needs the latest prior report. The engine owns decision `t` and masks
    knowable_at <= t per bar; the provider never sees `t`."""

    snapshot_id: str

    def get_fundamentals(self, symbols: list[str], end: datetime) -> pd.DataFrame: ...


@runtime_checkable
class NewsProvider(Protocol):
    """As-of consumption seam for point-in-time news (issue #132). Returns the FULL bitemporal
    history (including retraction tombstones) for `symbols` with knowable_at < end — no lower
    bound. The engine owns decision `t` and masks knowable_at <= t per bar; the provider never
    sees `t`."""

    snapshot_id: str

    def get_news(self, symbols: list[str], end: datetime) -> pd.DataFrame: ...


@runtime_checkable
class DataProvider(Protocol):
    def get_bars(
        self, symbols: list[str], start: datetime, end: datetime, timeframe: str
    ) -> pd.DataFrame: ...


@runtime_checkable
class Broker(Protocol):
    """The substitutable surface both brokers (sim + Alpaca paper) implement.

    `submit` sizes ONE OrderIntent against an equity snapshot the broker holds and returns a
    broker order id; a delta below the broker's minimum-notional threshold returns the string
    "noop" and submits nothing. Beyond this surface each broker exposes its own driving methods
    (the sim's equity()/fill_pending() replay hooks; Alpaca's account()/cancel_open_orders() HTTP
    calls), so the loops that need those are typed against the concrete class, not this protocol.
    """

    def get_positions(self) -> pd.Series: ...

    def submit(self, intent: OrderIntent) -> str: ...


@dataclass(frozen=True)
class LiveAuthorization:
    """A token representing a human go-live authorization, returned by
    `algua.registry.live_gate.verify_live_authorization` after it re-verifies the signature.

    DEFENSE IN DEPTH, NOT THE WALL: in-process this dataclass is forgeable (any code can construct
    one), so requiring it to build `AlpacaLiveBroker` is a tollbooth that catches accidental
    unauthorized construction — it is NOT the primary control. The real wall is (a) the live API
    keys living only in the trusted operator env and (b) the live loop calling
    `verify_live_authorization` (re-verifying the SSH signature against the trust anchor)
    immediately before EVERY live order."""

    strategy_id: int
    code_hash: str
    config_hash: str
    dependency_hash: str | None
    principal: str
    authorized_at: str


@dataclass(frozen=True)
class PendingLiveAuthorization:
    """A go-live signature that has been verified but NOT yet recorded. It carries everything
    needed to consume the challenge + write the `live_authorizations` row, so those writes can be
    performed ATOMICALLY with the stage CAS in a single transaction (#254) instead of committed
    separately before the transition. `signature_b64` is the base64-encoded raw SSH signature."""

    nonce: str
    expires_at: str
    principal: str
    signature_b64: str
