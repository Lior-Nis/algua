from __future__ import annotations

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

    def __post_init__(self) -> None:
        if self.decision_lag_bars < 1:
            raise ValueError("decision_lag_bars must be >= 1 (no same-bar fills)")
        if self.warmup_bars < 0:
            raise ValueError("warmup_bars must be >= 0")
        if self.max_weight_per_symbol <= 0:
            raise ValueError("max_weight_per_symbol must be > 0")


@dataclass(frozen=True)
class OrderIntent:
    symbol: str
    side: Side
    target_weight: float
    decision_ts: datetime


@runtime_checkable
class Strategy(Protocol):
    name: str
    execution: ExecutionContract

    def target_weights(self, features: pd.DataFrame) -> pd.Series: ...


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
