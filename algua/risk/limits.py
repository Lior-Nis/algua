from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Collection

    from algua.contracts.types import ExecutionContract

# Single tolerance for "is this weight materially different from another / from a limit".
# One named constant rather than a scatter of bare 1e-9 / 1e-6 literals so both brokers and
# every risk check agree on what counts as a difference (#31). 1e-9 is comfortably tighter than
# any tradeable weight delta yet wide enough to absorb float rounding.
WEIGHT_TOL = 1e-9

# Explicit "drawdown breaker off" sentinel. Passing None disables the check, rather than
# overloading a magic max_drawdown >= 1.0 to mean "off" (#32).
DRAWDOWN_DISABLED: None = None


class RiskBreach(ValueError):
    """A hard risk-limit breach. Subclasses ValueError so existing CLI error handling
    (json_errors) still renders it; the CLI inspects `.kind` to trip the kill-switch."""

    def __init__(self, kind: str, detail: str) -> None:
        super().__init__(detail)
        self.kind = kind
        self.detail = detail


def check_gross_exposure(weights: pd.Series, max_gross: float) -> None:
    if len(weights) == 0:
        return
    gross = float(weights.abs().sum())
    if gross > max_gross + WEIGHT_TOL:
        raise RiskBreach(
            "gross_exposure",
            f"gross exposure {gross:.4f} exceeds max_gross_exposure {max_gross:.4f}",
        )


def check_finite_weights(weights: pd.Series, strategy_name: str) -> None:
    """Fail-closed guard against non-finite target weights. A strategy returning NaN/inf for a
    named symbol, a non-numeric weight, or a duplicated symbol index must HARD-BREACH, not be
    silently flattened by a downstream fillna(0.0) (NaN-skipping .sum() / `NaN < 0` would let it
    through). The panel fast-path's omitted-cell NaN is filled to flat BEFORE this runs, so its
    sparse-NaN-as-flat convention is preserved; only real non-finite VALUES reach here (#135)."""
    if len(weights) == 0:
        return
    if weights.index.isnull().any():
        raise RiskBreach(
            "non_finite_weight",
            f"strategy '{strategy_name}' returned a null symbol label",
        )
    if weights.index.has_duplicates:
        dups = sorted(set(weights.index[weights.index.duplicated(keep=False)]))
        raise RiskBreach(
            "non_finite_weight",
            f"strategy '{strategy_name}' returned duplicate symbol weight(s) for {dups}",
        )
    # bool is a numpy numeric subtype, so is_numeric_dtype accepts a bool Series and
    # np.isfinite(True) is True — a True weight would silently coerce to 1.0. Reject it.
    if pd.api.types.is_bool_dtype(weights) or not pd.api.types.is_numeric_dtype(weights):
        raise RiskBreach(
            "non_finite_weight",
            f"strategy '{strategy_name}' returned non-numeric target weights",
        )
    finite = np.isfinite(weights.to_numpy())
    if not bool(finite.all()):
        bad = sorted(weights.index[~finite])
        raise RiskBreach(
            "non_finite_weight",
            f"strategy '{strategy_name}' returned non-finite target weight(s) for {bad}",
        )


def check_universe_membership(
    weights: pd.Series, allowed_symbols: Collection[str], strategy_name: str
) -> None:
    """Reject any NONZERO target weight for a symbol outside the operating universe — the
    structural twin of the value checks. Mirrors the PIT loop's `w != 0.0` 'nonzero' semantics
    exactly: any nonzero weight for a non-member is a strategy bug (if numeric noise ever makes
    this too strict it can move to WEIGHT_TOL without changing the architecture).
    Offenders/allowed are rendered with `key=str` so a non-string symbol label cannot raise a
    bare TypeError that escapes the RiskBreach -> BacktestError / live-kill-switch contract.
    Empty `allowed_symbols` + any nonzero weight => every nonzero weight breaches (no allowed
    universe); a caller meaning "flat" must skip the call (as the PIT loop does via
    `if not members: continue`).

    Precondition: `check_finite_weights` runs first in `validate_decision_weights`, so NaN weights
    are already rejected as `non_finite_weight` before they could surface here as `out_of_universe`
    (NaN `!= 0.0` is True in pandas)."""
    if len(weights) == 0:
        return
    allowed = set(allowed_symbols)
    offenders = [s for s in weights.index[weights != 0.0] if s not in allowed]
    if offenders:
        raise RiskBreach(
            "out_of_universe",
            f"strategy '{strategy_name}' returned nonzero target weight(s) for out-of-universe "
            f"symbol(s) {sorted(offenders, key=str)} (allowed: {sorted(allowed, key=str)})",
        )


def check_max_weight_per_symbol(weights: pd.Series, max_per_symbol: float) -> None:
    """Single-name concentration cap: reject any |weight| above the per-symbol limit. Caps the
    LARGEST position, where gross caps the sum — an agent can pass gross with 100% in one name, so
    this is the rail that stops it. Absolute value, so it holds for shorts too (#135)."""
    if len(weights) == 0:
        return
    over = weights[weights.abs() > max_per_symbol + WEIGHT_TOL]
    if len(over):
        worst = sorted(over.index)
        raise RiskBreach(
            "max_weight_per_symbol",
            f"single-name weight(s) for {worst} exceed max_weight_per_symbol "
            f"{max_per_symbol:.4f}",
        )


def check_short_policy(weights: pd.Series, allow_short: bool, strategy_name: str) -> None:
    """Declared long/short gate. When allow_short is False (the default, long-only), any negative
    target weight hard-breaches; when True, shorts are permitted (the per-symbol cap still bounds
    |weight|). Replaces the old undeclared check_long_only: the constraint is now a hashed contract
    field, not an invisible convention (#135)."""
    if not allow_short and len(weights) and bool((weights < 0).any()):
        negative = sorted(weights[weights < 0].index)
        raise RiskBreach(
            "long_only",
            f"long-only: strategy '{strategy_name}' returned negative target weight(s) "
            f"for {negative}",
        )


def check_drawdown(equity: float, peak: float, max_drawdown: float | None) -> None:
    if max_drawdown is None or peak <= 0:
        return  # disabled (explicit None sentinel), or no peak yet
    if equity < peak * (1.0 - max_drawdown):
        dd = 1.0 - (equity / peak)
        raise RiskBreach(
            "drawdown",
            f"drawdown {dd:.4f} exceeds max_drawdown {max_drawdown:.4f} "
            f"(equity {equity:.2f}, peak {peak:.2f})",
        )


def validate_decision_weights(
    weights: pd.Series, contract: ExecutionContract, strategy_name: str
) -> None:
    """The ONE decision-weight validation every path calls (paper/live decide + backtest loop +
    fast-path), so the rails can never drift between research and live. Order: finite (fail-closed)
    -> short policy -> per-symbol cap -> gross exposure (#135)."""
    check_finite_weights(weights, strategy_name)
    check_short_policy(weights, contract.allow_short, strategy_name)
    check_max_weight_per_symbol(weights, contract.max_weight_per_symbol)
    check_gross_exposure(weights, contract.max_gross_exposure)
