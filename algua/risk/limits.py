from __future__ import annotations

import pandas as pd

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


def check_long_only(weights: pd.Series, strategy_name: str) -> None:
    if len(weights) and bool((weights < 0).any()):
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
