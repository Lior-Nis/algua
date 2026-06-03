from __future__ import annotations

import pandas as pd


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
    if gross > max_gross + 1e-9:
        raise RiskBreach(
            "gross_exposure",
            f"gross exposure {gross:.4f} exceeds max_gross_exposure {max_gross:.4f}",
        )


def check_long_only(weights: pd.Series, strategy_name: str) -> None:
    if len(weights) and bool((weights < 0).any()):
        negative = sorted(weights[weights < 0].index)
        raise RiskBreach(
            "long_only",
            f"long-only: strategy '{strategy_name}' returned negative target weight(s) "
            f"for {negative}",
        )


def check_drawdown(equity: float, peak: float, max_drawdown: float) -> None:
    if max_drawdown >= 1.0 or peak <= 0:
        return  # disabled, or no peak yet
    if equity < peak * (1.0 - max_drawdown):
        dd = 1.0 - (equity / peak)
        raise RiskBreach(
            "drawdown",
            f"drawdown {dd:.4f} exceeds max_drawdown {max_drawdown:.4f} "
            f"(equity {equity:.2f}, peak {peak:.2f})",
        )
