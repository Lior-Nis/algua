"""Leading-indicator signal-drift computation (issue #343) — PURE (numpy/pandas/scipy only).

Compares a strategy's cross-sectional signal scores over a REFERENCE window (an earlier,
"frozen-baseline" era) against a RECENT window (later era), and reports drift that can precede a
visible hit to realized P&L. Two tiers, honestly separated:

* TIER A — the LEADING layer (label-free; uses ONLY signal scores, zero forward returns): a shift
  in the signal's own output distribution / turnover / coverage fires BEFORE any realized label.
  This tier alone drives the headline ``verdict``.
* TIER B — CORROBORATING, not leading (needs realized forward returns; coincident with the return
  window): rank-IC and hit-rate. These beat portfolio P&L on VARIANCE (a full N-symbol
  cross-section vs one portfolio number), NOT on TIME — they do not see the future. Reported
  separately and never manufacture the headline alarm by themselves.

Nothing here does I/O or touches the registry / gates / order path. The natural next step —
persisting the reference distribution as a versioned artifact at promotion and wiring cross-run
persistence into the paper/forward loop as a demotion trigger — is deliberately out of scope
(needs CODEOWNERS-protected promotion/schema changes; tracked as a follow-up).

Limitations (advisory, not proof): a PSI shape shift may be a strategy-INTENDED regime tilt;
coverage is a count-only proxy (symbol-SET churn is caught only by the membership-Jaccard metric);
the PSI bands are the industry-convention heuristic (0.10 / 0.25), not statistically validated; IC
bars are equal-weighted regardless of cross-sectional breadth (matching ``factor_eval.factor_ic``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from algua.backtest.factor_eval import factor_ic

# --- statuses (insufficient_data is NON-triggering: it never raises the verdict) ---
OK = "ok"
WARN = "warn"
ALARM = "alarm"
INSUFFICIENT = "insufficient_data"
_SEVERITY = {OK: 0, INSUFFICIENT: 0, WARN: 1, ALARM: 2}

# --- heuristic thresholds (documented as conventions, not statistically validated) ---
PSI_WARN = 0.10
PSI_ALARM = 0.25
TURNOVER_FLOOR = 0.02  # denominator floor so a near-flat reference can't explode the ratio
TURNOVER_RATIO_WARN = 1.5
TURNOVER_RATIO_ALARM = 2.5
COVERAGE_RATIO_WARN = 0.85  # recent/reference: coverage that DROPS is the concern
COVERAGE_RATIO_ALARM = 0.60
MEMBERSHIP_JACCARD_WARN = 0.75  # symbol-set overlap that DROPS is the concern
MEMBERSHIP_JACCARD_ALARM = 0.50
IC_RETENTION_WARN = 0.50  # recent IC retains < 50% of the reference IC
HITRATE_DROP_WARN = 0.08  # absolute drop in IC>0 hit-rate
HITRATE_DROP_ALARM = 0.15
_PSI_FLOOR = 1e-6  # replaces empty bins so ln() stays finite
_MIN_PSI_REF = 20  # minimum reference observations to compute a PSI at all


def _worst(statuses: list[str]) -> str:
    """Worst (most severe) status; empty / all-insufficient collapses to insufficient_data."""
    triggering = [s for s in statuses if _SEVERITY.get(s, 0) > 0]
    if triggering:
        return max(triggering, key=lambda s: _SEVERITY[s])
    return INSUFFICIENT if statuses and all(s == INSUFFICIENT for s in statuses) else OK


def _leading_verdict(
    psi: str, turnover: str, coverage: str, membership: str
) -> str:
    """Headline verdict from the leading (tier A) statuses, with an asymmetry: any metric may
    RAISE the verdict (warn/alarm), but a confident all-clear (`ok`) requires at least one
    DISTRIBUTION detector — PSI or rank-turnover — to have actually run and cleared. Coverage /
    membership are presence checks: they can alarm on universe churn but cannot, alone, certify the
    signal itself is undecayed. So a constant/degenerate signal (PSI + turnover both insufficient)
    surfaces as `insufficient_data`, never a false `ok`."""
    statuses = [psi, turnover, coverage, membership]
    triggering = [s for s in statuses if _SEVERITY[s] > 0]
    if triggering:
        return max(triggering, key=lambda s: _SEVERITY[s])
    if OK in (psi, turnover):
        return OK
    return INSUFFICIENT


def _split_index(index: pd.Index, split: pd.Timestamp | None, reference_frac: float) -> int:
    """Position `k` such that index[:k] is the reference window and index[k:] is the recent one.

    A pinned `split` timestamp puts every bar with timestamp <= split in the reference window;
    otherwise the split is the positional `reference_frac` quantile. Raises if either side is
    empty (a degenerate window is a caller/config error, not a drift finding)."""
    n = len(index)
    if n < 2:
        raise ValueError("drift needs at least 2 decision bars")
    if split is not None:
        k = int(index.searchsorted(split, side="right"))
    else:
        if not 0.0 < reference_frac < 1.0:
            raise ValueError(f"reference_frac must be in (0, 1), got {reference_frac}")
        k = int(round(n * reference_frac))
    if not 0 < k < n:
        raise ValueError(
            "reference/recent split leaves an empty window; widen the period or move the split"
        )
    return k


def _standardize_pooled(panel: pd.DataFrame) -> np.ndarray:
    """Pool per-bar cross-sectionally z-scored finite scores into one 1-D array.

    Per-bar standardization strips benign market-regime level/scale shifts, leaving distributional
    SHAPE. Bars with < 2 finite scores or zero cross-sectional variance contribute nothing (no NaN
    leakage). Returns an empty array when no bar qualifies."""
    out: list[np.ndarray] = []
    for _, row in panel.iterrows():
        vals = row.to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size < 2:
            continue
        std = vals.std()
        if not (std > 0):
            continue
        out.append((vals - vals.mean()) / std)
    return np.concatenate(out) if out else np.array([], dtype=float)


def population_stability_index(
    reference: np.ndarray, recent: np.ndarray, *, bins: int
) -> float | None:
    """PSI of `recent` vs `reference` over quantile bins FROZEN from the reference.

    Returns None (insufficient) when the reference is too small or too discrete to bin (< 2
    distinct interior edges). Bin edges are extended to +/- inf so tail mass is always captured;
    empty bins are floored so the log stays finite."""
    if reference.size < _MIN_PSI_REF or recent.size == 0:
        return None
    q = np.quantile(reference, np.linspace(0.0, 1.0, bins + 1))
    interior = np.unique(q)
    if interior.size < 3:  # < 2 usable bins -> effectively constant reference
        return None
    edges = interior.astype(float).copy()
    edges[0], edges[-1] = -np.inf, np.inf
    ref_prop = np.histogram(reference, bins=edges)[0] / reference.size
    rec_prop = np.histogram(recent, bins=edges)[0] / recent.size
    ref_prop = np.where(ref_prop <= 0, _PSI_FLOOR, ref_prop)
    rec_prop = np.where(rec_prop <= 0, _PSI_FLOOR, rec_prop)
    return float(np.sum((rec_prop - ref_prop) * np.log(rec_prop / ref_prop)))


def _psi_status(psi: float | None) -> str:
    if psi is None:
        return INSUFFICIENT
    if psi > PSI_ALARM:
        return ALARM
    if psi > PSI_WARN:
        return WARN
    return OK


def mean_signal_turnover(panel: pd.DataFrame) -> float | None:
    """Mean per-consecutive-bar L1 turnover of rank-normalized scores.

    For each adjacent bar pair, ranks are recomputed over ONLY the symbols scored (finite) in BOTH
    bars — so universe churn / missingness never masquerades as turnover (ranks are normalized to
    [0, 1] within that intersection). A pair whose either bar is fully tied (zero cross-sectional
    variation — e.g. a constant/degenerate signal) carries NO ordering information and is skipped,
    so a constant signal reports None (insufficient), never a spurious 0-turnover all-clear.
    Returns None when no pair has a >= 2-symbol overlap with usable ordering on both sides."""
    turns: list[float] = []
    prev: pd.Series | None = None
    for _, row in panel.iterrows():
        cur = row[np.isfinite(row.to_numpy(dtype=float))]
        if prev is not None:
            common = prev.index.intersection(cur.index)
            if len(common) >= 2:
                p_c, c_c = prev[common], cur[common]
                if p_c.nunique() >= 2 and c_c.nunique() >= 2:  # both bars carry an ordering
                    n = len(common) - 1
                    pr = p_c.rank(method="average").to_numpy(dtype=float)
                    cr = c_c.rank(method="average").to_numpy(dtype=float)
                    turns.append(float(np.mean(np.abs(cr - pr) / n)))
        prev = cur
    return float(np.mean(turns)) if turns else None


def _active_symbols(panel: pd.DataFrame) -> set[str]:
    """Symbols scored (finite) on at least one bar — the era's realized cross-section."""
    if panel.empty:
        return set()
    finite_any = np.isfinite(panel.to_numpy(dtype=float)).any(axis=0)
    return {str(c) for c, keep in zip(panel.columns, finite_any, strict=True) if keep}


def membership_jaccard(reference: pd.DataFrame, recent: pd.DataFrame) -> float | None:
    """Jaccard overlap of the two eras' active-symbol SETS (identity-aware, unlike count coverage).

    Catches a same-COUNT but different-SYMBOL cross-section (universe churn) that count-only
    coverage and the identity-free pooled PSI both miss. None when either era scored nothing."""
    ref_syms, rec_syms = _active_symbols(reference), _active_symbols(recent)
    union = ref_syms | rec_syms
    if not union:
        return None
    return len(ref_syms & rec_syms) / len(union)


def _membership_status(jaccard: float | None) -> str:
    if jaccard is None:
        return INSUFFICIENT
    if jaccard < MEMBERSHIP_JACCARD_ALARM:
        return ALARM
    if jaccard < MEMBERSHIP_JACCARD_WARN:
        return WARN
    return OK


def _ratio_status(
    recent: float, reference: float, *, floor: float, warn: float, alarm: float
) -> str:
    denom = max(reference, floor)
    ratio = recent / denom
    if ratio >= alarm:
        return ALARM
    if ratio >= warn:
        return WARN
    return OK


def mean_coverage(panel: pd.DataFrame) -> float | None:
    """Mean count of finite scores per bar (a weak count-only proxy for cross-sectional breadth)."""
    if panel.empty:
        return None
    counts = np.isfinite(panel.to_numpy(dtype=float)).sum(axis=1)
    return float(counts.mean()) if counts.size else None


def _coverage_status(recent: float | None, reference: float | None) -> str:
    if recent is None or reference is None or reference <= 0:
        return INSUFFICIENT
    ratio = recent / reference
    if ratio < COVERAGE_RATIO_ALARM:
        return ALARM
    if ratio < COVERAGE_RATIO_WARN:
        return WARN
    return OK


def _window_ic(scores: pd.DataFrame, fwd: pd.DataFrame, *, min_obs: int) -> dict[str, Any]:
    """factor_ic over a window, tagged insufficient when < min_obs usable bars or degenerate."""
    ic = factor_ic(scores, fwd)
    n = ic.get("n_obs") or 0
    ok = n >= min_obs and ic.get("mean_ic") is not None
    return {"mean_ic": ic.get("mean_ic"), "hit_rate": ic.get("hit_rate"), "n_obs": n, "ok": ok}


def _ic_decay(ref: dict[str, Any], rec: dict[str, Any]) -> tuple[str, float | None]:
    """(status, retention) for recent vs reference mean IC.

    Retention = recent/reference (only meaningful when the reference had positive edge). A recent
    IC <= 0 after a positive reference is a sign-flip/collapse -> alarm; retaining < 50% -> warn.
    If the reference itself had no positive edge, decay is not assessable -> insufficient."""
    if not (ref["ok"] and rec["ok"]):
        return INSUFFICIENT, None
    ref_ic, rec_ic = ref["mean_ic"], rec["mean_ic"]
    if ref_ic is None or rec_ic is None or ref_ic <= 0:
        return INSUFFICIENT, None
    retention = rec_ic / ref_ic
    if rec_ic <= 0:
        return ALARM, retention
    if retention < IC_RETENTION_WARN:
        return WARN, retention
    return OK, retention


def _hitrate_drift(ref: dict[str, Any], rec: dict[str, Any]) -> tuple[str, float | None]:
    if not (ref["ok"] and rec["ok"]) or ref["hit_rate"] is None or rec["hit_rate"] is None:
        return INSUFFICIENT, None
    drop = ref["hit_rate"] - rec["hit_rate"]
    if drop >= HITRATE_DROP_ALARM:
        return ALARM, drop
    if drop >= HITRATE_DROP_WARN:
        return WARN, drop
    return OK, drop


@dataclass
class DriftReport:
    """Advisory drift verdict. `verdict` is driven by the TIER A (leading, label-free) metrics
    ONLY; `corroborating` holds the coincident TIER B read; `note` flags a tier-B/tier-A
    divergence without ever escalating the headline."""

    verdict: str
    reference: dict[str, Any]
    recent: dict[str, Any]
    leading: dict[str, Any]  # tier A metrics
    corroborating: dict[str, Any]  # tier B metrics
    note: str | None = None
    limitations: list[str] = field(
        default_factory=lambda: [
            "advisory only: gates nothing, persists nothing, off the live/paper order path",
            "a PSI shape shift may be a strategy-intended regime tilt, not decay",
            "coverage is a count-only proxy; symbol-SET churn is caught only by membership_drift",
            "PSI bands (0.10/0.25) are the industry heuristic, not statistically validated",
            "TIER B (IC/hit-rate) is coincident with the return window, not leading",
            "single-run snapshot; sustained drift across runs is the real signal",
        ]
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "reference": self.reference,
            "recent": self.recent,
            "leading": self.leading,
            "corroborating": self.corroborating,
            "note": self.note,
            "limitations": list(self.limitations),
        }


def drift_report(
    score_panel: pd.DataFrame,
    forward_returns: pd.DataFrame | None,
    *,
    split: pd.Timestamp | None = None,
    reference_frac: float = 0.5,
    psi_bins: int = 10,
    min_obs: int = 20,
    forward_embargo: int = 0,
) -> DriftReport:
    """Compute the advisory drift report over a chronological reference/recent split.

    `forward_returns` is OPTIONAL: when None, TIER B is reported as insufficient_data and the
    verdict rests purely on the label-free leading layer. No look-ahead — reference statistics and
    PSI bin edges are frozen from the strictly-earlier window.

    `forward_embargo` = the label-realization lag of `forward_returns` in bars (decision lag +
    horizon). A label known at decision `t` is realized from prices at `t+embargo`, so the last
    `forward_embargo` reference decisions consume RECENT-era prices; they are PURGED from the
    reference IC so the frozen reference never peeks across the split (a purged-CV embargo). The
    default 0 assumes labels already aligned to the decision timestamp."""
    panel = score_panel.sort_index()
    k = _split_index(panel.index, split, reference_frac)
    ref_panel, rec_panel = panel.iloc[:k], panel.iloc[k:]
    split_ts = panel.index[k]

    # --- TIER A (leading, label-free) ---
    psi = population_stability_index(
        _standardize_pooled(ref_panel), _standardize_pooled(rec_panel), bins=psi_bins
    )
    psi_status = _psi_status(psi)

    ref_to = mean_signal_turnover(ref_panel)
    rec_to = mean_signal_turnover(rec_panel)
    turnover_status = (
        INSUFFICIENT
        if ref_to is None or rec_to is None
        else _ratio_status(
            rec_to, ref_to, floor=TURNOVER_FLOOR,
            warn=TURNOVER_RATIO_WARN, alarm=TURNOVER_RATIO_ALARM,
        )
    )

    ref_cov, rec_cov = mean_coverage(ref_panel), mean_coverage(rec_panel)
    coverage_status = _coverage_status(rec_cov, ref_cov)

    jaccard = membership_jaccard(ref_panel, rec_panel)
    membership_status = _membership_status(jaccard)

    leading = {
        "signal_distribution_psi": {"psi": psi, "status": psi_status, "bins": psi_bins},
        "turnover_drift": {
            "reference": ref_to, "recent": rec_to,
            "ratio": (
                rec_to / max(ref_to, TURNOVER_FLOOR)
                if (ref_to is not None and rec_to is not None) else None
            ),
            "status": turnover_status,
        },
        "coverage_drift": {
            "reference": ref_cov, "recent": rec_cov,
            "ratio": (rec_cov / ref_cov) if (ref_cov and rec_cov is not None) else None,
            "status": coverage_status,
        },
        "membership_drift": {"jaccard": jaccard, "status": membership_status},
    }
    verdict = _leading_verdict(
        psi_status, turnover_status, coverage_status, membership_status
    )

    # --- TIER B (corroborating, coincident) ---
    if forward_returns is None:
        ref_ic = rec_ic = {"mean_ic": None, "hit_rate": None, "n_obs": 0, "ok": False}
    else:
        fwd = forward_returns.sort_index()
        # Purge the last `forward_embargo` reference decisions: their labels are realized from
        # prices at/after the split, so keeping them would leak the recent window into the frozen
        # reference IC (a boundary look-ahead). Recent decisions look only further forward.
        ref_cut = max(k - max(forward_embargo, 0), 0)
        ref_ic = _window_ic(panel.iloc[:ref_cut], fwd, min_obs=min_obs)
        rec_ic = _window_ic(rec_panel, fwd, min_obs=min_obs)
    ic_status, retention = _ic_decay(ref_ic, rec_ic)
    hit_status, hit_drop = _hitrate_drift(ref_ic, rec_ic)
    corroborating = {
        "ic_decay": {
            "reference_mean_ic": ref_ic["mean_ic"], "recent_mean_ic": rec_ic["mean_ic"],
            "retention": retention, "reference_n": ref_ic["n_obs"], "recent_n": rec_ic["n_obs"],
            "status": ic_status,
        },
        "hit_rate_drift": {
            "reference": ref_ic["hit_rate"], "recent": rec_ic["hit_rate"],
            "drop": hit_drop, "status": hit_status,
        },
        "note": "TIER B is coincident with the return window (higher-power than portfolio P&L on "
                "variance, NOT leading in time); it never raises the headline verdict.",
    }

    note: str | None = None
    tier_b_worst = _worst([ic_status, hit_status])
    if verdict == OK and _SEVERITY[tier_b_worst] > 0:
        note = (
            "signal distribution stable but measured predictive power is eroding "
            "(corroborating tier B) — investigate before it reaches realized P&L"
        )

    def _bounds(idx: pd.Index) -> dict[str, Any]:
        if not len(idx):
            return {"bars": 0}
        return {"start": str(idx[0]), "end": str(idx[-1]), "bars": int(len(idx))}

    return DriftReport(
        verdict=verdict,
        reference={**_bounds(ref_panel.index)},
        recent={**_bounds(rec_panel.index), "split": str(split_ts)},
        leading=leading,
        corroborating=corroborating,
        note=note,
    )
