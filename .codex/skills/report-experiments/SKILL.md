---
name: report-experiments
description: Generate an organized, plotted experiment report for a strategy (sweep + walk-forward) into the kb vault — read MLflow-tracked data, render data-science SVGs, write a provenance-stamped, graph-linked markdown report. Use after sweeps/walk-forwards to curate and interpret the results; a natural step-7 (Record) extension of run-the-research-loop.
---

# Reporting experiments

Turn a strategy's raw MLflow runs into a **curated, interpreted, graph-linked** report in the kb:
plots + your reading of the results, living at `kb/strategies/<name>/reports/<stamp>/`, wikilinked
into the strategy and family notes. Read `operating-algua` first for the golden rules.

## The discipline (read before running)

1. **Not an MLflow duplicate.** MLflow stays the raw run store. This report is the *curated* layer
   ON TOP of it — it carries the **reading** of the results and the graph links, and pulls data
   *from* MLflow rather than re-storing runs. If you find yourself copying raw run tables, stop.
2. **Reproducibility stamp.** Every report records the MLflow run id(s), `snapshot_id`, `code_hash`,
   `config_hash`, `dependency_hash`, `seed` it was generated from — labelled **"identity as logged by
   the run"** (it pins the *run's* artifact, not the strategy's current source, which may have moved
   on). A plot without provenance is a liability.
3. **Binaries in git, but diffable.** The vault is version-controlled. The script emits **SVG**
   (text) deterministically (fixed `svg.hashsalt`, no embedded date, stable `MPLCONFIGDIR`, sorted
   inputs) so re-runs don't churn git. One report dir per run.
4. **Never surface the holdout.** The single-use OOS holdout is revealed only by `research promote`
   and is *not* in MLflow (the tracker strips it). The script rejects ANY `holdout*` metric
   defensively — do not add it back.

## v1 scope (tracked data only)

Plots come ONLY from data MLflow already holds, so there's **no backtest re-run and no new CLI**:
- **Sweep / HPO:** parameter heatmap (score over a 2-param grid), per-parameter sensitivity curves,
  top-N leaderboard — from the logged `sweep.json`.
- **Walk-forward:** per-window in-sample stability bars — from `result.json`'s `window_metrics`.
- **Cross-run:** `mean_sharpe` across the strategy's tracked runs over time.

**Deferred (out of scope):** equity curve, drawdown, rolling Sharpe, return distribution. They need
the portfolio **return series**, which no CLI command emits today; producing it from a skill would
mean reaching into `algua.backtest` internals (violates "never bypass the CLI"). The clean
enablement is a future `backtest`-emits-series change — file it, don't hack around it.

## Playbook

0. **Preflight.** Confirm the plot lib is importable; it ships transitively via mlflow:
   `uv run python -c "import matplotlib"` — if it fails, `uv add matplotlib` and retry. Resolve the
   tracking store the way the platform does: `MLFLOW_TRACKING_URI` (default `mlruns`).
1. **Pick the strategy + family.** `uv run algua registry show <name>` gives its `family` (for the
   `[[wikilink]]`). The strategy must have at least one **`--track`ed** sweep or walk-forward run; if
   not, run e.g. `uv run algua backtest sweep <name> --demo --param lookback=20,40,60 --param top_k=1,2 --track`
   and `uv run algua backtest walk-forward <name> --demo --windows 4 --track` first.
2. **Generate the figures + skeleton.** Save the reference script below to a temp file and run it:
   `uv run python /tmp/report_experiments.py <name> --family <slug>`. It writes
   `kb/strategies/<name>/reports/<stamp>/report.md` + the SVGs, with the provenance stamp filled in.
3. **Interpret — this is the point.** Edit the generated `report.md`: under each figure, write the
   *reading* (what the heatmap/sensitivity says about robustness; whether the windows are stable or a
   single window carries it; over/under-fitting tells). Reference `interpret-results` for what
   "good" looks like and the overfitting/look-ahead pitfalls. A report without interpretation is just
   pictures.
4. **Join the graph.** Add a one-line `[[report]]` pointer from the strategy note's free-text area
   (NOT the registry-owned synced blocks) and, if useful, the family note. The report already
   wikilinks back to `[[<name>]]` / `[[<family>]]`.
5. **Commit to the vault** (the notes *are* the knowledge): `git add kb/strategies/<name>/reports/...`
   and commit. Keep figures small; one report dir per run.

## Reference script

Save verbatim to a temp file (e.g. `/tmp/report_experiments.py`) and run it. It reads MLflow only;
anything that *drives the system* still goes through `uv run algua ...`.

```python
#!/usr/bin/env python
"""Generate a curated, plotted experiment report for an algua strategy FROM MLflow-tracked data.
MLflow stays the raw run store; this writes the curated, graph-linked layer into the kb vault. It
pulls FROM MLflow and never re-stores runs. v1: tracked data only (no equity/drawdown series)."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Stable matplotlib cache dir, set BEFORE importing pyplot -> reproducible rendering across envs.
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "algua-mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402

mpl.rcParams["svg.hashsalt"] = "algua-report"  # stable element ids -> diffable SVG
mpl.rcParams["figure.figsize"] = (7.0, 4.5)
mpl.rcParams["font.family"] = "DejaVu Sans"
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402


def _uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", "mlruns")


def _savefig(fig: Any, out: Path) -> None:
    fig.savefig(out, format="svg", metadata={"Date": None}, bbox_inches="tight")
    plt.close(fig)


def _artifact_json(run_id: str, path: str, uri: str) -> dict[str, Any] | None:
    """Download + parse a JSON artifact. ANY failure (missing/unreadable/malformed) -> None, so a
    corrupt run is skipped, never fatal."""
    try:
        local = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path=path, tracking_uri=uri
        )
        return json.loads(Path(local).read_text())
    except Exception:
        return None


def _first_with_artifact(runs: list, kind: str, artifact: str, uri: str):
    """Newest run (runs are pre-sorted DESC) of `kind` whose `artifact` loads — falls past a corrupt
    or artifact-less newest run to the next valid one. Returns (run, parsed) or (None, None)."""
    for r in runs:
        if r.data.tags.get("kind") == kind:
            d = _artifact_json(r.info.run_id, artifact, uri)
            if d is not None:
                return r, d
    return None, None


def _filter_holdout(metrics: dict[str, float]) -> dict[str, float]:
    """Reject ANY key whose lowercase form starts with "holdout" (holdout_metrics, holdout.sharpe,
    holdout_sharpe, holdoutSharpe, bare "holdout"): the single-use OOS holdout is never surfaced."""
    return {k: v for k, v in metrics.items() if not k.lower().startswith("holdout")}


def plot_sweep_heatmap(sweep: dict[str, Any], out: Path) -> str | None:
    grid = {k: v for k, v in (sweep.get("grid") or {}).items() if len(v) > 1}
    if len(grid) != 2:
        return None
    ranked = sweep.get("ranked") or []
    (px, xs), (py, ys) = sorted(grid.items())
    score = {
        (r["params"][px], r["params"][py]): r.get("score")
        for r in ranked
        if px in r.get("params", {}) and py in r.get("params", {})
    }
    z = [[score.get((x, y), float("nan")) for x in xs] for y in ys]
    fig, ax = plt.subplots()
    im = ax.imshow(z, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(xs)), [str(x) for x in xs])
    ax.set_yticks(range(len(ys)), [str(y) for y in ys])
    ax.set_xlabel(px)
    ax.set_ylabel(py)
    ax.set_title(f"{sweep.get('strategy', '?')} — {sweep.get('rank_by', 'score')} over ({px} × {py})")
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            v = score.get((x, y))
            if v is not None:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="w", fontsize=8)
    fig.colorbar(im, ax=ax, label=sweep.get("rank_by", "score"))
    _savefig(fig, out)
    return out.name


def plot_sensitivity(sweep: dict[str, Any], out: Path) -> str | None:
    swept = {k: v for k, v in (sweep.get("grid") or {}).items() if len(v) > 1}
    if not swept:
        return None
    ranked = sweep.get("ranked") or []
    rank_by = sweep.get("rank_by", "score")
    fig, axes = plt.subplots(1, len(swept), squeeze=False)
    for ax, (p, vals) in zip(axes[0], sorted(swept.items()), strict=True):
        means = []
        for val in vals:
            scores = [r.get("score") for r in ranked if r.get("params", {}).get(p) == val]
            scores = [s for s in scores if s is not None]
            means.append(sum(scores) / len(scores) if scores else float("nan"))
        ax.plot([str(v) for v in vals], means, marker="o")
        ax.set_xlabel(p)
        ax.set_ylabel(f"mean {rank_by}")
        ax.set_title(p)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"{sweep.get('strategy', '?')} — parameter sensitivity")
    _savefig(fig, out)
    return out.name


def plot_leaderboard(sweep: dict[str, Any], out: Path, top: int = 10) -> str | None:
    ranked = (sweep.get("ranked") or [])[:top]
    if not ranked:
        return None
    labels = [", ".join(f"{k}={v}" for k, v in sorted(r.get("params", {}).items())) for r in ranked]
    scores = [r.get("score", float("nan")) for r in ranked]
    fig, ax = plt.subplots(figsize=(7.0, max(3.0, 0.4 * len(ranked) + 1)))
    ax.barh(range(len(ranked)), scores, color="#4c72b0")
    ax.set_yticks(range(len(ranked)), labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(sweep.get("rank_by", "score"))
    ax.set_title(f"{sweep.get('strategy', '?')} — top {len(ranked)} combos")
    ax.grid(True, axis="x", alpha=0.3)
    _savefig(fig, out)
    return out.name


def plot_wf_stability(result: dict[str, Any], out: Path) -> str | None:
    wm = result.get("window_metrics") or []
    if not wm:
        return None
    idx = [w.get("index", i) for i, w in enumerate(wm)]
    sharpe = [float(w.get("sharpe") or 0.0) for w in wm]
    fig, ax = plt.subplots()
    colors = ["#55a868" if s >= 0 else "#c44e52" for s in sharpe]
    ax.bar(idx, sharpe, color=colors)
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xlabel("walk-forward window")
    ax.set_ylabel("sharpe")
    st = result.get("stability", {})
    sub = (
        f"mean {st.get('mean_sharpe', float('nan')):.2f} · "
        f"min {st.get('min_sharpe', float('nan')):.2f} · "
        f"pct+ {st.get('pct_positive_windows', float('nan')):.0%}"
    )
    ax.set_title(f"{result.get('strategy', '?')} — per-window stability\n{sub}", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    _savefig(fig, out)
    return out.name


def plot_cross_run(strategy: str, runs: list, out: Path) -> str | None:
    pts = []
    for r in runs:
        m = _filter_holdout(dict(r.data.metrics))
        if "mean_sharpe" in m:
            pts.append((r.info.start_time or 0, r.info.run_id, m["mean_sharpe"],
                        r.data.tags.get("kind", "?")))
    if len(pts) < 2:
        return None
    pts.sort()  # (start_time, run_id) — deterministic tie-break
    fig, ax = plt.subplots()
    ax.plot(range(len(pts)), [p[2] for p in pts], marker="o")
    ax.set_xticks(range(len(pts)), [p[3] for p in pts], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("mean_sharpe")
    ax.set_title(f"{strategy} — mean_sharpe across tracked runs (oldest→newest)")
    ax.grid(True, alpha=0.3)
    _savefig(fig, out)
    return out.name


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("strategy")
    ap.add_argument("--kb", default="kb", help="vault root")
    ap.add_argument("--family", default=None, help="family slug for the [[wikilink]]")
    args = ap.parse_args()

    uri = _uri()
    client = MlflowClient(tracking_uri=uri)
    exp = client.get_experiment_by_name(args.strategy)
    if exp is None:
        print(f"no MLflow experiment for {args.strategy!r} at {uri}")
        return 1
    runs = client.search_runs([exp.experiment_id])
    # Deterministic order: newest first, run_id breaks identical-start_time ties.
    runs = sorted(runs, key=lambda r: (r.info.start_time or 0, r.info.run_id), reverse=True)

    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    rdir = Path(args.kb) / "strategies" / args.strategy / "reports" / stamp
    rdir.mkdir(parents=True, exist_ok=True)

    figs: list[tuple[str, str]] = []
    provenance: dict[str, Any] = {}

    sweep_run, sweep = _first_with_artifact(runs, "sweep", "sweep.json", uri)
    if sweep is not None:
        best_cfg = ((sweep.get("ranked") or [{}])[0] or {}).get("config_hash")
        provenance["sweep"] = {
            "run_id": sweep_run.info.run_id, "snapshot_id": sweep.get("snapshot_id"),
            "code_hash": sweep.get("code_hash"), "dependency_hash": sweep.get("dependency_hash"),
            "best_config_hash": best_cfg, "seed": sweep.get("seed"),
            "n_combos": sweep.get("n_combos"), "rank_by": sweep.get("rank_by"),
        }
        for fn, name in (
            (plot_sweep_heatmap(sweep, rdir / "sweep_heatmap.svg"), "Parameter heatmap"),
            (plot_sensitivity(sweep, rdir / "sweep_sensitivity.svg"), "Parameter sensitivity"),
            (plot_leaderboard(sweep, rdir / "sweep_leaderboard.svg"), "Top-N leaderboard"),
        ):
            if fn:
                figs.append((name, fn))

    wf_run, result = _first_with_artifact(runs, "walk_forward", "result.json", uri)
    if result is not None:
        provenance["walk_forward"] = {
            "run_id": wf_run.info.run_id, "config_hash": result.get("config_hash"),
            "dependency_hash": result.get("dependency_hash"),
            "snapshot_id": result.get("snapshot_id"), "code_hash": result.get("code_hash"),
            "seed": result.get("seed"), "windows": result.get("windows"),
        }
        fn = plot_wf_stability(result, rdir / "wf_stability.svg")
        if fn:
            figs.append(("Walk-forward per-window stability", fn))

    fn = plot_cross_run(args.strategy, runs, rdir / "cross_run.svg")
    if fn:
        figs.append(("Cross-run leaderboard", fn))

    fam = args.family
    lines = [
        f"# Experiment report — {args.strategy}",
        "",
        f"> Generated {stamp} from MLflow ({uri}). Curated/interpreted layer over the raw run store",
        "> — pulls FROM MLflow, does not re-store runs. v1: tracked-data plots only.",
        "",
        f"Strategy: [[{args.strategy}]]" + (f" · Family: [[{fam}]]" if fam else ""),
        "",
        "## Provenance (identity AS LOGGED BY THE RUN)",
        "",
        "```json",
        json.dumps(provenance, indent=2, sort_keys=True),
        "```",
        "",
        "## Figures",
        "",
    ]
    for title, name in figs:
        lines += [f"### {title}", "", f"![{title}]({name})", "", "_Reading: TODO — interpret._", ""]
    if not figs:
        lines += ["_No plottable tracked runs (need a `--track`ed sweep or walk-forward)._", ""]
    (rdir / "report.md").write_text("\n".join(lines))
    print(f"wrote {rdir}/report.md with {len(figs)} figure(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

## Notes

- The script is **deterministic**: same MLflow runs → byte-identical SVGs (fixed hashsalt, no date
  metadata, stable `MPLCONFIGDIR`, `(start_time, run_id)` ordering, sorted labels/keys), so
  committing a regenerated report is a clean no-op diff.
- It **degrades gracefully**: missing/corrupt artifacts are skipped (it falls to the next valid run
  of that kind); a heatmap needs a 2-param grid; cross-run needs ≥2 runs with `mean_sharpe`; missing
  pieces are simply omitted, never fatal.
- The `_Reading: TODO_` placeholders under each figure are your cue (step 3) — replace them. Don't
  commit a report still saying TODO.
