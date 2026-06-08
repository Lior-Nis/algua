from __future__ import annotations

import keyword
import re
from pathlib import Path

import typer

from algua.cli._common import ok, registry_conn
from algua.cli.app import app, emit
from algua.cli.errors import json_errors
from algua.cli.registry_cmd import _kb_metadata
from algua.config.settings import get_settings
from algua.knowledge.sync import (
    family_doc_path,
    generate_indexes,
    strategy_doc_path,
    strategy_family,
    sync_all,
    sync_family_doc,
    sync_strategy_doc,
)
from algua.knowledge.templates import scaffold_family_doc, scaffold_strategy_doc
from algua.registry.store import SqliteStrategyRepository
from algua.strategies.loader import list_strategies

strategy_app = typer.Typer(help="Author and list strategies", no_args_is_help=True)
app.add_typer(strategy_app, name="strategy")

_FAMILY_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")

_TEMPLATE = '''\
"""Strategy: {name}. Edit compute_weights to express your cross-sectional logic."""
from __future__ import annotations

from typing import Any

import pandas as pd

from algua.contracts.types import ExecutionContract
from algua.strategies.base import StrategyConfig

CONFIG = StrategyConfig(
    name="{name}",
    universe=["AAPL", "MSFT", "NVDA"],
    execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
    params={{"lookback": 60, "top_k": 2}},
)


def compute_weights(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    lookback = int(params["lookback"])
    wide = view.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    if len(wide) <= lookback:
        return pd.Series(dtype="float64")
    scores = (wide.iloc[-1] / wide.iloc[-1 - lookback] - 1.0).dropna()
    winners = scores.sort_values(ascending=False).head(int(params["top_k"])).index
    if len(winners) == 0:
        return pd.Series(dtype="float64")
    return pd.Series(1.0 / len(winners), index=winners)
'''


@strategy_app.command("list")
@json_errors()
def list_() -> None:
    """List available strategies as JSON."""
    emit(list_strategies())


@strategy_app.command("new")
@json_errors()
def new(
    name: str,
    family: str = typer.Option(None, "--family", help="thesis family this belongs to"),
    derived_from: str = typer.Option(None, "--derived-from", help="parent strategy name"),
) -> None:
    """Scaffold a new strategy module + its knowledge-base doc (and family hub if needed)."""
    if not name.isidentifier() or keyword.iskeyword(name):
        raise ValueError(
            f"invalid strategy name {name!r}: must be a valid, non-keyword Python identifier"
        )
    if family is not None and not _FAMILY_RE.match(family):
        raise ValueError(
            f"invalid family {family!r}: must be a lowercase slug (a-z, 0-9, hyphen)"
        )
    path = Path(__file__).parent.parent / "strategies" / "examples" / f"{name}.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError(f"strategy already exists: {path}")
    path.write_text(_TEMPLATE.format(name=name))

    settings = get_settings()
    doc_path = strategy_doc_path(settings, name)
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    if not doc_path.exists():
        doc_path.write_text(
            scaffold_strategy_doc(name, family=family, derived_from=derived_from)
        )
    family_doc: str | None = None
    if family:
        fam_path = family_doc_path(settings, family)
        fam_path.parent.mkdir(parents=True, exist_ok=True)
        if not fam_path.exists():
            fam_path.write_text(scaffold_family_doc(family))
        family_doc = str(fam_path)

    emit(ok({
        "name": name, "path": str(path),
        "doc": str(doc_path), "family_doc": family_doc,
    }))


@strategy_app.command("doc")
@json_errors()
def doc(
    name: str = typer.Argument(None, help="strategy to sync; omit (or --all) for all"),
    all_: bool = typer.Option(False, "--all", help="sync every strategy + family doc"),
) -> None:
    """Regenerate the synced blocks of strategy/family docs + rebuild the indexes."""
    settings = get_settings()
    # Read full records at the CLI seam; the knowledge layer stays registry-free.
    with registry_conn() as conn:
        recs = {rec.name: rec for rec in SqliteStrategyRepository(conn).list_strategies()}
    stages = {n: r.stage.value for n, r in recs.items()}
    if all_ or name is None:
        summary = sync_all(settings, stages)
    else:
        meta = _kb_metadata(recs[name]) if name in recs else None
        if not sync_strategy_doc(settings, name, stage=stages.get(name), metadata=meta):
            raise ValueError(f"no strategy doc for {name!r}; run `strategy new` first")
        # Keep the family roster consistent with this strategy's freshly-synced stage.
        family = strategy_family(settings, name)
        if family:
            sync_family_doc(settings, family)
        generate_indexes(settings)
        summary = {"strategies": [name], "families": [family] if family else []}
    emit(ok(summary))
