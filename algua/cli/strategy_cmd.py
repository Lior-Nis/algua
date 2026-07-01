from __future__ import annotations

import keyword
import re
from pathlib import Path

import typer

from algua.cli._common import ok, registry_conn
from algua.cli.app import app, emit
from algua.cli.errors import json_errors
from algua.config.settings import get_settings
from algua.contracts.registry_metadata import Author, HypothesisStatus
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
from algua.registry.repository import StrategyNotFound, kb_metadata
from algua.registry.store import SqliteStrategyRepository
from algua.strategies.loader import list_strategies

strategy_app = typer.Typer(help="Author and list strategies", no_args_is_help=True)
app.add_typer(strategy_app, name="strategy")

_FAMILY_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")


def family_package_dir(family: str) -> str:
    """The on-disk package dir name for a family slug. `_FAMILY_RE` allows hyphens but Python
    packages can't, so hyphens map to underscores. This is the ONE place the slug->dir transform
    lives (CLI scaffolding + any future doctor check). The registry/kb keep the hyphen slug form."""
    return family.replace("-", "_")

_TEMPLATE = '''\
"""Strategy: {name}. Author `signal` (cross-sectional scores per symbol); the named construction
policy in CONFIG turns scores into target weights."""
from __future__ import annotations

from typing import Any

import pandas as pd

from algua.contracts.types import ExecutionContract
from algua.strategies.base import StrategyConfig

CONFIG = StrategyConfig(
    name="{name}",
    universe=["AAPL", "MSFT", "NVDA"],
    execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
    params={{"lookback": 60}},
    construction="top_k_equal_weight",
    construction_params={{"top_k": 2}},
)


def signal(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """Return a score per symbol (higher = more attractive). NOT weights — the construction policy
    maps scores to weights. See algua/portfolio/construction.py for available policies."""
    lookback = int(params["lookback"])
    wide = view.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    if len(wide) <= lookback:
        return pd.Series(dtype="float64")
    return (wide.iloc[-1] / wide.iloc[-1 - lookback] - 1.0).dropna()
'''


@strategy_app.command("list")
@json_errors
def list_() -> None:
    """List available strategies as JSON."""
    emit(list_strategies())


@strategy_app.command("new")
@json_errors
def new(
    name: str,
    family: str = typer.Option(None, "--family", help="thesis family this belongs to (required)"),
    derived_from: str = typer.Option(None, "--derived-from", help="parent strategy name"),
    tag: list[str] = typer.Option(None, "--tag", help="tag (repeatable)"),
    author: Author = typer.Option(Author.AGENT, "--author"),
    hypothesis_status: HypothesisStatus = typer.Option(
        HypothesisStatus.UNTESTED, "--hypothesis-status"
    ),
    description: str = typer.Option(None, "--description"),
) -> None:
    """Scaffold a new strategy module + kb doc AND register it (registry owns the metadata)."""
    # --- preflight: validate everything before any write ---
    if not name.isidentifier() or keyword.iskeyword(name) or name.startswith("_"):
        raise ValueError(
            f"invalid strategy name {name!r}: must be a valid, non-keyword Python identifier "
            f"not starting with '_' (the loader reserves `_`-prefixed modules as private/temp, so "
            f"such a strategy would be registered but unreachable)"
        )
    if not _FAMILY_RE.match(family or ""):
        raise ValueError(
            "--family is required; use a lowercase slug such as 'momentum' "
            "(a-z, 0-9, hyphen)"
        )
    path = (
        Path(__file__).parent.parent / "strategies" / family_package_dir(family) / f"{name}.py"
    )
    settings = get_settings()
    doc_path = strategy_doc_path(settings, name)
    if path.exists():
        raise ValueError(f"strategy already exists: {path}")
    if doc_path.exists():
        raise ValueError(f"strategy doc already exists: {doc_path}")
    with registry_conn() as conn:
        repo = SqliteStrategyRepository(conn)
        try:
            repo.get(name)
            raise ValueError(f"{name} is already registered")
        except StrategyNotFound:
            pass
        if derived_from is not None:
            if derived_from == name:
                raise ValueError(f"{name} cannot be derived from itself")
            repo.get(derived_from)  # StrategyNotFound if the parent is unknown
        # --- register first (fast, transactional) ---
        rec = repo.add(
            name, family=family, tags=tag or [], author=author,
            hypothesis_status=hypothesis_status, derived_from=derived_from,
            description=description,
        )
        # --- scaffold; roll the registration back on any failure ---
        fam_path: Path = family_doc_path(settings, family) if family else Path("/dev/null")
        fam_created = False
        # Track whether THIS call introduces the family package dir / its __init__.py, so rollback
        # can remove them — a failed scaffold must not leave an empty family package in the tree.
        pkg_dir = path.parent
        pkg_existed = pkg_dir.exists()
        init_py = pkg_dir / "__init__.py"
        init_created = False
        try:
            pkg_dir.mkdir(parents=True, exist_ok=True)
            if not init_py.exists():
                init_py.write_text("")
                init_created = True
            path.write_text(_TEMPLATE.format(name=name))
            doc_path.parent.mkdir(parents=True, exist_ok=True)
            doc_path.write_text(
                scaffold_strategy_doc(name, family=family, derived_from=derived_from)
            )
            family_doc: str | None = None
            if family:
                fam_path.parent.mkdir(parents=True, exist_ok=True)
                if not fam_path.exists():
                    fam_path.write_text(scaffold_family_doc(family))
                    fam_created = True
                family_doc = str(fam_path)
            sync_strategy_doc(settings, name, stage=rec.stage.value, metadata=kb_metadata(rec))
        except Exception as exc:
            repo.delete(name)
            # best-effort: remove half-written files so a retry isn't blocked
            path.unlink(missing_ok=True)
            doc_path.unlink(missing_ok=True)
            if fam_created:
                fam_path.unlink(missing_ok=True)
            # Remove a family package this call introduced — but ONLY if no OTHER strategy module
            # appeared in the meantime (e.g. a concurrent `strategy new` into the same fresh
            # family). Guard the __init__ unlink on the same "no other strategy" condition so we
            # never strand a sibling's file in a now-non-package dir.
            others = (
                [f for f in pkg_dir.glob("*.py") if f.name != "__init__.py"]
                if pkg_dir.exists()
                else []
            )
            if not others:
                if init_created:
                    init_py.unlink(missing_ok=True)
                if not pkg_existed and pkg_dir.exists() and not any(pkg_dir.iterdir()):
                    pkg_dir.rmdir()
            raise ValueError(f"scaffold failed for {name!r}: {exc}") from exc
    emit(ok({"name": name, "path": str(path), "doc": str(doc_path), "family_doc": family_doc}))


@strategy_app.command("doc")
@json_errors
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
        summary = sync_all(settings, stages, metadata={n: kb_metadata(r) for n, r in recs.items()})
    else:
        meta = kb_metadata(recs[name]) if name in recs else None
        if not sync_strategy_doc(settings, name, stage=stages.get(name), metadata=meta):
            raise ValueError(f"no strategy doc for {name!r}; run `strategy new` first")
        # Keep the family roster consistent with this strategy's freshly-synced stage.
        family = strategy_family(settings, name)
        if family:
            sync_family_doc(settings, family)
        generate_indexes(settings)
        summary = {"strategies": [name], "families": [family] if family else []}
    emit(ok(summary))
