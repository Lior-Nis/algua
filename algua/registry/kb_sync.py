"""``sync_kb_doc`` -- the best-effort kb-doc re-sync side effect shared by every stage-mutating
gate command.

Moved out of ``algua.cli._common`` because it opens its own registry connection: living in
``algua.registry`` (not ``algua.knowledge``) respects the import-linter contract "knowledge layer
stays off cli, registry, backtest, live, execution" (knowledge->registry is forbidden), while
registry->knowledge is legal.
"""

from __future__ import annotations

import sys

from algua.config.settings import get_settings
from algua.registry.db import registry_conn


def sync_kb_doc(name: str) -> None:
    """Best-effort: re-sync ``name``'s kb doc + family roster + indexes to current registry truth.

    The transactional side effect (#331) wired into every stage-mutating gate command (research
    promote, paper promote, backtest --register, registry transition) so the Obsidian vault is
    never stale-by-default. Two invariants make it safe:

    * OUT OF TRANSACTION — it opens its OWN short registry connection, which callers invoke only
      AFTER their write transaction has committed and closed, so vault file I/O never runs while a
      registry write lock is held.
    * NON-FATAL — any failure is swallowed (with a one-line stderr warning); a stale vault is a
      curation gap, never a reason to break or roll back a real promotion. The binding audit (the
      ``gate_evaluations`` row + result JSON) is always-on and reproducible-by-hash regardless.

    Lazy imports keep the mlflow-importing knowledge layer off the hot path of unrelated commands.
    """
    try:
        from algua.knowledge.sync import sync_strategy_and_dependents
        from algua.registry.repository import kb_metadata
        from algua.registry.store import SqliteStrategyRepository

        with registry_conn() as conn:
            rec = SqliteStrategyRepository(conn).get(name)
        sync_strategy_and_dependents(
            get_settings(), name, stage=rec.stage.value, metadata=kb_metadata(rec)
        )
    except Exception as exc:  # noqa: BLE001 — vault curation must never break a promotion
        print(f"warning: kb doc sync for {name!r} failed: {exc}", file=sys.stderr)
