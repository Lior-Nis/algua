"""Suite-wide isolation for filesystem surfaces a test must never write into for real.

``Settings.knowledge_dir`` defaults to the RELATIVE path ``kb`` — the developer's actual Obsidian
vault when pytest runs from the repo root. The gate commands sync the vault as a side effect
(``_common.sync_kb_doc``, wired into ``backtest --register`` / ``registry transition`` /
``research promote`` / ``paper promote``), and that sync now scaffolds a doc for a registered
strategy that lacks one. Without this fixture, any test exercising those commands would create
``kb/strategies/<fixture-strategy>.md`` in the working tree.

Only the DEFAULT is redirected: a test that sets ``ALGUA_KNOWLEDGE_DIR`` itself (several do, to
assert on the vault they built) still wins, because its own ``monkeypatch.setenv`` runs after this
conftest-level fixture.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_knowledge_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_KNOWLEDGE_DIR", str(tmp_path / "kb"))
