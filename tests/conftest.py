"""Suite-wide isolation for developer-environment surfaces a test must never inherit.

Three leaks, all caused by a test resolving a real, developer-machine surface by default when
pytest runs from the repo root.

**1. The knowledge vault (write leak).** ``Settings.knowledge_dir`` defaults to the RELATIVE path
``kb`` — the developer's actual Obsidian vault. The gate commands sync the vault as a side effect
(``_common.sync_kb_doc``, wired into ``backtest --register`` / ``registry transition`` /
``research promote`` / ``paper promote``), and that sync now scaffolds a doc for a registered
strategy that lacks one. Without this fixture, any test exercising those commands would create
``kb/strategies/<fixture-strategy>.md`` in the working tree.

Only the DEFAULT is redirected: a test that sets ``ALGUA_KNOWLEDGE_DIR`` itself (several do, to
assert on the vault they built) still wins, because its own ``monkeypatch.setenv`` runs after this
conftest-level fixture.

**2. The repo-root ``.env`` (read leak).** ``Settings.model_config`` sets ``env_file=".env"``, so
pydantic-settings reads the developer's real ``.env`` from the CWD on every instantiation. That
silently defeats ``monkeypatch.delenv``, which only clears the PROCESS environment: a test that
deletes ``ALGUA_ALPACA_API_KEY`` to assert credentials are ABSENT still gets a real key from the
file, and the assertion inverts. This is not hypothetical — ``test_cli_core.py``'s two ``doctor``
tests (``test_doctor_paper_flag_gates_paper_credentials``,
``test_doctor_advisory_rows_do_not_gate_exit``) passed in CI and in a fresh worktree while failing
for any developer who had actually configured credentials, which is precisely backwards: the suite
was green exactly where the credentials were missing.

Disabling ``env_file`` for the whole suite makes "a clean env" mean what those tests already say it
means, and matches ``get_settings``' own stated contract ("Fresh Settings each call (re-reads env;
keeps tests isolated)"). A test that genuinely needs a value still sets it with
``monkeypatch.setenv``, which is unaffected. Deliberately suite-wide rather than per-test: the
failure mode is silent and machine-dependent, so it must not depend on each future test author
remembering it.
"""

from __future__ import annotations

import pytest

from algua.config.settings import Settings


@pytest.fixture(autouse=True)
def _isolated_knowledge_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_KNOWLEDGE_DIR", str(tmp_path / "kb"))


@pytest.fixture(autouse=True)
def _no_repo_dotenv(monkeypatch):
    """Stop pydantic-settings reading the developer's real ``.env`` (see module docstring)."""
    monkeypatch.setitem(Settings.model_config, "env_file", None)


@pytest.fixture(autouse=True)
def _isolated_db_path(monkeypatch, tmp_path):
    """**3. The registry DB (write leak).** ``Settings.db_path`` defaults to the RELATIVE path
    ``data/algua.db`` — the developer's real, 1.8 MB registry DB. Before the strategy-run-tracking
    branch, an un-``--register``ed ``backtest run`` never touched the DB at all; that branch made
    ``run_backtest_task`` open ``registry_conn()`` UNCONDITIONALLY (to record the run row), so any
    test that exercises a backtest/walk-forward/sweep/gate path without setting
    ``ALGUA_DB_PATH`` itself now silently writes into the real registry the moment it runs on a
    developer machine (every CURRENT test happens to set it, by the established
    ``monkeypatch.setenv("ALGUA_DB_PATH", ...)`` idiom — but nothing enforces that a FUTURE test
    author remembers to). Same rationale as ``_isolated_knowledge_dir`` / ``_no_repo_dotenv``
    above: a silent, machine-dependent failure mode must not depend on each future test author
    remembering the idiom, so it is redirected suite-wide instead.

    Only the DEFAULT is redirected: a test that sets ``ALGUA_DB_PATH`` itself still wins, because
    its own ``monkeypatch.setenv`` runs AFTER this conftest-level fixture and simply overwrites the
    value this fixture set.
    """
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "_default_isolated_registry.db"))


@pytest.fixture(autouse=True)
def _isolated_registries():
    """Snapshot four of the five name->factory registries in ``algua/`` and restore them wholesale
    after each test: ``algua.data.importers``, ``algua.data.providers``, ``algua.execution.broker_
    factory``, and ``algua.tracking.factory``.

    Restore, not delete: the manual ``del _REGISTRY["dummy"]`` pattern this replaces reverts an
    ADDITION but silently keeps an OVERWRITE, and leaks entirely if the test errors first. For the
    broker registry that is a safety concern rather than tidiness -- a fake left registered under a
    live kind changes what a LATER test constructs across the paper/live boundary.

    Four registries, not the three the brief for this task named: ``algua.data.importers`` (local
    vendor-file importers, e.g. FirstRate/Databento) is a *separate* registry from
    ``algua.data.providers`` (network bar providers, e.g. yfinance/Alpaca) -- the manual cleanup in
    ``tests/test_data_ingest_streamed.py`` was on the importers registry, not providers. Both get
    the same fragile ``del``-in-``finally`` treatment today, so both are covered here.

    The fifth, ``algua.features.catalogue._REGISTRY``, is deliberately EXCLUDED: ``load_all_
    factors`` REBINDS the module global (``global _REGISTRY; ...; _REGISTRY = fresh``) rather than
    mutating it in place, so snapshotting it by reference here would go stale the moment a test
    called ``load_all_factors`` -- restoring the OLD dict object at teardown would not undo the
    rebind. It already has its own reset hook, ``catalogue._reset_registry()``, that its own tests
    call directly.
    """
    from algua.data.importers import _REGISTRY as importers
    from algua.data.providers import _REGISTRY as providers
    from algua.execution.broker_factory import _REGISTRY as brokers
    from algua.tracking.factory import _REGISTRY as trackers

    snapshots = [(r, dict(r)) for r in (importers, providers, brokers, trackers)]
    yield
    for registry, snapshot in snapshots:
        registry.clear()
        registry.update(snapshot)
