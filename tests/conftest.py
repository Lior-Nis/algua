"""Suite-wide isolation for developer-environment surfaces a test must never inherit.

Two leaks, both caused by ``Settings`` resolving defaults relative to the repo root when pytest
runs from there.

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
def _isolated_registries():
    """Snapshot every name->factory registry and restore it wholesale after each test.

    Restore, not delete: the manual ``del _REGISTRY["dummy"]`` pattern this replaces reverts an
    ADDITION but silently keeps an OVERWRITE, and leaks entirely if the test errors first. For the
    broker registry that is a safety concern rather than tidiness -- a fake left registered under a
    live kind changes what a LATER test constructs across the paper/live boundary.

    Four registries, not the three the brief for this task named: ``algua.data.importers`` (local
    vendor-file importers, e.g. FirstRate/Databento) is a *separate* registry from
    ``algua.data.providers`` (network bar providers, e.g. yfinance/Alpaca) -- the manual cleanup in
    ``tests/test_data_ingest_streamed.py`` was on the importers registry, not providers. Both get
    the same fragile ``del``-in-``finally`` treatment today, so both are covered here.
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
