# tests/container/test_entrypoint_guard.py
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT = REPO_ROOT / "scripts" / "entrypoint.sh"


@pytest.fixture
def stub_uv(tmp_path):
    """A fake `uv` on PATH that logs its args and exits 0, so we never run the real CLI."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "uv.log"
    stub = bindir / "uv"
    stub.write_text(f'#!/usr/bin/env bash\necho "$*" >> "{log}"\nexit 0\n')
    stub.chmod(0o755)
    return bindir, log


def _run(args, env_extra, stub_bindir):
    env = {**os.environ, "PATH": f"{stub_bindir}:{os.environ['PATH']}"}
    env.update(env_extra)
    return subprocess.run(
        [str(ENTRYPOINT), *args], env=env, capture_output=True, text=True
    )


def test_refuses_promote_on_per_run_db(stub_uv):
    bindir, log = stub_uv
    r = _run(["research", "promote", "s"],
             {"ALGUA_DB_PATH": "/app/runs/alpha/algua.db"}, bindir)
    assert r.returncode == 4
    assert not log.exists(), "uv must NOT be invoked when the guard fires"
    assert "refusing 'research promote'" in r.stderr


def test_allows_promote_on_per_run_db_with_explicit_optin(stub_uv):
    bindir, log = stub_uv
    r = _run(["research", "promote", "s"],
             {"ALGUA_DB_PATH": "/app/runs/alpha/algua.db", "ALGUA_ALLOW_PROMOTE": "1"}, bindir)
    assert r.returncode == 0
    assert "research promote s" in log.read_text()


def test_allows_promote_against_authoritative_db(stub_uv):
    bindir, log = stub_uv
    r = _run(["research", "promote", "s"],
             {"ALGUA_DB_PATH": "/data/algua.db"}, bindir)
    assert r.returncode == 0
    assert "research promote s" in log.read_text()


def test_non_promote_commands_pass_through(stub_uv):
    bindir, log = stub_uv
    r = _run(["doctor"], {"ALGUA_DB_PATH": "/app/runs/alpha/algua.db"}, bindir)
    assert r.returncode == 0
    assert "doctor" in log.read_text()


def test_entrypoint_is_executable():
    assert os.access(ENTRYPOINT, os.X_OK)
