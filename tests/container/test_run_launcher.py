# tests/container/test_run_launcher.py
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN = REPO_ROOT / "scripts" / "run.sh"


@pytest.fixture
def stub_docker(tmp_path):
    """A fake `docker` on PATH so the happy path doesn't need a daemon."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "docker.log"
    stub = bindir / "docker"
    stub.write_text(f'#!/usr/bin/env bash\necho "$*" >> "{log}"\nexit 0\n')
    stub.chmod(0o755)
    return bindir, log


def _run(args, cwd, stub_bindir, env_extra=None):
    env = {**os.environ, "PATH": f"{stub_bindir}:{os.environ['PATH']}"}
    env.update(env_extra or {})
    return subprocess.run(
        [str(RUN), *args], cwd=cwd, env=env, capture_output=True, text=True
    )


@pytest.mark.parametrize("bad", ["a/b", "..", ".", "has space", "../escape", ""])
def test_rejects_invalid_run_id(bad, tmp_path, stub_docker):
    bindir, log = stub_docker
    r = _run([bad, "doctor"], tmp_path, bindir)
    assert r.returncode == 2
    assert not (tmp_path / "runs").exists() or not list((tmp_path / "runs").iterdir())
    assert not log.exists(), "docker must not run for an invalid RUN_ID"


def test_collision_without_reuse_is_refused(tmp_path, stub_docker):
    bindir, log = stub_docker
    (tmp_path / "runs" / "alpha").mkdir(parents=True)
    r = _run(["alpha", "doctor"], tmp_path, bindir)
    assert r.returncode == 3
    assert not log.exists()


def test_valid_new_run_creates_dir_and_execs_docker(tmp_path, stub_docker):
    bindir, log = stub_docker
    r = _run(["alpha", "doctor"], tmp_path, bindir)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "runs" / "alpha").is_dir()
    assert "compose run --rm algua doctor" in log.read_text()


def test_reuse_flag_allows_existing_dir(tmp_path, stub_docker):
    bindir, log = stub_docker
    (tmp_path / "runs" / "alpha").mkdir(parents=True)
    r = _run(["alpha", "doctor"], tmp_path, bindir, {"ALGUA_REUSE": "1"})
    assert r.returncode == 0, r.stderr
    assert "compose run --rm algua doctor" in log.read_text()


def test_run_launcher_is_executable():
    assert os.access(RUN, os.X_OK)
