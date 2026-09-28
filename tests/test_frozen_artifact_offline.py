"""Offline acceptance: the real final-locator verifier with no checkout, Git, uv or network.

A child process verifies a recorded descriptor against a really published bundle and a really
published environment (whose own interpreter runs the isolated probe at its final locator). The
child imports the verifier from a copy of the `algua` package, drops every checkout entry from
`sys.path`, runs from an unrelated directory with no `git` or `uv` on `PATH`, and installs an audit
hook *before importing algua* that refuses any access to the checkout or its Git directories, any
network use and any subprocess except the published environment's interpreter. The dev
environment's third-party packages happen to live under the checkout (`.venv`) and stay readable:
they are the verifier's runtime, not checkout content.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from algua.registry.artifact_contract import BundleDescriptor
from algua.registry.artifact_recording import frozen_deployment_manifest
from algua.registry.artifact_store import publish_bundle
from algua.registry.db import connect, migrate
from algua.registry.environment_contract import EnvironmentDescriptor, EnvironmentKey
from algua.registry.environment_store import publish_environment
from algua.registry.frozen_manifest_contract import FrozenManifest
from algua.registry.frozen_source import FrozenFile
from algua.registry.planner_environment import current_interpreter_identity, inventory_environment
from algua.registry.store import SqliteStrategyRepository
from tests._venv_fixture import SITE_PACKAGES, uv_like_venv
from tests.test_frozen_artifact_ledger import _candidate

REPO = Path(__file__).resolve().parents[1]

CHILD = r"""
import json, os, shutil, sys
from pathlib import Path

config = json.loads(sys.argv[1])
forbidden = [Path(item) for item in config["forbidden"]]
allowed = [Path(item) for item in config["allowed"]]
interpreter = config["interpreter"]
violations = []
spawned = []

def _denied(raw):
    if isinstance(raw, int) or raw is None:
        return False
    path = Path(os.path.abspath(os.fsdecode(raw)))
    if any(path == root or root in path.parents for root in allowed):
        return False
    return any(path == root or root in path.parents for root in forbidden)

_PATH_EVENTS = {"open", "os.listdir", "os.scandir", "os.chdir", "os.mkdir", "os.remove",
                "os.rename", "os.rmdir", "os.symlink", "os.link", "os.chmod", "os.utime",
                "os.truncate", "sqlite3.connect", "shutil.copyfile", "shutil.rmtree"}
_NETWORK_EVENTS = {"socket.connect", "socket.getaddrinfo", "socket.gethostbyname",
                   "socket.gethostbyaddr", "socket.sendto", "socket.sendmsg"}
_SPAWN_EVENTS = {"os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork"}

def audit(event, args):
    if event in _PATH_EVENTS and args and _denied(args[0]):
        violations.append(f"{event}:{args[0]}")
        raise PermissionError("checkout access refused offline")
    if event in _NETWORK_EVENTS:
        violations.append(event)
        raise PermissionError("network refused offline")
    if event == "subprocess.Popen" and os.fspath(args[0]) == interpreter:
        spawned.append(list(map(os.fspath, args[1]))[:3])
    elif event in _SPAWN_EVENTS or event == "subprocess.Popen":
        violations.append(f"{event}:{args[0] if args else ''}")
        raise PermissionError("subprocess refused offline")

sys.path[:] = [config["code"]] + [
    entry for entry in sys.path if not _denied(entry or ".")
]
sys.addaudithook(audit)

from algua.registry.artifact_errors import FrozenArtifactError
from algua.registry.artifact_verification import verify_frozen_artifact
from algua.registry.db import connect
from algua.registry.store import SqliteStrategyRepository

import algua
outcome = {
    "code": str(Path(algua.__file__).resolve().parent),
    "git": shutil.which("git"), "uv": shutil.which("uv"), "cwd": os.getcwd(),
}
try:
    repo = SqliteStrategyRepository(connect(Path(config["db"])))
    result = verify_frozen_artifact(
        repo, config["digest"], store_root=Path(config["store"]))
    outcome.update(ok=True, strategy=result.strategy, artifact_id=result.artifact_id,
                   digest=result.manifest.digest)
except FrozenArtifactError as exc:
    outcome.update(ok=False, error=type(exc).__name__)
outcome["violations"] = violations
outcome["spawned"] = spawned
print(json.dumps(outcome))
"""


def _git_dirs() -> list[str]:
    result = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir", "--git-dir"],
        cwd=REPO, check=True, capture_output=True, text=True,
    )
    return result.stdout.split()


def _bundle_files() -> tuple[FrozenFile, ...]:
    files = (
        FrozenFile("_algua/protocol.json", "100644", b'{"descriptor_version":1}'),
        FrozenFile("_algua/resolved-config.json", "100644", b'{"name":"s"}'),
        FrozenFile("algua/__init__.py", "100644", b"VALUE = 1\n"),
        FrozenFile("algua/strategies/s.py", "100644", b"CONFIG = {'name': 's'}\n"),
    )
    return tuple(sorted(files, key=lambda item: item.path.encode()))


def _publish(tmp_path: Path) -> tuple[Path, Path, FrozenManifest]:
    store = tmp_path / "store"
    files = _bundle_files()
    bundle = BundleDescriptor.from_files(tuple(item.contract_entry for item in files))
    publish_bundle(store, files, bundle)
    built = uv_like_venv(tmp_path / "build")
    (built / SITE_PACKAGES / "six.py").write_text("VERSION = '1.17.0'\n")
    identity = current_interpreter_identity()
    key = EnvironmentKey(
        build_inputs_digest="1" * 64, dependency_hash="d" * 64, interpreter=identity,
        uv_version="uv 0.9.26", create_argv=("uv", "venv"), sync_argv=("uv", "sync"),
    )
    environment = EnvironmentDescriptor(key, inventory_environment(built).digest, identity)
    publish_environment(store, built, environment)
    manifest = FrozenManifest(
        source_ref="a" * 40, code_hash="b" * 32, config_hash="c" * 32,
        dependency_hash="d" * 64, resolved_config={"name": "s"}, universe_name="liquid-us",
        bundle=bundle, environment=environment,
    )
    db = tmp_path / "registry.db"
    conn = connect(db)
    migrate(conn)
    repo = SqliteStrategyRepository(conn)
    _strategy_id, gate_id = _candidate(repo, manifest)
    repo.record_frozen_artifact(
        "s", frozen_deployment_manifest(manifest), research_gate_id=gate_id)
    conn.close()
    return store, db, manifest


def _verify_offline(tmp_path: Path, store: Path, db: Path, manifest: FrozenManifest) -> dict:
    code = tmp_path / "code"
    shutil.copytree(REPO / "algua", code / "algua",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    elsewhere = tmp_path / "elsewhere"
    empty_bin = tmp_path / "empty-bin"
    elsewhere.mkdir()
    empty_bin.mkdir()
    interpreter = store / manifest.environment.locator / "bin/python"
    config = {
        "code": str(code), "db": str(db), "store": str(store), "digest": manifest.digest,
        "interpreter": str(interpreter),
        "forbidden": [str(REPO), *_git_dirs()],
        "allowed": [str(Path(sys.prefix).resolve()), str(Path(sys.prefix))],
    }
    env = {
        "PATH": str(empty_bin), "HOME": str(tmp_path / "home"),
        "GIT_DIR": str(tmp_path / "no-git"), "GIT_CEILING_DIRECTORIES": "/",
        "PYTHONDONTWRITEBYTECODE": "1", "LANG": "C.UTF-8",
    }
    result = subprocess.run(
        [sys.executable, "-c", CHILD, json.dumps(config)], cwd=elsewhere, env=env,
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    outcome = json.loads(result.stdout.strip().splitlines()[-1])
    assert outcome["code"] == str((code / "algua").resolve())
    assert outcome["git"] is None and outcome["uv"] is None
    assert outcome["cwd"] == str(elsewhere)
    assert outcome["violations"] == []
    return outcome


def test_offline_verification_runs_the_real_final_locator_verifier(tmp_path: Path) -> None:
    store, db, manifest = _publish(tmp_path)

    outcome = _verify_offline(tmp_path, store, db, manifest)

    assert outcome["ok"] is True
    assert outcome["strategy"] == "s"
    assert outcome["digest"] == manifest.digest
    assert outcome["artifact_id"] > 0
    # The only process was the published interpreter's isolated probe at its final locator.
    interpreter = str(store / manifest.environment.locator / "bin/python")
    assert outcome["spawned"] == [[interpreter, "-I", "-S"]]


@pytest.mark.parametrize(
    "damage,error",
    [("bundle", "FrozenBundleCorrupt"), ("environment", "FrozenEnvironmentCorrupt")],
)
def test_offline_verification_fails_closed_on_published_drift(
    tmp_path: Path, damage: str, error: str,
) -> None:
    store, db, manifest = _publish(tmp_path)
    if damage == "bundle":
        target = store / manifest.bundle.locator / "algua/__init__.py"
    else:
        target = store / manifest.environment.locator / SITE_PACKAGES / "six.py"
    target.chmod(0o644)
    target.write_text("VALUE = 2\n")
    target.chmod(0o444)

    outcome = _verify_offline(tmp_path, store, db, manifest)

    assert outcome["ok"] is False
    assert outcome["error"] == error
    assert target.read_text() == "VALUE = 2\n"


def test_offline_child_really_refuses_checkout_git_uv_and_network(tmp_path: Path) -> None:
    """The sandboxing harness itself must refuse what it claims to refuse."""
    probe = tmp_path / "probe.py"
    probe.write_text(CHILD.split("sys.addaudithook(audit)")[0] + "sys.addaudithook(audit)\n" + (
        "import socket, subprocess\n"
        "attempts = {}\n"
        f"for name, action in (('checkout', lambda: open({str(REPO / 'pyproject.toml')!r})),\n"
        f"                     ('git', lambda: open({_git_dirs()[0] + '/HEAD'!r})),\n"
        "                     ('network', lambda: socket.getaddrinfo('localhost', 80)),\n"
        "                     ('uv', lambda: subprocess.run(['uv', '--version']))):\n"
        "    try:\n"
        "        action()\n"
        "        attempts[name] = 'allowed'\n"
        "    except Exception:\n"
        "        attempts[name] = 'refused'\n"
        "print(json.dumps({'attempts': attempts, 'violations': len(violations)}))\n"
    ))
    config = {"code": str(tmp_path), "db": "", "store": "", "digest": "",
              "interpreter": "/nonexistent", "forbidden": [str(REPO), *_git_dirs()],
              "allowed": [str(Path(sys.prefix).resolve()), str(Path(sys.prefix))]}
    result = subprocess.run(
        [sys.executable, str(probe), json.dumps(config)], cwd=tmp_path,
        env={"PATH": os.environ["PATH"], "LANG": "C.UTF-8"}, capture_output=True, text=True,
        timeout=60,
    )
    outcome = json.loads(result.stdout.strip().splitlines()[-1])

    assert outcome == {
        "attempts": {"checkout": "refused", "git": "refused", "network": "refused",
                     "uv": "refused"},
        "violations": 4,
    }
