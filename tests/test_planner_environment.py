from __future__ import annotations

import os
import sys
import venv
from pathlib import Path

import pytest

from algua.registry.frozen_source import FrozenFile
from algua.registry.planner_environment import (
    CREATE_FLAGS,
    SYNC_FLAGS,
    EnvironmentIncompatible,
    build_environment_key,
    current_interpreter_identity,
    inventory_environment,
    provision_environment,
    scrubbed_environment,
    validate_lock,
    verify_environment,
)


def _inputs() -> tuple[FrozenFile, ...]:
    return (
        FrozenFile(".python-version", "100644", b"3.12\n"),
        FrozenFile("pyproject.toml", "100644", b"[project]\nname='algua'\nversion='0'\n"),
        FrozenFile("uv.lock", "100644", b"version=1\n[[package]]\nname='x'\n"
                   b"version='1'\nsource={registry='https://pypi.org/simple'}\n"
                   b"wheels=[{url='https://example/x.whl',hash='sha256:aa'}]\n"),
    )


def test_environment_key_binds_inputs_interpreter_uv_and_exact_flags() -> None:
    key = build_environment_key(_inputs(), "a" * 64, uv_version="uv 0.9.26")
    assert key.interpreter == current_interpreter_identity()
    assert key.create_argv == CREATE_FLAGS
    assert key.sync_argv == SYNC_FLAGS
    assert key.digest != build_environment_key(
        _inputs(), "a" * 64, uv_version="uv 0.9.27").digest


def test_normative_flags_exclude_project_resolution_and_links() -> None:
    assert "--relocatable" in CREATE_FLAGS
    for flag in (
        "--locked", "--no-dev", "--no-default-groups", "--no-editable",
        "--no-install-project", "--no-install-workspace", "--no-install-local",
        "--no-build", "--no-python-downloads", "--no-env-file", "--no-config",
    ):
        assert flag in SYNC_FLAGS
    assert SYNC_FLAGS[SYNC_FLAGS.index("--link-mode") + 1] == "copy"


def test_scrubbed_environment_has_no_inherited_authority(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALGUA_DB_PATH", "/secret")
    monkeypatch.setenv("ALPACA_API_KEY", "secret")
    monkeypatch.setenv("PYTHONPATH", "/checkout")
    monkeypatch.setenv("HTTPS_PROXY", "http://secret")
    env = scrubbed_environment(Path("/tmp/bin"))
    assert env == {
        "HOME": env["HOME"], "PATH": "/tmp/bin", "PYTHONDONTWRITEBYTECODE": "1",
        "LANG": env["LANG"], "LC_ALL": env["LC_ALL"],
    }


@pytest.mark.parametrize(
    "source",
    ["editable='.'", "path='../x'", "git='https://example/repo'", "url='file:///tmp/x.whl'"],
)
def test_lock_rejects_local_or_vcs_dependencies(source: str) -> None:
    lock = f"version=1\n[[package]]\nname='bad'\nversion='1'\nsource={{ {source} }}\n".encode()
    with pytest.raises(EnvironmentIncompatible):
        validate_lock(lock)


def test_lock_requires_a_wheel_for_every_registry_package() -> None:
    lock = (b"version=1\n[[package]]\nname='bad'\nversion='1'\n"
            b"source={registry='https://pypi.org/simple'}\nsdist={url='https://example/x.tgz'}\n")
    with pytest.raises(EnvironmentIncompatible, match="wheel"):
        validate_lock(lock)


def test_python_pin_must_match_running_minor() -> None:
    wrong = list(_inputs())
    wrong[0] = FrozenFile(".python-version", "100644", b"9.9\n")
    with pytest.raises(EnvironmentIncompatible, match="Python pin"):
        build_environment_key(tuple(wrong), "a" * 64, uv_version="uv 0.9.26")
    assert sys.version_info[:2] == (3, 12)


def test_environment_inventory_and_isolated_probe(tmp_path: Path) -> None:
    env = tmp_path / "env"
    venv.EnvBuilder(with_pip=False).create(env)
    lib64 = env / "lib64"
    if lib64.is_symlink():
        lib64.unlink()
    site = env / "lib/python3.12/site-packages"
    package = site / "numpy"
    dist = site / "numpy-2.3.3.dist-info"
    package.mkdir(parents=True)
    dist.mkdir()
    (package / "__init__.py").write_text("__version__ = '2.3.3'\n")
    (dist / "METADATA").write_text("Name: numpy\nVersion: 2.3.3\n")
    inventory = inventory_environment(env)
    assert [(item.name, item.version) for item in inventory.distributions] == [("numpy", "2.3.3")]
    assert inventory.digest
    verify_environment(env, current_interpreter_identity(), inventory.digest)


def test_environment_inventory_rejects_algua_hardlinks_and_unexpected_symlinks(
    tmp_path: Path,
) -> None:
    env = tmp_path / "env"
    site = env / "lib/python3.12/site-packages"
    dist = site / "algua-1.dist-info"
    dist.mkdir(parents=True)
    (dist / "METADATA").write_text("Name: algua\nVersion: 1\n")
    with pytest.raises(EnvironmentIncompatible, match="Algua"):
        inventory_environment(env)
    (dist / "METADATA").write_text("Name: safe\nVersion: 1\n")
    os.symlink("METADATA", dist / "bad-link")
    with pytest.raises(EnvironmentIncompatible, match="symlink"):
        inventory_environment(env)


def test_provision_uses_exact_uv_commands_and_private_inputs(tmp_path: Path) -> None:
    calls: list[tuple[list[str], Path, dict[str, str]]] = []
    environment = tmp_path / "environment"
    build_root = tmp_path / "inputs"

    def runner(argv, *, cwd, env, check, capture_output, text):
        calls.append((argv, cwd, env))
        if argv[1] == "venv":
            venv.EnvBuilder(with_pip=False).create(environment)
            lib64 = environment / "lib64"
            if lib64.is_symlink():
                lib64.unlink()
        return type("Completed", (), {"returncode": 0, "stderr": ""})()

    key = build_environment_key(_inputs(), "a" * 64, uv_version="uv 0.9.26")
    inventory = provision_environment(
        build_root, environment, _inputs(), key, runner=runner)

    assert [call[0] for call in calls] == [
        [item.replace("<exact-current-interpreter>", sys.executable).replace(
            "<environment>", str(environment)) for item in CREATE_FLAGS],
        [item.replace("<private-build-input-root>", str(build_root)) for item in SYNC_FLAGS],
    ]
    assert calls[1][2]["VIRTUAL_ENV"] == str(environment)
    assert (build_root / "uv.lock").read_bytes() == _inputs()[2].data
    assert inventory.digest
