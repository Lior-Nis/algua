"""Pinned interpreter and uv policy for shared frozen planner environments."""
from __future__ import annotations

import platform
import shutil
import subprocess
import sys
import sysconfig
from collections.abc import Callable
from pathlib import Path
from typing import Any

from algua.primitives.bounded_subprocess import (
    BoundedCompletion,
    OutputLimitExceeded,
    run_bounded,
)
from algua.registry.artifact_contract import BuildInputs
from algua.registry.environment_contract import (
    EnvironmentKey,
    InstalledInventory,
    InterpreterIdentity,
)
from algua.registry.frozen_source import FrozenFile
from algua.registry.planner_environment_errors import (
    EnvironmentIncompatible,
    EnvironmentUnavailable,
)
from algua.registry.planner_environment_inventory import (
    inventory_environment,
    scrubbed_environment,
)
from algua.registry.planner_environment_lock import locked_wheels, validate_lock
from algua.registry.planner_environment_outage import is_locked_wheel_outage
from algua.registry.planner_environment_probe import verify_environment

CREATE_FLAGS = (
    "uv", "venv", "--relocatable", "--python", "<exact-current-interpreter>",
    "--no-python-downloads", "<environment>",
)
SYNC_FLAGS = (
    "uv", "sync", "--project", "<private-build-input-root>", "--active", "--locked",
    "--no-dev", "--no-default-groups", "--no-editable", "--no-install-project",
    "--no-install-workspace", "--no-install-local", "--no-build", "--no-python-downloads",
    "--link-mode", "copy", "--no-env-file", "--no-config", "--no-progress",
)
_UV_VERSION_TIMEOUT_SECONDS = 30
_UV_VERSION_OUTPUT_BYTES = 4096
_UV_TIMEOUT_SECONDS = 900
_UV_OUTPUT_BYTES = 1024 * 1024
# Launching, output overflow and an unclassified timeout are never evidence of a temporary
# locked-wheel outage, so they stay inside the non-retryable incompatibility boundary.
_UV_FAILURES = (OSError, subprocess.SubprocessError, OutputLimitExceeded)
UvRunner = Callable[..., BoundedCompletion]


def installer_version() -> str:
    """Return the exact uv version used by the keyed provisioning policy."""
    uv = shutil.which("uv")
    if uv is None:
        raise EnvironmentIncompatible("uv is unavailable for frozen environment construction")
    try:
        result = run_bounded(
            [uv, "--version"], env=scrubbed_environment(Path(uv).parent),
            timeout=_UV_VERSION_TIMEOUT_SECONDS, max_stdout=_UV_VERSION_OUTPUT_BYTES,
            max_stderr=_UV_VERSION_OUTPUT_BYTES,
        )
    except _UV_FAILURES as exc:
        raise EnvironmentIncompatible("uv version could not be determined") from exc
    if result.returncode != 0:
        raise EnvironmentIncompatible("uv version could not be determined")
    try:
        version = result.stdout.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise EnvironmentIncompatible("uv returned an invalid version identity") from exc
    if not version or len(version) > 128 or "\n" in version:
        raise EnvironmentIncompatible("uv returned an invalid version identity")
    return version


def current_interpreter_identity() -> InterpreterIdentity:
    return InterpreterIdentity(
        implementation=platform.python_implementation(),
        version=platform.python_version(),
        cache_tag=sys.implementation.cache_tag or "unknown",
        soabi=sysconfig.get_config_var("SOABI") or "unknown",
        platform_tag=sysconfig.get_platform(),
        os_name=platform.system().lower(),
        machine=platform.machine().lower(),
    )


def _by_path(inputs: tuple[FrozenFile, ...]) -> dict[str, FrozenFile]:
    result = {item.path: item for item in inputs}
    expected = {".python-version", "pyproject.toml", "uv.lock"}
    if set(result) != expected:
        raise EnvironmentIncompatible("environment build inputs are incomplete")
    return result


def build_environment_key(
    inputs: tuple[FrozenFile, ...], dependency_hash: str, *, uv_version: str,
) -> EnvironmentKey:
    by_path = _by_path(inputs)
    try:
        pin = by_path[".python-version"].data.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise EnvironmentIncompatible("Python pin is not UTF-8") from exc
    current_minor = f"{sys.version_info.major}.{sys.version_info.minor}"
    if pin != current_minor:
        raise EnvironmentIncompatible("Python pin does not match the running interpreter")
    validate_lock(by_path["uv.lock"].data)
    entries = tuple(item.contract_entry for item in inputs)
    return EnvironmentKey(
        build_inputs_digest=BuildInputs(entries).digest,
        dependency_hash=dependency_hash,
        interpreter=current_interpreter_identity(),
        uv_version=uv_version,
        create_argv=CREATE_FLAGS,
        sync_argv=SYNC_FLAGS,
    )


def materialize_argv(template: tuple[str, ...], substitutions: dict[str, Path]) -> list[str]:
    result: list[str] = []
    for item in template:
        replacement: Any = substitutions.get(item)
        result.append(str(replacement) if replacement is not None else item)
    return result


def _expanded(template: tuple[str, ...], replacements: dict[str, str]) -> list[str]:
    return [replacements.get(item, item) for item in template]


def _run_uv(
    runner: UvRunner, argv: list[str], *, cwd: Path, env: dict[str, str],
) -> BoundedCompletion:
    return runner(
        argv, cwd=cwd, env=env, timeout=_UV_TIMEOUT_SECONDS, max_stdout=_UV_OUTPUT_BYTES,
        max_stderr=_UV_OUTPUT_BYTES,
    )


def provision_environment(
    build_root: Path,
    environment: Path,
    inputs: tuple[FrozenFile, ...],
    key: EnvironmentKey,
    *,
    runner: UvRunner = run_bounded,
) -> InstalledInventory:
    """Create one private environment; publication remains a separate atomic step."""
    if build_root.exists() or environment.exists():
        raise EnvironmentIncompatible("environment staging paths must be absent")
    build_root.mkdir(parents=True, mode=0o700)
    private_home = build_root / ".home"
    private_home.mkdir(mode=0o700)
    for item in inputs:
        destination = build_root / item.path
        destination.write_bytes(item.data)
        destination.chmod(0o600)
    uv = shutil.which("uv")
    if uv is None:
        raise EnvironmentIncompatible("uv is unavailable for frozen environment construction")
    observed_key = build_environment_key(
        inputs, key.dependency_hash, uv_version=installer_version(),
    )
    if observed_key != key:
        raise EnvironmentIncompatible("environment provisioning key drifted")
    base_env = scrubbed_environment(Path(uv).parent, home=private_home)
    create = _expanded(CREATE_FLAGS, {
        "<exact-current-interpreter>": sys.executable,
        "<environment>": str(environment),
    })
    sync = _expanded(SYNC_FLAGS, {"<private-build-input-root>": str(build_root)})
    try:
        created = _run_uv(runner, create, cwd=build_root, env=base_env)
    except _UV_FAILURES as exc:
        raise EnvironmentIncompatible("locked environment creation failed") from exc
    if created.returncode != 0:
        raise EnvironmentIncompatible("locked environment creation failed")
    lib64 = environment / "lib64"
    if lib64.is_symlink() and lib64.resolve() == (environment / "lib").resolve():
        lib64.unlink()
    sync_env = {**base_env, "VIRTUAL_ENV": str(environment)}
    try:
        synced = _run_uv(runner, sync, cwd=build_root, env=sync_env)
    except _UV_FAILURES as exc:
        raise EnvironmentIncompatible("locked environment provisioning failed") from exc
    if synced.returncode != 0:
        wheels = locked_wheels(_by_path(inputs)["uv.lock"].data)
        if is_locked_wheel_outage(synced.stdout, synced.stderr, wheels):
            raise EnvironmentUnavailable("a compatible locked wheel is temporarily unavailable")
        raise EnvironmentIncompatible("locked environment provisioning failed")
    inventory = inventory_environment(environment)
    verify_environment(environment, key.interpreter, inventory.digest)
    return inventory
