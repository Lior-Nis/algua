"""Pinned interpreter and uv policy for shared frozen planner environments."""
from __future__ import annotations

import platform
import re
import shutil
import subprocess
import sys
import sysconfig
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from algua.primitives.bounded_subprocess import (
    BoundedCompletion,
    OutputLimitExceeded,
    run_bounded,
)
from algua.registry.artifact_contract import BuildInputs
from algua.registry.environment_contract import (
    EnvironmentKey,
    InstalledDistribution,
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


def locked_wheels(raw: bytes) -> dict[str, tuple[str, str]]:
    """Validate the committed lock and map every locked wheel URL to its canonical identity.

    Total over arbitrary bytes: any malformation is `EnvironmentIncompatible`. Each registry
    package must carry a PEP 503 canonical name and a bounded version, and no wheel URL may be
    locked for two packages.
    """
    try:
        payload = tomllib.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise EnvironmentIncompatible("committed uv lock is invalid") from exc
    packages = payload.get("package")
    if not isinstance(packages, list):
        raise EnvironmentIncompatible("committed uv lock has no package inventory")
    result: dict[str, tuple[str, str]] = {}
    for package in packages:
        if not isinstance(package, dict):
            raise EnvironmentIncompatible("committed uv lock package is invalid")
        source = package.get("source")
        name: Any = package.get("name")
        if name == "algua" and source == {"editable": "."}:
            continue
        if not isinstance(source, dict) or set(source) != {"registry"}:
            raise EnvironmentIncompatible(
                "local, editable, URL and VCS dependencies are unsupported")
        version: Any = package.get("version")
        try:
            identity = InstalledDistribution(name, version)
        except ValueError as exc:
            raise EnvironmentIncompatible(
                "locked package name or version is not canonical") from exc
        wheels = package.get("wheels")
        if not isinstance(wheels, list) or not wheels:
            raise EnvironmentIncompatible("every locked registry package requires a wheel")
        for wheel in wheels:
            if not isinstance(wheel, dict):
                raise EnvironmentIncompatible("locked wheel metadata is invalid")
            url = wheel.get("url")
            digest = wheel.get("hash")
            if not isinstance(url, str) or not isinstance(digest, str):
                raise EnvironmentIncompatible("locked wheel URL or hash is not canonical")
            # `urlsplit` silently strips tabs, newlines and leading spaces and normalizes the
            # scheme or an empty query/fragment, so only a printable URL that round-trips exactly
            # is one canonical key for duplicate detection and outage evidence.
            if not url.isprintable():
                raise EnvironmentIncompatible("locked wheel URL contains non-printable characters")
            try:
                parsed = urlsplit(url)
                netloc = parsed.hostname
            except ValueError as exc:
                raise EnvironmentIncompatible("locked wheel URL is malformed") from exc
            if (
                urlunsplit(parsed) != url or parsed.scheme != "https" or not netloc
                or parsed.username is not None or parsed.password is not None or parsed.fragment
                or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None
            ):
                raise EnvironmentIncompatible("locked wheel URL or hash is not canonical")
            if url in result:
                raise EnvironmentIncompatible("a wheel URL is locked for more than one package")
            result[url] = (identity.name, identity.version)
    return result


def validate_lock(raw: bytes) -> None:
    locked_wheels(raw)


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
