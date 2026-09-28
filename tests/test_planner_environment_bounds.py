from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pytest

from algua.registry import planner_environment_inventory as inventory_module
from algua.registry.environment_contract import (
    MAX_DISTRIBUTION_METADATA_BYTES,
    MAX_ENVIRONMENT_BYTES,
    MAX_ENVIRONMENT_FILE_BYTES,
    MAX_ENVIRONMENT_FILES,
)
from algua.registry.planner_environment_errors import EnvironmentIncompatible
from algua.registry.planner_environment_inventory import inventory_environment
from tests._venv_fixture import SITE_PACKAGES, uv_like_venv


def _environment(tmp_path: Path) -> Path:
    env = uv_like_venv(tmp_path / "env")
    site = env / SITE_PACKAGES
    (site / "numpy").mkdir()
    (site / "numpy/__init__.py").write_text("__version__ = '2.3.3'\n")
    (site / "numpy-2.3.3.dist-info").mkdir()
    (site / "numpy-2.3.3.dist-info/METADATA").write_text("Name: numpy\nVersion: 2.3.3\n")
    return env


def _regular_files(env: Path) -> list[Path]:
    return [
        Path(directory) / name
        for directory, _dirs, names in os.walk(env)
        for name in names
        if not (Path(directory) / name).is_symlink()
    ]


def _count_calls(monkeypatch: pytest.MonkeyPatch, name: str) -> list[Path]:
    calls: list[Path] = []
    real = getattr(inventory_module, name)

    def counted(*args, **kwargs):
        calls.append(args[-1] if name == "_interpreter_link" else args[0])
        return real(*args, **kwargs)

    monkeypatch.setattr(inventory_module, name, counted)
    return calls


def _count_hashes(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    return _count_calls(monkeypatch, "_file_digest")


@pytest.mark.parametrize(
    "relative", ["lib/python3.12/site-packages/empty_pkg", "share/nested/empty", "include"],
)
def test_uninventoried_empty_directories_are_rejected(tmp_path: Path, relative: str) -> None:
    env = _environment(tmp_path)
    inventory_environment(env)
    (env / relative).mkdir(parents=True)

    with pytest.raises(EnvironmentIncompatible, match="directory"):
        inventory_environment(env)


def test_file_count_is_bounded_before_hashing_or_growth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _environment(tmp_path)
    baseline = inventory_environment(env)
    entries = len(baseline.files) + len(baseline.interpreter_links)
    monkeypatch.setattr(inventory_module, "MAX_ENVIRONMENT_FILES", entries)
    assert inventory_environment(env) == baseline

    (env / SITE_PACKAGES / "zz-empty.py").touch()
    hashed = _count_hashes(monkeypatch)
    linked = _count_calls(monkeypatch, "_interpreter_link")
    with pytest.raises(EnvironmentIncompatible, match="file-count"):
        inventory_environment(env)
    # Exactly `entries` entries were processed; the one past the bound was never read.
    assert len(hashed) + len(linked) == entries


def test_per_file_bytes_are_bounded_before_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _environment(tmp_path)
    largest = max(_regular_files(env), key=lambda path: path.stat().st_size)
    monkeypatch.setattr(inventory_module, "MAX_ENVIRONMENT_FILE_BYTES", largest.stat().st_size)
    inventory_environment(env)

    monkeypatch.setattr(
        inventory_module, "MAX_ENVIRONMENT_FILE_BYTES", largest.stat().st_size - 1)
    hashed = _count_hashes(monkeypatch)
    with pytest.raises(EnvironmentIncompatible, match="per-file"):
        inventory_environment(env)
    assert largest not in hashed


def test_aggregate_bytes_are_bounded_before_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _environment(tmp_path)
    total = sum(path.stat().st_size for path in _regular_files(env))
    monkeypatch.setattr(inventory_module, "MAX_ENVIRONMENT_BYTES", total)
    inventory_environment(env)

    monkeypatch.setattr(inventory_module, "MAX_ENVIRONMENT_BYTES", total - 1)
    hashed = _count_hashes(monkeypatch)
    with pytest.raises(EnvironmentIncompatible, match="aggregate"):
        inventory_environment(env)
    assert len(hashed) < len(_regular_files(env))


def test_streamed_file_digest_refuses_growth_past_its_bound(tmp_path: Path) -> None:
    path = tmp_path / "growing"
    path.write_bytes(b"x" * 10)

    assert inventory_module._file_digest(path, 10)[0] == 10
    with pytest.raises(EnvironmentIncompatible, match="per-file"):
        inventory_module._file_digest(path, 9)


def _record_reads(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    requested: list[int] = []
    real_open = Path.open

    class Recording:
        def __init__(self, handle):
            self._handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self._handle.close()

        def read(self, size=-1):
            requested.append(size)
            return self._handle.read(size)

    monkeypatch.setattr(Path, "open", lambda self, *args, **kwargs: Recording(
        real_open(self, *args, **kwargs)))
    return requested


@pytest.mark.parametrize("reader", ["_file_digest", "_read_metadata"])
def test_bounded_reads_never_request_bytes_past_the_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reader: str,
) -> None:
    path = tmp_path / "large"
    path.write_bytes(b"x" * (3 * 1024 * 1024))
    requested = _record_reads(monkeypatch)

    with pytest.raises(EnvironmentIncompatible):
        getattr(inventory_module, reader)(path, 10)

    assert requested and all(0 <= size <= 11 for size in requested)


def test_aggregate_growth_after_stat_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _environment(tmp_path)
    regular = _regular_files(env)
    total = sum(path.stat().st_size for path in regular)
    monkeypatch.setattr(inventory_module, "MAX_ENVIRONMENT_BYTES", total)
    real = inventory_module._file_digest
    calls: list[Path] = []

    def grown(path: Path, limit: int) -> tuple[int, str]:
        calls.append(path)
        size, digest = real(path, limit)
        # The last file read grew between its lstat pre-check and the streamed read; no later
        # pre-check exists to notice the inflated total.
        return (size + 1 if len(calls) == len(regular) else size), digest

    monkeypatch.setattr(inventory_module, "_file_digest", grown)
    with pytest.raises(EnvironmentIncompatible, match="aggregate"):
        inventory_environment(env)
    assert len(calls) == len(regular)


def test_a_directory_holding_only_interpreter_links_is_inventoried(tmp_path: Path) -> None:
    env = _environment(tmp_path)
    for script in (env / "bin").iterdir():
        if not script.is_symlink():
            script.unlink()

    inventory = inventory_environment(env)

    assert not any(item.path.startswith("bin/") for item in inventory.files)
    assert {link.path for link in inventory.interpreter_links} == {
        "bin/python", "bin/python3", "bin/python3.12"}


def _malformed_path(env: Path, shape: str) -> Path:
    site = env / SITE_PACKAGES
    if shape == "overlong":
        directory = site
        for index in range(5):
            directory = directory / (f"{index}" * 250)
        directory.mkdir(parents=True)
        return directory / "payload.py"
    if shape == "undecodable":
        return Path(os.fsdecode(bytes(site) + b"/\xffpayload.py"))
    names = {
        "non-nfc": "café.py",
        "trailing-space": "payload.py ",
        "trailing-dot": "payload.",
        "backslash": "pay\\load.py",
    }
    return site / names[shape]


@pytest.mark.parametrize(
    "shape",
    ["overlong", "undecodable", "non-nfc", "trailing-space", "trailing-dot", "backslash"],
)
def test_malformed_paths_are_refused_before_their_content_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shape: str,
) -> None:
    env = _environment(tmp_path)
    target = _malformed_path(env, shape)
    target.write_bytes(b"x" * 4096)
    hashed = _count_hashes(monkeypatch)

    with pytest.raises(EnvironmentIncompatible, match="path"):
        inventory_environment(env)

    assert target not in hashed


def test_distribution_metadata_is_read_through_an_explicit_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _environment(tmp_path)
    metadata = env / SITE_PACKAGES / "numpy-2.3.3.dist-info/METADATA"
    size = metadata.stat().st_size
    monkeypatch.setattr(inventory_module, "MAX_DISTRIBUTION_METADATA_BYTES", size)
    inventory_environment(env)

    monkeypatch.setattr(inventory_module, "MAX_DISTRIBUTION_METADATA_BYTES", size - 1)
    with pytest.raises(EnvironmentIncompatible, match="metadata"):
        inventory_environment(env)


def test_bounded_metadata_read_never_reads_past_its_bound(tmp_path: Path) -> None:
    path = tmp_path / "METADATA"
    path.write_text("Name: numpy\nVersion: 2.3.3\n" + "x" * 100)

    with pytest.raises(EnvironmentIncompatible, match="metadata"):
        inventory_module._read_metadata(path, 20)
    assert inventory_module._read_metadata(path, path.stat().st_size).startswith("Name: numpy")


def test_non_utf8_metadata_is_incompatible(tmp_path: Path) -> None:
    env = _environment(tmp_path)
    (env / SITE_PACKAGES / "numpy-2.3.3.dist-info/METADATA").write_bytes(b"Name: \xff\n")

    with pytest.raises(EnvironmentIncompatible, match="metadata"):
        inventory_environment(env)


def test_protected_environment_bounds_are_explicit() -> None:
    # Sized from the locked no-dev environment built with the normative uv flags (Story 1.3b
    # chunk-2 evidence): 23,849 files, 861.4 MiB, largest file 160.0 MiB, largest METADATA
    # 115.9 KiB, zero empty directories.
    assert MAX_ENVIRONMENT_FILES == 100_000
    assert MAX_ENVIRONMENT_FILE_BYTES == 512 * 1024 * 1024
    assert MAX_ENVIRONMENT_BYTES == 4 * 1024 * 1024 * 1024
    assert MAX_DISTRIBUTION_METADATA_BYTES == 1024 * 1024


def test_repository_environment_fits_the_protected_bounds() -> None:
    """The running dev environment is a superset of the locked no-dev planner environment."""
    if sys.prefix == sys.base_prefix:
        pytest.skip("the test interpreter is not running from the repository environment")
    files = 0
    total = 0
    largest = 0
    largest_metadata = 0
    for directory, dirnames, names in os.walk(sys.prefix):
        dirnames[:] = [name for name in dirnames if name != "__pycache__"]
        for name in names:
            path = Path(directory) / name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or path.suffix in {".pyc", ".pyo"}:
                continue
            files += 1
            total += info.st_size
            largest = max(largest, info.st_size)
            if name == "METADATA" and directory.endswith(".dist-info"):
                largest_metadata = max(largest_metadata, info.st_size)

    assert 0 < files <= MAX_ENVIRONMENT_FILES
    assert total <= MAX_ENVIRONMENT_BYTES
    assert largest <= MAX_ENVIRONMENT_FILE_BYTES
    assert 0 < largest_metadata <= MAX_DISTRIBUTION_METADATA_BYTES
