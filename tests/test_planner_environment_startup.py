"""Site-startup policy: the `-I -S` probe may certify an environment only if ordinary startup
would expose nothing beyond the environment's own site-packages.

The pinned startup files are uv 0.9.26's verbatim `uv venv` output (see `tests/_venv_fixture.py`).
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from algua.registry import planner_environment_inventory as inventory_module
from algua.registry.planner_environment import (
    EnvironmentIncompatible,
    inventory_environment,
)
from tests._venv_fixture import SITE_PACKAGES, UV_PTH, UV_SHIM, uv_like_venv

PINNED_PTH = "69ac3d8f27e679c81b94ab30b3b56e9cd138219b1ba94a1fa3606d5a76a1433d"
PINNED_SHIM = "6cf30c56faf2a55228914dbbd17f8088ed371ebb08f5e7fa6fd931f913fcaf1d"


def _outside_algua(tmp_path: Path) -> Path:
    """A checkout-like tree holding an importable `algua` outside the environment."""
    checkout = tmp_path / "checkout"
    (checkout / "algua").mkdir(parents=True)
    (checkout / "algua/__init__.py").write_text("")
    return checkout


def _ordinary_startup_finds_algua(env: Path) -> bool:
    result = subprocess.run(
        [str(env / "bin/python"), "-I", "-B", "-c",
         "import importlib.util; print(importlib.util.find_spec('algua') is not None)"],
        capture_output=True, text=True, timeout=30, check=True,
    )
    return result.stdout.strip() == "True"


def test_uv_startup_files_are_accepted_byte_for_byte(tmp_path: Path) -> None:
    env = uv_like_venv(tmp_path / "env")

    inventory = inventory_environment(env)

    digests = {item.path: item.sha256 for item in inventory.files}
    assert digests[f"{SITE_PACKAGES}/_virtualenv.pth"] == PINNED_PTH
    assert digests[f"{SITE_PACKAGES}/_virtualenv.py"] == PINNED_SHIM
    assert hashlib.sha256(UV_PTH).hexdigest() == PINNED_PTH
    assert hashlib.sha256(UV_SHIM).hexdigest() == PINNED_SHIM


def test_a_pth_path_entry_exposing_external_algua_is_refused(tmp_path: Path) -> None:
    env = uv_like_venv(tmp_path / "env")
    (env / SITE_PACKAGES / "checkout.pth").write_text(f"{_outside_algua(tmp_path)}\n")
    # The isolated probe alone would certify this environment: ordinary startup finds `algua`
    # through the path entry while `-I -S` never processes it.
    assert _ordinary_startup_finds_algua(env) is True

    with pytest.raises(EnvironmentIncompatible, match=r"\.pth"):
        inventory_environment(env)


@pytest.mark.parametrize(
    "name,content",
    [
        ("hook.pth", "import sys; sys.path.insert(0, {checkout!r})\n"),
        ("editable.pth", "import _editable_finder\n"),
        ("__editable__.algua-0.pth", "{checkout}\n"),
        ("distutils-precedence.pth", "import os; os.environ.setdefault('X', '1')\n"),
        ("comment-only.pth", "# nothing\n"),
        ("empty.pth", ""),
        ("nested/inert.pth", "{checkout}\n"),
    ],
)
def test_every_non_normative_pth_file_is_refused(
    tmp_path: Path, name: str, content: str,
) -> None:
    env = uv_like_venv(tmp_path / "env")
    target = env / SITE_PACKAGES / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content.format(checkout=str(_outside_algua(tmp_path))))

    with pytest.raises(EnvironmentIncompatible, match=r"\.pth"):
        inventory_environment(env)


@pytest.mark.parametrize(
    "content",
    [b"import _virtualenv\n", b"import _virtualenv; import os",
     b"import _virtualenv\n/tmp/checkout", b"/tmp/checkout", b""],
)
def test_the_uv_pth_must_match_its_pinned_bytes(tmp_path: Path, content: bytes) -> None:
    env = uv_like_venv(tmp_path / "env")
    (env / SITE_PACKAGES / "_virtualenv.pth").write_bytes(content)

    with pytest.raises(EnvironmentIncompatible, match="pinned"):
        inventory_environment(env)


def test_the_uv_shim_must_match_its_pinned_bytes(tmp_path: Path) -> None:
    env = uv_like_venv(tmp_path / "env")
    shim = env / SITE_PACKAGES / "_virtualenv.py"
    shim.write_bytes(UV_SHIM + b"\nimport sys; sys.path.append('/tmp/checkout')\n")

    with pytest.raises(EnvironmentIncompatible, match="pinned"):
        inventory_environment(env)


@pytest.mark.parametrize(
    "relative",
    [
        "sitecustomize.py", "usercustomize.py", "sitecustomize/__init__.py",
        "usercustomize/__init__.py", "sitecustomize.cpython-312-x86_64-linux-gnu.so",
        "sitecustomize/nested/module.py", "_virtualenv/__init__.py",
        "_virtualenv.cpython-312-x86_64-linux-gnu.so", "_virtualenv.abi3.so",
    ],
)
def test_executable_startup_hooks_and_shim_shadows_are_refused(
    tmp_path: Path, relative: str,
) -> None:
    env = uv_like_venv(tmp_path / "env")
    target = env / SITE_PACKAGES / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("import sys; sys.path.append('/tmp/checkout')\n")

    with pytest.raises(EnvironmentIncompatible, match="startup"):
        inventory_environment(env)


@pytest.mark.parametrize(
    "relative",
    ["sitecustomize-1.0.dist-info/METADATA", "mypkg/sitecustomize.py", "_virtualenvx.py",
     "not_a_hook.pth.txt"],
)
def test_names_that_are_not_startup_hooks_are_accepted(tmp_path: Path, relative: str) -> None:
    env = uv_like_venv(tmp_path / "env")
    target = env / SITE_PACKAGES / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("Name: sitecustomize\nVersion: 1.0\n" if relative.endswith("METADATA")
                      else "VALUE = 1\n")

    inventory_environment(env)


def test_startup_files_are_refused_before_their_content_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = uv_like_venv(tmp_path / "env")
    evil = env / SITE_PACKAGES / "evil.pth"
    evil.write_text("/tmp/checkout\n")
    hashed: list[Path] = []
    real = inventory_module._file_digest

    def recording(path: Path, limit: int):
        hashed.append(path)
        return real(path, limit)

    monkeypatch.setattr(inventory_module, "_file_digest", recording)
    with pytest.raises(EnvironmentIncompatible):
        inventory_environment(env)
    assert evil not in hashed


def _pyvenv(env: Path, text: str) -> None:
    (env / "pyvenv.cfg").write_text(text)


@pytest.mark.parametrize(
    "text",
    [
        "home = /usr/bin\ninclude-system-site-packages = true\n",
        "home = /usr/bin\ninclude-system-site-packages = TRUE\n",
        "home = /usr/bin\n",
        "home = /usr/bin\ninclude-system-site-packages = false\n"
        "include-system-site-packages = true\n",
        "home = /usr/bin\ninclude-system-site-packages = no\n",
        "home = /usr/bin\nInclude-System-Site-Packages=true\n",
    ],
    ids=["true", "true-uppercase", "absent-means-true", "last-wins", "not-false",
         "case-and-spacing"],
)
def test_system_site_packages_must_be_disabled_as_site_reads_it(
    tmp_path: Path, text: str,
) -> None:
    env = uv_like_venv(tmp_path / "env")
    _pyvenv(env, text)

    with pytest.raises(EnvironmentIncompatible, match="pyvenv"):
        inventory_environment(env)


@pytest.mark.parametrize(
    "text",
    [
        "home = /usr/bin\nimplementation = CPython\nuv = 0.9.26\nversion_info = 3.12.3\n"
        "include-system-site-packages = false\nrelocatable = true\n",
        "home = /usr/bin\n  INCLUDE-SYSTEM-SITE-PACKAGES =  False  \n",
    ],
    ids=["uv-0.9.26-verbatim", "case-and-spacing"],
)
def test_disabled_system_site_packages_are_accepted(tmp_path: Path, text: str) -> None:
    env = uv_like_venv(tmp_path / "env")
    _pyvenv(env, text)

    inventory_environment(env)


def test_pyvenv_cfg_is_required_at_the_root_and_nowhere_else(tmp_path: Path) -> None:
    env = uv_like_venv(tmp_path / "env")
    shutil.copy2(env / "pyvenv.cfg", env / "bin/pyvenv.cfg")
    with pytest.raises(EnvironmentIncompatible, match="pyvenv"):
        inventory_environment(env)

    (env / "bin/pyvenv.cfg").unlink()
    (env / "pyvenv.cfg").unlink()
    with pytest.raises(EnvironmentIncompatible, match="pyvenv"):
        inventory_environment(env)


def test_oversized_pyvenv_cfg_is_refused(tmp_path: Path) -> None:
    env = uv_like_venv(tmp_path / "env")
    _pyvenv(env, "include-system-site-packages = false\n" + "# pad\n" * 20000)

    with pytest.raises(EnvironmentIncompatible, match="pyvenv"):
        inventory_environment(env)


def test_pinned_startup_files_match_the_installed_uv(tmp_path: Path) -> None:
    """When uv 0.9.26 is on PATH, its real `uv venv` output is exactly what is pinned."""
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is not installed")
    version = subprocess.run([uv, "--version"], capture_output=True, text=True, timeout=30)
    if version.stdout.strip() != "uv 0.9.26":
        pytest.skip(f"pinned startup bytes are for uv 0.9.26, found {version.stdout.strip()!r}")
    env = tmp_path / "env"
    subprocess.run(
        [uv, "venv", "--relocatable", "--python", sys.executable, "--no-python-downloads",
         "--quiet", str(env)],
        check=True, capture_output=True, timeout=120, env={"PATH": str(Path(uv).parent),
                                                           "HOME": str(tmp_path)},
    )
    (env / "lib64").unlink()
    (env / SITE_PACKAGES / "six.py").write_text("VERSION = '1.17.0'\n")

    assert (env / SITE_PACKAGES / "_virtualenv.pth").read_bytes() == UV_PTH
    assert (env / SITE_PACKAGES / "_virtualenv.py").read_bytes() == UV_SHIM
    inventory_environment(env)
