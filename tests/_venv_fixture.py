"""A standard-library venv shaped like the relocatable environment `uv venv` creates.

`uv venv` (0.9.26) writes `_virtualenv.pth` and `_virtualenv.py` into site-packages, creates the
`lib64 -> lib` link that provisioning removes, and leaves no empty directory. The standard-library
builder instead leaves an empty `include/` tree and an empty site-packages, which the complete
inventory correctly refuses, so fixtures normalize to the uv shape here.
"""
from __future__ import annotations

import shutil
import venv
from pathlib import Path

SITE_PACKAGES = "lib/python3.12/site-packages"


def uv_like_venv(path: Path) -> Path:
    venv.EnvBuilder(with_pip=False, symlinks=True).create(path)
    lib64 = path / "lib64"
    if lib64.is_symlink():
        lib64.unlink()
    include = path / "include"
    if include.is_dir() and not any(item.is_file() for item in include.rglob("*")):
        shutil.rmtree(include)
    site = path / SITE_PACKAGES
    (site / "_virtualenv.py").write_text('"""uv virtualenv startup shim fixture."""\n')
    (site / "_virtualenv.pth").write_text("import _virtualenv\n")
    return path
