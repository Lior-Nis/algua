"""Pinned site-startup policy for frozen planner environments.

The isolated `-I -S` probe checks `find_spec('algua')` against the environment's own
site-packages only. That certification is meaningful only if ordinary site startup would expose
nothing more, so an environment is accepted only when:

* the one `.pth` file anywhere is uv 0.9.26's `_virtualenv.pth`, byte-for-byte
  `import _virtualenv` (no path entry; its single import line loads the pinned shim);
* `_virtualenv` resolves only to uv 0.9.26's pinned `_virtualenv.py`, which merely patches
  distutils/setuptools install configuration, and nothing in site-packages can shadow it;
* no `sitecustomize` or `usercustomize` module or package is importable from site-packages;
* `pyvenv.cfg` exists only at the environment root and disables system site-packages exactly
  as `site` reads it (an absent key means enabled; the last occurrence wins), so neither the
  system site directory nor the user site is added.

A different uv version changes these bytes; it then fails closed until the pins are reviewed.
"""
from __future__ import annotations

import sys

from algua.registry.planner_environment_errors import EnvironmentIncompatible

SITE_PACKAGES = f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
PYVENV_CFG = "pyvenv.cfg"
MAX_PYVENV_CFG_BYTES = 64 * 1024
_PINNED_STARTUP_FILES = {
    f"{SITE_PACKAGES}/_virtualenv.pth":
        "69ac3d8f27e679c81b94ab30b3b56e9cd138219b1ba94a1fa3606d5a76a1433d",
    f"{SITE_PACKAGES}/_virtualenv.py":
        "6cf30c56faf2a55228914dbbd17f8088ed371ebb08f5e7fa6fd931f913fcaf1d",
}
_STARTUP_MODULES = frozenset({"sitecustomize", "usercustomize", "_virtualenv"})


def startup_pin(relative: str) -> str | None:
    """Refuse a file ordinary startup would process, by name alone and before it is read.

    Returns the pinned digest the file's content must then match, or None for ordinary files.
    """
    pinned = _PINNED_STARTUP_FILES.get(relative)
    if pinned is not None:
        return pinned
    if relative.endswith(".pth"):
        raise EnvironmentIncompatible("environment contains a non-normative .pth startup file")
    if relative != PYVENV_CFG and relative.rsplit("/", 1)[-1] == PYVENV_CFG:
        raise EnvironmentIncompatible("environment has a pyvenv.cfg outside its root")
    prefix = f"{SITE_PACKAGES}/"
    if relative.startswith(prefix):
        top = relative[len(prefix):].split("/", 1)[0].split(".", 1)[0]
        if top in _STARTUP_MODULES:
            raise EnvironmentIncompatible("environment contains an executable startup hook")
    return None


def require_pinned(relative: str, pinned: str | None, digest: str) -> None:
    if pinned is not None and digest != pinned:
        raise EnvironmentIncompatible(f"{relative} is not the pinned normative uv startup file")


def require_isolated_site(pyvenv_cfg: str) -> None:
    """Require that `site` would leave system site-packages (and the user site) disabled."""
    system_site = "true"
    for line in pyvenv_cfg.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            if key.strip().lower() == "include-system-site-packages":
                system_site = value.strip().lower()
    if system_site != "false":
        raise EnvironmentIncompatible("pyvenv.cfg does not disable system site-packages")
