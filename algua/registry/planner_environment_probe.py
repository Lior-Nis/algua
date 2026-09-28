"""Isolated interpreter verification of frozen planner environments.

Carved out of the environment inventory: the inventory decides what the environment contains,
this module runs its interpreter isolated and checks the keyed identity it reports.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from algua.primitives.bounded_subprocess import OutputLimitExceeded, run_bounded
from algua.registry.artifact_contract import canonical_json
from algua.registry.environment_contract import InterpreterIdentity
from algua.registry.planner_environment_errors import EnvironmentIncompatible
from algua.registry.planner_environment_inventory import (
    inventory_environment,
    scrubbed_environment,
)
from algua.registry.planner_environment_startup import SITE_PACKAGES

_PROBE_TIMEOUT_SECONDS = 30
_PROBE_OUTPUT_BYTES = 4096
_PROBE_IDENTITY_FIELDS = frozenset(
    {"implementation", "version", "cache_tag", "soabi", "platform_tag", "os_name", "machine"})
# `-I -S`: no site module, so no `.pth` line, sitecustomize or usercustomize executes and no
# bytecode is imported from the environment. Only the environment's direct import root (argv) is
# added, so `find_spec('algua')` sees installed top-level packages without processing any `.pth`.
_PROBE = (
    "import importlib.util,json,platform,sys,sysconfig\n"
    "sys.path.extend(sys.argv[1:])\n"
    "print(json.dumps({'implementation':platform.python_implementation(),"
    "'version':platform.python_version(),'cache_tag':sys.implementation.cache_tag or 'unknown',"
    "'soabi':sysconfig.get_config_var('SOABI') or 'unknown',"
    "'platform_tag':sysconfig.get_platform(),'os_name':platform.system().lower(),"
    "'machine':platform.machine().lower(),"
    "'algua':importlib.util.find_spec('algua') is not None},"
    "sort_keys=True,separators=(',',':'),ensure_ascii=False))\n"
)


def _parse_probe(raw: bytes) -> tuple[dict[str, str], bool]:
    """Accept exactly one canonical identity object followed by one newline, nothing else.

    Canonical equality also refuses duplicate keys, whitespace and trailing output: none of them
    can round-trip to the canonical text of the decoded object.
    """
    try:
        text = raw.decode("utf-8")
        value: Any = json.loads(text)
        canonical = canonical_json(value) + "\n" if isinstance(value, dict) else None
    except ValueError as exc:
        raise EnvironmentIncompatible("environment interpreter probe output is malformed") from exc
    if (
        not isinstance(value, dict) or set(value) != {*_PROBE_IDENTITY_FIELDS, "algua"}
        or type(value["algua"]) is not bool
        or any(type(value[field]) is not str for field in _PROBE_IDENTITY_FIELDS)
        or text != canonical
    ):
        raise EnvironmentIncompatible(
            "environment interpreter probe output is not one canonical identity object")
    has_algua = value.pop("algua")
    return value, has_algua


def _probe(root: Path) -> tuple[dict[str, str], bool]:
    """Run the environment's interpreter isolated (`-I -S`) and return its identity facts."""
    python = root / "bin/python"
    import_root = root / SITE_PACKAGES
    try:
        result = run_bounded(
            [str(python), "-I", "-S", "-c", _PROBE, str(import_root)], cwd=root,
            env=scrubbed_environment(python.parent), timeout=_PROBE_TIMEOUT_SECONDS,
            max_stdout=_PROBE_OUTPUT_BYTES, max_stderr=_PROBE_OUTPUT_BYTES,
        )
    except (OSError, subprocess.SubprocessError, OutputLimitExceeded) as exc:
        raise EnvironmentIncompatible(
            "published environment interpreter verification failed") from exc
    if result.returncode != 0:
        raise EnvironmentIncompatible("published environment interpreter verification failed")
    return _parse_probe(result.stdout)


def verify_environment(
    root: Path, expected_interpreter: InterpreterIdentity, expected_inventory_digest: str,
) -> None:
    inventory = inventory_environment(root)
    if inventory.digest != expected_inventory_digest:
        raise EnvironmentIncompatible("installed environment inventory drifted")
    observed, has_algua = _probe(root)
    if has_algua or observed != expected_interpreter.to_dict():
        raise EnvironmentIncompatible("published environment interpreter identity drifted")
