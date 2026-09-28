from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from algua.primitives.bounded_subprocess import BoundedCompletion, OutputLimitExceeded
from algua.registry import planner_environment_probe as probe_module
from algua.registry.planner_environment import (
    EnvironmentIncompatible,
    current_interpreter_identity,
    inventory_environment,
    verify_environment,
)
from tests._venv_fixture import SITE_PACKAGES, uv_like_venv


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _valid_probe() -> dict[str, object]:
    return {**current_interpreter_identity().to_dict(), "algua": False}


def _verify(env: Path) -> None:
    verify_environment(env, current_interpreter_identity(), inventory_environment(env).digest)


def test_probe_never_runs_installed_startup_hooks(tmp_path: Path) -> None:
    """Defense in depth: the inventory refuses these hooks, and the probe never runs them."""
    env = uv_like_venv(tmp_path / "env")
    site = env / SITE_PACKAGES
    markers = {name: tmp_path / f"{name}-ran" for name in ("pth", "sitecustomize", "usercustomize")}
    (site / "evil.pth").write_text(
        f"import pathlib; pathlib.Path({str(markers['pth'])!r}).touch()\n")
    for name in ("sitecustomize", "usercustomize"):
        (site / f"{name}.py").write_text(
            f"import pathlib; pathlib.Path({str(markers[name])!r}).touch()\n"
            "print('{\"hijacked\": true}')\n")

    with pytest.raises(EnvironmentIncompatible):
        inventory_environment(env)
    identity, has_algua = probe_module._probe(env)

    assert identity == current_interpreter_identity().to_dict()
    assert has_algua is False
    assert not [name for name, marker in markers.items() if marker.exists()]


def test_probe_leaves_the_uv_startup_shim_unimported_and_writes_no_bytecode(
    tmp_path: Path,
) -> None:
    env = uv_like_venv(tmp_path / "env")
    assert (env / SITE_PACKAGES / "_virtualenv.pth").read_bytes() == b"import _virtualenv"

    _verify(env)
    _verify(env)  # a probe that imported the shim would have left bytecode for the inventory

    assert not list(env.rglob("__pycache__"))


def test_probe_detects_algua_through_the_explicit_environment_import_root(
    tmp_path: Path,
) -> None:
    env = uv_like_venv(tmp_path / "env")
    (env / SITE_PACKAGES / "algua").mkdir()
    (env / SITE_PACKAGES / "algua/__init__.py").write_text("")

    with pytest.raises(EnvironmentIncompatible, match="identity"):
        _verify(env)


def test_probe_runs_isolated_without_site_and_with_bounded_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = uv_like_venv(tmp_path / "env")
    calls: list[tuple[list[str], dict]] = []

    def fake(argv, **kwargs):
        calls.append((list(argv), kwargs))
        return BoundedCompletion(0, _canonical(_valid_probe()), b"")

    monkeypatch.setattr(probe_module, "run_bounded", fake)
    _verify(env)

    [(argv, kwargs)] = calls
    assert argv[:4] == [str(env / "bin/python"), "-I", "-S", "-c"]
    assert argv[5:] == [str(env / SITE_PACKAGES)]
    assert 0 < kwargs["max_stdout"] <= 4096 and 0 < kwargs["max_stderr"] <= 4096
    assert 0 < kwargs["timeout"] <= 60


@pytest.mark.parametrize(
    "failure",
    [
        OutputLimitExceeded("stdout"),
        OutputLimitExceeded("stderr"),
        subprocess.TimeoutExpired(["python"], 30),
        FileNotFoundError(2, "missing interpreter"),
        PermissionError(13, "not executable"),
    ],
)
def test_probe_seam_failures_are_incompatible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: BaseException,
) -> None:
    env = uv_like_venv(tmp_path / "env")

    def fail(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(probe_module, "run_bounded", fail)
    with pytest.raises(EnvironmentIncompatible):
        _verify(env)


def test_probe_nonzero_exit_is_incompatible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = uv_like_venv(tmp_path / "env")
    monkeypatch.setattr(
        probe_module, "run_bounded",
        lambda *_args, **_kwargs: BoundedCompletion(1, _canonical(_valid_probe()), b""),
    )
    with pytest.raises(EnvironmentIncompatible):
        _verify(env)


def _malformed_outputs() -> list[tuple[str, bytes]]:
    valid = _valid_probe()
    text = _canonical(valid)
    extra = {**valid, "extra": "x"}
    missing = {key: value for key, value in valid.items() if key != "soabi"}
    return [
        ("empty", b""),
        ("integer", b"1\n"),
        ("string", b'"x"\n'),
        ("null", b"null\n"),
        ("boolean", b"true\n"),
        ("list", b"[]\n"),
        ("extra-key", _canonical(extra)),
        ("missing-key", _canonical(missing)),
        ("algua-not-bool", _canonical({**valid, "algua": "false"})),
        ("algua-int", _canonical({**valid, "algua": 0})),
        ("identity-not-str", _canonical({**valid, "version": 3})),
        ("non-canonical-spacing", (json.dumps(valid, sort_keys=True) + "\n").encode()),
        ("unsorted", (json.dumps(dict(reversed(list(valid.items()))),
                                 separators=(",", ":")) + "\n").encode()),
        ("no-newline", text.rstrip(b"\n")),
        ("trailing-newline", text + b"\n"),
        ("trailing-object", text + text),
        ("trailing-text", text + b"junk"),
        ("leading-space", b" " + text),
        ("duplicate-key", text.replace(b'{"algua":false', b'{"algua":false,"algua":false')),
        ("not-utf8", b"\xff" + text),
    ]


@pytest.mark.parametrize(
    "output", [raw for _name, raw in _malformed_outputs()],
    ids=[name for name, _raw in _malformed_outputs()],
)
def test_probe_parser_refuses_everything_but_one_canonical_identity_object(output: bytes) -> None:
    with pytest.raises(EnvironmentIncompatible):
        probe_module._parse_probe(output)


def test_probe_parser_returns_identity_and_algua_flag() -> None:
    identity, has_algua = probe_module._parse_probe(_canonical(_valid_probe()))

    assert identity == current_interpreter_identity().to_dict()
    assert has_algua is False


@pytest.mark.parametrize(
    "output", [raw for _name, raw in _malformed_outputs()],
    ids=[name for name, _raw in _malformed_outputs()],
)
def test_probe_requires_exactly_one_canonical_identity_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, output: bytes,
) -> None:
    env = uv_like_venv(tmp_path / "env")
    monkeypatch.setattr(
        probe_module, "run_bounded",
        lambda *_args, **_kwargs: BoundedCompletion(0, output, b""),
    )
    with pytest.raises(EnvironmentIncompatible):
        _verify(env)


def test_canonical_probe_output_is_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = uv_like_venv(tmp_path / "env")
    monkeypatch.setattr(
        probe_module, "run_bounded",
        lambda *_args, **_kwargs: BoundedCompletion(0, _canonical(_valid_probe()), b""),
    )
    _verify(env)
