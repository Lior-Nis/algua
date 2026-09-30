"""Story 1.3c contract §5: the frozen planner child (`algua.live.frozen_child`).

Every end-to-end test launches a real child with the exact §5 argv, environment, cwd and stdin
against a real Story 1.3b bundle layout: a copy of the working tree's `algua/` plus a fixture
strategy family, with `_algua/protocol.json` and `_algua/resolved-config.json` produced by 1.3b's
own writer, at `frozen/bundles/sha256/<xx>/<digest>`. The environment is a venv-like tree at
`frozen/environments/sha256/<xx>/<digest>` — `pyvenv.cfg` plus `bin/python` linked to the base
interpreter, exactly how a relocatable venv is laid out — so the child's real `sys.prefix` carries
the environment digest. Its site-packages is linked to the development environment's, so the
repository's editable install IS on the child's import path: the bundle must still be the only
`algua` source, which the module-origin check and the probe below prove.

The invocation directory is sealed 0555 with 0444 files, as the supervisor will seal it, and every
launch asserts the child left it exactly as it found it. Results are compared as canonical wire
bytes against the in-process planner run on the same inputs.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from algua.live import frozen_child
from algua.live.frozen_child import foreign_algua_modules
from algua.live.frozen_wire import (
    BARS_FILE,
    BOOTSTRAP,
    CHILD_MODULE,
    EXIT_BAD_REQUEST,
    EXIT_OK,
    EXIT_UNSUPPORTED,
    MAX_REQUEST_BYTES,
    REQUEST_FILE,
    WireIdentity,
    encode_request,
)
from algua.live.frozen_wire_arrow import encode_bars
from algua.live.frozen_wire_result import decode_result, encode_result
from algua.live.planner import phase_a, phase_b
from algua.live.planner_contract import (
    Decision,
    EarlyNoDecision,
    EarlyPlannerInput,
    LatePlannerInput,
    PlannerRiskFailure,
    SnapshotRequired,
)
from tests._frozen_harness import (
    CHECKOUT_ROOT,
    ENV_DIGEST,
    FAMILY,
    LEAKED_MODULE,
    LEAKY,
    MANIFEST,
    REQUEST_ID,
    STRATEGY,
    Store,
    frozen_store,
)
from tests._frozen_harness import digest as _digest
from tests._frozen_harness import early_input as _early
from tests._frozen_harness import fixture_bars as _bars
from tests._frozen_harness import in_process as _in_process
from tests._frozen_harness import late_input as _late
from tests._frozen_harness import overlaid as _overlaid

FORBIDDEN_LAYERS = ("registry", "data", "cli", "execution", "operator")


def _wire(
    early: EarlyPlannerInput,
    late: LatePlannerInput | None = None,
    *,
    bundle: str = "good",
    environment: str = ENV_DIGEST,
) -> tuple[bytes, bytes]:
    identity = WireIdentity(
        early.strategy_name, 7, 11, MANIFEST, _digest(bundle), environment
    )
    return encode_request("a" if late is None else "b", early, late, identity)


# --- the store: one environment and lazily built bundles, shared by the module ------------------


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory):
    with frozen_store(tmp_path_factory.mktemp("store")) as built:
        yield built


def _launch(
    store: Store, key: str, tmp_path: Path, request: bytes, bars: bytes, *, code: str = BOOTSTRAP
) -> subprocess.CompletedProcess[bytes]:
    bundle = store.bundle(key)
    invocation = tmp_path / "invocation"
    invocation.mkdir(mode=0o700)
    (invocation / REQUEST_FILE).write_bytes(request)
    (invocation / BARS_FILE).write_bytes(bars)
    for name in (REQUEST_FILE, BARS_FILE):
        (invocation / name).chmod(0o444)
    invocation.chmod(0o555)
    try:
        done = subprocess.run(
            [str(store.python), "-I", "-B", "-c", code, str(bundle), str(invocation)],
            cwd=bundle,
            env={"PATH": str(store.python.parent), "HOME": "/nonexistent", "LANG": "C.UTF-8",
                 "LC_ALL": "C.UTF-8", "TZ": "UTC"},
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=60,
            check=False,
            start_new_session=True,
        )
        # The child writes no file: the sealed invocation directory is exactly as it was.
        assert sorted(os.listdir(invocation)) == sorted([REQUEST_FILE, BARS_FILE])
        assert (invocation / REQUEST_FILE).read_bytes() == request
        assert (invocation / BARS_FILE).read_bytes() == bars
    finally:
        invocation.chmod(0o700)
    return done


# --- wire-v1 entry constants --------------------------------------------------------------------


def test_the_wire_v1_entry_point_is_pinned():
    assert CHILD_MODULE == "algua.live.frozen_child" == frozen_child.__name__
    assert (EXIT_OK, EXIT_BAD_REQUEST, EXIT_UNSUPPORTED) == (0, 2, 3)
    assert BOOTSTRAP == (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from algua.live.frozen_child import main; sys.exit(main(sys.argv[2]))"
    )
    compile(BOOTSTRAP, "<bootstrap>", "exec")


# --- parity with the in-process planner ---------------------------------------------------------


@pytest.mark.parametrize("case", ["snapshot", "no_bars"])
def test_phase_a_equals_the_in_process_planner(store, tmp_path, case):
    strategy = _in_process()
    early = _early(strategy, empty=case == "no_bars")
    expected = phase_a(_overlaid(strategy), early)
    assert isinstance(expected, SnapshotRequired if case == "snapshot" else EarlyNoDecision)

    done = _launch(store, "good", tmp_path, *_wire(early))

    assert done.returncode == EXIT_OK, done.stderr.decode()
    assert done.stdout == encode_result("a", REQUEST_ID, expected)
    decode_result(done.stdout, phase="a", request_id=REQUEST_ID)


@pytest.mark.parametrize("case", ["disabled", "enabled", "drawdown"])
def test_phase_b_equals_the_in_process_planner(store, tmp_path, case):
    strategy = _in_process()
    early = _early(strategy)
    late = _late(strategy, early, case)
    expected = phase_b(_overlaid(strategy), late)
    if case == "drawdown":
        assert isinstance(expected, PlannerRiskFailure) and expected.kind == "drawdown"
    else:
        assert isinstance(expected, Decision) and expected.ordered_intents

    done = _launch(store, "good", tmp_path, *_wire(early, late))

    assert done.returncode == EXIT_OK, done.stderr.decode()
    assert done.stdout == encode_result("b", REQUEST_ID, expected)
    decode_result(done.stdout, phase="b", request_id=REQUEST_ID)


# --- refusals: EXIT_UNSUPPORTED -----------------------------------------------------------------


def _unsupported_request(case: str) -> tuple[str, tuple[bytes, bytes]]:
    strategy = _in_process()
    if case == "bundle_digest":
        return "good", _wire(_early(strategy), bundle="elsewhere")
    if case == "environment_digest":
        return "good", _wire(_early(strategy), environment=_digest("other environment"))
    if case == "resolved_config_file":  # request = the bundle's strategy, not its recorded file
        return "skewed", _wire(_early(strategy), bundle="skewed")
    if case == "strategy_config":  # request = the recorded file, not the bundle's strategy
        return "skewed", _wire(_early(_in_process(lookback=2)), bundle="skewed")
    if case == "unknown_strategy":
        return "good", _wire(_early(strategy, name="no_such_strategy"))
    if case == "module_origin":
        return "leaky", _wire(_early(_in_process(LEAKY)), bundle="leaky")
    return case, _wire(_early(strategy), bundle=case)  # a protocol.json refusal


@pytest.mark.parametrize(
    "case,reason",
    [
        ("bundle_digest", "bundle_digest"),
        ("environment_digest", "environment_digest"),
        ("resolved_config_file", "resolved-config.json"),
        ("strategy_config", "does not dump to the recorded config"),
        ("unknown_strategy", "no_such_strategy"),
        ("no_protocol", "protocol.json"),
        ("wire_v2", "protocol.json"),
        ("boundary_v2", "protocol.json"),
        ("module_origin", LEAKED_MODULE),
    ],
)
def test_the_child_refuses_what_is_not_its_bundle(store, tmp_path, case, reason):
    key, (request, bars) = _unsupported_request(case)

    done = _launch(store, key, tmp_path, request, bars)

    assert done.returncode == EXIT_UNSUPPORTED, done.stderr.decode()
    assert done.stdout == b""
    assert reason in done.stderr.decode()


def test_main_refuses_unless_its_package_is_the_one_bootstrap_put_first(monkeypatch, capsys,
                                                                        tmp_path):
    monkeypatch.setattr(sys, "path", [str(tmp_path), *sys.path])

    assert frozen_child.main(str(tmp_path)) == EXIT_UNSUPPORTED
    assert "sys.path[0]" in capsys.readouterr().err


# --- refusals: EXIT_BAD_REQUEST -----------------------------------------------------------------


@pytest.mark.parametrize("case", ["not_canonical", "oversize", "bars_digest"])
def test_a_request_the_codec_refuses_is_a_bad_request(store, tmp_path, case):
    request, bars = _wire(_early(_in_process()))
    if case == "not_canonical":
        request += b" "
    elif case == "oversize":
        request = b"{" + b" " * MAX_REQUEST_BYTES
    else:
        bars = encode_bars(_bars(bump=1.0))

    done = _launch(store, "good", tmp_path, request, bars)

    assert done.returncode == EXIT_BAD_REQUEST, done.stderr.decode()
    assert done.stdout == b""


# --- module origin ------------------------------------------------------------------------------


def _module(file: str) -> SimpleNamespace:
    return SimpleNamespace(__file__=file)


def test_foreign_algua_modules_names_every_algua_module_outside_the_bundle(tmp_path):
    root = tmp_path / "bundle"
    modules: dict[str, object] = {
        "algua": _module(str(root / "algua/__init__.py")),
        "algua.live.planner": _module(str(root / "algua/live/planner.py")),
        "algua.leak": _module(str(tmp_path / "checkout/algua/leak.py")),
        "algua.sibling": _module(str(tmp_path / "bundle-other/algua/sibling.py")),
        "algua.dotdot": _module(str(root / "algua/../../checkout/algua/dotdot.py")),
        "algua.namespace": ModuleType("algua.namespace"),  # no __file__
        "algua.blocked": None,
        "pandas": _module("/elsewhere/pandas/__init__.py"),
        "builtins": ModuleType("builtins"),
        "alguax": _module("/elsewhere/alguax.py"),
    }

    assert foreign_algua_modules(modules, str(root)) == [
        "algua.blocked", "algua.dotdot", "algua.leak", "algua.namespace", "algua.sibling",
    ]


def test_foreign_algua_modules_compares_resolved_paths(tmp_path):
    root = tmp_path / "bundle"
    (root / "algua").mkdir(parents=True)
    (root / "algua/inside.py").write_text("")
    outside = tmp_path / "outside.py"
    outside.write_text("")
    (root / "algua/link.py").symlink_to(outside)
    alias = tmp_path / "alias"
    alias.symlink_to(root)
    modules = {
        "algua.link": _module(str(root / "algua/link.py")),
        "algua.inside": _module(str(alias / "algua/inside.py")),
    }

    assert foreign_algua_modules(modules, str(root)) == ["algua.link"]


PROBE = (
    "import json, sys\n"
    "sys.path.insert(0, sys.argv[1])\n"
    "from algua.live.frozen_child import main\n"
    "code = main(sys.argv[2])\n"
    "import algua\n"
    "loaded = sorted(n for n in sys.modules if n == 'algua' or n.startswith('algua.'))\n"
    "sys.stderr.write('\\nPROBE ' + json.dumps({'code': code, 'algua': algua.__file__,\n"
    "    'path': sys.path, 'prefix': sys.prefix, 'loaded': loaded}))\n"
)


def test_the_bundle_is_the_only_algua_source_despite_the_editable_install(store, tmp_path):
    strategy = _in_process()
    early = _early(strategy)
    late = _late(strategy, early, "enabled")

    done = _launch(store, "good", tmp_path, *_wire(early, late), code=PROBE)

    probe = json.loads(done.stderr.decode().rpartition("\nPROBE ")[2])
    bundle = store.bundle("good")
    assert probe["code"] == EXIT_OK and done.returncode == 0
    assert probe["algua"] == str(bundle / "algua/__init__.py")
    assert probe["path"][0] == str(bundle)
    assert str(CHECKOUT_ROOT) in probe["path"]  # the editable install is live in the child
    assert Path(probe["prefix"]).name == ENV_DIGEST
    assert f"algua.strategies.{FAMILY}.{STRATEGY}" in probe["loaded"]
    assert "algua.live.planner_late" in probe["loaded"]
    layers = {name.split(".")[1] for name in probe["loaded"] if "." in name}
    assert not layers & set(FORBIDDEN_LAYERS), sorted(layers & set(FORBIDDEN_LAYERS))
