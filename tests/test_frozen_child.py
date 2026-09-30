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

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pandas as pd
import pytest

import algua
from algua.contracts.canonical import canonical_json
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
    BOUNDARY_VERSION,
    CapturedStrategyState,
    Decision,
    EarlyNoDecision,
    EarlyPlannerInput,
    LatePlannerInput,
    PlannerRiskFailure,
    SnapshotRequired,
    VenueBeliefDisabled,
    VenueBeliefEnabled,
)
from algua.portfolio.construction import get_construction_policy
from algua.portfolio.overlays import resolve_overlays
from algua.registry.artifact_preparation import _bundle_files
from algua.strategies.base import LoadedStrategy, config_hash

CHECKOUT_ALGUA = Path(algua.__file__).parent
CHECKOUT_ROOT = CHECKOUT_ALGUA.parent
FAMILY = "frozen_child_family"
STRATEGY = "frozen_child_fixture"
LEAKY = "frozen_child_leaky"
LEAKED_MODULE = "algua.contracts.leaked_from_checkout"
GATE = ("AAA", "BBB")  # the gate universe; the strategy's own CONFIG universe is wider
NOW = datetime(2023, 1, 6, tzinfo=UTC)
REQUEST_ID = "0123456789abcdef0123456789abcdef"
MANIFEST = "d" * 64
FORBIDDEN_LAYERS = ("registry", "data", "cli", "execution", "operator")


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


ENV_DIGEST = _digest("environment")


# --- the fixture strategy -----------------------------------------------------------------------


def _source(name: str, *, lookback: int = 1, leak: bool = False) -> str:
    source = (
        '"""Frozen-child fixture: each symbol\'s last close is its score; top-1 equal weight."""\n'
        "from __future__ import annotations\n"
        "from typing import Any\n"
        "import pandas as pd\n"
        "from algua.contracts.types import ExecutionContract\n"
        "from algua.strategies.base import StrategyConfig\n"
        f"CONFIG = StrategyConfig(name={name!r}, universe=['AAA', 'BBB', 'CCC'],\n"
        "    execution=ExecutionContract(rebalance_frequency='1d', decision_lag_bars=1),\n"
        f"    params={{'lookback': {lookback}}}, construction='top_k_equal_weight',\n"
        "    construction_params={'top_k': 1}, feature_lookback=1)\n"
        "def signal(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:\n"
        "    return view.groupby('symbol')['close'].last().astype('float64')\n"
    )
    if leak:  # load one algua module from the mutable checkout, as a leaking import path would
        source += (
            "import importlib.util, sys\n"
            f"_spec = importlib.util.spec_from_file_location({LEAKED_MODULE!r},\n"
            f"    {str(CHECKOUT_ALGUA / 'contracts' / 'planner.py')!r})\n"
            "_leaked = importlib.util.module_from_spec(_spec)\n"
            "sys.modules[_spec.name] = _leaked\n"
            "_spec.loader.exec_module(_leaked)\n"
        )
    return source


def _in_process(name: str = STRATEGY, *, lookback: int = 1) -> LoadedStrategy:
    """The same strategy source, loaded in this process the way `load_strategy` binds it."""
    namespace: dict[str, Any] = {}
    exec(compile(_source(name, lookback=lookback), f"<{name}>", "exec"), namespace)
    config = namespace["CONFIG"]
    return LoadedStrategy(
        config=config,
        signal_fn=namespace["signal"],
        construct_fn=get_construction_policy(config.construction),
        overlay_fns=resolve_overlays(config.overlays, feature_lookback=config.feature_lookback),
    )


def _recorded(strategy: LoadedStrategy) -> dict[str, Any]:
    """The recorded config exactly as Story 1.3b captures it."""
    return json.loads(canonical_json(strategy.config.model_dump(mode="json")))


def _overlaid(strategy: LoadedStrategy) -> LoadedStrategy:
    return replace(strategy, config=strategy.config.model_copy(update={"universe": list(GATE)}))


# --- planner inputs -----------------------------------------------------------------------------


def _bars(*, empty: bool = False, bump: float = 0.0) -> pd.DataFrame:
    closes = {"AAA": 10.0, "BBB": 12.0, "CCC": 20.0, "OLD": 5.0}
    rows = [
        {"timestamp": datetime(2023, 1, day, tzinfo=UTC), "symbol": symbol, "open": close,
         "high": close, "low": close, "close": close + bump, "adj_close": close,
         "volume": 100.0}
        for day in (3, 4, 5)
        for symbol, close in closes.items()
    ]
    frame = pd.DataFrame(rows).set_index("timestamp")
    return frame.iloc[0:0] if empty else frame


def _early(
    strategy: LoadedStrategy, *, name: str | None = None, empty: bool = False
) -> EarlyPlannerInput:
    return EarlyPlannerInput(
        boundary_version=BOUNDARY_VERSION,
        request_id=REQUEST_ID,
        strategy_name=strategy.name if name is None else name,
        deployment_id=7,
        artifact_id=11,
        manifest_digest=MANIFEST,
        config_hash=config_hash(strategy),
        resolved_config_json=json.dumps(
            _recorded(strategy), sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ),
        now=NOW,
        timeframe="1d",
        calendar_code="XNYS",
        raw_bars=_bars(empty=empty),
        early_positions={} if empty else {"OLD": 2.0},
        gate_universe=GATE,
        max_drawdown=0.1,
    )


def _late(strategy: LoadedStrategy, early: EarlyPlannerInput, case: str) -> LatePlannerInput:
    first = phase_a(_overlaid(strategy), early)
    assert isinstance(first, SnapshotRequired)
    captured = CapturedStrategyState(
        request_id=REQUEST_ID,
        sizing_equity=100.0,
        drawdown_equity=100.0,
        quantities={"OLD": 2.0},
        market_values={"OLD": 10.0},
        persisted_peak_equity=200.0 if case == "drawdown" else 100.0,
        venue_belief=(
            VenueBeliefEnabled({"OLD": 2.0}) if case == "enabled" else VenueBeliefDisabled()
        ),
    )
    return LatePlannerInput(early, first.phase_a_binding, captured)


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


PROTOCOL = "_algua/protocol.json"


def _generated(resolved: dict[str, Any]) -> dict[str, bytes]:
    """`_algua/` exactly as Story 1.3b's preparation writes it."""
    return {item.path: item.data for item in _bundle_files((), resolved)}


STAMP = _generated({})[PROTOCOL]  # 1.3b's protocol.json bytes (independent of the config)


def _protocol(**changes: Any) -> bytes:
    protocol = json.loads(STAMP)
    protocol.update(changes)
    return canonical_json(protocol).encode("utf-8")


# bundle key -> (strategy modules, recorded config, protocol.json bytes; None omits the file)
BUNDLES: dict[str, tuple[dict[str, str], dict[str, Any], bytes | None]] = {
    "good": ({STRATEGY: _source(STRATEGY)}, _recorded(_in_process()), STAMP),
    "skewed": ({STRATEGY: _source(STRATEGY)}, _recorded(_in_process(lookback=2)), STAMP),
    "leaky": ({LEAKY: _source(LEAKY, leak=True)}, _recorded(_in_process(LEAKY)), STAMP),
    "no_protocol": ({STRATEGY: _source(STRATEGY)}, _recorded(_in_process()), None),
    "wire_v2": (
        {STRATEGY: _source(STRATEGY)}, _recorded(_in_process()),
        _protocol(frozen_wire={"name": "frozen-planner", "version": 2}),
    ),
    "boundary_v2": (
        {STRATEGY: _source(STRATEGY)}, _recorded(_in_process()),
        _protocol(planner_boundary_version=2),
    ),
}


def _seal(root: Path) -> None:
    for directory, _, files in os.walk(root, topdown=False):
        for name in files:
            (Path(directory) / name).chmod(0o444)
        Path(directory).chmod(0o555)


def _unseal(root: Path) -> None:
    for directory, _, _ in os.walk(root, topdown=True):
        Path(directory).chmod(0o755)


@dataclass
class Store:
    root: Path
    python: Path
    built: dict[str, Path] = field(default_factory=dict)

    def bundle(self, key: str) -> Path:
        if key not in self.built:
            self.built[key] = self._build(key)
        return self.built[key]

    def _build(self, key: str) -> Path:
        strategies, resolved, protocol = BUNDLES[key]
        digest = _digest(key)
        bundle = self.root / "frozen/bundles/sha256" / digest[:2] / digest
        shutil.copytree(CHECKOUT_ALGUA, bundle / "algua",
                        ignore=shutil.ignore_patterns("__pycache__"))
        family = bundle / "algua/strategies" / FAMILY
        family.mkdir()
        (family / "__init__.py").write_text("")
        for name, source in strategies.items():
            (family / f"{name}.py").write_text(source)
        files = {path: data for path, data in _generated(resolved).items() if path != PROTOCOL}
        if protocol is not None:
            files[PROTOCOL] = protocol
        (bundle / "_algua").mkdir()
        for path, data in files.items():
            (bundle / path).write_bytes(data)
        _seal(bundle)
        return bundle


def _environment(root: Path) -> Path:
    """A venv-like environment at its digest locator; returns its `bin/python`."""
    env = root / "frozen/environments/sha256" / ENV_DIGEST[:2] / ENV_DIGEST
    base = Path(os.path.realpath(sys.executable))
    (env / "bin").mkdir(parents=True)
    (env / "bin/python").symlink_to(base)
    lib = env / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}"
    lib.mkdir(parents=True)
    (lib / "site-packages").symlink_to(sysconfig.get_path("purelib"))
    (env / "pyvenv.cfg").write_text(
        f"home = {base.parent}\ninclude-system-site-packages = false\n"
        f"version = {platform.python_version()}\n"
    )
    return env / "bin/python"


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("store")
    built = Store(root, _environment(root))
    yield built
    for bundle in built.built.values():
        _unseal(bundle)


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
