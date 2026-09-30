"""Shared real-bundle harness for the frozen planner tests (Story 1.3c contract §10).

A Story 1.3b bundle layout built from a copy of the working tree's `algua/` plus a fixture strategy
family, with `_algua/protocol.json` and `_algua/resolved-config.json` produced by 1.3b's own writer,
at `frozen/bundles/sha256/<xx>/<digest>`. The environment is a venv-like tree at
`frozen/environments/sha256/<xx>/<digest>` — `pyvenv.cfg` plus `bin/python` linked to the base
interpreter, exactly how a relocatable venv is laid out — so the child's real `sys.prefix` carries
the environment digest. Its site-packages is linked to the development environment's, so the
repository's editable install IS on the child's import path: the bundle must still be the only
`algua` source, which the child's module-origin check enforces.

Also the fixture strategy's in-process twin and the planner inputs every parity test shares.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
import sysconfig
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

import algua
from algua.contracts.canonical import canonical_json
from algua.live.planner import phase_a
from algua.live.planner_contract import (
    BOUNDARY_VERSION,
    CapturedStrategyState,
    EarlyPlannerInput,
    LatePlannerInput,
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
DEPLOYMENT_ID = 7
ARTIFACT_ID = 11
CALENDAR = "XNYS"


def digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


ENV_DIGEST = digest("environment")


# --- the fixture strategy -----------------------------------------------------------------------


def strategy_source(name: str, *, lookback: int = 1, leak: bool = False) -> str:
    text = (
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
        text += (
            "import importlib.util, sys\n"
            f"_spec = importlib.util.spec_from_file_location({LEAKED_MODULE!r},\n"
            f"    {str(CHECKOUT_ALGUA / 'contracts' / 'planner.py')!r})\n"
            "_leaked = importlib.util.module_from_spec(_spec)\n"
            "sys.modules[_spec.name] = _leaked\n"
            "_spec.loader.exec_module(_leaked)\n"
        )
    return text


def in_process(name: str = STRATEGY, *, lookback: int = 1) -> LoadedStrategy:
    """The same strategy source, loaded in this process the way `load_strategy` binds it."""
    namespace: dict[str, Any] = {}
    exec(compile(strategy_source(name, lookback=lookback), f"<{name}>", "exec"), namespace)
    config = namespace["CONFIG"]
    return LoadedStrategy(
        config=config,
        signal_fn=namespace["signal"],
        construct_fn=get_construction_policy(config.construction),
        overlay_fns=resolve_overlays(config.overlays, feature_lookback=config.feature_lookback),
    )


def recorded(strategy: LoadedStrategy) -> dict[str, Any]:
    """The recorded config exactly as Story 1.3b captures it."""
    return json.loads(canonical_json(strategy.config.model_dump(mode="json")))


def recorded_json(strategy: LoadedStrategy) -> str:
    """The recorded config in the planner's canonical form (`resolved_config_json`)."""
    return json.dumps(
        recorded(strategy), sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    )


def overlaid(strategy: LoadedStrategy) -> LoadedStrategy:
    return replace(strategy, config=strategy.config.model_copy(update={"universe": list(GATE)}))


# --- planner inputs -----------------------------------------------------------------------------


def fixture_bars(*, empty: bool = False, bump: float = 0.0) -> pd.DataFrame:
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


def early_input(
    strategy: LoadedStrategy, *, name: str | None = None, empty: bool = False
) -> EarlyPlannerInput:
    return EarlyPlannerInput(
        boundary_version=BOUNDARY_VERSION,
        request_id=REQUEST_ID,
        strategy_name=strategy.name if name is None else name,
        deployment_id=DEPLOYMENT_ID,
        artifact_id=ARTIFACT_ID,
        manifest_digest=MANIFEST,
        config_hash=config_hash(strategy),
        resolved_config_json=recorded_json(strategy),
        now=NOW,
        timeframe="1d",
        calendar_code=CALENDAR,
        raw_bars=fixture_bars(empty=empty),
        early_positions={} if empty else {"OLD": 2.0},
        gate_universe=GATE,
        max_drawdown=0.1,
    )


def late_input(strategy: LoadedStrategy, early: EarlyPlannerInput, case: str) -> LatePlannerInput:
    first = phase_a(overlaid(strategy), early)
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


# --- the store: one environment and lazily built bundles ----------------------------------------


PROTOCOL = "_algua/protocol.json"


def generated(resolved: dict[str, Any]) -> dict[str, bytes]:
    """`_algua/` exactly as Story 1.3b's preparation writes it."""
    return {item.path: item.data for item in _bundle_files((), resolved)}


STAMP = generated({})[PROTOCOL]  # 1.3b's protocol.json bytes (independent of the config)


def protocol_bytes(**changes: Any) -> bytes:
    stamp = json.loads(STAMP)
    stamp.update(changes)
    return canonical_json(stamp).encode("utf-8")


# bundle key -> (strategy modules, recorded config, protocol.json bytes; None omits the file)
BUNDLES: dict[str, tuple[dict[str, str], dict[str, Any], bytes | None]] = {
    "good": ({STRATEGY: strategy_source(STRATEGY)}, recorded(in_process()), STAMP),
    "skewed": ({STRATEGY: strategy_source(STRATEGY)}, recorded(in_process(lookback=2)), STAMP),
    "leaky": ({LEAKY: strategy_source(LEAKY, leak=True)}, recorded(in_process(LEAKY)), STAMP),
    "no_protocol": ({STRATEGY: strategy_source(STRATEGY)}, recorded(in_process()), None),
    "wire_v2": (
        {STRATEGY: strategy_source(STRATEGY)}, recorded(in_process()),
        protocol_bytes(frozen_wire={"name": "frozen-planner", "version": 2}),
    ),
    "boundary_v2": (
        {STRATEGY: strategy_source(STRATEGY)}, recorded(in_process()),
        protocol_bytes(planner_boundary_version=2),
    ),
}


# --- a strategy that writes to stdout, the child's only result channel ---------------------------

NOISE_ON_LOAD = "noise: print on load"
#: What the noisy fixture writes while it scores (Phase B), in this order.
NOISE_WHILE_SCORING = ("noise: print", "noise: stderr", "noise: sys.__stdout__", "noise: fd 1")


def noisy_source(name: str = STRATEGY) -> str:
    """The fixture strategy, writing to stdout on load and while it scores, as chatty library or C
    code would: ``print``, then a stderr write, then the original ``sys.__stdout__``, then fd 1."""
    printed, errored, dunder, raw = NOISE_WHILE_SCORING
    return strategy_source(name).replace(
        "def signal(", f"import os, sys\nprint({NOISE_ON_LOAD!r})\ndef signal("
    ).replace(
        "    return view.groupby",
        f"    print({printed!r})\n"
        f"    sys.stderr.write({errored + chr(10)!r})\n"
        f"    sys.__stdout__.write({dunder + chr(10)!r})\n"
        "    sys.__stdout__.flush()\n"
        f"    os.write(1, {(raw + chr(10)).encode()!r})\n"
        "    return view.groupby",
    )


BUNDLES["noisy"] = ({STRATEGY: noisy_source()}, recorded(in_process()), STAMP)


def seal(root: Path) -> None:
    for directory, _, files in os.walk(root, topdown=False):
        for name in files:
            (Path(directory) / name).chmod(0o444)
        Path(directory).chmod(0o555)


def unseal(root: Path) -> None:
    for directory, _, _ in os.walk(root, topdown=True):
        Path(directory).chmod(0o755)


@dataclass
class Store:
    root: Path
    python: Path
    built: dict[str, Path] = field(default_factory=dict)

    @property
    def environment(self) -> Path:
        return self.python.parent.parent

    def bundle(self, key: str) -> Path:
        if key not in self.built:
            self.built[key] = self._build(key)
        return self.built[key]

    def _build(self, key: str) -> Path:
        strategies, resolved, stamp = BUNDLES[key]
        bundle_digest = digest(key)
        bundle = self.root / "frozen/bundles/sha256" / bundle_digest[:2] / bundle_digest
        shutil.copytree(CHECKOUT_ALGUA, bundle / "algua",
                        ignore=shutil.ignore_patterns("__pycache__"))
        family = bundle / "algua/strategies" / FAMILY
        family.mkdir()
        (family / "__init__.py").write_text("")
        for name, text in strategies.items():
            (family / f"{name}.py").write_text(text)
        files = {path: data for path, data in generated(resolved).items() if path != PROTOCOL}
        if stamp is not None:
            files[PROTOCOL] = stamp
        (bundle / "_algua").mkdir()
        for path, data in files.items():
            (bundle / path).write_bytes(data)
        seal(bundle)
        return bundle


def build_environment(root: Path, env_digest: str = ENV_DIGEST) -> Path:
    """A venv-like environment at its digest locator; returns its `bin/python`."""
    env = root / "frozen/environments/sha256" / env_digest[:2] / env_digest
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


@contextmanager
def frozen_store(root: Path) -> Iterator[Store]:
    """A store under ``root`` whose sealed bundles are unsealed again on exit."""
    built = Store(root, build_environment(root))
    try:
        yield built
    finally:
        for bundle in built.built.values():
            unseal(bundle)
