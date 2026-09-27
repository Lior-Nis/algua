import sys
from pathlib import Path

import pytest

from algua.strategies.base import LoadedStrategy
from algua.strategies.loader import StrategyNotFound, list_strategies, load_strategy


def test_load_bundled_momentum():
    strat = load_strategy("cross_sectional_momentum")
    assert isinstance(strat, LoadedStrategy)
    assert strat.name == "cross_sectional_momentum"


def test_unknown_strategy_raises():
    with pytest.raises(StrategyNotFound):
        load_strategy("does_not_exist")


def test_strategy_missing_signal_raises():
    import algua.strategies.momentum as fam

    # Non-`_` name: `_`-prefixed modules are skipped at discovery (so this must be discoverable to
    # exercise the missing-signal branch). Cleaned up in `finally`.
    mod_path = Path(fam.__path__[0]) / "tmp_no_signal_fn.py"
    mod_path.write_text(
        "from algua.contracts.types import ExecutionContract\n"
        "from algua.strategies.base import StrategyConfig\n"
        "CONFIG = StrategyConfig(name='tmp_no_signal_fn', universe=['AAPL'],\n"
        "    execution=ExecutionContract(rebalance_frequency='1d'),\n"
        "    construction='equal_weight_positive')\n"
    )
    try:
        with pytest.raises(StrategyNotFound, match="signal"):
            load_strategy("tmp_no_signal_fn")
    finally:
        mod_path.unlink()


def test_list_includes_bundled():
    assert "cross_sectional_momentum" in list_strategies()


def test_loader_resolves_and_binds_construction():
    s = load_strategy("cross_sectional_momentum")
    assert s.config.construction == "top_k_equal_weight"
    assert s.signal_fn is not None and s.signal_panel_fn is not None
    assert callable(s.construct_fn)


def test_loader_rejects_unknown_construction():
    # A module whose CONFIG names a missing policy must fail at load. Non-`_` name in a real
    # family dir so `_index` discovers it (`_`-prefixed modules are skipped). Cleaned up after.
    import algua.strategies.momentum as fam
    mod_path = Path(fam.__path__[0]) / "tmp_bad_policy.py"
    mod_path.write_text(
        "import pandas as pd\n"
        "from algua.contracts.types import ExecutionContract\n"
        "from algua.strategies.base import StrategyConfig\n"
        "CONFIG = StrategyConfig(name='tmp_bad_policy', universe=['A'],\n"
        "    execution=ExecutionContract(rebalance_frequency='1d'),\n"
        "    construction='nope_not_real')\n"
        "def signal(view, params):\n"
        "    return pd.Series(dtype='float64')\n"
    )
    try:
        with pytest.raises(StrategyNotFound):
            load_strategy("tmp_bad_policy")
    finally:
        mod_path.unlink()


def test_loader_rejects_config_name_mismatch():
    """A module whose CONFIG.name differs from the file/registry stem fails closed (#275),
    so a strategy's identity can't silently fragment across MLflow/docs/registry."""
    import algua.strategies.momentum as fam
    mod_path = Path(fam.__path__[0]) / "tmp_name_mismatch.py"
    mod_path.write_text(
        "import pandas as pd\n"
        "from algua.contracts.types import ExecutionContract\n"
        "from algua.strategies.base import StrategyConfig\n"
        # CONFIG.name deliberately != the file stem 'tmp_name_mismatch'
        "CONFIG = StrategyConfig(name='some_other_name', universe=['A'],\n"
        "    execution=ExecutionContract(rebalance_frequency='1d'),\n"
        "    construction='equal_weight_positive')\n"
        "def signal(view, params):\n"
        "    return pd.Series(dtype='float64')\n"
    )
    try:
        with pytest.raises(StrategyNotFound, match="must match"):
            load_strategy("tmp_name_mismatch")
    finally:
        mod_path.unlink()


def test_load_by_bare_name_across_families(tmp_path):
    # Both bundled strategies resolve by bare name regardless of which family dir they live in.
    assert load_strategy("cross_sectional_momentum").name == "cross_sectional_momentum"
    assert load_strategy("fundamentals_earnings_tilt").name == "fundamentals_earnings_tilt"


def test_duplicate_bare_name_across_families_raises():
    """A bare name appearing in two family dirs is a hard, fail-closed error (raised during
    discovery, before any module is imported or its contract checked)."""
    import algua.strategies as sp
    root = Path(sp.__file__).parent
    # Non-`_` family dirs: `_`-prefixed dirs are skipped at discovery, so the dup must be in real
    # (discoverable) family dirs. Cleaned up in `finally`.
    fam_a, fam_b = root / "dupfam_a", root / "dupfam_b"
    body = (
        "import pandas as pd\n"
        "from algua.contracts.types import ExecutionContract\n"
        "from algua.strategies.base import StrategyConfig\n"
        "CONFIG = StrategyConfig(name='dupe', universe=['AAA'],\n"
        "    execution=ExecutionContract(rebalance_frequency='1d'),\n"
        "    construction='equal_weight_positive')\n"
        "def signal(view, params):\n    return pd.Series(dtype='float64')\n"
    )
    try:
        for fam in (fam_a, fam_b):
            fam.mkdir()
            (fam / "__init__.py").write_text("")
            (fam / "dupe.py").write_text(body)
        with pytest.raises(StrategyNotFound, match="duplicate"):
            load_strategy("dupe")
    finally:
        import shutil
        shutil.rmtree(fam_a, ignore_errors=True)
        shutil.rmtree(fam_b, ignore_errors=True)


def test_loading_one_strategy_does_not_import_siblings():
    """The single-import contract: loading one strategy must not pull a sibling into sys.modules."""
    sys.modules.pop("algua.strategies.fundamentals.fundamentals_earnings_tilt", None)
    load_strategy("cross_sectional_momentum")
    assert "algua.strategies.fundamentals.fundamentals_earnings_tilt" not in sys.modules


def test_family_init_files_are_empty():
    """Enforce the empty-`__init__` convention the single-import contract relies on: a family
    `__init__.py` that imported a sibling would silently pull extra modules into the code_hash
    closure when any strategy in that family is loaded. Committed family inits must stay empty."""
    import algua.strategies as sp
    root = Path(sp.__file__).parent
    for fam in root.iterdir():
        if not (fam.is_dir() and not fam.name.startswith("_") and (fam / "__init__.py").exists()):
            continue
        body = (fam / "__init__.py").read_text().strip()
        assert body == "", f"family package {fam.name}/__init__.py must be empty, has: {body!r}"


def test_load_tradable_strategy_loads_plain_strategy():
    from algua.strategies.loader import load_tradable_strategy
    s = load_tradable_strategy("cross_sectional_momentum")
    assert s.config.name == "cross_sectional_momentum"


def test_load_tradable_strategy_rejects_fundamentals_strategy(monkeypatch):
    # A needs_fundamentals strategy must be refused by the tradability gate.
    import algua.strategies.loader as loader

    sentinel = object()

    def _fake_load(name):
        return sentinel

    def _boom(strategy):
        assert strategy is sentinel
        raise ValueError("needs_fundamentals: not tradable without a fundamentals lane")

    monkeypatch.setattr(loader, "load_strategy", _fake_load)
    monkeypatch.setattr(loader, "assert_tradable_without_fundamentals", _boom)
    with pytest.raises(ValueError, match="needs_fundamentals"):
        loader.load_tradable_strategy("x")


def test_reload_purges_strategy_and_helper_module_state(tmp_path):
    """Warm-worker state hygiene (#326): load_strategy(reload=True) must reset BOTH the strategy
    module's own globals AND its author-written first-party helper modules (part of the artifact
    identity, hashed into code_hash) — so a batch worker's task N+1 never inherits task N's state.
    Regression for the Codex GATE-2 finding: a root-only reload leaks a sibling helper's state."""
    import algua.strategies.momentum as fam

    fam_dir = Path(fam.__path__[0])
    # `_`-prefixed: a real first-party helper module, skipped by strategy discovery.
    helper = fam_dir / "_reload_probe_helper.py"
    strat = fam_dir / "reload_probe_strat.py"
    # The helper carries mutable module-level state that its score() advances on each call.
    helper.write_text(
        "count = 0\n"
        "def bump():\n"
        "    global count\n"
        "    count += 1\n"
        "    return count\n"
    )
    strat.write_text(
        "import pandas as pd\n"
        "from algua.contracts.types import ExecutionContract\n"
        "from algua.strategies.base import StrategyConfig\n"
        "from algua.strategies.momentum._reload_probe_helper import bump\n"
        "CONFIG = StrategyConfig(name='reload_probe_strat', universe=['AAPL'],\n"
        "    execution=ExecutionContract(rebalance_frequency='1d'),\n"
        "    construction='equal_weight_positive')\n"
        "def signal(view, params):\n"
        "    return pd.Series({'AAPL': float(bump())})\n"
    )
    try:
        # Task 1: helper.count advances to 1 via the signal.
        s1 = load_strategy("reload_probe_strat", reload=True)
        assert s1.signal_fn(None, {})["AAPL"] == 1.0
        # Task 2 in the SAME process: reload must reset the helper's `count` to 0 so the next
        # signal call starts at 1 again — identical to a cold process. A root-only reload would
        # leak `count` and return 2.0 here.
        s2 = load_strategy("reload_probe_strat", reload=True)
        assert s2.signal_fn(None, {})["AAPL"] == 1.0, "helper module state leaked across reload"
    finally:
        for f in (helper, strat):
            f.unlink(missing_ok=True)
        for m in ("algua.strategies.momentum.reload_probe_strat",
                  "algua.strategies.momentum._reload_probe_helper"):
            sys.modules.pop(m, None)


def test_reload_is_dependency_safe_for_helper_to_helper_imports(tmp_path):
    """Warm reload (#326) must be dependency-order-safe: a helper A that re-exports an object from
    a deeper helper B, whose object carries mutable class-attached state, must still reset across a
    warm reload (sys.modules insertion order is dependency-first, so B reloads before A before the
    strategy). Regression for the Codex GATE-2 helper-to-helper reload-order concern."""
    import algua.strategies.momentum as fam

    d = Path(fam.__path__[0])
    state = d / "_dep_state_probe.py"   # B: owns the mutable state
    mid = d / "_dep_mid_probe.py"       # A: re-exports B's object
    strat = d / "dep_order_probe_strat.py"
    state.write_text(
        "class F:\n    count = 0\n"
        "def tick():\n    F.count += 1\n    return F.count\n"
    )
    mid.write_text("from algua.strategies.momentum._dep_state_probe import tick\n")
    strat.write_text(
        "import pandas as pd\n"
        "from algua.contracts.types import ExecutionContract\n"
        "from algua.strategies.base import StrategyConfig\n"
        "from algua.strategies.momentum._dep_mid_probe import tick\n"
        "CONFIG = StrategyConfig(name='dep_order_probe_strat', universe=['AAPL'],\n"
        "    execution=ExecutionContract(rebalance_frequency='1d'),\n"
        "    construction='equal_weight_positive')\n"
        "def signal(view, params):\n    return pd.Series({'AAPL': float(tick())})\n"
    )
    try:
        s1 = load_strategy("dep_order_probe_strat", reload=True)
        assert s1.signal_fn(None, {})["AAPL"] == 1.0
        s2 = load_strategy("dep_order_probe_strat", reload=True)
        assert s2.signal_fn(None, {})["AAPL"] == 1.0, "transitive helper state leaked across reload"
    finally:
        for f in (state, mid, strat):
            f.unlink(missing_ok=True)
        for m in ("algua.strategies.momentum.dep_order_probe_strat",
                  "algua.strategies.momentum._dep_mid_probe",
                  "algua.strategies.momentum._dep_state_probe"):
            sys.modules.pop(m, None)


def test_reload_resets_reexported_class_state(tmp_path):
    """Warm reload (#326) resets state even when a helper A re-exports a CLASS (not a function)
    from a deeper helper B and the strategy mutates a class attribute. sys.modules insertion order
    is dependency-first (B before A before the strategy), so reloading in that order rebinds A's
    re-exported name to the freshly reloaded class. Regression for the Codex GATE-2 class/object
    re-export concern (which predicted a leak here)."""
    import algua.strategies.momentum as fam

    d = Path(fam.__path__[0])
    state = d / "_cls_state_probe.py"   # B: owns the class with mutable class-attr state
    mid = d / "_cls_mid_probe.py"       # A: re-exports the CLASS object itself
    strat = d / "cls_reexport_probe_strat.py"
    state.write_text("class Counter:\n    count = 0\n")
    mid.write_text("from algua.strategies.momentum._cls_state_probe import Counter\n")
    strat.write_text(
        "import pandas as pd\n"
        "from algua.contracts.types import ExecutionContract\n"
        "from algua.strategies.base import StrategyConfig\n"
        "from algua.strategies.momentum._cls_mid_probe import Counter\n"
        "CONFIG = StrategyConfig(name='cls_reexport_probe_strat', universe=['AAPL'],\n"
        "    execution=ExecutionContract(rebalance_frequency='1d'),\n"
        "    construction='equal_weight_positive')\n"
        "def signal(view, params):\n"
        "    Counter.count += 1\n"
        "    return pd.Series({'AAPL': float(Counter.count)})\n"
    )
    try:
        s1 = load_strategy("cls_reexport_probe_strat", reload=True)
        assert s1.signal_fn(None, {})["AAPL"] == 1.0
        s2 = load_strategy("cls_reexport_probe_strat", reload=True)
        assert s2.signal_fn(None, {})["AAPL"] == 1.0, "re-exported class state leaked across reload"
    finally:
        for f in (state, mid, strat):
            f.unlink(missing_ok=True)
        for m in ("algua.strategies.momentum.cls_reexport_probe_strat",
                  "algua.strategies.momentum._cls_mid_probe",
                  "algua.strategies.momentum._cls_state_probe"):
            sys.modules.pop(m, None)


def test_load_strategy_config_refreshes_a_warm_module_before_retaining_config():
    """Frozen preparation retains CONFIG from the declared-config read: a warm process that already
    imported the module must not freeze its stale configuration after the source changed."""
    import algua.strategies.momentum as fam
    from algua.strategies.loader import load_strategy_config

    strat = Path(fam.__path__[0]) / "config_refresh_probe.py"
    template = (
        "import pandas as pd\n"
        "from algua.contracts.types import ExecutionContract\n"
        "from algua.strategies.base import StrategyConfig\n"
        "CONFIG = StrategyConfig(name='config_refresh_probe', universe=[{universe}],\n"
        "    execution=ExecutionContract(rebalance_frequency='1d'),\n"
        "    construction='equal_weight_positive')\n"
        "def signal(view, params):\n"
        "    return pd.Series(dtype='float64')\n"
    )
    try:
        strat.write_text(template.format(universe="'AAPL'"))
        assert load_strategy_config("config_refresh_probe").universe == ["AAPL"]
        strat.write_text(template.format(universe="'AAPL', 'MSFT'"))
        assert load_strategy_config("config_refresh_probe").universe == ["AAPL", "MSFT"]
        assert load_strategy("config_refresh_probe").config.universe == ["AAPL", "MSFT"]
    finally:
        strat.unlink(missing_ok=True)
        sys.modules.pop("algua.strategies.momentum.config_refresh_probe", None)


def test_load_strategy_config_bypasses_timestamp_valid_stale_bytecode():
    """A same-size, same-mtime source edit leaves the cached bytecode's timestamp stamp valid; the
    warm refresh must still compile the CURRENT source rather than retain the stale CONFIG."""
    import importlib.util
    import os
    import py_compile

    import algua.strategies.momentum as fam
    from algua.strategies.loader import load_strategy_config

    strat = Path(fam.__path__[0]) / "bytecode_refresh_probe.py"
    cached = Path(importlib.util.cache_from_source(str(strat)))
    template = (
        "import pandas as pd\n"
        "from algua.contracts.types import ExecutionContract\n"
        "from algua.strategies.base import StrategyConfig\n"
        "CONFIG = StrategyConfig(name='bytecode_refresh_probe', universe=['{symbol}'],\n"
        "    execution=ExecutionContract(rebalance_frequency='1d'),\n"
        "    construction='equal_weight_positive')\n"
        "def signal(view, params):\n"
        "    return pd.Series(dtype='float64')\n"
    )
    try:
        strat.write_text(template.format(symbol="AAPL"))
        assert load_strategy_config("bytecode_refresh_probe").universe == ["AAPL"]
        py_compile.compile(
            str(strat), cfile=str(cached),
            invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP,
        )
        stamp = strat.stat()
        strat.write_text(template.format(symbol="MSFT"))
        os.utime(strat, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        assert strat.stat().st_size == stamp.st_size
        assert load_strategy_config("bytecode_refresh_probe").universe == ["MSFT"]
    finally:
        strat.unlink(missing_ok=True)
        cached.unlink(missing_ok=True)
        sys.modules.pop("algua.strategies.momentum.bytecode_refresh_probe", None)


def test_load_strategy_config_reloads_the_family_closure_dependency_first():
    """``sys.modules`` order reflects when each module last finished executing, not its current
    source's imports: a helper loaded before an independently imported deeper helper, and only
    later edited to import it, sits BEFORE it. The refresh must still reload the deeper helper
    first, or the helper re-binds a stale value and the retained CONFIG is stale."""
    import importlib

    import algua.strategies.momentum as fam
    from algua.strategies.loader import load_strategy_config

    d = Path(fam.__path__[0])
    deep = d / "_closure_deep_probe.py"
    mid = d / "_closure_mid_probe.py"
    strat = d / "closure_order_probe.py"
    mid.write_text("UNIVERSE = ['AAPL']\n")
    strat.write_text(
        "import pandas as pd\n"
        "from algua.contracts.types import ExecutionContract\n"
        "from algua.strategies.base import StrategyConfig\n"
        "from algua.strategies.momentum._closure_mid_probe import UNIVERSE\n"
        "CONFIG = StrategyConfig(name='closure_order_probe', universe=list(UNIVERSE),\n"
        "    execution=ExecutionContract(rebalance_frequency='1d'),\n"
        "    construction='equal_weight_positive')\n"
        "def signal(view, params):\n"
        "    return pd.Series(dtype='float64')\n"
    )
    try:
        assert load_strategy_config("closure_order_probe").universe == ["AAPL"]
        # Another family member imports the deeper helper independently, AFTER the helper that
        # will later depend on it is already loaded.
        deep.write_text("UNIVERSE = ['AAPL', 'MSFT']\n")
        importlib.invalidate_caches()
        importlib.import_module("algua.strategies.momentum._closure_deep_probe")
        mid.write_text("from algua.strategies.momentum._closure_deep_probe import UNIVERSE\n")
        deep.write_text("UNIVERSE = ['MSFT']\n")
        assert load_strategy_config("closure_order_probe").universe == ["MSFT"]
        assert load_strategy("closure_order_probe").config.universe == ["MSFT"]
    finally:
        for f in (deep, mid, strat):
            f.unlink(missing_ok=True)
        for m in ("algua.strategies.momentum.closure_order_probe",
                  "algua.strategies.momentum._closure_mid_probe",
                  "algua.strategies.momentum._closure_deep_probe"):
            sys.modules.pop(m, None)


def test_load_strategy_config_fails_closed_on_a_cyclic_family_import_graph():
    """A cyclic static import component has no dependency-safe refresh order: the declared-config
    read refuses it as not found (frozen preparation's fail-closed mapping) and leaves the warm
    family closure exactly as it was."""
    import algua.strategies.momentum as fam
    from algua.strategies.loader import load_strategy_config

    d = Path(fam.__path__[0])
    left, right = d / "_cycle_left_probe.py", d / "_cycle_right_probe.py"
    strat = d / "cycle_probe_strat.py"
    left.write_text("UNIVERSE = ['AAPL']\n")
    strat.write_text(
        "import pandas as pd\n"
        "from algua.contracts.types import ExecutionContract\n"
        "from algua.strategies.base import StrategyConfig\n"
        "from algua.strategies.momentum._cycle_left_probe import UNIVERSE\n"
        "CONFIG = StrategyConfig(name='cycle_probe_strat', universe=list(UNIVERSE),\n"
        "    execution=ExecutionContract(rebalance_frequency='1d'),\n"
        "    construction='equal_weight_positive')\n"
        "def signal(view, params):\n"
        "    return pd.Series(dtype='float64')\n"
    )
    try:
        assert load_strategy_config("cycle_probe_strat").universe == ["AAPL"]
        warm = sys.modules["algua.strategies.momentum.cycle_probe_strat"]
        left.write_text(
            "from algua.strategies.momentum import _cycle_right_probe\nUNIVERSE = ['MSFT']\n")
        right.write_text("from algua.strategies.momentum import _cycle_left_probe\n")
        with pytest.raises(StrategyNotFound, match="cyclic"):
            load_strategy_config("cycle_probe_strat")
        assert sys.modules["algua.strategies.momentum.cycle_probe_strat"] is warm
        assert load_strategy("cycle_probe_strat").config.universe == ["AAPL"]
    finally:
        for f in (left, right, strat):
            f.unlink(missing_ok=True)
        for m in ("algua.strategies.momentum.cycle_probe_strat",
                  "algua.strategies.momentum._cycle_left_probe",
                  "algua.strategies.momentum._cycle_right_probe"):
            sys.modules.pop(m, None)
