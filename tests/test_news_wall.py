import ast
import pathlib
import tomllib

REPO = pathlib.Path(__file__).resolve().parents[1]


def _contracts():
    data = tomllib.loads((REPO / "pyproject.toml").read_text())
    return data["tool"]["importlinter"]["contracts"]


def test_live_execution_barred_from_data_lane():
    cs = _contracts()
    assert any(
        set(c.get("source_modules", [])) == {"algua.live", "algua.execution"}
        and c.get("forbidden_modules") == ["algua.data"]
        for c in cs
    ), "missing: algua.live/algua.execution forbidden from algua.data"


def test_query_news_lives_in_walled_hindsight_module():
    import algua.data.hindsight as h

    assert h.query_news.__module__ == "algua.data.hindsight"


def test_news_full_history_read_not_on_a_decision_lane_module():
    # The only full-history news read surfaces are store.read_news (in algua.data, walled from all
    # decision/execution lanes) and hindsight.query_news (in the walled module). Assert no news read
    # leaked onto contracts.types (an importable base layer).
    import algua.contracts.types as t

    assert not hasattr(t, "read_news") and not hasattr(t, "query_news")


def test_all_decision_lanes_barred_from_data_lane_in_config():
    # The complete decision/execution lane set must be forbidden from algua.data — across several
    # contracts. (Guards against the easy misread that only live/execution were added.)
    cs = _contracts()
    barred = {
        m
        for c in cs
        if "algua.data" in c.get("forbidden_modules", [])
        for m in c.get("source_modules", [])
    }
    for lane in [
        "algua.backtest", "algua.features", "algua.strategies",
        "algua.contracts", "algua.live", "algua.execution",
    ]:
        assert lane in barred, f"{lane} is not forbidden from algua.data by any contract"


def test_no_static_data_import_in_live_or_execution():
    """Defense beyond config (mirrors the fundamentals wall test): assert no module under
    algua/live or algua/execution statically imports algua.data — the property the new contract
    protects. This is what makes store.read_news unreachable from the live decision lane."""
    offenders = []
    for pkg in ["algua/live", "algua/execution"]:
        for path in (REPO / pkg).rglob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                mod = node.module or "" if isinstance(node, ast.ImportFrom) else ""
                if mod.startswith("algua.data"):
                    offenders.append(str(path))
                if isinstance(node, ast.Import):
                    for a in node.names:
                        if a.name.startswith("algua.data"):
                            offenders.append(str(path))
    assert not offenders, f"live/execution import algua.data: {offenders}"
