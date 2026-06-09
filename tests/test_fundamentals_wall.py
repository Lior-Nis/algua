import ast
import pathlib
import tomllib

REPO = pathlib.Path(__file__).resolve().parents[1]


def _contracts():
    data = tomllib.loads((REPO / "pyproject.toml").read_text())
    return data["tool"]["importlinter"]["contracts"]


def test_strategies_barred_from_data_lane():
    cs = _contracts()
    assert any(
        c.get("source_modules") == ["algua.strategies"]
        and "algua.data" in c.get("forbidden_modules", [])
        for c in cs
    ), "missing: algua.strategies forbidden from algua.data"


def test_contracts_barred_from_data_lane():
    cs = _contracts()
    contracts_rule = next(c for c in cs if c.get("source_modules") == ["algua.contracts"])
    assert "algua.data" in contracts_rule["forbidden_modules"]


def test_hindsight_module_walled():
    cs = _contracts()
    rule = next(
        (c for c in cs if c.get("forbidden_modules") == ["algua.data.hindsight"]), None
    )
    assert rule is not None, "missing dedicated algua.data.hindsight forbidden contract"
    for src in ["algua.backtest", "algua.features", "algua.contracts",
                "algua.strategies", "algua.live", "algua.execution"]:
        assert src in rule["source_modules"]


def test_no_static_data_import_in_pure_layers():
    """Defense beyond config: assert no module under algua/strategies or algua/contracts imports
    algua.data (the actual property the wall protects)."""
    offenders = []
    for pkg in ["algua/strategies", "algua/contracts"]:
        for path in (REPO / pkg).rglob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and (
                    (node.module or "").startswith("algua.data")
                ):
                    offenders.append(str(path))
                if isinstance(node, ast.Import):
                    for a in node.names:
                        if a.name.startswith("algua.data"):
                            offenders.append(str(path))
    assert not offenders, f"pure layers import algua.data: {offenders}"
