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
