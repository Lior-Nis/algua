import json

import pytest
from typer.testing import CliRunner

from algua.cli.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _tmp_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path / "data"))


def _json(result):
    assert result.exit_code == 0, result.stdout
    return json.loads(result.stdout)


def test_data_ingest_and_inspect(tmp_path):
    source = tmp_path / "bars.csv"
    source.write_text("ts,symbol,close\n2026-01-02,AAPL,100\n", encoding="utf-8")

    ingested = _json(
        runner.invoke(
            app,
            [
                "data",
                "ingest",
                "daily-bars",
                "--provider",
                "local",
                "--symbols",
                "AAPL",
                "--start",
                "2026-01-02",
                "--end",
                "2026-01-02",
                "--as-of",
                "2026-01-03T00:00:00+00:00",
                "--source",
                "fixture",
                "--from-file",
                str(source),
            ],
        )
    )

    assert ingested["ok"] is True
    snapshot_id = ingested["snapshot"]["snapshot_id"]

    listed = _json(runner.invoke(app, ["data", "inspect"]))
    assert [r["snapshot_id"] for r in listed] == [snapshot_id]

    shown = _json(runner.invoke(app, ["data", "inspect", "--snapshot-id", snapshot_id]))
    assert shown["dataset"] == "daily-bars"
    assert shown["symbols"] == ["AAPL"]


def test_data_errors_emit_json(tmp_path):
    result = runner.invoke(
        app,
        [
            "data",
            "ingest",
            "daily-bars",
            "--provider",
            "local",
            "--symbols",
            "AAPL",
            "--start",
            "2026-01-02",
            "--end",
            "2026-01-02",
            "--as-of",
            "2026-01-03T00:00:00+00:00",
            "--source",
            "fixture",
            "--from-file",
            str(tmp_path / "missing.csv"),
        ],
    )

    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False
