import json

import pandas as pd
import pytest
from typer.testing import CliRunner

from algua.cli.main import app
from algua.data.contracts import ProviderBars

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
                "bars",
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
    assert shown["dataset"] == "bars"
    assert shown["symbols"] == ["AAPL"]


def test_data_ingest_rejects_unknown_dataset(tmp_path):
    # #168: the `data ingest` positional is the closed Dataset enum — a typo fails closed
    # with a JSON error instead of recording a snapshot no typed reader can ever match.
    source = tmp_path / "bars.csv"
    source.write_text("ts,symbol,close\n2026-01-02,AAPL,100\n", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "data", "ingest", "daily-bars",
            "--provider", "local", "--symbols", "AAPL",
            "--start", "2026-01-02", "--end", "2026-01-02",
            "--as-of", "2026-01-03T00:00:00+00:00", "--source", "fixture",
            "--from-file", str(source),
        ],
    )

    assert result.exit_code != 0
    assert json.loads(result.stdout)["ok"] is False


def test_data_inspect_rejects_unknown_dataset_filter():
    result = runner.invoke(app, ["data", "inspect", "--dataset", "daily-bars"])

    assert result.exit_code != 0
    assert json.loads(result.stdout)["ok"] is False


@pytest.mark.parametrize(
    "argv",
    [
        ["data", "inspect", "--summary", "--snapshot-id", "x"],
        ["data", "inspect", "--summary", "--dataset", "daily-bars"],
        ["data", "inspect", "--snapshot-id", "x", "--dataset", "daily-bars"],
        ["data", "inspect", "--summary", "--snapshot-id", "x", "--dataset", "daily-bars"],
    ],
)
def test_data_inspect_rejects_multiple_selectors(argv):
    """Passing >1 of --summary/--snapshot-id/--dataset fails closed instead of silently
    resolving by return-order precedence (#260)."""
    result = runner.invoke(app, argv)
    assert result.exit_code != 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert "only one of" in payload["error"]


def test_data_ingest_bars_with_provider(monkeypatch):
    from algua.cli import data_cmd

    class FakeProvider:
        def get_bars(self, _request):
            return ProviderBars(
                frame=pd.DataFrame(
                    {
                        "ts": ["2026-01-02T00:00:00+00:00"],
                        "symbol": ["AAPL"],
                        "open": [99.0],
                        "high": [101.0],
                        "low": [98.0],
                        "close": [100.0],
                        "adj_close": [99.5],
                        "volume": [1000.0],
                    }
                ),
                source_metadata={"provider": "fake"},
            )

    monkeypatch.setattr(data_cmd, "_bar_provider", lambda _name: FakeProvider())

    out = _json(
        runner.invoke(
            app,
            [
                "data",
                "ingest-bars",
                "--provider",
                "fake",
                "--symbols",
                "AAPL",
                "--start",
                "2026-01-02",
                "--end",
                "2026-01-03",
                "--as-of",
                "2026-01-04T00:00:00+00:00",
            ],
        )
    )

    assert out["snapshot"]["dataset"] == "bars"
    assert out["snapshot"]["storage_format"] == "parquet_dataset"


def test_data_ingest_bars_provider_failure_emits_json(monkeypatch):
    from algua.cli import data_cmd
    from algua.data.providers.errors import ProviderError

    class FailingProvider:
        def get_bars(self, _request):
            raise ProviderError("yfinance download failed: read timed out")

    monkeypatch.setattr(data_cmd, "_bar_provider", lambda _name: FailingProvider())

    result = runner.invoke(
        app,
        [
            "data",
            "ingest-bars",
            "--provider",
            "yfinance",
            "--symbols",
            "AAPL",
            "--start",
            "2026-01-02",
            "--end",
            "2026-01-03",
            "--as-of",
            "2026-01-04T00:00:00+00:00",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert "download failed" in payload["error"]


def test_data_ingest_universe_and_summary():
    out = _json(
        runner.invoke(
            app,
            [
                "data",
                "ingest-universe",
                "core",
                "--symbols",
                "AAPL,MSFT",
                "--effective-date",
                "2026-01-02",
                "--as-of",
                "2026-01-03T00:00:00+00:00",
            ],
        )
    )

    assert out["snapshot"]["dataset"] == "universes"
    summary = _json(runner.invoke(app, ["data", "inspect", "--summary"]))
    assert summary["datasets"][0]["dataset"] == "universes"
    assert summary["datasets"][0]["symbols"] == ["AAPL", "MSFT"]


def test_data_errors_emit_json(tmp_path):
    result = runner.invoke(
        app,
        [
            "data",
            "ingest",
            "bars",
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


def _write_raw_p(path, ts, closes):
    pd.DataFrame(
        {
            "ts": ts,
            "open": closes,
            "high": [c + 1 for c in closes],
            "low": [c - 1 for c in closes],
            "close": closes,
            "volume": [100.0] * len(closes),
        }
    ).to_parquet(path)


def test_cli_databento_import(tmp_path, monkeypatch):
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path / "store"))
    raw = tmp_path / "raw"
    raw.mkdir()
    ca = tmp_path / "ca.parquet"
    _write_raw_p(
        raw / "AAPL.parquet",
        pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC"),
        [100.0, 110.0, 50.0, 55.0],
    )
    pd.DataFrame(
        [{"symbol": "AAPL", "ex_date": "2024-01-03", "kind": "split", "value": 2.0}]
    ).to_parquet(ca)
    res = runner.invoke(
        app,
        [
            "data", "import-bars",
            "--vendor", "databento",
            "--raw-dir", str(raw),
            "--corp-actions", str(ca),
            "--as-of", "2024-06-01T00:00:00Z",
        ],
    )
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert payload["ok"] is True
    # provenance carries the CA hash
    snap = payload["snapshot"]
    assert "corp_actions_sha256" in json.dumps(snap)


def test_cli_databento_requires_corp_actions(tmp_path, monkeypatch):
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path / "store"))
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_raw_p(
        raw / "AAPL.parquet",
        pd.date_range("2024-01-01", periods=1, freq="D", tz="UTC"),
        [100.0],
    )
    res = runner.invoke(
        app,
        [
            "data", "import-bars",
            "--vendor", "databento",
            "--raw-dir", str(raw),
            "--as-of", "2024-06-01T00:00:00Z",
        ],
    )
    assert res.exit_code != 0
    assert "corp-actions" in res.output


def test_data_verify_all_healthy_exits_zero(tmp_path):
    source = tmp_path / "bars.csv"
    source.write_text("ts,symbol,close\n2026-01-02,AAPL,100\n", encoding="utf-8")
    # arrange: ingest a snapshot so the data dir is non-empty
    _json(
        runner.invoke(
            app,
            [
                "data",
                "ingest",
                "bars",
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
    result = runner.invoke(app, ["data", "verify"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["failed"] == 0
    assert payload["verified"] >= 1


def test_data_verify_flags_damage_and_exits_nonzero(tmp_path, monkeypatch):
    import os

    data_dir = tmp_path / "data"
    monkeypatch.setenv("ALGUA_DATA_DIR", str(data_dir))

    source = tmp_path / "bars.csv"
    source.write_text("ts,symbol,close\n2026-01-02,AAPL,100\n", encoding="utf-8")
    _json(
        runner.invoke(
            app,
            [
                "data",
                "ingest",
                "bars",
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
    # find the payload file in data_dir and corrupt it
    payload_files = list(data_dir.rglob("*.parquet")) + list(data_dir.rglob("*.csv"))
    assert payload_files, "expected at least one payload file on disk"
    target = payload_files[0]
    with target.open("r+b") as fh:
        fh.seek(0)
        fh.write(b"\x00" * min(64, os.path.getsize(target)))

    result = runner.invoke(app, ["data", "verify"])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["failed"] == 1
    assert any(not s["ok"] for s in payload["snapshots"])


def test_data_verify_unknown_snapshot_id_exits_nonzero():
    result = runner.invoke(app, ["data", "verify", "--snapshot-id", "deadbeefdeadbeef"])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False


def test_cli_databento_rejects_adjusted_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path / "store"))
    raw = tmp_path / "raw"
    raw.mkdir()
    ca = tmp_path / "ca.parquet"
    _write_raw_p(raw / "AAPL.parquet",
                 pd.date_range("2024-01-01", periods=1, freq="D", tz="UTC"), [100.0])
    pd.DataFrame(
        [{"symbol": "AAPL", "ex_date": "2024-01-03", "kind": "split", "value": 2.0}]
    ).to_parquet(ca)
    res = runner.invoke(app, [
        "data", "import-bars", "--vendor", "databento", "--raw-dir", str(raw),
        "--corp-actions", str(ca), "--adjusted-dir", str(raw), "--as-of", "2024-06-01T00:00:00Z",
    ])
    assert res.exit_code != 0
    assert "adjusted-dir" in res.output
