import json

import pandas as pd
from typer.testing import CliRunner

from algua.cli.main import app

runner = CliRunner()


def test_ingest_then_query_fundamentals_roundtrip(tmp_path, monkeypatch):
    # match tests/test_cli_data.py convention
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path / "data"))
    src = tmp_path / "funds.csv"
    pd.DataFrame(
        [
            ["AAPL", "2025-03-31", "revenue", 100.0, "2025-05-01T13:00:00Z", "vendorX"],
            ["AAPL", "2025-03-31", "revenue", 110.0, "2025-08-01T13:00:00Z", "vendorX"],
        ],
        columns=["symbol", "fiscal_period_end", "metric", "value", "knowable_at", "source"],
    ).to_csv(src, index=False)

    ing = runner.invoke(app, ["data", "ingest-fundamentals", "--from-file", str(src),
                              "--provider", "vendorX", "--symbols", "AAPL",
                              "--as-of", "2025-09-01T00:00:00Z", "--source", "vendorX"])
    assert ing.exit_code == 0, ing.output
    sid = json.loads(ing.output)["snapshot"]["snapshot_id"]

    q = runner.invoke(app, ["data", "query-fundamentals",
                            "--snapshot-id", sid, "--symbols", "AAPL"])
    assert q.exit_code == 0, q.output
    rows = json.loads(q.output)
    assert len(rows) == 2  # full hindsight
    assert rows[0]["symbol"] == "AAPL"
