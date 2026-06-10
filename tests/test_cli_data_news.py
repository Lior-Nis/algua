import json

import pandas as pd
from typer.testing import CliRunner

from algua.cli.main import app

runner = CliRunner()


def _write_csv(tmp_path):
    p = tmp_path / "news.csv"
    pd.DataFrame([
        {"source": "reuters", "article_id": "a1", "symbols": "AAPL,MSFT",
         "published_at": "2025-01-02T13:00:00Z", "knowable_at": "2025-01-02T13:00:00Z",
         "headline": "h", "url": "http://x/1", "body": "b"},
    ]).to_csv(p, index=False)
    return p


def test_ingest_then_query_news(tmp_path, monkeypatch):
    # match the fundamentals CLI test's env
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path / "data"))
    csv = _write_csv(tmp_path)
    r = runner.invoke(app, ["data", "ingest-news", "--provider", "f",
                            "--as-of", "2025-01-03T00:00:00Z", "--from-file", str(csv)])
    assert r.exit_code == 0, r.output
    snap = json.loads(r.output)["snapshot"]["snapshot_id"]
    q = runner.invoke(app, ["data", "query-news", "--snapshot-id", snap])
    assert q.exit_code == 0, q.output
    rows = json.loads(q.output)
    assert sorted(x["symbol"] for x in rows) == ["AAPL", "MSFT"]
    # the non-null url must round-trip EXACTLY (a parquet/JSON null regression must fail here)
    assert all(x["url"] == "http://x/1" and x["body"] == "b" for x in rows)


def test_query_news_emits_null_for_missing_url_body(tmp_path, monkeypatch):
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path / "data"))
    p = tmp_path / "news_null.csv"
    pd.DataFrame([
        {"source": "reuters", "article_id": "a9", "symbols": "AAPL",
         "published_at": "2025-01-02T13:00:00Z", "knowable_at": "2025-01-02T13:00:00Z",
         "headline": "h", "url": "", "body": ""},  # empty CSV cells -> null
    ]).to_csv(p, index=False)
    r = runner.invoke(app, ["data", "ingest-news", "--provider", "f",
                            "--as-of", "2025-01-03T00:00:00Z", "--from-file", str(p)])
    assert r.exit_code == 0, r.output
    snap = json.loads(r.output)["snapshot"]["snapshot_id"]
    q = runner.invoke(app, ["data", "query-news", "--snapshot-id", snap])
    assert q.exit_code == 0, q.output
    row = json.loads(q.output)[0]
    assert row["url"] is None and row["body"] is None
