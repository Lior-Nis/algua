import json
import os
import sqlite3

import pytest
from typer.testing import CliRunner

from algua.cli.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _tmp_db(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))
    yield tmp_path


def _run(*args):
    return runner.invoke(app, ["research", "idea", *args])


def _json(r):
    return json.loads(r.stdout)


# Distinct multi-word tags per index — a bare digit ("0"/"1") is stripped by the dedup
# tokenizer's len(t) > 1 filter, so two ideas differing only by digit collide (jaccard 1.0);
# these tags carry enough unique tokens to keep the seeded ideas' signatures apart.
_SEED_TAGS = ["alpha lunar drift pattern", "bravo solar spike anomaly",
             "charlie comet flare surge", "delta nova burst swing"]


def _seed(n=2):
    ids = []
    for i in range(n):
        r = _run("add", "--title", f"seed idea {_SEED_TAGS[i]}", "--hypothesis",
                 f"seed hyp {_SEED_TAGS[i]} persists", "--source-type", "inspiration",
                 "--category", "momentum", "--market", "us_equities", "--horizon", "daily",
                 "--falsification", "f", "--inspiration", f"i{i}|blog/x|niche")
        assert r.exit_code == 0, r.output
        ids.append(_json(r)["id"])
    return ids


def test_claim_then_record_outcome_then_depth():
    _seed(2)
    r = _run("claim", "--run", "r1", "--limit", "1")
    assert r.exit_code == 0, r.output
    (c,) = _json(r)["claimed"]
    assert c["claimed_by"] == "r1" and c["claim_token"]
    r = _run("record-outcome", str(c["id"]), "--token", c["claim_token"],
             "--outcome", "walkforward_refuted", "--reason", "min sharpe < 0")
    assert r.exit_code == 0 and _json(r)["status"] == "refuted"
    r = _run("record-outcome", str(c["id"]), "--token", c["claim_token"],
             "--outcome", "run_error", "--reason", "again")
    assert r.exit_code == 1 and _json(r)["code"] == "claim_token_mismatch"
    d = _json(_run("depth"))
    assert d["open_unclaimed"] == 1 and d["refill_at"] == 72 and d["below_refill"] is True


def test_claim_empty_pool_is_ok_with_empty_list():
    r = _run("claim", "--run", "r1", "--limit", "3")
    assert r.exit_code == 0 and _json(r)["claimed"] == []


def test_import_from_scratch_db(tmp_path):
    _seed(1)
    auth = tmp_path / "r.db"
    scratch = tmp_path / "scratch.db"
    src, dst = sqlite3.connect(auth), sqlite3.connect(scratch)
    with dst:
        src.backup(dst)
    src.close()
    dst.close()
    # add one more idea to SCRATCH via the CLI pointed at it
    env = dict(os.environ, ALGUA_DB_PATH=str(scratch))
    r = runner.invoke(app, ["research", "idea", "add", "--title", "scratch only idea words",
                            "--hypothesis", "scratch hyp words", "--source-type", "inspiration",
                            "--category", "momentum", "--market", "us_equities", "--horizon",
                            "daily", "--falsification", "f", "--inspiration", "j|blog/y|rare"],
                       env=env)
    assert r.exit_code == 0, r.output
    critic = tmp_path / "critic.jsonl"
    critic.write_text(json.dumps({"title": "bad", "hypothesis": "beta",
                                  "reason_kind": "beta_in_disguise",
                                  "reason": "long-short beta in disguise"}) + "\n")
    r = _run("import", "--from", str(scratch), "--run", "leap-1", "--max", "6",
             "--seeded-max-id", "1", "--critic-file", str(critic))
    assert r.exit_code == 0, r.output
    body = _json(r)
    assert len(body["imported"]) == 1 and body["critic_rows"] == 1
    assert body["critic_errors"] == 0
    assert len(_json(_run("list"))) == 2


def test_import_bad_critic_lines_are_counted_never_fatal(tmp_path):
    # The critic file is agent output. A malformed LINE and a row with no usable reason used to
    # raise out of `import` AFTER the ideas had been committed — an exit-1 for a run whose
    # authoritative work had actually succeeded. Both are now counted, and the run still exits 0.
    _seed(1)
    auth = tmp_path / "r.db"
    scratch = tmp_path / "scratch.db"
    src, dst = sqlite3.connect(auth), sqlite3.connect(scratch)
    with dst:
        src.backup(dst)
    src.close()
    dst.close()
    critic = tmp_path / "critic.jsonl"
    critic.write_text("\n".join([
        json.dumps({"title": "a", "hypothesis": "h1", "reason_kind": "beta_in_disguise",
                    "reason": "long-short beta in disguise"}),
        "{not json at all",
        json.dumps({"title": "c", "hypothesis": "h3", "reason_kind": "no_mechanism",
                    "reason": "   "}),
        json.dumps({"title": "d", "hypothesis": "h4", "reason_kind": "no_mechanism",
                    "reason": "states no mechanism at all"}),
    ]) + "\n")

    r = _run("import", "--from", str(scratch), "--run", "leap-1", "--max", "6",
             "--seeded-max-id", "1", "--critic-file", str(critic))

    assert r.exit_code == 0, r.output
    body = _json(r)
    assert body["ok"] is True
    assert body["critic_rows"] == 2 and body["critic_errors"] == 2


def test_import_from_non_sqlite_file_fails_closed_without_leaking_connection(tmp_path):
    # A corrupt/non-sqlite --from target must fail closed through the JSON envelope (never a raw
    # traceback), and must not leak the scratch connection: `migrate(scratch)` has to run INSIDE
    # the try/finally that closes it (review fix round 1 — it previously ran before the try).
    bogus = tmp_path / "not_a_db.txt"
    bogus.write_bytes(b"this is not a sqlite file, just some text bytes")
    r = _run("import", "--from", str(bogus), "--run", "leap-1", "--max", "6")
    assert r.exit_code == 1
    body = _json(r)
    assert body["ok"] is False
    assert "Traceback" not in r.output


def test_scorecard_and_refuted_read_paths():
    _seed(1)
    (c,) = _json(_run("claim", "--run", "r1", "--limit", "1"))["claimed"]
    _run("record-outcome", str(c["id"]), "--token", c["claim_token"], "--outcome",
         "integrity_fail", "--reason", "pit universe missing")
    sc = _json(_run("scorecard", "--days", "30"))
    assert sc["by_venue"]["blog/x"]["n"] == 1
    ref = _json(_run("refuted", "--limit", "5"))
    assert ref[0]["reason"] == "pit universe missing"
