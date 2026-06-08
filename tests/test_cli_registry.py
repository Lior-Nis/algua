import json
import sqlite3

import pytest
from typer.testing import CliRunner

from algua.cli.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _tmp_db(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))


def _json(result):
    assert result.exit_code == 0, result.stdout
    return json.loads(result.stdout)


def test_add_and_list():
    added = _json(runner.invoke(app, ["registry", "add", "alpha"]))
    assert added["ok"] is True  # success envelope discriminator
    listed = _json(runner.invoke(app, ["registry", "list"]))
    # `registry list` is the documented exception: a bare JSON array (collection), not an envelope.
    assert isinstance(listed, list)
    assert [s["name"] for s in listed] == ["alpha"]
    assert listed[0]["stage"] == "idea"


def test_transition_legal():
    runner.invoke(app, ["registry", "add", "alpha"])
    out = _json(runner.invoke(
        app, ["registry", "transition", "alpha", "--to", "backtested",
              "--actor", "agent", "--reason", "ran"]))
    assert out["stage"] == "backtested"


def test_transition_illegal_exits_nonzero():
    runner.invoke(app, ["registry", "add", "alpha"])
    result = runner.invoke(
        app, ["registry", "transition", "alpha", "--to", "live", "--actor", "agent"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False


def _advance_to_paper(strategy: str) -> None:
    runner.invoke(app, ["registry", "add", strategy])
    for stage in ("backtested", "shortlisted", "paper"):
        runner.invoke(app, ["registry", "transition", strategy,
                            "--to", stage, "--actor", "agent"])


def _make_key(tmp_path, name="id"):
    import subprocess
    key = tmp_path / name
    subprocess.run(["ssh-keygen", "-t", "ed25519", "-N", "", "-C", "test", "-f", str(key)],
                   check=True, capture_output=True)
    return key, (tmp_path / f"{name}.pub").read_text().strip()


def _allowed_signers_file(tmp_path, principal, pub):
    keytype, keyblob = pub.split()[0], pub.split()[1]
    path = tmp_path / "allowed_signers"
    path.write_text(f'{principal} namespaces="algua-go-live" {keytype} {keyblob}\n')
    return path


def _sign_file(key, content_path):
    import subprocess
    subprocess.run(["ssh-keygen", "-Y", "sign", "-n", "algua-go-live", "-f", str(key),
                    str(content_path)],
                   check=True, capture_output=True)
    return content_path.parent / (content_path.name + ".sig")


def test_full_path_to_live_signed_ceremony(tmp_path, monkeypatch):
    """Two-step signed go-live: challenge then signature."""
    strategy = "cross_sectional_momentum"
    _advance_to_paper(strategy)

    key, pub = _make_key(tmp_path)
    signers = _allowed_signers_file(tmp_path, "lior", pub)
    monkeypatch.setattr("algua.cli.registry_cmd.ALLOWED_SIGNERS_PATH", signers)

    # Allocate capital so the go-live guard is satisfied (guard: requires allocation)
    monkeypatch.setattr("algua.cli.live_cmd._live_account_equity", lambda: 100_000.0)
    assert runner.invoke(app, ["live", "allocate", strategy, "--capital", "10000"]).exit_code == 0

    # Step 1: transition --to live with no --signature -> challenge issued, stage stays paper
    result1 = runner.invoke(app, ["registry", "transition", strategy, "--to", "live",
                                  "--actor", "human"])
    assert result1.exit_code == 0, result1.stdout
    out1 = json.loads(result1.stdout)
    assert out1["ok"] is True
    assert out1["action"] == "go_live_challenge"
    assert "challenge" in out1

    show_after_challenge = json.loads(
        runner.invoke(app, ["registry", "show", strategy]).stdout)
    assert show_after_challenge["stage"] == "paper", "stage must still be paper after step-1"

    # Sign the challenge value
    challenge_file = tmp_path / "challenge.txt"
    challenge_file.write_text(out1["challenge"])
    sig_file = _sign_file(key, challenge_file)

    # Step 2: transition --to live --signature <file>.sig -> accepted, stage becomes live
    result2 = runner.invoke(app, ["registry", "transition", strategy, "--to", "live",
                                  "--actor", "human", "--signature", str(sig_file)])
    assert result2.exit_code == 0, result2.stdout
    out2 = json.loads(result2.stdout)
    assert out2["ok"] is True
    assert out2["stage"] == "live"

    # Replay: same .sig again must NOT succeed (challenge consumed / strategy already live)
    result3 = runner.invoke(app, ["registry", "transition", strategy, "--to", "live",
                                  "--actor", "human", "--signature", str(sig_file)])
    assert result3.exit_code != 0 or json.loads(result3.stdout).get("ok") is False, \
        "replayed signature must not produce a fresh go-live"


def test_go_live_without_signature_stays_paper(tmp_path, monkeypatch):
    """Step-1 alone (no --signature) returns challenge and leaves stage paper."""
    strategy = "cross_sectional_momentum"
    _advance_to_paper(strategy)

    key, pub = _make_key(tmp_path)
    signers = _allowed_signers_file(tmp_path, "lior", pub)
    monkeypatch.setattr("algua.cli.registry_cmd.ALLOWED_SIGNERS_PATH", signers)

    # Allocate capital so the go-live guard is satisfied (guard: requires allocation)
    monkeypatch.setattr("algua.cli.live_cmd._live_account_equity", lambda: 100_000.0)
    assert runner.invoke(app, ["live", "allocate", strategy, "--capital", "10000"]).exit_code == 0

    result = runner.invoke(app, ["registry", "transition", strategy, "--to", "live",
                                 "--actor", "human"])
    assert result.exit_code == 0, result.stdout
    out = json.loads(result.stdout)
    assert out["action"] == "go_live_challenge"
    assert "challenge" in out

    show = json.loads(runner.invoke(app, ["registry", "show", strategy]).stdout)
    assert show["stage"] == "paper"


def test_go_live_with_signature_but_no_pending_challenge(tmp_path, monkeypatch):
    """Presenting a signature without a prior challenge must fail."""

    strategy = "cross_sectional_momentum"
    _advance_to_paper(strategy)

    key, pub = _make_key(tmp_path)
    signers = _allowed_signers_file(tmp_path, "lior", pub)
    monkeypatch.setattr("algua.cli.registry_cmd.ALLOWED_SIGNERS_PATH", signers)

    # Sign an arbitrary payload (no challenge was issued)
    arb = tmp_path / "arbitrary.txt"
    arb.write_text("not a real challenge")
    sig_file = _sign_file(key, arb)

    result = runner.invoke(app, ["registry", "transition", strategy, "--to", "live",
                                 "--actor", "human", "--signature", str(sig_file)])
    assert result.exit_code != 0 or json.loads(result.stdout).get("ok") is False
    show = json.loads(runner.invoke(app, ["registry", "show", strategy]).stdout)
    assert show["stage"] == "paper"


def test_unknown_strategy_emits_json_error():
    result = runner.invoke(app, ["registry", "show", "ghost"])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert "ghost" in payload["error"]


def test_invalid_stage_value_emits_json_error():
    runner.invoke(app, ["registry", "add", "alpha"])
    result = runner.invoke(
        app, ["registry", "transition", "alpha", "--to", "bogus", "--actor", "agent"])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False


def test_duplicate_add_emits_json_error():
    runner.invoke(app, ["registry", "add", "alpha"])
    result = runner.invoke(app, ["registry", "add", "alpha"])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False


def test_registry_command_closes_connection(monkeypatch, tmp_path):
    # The connect+migrate+close lifecycle now lives in cli._common.registry_conn (the single
    # shared idiom), so the connection factory is patched there.
    from algua.cli import _common

    closed = []

    class TrackingConnection(sqlite3.Connection):
        def close(self) -> None:
            closed.append(True)
            super().close()

    def connect_tracking(_db_path):
        conn = sqlite3.connect(tmp_path / "tracked.db", factory=TrackingConnection)
        conn.row_factory = sqlite3.Row
        return conn

    monkeypatch.setattr(_common, "connect", connect_tracking)

    result = runner.invoke(app, ["registry", "add", "alpha"])

    assert result.exit_code == 0, result.stdout
    assert closed == [True]


def test_enroll_approver_appends_line(tmp_path, monkeypatch):
    import json as _json

    signers = tmp_path / "allowed_signers"
    signers.write_text("# header\n")
    monkeypatch.setattr("algua.cli.registry_cmd.ALLOWED_SIGNERS_PATH", signers)
    r = runner.invoke(app, ["registry", "enroll-approver", "--name", "lior",
                            "--pubkey", "ssh-ed25519 AAAAC3NzaC1lZDI1 lior@dev"])
    assert r.exit_code == 0, r.stdout
    assert _json.loads(r.stdout)["ok"] is True
    body = signers.read_text()
    assert 'lior namespaces="algua-go-live" ssh-ed25519 AAAAC3NzaC1lZDI1' in body
    r2 = runner.invoke(app, ["registry", "enroll-approver", "--name", "x",
                             "--pubkey", "ssh-ed25519 AAAAC3NzaC1lZDI1 other"])
    assert r2.exit_code == 1  # duplicate pubkey rejected


def test_enroll_approver_rejects_bad_principal(tmp_path, monkeypatch):
    import json as _json
    signers = tmp_path / "allowed_signers"
    signers.write_text("# header\n")
    monkeypatch.setattr("algua.cli.registry_cmd.ALLOWED_SIGNERS_PATH", signers)
    # a --name with whitespace/newline must be rejected (no allowed_signers line injection)
    r = runner.invoke(app, ["registry", "enroll-approver", "--name", "bad name",
                            "--pubkey", "ssh-ed25519 AAAAaaa x"])
    assert r.exit_code == 1 and _json.loads(r.stdout)["ok"] is False
    r2 = runner.invoke(app, ["registry", "enroll-approver", "--name", "evil\nattacker",
                             "--pubkey", "ssh-ed25519 BBBBbbb x"])
    assert r2.exit_code == 1
    assert "namespaces" not in signers.read_text().replace("# header", "")  # nothing enrolled


# ---------------------------------------------------------------------------
# Go-live guard helpers (Task 5: requires allocation + ≤1 live strategy)
# ---------------------------------------------------------------------------

def _seed_paper(monkeypatch, tmp_path, name: str) -> str:
    """Register a strategy and advance it to the paper stage; return its name."""
    _advance_to_paper(name)
    return name


def _force_live(monkeypatch, tmp_path, name: str) -> None:
    """Register a strategy and force-set its stage to 'live' directly in the DB."""
    from contextlib import closing

    from algua.config.settings import get_settings
    from algua.registry.db import connect, migrate
    runner.invoke(app, ["registry", "add", name])
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        conn.execute("UPDATE strategies SET stage='live' WHERE name=?", (name,))
        conn.commit()


def _allocate(monkeypatch, tmp_path, name: str, capital: float) -> None:
    """Set a live allocation for the named strategy via the CLI (monkeypatching equity read)."""
    monkeypatch.setattr("algua.cli.live_cmd._live_account_equity", lambda: 1_000_000.0)
    r = runner.invoke(app, ["live", "allocate", name, "--capital", str(capital)])
    assert r.exit_code == 0, f"_allocate failed: {r.stdout}"


def test_go_live_requires_allocation(monkeypatch, tmp_path):
    # a strategy at paper with NO allocation cannot be issued a go-live challenge
    name = _seed_paper(monkeypatch, tmp_path, "s1")
    r = runner.invoke(app, ["registry", "transition", name, "--to", "live", "--actor", "human"])
    assert r.exit_code == 1 and "allocation" in r.stdout.lower()


def test_list_emits_metadata_fields():
    runner.invoke(app, ["registry", "add", "alpha"])
    out = _json(runner.invoke(app, ["registry", "list"]))
    assert out[0]["author"] == "agent"
    assert out[0]["hypothesis_status"] == "untested"
    assert out[0]["tags"] == []
    assert out[0]["family"] is None


def test_show_emits_metadata_fields():
    runner.invoke(app, ["registry", "add", "alpha"])
    out = _json(runner.invoke(app, ["registry", "show", "alpha"]))
    assert out["author"] == "agent"
    assert out["hypothesis_status"] == "untested"
    assert out["tags"] == []


def test_add_with_metadata_flags():
    out = _json(runner.invoke(app, [
        "registry", "add", "mr1", "--family", "mean-reversion",
        "--tag", "slow", "--tag", "carry", "--author", "human",
        "--hypothesis-status", "supported", "--description", "desc",
    ]))
    assert out["ok"] is True
    rec = _json(runner.invoke(app, ["registry", "show", "mr1"]))
    assert rec["family"] == "mean-reversion"
    assert rec["tags"] == ["carry", "slow"]
    assert rec["author"] == "human"
    assert rec["hypothesis_status"] == "supported"


def test_invalid_family_slug_emits_json_error():
    runner.invoke(app, ["registry", "add", "alpha"])
    result = runner.invoke(app, ["registry", "add", "beta", "--family", "Bad_Family"])
    assert result.exit_code != 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is False

    result2 = runner.invoke(app, ["registry", "set", "alpha", "--family", "-bad"])
    assert result2.exit_code != 0
    payload2 = json.loads(result2.stdout)
    assert payload2["ok"] is False


def test_set_changes_metadata_and_reports_before_after(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_KNOWLEDGE_DIR", str(tmp_path / "vault"))
    runner.invoke(app, ["registry", "add", "a", "--hypothesis-status", "untested"])
    out = _json(runner.invoke(app, [
        "registry", "set", "a", "--hypothesis-status", "supported", "--add-tag", "carry",
    ]))
    assert out["changed"]["hypothesis_status"] == {"before": "untested", "after": "supported"}
    rec = _json(runner.invoke(app, ["registry", "show", "a"]))
    assert rec["hypothesis_status"] == "supported"
    assert rec["tags"] == ["carry"]


def test_list_filters_compose():
    runner.invoke(app, ["registry", "add", "a", "--family", "mean-reversion", "--tag", "slow"])
    runner.invoke(app, ["registry", "add", "b", "--family", "momentum", "--tag", "fast"])
    out = _json(runner.invoke(app, ["registry", "list", "--family", "mean-reversion"]))
    assert [r["name"] for r in out] == ["a"]
    out2 = _json(runner.invoke(app, ["registry", "list", "--tag", "fast"]))
    assert [r["name"] for r in out2] == ["b"]


def test_go_live_allows_second_live_strategy_with_allocation(monkeypatch, tmp_path):
    # one strategy already live; a SECOND with an allocation now reaches the go-live challenge
    from algua.registry.repository import ArtifactIdentity
    _force_live(monkeypatch, tmp_path, "already")
    name = _seed_paper(monkeypatch, tmp_path, "s2")
    _allocate(monkeypatch, tmp_path, name, 1000.0)
    # patch compute_artifact_hashes so s2 (no real module) doesn't raise StrategyNotFound
    monkeypatch.setattr(
        "algua.cli.registry_cmd.compute_artifact_hashes",
        lambda n: ArtifactIdentity(code_hash="aabb", config_hash="ccdd", dependency_hash="eeff"),
    )
    r = runner.invoke(app, ["registry", "transition", name, "--to", "live", "--actor", "human"])
    assert r.exit_code == 0  # a challenge is issued (no ≤1-live refusal)
    assert json.loads(r.stdout)["action"] == "go_live_challenge"
