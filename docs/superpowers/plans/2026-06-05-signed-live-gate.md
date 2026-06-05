# Signed `paper → live` Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Going live requires an Ed25519 SSH signature over an artifact-bound, single-use challenge; the private key never touches the computer, so an autonomous agent can never promote a strategy to `live`.

**Architecture:** A `live_challenges` table (schema bump) + an `algua/registry/live_gate.py` module (challenge build/issue/consume + `ssh-keygen`-based signature verify). `registry transition --to live` becomes two-step: no `--signature` issues a challenge; with `--signature` it verifies + consumes + transitions. Approver **public** keys live in a CODEOWNERS-protected `approvers/allowed_signers` file. The honor-system `registry approve` is removed.

**Tech Stack:** Python 3.12, sqlite3, Typer, `ssh-keygen` (system binary, shelled out — no new pip dep), pytest, ruff, mypy, import-linter. Spec: `docs/superpowers/specs/2026-06-05-signed-live-gate-design.md`.

---

## File structure

| File | Responsibility |
|---|---|
| `algua/registry/db.py` (modify) | `live_challenges` table; `SCHEMA_VERSION` 6 → 7. |
| `algua/registry/live_gate.py` (new) | `build_challenge`, `issue_challenge`, `verify_and_consume`, `verify_signature`, `SignatureError`. |
| `approvers/allowed_signers` (new) | The CODEOWNERS-protected trust anchor (enrolled approver pubkeys). |
| `CODEOWNERS` (modify) | Protect `approvers/`, `live_gate.py`, `transitions.py`. |
| `algua/cli/registry_cmd.py` (modify) | `enroll-approver`; two-step `transition --to live`; remove `approve`. |
| docs (`CLAUDE.md`, `docs/agent/operating.md`, `docs/agent/research-lifecycle.md`) | replace `registry approve` with the signed flow. |

---

### Task 1: `live_challenges` table (schema v7) + challenge persistence

**Files:** Modify `algua/registry/db.py`; Create `algua/registry/live_gate.py`; Test `tests/test_paper_db.py`, `tests/test_live_gate.py` (new).

Context: `db.py` has `SCHEMA_VERSION = 6` + a `_SCHEMA` string. `tests/test_paper_db.py` has `_tables`/`connect`/`migrate`. The version-assertion test asserts the `SCHEMA_VERSION` constant (no number edit needed). `now` is injected for testability throughout the codebase.

- [ ] **Step 1: Failing tests.** Create `tests/test_live_gate.py`:
```python
from datetime import UTC, datetime, timedelta

from algua.registry import live_gate
from algua.registry.db import connect, migrate


def _conn(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    conn.execute("INSERT INTO strategies(id, name, stage, created_at, updated_at) "
                 "VALUES (1, 's', 'paper', '2026-01-01', '2026-01-01')")
    conn.commit()
    return conn


def test_build_challenge_is_deterministic():
    a = live_gate.build_challenge("s", 1, "ch", "cfg", "nonce123", "2026-06-05T00:10:00+00:00")
    b = live_gate.build_challenge("s", 1, "ch", "cfg", "nonce123", "2026-06-05T00:10:00+00:00")
    assert a == b and "nonce=nonce123" in a and a.startswith("algua-go-live")


def test_issue_then_find_then_consume(tmp_path):
    conn = _conn(tmp_path)
    now = datetime(2026, 6, 5, tzinfo=UTC)
    issued = live_gate.issue_challenge(conn, 1, "s", "ch", "cfg", now=now)
    assert "nonce" in issued and "challenge" in issued and "expires_at" in issued
    row = live_gate.find_pending_challenge(conn, 1, "ch", "cfg", now=now)
    assert row is not None and row["nonce"] == issued["nonce"]
    assert live_gate.consume_challenge(conn, issued["nonce"], now=now) is True
    assert live_gate.consume_challenge(conn, issued["nonce"], now=now) is False  # single-use
    assert live_gate.find_pending_challenge(conn, 1, "ch", "cfg", now=now) is None  # consumed


def test_find_pending_rejects_expired_and_wrong_hash(tmp_path):
    conn = _conn(tmp_path)
    now = datetime(2026, 6, 5, tzinfo=UTC)
    live_gate.issue_challenge(conn, 1, "s", "ch", "cfg", now=now)
    later = now + timedelta(hours=1)
    assert live_gate.find_pending_challenge(conn, 1, "ch", "cfg", now=later) is None  # expired
    assert live_gate.find_pending_challenge(conn, 1, "DIFFERENT", "cfg", now=now) is None
```
Append to `tests/test_paper_db.py`:
```python
def test_migrate_creates_live_challenges_table(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    assert "live_challenges" in _tables(conn)
```

- [ ] **Step 2: Run** `uv run pytest tests/test_live_gate.py tests/test_paper_db.py -q` → FAIL.

- [ ] **Step 3: Table + bump.** In `db.py`: `SCHEMA_VERSION = 6` → `7`; append to `_SCHEMA`:
```sql
CREATE TABLE IF NOT EXISTS live_challenges (
    nonce        TEXT PRIMARY KEY,
    strategy_id  INTEGER NOT NULL REFERENCES strategies(id),
    code_hash    TEXT NOT NULL,
    config_hash  TEXT NOT NULL,
    issued_at    TEXT NOT NULL,
    expires_at   TEXT NOT NULL,
    consumed_at  TEXT
);
```

- [ ] **Step 4: Create `algua/registry/live_gate.py`** (challenge half only; signature half is Task 2):
```python
from __future__ import annotations

import secrets
import sqlite3
from datetime import UTC, datetime, timedelta

_NAMESPACE = "algua-go-live"
_TTL = timedelta(minutes=10)


def _now() -> datetime:
    return datetime.now(UTC)


def build_challenge(strategy: str, strategy_id: int, code_hash: str, config_hash: str,
                    nonce: str, expires_at: str) -> str:
    """The exact bytes the operator signs. One definition, used to both issue and verify so the
    two can never drift. Binds the go-live to a specific strategy + artifact + single-use nonce."""
    return (
        f"{_NAMESPACE}\nstrategy={strategy}\nstrategy_id={strategy_id}\n"
        f"code_hash={code_hash}\nconfig_hash={config_hash}\nnonce={nonce}\nexpires_at={expires_at}"
    )


def issue_challenge(conn: sqlite3.Connection, strategy_id: int, strategy: str, code_hash: str,
                    config_hash: str, *, now: datetime | None = None) -> dict[str, str]:
    """Create + persist a pending go-live challenge; return {nonce, challenge, expires_at}."""
    now = now or _now()
    nonce = secrets.token_hex(32)
    expires_at = (now + _TTL).isoformat()
    conn.execute(
        "INSERT INTO live_challenges(nonce, strategy_id, code_hash, config_hash, issued_at, "
        "expires_at, consumed_at) VALUES (?,?,?,?,?,?,NULL)",
        (nonce, strategy_id, code_hash, config_hash, now.isoformat(), expires_at),
    )
    conn.commit()
    return {"nonce": nonce, "expires_at": expires_at,
            "challenge": build_challenge(strategy, strategy_id, code_hash, config_hash, nonce,
                                         expires_at)}


def find_pending_challenge(conn: sqlite3.Connection, strategy_id: int, code_hash: str,
                           config_hash: str, *, now: datetime | None = None) -> sqlite3.Row | None:
    """The newest unconsumed, unexpired challenge matching the strategy + recomputed hashes."""
    now = now or _now()
    return conn.execute(
        "SELECT * FROM live_challenges WHERE strategy_id=? AND code_hash=? AND config_hash=? "
        "AND consumed_at IS NULL AND expires_at > ? ORDER BY issued_at DESC LIMIT 1",
        (strategy_id, code_hash, config_hash, now.isoformat()),
    ).fetchone()


def consume_challenge(conn: sqlite3.Connection, nonce: str, *,
                      now: datetime | None = None) -> bool:
    """Mark a challenge consumed (single-use). Returns False if already consumed/missing."""
    now = now or _now()
    cur = conn.execute(
        "UPDATE live_challenges SET consumed_at=? WHERE nonce=? AND consumed_at IS NULL",
        (now.isoformat(), nonce),
    )
    conn.commit()
    return cur.rowcount > 0
```

- [ ] **Step 5: Run** `uv run pytest tests/test_live_gate.py tests/test_paper_db.py tests/test_registry_db.py -q` → PASS.

- [ ] **Step 6: Gate + commit** — `uv run ruff check . && uv run mypy algua && uv run pytest tests/test_live_gate.py tests/test_paper_db.py tests/test_registry_db.py -q`
```bash
git add algua/registry/db.py algua/registry/live_gate.py tests/test_live_gate.py tests/test_paper_db.py
git commit -m "feat(registry): live_challenges table + challenge persistence (schema v7)"
```

---

### Task 2: SSH signature verification

**Files:** Modify `algua/registry/live_gate.py`; Test `tests/test_live_gate.py`.

Context: verification shells to `ssh-keygen -Y find-principals` (is the signer enrolled?) then `ssh-keygen -Y verify -I <principal>`, both reading the signed payload on stdin and taking the signature as a file (`-s`). `ssh-keygen` is at `/usr/bin/ssh-keygen`.

- [ ] **Step 1: Failing tests** — append to `tests/test_live_gate.py`:
```python
import subprocess


def _make_key(tmp_path, name="id"):
    key = tmp_path / name
    subprocess.run(["ssh-keygen", "-t", "ed25519", "-N", "", "-C", "test", "-f", str(key)],
                   check=True, capture_output=True)
    return key, (tmp_path / f"{name}.pub").read_text().strip()


def _allowed_signers(tmp_path, principal, pub):
    # allowed_signers line: "<principal> namespaces=... <keytype> <keyblob>"
    keytype, keyblob = pub.split()[0], pub.split()[1]
    path = tmp_path / "allowed_signers"
    path.write_text(f'{principal} namespaces="algua-go-live" {keytype} {keyblob}\n')
    return path


def _sign(key, payload, tmp_path):
    data = tmp_path / "payload.txt"
    data.write_text(payload)
    subprocess.run(["ssh-keygen", "-Y", "sign", "-n", "algua-go-live", "-f", str(key), str(data)],
                   check=True, capture_output=True)
    return (tmp_path / "payload.txt.sig").read_bytes()


def test_verify_signature_happy_path(tmp_path):
    key, pub = _make_key(tmp_path)
    signers = _allowed_signers(tmp_path, "lior", pub)
    payload = "algua-go-live\nstrategy=s\nnonce=abc"
    sig = _sign(key, payload, tmp_path)
    assert live_gate.verify_signature(signers, payload, sig) == "lior"


def test_verify_signature_rejects_unenrolled_key(tmp_path):
    key, _pub = _make_key(tmp_path, "mine")
    _other_key, other_pub = _make_key(tmp_path, "other")
    signers = _allowed_signers(tmp_path, "lior", other_pub)  # enroll a DIFFERENT key
    payload = "algua-go-live\nx=1"
    sig = _sign(key, payload, tmp_path)
    assert live_gate.verify_signature(signers, payload, sig) is None


def test_verify_signature_rejects_tampered_payload(tmp_path):
    key, pub = _make_key(tmp_path)
    signers = _allowed_signers(tmp_path, "lior", pub)
    sig = _sign(key, "original-payload", tmp_path)
    assert live_gate.verify_signature(signers, "DIFFERENT-payload", sig) is None
```

- [ ] **Step 2: Run** `uv run pytest tests/test_live_gate.py -q -k verify_signature` → FAIL.

- [ ] **Step 3: Implement** — add to `algua/registry/live_gate.py` (add `import subprocess`, `import tempfile`, `from pathlib import Path` to the imports):
```python
class SignatureError(RuntimeError):
    """ssh-keygen is unavailable or failed in an unexpected way (not a plain bad signature)."""


def verify_signature(allowed_signers_path: Path, payload: str, signature: bytes) -> str | None:
    """Verify an SSH signature over `payload` against the enrolled keys in `allowed_signers_path`.
    Returns the matched principal on success, or None if the signature is invalid / the signer is
    not enrolled. Raises SignatureError only when ssh-keygen itself can't run."""
    with tempfile.NamedTemporaryFile("wb", suffix=".sig", delete=True) as sigf:
        sigf.write(signature)
        sigf.flush()
        data = payload.encode()
        try:
            found = subprocess.run(
                ["ssh-keygen", "-Y", "find-principals", "-f", str(allowed_signers_path),
                 "-s", sigf.name],
                input=data, capture_output=True,
            )
        except FileNotFoundError as exc:  # ssh-keygen not installed
            raise SignatureError("ssh-keygen not found on PATH") from exc
        if found.returncode != 0:
            return None  # signer not enrolled (or malformed signature)
        principal = found.stdout.decode().splitlines()[0].strip() if found.stdout.strip() else ""
        if not principal:
            return None
        verified = subprocess.run(
            ["ssh-keygen", "-Y", "verify", "-f", str(allowed_signers_path), "-I", principal,
             "-n", _NAMESPACE, "-s", sigf.name],
            input=data, capture_output=True,
        )
        return principal if verified.returncode == 0 else None


def verify_and_consume(conn: sqlite3.Connection, strategy: str, strategy_id: int, code_hash: str,
                       config_hash: str, signature: bytes, allowed_signers_path: Path, *,
                       now: datetime | None = None) -> str | None:
    """Find the pending challenge for this artifact, verify the signature over its exact payload,
    and atomically consume it. Returns the approver principal on success, else None."""
    now = now or _now()
    row = find_pending_challenge(conn, strategy_id, code_hash, config_hash, now=now)
    if row is None:
        return None
    payload = build_challenge(strategy, strategy_id, code_hash, config_hash, row["nonce"],
                              row["expires_at"])
    principal = verify_signature(allowed_signers_path, payload, signature)
    if principal is None:
        return None
    return principal if consume_challenge(conn, row["nonce"], now=now) else None
```

- [ ] **Step 4: Run** `uv run pytest tests/test_live_gate.py -q` → PASS.

- [ ] **Step 5: Gate + commit** — `uv run ruff check . && uv run mypy algua && uv run pytest tests/test_live_gate.py -q`
```bash
git add algua/registry/live_gate.py tests/test_live_gate.py
git commit -m "feat(registry): ssh-keygen signature verification for the live gate"
```

---

### Task 3: `approvers/allowed_signers` + `enroll-approver` + CODEOWNERS

**Files:** Create `approvers/allowed_signers`; Modify `CODEOWNERS`, `algua/cli/registry_cmd.py`; Test `tests/test_cli_registry.py`.

Context: `registry_cmd.py` imports `from algua.cli._common import ok, registry_conn`, `emit`, `json_errors`, `typer`. The default allowed-signers path is `approvers/allowed_signers` relative to the repo root (CWD when the CLI runs). `tests/test_cli_registry.py` uses a Typer `CliRunner`.

- [ ] **Step 1: Create the trust-anchor file** `approvers/allowed_signers` with a header (no keys yet):
```
# Enrolled go-live approver PUBLIC keys (SSH allowed_signers format, namespace algua-go-live).
# This file is the TRUST ANCHOR for the paper->live gate and is CODEOWNERS-protected: an agent
# may edit it in a worktree but the real gate uses the reviewed copy on main. One line per key:
#   <principal> namespaces="algua-go-live" ssh-ed25519 AAAA... comment
```

- [ ] **Step 2: Protect it in `CODEOWNERS`** — append:
```
/approvers/                     @Lior-Nis   # go-live approver public keys (the signature trust anchor)
/algua/registry/live_gate.py    @Lior-Nis   # the signed paper->live gate
/algua/registry/transitions.py  @Lior-Nis   # live-gate enforcement
```

- [ ] **Step 3: Failing test** — append to `tests/test_cli_registry.py`:
```python
def test_enroll_approver_appends_line(tmp_path, monkeypatch):
    import json as _json

    from algua.cli.main import app
    from typer.testing import CliRunner
    signers = tmp_path / "allowed_signers"
    signers.write_text("# header\n")
    monkeypatch.setattr("algua.cli.registry_cmd.ALLOWED_SIGNERS_PATH", signers)
    r = CliRunner().invoke(app, ["registry", "enroll-approver", "--name", "lior",
                                 "--pubkey", "ssh-ed25519 AAAAC3NzaC1lZDI1 lior@dev"])
    assert r.exit_code == 0, r.stdout
    assert _json.loads(r.stdout)["ok"] is True
    body = signers.read_text()
    assert 'lior namespaces="algua-go-live" ssh-ed25519 AAAAC3NzaC1lZDI1' in body
    # duplicate pubkey rejected
    r2 = CliRunner().invoke(app, ["registry", "enroll-approver", "--name", "x",
                                  "--pubkey", "ssh-ed25519 AAAAC3NzaC1lZDI1 other"])
    assert r2.exit_code == 1
```
(Use the test file's existing `runner`/imports if present rather than re-importing.)

- [ ] **Step 4: Implement** — in `algua/cli/registry_cmd.py`, add near the top:
```python
from pathlib import Path

ALLOWED_SIGNERS_PATH = Path("approvers/allowed_signers")
```
and the command:
```python
@registry_app.command("enroll-approver")
@json_errors(ValueError)
def enroll_approver(
    name: str = typer.Option(..., "--name", help="approver identity (the allowed_signers principal)"),
    pubkey: str = typer.Option(..., "--pubkey", help="the approver's SSH public key (ssh-ed25519 AAAA...)"),
) -> None:
    """Enroll a go-live approver PUBLIC key. The trust comes from committing this through code-owner
    review — the live gate uses the reviewed copy on main."""
    if not name.strip():
        raise ValueError("--name must not be empty")
    parts = pubkey.split()
    if len(parts) < 2 or not parts[0].startswith("ssh-"):
        raise ValueError("--pubkey must be an SSH public key, e.g. 'ssh-ed25519 AAAA... comment'")
    keytype, keyblob = parts[0], parts[1]
    existing = ALLOWED_SIGNERS_PATH.read_text() if ALLOWED_SIGNERS_PATH.exists() else ""
    if keyblob in existing:
        raise ValueError("that public key is already enrolled")
    line = f'{name} namespaces="algua-go-live" {keytype} {keyblob}\n'
    with ALLOWED_SIGNERS_PATH.open("a") as fh:
        fh.write(line)
    emit(ok({"enrolled": name, "keytype": keytype}))
```

- [ ] **Step 5: Run** `uv run pytest tests/test_cli_registry.py -q` → PASS.

- [ ] **Step 6: Gate + commit** — `uv run ruff check . && uv run mypy algua && uv run pytest tests/test_cli_registry.py -q`
```bash
git add approvers/allowed_signers CODEOWNERS algua/cli/registry_cmd.py tests/test_cli_registry.py
git commit -m "feat(registry): enroll-approver + CODEOWNERS-protected allowed_signers anchor"
```

---

### Task 4: two-step `transition --to live` + remove honor-system `approve`

**Files:** Modify `algua/cli/registry_cmd.py`; docs (`CLAUDE.md`, `docs/agent/operating.md`, `docs/agent/research-lifecycle.md`); Test `tests/test_cli_registry.py`.

Context: the current `transition` command calls `transition_strategy(repo, name, Stage(to), Actor(actor), reason)`. `transition_strategy(repo, name, to, actor, reason, approval_verifier=None)` runs `_validate_live_gate` for LIVE (requires `actor is HUMAN` + `approval_verifier(repo, sid, code_hash, config_hash) -> bool`). `compute_artifact_hashes(name)` and `record_approval(repo, name, approved_by)` live in `algua/registry/approvals.py`. `validate_transition(from, to)` raises `TransitionError` on a disallowed edge. The existing `test_full_path_to_live_with_approval` uses `registry approve` then `transition --to live` — it must become the signed flow.

- [ ] **Step 1: Rewrite the e2e test** — in `tests/test_cli_registry.py`, replace `test_full_path_to_live_with_approval` with the signed flow (reuse the `_make_key`/`_sign` helpers from `tests/test_live_gate.py` by importing them, or inline equivalents):
```python
def test_signed_go_live_end_to_end(tmp_path, monkeypatch):
    import json as _json
    import subprocess

    from typer.testing import CliRunner

    from algua.cli.main import app
    runner = CliRunner()
    # ... bring a strategy to 'paper' the way other tests in this file do (add + transitions) ...
    name = _to_paper_strategy(runner)  # use this file's existing helper/flow to reach paper

    # enroll a throwaway key as an approver
    key = tmp_path / "id"
    subprocess.run(["ssh-keygen", "-t", "ed25519", "-N", "", "-f", str(key)], check=True,
                   capture_output=True)
    pub = (tmp_path / "id.pub").read_text().strip()
    signers = tmp_path / "allowed_signers"
    signers.write_text(f'lior namespaces="algua-go-live" {pub.split()[0]} {pub.split()[1]}\n')
    monkeypatch.setattr("algua.cli.registry_cmd.ALLOWED_SIGNERS_PATH", signers)

    # step 1: issue challenge (no transition)
    out = runner.invoke(app, ["registry", "transition", name, "--to", "live", "--actor", "human"])
    issued = _json.loads(out.stdout)
    assert issued["action"] == "go_live_challenge" and "challenge" in issued
    show = runner.invoke(app, ["registry", "show", name])
    assert '"stage": "live"' not in show.stdout  # still paper

    # step 2: sign the challenge, transition
    payload = tmp_path / "p"
    payload.write_text(issued["challenge"])
    subprocess.run(["ssh-keygen", "-Y", "sign", "-n", "algua-go-live", "-f", str(key), str(payload)],
                   check=True, capture_output=True)
    out2 = runner.invoke(app, ["registry", "transition", name, "--to", "live", "--actor", "human",
                               "--signature", str(payload) + ".sig"])
    assert out2.exit_code == 0, out2.stdout
    assert _json.loads(out2.stdout)["stage"] == "live"

    # replay the same signature -> rejected (consumed)
    out3 = runner.invoke(app, ["registry", "transition", name, "--to", "paper", "--actor", "agent"])
    runner.invoke(app, ["registry", "transition", name, "--to", "live", "--actor", "human",
                        "--signature", str(payload) + ".sig"])
    # a fresh signature would be needed; the consumed challenge no longer matches
```
Also add a negative: `transition --to live` with `--signature` but no pending challenge → `{ok:false}`, stays paper. (Write `_to_paper_strategy` as a small local helper mirroring how the file's other tests reach `paper`, or reuse an existing one.)

- [ ] **Step 2: Run** `uv run pytest tests/test_cli_registry.py -q` → FAIL.

- [ ] **Step 3: Rewrite the `transition` command** in `algua/cli/registry_cmd.py`. Add imports: `from algua.registry import live_gate`, `from algua.registry.approvals import compute_artifact_hashes, record_approval`, `from algua.registry.live_gate import SignatureError`, `from algua.contracts.lifecycle import validate_transition` (and keep `Actor, Stage`). Replace the command:
```python
@registry_app.command("transition")
@json_errors(ValueError, LookupError, TransitionError, SignatureError)
def transition(
    name: str,
    to: str = typer.Option(..., "--to"),
    actor: str = typer.Option(..., "--actor"),
    reason: str = typer.Option(None, "--reason"),
    signature: str = typer.Option(None, "--signature",
                                  help="path to the SSH signature over the printed go-live challenge"),
) -> None:
    """Advance a strategy's lifecycle stage. Going live is a two-step signed ceremony: run with no
    --signature to print a challenge, sign it with your enrolled key, then re-run with --signature."""
    target = Stage(to)
    with registry_conn() as conn:
        repo = SqliteStrategyRepository(conn)
        if target is Stage.LIVE and signature is None:
            rec = repo.get(name)
            validate_transition(rec.stage, Stage.LIVE)  # reject non-paper before issuing
            code_hash, config_hash = compute_artifact_hashes(name)
            issued = live_gate.issue_challenge(conn, rec.id, name, code_hash, config_hash)
            emit(ok({
                "action": "go_live_challenge", "strategy": name, **issued,
                "instructions": ("sign the 'challenge' value with your enrolled key: "
                                 "ssh-keygen -Y sign -n algua-go-live -f <key> <file>; "
                                 "then re-run this command with --signature <file>.sig"),
            }))
            return

        verifier = None
        approver: dict[str, str] = {}
        if target is Stage.LIVE:
            sig_bytes = Path(signature).read_bytes()
            rec0 = repo.get(name)

            def _verify(_repo: object, sid: int, ch: str, cfg: str) -> bool:
                principal = live_gate.verify_and_consume(
                    conn, name, sid, ch, cfg, sig_bytes, ALLOWED_SIGNERS_PATH)
                if principal is None:
                    return False
                approver["id"] = principal
                return True

            verifier = _verify

        rec = transition_strategy(repo, name, target, Actor(actor), reason,
                                  approval_verifier=verifier)
        if target is Stage.LIVE:
            record_approval(repo, name, approver["id"])  # audit row, bound to the recomputed hashes
    emit(ok({"name": rec.name, "stage": rec.stage.value}))
```
(Note: `rec0` is unused above — drop it; it's only here as a reminder that `repo.get` may be called inside the gate. Keep the body lean.)

- [ ] **Step 4: Remove the honor-system `approve`** — delete the `@registry_app.command("approve")` function from `registry_cmd.py`. Keep the `record_approval` import (now used by `transition`).

- [ ] **Step 5: Update docs** — in `CLAUDE.md` (command surface), `docs/agent/operating.md`, and `docs/agent/research-lifecycle.md`, replace the `registry approve ...` line/flow with the signed two-step go-live (`transition --to live` → sign → `--signature`). Keep it short; mirror the spec's §5.

- [ ] **Step 6: Run** `uv run pytest tests/test_cli_registry.py tests/test_registry_approvals.py -q` → PASS (adjust any remaining `approve`-referencing assertions to the signed flow; `record_approval`/`has_valid_approval` unit tests in `test_registry_approvals.py` stay).

- [ ] **Step 7: Full gate + commit** — `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports` (all pass; contracts kept, 0 broken).
```bash
git add algua/cli/registry_cmd.py CLAUDE.md docs/agent/operating.md docs/agent/research-lifecycle.md tests/test_cli_registry.py
git commit -m "feat(cli): signed two-step paper->live transition; remove honor-system approve"
```

---

## Self-review notes

- **Spec coverage:** `live_challenges` + persistence (§4 → Task 1); `ssh-keygen` find-principals→verify + `verify_and_consume` (§5,§7 → Task 2); `approvers/allowed_signers` + CODEOWNERS + `enroll-approver` (§3 → Task 3); two-step issue/verify transition + removal of `approve` (§5,§6 → Task 4). Tests for the lifecycle, hermetic signature verify, e2e signed transition, replay/expiry/tamper/unenrolled negatives, enrollment (§8 → Tasks 1–4). The version-assertion test needs no edit (asserts the `SCHEMA_VERSION` constant). Live acceptance is documented in the spec (§8), not code.
- **Type consistency:** `build_challenge(strategy, strategy_id, code_hash, config_hash, nonce, expires_at)` is called identically in `issue_challenge` and `verify_and_consume`. `verify_signature(path, payload, signature) -> str|None` and `verify_and_consume(conn, strategy, strategy_id, code_hash, config_hash, signature, allowed_signers_path) -> str|None` are used as defined in Task 4's `_verify`. `ALLOWED_SIGNERS_PATH` (Task 3) is referenced by the transition command (Task 4) and is monkeypatchable in tests.
- **No placeholders:** the only "fill from existing helpers" note is reaching `paper` stage in the e2e test, which must mirror this test file's existing add+transition flow (the implementer reads the file). `record_approval(repo, name, approved_by)` matches the `algua/registry/approvals.py` signature.
- **Security invariant:** the trust anchor (`approvers/allowed_signers`) and the gate code (`live_gate.py`, `transitions.py`, `store.py`) are CODEOWNERS-protected, so an agent's worktree edits never reach the `main` code that runs go-live. The private key is never on the computer. Both are required for the wall to hold.
