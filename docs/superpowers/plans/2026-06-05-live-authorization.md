# Trade-Time Live Authorization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist the signed go-live evidence in a `live_authorizations` table and build `verify_live_authorization`, the trade-time primitive that re-verifies a human signature against the trust anchor before any live order.

**Architecture:** On a successful signed `transition --to live`, `verify_and_consume` also writes a `live_authorizations` row (challenge + base64 signature + principal + recomputed identity). The new `verify_live_authorization(conn, repo, name, allowed_signers_path)` recomputes the current artifact identity, finds an unrevoked matching row, and re-runs `ssh-keygen -Y verify` on the stored challenge+signature against the CURRENT `approvers/allowed_signers` — trusting nothing forgeable. No live venue, no live orders.

**Tech Stack:** Python 3.12, sqlite3, `ssh-keygen` (system binary), pytest, ruff, mypy, import-linter. Spec: `docs/superpowers/specs/2026-06-05-live-authorization-design.md`.

---

## File structure

| File | Responsibility |
|---|---|
| `algua/registry/db.py` (modify) | `live_authorizations` table; `SCHEMA_VERSION` 11 → 12. |
| `algua/registry/live_gate.py` (modify) | `verify_and_consume` writes the evidence row; new `verify_live_authorization` + `LiveAuthorizationError`; own `ALLOWED_SIGNERS_PATH`. |
| `algua/cli/registry_cmd.py` (modify) | import `ALLOWED_SIGNERS_PATH` from `live_gate` instead of defining it. |

---

### Task 1: `live_authorizations` table (schema v12) + write evidence at go-live

**Files:** Modify `algua/registry/db.py`, `algua/registry/live_gate.py`; Test `tests/test_paper_db.py`, `tests/test_live_gate.py`.

Context: `db.py` has `SCHEMA_VERSION = 11` + a `_SCHEMA` string; the version-assertion test asserts the constant. `algua/registry/live_gate.py` has `verify_and_consume(conn, strategy, strategy_id, code_hash, config_hash, dependency_hash, signature, allowed_signers_path, *, now=None) -> str | None` which already finds the pending challenge, rebuilds the `payload` (via `build_challenge`), `verify_signature`s it, and `consume_challenge`s the nonce. `tests/test_live_gate.py` has helpers `_conn(tmp_path)` (connect+migrate+insert strategy id=1 'paper'), `_make_key(tmp_path, name="id")`, `_allowed_signers(tmp_path, principal, pub)`, `_sign(key, payload, tmp_path)`.

- [ ] **Step 1: Failing tests.** Append to `tests/test_paper_db.py`:
```python
def test_migrate_creates_live_authorizations_table(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    assert "live_authorizations" in _tables(conn)
```
Append to `tests/test_live_gate.py`:
```python
def test_verify_and_consume_writes_live_authorization(tmp_path):
    import base64

    conn = _conn(tmp_path)
    now = datetime(2026, 6, 5, tzinfo=UTC)
    issued = live_gate.issue_challenge(conn, 1, "s", "ch", "cfg", "dep", now=now)
    key, pub = _make_key(tmp_path)
    signers = _allowed_signers(tmp_path, "lior", pub)
    sig = _sign(key, issued["challenge"], tmp_path)
    principal = live_gate.verify_and_consume(conn, "s", 1, "ch", "cfg", "dep", sig, signers, now=now)
    assert principal == "lior"
    row = conn.execute("SELECT * FROM live_authorizations WHERE strategy_id=1").fetchone()
    assert row is not None
    assert row["code_hash"] == "ch" and row["config_hash"] == "cfg" and row["dependency_hash"] == "dep"
    assert row["principal"] == "lior" and row["challenge"] == issued["challenge"]
    # the stored signature round-trips and re-verifies against the same anchor
    assert live_gate.verify_signature(signers, row["challenge"],
                                      base64.b64decode(row["signature"])) == "lior"


def test_verify_and_consume_no_authorization_on_bad_signature(tmp_path):
    conn = _conn(tmp_path)
    now = datetime(2026, 6, 5, tzinfo=UTC)
    issued = live_gate.issue_challenge(conn, 1, "s", "ch", "cfg", "dep", now=now)
    key, _pub = _make_key(tmp_path, "mine")
    _ok, other_pub = _make_key(tmp_path, "other")
    signers = _allowed_signers(tmp_path, "lior", other_pub)  # different key enrolled
    sig = _sign(key, issued["challenge"], tmp_path)
    assert live_gate.verify_and_consume(conn, "s", 1, "ch", "cfg", "dep", sig, signers, now=now) is None
    assert conn.execute("SELECT COUNT(*) FROM live_authorizations").fetchone()[0] == 0  # no row
```

- [ ] **Step 2: Run** `uv run pytest tests/test_paper_db.py tests/test_live_gate.py -q` → FAIL.

- [ ] **Step 3: Table + bump.** In `db.py`: `SCHEMA_VERSION = 11` → `12`; append to `_SCHEMA`:
```sql
CREATE TABLE IF NOT EXISTS live_authorizations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id     INTEGER NOT NULL REFERENCES strategies(id),
    code_hash       TEXT NOT NULL,
    config_hash     TEXT NOT NULL,
    dependency_hash TEXT,
    challenge       TEXT NOT NULL,
    signature       TEXT NOT NULL,
    principal       TEXT NOT NULL,
    authorized_at   TEXT NOT NULL,
    revoked_at      TEXT
);
```

- [ ] **Step 4: Write the evidence row.** In `algua/registry/live_gate.py`, add `import base64` to the imports, and change the tail of `verify_and_consume` from:
```python
    principal = verify_signature(allowed_signers_path, payload, signature)
    if principal is None:
        return None
    return principal if consume_challenge(conn, row["nonce"], now=now) else None
```
to:
```python
    principal = verify_signature(allowed_signers_path, payload, signature)
    if principal is None:
        return None
    if not consume_challenge(conn, row["nonce"], now=now):
        return None
    # Persist the durable, re-verifiable proof of this go-live: the exact signed challenge, the
    # signature, the approver, and the artifact identity. Trade-time live_authorization re-verifies
    # THIS against the trust anchor rather than trusting the agent-writable stage/approvals rows.
    conn.execute(
        "INSERT INTO live_authorizations(strategy_id, code_hash, config_hash, dependency_hash, "
        "challenge, signature, principal, authorized_at) VALUES (?,?,?,?,?,?,?,?)",
        (strategy_id, code_hash, config_hash, dependency_hash, payload,
         base64.b64encode(signature).decode(), principal, now.isoformat()),
    )
    conn.commit()
    return principal
```

- [ ] **Step 5: Run** `uv run pytest tests/test_paper_db.py tests/test_live_gate.py tests/test_registry_db.py -q` → PASS.

- [ ] **Step 6: Gate + commit** — `uv run ruff check . && uv run mypy algua && uv run pytest tests/test_paper_db.py tests/test_live_gate.py tests/test_registry_db.py -q`
```bash
git add algua/registry/db.py algua/registry/live_gate.py tests/test_paper_db.py tests/test_live_gate.py
git commit -m "feat(registry): live_authorizations table + persist go-live evidence (schema v12)"
```

---

### Task 2: `verify_live_authorization` primitive + shared trust anchor

**Files:** Modify `algua/registry/live_gate.py`, `algua/cli/registry_cmd.py`; Test `tests/test_live_gate.py`.

Context: `algua/cli/registry_cmd.py:21` defines `ALLOWED_SIGNERS_PATH = Path(__file__).resolve().parents[2] / "approvers" / "allowed_signers"` and uses it in `enroll-approver` and the `transition` verify branch. `algua/registry/repository.py` has `ArtifactIdentity(NamedTuple)` (`code_hash`, `config_hash`, `dependency_hash`) and `StrategyRecord` (`.id`, `.stage: Stage`). `algua/registry/approvals.py::compute_artifact_hashes(name) -> ArtifactIdentity`. `SqliteStrategyRepository(conn).get(name) -> StrategyRecord`. `Stage` is `algua.contracts.lifecycle.Stage`. `live_gate.py` already has `verify_signature` + `SignatureError` and (from Task 1) `import base64`. `live_gate.py` is in the registry layer — the CLI may import from it, not vice-versa (import-linter).

- [ ] **Step 1: Failing tests.** Append to `tests/test_live_gate.py` (these monkeypatch `compute_artifact_hashes` so the identity is controlled without a real strategy module):
```python
def _live_strategy(conn, stage="live"):
    conn.execute("UPDATE strategies SET stage=? WHERE id=1", (stage,))
    conn.commit()


def _seed_authorization(conn, tmp_path, *, code="ch", cfg="cfg", dep="dep", principal="lior"):
    # build a real signed authorization row for strategy_id=1
    key, pub = _make_key(tmp_path)
    signers = _allowed_signers(tmp_path, principal, pub)
    issued = live_gate.issue_challenge(conn, 1, "s", code, cfg, dep,
                                       now=datetime(2026, 6, 5, tzinfo=UTC))
    sig = _sign(key, issued["challenge"], tmp_path)
    live_gate.verify_and_consume(conn, "s", 1, code, cfg, dep, sig, signers,
                                 now=datetime(2026, 6, 5, tzinfo=UTC))
    return signers


def _identity(monkeypatch, code="ch", cfg="cfg", dep="dep"):
    from algua.registry.repository import ArtifactIdentity
    monkeypatch.setattr("algua.registry.approvals.compute_artifact_hashes",
                        lambda name: ArtifactIdentity(code, cfg, dep))


def test_verify_live_authorization_happy_path(tmp_path, monkeypatch):
    from algua.registry.store import SqliteStrategyRepository
    conn = _conn(tmp_path)
    signers = _seed_authorization(conn, tmp_path)
    _live_strategy(conn)
    _identity(monkeypatch)
    row = live_gate.verify_live_authorization(conn, SqliteStrategyRepository(conn), "s", signers)
    assert row["principal"] == "lior"


def test_verify_live_authorization_rejects_non_live(tmp_path, monkeypatch):
    import pytest

    from algua.registry.live_gate import LiveAuthorizationError
    from algua.registry.store import SqliteStrategyRepository
    conn = _conn(tmp_path)
    signers = _seed_authorization(conn, tmp_path)  # stage stays 'paper'
    _identity(monkeypatch)
    with pytest.raises(LiveAuthorizationError):
        live_gate.verify_live_authorization(conn, SqliteStrategyRepository(conn), "s", signers)


def test_verify_live_authorization_rejects_changed_code(tmp_path, monkeypatch):
    import pytest

    from algua.registry.live_gate import LiveAuthorizationError
    from algua.registry.store import SqliteStrategyRepository
    conn = _conn(tmp_path)
    signers = _seed_authorization(conn, tmp_path, code="ch")
    _live_strategy(conn)
    _identity(monkeypatch, code="CHANGED")  # current code != approved code
    with pytest.raises(LiveAuthorizationError):
        live_gate.verify_live_authorization(conn, SqliteStrategyRepository(conn), "s", signers)


def test_verify_live_authorization_rejects_revoked_or_unenrolled(tmp_path, monkeypatch):
    import pytest

    from algua.registry.live_gate import LiveAuthorizationError
    from algua.registry.store import SqliteStrategyRepository
    conn = _conn(tmp_path)
    signers = _seed_authorization(conn, tmp_path)
    _live_strategy(conn)
    _identity(monkeypatch)
    # revoked row -> rejected
    conn.execute("UPDATE live_authorizations SET revoked_at='2026-06-05' WHERE strategy_id=1")
    conn.commit()
    with pytest.raises(LiveAuthorizationError):
        live_gate.verify_live_authorization(conn, SqliteStrategyRepository(conn), "s", signers)
    # un-revoke but remove the approver key from the anchor -> re-verify fails
    conn.execute("UPDATE live_authorizations SET revoked_at=NULL WHERE strategy_id=1")
    conn.commit()
    signers.write_text("# no keys enrolled\n")
    with pytest.raises(LiveAuthorizationError):
        live_gate.verify_live_authorization(conn, SqliteStrategyRepository(conn), "s", signers)
```

- [ ] **Step 2: Run** `uv run pytest tests/test_live_gate.py -q -k live_authorization` → FAIL.

- [ ] **Step 3: Move the anchor + add the primitive.** In `algua/registry/live_gate.py`: ensure `from pathlib import Path` is imported (it is), and add near the top (after `_TTL`):
```python
# The go-live trust anchor: enrolled approver PUBLIC keys, resolved from the INSTALLED source tree
# (not CWD) so the gate always reads the vetted, CODEOWNERS-reviewed copy. Shared by the CLI and
# the trade-time verifier so there is exactly one anchor.
ALLOWED_SIGNERS_PATH = Path(__file__).resolve().parents[2] / "approvers" / "allowed_signers"
```
Add at the end of the module:
```python
class LiveAuthorizationError(RuntimeError):
    """The current live artifact is NOT covered by a re-verifiable human go-live signature."""


def verify_live_authorization(conn: sqlite3.Connection, repo: object, name: str,
                              allowed_signers_path: Path) -> sqlite3.Row:
    """Trade-time wall: prove the strategy's CURRENT artifact is human-authorized for live by
    re-verifying a stored signature against the CURRENT trust anchor. Trusts nothing forgeable —
    not the `stage` column, not an `approvals` row. Raises LiveAuthorizationError on any failure;
    returns the authorization row on success. The future live loop calls this before every order."""
    from algua.contracts.lifecycle import Stage
    from algua.registry.approvals import compute_artifact_hashes

    rec = repo.get(name)
    if rec.stage is not Stage.LIVE:
        raise LiveAuthorizationError(f"{name} is not live (stage={rec.stage.value})")
    identity = compute_artifact_hashes(name)
    row = conn.execute(
        "SELECT * FROM live_authorizations WHERE strategy_id=? AND code_hash=? AND config_hash=? "
        "AND dependency_hash IS ? AND revoked_at IS NULL ORDER BY id DESC LIMIT 1",
        (rec.id, identity.code_hash, identity.config_hash, identity.dependency_hash),
    ).fetchone()
    if row is None:
        raise LiveAuthorizationError(
            f"no unrevoked live authorization matching the current artifact of {name}")
    principal = verify_signature(allowed_signers_path, row["challenge"],
                                 base64.b64decode(row["signature"]))
    if principal is None or principal != row["principal"]:
        raise LiveAuthorizationError(
            f"live authorization signature for {name} failed re-verification against the anchor")
    return row
```

- [ ] **Step 4: Share the anchor with the CLI.** In `algua/cli/registry_cmd.py`: delete the local `ALLOWED_SIGNERS_PATH = ...` line (with its comment) and import it instead — add `from algua.registry.live_gate import ALLOWED_SIGNERS_PATH, SignatureError` (merge with the existing `from algua.registry.live_gate import SignatureError` line if present). All existing `ALLOWED_SIGNERS_PATH` uses now reference the imported name.

- [ ] **Step 5: Run** `uv run pytest tests/test_live_gate.py tests/test_cli_registry.py -q` → PASS.

- [ ] **Step 6: Full gate + commit** — `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports` (all pass; contracts kept, 0 broken).
```bash
git add algua/registry/live_gate.py algua/cli/registry_cmd.py tests/test_live_gate.py
git commit -m "feat(registry): verify_live_authorization trade-time primitive + shared anchor"
```

---

## Self-review notes

- **Spec coverage:** `live_authorizations` table (§3 → Task 1); evidence written in `verify_and_consume` at go-live (§4 → Task 1); `verify_live_authorization` + `LiveAuthorizationError` re-verifying identity + signature (§5 → Task 2); anchor moved to `live_gate` shared by CLI + primitive (§2 → Task 2); tests for schema, evidence write, happy path, and every rejection — non-live, no row, changed code, revoked, unenrolled key (§6 → Tasks 1–2). The version-assertion test needs no edit (asserts the `SCHEMA_VERSION` constant). Live acceptance is the future live-loop slice, not this one.
- **Type consistency:** `verify_and_consume(... ) -> str | None` is unchanged in signature (only its body now also writes a row). `verify_live_authorization(conn, repo, name, allowed_signers_path) -> sqlite3.Row` (raises) is used identically in Task 2's tests. `ArtifactIdentity` fields (`code_hash`/`config_hash`/`dependency_hash`) and `StrategyRecord.stage`/`.id` match `repository.py`. `ALLOWED_SIGNERS_PATH` has ONE definition (live_gate) after Task 2.
- **No placeholders:** every code step is complete. Tests monkeypatch `algua.registry.approvals.compute_artifact_hashes` (imported inside the primitive at call time, so patching the source module takes effect) to control the artifact identity without a real strategy module.
- **Security invariant:** the primitive's authority is steps 3–4 (an unrevoked row whose stored signature RE-VERIFIES against the current anchor for the current identity), never the `stage` column alone — an agent that forges `stage='live'` or inserts a fake row still can't produce a signature that verifies without the private key.
