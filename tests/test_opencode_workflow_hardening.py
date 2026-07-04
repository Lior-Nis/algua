"""Regression tests for the opencode workflow hardening (#456).

These lock the invariants that close the two prompt-injection / secret-exfiltration
vectors described in
docs/superpowers/specs/2026-07-03-harden-opencode-workflow-456-design.md so a future
edit cannot silently re-open the hole:

* trusted-trigger allow-list (OWNER/COLLABORATOR only, MEMBER excluded);
* whitespace-tokenized command match (no /octopus false-accept);
* manual `environment:` approval gate + fail-closed guard step (first, before opencode);
* `actions: read` (authorizes the guard's environments-API read) and no `id-token: write`;
* every action SHA-pinned;
* `types: [created]` only (immutable trigger);
* CODEOWNERS covers `.github/`.

The authorization truth table is evaluated over the REAL `if:` string lifted from the
workflow, using a tiny hand-written recursive-descent parser/evaluator (NO `eval`).
"""

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "opencode.yml"
CODEOWNERS_PATH = REPO_ROOT / "CODEOWNERS"


def _load_workflow() -> dict:
    with open(WORKFLOW_PATH) as f:
        return yaml.safe_load(f)


def _job() -> dict:
    return _load_workflow()["jobs"]["opencode"]


# --------------------------------------------------------------------------- #
# Static invariants                                                           #
# --------------------------------------------------------------------------- #


def test_no_id_token_permission():
    """id-token: write must be gone (no OIDC minting for the agent)."""
    permissions = _job()["permissions"]
    assert "id-token" not in permissions, (
        "id-token should not be in opencode workflow permissions"
    )


def test_actions_read_permission_present():
    """actions: read must be present — it authorizes the guard's environments-API read."""
    permissions = _job()["permissions"]
    assert permissions.get("actions") == "read", (
        "job permissions must include `actions: read` to authorize the "
        "environments/opencode API call in the fail-closed guard (GATE-1 finding 3)"
    )


def test_read_only_scopes_kept():
    """The existing read scopes must remain."""
    permissions = _job()["permissions"]
    for scope in ("contents", "pull-requests", "issues"):
        assert permissions.get(scope) == "read", f"{scope}: read must be kept"


def test_all_uses_are_sha_pinned():
    """Every uses: must be pinned to a 40-hex commit SHA (no @latest / @vN / branch)."""
    with open(WORKFLOW_PATH) as f:
        content = f.read()

    matches = re.findall(r"uses:\s*(\S+)", content)
    assert len(matches) > 0, "Should find at least one 'uses:' in the workflow"

    sha_pattern = r"^[\w.\-/]+@[0-9a-f]{40}$"
    for uses_value in matches:
        assert re.match(sha_pattern, uses_value), (
            f"'{uses_value}' should be SHA-pinned (40-character hex commit hash)"
        )
        assert "@latest" not in uses_value, f"'{uses_value}' should not contain @latest"
        assert not re.search(r"@v\d", uses_value), (
            f"'{uses_value}' should not contain @vX version tags"
        )


def test_environment_gate_present():
    """The job must attach to the protected `opencode` deployment environment."""
    assert _job().get("environment") == "opencode", (
        "the opencode job must declare `environment: opencode` (the manual approval gate)"
    )


def test_trigger_gated_on_author_association():
    """The job `if:` must reference author_association (trusted-trigger allow-list)."""
    if_condition = _job().get("if", "")
    assert "author_association" in if_condition, (
        "the opencode job 'if' condition should check author_association"
    )


def test_triggers_created_only():
    """Both event types must be restricted to types: [created] exactly (immutable trigger)."""
    wf = _load_workflow()
    # PyYAML parses the unquoted `on:` key as the boolean True.
    on_block = wf[True] if True in wf else wf["on"]
    for event in ("issue_comment", "pull_request_review_comment"):
        assert event in on_block, f"{event} trigger must be present"
        assert on_block[event] == {"types": ["created"]}, (
            f"{event} must be restricted to types: [created] exactly "
            f"(no edited/submitted), got {on_block[event]!r}"
        )


def _steps() -> list:
    return _job()["steps"]


def _guard_step() -> dict:
    return _steps()[0]


def _opencode_step_index() -> int:
    for i, step in enumerate(_steps()):
        uses = step.get("uses", "")
        if "anomalyco/opencode/github" in uses:
            return i
    raise AssertionError("anomalyco/opencode/github step not found")


def test_guard_step_is_first_and_before_opencode():
    """The fail-closed guard must be the job's first step, before the opencode step."""
    guard = _guard_step()
    run_text = guard.get("run", "")
    assert "run" in guard, "the first step must be a run: guard step"
    assert "environments/opencode" in run_text, (
        "the guard must query the environments/opencode API"
    )
    assert _opencode_step_index() > 0, (
        "the opencode step must come after the guard step (guard index 0)"
    )


def test_guard_step_assertions():
    """Guard run: text must assert required reviewers, prevent_self_review, admin bypass, exit 1."""
    run_text = _guard_step().get("run", "")
    assert "required_reviewers" in run_text, "guard must check the required-reviewers rule presence"
    assert "prevent_self_review" in run_text, (
        "guard must assert prevent_self_review (self-approvable env must fail closed)"
    )
    assert "can_admins_bypass" in run_text, (
        "guard must reference can_admins_bypass (admin-bypass posture)"
    )
    assert "exit 1" in run_text, "guard must contain at least one fail-closed `exit 1` branch"


def test_guard_drift_scoped_to_review_comment_only():
    """Drift compare is scoped to pull_request_review_comment (no tautological issue_comment)."""
    run_text = _guard_step().get("run", "")
    # Drift is only meaningful with an event-time anchored SHA — only review comments carry one.
    assert "pull_request_review_comment" in run_text, (
        "the drift compare must be gated on the pull_request_review_comment event"
    )
    assert ("commit_id" in run_text) or ("pull_request.head.sha" in run_text), (
        "the drift compare must reference the review-comment anchored SHA "
        "(commit_id / pull_request.head.sha)"
    )
    # Guard against a future edit re-introducing a tautological issue_comment drift check:
    # any drift comparison for issue_comment would compare the current head to itself.
    assert not re.search(r"issue_comment[\s\S]*head\.sha", run_text), (
        "the guard must not resolve-and-compare a PR head for issue_comment (tautological)"
    )


def test_no_opencode_api_key_in_guard():
    """The guard must not reference OPENCODE_API_KEY (read-only github.token only)."""
    assert "OPENCODE_API_KEY" not in _guard_step().get("run", ""), (
        "the guard step must not reference OPENCODE_API_KEY"
    )
    assert "OPENCODE_API_KEY" not in str(_guard_step().get("env", {}))


def test_codeowners_guards_github_dir():
    """CODEOWNERS must have a rule covering .github/."""
    with open(CODEOWNERS_PATH) as f:
        content = f.read()

    found = False
    for line in content.split("\n"):
        if line.strip().startswith("#") or not line.strip():
            continue
        tokens = line.strip().split()
        if tokens and (
            tokens[0].startswith("/.github") or tokens[0].startswith(".github")
        ):
            found = True
            break
    assert found, "CODEOWNERS should have a rule covering .github/"


# --------------------------------------------------------------------------- #
# Authorization truth-table — hand-written parser/evaluator (NO eval)         #
# --------------------------------------------------------------------------- #


class _Tok:
    def __init__(self, kind: str, value: str):
        self.kind = kind
        self.value = value

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"Tok({self.kind}, {self.value!r})"


def _tokenize(expr: str) -> list:
    """Tokenize the fixed GitHub-expression sub-grammar the workflow uses."""
    tokens: list = []
    i = 0
    n = len(expr)
    while i < n:
        c = expr[i]
        if c.isspace():
            i += 1
            continue
        if c == "'":
            j = i + 1
            while j < n and expr[j] != "'":
                j += 1
            if j >= n:
                raise ValueError("unterminated string literal")
            tokens.append(_Tok("string", expr[i + 1 : j]))
            i = j + 1
            continue
        if expr.startswith("||", i):
            tokens.append(_Tok("or", "||"))
            i += 2
            continue
        if expr.startswith("&&", i):
            tokens.append(_Tok("and", "&&"))
            i += 2
            continue
        if expr.startswith("==", i):
            tokens.append(_Tok("eq", "=="))
            i += 2
            continue
        if c == "(":
            tokens.append(_Tok("lparen", "("))
            i += 1
            continue
        if c == ")":
            tokens.append(_Tok("rparen", ")"))
            i += 1
            continue
        if c == ",":
            tokens.append(_Tok("comma", ","))
            i += 1
            continue
        if c.isalpha() or c == "_":
            j = i
            while j < n and (expr[j].isalnum() or expr[j] in "._"):
                j += 1
            tokens.append(_Tok("ident", expr[i:j]))
            i = j
            continue
        raise ValueError(f"unexpected character {c!r} at position {i}")
    tokens.append(_Tok("eof", ""))
    return tokens


def _truthy(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return len(v) > 0
    raise ValueError(f"cannot coerce {v!r} to truthiness")


class _Parser:
    """Recursive-descent parser/evaluator for the workflow `if:` sub-grammar.

    Grammar:
        expr    := or
        or      := and ('||' and)*
        and     := cmp ('&&' cmp)*
        cmp     := primary ('==' primary)?
        primary := '(' expr ')' | call | stringlit | ctxref
        call    := ('contains'|'startsWith'|'endsWith') '(' primary ',' primary ')'

    Evaluated against the two inputs (assoc, body). No eval, no __builtins__; any
    unexpected token raises loudly.
    """

    def __init__(self, tokens: list, assoc: str, body: str):
        self.tokens = tokens
        self.pos = 0
        self.assoc = assoc
        self.body = body

    def _peek(self) -> _Tok:
        return self.tokens[self.pos]

    def _next(self) -> _Tok:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def _expect(self, kind: str) -> _Tok:
        tok = self._next()
        if tok.kind != kind:
            raise ValueError(f"expected {kind}, got {tok.kind} ({tok.value!r})")
        return tok

    def parse(self):
        value = self._parse_or()
        if self._peek().kind != "eof":
            raise ValueError(f"trailing tokens: {self._peek()}")
        return value

    def _parse_or(self):
        value = self._parse_and()
        while self._peek().kind == "or":
            self._next()
            right = self._parse_and()
            value = _truthy(value) or _truthy(right)
        return value

    def _parse_and(self):
        value = self._parse_cmp()
        while self._peek().kind == "and":
            self._next()
            right = self._parse_cmp()
            value = _truthy(value) and _truthy(right)
        return value

    def _parse_cmp(self):
        left = self._parse_primary()
        if self._peek().kind == "eq":
            self._next()
            right = self._parse_primary()
            return left == right
        return left

    def _parse_primary(self):
        tok = self._peek()
        if tok.kind == "lparen":
            self._next()
            value = self._parse_or()
            self._expect("rparen")
            return value
        if tok.kind == "string":
            self._next()
            return tok.value
        if tok.kind == "ident":
            self._next()
            if self._peek().kind == "lparen":
                return self._eval_call(tok.value)
            return self._resolve_ref(tok.value)
        raise ValueError(f"unexpected token in primary: {tok}")

    def _eval_call(self, name: str):
        self._expect("lparen")
        a = self._parse_primary()
        self._expect("comma")
        b = self._parse_primary()
        self._expect("rparen")
        if not isinstance(a, str) or not isinstance(b, str):
            raise ValueError(f"{name} expects string args, got {a!r}, {b!r}")
        if name == "contains":
            return b in a
        if name == "startsWith":
            return a.startswith(b)
        if name == "endsWith":
            return a.endswith(b)
        raise ValueError(f"unsupported call {name!r}")

    def _resolve_ref(self, ref: str) -> str:
        if ref == "github.event.comment.author_association":
            return self.assoc
        if ref == "github.event.comment.body":
            return self.body
        raise ValueError(f"unsupported context ref {ref!r}")


def _eval_if(assoc: str, body: str) -> bool:
    if_condition = _job()["if"]
    tokens = _tokenize(if_condition)
    return _truthy(_Parser(tokens, assoc, body).parse())


_TRUTH_TABLE = [
    ("NONE", "/oc review", False),
    ("NONE", "/opencode go", False),
    ("NONE", "please /oc", False),
    ("NONE", "please /opencode", False),
    ("COLLABORATOR", "no command here", False),
    ("COLLABORATOR", "/oc", True),
    ("COLLABORATOR", "/opencode", True),
    ("COLLABORATOR", "hey /oc please", True),
    ("COLLABORATOR", "hey /opencode", True),
    ("COLLABORATOR", "review /oc", True),
    ("OWNER", "/oc", True),
    ("MEMBER", "/oc", False),
    ("CONTRIBUTOR", "/oc", False),
    ("COLLABORATOR", "/octopus", False),
    ("COLLABORATOR", "/opencode-now", False),
    ("COLLABORATOR", "run /ocarina", False),
]


@pytest.mark.parametrize("assoc,body,expected", _TRUTH_TABLE)
def test_authorization_truth_table(assoc, body, expected):
    """Evaluate the REAL `if:` string against the design's 16-row truth table."""
    assert _eval_if(assoc, body) is expected, (
        f"if: for (assoc={assoc!r}, body={body!r}) should be {expected}"
    )


def test_evaluator_rejects_unexpected_tokens():
    """The evaluator must fail loudly on any construct outside the whitelisted grammar."""
    with pytest.raises(ValueError):
        _Parser(_tokenize("github.event.foo.bar == 'x'"), "OWNER", "/oc").parse()
    with pytest.raises(ValueError):
        _tokenize("1 & 2")
