"""SR 11-7 model-governance / validation-record inventory (issue #393).

A *governance record* is the lightweight validation dossier an org needs to scale to many
models: an accountable owner, intended-use assumptions, known limitations/failure modes, a
validation summary, the linked passing gate-evaluation, and — the load-bearing field — a
scheduled next-review date a monitor can enforce. Without a next-review/owner field the system
cannot even detect that a live model is overdue for revalidation.

Design (KISS, no schema migration): the record lives as *governance-owned* frontmatter keys on
the strategy's existing kb-vault doc (``<knowledge_dir>/strategies/<name>.md``), reusing the same
frontmatter + block infra as the rest of the vault. This module is PURE knowledge-layer: it never
imports the registry and never opens the DB. Binding the record to a REAL strategy (registry row
+ gate-evaluation integrity) is enforced by the caller at the CLI seam, which resolves the doc
path only after ``repo.get(name)`` succeeds.

``governance_*`` keys are governance-owned: they are NOT registry-owned (``registry set`` /
``sync._apply_owned_metadata`` must never add them) and NOT hand-authored. Edit them via
``algua governance record``; the render overwrites the owned keys + the GOVERNANCE block.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from algua.knowledge.frontmatter import parse_doc, render_doc, replace_block

# Governance-owned frontmatter keys. Kept as a frozenset so a future projector can assert it never
# overlaps the registry-owned set.
OWNED_KEYS = frozenset(
    {
        "governance_owner",
        "governance_assumptions",
        "governance_limitations",
        "governance_validation_summary",
        "governance_last_validated",
        "governance_next_review",
        "governance_gate_eval_id",
    }
)

# Any ALGUA marker substring in user text would corrupt the rendered block: replace_block refuses
# on duplicated/malformed markers on the NEXT edit. Reject rather than silently mangle.
_MARKER_TOKEN = "ALGUA:"


@dataclass(frozen=True)
class GovernanceRecord:
    """One strategy's governance dossier, normalized from frontmatter.

    ``next_review``/``last_validated`` are normalized to ``date | None`` — a value that does not
    normalize to a real ISO date (missing, blank, a list/dict, a bare ``datetime``, or an
    unparseable string) becomes ``None`` and is treated as OVERDUE by ``is_overdue`` (fail-closed).
    """

    name: str
    owner: str | None
    assumptions: list[str]
    limitations: list[str]
    validation_summary: str | None
    last_validated: date | None
    next_review: date | None
    gate_eval_id: int | None
    # True iff the doc carries a governance_gate_eval_id key that does NOT normalize to a positive
    # int (e.g. 'abc', -1, []). Distinct from a simply-absent key: a malformed citation is
    # unverifiable evidence and must read fail-closed, not as 'no citation'.
    gate_eval_id_malformed: bool
    present: bool  # True iff the doc actually carries a governance record


def _norm_date(value: Any) -> date | None:
    """Normalize a frontmatter date value to ``date | None`` — fail-closed.

    ``yaml.safe_load`` yields a ``datetime.date`` for an unquoted ISO date and a ``str`` for a
    quoted one; we accept both. A ``datetime`` (has a time-of-day), a list/dict, a blank string,
    or an unparseable string all normalize to ``None`` (⇒ overdue).
    """
    if isinstance(value, date):
        # bool is an int, not a date; datetime IS a date subclass but carries a time — reject it so
        # a stray timestamp can't masquerade as a clean review date.
        from datetime import datetime

        if isinstance(value, datetime):
            return None
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return date.fromisoformat(text)
        except ValueError:
            return None
    return None


def _norm_list(value: Any) -> list[str]:
    """Normalize a repeated field to ``list[str]`` (drops blanks). A bare string becomes a
    one-element list so a hand-authored scalar still round-trips sanely."""
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if str(v).strip()]


def _norm_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _reject_markers(*texts: str) -> None:
    for t in texts:
        if _MARKER_TOKEN in t:
            raise ValueError(
                f"governance text may not contain the marker token {_MARKER_TOKEN!r} "
                "(it would corrupt the rendered GOVERNANCE block)"
            )


def read_governance(doc_path: Path, name: str) -> GovernanceRecord:
    """Parse the governance record from a strategy doc. Absent doc / no governance keys ⇒ a
    record with ``present=False`` and every field cleared (so it reads as OVERDUE).

    ``doc_path`` is resolved by the caller (via ``sync.strategy_doc_path``) AFTER the strategy is
    confirmed to exist in the registry — this module never touches the registry.
    """
    if not doc_path.exists():
        return GovernanceRecord(
            name=name, owner=None, assumptions=[], limitations=[], validation_summary=None,
            last_validated=None, next_review=None, gate_eval_id=None,
            gate_eval_id_malformed=False, present=False,
        )
    fm, _ = parse_doc(doc_path.read_text())
    present = any(k in fm for k in OWNED_KEYS)
    gate_eval_id, gate_malformed = _norm_gate_id(fm)
    return GovernanceRecord(
        name=name,
        owner=_norm_str(fm.get("governance_owner")),
        assumptions=_norm_list(fm.get("governance_assumptions")),
        limitations=_norm_list(fm.get("governance_limitations")),
        validation_summary=_norm_str(fm.get("governance_validation_summary")),
        last_validated=_norm_date(fm.get("governance_last_validated")),
        next_review=_norm_date(fm.get("governance_next_review")),
        gate_eval_id=gate_eval_id,
        gate_eval_id_malformed=gate_malformed,
        present=present,
    )


def _norm_gate_id(fm: dict[str, Any]) -> tuple[int | None, bool]:
    """Normalize ``governance_gate_eval_id`` to ``(id | None, malformed)``.

    Key absent ⇒ ``(None, False)``. Key present but not a POSITIVE int (``'abc'``, ``-1``, ``[]``,
    ``0``, a bool) ⇒ ``(None, True)`` so the caller can fail-close on unverifiable evidence rather
    than mistake it for 'no citation'.
    """
    if "governance_gate_eval_id" not in fm:
        return None, False
    raw = fm.get("governance_gate_eval_id")
    if raw is None:
        return None, False
    value: int | None = None
    if isinstance(raw, int) and not isinstance(raw, bool):
        value = raw
    elif isinstance(raw, str) and raw.strip().lstrip("+").isdigit():
        value = int(raw.strip())
    if value is None or value <= 0:
        return None, True
    return value, False


def is_overdue(record: GovernanceRecord, today: date) -> bool:
    """Fail-closed overdue check. A record with no valid ``next_review`` (missing / blank /
    malformed / absent doc / no governance record at all) is OVERDUE, not silently ok. A valid
    ``next_review`` strictly before ``today`` is overdue; ``next_review == today`` is due-today,
    NOT yet overdue.
    """
    if record.next_review is None:
        return True
    return record.next_review < today


def _render_block(record: GovernanceRecord) -> str:
    def _bullets(items: list[str]) -> str:
        return "\n".join(f"- {i}" for i in items) if items else "_none recorded_"

    return (
        f"**Owner:** {record.owner or '_unassigned_'}\n\n"
        f"**Last validated:** {record.last_validated.isoformat() if record.last_validated else '—'}"
        f"  ·  **Next review:** "
        f"{record.next_review.isoformat() if record.next_review else '_UNSET (overdue)_'}"
        f"  ·  **Gate eval:** {record.gate_eval_id if record.gate_eval_id is not None else '—'}\n\n"
        f"**Assumptions:**\n{_bullets(record.assumptions)}\n\n"
        f"**Limitations:**\n{_bullets(record.limitations)}\n\n"
        f"**Validation summary:** {record.validation_summary or '_none_'}"
    )


def record_governance(
    doc_path: Path,
    name: str,
    *,
    owner: str,
    assumptions: list[str],
    limitations: list[str],
    validation_summary: str | None,
    next_review: date,
    last_validated: date | None,
    gate_eval_id: int | None,
) -> GovernanceRecord:
    """Write the governance-owned frontmatter keys + the rendered GOVERNANCE block onto an
    EXISTING strategy doc. Raises ``FileNotFoundError`` if the doc is absent — the caller creates
    the strategy + its doc first; governance never fabricates a doc for a phantom strategy.

    The owned keys are fully overwritten every call (governance-owned, like the registry-owned
    keys in ``sync._apply_owned_metadata``); any other frontmatter key is preserved.
    """
    if not doc_path.exists():
        raise FileNotFoundError(f"no strategy doc for {name!r} at {doc_path}")
    assumptions = _norm_list(assumptions)
    limitations = _norm_list(limitations)
    owner = owner.strip()
    if not owner:
        raise ValueError("governance record requires an accountable --owner")
    validation_summary = _norm_str(validation_summary)
    _reject_markers(owner, *assumptions, *limitations, validation_summary or "")

    fm, body = parse_doc(doc_path.read_text())
    fm["governance_owner"] = owner
    fm["governance_assumptions"] = assumptions
    fm["governance_limitations"] = limitations
    fm["governance_validation_summary"] = validation_summary
    fm["governance_last_validated"] = last_validated.isoformat() if last_validated else None
    fm["governance_next_review"] = next_review.isoformat()
    fm["governance_gate_eval_id"] = gate_eval_id

    record = GovernanceRecord(
        name=name, owner=owner, assumptions=assumptions, limitations=limitations,
        validation_summary=validation_summary, last_validated=last_validated,
        next_review=next_review, gate_eval_id=gate_eval_id,
        gate_eval_id_malformed=False, present=True,
    )
    body = replace_block(body, "GOVERNANCE", _render_block(record))
    doc_path.write_text(render_doc(fm, body))
    return record


def record_to_json(record: GovernanceRecord, *, today: date) -> dict[str, Any]:
    """Serialize a record for the CLI JSON surface, including the derived overdue verdict."""
    return {
        "name": record.name,
        "owner": record.owner,
        "assumptions": record.assumptions,
        "limitations": record.limitations,
        "validation_summary": record.validation_summary,
        "last_validated": record.last_validated.isoformat() if record.last_validated else None,
        "next_review": record.next_review.isoformat() if record.next_review else None,
        "gate_eval_id": record.gate_eval_id,
        "gate_eval_id_malformed": record.gate_eval_id_malformed,
        "present": record.present,
        "overdue": is_overdue(record, today),
    }
