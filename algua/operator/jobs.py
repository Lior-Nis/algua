"""Per-job manifest binding ``--job`` to a FULL canonical argv + completion predicate (#486).

Pure declarative config — stdlib only, no I/O, no cross-module import edge, so ``algua.operator``
stays an import-linter leaf. Each allowlisted job declares:

* ``argv_template`` — the EXACT canonical argv the wrapper is permitted to run for this job, with
  ``"{name}"`` placeholder tokens for the lone variable positions. The wrapper matches the trailing
  ``operator run -- <command…>`` against this template by :meth:`OperatorJob.bind` — an exact-arity
  STRUCTURAL match, never a head prefix — so a mistyped / extended / rogue trailing command can
  never mark a session done (round-3 fix #4). Fail-closed on any deviation.
* ``is_completed`` — the completion predicate over ``(rc, parsed-stdout)`` deciding whether a run
  positively COMPLETED the session (marker recorded) vs merely exited 0 (e.g. a reconcile-deferred
  cycle, which must NOT be recorded or the session's trading is silently lost). See §D4.
* ``expected_duration_seconds`` — the stuck-lock grace threshold (§D2): an ``operator.lock`` held
  longer than this is a wedged holder, surfaced via an alert.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

__all__ = ["OPERATOR_JOBS", "CommandMismatch", "OperatorJob"]


class CommandMismatch(Exception):
    """The trailing command does not structurally match the job's canonical argv template."""


def _is_placeholder(token: str) -> bool:
    """A template token of the form ``{name}`` captures one non-empty variable value."""
    return len(token) >= 3 and token[0] == "{" and token[-1] == "}"


@dataclass(frozen=True)
class OperatorJob:
    """One allowlisted operator job: its canonical argv, grace, and completion predicate."""

    key: str
    argv_template: tuple[str, ...]
    expected_duration_seconds: float
    is_completed: Callable[[int, dict | None], bool]

    def bind(self, command: tuple[str, ...]) -> dict[str, str]:
        """Exact-arity full-argv match of ``command`` against ``argv_template``.

        Every fixed token must equal its template token; every ``{name}`` placeholder captures a
        non-empty value. ANY deviation — wrong arity, an altered/extra/missing token, an empty
        placeholder value — raises :class:`CommandMismatch`. Returns the captured ``{name: value}``.
        """
        if len(command) != len(self.argv_template):
            raise CommandMismatch(
                f"job {self.key!r} expects {len(self.argv_template)} argv tokens "
                f"({list(self.argv_template)}), got {len(command)} ({list(command)})"
            )
        captured: dict[str, str] = {}
        for tmpl, got in zip(self.argv_template, command, strict=True):
            if _is_placeholder(tmpl):
                name = tmpl[1:-1]
                if not got:
                    raise CommandMismatch(
                        f"job {self.key!r} placeholder {tmpl} captured an empty value"
                    )
                captured[name] = got
            elif tmpl != got:
                raise CommandMismatch(
                    f"job {self.key!r} token mismatch: expected {tmpl!r}, got {got!r}"
                )
        return captured


def _paper_completed(rc: int, payload: dict | None) -> bool:
    """A cycle COMPLETED the session iff the driver asserted success, did not defer, and — when
    it had tenants — decided on a named snapshot AND at least one tenant actually ticked. A cycle
    whose every tenant failed tick-time setup, or that ticked without provenance, is NOT
    completed: the marker stays unwritten and the next fire retries (#556)."""
    p = payload or {}
    if not (rc == 0 and p.get("ok") is True and not p.get("deferred")):
        return False
    strategies = p.get("strategies") or []
    if not strategies:
        return True
    snap = p.get("snapshot")
    has_snapshot = isinstance(snap, dict) and isinstance(snap.get("id"), str) and bool(snap["id"])
    any_ticked = any(isinstance(s, dict) and s.get("ok") is True for s in strategies)
    return has_snapshot and any_ticked


OPERATOR_JOBS: dict[str, OperatorJob] = {
    "paper": OperatorJob(
        key="paper",
        # #556: no variable token. The lane refreshes its own bars; `--refresh` is a fixed flag,
        # so the exact-arity match still rejects any appended/altered token (incl. `--snapshot X`),
        # and the env-var-left-empty failure mode of the old {snapshot} placeholder cannot recur.
        argv_template=("algua", "paper", "run-all", "--refresh"),
        expected_duration_seconds=900.0,
        # Require the driver's OWN positive verdict (`ok: true`), not just rc==0: `paper run-all`
        # today always exits non-zero on an `ok:false` outcome, but this predicate is the ONLY
        # thing standing between a future/altered driver behavior and silently marking a broken
        # session complete (GATE-2 finding, #486) — `rc==0` alone is not proof of success. A
        # `deferred:true` cycle exits 0 with `ok:true` but did NOT trade the session (a transient
        # reconcile condition) — also NOT completed, so the marker is left unwritten and the next
        # fire retries. PLUS (#556): a cycle that TICKED strategies must carry the snapshot it
        # decided on (`snapshot.id`) — provenance is part of "completed"; a no-work cycle (no
        # strategies) needs none.
        is_completed=lambda rc, payload: _paper_completed(rc, payload),
    ),
}
# Merge-back does NOT get an OPERATOR_JOBS entry (factory slice 3). `algua operator run --job X` is
# SESSION-GATED (`algua.operator.session_runner.run_session` calls `session_gate` unconditionally,
# once per session of the configured exchange) — but
# merge-back must fire repeatedly through the day (once per drain cycle), so reusing this wrapper
# would silently cap it at once per session. The candidate-selection gap this comment used to note
# (WHICH branch/strategy/universe/window to merge back) is closed by a durable queue
# (`data/mergeback-queue.json`, populated by the research driver + drained by
# `.codex/scripts/drain-mergeback-queue.sh`), but the drainer does NOT invoke merge-back through
# THIS manifest. It uses the separate, purely additive `algua operator lock-run -- <command…>`
# command instead: same `operator.lock` (so merge-back and the paper job still share real
# kernel-enforced mutual exclusion), but no session marker and no completion recording — exactly
# what a fires-many-times-per-day job needs, without reintroducing the once-per-session bug a naive
# OPERATOR_JOBS entry would cause.
