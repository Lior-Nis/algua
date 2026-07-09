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


OPERATOR_JOBS: dict[str, OperatorJob] = {
    "paper": OperatorJob(
        key="paper",
        argv_template=("algua", "paper", "run-all", "--snapshot", "{snapshot}"),
        expected_duration_seconds=900.0,
        # Require the driver's OWN positive verdict (`ok: true`), not just rc==0: `paper run-all`
        # today always exits non-zero on an `ok:false` outcome, but this predicate is the ONLY
        # thing standing between a future/altered driver behavior and silently marking a broken
        # session complete (GATE-2 finding, #486) — `rc==0` alone is not proof of success. A
        # `deferred:true` cycle exits 0 with `ok:true` but did NOT trade the session (a transient
        # reconcile condition) — also NOT completed, so the marker is left unwritten and the next
        # fire retries.
        is_completed=lambda rc, payload: (
            rc == 0 and (payload or {}).get("ok") is True and not (payload or {}).get("deferred")
        ),
    ),
}
# The "research" job (`algua paper merge-back …`, completed ⇔ rc0, grace 3600) is DEFERRED to a
# follow-up (round-3 fix #1): a static ExecStart cannot resolve WHICH candidate branch/strategy to
# merge back — that selection mechanism must land first. It drops in here as a new manifest entry +
# unit pair with no wrapper change.
