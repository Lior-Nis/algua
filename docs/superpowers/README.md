# Design and implementation history

For new work, read the [Vision of Record](../PRD.md), [current architecture](../architecture.md),
and [vision reconciliation](../vision-reconciliation.md) first.

Files in `specs/` and `plans/` preserve decisions, evidence and implementation instructions as of
their dates. A document marked approved records approval of that design at that time; it does
not establish that all its features shipped or that its roadmap still governs.

Before reusing a plan, check its product assumptions against the current vision, its described
state against code/tests, and its operating commands against `CLAUDE.md`. Preserve useful design
rationale and current integrity contracts. Escalate changes to safety or authority boundaries
through the existing review process.

The September 6 PRD and its numbered sections remain available in
[the pre-reconciliation revision](https://github.com/Lior-Nis/algua/blob/d9ed65455a626d3120559eca30fc0ed0818f9084/docs/PRD.md).
Older numerical PRD references refer to that revision unless a notice explicitly maps them to
the current vision. Use section titles alongside numbers in new references.
