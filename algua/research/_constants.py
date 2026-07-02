from __future__ import annotations

# Minimum holdout sample (Wall C). A holdout with fewer observations is underpowered and fails
# closed — complements the 1/sqrt(T) haircut, which is ZERO at N=1. ~one trading quarter. Protected
#
# Shared power floor: consumed by the gate criteria (``gates.GateCriteria`` default) and the
# idiosyncratic-alpha overlap floor (``regime.IR_MIN_OVERLAP_BARS``). It lives in this pure leaf so
# ``regime`` can reference it without importing ``gates`` (which would create a gates<->regime
# cycle, since ``gates`` composes ``regime``).
MIN_HOLDOUT_OBSERVATIONS = 63
