"""What a strategy name must satisfy for `client_order_id` to be unambiguous.

A LEAF on purpose (stdlib only). The rule has to be enforced where names are MINTED -- the registry
-- and applied where ids are BUILT -- `execution.order_state`. `order_state` imports
`algua.live.paper_loop`, so the registry importing it would drag the whole live lane across the
"registry stays off the live lane" contract. Keeping the rule here lets both sides share it without
either reaching into the other's layer.
"""

from __future__ import annotations

import re

#: Strip anything outside [A-Za-z0-9_-] so a symbol or strategy name with odd characters cannot
#: produce an id the venue rejects.
COID_SANITIZE = re.compile(r"[^A-Za-z0-9_-]")
COID_MAX_CHARS = 128  # Alpaca's documented client_order_id ceiling
#: Longest strategy name that can never overflow the id: 128 minus the "-YYYYmmddTHHMMSSZ-" stamp
#: and a generous symbol allowance. Enforced at REGISTRATION so the raise below is unreachable.
MAX_STRATEGY_NAME_CHARS = 100


def assert_coid_safe_name(name: str) -> None:
    """Refuse a strategy name that could make `client_order_id` ambiguous.

    TWO failure modes, both ending with one id standing for two different decisions:

    * LENGTH -- the id used to be truncated with `[:128]`, and truncation cuts from the RIGHT, so a
      long enough name pushed the timestamp and symbol off the end and every session and every
      symbol collapsed onto one id.
    * NON-ASCII -- sanitisation maps every character outside [A-Za-z0-9_-] to "_", so two distinct
      names can reduce to the same string.

    Either defeats duplicate-order recovery: it compares the returned symbol and side, but two
    decisions colliding on both pass every check, so the current order is silently attributed to a
    stale one and never submitted.

    Checked at registration rather than at submit time because a name is CONFIGURATION: it is fixed
    once, so this fails immediately and identically rather than intermittently mid-tick.
    """
    if len(name) > MAX_STRATEGY_NAME_CHARS:
        raise ValueError(
            f"strategy name is {len(name)} chars, over the {MAX_STRATEGY_NAME_CHARS} limit that "
            f"keeps client_order_id unambiguous (Alpaca caps the id at {COID_MAX_CHARS})"
        )
    if not name.isascii():
        raise ValueError(
            f"strategy name {name!r} is not ASCII; client_order_id sanitisation would map it onto "
            f"the same id as another name, making two decisions indistinguishable"
        )
