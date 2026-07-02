"""Pure network-endpoint guards shared across lanes.

This is a stdlib-only leaf (imports no other ``algua`` module) so every lane that dials an
outbound credential-bearing endpoint — the data-ingestion provider (``algua.data``), the
trading broker (``algua.execution``), and config validation (``algua.config``) — can share the
SAME https-scheme + host-allowlist wall without crossing an import boundary.

The wall exists because Alpaca credentials (``APCA-API-KEY-ID`` / ``APCA-API-SECRET-KEY``)
travel as request headers: a base URL that is plaintext ``http`` or points at a non-Alpaca host
would leak those account-scoped secrets in the clear or to an attacker (SSRF). Every such URL
must be validated BEFORE the credentials attach.
"""
from __future__ import annotations

from urllib.parse import urlparse


def require_https_allowlisted_host(url: str, allowed_hosts: frozenset[str]) -> None:
    """Raise ``ValueError`` unless ``url`` is https to a host in ``allowed_hosts``.

    Fails closed on every ambiguous input: a non-https scheme, an unparseable/hostless URL
    (``hostname`` is ``None``), a host outside the allowlist, an empty allowlist (which admits
    nothing), and a malformed port (a non-numeric/out-of-range ``:port`` — ``parsed.port``
    raises ``ValueError`` on access, which we translate into a closed refusal rather than let
    the endpoint through unvalidated). ``urlparse(...).hostname`` lowercases the host and
    resolves userinfo tricks — ``https://data.alpaca.markets@evil.test`` parses to host
    ``evil.test`` and is rejected — so the allowlist is enforced on the REAL destination.
    """
    parsed = urlparse(url)
    try:
        parsed.port  # noqa: B018 — property access; a non-numeric/out-of-range port raises here
    except ValueError:
        raise ValueError(f"refusing endpoint {url!r}: malformed port") from None
    if (
        parsed.scheme != "https"
        or parsed.hostname is None
        or parsed.hostname not in allowed_hosts
    ):
        raise ValueError(
            f"refusing endpoint {url!r} (host {parsed.hostname!r}); "
            f"must be https to one of {sorted(allowed_hosts)}"
        )
