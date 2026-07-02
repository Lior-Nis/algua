import pytest

from algua.contracts.net import require_https_allowlisted_host

_HOSTS = frozenset({"data.alpaca.markets"})


@pytest.mark.parametrize(
    "url",
    [
        "https://data.alpaca.markets/v2",
        "https://data.alpaca.markets",
        "HTTPS://DATA.ALPACA.MARKETS/v2",  # urlparse lowercases scheme + host
    ],
)
def test_accepts_https_allowlisted_host(url):
    # Returns None (no raise) on an accepted URL.
    assert require_https_allowlisted_host(url, _HOSTS) is None


@pytest.mark.parametrize(
    "url",
    [
        "http://data.alpaca.markets/v2",  # plaintext
        "https://evil.test/v2",  # foreign host
        "https://data.alpaca.markets@evil.test/v2",  # userinfo — real host is evil.test
        "https://data.alpaca.markets./v2",  # trailing-dot host not in the allowlist
        "data.alpaca.markets/v2",  # missing scheme -> not https
        "https:///v2",  # hostless
        "",  # empty
        "ftp://data.alpaca.markets",  # wrong scheme
        "https://data.alpaca.markets:99999/v2",  # allowlisted host but out-of-range port
        "https://data.alpaca.markets:bad/v2",  # allowlisted host but non-numeric port
    ],
)
def test_rejects_unsafe_url(url):
    with pytest.raises(ValueError):
        require_https_allowlisted_host(url, _HOSTS)


def test_empty_allowlist_fails_closed():
    # An empty allowlist admits nothing, even a well-formed https URL.
    with pytest.raises(ValueError):
        require_https_allowlisted_host("https://data.alpaca.markets/v2", frozenset())
