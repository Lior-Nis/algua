# algua/research/idea_dedup.py
from __future__ import annotations

import re

# Small, explicit stopword set: generic finance/strategy filler that would otherwise inflate
# token overlap between unrelated ideas. Not a linguistics project — just the obvious noise.
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "by", "is", "are",
    "be", "that", "this", "as", "at", "from", "it", "its", "we", "using", "use", "based",
    "strategy", "signal", "returns", "return", "stock", "stocks", "market",
})
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_UNKNOWN_FAMILY = "unknown"


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1}


def signature(title: str, hypothesis: str) -> str:
    """Normalized, order-independent dedup signature: the sorted, deduped, stopword-stripped
    token set of title + hypothesis, space-joined. Stored on the idea so a collision is a cheap
    token-set comparison and the signature stays human-inspectable."""
    return " ".join(sorted(_tokens(f"{title} {hypothesis}")))


def _sig_tokens(sig: str) -> set[str]:
    return set(sig.split())


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def families_comparable(f1: str | None, f2: str | None) -> bool:
    """Whether two ideas are in scope to be compared. A NULL or "unknown" family on EITHER side
    is compared against everything (fail-safe: a missing/mis-tagged family can never silently
    suppress collision detection). Two concrete families compare only when equal."""
    if f1 in (None, _UNKNOWN_FAMILY) or f2 in (None, _UNKNOWN_FAMILY):
        return True
    return f1 == f2


def is_collision(
    cand_signature: str,
    cand_family: str | None,
    other_signature: str,
    other_family: str | None,
    *,
    threshold: float = 0.6,
) -> bool:
    """True when two ideas are in comparable families AND their signatures' Jaccard meets the
    threshold. Coarse and recall-oriented within a family; the agent semantic layer is the
    backstop for paraphrase-level evasion the token set can't catch."""
    if not families_comparable(cand_family, other_family):
        return False
    return jaccard(_sig_tokens(cand_signature), _sig_tokens(other_signature)) >= threshold
