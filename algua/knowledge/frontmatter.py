from __future__ import annotations

import re
from typing import Any

import yaml

_DELIM = "---"
# Frontmatter is delimited by `---` on its OWN line, opening and closing. Matching the bare `---`
# SUBSTRING (the old text.split('---', 2)) split inside YAML values too — a description like
# 'a---b' truncated the frontmatter and spilled raw YAML into the body on the next sync (#252).
# The closing delimiter is a `---` line followed by a newline OR end-of-doc; the optional newline
# is then consumed so the body excludes it (matching the old leading-newline strip).
_FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---(?:\n|\Z)(.*)\Z", re.DOTALL)


def parse_doc(text: str) -> tuple[dict[str, Any], str]:
    """Split a markdown doc into (frontmatter dict, body). Empty dict if no frontmatter.

    Frontmatter is the block between an opening `---` line and the first subsequent `---` line;
    a `---` appearing INSIDE a YAML value (or in the body) is never treated as the delimiter."""
    m = _FRONTMATTER_RE.match(text)
    if m is None:
        return {}, text
    fm = yaml.safe_load(m.group(1)) or {}
    return fm, m.group(2)


def render_doc(frontmatter: dict[str, Any], body: str) -> str:
    """Render frontmatter + body back into a markdown doc."""
    fm_text = yaml.safe_dump(frontmatter, sort_keys=False).strip()
    return f"{_DELIM}\n{fm_text}\n{_DELIM}\n{body}"


def replace_block(text: str, marker: str, content: str) -> str:
    """Replace the bytes between <!-- ALGUA:{marker} --> and <!-- /ALGUA:{marker} -->.

    If both markers are absent, append a fresh block at the end. Prose is never touched.
    A malformed marker state (exactly one marker, duplicated markers, or the close marker
    before the open) is refused with a ValueError — silently appending in those cases would
    let a later sync swallow authored prose between a stray marker and an appended one.
    """
    start = f"<!-- ALGUA:{marker} -->"
    end = f"<!-- /ALGUA:{marker} -->"
    block = f"{start}\n{content}\n{end}"
    n_start = text.count(start)
    n_end = text.count(end)
    if n_start == 0 and n_end == 0:
        sep = "" if text.endswith("\n") or text == "" else "\n"
        return f"{text}{sep}{block}\n"
    if n_start != 1 or n_end != 1:
        raise ValueError(
            f"malformed ALGUA:{marker} markers ({n_start} open, {n_end} close); refusing to edit"
        )
    i = text.find(start)
    j = text.find(end)
    if j < i:
        raise ValueError(f"ALGUA:{marker} close marker precedes open marker; refusing to edit")
    return text[:i] + block + text[j + len(end):]
