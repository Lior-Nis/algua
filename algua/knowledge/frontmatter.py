from __future__ import annotations

from typing import Any

import yaml

_DELIM = "---"


def parse_doc(text: str) -> tuple[dict[str, Any], str]:
    """Split a markdown doc into (frontmatter dict, body). Empty dict if no frontmatter."""
    if not text.startswith(_DELIM):
        return {}, text
    parts = text.split(_DELIM, 2)
    if len(parts) < 3:
        return {}, text
    fm = yaml.safe_load(parts[1]) or {}
    body = parts[2]
    if body.startswith("\n"):
        body = body[1:]
    return fm, body


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
