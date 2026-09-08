"""The inspirations domain of the vault (spec 2026-09-08 §4/§5): one note per thing the web
says works. Written ONLY by the trusted forage driver via `accept_new_notes`; frontmatter
edited ONLY by the trusted leap driver via `mark_used` / `mark_exhausted`. Pure vault I/O:
imports config + knowledge only."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import yaml

from algua.config.settings import Settings
from algua.knowledge.frontmatter import parse_doc, render_doc
from algua.knowledge.sync import _safe_path, kb_sync_lock

NOTE_ID_RE = re.compile(r"^\d{4}-\d{2}-\d{2}-[a-z0-9][a-z0-9-]{2,60}$")
SOURCE_KINDS = frozenset({"book_summary", "paper", "forum", "video", "blog", "other"})
MARKETS = frozenset({"us_equities", "crypto", "forex", "prediction", "any"})
HORIZONS = frozenset({"intraday", "daily", "weekly", "monthly", "event"})
OBSCURITY = frozenset({"canon", "common", "niche", "rare"})
NOTE_STATUSES = frozenset({"fresh", "used", "exhausted"})
REQUIRED = ("id", "found_at", "source_url", "venue", "source_kind", "category", "market",
            "horizon", "mechanism", "obscurity", "status")
MAX_NOTE_BYTES = 16384
_TRACKING = re.compile(r"^(utm_.*|fbclid|gclid|mc_cid|mc_eid)$")


def inspirations_dir(settings: Settings) -> Path:
    return settings.knowledge_dir / "inspirations"


def canonical_url(url: str) -> str:
    parts = urlsplit(url.strip())
    query = urlencode([(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
                       if not _TRACKING.match(k)])
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path, query, ""))


def url_hash(url: str) -> str:
    return hashlib.sha256(canonical_url(url).encode()).hexdigest()


def parse_note(text: str) -> tuple[dict[str, Any], str]:
    return parse_doc(text)


def validate_note(fm: dict[str, Any], *, stem: str, categories: set[str]) -> list[str]:
    problems = [f"missing: {k}" for k in REQUIRED if not fm.get(k)]
    if fm.get("id") != stem:
        problems.append("id != filename stem")
    checks = (("source_kind", SOURCE_KINDS), ("market", MARKETS), ("horizon", HORIZONS),
              ("obscurity", OBSCURITY), ("status", NOTE_STATUSES))
    for key, allowed in checks:
        if fm.get(key) and str(fm[key]) not in allowed:
            problems.append(f"{key}: {fm[key]}")
    if fm.get("category") and str(fm["category"]) not in categories:
        problems.append(f"category: {fm['category']}")
    if fm.get("status") not in (None, "fresh"):
        problems.append("status must be fresh on acceptance")
    return problems


class SeenFile:
    def __init__(self, path: Path) -> None:
        self.path = path

    def hashes(self) -> set[str]:
        if not self.path.exists():
            return set()
        out = set()
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                out.add(json.loads(line)["hash"])
            except Exception:
                continue
        return out

    def append(self, url: str, *, run_stamp: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"hash": url_hash(url), "url": canonical_url(url),
                                "first_seen": datetime.now(UTC).isoformat(),
                                "run": run_stamp}) + "\n")


def accept_new_notes(*, staged_dir: Path, settings: Settings, seen_path: Path,
                     categories: set[str], run_stamp: str, max_notes: int) -> dict:
    """Trusted acceptance of what the forage agent staged (spec §5 policy)."""
    seen = SeenFile(seen_path)
    seen_hashes = seen.hashes()
    dest = inspirations_dir(settings)
    accepted: list[str] = []
    rejected: list[dict] = []
    for path in sorted(staged_dir.glob("*.md")) if staged_dir.exists() else []:
        reasons: list[str] = []
        stem = path.stem
        if path.is_symlink() or not path.is_file():
            reasons.append("not a regular file")
        if not NOTE_ID_RE.match(stem):
            reasons.append(f"bad filename: {path.name}")
        elif path.stat().st_size > MAX_NOTE_BYTES:
            reasons.append(f"too large: {path.stat().st_size} bytes")
        if reasons:
            rejected.append({"file": path.name, "reasons": reasons})
            continue
        fm, _ = parse_note(path.read_text(encoding="utf-8"))
        reasons = validate_note(fm, stem=stem, categories=categories)
        if not reasons and url_hash(str(fm["source_url"])) in seen_hashes:
            reasons.append("already seen: source_url")
        if not reasons and (dest / path.name).exists():
            reasons.append("id already exists in the vault")
        if not reasons and len(accepted) >= max_notes:
            reasons.append("max_notes reached")
        if reasons:
            rejected.append({"file": path.name, "reasons": reasons})
            continue
        with kb_sync_lock(settings):
            target = _safe_path(dest, path.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        seen.append(str(fm["source_url"]), run_stamp=run_stamp)
        seen_hashes.add(url_hash(str(fm["source_url"])))
        accepted.append(stem)
    return {"accepted": accepted, "rejected": rejected}


def _edit_frontmatter(settings: Settings, inspiration_id: str, mutate) -> dict[str, Any]:
    if not NOTE_ID_RE.match(inspiration_id):
        raise ValueError(f"bad inspiration id {inspiration_id!r}")
    path = _safe_path(inspirations_dir(settings), f"{inspiration_id}.md")
    with kb_sync_lock(settings):
        fm, body = parse_note(path.read_text(encoding="utf-8"))
        mutate(fm)
        path.write_text(render_doc(fm, body), encoding="utf-8")
    return fm


def mark_used(settings: Settings, inspiration_id: str, *, idea_id: int) -> dict[str, Any]:
    def _m(fm):
        leaps = list(fm.get("leaps") or [])
        if idea_id not in leaps:
            leaps.append(idea_id)
        fm["leaps"] = leaps
        if fm.get("status") == "fresh":
            fm["status"] = "used"
    return _edit_frontmatter(settings, inspiration_id, _m)


def mark_exhausted(settings: Settings, inspiration_id: str) -> dict[str, Any]:
    def _m(fm):
        fm["status"] = "exhausted"
    return _edit_frontmatter(settings, inspiration_id, _m)


def list_notes(settings: Settings, *, status: str | None = None,
               limit: int | None = None) -> list[dict[str, Any]]:
    d = inspirations_dir(settings)
    notes = []
    for path in sorted(d.glob("*.md"), reverse=True) if d.exists() else []:
        if not NOTE_ID_RE.match(path.stem):
            continue
        fm, _ = parse_note(path.read_text(encoding="utf-8"))
        if status is None or fm.get("status") == status:
            notes.append(fm)
        if limit is not None and len(notes) >= limit:
            break
    return notes


class SourcesRegistry:
    """`_sources.yaml`: `{venues: [...]}`. Only trusted code writes it."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        data = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        return list(data.get("venues") or [])

    def _save(self, venues: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(yaml.safe_dump({"venues": venues}, sort_keys=False),
                             encoding="utf-8")

    def slice(self, categories: set[str], *, k: int) -> list[dict[str, Any]]:
        hits = [v for v in self.load() if set(v.get("categories") or []) & categories]
        hits.sort(key=lambda v: -(v.get("yield") or {}).get("integrity_yield", 0.0) or 0.0)
        return hits[:k]

    def write_yield(self, venue_key: str, yield_obj: dict[str, Any]) -> None:
        venues = self.load()
        for v in venues:
            if v.get("key") == venue_key:
                v["yield"] = yield_obj
        self._save(venues)

    def propose(self, venue: dict[str, Any]) -> None:
        venues = self.load()
        if any(v.get("key") == venue.get("key") for v in venues):
            return
        venues.append({**venue, "added_by": "forage",
                       "added_at": datetime.now(UTC).date().isoformat()})
        self._save(venues)
