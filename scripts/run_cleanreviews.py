# scripts/run_clean_reviews.py
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional
from urllib.parse import urlparse


WS_RE = re.compile(r"\s+")
ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")


def iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def normalize_text(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s = ZERO_WIDTH_RE.sub("", s)
    s = s.strip()
    s = WS_RE.sub(" ", s)
    return s or None


def parse_iso_datetime(s: Optional[str]) -> Optional[str]:
    """Return ISO string if parseable; otherwise None (don’t crash pipeline)."""
    if not s:
        return None
    s = s.strip()
    # Your timestamps look like: 2025-12-10T04:19:16.000+00:00
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.isoformat()
    except Exception:
        return None


def product_key_from_url(url: str) -> str:
    """
    Make a stable join key from URL path, ignoring query fragments.
    Example: /products/melixir-vegan-lip-butter/v/01-agave-clear
    """
    p = urlparse(url)
    path = (p.path or "").rstrip("/")
    return path.lower()


def review_fingerprint(r: Dict[str, Any]) -> str:
    """
    Dedup key:
    - if provider gave review_id, use it (you don’t have it)
    - else hash stable fields
    """
    base = "|".join([
        (r.get("product_url") or "").strip(),
        (r.get("reviewer_name") or "").strip().lower(),
        (r.get("headline") or "").strip().lower(),
        (r.get("review_text") or "").strip().lower(),
        (r.get("date_published") or r.get("date_created") or "").strip(),
        str(r.get("rating") or "").strip(),
    ])
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


def coerce_rating(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        v = int(x)
        if 1 <= v <= 5:
            return v
        return None
    except Exception:
        return None


def clean_one(r: Dict[str, Any]) -> Dict[str, Any]:
    product_url = (r.get("product_url") or "").strip()
    out = dict(r)

    out["product_url"] = product_url
    out["product_key"] = product_key_from_url(product_url) if product_url else None

    out["product_brand"] = normalize_text(out.get("product_brand"))
    out["product_title"] = normalize_text(out.get("product_title"))
    out["reviewer_name"] = normalize_text(out.get("reviewer_name"))
    out["headline"] = normalize_text(out.get("headline"))
    out["review_text"] = normalize_text(out.get("review_text"))

    out["rating"] = coerce_rating(out.get("rating"))
    out["date_created"] = parse_iso_datetime(out.get("date_created"))
    out["date_published"] = parse_iso_datetime(out.get("date_published"))
    out["scraped_at"] = parse_iso_datetime(out.get("scraped_at")) or out.get("scraped_at")

    # Basic drop-in flags
    out["has_text"] = bool(out.get("review_text")) and len(out["review_text"]) >= 10

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-jsonl", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--min-text-len", type=int, default=10)
    args = ap.parse_args()

    seen = set()
    kept = []

    total = 0
    deduped = 0
    dropped_empty = 0

    for r in iter_jsonl(args.in_jsonl):
        total += 1
        c = clean_one(r)

        fp = review_fingerprint(c)
        if fp in seen:
            deduped += 1
            continue
        seen.add(fp)

        txt = c.get("review_text") or ""
        if txt and len(txt) < args.min_text_len:
            dropped_empty += 1
            continue

        kept.append(c)

    n = write_jsonl(args.out_jsonl, kept)

    print(f"Input: {total}")
    print(f"Kept: {n}")
    print(f"Deduped: {deduped}")
    print(f"Dropped short text (<{args.min_text_len}): {dropped_empty}")


if __name__ == "__main__":
    main()