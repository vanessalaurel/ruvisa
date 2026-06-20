#!/usr/bin/env python3
"""Offline export of adaptive ranking rows for paper tables (§4.4).

Uses the same _adaptive_score path as recommend_products / POST /recommend.
Run from repo root: python scripts/export_adaptive_ranking_snapshot.py [user_id]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

# Repo root on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_user_snapshot(user_id: str, db_path: Path) -> tuple[str, list[float]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT skin_type FROM users WHERE user_id = ?", (user_id,)
    ).fetchone()
    skin = (row["skin_type"] if row else None) or "combination"
    a = conn.execute(
        """SELECT concern_vector FROM analyses
           WHERE user_id = ? ORDER BY created_at DESC LIMIT 1""",
        (user_id,),
    ).fetchone()
    conn.close()
    cv: list[float] = []
    if a and a[0]:
        raw = a[0]
        cv = json.loads(raw) if isinstance(raw, str) else list(raw)
    if len(cv) < 7:
        cv = (cv + [0.0] * 7)[:7]
    return skin, cv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("user_id", nargs="?", default="a17ad82ed25544fe")
    parser.add_argument("--top-n", type=int, default=10, help="rows for Table 6")
    parser.add_argument("--table7-k", type=int, default=5, help="SKUs for Table 7")
    args = parser.parse_args()

    db_path = ROOT / "data" / "skincare.db"
    if not db_path.is_file():
        print(f"No database at {db_path}", file=sys.stderr)
        sys.exit(1)

    skin_type, user_vec = _load_user_snapshot(args.user_id, db_path)

    from agent.tools import (
        _adaptive_score,
        _compute_outcome_penalties,
        _load_product_data,
    )

    products, reviews = _load_product_data()
    direct_pen, failed_ings, worsened, direct_boosts, improved_ings = (
        _compute_outcome_penalties(args.user_id)
    )

    scored: list[dict] = []
    for url, p in products.items():
        rs = reviews.get(url, {})
        r = _adaptive_score(
            url,
            p,
            rs,
            list(user_vec),
            skin_type,
            direct_pen,
            failed_ings,
            worsened,
            None,
            direct_boosts,
            improved_ings,
        )
        if r:
            scored.append(
                {
                    "product_url": url,
                    "title": r["title"],
                    "base_similarity": r["base_similarity"],
                    "modifier": r["penalty"],
                    "adaptive_score": r["adaptive_score"],
                    "skin_match": r["skin_match"],
                    "price_value": p.get("price_value"),
                }
            )

    def sort_key_adaptive(x):
        return (-int(x["skin_match"]), -x["adaptive_score"], x["price_value"] or 9999)

    def sort_key_flat(x):
        # modifier treated as 1: rank by cosine only (same boosted user vec as production).
        return (-int(x["skin_match"]), -x["base_similarity"], x["price_value"] or 9999)

    by_adaptive = sorted(scored, key=sort_key_adaptive)
    by_flat = sorted(scored, key=sort_key_flat)

    url_to_rank_adaptive = {x["product_url"]: i + 1 for i, x in enumerate(by_adaptive)}
    url_to_rank_flat = {x["product_url"]: i + 1 for i, x in enumerate(by_flat)}

    print("## Metadata (verbatim for caption)")
    print(f"- user_id: `{args.user_id}`")
    print(f"- skin_type: {skin_type}")
    print(f"- concern_vector (7-dim, latest analysis): {json.dumps(user_vec)}")
    print(f"- catalog size (scored products): {len(scored)}")
    print()

    print("## Table 6 — Ranked products (one user snapshot)")
    print()
    print("| rank | product_title | base_similarity | modifier | adaptive_score | skin_match |")
    print("| ---: | --- | ---: | ---: | ---: | :---: |")
    for i, row in enumerate(by_adaptive[: args.top_n], 1):
        title = row["title"].replace("|", "\\|")
        if len(title) > 70:
            title = title[:67] + "..."
        sm = "yes" if row["skin_match"] else "no"
        print(
            f"| {i} | {title} | {row['base_similarity']:.4f} | "
            f"{row['modifier']:.3f} | {row['adaptive_score']:.4f} | {sm} |"
        )
    print()

    topk = by_adaptive[: args.table7_k]
    print("## Table 7 — Same top-{} SKUs: rank under adaptive vs modifier=1 (cosine-only sort)".format(args.table7_k))
    print()
    print(
        "| adaptive_rank | flat_rank | product_title | base_similarity | "
        "modifier | score_if_mod_1 (=base_sim) | adaptive_score |"
    )
    print("| ---: | ---: | --- | ---: | ---: | ---: | ---: |")
    for i, row in enumerate(topk, 1):
        title = row["title"].replace("|", "\\|")
        if len(title) > 55:
            title = title[:52] + "..."
        fr = url_to_rank_flat[row["product_url"]]
        print(
            f"| {i} | {fr} | {title} | {row['base_similarity']:.4f} | "
            f"{row['modifier']:.3f} | {row['base_similarity']:.4f} | {row['adaptive_score']:.4f} |"
        )
    print()

    # One-line check for prose (reordering)
    first_ad = by_adaptive[0]["product_url"] if by_adaptive else None
    first_fl = by_flat[0]["product_url"] if by_flat else None
    print("## Prose check")
    print(f"- Top-1 product URL (adaptive sort): {first_ad}")
    print(f"- Top-1 product URL (flat modifier=1 sort): {first_fl}")
    print(f"- Top-1 differs: {first_ad != first_fl}")
    any_penalized = any(x["modifier"] < 1.0 for x in topk)
    print(f"- Any of top-{args.table7_k} has modifier < 1: {any_penalized}")


if __name__ == "__main__":
    main()
