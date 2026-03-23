#!/usr/bin/env python3
"""Migrate products + review effectiveness scores into SQLite (data/skincare.db).

The API and recommender read from SQLite first so the app does not parse large
JSONL files on every request. Run this after updating:

  labeling/products_evidence_labeled.jsonl
  labeling/product_review_scores.json

Then set SKINCARE_JSONL_FALLBACK=0 in production to avoid accidental JSONL loads
if the database is empty.
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from db.database import get_db, init_db

PRODUCTS_JSONL = PROJECT_ROOT / "labeling" / "products_evidence_labeled.jsonl"
REVIEW_SCORES_JSON = PROJECT_ROOT / "labeling" / "product_review_scores.json"


def migrate(products_jsonl: Path | None = None, review_scores_json: Path | None = None):
    products_jsonl = products_jsonl or PRODUCTS_JSONL
    review_scores_json = review_scores_json or REVIEW_SCORES_JSON

    init_db()
    conn = get_db()

    # Products
    if not products_jsonl.exists():
        print(f"Products file not found: {products_jsonl}")
        conn.close()
        return
    conn.execute("DELETE FROM products")
    count = 0
    with open(products_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            url = d.get("product_url", "")
            if not url:
                continue
            cat = d.get("category")
            category = cat[-1] if isinstance(cat, list) and cat else (str(cat) if cat else "Unknown")
            conn.execute(
                """INSERT OR REPLACE INTO products (product_url, category, price_value, rating, brand, title, data)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    url,
                    category,
                    d.get("price_value"),
                    d.get("rating"),
                    d.get("brand", ""),
                    d.get("title", ""),
                    json.dumps(d, ensure_ascii=False),
                ),
            )
            count += 1
    conn.commit()
    print(f"Migrated {count} products")

    # Product review scores
    if not review_scores_json.exists():
        print(f"Review scores file not found: {review_scores_json}")
        conn.close()
        return
    conn.execute("DELETE FROM product_review_scores")
    with open(review_scores_json) as f:
        scores = json.load(f)
    for url, data in scores.items():
        if url:
            conn.execute(
                "INSERT OR REPLACE INTO product_review_scores (product_url, data) VALUES (?, ?)",
                (url, json.dumps(data, ensure_ascii=False)),
            )
    conn.commit()
    print(f"Migrated {len(scores)} product review scores")
    conn.close()
    print("Done.")
    print("Restart the API server so in-memory product caches reload from SQLite.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Load products + review scores into SQLite")
    ap.add_argument("--products-jsonl", type=Path, default=None, help="Path to products_evidence_labeled.jsonl")
    ap.add_argument("--review-scores-json", type=Path, default=None, help="Path to product_review_scores.json")
    args = ap.parse_args()
    migrate(products_jsonl=args.products_jsonl, review_scores_json=args.review_scores_json)
