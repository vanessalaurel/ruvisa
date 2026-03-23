"""
Product Ranking Engine v3 -- Cosine Similarity

Scores each product by measuring the alignment between:
  - Ingredient vector (DeBERTa): 7-dim binary [0/1] per concern
  - Review vector (review labeler): 7-dim effectiveness [-1, +1] per concern

High cosine similarity = reviews validate what the ingredients claim to do.
Negative similarity   = reviews contradict the ingredient claims.

Final score:
  product_score = 0.7 * cosine_sim(ingredient_vec, review_vec)
                + 0.3 * (star_rating / 5.0)

Skin type is a SOFT FILTER: matching products shown first, non-matching
shown below with a marker. It does NOT affect the score.

Usage:
  python rank_products.py                                   # all
  python rank_products.py --skin_type oily                  # soft filter
  python rank_products.py --skin_type dry --budget 500      # + budget cap
"""

import argparse
import json
import math
import os
from collections import defaultdict

CONCERNS = [
    "acne", "comedonal_acne", "pigmentation",
    "acne_scars_texture", "pores", "redness", "wrinkles",
]

SKIN_TYPES = ["dry", "oily", "sensitive", "normal", "combination"]

CONCERN_DISPLAY = {
    "acne": "Acne",
    "comedonal_acne": "Comedonal Acne (Blackheads/Whiteheads)",
    "pigmentation": "Pigmentation & Dark Spots",
    "acne_scars_texture": "Acne Scars & Texture",
    "pores": "Pores",
    "redness": "Redness & Irritation",
    "wrinkles": "Wrinkles & Anti-Aging",
}

ALPHA = 0.7   # weight for cosine similarity
BETA  = 0.3   # weight for normalized star rating


# ── Math helpers ───────────────────────────────────────────────────────────────

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def norm(v):
    return math.sqrt(sum(x * x for x in v))


def cosine_similarity(a, b):
    """Cosine similarity between two equal-length vectors. Returns 0 if
    either vector is all zeros."""
    n_a, n_b = norm(a), norm(b)
    if n_a < 1e-10 or n_b < 1e-10:
        return 0.0
    return dot(a, b) / (n_a * n_b)


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data():
    base = os.path.dirname(__file__)
    products = {}
    with open(os.path.join(base, "products_evidence_labeled.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            products[d["product_url"]] = d

    with open(os.path.join(base, "product_review_scores.json")) as f:
        review_scores = json.load(f)

    return products, review_scores


# ── Vector builders ────────────────────────────────────────────────────────────

def build_ingredient_vector(product):
    """7-dim vector from INCI position-weighted evidence scores.
    Falls back to binary label if evidence_scores is missing."""
    ev = product.get("evidence_scores", {})
    return [float(ev.get(c, 0)) or float(product.get(c, 0)) for c in CONCERNS]


def build_review_vector(review_scores_entry):
    """7-dim effectiveness vector from aggregated review labels.
    Missing/null mapped to 0."""
    cs = review_scores_entry.get("concern_scores", {})
    vec = []
    for c in CONCERNS:
        eff = cs.get(c, {}).get("effectiveness")
        vec.append(eff if eff is not None else 0.0)
    return vec


def total_review_mentions(review_scores_entry):
    cs = review_scores_entry.get("concern_scores", {})
    return sum(cs.get(c, {}).get("total_mentions", 0) for c in CONCERNS)


# ── Ranking ────────────────────────────────────────────────────────────────────

def rank_products(products, review_scores, user_skin_type=None, budget=None):
    """Build ranked leaderboards: {concern: {category: [ranked products]}}"""

    # Budget filter
    eligible = {}
    for url, p in products.items():
        if budget and (p.get("price_value") or float("inf")) > budget:
            continue
        eligible[url] = p

    # Pre-compute per-product scores (concern-independent)
    product_scores = {}
    for url, p in eligible.items():
        rs_entry = review_scores.get(url, {})
        i_vec = build_ingredient_vector(p)
        r_vec = build_review_vector(rs_entry)
        cos_sim = cosine_similarity(i_vec, r_vec)
        rating_norm = (p.get("rating") or 0) / 5.0
        score = ALPHA * cos_sim + BETA * rating_norm

        skin_match = None
        if user_skin_type:
            skin_match = bool(p.get(f"skin_{user_skin_type}", 0))

        product_scores[url] = {
            "ingredient_vec": i_vec,
            "review_vec": r_vec,
            "cosine_similarity": cos_sim,
            "rating_norm": rating_norm,
            "product_score": score,
            "skin_match": skin_match,
            "review_mentions": total_review_mentions(rs_entry),
        }

    # Group by category
    by_category = defaultdict(list)
    for url, p in eligible.items():
        cat = p["category"][-1] if p.get("category") else "Unknown"
        by_category[cat].append(p)

    rankings = {}

    for concern in CONCERNS:
        rankings[concern] = {}

        for cat, cat_products in sorted(by_category.items()):
            scored = []
            for p in cat_products:
                url = p["product_url"]
                ps = product_scores[url]

                # Only include products that have EITHER ingredient relevance
                # OR review data for this specific concern
                has_ingredient = p.get(concern, 0) == 1
                rs_entry = review_scores.get(url, {})
                concern_cs = rs_entry.get("concern_scores", {}).get(concern, {})
                has_review = concern_cs.get("total_mentions", 0) > 0

                if not has_ingredient and not has_review:
                    continue

                review_eff = concern_cs.get("effectiveness")
                review_mentions = concern_cs.get("total_mentions", 0)

                entry = {
                    "product_url": url,
                    "brand": p.get("brand", ""),
                    "title": p.get("title", ""),
                    "full_name": p.get("full_name", ""),
                    "category": cat,
                    "price": p.get("price", ""),
                    "price_value": p.get("price_value"),
                    "rating": p.get("rating"),
                    "review_count": p.get("review_count", 0),
                    "ingredient_match": int(has_ingredient),
                    "review_effectiveness": review_eff,
                    "review_mentions": review_mentions,
                    "skin_match": ps["skin_match"],
                    "cosine_similarity": round(ps["cosine_similarity"], 4),
                    "product_score": round(ps["product_score"], 4),
                    "ingredient_vec": [int(x) for x in ps["ingredient_vec"]],
                    "review_vec": [round(x, 3) for x in ps["review_vec"]],
                }
                scored.append(entry)

            # Sort: skin match first (soft filter), then score desc, then price asc
            def sort_key(x):
                skin_priority = 0 if x["skin_match"] is None else (0 if x["skin_match"] else 1)
                return (skin_priority, -x["product_score"], x["price_value"] or 0)

            scored.sort(key=sort_key)

            for rank, item in enumerate(scored, 1):
                item["rank"] = rank

            if scored:
                rankings[concern][cat] = scored

    return rankings


# ── Display ────────────────────────────────────────────────────────────────────

def print_leaderboard(rankings, user_skin_type=None, budget=None, top_n=5):
    print("=" * 100)
    print("PRODUCT RANKING LEADERBOARD (Cosine Similarity)")
    print("Score = 0.7 * cos_sim(ingredient_vec, review_vec) + 0.3 * rating/5")
    if user_skin_type:
        print(f"Skin type: {user_skin_type.upper()} (soft filter — MATCH shown first)")
    if budget:
        print(f"Budget: <= HKD {budget:.0f}")
    print("=" * 100)

    for concern in CONCERNS:
        print(f"\n{'─' * 100}")
        print(f"  {CONCERN_DISPLAY[concern].upper()}")
        print(f"{'─' * 100}")

        concern_data = rankings.get(concern, {})
        if not concern_data:
            print("  No products found for this concern.")
            continue

        for cat, prods in sorted(concern_data.items()):
            top = prods[:top_n]
            if not top:
                continue

            print(f"\n  [{cat}]")
            for p in top:
                ing = "ING" if p["ingredient_match"] else "   "

                if p["review_effectiveness"] is not None and p["review_mentions"] > 0:
                    rev = f"REV={p['review_effectiveness']:+.2f}({p['review_mentions']})"
                else:
                    rev = "REV=n/a"

                skin = ""
                if p["skin_match"] is not None:
                    skin = "MATCH" if p["skin_match"] else "     "

                cos_str = f"cos={p['cosine_similarity']:+.3f}"

                print(
                    f"    #{p['rank']:2d}  score={p['product_score']:.3f}  "
                    f"{cos_str}  {ing}  {rev:<16s} {skin:<5s} "
                    f"★{p['rating'] or 0:.1f}  {p['price']:>10s}  "
                    f"{p['full_name'][:45]}"
                )


def print_summary_table(rankings):
    print("\n" + "=" * 100)
    print("SUMMARY: #1 PRODUCT PER CONCERN x CATEGORY")
    print("=" * 100)

    cats_seen = set()
    for concern in CONCERNS:
        for cat in rankings.get(concern, {}):
            cats_seen.add(cat)
    cats_sorted = sorted(cats_seen)

    header = f"{'Concern':<26}"
    for cat in cats_sorted:
        header += f" | {cat[:14]:<14}"
    print(header)
    print("─" * len(header))

    for concern in CONCERNS:
        row = f"{CONCERN_DISPLAY[concern][:25]:<26}"
        concern_data = rankings.get(concern, {})
        for cat in cats_sorted:
            prods = concern_data.get(cat, [])
            if prods:
                p = prods[0]
                cell = f"{p['brand'][:14]}"
            else:
                cell = "—"
            row += f" | {cell:<14}"
        print(row)


def save_output(rankings, path):
    with open(path, "w") as f:
        json.dump(rankings, f, indent=2, ensure_ascii=False)
    print(f"\nFull rankings saved to {path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Product Recommender — Cosine Similarity Engine")
    parser.add_argument("--skin_type", type=str, default=None,
                        choices=SKIN_TYPES,
                        help="User skin type (soft filter)")
    parser.add_argument("--budget", type=float, default=None,
                        help="Max price in HKD")
    parser.add_argument("--top_n", type=int, default=5,
                        help="Top products per category to display")
    args = parser.parse_args()

    products, review_scores = load_data()
    print(f"Products loaded: {len(products)}")
    print(f"Products with reviews: {len(review_scores)}")

    rankings = rank_products(products, review_scores,
                             user_skin_type=args.skin_type,
                             budget=args.budget)

    total_ranked = sum(
        len(prods)
        for concern_data in rankings.values()
        for prods in concern_data.values()
    )
    print(f"Total ranked entries: {total_ranked}")

    print_leaderboard(rankings, user_skin_type=args.skin_type,
                      budget=args.budget, top_n=args.top_n)
    print_summary_table(rankings)

    output_path = os.path.join(os.path.dirname(__file__), "product_rankings.json")
    save_output(rankings, output_path)


if __name__ == "__main__":
    main()
