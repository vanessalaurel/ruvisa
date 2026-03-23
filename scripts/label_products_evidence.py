"""
Relabel products using INCIDecoder evidence-based concern lookup
instead of marketing-claim-based labels.

Incorporates INCI position weighting: ingredients listed earlier have
higher concentration per INCI rules, so they receive higher weight.

Reads:
  labeling/concern_lookup.json        – {ingredient_lower: [concerns]}
  labeling/products_labeled.jsonl     – existing product data

Writes:
  labeling/products_evidence_labeled.jsonl – same schema, evidence-based labels
    Binary labels (0/1)        -> used for DeBERTa training
    evidence_scores (0.0-1.0)  -> used for recommendation ranking
"""

import json
import math
import re
from collections import Counter
from pathlib import Path

LABELING_DIR = Path(__file__).resolve().parent.parent / "labeling"
CONCERNS = [
    "acne", "comedonal_acne", "pigmentation",
    "acne_scars_texture", "pores", "redness", "wrinkles",
]


def load_concern_lookup(path: Path) -> dict[str, list[str]]:
    with open(path) as f:
        return json.load(f)


def normalize(s: str) -> str:
    """Lowercase, collapse whitespace, strip non-alphanumeric edges."""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def build_match_index(lookup: dict[str, list[str]]) -> list[tuple[str, list[str]]]:
    """
    Build a sorted list of (normalized_name, concerns) for matching.
    Longer names first so we prefer specific matches.
    """
    items = [(normalize(k), v) for k, v in lookup.items()]
    items.sort(key=lambda x: len(x[0]), reverse=True)
    return items


def match_ingredient(
    raw_ing: str,
    match_index: list[tuple[str, list[str]]],
    min_substr_len: int = 5,
) -> list[str]:
    """
    Match a single product ingredient string against the evidence lookup.
    Returns the list of concerns, or empty list if no match.

    Strategy:
      1. Exact match after normalization
      2. Substring: evidence name found within the product ingredient
      3. Substring: product ingredient found within an evidence name
    """
    normed = normalize(raw_ing)
    if not normed:
        return []

    for evi_name, concerns in match_index:
        if normed == evi_name:
            return concerns

    for evi_name, concerns in match_index:
        if len(evi_name) >= min_substr_len and evi_name in normed:
            return concerns

    for evi_name, concerns in match_index:
        if len(normed) >= min_substr_len and normed in evi_name:
            return concerns

    return []


def inci_position_weight(position: int, total: int) -> float:
    """
    Compute a weight for an ingredient based on its INCI list position.

    INCI rules: ingredients >1% concentration are listed in descending
    order. Ingredients <=1% can appear in any order after that.

    Uses exponential decay so top ingredients get substantially more
    weight. Position 0 (first) -> ~1.0, last -> 0.1 floor.
    """
    if total <= 1:
        return 1.0
    t = position / (total - 1)
    return max(0.1, math.exp(-2.3 * t))


def label_product_evidence(
    product: dict,
    match_index: list[tuple[str, list[str]]],
) -> tuple[dict[str, int], dict[str, float], list[str]]:
    """
    Label a product based on ingredient evidence.
    Returns (binary_labels, position_weighted_scores, matched_ingredients).
    """
    labels = {c: 0 for c in CONCERNS}
    scores = {c: 0.0 for c in CONCERNS}
    matched = []

    ingredients = product.get("ingredients", [])
    if not isinstance(ingredients, list):
        return labels, scores, matched

    total = len(ingredients)

    for idx, raw_ing in enumerate(ingredients):
        if not isinstance(raw_ing, str):
            continue
        concerns = match_ingredient(raw_ing, match_index)
        if concerns:
            w = inci_position_weight(idx, total)
            matched.append(raw_ing)
            for c in concerns:
                if c in labels:
                    labels[c] = 1
                    scores[c] = max(scores[c], w)

    scores = {c: round(v, 4) for c, v in scores.items()}
    return labels, scores, matched


def main():
    lookup_path = LABELING_DIR / "concern_lookup.json"
    products_path = LABELING_DIR / "products_labeled.jsonl"
    output_path = LABELING_DIR / "products_evidence_labeled.jsonl"

    print("Loading concern lookup...")
    lookup = load_concern_lookup(lookup_path)
    match_index = build_match_index(lookup)
    print(f"  {len(match_index)} evidence ingredients loaded")

    print(f"Labeling products from {products_path.name}...")

    products = []
    with open(products_path) as f:
        for line in f:
            line = line.strip()
            if line:
                products.append(json.loads(line))

    agree = Counter()
    disagree = Counter()
    evidence_counts = Counter()
    old_counts = Counter()
    total_matched_ings = 0

    score_sums = {c: [] for c in CONCERNS}

    with open(output_path, "w") as fout:
        for prod in products:
            new_labels, new_scores, matched_ings = label_product_evidence(
                prod, match_index
            )
            total_matched_ings += len(matched_ings)

            for c in CONCERNS:
                old_val = prod.get(c, 0)
                new_val = new_labels[c]
                old_counts[c] += old_val
                evidence_counts[c] += new_val
                if old_val == new_val:
                    agree[c] += 1
                else:
                    disagree[c] += 1

                prod[c] = new_val
                if new_scores[c] > 0:
                    score_sums[c].append(new_scores[c])

            prod["evidence_scores"] = new_scores
            prod["evidence_matched_ingredients"] = matched_ings

            fout.write(json.dumps(prod, ensure_ascii=False) + "\n")

    n = len(products)
    print(f"\nLabeled {n} products -> {output_path.name}")
    print(f"Average matched ingredients per product: {total_matched_ings / n:.1f}")

    print(f"\n{'Concern':<22} {'Old(claim)':>10} {'New(evidence)':>14} {'Agreement':>10} {'Changed':>10}")
    print("-" * 70)
    for c in CONCERNS:
        print(
            f"  {c:<20} {old_counts[c]:>10} {evidence_counts[c]:>14} "
            f"{agree[c]:>9} ({100*agree[c]/n:.0f}%) {disagree[c]:>6} ({100*disagree[c]/n:.0f}%)"
        )

    zero_labels = sum(
        1 for prod in products
        if all(prod.get(c, 0) == 0 for c in CONCERNS)
    )
    print(f"\nProducts with all-zero evidence labels: {zero_labels}/{n}")

    print(f"\n--- INCI Position-Weighted Evidence Scores ---")
    print(f"{'Concern':<22} {'Products':>10} {'Mean Score':>12} {'Min':>8} {'Max':>8}")
    print("-" * 62)
    for c in CONCERNS:
        vals = score_sums[c]
        if vals:
            print(
                f"  {c:<20} {len(vals):>10} {sum(vals)/len(vals):>12.3f} "
                f"{min(vals):>8.3f} {max(vals):>8.3f}"
            )
        else:
            print(f"  {c:<20} {0:>10} {'N/A':>12} {'N/A':>8} {'N/A':>8}")


if __name__ == "__main__":
    main()
