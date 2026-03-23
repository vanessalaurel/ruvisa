"""
Multi-Product Routine Optimization

Recommends an entire routine (Cleanser → Toner → Serum → Moisturizer → SPF)
using constrained optimization:
  - Maximize: total concern coverage (each concern addressed by best product)
  - Minimize: ingredient conflicts between products
  - Subject to: budget, one product per category, category diversity

Frame: maximize coverage - λ * conflict_penalty, s.t. cost ≤ budget.
"""

import json
import math
from collections import defaultdict
from pathlib import Path

CONCERNS = [
    "acne", "comedonal_acne", "pigmentation",
    "acne_scars_texture", "pores", "redness", "wrinkles",
]

# Routine steps and their product category mappings (last segment of category path)
ROUTINE_STEPS = ["Cleanser", "Toner", "Serum", "Moisturizer", "SPF"]
CATEGORY_MAP = {
    "Cleanser": ["Facial Cleanser"],
    "Toner": ["Toner"],
    "Serum": ["Serum & Booster"],
    "Moisturizer": ["Day Moisturiser", "Night Cream"],
    "SPF": ["Day Moisturiser", "Serum & Booster", "Facial Mist"],  # SPF in moisturizer/serum/mist
}

# Active ingredients that can conflict (normalized keywords)
# Severity: 1.0 = strong incompatibility, 0.5 = use caution
INGREDIENT_CONFLICTS = [
    ({"retinol", "retinyl", "adapalene", "tretinoin"}, {"benzoyl peroxide"}, 1.0),
    ({"retinol", "retinyl", "adapalene", "tretinoin"}, {"glycolic", "lactic acid", "mandelic", "pha", "gluconolactone"}, 0.9),
    ({"retinol", "retinyl", "adapalene", "tretinoin"}, {"salicylic"}, 0.9),
    ({"ascorbic", "l-ascorbic", "vitamin c", "vitamin c"}, {"benzoyl peroxide"}, 1.0),
    ({"ascorbic", "l-ascorbic", "vitamin c"}, {"niacinamide"}, 0.5),
    ({"glycolic", "lactic acid", "mandelic", "aha"}, {"salicylic", "bha"}, 0.7),
    ({"glycolic", "lactic acid", "mandelic", "aha"}, {"glycolic", "lactic acid", "mandelic", "aha"}, 0.5),
    ({"copper peptide", "copper", "ghk-cu"}, {"ascorbic", "l-ascorbic", "vitamin c"}, 0.8),
]

# Flatten for lookup: each ingredient maps to its "group" name
def _build_conflict_lookup():
    groups = {}
    for g1, g2, sev in INGREDIENT_CONFLICTS:
        for k in g1:
            groups[k] = (g1, "a")
        for k in g2:
            groups[k] = (g2, "b")
    return groups


def _extract_actives(product) -> set:
    """Extract active ingredient keywords from product for conflict check."""
    out = set()
    for src in [product.get("evidence_matched_ingredients", []), product.get("ingredients", [])]:
        for ing in (src or []):
            s = str(ing).lower()
            out.add(s)
            for part in s.replace("-", " ").replace("_", " ").split():
                if len(part) > 3:
                    out.add(part)
    return out


def _conflict_score(actives_a: set, actives_b: set) -> float:
    """Compute conflict severity between two products' active ingredients."""
    total = 0.0
    for g1, g2, sev in INGREDIENT_CONFLICTS:
        has1 = any(any(k in a for k in g1) for a in actives_a)
        has2 = any(any(k in a for k in g2) for a in actives_b)
        if has1 and has2:
            total += sev
    return min(1.0, total)


def _get_product_category(product) -> str | None:
    c = product.get("category")
    if not c:
        return None
    return c[-1] if isinstance(c, list) else str(c)




def _load_products_by_category(products: dict) -> dict[str, list]:
    by_step = defaultdict(list)
    for url, p in products.items():
        cat = _get_product_category(p)
        if not cat:
            continue
        for step, cats in CATEGORY_MAP.items():
            if cat not in cats:
                continue
            if step == "SPF":
                title = (p.get("title") or "").lower()
                if "spf" in title or "sunscreen" in title or "sun screen" in title:
                    by_step[step].append((url, p))
            else:
                by_step[step].append((url, p))
            break
    return dict(by_step)


def _coverage_score(routine_products: list[tuple], user_vec: list[float], build_pvec) -> float:
    """
    Total concern coverage: for each concern, take MAX product score (avoid redundancy).
    Weight by user concern so we prioritize what matters to them.
    """
    if not routine_products:
        return 0.0
    coverage = [0.0] * len(CONCERNS)
    for url, p in routine_products:
        pvec = build_pvec(url, p)
        for i in range(len(CONCERNS)):
            coverage[i] = max(coverage[i], pvec[i])
    return sum(uc * cov for uc, cov in zip(user_vec, coverage))


def _conflict_penalty(routine_products: list[tuple]) -> float:
    """Sum of pairwise conflict scores."""
    if len(routine_products) < 2:
        return 0.0
    actives_list = [_extract_actives(p) for _, p in routine_products]
    total = 0.0
    for i in range(len(actives_list)):
        for j in range(i + 1, len(actives_list)):
            total += _conflict_score(actives_list[i], actives_list[j])
    return total


def _total_cost(routine_products: list[tuple]) -> float:
    return sum(p.get("price_value") or 0 for _, p in routine_products)


def optimize_routine(
    products: dict,
    reviews: dict,
    user_vec: list[float],
    skin_type: str,
    budget: float | None = None,
    top_k_per_step: int = 8,
    lambda_conflict: float = 2.0,
    build_product_vector=None,
    exclude_urls: set | None = None,
) -> dict:
    """
    Find best routine: one product per step maximizing coverage - λ*conflicts.

    Returns:
        {
            "routine": [{"step": str, "product": dict, "product_url": str}, ...],
            "coverage": float,
            "conflict_penalty": float,
            "total_cost": float,
            "score": float
        }
    """
    if build_product_vector is None:
        def _default_pvec(_url, p):
            ev = p.get("evidence_scores", {})
            return [float(ev.get(c, 0)) or float(p.get(c, 0)) for c in CONCERNS]
        build_product_vector = _default_pvec

    by_step = _load_products_by_category(products)

    # Score each product by user fit (cosine sim with user vec)
    def prod_score(url, p):
        pvec = build_product_vector(url, p)
        if all(v == 0 for v in pvec):
            return 0.0
        dot = sum(a * b for a, b in zip(user_vec, pvec))
        na = math.sqrt(sum(a * a for a in user_vec)) or 1e-9
        nb = math.sqrt(sum(b * b for b in pvec)) or 1e-9
        sim = dot / (na * nb)
        skin_match = 1.2 if p.get(f"skin_{skin_type}", 0) else 1.0
        return max(0, sim * skin_match)

    exclude_urls = exclude_urls or set()

    # Top K per step
    candidates = {}
    for step in ROUTINE_STEPS:
        pool = [(url, p) for url, p in by_step.get(step, []) if url not in exclude_urls]
        scored = [(url, p, prod_score(url, p)) for url, p in pool]
        scored.sort(key=lambda x: -x[2])
        candidates[step] = [(url, p) for url, p, _ in scored[:top_k_per_step]]

    # Exhaustive search over combinations (small search space)
    from itertools import product as iter_product

    steps_with_candidates = [s for s in ROUTINE_STEPS if candidates.get(s)]
    if not steps_with_candidates:
        return {"routine": [], "coverage": 0, "conflict_penalty": 0, "total_cost": 0, "score": 0}

    best_score = -1e9
    best_routine = []
    best_coverage = 0.0
    best_conflict = 0.0
    best_cost = 0.0

    for combo in iter_product(*[candidates[s] for s in steps_with_candidates]):
        routine = list(zip(steps_with_candidates, combo))
        products_list = [(url, p) for _, (url, p) in routine]

        cost = _total_cost(products_list)
        if budget is not None and cost > budget:
            continue

        cov = _coverage_score(products_list, user_vec, build_product_vector)
        conflict = _conflict_penalty(products_list)
        score = cov - lambda_conflict * conflict

        if score > best_score:
            best_score = score
            best_routine = routine
            best_coverage = cov
            best_conflict = conflict
            best_cost = cost

    result_routine = []
    for step, (url, p) in best_routine:
        result_routine.append({
            "step": step,
            "product_url": url,
            "product": {
                "product_url": url,
                "brand": p.get("brand"),
                "title": p.get("title"),
                "category": _get_product_category(p),
                "price": p.get("price"),
                "price_value": p.get("price_value"),
                "rating": p.get("rating"),
                "evidence_matched_ingredients": p.get("evidence_matched_ingredients", [])[:5],
                "evidence_scores": p.get("evidence_scores", {}),
            },
        })

    return {
        "routine": result_routine,
        "coverage": round(best_coverage, 4),
        "conflict_penalty": round(best_conflict, 4),
        "total_cost": round(best_cost, 2),
        "score": round(best_score, 4),
    }
