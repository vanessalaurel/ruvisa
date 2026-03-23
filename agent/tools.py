"""LangChain tools wrapping the existing skin analysis and recommender pipeline."""

import json
import logging
import math
import os
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool

import db.crud as crud
from db.database import init_db

init_db()

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABELING_DIR = PROJECT_ROOT / "labeling"

CONCERNS = [
    "acne", "comedonal_acne", "pigmentation",
    "acne_scars_texture", "pores", "redness", "wrinkles",
]

_products_cache: dict | None = None
_reviews_cache: dict | None = None


def invalidate_product_cache() -> None:
    """Clear in-memory catalog cache (e.g. after running migrate_products_to_sqlite.py)."""
    global _products_cache, _reviews_cache
    _products_cache = None
    _reviews_cache = None


def _jsonl_fallback_enabled() -> bool:
    """If false, never read products_evidence_labeled.jsonl (use SQLite only)."""
    return os.environ.get("SKINCARE_JSONL_FALLBACK", "1").lower() not in ("0", "false", "no")


def _load_jsonl_catalog():
    """Load full catalog from JSONL + product_review_scores.json (slow; dev / rescue only)."""
    products = {}
    path = LABELING_DIR / "products_evidence_labeled.jsonl"
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            products[d["product_url"]] = d
    reviews = {}
    rpath = LABELING_DIR / "product_review_scores.json"
    with open(rpath) as f:
        reviews = json.load(f)
    return products, reviews


def _load_product_data():
    """Load products and review scores from SQLite (fast). Optional JSONL fallback if DB empty."""
    global _products_cache, _reviews_cache
    if _products_cache is not None:
        return _products_cache, _reviews_cache

    products: dict = {}
    reviews: dict = {}

    try:
        products = crud.get_all_products()
        reviews = crud.get_all_product_review_scores()
    except Exception:
        logger.exception("Failed to load catalog from SQLite")
        products, reviews = {}, {}

    if products:
        if not reviews:
            rpath = LABELING_DIR / "product_review_scores.json"
            if rpath.exists():
                try:
                    with open(rpath) as f:
                        reviews = json.load(f)
                    logger.info(
                        "Merged %d review-score rows from JSON (SQLite had none)",
                        len(reviews),
                    )
                except Exception:
                    logger.exception("Could not read %s", rpath)
        _products_cache = products
        _reviews_cache = reviews
        logger.info(
            "Catalog from SQLite: %d products, %d review-score rows",
            len(products),
            len(reviews),
        )
        return products, reviews

    # Empty DB
    if not _jsonl_fallback_enabled():
        logger.error(
            "SQLite has no products and SKINCARE_JSONL_FALLBACK is disabled. "
            "Run: python scripts/migrate_products_to_sqlite.py"
        )
        _products_cache, _reviews_cache = {}, {}
        return _products_cache, _reviews_cache

    logger.warning(
        "SQLite catalog empty — loading JSONL (slow). For production, run "
        "scripts/migrate_products_to_sqlite.py and keep SKINCARE_JSONL_FALLBACK=0."
    )
    products, reviews = _load_jsonl_catalog()
    _products_cache = products
    _reviews_cache = reviews
    return products, reviews


def _cosine_sim(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(x * x for x in b)) or 1e-9
    return dot / (na * nb)


def _build_product_vector(product, review_entry):
    ev = product.get("evidence_scores", {})
    cs = review_entry.get("concern_scores", {}) if review_entry else {}
    vec = []
    for c in CONCERNS:
        ing = float(ev.get(c, 0)) or float(product.get(c, 0))
        eff = cs.get(c, {}).get("effectiveness")
        if eff is not None:
            rev = (eff + 1.0) / 2.0
            val = 0.5 * ing + 0.5 * rev
        else:
            val = ing
        vec.append(round(val, 4))
    return vec


def _build_user_vec(acne, wrinkle, pigmentation, pores, redness):
    return [
        acne,
        min(acne, 0.5),
        pigmentation,
        min(acne * 0.3, 0.5),
        pores,
        redness,
        wrinkle,
    ]


def _ingredient_set(product):
    """Lowercase set of evidence-matched ingredients for similarity check."""
    return {i.lower() for i in product.get("evidence_matched_ingredients", [])}


def _ingredient_overlap(set_a, set_b):
    """Jaccard similarity between two ingredient sets."""
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _compute_outcome_penalties(user_id: str):
    """Build per-product penalty map from past outcomes.

    Returns:
        direct_penalties:  {product_url: multiplier}  (0.1 = heavy penalty)
        failed_ingredients: set of ingredient sets from failed products
        worsened_concerns:  {concern_name: delta}  (positive = worsened)
    """
    failed = crud.get_failed_product_urls(user_id)
    products, _ = _load_product_data()

    direct_penalties = {}
    failed_ingredient_sets = []
    worsened_concerns = {}

    for url, info in failed.items():
        outcome = info["outcome"]
        if outcome == "worsened":
            direct_penalties[url] = 0.05
        elif outcome == "mixed":
            direct_penalties[url] = 0.3
        elif outcome == "no_change":
            direct_penalties[url] = 0.4

        if url in products:
            failed_ingredient_sets.append(_ingredient_set(products[url]))

        deltas = info.get("concern_deltas", {})
        for c, d in deltas.items():
            if d > 0.03:
                worsened_concerns[c] = max(worsened_concerns.get(c, 0), d)

    return direct_penalties, failed_ingredient_sets, worsened_concerns


def _adaptive_score(
    url, product, review_entry, user_vec, skin_type,
    direct_penalties, failed_ingredient_sets, worsened_concerns,
    budget,
):
    """Score a single product with adaptive penalties.

    final_score = cosine_sim(boosted_user_vec, product_vec) * modifier

    modifier accounts for:
      1. Direct penalty if this exact product failed before
      2. Ingredient-similarity penalty if product shares ingredients with failed ones
      3. Concern boost so worsened concerns weigh more in matching
    """
    if budget and (product.get("price_value") or float("inf")) > budget:
        return None

    pvec = _build_product_vector(product, review_entry)
    if all(v == 0 for v in pvec):
        return None

    boosted_vec = list(user_vec)
    for c, delta in worsened_concerns.items():
        if c in CONCERNS:
            idx = CONCERNS.index(c)
            boosted_vec[idx] = min(1.0, boosted_vec[idx] * (1.0 + delta * 3))

    sim = _cosine_sim(boosted_vec, pvec)
    if sim <= 0:
        return None

    modifier = direct_penalties.get(url, 1.0)

    if modifier == 1.0 and failed_ingredient_sets:
        prod_ings = _ingredient_set(product)
        for failed_set in failed_ingredient_sets:
            overlap = _ingredient_overlap(prod_ings, failed_set)
            if overlap > 0.6:
                modifier = min(modifier, 0.2)
            elif overlap > 0.4:
                modifier = min(modifier, 0.5)
            elif overlap > 0.2:
                modifier = min(modifier, 0.75)

    final = sim * modifier
    skin_match = bool(product.get(f"skin_{skin_type}", 0))
    cat = product["category"][-1] if product.get("category") else "Unknown"

    return {
        "product_url": url,
        "brand": product.get("brand", ""),
        "title": product.get("title", ""),
        "category": cat,
        "price": product.get("price", ""),
        "price_value": product.get("price_value"),
        "rating": product.get("rating"),
        "base_similarity": round(sim, 4),
        "penalty": round(modifier, 3),
        "adaptive_score": round(final, 4),
        "skin_match": skin_match,
        "evidence_ingredients": product.get("evidence_matched_ingredients", [])[:5],
    }


@tool
def recommend_products(
    user_id: str,
    skin_type: str,
    acne_score: float = 0.0,
    wrinkle_score: float = 0.0,
    pigmentation_score: float = 0.0,
    pores_score: float = 0.0,
    redness_score: float = 0.0,
    budget: Optional[float] = None,
    top_n: int = 5,
) -> str:
    """Recommend skincare products using adaptive scoring.
    Automatically penalizes products that didn't work in past scans and
    boosts concerns that worsened. Provide concern scores 0.0-1.0.
    skin_type: oily, dry, sensitive, normal, combination."""
    products, reviews = _load_product_data()

    user_vec = _build_user_vec(acne_score, wrinkle_score, pigmentation_score,
                               pores_score, redness_score)

    direct_penalties, failed_ings, worsened = _compute_outcome_penalties(user_id)
    has_history = bool(direct_penalties)

    scored = []
    for url, p in products.items():
        rs = reviews.get(url, {})
        result = _adaptive_score(
            url, p, rs, user_vec, skin_type,
            direct_penalties, failed_ings, worsened, budget,
        )
        if result:
            scored.append(result)

    scored.sort(key=lambda x: (-int(x["skin_match"]), -x["adaptive_score"],
                                x["price_value"] or 9999))

    top = scored[:top_n]
    if not top:
        return "No products found matching your criteria."

    header = f"Top {len(top)} recommendations for {skin_type} skin"
    if has_history:
        header += " (adaptive — penalizing previously ineffective products)"
    header += ":\n"

    lines = [header]
    for i, p in enumerate(top, 1):
        match_tag = " [SKIN MATCH]" if p["skin_match"] else ""
        penalty_tag = ""
        if p["penalty"] < 1.0:
            penalty_tag = f" [PENALIZED x{p['penalty']}]"

        lines.append(
            f"{i}. {p['brand']} - {p['title']} ({p['category']}){match_tag}{penalty_tag}\n"
            f"   Price: {p['price']} | Rating: {p['rating']} | "
            f"Score: {p['adaptive_score']:.3f} (base: {p['base_similarity']:.3f})\n"
            f"   Key ingredients: {', '.join(p['evidence_ingredients'][:3])}\n"
        )
    return "\n".join(lines)


@tool
def get_product_info(product_name: str) -> str:
    """Look up detailed info for a specific product by name or brand.
    Returns product details including ingredients, evidence scores, and price."""
    products, reviews = _load_product_data()

    query = product_name.lower()
    matches = []
    for url, p in products.items():
        title = (p.get("title") or "").lower()
        brand = (p.get("brand") or "").lower()
        full = (p.get("full_name") or "").lower()
        if query in title or query in brand or query in full:
            rs = reviews.get(url, {})
            ev = p.get("evidence_scores", {})
            matches.append({
                "brand": p.get("brand"),
                "title": p.get("title"),
                "price": p.get("price"),
                "rating": p.get("rating"),
                "skin_type": p.get("skin_type"),
                "evidence_scores": {c: ev.get(c, 0) for c in CONCERNS},
                "top_ingredients": p.get("evidence_matched_ingredients", [])[:8],
                "review_mentions": rs.get("concern_scores", {}),
            })

    if not matches:
        return f"No products found matching '{product_name}'."

    lines = [f"Found {len(matches)} product(s):\n"]
    for m in matches[:5]:
        lines.append(
            f"- {m['brand']} - {m['title']}\n"
            f"  Price: {m['price']} | Rating: {m['rating']}\n"
            f"  Skin type: {m['skin_type']}\n"
            f"  Evidence scores: {m['evidence_scores']}\n"
            f"  Key ingredients: {', '.join(m['top_ingredients'][:5])}\n"
        )
    return "\n".join(lines)


@tool
def search_products(
    concern: str,
    skin_type: Optional[str] = None,
    max_price: Optional[float] = None,
    min_rating: Optional[float] = None,
    sort_by: str = "evidence",
    limit: int = 10,
) -> str:
    """Search products by skin concern, skin type, price, and rating.
    concern: one of acne, comedonal_acne, pigmentation, acne_scars_texture, pores, redness, wrinkles.
    sort_by: 'evidence' (highest evidence score), 'price' (cheapest), or 'rating' (highest rated)."""
    products, _ = _load_product_data()

    if concern not in CONCERNS:
        return f"Invalid concern '{concern}'. Must be one of: {', '.join(CONCERNS)}"

    results = []
    for url, p in products.items():
        ev = p.get("evidence_scores", {})
        score = ev.get(concern, 0)
        if score <= 0:
            continue

        if skin_type and not p.get(f"skin_{skin_type}", 0):
            continue
        pv = p.get("price_value") or float("inf")
        if max_price and pv > max_price:
            continue
        if min_rating and (p.get("rating") or 0) < min_rating:
            continue

        cat = p["category"][-1] if p.get("category") else "Unknown"
        results.append({
            "brand": p.get("brand", ""),
            "title": p.get("title", ""),
            "category": cat,
            "price_value": p.get("price_value"),
            "price": p.get("price", ""),
            "rating": p.get("rating"),
            "evidence_score": round(score, 3),
            "ingredients": p.get("evidence_matched_ingredients", [])[:3],
        })

    if sort_by == "price":
        results.sort(key=lambda x: x["price_value"] or 9999)
    elif sort_by == "rating":
        results.sort(key=lambda x: -(x["rating"] or 0))
    else:
        results.sort(key=lambda x: -x["evidence_score"])

    top = results[:limit]
    if not top:
        return f"No products found for {concern}."

    lines = [f"Top {len(top)} products for {concern}:\n"]
    for i, r in enumerate(top, 1):
        lines.append(
            f"{i}. {r['brand']} - {r['title']} ({r['category']})\n"
            f"   Evidence: {r['evidence_score']} | "
            f"Price: {r['price']} | Rating: {r['rating']}\n"
        )
    return "\n".join(lines)


@tool
def get_user_profile(user_id: str) -> str:
    """Get a user's profile including skin type, analysis history, and purchase history."""
    user = crud.get_user(user_id)
    if not user:
        return f"No profile found for user '{user_id}'."

    analyses = crud.get_analysis_history(user_id, limit=5)
    purchases = crud.get_purchase_history(user_id, limit=5)

    lines = [
        f"User: {user['name'] or user['user_id']}",
        f"Skin type: {user['skin_type'] or 'not set'}",
        f"Member since: {user['created_at']}",
        f"Total analyses: {len(analyses)}",
    ]

    if analyses:
        latest = analyses[0]
        lines.append(f"\nLatest analysis ({latest['created_at']}):")
        if latest.get("concern_vector"):
            cv = latest["concern_vector"]
            if isinstance(cv, str):
                cv = json.loads(cv)
            for c, v in zip(CONCERNS, cv):
                if v > 0:
                    lines.append(f"  {c}: {v:.2f}")

    if purchases:
        lines.append(f"\nRecent purchases ({len(purchases)}):")
        for p in purchases[:3]:
            lines.append(f"  - {p['product_title']} (${p['price']})")

    return "\n".join(lines)


@tool
def compare_analyses(user_id: str) -> str:
    """Compare a user's latest skin analysis with their previous one to track changes over time."""
    analyses = crud.get_analysis_history(user_id, limit=2)
    if len(analyses) < 2:
        return "Need at least 2 analyses to compare. Only found " + str(len(analyses)) + "."

    latest = analyses[0]
    previous = analyses[1]

    lines = [
        f"Skin progress for {user_id}:",
        f"Previous: {previous['created_at']}",
        f"Latest:   {latest['created_at']}",
        "",
    ]

    cv_latest = latest.get("concern_vector", [])
    cv_prev = previous.get("concern_vector", [])
    if isinstance(cv_latest, str):
        cv_latest = json.loads(cv_latest)
    if isinstance(cv_prev, str):
        cv_prev = json.loads(cv_prev)

    if cv_latest and cv_prev and len(cv_latest) == 7 and len(cv_prev) == 7:
        lines.append("Concern score changes:")
        for c, new, old in zip(CONCERNS, cv_latest, cv_prev):
            delta = new - old
            if abs(delta) > 0.01:
                direction = "improved" if delta < 0 else "worsened"
                lines.append(f"  {c}: {old:.2f} -> {new:.2f} ({direction}, {delta:+.2f})")
            else:
                lines.append(f"  {c}: {new:.2f} (no change)")

    acne_l = latest.get("acne_summary", {})
    acne_p = previous.get("acne_summary", {})
    if isinstance(acne_l, str):
        acne_l = json.loads(acne_l)
    if isinstance(acne_p, str):
        acne_p = json.loads(acne_p)
    if acne_l and acne_p:
        cnt_new = acne_l.get("total_detections", 0)
        cnt_old = acne_p.get("total_detections", 0)
        lines.append(f"\nAcne detections: {cnt_old} -> {cnt_new}")

    wrk_l = latest.get("wrinkle_summary", {})
    wrk_p = previous.get("wrinkle_summary", {})
    if isinstance(wrk_l, str):
        wrk_l = json.loads(wrk_l)
    if isinstance(wrk_p, str):
        wrk_p = json.loads(wrk_p)
    if wrk_l and wrk_p:
        sev_new = wrk_l.get("severity", "N/A")
        sev_old = wrk_p.get("severity", "N/A")
        lines.append(f"Wrinkle severity: {sev_old} -> {sev_new}")

    return "\n".join(lines)


@tool
def track_purchase(
    user_id: str,
    product_name: str,
    price: Optional[float] = None,
) -> str:
    """Record that a user purchased a product. Looks up the product URL by name."""
    products, _ = _load_product_data()

    query = product_name.lower()
    product_url = None
    title = product_name
    for url, p in products.items():
        t = (p.get("title") or "").lower()
        b = (p.get("brand") or "").lower()
        if query in t or query in b:
            product_url = url
            title = f"{p.get('brand', '')} - {p.get('title', '')}"
            if price is None:
                price = p.get("price_value")
            break

    crud.create_user(user_id)
    pid = crud.save_purchase(user_id, product_url or "", title, price)
    return f"Purchase recorded (id={pid}): {title} at ${price or 'N/A'}"


@tool
def evaluate_outcomes(user_id: str) -> str:
    """Evaluate how previous product recommendations affected the user's skin.
    Compares the last two scans, checks which products were recommended/purchased
    between them, and records whether each product improved, worsened, or had
    no effect on skin concerns. Call this after a new scan to update the
    adaptive recommendation engine."""
    result = crud.evaluate_product_outcomes(user_id)

    if result["evaluated_count"] == 0:
        return result.get("reason", "No products to evaluate.")

    lines = [f"Evaluated {result['evaluated_count']} product(s):\n"]

    deltas = result.get("deltas", {})
    worsened = [c for c, d in deltas.items() if d > 0.03]
    improved = [c for c, d in deltas.items() if d < -0.03]

    if worsened:
        lines.append(f"Concerns that WORSENED: {', '.join(worsened)}")
    if improved:
        lines.append(f"Concerns that IMPROVED: {', '.join(improved)}")
    lines.append("")

    for o in result["outcomes"]:
        products, _ = _load_product_data()
        p = products.get(o["product_url"], {})
        name = f"{p.get('brand', '')} - {p.get('title', o['product_url'])}"
        lines.append(f"  - {name}: {o['outcome'].upper()}")

    lines.append(
        "\nThe recommendation engine will now penalize ineffective products "
        "and prioritize alternatives with different ingredient profiles."
    )
    return "\n".join(lines)


@tool
def recommend_routine(
    user_id: str,
    skin_type: str,
    acne_score: float = 0.0,
    wrinkle_score: float = 0.0,
    pigmentation_score: float = 0.0,
    pores_score: float = 0.0,
    redness_score: float = 0.0,
    budget: Optional[float] = None,
) -> str:
    """Recommend an optimized full skincare routine (Cleanser → Toner → Serum → Moisturizer → SPF).
    Products are chosen to maximize concern coverage while minimizing ingredient conflicts.
    Provide concern scores 0.0-1.0 and optional budget."""
    from labeling.routine_optimizer import optimize_routine

    products, reviews = _load_product_data()
    user_vec = _build_user_vec(acne_score, wrinkle_score, pigmentation_score, pores_score, redness_score)

    direct_pen, _, _ = _compute_outcome_penalties(user_id)
    exclude = set(direct_pen.keys()) if direct_pen else None

    result = optimize_routine(
        products, reviews, user_vec, skin_type,
        budget=budget,
        build_product_vector=lambda url, p: _build_product_vector(p, reviews.get(url, {})),
        exclude_urls=exclude,
    )

    routine = result.get("routine", [])
    if not routine:
        return "No routine could be built. Try adjusting your budget or skin type."

    lines = [
        f"Optimized routine (coverage: {result.get('coverage', 0):.2f}, "
        f"conflict penalty: {result.get('conflict_penalty', 0):.2f}, "
        f"total: ${result.get('total_cost', 0):.0f}):\n",
    ]
    for r in routine:
        prod = r.get("product", {})
        if prod:
            lines.append(f"• {r['step']}: {prod.get('brand', '')} - {prod.get('title', '')} ({prod.get('price', 'N/A')})")
            ings = prod.get("evidence_matched_ingredients", [])[:3]
            if ings:
                lines.append(f"  Key ingredients: {', '.join(ings)}\n")
    return "\n".join(lines)


ALL_TOOLS = [
    recommend_products,
    recommend_routine,
    get_product_info,
    search_products,
    get_user_profile,
    compare_analyses,
    track_purchase,
    evaluate_outcomes,
]
