"""
Skincare Knowledge Graph
========================

Makes the ingredient -> function -> concern relationships (already implicit in
``concern_lookup.json`` / ``ingredient_evidence.json``) explicit as a directed
graph, and adds the one thing a flat 7-dim vector cannot encode: ingredient
**synergy** and **conflict** relationships.

The graph is a layer *on top of* the existing cosine recommender. It is used to:

  1. Adjust ranking with a conflict penalty / synergy boost
     (:func:`conflict_synergy_factor`), folded into the adaptive modifier.
  2. Produce human-readable explanations of why a product matches a user
     (:func:`explain_product_match`) for the agentic layer.

Node types (``(kind, key)`` tuples):
    ("concern",    name)    7 skin concerns
    ("function",   name)    INCIDecoder function categories
    ("ingredient", name)    individual INCI ingredients (lowercased)
    ("active",     key)     canonical active groups used for conflict/synergy
    ("product",    url)     catalogue products

Edge ``rel`` values:
    HAS_FUNCTION    ingredient  -> function
    TARGETS         function    -> concern   (weight w)
    CONTAINS        product     -> ingredient (weight w = INCI position weight)
    HAS_ACTIVE      product     -> active
    REVIEW_SUPPORTS product     -> concern   (weight w = review effectiveness)
    CONFLICTS_WITH  active      <-> active    (weight w = severity)
    SYNERGIZES_WITH active      <-> active    (weight w = strength)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import networkx as nx

logger = logging.getLogger(__name__)

LABELING_DIR = Path(__file__).resolve().parent

CONCERNS = [
    "acne", "comedonal_acne", "pigmentation",
    "acne_scars_texture", "pores", "redness", "wrinkles",
]

# INCIDecoder function -> concern(s) with a relevance weight in (0, 1].
# Mirrors the function-to-concern mapping documented for the ingredient analyzer.
FUNCTION_TO_CONCERN: dict[str, dict[str, float]] = {
    "anti-acne": {"acne": 1.0, "comedonal_acne": 1.0},
    "skin-brightening": {"pigmentation": 1.0},
    "soothing": {"redness": 1.0},
    "exfoliant": {"pores": 1.0, "acne_scars_texture": 1.0, "comedonal_acne": 0.7},
    "cell-communicating-ingredient": {"wrinkles": 1.0},
    "astringent": {"pores": 1.0},
    "antioxidant": {"pigmentation": 0.6, "wrinkles": 0.4},
}

# Canonical active groups -> matching keywords (substring, lowercased).
# Used to detect actives robustly despite INCI naming variance.
ACTIVE_KEYWORDS: dict[str, list[str]] = {
    "retinoids": ["retinol", "retinal", "retinyl", "retinaldehyde", "adapalene", "tretinoin"],
    "benzoyl_peroxide": ["benzoyl peroxide"],
    "vitamin_c": ["ascorbic", "l-ascorbic", "ascorbyl", "vitamin c"],
    "aha": ["glycolic", "lactic acid", "mandelic", "aha", "gluconolactone", "pha"],
    "bha": ["salicylic", "bha"],
    "niacinamide": ["niacinamide"],
    "vitamin_e": ["tocopherol", "tocopheryl", "vitamin e"],
    "ferulic_acid": ["ferulic"],
    "zinc": ["zinc pca", "zinc gluconate", "zinc"],
    "hyaluronic_acid": ["hyaluronic", "sodium hyaluronate"],
    "ceramide": ["ceramide"],
    "peptide": ["peptide", "ghk", "copper peptide"],
    "glycerin": ["glycerin", "glycerine"],
    "centella": ["centella", "cica", "madecassoside", "asiaticoside"],
    "panthenol": ["panthenol", "pro-vitamin b5", "provitamin b5"],
}

# Conflict knowledge (canonical active keys, severity, note).
# Severity 1.0 = strong incompatibility, lower = use-with-caution.
INGREDIENT_CONFLICTS: list[tuple[str, str, float, str]] = [
    ("retinoids", "benzoyl_peroxide", 1.0,
     "Benzoyl peroxide can oxidize and deactivate retinoids when layered."),
    ("retinoids", "aha", 0.9,
     "Retinoids combined with AHAs raise irritation and over-exfoliation risk."),
    ("retinoids", "bha", 0.9,
     "Retinoids combined with BHA can over-exfoliate and irritate the skin."),
    ("vitamin_c", "benzoyl_peroxide", 1.0,
     "Benzoyl peroxide oxidizes vitamin C and reduces its efficacy."),
    ("vitamin_c", "niacinamide", 0.5,
     "High-dose vitamin C with niacinamide is debated and may reduce efficacy in some formulas."),
    ("aha", "bha", 0.7,
     "Combining AHA and BHA increases over-exfoliation and barrier-irritation risk."),
    ("peptide", "vitamin_c", 0.8,
     "Copper peptides with vitamin C may reduce each other's stability."),
]

# Synergy knowledge (canonical active keys, strength, note). This is the new
# domain knowledge that a flat ingredient/review vector cannot represent.
INGREDIENT_SYNERGIES: list[tuple[str, str, float, str]] = [
    ("vitamin_c", "vitamin_e", 0.9,
     "Vitamin C and E regenerate each other and strengthen antioxidant protection."),
    ("vitamin_c", "ferulic_acid", 0.9,
     "Ferulic acid stabilizes vitamin C and enhances photoprotection."),
    ("vitamin_e", "ferulic_acid", 0.7,
     "Ferulic acid stabilizes vitamin E in antioxidant systems."),
    ("niacinamide", "zinc", 0.8,
     "Niacinamide with zinc supports oil control for blemish-prone skin."),
    ("niacinamide", "hyaluronic_acid", 0.6,
     "Niacinamide pairs with hyaluronic acid for barrier support and hydration."),
    ("retinoids", "hyaluronic_acid", 0.7,
     "Hyaluronic acid buffers retinoid dryness and irritation."),
    ("retinoids", "niacinamide", 0.6,
     "Niacinamide can reduce retinoid irritation and support the skin barrier."),
    ("bha", "niacinamide", 0.6,
     "Salicylic acid with niacinamide aids oil control with less irritation."),
    ("ceramide", "niacinamide", 0.7,
     "Ceramides with niacinamide strengthen the skin barrier."),
    ("hyaluronic_acid", "glycerin", 0.5,
     "Humectant pairing for layered, longer-lasting hydration."),
    ("centella", "niacinamide", 0.6,
     "Centella with niacinamide soothes redness and supports the barrier."),
    ("panthenol", "hyaluronic_acid", 0.5,
     "Panthenol with hyaluronic acid boosts soothing hydration."),
]

# Tunables for the ranking factor.
_CONFLICT_STEP = 0.12   # per-conflict penalty scaled by severity
_SYNERGY_STEP = 0.08    # per-synergy boost scaled by strength
_FACTOR_FLOOR = 0.70
_FACTOR_CEIL = 1.25


def _normalize(name: str) -> str:
    return str(name).strip().lower()


def _product_actives(product: dict) -> set[str]:
    """Return the set of canonical active keys present in a product."""
    haystack = []
    for src in (product.get("evidence_matched_ingredients"), product.get("ingredients")):
        for ing in (src or []):
            haystack.append(_normalize(ing))
    blob = " | ".join(haystack)
    present = set()
    for key, keywords in ACTIVE_KEYWORDS.items():
        if any(kw in blob for kw in keywords):
            present.add(key)
    return present


def build_kg(products: dict, reviews: dict | None = None) -> nx.DiGraph:
    """Build the skincare knowledge graph from catalogue + evidence data.

    Args:
        products: {url: product_record} (same shape as the recommender loader).
        reviews:  {url: review_score_entry} (optional; adds REVIEW_SUPPORTS edges).
    """
    reviews = reviews or {}
    G = nx.DiGraph()

    for c in CONCERNS:
        G.add_node(("concern", c), kind="concern", label=c)

    # Ingredient -> function -> concern from the evidence mapping.
    evidence_path = LABELING_DIR / "ingredient_evidence.json"
    if evidence_path.exists():
        ingredient_functions = json.loads(evidence_path.read_text())
        for ing, funcs in ingredient_functions.items():
            ing_node = ("ingredient", _normalize(ing))
            G.add_node(ing_node, kind="ingredient", label=ing)
            for f in funcs:
                fn = ("function", f)
                if fn not in G:
                    G.add_node(fn, kind="function", label=f)
                G.add_edge(ing_node, fn, rel="HAS_FUNCTION", w=1.0)
                for concern, w in FUNCTION_TO_CONCERN.get(f, {}).items():
                    G.add_edge(fn, ("concern", concern), rel="TARGETS", w=w)
    else:
        logger.warning("ingredient_evidence.json not found at %s", evidence_path)

    # Active group nodes + conflict/synergy edges (the new knowledge layer).
    for key in ACTIVE_KEYWORDS:
        G.add_node(("active", key), kind="active", label=key)
    for a, b, sev, note in INGREDIENT_CONFLICTS:
        G.add_edge(("active", a), ("active", b), rel="CONFLICTS_WITH", w=sev, note=note)
        G.add_edge(("active", b), ("active", a), rel="CONFLICTS_WITH", w=sev, note=note)
    for a, b, strength, note in INGREDIENT_SYNERGIES:
        G.add_edge(("active", a), ("active", b), rel="SYNERGIZES_WITH", w=strength, note=note)
        G.add_edge(("active", b), ("active", a), rel="SYNERGIZES_WITH", w=strength, note=note)

    # Products: CONTAINS (matched ingredients), HAS_ACTIVE, REVIEW_SUPPORTS.
    for url, p in products.items():
        pnode = ("product", url)
        G.add_node(pnode, kind="product",
                   label=p.get("full_name") or p.get("title") or url)

        matched = p.get("evidence_matched_ingredients") or []
        total = len(p.get("ingredients") or matched) or 1
        for idx, ing in enumerate(matched):
            ing_node = ("ingredient", _normalize(ing))
            if ing_node not in G:
                G.add_node(ing_node, kind="ingredient", label=ing)
            pos_w = round(1.0 - (idx / total), 4)
            G.add_edge(pnode, ing_node, rel="CONTAINS", w=pos_w)

        for key in _product_actives(p):
            G.add_edge(pnode, ("active", key), rel="HAS_ACTIVE", w=1.0)

        cs = (reviews.get(url, {}) or {}).get("concern_scores", {})
        for c, sc in cs.items():
            eff = sc.get("effectiveness")
            if eff is not None and c in CONCERNS:
                G.add_edge(pnode, ("concern", c), rel="REVIEW_SUPPORTS",
                           w=round((eff + 1.0) / 2.0, 4),
                           mentions=sc.get("total_mentions", 0))

    logger.info("Knowledge graph built: %d nodes, %d edges",
                G.number_of_nodes(), G.number_of_edges())
    return G


def conflict_synergy_factor(G: nx.DiGraph, url: str) -> tuple[float, list[str]]:
    """Multiplicative ranking factor from ingredient conflicts/synergies.

    Returns (factor, reasons) where factor < 1 penalizes conflicting products
    and factor > 1 boosts synergistic ones, clamped to [0.70, 1.25].
    """
    pnode = ("product", url)
    if pnode not in G:
        return 1.0, []

    actives = {n[1] for _, n, d in G.out_edges(pnode, data=True)
               if d.get("rel") == "HAS_ACTIVE"}
    if len(actives) < 2:
        return 1.0, []

    factor = 1.0
    reasons: list[str] = []
    seen: set[frozenset] = set()

    for a in actives:
        anode = ("active", a)
        if anode not in G:
            continue
        for _, other, d in G.out_edges(anode, data=True):
            b = other[1]
            if b not in actives:
                continue
            pair = frozenset((a, b))
            if pair in seen:
                continue
            rel = d.get("rel")
            if rel == "CONFLICTS_WITH":
                seen.add(pair)
                factor *= (1.0 - _CONFLICT_STEP * d.get("w", 1.0))
                reasons.append(f"[conflict] {a} + {b}: {d.get('note', '')}")
            elif rel == "SYNERGIZES_WITH":
                seen.add(pair)
                factor *= (1.0 + _SYNERGY_STEP * d.get("w", 1.0))
                reasons.append(f"[synergy] {a} + {b}: {d.get('note', '')}")

    factor = max(_FACTOR_FLOOR, min(_FACTOR_CEIL, factor))
    return round(factor, 4), reasons


def explain_product_match(G: nx.DiGraph, url: str, user_vec: list[float],
                          top_k: int = 3) -> list[str]:
    """Build human-readable explanation paths for why a product matches a user.

    Walks Product -> Ingredient -> Function -> Concern and Product ->
    REVIEW_SUPPORTS -> Concern for the user's most severe concerns.
    """
    pnode = ("product", url)
    if pnode not in G:
        return []

    ranked_concerns = sorted(
        zip(CONCERNS, user_vec), key=lambda x: x[1], reverse=True
    )
    target_concerns = [c for c, v in ranked_concerns if v > 0][:top_k]
    if not target_concerns:
        target_concerns = [c for c, _ in ranked_concerns[:top_k]]

    contained = [n for _, n, d in G.out_edges(pnode, data=True)
                 if d.get("rel") == "CONTAINS"]
    review_support = {n[1]: d for _, n, d in G.out_edges(pnode, data=True)
                      if d.get("rel") == "REVIEW_SUPPORTS"}

    explanations: list[str] = []
    for concern in target_concerns:
        ing_hits: list[str] = []
        for ing_node in contained:
            for _, fn, e1 in G.out_edges(ing_node, data=True):
                if e1.get("rel") != "HAS_FUNCTION":
                    continue
                if G.has_edge(fn, ("concern", concern)):
                    func_label = fn[1]
                    ing_label = G.nodes[ing_node].get("label", ing_node[1])
                    ing_hits.append(f"{ing_label} ({func_label})")
                    break
        parts = []
        if ing_hits:
            uniq = list(dict.fromkeys(ing_hits))[:3]
            parts.append("contains " + ", ".join(uniq))
        if concern in review_support:
            m = review_support[concern].get("mentions", 0)
            parts.append(f"{m} review(s) reported improvement")
        if parts:
            explanations.append(f"For {concern}: " + "; ".join(parts) + ".")

    _, synergy_conflict = conflict_synergy_factor(G, url)
    explanations.extend(synergy_conflict)
    return explanations


_kg_cache: nx.DiGraph | None = None


def get_kg(products: dict, reviews: dict | None = None,
           rebuild: bool = False) -> nx.DiGraph:
    """Return a cached knowledge graph, building it on first use."""
    global _kg_cache
    if _kg_cache is None or rebuild:
        _kg_cache = build_kg(products, reviews)
    return _kg_cache


def invalidate_kg_cache() -> None:
    global _kg_cache
    _kg_cache = None
