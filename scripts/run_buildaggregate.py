# scripts/run_build_review_aggregates.py
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List


def iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


# -----------------------------
# Lesion-class + skin-issue patterns
# -----------------------------
# Design:
# - We separate:
#   (A) "conditions mentioned" (acne, nodules, scars, blackheads...)
#   (B) "improvement signals" (reduced, cleared, minimized...)
#   (C) "worsening signals" (caused, worse, breakout...)
#
# Then we create combined labels such as:
#   - pro: "improved_blackheads" = condition mentioned AND improvement wording
#   - con: "worse_blackheads"    = condition mentioned AND worsening wording
#
# This reduces false positives like "I have acne" (mention only) being counted as "acne_help".
#
# Rates are computed over n_with_text (text reviews only).


IMPROVE_RE = re.compile(
    r"\b("
    r"clear(?:ed|ing)?|"
    r"gone|"
    r"improv(?:e|ed|ing|ement)s?|"
    r"reduc(?:e|ed|es|ing)|"
    r"help(?:ed|s)?|"
    r"prevent(?:ed|s|ing)?|"
    r"calm(?:ed|s|ing)?|"
    r"sooth(?:ed|es|ing)?|"
    r"minimi[sz](?:e|ed|es|ing)|"
    r"fade(?:d|s|ing)?|"
    r"flatten(?:ed|s|ing)?|"
    r"heal(?:ed|s|ing)?|"
    r"less(?:en(?:ed|s|ing)?)?"
    r")\b",
    re.I,
)

WORSEN_RE = re.compile(
    r"\b("
    r"worsen(?:ed|s|ing)?|"
    r"worse|"
    r"cause(?:d|s|ing)?|"
    r"trigger(?:ed|s|ing)?|"
    r"flare(?:d|s|ing)?|"
    r"break ?out|broke me out|"
    r"more|"
    r"increas(?:e|ed|es|ing)|"
    r"clog(?:ged|s|ging)?"
    r")\b",
    re.I,
)

NEGATION_RE = re.compile(r"\b(no|not|never|without|didn'?t|doesn'?t|isn'?t|wasn'?t|weren'?t)\b", re.I)

# Condition/lesion mentions (your lesion classes + related terms)
CONDITIONS = {
    "acne": re.compile(r"\b(acne|breakouts?|blemish(?:es)?|pimple(?:s)?)\b", re.I),
    "nodule": re.compile(r"\b(nodule(?:s)?|nodular acne|cyst(?:ic)?|cystic acne)\b", re.I),
    "acne_scars": re.compile(
        r"\b(acne scars?|pitted scars?|atrophic scars?|boxcar scars?|ice[- ]?pick scars?|rolling scars?)\b",
        re.I,
    ),
    "blackhead": re.compile(r"\b(blackhead(?:s)?|black heads?|open comedone(?:s)?)\b", re.I),
    "whitehead": re.compile(r"\b(whitehead(?:s)?|white heads?|closed comedone(?:s)?)\b", re.I),
    "flatwart": re.compile(r"\b(flat wart(?:s)?|flatwart(?:s)?|verruca plana)\b", re.I),
    # extras you mentioned / closely related
    "pores": re.compile(r"\b(pore(?:s)?|pores look smaller|large pores?)\b", re.I),
    "texture": re.compile(r"\b(texture|bumpy|rough|uneven|congestion)\b", re.I),
    "inflammation": re.compile(r"\b(inflamm(?:ation|atory)|inflamed|swollen|swelling)\b", re.I),
    "redness": re.compile(r"\b(red(?:ness)?|erythema|flushed)\b", re.I),
}

# “Clean pores / unclogging” tends to be written without the word "improve"
PORE_CLEAN_RE = re.compile(
    r"\b("
    r"unclog(?:s|ged|ging)? (?:my )?pores?|"
    r"clean(?:ed|s|ing)? (?:out )?(?:my )?pores?|"
    r"pores? (?:feel|look) (?:so )?clean|"
    r"deep[- ]?clean(?:ing)?|"
    r"cleans(?:ed|es)? pores?"
    r")\b",
    re.I,
)

PORE_MINIMIZE_RE = re.compile(
    r"\b("
    r"minimi[sz](?:e|ed|es|ing) pores?|"
    r"pores? (?:look|seem|appear) (?:smaller|minimized|reduced)|"
    r"reduc(?:e|ed|es|ing) (?:the )?appearance of pores?"
    r")\b",
    re.I,
)

SCAR_IMPROVE_RE = re.compile(
    r"\b("
    r"fade(?:d|s|ing)? (?:my )?(?:acne )?scars?|"
    r"lighten(?:ed|s|ing)? (?:my )?(?:acne )?scars?|"
    r"scars? (?:are|is) (?:gone|faded|lighter|less noticeable)|"
    r"improv(?:ed|es|ing) (?:my )?(?:acne )?scars?"
    r")\b",
    re.I,
)

SKIN_TYPES = {
    "normal": re.compile(r"\b(normal (?:skin|type))\b", re.I),

    "dry": re.compile(
        r"\b(dry (?:skin|type)|dehydrat(?:ed|ion)|flak(?:y|ing)|chapped|very dry)\b",
        re.I,
    ),

    "oily": re.compile(
        r"\b(oily (?:skin|type)|oil(?:y|iness)|greas(?:y|iness)|sebum)\b",
        re.I,
    ),

    "combination": re.compile(
        r"\b(combination (?:skin|type)|combo skin)\b",
        re.I,
    ),

    "sensitive": re.compile(
        r"\b(sensitive (?:skin|type)|sensitiz(?:ed|ing)|reactive skin|easily irritat(?:ed|ing))\b",
        re.I,
    ),
}
ATTR = {
    "lightweight_fast_absorb": re.compile(
        r"\b(light(?:weight)?|fast absorb(?:ing)?|quick absorb(?:ing)?|absorbs? quickly|sink(?:s|ing)? in|non[- ]?greasy)\b",
        re.I,
    ),
    "heavy_thick": re.compile(
        r"\b(thick|heavy|occlusive|waxy|balmy|too rich)\b",
        re.I,
    ),
    "greasy_oily_finish": re.compile(
        r"\b(greasy|oily finish|shiny|slick)\b",
        re.I,
    ),
    "hydrating_moisturizing": re.compile(
        r"\b(hydrat(?:e|ing|ion)|moisturiz(?:e|ing|ed)|nourish(?:ing)?|soft(?:en|ens|ened)?)\b",
        re.I,
    ),
    "irritating_sensitive": re.compile(
        r"\b(irritat(?:e|ed|ing)|burn(?:ing|s|t)?|sting(?:ing|s)?|rash|reaction|red(?:ness)?)\b",
        re.I,
    ),
    "fragrance_scent": re.compile(
        r"\b(fragrance|perfume|scent(?:ed)?|smell)\b",
        re.I,
    ),
}
# -----------------------------
# Texture/finish performance signals -> suitability inference
# -----------------------------

OILY_POS_RE = re.compile(
    r"\b("
    r"oil[- ]?free|"
    r"non[- ]?comedogenic|won'?t\s+clog(?:\s+my)?\s+pores?|"
    r"light(?:weight)?|"
    r"fast|quick\s+absorb(?:ing)?|absorbs?\s+quickly|"
    r"sink(?:s|ing)?\s+in|"
    r"non[- ]?greasy|not\s+greasy|"
    r"matte|mattifying|"
    r"oil\s*control|controls?\s+oil|"
    r"less\s+shiny|reduced\s+shine|"
    r"water[- ]?based|gel(?:\s+cream)?|watery|"
    r"foaming\s+cleanser|"
    r"doesn'?t\s+feel\s+heavy"
    r")\b",
    re.I,
)

OILY_NEG_RE = re.compile(
    r"\b("
    r"greasy|oily\s+finish|slick|shiny|"
    r"heavy|thick|too\s+rich|"
    r"waxy|balm(?:y)?|"
    r"clog(?:s|ged|ging)?\s+pores?|"
    r"broke\s+me\s+out|break\s*out|"
    r"congest(?:ed|ion)"
    r")\b",
    re.I,
)

DRY_POS_RE = re.compile(
    r"\b("
    r"rich|thick|"
    r"cream(?:y)?|"
    r"balm(?:y)?|"
    r"occlusive|"
    r"oils?|facial\s+oil|"
    r"deeply\s+hydrating|"
    r"very\s+hydrating|"
    r"moisturiz(?:e|ing|ed)|"
    r"nourish(?:ing)?|"
    r"plump(?:ed|ing)?|"
    r"barrier\s+(?:repair|restore|restoring|strengthen)|"
    r"sealed\s+in\s+moisture|"
    r"help(?:ed)?\s+(?:my\s+)?tight(?:ness)?|"
    r"stopped\s+flak(?:ing)?|help(?:ed)?\s+with\s+flak(?:y|ing)"
    r")\b",
    re.I,
)

DRY_NEG_RE = re.compile(
    r"\b("
    r"dry(?:ing)?|"
    r"tight|"
    r"stripping|"
    r"left\s+my\s+skin\s+dry|"
    r"made\s+my\s+skin\s+dry|"
    r"flak(?:y|ing)\s+(?:after|again)|"
    r"irritat(?:e|ed|ing)\s+my\s+dry\s+skin"
    r")\b",
    re.I,
)

SENSITIVE_POS_RE = re.compile(
    r"\b("
    r"gentle|"
    r"soothing|calm(?:ing)?|"
    r"no\s+irritation|didn'?t\s+irritate|"
    r"no\s+sting(?:ing)?|didn'?t\s+sting|"
    r"no\s+burn(?:ing)?|didn'?t\s+burn|"
    r"fragrance[- ]?free|unscented|"
    r"calmed\s+red(?:ness)?|"
    r"reduced\s+red(?:ness)?"
    r")\b",
    re.I,
)

SENSITIVE_NEG_RE = re.compile(
    r"\b("
    r"irritat(?:e|ed|ing)|"
    r"sting(?:s|ing)?|"
    r"burn(?:s|t|ing)?|"
    r"rash|reaction|"
    r"made\s+me\s+red|"
    r"fragrance|perfume|scent(?:ed)?"
    r")\b",
    re.I,
)

COMBO_POS_RE = re.compile(
    r"\b("
    r"balanced|"
    r"not\s+too\s+(?:dry|oily)|"
    r"lightweight\s+but\s+hydrating|"
    r"hydrates?\s+without\s+(?:grease|greasiness|oil)|"
    r"good\s+for\s+t[- ]?zone|"
    r"t[- ]?zone\s+(?:didn'?t|get)\s+oily|"
    r"cheeks?\s+(?:felt|feel)\s+hydrated"
    r")\b",
    re.I,
)

NORMAL_POS_RE = re.compile(
    r"\b("
    r"everyday|daily|"
    r"works?\s+well\s+for\s+me|"
    r"nice\s+texture|"
    r"absorbs?\s+well|"
    r"layers?\s+well|"
    r"not\s+too\s+heavy|not\s+too\s+light"
    r")\b",
    re.I,
)


def has_negation_near(text: str, m: re.Match, window: int = 20) -> bool:
    """Heuristic: if there's a negation word shortly before the match, treat it as negated."""
    start = max(0, m.start() - window)
    prefix = text[start : m.start()]
    return bool(NEGATION_RE.search(prefix))


def condition_improved(text: str, cond_re: re.Pattern) -> bool:
    """Condition mentioned AND improvement verb present (with simple negation handling)."""
    cm = cond_re.search(text)
    if not cm:
        return False
    im = IMPROVE_RE.search(text)
    if not im:
        return False
    if has_negation_near(text, im):
        return False
    return True


def condition_worsened(text: str, cond_re: re.Pattern) -> bool:
    """Condition mentioned AND worsening verb present (with simple negation handling)."""
    cm = cond_re.search(text)
    if not cm:
        return False
    wm = WORSEN_RE.search(text)
    if not wm:
        return False
    if has_negation_near(text, wm):
        return False
    return True

def mentioned_skin_types(text: str) -> List[str]:
    hits = []
    for st, st_re in SKIN_TYPES.items():
        if st_re.search(text):
            hits.append(st)
    return hits

# Pros/Cons labels built around your lesion classes + pores/texture/inflammation
PRO_CON_RULES = {
    "pros": {
        # direct combined condition+improve
        "acne_improved": lambda t: condition_improved(t, CONDITIONS["acne"]),
        "nodule_inflammation_reduced": lambda t: condition_improved(t, CONDITIONS["nodule"])
        or (CONDITIONS["nodule"].search(t) and CONDITIONS["inflammation"].search(t) and IMPROVE_RE.search(t)),
        "blackheads_improved": lambda t: condition_improved(t, CONDITIONS["blackhead"]),
        "whiteheads_improved": lambda t: condition_improved(t, CONDITIONS["whitehead"]),
        "texture_improved": lambda t: condition_improved(t, CONDITIONS["texture"]),
        "redness_calmed": lambda t: condition_improved(t, CONDITIONS["redness"]),
        "pores_unclogged": lambda t: bool(PORE_CLEAN_RE.search(t)) and not has_negation_near(t, PORE_CLEAN_RE.search(t)),
        "pores_look_smaller": lambda t: bool(PORE_MINIMIZE_RE.search(t)) and not has_negation_near(t, PORE_MINIMIZE_RE.search(t)),
        # scars are tricky; use a more explicit scar-improve pattern
        "acne_scars_improved": lambda t: bool(SCAR_IMPROVE_RE.search(t)) and not has_negation_near(t, SCAR_IMPROVE_RE.search(t)),
        # also allow generic improvement + scars mention (less strict)
        "scars_mentioned_and_improved": lambda t: condition_improved(t, CONDITIONS["acne_scars"]),
    },
    "cons": {
        "acne_worse": lambda t: condition_worsened(t, CONDITIONS["acne"]),
        "nodule_worse": lambda t: condition_worsened(t, CONDITIONS["nodule"]),
        "blackheads_worse": lambda t: condition_worsened(t, CONDITIONS["blackhead"]),
        "whiteheads_worse": lambda t: condition_worsened(t, CONDITIONS["whitehead"]),
        "texture_worse": lambda t: condition_worsened(t, CONDITIONS["texture"]),
        "pores_clogged": lambda t: bool(re.search(r"\b(clog(?:ged|s|ging)? pores?|congest(?:ed|ion))\b", t, re.I))
        and not has_negation_near(t, re.search(r"\b(clog(?:ged|s|ging)? pores?|congest(?:ed|ion))\b", t, re.I)),
        "irritation": lambda t: bool(re.search(r"\b(irritat(?:e|ed|ing)|burn(?:s|t|ing)?|sting(?:s|ing)?|rash)\b", t, re.I)),
        "drying": lambda t: bool(re.search(r"\b(dry(ing)?|tight|stripping|crack(?:s|ed|ing)?)\b", t, re.I)),
    },
}
SKIN_TYPE_RULES = {
    "oily": {
        "pros": {
            "oily_skin_friendly": lambda t: bool(ATTR["lightweight_fast_absorb"].search(t))
            and not has_negation_near(t, ATTR["lightweight_fast_absorb"].search(t)),
        },
        "cons": {
            "oily_skin_unfriendly": lambda t: bool(ATTR["heavy_thick"].search(t) or ATTR["greasy_oily_finish"].search(t)),
        },
    },
    "sensitive": {
        "pros": {
            "sensitive_skin_friendly": lambda t: bool(re.search(r"\b(gentle|soothing|calm(?:ing)?|no irritation|didn'?t irritate)\b", t, re.I))
            and not has_negation_near(t, re.search(r"\b(gentle|soothing|calm(?:ing)?|no irritation|didn'?t irritate)\b", t, re.I)),
        },
        "cons": {
            "sensitive_skin_unfriendly": lambda t: bool(ATTR["irritating_sensitive"].search(t))
            and not has_negation_near(t, ATTR["irritating_sensitive"].search(t)),
        },
    },

    # Optional: dry skin likes hydration/richness; cons if drying/tight
    "dry": {
        "pros": {
            "dry_skin_friendly": lambda t: bool(ATTR["hydrating_moisturizing"].search(t))
            and not has_negation_near(t, ATTR["hydrating_moisturizing"].search(t)),
        },
        "cons": {
            "dry_skin_unfriendly": lambda t: bool(re.search(r"\b(dry(ing)?|tight|stripping)\b", t, re.I))
            and not has_negation_near(t, re.search(r"\b(dry(ing)?|tight|stripping)\b", t, re.I)),
        },
    },
}
INFERRED_SUITABILITY_RULES = {
    "oily_skin_suitable_inferred": lambda t: (
        bool(OILY_POS_RE.search(t)) and not has_negation_near(t, OILY_POS_RE.search(t))
        and not (OILY_NEG_RE.search(t) and not has_negation_near(t, OILY_NEG_RE.search(t)))
    ),
    "dry_skin_suitable_inferred": lambda t: (
        bool(DRY_POS_RE.search(t)) and not has_negation_near(t, DRY_POS_RE.search(t))
        and not (DRY_NEG_RE.search(t) and not has_negation_near(t, DRY_NEG_RE.search(t)))
    ),
    "sensitive_skin_suitable_inferred": lambda t: (
        bool(SENSITIVE_POS_RE.search(t)) and not has_negation_near(t, SENSITIVE_POS_RE.search(t))
        and not (SENSITIVE_NEG_RE.search(t) and not has_negation_near(t, SENSITIVE_NEG_RE.search(t)))
    ),
    "combination_skin_suitable_inferred": lambda t: (
        bool(COMBO_POS_RE.search(t)) and not has_negation_near(t, COMBO_POS_RE.search(t))
    ),
    "normal_skin_suitable_inferred": lambda t: (
        bool(NORMAL_POS_RE.search(t)) and not has_negation_near(t, NORMAL_POS_RE.search(t))
    ),
}


def top_k_from_counts_and_rates(
    counts: Dict[str, int],
    rates: Dict[str, float],
    k: int = 5,
    min_count: int = 1,
) -> List[Dict[str, Any]]:
    items = [{"label": lab, "count": cnt, "rate": rates.get(lab, 0.0)} for lab, cnt in counts.items()]
    items = [x for x in items if x["count"] >= min_count]
    items.sort(key=lambda x: (x["count"], x["rate"]), reverse=True)
    return items[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-clean-jsonl", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--top-k", type=int, default=5, help="How many pros/cons bullets to output per product")
    args = ap.parse_args()

    pros_labels = list(PRO_CON_RULES["pros"].keys())
    cons_labels = list(PRO_CON_RULES["cons"].keys())

    g = defaultdict(
        lambda: {
            "product_key": None,
            "product_url": None,
            "product_brand": None,
            "product_title": None,
            "n": 0,
            "sum_rating": 0.0,
            "n_rating": 0,
            "rating_hist": {str(i): 0 for i in range(1, 6)},
            "n_with_text": 0,
            "sum_text_len": 0,
            # condition mentions (optional but useful)
            "conditions_mentioned": {k: 0 for k in CONDITIONS.keys()},
            # pros/cons
            "pros_count": {k: 0 for k in pros_labels},
            "cons_count": {k: 0 for k in cons_labels},
            "skin_type_mentions": {k: 0 for k in SKIN_TYPES.keys()},

            "skin_type_pros_count": {
                st: {lab: 0 for lab in SKIN_TYPE_RULES.get(st, {}).get("pros", {}).keys()}
                for st in SKIN_TYPES.keys()
            },
            "skin_type_cons_count": {
                st: {lab: 0 for lab in SKIN_TYPE_RULES.get(st, {}).get("cons", {}).keys()}
                for st in SKIN_TYPES.keys()
            },
            "inferred_suitability_count": {k: 0 for k in INFERRED_SUITABILITY_RULES.keys()},
        }
    )

    for r in iter_jsonl(args.in_clean_jsonl):
        pk = r.get("product_key") or r.get("product_url")
        if not pk:
            continue

        agg = g[pk]
        agg["product_key"] = pk
        agg["product_url"] = agg["product_url"] or r.get("product_url")
        agg["product_brand"] = agg["product_brand"] or r.get("product_brand")
        agg["product_title"] = agg["product_title"] or r.get("product_title")

        agg["n"] += 1

        rating = r.get("rating")
        if isinstance(rating, int) and 1 <= rating <= 5:
            agg["sum_rating"] += rating
            agg["n_rating"] += 1
            agg["rating_hist"][str(rating)] += 1

        text = (r.get("review_text") or "").strip()
        # inferred suitability counts (per review)
        
        if not text:
            continue
        
        

        agg["n_with_text"] += 1
        agg["sum_text_len"] += len(text)
        for lab, fn in INFERRED_SUITABILITY_RULES.items():
            try:
                if fn(text):
                    agg["inferred_suitability_count"][lab] += 1
            except Exception:
                pass

        # --- Skin type mentions + skin-type-specific pros/cons (per review) ---
        sts = mentioned_skin_types(text)
        for st in sts:
            agg["skin_type_mentions"][st] += 1

            for lab, fn in SKIN_TYPE_RULES.get(st, {}).get("pros", {}).items():
                try:
                    if fn(text):
                        agg["skin_type_pros_count"][st][lab] += 1
                except Exception:
                    pass

            for lab, fn in SKIN_TYPE_RULES.get(st, {}).get("cons", {}).items():
                try:
                    if fn(text):
                        agg["skin_type_cons_count"][st][lab] += 1
                except Exception:
                    pass

        # Track condition mentions (for your lesion-class analytics)
        for cond_name, cond_re in CONDITIONS.items():
            if cond_re.search(text):
                agg["conditions_mentioned"][cond_name] += 1

        # Pros/cons rules (presence/absence per review)
        for lab, fn in PRO_CON_RULES["pros"].items():
            try:
                if fn(text):
                    agg["pros_count"][lab] += 1
            except Exception:
                # never break aggregation on a rule error
                pass

        for lab, fn in PRO_CON_RULES["cons"].items():
            try:
                if fn(text):
                    agg["cons_count"][lab] += 1
            except Exception:
                pass

    rows: List[Dict[str, Any]] = []
    for pk, agg in g.items():
        n = agg["n"]
        n_rating = agg["n_rating"]
        n_with_text = agg["n_with_text"]

        avg_rating = (agg["sum_rating"] / n_rating) if n_rating else None
        avg_text_len = (agg["sum_text_len"] / n_with_text) if n_with_text else None

        # Rates over text reviews only
        pros_rate = {k: (v / n_with_text) if n_with_text else 0.0 for k, v in agg["pros_count"].items()}
        cons_rate = {k: (v / n_with_text) if n_with_text else 0.0 for k, v in agg["cons_count"].items()}
        conditions_rate = {
            k: (v / n_with_text) if n_with_text else 0.0 for k, v in agg["conditions_mentioned"].items()
        }
        # Skin-type mention rate over text reviews (how often people state their skin type)
        skin_type_mentions_rate = {
            st: (cnt / n_with_text) if n_with_text else 0.0
            for st, cnt in agg["skin_type_mentions"].items()
        }

        # Skin-type pros/cons rates should be over reviews that mention that skin type
        skin_type_pros_rate = {}
        skin_type_cons_rate = {}

        for st in SKIN_TYPES.keys():
            denom = agg["skin_type_mentions"].get(st, 0)
            skin_type_pros_rate[st] = {
                lab: (cnt / denom) if denom else 0.0
                for lab, cnt in agg["skin_type_pros_count"][st].items()
            }
            skin_type_cons_rate[st] = {
                lab: (cnt / denom) if denom else 0.0
                for lab, cnt in agg["skin_type_cons_count"][st].items()
            }

        top_pros = top_k_from_counts_and_rates(agg["pros_count"], pros_rate, k=args.top_k)
        top_cons = top_k_from_counts_and_rates(agg["cons_count"], cons_rate, k=args.top_k)

        rows.append(
            {
                "source": "sephora_hk",
                "product_key": agg["product_key"],
                "product_url": agg["product_url"],
                "product_brand": agg["product_brand"],
                "product_title": agg["product_title"],
                "review_count": n,
                "rating_count": n_rating,
                "avg_rating": avg_rating,
                "rating_hist": agg["rating_hist"],
                "pct_with_text": (n_with_text / n) if n else 0.0,
                "avg_text_len": avg_text_len,
                # lesion-class analytics (mentions, not necessarily improvement)
                "conditions_mentioned_count": agg["conditions_mentioned"],
                "conditions_mentioned_rate": conditions_rate,
                # pros/cons analytics
                "pros_count": agg["pros_count"],
                "pros_rate": pros_rate,
                "cons_count": agg["cons_count"],
                "cons_rate": cons_rate,
                # UI-ready bullets
                "top_pros": top_pros,
                "top_cons": top_cons,
                # ranking
                "confidence": math.log1p(n),
                "skin_type_mentions_count": agg["skin_type_mentions"],
                "skin_type_mentions_rate": {k: (v / n_with_text) if n_with_text else 0.0 for k, v in agg["skin_type_mentions"].items()},
                "skin_type_pros_count": agg["skin_type_pros_count"],
                "skin_type_pros_rate": skin_type_pros_rate,
                "skin_type_cons_count": agg["skin_type_cons_count"],
                "skin_type_cons_rate": skin_type_cons_rate,
                 "inferred_suitability_count": agg["inferred_suitability_count"],
                "inferred_suitability_rate" 
                : {
                                                k: (v / n_with_text) if n_with_text else 0.0
                                                for k, v in agg["inferred_suitability_count"].items()
                                            }
                                           
            }
        )

    rows.sort(key=lambda r: r["review_count"], reverse=True)
    write_jsonl(args.out_jsonl, rows)
    print(f"Wrote {len(rows)} product aggregates -> {args.out_jsonl}")


if __name__ == "__main__":
    main()
