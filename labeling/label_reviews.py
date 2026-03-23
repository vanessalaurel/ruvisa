"""
Review Labeler for Skin Concern Effectiveness + Skin Type Suitability

For each review, determines:
  1. Per-concern effectiveness: positive (+1), negative (-1), neutral (0)
     for: acne, comedonal_acne, pigmentation, acne_scars_texture, pores, redness, wrinkles
  2. Per-skin-type suitability: positive (+1), negative (-1), neutral (0)
     for: dry, oily, sensitive, normal, combination

Approach:
  1. Keyword matching to detect which concerns/skin types a review discusses
  2. Context-aware pattern matching for positive/negative sentiment
  3. Star rating as a secondary signal when keywords are present but
     sentiment direction is ambiguous
"""

import json
import os
import re
from collections import Counter, defaultdict

CONCERNS = [
    "acne", "comedonal_acne", "pigmentation",
    "acne_scars_texture", "pores", "redness", "wrinkles",
]

CONCERN_KEYWORDS = {
    "acne": [
        "acne", "pimple", "pimples", "breakout", "breakouts", "break out",
        "breaking out", "broke out", "blemish", "blemishes", "zit", "zits",
        "cystic acne",
    ],
    "comedonal_acne": [
        "blackhead", "blackheads", "whitehead", "whiteheads",
        "comedone", "comedones", "comedonal", "clogged pore",
        "clogged pores",
    ],
    "pigmentation": [
        "dark spot", "dark spots", "hyperpigmentation", "pigmentation",
        "discoloration", "discolouration", "melasma", "sun spot",
        "sun spots", "uneven tone", "uneven skin tone",
        "dark mark", "dark marks",
    ],
    "acne_scars_texture": [
        "acne scar", "acne scars", "scarring",
        "uneven texture", "skin texture", "rough skin", "roughness",
        "bumpy skin", "ice pick",
    ],
    "pores": [
        "my pore", "my pores", "large pore", "enlarged pore",
        "visible pore", "open pore", "minimize pore", "minimise pore",
        "refine pore", "tighten pore", "shrink pore",
        "clogged pore", "clogged pores",
    ],
    "redness": [
        "redness", "rosacea", "my skin red", "face red",
        "inflamed", "inflammation",
    ],
    "wrinkles": [
        "wrinkle", "wrinkles", "fine line", "fine lines",
        "anti-aging", "anti aging", "antiaging",
        "crow feet", "crow's feet", "laugh line", "laugh lines",
        "sagging",
    ],
}

# Regex-based patterns for concerns that need more context than simple substring
# These are checked after keyword matching, adding more mentions
CONCERN_REGEX = {
    "pigmentation": [
        r"brighten\w*\s+(?:my\s+)?(?:skin|face|complexion|dark)",
    ],
    "acne_scars_texture": [
        r"(?:my|skin)\s+(?:texture|scars?)",
        r"(?:smooth|soften)\w*\s+(?:my\s+)?(?:skin|face)",
    ],
    "pores": [
        r"(?:my|visible|large|big|huge|open|clogged|enlarged)\s+pores?",
        r"pores?\s+(?:look|appear|are|seem|feel)\s+(?:smaller|bigger|larger|cleaner|clearer|tighter)",
    ],
    "redness": [
        r"(?:calm|soothe|reduc)\w*\s+(?:my\s+)?(?:redness|irritation|inflammation)",
        r"(?:skin|face)\s+(?:got|became|turn(?:ed)?|is|was)\s+red",
        r"(?:irritat|sting|burn)\w+\s+(?:my\s+)?(?:skin|face)",
    ],
    "wrinkles": [
        r"(?:firm|plump|tighten)\w*\s+(?:my\s+)?(?:skin|face)",
        r"(?:aging|ageing)\s+(?:skin|concern|sign)",
    ],
}

# {kw} is replaced with the escaped keyword. W+ matches 1-5 intervening words.
W = r"(?:\s+\S+){0,5}\s+"  # up to 5 words gap

POSITIVE_PATTERNS = [
    r"help(?:s|ed)?" + W + r"{kw}",
    r"help(?:s|ed)?\s+(?:with\s+)?(?:my\s+)?{kw}",
    r"reduc(?:e[ds]?|ing)" + W + r"{kw}",
    r"clear(?:s|ed|ing)?\s+(?:up\s+)?" + W + r"{kw}",
    r"(?:got|get|getting)\s+rid\s+of" + W + r"{kw}",
    r"improv(?:e[ds]?|ing)" + W + r"{kw}",
    r"diminish(?:es|ed|ing)?" + W + r"{kw}",
    r"(?:less|fewer|no\s+more|minimize[ds]?)\s+{kw}",
    r"{kw}" + W + r"(?:went\s+away|disappeared|cleared|improved|gone|reduced|faded)",
    r"(?:fad(?:e[ds]?|ing)|lighten(?:s|ed|ing)?)" + W + r"{kw}",
    r"(?:great|good|amazing|excellent|perfect|fantastic|love|wonderful)\s+for\s+{kw}",
    r"no\s+(?:more\s+)?{kw}",
    r"(?:prevent|prevents|prevented|preventing)" + W + r"{kw}",
    r"{kw}\s+(?:are|is|was|were)\s+(?:much\s+)?(?:better|less|smaller|fewer|gone)",
    r"(?:healed?|healing)" + W + r"{kw}",
    r"(?:fight|fights|combat|combats)" + W + r"{kw}",
    r"(?:tighten|minimiz|shrink|refin)\w*" + W + r"{kw}",
    r"(?:smooth|soften)\w*" + W + r"{kw}",
]

NEGATIVE_PATTERNS = [
    r"(?:caus|gave|give|trigger)\w*" + W + r"{kw}",
    r"(?:made|make|making)" + W + r"{kw}\s+worse",
    r"(?:more|increased?|worsen)\s+{kw}",
    r"{kw}\s+(?:got|became|become)\s+worse",
    r"{kw}\s+(?:increased|appeared|flared|worsened)",
    r"(?:didn.t|did\s+not|doesn.t|does\s+not|won.t)\s+help" + W + r"{kw}",
    r"no\s+(?:effect|improvement|change|difference)" + W + r"{kw}",
]

SKIN_TYPES = ["dry", "oily", "sensitive", "normal", "combination"]

# Patterns that detect the reviewer's skin type claim
# e.g. "I have oily skin", "my dry skin", "as someone with sensitive skin"
SKIN_TYPE_CLAIM = {
    "dry": [
        r"(?:i\s+have|my|i.m|i\s+am)\s+(?:\w+\s+){0,2}dry\s+skin",
        r"dry\s+(?:skin\s+)?(?:type|person|girl|guy)",
        r"(?:as\s+)?(?:a\s+)?(?:someone|person)\s+with\s+dry\s+skin",
        r"my\s+skin\s+(?:is|was|tends?\s+to\s+be)\s+(?:very\s+)?dry",
        r"dry[\s/-]+(?:to[\s/-]+)?(?:normal|combo|combination)\s+skin",
        r"normal[\s/-]+(?:to[\s/-]+)?dry\s+skin",
    ],
    "oily": [
        r"(?:i\s+have|my|i.m|i\s+am)\s+(?:\w+\s+){0,2}oily\s+skin",
        r"oily\s+(?:skin\s+)?(?:type|person|girl|guy)",
        r"(?:as\s+)?(?:a\s+)?(?:someone|person)\s+with\s+oily\s+skin",
        r"my\s+skin\s+(?:is|was|tends?\s+to\s+be)\s+(?:very\s+)?oily",
        r"oily[\s/-]+(?:to[\s/-]+)?(?:combo|combination)\s+skin",
    ],
    "sensitive": [
        r"(?:i\s+have|my|i.m|i\s+am)\s+(?:\w+\s+){0,2}sensitive\s+skin",
        r"sensitive\s+(?:skin\s+)?(?:type|person|girl|guy)",
        r"(?:as\s+)?(?:a\s+)?(?:someone|person)\s+with\s+sensitive\s+skin",
        r"my\s+skin\s+(?:is|was)\s+(?:very\s+)?sensitive",
        r"my\s+sensitive\s+skin",
    ],
    "normal": [
        r"(?:i\s+have|my|i.m|i\s+am)\s+(?:\w+\s+){0,2}normal\s+skin",
        r"normal\s+(?:skin\s+)?type",
        r"my\s+skin\s+(?:is|was)\s+normal",
        r"normal[\s/-]+(?:to[\s/-]+)?(?:dry|oily|combo|combination)\s+skin",
    ],
    "combination": [
        r"(?:i\s+have|my|i.m|i\s+am)\s+(?:\w+\s+){0,2}(?:combination|combo)\s+skin",
        r"(?:combination|combo)\s+(?:skin\s+)?(?:type|person|girl|guy)",
        r"(?:as\s+)?(?:a\s+)?(?:someone|person)\s+with\s+(?:combination|combo)\s+skin",
        r"my\s+skin\s+(?:is|was)\s+(?:combination|combo)",
        r"(?:oily|dry)[\s/-]+(?:to[\s/-]+)?(?:combo|combination)\s+skin",
    ],
}

# Patterns indicating the product works/doesn't work for a skin type
SKIN_POSITIVE = [
    r"(?:great|good|perfect|amazing|love|wonderful|ideal|best)\s+(?:for|on)\s+(?:my\s+)?{st}\s+skin",
    r"(?:works?|worked)\s+(?:really\s+)?(?:well|great|perfectly)\s+(?:for|on)\s+(?:my\s+)?{st}\s+skin",
    r"(?:suitable|recommend(?:ed)?)\s+for\s+{st}\s+skin",
    r"{st}\s+skin\s+(?:loves?|approved)",
    r"(?:my\s+)?{st}\s+skin\s+(?:looks?|feels?)\s+(?:amazing|great|better|hydrated|moisturized|balanced|calm|soft|smooth)",
]

SKIN_NEGATIVE = [
    r"(?:not\s+(?:good|great|suitable|ideal)|bad|terrible|horrible|awful)\s+for\s+{st}\s+skin",
    r"(?:too\s+(?:dry|oily|heavy|greasy|rich|light))\s+for\s+(?:my\s+)?{st}\s+skin",
    r"(?:doesn.t|does\s+not|didn.t|did\s+not)\s+work\s+(?:for|on)\s+(?:my\s+)?{st}\s+skin",
    r"(?:my\s+)?{st}\s+skin\s+(?:didn.t|doesn.t|does\s+not)\s+(?:like|tolerate|agree)",
    r"(?:not\s+suitable|not\s+recommend(?:ed)?)\s+for\s+{st}\s+skin",
    r"(?:harsh|irritating|drying|stripping)\s+(?:for|on)\s+(?:my\s+)?{st}\s+skin",
]

GENERAL_POSITIVE = [
    r"love\s+(?:this|it)",
    r"(?:highly|definitely|would)\s+recommend",
    r"holy\s+grail", r"hg\b", r"repurchas",
    r"(?:amazing|incredible|fantastic|wonderful|excellent|perfect)\s+product",
    r"game\s*changer", r"best\s+(?:product|thing|purchase)",
    r"(?:will|going\s+to)\s+(?:buy|repurchase|get)\s+again",
    r"work(?:s|ed)?\s+(?:really\s+)?(?:well|great|amazing|perfectly)",
    r"skin\s+(?:looks?|feels?)\s+(?:amazing|great|better|wonderful|healthy|clear|smooth|soft|glow)",
]

GENERAL_NEGATIVE = [
    r"(?:broke|break)\w*\s+(?:me\s+)?out",
    r"(?:waste|wasted)\s+(?:of\s+)?money",
    r"(?:didn.t|did\s+not|doesn.t|does\s+not)\s+work",
    r"made\s+(?:my\s+)?skin\s+worse",
    r"(?:return|returned)\s+(?:it|this)",
    r"(?:regret|disappointed|disappointing|terrible|horrible|awful)",
    r"(?:allergic|allergy)\s+reaction",
    r"not\s+worth",
    r"(?:do|would)\s+not\s+recommend",
]


def find_concern_mentions(text):
    """Detect which concerns are mentioned in review text."""
    text_lower = text.lower()
    mentioned = {}
    for concern, keywords in CONCERN_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                mentioned[concern] = True
                break
    for concern, patterns in CONCERN_REGEX.items():
        if concern not in mentioned:
            for pat in patterns:
                if re.search(pat, text_lower):
                    mentioned[concern] = True
                    break
    return set(mentioned.keys())


def score_concern_sentiment(text, concern):
    """Score whether review text is positive or negative about a specific concern."""
    text_lower = text.lower()
    keywords = CONCERN_KEYWORDS[concern]
    pos_score = 0
    neg_score = 0

    for kw in keywords:
        if kw not in text_lower:
            continue
        for pattern_template in POSITIVE_PATTERNS:
            pattern = pattern_template.replace("{kw}", re.escape(kw))
            if re.search(pattern, text_lower):
                pos_score += 1

        for pattern_template in NEGATIVE_PATTERNS:
            pattern = pattern_template.replace("{kw}", re.escape(kw))
            if re.search(pattern, text_lower):
                neg_score += 1

    return pos_score, neg_score


def check_general_sentiment(text):
    """Check for general positive/negative sentiment patterns."""
    text_lower = text.lower()
    pos = sum(1 for p in GENERAL_POSITIVE if re.search(p, text_lower))
    neg = sum(1 for p in GENERAL_NEGATIVE if re.search(p, text_lower))
    return pos, neg


def label_skin_type(text, rating):
    """
    Detect which skin type the reviewer claims and whether the product
    was positive or negative for that skin type.

    Returns dict: {skin_type: +1, -1, or 0}
    """
    text_lower = text.lower()
    rating_positive = rating >= 4
    rating_negative = rating <= 2
    labels = {}

    for st in SKIN_TYPES:
        claimed = False
        for pat in SKIN_TYPE_CLAIM[st]:
            if re.search(pat, text_lower):
                claimed = True
                break

        if not claimed:
            labels[st] = 0
            continue

        pos = sum(1 for p in SKIN_POSITIVE if re.search(p.replace("{st}", st), text_lower))
        neg = sum(1 for p in SKIN_NEGATIVE if re.search(p.replace("{st}", st), text_lower))

        if pos > neg:
            labels[st] = 1
        elif neg > pos:
            labels[st] = -1
        elif rating_positive:
            labels[st] = 1
        elif rating_negative:
            labels[st] = -1
        else:
            labels[st] = 0

    return labels


def label_review(review):
    """
    Label a single review with per-concern effectiveness and per-skin-type
    suitability scores.

    Returns (concern_labels, skin_type_labels) where each is a dict of
    {key: +1, -1, or 0}.
    """
    text = (review.get("review_text") or "") + " " + (review.get("headline") or "")
    rating = int(review.get("rating", 3))
    rating_positive = rating >= 4
    rating_negative = rating <= 2

    labels = {}
    mentioned = find_concern_mentions(text)

    for concern in CONCERNS:
        if concern not in mentioned:
            labels[concern] = 0
            continue

        pos_score, neg_score = score_concern_sentiment(text, concern)

        if pos_score > neg_score:
            labels[concern] = 1
        elif neg_score > pos_score:
            labels[concern] = -1
        else:
            if rating_positive:
                labels[concern] = 1
            elif rating_negative:
                labels[concern] = -1
            else:
                labels[concern] = 0

    text_lower = text.lower()
    if re.search(r"(?:broke|break|breaking)\s+(?:me\s+)?out", text_lower):
        negated = re.search(
            r"(?:stop|no\s+more|prevent|didn.t|doesn.t|does\s+not|don.t|do\s+not"
            r"|without|never|won.t|will\s+not|hasn.t|not)\s+(?:\S+\s+){0,3}"
            r"(?:break|broke|breaking)\s+(?:me\s+)?out", text_lower)
        if not negated:
            labels["acne"] = -1

    for concern in mentioned:
        if labels[concern] == 0:
            if rating_positive:
                labels[concern] = 1
            elif rating_negative:
                labels[concern] = -1

    skin_labels = label_skin_type(text, rating)

    return labels, skin_labels


def aggregate_product_scores(labeled_reviews):
    """Aggregate review labels per product into effectiveness + skin type scores."""
    product_data = defaultdict(lambda: {
        "review_count": 0,
        "avg_rating": 0.0,
        "concern_scores": {c: {"positive": 0, "negative": 0, "total_mentions": 0}
                           for c in CONCERNS},
        "skin_type_scores": {st: {"positive": 0, "negative": 0, "total_mentions": 0}
                             for st in SKIN_TYPES},
    })

    for rev in labeled_reviews:
        url = rev["product_url"]
        pd = product_data[url]
        pd["review_count"] += 1
        pd["avg_rating"] += float(rev.get("rating", 0))
        pd["product_brand"] = rev.get("product_brand", "")
        pd["product_title"] = rev.get("product_title", "")

        for concern in CONCERNS:
            val = rev["concern_labels"].get(concern, 0)
            if val == 1:
                pd["concern_scores"][concern]["positive"] += 1
                pd["concern_scores"][concern]["total_mentions"] += 1
            elif val == -1:
                pd["concern_scores"][concern]["negative"] += 1
                pd["concern_scores"][concern]["total_mentions"] += 1

        for st in SKIN_TYPES:
            val = rev.get("skin_type_labels", {}).get(st, 0)
            if val == 1:
                pd["skin_type_scores"][st]["positive"] += 1
                pd["skin_type_scores"][st]["total_mentions"] += 1
            elif val == -1:
                pd["skin_type_scores"][st]["negative"] += 1
                pd["skin_type_scores"][st]["total_mentions"] += 1

    for url, pd in product_data.items():
        if pd["review_count"] > 0:
            pd["avg_rating"] /= pd["review_count"]
        for concern in CONCERNS:
            cs = pd["concern_scores"][concern]
            total = cs["total_mentions"]
            if total > 0:
                cs["effectiveness"] = round(
                    (cs["positive"] - cs["negative"]) / total, 3)
            else:
                cs["effectiveness"] = None
        for st in SKIN_TYPES:
            ss = pd["skin_type_scores"][st]
            total = ss["total_mentions"]
            if total > 0:
                ss["suitability"] = round(
                    (ss["positive"] - ss["negative"]) / total, 3)
            else:
                ss["suitability"] = None

    return dict(product_data)


def main():
    input_path = os.path.join(os.path.dirname(__file__),
                              "..", "data", "raw", "sephora", "reviews", "reviews.jsonl")
    input_path = os.path.normpath(input_path)

    output_reviews = os.path.join(os.path.dirname(__file__), "reviews_labeled.jsonl")
    output_product_scores = os.path.join(os.path.dirname(__file__),
                                         "product_review_scores.json")

    print("=" * 60)
    print("Review Labeler - Skin Concern Effectiveness")
    print("=" * 60)
    print(f"Input:  {input_path}")
    print(f"Output: {output_reviews}")
    print(f"Aggregate: {output_product_scores}")

    reviews = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                reviews.append(json.loads(line))
    print(f"\nLoaded {len(reviews)} reviews")

    labeled = []
    concern_stats = {c: Counter() for c in CONCERNS}
    skin_stats = {st: Counter() for st in SKIN_TYPES}
    total_labeled = 0

    for rev in reviews:
        concern_labels, skin_labels = label_review(rev)
        rev["concern_labels"] = concern_labels
        rev["skin_type_labels"] = skin_labels
        labeled.append(rev)

        has_any = False
        for concern in CONCERNS:
            val = concern_labels[concern]
            if val != 0:
                concern_stats[concern][val] += 1
                has_any = True
        for st in SKIN_TYPES:
            val = skin_labels[st]
            if val != 0:
                skin_stats[st][val] += 1
                has_any = True
        if has_any:
            total_labeled += 1

    with open(output_reviews, "w") as f:
        for rev in labeled:
            f.write(json.dumps(rev, ensure_ascii=False) + "\n")
    print(f"\nLabeled reviews saved to {output_reviews}")

    print(f"\n{'Concern':<22} {'Positive':>8} {'Negative':>8} {'Total':>8}")
    print("-" * 50)
    for concern in CONCERNS:
        pos = concern_stats[concern][1]
        neg = concern_stats[concern][-1]
        total = pos + neg
        print(f"{concern:<22} {pos:>8} {neg:>8} {total:>8}")

    print(f"\n{'Skin Type':<22} {'Positive':>8} {'Negative':>8} {'Total':>8}")
    print("-" * 50)
    for st in SKIN_TYPES:
        pos = skin_stats[st][1]
        neg = skin_stats[st][-1]
        total = pos + neg
        print(f"{st:<22} {pos:>8} {neg:>8} {total:>8}")

    print(f"\nReviews with at least one label: {total_labeled} / {len(reviews)}")

    product_scores = aggregate_product_scores(labeled)
    with open(output_product_scores, "w") as f:
        json.dump(product_scores, f, indent=2, ensure_ascii=False)
    print(f"Product aggregate scores saved to {output_product_scores}")
    print(f"Products with review data: {len(product_scores)}")

    print("\n--- Top products by concern effectiveness ---")
    for concern in CONCERNS:
        ranked = []
        for url, pd in product_scores.items():
            cs = pd["concern_scores"][concern]
            if cs["total_mentions"] >= 3:
                ranked.append((pd["product_title"], cs["effectiveness"],
                               cs["total_mentions"], pd["avg_rating"]))
        ranked.sort(key=lambda x: (-x[1], -x[2]))
        if ranked:
            top = ranked[0]
            print(f"  {concern:<22}: {top[0][:35]} "
                  f"(eff={top[1]:+.2f}, mentions={top[2]}, rating={top[3]:.1f})")


if __name__ == "__main__":
    main()
