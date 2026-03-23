import re
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple, Optional

LESION_LABELS: List[str] = [
    "acne",                # merged: acne_inflammatory + acne_nodulocystic
    "comedonal_acne",      # merged: blackheads + whiteheads
    "pigmentation",
    "acne_scars_texture",
    "pores",
    "redness",
    "wrinkles",
]

# ----------------------------
# Normalization helpers
# ----------------------------

def _norm(x: Any) -> str:
    s = "" if x is None else str(x)
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _join_field(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, list):
        return " | ".join(str(x) for x in v if x is not None)
    return str(v)

def build_text(row: Dict[str, Any]) -> str:
    fields = [
        row.get("title", ""),
        row.get("full_name", ""),
        row.get("description_raw", ""),
        row.get("what_it_is", ""),
        row.get("what_it_does", ""),
        _join_field(row.get("category", [])),
        _join_field(row.get("product_claims", [])),
        row.get("skin_concerns", ""),
        row.get("skin_type", ""),
    ]
    return _norm(" | ".join([_join_field(f) for f in fields if _join_field(f)]))

def parse_concerns(row: Dict[str, Any]) -> Set[str]:
    raw = _norm(row.get("skin_concerns", ""))

    raw = re.split(r"\bfinish\s*:\s*", raw, maxsplit=1)[0].strip()

    if not raw:
        return set()

    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return set(parts)

def parse_skin_types(row: Dict[str, Any]) -> Set[str]:
    st = row.get("skin_type", "")
    if isinstance(st, list):
        raw = ", ".join(str(x) for x in st)
    else:
        raw = str(st or "")
    raw = _norm(raw)
    if not raw:
        return set()
    return {p.strip() for p in raw.split(",") if p.strip()}

# ----------------------------
# Regex compilation + matching
# ----------------------------

_NEGATION_RE = re.compile(
    r"\b(?:no|not|never|without|avoid|unlikely|doesn['']?t|do\s+not|don['']?t|won['']?t|cannot|can['']?t)\b",
    re.IGNORECASE,
)

def _term_to_regex(term: str) -> str:
    t = _norm(term)
    t = re.escape(t)
    t = t.replace(r"\ ", r"[\s\-]+")
    return rf"\b{t}\b"

def compile_terms(terms: List[str]) -> re.Pattern:
    patterns = [_term_to_regex(t) for t in terms if _norm(t)]
    if not patterns:
        return re.compile(r"(?!x)x")
    return re.compile("|".join(patterns), re.IGNORECASE)

def _is_negated(text: str, match_start: int, window_chars: int = 45) -> bool:
    left = text[max(0, match_start - window_chars): match_start]
    return _NEGATION_RE.search(left) is not None

def find_hits(text: str, pattern: re.Pattern, *, allow_negated: bool = False) -> int:
    if not text:
        return 0
    count = 0
    for m in pattern.finditer(text):
        if (not allow_negated) and _is_negated(text, m.start()):
            continue
        count += 1
    return count

# ----------------------------
# Term sets (strong vs weak)
# ----------------------------

TERMS_STRONG: Dict[str, List[str]] = {
    "acne": [
        # inflammatory
        "acne", "breakout", "breakouts", "blemish", "blemishes",
        "pimple", "pimples", "papule", "pustule", "spot treatment",
        "anti-blemish", "blemish control", "helps clear acne", "helps clear breakouts",
        # nodulocystic
        "cystic acne", "cystic", "cysts", "nodular acne", "nodular", "nodule",
        "deep acne", "painful acne",
    ],
    "comedonal_acne": [
        "blackhead", "black heads", "black-head",
        "whitehead", "white heads", "white-head",
        "closed comedones", "comedones", "comedonal",
    ],
    "pigmentation": [
        "hyperpigmentation", "dark spots", "discoloration", "uneven skin tone",
        "pih", "pie", "post-inflammatory hyperpigmentation", "post inflammatory hyperpigmentation",
        "post-inflammatory erythema", "post inflammatory erythema",
        "acne marks", "blemish marks", "brown marks", "red marks",
    ],
    "acne_scars_texture": [
        "acne scars", "acne scarring", "atrophic", "pitted",
        "ice pick", "boxcar", "rolling scar",
    ],
    "pores": [
        "minimize pores", "minimise pores", "pore refining", "refine the look of pores",
        "tighten pores", "decongest", "clogged pores",
    ],
    "redness": [
        "redness", "reduce redness", "reduces redness", "calms redness", "calm redness",
        "rosacea",
    ],
    "wrinkles": [
        "wrinkle", "wrinkles", "fine line", "fine lines",
        "anti-aging", "anti ageing", "anti-ageing",
        "crows feet", "crow's feet",
    ],
}

TERMS_WEAK: Dict[str, List[str]] = {
    "pores": ["pores", "pore"],
    "redness": ["soothe", "soothing", "calming", "calms", "irritation"],
    "wrinkles": ["firming", "firms", "plumping", "elasticity", "loss of firmness"],
    "pigmentation": ["brightening", "brighten", "radiance", "glow"],
    "acne_scars_texture": ["texture", "uneven texture", "uneven skin texture"],
}

PAT_STRONG = {k: compile_terms(v) for k, v in TERMS_STRONG.items()}
PAT_WEAK = {k: compile_terms(v) for k, v in TERMS_WEAK.items()}

# ----------------------------
# Map scraped concerns -> lesion labels
# ----------------------------

CONCERN_TO_LABEL: Dict[str, str] = {
    "breakouts / blemishes": "acne",
    "blackheads": "comedonal_acne",
    "whiteheads": "comedonal_acne",
    "pores": "pores",
    "redness": "redness",
    "fine lines & wrinkles": "wrinkles",
    "fine lines and wrinkles": "wrinkles",
    "ageing": "wrinkles",
    "aging": "wrinkles",
    "pigmentation & dark spots": "pigmentation",
    "uneven skin tone": "pigmentation",
    "uneven skin texture": "acne_scars_texture",
}

def _concerns_to_seed_labels(concerns: Set[str]) -> Set[str]:
    out = set()
    for c in concerns:
        c0 = _norm(c)
        if c0 in CONCERN_TO_LABEL:
            out.add(CONCERN_TO_LABEL[c0])
    return out

# ----------------------------
# Main labeling function
# ----------------------------

@dataclass
class LabelDebug:
    strong_hits: Dict[str, int]
    weak_hits: Dict[str, int]
    seeded_from_concerns: Set[str]

def label_product(row: Dict[str, Any], *, weak_threshold: int = 2, return_debug: bool = False):
    concerns = parse_concerns(row)
    _ = parse_skin_types(row)
    text = build_text(row)

    seeded = _concerns_to_seed_labels(concerns)

    acne_context = (
        ("breakouts / blemishes" in concerns)
        or ("blackheads" in concerns)
        or (find_hits(text, compile_terms(["acne-prone", "acne prone", "acne"]), allow_negated=False) > 0)
    )

    strong_hits: Dict[str, int] = {}
    weak_hits: Dict[str, int] = {}

    for lab in LESION_LABELS:
        strong_hits[lab] = find_hits(text, PAT_STRONG.get(lab, compile_terms([])))
        weak_hits[lab] = find_hits(text, PAT_WEAK.get(lab, compile_terms([])))

    y: Dict[str, int] = {k: 0 for k in LESION_LABELS}

    for lab in seeded:
        y[lab] = 1

    for lab in LESION_LABELS:
        if strong_hits.get(lab, 0) >= 1:
            y[lab] = 1

    if y["pores"] == 0 and weak_hits.get("pores", 0) >= max(weak_threshold, 2):
        y["pores"] = 1

    if y["redness"] == 0 and weak_hits.get("redness", 0) >= max(weak_threshold, 2):
        y["redness"] = 1

    if y["wrinkles"] == 0 and weak_hits.get("wrinkles", 0) >= max(weak_threshold, 2):
        y["wrinkles"] = 1

    if y["pigmentation"] == 0 and weak_hits.get("pigmentation", 0) >= max(weak_threshold, 2):
        y["pigmentation"] = 1

    if y["acne_scars_texture"] == 0:
        if acne_context and weak_hits.get("acne_scars_texture", 0) >= 1:
            y["acne_scars_texture"] = 1
        elif weak_hits.get("acne_scars_texture", 0) >= max(weak_threshold, 2):
            y["acne_scars_texture"] = 1

    if return_debug:
        return y, LabelDebug(strong_hits=strong_hits, weak_hits=weak_hits, seeded_from_concerns=seeded)
    return y

# Backward compat alias
label_9 = label_product

def labels_to_vector(y: Dict[str, int]) -> List[int]:
    return [int(y.get(k, 0)) for k in LESION_LABELS]
