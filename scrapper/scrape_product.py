from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import orjson
from selectolax.parser import HTMLParser

from playwright.sync_api import sync_playwright
from scrapper.common import append_jsonl, save_text_gz, utc_now_iso

IN_URLS = Path("data/raw/sephora/urls/product_urls.jsonl")
OUT_PRODUCTS = Path("data/raw/sephora/products/products.jsonl")
HTML_DIR = Path("data/raw/sephora/html/product")


# -----------------------------
# Helpers
# -----------------------------

def _load_urls(path: Path, category_filter: Optional[List[str]] = None) -> List[Tuple[str, str]]:
    """
    Load product URLs from JSONL file.
    
    Args:
        path: Path to the URLs JSONL file
        category_filter: Optional list of category slugs to include (e.g., ["facial-cleanser", "toner"])
    
    Returns:
        List of (url, category) tuples
    """
    urls: List[Tuple[str, str]] = []
    with path.open("rb") as f:
        for line in f:
            if not line.strip():
                continue
            obj = orjson.loads(line)
            u = obj.get("product_url")
            cat = obj.get("category", "unknown")
            
            if not u:
                continue
            
            # Filter by category if specified
            if category_filter and cat not in category_filter:
                continue
            
            urls.append((u, cat))

    # Deduplicate while preserving order
    seen: set[str] = set()
    out: List[Tuple[str, str]] = []
    for u, cat in urls:
        if u in seen:
            continue
        seen.add(u)
        out.append((u, cat))
    return out


def _load_scraped_urls(path: Path) -> set[str]:
    """Load already scraped product URLs to enable resumable scraping."""
    scraped: set[str] = set()
    if not path.exists():
        return scraped
    
    with path.open("rb") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = orjson.loads(line)
                u = obj.get("product_url")
                if u:
                    scraped.add(u)
            except Exception:
                pass
    return scraped


def _text(el) -> str:
    return (el.text() or "").strip() if el else ""


def _first_text(tree: HTMLParser, css: str) -> Optional[str]:
    el = tree.css_first(css)
    t = _text(el)
    return t or None


def _norm_ws(s: str) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s


def _maybe_none(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s2 = s.strip()
    return s2 if s2 else None


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for x in items:
        x2 = x.strip()
        if not x2:
            continue
        k = x2.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(x2)
    return out


# -----------------------------
# Price
# -----------------------------

_PRICE_RE = re.compile(r"(\$)\s*([\d,]+(?:\.\d{2})?)")

def _extract_price(tree: HTMLParser, html: str) -> Tuple[Optional[str], Optional[float], Optional[str]]:
    """
    Returns (price_display, price_value, price_currency).
    Keeps your original string output but also returns numeric & currency.
    """
    # Prefer meta if present (sometimes OG/JSON contains it)
    meta_amount = tree.css_first('meta[property="product:price:amount"], meta[itemprop="price"]')
    meta_curr = tree.css_first('meta[property="product:price:currency"], meta[itemprop="priceCurrency"]')
    if meta_amount:
        amt = meta_amount.attributes.get("content")
        cur = meta_curr.attributes.get("content") if meta_curr else None
        if amt:
            try:
                val = float(amt.replace(",", "").strip())
                display = f"${val:.2f}" if (cur in (None, "", "HKD") and "$" not in (amt or "")) else amt
                return display, val, (cur or "HKD")
            except ValueError:
                pass

    # Try specific price selectors (keep it conservative)
    for el in tree.css('[class*="Price"], [class*="price"], [data-at*="price"]'):
        text = _norm_ws(_text(el))
        m = _PRICE_RE.search(text)
        if m:
            sym, num = m.group(1), m.group(2)
            val = float(num.replace(",", ""))
            return f"{sym}{val:.2f}", val, "HKD"
    
    # Fallback: search anywhere in HTML
    m = _PRICE_RE.search(html)
    if m:
        sym, num = m.group(1), m.group(2)
        val = float(num.replace(",", ""))
        return f"{sym}{val:.2f}", val, "HKD"

    return None, None, None


# -----------------------------
# Brand / Title / Image
# -----------------------------

def _extract_brand(tree: HTMLParser) -> Optional[str]:
    selectors = [
        '[data-at="brand_name"]',
        '.brand-name a',
        '[class*="brand"] a',
        '.product-brand',
        '[class*="brand-name"]',
        'meta[property="product:brand"]',
    ]
    for sel in selectors:
        el = tree.css_first(sel)
        if not el:
            continue
        if el.tag == "meta":
            v = el.attributes.get("content")
            v = v.strip() if v else ""
        else:
            v = _text(el)
        if v and v.lower() not in {"brands", "brand"}:
            return v
    return None


def _extract_image_url(tree: HTMLParser) -> Optional[str]:
    og_img = tree.css_first('meta[property="og:image"]')
    if og_img:
        url = og_img.attributes.get("content")
        if url:
            return url

    selectors = [
        'img[data-at*="product_image"]',
        'img[class*="product"]',
        '.product-image img',
        '[class*="gallery"] img',
    ]
    for sel in selectors:
        el = tree.css_first(sel)
        if el:
            src = el.attributes.get("src") or el.attributes.get("data-src")
            if src:
                return src
    return None


def _extract_variant(url: str) -> Optional[str]:
    match = re.search(r"/v/([^/?\s]+)", url)
    return match.group(1) if match else None


# -----------------------------
# Description
# -----------------------------

def _best_text_from_candidates(tree: HTMLParser, selectors: List[str], min_len: int = 80) -> Optional[str]:
    best = ""
    for sel in selectors:
        for el in tree.css(sel):
            t = _text(el)
            if len(t) > len(best):
                best = t
    best = best.strip()
    return best if len(best) >= min_len else (best or None)


def _extract_description(tree: HTMLParser) -> Dict[str, Any]:
    """
    Extract description_raw plus structured fields.
    Uses broader candidate selectors than just [class*="description"].
    """
    # Note: :contains and :has selectors cause crashes in selectolax
    selectors = [
        '[data-at="product_description"]',
        '[class*="description"]',
        '[class*="Description"]',
    ]
    full_text = _best_text_from_candidates(tree, selectors, min_len=60)
    if not full_text:
        return {}
    
    full_text = full_text.replace("\xa0", " ")
    full_text = re.sub(r"\s+\+\s*View Full Description\s*$", "", full_text, flags=re.I)
    full_text = full_text.strip()
    
    result: Dict[str, Any] = {"description_raw": full_text}
    
    # More robust section parsing: capture until next known label
    def _cap(label: str, next_labels: List[str]) -> Optional[str]:
        # e.g. Skin Type: ... Skin Concerns:
        nxt = "|".join(map(re.escape, next_labels))
        pat = rf"{re.escape(label)}\s*[:\-]\s*(.+?)(?=(?:{nxt})\s*[:\-]|$)"
        m = re.search(pat, full_text, flags=re.I | re.S)
        return _norm_ws(m.group(1)) if m else None

    skin_type = _cap("Skin Type", ["Skin Concerns", "Formulation", "Skincare By Age", "What it is", "What it does"])
    skin_concerns = _cap("Skin Concerns", ["Formulation", "Skincare By Age", "What it is", "What it does"])
    formulation = _cap("Formulation", ["Skincare By Age", "What it is", "What it does"])
    age_range = _cap("Skincare By Age", ["What it is", "What it does"])

    if skin_type:
        result["skin_type"] = skin_type
    if skin_concerns:
        result["skin_concerns"] = skin_concerns
    if formulation:
        result["formulation"] = formulation
    if age_range:
        result["age_range"] = age_range

    what_it_is = _cap("What it is", ["What it does", "How to use", "Ingredients", "Disclaimer"])
    if what_it_is:
        result["what_it_is"] = what_it_is

    what_it_does = _cap("What it does", ["How to use", "Ingredients", "Disclaimer"])
    if what_it_does:
        result["what_it_does"] = what_it_does
    
    return result


# -----------------------------
# Claims & Ingredients
# -----------------------------

# Detect a likely INCI start for mixed-case or uppercase lists.
_INCI_START_RE = re.compile(
    r"\b("
    r"AQUA\s*/\s*WATER\s*/\s*EAU|"
    r"Water\s*\(Aqua\/Eau\)|"
    r"Aqua\s*\(Water\)|"
    r"WATER\b|AQUA\b|EAU\b"
    r")",
    flags=re.I,
)

def _cut_at_ingredients_markers(text: str) -> str:
    """
    Cut a mixed blob before ingredients list begins, to keep claims clean.
    """
    if not text:
        return ""
    t = text

    # First: explicit markers
    markers = [
        "Ingredients:",
        "INGREDIENTS:",
        "+ View Full Ingredients",
        "View Full Ingredients",
    ]
    idxs = [t.find(m) for m in markers if t.find(m) != -1]
    if idxs:
        t = t[: min(idxs)]

    # Second: cut at INCI start if present
    m = _INCI_START_RE.search(t)
    if m:
        # if INCI starts very early, don't cut (it might be the ingredients section)
        if m.start() > 20:
            t = t[: m.start()]

    return t.strip()


def _extract_product_claims(tree: HTMLParser) -> Optional[List[str]]:
    """
    Try to extract claims from badge-like UI elements first.
    Fallback: parse from any "claims" text but cut before ingredients.
    """
    # 1) badge/tag UI (most reliable)
    badge_selectors = [
        '[data-at*="product_badge"]',
        '[class*="Badge"]',
        '[class*="badge"]',
        '[class*="tag"]',
        '[class*="Tag"]',
        '[class*="callout"]',
    ]
    candidates: List[str] = []
    for sel in badge_selectors:
        for el in tree.css(sel):
            t = _norm_ws(_text(el))
            # Filter obvious noise
            if not t or len(t) > 80:
                continue
            # Common claim words
            if any(k in t.lower() for k in ["free", "clean", "vegan", "oil", "sulphate", "sulfate", "fragrance", "paraben", "bha", "aha", "vitamin", "antioxid"]):
                candidates.append(t)

    candidates = _dedupe_keep_order(candidates)
    if candidates:
        return candidates

    # 2) fallback: any text block mentioning claims
    # Note: :contains selector crashes selectolax, removed it
    blob = _best_text_from_candidates(
        tree,
        selectors=[
            '[class*="claim"]',
            '[class*="Claim"]',
        ],
        min_len=10,
    )
    if not blob:
        return None

    blob = _cut_at_ingredients_markers(blob)
    if not blob:
        return None

    # Extract after "Product Claims:" if present
    m = re.search(r"Product Claims\s*[:\-]\s*(.+)", blob, flags=re.I | re.S)
    s = m.group(1) if m else blob
    # Split on commas / bullets / pipes
    parts = re.split(r"[\n•\|\u2022]+|,\s*", s)
    parts = [p.strip(" .;:-") for p in parts if p.strip(" .;:-")]
    parts = [p for p in parts if len(p) <= 60]
    parts = _dedupe_keep_order(parts)
    return parts or None


def _split_ingredients(ingredients_raw: str) -> List[str]:
    """
    Split by commas but keep commas inside parentheses.
    """
    out: List[str] = []
    cur = ""
    depth = 0
    for ch in ingredients_raw:
        if ch == "(":
            depth += 1
            cur += ch
        elif ch == ")":
            depth = max(0, depth - 1)
            cur += ch
        elif ch == "," and depth == 0:
            part = cur.strip()
            if part:
                out.append(part)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def _find_ingredients_text(tree: HTMLParser) -> Optional[str]:
    """
    Locate the ingredients section more reliably than selecting all [class*="ingredient"].
    Strategy:
      1) Look for elements containing "Ingredients" / "View Full Ingredients"
      2) Prefer the longest text that also looks like INCI (has commas + common starters)
    """
    candidates: List[str] = []

    # broad search: elements likely to contain ingredients
    for el in tree.css('section, div, article'):
        t = _text(el)
        if not t or len(t) < 80:
            continue
        if ("Ingredients" in t) or ("INGREDIENTS" in t) or ("View Full Ingredients" in t) or _INCI_START_RE.search(t):
            candidates.append(t)

    if not candidates:
        # fallback to class-based selection
        for el in tree.css('[class*="ingredient"], [class*="Ingredient"]'):
            t = _text(el)
            if t and len(t) >= 80:
                candidates.append(t)

    if not candidates:
        return None

    # score candidates
    def score(t: str) -> int:
        s = 0
        if "Ingredients" in t or "INGREDIENTS" in t:
            s += 50
        if "View Full Ingredients" in t:
            s += 20
        if _INCI_START_RE.search(t):
            s += 50
        if t.count(",") >= 8:
            s += 20
        return s + min(len(t) // 50, 40)

    best = max(candidates, key=score)
    return best


def _is_valid_inci_list(text: str) -> bool:
    """
    Check if text looks like a valid INCI ingredient list.
    Valid INCI lists:
    - Start with common ingredients (case-insensitive)
    - Don't contain HTML/CSS/JavaScript code
    - Don't contain navigation menu text
    - Don't contain prices, product recommendations, or instructions
    """
    if not text or len(text) < 20:
        return False
    
    # Reject if it contains obvious garbage (case-insensitive check)
    garbage_indicators = [
        # HTML/JS/CSS
        "<!doctype", "<html", "<script", "<style", "function()",
        "@media", "!important", "font-family:", "background-color:",
        # URLs
        "http://", "https://", ".com", ".hk",
        # Navigation
        "sign in", "sign up", "add to bag", "shopping", "my account",
        "categories", "brands",
        # Prices and products
        "$", "hkd", "holiday limited edition", "recommended for you",
        # Instructions (How To)
        "massage over", "rinse thoroughly", "use twice a day", "apply to",
        "how to", "ship to", "shipping", "massage gently", "step 1",
        # JSON-LD
        "@context", "@type", "schema.org", "breadcrumblist",
        # Other noise
        "view full", "disclaimer", "planet aware", "clean at sephora",
    ]
    text_lower = text.lower()
    for indicator in garbage_indicators:
        if indicator in text_lower:
            return False
    
    # Check if text starts with common INCI ingredients (case-insensitive)
    text_upper = text.upper().strip()
    # Remove "INGREDIENTS:" prefix if present
    text_upper = re.sub(r"^INGREDIENTS\s*:\s*", "", text_upper)
    
    common_starters = [
        "WATER", "AQUA", "EAU", "ALCOHOL", "GLYCERIN", "ISOPROPYL",
        "CYCLOPENTASILOXANE", "DIMETHICONE", "BUTYLENE", "PROPYLENE",
        "CETYL", "STEARIC", "SODIUM", "POTASSIUM", "CAPRYLIC",
        "HELIANTHUS", "CETEARYL", "GLYCERYL", "SQUALANE", "ISONONYL",
        "OCTYLDODECANOL", "ETHYLHEXYL", "CAPRYL", "POLYGLYCERYL",
        "MINERAL", "PARAFFINUM", "PETROLATUM", "LANOLIN", "ORYZA",
        "PEG-", "PPG-", "BUTYROSPERMUM", "CERA", "OLEA", "RICINUS",
        "PRUNUS", "ROSA", "CAMELLIA", "ARGANIA", "SIMMONDSIA", "PERSEA",
        "COCOS", "THEOBROMA", "MANGIFERA", "VITIS", "CITRUS", "ALOE",
    ]
    starts_with_inci = any(text_upper.startswith(s) for s in common_starters)
    
    # Also check for valid ingredient list pattern: many spaces, some parens (botanical names)
    has_list_pattern = (
        text.count(" ") >= 10 and 
        len(text) >= 50 and
        ("(" in text or text.count(" ") >= 20)
    )
    
    return starts_with_inci or has_list_pattern


def _extract_ingredients(tree: HTMLParser) -> Dict[str, Any]:
    """
    Extract:
      - product_claims (clean list)
      - ingredients_raw (string)
      - ingredients (list)
      - ingredients_count
    """
    result: Dict[str, Any] = {}

    # Extract product claims from the specific Sephora element
    # e.g., "Clean at Sephora, Paraben-free"
    claims_el = tree.css_first('.variant-ingredients-values')
    if claims_el:
        claims_text = _text(claims_el)
        if claims_text:
            claims = [c.strip() for c in claims_text.split(',') if c.strip()]
            if claims:
                result["product_claims"] = claims

    # Extract actual INCI ingredients list from specific Sephora element
    # This is the actual ingredient list like "WATER, STEARIC ACID, GLYCERIN..."
    ing_el = tree.css_first('.product-ingredients-values')
    if ing_el:
        text = _text(ing_el).replace("\xa0", " ")
        text = re.sub(r"\s+", " ", text).strip()
        
        # Strip "Ingredients:" prefix if present (case-insensitive)
        text = re.sub(r"^ingredients\s*:\s*", "", text, flags=re.IGNORECASE)
        
        if text and _is_valid_inci_list(text):
            result["ingredients_raw"] = text
            
            # Only split if comma-separated (standard INCI format)
            # Space-separated lists are kept as raw text only
            if text.count(",") >= 3:
                ings = _split_ingredients(text)
                ings = [re.sub(r"\s+", " ", i).strip() for i in ings]
                ings = [i.rstrip(".;") for i in ings if i]
                if ings:
                    result["ingredients"] = ings
                    result["ingredients_count"] = len(ings)
            else:
                # Space-separated format - keep as raw, estimate count
                # Estimate by counting common suffix words
                word_count = len(text.split())
                # Rough estimate: ingredient names are ~3 words on average
                result["ingredients_count_estimated"] = max(1, word_count // 3)

    # No fallback - only use the specific selector to avoid garbage
    # Some products simply don't have ingredient lists available
    return result


# -----------------------------
# Rating
# -----------------------------

def _extract_rating(tree: HTMLParser, html: str) -> Dict[str, Any]:
    """
    Extract rating + review_count.
    - rating is None if not found (don’t default to 0.0).
    """
    result: Dict[str, Any] = {}

    # 1) meta/itemprop (most reliable)
    rv = tree.css_first('meta[itemprop="ratingValue"]')
    if rv:
        c = rv.attributes.get("content")
        if c:
            try:
                result["rating"] = float(c)
                return result
            except ValueError:
                pass

    # 2) JSON-ish fallback in HTML
    m = re.search(r'"ratingValue"\s*:\s*"?(\d+(?:\.\d+)?)"?', html)
    if m:
        try:
            result["rating"] = float(m.group(1))
        except ValueError:
            pass

    # 3) visible text
    rating_el = tree.css_first('[data-at*="rating"], [class*="rating"], [class*="Rating"], [class*="star"]')
    if rating_el and "rating" not in result:
        t = _norm_ws(_text(rating_el))
        m2 = re.search(r"\b(\d(?:\.\d)?)\b\s*(?:/\s*5)?", t)
        if m2:
            try:
                result["rating"] = float(m2.group(1))
            except ValueError:
                pass

    # Review count (try JSON then visible)
    m3 = re.search(r'"reviewCount"\s*:\s*(\d+)', html)
    if m3:
        result["review_count"] = int(m3.group(1))
    else:
        review_el = tree.css_first('[class*="review-count"], [class*="Review"], [class*="reviews"], [data-at*="review"]')
        if review_el:
            t = _norm_ws(_text(review_el))
            m4 = re.search(r"(\d+)\s*(?:review|reviews|rating|ratings)", t, flags=re.I)
            if m4:
                result["review_count"] = int(m4.group(1))
    
    return result


# -----------------------------
# Category / Breadcrumbs
# -----------------------------

def _extract_category(tree: HTMLParser) -> Optional[List[str]]:
    breadcrumb_els = tree.css('[class*="breadcrumb"] a, nav[aria-label*="breadcrumb"] a')
    if breadcrumb_els:
        categories = [_text(el) for el in breadcrumb_els if _text(el)]
        categories = [c for c in categories if c.lower() not in {"home", "sephora", ""}]
        categories = _dedupe_keep_order(categories)
        if categories:
            return categories
    return None


# -----------------------------
# Main parse
# -----------------------------

def parse_product(html: str, url: str, category: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Parse product HTML and return structured data.
    
    Args:
        html: The HTML content of the product page
        url: The product URL
        category: Optional category path (e.g., ["Skincare", "Cleanser & Exfoliator", "Facial Cleanser"])
                  If not provided, will attempt to extract from page breadcrumbs.
    """
    tree = HTMLParser(html)

    title = (
        _first_text(tree, '[data-at="product_name"]')
        or _first_text(tree, "h1")
        or _first_text(tree, ".product-name")
    )

    og_title = None
    og = tree.css_first('meta[property="og:title"]')
    if og:
        og_title = og.attributes.get("content")
    
    brand = _extract_brand(tree)
    
    # If brand not found, try to infer from og_title vs title
    if not brand and og_title and title and title in og_title:
        maybe = og_title.replace(title, "").strip(" -|")
        brand = maybe or None

    price_display, price_value, price_currency = _extract_price(tree, html)
    variant = _extract_variant(url)
    image_url = _extract_image_url(tree)
    
    # Use provided category or try to extract from page
    if category is None:
        category = _extract_category(tree)

    rating_data = _extract_rating(tree, html)
    desc_data = _extract_description(tree)
    ing_data = _extract_ingredients(tree)
    
    result: Dict[str, Any] = {
        "source": "sephora_hk",
        "product_url": url,
        "brand": brand,
        "title": title,
        "full_name": og_title,
        "price": price_display,
        "price_value": price_value,
        "price_currency": price_currency,
        "variant": variant,
        "image_url": image_url,
        "category": category,
        "scraped_at": utc_now_iso(),
    }
    
    # Only include rating if present
    if "rating" in rating_data:
        result["rating"] = rating_data["rating"]
    if "review_count" in rating_data:
        result["review_count"] = rating_data["review_count"]

    result.update(desc_data)
    result.update(ing_data)
    
    return result


# Category slug to display name mapping
CATEGORY_DISPLAY_NAMES = {
    "facial-cleanser": ["Skincare", "Cleanser & Exfoliator", "Facial Cleanser"],
    "scrub-and-exfoliator": ["Skincare", "Cleanser & Exfoliator", "Scrub & Exfoliator"],
    "toner": ["Skincare", "Toner"],
    "day-moisturiser": ["Skincare", "Moisturiser", "Day Moisturiser"],
    "night-cream": ["Skincare", "Moisturiser", "Night Cream"],
    "facial-mist": ["Skincare", "Moisturiser", "Facial Mist"],
    "mask": ["Skincare", "Masks & Treatments", "Mask"],
    "serum-and-booster": ["Skincare", "Masks & Treatments", "Serum & Booster"],
    "face-oil": ["Skincare", "Masks & Treatments", "Face Oil"],
    "eye-care": ["Skincare", "Masks & Treatments", "Eye Care"],
    "lip-care": ["Skincare", "Masks & Treatments", "Lip Care"],
    "suncare": ["Skincare", "Sun Care"],
}


def scrape_products(
    limit: Optional[int] = None,
    save_html: bool = True,
    category: Optional[List[str]] = None,
    category_filter: Optional[List[str]] = None,
    resume: bool = True,
) -> None:
    """
    Scrape product details from URLs in the product_urls.jsonl file.
    
    Args:
        limit: Maximum number of products to scrape (None = all)
        save_html: Whether to save HTML to disk
        category: Category path to assign to all products (overrides auto-detect)
                  e.g., ["Skincare", "Cleanser & Exfoliator", "Facial Cleanser"]
        category_filter: List of category slugs to scrape (e.g., ["facial-cleanser", "toner"])
                         If None, scrapes all categories
        resume: If True, skip already scraped URLs (default: True)
    """
    import time
    import random
    
    # Load URLs with optional category filter
    url_data = _load_urls(IN_URLS, category_filter=category_filter)
    
    # Load already scraped URLs for resumable scraping
    scraped_urls: set[str] = set()
    if resume:
        scraped_urls = _load_scraped_urls(OUT_PRODUCTS)
        print(f"Loaded {len(scraped_urls)} already scraped URLs")
    
    # Filter out already scraped
    url_data = [(u, cat) for u, cat in url_data if u not in scraped_urls]
    
        if limit:
        url_data = url_data[:limit]

    total = len(url_data)
    if total == 0:
        print("No new products to scrape!")
        return
    
    filter_str = ", ".join(category_filter) if category_filter else "all"
    print(f"Scraping {total} products (filter: {filter_str})...")
    
    success_count = 0
    fail_count = 0
    
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)
        
        for i, (url, cat_slug) in enumerate(url_data):
            # Determine category to use
            if category:
                product_category = category
            else:
                product_category = CATEGORY_DISPLAY_NAMES.get(cat_slug, [cat_slug])
            
            print(f"[{i+1}/{total}] {url.split('/')[-2][:40]}...")
            
            try:
                # Polite delay
                time.sleep(random.uniform(2.0, 4.0))
                
                context = browser.new_context(
                    user_agent="Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0",
                    viewport={"width": 1920, "height": 1080},
                )
                page = context.new_page()
                page.goto(url, wait_until="networkidle", timeout=90000)
                page.wait_for_timeout(3000)
                html = page.content()
                context.close()

            if save_html:
                fname = re.sub(r"[^a-zA-Z0-9]+", "_", url)[:180] + ".html.gz"
                save_text_gz(HTML_DIR / fname, html)

                obj = parse_product(html, url, category=product_category)
                print(f"         {obj.get('brand')} - {obj.get('title')} - {obj.get('price')}")
            append_jsonl(OUT_PRODUCTS, obj)
                success_count += 1
                
            except Exception as e:
                fail_count += 1
                print(f"         ERROR: {type(e).__name__}: {str(e)[:60]}")
                continue  # Skip this product and continue with the next
        
        browser.close()

    print(f"Done. Scraped {success_count}/{total} products ({fail_count} failed). Wrote: {OUT_PRODUCTS}")