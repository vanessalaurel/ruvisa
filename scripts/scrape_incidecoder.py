"""
Scrape INCIDecoder ingredient-function pages to build an evidence-based
ingredient-to-skin-concern lookup table.

Outputs:
  labeling/ingredient_evidence.json  – {ingredient: [functions]}
  labeling/concern_lookup.json       – {ingredient: [concerns]}
"""

import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://incidecoder.com"
OUT_DIR = Path(__file__).resolve().parent.parent / "labeling"

FUNCTION_PAGES = [
    "anti-acne",
    "skin-brightening",
    "soothing",
    "exfoliant",
    "cell-communicating-ingredient",
    "astringent",
    "antioxidant",
]

FUNCTION_TO_CONCERN = {
    "anti-acne":                     ["acne", "comedonal_acne"],
    "skin-brightening":              ["pigmentation"],
    "soothing":                      ["redness"],
    "exfoliant":                     ["pores", "acne_scars_texture", "comedonal_acne"],
    "cell-communicating-ingredient": ["wrinkles"],
    "astringent":                    ["pores"],
    "antioxidant":                   ["pigmentation"],
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
}

REQUEST_DELAY = 2.0


def fetch_page(url: str) -> BeautifulSoup | None:
    """Fetch a URL and return parsed soup, or None on failure."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "html.parser")
    except requests.RequestException as e:
        print(f"  [WARN] Failed to fetch {url}: {e}")
        return None


def extract_ingredients_from_page(soup: BeautifulSoup) -> list[str]:
    """Extract ingredient names from a function page."""
    ingredients = []
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if href.startswith("/ingredients/") and href != "/ingredients":
            name = link.get_text(strip=True)
            if name:
                ingredients.append(name)
    return ingredients


def get_next_page_url(soup: BeautifulSoup, current_url: str) -> str | None:
    """Find the 'Next page >>' link if present."""
    for link in soup.find_all("a", href=True):
        text = link.get_text(strip=True)
        if "next page" in text.lower():
            return urljoin(current_url, link["href"])
    return None


def scrape_function(func_slug: str) -> list[str]:
    """Scrape all ingredients from a function page (with pagination)."""
    all_ingredients = []
    url = f"{BASE_URL}/ingredient-functions/{func_slug}"
    page_num = 1

    while url:
        print(f"  Scraping {func_slug} page {page_num}: {url}")
        soup = fetch_page(url)
        if soup is None:
            break

        page_ings = extract_ingredients_from_page(soup)
        if not page_ings:
            break

        all_ingredients.extend(page_ings)
        url = get_next_page_url(soup, url)
        page_num += 1
        time.sleep(REQUEST_DELAY)

    return all_ingredients


def build_evidence_table() -> dict[str, list[str]]:
    """
    Scrape all function pages and return {ingredient_name: [functions]}.
    """
    evidence: dict[str, list[str]] = {}

    for func_slug in FUNCTION_PAGES:
        print(f"\n--- Scraping function: {func_slug} ---")
        ingredients = scrape_function(func_slug)
        print(f"  Found {len(ingredients)} ingredients")

        for ing_name in ingredients:
            key = ing_name.strip()
            if key not in evidence:
                evidence[key] = []
            if func_slug not in evidence[key]:
                evidence[key].append(func_slug)

        time.sleep(REQUEST_DELAY)

    return evidence


def build_concern_lookup(evidence: dict[str, list[str]]) -> dict[str, list[str]]:
    """
    Convert {ingredient: [functions]} to {ingredient_lower: [concerns]}.
    """
    lookup: dict[str, list[str]] = {}

    for ing_name, functions in evidence.items():
        concerns = set()
        for func in functions:
            for concern in FUNCTION_TO_CONCERN.get(func, []):
                concerns.add(concern)

        if concerns:
            lookup[ing_name.lower().strip()] = sorted(concerns)

    return lookup


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("INCIDecoder Evidence Scraper")
    print("=" * 60)

    evidence = build_evidence_table()

    evidence_path = OUT_DIR / "ingredient_evidence.json"
    with open(evidence_path, "w") as f:
        json.dump(evidence, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(evidence)} ingredients to {evidence_path}")

    concern_lookup = build_concern_lookup(evidence)

    lookup_path = OUT_DIR / "concern_lookup.json"
    with open(lookup_path, "w") as f:
        json.dump(concern_lookup, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(concern_lookup)} ingredients with concerns to {lookup_path}")

    print("\n--- Concern coverage summary ---")
    all_concerns = ["acne", "comedonal_acne", "pigmentation",
                    "acne_scars_texture", "pores", "redness", "wrinkles"]
    for concern in all_concerns:
        count = sum(1 for ings in concern_lookup.values() if concern in ings)
        print(f"  {concern}: {count} ingredients")


if __name__ == "__main__":
    main()
