from __future__ import annotations

import re
import time
import random
from pathlib import Path
from typing import List, Optional, Set, Dict, Any
from urllib.parse import urljoin, urlparse, parse_qs, urlencode, urlunparse

from selectolax.parser import HTMLParser
from playwright.sync_api import sync_playwright

from scrapper.common import append_jsonl, save_text_gz, utc_now_iso


OUT_URLS = Path("data/raw/sephora/urls/product_urls.jsonl")
HTML_DIR = Path("data/raw/sephora/html/category")

# Tracking parameters to strip from URLs
TRACKING_PARAMS = {
    "icid2", "icid", "om_mmc", "utm_source", "utm_medium", "utm_campaign",
    "utm_content", "utm_term", "ref", "source", "origin", "sref",
}


def canonicalize_url(url: str) -> str:
    """
    Canonicalize a product URL by:
    - Stripping tracking parameters (icid2, utm_*, etc.)
    - Keeping only the essential /products/.../v/... path
    - Normalizing to https
    """
    p = urlparse(url)
    
    # Parse query string and remove tracking params
    qs = parse_qs(p.query)
    clean_qs = {k: v for k, v in qs.items() if k.lower() not in TRACKING_PARAMS}
    
    # Rebuild URL without tracking params
    new_query = urlencode(clean_qs, doseq=True) if clean_qs else ""
    
    # Ensure https
    scheme = "https" if p.scheme in ("http", "https", "") else p.scheme
    
    return urlunparse((scheme, p.netloc, p.path, p.params, new_query, ""))


def _normalize_url(base: str, href: str) -> str:
    return urljoin(base, href)


def _with_page(url: str, page: int) -> str:
    """Add or update page parameter in URL."""
    p = urlparse(url)
    qs = parse_qs(p.query)
    qs["page"] = [str(page)]
    new_query = urlencode(qs, doseq=True)
    return urlunparse((p.scheme, p.netloc, p.path, p.params, new_query, p.fragment))


def _extract_max_page(html: str) -> Optional[int]:
    """
    Extract the maximum page number from pagination.
    Sephora HK uses pagination like "1 2 3 4 5" at the bottom of category pages.
    """
    tree = HTMLParser(html)
    
    # Look for pagination elements
    pagination_els = tree.css('[class*="pagination"] a, [class*="Pagination"] a, .page-number')
    if pagination_els:
        nums = []
        for el in pagination_els:
            text = el.text() or ""
            if text.strip().isdigit():
                nums.append(int(text.strip()))
        if nums:
            return max(nums)
    
    # Fallback: look for page numbers in the HTML
    # Be more specific to avoid matching product counts, prices, etc.
    page_matches = re.findall(r'[?&]page=(\d+)', html)
    if page_matches:
        return max(int(p) for p in page_matches)
    
        return None


def _extract_product_links(html: str, base_url: str) -> List[str]:
    """
    Extract product links from Sephora HK category HTML.
    Product URLs follow pattern: /products/PRODUCT-NAME/v/VARIANT
    """
    tree = HTMLParser(html)

    # Get base domain
    parsed_base = urlparse(base_url)
    base_domain = f"{parsed_base.scheme}://{parsed_base.netloc}"

    hrefs: List[str] = []
    for a in tree.css("a"):
        href = a.attributes.get("href")
        if not href:
            continue
        # Normalize to absolute URL
        if href.startswith("/"):
            href = base_domain + href
        elif not href.startswith("http"):
            href = _normalize_url(base_url, href)
        hrefs.append(href)

    out: List[str] = []
    for u in hrefs:
        # Must be sephora.hk domain
        if "sephora.hk" not in u:
            continue
        
        # Skip non-product pages
        if any(x in u for x in [
            "/categories/", "/register", "/sign", "/help", 
            "/stores", "/events", "/bag", "/cart", "/search",
            "/account", "/wishlist", "/checkout"
        ]):
            continue

        # Match Sephora HK product URL pattern: /products/PRODUCT-NAME/v/VARIANT
        if "/products/" in u and "/v/" in u:
            # Canonicalize to remove tracking params
            canonical = canonicalize_url(u)
            out.append(canonical)

    # De-dupe while preserving order
    seen: Set[str] = set()
    uniq: List[str] = []
    for u in out:
        if u in seen:
            continue
        seen.add(u)
        uniq.append(u)
    return uniq


def _load_existing_urls(path: Path) -> Set[str]:
    """Load already discovered URLs to avoid duplicates across runs."""
    existing: Set[str] = set()
    if path.exists():
        with path.open("r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    import orjson
                    obj = orjson.loads(line)
                    url = obj.get("product_url")
                    if url:
                        existing.add(canonicalize_url(url))
                except Exception:
                    pass
    return existing


def _get_category_slug(url: str) -> str:
    """Extract a slug from category URL for file naming."""
    path = urlparse(url).path
    # /categories/skincare/cleanser-and-exfoliator/facial-cleanser -> facial-cleanser
    parts = [p for p in path.split("/") if p]
    return parts[-1] if parts else "category"


def discover_category(
    category_url: str,
    max_pages: Optional[int] = None,
    save_html: bool = True,
    existing_urls: Optional[Set[str]] = None,
) -> Set[str]:
    """
    Discover product URLs from a category page with pagination.
    
    Args:
        category_url: The category page URL
        max_pages: Maximum pages to scrape (None = all available)
        save_html: Whether to save HTML for debugging
        existing_urls: Set of already discovered URLs to skip
    
    Returns:
        Set of newly discovered product URLs
    """
    if existing_urls is None:
        existing_urls = set()
    
    category_slug = _get_category_slug(category_url)
    new_urls: Set[str] = set()
    
    print(f"\nDiscovering: {category_slug}")
    print(f"URL: {category_url}")
    
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0",
            viewport={"width": 1920, "height": 1080},
        )

    try:
            # First page to detect max pages
            page = context.new_page()
            page.goto(category_url, wait_until="networkidle", timeout=90000)
            page.wait_for_timeout(3000)
            html1 = page.content()
            page.close()
            
        if save_html:
                html_subdir = HTML_DIR / category_slug
                html_subdir.mkdir(parents=True, exist_ok=True)
                save_text_gz(html_subdir / "page_1.html.gz", html1)

        detected_max = _extract_max_page(html1)
            total_pages = detected_max or 1
            if max_pages:
                total_pages = min(total_pages, max_pages)
            
            print(f"Detected {detected_max or 'unknown'} pages, will scrape {total_pages}")
            
            # Process first page
            links = _extract_product_links(html1, base_url=category_url)
            for link in links:
                canonical = canonicalize_url(link)
                if canonical not in existing_urls and canonical not in new_urls:
                    new_urls.add(canonical)
                    append_jsonl(OUT_URLS, {
                        "source": "sephora_hk",
                        "category_url": category_url,
                        "category": category_slug,
                        "product_url": canonical,
                        "discovered_from_page": 1,
                        "discovered_at": utc_now_iso(),
                    })
            
            print(f"  Page 1: {len(links)} products found, {len(new_urls)} new")
            
            # Process remaining pages
            for page_num in range(2, total_pages + 1):
                time.sleep(random.uniform(2.0, 4.0))
                
                page_url = _with_page(category_url, page_num)
                page = context.new_page()
                
                try:
                    page.goto(page_url, wait_until="networkidle", timeout=90000)
                    page.wait_for_timeout(3000)
                    html = page.content()
                except Exception as e:
                    print(f"  Page {page_num}: ERROR - {e}")
                    page.close()
                    continue
                finally:
                    page.close()
                
                if save_html:
                    save_text_gz(html_subdir / f"page_{page_num}.html.gz", html)
                
                links = _extract_product_links(html, base_url=category_url)
                new_on_page = 0
                for link in links:
                    canonical = canonicalize_url(link)
                    if canonical not in existing_urls and canonical not in new_urls:
                        new_urls.add(canonical)
                        new_on_page += 1
                        append_jsonl(OUT_URLS, {
                            "source": "sephora_hk",
                            "category_url": category_url,
                            "category": category_slug,
                            "product_url": canonical,
                            "discovered_from_page": page_num,
                            "discovered_at": utc_now_iso(),
                        })
                
                print(f"  Page {page_num}: {len(links)} products found, {new_on_page} new")
                
                # Stop if no new products found (we've seen all)
                if len(links) == 0:
                    print(f"  No products on page {page_num}, stopping pagination")
                    break
        
        finally:
            context.close()
            browser.close()
    
    print(f"Category {category_slug}: {len(new_urls)} new URLs discovered")
    return new_urls


def discover_all_categories(
    category_urls: List[str],
    max_pages_per_category: Optional[int] = None,
    save_html: bool = True,
    clear_existing: bool = False,
) -> None:
    """
    Discover product URLs from multiple category pages with deduplication.
    
    Args:
        category_urls: List of category page URLs to scrape
        max_pages_per_category: Max pages per category (None = all)
        save_html: Whether to save HTML for debugging
        clear_existing: If True, clear existing URLs file before starting
    """
    if clear_existing and OUT_URLS.exists():
        OUT_URLS.unlink()
        print(f"Cleared existing URLs file: {OUT_URLS}")
    
    # Load existing URLs to avoid duplicates
    existing_urls = _load_existing_urls(OUT_URLS)
    print(f"Loaded {len(existing_urls)} existing URLs")
    
    total_new = 0
    
    for i, category_url in enumerate(category_urls):
        print(f"\n{'='*60}")
        print(f"Category {i+1}/{len(category_urls)}")
        
        try:
            new_urls = discover_category(
                category_url=category_url,
                max_pages=max_pages_per_category,
                save_html=save_html,
                existing_urls=existing_urls,
            )
            
            # Add to existing set for next category
            existing_urls.update(new_urls)
            total_new += len(new_urls)
            
        except Exception as e:
            print(f"ERROR scraping category: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"DONE. Total new URLs: {total_new}")
    print(f"Total unique URLs in file: {len(existing_urls)}")
        print(f"Wrote: {OUT_URLS}")