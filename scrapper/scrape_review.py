"""
Scrape product reviews from Sephora HK product pages.

Reviews are embedded in JSON-LD format via Bazaarvoice.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import orjson
from selectolax.parser import HTMLParser

from scrapper.common import append_jsonl, utc_now_iso, save_text_gz
from scrapper.common_browser import BrowserFetcher, BrowserConfig, get_rendered_html, close_browser


# Paths
PRODUCTS_FILE = Path("data/raw/sephora/products/products.jsonl")
REVIEWS_FILE = Path("data/raw/sephora/reviews/reviews.jsonl")
HTML_DIR = Path("data/raw/sephora/html/review")


@dataclass
class Review:
    """A single product review."""
    product_url: str
    product_brand: str
    product_title: str
    reviewer_name: str
    rating: int
    headline: str
    review_text: str
    date_created: Optional[str] = None
    date_published: Optional[str] = None
    has_images: bool = False
    has_videos: bool = False


def _extract_reviews_from_jsonld(html: str, product_url: str, product_brand: str, product_title: str) -> List[Dict[str, Any]]:
    """
    Extract reviews from the JSON-LD script tag embedded by Bazaarvoice.
    """
    reviews = []
    
    tree = HTMLParser(html)
    
    # Look for the Bazaarvoice JSON-LD reviews data
    script_tags = tree.css('script[type="application/ld+json"]')
    
    for script in script_tags:
        text = script.text(strip=True)
        if not text:
            continue
        
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            continue
        
        # Check if this is the reviews data
        if isinstance(data, dict) and "review" in data:
            review_list = data.get("review", [])
            
            for r in review_list:
                if not isinstance(r, dict):
                    continue
                
                # Extract review data
                rating_obj = r.get("reviewRating", {})
                rating_value = rating_obj.get("ratingValue") if isinstance(rating_obj, dict) else None
                
                author_obj = r.get("author", {})
                author_name = author_obj.get("name", "Anonymous") if isinstance(author_obj, dict) else "Anonymous"
                
                review = {
                    "source": "sephora_hk",
                    "product_url": product_url,
                    "product_brand": product_brand,
                    "product_title": product_title,
                    "reviewer_name": author_name,
                    "rating": int(rating_value) if rating_value else None,
                    "headline": r.get("headline", ""),
                    "review_text": r.get("reviewBody", ""),
                    "date_created": r.get("dateCreated"),
                    "date_published": r.get("datePublished"),
                    "has_images": bool(r.get("image")),
                    "has_videos": bool(r.get("video")),
                    "scraped_at": utc_now_iso(),
                }
                
                # Only add if we have some content
                if review["review_text"] or review["headline"]:
                    reviews.append(review)
    
    return reviews


def _extract_additional_reviews_from_html(html: str, product_url: str, product_brand: str, product_title: str) -> List[Dict[str, Any]]:
    """
    Extract additional review info from visible HTML elements.
    This captures reviews that might be loaded via JavaScript but not in JSON-LD.
    """
    reviews = []
    tree = HTMLParser(html)
    
    # Look for review containers in the rendered HTML
    # Bazaarvoice typically renders reviews in specific containers
    review_containers = tree.css('.bv-content-review, .bv-review, [data-bv-content="review"]')
    
    for container in review_containers:
        try:
            # Try to extract rating
            rating_elem = container.css_first('.bv-rating-stars-container, .bv-rating')
            rating = None
            if rating_elem:
                rating_text = rating_elem.attributes.get('data-bv-rating', '')
                if rating_text:
                    try:
                        rating = int(float(rating_text))
                    except ValueError:
                        pass
            
            # Extract reviewer name
            name_elem = container.css_first('.bv-author, .bv-content-author-name')
            reviewer_name = name_elem.text(strip=True) if name_elem else "Anonymous"
            
            # Extract headline
            headline_elem = container.css_first('.bv-content-title')
            headline = headline_elem.text(strip=True) if headline_elem else ""
            
            # Extract review text
            text_elem = container.css_first('.bv-content-summary-body-text, .bv-content-summary')
            review_text = text_elem.text(strip=True) if text_elem else ""
            
            # Extract date
            date_elem = container.css_first('.bv-content-datetime-stamp, .bv-content-datetime')
            date_text = date_elem.text(strip=True) if date_elem else None
            
            if review_text or headline:
                review = {
                    "source": "sephora_hk",
                    "product_url": product_url,
                    "product_brand": product_brand,
                    "product_title": product_title,
                    "reviewer_name": reviewer_name,
                    "rating": rating,
                    "headline": headline,
                    "review_text": review_text,
                    "date_created": date_text,
                    "date_published": date_text,
                    "has_images": False,
                    "has_videos": False,
                    "scraped_at": utc_now_iso(),
                }
                reviews.append(review)
                
        except Exception:
            continue
    
    return reviews


def parse_reviews(html: str, product_url: str, product_brand: str, product_title: str) -> List[Dict[str, Any]]:
    """
    Parse all reviews from a product page HTML.
    """
    # First extract from JSON-LD (most reliable)
    reviews = _extract_reviews_from_jsonld(html, product_url, product_brand, product_title)
    
    # If no JSON-LD reviews, try HTML parsing
    if not reviews:
        reviews = _extract_additional_reviews_from_html(html, product_url, product_brand, product_title)
    
    return reviews


def _load_products() -> List[Dict[str, Any]]:
    """Load all products from products.jsonl."""
    products = []
    if not PRODUCTS_FILE.exists():
        return products
    
    with open(PRODUCTS_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    products.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    return products


def _load_scraped_review_urls() -> Set[str]:
    """Load URLs that have already been scraped for reviews."""
    scraped = set()
    if not REVIEWS_FILE.exists():
        return scraped
    
    with open(REVIEWS_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    url = data.get("product_url", "")
                    if url:
                        scraped.add(url)
                except json.JSONDecodeError:
                    continue
    
    return scraped


def _url_to_filename(url: str) -> str:
    """Convert URL to a safe filename."""
    # Remove protocol and replace special chars
    safe = url.replace("https://", "").replace("http://", "")
    safe = re.sub(r'[^\w\-.]', '_', safe)
    return safe[:200]  # Limit length


def scrape_reviews(
    limit: int | None = None,
    save_html: bool = True,
    resume: bool = True,
) -> None:
    """
    Scrape reviews for all products in products.jsonl.
    
    Args:
        limit: Maximum number of products to scrape reviews for
        save_html: Whether to save HTML files
        resume: Whether to skip already scraped products
    """
    # Ensure output directory exists
    REVIEWS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if save_html:
        HTML_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load products
    products = _load_products()
    if not products:
        print("No products found in products.jsonl")
        return
    
    print(f"Loaded {len(products)} products")
    
    # Filter to products with reviews
    products_with_reviews = [p for p in products if p.get("review_count", 0) > 0]
    print(f"Products with reviews: {len(products_with_reviews)}")
    
    # Load already scraped URLs
    scraped_urls = set()
    if resume:
        scraped_urls = _load_scraped_review_urls()
        print(f"Already scraped reviews for {len(scraped_urls)} product URLs")
    
    # Filter to unscraped products
    to_scrape = [p for p in products_with_reviews if p["product_url"] not in scraped_urls]
    print(f"Products to scrape: {len(to_scrape)}")
    
    if limit:
        to_scrape = to_scrape[:limit]
        print(f"Limited to {len(to_scrape)} products")
    
    if not to_scrape:
        print("No products to scrape!")
        return
    
    # Create browser fetcher
    fetcher = BrowserFetcher(BrowserConfig(
        timeout_ms=90000,
        max_retries=3,
        min_delay=3.0,
        max_delay=6.0,
        wait_after_load_ms=5000,  # Wait longer for reviews to load
    ))
    
    total_reviews = 0
    
    try:
        for i, product in enumerate(to_scrape):
            url = product["product_url"]
            brand = product.get("brand", "")
            title = product.get("title", "")
            review_count = product.get("review_count", 0)
            
            print(f"\n[{i+1}/{len(to_scrape)}] {brand} - {title}")
            print(f"  URL: {url}")
            print(f"  Expected reviews: {review_count}")
            
            try:
                # Fetch the page
                html = fetcher.get_html(url)
                
                # Save HTML if requested
                if save_html:
                    filename = _url_to_filename(url) + ".html.gz"
                    save_text_gz(HTML_DIR / filename, html)
                
                # Parse reviews
                reviews = parse_reviews(html, url, brand, title)
                
                print(f"  Extracted {len(reviews)} reviews")
                
                # Save reviews
                for review in reviews:
                    append_jsonl(REVIEWS_FILE, review)
                    total_reviews += 1
                
            except Exception as e:
                print(f"  ERROR: {e}")
                continue
    
    finally:
        # Close browser
        fetcher.close()
    
    print(f"\n{'='*60}")
    print(f"DONE. Total reviews scraped: {total_reviews}")
    print(f"Reviews saved to: {REVIEWS_FILE}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Scrape product reviews from Sephora HK")
    parser.add_argument("--limit", type=int, help="Maximum number of products to scrape")
    parser.add_argument("--no-html", action="store_true", help="Don't save HTML files")
    parser.add_argument("--no-resume", action="store_true", help="Don't skip already scraped products")
    
    args = parser.parse_args()
    
    scrape_reviews(
        limit=args.limit,
        save_html=not args.no_html,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()

