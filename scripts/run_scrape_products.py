#!/usr/bin/env python3
"""
Entrypoint script to scrape product details from Sephora.

Usage:
    # Continue scraping facial-cleanser (resumes from where it left off)
    python scripts/run_scrape_products.py --filter facial-cleanser
    
    # Scrape multiple categories
    python scripts/run_scrape_products.py --filter facial-cleanser,scrub-and-exfoliator,toner
    
    # Scrape all skincare
    python scripts/run_scrape_products.py --all-skincare
    
    # Scrape with limit
    python scripts/run_scrape_products.py --filter facial-cleanser --limit 50
    
    # Start fresh (don't skip already scraped)
    python scripts/run_scrape_products.py --filter facial-cleanser --no-resume
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scrapper.scrape_product import scrape_products

# Available category slugs (from product_urls.jsonl)
SKINCARE_CATEGORIES = [
    "facial-cleanser",
    "scrub-and-exfoliator",
    "toner",
    "day-moisturiser",
    "night-cream",
    "facial-mist",
    "mask",
    "serum-and-booster",
    "face-oil",
    "eye-care",
    "lip-care",
    "suncare",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape Sephora product details")
    parser.add_argument(
        "--filter", "-f",
        type=str,
        default=None,
        help="Comma-separated category slugs to scrape (e.g., 'facial-cleanser,toner')"
    )
    parser.add_argument(
        "--all-skincare",
        action="store_true",
        help="Scrape all skincare categories"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Maximum number of products to scrape (default: all)"
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Don't save HTML files"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't skip already scraped products (start fresh)"
    )
    
    args = parser.parse_args()
    
    # Determine category filter
    if args.all_skincare:
        category_filter = SKINCARE_CATEGORIES
    elif args.filter:
        category_filter = [c.strip() for c in args.filter.split(",")]
    else:
        # Default: facial-cleanser only
        category_filter = ["facial-cleanser"]
    
    print(f"Categories to scrape: {category_filter}")
    
    scrape_products(
        limit=args.limit,
        save_html=not args.no_html,
        category_filter=category_filter,
        resume=not args.no_resume,
    )
