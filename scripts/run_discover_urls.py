#!/usr/bin/env python3
"""
Entrypoint script to discover product URLs from Sephora.

Usage:
    # Discover from single category (facial cleanser)
    python scripts/run_discover_urls.py
    
    # Discover from all skincare categories
    python scripts/run_discover_urls.py --all-skincare
    
    # Clear existing and start fresh
    python scripts/run_discover_urls.py --all-skincare --clear
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scrapper.discover_url import discover_category, discover_all_categories

# Sephora HK category URLs
# Based on https://www.sephora.hk website structure
SKINCARE_CATEGORIES = [
    # Cleanser & Exfoliator
    "https://www.sephora.hk/categories/skincare/cleanser-and-exfoliator/facial-cleanser",
    "https://www.sephora.hk/categories/skincare/cleanser-and-exfoliator/scrub-and-exfoliator",
    # Toner
    "https://www.sephora.hk/categories/skincare/toner",
    # Moisturiser
    "https://www.sephora.hk/categories/skincare/moisturiser/day-moisturiser",
    "https://www.sephora.hk/categories/skincare/moisturiser/night-cream",
    "https://www.sephora.hk/categories/skincare/moisturiser/facial-mist",
    # Masks & Treatments
    "https://www.sephora.hk/categories/skincare/masks-and-treatments/mask",
    "https://www.sephora.hk/categories/skincare/masks-and-treatments/serum-and-booster",
    "https://www.sephora.hk/categories/skincare/masks-and-treatments/face-oil",
    "https://www.sephora.hk/categories/skincare/masks-and-treatments/eye-care",
    "https://www.sephora.hk/categories/skincare/masks-and-treatments/lip-care",
    # Sun Care (note: suncare, not sun-care)
    "https://www.sephora.hk/categories/skincare/suncare",
]

MAKEUP_CATEGORIES = [
    # Face
    "https://www.sephora.hk/categories/makeup/face/foundation",
    "https://www.sephora.hk/categories/makeup/face/concealer-and-corrector",
    "https://www.sephora.hk/categories/makeup/face/powder",
    "https://www.sephora.hk/categories/makeup/face/blush",
    "https://www.sephora.hk/categories/makeup/face/bronzer",
    # Eyes
    "https://www.sephora.hk/categories/makeup/eyes/eyeshadow",
    "https://www.sephora.hk/categories/makeup/eyes/eyeliner",
    "https://www.sephora.hk/categories/makeup/eyes/mascara",
    # Lips
    "https://www.sephora.hk/categories/makeup/lips/lipstick",
    "https://www.sephora.hk/categories/makeup/lips/lip-gloss",
    "https://www.sephora.hk/categories/makeup/lips/lip-stain-and-tint",
]

ALL_CATEGORIES = SKINCARE_CATEGORIES + MAKEUP_CATEGORIES


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Discover Sephora product URLs")
    parser.add_argument(
        "--category", "-c",
        type=str,
        default=None,
        help="Single category URL to scrape"
    )
    parser.add_argument(
        "--all-skincare",
        action="store_true",
        help="Scrape all skincare categories"
    )
    parser.add_argument(
        "--all-makeup",
        action="store_true",
        help="Scrape all makeup categories"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Scrape all categories (skincare + makeup)"
    )
    parser.add_argument(
        "--max-pages", "-p",
        type=int,
        default=None,
        help="Maximum pages per category (default: all)"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing URLs file before starting"
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Don't save HTML files"
    )
    
    args = parser.parse_args()
    
    if args.all:
        categories = ALL_CATEGORIES
    elif args.all_skincare:
        categories = SKINCARE_CATEGORIES
    elif args.all_makeup:
        categories = MAKEUP_CATEGORIES
    elif args.category:
        categories = [args.category]
    else:
        # Default: just facial cleanser
        categories = [SKINCARE_CATEGORIES[0]]
    
    print(f"Will scrape {len(categories)} categories")
    
    discover_all_categories(
        category_urls=categories,
        max_pages_per_category=args.max_pages,
        save_html=not args.no_html,
        clear_existing=args.clear,
    )
