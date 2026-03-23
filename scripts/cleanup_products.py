#!/usr/bin/env python3
"""
Cleanup script to remove garbage ingredient data from scraped products.

Usage:
    python scripts/cleanup_products.py
    python scripts/cleanup_products.py --dry-run  # Preview without changing
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

PRODUCTS_FILE = Path("data/raw/sephora/products/products.jsonl")

# Indicators of garbage in ingredients
GARBAGE_INDICATORS = [
    # HTML/JS/CSS
    "<!DOCTYPE", "<html", "<script", "<style", "function()",
    "@media", "!important", "font-family:", "background-color:",
    # URLs
    "http://", "https://", ".com", ".hk",
    # Navigation
    "Sign In", "Sign Up", "Add To Bag", "Shopping", "My Account",
    "Categories", "Brands", "Makeup", "Skincare", "Hair",
    # Prices and products
    "$", "HKD", "Holiday Limited Edition", "Recommended for You",
    # Instructions
    "Massage over", "Rinse thoroughly", "Use twice a day", "Apply to",
    "How To", "Ship to", "Shipping", "massage gently",
    # JSON-LD
    "@context", "@type", "schema.org", "BreadcrumbList",
    # Clean at Sephora description
    "Planet Aware", "Clean at Sephora", "what makes a brand",
    "formulated without", "Learn more about Sephora",
    # Other noise
    "View Full", "Disclaimer", "skin's micro-flora", "skin's microbiome",
]


def is_garbage_ingredients(ingredients_raw: str) -> bool:
    """Check if ingredients_raw contains garbage."""
    if not ingredients_raw:
        return False
    
    for indicator in GARBAGE_INDICATORS:
        if indicator.lower() in ingredients_raw.lower():
            return True
    return False


def cleanup_products(dry_run: bool = False) -> None:
    """Clean up products with garbage ingredients."""
    if not PRODUCTS_FILE.exists():
        print(f"File not found: {PRODUCTS_FILE}")
        return
    
    with PRODUCTS_FILE.open('r') as f:
        lines = f.readlines()
    
    cleaned_count = 0
    output_lines = []
    bad_products = []
    
    for i, line in enumerate(lines, 1):
        obj = json.loads(line)
        ingredients_raw = obj.get('ingredients_raw', '')
        
        if is_garbage_ingredients(ingredients_raw):
            bad_products.append({
                'line': i,
                'brand': obj.get('brand'),
                'title': obj.get('title'),
            })
            
            if not dry_run:
                # Remove garbage fields
                obj.pop('ingredients_raw', None)
                obj.pop('ingredients', None)
                obj.pop('ingredients_count', None)
            cleaned_count += 1
        
        output_lines.append(json.dumps(obj, ensure_ascii=False) + '\n')
    
    print(f"Total products: {len(lines)}")
    print(f"Products with garbage ingredients: {cleaned_count}")
    
    if bad_products:
        print("\nAffected products:")
        for p in bad_products:
            print(f"  #{p['line']}: {p['brand']} - {p['title'][:50]}")
    
    if dry_run:
        print("\n[DRY RUN] No changes made.")
    else:
        with PRODUCTS_FILE.open('w') as f:
            f.writelines(output_lines)
        print(f"\nCleaned {cleaned_count} products (removed garbage ingredients)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean up garbage ingredients from products")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without modifying the file"
    )
    args = parser.parse_args()
    
    cleanup_products(dry_run=args.dry_run)

