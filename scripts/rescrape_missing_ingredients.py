#!/usr/bin/env python3
"""
Re-scrape products that are missing ingredients.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scrapper.common_browser import BrowserFetcher, BrowserConfig
from scrapper.scrape_product import parse_product

PRODUCTS_FILE = Path("data/raw/sephora/products/products.jsonl")


def main():
    if not PRODUCTS_FILE.exists():
        print("No products file found.")
        return
    
    # Read all products
    with open(PRODUCTS_FILE, "r") as f:
        lines = f.readlines()
    
    products = [json.loads(line) for line in lines]
    print(f"Total products: {len(products)}")
    
    # Find products missing ingredients
    missing = []
    for i, p in enumerate(products):
        if not p.get("ingredients"):
            missing.append((i, p))
    
    print(f"Products missing ingredients: {len(missing)}")
    
    if not missing:
        print("All products have ingredients!")
        return
    
    # Re-scrape with browser
    cfg = BrowserConfig(
        headless=True,
        timeout_ms=90000,
        min_delay=2.0,
        max_delay=4.0,
    )
    fetcher = BrowserFetcher(cfg)
    
    try:
        updated = 0
        for idx, (prod_idx, product) in enumerate(missing):
            url = product.get("product_url")
            if not url:
                print(f"  [{idx+1}/{len(missing)}] No URL for product, skipping...")
                continue
            
            print(f"  [{idx+1}/{len(missing)}] Re-scraping: {product.get('brand')} - {product.get('title')[:30]}...")
            
            try:
                html = fetcher.get_html(url)
                new_data = parse_product(html, url, category=product.get("category"))
                
                if new_data.get("ingredients"):
                    # Update the product with new ingredient data
                    products[prod_idx]["ingredients_raw"] = new_data.get("ingredients_raw")
                    products[prod_idx]["ingredients"] = new_data.get("ingredients")
                    products[prod_idx]["ingredients_count"] = new_data.get("ingredients_count")
                    if new_data.get("product_claims"):
                        products[prod_idx]["product_claims"] = new_data.get("product_claims")
                    updated += 1
                    print(f"    ✓ Found {new_data.get('ingredients_count')} ingredients")
                else:
                    print(f"    ✗ Still no ingredients on page")
                    
            except Exception as e:
                print(f"    ✗ Error: {e}")
                
    finally:
        fetcher.close()
    
    # Write updated products
    with open(PRODUCTS_FILE, "w") as f:
        for p in products:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    
    print(f"\nUpdated {updated} products with ingredients.")


if __name__ == "__main__":
    main()

