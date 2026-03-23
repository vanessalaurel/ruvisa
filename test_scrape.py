#!/usr/bin/env python3
"""Minimal test script for debugging segfault."""

import sys
sys.path.insert(0, '.')

from tqdm import tqdm
from pathlib import Path
from playwright.sync_api import sync_playwright

from scrapper.common import append_jsonl
from scrapper.scrape_product import parse_product, _load_urls

IN_URLS = Path('data/raw/sephora/urls/product_urls.jsonl')
OUT_PRODUCTS = Path('data/raw/sephora/products/products.jsonl')

def main():
    print('Loading URLs...')
    urls = _load_urls(IN_URLS)[:3]
    print(f'Got {len(urls)} URLs')
    
    print('Starting playwright...')
    with sync_playwright() as p:
        print('Launching Firefox...')
        browser = p.firefox.launch(headless=True)
        print('Firefox launched!')
        
        for url in tqdm(urls, desc='Scraping'):
            context = browser.new_context(
                user_agent='Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0',
                viewport={'width': 1920, 'height': 1080},
            )
            page = context.new_page()
            page.goto(url, wait_until='networkidle', timeout=90000)
            page.wait_for_timeout(3000)
            html = page.content()
            context.close()
            
            obj = parse_product(html, url)
            print(f'  {obj.get("brand")} - {obj.get("title")}')
            append_jsonl(OUT_PRODUCTS, obj)
        
        browser.close()
    
    print('Done!')

if __name__ == '__main__':
    main()

