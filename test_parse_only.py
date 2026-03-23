#!/usr/bin/env python3
"""Test parse_product separately."""
import sys
sys.path.insert(0, '.')

import gzip
from pathlib import Path

print('1: imports', flush=True)
from selectolax.parser import HTMLParser
print('2: selectolax imported', flush=True)

# Load saved HTML
html_dir = Path('data/raw/sephora/html/product')
gz_files = list(html_dir.glob('*.html.gz'))
if not gz_files:
    print('No HTML files found!')
    sys.exit(1)

html_path = gz_files[0]
print(f'3: Loading {html_path.name}', flush=True)

with gzip.open(html_path, 'rt') as f:
    html = f.read()
print(f'4: Got {len(html)} chars', flush=True)

print('5: Importing parse_product', flush=True)
from scrapper.scrape_product import parse_product
print('6: parse_product imported', flush=True)

print('7: Calling parse_product', flush=True)
obj = parse_product(html, 'http://test.com/product/v/123')
print('8: parse_product done', flush=True)
print(f'9: Result: {obj.get("brand")} - {obj.get("title")} - {obj.get("price")}', flush=True)

