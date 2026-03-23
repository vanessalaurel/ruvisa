#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
sys.stdout.flush()
print('A', flush=True)
from scrapper.scrape_product import scrape_products
print('B', flush=True)
scrape_products(limit=1, save_html=False)
print('C', flush=True)

