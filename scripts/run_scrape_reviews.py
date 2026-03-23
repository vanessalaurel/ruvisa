#!/usr/bin/env python3
"""
Entrypoint script to scrape product reviews from Sephora.

Usage:
    python scripts/run_scrape_reviews.py
    python scripts/run_scrape_reviews.py --limit 50
    python scripts/run_scrape_reviews.py --no-resume
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scrapper.scrape_review import main

if __name__ == "__main__":
    main()
