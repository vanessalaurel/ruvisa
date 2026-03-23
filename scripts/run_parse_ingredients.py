#!/usr/bin/env python3
"""
Entrypoint script to parse and normalize product ingredients.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scrapper.parse_ing import main

if __name__ == "__main__":
    main()

