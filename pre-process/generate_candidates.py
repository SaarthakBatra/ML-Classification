#!/usr/bin/env python3
"""
Top-level entrypoint for Candidate Pair Generation.
Run from AMAZON/result directory:
    python generate_candidates.py [options]
"""

import os
import sys

# Ensure src and config are in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from cli.main import main

if __name__ == "__main__":
    main()

