"""
Backward-compatibility shim for normalizer engine.
Re-exports modular components from src/normalizers.
"""

import os
import sys

# Ensure proper path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, RESULT_DIR)
sys.path.insert(0, BASE_DIR)

from normalizers.aggressive import normalize_aggressive_name, extract_address_features
from normalizers.moderate import normalize_moderate
from normalizers.batch import process_dataframe
from normalizers.base import strip_accents, clean_url_slug

__all__ = [
    "normalize_aggressive_name",
    "extract_address_features",
    "normalize_moderate",
    "process_dataframe",
    "strip_accents",
    "clean_url_slug",
]
