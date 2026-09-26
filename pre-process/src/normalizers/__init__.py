"""
Normalizers Package for Business Entity Resolution.
Exposes public normalization routines and batch processors.
"""

from normalizers.aggressive import normalize_aggressive_name, extract_address_features
from normalizers.moderate import normalize_moderate
from normalizers.batch import process_dataframe
from normalizers.base import strip_accents, clean_url_slug
from normalizers.cleaner import (
    build_clean_name_expr,
    build_clean_address_expr,
    build_clean_country_expr,
    clean_single_name,
    clean_single_address,
)

__all__ = [
    "normalize_aggressive_name",
    "extract_address_features",
    "normalize_moderate",
    "process_dataframe",
    "strip_accents",
    "clean_url_slug",
    "build_clean_name_expr",
    "build_clean_address_expr",
    "build_clean_country_expr",
    "clean_single_name",
    "clean_single_address",
]
