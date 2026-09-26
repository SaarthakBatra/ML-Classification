"""
Ultra-High-Performance Polars Text Cleaner (Phase 0 Preprocessing Engine).

Implements the complete text normalization specification from Preprocessing_1.0.md:
1. Unicode Normalization: NFKD encoding & diacritics stripping (café -> cafe)
2. Case & Whitespace: Lowercasing, trimming, and collapsing whitespace
3. Punctuation Stripping & Acronym Protection:
   - Periods removed without space replacement (I.B.M. -> IBM)
   - '@' -> ' at '
   - '&' -> ' and '
   - All other punctuation stripped except alphanumeric and spaces
4. Abbreviation Expansion:
   - Ordinal numbers: first -> 1st, second -> 2nd, etc.
   - Business suffixes: corp -> corporation, inc -> incorporated, ltd -> limited,
     pvt -> private, co -> company, llc -> limited liability company, etc.
   - Address terms: st -> street, rd -> road, ave -> avenue, blvd -> boulevard,
     apt -> apartment, dr -> drive, ln -> lane, etc.
5. Empty String Safety Fallback:
   - Empty name -> UNKNOWN_NAME
   - Empty address -> UNKNOWN_ADDRESS
6. Country Normalization:
   - Standardized mappings (usa -> US, ind -> INDIA, fr -> FRANCE)
"""

import re
from typing import Optional, List, Tuple
import polars as pl
from normalizers.base import (
    ORDINAL_EXPANSIONS,
    BUSINESS_SUFFIX_EXPANSIONS,
    ADDR_ABBREVIATIONS,
    COUNTRY_ALIASES,
    UNKNOWN_NAME,
    UNKNOWN_ADDRESS,
    LITERAL_NULLS,
    strip_accents,
    clean_url_slug,
    transliterate_text,
)

# Literal null regex matching exact strings
LITERAL_NULL_REGEX = r'^(?i:n/a|na|null|none|-|nan|undefined)$'

# Address expansion patterns as (compiled_regex, replacement_string)
ADDRESS_EXPANSIONS: List[Tuple[str, str]] = [
    (r'\bst\b', 'street'),
    (r'\brd\b', 'road'),
    (r'\bave\b', 'avenue'),
    (r'\bblvd\b', 'boulevard'),
    (r'\bdr\b', 'drive'),
    (r'\bln\b', 'lane'),
    (r'\bct\b', 'court'),
    (r'\bapt\b', 'apartment'),
    (r'\bste\b', 'suite'),
    (r'\bpkwy\b', 'parkway'),
    (r'\br\b', 'rue'),
    (r'\bpl\b', 'place'),
    (r'\bsq\b', 'square'),
    (r'\bhwy\b', 'highway'),
    (r'\bcir\b', 'circle'),
    (r'\bter\b', 'terrace'),
    (r'\bbldg\b', 'building'),
    (r'\bfl\b', 'floor'),
]


def build_clean_name_expr(col_name: str = "business_name") -> pl.Expr:
    """
    Builds a vectorized Polars expression that standardizes business names
    strictly according to Preprocessing_1.0.md.

    Complexity: Fully parallelized in Rust across CPU cores.
    """
    expr = pl.col(col_name).fill_null("").cast(pl.Utf8)

    # 1. Literal null check
    expr = pl.when(expr.str.strip_chars().str.contains(LITERAL_NULL_REGEX)).then(pl.lit("")).otherwise(expr)

    # 2. Unicode NFKD normalization and diacritic stripping
    expr = expr.str.normalize("NFKD").str.replace_all(r"[\u0300-\u036f]", "")

    # 3. Lowercase
    expr = expr.str.to_lowercase()

    # 4. Acronym protection: remove periods without space replacement (I.B.M. -> ibm)
    expr = expr.str.replace_all(r"\.", "")

    # 5. At symbol mapping & ampersand mapping
    expr = expr.str.replace_all(r"@", " at ")
    expr = expr.str.replace_all(r"&", " and ")

    # 6. Remove non-alphanumeric characters (keep alphanumeric and spaces)
    expr = expr.str.replace_all(r"[^\w\s]", " ")

    # 7. Ordinal mapping (word -> number)
    for word, repl in ORDINAL_EXPANSIONS:
        expr = expr.str.replace_all(rf"\b{word}\b", repl)

    # 8. Business suffix expansion (corp -> corporation, inc -> incorporated, etc.)
    for pattern, repl in BUSINESS_SUFFIX_EXPANSIONS:
        # If pattern already contains \b, use it directly, otherwise wrap with \b
        regex = pattern if pattern.startswith(r"\b") else rf"\b{pattern}\b"
        expr = expr.str.replace_all(regex, repl)

    # 9. Collapse multiple internal spaces and trim
    expr = expr.str.replace_all(r"\s+", " ").str.strip_chars()

    # 10. Empty String Safety Fallback
    return pl.when(expr == "").then(pl.lit(UNKNOWN_NAME)).otherwise(expr)


def build_clean_address_expr(col_name: str = "business_address") -> pl.Expr:
    """
    Builds a vectorized Polars expression that standardizes business addresses
    strictly according to Preprocessing_1.0.md.

    Complexity: Fully parallelized in Rust across CPU cores.
    """
    expr = pl.col(col_name).fill_null("").cast(pl.Utf8)

    # 1. Literal null check
    expr = pl.when(expr.str.strip_chars().str.contains(LITERAL_NULL_REGEX)).then(pl.lit("")).otherwise(expr)

    # 2. Unicode NFKD normalization and diacritic stripping
    expr = expr.str.normalize("NFKD").str.replace_all(r"[\u0300-\u036f]", "")

    # 3. Lowercase
    expr = expr.str.to_lowercase()

    # 4. Remove periods without space replacement (e.g. St. -> st, Ave. -> ave)
    expr = expr.str.replace_all(r"\.", "")

    # 5. At symbol mapping & ampersand mapping
    expr = expr.str.replace_all(r"@", " at ")
    expr = expr.str.replace_all(r"&", " and ")

    # 6. Remove non-alphanumeric characters
    expr = expr.str.replace_all(r"[^\w\s]", " ")

    # 7. Ordinal mapping (word -> number)
    for word, repl in ORDINAL_EXPANSIONS:
        expr = expr.str.replace_all(rf"\b{word}\b", repl)

    # 8. Address abbreviation expansion (st -> street, rd -> road, etc.)
    for pattern, repl in ADDRESS_EXPANSIONS:
        expr = expr.str.replace_all(pattern, repl)

    # 9. Collapse multiple internal spaces and trim
    expr = expr.str.replace_all(r"\s+", " ").str.strip_chars()

    # 10. Empty String Safety Fallback
    return pl.when(expr == "").then(pl.lit(UNKNOWN_ADDRESS)).otherwise(expr)


def build_clean_country_expr(col_name: str = "country") -> pl.Expr:
    """
    Builds a vectorized Polars expression that standardizes country codes:
    - Trims whitespace
    - Maps known aliases ('usa' -> 'US', 'india' -> 'INDIA', 'france' -> 'FRANCE')
    - Converts to uppercase
    """
    expr = pl.col(col_name).fill_null("").cast(pl.Utf8).str.strip_chars().str.to_lowercase()

    # Build when-then branches for known aliases
    when_expr = None
    for alias, standard in COUNTRY_ALIASES.items():
        if when_expr is None:
            when_expr = pl.when(expr == alias).then(pl.lit(standard))
        else:
            when_expr = when_expr.when(expr == alias).then(pl.lit(standard))

    if when_expr is not None:
        expr = when_expr.otherwise(expr.str.to_uppercase())
    else:
        expr = expr.str.to_uppercase()

    return expr


def clean_single_name(name: str) -> str:
    """Python scalar function for clean_name (for testing and single-row usage)."""
    if not name or name.strip().lower() in LITERAL_NULLS:
        return UNKNOWN_NAME

    text = name.strip()
    text = transliterate_text(text)
    text = strip_accents(text)
    text = clean_url_slug(text)
    text = text.lower()
    text = text.replace('.', '')
    text = text.replace('@', ' at ').replace('&', ' and ')
    text = re.sub(r'[^\w\s]', ' ', text)

    for word, repl in ORDINAL_EXPANSIONS:
        text = re.sub(rf'\b{word}\b', repl, text)

    for pattern, repl in BUSINESS_SUFFIX_EXPANSIONS:
        regex = pattern if pattern.startswith(r'\b') else rf'\b{pattern}\b'
        text = re.sub(regex, repl, text)

    text = re.sub(r'\s+', ' ', text).strip()
    return text if text else UNKNOWN_NAME


def clean_single_address(address: str) -> str:
    """Python scalar function for clean_address (for testing and single-row usage)."""
    if not address or address.strip().lower() in LITERAL_NULLS:
        return UNKNOWN_ADDRESS

    text = address.strip()
    text = transliterate_text(text)
    text = strip_accents(text)
    text = text.lower()
    text = text.replace('.', '')
    text = text.replace('@', ' at ').replace('&', ' and ')
    text = re.sub(r'[^\w\s]', ' ', text)

    for word, repl in ORDINAL_EXPANSIONS:
        text = re.sub(rf'\b{word}\b', repl, text)

    for pattern, repl in ADDRESS_EXPANSIONS:
        text = re.sub(pattern, repl, text)

    text = re.sub(r'\s+', ' ', text).strip()
    return text if text else UNKNOWN_ADDRESS
