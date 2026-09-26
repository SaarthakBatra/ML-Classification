"""
Track A: Aggressive Normalization Engine.

Designed for high-recall deterministic blocking keys, inverted indices, and token collisions:
- Diacritic/accent neutralization (NFKD)
- Domain slug unpacking (extracting core entity name from URL)
- Ampersand conversion
- Multilingual corporate legal suffix stripping
- Punctuation removal
- Alphabetical token sorting (eliminating word transposition variance)
- Premise address numeric token extraction for physical geocoding
"""

from typing import List, Tuple
import re
from normalizers.base import (
    strip_accents,
    clean_url_slug,
    transliterate_text,
    RE_LEGAL_SUFFIXES,
    RE_NON_ALPHANUM,
    RE_NUMERICS,
    ADDR_ABBREVIATIONS,
    ORDINAL_EXPANSIONS,
    LITERAL_NULLS,
)


def normalize_aggressive_name(name: str) -> str:
    """
    Aggressive name normalization for inverted indices and blocking keys.

    Complexity: O(L + N log N) where L is string length, N is token count.
    Output: Alphabetically sorted, suffix-stripped lowercase tokens.
    """
    if not name or name.strip().lower() in LITERAL_NULLS:
        return ""

    text = name.strip()
    text = transliterate_text(text)
    text = strip_accents(text)
    text = clean_url_slug(text)
    text = text.replace('&', ' and ').replace('@', ' at ')
    text = text.lower()
    # Acronym protection: remove dots with no replacement (I.B.M. -> ibm)
    text = text.replace('.', '')
    text = RE_LEGAL_SUFFIXES.sub(' ', text)
    text = RE_NON_ALPHANUM.sub(' ', text)

    tokens = text.split()
    tokens.sort()
    return " ".join(tokens)


def extract_address_features(address: str, max_digits: int = 6) -> Tuple[str, List[int]]:
    """
    Extracts aggressive address token representation and sorted numeric tokens.

    Parameters:
        address: Raw address string
        max_digits: Upper bound on digit count to filter out phone numbers / tracking IDs

    Returns:
        Tuple of (clean_address_tokens, sorted_distinct_numeric_tokens)
    """
    if not address or address.strip().lower() in LITERAL_NULLS:
        return "", []

    text = address.strip()
    text = transliterate_text(text)
    text = strip_accents(text)
    text = text.replace('&', ' and ').replace('@', ' at ')
    text = text.lower()
    text = text.replace('.', '')

    # Expand ordinals (first -> 1st, etc.)
    for word, repl in ORDINAL_EXPANSIONS:
        text = re.sub(rf'\b{word}\b', repl, text)

    # Expand common street and locality abbreviations
    for pattern, replacement in ADDR_ABBREVIATIONS:
        text = pattern.sub(replacement, text)

    # Extract all distinctive premise/building/street numbers
    numbers = [int(n) for n in RE_NUMERICS.findall(text) if len(n) <= max_digits]
    sorted_numbers = sorted(list(set(numbers)))

    # Clean text tokens (excluding pure numbers)
    clean_text = RE_NON_ALPHANUM.sub(' ', text)
    tokens = [t for t in clean_text.split() if not t.isdigit()]

    return " ".join(tokens), sorted_numbers

