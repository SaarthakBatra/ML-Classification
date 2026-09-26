"""
Track B: Moderate Normalization Engine.

Designed for multilingual dense embeddings (SentenceTransformers / FAISS):
- Preserves semantic syntax, non-Latin scripts (Tamil, Devanagari, etc.), and French accents
- Normalizes Unicode to Canonical Composition (NFC)
- Expands basic street abbreviations without destroying natural language context
- Produces bi-encoder input strings: `embed_name` and `embed_combined`
"""

import unicodedata
from typing import Tuple
from normalizers.base import (
    ADDR_ABBREVIATIONS,
    RE_SPACES,
    LITERAL_NULLS,
)


def normalize_moderate(name: str, address: str) -> Tuple[str, str]:
    """
    Moderate normalization preserving cross-lingual semantic context.

    Parameters:
        name: Raw business name
        address: Raw business address

    Returns:
        Tuple of (embed_name, embed_combined)
    """
    clean_name = (name or "").strip()
    clean_addr = (address or "").strip()

    if clean_name.lower() in LITERAL_NULLS:
        clean_name = ""
    if clean_addr.lower() in LITERAL_NULLS:
        clean_addr = ""

    # NFC Normalization for Unicode consistency across distinct OS encodings
    clean_name = unicodedata.normalize('NFC', clean_name)
    clean_addr = unicodedata.normalize('NFC', clean_addr)

    # Moderate address expansion
    for pattern, replacement in ADDR_ABBREVIATIONS:
        clean_addr = pattern.sub(replacement, clean_addr)

    clean_name = RE_SPACES.sub(' ', clean_name)
    clean_addr = RE_SPACES.sub(' ', clean_addr)

    if clean_name and clean_addr:
        embed_combined = f"{clean_name} | {clean_addr}"
    elif clean_name:
        embed_combined = clean_name
    elif clean_addr:
        embed_combined = clean_addr
    else:
        embed_combined = ""

    return clean_name, embed_combined
