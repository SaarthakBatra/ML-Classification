"""
Core Text Normalization Primitives & Regular Expressions.

Implements high-speed compiled patterns for:
- Legal entity suffixes across US, India, and France
- Standard address street abbreviations
- URL and web domain slug unpackers
- Diacritic / accent decomposition (NFKD)
"""

import re
import unicodedata
from typing import Set, Tuple, List

# 1. URL / Domain Pattern
RE_URL = re.compile(
    r'(?:https?://)?(?:www\.)?([a-zA-Z0-9-]+)\.(?:com|in|co\.in|org|net|fr|gov|io|edu|biz|info)(?:/[^\s]*)?',
    re.IGNORECASE
)

# 2. Comprehensive Legal Suffixes across US, India, France
# Word boundaries are strictly enforced to avoid corrupting tokens like 'costco'
LEGAL_SUFFIXES = (
    r'\b('
    # US / UK common
    r'incorporated|corporation|limited liability company|limited|company|'
    r'associates|enterprises|foundation|holdings|group|'
    r'inc|corp|llc|llp|ltd|co|'
    # India common
    r'private limited|private|pvt ltd|pvt|opc|'
    # Transliterated Indian legal suffixes (from Devanagari, Bengali, Telugu, etc.)
    r'praivet limited|praivett limited|praivet|praivett|'
    # France common
    r'sarl|sasu|sas|sci|eurl|gie|snc|sca|cie|sa'
    r')\b'
)
RE_LEGAL_SUFFIXES = re.compile(LEGAL_SUFFIXES, re.IGNORECASE)

# 3. Standard Address Abbreviations (Word-bounded)
ADDR_ABBREVIATIONS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r'\bst\b', re.IGNORECASE), 'street'),
    (re.compile(r'\brd\b', re.IGNORECASE), 'road'),
    (re.compile(r'\bave\b', re.IGNORECASE), 'avenue'),
    (re.compile(r'\bblvd\b', re.IGNORECASE), 'boulevard'),
    (re.compile(r'\bdr\b', re.IGNORECASE), 'drive'),
    (re.compile(r'\bln\b', re.IGNORECASE), 'lane'),
    (re.compile(r'\bct\b', re.IGNORECASE), 'court'),
    (re.compile(r'\bapt\b', re.IGNORECASE), 'apartment'),
    (re.compile(r'\bste\b', re.IGNORECASE), 'suite'),
    (re.compile(r'\bpkwy\b', re.IGNORECASE), 'parkway'),
    (re.compile(r'\br\b', re.IGNORECASE), 'rue'),  # French road indicator
]

# 4. Numerics and Alphanumerics
RE_NUMERICS = re.compile(r'\b\d+')
RE_NON_ALPHANUM = re.compile(r'[^\w\s]', re.UNICODE)
RE_SPACES = re.compile(r'\s+')

# 5. Literal null representations
LITERAL_NULLS: Set[str] = {'null', 'none', 'n/a', 'na', '-', 'nan', 'undefined'}


try:
    import anyascii

    def transliterate_text(text: str) -> str:
        """
        Transliterates non-Latin scripts (Devanagari, Bengali, Telugu, Tamil, etc.)
        and non-ASCII characters into phonetic ASCII Latin representations.
        """
        if not text:
            return ""
        return anyascii.anyascii(text)
except ImportError:
    def transliterate_text(text: str) -> str:
        """Fallback when anyascii is not available."""
        return text


def strip_accents(text: str) -> str:
    """Decompose and strip diacritics/accents (e.g. café -> cafe, Société -> Societe)."""
    if not text:
        return ""
    nfkd = unicodedata.normalize('NFKD', text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def clean_url_slug(text: str) -> str:
    """Extract and space-delimit domain names (e.g. 'foo-bar.com' -> 'foo bar')."""
    if not text:
        return ""
    def _repl(match):
        slug = match.group(1)
        return slug.replace('-', ' ').replace('_', ' ')
    return RE_URL.sub(_repl, text)

