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
    (re.compile(r'\bpl\b', re.IGNORECASE), 'place'),
    (re.compile(r'\bsq\b', re.IGNORECASE), 'square'),
    (re.compile(r'\bhwy\b', re.IGNORECASE), 'highway'),
    (re.compile(r'\bcir\b', re.IGNORECASE), 'circle'),
    (re.compile(r'\bter\b', re.IGNORECASE), 'terrace'),
    (re.compile(r'\bbldg\b', re.IGNORECASE), 'building'),
    (re.compile(r'\bfl\b', re.IGNORECASE), 'floor'),
]

# 4. Ordinal Number Mappings (Phase 0 spec: word -> number, e.g. first -> 1st)
ORDINAL_EXPANSIONS: List[Tuple[str, str]] = [
    ('first', '1st'),
    ('second', '2nd'),
    ('third', '3rd'),
    ('fourth', '4th'),
    ('fifth', '5th'),
    ('sixth', '6th'),
    ('seventh', '7th'),
    ('eighth', '8th'),
    ('ninth', '9th'),
    ('tenth', '10th'),
    ('eleventh', '11th'),
    ('twelfth', '12th'),
    ('thirteenth', '13th'),
    ('fourteenth', '14th'),
    ('fifteenth', '15th'),
    ('sixteenth', '16th'),
    ('seventeenth', '17th'),
    ('eighteenth', '18th'),
    ('nineteenth', '19th'),
    ('twentieth', '20th'),
]

# 5. Business Suffix Expansions (Phase 0 spec: corp -> corporation, inc -> incorporated, etc.)
BUSINESS_SUFFIX_EXPANSIONS: List[Tuple[str, str]] = [
    # Multi-word first
    ('private limited', 'private limited'),
    ('pvt ltd', 'private limited'),
    ('praivet limited', 'private limited'),
    ('praivett limited', 'private limited'),
    ('limited liability company', 'limited liability company'),
    ('limited liability partnership', 'limited liability partnership'),
    # Single-word legal suffixes
    ('corp', 'corporation'),
    ('inc', 'incorporated'),
    ('ltd', 'limited'),
    ('pvt', 'private'),
    ('co', 'company'),
    ('llc', 'limited liability company'),
    ('llp', 'limited liability partnership'),
    # France legal entity expansions
    ('sarl', 'societe a responsabilite limitee'),
    ('sasu', 'societe par actions simplifiee unipersonnelle'),
    ('sas', 'societe par actions simplifiee'),
    ('eurl', 'entreprise unipersonnelle a responsabilite limitee'),
    ('sa', 'societe anonyme'),
    ('gie', 'groupement d interet economique'),
    ('sci', 'societe civile immobiliere'),
]

# 6. Country Alias Normalization Map
COUNTRY_ALIASES = {
    'usa': 'US',
    'united states': 'US',
    'united states of america': 'US',
    'u.s.a.': 'US',
    'u.s.': 'US',
    'us': 'US',
    'india': 'INDIA',
    'ind': 'INDIA',
    'in': 'INDIA',
    'bharat': 'INDIA',
    'france': 'FRANCE',
    'fr': 'FRANCE',
}

# 7. Fallback Identifiers for Empty String Safety (Phase 0 spec)
UNKNOWN_NAME = "UNKNOWN_NAME"
UNKNOWN_ADDRESS = "UNKNOWN_ADDRESS"

# 8. Numerics and Alphanumerics
RE_NUMERICS = re.compile(r'\b\d+')
RE_NON_ALPHANUM = re.compile(r'[^\w\s]', re.UNICODE)
RE_SPACES = re.compile(r'\s+')

# 9. Literal null representations
LITERAL_NULLS: Set[str] = {'null', 'none', 'n/a', 'na', '-', 'nan', 'undefined', ''}


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

