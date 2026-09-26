"""
Unit Tests for Dual-Track Normalization Engine.
Verifies behavior on real edge cases from US, Indian, and French entities.
"""

import unittest
import os
import sys
import polars as pl

# Ensure src and result root in sys.path
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.dirname(TEST_DIR)
sys.path.insert(0, RESULT_DIR)
sys.path.insert(0, os.path.join(RESULT_DIR, "src"))

from normalizers.aggressive import normalize_aggressive_name, extract_address_features
from normalizers.moderate import normalize_moderate
from normalizers.batch import process_dataframe
from normalizers.base import strip_accents, clean_url_slug


class TestNormalizer(unittest.TestCase):
    """Test suite for text normalization routines."""

    def test_strip_accents(self):
        """Verify diacritic decomposition."""
        self.assertEqual(strip_accents("Société Générale"), "Societe Generale")
        self.assertEqual(strip_accents("Dréxkor"), "Drexkor")
        self.assertEqual(strip_accents("café"), "cafe")
        self.assertEqual(strip_accents(""), "")

    def test_clean_url_slug(self):
        """Verify domain unpacking into constituent tokens."""
        self.assertEqual(clean_url_slug("maurewilliamscolombier.com"), "maurewilliamscolombier")
        self.assertEqual(clean_url_slug("https://www.apple-store.com/us"), "apple store")
        self.assertEqual(clean_url_slug("my-company.co.in"), "my company")

    def test_aggressive_name_normalization(self):
        """Verify suffix stripping, punctuation removal, and alphabetical token sorting."""
        cases = [
            ("Maure Williams Colombier Inc", "colombier maure williams"),
            ("colombier maure williams", "colombier maure williams"),
            ("Raj Investments LLP", "investments raj"),
            ("Dahlia Power Reliable Scientific LLC", "dahlia power reliable scientific"),
            ("Société Générale & Frères SARL", "and freres generale societe"),
            ("", ""),
            ("null", ""),
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(normalize_aggressive_name(raw), expected)

    def test_address_feature_extraction(self):
        """Verify address abbreviation expansion and sorted numeric token extraction."""
        text, nums = extract_address_features("85 Wayne Avenue, Ticonderoga, NY")
        self.assertEqual(nums, [85])
        self.assertIn("wayne", text)
        self.assertIn("avenue", text)

        text, nums = extract_address_features("630 45th Terrace, Kansas City, MO")
        self.assertEqual(nums, [45, 630])

        text, nums = extract_address_features("6(29), C.I.T. Colony, 2Nd Main Road Mylapore, Chennai")
        self.assertEqual(nums, [2, 6, 29])

        text, nums = extract_address_features("12 Allée des Roses, 75008 Paris")
        self.assertEqual(nums, [12, 75008])

        text, nums = extract_address_features("")
        self.assertEqual(nums, [])
        self.assertEqual(text, "")

    def test_moderate_normalization(self):
        """Verify semantic preservation of non-Latin scripts and address expansions."""
        name, combined = normalize_moderate(
            "Raj Investments LLP",
            "6(29), C.I.T. Colony, Chennai"
        )
        self.assertEqual(name, "Raj Investments LLP")
        self.assertEqual(combined, "Raj Investments LLP | 6(29), C.I.T. Colony, Chennai")

        # Tamil script preservation
        tamil_name = "ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி"
        tamil_addr = "6(29), C.i.t. Colony, Chennai, தமிழ்நாடு"
        name_t, comb_t = normalize_moderate(tamil_name, tamil_addr)
        self.assertEqual(name_t, tamil_name)
        self.assertIn(tamil_name, comb_t)
        self.assertIn("தமிழ்நாடு", comb_t)

    def test_batch_dataframe_processing(self):
        """Verify Polars DataFrame batch processing generates expected schema."""
        df = pl.DataFrame({
            "entity_id": ["S1-001", "S2-001"],
            "business_name": ["Maure Williams Colombier Inc", "Société Nouvelle SARL"],
            "business_address": ["85 Wayne Ave, NY", "14 Rue du Faubourg, Paris"],
            "country": ["US", "FRANCE"]
        })
        processed = process_dataframe(df)

        self.assertIn("agg_name", processed.columns)
        self.assertIn("agg_addr", processed.columns)
        self.assertIn("addr_nums", processed.columns)
        self.assertIn("embed_name", processed.columns)
        self.assertIn("embed_combined", processed.columns)

        self.assertEqual(processed["agg_name"][0], "colombier maure williams")
        self.assertEqual(processed["addr_nums"][0].to_list(), [85])
        self.assertEqual(processed["country"][0], "US")
        self.assertEqual(processed["country"][1], "FRANCE")


if __name__ == "__main__":
    unittest.main()
