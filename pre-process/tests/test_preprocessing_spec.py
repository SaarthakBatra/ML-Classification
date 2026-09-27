"""
Comprehensive Unit Tests for Preprocessing_1.0.md Specification.

Validates all 13 core requirements:
1. Fake-null literal parsing
2. Unicode NFKD diacritics stripping (French accents)
3. Lowercasing and whitespace collapsing
4. Acronym protection (I.B.M. -> IBM)
5. Symbol mapping (@ -> 'at', & -> 'and')
6. Special character stripping
7. Ordinal number mapping (first -> 1st)
8. Business suffix expansion (corp -> corporation, inc -> incorporated)
9. Address abbreviation expansion (st -> street, ave -> avenue)
10. Empty string safety fallback (UNKNOWN_NAME, UNKNOWN_ADDRESS)
11. Country alias normalization (usa -> US, ind -> INDIA, fr -> FRANCE)
12. Target union with explicit source column (S2, S3)
13. Strict intra-country partitioning
"""

import os
import sys
import unittest
import tempfile
import polars as pl

# Setup sys.path
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PRE_PROCESS_DIR = os.path.dirname(TEST_DIR)
SRC_DIR = os.path.join(PRE_PROCESS_DIR, "src")
for p in [PRE_PROCESS_DIR, SRC_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from normalizers.cleaner import (
    build_clean_name_expr,
    build_clean_address_expr,
    build_clean_country_expr,
    clean_single_name,
    clean_single_address,
    UNKNOWN_NAME,
    UNKNOWN_ADDRESS,
)
from pipeline.preprocessor import clean_dataframe, run_preprocessing_pipeline


class TestPreprocessingSpec(unittest.TestCase):
    """Test suite validating Preprocessing_1.0.md adherence."""

    def test_unicode_diacritics_stripping(self):
        """NFKD diacritics stripping (café -> cafe, Société -> societe)."""
        self.assertEqual(clean_single_name("café de la paix"), "cafe de la paix")
        self.assertEqual(clean_single_name("Société Générale"), "societe generale")
        self.assertEqual(clean_single_name("Dréxkor"), "drexkor")

    def test_acronym_protection_and_case(self):
        """Periods removed without space replacement (I.B.M. -> ibm, identical to IBM)."""
        self.assertEqual(clean_single_name("I.B.M."), "ibm")
        self.assertEqual(clean_single_name("IBM"), "ibm")
        self.assertEqual(clean_single_name("I.B.M."), clean_single_name("IBM"))
        self.assertEqual(clean_single_name("U.S.A. Tech"), "usa tech")

    def test_symbol_mapping(self):
        """@ -> at, & -> and mapping."""
        self.assertEqual(clean_single_name("Coffee @ Paris"), "coffee at paris")
        self.assertEqual(clean_single_name("A & B Enterprises"), "a and b enterprises")
        self.assertEqual(clean_single_address("Corner of 5th & Main"), "corner of 5th and main")

    def test_ordinal_number_mapping(self):
        """Word ordinals mapped to numbers (first -> 1st, second -> 2nd, etc.)."""
        self.assertIn("1st avenue", clean_single_address("First Avenue"))
        self.assertIn("2nd street", clean_single_address("Second St"))
        self.assertIn("3rd floor", clean_single_address("Third Fl"))
        self.assertIn("4th floor", clean_single_address("Fourth Floor"))
        self.assertIn("10th avenue", clean_single_address("Tenth Ave"))

    def test_business_suffix_expansion(self):
        """Legal suffixes expanded (corp -> corporation, inc -> incorporated, ltd -> limited)."""
        self.assertEqual(clean_single_name("Zephay Labs Inc"), "zephay labs incorporated")
        self.assertEqual(clean_single_name("Vision Partners Corp"), "vision partners corporation")
        self.assertEqual(clean_single_name("Nexus Retail Ltd"), "nexus retail limited")
        self.assertEqual(clean_single_name("Apex Ventures LLC"), "apex ventures limited liability company")
        self.assertEqual(clean_single_name("Horizon Global Co"), "horizon global company")
        self.assertEqual(clean_single_name("Starlight Pvt Ltd"), "starlight private limited")

    def test_address_abbreviation_expansion(self):
        """Address abbreviations expanded (st -> street, rd -> road, ave -> avenue, etc.)."""
        self.assertEqual(clean_single_address("1064 Newton Rd"), "1064 newton road")
        self.assertEqual(clean_single_address("85 Wayne Ave"), "85 wayne avenue")
        self.assertEqual(clean_single_address("12 Main St"), "12 main street")
        self.assertEqual(clean_single_address("700 Ocean Blvd"), "700 ocean boulevard")
        self.assertEqual(clean_single_address("Apt 4B"), "apartment 4b")
        self.assertEqual(clean_single_address("Suite 100"), "suite 100")

    def test_empty_string_safety_fallback(self):
        """Empty, whitespace, or literal fake-null strings map to fallback values."""
        null_literals = ["", "   ", "n/a", "na", "null", "NULL", "none", "-", "--", "undefined", "nan", "???"]
        for val in null_literals:
            with self.subTest(val=val):
                self.assertEqual(clean_single_name(val), UNKNOWN_NAME)
                self.assertEqual(clean_single_address(val), UNKNOWN_ADDRESS)

    def test_country_normalization(self):
        """Country alias standardization."""
        df = pl.DataFrame({
            "country": ["usa", "united states", "u.s.a.", "US", "ind", "india", "bharat", "in", "France", "fr"]
        })
        clean_df = df.with_columns(build_clean_country_expr().alias("country"))
        expected = ["US", "US", "US", "US", "INDIA", "INDIA", "INDIA", "INDIA", "FRANCE", "FRANCE"]
        self.assertEqual(clean_df["country"].to_list(), expected)

    def test_vectorized_polars_clean_dataframe(self):
        """Test clean_dataframe function with Polars DataFrame."""
        df = pl.DataFrame({
            "entity_id": ["S1-001", "S1-002", "S1-003"],
            "business_name": ["Zephay Labs Inc.", "I.B.M. & Sons", "---"],
            "business_address": ["2621 Cotten Rd", "1st First Ave", "n/a"],
            "country": ["usa", "France", "india"],
        })
        cleaned = clean_dataframe(df, source_tag="S1")

        self.assertIn("clean_name", cleaned.columns)
        self.assertIn("clean_address", cleaned.columns)
        self.assertIn("source", cleaned.columns)

        self.assertEqual(cleaned["clean_name"][0], "zephay labs incorporated")
        self.assertEqual(cleaned["clean_name"][1], "ibm and sons")
        self.assertEqual(cleaned["clean_name"][2], UNKNOWN_NAME)

        self.assertEqual(cleaned["clean_address"][0], "2621 cotten road")
        self.assertEqual(cleaned["clean_address"][1], "1st 1st avenue")
        self.assertEqual(cleaned["clean_address"][2], UNKNOWN_ADDRESS)

        self.assertEqual(cleaned["country"].to_list(), ["US", "FRANCE", "INDIA"])
        self.assertEqual(cleaned["source"].to_list(), ["S1", "S1", "S1"])

    def test_blocking_representations_preserve_semantic_script_and_normalize_lexical_text(self):
        cleaned = clean_dataframe(pl.DataFrame({
            "entity_id": ["S1-1"],
            "business_name": ["एसएस Food Pvt. Ltd."],
            "business_address": ["12 Main St"],
            "country": ["India"],
        }), source_tag="S1")

        self.assertEqual(cleaned["name_for_faiss"][0], "एसएस food pvt. ltd.")
        self.assertEqual(cleaned["name_for_bm25"][0], cleaned["clean_name"][0])
        self.assertEqual(cleaned["addr_for_bm25"][0], cleaned["clean_address"][0])

    def test_end_to_end_partitioning_pipeline(self):
        """Test complete pipeline execution with mock TSV files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            s1_file = os.path.join(tmp_dir, "s1.tsv")
            s2_file = os.path.join(tmp_dir, "s2.tsv")
            s3_file = os.path.join(tmp_dir, "s3.tsv")
            out_dir = os.path.join(tmp_dir, "preprocessed")

            # Create mock TSVs
            pl.DataFrame({
                "entity_id": ["S1-1", "S1-2", "S1-3"],
                "business_name": ["Acme Corp", "Boulangerie Parisienne", "Bharat Electronics Ltd"],
                "business_address": ["123 Main St", "10 Rue de la Paix", "5th Cross Road, Bangalore"],
                "country": ["US", "France", "India"]
            }).write_csv(s1_file, separator="\t")

            pl.DataFrame({
                "entity_id": ["S2-1", "S2-2"],
                "business_name": ["Acme Corporation", "Boulangerie Parisienne SAS"],
                "business_address": ["123 Main Street", "10 Rue de la Paix"],
                "country": ["usa", "FR"]
            }).write_csv(s2_file, separator="\t")

            pl.DataFrame({
                "entity_id": ["S3-1", "S3-2"],
                "business_name": ["Acme Inc", "Bharat Electronics Limited"],
                "business_address": ["123 Main St", "5th Cross Rd"],
                "country": ["US", "india"]
            }).write_csv(s3_file, separator="\t")

            # Run pipeline
            partitions = run_preprocessing_pipeline(
                s1_path=s1_file,
                s2_path=s2_file,
                s3_path=s3_file,
                output_dir=out_dir,
                export_formats=("parquet", "tsv"),
                verbose=False
            )

            # Verify partitions exist
            self.assertIn("US", partitions)
            self.assertIn("FRANCE", partitions)
            self.assertIn("INDIA", partitions)

            # Verify US partition
            q_us = partitions["US"]["query"]
            t_us = partitions["US"]["target"]
            self.assertEqual(len(q_us), 1)
            self.assertEqual(len(t_us), 2)  # S2-1 and S3-1
            self.assertEqual(sorted(t_us["source"].to_list()), ["S2", "S3"])

            # Verify Target columns
            self.assertIn("entity_id", t_us.columns)
            self.assertIn("clean_name", t_us.columns)
            self.assertIn("clean_address", t_us.columns)
            self.assertIn("source", t_us.columns)

            # Verify exported files exist on disk
            self.assertTrue(os.path.isfile(os.path.join(out_dir, "Query_US.parquet")))
            self.assertTrue(os.path.isfile(os.path.join(out_dir, "Query_US.tsv")))
            self.assertTrue(os.path.isfile(os.path.join(out_dir, "Target_US.parquet")))
            self.assertTrue(os.path.isfile(os.path.join(out_dir, "Target_US.tsv")))
            self.assertTrue(os.path.isfile(os.path.join(out_dir, "manifest.json")))


if __name__ == "__main__":
    unittest.main()
