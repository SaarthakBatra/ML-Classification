"""
Unit and Integration Benchmark Tests for Blocking Channels B1, B2, B3, B4 and Fusion.
Tests against the cached validation split (2,000 Source 1 queries).
"""

import unittest
import os
import sys
import time

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.dirname(TEST_DIR)
sys.path.insert(0, RESULT_DIR)
sys.path.insert(0, os.path.join(RESULT_DIR, "src"))

from validation_dataset import load_validation_split
from normalizers.batch import process_dataframe
from blocking.b1_numeric_geo import NumericGeoIndexer
from blocking.b2_name_tokens import CoreNameTokenIndexer
from blocking.b3_fuzzy import FuzzyCoreIndexer
from blocking.b4_address_tokens import AddressTokenIndexer
from fusion.rank_fusion import fuse_candidate_channels
from evaluation.metrics import evaluate_blocking


class TestBlockingPipeline(unittest.TestCase):
    """Integration test suite for blocking channels on the 2,000 validation split."""

    @classmethod
    def setUpClass(cls):
        """Loads and normalizes the validation dataset once for all test methods."""
        print("\nLoading and normalizing validation dataset for test suite...")
        s1_df, target_df, gt_map = load_validation_split(2000)
        cls.s1_norm = process_dataframe(s1_df)
        cls.target_norm = process_dataframe(target_df)
        cls.gt_map = gt_map
        cls.s1_ids = cls.s1_norm["entity_id"].to_list()
        cls.n_targets = len(cls.target_norm)
        print(f"Ready: {len(cls.s1_norm):,} queries against {cls.n_targets:,} targets.")

    def test_b1_numeric_geo(self):
        """Test Channel B1: Numeric Premise Geo Indexer."""
        b1 = NumericGeoIndexer()
        b1.fit(self.target_norm)
        cands = b1.retrieve(self.s1_norm, max_cands_per_entity=30)
        metrics = evaluate_blocking(cands, self.gt_map, self.n_targets, verbose=False)

        # Baseline recall: ~60.8%, reduction ratio: > 99.98%
        self.assertGreater(metrics["pair_recall"], 0.55)
        self.assertGreater(metrics["reduction_ratio"], 0.999)
        self.assertLessEqual(metrics["max_candidates_per_s1"], 30)

    def test_b2_core_name_tokens(self):
        """Test Channel B2: Core Name Token Indexer + Jaccard."""
        b2 = CoreNameTokenIndexer(max_token_freq_ratio=0.01, jaccard_threshold=0.30)
        b2.fit(self.target_norm)
        cands = b2.retrieve(self.s1_norm, max_cands_per_entity=30)
        metrics = evaluate_blocking(cands, self.gt_map, self.n_targets, verbose=False)

        # Baseline recall: ~79.3%, reduction ratio: > 99.97%
        self.assertGreater(metrics["pair_recall"], 0.75)
        self.assertGreater(metrics["reduction_ratio"], 0.999)
        self.assertLessEqual(metrics["max_candidates_per_s1"], 30)

    def test_b3_fuzzy_core_indexer(self):
        """Test Channel B3: Rapid prefix-bucketing & RapidFuzz."""
        b3 = FuzzyCoreIndexer(min_ratio=70.0)
        b3.fit(self.target_norm)
        cands = b3.retrieve(self.s1_norm, max_cands_per_entity=20)
        metrics = evaluate_blocking(cands, self.gt_map, self.n_targets, verbose=False)

        # Baseline recall: ~63.8%, reduction ratio: > 99.98%
        self.assertGreater(metrics["pair_recall"], 0.60)
        self.assertGreater(metrics["reduction_ratio"], 0.999)
        self.assertLessEqual(metrics["max_candidates_per_s1"], 20)

    def test_b4_address_token_indexer(self):
        """Test Channel B4: Address Token Indexer + Jaccard."""
        b4 = AddressTokenIndexer(max_freq_ratio=0.015, min_jaccard=0.28)
        b4.fit(self.target_norm)
        cands = b4.retrieve(self.s1_norm, max_cands_per_entity=30)
        metrics = evaluate_blocking(cands, self.gt_map, self.n_targets, verbose=False)

        # Baseline recall: ~78.1%, reduction ratio: > 99.98%
        self.assertGreater(metrics["pair_recall"], 0.74)
        self.assertGreater(metrics["reduction_ratio"], 0.999)
        self.assertLessEqual(metrics["max_candidates_per_s1"], 30)

    def test_fused_blocking(self):
        """Test 4-Channel Fusion (B1 + B2 + B3 + B4) with budget cap K <= 20."""
        b1 = NumericGeoIndexer().fit(self.target_norm)
        b2 = CoreNameTokenIndexer().fit(self.target_norm)
        b3 = FuzzyCoreIndexer().fit(self.target_norm)
        b4 = AddressTokenIndexer().fit(self.target_norm)

        c1 = b1.retrieve(self.s1_norm, 30)
        c2 = b2.retrieve(self.s1_norm, 30)
        c3 = b3.retrieve(self.s1_norm, 20)
        c4 = b4.retrieve(self.s1_norm, 30)

        fused = fuse_candidate_channels(
            channel_results=[
                ("B1_numeric", c1, 3.0),
                ("B2_tokens", c2, 2.5),
                ("B3_fuzzy", c3, 2.0),
                ("B4_addr_tokens", c4, 2.5),
            ],
            s1_ids=self.s1_ids,
            hard_budget_cap=20
        )

        metrics = evaluate_blocking(fused, self.gt_map, self.n_targets, verbose=False)

        # Baseline fused recall: ~95.7%, reduction ratio: > 99.97%, budget cap <= 20
        self.assertGreater(metrics["pair_recall"], 0.95)
        self.assertGreater(metrics["reduction_ratio"], 0.9997)
        self.assertLessEqual(metrics["max_candidates_per_s1"], 20)
        self.assertLess(metrics["zero_candidate_s1"] / len(self.s1_ids), 0.02)


if __name__ == "__main__":
    unittest.main()
