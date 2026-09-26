"""
Unit and Integration Benchmark Tests for Channel B5 (Dense Multilingual Bi-Encoder) and 5-Channel Fusion.
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
from blocking.b5_dense import DenseEmbeddingRetriever
from fusion.rank_fusion import fuse_candidate_channels
from evaluation.metrics import evaluate_blocking


class TestDenseBlockingPipeline(unittest.TestCase):
    """Integration test suite for Channel B5 and full 5-channel fusion on 1,000 queries."""

    @classmethod
    def setUpClass(cls):
        """Loads, normalizes, and fits B5 once for all tests in class."""
        print("\nLoading and normalizing 1,000 query split for dense retriever test...")
        s1_df, target_df, gt_map = load_validation_split(1000)
        cls.s1_norm = process_dataframe(s1_df)
        cls.target_norm = process_dataframe(target_df)
        cls.gt_map = gt_map
        cls.s1_ids = cls.s1_norm["entity_id"].to_list()
        cls.n_targets = len(cls.target_norm)
        print("Fitting DenseEmbeddingRetriever...")
        cls.b5 = DenseEmbeddingRetriever(batch_size=256)
        cls.b5.fit(cls.target_norm, text_col="embed_combined")
        cls.b5_cands = cls.b5.retrieve(cls.s1_norm, text_col="embed_combined", top_k=15, min_similarity=0.50)

    def test_b5_dense_retriever(self):
        """Test Channel B5: Multilingual Bi-Encoder Retriever."""
        metrics = evaluate_blocking(self.b5_cands, self.gt_map, self.n_targets, verbose=False)

        # Baseline recall: ~85.7%
        self.assertGreater(metrics["pair_recall"], 0.80)
        self.assertGreater(metrics["reduction_ratio"], 0.999)
        self.assertLessEqual(metrics["max_candidates_per_s1"], 15)

    def test_five_channel_fusion(self):
        """Test full 5-channel fusion (B1 + B2 + B3 + B4 + B5) with budget cap K <= 20."""
        b1 = NumericGeoIndexer().fit(self.target_norm)
        b2 = CoreNameTokenIndexer().fit(self.target_norm)
        b3 = FuzzyCoreIndexer().fit(self.target_norm)
        b4 = AddressTokenIndexer().fit(self.target_norm)

        c1 = b1.retrieve(self.s1_norm, 30)
        c2 = b2.retrieve(self.s1_norm, 30)
        c3 = b3.retrieve(self.s1_norm, 20)
        c4 = b4.retrieve(self.s1_norm, 30)
        c5 = self.b5_cands

        fused = fuse_candidate_channels(
            channel_results=[
                ("B1_numeric", c1, 3.0),
                ("B2_tokens", c2, 2.5),
                ("B3_fuzzy", c3, 2.0),
                ("B4_addr_tokens", c4, 2.5),
                ("B5_dense", c5, 2.5),
            ],
            s1_ids=self.s1_ids,
            hard_budget_cap=20
        )

        metrics = evaluate_blocking(fused, self.gt_map, self.n_targets, verbose=False)

        # Baseline recall: ~97.38%, zero-candidate count = 0
        self.assertGreater(metrics["pair_recall"], 0.96)
        self.assertGreater(metrics["reduction_ratio"], 0.9995)
        self.assertLessEqual(metrics["max_candidates_per_s1"], 20)


if __name__ == "__main__":
    unittest.main()
