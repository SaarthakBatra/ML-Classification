"""
Comprehensive Benchmark for Multi-Channel Blocking and Fusion.
Measures latency and marginal recall gains on the 2,000 Source 1 validation split.
"""

import os
import sys
import time

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.dirname(BENCH_DIR)
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


def run_benchmark():
    print("=" * 70)
    print("BENCHMARK: MULTI-CHANNEL BLOCKING AND FUSION LATENCY & RECALL")
    print("=" * 70)

    s1_df, target_df, gt_map = load_validation_split(2000)
    print(f"Validation pool: {len(s1_df):,} S1 queries, {len(target_df):,} targets")

    t_norm = time.time()
    s1_norm = process_dataframe(s1_df)
    target_norm = process_dataframe(target_df)
    print(f"Normalized all records in {time.time()-t_norm:.2f}s")

    s1_ids = s1_norm["entity_id"].to_list()
    n_targets = len(target_norm)

    # Channel B1
    t0 = time.time()
    b1 = NumericGeoIndexer().fit(target_norm)
    c1 = b1.retrieve(s1_norm, 30)
    t_b1 = time.time() - t0
    m1 = evaluate_blocking(c1, gt_map, n_targets, verbose=False)

    # Channel B2
    t0 = time.time()
    b2 = CoreNameTokenIndexer().fit(target_norm)
    c2 = b2.retrieve(s1_norm, 30)
    t_b2 = time.time() - t0
    m2 = evaluate_blocking(c2, gt_map, n_targets, verbose=False)

    # Channel B3
    t0 = time.time()
    b3 = FuzzyCoreIndexer().fit(target_norm)
    c3 = b3.retrieve(s1_norm, 20)
    t_b3 = time.time() - t0
    m3 = evaluate_blocking(c3, gt_map, n_targets, verbose=False)

    # Channel B4
    t0 = time.time()
    b4 = AddressTokenIndexer().fit(target_norm)
    c4 = b4.retrieve(s1_norm, 30)
    t_b4 = time.time() - t0
    m4 = evaluate_blocking(c4, gt_map, n_targets, verbose=False)

    # Fused
    t0 = time.time()
    fused = fuse_candidate_channels(
        channel_results=[
            ("B1_numeric", c1, 3.0),
            ("B2_tokens", c2, 2.5),
            ("B3_fuzzy", c3, 2.0),
            ("B4_addr_tokens", c4, 2.5),
        ],
        s1_ids=s1_ids,
        hard_budget_cap=20
    )
    t_fused = time.time() - t0
    m_fused = evaluate_blocking(fused, gt_map, n_targets, verbose=False)

    print("\nBENCHMARK SUMMARY:")
    print("-" * 70)
    print(f"  Channel B1 (Numeric Geo):  {m1['pair_recall']*100:.2f}% recall | {m1['avg_candidates_per_s1']:.1f} cands/s1 | {t_b1:.3f}s")
    print(f"  Channel B2 (Name Tokens):  {m2['pair_recall']*100:.2f}% recall | {m2['avg_candidates_per_s1']:.1f} cands/s1 | {t_b2:.3f}s")
    print(f"  Channel B3 (Fuzzy Ratio):  {m3['pair_recall']*100:.2f}% recall | {m3['avg_candidates_per_s1']:.1f} cands/s1 | {t_b3:.3f}s")
    print(f"  Channel B4 (Addr Tokens):  {m4['pair_recall']*100:.2f}% recall | {m4['avg_candidates_per_s1']:.1f} cands/s1 | {t_b4:.3f}s")
    print(f"  --> FUSED (Cap K=20):      {m_fused['pair_recall']*100:.2f}% recall | {m_fused['avg_candidates_per_s1']:.1f} cands/s1 | {t_fused:.3f}s")
    print(f"  --> Reduction Ratio:       {m_fused['reduction_ratio']*100:.4f}%")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()
