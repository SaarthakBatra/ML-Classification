"""
Comprehensive Candidate Pairs Distribution & Statistical Analyzer.

Computes:
1. Cartesian Space Reduction & Reduction Ratio
2. Candidate Count Statistics (Mean, Median, Std, P25, P50, P75, P90, P95, P99, Max, Min)
3. Zero-Candidate Rate
4. Target Source Split (Source 2 vs Source 3 distribution)
5. Candidate Count Histograms
"""

import os
from typing import Dict, Any
import numpy as np


def analyze_candidate_file(
    candidate_path: str,
    n_total_targets: int = 9969589,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Analyzes generated candidate_pairs.tsv file.

    Parameters:
        candidate_path: Absolute or relative path to candidate_pairs.tsv
        n_total_targets: Total number of records in target pool (S2 + S3)
        verbose: Whether to print human-readable console report

    Returns:
        Dictionary of computed summary statistics
    """
    if not os.path.isfile(candidate_path):
        raise FileNotFoundError(f"File not found: {candidate_path}")

    counts = []
    s2_counts = 0
    s3_counts = 0
    zero_cand_s1s = 0
    total_pairs = 0
    s1_ids = []

    with open(candidate_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        assert header == ["source1_entity_id", "candidate_entity_ids"], f"Invalid header: {header}"

        for line in f:
            parts = line.strip().split("\t")
            s1_id = parts[0]
            s1_ids.append(s1_id)
            cands_str = parts[1] if len(parts) > 1 else ""
            if not cands_str:
                counts.append(0)
                zero_cand_s1s += 1
            else:
                cands = cands_str.split(",")
                num_cands = len(cands)
                counts.append(num_cands)
                total_pairs += num_cands
                for c in cands:
                    if c.startswith("S2-"):
                        s2_counts += 1
                    elif c.startswith("S3-"):
                        s3_counts += 1

    counts_arr = np.array(counts)
    n_s1 = len(counts_arr)

    cartesian_space = n_s1 * n_total_targets
    reduction_ratio = 1.0 - (total_pairs / cartesian_space) if cartesian_space > 0 else 1.0

    stats = {
        "n_s1": n_s1,
        "n_total_targets": n_total_targets,
        "total_pairs": total_pairs,
        "cartesian_space": cartesian_space,
        "reduction_ratio": reduction_ratio,
        "mean_cands": float(np.mean(counts_arr)),
        "median_cands": float(np.median(counts_arr)),
        "std_cands": float(np.std(counts_arr)),
        "min_cands": int(np.min(counts_arr)),
        "p25": float(np.percentile(counts_arr, 25)),
        "p75": float(np.percentile(counts_arr, 75)),
        "p90": float(np.percentile(counts_arr, 90)),
        "p95": float(np.percentile(counts_arr, 95)),
        "p99": float(np.percentile(counts_arr, 99)),
        "max_cands": int(np.max(counts_arr)),
        "zero_candidate_s1s": zero_cand_s1s,
        "zero_candidate_pct": zero_cand_s1s / n_s1 * 100.0 if n_s1 > 0 else 0.0,
        "s2_counts": s2_counts,
        "s3_counts": s3_counts,
    }

    if verbose:
        print("=" * 75)
        print(f"METRICS REPORT FOR CANDIDATE PAIRS: {os.path.basename(candidate_path)}")
        print("=" * 75)
        print("\n--- 1. REDUCTION RATIO & SEARCH SPACE ---")
        print(f"  Source 1 Query Entities:       {n_s1:,}")
        print(f"  Total Target Search Pool:      {n_total_targets:,}")
        print(f"  Total Cartesian Comparisons:   {cartesian_space:,}")
        print(f"  Filtered Candidate Pairs:      {total_pairs:,}")
        print(f"  Pairs Eliminated:              {cartesian_space - total_pairs:,}")
        print(f"  --> REDUCTION RATIO (RR):       {reduction_ratio * 100:.6f}%")
        print(f"  --> Search Space Pruned:       {(1.0 - total_pairs/cartesian_space)*100:.4f}% of Cartesian product")

        print("\n--- 2. CANDIDATE DISTRIBUTION METRICS (PER S1 ENTITY) ---")
        print(f"  Mean Candidates / S1:          {stats['mean_cands']:.2f}")
        print(f"  Median Candidates (P50):       {stats['median_cands']:.1f}")
        print(f"  Standard Deviation:            {stats['std_cands']:.2f}")
        print(f"  Min Candidates:                {stats['min_cands']}")
        print(f"  25th Percentile (P25):         {stats['p25']:.1f}")
        print(f"  75th Percentile (P75):         {stats['p75']:.1f}")
        print(f"  90th Percentile (P90):         {stats['p90']:.1f}")
        print(f"  95th Percentile (P95):         {stats['p95']:.1f}")
        print(f"  99th Percentile (P99):         {stats['p99']:.1f}")
        print(f"  Max Candidates (Budget Cap):   {stats['max_cands']}")
        print(f"  Zero-Candidate S1s:            {zero_cand_s1s} ({stats['zero_candidate_pct']:.2f}%)")

        print("\n--- 3. TARGET SOURCE CANDIDATE DISTRIBUTION ---")
        s2_pct = s2_counts / total_pairs * 100 if total_pairs > 0 else 0
        s3_pct = s3_counts / total_pairs * 100 if total_pairs > 0 else 0
        s2_s3_ratio = s2_counts / s3_counts if s3_counts > 0 else 0
        print(f"  Source 2 (S2) Candidates:      {s2_counts:,} ({s2_pct:.2f}%)")
        print(f"  Source 3 (S3) Candidates:      {s3_counts:,} ({s3_pct:.2f}%)")
        print(f"  S2 : S3 Candidate Ratio:       {s2_s3_ratio:.2f} : 1.00")

        print("\n--- 4. CANDIDATE COUNT HISTOGRAM ---")
        bucket_ranges = [(0, 0), (1, 5), (6, 10), (11, 15), (16, 19), (20, 20)]
        for low, high in bucket_ranges:
            if low == high:
                cnt = int(np.sum(counts_arr == low))
                label = f"K = {low:2d}"
            else:
                cnt = int(np.sum((counts_arr >= low) & (counts_arr <= high)))
                label = f"{low:2d} <= K <= {high:2d}"
            bar = "█" * int(cnt / n_s1 * 40)
            print(f"  {label}: {cnt:5d} ({cnt/n_s1*100:5.1f}%) | {bar}")

        print("=" * 75)

    return stats
