"""
Evaluation Metrics Harness for Candidate Pair Generation.

Computes:
- Pair Recall: Fraction of all true matches present in generated candidate pairs
- Reduction Ratio: Search space pruning relative to full Cartesian product
- Average / Max candidates per entity
- Zero-candidate query counts
"""

from typing import Dict, List, Set


def evaluate_blocking(
    candidate_map: Dict[str, List[str]],
    gt_map: Dict[str, Set[str]],
    n_targets: int,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Evaluates candidate generation quality against ground truth.

    Parameters:
        candidate_map: Dict mapping query s1_id -> list of candidate target IDs
        gt_map: Dict mapping query s1_id -> set of ground truth target IDs
        n_targets: Total number of targets in search pool
        verbose: Whether to print formatted report

    Returns:
        Dictionary of computed metrics
    """
    total_true_matches = 0
    found_true_matches = 0
    total_candidate_pairs = 0
    zero_candidates = 0
    max_candidates = 0

    for s1_id, true_set in gt_map.items():
        candidates = set(candidate_map.get(s1_id, []))
        total_candidate_pairs += len(candidates)
        if len(candidates) == 0:
            zero_candidates += 1
        if len(candidates) > max_candidates:
            max_candidates = len(candidates)

        total_true_matches += len(true_set)
        found_true_matches += len(true_set.intersection(candidates))

    n_s1 = len(gt_map)
    recall = found_true_matches / total_true_matches if total_true_matches > 0 else 1.0
    total_possible_pairs = n_s1 * n_targets
    rr = 1.0 - (total_candidate_pairs / total_possible_pairs) if total_possible_pairs > 0 else 1.0
    avg_cands = total_candidate_pairs / n_s1 if n_s1 > 0 else 0.0

    metrics = {
        "pair_recall": recall,
        "reduction_ratio": rr,
        "total_true_matches": total_true_matches,
        "found_true_matches": found_true_matches,
        "avg_candidates_per_s1": avg_cands,
        "max_candidates_per_s1": max_candidates,
        "zero_candidate_s1": zero_candidates,
    }

    if verbose:
        print("\n" + "=" * 55)
        print("BLOCKING EVALUATION REPORT")
        print("=" * 55)
        print(f"  Total S1 Entities Evaluated: {n_s1:,}")
        print(f"  Target Search Pool Size:     {n_targets:,}")
        print(f"  Total True Matches:          {total_true_matches:,}")
        print(f"  True Matches Captured:       {found_true_matches:,}")
        print(f"  --> Pair Recall:             {recall * 100:.2f}%")
        print(f"  --> Reduction Ratio:         {rr * 100:.4f}%")
        print(f"  Avg Candidates / S1:         {avg_cands:.1f}")
        print(f"  Max Candidates / S1:         {max_candidates}")
        print(f"  Zero-Candidate S1s:          {zero_candidates} ({zero_candidates/n_s1*100:.1f}%)")
        print("=" * 55)

    return metrics


def evaluate_candidate_file(
    candidate_path: str,
    ground_truth_path: str,
    n_targets: int = 9969589,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Evaluates candidate_pairs.tsv against ground_truth.tsv.
    Reads candidate pairs and ground truth matches for evaluated S1 entities,
    computing Pair Recall, Reduction Ratio, and distribution metrics.
    """
    import os

    if not os.path.isfile(candidate_path):
        raise FileNotFoundError(f"Candidate file not found: {candidate_path}")
    if not os.path.isfile(ground_truth_path):
        raise FileNotFoundError(f"Ground truth file not found: {ground_truth_path}")

    if verbose:
        print(f"Loading candidate pairs from: {candidate_path}")
    candidate_map: Dict[str, List[str]] = {}
    with open(candidate_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        for line in f:
            parts = line.strip().split("\t")
            if not parts or not parts[0]:
                continue
            s1_id = parts[0]
            cands = parts[1].split(",") if len(parts) > 1 and parts[1] else []
            candidate_map[s1_id] = cands

    s1_evaluated = set(candidate_map.keys())
    if verbose:
        print(f"Loaded {len(candidate_map):,} S1 candidates.")
        print(f"Loading relevant ground truth matches from: {ground_truth_path}")

    gt_map: Dict[str, Set[str]] = {}
    with open(ground_truth_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        for line in f:
            parts = line.strip().split("\t")
            if not parts or not parts[0]:
                continue
            s1_id = parts[0]
            if s1_id in s1_evaluated:
                matches = set(m.strip() for m in parts[1].split(",") if m.strip()) if len(parts) > 1 and parts[1] else set()
                gt_map[s1_id] = matches

    # Fill in any evaluated S1 IDs not found in GT as having 0 true matches
    for s1_id in s1_evaluated:
        if s1_id not in gt_map:
            gt_map[s1_id] = set()

    return evaluate_blocking(
        candidate_map=candidate_map,
        gt_map=gt_map,
        n_targets=n_targets,
        verbose=verbose
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Candidate Pairs TSV against Ground Truth")
    parser.add_argument("--candidate", "-c", required=True, help="Path to candidate_pairs.tsv")
    parser.add_argument("--ground-truth", "-gt", required=True, help="Path to ground_truth.tsv")
    parser.add_argument("--targets", "-n", type=int, default=9969589, help="Total target records in pool")
    args = parser.parse_args()

    evaluate_candidate_file(
        candidate_path=args.candidate,
        ground_truth_path=args.ground_truth,
        n_targets=args.targets,
        verbose=True
    )
