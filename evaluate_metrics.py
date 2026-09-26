#!/usr/bin/env python3
"""
Top-level entrypoint for Candidate Pair Evaluation against Ground Truth.
Run from AMAZON/result directory:
    python evaluate_metrics.py --candidate <path> --ground-truth <path>
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from evaluation.metrics import evaluate_candidate_file
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="evaluate_metrics",
        description="Evaluate Candidate Pairs TSV against Ground Truth for Pair Recall & Reduction Ratio."
    )
    parser.add_argument(
        "--candidate", "-c",
        type=str,
        default=os.path.join(BASE_DIR, "output", "candidate_pairs.tsv"),
        help="Path to candidate_pairs.tsv"
    )
    parser.add_argument(
        "--ground-truth", "-gt",
        type=str,
        default=os.path.join(os.path.dirname(BASE_DIR), "dataset", "train", "train_ground_truth.tsv"),
        help="Path to ground truth TSV (e.g. train_ground_truth.tsv)"
    )
    parser.add_argument(
        "--targets", "-n",
        type=int,
        default=9969589,
        help="Total target pool size for Reduction Ratio calculation"
    )
    args = parser.parse_args()

    evaluate_candidate_file(
        candidate_path=args.candidate,
        ground_truth_path=args.ground_truth,
        n_targets=args.targets,
        verbose=True
    )
