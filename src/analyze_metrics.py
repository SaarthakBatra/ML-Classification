"""
Backward-compatibility shim for candidate pair metrics analyzer.
Re-exports analyze_candidate_file from src/evaluation/analyzer.py.
"""

import os
import sys
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, RESULT_DIR)
sys.path.insert(0, BASE_DIR)

from evaluation.analyzer import analyze_candidate_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze candidate pairs TSV")
    parser.add_argument(
        "--candidate",
        type=str,
        default=os.path.join(RESULT_DIR, "output", "candidate_pairs.tsv"),
        help="Path to candidate pairs TSV"
    )
    parser.add_argument(
        "--targets",
        type=int,
        default=9969589,
        help="Total number of target records in target pool"
    )
    args = parser.parse_args()
    analyze_candidate_file(args.candidate, args.targets, verbose=True)
