#!/usr/bin/env python3
"""
Top-level entrypoint for Statistical Metrics & Distribution Analysis of Candidate Pairs.
Run from AMAZON/result directory:
    python analyze_metrics.py --candidate <path>
"""

import os
import sys
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from evaluation.analyzer import analyze_candidate_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="analyze_metrics",
        description="Compute statistical distributions, reduction ratio, percentiles, and histograms for candidate_pairs.tsv"
    )
    parser.add_argument(
        "--candidate", "-c",
        type=str,
        default=os.path.join(BASE_DIR, "output", "candidate_pairs.tsv"),
        help="Path to candidate_pairs.tsv (default: output/candidate_pairs.tsv)"
    )
    parser.add_argument(
        "--targets", "-n",
        type=int,
        default=9969589,
        help="Total number of records in target pool (S2 + S3)"
    )
    args = parser.parse_args()

    analyze_candidate_file(args.candidate, args.targets, verbose=True)
