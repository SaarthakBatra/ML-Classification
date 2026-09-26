#!/usr/bin/env python3
"""
CLI Runner for Phase 0 Preprocessing & Partitioning Pipeline.

Implements the complete text normalization and country-partitioning workflow
specified in Preprocessing_1.0.md.

Usage:
    python run_preprocess.py --dataset test
    python run_preprocess.py --dataset train
    python run_preprocess.py --s1 path/to/s1.tsv --s2 path/to/s2.tsv --s3 path/to/s3.tsv
    python run_preprocess.py --sample-s1 10000 --format parquet
"""

import os
import sys
import argparse
from typing import Tuple

# Setup sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AMAZON_DIR = os.path.dirname(SCRIPT_DIR)
PRE_PROCESS_DIR = os.path.join(SCRIPT_DIR, "pre-process")
SRC_DIR = os.path.join(PRE_PROCESS_DIR, "src")

for path in [SCRIPT_DIR, PRE_PROCESS_DIR, SRC_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

from pipeline.preprocessor import run_preprocessing_pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        prog="run_preprocess",
        description=(
            "=======================================================================\n"
            "Phase 0 Preprocessing & Partitioning — Polars Engine\n"
            "Strictly implements Preprocessing_1.0.md specifications:\n"
            "- Unicode NFKD diacritics stripping (café -> cafe)\n"
            "- Acronym protection (I.B.M. -> IBM)\n"
            "- Business suffix expansion (corp -> corporation, inc -> incorporated)\n"
            "- Address abbreviation expansion (st -> street, rd -> road, ave -> avenue)\n"
            "- Ordinal number mapping (first -> 1st, second -> 2nd)\n"
            "- Empty string safety fallback (UNKNOWN_NAME, UNKNOWN_ADDRESS)\n"
            "- Strict intra-country partitioning (Query_<Country>, Target_<Country>)\n"
            "- Unified Target Pool (S2 ∪ S3) with explicit 'source' tracking\n"
            "======================================================================="
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--dataset",
        choices=["test", "train"],
        default="test",
        help="Target dataset split to preprocess (default: 'test')",
    )
    parser.add_argument(
        "--s1",
        type=str,
        default=None,
        help="Explicit path to Source 1 TSV file (overrides --dataset)",
    )
    parser.add_argument(
        "--s2",
        type=str,
        default=None,
        help="Explicit path to Source 2 TSV file (overrides --dataset)",
    )
    parser.add_argument(
        "--s3",
        type=str,
        default=None,
        help="Explicit path to Source 3 TSV file (overrides --dataset)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Output directory for partitioned datasets (default: pre-process/data/preprocessed/<dataset>)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="parquet,tsv",
        help="Comma-separated serialization formats: 'parquet', 'tsv', or 'parquet,tsv' (default: 'parquet,tsv')",
    )
    parser.add_argument(
        "--sample-s1",
        type=int,
        default=None,
        help="Sample N rows from Source 1 for rapid prototyping/benchmarking",
    )
    parser.add_argument(
        "--sample-target",
        type=int,
        default=None,
        help="Sample N rows from Source 2 and 3 for rapid prototyping/benchmarking",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable detailed progress logging",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Determine default paths based on dataset choice
    dataset_dir = os.path.join(AMAZON_DIR, "dataset", args.dataset)
    s1_path = args.s1 or os.path.join(dataset_dir, f"{args.dataset}_source1.tsv")
    s2_path = args.s2 or os.path.join(dataset_dir, f"{args.dataset}_source2.tsv")
    s3_path = args.s3 or os.path.join(dataset_dir, f"{args.dataset}_source3.tsv")

    output_dir = args.output_dir or os.path.join(PRE_PROCESS_DIR, "data", "preprocessed", args.dataset)

    export_formats = tuple(f.strip().lower() for f in args.format.split(",") if f.strip())
    if not export_formats:
        export_formats = ("parquet", "tsv")

    # Execute pipeline
    run_preprocessing_pipeline(
        s1_path=s1_path,
        s2_path=s2_path,
        s3_path=s3_path,
        output_dir=output_dir,
        export_formats=export_formats,
        sample_s1=args.sample_s1,
        sample_target=args.sample_target,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
