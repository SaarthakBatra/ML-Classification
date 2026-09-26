"""
Accessible & High-Performance Command-Line Interface (CLI).

Provides intuitive argument parsing, colored logging, progress feedback,
and automated validation for the candidate pair generation pipeline.
"""

import os
import sys
import argparse
from typing import List

# Ensure result root and src are in path
RESULT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AMAZON_DIR = os.path.dirname(RESULT_DIR)
sys.path.insert(0, RESULT_DIR)
sys.path.insert(0, os.path.join(RESULT_DIR, "src"))

from config.settings import PipelineConfig, FusionConfig
from pipeline.generator import generate_candidate_pairs_pipeline
from evaluation.analyzer import analyze_candidate_file


def parse_arguments() -> PipelineConfig:
    """Parses command-line arguments and returns a strongly-typed PipelineConfig."""
    default_test_dir = os.path.join(AMAZON_DIR, "dataset", "test")
    default_s1 = os.path.join(default_test_dir, "test_source1.tsv")
    default_s2 = os.path.join(default_test_dir, "test_source2.tsv")
    default_s3 = os.path.join(default_test_dir, "test_source3.tsv")
    default_out = os.path.join(RESULT_DIR, "output", "candidate_pairs.tsv")

    parser = argparse.ArgumentParser(
        prog="generate_candidates",
        description=(
            "=======================================================================\n"
            "Business Entity Resolution — High-Recall Multi-Channel Blocking Engine\n"
            "=======================================================================\n"
            "Generates candidate pairs for Source 1 reference queries against the\n"
            "combined target pool (Source 2 ∪ Source 3) partitioned by country.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Input File Arguments
    data_group = parser.add_argument_group("Dataset Paths")
    data_group.add_argument(
        "--s1",
        type=str,
        default=default_s1,
        help="Path to Source 1 TSV (Query Reference, e.g. test_source1.tsv)"
    )
    data_group.add_argument(
        "--s2",
        type=str,
        default=default_s2,
        help="Path to Source 2 TSV (Target Pool, e.g. test_source2.tsv)"
    )
    data_group.add_argument(
        "--s3",
        type=str,
        default=default_s3,
        help="Path to Source 3 TSV (Target Pool, e.g. test_source3.tsv)"
    )
    data_group.add_argument(
        "--output",
        type=str,
        default=default_out,
        help="Path where candidate_pairs.tsv will be serialized"
    )
    data_group.add_argument(
        "--test-dir",
        type=str,
        default=default_test_dir,
        help="Directory containing test source files for submission validation"
    )

    # Pipeline Tuning Arguments
    tuning_group = parser.add_argument_group("Blocking & Fusion Hyperparameters")
    tuning_group.add_argument(
        "--channels",
        type=str,
        default="b1,b2,b3,b4,b5",
        help="Comma-separated blocking channels to execute. Options: b1, b2, b3, b4, b5 (default: b1,b2,b3,b4,b5)"
    )
    tuning_group.add_argument(
        "--budget",
        type=int,
        default=20,
        help="Maximum candidate pairs to preserve per Source 1 entity (default: 20)"
    )
    tuning_group.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "mps", "cuda", "cpu"],
        help="Compute device for Channel B5 multilingual neural embeddings (default: auto)"
    )

    # Development & Validation Options
    dev_group = parser.add_argument_group("Accessibility, Testing & Validation")
    dev_group.add_argument(
        "--sample-s1",
        type=int,
        default=None,
        help="Limit Source 1 to first N records for rapid prototyping or benchmarking"
    )
    dev_group.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Automatically invoke utils/validate_submission.py upon pipeline completion (default: True)"
    )
    dev_group.add_argument(
        "--no-validate",
        dest="validate",
        action="store_false",
        help="Disable automatic submission validation"
    )
    dev_group.add_argument(
        "--analyze",
        action="store_true",
        default=True,
        help="Run comprehensive statistical distribution analysis on generated output (default: True)"
    )

    args = parser.parse_args()

    channels_list = [c.strip().lower() for c in args.channels.split(",") if c.strip()]

    config = PipelineConfig(
        s1_path=args.s1,
        s2_path=args.s2,
        s3_path=args.s3,
        output_path=args.output,
        test_dir=args.test_dir,
        channels=channels_list,
        sample_s1=args.sample_s1,
        device=args.device,
        validate_output=args.validate,
        fusion=FusionConfig(budget_cap=args.budget)
    )

    return config, args.analyze


def main():
    """Main CLI entrypoint."""
    config, run_analyze = parse_arguments()

    # Execute pipeline
    output_path = generate_candidate_pairs_pipeline(config, verbose=True)

    # Execute statistical analysis
    if run_analyze and os.path.isfile(output_path):
        print("\n" + "=" * 75)
        print("CANDIDATE PAIRS STATISTICAL METRICS REPORT")
        print("=" * 75)
        analyze_candidate_file(output_path, verbose=True)


if __name__ == "__main__":
    main()
