#!/usr/bin/env python3
"""
Candidate Pair Generation CLI & Pipeline Entrypoint.
Backward-compatible wrapper routing to modular pipeline implementation in src/pipeline and src/cli.
"""

import os
import sys

# Ensure proper paths
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.dirname(SRC_DIR)
sys.path.insert(0, RESULT_DIR)
sys.path.insert(0, SRC_DIR)

from cli.main import main
from pipeline.generator import generate_candidate_pairs_pipeline
from pipeline.loader import load_dataset_schema

# Compatibility function signature for programmatic callers
def generate_candidate_pairs(
    s1_path: str,
    s2_path: str,
    s3_path: str,
    output_path: str,
    budget_cap: int = 20,
    channels: list = [],
    sample_s1: int = 0,
    device: str = "auto"
) -> str:
    """Compatibility API for legacy function calls."""
    from config.settings import PipelineConfig, FusionConfig
    if channels is None:
        channels = ["b1", "b2", "b3", "b4", "b5"]

    cfg = PipelineConfig(
        s1_path=s1_path,
        s2_path=s2_path,
        s3_path=s3_path,
        output_path=output_path,
        channels=channels,
        sample_s1=sample_s1,
        device=device,
        fusion=FusionConfig(budget_cap=budget_cap)
    )
    return generate_candidate_pairs_pipeline(cfg)


if __name__ == "__main__":
    main()
