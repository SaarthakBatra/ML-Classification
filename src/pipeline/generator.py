"""
End-to-End Candidate Pair Generator Orchestrator.

Orchestrates:
1. Loading and validation of S1, S2, and S3 datasets
2. Country partitioning and bounded-memory execution
3. Competition-compliant TSV file export
4. Validation via official utils/validate_submission.py
"""

import os
import sys
import gc
import time
import subprocess
from typing import Dict, List, Optional
import polars as pl

from config.settings import PipelineConfig, FusionConfig
from pipeline.loader import load_dataset_schema
from pipeline.partition import process_country_partition


def generate_candidate_pairs_pipeline(
    config: PipelineConfig,
    verbose: bool = True
) -> str:
    """
    Executes the candidate pair generation pipeline per PipelineConfig.

    Returns:
        Path to generated candidate_pairs.tsv file.
    """
    total_start = time.time()
    channels = config.channels

    if verbose:
        print("=" * 75)
        print("BUSINESS ENTITY RESOLUTION — CANDIDATE PAIR GENERATION PIPELINE")
        print(f"  Active Channels: {', '.join(c.upper() for c in channels)}")
        print(f"  Budget Cap:      {config.fusion.budget_cap} candidates per S1 entity")
        print(f"  Output Path:     {config.output_path}")
        print("=" * 75)

    # 1. Load Data Sources
    s1_df = load_dataset_schema(config.s1_path, "Source 1 (Reference)", verbose=verbose)
    if config.sample_s1 and config.sample_s1 < len(s1_df):
        if verbose:
            print(f"  [Sample Mode] Limiting Source 1 to first {config.sample_s1:,} entities")
        s1_df = s1_df.slice(0, config.sample_s1)

    s2_df = load_dataset_schema(config.s2_path, "Source 2 (Target Pool)", verbose=verbose)
    s3_df = load_dataset_schema(config.s3_path, "Source 3 (Target Pool)", verbose=verbose)

    target_df = pl.concat([s2_df, s3_df], how="vertical_relaxed")
    del s2_df, s3_df
    gc.collect()

    if verbose:
        print(f"  Combined Target Pool (S2 ∪ S3): {len(target_df):,} records")

    # Preserve exact original S1 ordering for final output
    original_s1_ids = s1_df["entity_id"].to_list()
    all_fused_candidates: Dict[str, List[str]] = {eid: [] for eid in original_s1_ids}

    # Discover unique country partitions
    unique_countries = [c for c in s1_df["country"].unique().to_list() if c]
    if verbose:
        print(f"\nDiscovered {len(unique_countries)} country partitions in Source 1: {unique_countries}")

    # Process Partition by Partition
    for country_idx, country in enumerate(unique_countries, 1):
        if verbose:
            print("\n" + "-" * 75)
            print(f"PROCESSING PARTITION [{country_idx}/{len(unique_countries)}]: Country = '{country}'")
            print("-" * 75)

        s1_c = s1_df.filter(pl.col("country") == country)
        target_c = target_df.filter(pl.col("country") == country)

        partition_candidates = process_country_partition(
            s1_partition=s1_c,
            target_partition=target_c,
            country_name=country,
            channels=channels,
            blocking_config=config.blocking,
            fusion_config=config.fusion,
            normalizer_config=config.normalizer,
            device=config.device,
            verbose=verbose
        )

        # Merge partition results
        for eid, cands in partition_candidates.items():
            all_fused_candidates[eid] = cands

        del s1_c, target_c, partition_candidates
        gc.collect()

    # 4. Serialize Competition-Compliant TSV
    if verbose:
        print("\n" + "=" * 75)
        print("STAGE: Writing Submission Candidate Pairs TSV")
        print("=" * 75)

    os.makedirs(os.path.dirname(os.path.abspath(config.output_path)), exist_ok=True)
    t_export = time.time()
    total_candidates = 0
    zero_candidates = 0

    with open(config.output_path, "w", encoding="utf-8") as f:
        # Strict Header per competition guidelines: source1_entity_id \t candidate_entity_ids
        f.write("source1_entity_id\tcandidate_entity_ids\n")

        for s1_id in original_s1_ids:
            cands = all_fused_candidates.get(s1_id, [])
            valid_cands = [
                c for c in cands
                if c.startswith(("S2-", "S3-")) and not c.startswith("S1-")
            ]
            total_candidates += len(valid_cands)
            if len(valid_cands) == 0:
                zero_candidates += 1
                f.write(f"{s1_id}\t\n")
            else:
                f.write(f"{s1_id}\t{','.join(valid_cands)}\n")

    n_s1 = len(original_s1_ids)
    n_tgt = len(target_df)
    avg_cands = total_candidates / n_s1 if n_s1 > 0 else 0
    cartesian = n_s1 * n_tgt
    rr = 1.0 - (total_candidates / cartesian) if cartesian > 0 else 1.0

    if verbose:
        print(f"Exported to: {config.output_path} in {time.time()-t_export:.2f}s")
        print(f"  Total S1 Entities:     {n_s1:,}")
        print(f"  Total Candidate Pairs: {total_candidates:,}")
        print(f"  Avg Candidates / S1:   {avg_cands:.2f}")
        print(f"  Zero-Candidate S1s:    {zero_candidates} ({zero_candidates/n_s1*100:.2f}%)")
        print(f"  Reduction Ratio:       {rr*100:.5f}%")
        print(f"Total Pipeline Runtime:  {time.time()-total_start:.2f}s")
        print("=" * 75)

    # 5. Optional Official Validation
    if config.validate_output:
        # Check AMAZON/utils first, then fallback to result/utils
        amazon_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        validator_script = os.path.join(amazon_dir, "utils", "validate_submission.py")
        if not os.path.isfile(validator_script):
            validator_script = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "utils",
                "validate_submission.py"
            )
        if os.path.isfile(validator_script):
            if verbose:
                print("\n" + "=" * 75)
                print("RUNNING OFFICIAL SUBMISSION VALIDATOR...")
                print("=" * 75)
            cmd = [
                sys.executable,
                validator_script,
                "--candidate", config.output_path,
                "--test-dir", config.test_dir
            ]
            res = subprocess.run(cmd)
            if res.returncode == 0:
                if verbose:
                    print("\n>>> VALIDATION PASSED! File is 100% compliant with submission rules. <<<")
            else:
                if verbose:
                    print(f"\n>>> VALIDATOR RETURNED CODE {res.returncode}. Please review messages. <<<")

    return config.output_path
