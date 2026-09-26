"""
Country-Partitioned Processing Engine.

Leverages the strict intra-country match constraint to shard query and target sets.
Drastically bounds peak RAM usage and eliminates cross-border false positive comparisons.
"""

import gc
import time
from typing import Dict, List, Tuple
import polars as pl
from config.settings import BlockingConfig, FusionConfig, NormalizerConfig
from normalizers.batch import process_dataframe
from blocking.registry import build_indexer
from fusion.rank_fusion import fuse_candidate_channels


def process_country_partition(
    s1_partition: pl.DataFrame,
    target_partition: pl.DataFrame,
    country_name: str,
    channels: List[str],
    blocking_config: BlockingConfig,
    fusion_config: FusionConfig,
    normalizer_config: NormalizerConfig,
    device: str = "auto",
    verbose: bool = True
) -> Dict[str, List[str]]:
    """
    Executes normalization, multi-channel blocking, and consensus fusion for a single country.

    Parameters:
        s1_partition: Source 1 records filtered to current country
        target_partition: Combined target records (S2 ∪ S3) filtered to current country
        country_name: Country code (e.g. 'US', 'INDIA', 'FRANCE')
        channels: List of channel codes to execute (e.g. ['b1', 'b2', 'b3', 'b4'])
        blocking_config: Hyperparameters for indexers
        fusion_config: Hyperparameters for fusion
        normalizer_config: Hyperparameters for text cleaners
        device: Hardware device for neural channels
        verbose: Whether to log timing details

    Returns:
        Dict mapping: s1_entity_id -> list of candidate target entity IDs (budget-capped)
    """
    n_s1 = len(s1_partition)
    n_target = len(target_partition)

    if n_s1 == 0 or n_target == 0:
        if verbose:
            print(f"  [Partition {country_name}] Empty query or target pool. Skipping.")
        return {eid: [] for eid in s1_partition["entity_id"].to_list()}

    # 1. Dual-Track Normalization
    t_norm = time.time()
    s1_norm = process_dataframe(s1_partition, config=normalizer_config)
    target_norm = process_dataframe(target_partition, config=normalizer_config)
    if verbose:
        print(f"  [Partition {country_name}] Normalized {n_s1:,} S1 and {n_target:,} targets in {time.time()-t_norm:.2f}s")

    # 2. Multi-Channel Blocking
    channel_results: List[Tuple[str, Dict[str, List[str]], float]] = []

    for ch in channels:
        ch_clean = ch.lower().strip()
        t_ch = time.time()
        indexer = build_indexer(ch_clean, blocking_config, device=device)

        # Retrieve weight and budget for channel
        if ch_clean == "b1":
            weight = blocking_config.b1_weight
            max_cands = blocking_config.b1_max_cands
        elif ch_clean == "b2":
            weight = blocking_config.b2_weight
            max_cands = blocking_config.b2_max_cands
        elif ch_clean == "b3":
            weight = blocking_config.b3_weight
            max_cands = blocking_config.b3_max_cands
        elif ch_clean == "b4":
            weight = blocking_config.b4_weight
            max_cands = blocking_config.b4_max_cands
        elif ch_clean == "b5":
            weight = blocking_config.b5_weight
            max_cands = blocking_config.b5_top_k
        else:
            weight = 1.0
            max_cands = 20

        # Fit & Retrieve
        indexer.fit(target_norm)
        cands = indexer.retrieve(s1_norm, max_cands_per_entity=max_cands)
        channel_results.append((ch_clean, cands, weight))

        if verbose:
            print(f"    Channel [{ch_clean.upper()}]: fit + retrieve completed in {time.time()-t_ch:.2f}s")

    # 3. Consensus Fusion & Pruning
    t_fuse = time.time()
    s1_ids = s1_partition["entity_id"].to_list()
    fused_candidates = fuse_candidate_channels(
        channel_results=channel_results,
        s1_ids=s1_ids,
        hard_budget_cap=fusion_config.budget_cap,
        rank_discount_slope=fusion_config.rank_discount_slope
    )
    if verbose:
        print(f"  [Partition {country_name}] Fusion completed in {time.time()-t_fuse:.2f}s")

    # Explicit garbage collection
    del s1_norm, target_norm, channel_results
    gc.collect()

    return fused_candidates
