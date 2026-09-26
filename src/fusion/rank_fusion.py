"""
Consensus Ranking & Candidate Fusion Engine.

Combines heterogeneous candidate lists generated from multiple complementary blocking channels.
Applies positional rank discounts, accumulates multi-channel consensus scores,
and enforces a strict hard budget cap (K <= 20) per Source 1 entity.
"""

from collections import defaultdict
from typing import Dict, List, Tuple


def fuse_candidate_channels(
    channel_results: List[Tuple[str, Dict[str, List[str]], float]],
    s1_ids: List[str],
    hard_budget_cap: int = 20,
    rank_discount_slope: float = 0.5
) -> Dict[str, List[str]]:
    """
    Fuses candidate lists from multiple channels using weighted positional rank scoring.

    Parameters:
        channel_results: List of tuples (channel_name, candidate_dict, channel_weight)
        s1_ids: Ordered list of query entity IDs
        hard_budget_cap: Maximum candidates retained per query entity
        rank_discount_slope: Controls the slope of positional rank discounting

    Returns:
        Dictionary mapping: s1_id -> sorted list of top-ranked candidate target entity IDs
    """
    fused: Dict[str, List[str]] = {}

    for s1_id in s1_ids:
        score_map: Dict[str, float] = defaultdict(float)

        for channel_name, c_dict, weight in channel_results:
            cands = c_dict.get(s1_id, [])
            n_cands = len(cands)
            if n_cands == 0:
                continue

            for rank, cid in enumerate(cands):
                # Positional rank discount: top rank (rank=0) receives full weight
                rank_weight = (n_cands - rank) / n_cands
                # Score formula: weight * ( (1 - slope) + slope * rank_weight )
                score_map[cid] += weight * ((1.0 - rank_discount_slope) + rank_discount_slope * rank_weight)

        # Sort descending by cumulative score
        ranked = sorted(score_map.keys(), key=lambda c: score_map[c], reverse=True)
        fused[s1_id] = ranked[:hard_budget_cap]

    return fused
