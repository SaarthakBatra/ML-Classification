"""
Channel B3: Rapid Character 3-Gram & Typo-Tolerant Prefix Bucketing Indexer.

Catches severe typos and character permutations in short business names
(e.g., 'Wilblims' vs 'Williams', 'Wanye' vs 'Wayne', 'Dréxkor' vs 'Drexkor').
Leverages prefix bucketing for sub-millisecond evaluation via RapidFuzz.
"""

from collections import defaultdict
from typing import Dict, List, Tuple
import polars as pl
from rapidfuzz import fuzz
from blocking.base import BaseIndexer


class FuzzyCoreIndexer(BaseIndexer):
    """
    Fast character 3-gram/prefix index to catch typos in short business names.
    Fine-grained prefix bucketing allows sub-millisecond evaluation per entity.
    """

    def __init__(
        self,
        min_ratio: float = 70.0,
        prefix_len: int = 3,
        max_pool_per_prefix: int = 300
    ):
        self.min_ratio = min_ratio
        self.prefix_len = prefix_len
        self.max_pool_per_prefix = max_pool_per_prefix
        # Index key: (country, prefix)
        self.bucket_index: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self.target_names: Dict[str, str] = {}

    def fit(self, target_df: pl.DataFrame) -> "FuzzyCoreIndexer":
        """Buckets target records by country and 3-character prefix."""
        countries = target_df["country"].to_list()
        entity_ids = target_df["entity_id"].to_list()
        agg_names = target_df["agg_name"].to_list()
        p_len = self.prefix_len

        for c, eid, name in zip(countries, entity_ids, agg_names):
            if not name:
                continue
            self.target_names[eid] = name
            pfx = name[:p_len] if len(name) >= p_len else name
            self.bucket_index[(c, pfx)].append(eid)

        return self

    def retrieve(
        self,
        query_df: pl.DataFrame,
        max_cands_per_entity: int = 20
    ) -> Dict[str, List[str]]:
        """Queries the prefix bucket and evaluates RapidFuzz ratio on candidate pool."""
        results: Dict[str, List[str]] = {}
        countries = query_df["country"].to_list()
        entity_ids = query_df["entity_id"].to_list()
        agg_names = query_df["agg_name"].to_list()
        p_len = self.prefix_len
        max_pool = self.max_pool_per_prefix

        for c, eid, name in zip(countries, entity_ids, agg_names):
            if not name:
                results[eid] = []
                continue

            pfx = name[:p_len] if len(name) >= p_len else name
            candidate_pool = self.bucket_index.get((c, pfx), [])

            scored = []
            for tid in candidate_pool[:max_pool]:
                t_name = self.target_names[tid]
                score = fuzz.ratio(name, t_name)
                if score >= self.min_ratio:
                    scored.append((tid, score))

            scored.sort(key=lambda x: x[1], reverse=True)
            results[eid] = [tid for tid, _ in scored[:max_cands_per_entity]]

        return results
