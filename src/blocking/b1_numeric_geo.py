"""
Channel B1: Numeric Premise & Street Geo-Token Indexer.

Matches records based on physical address numeric pairs and distinctive street numbers.
Ideal for entities with identical premise locations but divergent business names (DBAs).
"""

from collections import defaultdict
from typing import Dict, List, Tuple
import polars as pl
from blocking.base import BaseIndexer


class NumericGeoIndexer(BaseIndexer):
    """
    Inverted Index that matches records by physical premise tokens:
    - Pairs of distinctive address numbers: (country, num1, num2)
    - Single distinctive numbers paired with street prefix tokens: (country, num, token[:4])
    """

    def __init__(self):
        self.num_pair_index: Dict[Tuple[str, int, int], List[str]] = defaultdict(list)
        self.single_num_token_index: Dict[Tuple[str, int, str], List[str]] = defaultdict(list)

    def fit(self, target_df: pl.DataFrame) -> "NumericGeoIndexer":
        """Builds in-memory inverted indices for numeric premise pairs."""
        countries = target_df["country"].to_list()
        entity_ids = target_df["entity_id"].to_list()
        addr_nums_list = target_df["addr_nums"].to_list()
        agg_addrs = target_df["agg_addr"].to_list()

        for c, eid, nums, addr in zip(countries, entity_ids, addr_nums_list, agg_addrs):
            if not nums:
                continue

            # 1. Index numeric pairs
            if len(nums) >= 2:
                for i in range(len(nums)):
                    for j in range(i + 1, min(len(nums), i + 4)):
                        key = (c, nums[i], nums[j])
                        self.num_pair_index[key].append(eid)

            # 2. Index single distinctive number paired with address tokens
            elif len(nums) == 1 and nums[0] >= 10:
                street_tokens = [t for t in addr.split() if len(t) >= 4]
                for st in street_tokens[:3]:
                    self.single_num_token_index[(c, nums[0], st[:4])].append(eid)

        return self

    def retrieve(
        self,
        query_df: pl.DataFrame,
        max_cands_per_entity: int = 30
    ) -> Dict[str, List[str]]:
        """Queries the numeric index to find premise collisions."""
        results: Dict[str, List[str]] = {}
        countries = query_df["country"].to_list()
        entity_ids = query_df["entity_id"].to_list()
        addr_nums_list = query_df["addr_nums"].to_list()
        agg_addrs = query_df["agg_addr"].to_list()

        for c, eid, nums, addr in zip(countries, entity_ids, addr_nums_list, agg_addrs):
            cands = set()
            if nums:
                if len(nums) >= 2:
                    for i in range(len(nums)):
                        for j in range(i + 1, min(len(nums), i + 4)):
                            key = (c, nums[i], nums[j])
                            matched = self.num_pair_index.get(key)
                            if matched:
                                cands.update(matched[:max_cands_per_entity])

                elif len(nums) == 1 and nums[0] >= 10:
                    street_tokens = [t for t in addr.split() if len(t) >= 4]
                    for st in street_tokens[:3]:
                        matched = self.single_num_token_index.get((c, nums[0], st[:4]))
                        if matched:
                            cands.update(matched[:max_cands_per_entity])

            results[eid] = list(cands)[:max_cands_per_entity]

        return results
