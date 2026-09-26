"""
Channel B4: Address Token Inverted Index + O(1) Set Jaccard Filtering.

Matches entities sharing the same physical street, locality, or building tokens
even when business names significantly diverge (DBAs, acquired entities).
"""

from collections import defaultdict
from typing import Dict, List, Set, Tuple
import polars as pl
from blocking.base import BaseIndexer


class AddressTokenIndexer(BaseIndexer):
    """
    Inverted Index on distinctive address tokens (street, locality, city).
    Catches entities sharing the same physical location even when names diverge (DBAs).
    Optimized for multi-million records using O(1) mathematical Jaccard arithmetic.
    """

    def __init__(self, max_freq_ratio: float = 0.015, min_jaccard: float = 0.28):
        self.max_freq_ratio = max_freq_ratio
        self.min_jaccard = min_jaccard
        self.inverted_index: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self.target_token_counts: Dict[str, int] = {}
        self.stop_tokens: Set[Tuple[str, str]] = set()

    def fit(self, target_df: pl.DataFrame) -> "AddressTokenIndexer":
        """Builds in-memory posting lists for distinctive address tokens."""
        counts: Dict[Tuple[str, str], int] = defaultdict(int)
        n_records = len(target_df)
        max_freq = min(2000, max(50, int(n_records * self.max_freq_ratio)))

        countries = target_df["country"].to_list()
        entity_ids = target_df["entity_id"].to_list()
        agg_addrs = target_df["agg_addr"].to_list()

        # First pass: count frequencies
        for c, eid, addr in zip(countries, entity_ids, agg_addrs):
            if not addr:
                continue
            toks = set(t for t in addr.split() if len(t) > 2)
            self.target_token_counts[eid] = len(toks)
            for t in toks:
                counts[(c, t)] += 1

        # Identify frequent address stop-tokens
        for (c, t), cnt in counts.items():
            if cnt > max_freq:
                self.stop_tokens.add((c, t))

        # Second pass: build posting lists
        for c, eid, addr in zip(countries, entity_ids, agg_addrs):
            if not addr:
                continue
            toks = set(t for t in addr.split() if len(t) > 2)
            for t in toks:
                if (c, t) not in self.stop_tokens:
                    self.inverted_index[(c, t)].append(eid)

        return self

    def retrieve(
        self,
        query_df: pl.DataFrame,
        max_cands_per_entity: int = 30
    ) -> Dict[str, List[str]]:
        """Queries the address token index and computes Jaccard similarity."""
        results: Dict[str, List[str]] = {}
        countries = query_df["country"].to_list()
        entity_ids = query_df["entity_id"].to_list()
        agg_addrs = query_df["agg_addr"].to_list()

        for c, eid, addr in zip(countries, entity_ids, agg_addrs):
            if not addr:
                results[eid] = []
                continue

            query_toks = set(t for t in addr.split() if len(t) > 2)
            c_counts: Dict[str, int] = defaultdict(int)

            for t in query_toks:
                if (c, t) in self.stop_tokens:
                    continue
                postings = self.inverted_index.get((c, t))
                if postings:
                    for tid in postings:
                        c_counts[tid] += 1

            scored = []
            len_q = len(query_toks)
            for tid, shared in c_counts.items():
                t_count = self.target_token_counts.get(tid, 0)
                union_len = len_q + t_count - shared
                jaccard = shared / union_len if union_len > 0 else 0.0
                if jaccard >= self.min_jaccard:
                    scored.append((tid, jaccard))

            scored.sort(key=lambda x: x[1], reverse=True)
            results[eid] = [tid for tid, _ in scored[:max_cands_per_entity]]

        return results
