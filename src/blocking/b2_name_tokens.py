"""
Channel B2: Core Name Token Inverted Index + Dynamic IDF Stopword Filtering.

High-throughput name matching using O(1) set-size arithmetic:
|A ∪ B| = |A| + |B| - |A ∩ B|
Jaccard = |A ∩ B| / (|A| + |B| - |A ∩ B|)
"""

from collections import defaultdict
from typing import Dict, List, Set, Tuple
import polars as pl
from blocking.base import BaseIndexer


class CoreNameTokenIndexer(BaseIndexer):
    """
    Inverted Index on sorted core name tokens with dynamic IDF weighting:
    - Filters out high-frequency stop words (e.g. 'store', 'center', 'sharma')
    - Computes fast token Jaccard using O(1) set-size arithmetic
    """

    def __init__(self, max_token_freq_ratio: float = 0.01, jaccard_threshold: float = 0.30):
        self.max_freq_ratio = max_token_freq_ratio
        self.jaccard_threshold = jaccard_threshold
        self.inverted_index: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self.target_token_counts: Dict[str, int] = {}
        self.stop_tokens: Set[Tuple[str, str]] = set()

    def fit(self, target_df: pl.DataFrame) -> "CoreNameTokenIndexer":
        """Builds in-memory posting lists for core name tokens."""
        countries = target_df["country"].to_list()
        entity_ids = target_df["entity_id"].to_list()
        agg_names = target_df["agg_name"].to_list()

        token_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        n_records = len(target_df)
        max_allowed_freq = min(1500, max(50, int(n_records * self.max_freq_ratio)))

        # First pass: count frequencies to identify stop words
        for c, eid, name in zip(countries, entity_ids, agg_names):
            if not name:
                continue
            tokens = set(name.split())
            self.target_token_counts[eid] = len(tokens)
            for tok in tokens:
                token_counts[(c, tok)] += 1

        # Identify stop-words per country
        for (c, tok), count in token_counts.items():
            if count > max_allowed_freq:
                self.stop_tokens.add((c, tok))

        # Second pass: build posting lists for non-stopwords
        for c, eid, name in zip(countries, entity_ids, agg_names):
            if not name:
                continue
            tokens = set(name.split())
            for tok in tokens:
                if (c, tok) not in self.stop_tokens:
                    self.inverted_index[(c, tok)].append(eid)

        return self

    def retrieve(
        self,
        query_df: pl.DataFrame,
        max_cands_per_entity: int = 30
    ) -> Dict[str, List[str]]:
        """Retrieves and ranks candidate records by token Jaccard similarity."""
        results: Dict[str, List[str]] = {}
        countries = query_df["country"].to_list()
        entity_ids = query_df["entity_id"].to_list()
        agg_names = query_df["agg_name"].to_list()

        for c, eid, name in zip(countries, entity_ids, agg_names):
            if not name:
                results[eid] = []
                continue

            query_tokens = set(name.split())
            candidate_counts: Dict[str, int] = defaultdict(int)

            # Accumulate candidate occurrences across query tokens
            for tok in query_tokens:
                if (c, tok) in self.stop_tokens:
                    continue
                postings = self.inverted_index.get((c, tok))
                if postings:
                    for tid in postings:
                        candidate_counts[tid] += 1

            # Filter by Jaccard using exact set arithmetic: |A ∪ B| = |A| + |B| - |A ∩ B|
            scored_candidates = []
            len_q = len(query_tokens)
            for tid, shared_count in candidate_counts.items():
                t_count = self.target_token_counts.get(tid, 0)
                union_len = len_q + t_count - shared_count
                jaccard = shared_count / union_len if union_len > 0 else 0.0

                if jaccard >= self.jaccard_threshold:
                    scored_candidates.append((tid, jaccard))

            # Sort descending by Jaccard
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            results[eid] = [tid for tid, _ in scored_candidates[:max_cands_per_entity]]

        return results
