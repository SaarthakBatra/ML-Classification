"""
Abstract Base Class for Blocking Indexers.

Enforces a uniform API across all blocking channels:
- Channel B1: Numeric Premise Geo Indexer
- Channel B2: Core Name Token Indexer
- Channel B3: Fuzzy Core Name Indexer
- Channel B4: Address Token Indexer
- Channel B5: Dense Embedding Bi-Encoder Retriever
"""

from abc import ABC, abstractmethod
from typing import Dict, List
import polars as pl


class BaseIndexer(ABC):
    """
    Abstract Base Class defining the contract for blocking channels.
    """

    @abstractmethod
    def fit(self, target_df: pl.DataFrame) -> "BaseIndexer":
        """
        Builds an in-memory index from the target pool (Source 2 ∪ Source 3).

        Parameters:
            target_df: Polars DataFrame containing normalized target records.

        Returns:
            self
        """
        pass

    @abstractmethod
    def retrieve(
        self,
        query_df: pl.DataFrame,
        max_cands_per_entity: int = 30
    ) -> Dict[str, List[str]]:
        """
        Queries the index for candidate matches for each query entity in query_df (Source 1).

        Parameters:
            query_df: Polars DataFrame containing normalized query records.
            max_cands_per_entity: Maximum candidate pairs retrieved per query entity.

        Returns:
            Dictionary mapping: query_entity_id -> list of candidate target_entity_ids.
        """
        pass
