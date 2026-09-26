"""
Blocking Channel Registry & Factory.

Enables clean dynamic instantiation of blocking indexers by channel code ('b1', 'b2', 'b3', 'b4', 'b5').
"""

from typing import Dict, Type
from config.settings import BlockingConfig
from blocking.base import BaseIndexer
from blocking.b1_numeric_geo import NumericGeoIndexer
from blocking.b2_name_tokens import CoreNameTokenIndexer
from blocking.b3_fuzzy import FuzzyCoreIndexer
from blocking.b4_address_tokens import AddressTokenIndexer
from blocking.b5_dense import DenseEmbeddingRetriever

INDEXER_REGISTRY: Dict[str, Type[BaseIndexer]] = {
    "b1": NumericGeoIndexer,
    "b2": CoreNameTokenIndexer,
    "b3": FuzzyCoreIndexer,
    "b4": AddressTokenIndexer,
    "b5": DenseEmbeddingRetriever,
}


def build_indexer(channel: str, config: BlockingConfig, device: str = "auto") -> BaseIndexer:
    """
    Factory function to construct and configure an indexer instance.

    Parameters:
        channel: Channel key ('b1', 'b2', 'b3', 'b4', 'b5')
        config: Strongly-typed BlockingConfig instance
        device: Device for neural channels ('auto', 'cpu', 'cuda', 'mps')

    Returns:
        Configured BaseIndexer instance
    """
    ch = channel.lower().strip()
    if ch == "b1":
        return NumericGeoIndexer()
    elif ch == "b2":
        return CoreNameTokenIndexer(
            max_token_freq_ratio=config.b2_max_token_freq_ratio,
            jaccard_threshold=config.b2_jaccard_threshold
        )
    elif ch == "b3":
        return FuzzyCoreIndexer(
            min_ratio=config.b3_min_ratio,
            prefix_len=config.b3_prefix_length,
            max_pool_per_prefix=config.b3_max_pool_per_prefix
        )
    elif ch == "b4":
        return AddressTokenIndexer(
            max_freq_ratio=config.b4_max_token_freq_ratio,
            min_jaccard=config.b4_jaccard_threshold
        )
    elif ch == "b5":
        return DenseEmbeddingRetriever(
            model_name=config.b5_model_name,
            batch_size=config.b5_batch_size,
            device=device
        )
    else:
        raise ValueError(f"Unknown blocking channel: '{channel}'. Supported channels: {list(INDEXER_REGISTRY.keys())}")
