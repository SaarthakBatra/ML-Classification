"""
Blocking Package for Multi-Channel Candidate Pair Generation.
"""

from blocking.base import BaseIndexer
from blocking.b1_numeric_geo import NumericGeoIndexer
from blocking.b2_name_tokens import CoreNameTokenIndexer
from blocking.b3_fuzzy import FuzzyCoreIndexer
from blocking.b4_address_tokens import AddressTokenIndexer
from blocking.b5_dense import DenseEmbeddingRetriever
from blocking.registry import build_indexer, INDEXER_REGISTRY

__all__ = [
    "BaseIndexer",
    "NumericGeoIndexer",
    "CoreNameTokenIndexer",
    "FuzzyCoreIndexer",
    "AddressTokenIndexer",
    "DenseEmbeddingRetriever",
    "build_indexer",
    "INDEXER_REGISTRY",
]
