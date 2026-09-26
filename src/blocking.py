"""
Backward-compatibility shim for blocking layer.
Re-exports modular components from src/blocking, src/fusion, and src/evaluation.
"""

import os
import sys

# Ensure proper path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, RESULT_DIR)
sys.path.insert(0, BASE_DIR)

from blocking.b1_numeric_geo import NumericGeoIndexer
from blocking.b2_name_tokens import CoreNameTokenIndexer
from blocking.b3_fuzzy import FuzzyCoreIndexer
from blocking.b4_address_tokens import AddressTokenIndexer
from blocking.b5_dense import DenseEmbeddingRetriever
from blocking.registry import build_indexer, INDEXER_REGISTRY
from fusion.rank_fusion import fuse_candidate_channels
from evaluation.metrics import evaluate_blocking

__all__ = [
    "NumericGeoIndexer",
    "CoreNameTokenIndexer",
    "FuzzyCoreIndexer",
    "AddressTokenIndexer",
    "DenseEmbeddingRetriever",
    "build_indexer",
    "INDEXER_REGISTRY",
    "fuse_candidate_channels",
    "evaluate_blocking",
]
