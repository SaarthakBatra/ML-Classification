"""
Backward-compatibility shim for DenseEmbeddingRetriever.
Re-exports DenseEmbeddingRetriever from src/blocking/b5_dense.py.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, RESULT_DIR)
sys.path.insert(0, BASE_DIR)

from blocking.b5_dense import DenseEmbeddingRetriever

__all__ = ["DenseEmbeddingRetriever"]
