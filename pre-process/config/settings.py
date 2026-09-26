"""
Pipeline Configuration & Hyperparameter Management.

Provides strongly-typed dataclasses for all stages of the Business Entity Resolution
Candidate Generation Pipeline (ML Challenge 2026).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class NormalizerConfig:
    """Hyperparameters for dual-track text normalization."""
    # Maximum digits for premise address numbers to avoid phone numbers / tracking IDs
    max_numeric_length: int = 6
    # Suffix removal enabled
    strip_legal_suffixes: bool = True
    # Strip diacritics / accents (essential for French dataset compatibility)
    strip_accents: bool = True


@dataclass(frozen=True)
class BlockingConfig:
    """Hyperparameters and thresholds for multi-channel blocking indexers."""
    # Channel B1: Numeric premise pairs
    b1_max_cands: int = 30
    b1_weight: float = 3.0

    # Channel B2: Core name token inverted index + O(1) set Jaccard
    b2_max_token_freq_ratio: float = 0.01  # Max frequency ratio before treating token as stopword
    b2_jaccard_threshold: float = 0.30     # Minimum Jaccard similarity
    b2_max_cands: int = 30
    b2_weight: float = 2.5

    # Channel B3: Rapid prefix-bucketing & typo-tolerant RapidFuzz ratio
    b3_min_ratio: float = 70.0             # Minimum character fuzzy ratio
    b3_prefix_length: int = 3              # Prefix bucket length
    b3_max_pool_per_prefix: int = 300      # Max pool size to evaluate fuzzy ratio on
    b3_max_cands: int = 20
    b3_weight: float = 2.0

    # Channel B4: Address token inverted index + O(1) set Jaccard
    b4_max_token_freq_ratio: float = 0.015 # Max frequency ratio for address stop-tokens
    b4_jaccard_threshold: float = 0.28     # Minimum address Jaccard similarity
    b4_max_cands: int = 30
    b4_weight: float = 2.5

    # Channel B5: Multilingual Dense Bi-Encoder
    b5_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
    b5_batch_size: int = 256
    b5_min_similarity: float = 0.50
    b5_top_k: int = 15
    b5_weight: float = 2.5


@dataclass(frozen=True)
class FusionConfig:
    """Hyperparameters for consensus fusion and pruning."""
    # Hard budget cap per Source 1 entity (competition requirement: budget <= 20)
    budget_cap: int = 20
    # Positional rank discount factor
    rank_discount_slope: float = 0.5


import os

_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
_RESULT_DIR = os.path.dirname(_CONFIG_DIR)
_AMAZON_DIR = os.path.dirname(_RESULT_DIR)
_DEFAULT_TEST_DIR = os.path.join(_AMAZON_DIR, "dataset", "test")


@dataclass
class PipelineConfig:
    """Global end-to-end pipeline execution configuration."""
    s1_path: str = os.path.join(_DEFAULT_TEST_DIR, "test_source1.tsv")
    s2_path: str = os.path.join(_DEFAULT_TEST_DIR, "test_source2.tsv")
    s3_path: str = os.path.join(_DEFAULT_TEST_DIR, "test_source3.tsv")
    output_path: str = os.path.join(_RESULT_DIR, "output", "candidate_pairs.tsv")
    test_dir: str = _DEFAULT_TEST_DIR
    channels: List[str] = field(default_factory=lambda: ["b1", "b2", "b3", "b4", "b5"])
    sample_s1: Optional[int] = None
    device: str = "auto"
    validate_output: bool = True

    # Nested configurations
    normalizer: NormalizerConfig = field(default_factory=NormalizerConfig)
    blocking: BlockingConfig = field(default_factory=BlockingConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
