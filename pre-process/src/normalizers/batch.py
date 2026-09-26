"""
Vectorized Polars Batch Processing for Dual-Track Normalization.

Processes hundreds of thousands of records per second with minimal memory footprint.
"""

from typing import Optional
import polars as pl
from config.settings import NormalizerConfig
from normalizers.aggressive import normalize_aggressive_name, extract_address_features
from normalizers.moderate import normalize_moderate
from normalizers.cleaner import (
    build_clean_name_expr,
    build_clean_address_expr,
    build_clean_country_expr,
)


def process_dataframe(
    df: pl.DataFrame,
    config: Optional[NormalizerConfig] = None
) -> pl.DataFrame:
    """
    Apply dual-track normalization across a Polars DataFrame.

    Required input schema columns:
        - entity_id (str)
        - business_name (str)
        - business_address (str)
        - country (str)

    Appends feature columns:
        - clean_name: Phase 0 fully expanded and standardized business name
        - clean_address: Phase 0 fully expanded and standardized business address
        - agg_name: Suffix-stripped, alphabetically sorted core tokens (Track A)
        - agg_addr: Clean address tokens (Track A)
        - addr_nums: Sorted list of premise address integers (Track A)
        - embed_name: Clean business name (Track B)
        - embed_comb: Combined name + address (Track B)
        - embed_combined: Alias for embed_comb for downstream bi-encoder models
    """
    if config is None:
        config = NormalizerConfig()

    # Ensure required columns are present and clean nulls + standard country
    df = df.with_columns([
        pl.col("business_name").fill_null("").cast(pl.Utf8),
        pl.col("business_address").fill_null("").cast(pl.Utf8),
        build_clean_country_expr("country").alias("country"),
        build_clean_name_expr("business_name").alias("clean_name"),
        build_clean_address_expr("business_address").alias("clean_address"),
    ])

    names = df["business_name"].to_list()
    addresses = df["business_address"].to_list()

    agg_names = []
    agg_addrs = []
    addr_nums = []
    embed_names = []
    embed_combs = []

    max_digits = config.max_numeric_length

    for name, addr in zip(names, addresses):
        # Track A: Aggressive
        an = normalize_aggressive_name(name)
        aa, nums = extract_address_features(addr, max_digits=max_digits)

        # Track B: Moderate
        en, ec = normalize_moderate(name, addr)

        agg_names.append(an)
        agg_addrs.append(aa)
        addr_nums.append(nums)
        embed_names.append(en)
        embed_combs.append(ec)

    return df.with_columns([
        pl.Series("agg_name", agg_names, dtype=pl.Utf8),
        pl.Series("agg_addr", agg_addrs, dtype=pl.Utf8),
        pl.Series("addr_nums", addr_nums, dtype=pl.List(pl.Int64)),
        pl.Series("embed_name", embed_names, dtype=pl.Utf8),
        pl.Series("embed_comb", embed_combs, dtype=pl.Utf8),
        pl.Series("embed_combined", embed_combs, dtype=pl.Utf8),
    ])
