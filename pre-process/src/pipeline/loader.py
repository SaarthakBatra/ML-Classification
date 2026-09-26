"""
Dataset Loader & Schema Standardizer.

Loads TSV datasets using Polars with schema enforcement and ragged line handling.
"""

import os
import time
from typing import Optional
import polars as pl


def load_dataset_schema(
    path: str,
    name: str = "Dataset",
    source_tag: Optional[str] = None,
    verbose: bool = True
) -> pl.DataFrame:
    """
    Loads a TSV dataset using Polars with normalized schema and null handling.

    Parameters:
        path: Filepath to TSV dataset
        name: Human-readable dataset identifier for logging
        source_tag: Optional source identifier ('S1', 'S2', 'S3') to append
        verbose: Whether to log load statistics

    Returns:
        Polars DataFrame with standardized lowercase column names
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    from normalizers.cleaner import build_clean_country_expr, LITERAL_NULL_REGEX

    t0 = time.time()
    df = pl.read_csv(
        path,
        separator="\t",
        truncate_ragged_lines=True,
        infer_schema_length=10000,
        null_values=["", "null", "NULL", "None", "none", "NaN", "nan", "n/a", "N/A", "na", "NA", "-", "undefined"]
    )

    # Standardize column names to lowercase
    col_map = {col: col.strip().lower() for col in df.columns}
    df = df.rename(col_map)

    # Ensure required columns exist
    required_cols = ["entity_id", "business_name", "business_address", "country"]
    for col in required_cols:
        if col not in df.columns:
            df = df.with_columns(pl.lit("").alias(col))

    # Strip whitespace, replace literal fake-nulls, normalize country
    name_col = pl.col("business_name").fill_null("").cast(pl.Utf8)
    addr_col = pl.col("business_address").fill_null("").cast(pl.Utf8)

    name_clean = pl.when(name_col.str.strip_chars().str.contains(LITERAL_NULL_REGEX)).then(pl.lit("")).otherwise(name_col)
    addr_clean = pl.when(addr_col.str.strip_chars().str.contains(LITERAL_NULL_REGEX)).then(pl.lit("")).otherwise(addr_col)

    columns_to_add = [
        pl.col("entity_id").cast(pl.Utf8).str.strip_chars(),
        name_clean.alias("business_name"),
        addr_clean.alias("business_address"),
        build_clean_country_expr("country").alias("country"),
    ]

    if source_tag is not None:
        columns_to_add.append(pl.lit(source_tag).alias("source"))

    df = df.with_columns(columns_to_add)

    if verbose:
        print(f"  Loaded {name} ({len(df):,} rows) in {time.time()-t0:.2f}s from: {path}")

    return df
