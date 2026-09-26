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
    verbose: bool = True
) -> pl.DataFrame:
    """
    Loads a TSV dataset using Polars with normalized schema and null handling.

    Parameters:
        path: Filepath to TSV dataset
        name: Human-readable dataset identifier for logging
        verbose: Whether to log load statistics

    Returns:
        Polars DataFrame with standardized lowercase column names
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    t0 = time.time()
    df = pl.read_csv(
        path,
        separator="\t",
        truncate_ragged_lines=True,
        infer_schema_length=10000,
        null_values=["", "null", "NULL", "None", "NaN", "nan"]
    )

    # Standardize column names to lowercase
    col_map = {col: col.strip().lower() for col in df.columns}
    df = df.rename(col_map)

    # Ensure required columns exist
    required_cols = ["entity_id", "business_name", "business_address", "country"]
    for col in required_cols:
        if col not in df.columns:
            df = df.with_columns(pl.lit("").alias(col))

    # Strip whitespace from entity_id and standardize country uppercase
    df = df.with_columns([
        pl.col("entity_id").cast(pl.Utf8).str.strip_chars(),
        pl.col("country").fill_null("").cast(pl.Utf8).str.strip_chars().str.to_uppercase(),
    ])

    if verbose:
        print(f"  Loaded {name} ({len(df):,} rows) in {time.time()-t0:.2f}s from: {path}")

    return df
