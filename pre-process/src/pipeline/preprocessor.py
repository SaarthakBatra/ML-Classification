"""
End-to-End Preprocessing & Partitioning Engine (Phase 0).

Strictly implements the specification defined in Preprocessing_1.0.md:
1. High-speed loading of TSV datasets using Polars
2. Literal fake-null parsing (["n/a", "na", "null", "none", "-", "nan", "undefined"] -> "")
3. Unicode normalization (NFKD), diacritics stripping (café -> cafe)
4. Acronym protection (dots removed without spaces: I.B.M. -> IBM)
5. Symbol mapping (@ -> ' at ', & -> ' and ')
6. Special character removal preserving alphanumeric and spaces
7. Ordinal number mapping (first -> 1st, second -> 2nd, etc.)
8. Business suffix expansion (corp -> corporation, inc -> incorporated, ltd -> limited, etc.)
9. Address abbreviation expansion (st -> street, rd -> road, ave -> avenue, etc.)
10. Empty string safety fallback (clean_name -> UNKNOWN_NAME, clean_address -> UNKNOWN_ADDRESS)
11. Country normalization and strict partitioning (Query_<Country>, Target_<Country>)
12. Vertical target pool union (S2 + S3) with preserved entity_id and explicit 'source' tracking column
13. High-throughput export to Parquet and/or TSV formats
"""

import os
import sys
import gc
import json
import time
from typing import Dict, List, Optional, Tuple, Set
import polars as pl

from pipeline.loader import load_dataset_schema
from normalizers.cleaner import (
    build_clean_name_expr,
    build_clean_address_expr,
    build_clean_country_expr,
    UNKNOWN_NAME,
    UNKNOWN_ADDRESS,
)


def clean_dataframe(
    df: pl.DataFrame,
    source_tag: Optional[str] = None,
    verbose: bool = False
) -> pl.DataFrame:
    """
    Applies vectorized Rust-native Polars expressions to clean and standardize a DataFrame
    per Preprocessing_1.0.md.

    Required columns: entity_id, business_name, business_address, country
    Output columns: entity_id, business_name, business_address, clean_name, clean_address, country, source
    """
    t0 = time.time()

    exprs = [
        pl.col("entity_id").cast(pl.Utf8).str.strip_chars(),
        build_clean_country_expr("country").alias("country"),
        build_clean_name_expr("business_name").alias("clean_name"),
        build_clean_address_expr("business_address").alias("clean_address"),
    ]

    if "source" not in df.columns and source_tag is not None:
        exprs.append(pl.lit(source_tag).alias("source"))
    elif "source" in df.columns:
        exprs.append(pl.col("source").cast(pl.Utf8))

    cleaned = df.with_columns(exprs)

    if verbose:
        print(f"    Normalized {len(df):,} records in {time.time()-t0:.2f}s")

    return cleaned


def run_preprocessing_pipeline(
    s1_path: str,
    s2_path: str,
    s3_path: str,
    output_dir: str,
    export_formats: Tuple[str, ...] = ("parquet", "tsv"),
    sample_s1: Optional[int] = None,
    sample_target: Optional[int] = None,
    verbose: bool = True
) -> Dict[str, Dict[str, pl.DataFrame]]:
    """
    Executes the complete Phase 0 Preprocessing & Partitioning pipeline.

    Parameters:
        s1_path: Path to Source 1 TSV file (Reference Query pool)
        s2_path: Path to Source 2 TSV file (Target pool)
        s3_path: Path to Source 3 TSV file (Target pool)
        output_dir: Destination folder for preprocessed partitions
        export_formats: Tuple of formats to export: ('parquet', 'tsv')
        sample_s1: Optional limit on Source 1 rows for testing
        sample_target: Optional limit on Source 2/3 rows for testing
        verbose: Whether to log detailed execution statistics

    Returns:
        Dict mapping: country -> {'query': pl.DataFrame, 'target': pl.DataFrame}
    """
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print("=" * 80)
        print("AMAZON ML CHALLENGE 2026 — PHASE 0 PREPROCESSING & PARTITIONING")
        print("=" * 80)
        print(f"  Source 1 (Query):  {s1_path}")
        print(f"  Source 2 (Target): {s2_path}")
        print(f"  Source 3 (Target): {s3_path}")
        print(f"  Output Directory:  {output_dir}")
        print(f"  Export Formats:    {', '.join(export_formats)}")
        print("=" * 80)

    # 1. Load and Standardize Raw Sources
    if verbose:
        print("\n[Stage 1/4] Loading Raw TSVs into Polars Engine...")

    s1_df = load_dataset_schema(s1_path, name="Source 1 (Query)", source_tag="S1", verbose=verbose)
    if sample_s1 and sample_s1 < len(s1_df):
        if verbose:
            print(f"  [Sample Mode] Limiting Source 1 to {sample_s1:,} rows")
        s1_df = s1_df.slice(0, sample_s1)

    s2_df = load_dataset_schema(s2_path, name="Source 2 (Target)", source_tag="S2", verbose=verbose)
    if sample_target and sample_target < len(s2_df):
        s2_df = s2_df.slice(0, sample_target)

    s3_df = load_dataset_schema(s3_path, name="Source 3 (Target)", source_tag="S3", verbose=verbose)
    if sample_target and sample_target < len(s3_df):
        s3_df = s3_df.slice(0, sample_target)

    # 2. Text Normalization Pipeline (Phase 0 Spec)
    if verbose:
        print("\n[Stage 2/4] Vectorized Text Cleaning & Normalization...")

    t_norm = time.time()
    s1_clean = clean_dataframe(s1_df, source_tag="S1", verbose=verbose)
    del s1_df
    gc.collect()

    s2_clean = clean_dataframe(s2_df, source_tag="S2", verbose=verbose)
    del s2_df
    gc.collect()

    s3_clean = clean_dataframe(s3_df, source_tag="S3", verbose=verbose)
    del s3_df
    gc.collect()

    if verbose:
        print(f"  Total normalization completed in {time.time()-t_norm:.2f}s")

    # 3. Target Union (S2 + S3) with Source Column Tracking
    if verbose:
        print("\n[Stage 3/4] Unifying Target Pool (S2 ∪ S3) with Source Origin Tagging...")

    t_union = time.time()
    target_clean = pl.concat([s2_clean, s3_clean], how="vertical_relaxed")
    del s2_clean, s3_clean
    gc.collect()

    n_queries = len(s1_clean)
    n_targets = len(target_clean)
    if verbose:
        print(f"  Combined Target Pool: {n_targets:,} records ({time.time()-t_union:.2f}s)")

    # 4. Country-Wise Partitioning & Serialization
    if verbose:
        print("\n[Stage 4/4] Country-Wise Partitioning & Output Serialization...")

    unique_countries = sorted([c for c in s1_clean["country"].unique().to_list() if c])
    if verbose:
        print(f"  Discovered {len(unique_countries)} Country Partitions: {unique_countries}")

    partition_results: Dict[str, Dict[str, pl.DataFrame]] = {}
    manifest_records = {}

    for c_idx, country in enumerate(unique_countries, 1):
        t_country = time.time()
        c_code = country.upper()

        query_c = s1_clean.filter(pl.col("country") == country)
        target_c = target_clean.filter(pl.col("country") == country)

        n_q = len(query_c)
        n_t = len(target_c)

        if verbose:
            print(f"\n  --- Partition [{c_idx}/{len(unique_countries)}]: {c_code} ---")
            print(f"      Query_{c_code}:  {n_q:,} entities")
            print(f"      Target_{c_code}: {n_t:,} entities (S2: {len(target_c.filter(pl.col('source')=='S2')):,}, S3: {len(target_c.filter(pl.col('source')=='S3')):,})")

        exported_files = {}

        # Export Query Partition
        query_base = os.path.join(output_dir, f"Query_{c_code}")
        if "parquet" in export_formats:
            q_parquet = f"{query_base}.parquet"
            query_c.write_parquet(q_parquet, compression="snappy")
            exported_files[f"Query_{c_code}_parquet"] = q_parquet
        if "tsv" in export_formats:
            q_tsv = f"{query_base}.tsv"
            query_c.write_csv(q_tsv, separator="\t")
            exported_files[f"Query_{c_code}_tsv"] = q_tsv

        # Export Target Partition
        target_base = os.path.join(output_dir, f"Target_{c_code}")
        if "parquet" in export_formats:
            t_parquet = f"{target_base}.parquet"
            target_c.write_parquet(t_parquet, compression="snappy")
            exported_files[f"Target_{c_code}_parquet"] = t_parquet
        if "tsv" in export_formats:
            t_tsv = f"{target_base}.tsv"
            target_c.write_csv(t_tsv, separator="\t")
            exported_files[f"Target_{c_code}_tsv"] = t_tsv

        manifest_records[c_code] = {
            "country": c_code,
            "query_count": n_q,
            "target_count": n_t,
            "target_s2_count": len(target_c.filter(pl.col("source") == "S2")),
            "target_s3_count": len(target_c.filter(pl.col("source") == "S3")),
            "files": exported_files,
            "partition_time_seconds": round(time.time() - t_country, 3),
        }

        partition_results[c_code] = {
            "query": query_c,
            "target": target_c,
        }

        if verbose:
            print(f"      Partition {c_code} exported in {time.time()-t_country:.2f}s")

    # 5. Write Execution Manifest
    manifest_path = os.path.join(output_dir, "manifest.json")
    manifest_data = {
        "pipeline": "Phase 0 Preprocessing & Partitioning",
        "spec": "Preprocessing_1.0.md",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_runtime_seconds": round(time.time() - start_time, 2),
        "total_queries": n_queries,
        "total_targets": n_targets,
        "partitions": manifest_records,
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    if verbose:
        print("\n" + "=" * 80)
        print("PHASE 0 PREPROCESSING & PARTITIONING COMPLETED SUCCESSFULLY")
        print(f"  Total Runtime:     {time.time()-start_time:.2f}s")
        print(f"  Total Queries:     {n_queries:,}")
        print(f"  Total Targets:     {n_targets:,}")
        print(f"  Output Manifest:   {manifest_path}")
        print("=" * 80)

    return partition_results
