"""
Fast Validation Split Builder.
Extracts 2,000 Source 1 records and ALL of their corresponding true matches from
Source 2 and Source 3, plus a distractor pool of 50,000 records.
Caches to parquet in result/data/ for high-speed repeated testing.
"""

import os
import sys
import time
from typing import Dict, Set, Tuple
import polars as pl

RESULT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.dirname(RESULT_DIR)
TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")
DATA_DIR = os.path.join(RESULT_DIR, "data")


def build_and_cache_validation_split(n_s1: int = 2000, n_distractors: int = 50000) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    out_s1_path = os.path.join(DATA_DIR, f"val_s1_{n_s1}.parquet")
    out_target_path = os.path.join(DATA_DIR, f"val_target_{n_s1}.parquet")
    out_gt_path = os.path.join(DATA_DIR, f"val_gt_{n_s1}.parquet")

    if os.path.exists(out_s1_path) and os.path.exists(out_target_path) and os.path.exists(out_gt_path):
        print(f"Validation split already exists at {DATA_DIR}")
        return DATA_DIR

    print(f"Building validation split for {n_s1:,} S1 entities...")
    t0 = time.time()

    # 1. Read S1 sample
    s1_df = pl.read_csv(
        os.path.join(TRAIN_DIR, "train_source1.tsv"),
        separator="\t",
        n_rows=n_s1
    )
    s1_ids = set(s1_df["entity_id"].to_list())

    # 2. Get Ground Truth
    gt_df = pl.read_csv(
        os.path.join(TRAIN_DIR, "train_ground_truth.tsv"),
        separator="\t"
    ).filter(pl.col("source1_entity_id").is_in(s1_ids))

    target_s2_needed = set()
    target_s3_needed = set()

    for row in gt_df.iter_rows(named=True):
        matched = row["matched_entity_ids"]
        if matched:
            for m_id in matched.split(","):
                m_id = m_id.strip()
                if m_id.startswith("S2-"):
                    target_s2_needed.add(m_id)
                elif m_id.startswith("S3-"):
                    target_s3_needed.add(m_id)

    print(f"Required True Matches: {len(target_s2_needed):,} from S2, {len(target_s3_needed):,} from S3")

    # 3. Read S2: True matches + distractors
    print("Extracting S2 target records...")
    s2_full = pl.read_csv(
        os.path.join(TRAIN_DIR, "train_source2.tsv"),
        separator="\t"
    )
    s2_matches = s2_full.filter(pl.col("entity_id").is_in(target_s2_needed))
    s2_distractors = s2_full.filter(~pl.col("entity_id").is_in(target_s2_needed)).head(n_distractors // 2)
    s2_target = pl.concat([s2_matches, s2_distractors]).with_columns(pl.lit("S2").alias("source"))

    # 4. Read S3: True matches + distractors
    print("Extracting S3 target records...")
    s3_full = pl.read_csv(
        os.path.join(TRAIN_DIR, "train_source3.tsv"),
        separator="\t"
    )
    s3_matches = s3_full.filter(pl.col("entity_id").is_in(target_s3_needed))
    s3_distractors = s3_full.filter(~pl.col("entity_id").is_in(target_s3_needed)).head(n_distractors // 2)
    s3_target = pl.concat([s3_matches, s3_distractors]).with_columns(pl.lit("S3").alias("source"))

    # 5. Union targets
    target_df = pl.concat([s2_target, s3_target])

    # Save to parquet
    s1_df.write_parquet(out_s1_path)
    gt_df.write_parquet(out_gt_path)
    target_df.write_parquet(out_target_path)

    elapsed = time.time() - t0
    print(f"Validation split created and cached in {elapsed:.2f}s:")
    print(f"  S1 Records:      {len(s1_df):,}")
    print(f"  Target Records:  {len(target_df):,} (includes 100% of true matches + {n_distractors:,} distractors)")
    print(f"  True Match IDs:  {len(target_s2_needed) + len(target_s3_needed):,}")
    return DATA_DIR


def load_validation_split(n_s1: int = 2000):
    s1_path = os.path.join(DATA_DIR, f"val_s1_{n_s1}.parquet")
    target_path = os.path.join(DATA_DIR, f"val_target_{n_s1}.parquet")
    gt_path = os.path.join(DATA_DIR, f"val_gt_{n_s1}.parquet")

    if not (os.path.exists(s1_path) and os.path.exists(target_path) and os.path.exists(gt_path)):
        build_and_cache_validation_split(n_s1)

    s1_df = pl.read_parquet(s1_path)
    target_df = pl.read_parquet(target_path)
    gt_df = pl.read_parquet(gt_path)

    gt_map: Dict[str, Set[str]] = {}
    for row in gt_df.iter_rows(named=True):
        matched = row["matched_entity_ids"]
        gt_map[row["source1_entity_id"]] = set(m.strip() for m in matched.split(",") if m.strip()) if matched else set()

    return s1_df, target_df, gt_map


if __name__ == "__main__":
    build_and_cache_validation_split(2000, 50000)
