"""
High-throughput benchmark for dual-track normalizer on 100,000 real training rows.
Measures row throughput (records/second) and latency.
"""

import sys
import os
import time

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.dirname(BENCH_DIR)
sys.path.insert(0, RESULT_DIR)
sys.path.insert(0, os.path.join(RESULT_DIR, "src"))

import polars as pl
from normalizers.batch import process_dataframe

DATASET_PATH = os.path.join(os.path.dirname(RESULT_DIR), "dataset", "train", "train_source1.tsv")


def run_benchmark(n_rows: int = 100000):
    print("=" * 65)
    print(f"BENCHMARK: Dual-Track Normalizer ({n_rows:,} rows)")
    print("=" * 65)

    if not os.path.isfile(DATASET_PATH):
        print(f"Warning: Dataset not found at {DATASET_PATH}. Skipping.")
        return

    print(f"Reading first {n_rows:,} rows...")
    t0 = time.time()
    df = pl.read_csv(DATASET_PATH, separator="\t", n_rows=n_rows)
    t_read = time.time() - t0
    print(f"Loaded {len(df):,} rows in {t_read:.2f}s ({len(df)/t_read:,.0f} rows/s)")

    print("Executing Dual-Track Normalizer...")
    t1 = time.time()
    processed_df = process_dataframe(df)
    t_process = time.time() - t1
    throughput = len(processed_df) / t_process
    print(f"Processed {len(processed_df):,} rows in {t_process:.2f}s ({throughput:,.0f} rows/s)")

    print("\nSample Output:")
    sample = processed_df.select([
        "entity_id", "agg_name", "agg_addr", "addr_nums", "embed_name"
    ]).head(5)
    print(sample)
    print("=" * 65)


if __name__ == "__main__":
    run_benchmark(100000)
