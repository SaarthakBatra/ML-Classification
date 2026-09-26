# Business Entity Resolution Workflow

The project is organized as two explicit workflow stages. The only virtual environment belongs at `pre-process/venv/`; both root-level runners use that interpreter.

```text
Raw S1, S2, S3 TSVs
        │
        ▼
run_preprocess.py
        │
        ▼
Query_<COUNTRY>.parquet, Target_<COUNTRY>.parquet, manifest.json
        │
        ▼
run_blocking.py
        │
        ▼
candidate_pairs.tsv, matching_results.tsv, manifest.json
```

## 1. Preprocessing

`run_preprocess.py` cleans the three source files, unions S2 and S3 into the target pool, normalizes country values, and writes separate query and target partitions for every country discovered in S1.

For the provided test split:

```bash
pre-process/venv/bin/python run_preprocess.py --dataset test
```

For training data:

```bash
pre-process/venv/bin/python run_preprocess.py --dataset train
```

For explicit input files:

```bash
pre-process/venv/bin/python run_preprocess.py \
  --s1 /path/to/source1.tsv \
  --s2 /path/to/source2.tsv \
  --s3 /path/to/source3.tsv \
  --output-dir pre-process/data/preprocessed/custom
```

The output directory contains a `manifest.json` that is the contract consumed by the blocking stage. Keep the manifest and its referenced Parquet files together.

## 2. Blocking

`run_blocking.py` loads preprocessing's manifest. For each country, it builds a character 3-gram BM25 index over normalized addresses and a multilingual FAISS index over normalized names. It then retrieves candidates in cycles, blacklists candidates already evaluated for a query, promotes candidates retrieved by both streams, and stops when the index is exhausted, 60 candidates have been seen, or the third cycle produces no accepted matches.

```bash
pre-process/venv/bin/python run_blocking.py \
  --preprocessed-dir pre-process/data/preprocessed/test \
  --output-dir blocking/output/test \
  --device cpu
```

Use `--device auto` to select CUDA or Apple Silicon acceleration when available. The first run requires the multilingual sentence-transformer model to be available locally or downloadable by SentenceTransformers.

Blocking writes:

- `candidate_pairs.tsv`: one row per retrieved candidate, with source ID, target ID, cycle, and stream ranks.
- `matching_results.tsv`: accepted matches. It is currently empty because feature engineering and the XGBoost evaluator are not implemented yet.
- `manifest.json`: runtime summary and output locations.

## Tests

```bash
pre-process/venv/bin/python pre-process/run_tests.py
pre-process/venv/bin/python -m unittest discover -s blocking/tests -p 'test_*.py'
```
