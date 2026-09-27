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

`run_preprocess.py` cleans the three source files, unions S2 and S3 into the target pool, normalizes country values, and writes separate query and target partitions for every country discovered in S1. Each partition contains `name_for_faiss` (lowercased original script), `name_for_bm25`, and `addr_for_bm25`.

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

`run_blocking.py` loads preprocessing's manifest. For each country, it combines three independent retrieval streams: character 3-gram BM25-style search over normalized names, character 3-gram BM25-style search over normalized addresses, and BGE-M3 dense name search through FAISS. Candidates returned by more streams rank first. The default ceiling is 90 unseen candidates per query, not 20.

```bash
pre-process/venv/bin/python run_blocking.py \
  --preprocessed-dir pre-process/data/preprocessed/test \
  --output-dir blocking/output/test \
  --device cuda
```

For Kaggle, install the blocking dependencies first:

```bash
pip install -r blocking/requirements.txt
```

Use `--device cuda` on Kaggle. BGE-M3 weights are downloaded from Hugging Face on the first run.

Blocking writes:

- `candidate_pairs.tsv`: one row per S1 ID with all retrieved target IDs.
- `candidate_pairs_per_cycle.tsv`: retrieval provenance by cycle.
- `matching_results.tsv`: accepted matches. It is currently empty because feature engineering and the XGBoost evaluator are not implemented yet.
- `manifest.json`: runtime summary and output locations.

## Tests

```bash
pre-process/venv/bin/python pre-process/run_tests.py
pre-process/venv/bin/python -m unittest discover -s blocking/tests -p 'test_*.py'
```
