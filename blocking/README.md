# Blocking workflow

This workflow consumes the country-partitioned artifacts produced by `pre-process`. It builds a character 3-gram BM25 address index and a multilingual FAISS name index for every country, then runs the progressive cyclic retrieval policy from `Blocking_1.0.md`.

```bash
pre-process/venv/bin/python run_blocking.py \
  --preprocessed-dir pre-process/data/preprocessed/test \
  --output-dir blocking/output/test \
  --device cpu
```

`candidate_pairs.tsv` contains every unseen candidate admitted during each retrieval cycle, including its stream ranks. `matching_results.tsv` is intentionally empty until the feature-engineering and classifier workflow supplies an evaluator; the blocking workflow records this limitation in its manifest instead of fabricating match decisions.
