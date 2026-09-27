# Blocking workflow

This workflow consumes the country-partitioned artifacts produced by `pre-process`. It builds character 3-gram retrieval indices for normalized names and addresses plus a BGE-M3 FAISS name index for every country, then runs the progressive cyclic retrieval policy from `Blocking_1.0.md`.

```bash
pre-process/venv/bin/python run_blocking.py \
  --preprocessed-dir pre-process/data/preprocessed/test \
  --output-dir blocking/output/test \
  --device cuda
```

`candidate_pairs.tsv` contains one aggregated candidate list per S1 entity. `candidate_pairs_per_cycle.tsv` keeps cycle provenance. `matching_results.tsv` is intentionally empty until the feature-engineering and classifier workflow supplies an evaluator.
