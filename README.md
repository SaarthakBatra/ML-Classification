# Business Entity Resolution — Modular Candidate Generation Pipeline

High-performance, country-partitioned multi-channel candidate generation pipeline engineered for the **ML Challenge 2026**.

---

## 1. System Architecture & Folder Arrangement

The codebase is organized into domain-driven, single-responsibility modules:

```text
AMAZON/result/
├── config/                      # Hyperparameter & Pipeline Configuration
│   ├── __init__.py
│   └── settings.py              # Typed dataclasses: NormalizerConfig, BlockingConfig, FusionConfig, PipelineConfig
│
├── src/                         # Modular Core Implementation
│   ├── __init__.py
│   ├── normalizers/             # Dual-Track Normalization Layer
│   │   ├── __init__.py
│   │   ├── base.py              # Regex primitives (legal suffixes, URLs, street abbreviations)
│   │   ├── aggressive.py        # Track A: Diacritic stripping, legal suffix removal, numeric extraction
│   │   ├── moderate.py          # Track B: Multilingual semantic preservation (Tamil, Hindi, French)
│   │   └── batch.py             # Vectorized Polars DataFrame batch processor (>70,000 rows/sec)
│   │
│   ├── blocking/                # Multi-Channel Blocking Layer
│   │   ├── __init__.py
│   │   ├── base.py              # Abstract BaseIndexer contract (fit, retrieve)
│   │   ├── b1_numeric_geo.py    # Channel B1: Physical premise address numeric pairs
│   │   ├── b2_name_tokens.py    # Channel B2: Core name token inverted index + O(1) Jaccard
│   │   ├── b3_fuzzy.py          # Channel B3: Rapid prefix-bucketing & typo-tolerant RapidFuzz ratio
│   │   ├── b4_address_tokens.py # Channel B4: Address token inverted index + O(1) Jaccard
│   │   ├── b5_dense.py          # Channel B5: Multilingual Bi-Encoder (SentenceTransformers + FAISS)
│   │   └── registry.py          # Indexer Factory & Dynamic Registry
│   │
│   ├── fusion/                  # Consensus Ranking & Candidate Pruning Layer
│   │   ├── __init__.py
│   │   └── rank_fusion.py       # Weighted positional rank discount & hard budget cap (K <= 20)
│   │
│   ├── evaluation/              # Metrics & Statistical Distribution Layer
│   │   ├── __init__.py
│   │   ├── metrics.py           # Ground-truth Pair Recall & Reduction Ratio evaluator
│   │   └── analyzer.py          # Distribution percentiles, histograms, S2/S3 splits
│   │
│   ├── pipeline/                # End-to-End Pipeline Orchestration Layer
│   │   ├── __init__.py
│   │   ├── loader.py            # TSV schema normalization & missing column handling
│   │   ├── partition.py         # Country-partitioned execution & memory bounds
│   │   └── generator.py         # TSV streaming serialization & official validation
│   │
│   ├── cli/                     # Accessible Command-Line Interface (a11y)
│   │   ├── __init__.py
│   │   └── main.py              # CLI with informative logging and auto-validation
│   │
│   ├── blocking.py              # Backward-compatible shim
│   ├── normalizer.py            # Backward-compatible shim
│   ├── dense_retriever.py       # Backward-compatible shim
│   └── analyze_metrics.py       # Backward-compatible shim
│
├── tests/                       # Automated Test Suite (unittest)
│   ├── __init__.py
│   ├── test_normalizer.py       # Unit tests for diacritics, URLs, abbreviations, batching
│   ├── test_blocking.py         # Unit & benchmark tests for B1, B2, B3, B4, and fusion
│   └── test_dense_blocking.py   # Unit & benchmark tests for B5 and 5-channel fusion
│
├── benchmarks/                  # Performance & Latency Benchmarks
│   ├── __init__.py
│   ├── benchmark_normalizer.py  # Throughput stress test (100k rows)
│   └── benchmark_blocking.py    # Per-channel latency & recall gain benchmark
│
├── generate_candidates.py       # Top-level CLI entrypoint
├── run_tests.py                 # Top-level test runner
└── README.md                    # System documentation
```

---

## 2. Multi-Channel Blocking Mechanics

| Channel | Core Strategy | Feature Representation | Complexity / Latency |
|:---|:---|:---|:---|
| **B1: Numeric Geo** | Inverted index on premise numeric pairs | Sorted premise digits + street token | Inverted Index: $O(1)$ lookup |
| **B2: Name Tokens** | Inverted index on core name tokens | Suffix-stripped alphabetically sorted tokens | $O(\|Q\| \cdot \|Postings\|)$ |
| **B3: Fuzzy Ratio** | 3-char prefix bucketing + RapidFuzz | Character 3-grams | Prefix partitioned: $< 0.1$ms |
| **B4: Address Tokens** | Inverted index on address tokens | Street, locality, city tokens | $O(1)$ set arithmetic Jaccard |
| **B5: Dense Vector** | Bi-Encoder (MiniLM-L12) + FAISS | Multilingual text embedding | Vector search: $O(N \cdot D)$ |

### Fast Mathematical Jaccard via Set Arithmetic
Rather than computing costly set intersections and unions at query time, Jaccard is computed in $O(1)$ scalar math:
$$\text{Jaccard}(A, B) = \frac{|A \cap B|}{|A| + |B| - |A \cap B|}$$

### Positional Rank Discounting
When fusing candidates from multiple channels, candidates at the top of each channel list receive a positional rank advantage:
$$\text{Score}(c) = \sum_{m \in \text{Channels}} w_m \cdot \left( (1 - \lambda) + \lambda \cdot \frac{K_m - \text{rank}_m(c)}{K_m} \right)$$
where $\lambda = 0.5$ is the rank discount slope, and $K_m$ is the channel candidate pool size.

---

## 3. Quick Start & Execution

### Running the Test Suite
To run all modular unit and integration tests:
```bash
python run_tests.py
```

### Running Throughput Benchmarks
```bash
python benchmarks/benchmark_normalizer.py
python benchmarks/benchmark_blocking.py
```

### Generating Candidate Pairs
To execute the pipeline and generate `candidate_pairs.tsv`:
```bash

# Full train dataset generation:
python generate_candidates.py \
    --s1 ../dataset/train/train_source1.tsv \
    --s2 ../dataset/train/train_source2.tsv \
    --s3 ../dataset/train/train_source3.tsv \
    --output output/candidate_pairs_train.tsv \
    --budget 20 \
    --channels b1,b2,b3,b4,b5 \
    --no-validate

# Rapid dry-run on sample (e.g. first 2,000 S1 queries):
python generate_candidates.py \
    --sample-s1 2000 \
    --output output/candidate_pairs_sample.tsv
```

### Evaluating & Studying Metrics

#### 1. Ground-Truth Quality Metrics (Pair Recall & Search Space Pruning)
Evaluate generated candidate pairs against ground truth:
```bash
python evaluate_metrics.py \
    --candidate output/candidate_pairs_train.tsv \
    --ground-truth ../dataset/train/train_ground_truth.tsv
```

#### 2. Candidate Distribution & Statistical Analysis (Test or Train)
Compute percentiles, reduction ratio, zero-candidate counts, and histograms:
```bash
python analyze_metrics.py --candidate output/candidate_pairs.tsv
```

### Validating Submission Compliance
Output files strictly adhere to challenge submission rules:
```bash
python ../utils/validate_submission.py \
    --candidate output/candidate_pairs.tsv \
    --test-dir ../dataset/test
```

