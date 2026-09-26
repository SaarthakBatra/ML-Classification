# Implementation Plan v2.3 — Amazon ML Challenge ER Pipeline
### Status: UPDATED — 3-Stream Blocking + Multilingual Transliteration

---

## Change Log

### v2.2 → v2.3 (Multilingual Script Updates)
| Issue | v2.2 | v2.3 Fix |
|---|---|---|
| 🔴 CRITICAL BUG | `encode("ascii")` drops 542k true non-Latin pairs | Script detection + phonetically transliterate Indic scripts |
| 🟠 ARCHITECTURE | Single normalized text column | Dual columns: `name_for_faiss` (orig) vs `name_for_bm25` (Latin/ITRANS) |
| 🟡 FEATURE | Missing cross-script signal | Added `is_cross_script` ML meta-feature |

### v2.1 → v2.2 (3-Stream Architecture)
| # | Change | Detail |
|---|---|---|
| 7 | 2 streams → **3 streams** (BM25-name + BM25-addr + FAISS-name) | No cross-contamination; independent signals; ≤60 candidates/cycle (3×20); `MAX_SEEN` raised 60 → 90 |
| 9 | Feature table expanded: +2 features | `name_bm25_score` and `addr_bm25_score` now separate features; RRF formula updated to 3-stream |

### v1.0 → v2.1 (Audit Fixes)
| Issue | v1.0 | v2.0/v2.1 Fix |
|---|---|---|
| 🔴 BUG 1 | `approach_2.md` inverted direction | Marked DEPRECATED; correct direction documented |
| 🔴 BUG 2 | 1:1 Bipartite post-processing drops valid matches | Removed entirely |
| 🔴 BUG 3 | `candidate_pairs.tsv` written per-cycle (multiple rows) | Dual output: per-cycle file + per-run accumulator file |
| 🟠 GAP 2 | No training data generation step | Explicit 2-mode orchestrator added (train | infer) |
| 🟠 GAP 3 | `seen_list` scope undefined | `seen_list` replaced by global `seen_dict` for O(1) lookups |

---

## 1. Goal Description

Build a robust, scalable Python pipeline that resolves business entity records from 3 independent, noisy sources. The pipeline ingests raw `.tsv` files, cleanly handles multilingual text (Devanagari, Tamil, etc.), runs a Progressive Cyclic DAG combining FAISS semantic search and BM25 lexical search for candidate generation, extracts an 18-dimensional feature vector per candidate pair, and applies a calibrated XGBoost classifier to produce perfectly formatted `matching_results.tsv` and `candidate_pairs.tsv` output files.

---

## 2. Corrected Project Structure

```text
Amazon ML Challange/
├── src/
│   ├── main.py                       # CLI entry point (--mode train | infer)
│   ├── config.py                     # All hyperparameters, paths, thresholds
│   ├── data_layer/
│   │   ├── loader.py                 # Polars TSV loader + null imputation
│   │   └── cleaner.py                # Transliteration + dual representation normalization
│   ├── blocking_layer/
│   │   ├── semantic_index.py         # FAISS + Sentence Transformers (name only)
│   │   ├── lexical_name_index.py     # BM25 char-3gram on name
│   │   ├── lexical_addr_index.py     # BM25 char-3gram on address
│   │   └── seen_dict.py              # Global seen_dict: init, update, write_all_outputs
│   ├── feature_layer/
│   │   └── feature_extractor.py      # 18-dim feature vector per candidate pair
│   ├── ml_layer/
│   │   ├── trainer.py                # XGBoost GroupKFold training + threshold opt
│   │   └── classifier.py            # Load model + online inference
│   └── pipeline/
│       └── orchestrator.py           # 2-mode cyclic DAG (train | infer)
├── output/
│   ├── matching_results.tsv          # Final matches (leaderboard upload)
│   ├── candidate_pairs.tsv           # Per-run accumulator (one row per S1 entity)
│   └── candidate_pairs_per_cycle.tsv # Per-cycle raw log (for debugging)
├── models/
│   └── xgb_model.pkl                 # Saved trained model + optimal threshold
├── notebooks/
│   └── eda.ipynb                     # Step 0: EDA before any code is written
├── utils/
│   └── (validate_submission.py lives in student_resource/utils/)
├── requirements.txt
├── run.sh
└── README.md                         # Exact reproduction instructions
```

---

## 3. Python Dependencies (`requirements.txt`)

```
polars>=0.20.0
numpy>=1.26.0
scikit-learn>=1.4.0
faiss-cpu>=1.7.4
rank-bm25>=0.2.2
sentence-transformers>=2.6.0
xgboost>=2.0.0
lightgbm>=4.3.0
jellyfish>=1.0.0
pyphonetics>=0.5.0
textdistance>=4.6.0
torch>=2.2.0
indic-transliteration>=2.3.62
```

---

## 4. CLI Interface — `run.sh` & `main.py`

```bash
# TRAIN MODE: Runs on training data to generate candidates, label them,
#             train XGBoost, and save model + optimal threshold.
bash run.sh --mode train \
            --data-dir dataset/train \
            --ground-truth dataset/train/train_ground_truth.tsv \
            --model-out models/xgb_model.pkl

# INFER MODE: Loads saved model, runs cyclic DAG on test data,
#             and generates both output files.
bash run.sh --mode infer \
            --data-dir dataset/test \
            --model-in models/xgb_model.pkl \
            --output-dir output/
```

> [!IMPORTANT]
> `run.sh` simply calls `python3 src/main.py` with the same flags. `config.py` stores default paths, thresholds, and hyperparameters. No path is hardcoded anywhere else.

---

## 5. Architectural Decisions (Unchanged from v1.0)

**Decision 1: Polars as Data Engine**
All dataframe operations use Polars (Rust-backed, multi-threaded). Prevents OOM on 2.2M+ row datasets. All string operations use `.str.replace_all()` with regex.

**Decision 2: Local GPU (NVIDIA RTX 3060, 6GB VRAM)**
`paraphrase-multilingual-MiniLM-L12-v2` (~118M params, ~470MB in VRAM). Runs inference with `batch_size=256` on CUDA. Embedding 2.2M records takes ~15–30 min locally instead of days on CPU.

**Decision 3: 2-Mode Orchestrator (RECOMMENDED ARCHITECTURE)**
A single `orchestrator.py` with a `mode` flag is the cleanest architecture:
- **Train mode:** Runs the full Cyclic DAG on training data → generates candidate pairs → joins with `train_ground_truth.tsv` → extracts features → trains and saves XGBoost.
- **Infer mode:** Loads saved model → runs Cyclic DAG on test data → produces output files.

This avoids code duplication, ensures the train and test pipelines are identical (preventing silent distribution drift), and makes the submission's reproduction README trivial.

---

## 6. Phase 0: Preprocessing & Partitioning v2.0 (Multilingual Safe)

Our EDA revealed that **7.10%** of all true match pairs in the ground truth are cross-script (Latin S1 ↔ Indic S2/S3). The previous `NFKD + ASCII encode` pipeline silently destroyed these. We now use a dual-representation transliteration pipeline.

### 6A. High-Speed Data Loading & Imputation
*   **Engine:** `polars.read_csv(separator='\t')`
*   **Literal Null Parsing:** Map common fake-null string literals (`["n/a", "na", "null", "none", "-"]`) to `""`.
*   **Missing Values:** Immediately fill nulls with empty strings `""` to prevent `NoneType` errors.
*   **Data Types:** `country` will be cast to a Categorical type to save memory.

### 6B. Multilingual Script Detection & Transliteration
*   **Script Detection:** We will use regex patterns (`[\u0900-\u097F]` for Devanagari, etc.) to detect Indic scripts (Devanagari, Bengali, Tamil, Telugu, Gujarati, Kannada, Malayalam, Gurmukhi).
*   **Transliteration:** If an Indic script is detected, we will use the `indic-transliteration` library to convert it to the **ITRANS** Roman scheme (phonetic Latin). This allows BM25 char 3-grams to partially overlap with the Latin S1 queries.

### 6C. Dual Representation Text Normalization Pipeline
Since FAISS (MiniLM) and BM25 have fundamentally different requirements for text matching, we generate **two independent representations** for `business_name`.

#### Name for FAISS (`name_for_faiss`)
MiniLM is a multilingual model natively supporting 50+ languages. We must preserve the original script.
*   **Process:** Lowercase -> Trim leading/trailing whitespace -> Collapse multiple spaces into one.
*   **Crucial Rule:** NO NFKD/ASCII encoding, NO transliteration, NO punctuation stripping. (Preserves meaning for semantic embeddings).

#### Name for BM25 (`name_for_bm25`)
BM25 relies on exact character 3-gram overlaps. It requires an aggressively sanitized Latin-only representation.
*   **Transliteration:** If Indic, transliterate to ITRANS.
*   **Unicode Normalization:** Safe `NFKD` encoding and ASCII coercion (`encode("ascii", "ignore").decode()`).
*   **Case & Space:** Lowercase and strip.
*   **Punctuation Stripping:** 
    *   Protect acronyms: Remove periods `.` with no space replacement (`I.B.M.` -> `IBM`).
    *   Map `@` -> ` at ` and `&` -> ` and `.
    *   Remove all remaining non-alphanumeric characters.
*   **Abbreviation Expansion:** Use `\b` anchored regex to expand `corp`->`corporation`, `inc`->`incorporated`, etc.

#### Address Normalization (`addr_for_bm25`)
FAISS is not used on addresses. Addresses are highly structured, so we only need a BM25 representation.
*   **Token-Level Transliteration:** Split the address into words. Transliterate only the Indic tokens, keeping Latin tokens intact.
*   **Standardization:** Apply the same pipeline as `name_for_bm25` (ASCII coercion, lowercase, symbol mapping, punctuation stripping).
*   **Address Abbreviations:** Expand `st`->`street`, `rd`->`road`, etc., using `\b` anchors.

### 6D. Empty String Safety Fallback
*   If any resulting representation is an empty string `""`, we replace it with `f"nullname {entity_id}"` (or `nulladdr {entity_id}`). This creates a globally unique singleton string, avoiding artificial clustering in indices.

### 6E. Detailed Output Expectations (Partitioned Parquet)
The preprocessing script outputs serialized files (e.g., `.parquet` for maximum I/O speed into Phase 1) partitioned by country and split into query vs. target sets (e.g., `Query_india.parquet`, `Target_india.parquet`).

Schema per row:
*   `entity_id` (String): Primary key.
*   `country` (String): Normalized country name.
*   `source` (String): `"S1"`, `"S2"`, or `"S3"`.
*   `name_original` & `addr_original` (String): Raw fields.
*   `name_for_faiss` (String): Lowercased, original script preserved.
*   `name_for_bm25` (String): Latin-only, transliterated, fully normalized.
*   `addr_for_bm25` (String): Latin-only, transliterated, normalized.
*   `is_cross_script` (Int8): `1` if the original name contained Indic scripts, `0` if purely Latin.

---

## 7. Phase 1–2: Blocking Layer (3-Stream)

### 7A. Final Stream Assignment

| # | Stream | Index | Field | Why This, Not Mixed |
|---|---|---|---|---|
| 1 | **BM25-Name** | `BM25Okapi` | `name_for_bm25` (char 3-grams) | Lexically catches name typos, abbreviation variants, transpositions (`Retail Bata` vs `Batra Retail`). Name is short — won't be drowned by address. |
| 2 | **BM25-Addr** | `BM25Okapi` | `addr_for_bm25` (char 3-grams) | Lexically catches address typos, numeric code overlap (ZIP, building numbers). Address is long — its own dedicated index prevents it from drowning the name signal. |
| 3 | **FAISS-Name** | `IndexFlatIP` | `name_for_faiss` (MiniLM embeddings) | Semantically catches synonyms and abbreviations (`IBM` ≈ `International Business Machines`, `Corp` ≈ `Corporation`). |

### 7B. Why FAISS Is Forbidden on Addresses
Semantic models capture *meaning*, not structure. Addresses are structured codes — not prose.
- **Adjacent building numbers look semantically identical:** `123 Main St` vs `124 Main St` embed to nearly identical vectors.
- **BM25 char-3grams already handle address typos natively:** Overlapping 3-grams handle typos perfectly without semantic models.

### 7C. Per-Cycle Candidate Budget

| Stream | k per cycle | Max new candidates after blacklist filter |
|---|---|---|
| BM25-Name | 20 + total_seen | 20 |
| BM25-Addr | 20 + total_seen | 20 |
| FAISS-Name | 20 + total_seen | 20 |
| **Union (≤3×20, deduplicated)** | — | **≤ 60 per cycle** |

**MAX_SEEN updated: 60 → 90**
- 3 streams × 20 candidates × minimum 3 cycles = **90 worst-case unique candidates**
- Stopping condition: `seen_dict[s1_id]["total"] >= 90`

---

## 8. The Cyclic DAG Orchestrator

### 8A. Global `seen_dict` — Design & Data Structure

**`seen_list` (local per-entity list) has been replaced by `seen_dict` (global nested dictionary).** `seen_dict` is initialized once before the outer loop and never cleared — it is the single source of truth for all 3 output files.

```python
seen_dict["S1-00001"] = {
    "accepted"  : set(),    # Confirmed matches → matching_results.tsv
    "rejected"  : set(),    # ML-rejected candidates (blacklisted from next cycle)
    "seen_ids"  : set(),    # Union of accepted + rejected (O(1) blacklist lookup)
    "total"     : int,      # len(seen_ids) — used for k = 20 + total
    "per_cycle" : dict,     # { cycle_num (int): [list of candidate_ids (str)] }
}
```

### 8B. Dual Candidate Output Files

| File | Rows per S1 Entity | Written When | Source in `seen_dict` |
|---|---|---|---|
| `candidate_pairs.tsv` | **Exactly 1** | After ALL entities processed | `accepted` ∪ `rejected` |
| `candidate_pairs_per_cycle.tsv` | **One per cycle** | After ALL entities processed | `per_cycle[1]`, `per_cycle[2]`, ... |
| `matching_results.tsv` | **Exactly 1** | After ALL entities processed | `accepted` only |

### 8C. Full Orchestrator Pseudocode (3 Streams)

```python
def run_orchestrator(mode, query_c, target_c,
                     bm25_name_index, bm25_addr_index, faiss_index,
                     s1_name_embs, model, threshold):

    seen_dict = {}

    for s1_entity in query_c.iter_rows(named=True):
        s1_id = s1_entity["entity_id"]
        init_entity(seen_dict, s1_id)

        cycle  = 1
        active = True
        new_matches_this_cycle = 0

        # Pre-compute query representations once per entity
        s1_name_3grams = to_char_3grams(s1_entity["name_for_bm25"])
        s1_addr_3grams = to_char_3grams(s1_entity["addr_for_bm25"])
        s1_name_emb    = s1_name_embs[s1_entity["row_index"]]

        while active:
            k = 20 + seen_dict[s1_id]["total"]

            # ── RETRIEVE (3 independent streams) ────────────────────────────────────
            bm25_name_results = bm25_name_index.get_top_n(s1_name_3grams, target_ids, n=k)
            bm25_addr_results = bm25_addr_index.get_top_n(s1_addr_3grams, target_ids, n=k)
            faiss_scores, faiss_indices = faiss_index.search(s1_name_emb.reshape(1, -1), k)
            faiss_results = format_faiss_results(faiss_scores, faiss_indices, target_ids)

            # ── FILTER BLACKLIST ──────────────────────────
            blacklist = seen_dict[s1_id]["seen_ids"]
            new_bm25_name = [x for x in bm25_name_results if x["id"] not in blacklist][:20]
            new_bm25_addr = [x for x in bm25_addr_results if x["id"] not in blacklist][:20]
            new_faiss     = [x for x in faiss_results     if x["id"] not in blacklist][:20]

            # ── UNION & PRIORITY SORT ────────────────────────────────────────────────
            # Priority: in all 3 streams > in 2 streams > in 1 stream only
            union_candidates = priority_deduplicate(new_bm25_name, new_bm25_addr, new_faiss)

            if len(union_candidates) == 0:
                active = False
                break

            # ── FEATURE EXTRACTION & ML SCORING ─────────────────────────────────────
            features = extract_features(s1_entity, union_candidates)
            new_matches_this_cycle = 0

            if mode == "train":
                store_training_pairs(s1_id, union_candidates, features)
                for cand in union_candidates:
                    update_seen(seen_dict, s1_id, cand["id"], accepted=False, cycle=cycle)
            else:
                probs = model.predict_proba(features)[:, 1]
                for cand, prob in zip(union_candidates, probs):
                    is_match = prob > threshold
                    update_seen(seen_dict, s1_id, cand["id"], accepted=is_match, cycle=cycle)
                    if is_match:
                        new_matches_this_cycle += 1

            # ── STOPPING CONDITION ────────────────────────────────────────────────────
            total_seen = seen_dict[s1_id]["total"]
            if total_seen >= 90 or (cycle >= 3 and new_matches_this_cycle == 0):
                active = False
            else:
                cycle += 1

    # Single pass output writer
    write_all_outputs(seen_dict, mode)
```

---

## 9. Phase 3: Feature Engineering (18-Dim Vector)

| Dim | Feature | Method | Stream Source |
|---|---|---|---|
| 1 | Name Jaro-Winkler | `jellyfish.jaro_winkler_similarity` | — |
| 2 | Name Monge-Elkan | `textdistance.MongeElkan` | — |
| 3 | Name Levenshtein (normalized) | `jellyfish.levenshtein_distance / max_len` | — |
| 4 | Name Phonetic Match | `pyphonetics` Double Metaphone binary flag | — |
| 5 | Acronym Score | Jaro-Winkler(initials(longer), shorter) | — |
| 6 | Name Semantic Cosine | MiniLM embedding dot product | FAISS-Name |
| 7 | Exact Name Match | Binary flag | — |
| 8 | Addr Token Jaccard | Word-level overlap | — |
| 9 | Addr Numeric Jaccard | Extracted number set overlap | — |
| 10 | Addr Numeric Exact | Binary: sets identical? | — |
| 11 | **Name BM25 Score (normalized)** | `name_bm25_score / max_name_bm25_for_query` | BM25-Name |
| 12 | **Addr BM25 Score (normalized)** | `addr_bm25_score / max_addr_bm25_for_query` | BM25-Addr |
| 13 | Exact Address Match | Binary flag | — |
| 14 | Name Length Ratio | `min(len_a, len_b) / max(len_a, len_b)` | — |
| 15 | Addr Length Ratio | Same formula | — |
| 16 | **RRF Score (3-stream)** | `1/(60+bm25_name_rank) + 1/(60+bm25_addr_rank) + 1/(60+faiss_rank)` | All 3 |
| 17 | **Stream Overlap Count** | `0/1/2/3` — how many streams retrieved this candidate | All 3 |
| Meta | **Cross-Script Target?** | `is_cross_script` binary flag from preprocessing | — |
| Meta | Source Origin | One-hot: S2=0, S3=1 | — |
| Meta | Country Target Encoded | Learned mean-encoding per country | — |

---

## 10. Phase 4: ML Classification

- **Model:** XGBoost (primary) / LightGBM (fallback)
- **Class Imbalance:** `scale_pos_weight = count(negatives) / count(positives)`
- **Monotonic Constraints:** All similarity features forced monotonically increasing
- **Validation:** GroupKFold (k=5) grouped by `source1_entity_id`
- **Threshold Calibration:** Grid search `[0.5 → 0.99, step=0.01]` on Macro F_0.5

> [!WARNING]
> **1:1 Bipartite Greedy Post-Processing has been REMOVED.** The problem statement allows S1 → many S2/S3 matches and does not constrain S2/S3 exclusivity. Applying 1:1 assignment would incorrectly drop valid true matches.

---

## 11. Development Steps (9 Steps)

```
Step 0:  EDA (notebooks/eda.ipynb) [COMPLETED - Verified cross-script necessity]

Step 1:  Initialize project structure + requirements.txt + config.py + run.sh.

Step 2:  Write data_layer:
         - loader.py (Polars TSV + null imputation + fake-null detection)
         - cleaner.py (Transliteration via indic-transliteration → dual column generation)

Step 3:  Write blocking_layer:
         - lexical_name_index.py (BM25 on name char-3grams)
         - lexical_addr_index.py (BM25 on address char-3grams)
         - semantic_index.py (FAISS + MiniLM on name ONLY, CUDA)
         - seen_dict.py (global dict: init_entity(), update_seen(), write_all_outputs())

Step 4:  Write feature_layer:
         - feature_extractor.py (all 18 features + meta-features)

Step 5:  Write ml_layer:
         - trainer.py (GroupKFold + monotonic constraints + F_0.5 threshold search)
         - classifier.py (load model + predict_proba + threshold apply)

Step 6:  Write pipeline/orchestrator.py
         - Cyclic DAG (global seen_dict, single-pass output writer, mode=train|infer)

Step 7:  Run TRAIN MODE end-to-end:
         bash run.sh --mode train ...
         → Generates labeled training pairs → trains XGBoost → saves model + threshold.

Step 8:  Run INFER MODE end-to-end:
         bash run.sh --mode infer ...
         → Generates output/matching_results.tsv and output/candidate_pairs.tsv

Step 9:  Validate & Submit:
         python3 utils/validate_submission.py \
             --matching output/matching_results.tsv \
             --candidate output/candidate_pairs.tsv \
             --test-dir dataset/test
         → Must print PASS before any leaderboard upload.
```
