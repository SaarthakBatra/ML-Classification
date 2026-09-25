# Implementation Plan: Amazon ML Challenge ER Pipeline

This document outlines the technical execution plan for building the Progressive Cyclic DAG Entity Resolution pipeline we designed. 

## 1. Goal Description
The objective is to implement a robust, scalable Python codebase that takes raw `train_source1.tsv`, `train_source2.tsv`, and `train_source3.tsv` files, executes dynamic country-wise blocking using FAISS/BM25, extracts advanced pairwise features (Jaro-Winkler, Phonetics, Sentence-Transformers), and uses an XGBoost classifier within a Cyclic DAG to output perfectly formatted matches.

> [!IMPORTANT]
> **User Review Required:** Please review the proposed directory structure and Python dependencies below. Once approved, I will begin writing the actual code.

## 2. Proposed Project Structure
I propose organizing the Python code into modular layers (following the strict `modules/<module_name>/` scoping rule). 

```text
Amazon ML Challange/
├── src/
│   ├── main.py                     # The orchestrator script
│   ├── config.py                   # Hyperparameters (Thresholds, Paths)
│   ├── data_layer/
│   │   ├── loader.py               # Preprocessing and Country Partitioning (Phase 0)
│   │   └── cleaner.py              # Basic text normalization
│   ├── blocking_layer/
│   │   ├── semantic_index.py       # FAISS + Sentence Transformers (Phase 1/2)
│   │   └── lexical_index.py        # BM25 + TF-IDF (Phase 1/2)
│   ├── feature_layer/
│   │   ├── feature_extractor.py    # Jaro, Monge-Elkan, Phonetics (Phase 3)
│   ├── ml_layer/
│   │   ├── classifier.py           # XGBoost training and inference (Phase 4)
│   └── pipeline/
│       └── cyclic_dag.py           # The loop logic (Retrieval -> Filter -> ML -> Stop)
├── notebooks/                      # For EDA and manual threshold F_0.5 testing
├── requirements.txt
└── run.sh
```

## 3. Python Dependencies
We will create a `requirements.txt` with the following highly optimized libraries:
*   `pandas`, `numpy`: Data manipulation
*   `scikit-learn`: TF-IDF, Cross-Validation (GroupKFold)
*   `faiss-cpu`: High-speed vector similarity search
*   `rank_bm25`: Fast lexical inverted indexing
*   `sentence-transformers`: For `paraphrase-multilingual-MiniLM-L12-v2`
*   `xgboost`: State-of-the-art tabular classification
*   `jellyfish`, `pyphonetics`, `textdistance`: Extremely fast string distance metrics (Jaro-Winkler, Soundex).

## 4. Architectural Decisions & Recommendations

**Decision 1: Data Processing Engine (Polars)**
There is no strict rule enforcing Pandas. Therefore, we will use **Polars**. Polars is written in Rust, natively multi-threaded, and handles large-scale string operations and dataset joins vastly faster than Pandas while using significantly less RAM. This will prevent out-of-memory crashes on the 2.2M+ row datasets.

**Decision 2: Execution Environment (Local Processing with CUDA)**
Your **NVIDIA RTX 3060 (6GB VRAM)** is actually perfect for this! 
*   The embedding model we chose (`paraphrase-multilingual-MiniLM-L12-v2`) is incredibly lightweight (only ~470MB in memory). 
*   It will easily fit into your 6GB VRAM, allowing us to run inference with a high batch size (e.g., 256 or 512). 
*   What would take days on a CPU will only take about 15–30 minutes locally on your RTX 3060. We will set up the pipeline to use `torch` with `cuda` device mapping so everything stays completely local!

## 5. Project DAG & Cycle Breakdown

### 5.1 High-Level Data Flow & Cycle Logic
This flowchart illustrates the overarching architecture, showing exactly what data is generated, where it is cached, and how the engines interact within the cycle.

```mermaid
graph TD
    %% Global Data Stores
    D1[("Target Database (S2 + S3)")]
    D2[("Query Database (S1)")]
    D3[("Seen List Cache (Per S1 Entity)")]
    D4[("Output File (matching_results.tsv)")]
    
    %% Core Engines
    E1["1. Blocking Engine (FAISS & BM25)"]
    E2["2. Feature Extraction Engine"]
    E3["3. ML Classification Engine (XGBoost)"]
    
    %% Flow
    D2 -->|S1 Query| E1
    D1 -->|Target Index Search| E1
    D3 -.->|Blacklist Filter| E1
    
    E1 -->|Top 20 Unseen Candidates| E2
    E2 -->|15-Dim Feature Vectors| E3
    E3 -->|Probabilities| R{Threshold Met?}
    
    R -->|Yes| D4
    R -->|Yes/No| D3
    
    %% Cycle Logic
    R --> C{"Stop Metric Reached?<br>(len >= 60 OR 0 New Matches in Cycle 3)"}
    C -->|No: Increment Cycle| E1
    C -->|Yes: Terminate Cycle| Next[Proceed to Next S1 Query]
```

### 5.2 Detailed Execution Architecture
This flowchart breaks down the internal mechanics of each engine and the exact stepwise logic of the DAG.

```mermaid
graph TD
    %% Phase 0
    A[Raw TSV Files] --> B(Phase 0: Preprocessing & Country Partitioning)
    B --> C{For Each Country C}
    
    %% Indices
    C --> D[Target_C: Source 2 + Source 3]
    C --> E[Query_C: Source 1]
    D --> F[Build FAISS Index Semantic]
    D --> G[Build BM25 Index Lexical]
    
    %% The Loop
    E --> H{For Each Query Entity S1}
    H --> I[Initialize: seen_list = empty, cycle = 1, active = True]
    I --> J{While active == True}
    
    %% Retrieval
    J -->|Yes| K["Set k = 20 + len(seen_list)"]
    F -.-> K
    G -.-> K
    K --> L[Retrieve Top-k from FAISS & BM25]
    L --> M[Post-Filter: Remove IDs already in seen_list]
    M --> N[Take Exact Top 20 Unseen Candidates]
    
    %% Evaluation
    N --> O[Phase 3: Feature Extraction]
    O --> O1[Lexical: Jaro-Winkler, Monge-Elkan]
    O --> O2[Semantic: MiniLM Cosine Similarity]
    O --> O3[Phonetic: Soundex Hash Match]
    O1 --> P[Phase 4: XGBoost Classification]
    O2 --> P
    O3 --> P
    
    %% Routing
    P --> Q{XGBoost Probability > Threshold?}
    Q -->|Yes| R[Mark Matched, Add to seen_list]
    Q -->|No| S[Mark Rejected, Add to seen_list]
    
    %% Stopping Metric
    R --> T{"len(seen_list) >= 60 OR (Cycle >= 3 AND New Matches == 0)?"}
    S --> T
    T -->|Yes| U[Set active = False, Yield Final Matches]
    T -->|No| V[cycle += 1, Loop Back]
    
    V --> J
    U --> W[Output: matching_results.tsv]
```

## 6. Development Steps
1.  Initialize the project directory and create `requirements.txt`.
2.  Write the `data_layer` (Partitioning by country).
3.  Write the `blocking_layer` (Building FAISS and BM25 indices).
4.  Write the `feature_layer` (The 15-dimensional vector extraction).
5.  Write the `ml_layer` (XGBoost logic and F_0.5 optimizer).
6.  Construct the `cyclic_dag.py` orchestrator to tie them all together.
