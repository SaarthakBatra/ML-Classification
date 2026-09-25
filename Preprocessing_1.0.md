# Phase 0: Preprocessing & Partitioning (Polars Engine)

In Phase 0, our primary goal is to load the raw massive datasets (`train_source1.tsv`, etc.), aggressively clean and standardize the noisy text, and partition the datasets by country to prepare for the Phase 1 Blocking phase.

By utilizing **Polars** instead of Pandas, we ensure that these memory-intensive string operations execute natively in Rust across all available CPU threads, guaranteeing speed and preventing Out-Of-Memory (OOM) crashes.

---

## 1. High-Speed Data Loading & Imputation
*   **Engine:** `polars.read_csv(separator='\t')`
*   **Literal Null Parsing:** Before applying standard null-fills, we must explicitly map common fake-null string literals (`["n/a", "na", "null", "none", "-"]`) to `""`.
*   **Missing Values:** Addresses and business names often contain nulls. We will immediately fill nulls with empty strings `""` to prevent `NoneType` errors during string manipulation.
*   **Data Types:** `country` will be cast to a categorical type to save memory.

## 2. Text Normalization Pipeline
Both `business_name` and `business_address` undergo a strict normalization pipeline. This is critical because `IBM` and `i.b.m.` must be mathematically identical before they hit the blocking index.

### A. Case & Whitespace
*   **Unicode Normalization:** Apply `NFKD` encoding to strip diacritics and accents (e.g., `café` -> `cafe`), which is critical for the unseen French test set.
*   Lowercasing all text.
*   Trimming leading and trailing whitespaces.
*   Collapsing multiple internal spaces into a single space.

### B. Punctuation Stripping
*   *Acronym Protection:* Remove periods `.` with no space replacement (e.g., `I.B.M.` -> `IBM`) to prevent acronyms from being splintered into separate characters.
*   *At Symbol Mapping:* Convert `@` to the word ` at ` before stripping, to preserve intent (e.g., `Coffee @ Paris`).
*   *Ampersand Mapping:* Convert `&` to the word `and` before stripping, as "A & B" and "A and B" are highly common variations.
*   Removing all remaining special characters except alphanumeric characters and spaces.

### C. Abbreviation Expansion (Dictionary Mapping)
To assist the lexical BM25 index, we must standardize common legal suffixes and street terms.
*   **Ordinal Numbers:** Map words to numbers (`first` -> `1st`, `second` -> `2nd`, `third` -> `3rd`) to align disparate address formats.
*   **Business Suffixes:** `corp` -> `corporation`, `inc` -> `incorporated`, `ltd` -> `limited`, `pvt` -> `private`, `co` -> `company`, `llc` -> `limited liability company`.
*   **Address Terms:** `st` -> `street`, `rd` -> `road`, `ave` -> `avenue`, `blvd` -> `boulevard`, `apt` -> `apartment`.

*(Note: Polars handles this via ultra-fast `.str.replace_all()` using regex).*

### D. Empty String Safety Fallback
*   If, after aggressive cleaning, a `business_name` or `business_address` is reduced to an empty string `""`, it will crash downstream matrices.
*   We will apply a final fallback, replacing `""` with a placeholder like `"UNKNOWN_NAME"` or `"UNKNOWN_ADDRESS"`.

## 3. Country-Wise Partitioning
As established in our pipeline rules, an entity in India will never match an entity in the US.
*   **Country Normalization:** Aggressively clean the `country` column itself (lowercase, strip whitespace, and map variations like `usa` -> `us`) to prevent accidental dataset splintering.
*   We partition S1, S2, and S3 datasets into strict country-specific subsets.
*   *Example:* `S1_US`, `S1_IN`, `S1_FR`.
*   The pipeline will process these partitions completely independently in a loop, clearing memory between countries.

## 4. Target Union (S2 + S3)
To feed our Blocking engine, we do not want to search S2 and S3 separately (which would require managing two separate FAISS/BM25 indices per country).
*   **Unioning:** We will vertically concatenate (union) `S2_Country` and `S3_Country` into a single, unified target dataset: `Target_Country`.
*   **Tracking:** We will strictly preserve the original `entity_id` without any modifications. Instead, we will append a new explicit column named `source` containing either `"S2"` or `"S3"`. This allows the downstream ML model to seamlessly identify the origin source without fragile string parsing.

---

## Summary of Phase 0 Outputs
At the end of this phase, for any given country (e.g., `US`), we yield exactly two ultra-clean, memory-optimized Polars DataFrames:
1.  **`Query_US`:** The cleaned S1 entities.
2.  **`Target_US`:** The combined, cleaned S2 and S3 entities.

These are seamlessly passed into Phase 1 (FAISS & BM25 Index Building).
