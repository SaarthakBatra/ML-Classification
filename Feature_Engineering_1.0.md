# Phase 3: Feature Engineering Analysis (Bulletproof Edition)

In Phase 2 (Blocking 1.0), we generated a manageable list of candidate pairs. In **Phase 3 (Feature Engineering)**, the goal is to transform every candidate pair into a numerical **Feature Vector** so the final XGBoost/LightGBM model can make a highly informed binary decision (Match vs. Non-Match).

After reviewing state-of-the-art frameworks (like **Magellan** from UW-Madison, **DeepMatcher**, and **PyJedAI**), this feature space has been expanded to be completely bulletproof, incorporating advanced phonetic, lexical, and semantic measures.

---

## 1. Name-Based Features (Lexical, Semantic, & Phonetic)
Business names suffer heavily from abbreviations, acronyms, and phonetic misspellings.

### A. String Distance Metrics (The Magellan Standard)
Magellan advocates for generating a cross-product of string distances.
*   **Jaro-Winkler Similarity:** Heavily weights prefix matches (e.g., `Amazon Inc` vs `Amazon Retail`).
*   **Normalized Levenshtein (Edit) Distance:** The standard character-edit distance.
*   **Monge-Elkan Distance:** An industrial standard for multi-word strings. It finds the best character-level match for each word in String A against any word in String B. It perfectly handles word-reordering combined with typos (e.g., `Batra Retail Inc` vs `Retail Bata`).

### B. Phonetic Encoding (The Industrial Standard)
Humans often misspell business names based on how they sound. Lexical distances fail here, but phonetic algorithms excel.
*   **Double Metaphone / Soundex:** Convert both names to their phonetic hash (e.g., `Kmart` -> `KMRT`, `Caymart` -> `KMRT`). 
*   **Feature:** A binary flag (`1` or `0`) indicating if the phonetic hashes match perfectly, or the Levenshtein distance between the two phonetic hashes.

### C. Acronym Match Score (Continuous)
*   Instead of a simple binary acronym match, we extract the initials of the longer string and calculate the **Jaro-Winkler distance** between those initials and the shorter string.
*   *Example:* `IBM` vs `International Business Machines India`. Initials = `IBMI`. Jaro-Winkler between `IBM` and `IBMI` yields a high continuous score (e.g., `0.93`), allowing the ML model to learn partial acronym matches rather than a strict 1/0 cutoff.

### D. Semantic Embedding Score
*   **Bi-Encoder Cosine Similarity:** Reuse the dense vector cosine similarity calculated in Phase 2. This single continuous feature tells the model if the names mean the same thing, regardless of spelling or phonetics.

---

## 2. Address-Based Features (Lexical & Numeric)
Addresses are highly noisy and suffer from missing components. Semantic embeddings are poor here; we rely entirely on character and numeric extraction.

### A. The Numeric Overlap Feature (Country-Agnostic)
We cannot hardcode Zip Code or PIN code regexes because French postcodes follow different rules.
*   **The Approach:** Use a generic Regex to extract all contiguous numbers from Address A and Address B into sets.
    *   *Example A:* `1795 Westchester Drive, 27262` -> `{1795, 27262}`
    *   *Example B:* `Westchester Dr 1795, 27262` -> `{1795, 27262}`
*   **Numeric Jaccard Similarity:** (Intersection over Union of the extracted number sets).
*   **Numeric Exact Match:** Binary `1` if the sets are identical, `0` if not, `-1` if no numbers exist in one of the strings. 

### B. Structural Address Overlap
*   **Token Jaccard Similarity:** Word-level overlap. Highly robust to reordered address components (e.g., `123 Main St, NY` vs `NY, 123 Main St`).
*   **Normalized BM25 Score:** The raw retrieval score from Phase 2 BM25 search must be **Normalized per query** (e.g., `Score / Max_Score_in_Query`). Raw BM25 scores are unbounded and fluctuate massively based on the country index size. Normalizing bounds it between 0.0 and 1.0, preventing the ML model from overfitting to country sizes.
*   **Exact Address Match:** A binary `1` or `0` flag if the normalized addresses are a perfect 1:1 identical match.

---

## 3. Structural Differentials (Length & Missing Data)
Sometimes the length of the string tells a story (e.g., one source is just an acronym).

*   **Name Length Ratio:** `min(len(A), len(B)) / max(len(A), len(B))`
*   **Address Length Ratio:** Useful to flag when one address is just a city name while the other is a full postal address.
*   **Missing Value Flags:** Binary features (`1` or `0`) indicating if the address is completely missing (null) in either source.

---

## 4. Meta-Features
Contextual information about the candidate generation process.

*   **Source Origin (One-Hot Encoded):** Is the candidate from Source 2 or Source 3? XGBoost can learn if Source 2 is generally noisier than Source 3 and adjust its internal weights automatically.
*   **Reciprocal Rank Fusion (RRF) Score:** Since candidates are retrieved from both BM25 and FAISS streams, a single rank is ambiguous. We calculate the combined RRF score: `(1 / (60 + BM25_Rank)) + (1 / (60 + FAISS_Rank))`. If a candidate is missing from a stream, its rank penalty is infinity (score = 0 for that half). This provides a single, elegant metric representing the combined search engine confidence.
*   **Country Label Target Encoder:** Feeding the country code as a categorical feature allows the tree model to learn country-specific matching thresholds (e.g., French addresses might naturally have lower Jaro-Winkler scores on average than US addresses).
*   **Exact Name Match:** A binary `1` or `0` flag if the normalized names are a perfect 1:1 identical match. (Note: Instead of trying to calculate a single composite "Name Match Score", we provide this flag alongside the individual Jaro and Monge-Elkan scores. XGBoost is mathematically designed to find the optimal composite weighting internally).

---

## Summary of the Feature Vector

For every candidate pair fed into XGBoost, the model will see a rich 15+ dimensional vector looking roughly like this:
`[ Name_Jaro: 0.95, Name_MongeElkan: 0.99, Name_PhoneticMatch: 1, Name_AcronymMatch: 0, Name_Semantic: 0.98, Addr_Jaccard: 0.70, Num_Jaccard: 1.0, Rank: 1, Source: 2, ... ]`

By feeding these granular, engineered metrics to the model, XGBoost can easily learn the optimal decision boundaries required to maximize the F_0.5 metric, eliminating the need for complex, fragile rule-based cascades.
