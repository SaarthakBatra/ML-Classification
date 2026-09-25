# Phase 4: Machine Learning Classification Analysis

In Phase 3, we engineered a dense, ~15-dimensional numerical feature vector for every candidate pair (e.g., `S1-001` vs `S2-999`). 

In **Phase 4**, the goal is to feed these vectors into a Machine Learning model that outputs the probability that the pair is a true match. Finally, we must calibrate this probability to maximize the challenge's specific evaluation metric (**F_0.5**).

---

## 1. Model Selection (State of the Art for Tabular ER)

When dealing with Entity Resolution, you generally have two paths: passing raw text into Deep Learning (Transformers) or passing engineered numerical features into Tree-based models. 

**The Industrial Standard:** Because we extracted highly complex, non-linear numerical features in Phase 3 (Jaro-Winkler, Jaccard overlaps, Semantic Cosines), **Gradient Boosting Machines (GBMs) like XGBoost or LightGBM** are the undisputed state-of-the-art.

### Why LightGBM / XGBoost over Deep Learning?
*   **Tabular Dominance:** Research consistently shows that GBMs outperform Deep Learning on structured, tabular datasets.
*   **Non-Linear Feature Interactions:** XGBoost easily learns rules like: *"If Name Jaro-Winkler > 0.9 BUT Numeric Address Jaccard == 0.0, then probability = 0%"*.
*   **Monotonic Constraints (Overfitting Prevention):** We will hardcode constraints (`monotone_constraints = (1, 1, 1...)`) for all similarity features, forcing the model to never decrease match probability when a similarity score increases. This mathematically eliminates overfitting to non-logical training outliers.
*   **Missing Values:** XGBoost natively handles missing data (e.g., if a business has no address, the Numeric Jaccard feature is `NaN`. XGBoost automatically learns which way to split `NaN`s).
*   **Computational Scale:** We will be training on millions of candidate pairs. LightGBM can train on millions of rows in minutes on a standard CPU, whereas Deep Learning would require massive GPU clustering and days of training.

---

## 2. Handling the "Needle in a Haystack" (Class Imbalance)

In our Candidate Generation (Phase 2), we retrieve the Top-20 matches from S2 and S3 for every S1 entity. 
*   If an S1 entity has 2 true matches in reality, our dataset for that entity has **2 Positives (Label 1)** and **18 Negatives (Label 0)**. 
*   Across 2.2 million entities, the dataset is massively imbalanced (~90% Label 0).

### The Bulletproof Training Strategy:
1.  **Scale Pos Weight:** In XGBoost/LightGBM, we must set the `scale_pos_weight` parameter (usually `count(Negative) / count(Positive)`). This forces the model to penalize itself much harder when it misclassifies a true match.
2.  **Hard Negative Mining:** The 18 negative pairs aren't just random businesses; they are the *closest* non-matches retrieved by BM25/FAISS. Training the model on these "Hard Negatives" is highly effective. It forces the model to learn the microscopic differences between a true match (`Apple Inc`) and a very similar false match (`Apple Store`).

---

## 3. Cross-Validation (Preventing Data Leakage)

If we do a standard `train_test_split`, we might put Candidate `(S1-001, S2-005)` in the train set and `(S1-001, S3-009)` in the validation set. This causes **Data Leakage** because the model has already "memorized" aspects of `S1-001`.

*   **The Standard:** Use **GroupKFold** cross-validation, grouping by `source1_entity_id`. This guarantees that an S1 entity and *all* of its candidates are either entirely in the training set or entirely in the validation set.

---

## 4. Offline Training & Threshold Optimization
*This phase occurs entirely offline on the `train.csv` dataset.*

The challenge evaluates using **F_0.5**, which weighs Precision twice as heavily as Recall. False Positives are punished severely.
1.  **Train:** Train the model using GroupKFold and Monotonic Constraints.
2.  **Calibrate:** Run inference on the out-of-fold validation set. Iterate through possible thresholds (e.g., `0.50` to `0.99`). Calculate the Macro-Averaged F_0.5 score for each.
3.  **Result:** Extract the optimal threshold (e.g., `> 0.88`). We only predict a match if the model is absolutely certain. Save the pre-trained model and optimal threshold to disk.

---

## 5. Online Inference & Greedy Bipartite Resolution
*This phase is integrated directly inside the `Blocking_1.0.md` Cyclic DAG for the test set.*

1.  **Online Scoring:** Inside the cycle loop, the pre-trained model scores the batch of `<= 40` unseen candidates.
2.  **Global Probability Tracking:** All candidates that pass the Optimal Threshold are appended to a global `potential_matches` cache with their exact probability score.
3.  **Greedy 1:1 Bipartite Resolution (Post-Processing):** After all S1 entities have finished all their cycles, we resolve conflicts to maximize Precision:
    *   Sort the entire `potential_matches` list globally by `Probability` (Descending).
    *   Iterate through the list. If an `S2` or `S3` entity has already been assigned to an `S1` entity, **drop it** (preventing an S2 entity from being matched to multiple S1 entities).
    *   This Greedy 1:1 Assignment guarantees no bipartite conflicts and eliminates massive False Positives.
4.  **Final Output:** `groupby('source1_entity_id')` on the resolved list, and save to `matching_results.tsv`. If an S1 entity has 0 resolved matches, output an empty string.

---

## Summary of Phase 4
By using **LightGBM/XGBoost** paired with **GroupKFold** validation, training on **Hard Negatives**, and performing an **Exhaustive Threshold Search for F_0.5**, this classification pipeline is robust, highly interpretable, and computationally efficient enough to run locally without exceeding memory limits.
