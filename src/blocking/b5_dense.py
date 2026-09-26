"""
Channel B5: Dense Multilingual Embedding Retrieval Channel.

Zero-shot cross-lingual & semantic retrieval using SentenceTransformers + FAISS.
Sharded by country for ultra-fast, strictly partitioned intra-country vector search.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import polars as pl
import faiss
import torch
from sentence_transformers import SentenceTransformer
from blocking.base import BaseIndexer


_MODEL_CACHE: Dict[Tuple[str, str], SentenceTransformer] = {}


class DenseEmbeddingRetriever(BaseIndexer):
    """
    Multilingual Dense Bi-Encoder Retriever.
    Indexes target entity text representations using SentenceTransformers
    and searches them via FAISS Inner-Product (cosine similarity on L2-normalized vectors).
    """

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        batch_size: int = 256,
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.batch_size = batch_size

        if device is None or device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        cache_key = (self.model_name, self.device)
        if cache_key not in _MODEL_CACHE:
            _MODEL_CACHE[cache_key] = SentenceTransformer(self.model_name, device=self.device)
        self.model = _MODEL_CACHE[cache_key]

        self.indices: Dict[str, faiss.IndexFlatIP] = {}
        self.country_target_ids: Dict[str, List[str]] = {}

    def fit(
        self,
        target_df: pl.DataFrame,
        text_col: str = "embed_combined"
    ) -> "DenseEmbeddingRetriever":
        """
        Encodes target records and constructs per-country FAISS indices.
        """
        unique_countries = target_df["country"].unique().to_list()

        for country in unique_countries:
            c_df = target_df.filter(pl.col("country") == country)
            entity_ids = c_df["entity_id"].to_list()
            raw_texts = c_df[text_col].fill_null("").to_list() if text_col in c_df.columns else []
            texts = [str(t).strip() if (t and str(t).strip()) else "empty" for t in raw_texts]

            if not texts:
                continue

            # Encode in batches with L2 normalization for exact cosine similarity
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True
            )
            embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

            dim = embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(embeddings)

            self.indices[country] = index
            self.country_target_ids[country] = entity_ids

        return self

    def retrieve(
        self,
        query_df: pl.DataFrame,
        max_cands_per_entity: int = 15,
        text_col: str = "embed_combined",
        top_k: Optional[int] = None,
        min_similarity: float = 0.50
    ) -> Dict[str, List[str]]:
        """
        Encodes query records and queries country FAISS indices.
        Returns mapping: query_entity_id -> list of candidate target_entity_ids.
        """
        k = top_k if top_k is not None else max_cands_per_entity
        results: Dict[str, List[str]] = {eid: [] for eid in query_df["entity_id"].to_list()}
        unique_countries = query_df["country"].unique().to_list()

        for country in unique_countries:
            if country not in self.indices:
                continue

            c_df = query_df.filter(pl.col("country") == country)
            s1_ids = c_df["entity_id"].to_list()
            raw_texts = c_df[text_col].fill_null("").to_list() if text_col in c_df.columns else []
            texts = [str(t).strip() if (t and str(t).strip()) else "empty" for t in raw_texts]

            if not texts:
                continue

            query_embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True
            )
            query_embeddings = np.ascontiguousarray(query_embeddings, dtype=np.float32)

            index = self.indices[country]
            target_ids = self.country_target_ids[country]

            # Query FAISS
            scores, indices = index.search(query_embeddings, k)


            for i, s1_id in enumerate(s1_ids):
                row_indices = indices[i]
                row_scores = scores[i]
                cands = []
                for idx, score in zip(row_indices, row_scores):
                    if idx >= 0 and idx < len(target_ids) and score >= min_similarity:
                        cands.append(target_ids[idx])
                results[s1_id] = cands

        return results
