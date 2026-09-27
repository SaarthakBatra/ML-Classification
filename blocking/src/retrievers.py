from typing import Dict, List, Protocol

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer


class TextEncoder(Protocol):
    def encode(self, texts: List[str]) -> np.ndarray: ...


class CharacterBM25Retriever:
    def __init__(self, text_column: str, ngram_size: int = 3, k1: float = 1.5, b: float = 0.75) -> None:
        self.text_column = text_column
        self.ngram_size = ngram_size
        self.k1 = k1
        self.b = b
        self.vectorizer: CountVectorizer | None = None
        self.target_ids: List[str] = []
        self.target_weights: csr_matrix | None = None
        self.idf: np.ndarray | None = None
        self.query_texts: Dict[str, str] = {}

    def fit(self, target_df: pl.DataFrame) -> "CharacterBM25Retriever":
        self.target_ids = target_df["entity_id"].cast(pl.Utf8).to_list()
        texts = self._texts(target_df, self.text_column)
        if not texts:
            return self
        vectorizer = CountVectorizer(analyzer="char_wb", ngram_range=(self.ngram_size, self.ngram_size))
        try:
            counts = vectorizer.fit_transform(texts).tocsr()
        except ValueError:
            return self
        document_lengths = np.asarray(counts.sum(axis=1)).ravel().astype(np.float32)
        average_length = float(document_lengths.mean()) or 1.0
        document_frequency = np.bincount(counts.indices, minlength=counts.shape[1])
        idf = np.log((len(texts) - document_frequency + 0.5) / (document_frequency + 0.5) + 1.0).astype(np.float32)
        rows = np.repeat(np.arange(counts.shape[0]), np.diff(counts.indptr))
        normalizer = self.k1 * (1.0 - self.b + self.b * document_lengths / average_length)
        weighted_data = counts.data.astype(np.float32) * (self.k1 + 1.0) / (counts.data + normalizer[rows])
        self.vectorizer = vectorizer
        self.target_weights = csr_matrix((weighted_data, counts.indices, counts.indptr), shape=counts.shape)
        self.idf = idf
        return self

    def bind_queries(self, query_df: pl.DataFrame) -> "CharacterBM25Retriever":
        self.query_texts = dict(zip(query_df["entity_id"].cast(pl.Utf8).to_list(), self._texts(query_df, self.text_column)))
        return self

    def retrieve(self, query_id: str, count: int) -> List[str]:
        if count < 1 or self.vectorizer is None or self.target_weights is None or self.idf is None:
            return []
        query_text = self.query_texts.get(query_id, "")
        query_counts = self.vectorizer.transform([query_text]).tocsr()
        if query_counts.nnz == 0:
            return []
        weighted_query = query_counts.multiply(self.idf).T
        scores = self.target_weights.dot(weighted_query).tocoo()
        if scores.nnz == 0:
            return []
        top = np.argsort(-scores.data)[:count]
        return [self.target_ids[int(scores.row[index])] for index in top if scores.data[index] > 0]

    def _texts(self, frame: pl.DataFrame, column: str) -> List[str]:
        if column not in frame.columns:
            return [""] * len(frame)
        return [str(value).strip() for value in frame[column].fill_null("").to_list()]


class BGEM3Encoder:
    def __init__(self, model_name: str, batch_size: int, device: str) -> None:
        from FlagEmbedding import BGEM3FlagModel

        self.batch_size = batch_size
        options = {"use_fp16": device == "cuda"}
        if device == "cuda":
            options["devices"] = ["cuda:0"]
        self.model = BGEM3FlagModel(model_name, **options)

    def encode(self, texts: List[str]) -> np.ndarray:
        return np.ascontiguousarray(
            self.model.encode(
                texts,
                batch_size=self.batch_size,
                max_length=256,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )["dense_vecs"],
            dtype=np.float32,
        )


class FaissSemanticRetriever:
    def __init__(self, encoder: TextEncoder, text_column: str = "name_for_faiss") -> None:
        self.encoder = encoder
        self.text_column = text_column
        self.target_ids: List[str] = []
        self.query_texts: Dict[str, str] = {}
        self.query_embeddings: Dict[str, np.ndarray] = {}
        self.index = None

    def fit(self, target_df: pl.DataFrame) -> "FaissSemanticRetriever":
        import faiss

        self.target_ids = target_df["entity_id"].cast(pl.Utf8).to_list()
        texts = self._texts(target_df, self.text_column)
        if not texts:
            return self
        embeddings = self.encoder.encode(texts)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        return self

    def bind_queries(self, query_df: pl.DataFrame) -> "FaissSemanticRetriever":
        query_ids = query_df["entity_id"].cast(pl.Utf8).to_list()
        texts = self._texts(query_df, self.text_column)
        self.query_texts = dict(zip(query_ids, texts))
        if texts:
            embeddings = self.encoder.encode(texts)
            self.query_embeddings = {
                query_id: embeddings[index]
                for index, query_id in enumerate(query_ids)
            }
        return self

    def retrieve(self, query_id: str, count: int) -> List[str]:
        if count < 1 or self.index is None:
            return []
        embedding = self.query_embeddings.get(query_id)
        if embedding is None:
            return []
        _, indices = self.index.search(embedding.reshape(1, -1), min(count, len(self.target_ids)))
        return [self.target_ids[index] for index in indices[0] if index >= 0]

    def _texts(self, frame: pl.DataFrame, column: str) -> List[str]:
        if column not in frame.columns:
            return [""] * len(frame)
        return [str(value).strip() for value in frame[column].fill_null("").to_list()]
