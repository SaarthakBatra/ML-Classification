import os
import sys
import json
import tempfile
import unittest

import numpy as np
import polars as pl


WORKFLOW_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(WORKFLOW_DIR, "src")
sys.path.insert(0, SRC_DIR)

from cyclic import CyclicCandidateGenerator
from retrievers import CharacterBM25Retriever, FaissSemanticRetriever
import workflow


class RankedStream:
    def __init__(self, rankings):
        self.rankings = rankings
        self.requests = []

    def retrieve(self, query_id, count):
        self.requests.append((query_id, count))
        return self.rankings[query_id][:count]


class CountingEncoder:
    def __init__(self):
        self.calls = []

    def encode(self, texts):
        self.calls.append(list(texts))
        return np.asarray([[1.0, 0.0] if "alpha" in text else [0.0, 1.0] for text in texts], dtype=np.float32)


class FixedRetriever:
    def __init__(self, *args):
        self.target_ids = []

    def fit(self, target_frame):
        self.target_ids = target_frame["entity_id"].to_list()
        return self

    def bind_queries(self, query_frame):
        return self

    def retrieve(self, query_id, count):
        return self.target_ids[:count]


class TestCyclicCandidateGenerator(unittest.TestCase):
    def test_character_bm25_returns_ranked_address_candidates(self):
        targets = pl.DataFrame(
            {
                "entity_id": ["S2-1", "S2-2", "S2-3"],
                "clean_address": ["10 rue de la paix paris", "200 market street san francisco", "11 rue victor hugo paris"],
            }
        )
        queries = pl.DataFrame({"entity_id": ["S1-1"], "clean_address": ["10 rue de la paix paris"]})

        retriever = CharacterBM25Retriever().fit(targets).bind_queries(queries)

        self.assertEqual(retriever.retrieve("S1-1", 2)[0], "S2-1")

    def test_semantic_queries_are_encoded_once_before_cyclic_retrieval(self):
        targets = pl.DataFrame({"entity_id": ["S2-1", "S2-2"], "clean_name": ["alpha cafe", "beta cafe"]})
        queries = pl.DataFrame({"entity_id": ["S1-1"], "clean_name": ["alpha coffee"]})
        encoder = CountingEncoder()

        retriever = FaissSemanticRetriever(encoder).fit(targets).bind_queries(queries)
        retriever.retrieve("S1-1", 2)
        retriever.retrieve("S1-1", 2)

        self.assertEqual(encoder.calls, [["alpha cafe", "beta cafe"], ["alpha coffee"]])

    def test_workflow_consumes_preprocessed_manifest_and_writes_candidates(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            preprocessed_dir = os.path.join(temporary_directory, "preprocessed")
            output_dir = os.path.join(temporary_directory, "output")
            os.mkdir(preprocessed_dir)
            query_path = os.path.join(preprocessed_dir, "Query_US.parquet")
            target_path = os.path.join(preprocessed_dir, "Target_US.parquet")
            pl.DataFrame({"entity_id": ["S1-1"], "clean_name": ["alpha"], "clean_address": ["10 main street"]}).write_parquet(query_path)
            pl.DataFrame({"entity_id": ["S2-1"], "clean_name": ["alpha"], "clean_address": ["10 main street"]}).write_parquet(target_path)
            with open(os.path.join(preprocessed_dir, "manifest.json"), "w", encoding="utf-8") as handle:
                json.dump({"partitions": {"US": {"files": {"Query_US_parquet": query_path, "Target_US_parquet": target_path}}}}, handle)

            original_encoder = workflow.SentenceTransformerEncoder
            original_lexical = workflow.CharacterBM25Retriever
            original_semantic = workflow.FaissSemanticRetriever
            workflow.SentenceTransformerEncoder = lambda *args: object()
            workflow.CharacterBM25Retriever = FixedRetriever
            workflow.FaissSemanticRetriever = FixedRetriever
            self.addCleanup(setattr, workflow, "SentenceTransformerEncoder", original_encoder)
            self.addCleanup(setattr, workflow, "CharacterBM25Retriever", original_lexical)
            self.addCleanup(setattr, workflow, "FaissSemanticRetriever", original_semantic)

            summary = workflow.run_blocking_workflow(
                workflow.BlockingWorkflowConfig(preprocessed_dir=preprocessed_dir, output_dir=output_dir, device="cpu")
            )

            self.assertEqual(summary["total_candidates"], 1)
            with open(os.path.join(output_dir, "candidate_pairs.tsv"), encoding="utf-8") as handle:
                self.assertEqual(len(handle.readlines()), 2)

    def test_blacklists_prior_cycles_and_orders_shared_candidates_first(self):
        lexical = RankedStream({"S1-1": ["S2-a", "S2-b", "S2-c", "S2-d", "S2-e", "S2-f"]})
        semantic = RankedStream({"S1-1": ["S2-b", "S2-e", "S2-g", "S2-a", "S2-h", "S2-i"]})
        generator = CyclicCandidateGenerator(lexical, semantic, per_stream_batch_size=2)

        result = generator.generate(["S1-1"], evaluator=lambda _, candidates: set())

        records = result.records_by_query["S1-1"]
        self.assertEqual([record.candidate_id for record in records[:3]], ["S2-b", "S2-a", "S2-e"])
        self.assertEqual(len({record.candidate_id for record in records}), len(records))
        self.assertEqual([record.cycle for record in records], [1, 1, 1, 2, 2, 2, 2, 3, 3])
        self.assertEqual(lexical.requests, [("S1-1", 2), ("S1-1", 5), ("S1-1", 9)])
        self.assertEqual(semantic.requests, [("S1-1", 2), ("S1-1", 5), ("S1-1", 9)])

    def test_stops_after_third_empty_match_cycle(self):
        lexical = RankedStream({"S1-1": [f"S2-{index}" for index in range(20)]})
        semantic = RankedStream({"S1-1": [f"S3-{index}" for index in range(20)]})
        generator = CyclicCandidateGenerator(lexical, semantic, per_stream_batch_size=2)

        result = generator.generate(["S1-1"], evaluator=lambda _, candidates: set())

        self.assertEqual(result.cycle_counts["S1-1"], 3)
        self.assertEqual(len(result.records_by_query["S1-1"]), 12)
        self.assertEqual(lexical.requests, [("S1-1", 2), ("S1-1", 6), ("S1-1", 10)])

    def test_stops_immediately_when_both_indices_are_exhausted(self):
        lexical = RankedStream({"S1-1": []})
        semantic = RankedStream({"S1-1": []})
        generator = CyclicCandidateGenerator(lexical, semantic, per_stream_batch_size=2)

        result = generator.generate(["S1-1"], evaluator=lambda _, candidates: set())

        self.assertEqual(result.records_by_query["S1-1"], [])
        self.assertEqual(result.cycle_counts["S1-1"], 1)


if __name__ == "__main__":
    unittest.main()
