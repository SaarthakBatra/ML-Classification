import csv
import json
import os
from dataclasses import dataclass
from typing import Dict, List

import polars as pl

from cyclic import CandidateRecord, CyclicCandidateGenerator, GenerationResult
from retrievers import BGEM3Encoder, CharacterBM25Retriever, FaissSemanticRetriever


@dataclass(frozen=True)
class BlockingWorkflowConfig:
    preprocessed_dir: str
    output_dir: str
    model_name: str = "BAAI/bge-m3"
    batch_size: int = 256
    device: str = "auto"
    per_stream_batch_size: int = 20
    max_seen_candidates: int = 90
    min_cycles_before_empty_stop: int = 3


def run_blocking_workflow(config: BlockingWorkflowConfig) -> Dict[str, object]:
    manifest = _load_manifest(config.preprocessed_dir)
    device = _resolve_device(config.device)
    os.makedirs(config.output_dir, exist_ok=True)

    all_records: List[CandidateRecord] = []
    all_matches: Dict[str, List[str]] = {}
    country_summary: Dict[str, Dict[str, int]] = {}
    encoder = BGEM3Encoder(config.model_name, config.batch_size, device)

    for country, partition in manifest["partitions"].items():
        query_path = _partition_file(partition, f"Query_{country}_parquet")
        target_path = _partition_file(partition, f"Target_{country}_parquet")
        query_frame = pl.read_parquet(query_path)
        target_frame = pl.read_parquet(target_path)
        query_ids = query_frame["entity_id"].cast(pl.Utf8).to_list()

        if not len(target_frame):
            result = GenerationResult(
                records_by_query={query_id: [] for query_id in query_ids},
                matched_ids_by_query={query_id: set() for query_id in query_ids},
                cycle_counts={query_id: 0 for query_id in query_ids},
            )
        else:
            lexical = CharacterBM25Retriever("addr_for_bm25").fit(target_frame).bind_queries(query_frame)
            names = CharacterBM25Retriever("name_for_bm25").fit(target_frame).bind_queries(query_frame)
            semantic = FaissSemanticRetriever(encoder).fit(target_frame).bind_queries(query_frame)
            generator = CyclicCandidateGenerator(
                lexical_retriever=lexical,
                semantic_retriever=semantic,
                name_retriever=names,
                per_stream_batch_size=config.per_stream_batch_size,
                max_seen_candidates=config.max_seen_candidates,
                min_cycles_before_empty_stop=config.min_cycles_before_empty_stop,
            )
            result = generator.generate(query_ids, evaluator=_unscored_evaluator)

        country_records = [record for records in result.records_by_query.values() for record in records]
        all_records.extend(country_records)
        all_matches.update({query_id: sorted(matches) for query_id, matches in result.matched_ids_by_query.items()})
        country_summary[country] = {
            "query_count": len(query_ids),
            "target_count": len(target_frame),
            "candidate_count": len(country_records),
        }

    candidate_path = os.path.join(config.output_dir, "candidate_pairs.tsv")
    cycle_path = os.path.join(config.output_dir, "candidate_pairs_per_cycle.tsv")
    match_path = os.path.join(config.output_dir, "matching_results.tsv")
    _write_candidate_pairs(candidate_path, all_records, all_matches)
    _write_cycle_candidates(cycle_path, all_records)
    _write_matches(match_path, all_matches)
    summary = {
        "workflow": "Blocking 1.0 three-stream cyclic retrieval",
        "preprocessed_dir": os.path.abspath(config.preprocessed_dir),
        "candidate_pairs_path": candidate_path,
        "candidate_pairs_per_cycle_path": cycle_path,
        "matching_results_path": match_path,
        "classifier_status": "unscored; no feature-engineering or classifier stage is implemented",
        "countries": country_summary,
        "total_candidates": len(all_records),
    }
    with open(os.path.join(config.output_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def _load_manifest(preprocessed_dir: str) -> Dict[str, object]:
    manifest_path = os.path.join(preprocessed_dir, "manifest.json")
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest.get("partitions"), dict):
        raise ValueError(f"Preprocessing manifest has no partitions mapping: {manifest_path}")
    return manifest


def _partition_file(partition: Dict[str, object], key: str) -> str:
    files = partition.get("files")
    if not isinstance(files, dict) or not isinstance(files.get(key), str):
        raise ValueError(f"Preprocessing manifest is missing {key}")
    return files[key]


def _resolve_device(requested_device: str) -> str:
    if requested_device != "auto":
        return requested_device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _unscored_evaluator(_: str, __: List[CandidateRecord]) -> List[str]:
    return []


def _write_candidate_pairs(path: str, records: List[CandidateRecord], query_matches: Dict[str, List[str]]) -> None:
    candidates_by_query: Dict[str, List[str]] = {}
    for record in records:
        candidates_by_query.setdefault(record.query_id, []).append(record.candidate_id)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        for query_id in query_matches:
            writer.writerow([query_id, ",".join(sorted(set(candidates_by_query.get(query_id, []))))])


def _write_cycle_candidates(path: str, records: List[CandidateRecord]) -> None:
    candidates_by_cycle: Dict[tuple[str, int], List[str]] = {}
    for record in records:
        candidates_by_cycle.setdefault((record.query_id, record.cycle), []).append(record.candidate_id)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["source1_entity_id", "cycle_number", "candidate_entity_ids"])
        for (query_id, cycle), candidate_ids in candidates_by_cycle.items():
            writer.writerow([query_id, cycle, ",".join(candidate_ids)])


def _write_matches(path: str, matched_ids_by_query: Dict[str, List[str]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["source1_entity_id", "matched_entity_ids"])
        for query_id, candidate_ids in matched_ids_by_query.items():
            writer.writerow([query_id, ",".join(candidate_ids)])
