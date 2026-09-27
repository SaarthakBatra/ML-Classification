from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Protocol, Set


class RankedRetriever(Protocol):
    def retrieve(self, query_id: str, count: int) -> List[str]: ...


@dataclass(frozen=True)
class CandidateRecord:
    query_id: str
    candidate_id: str
    cycle: int
    lexical_rank: int | None
    semantic_rank: int | None
    name_rank: int | None


@dataclass(frozen=True)
class GenerationResult:
    records_by_query: Dict[str, List[CandidateRecord]]
    matched_ids_by_query: Dict[str, Set[str]]
    cycle_counts: Dict[str, int]


Evaluator = Callable[[str, List[CandidateRecord]], Iterable[str]]


class CyclicCandidateGenerator:
    def __init__(
        self,
        lexical_retriever: RankedRetriever,
        semantic_retriever: RankedRetriever,
        name_retriever: RankedRetriever | None = None,
        per_stream_batch_size: int = 20,
        max_seen_candidates: int = 60,
        min_cycles_before_empty_stop: int = 3,
    ) -> None:
        if per_stream_batch_size < 1:
            raise ValueError("per_stream_batch_size must be positive")
        if max_seen_candidates < 1:
            raise ValueError("max_seen_candidates must be positive")
        if min_cycles_before_empty_stop < 1:
            raise ValueError("min_cycles_before_empty_stop must be positive")
        self.lexical_retriever = lexical_retriever
        self.semantic_retriever = semantic_retriever
        self.name_retriever = name_retriever
        self.per_stream_batch_size = per_stream_batch_size
        self.max_seen_candidates = max_seen_candidates
        self.min_cycles_before_empty_stop = min_cycles_before_empty_stop

    def generate(
        self,
        query_ids: Iterable[str],
        evaluator: Evaluator,
    ) -> GenerationResult:
        records_by_query: Dict[str, List[CandidateRecord]] = {}
        matched_ids_by_query: Dict[str, Set[str]] = {}
        cycle_counts: Dict[str, int] = {}

        for query_id in query_ids:
            seen_ids: Set[str] = set()
            matched_ids: Set[str] = set()
            records: List[CandidateRecord] = []
            cycle = 1

            while True:
                retrieve_count = self.per_stream_batch_size + len(seen_ids)
                lexical = self._take_unseen(
                    self.lexical_retriever.retrieve(query_id, retrieve_count), seen_ids
                )
                semantic = self._take_unseen(
                    self.semantic_retriever.retrieve(query_id, retrieve_count), seen_ids
                )
                names = self._take_unseen(
                    self.name_retriever.retrieve(query_id, retrieve_count), seen_ids
                ) if self.name_retriever else []
                cycle_records = self._merge(query_id, cycle, lexical, semantic, names)

                remaining = self.max_seen_candidates - len(seen_ids)
                cycle_records = cycle_records[:remaining]

                if not cycle_records:
                    break

                records.extend(cycle_records)
                seen_ids.update(record.candidate_id for record in cycle_records)
                accepted_ids = set(evaluator(query_id, cycle_records))
                unknown_ids = accepted_ids.difference(record.candidate_id for record in cycle_records)
                if unknown_ids:
                    raise ValueError(f"Evaluator returned candidates not retrieved for {query_id}: {sorted(unknown_ids)}")
                matched_ids.update(accepted_ids)

                if len(seen_ids) >= self.max_seen_candidates:
                    break
                if cycle >= self.min_cycles_before_empty_stop and not accepted_ids:
                    break
                cycle += 1

            records_by_query[query_id] = records
            matched_ids_by_query[query_id] = matched_ids
            cycle_counts[query_id] = cycle

        return GenerationResult(records_by_query, matched_ids_by_query, cycle_counts)

    def _take_unseen(self, candidate_ids: List[str], seen_ids: Set[str]) -> List[str]:
        selected: List[str] = []
        selected_ids: Set[str] = set()
        for candidate_id in candidate_ids:
            if candidate_id in seen_ids or candidate_id in selected_ids:
                continue
            selected.append(candidate_id)
            selected_ids.add(candidate_id)
            if len(selected) == self.per_stream_batch_size:
                break
        return selected

    def _merge(
        self,
        query_id: str,
        cycle: int,
        lexical: List[str],
        semantic: List[str],
        names: List[str],
    ) -> List[CandidateRecord]:
        lexical_ranks = {candidate_id: rank for rank, candidate_id in enumerate(lexical, 1)}
        semantic_ranks = {candidate_id: rank for rank, candidate_id in enumerate(semantic, 1)}
        name_ranks = {candidate_id: rank for rank, candidate_id in enumerate(names, 1)}
        ordered = sorted(
            set(lexical_ranks) | set(semantic_ranks) | set(name_ranks),
            key=lambda candidate_id: (
                -(candidate_id in lexical_ranks) - (candidate_id in semantic_ranks) - (candidate_id in name_ranks),
                min(rank for rank in (lexical_ranks.get(candidate_id), semantic_ranks.get(candidate_id), name_ranks.get(candidate_id)) if rank is not None),
                semantic_ranks.get(candidate_id, float("inf")),
                lexical_ranks.get(candidate_id, float("inf")),
                name_ranks.get(candidate_id, float("inf")),
                candidate_id,
            ),
        )
        return [
            CandidateRecord(
                query_id=query_id,
                candidate_id=candidate_id,
                cycle=cycle,
                lexical_rank=lexical_ranks.get(candidate_id),
                semantic_rank=semantic_ranks.get(candidate_id),
                name_rank=name_ranks.get(candidate_id),
            )
            for candidate_id in ordered
        ]
