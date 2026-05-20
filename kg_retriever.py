"""Knowledge graph retrieval using NetworkX."""
from __future__ import annotations

import logging
from collections import Counter

import jieba
import networkx as nx

import config
from kg_store import load_triples

logger = logging.getLogger(__name__)


def build_graph(triples: list[dict]) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    for triple in triples:
        head = triple.get("head")
        tail = triple.get("tail")
        relation = triple.get("relation")
        if not head or not tail or not relation:
            continue
        graph.add_edge(
            head,
            tail,
            relation=relation,
            chunk_id=triple.get("chunk_id"),
            source=triple.get("source"),
        )
    return graph


class KGRetriever:
    def __init__(self):
        self._graph = nx.MultiDiGraph()
        self._entities: set[str] = set()
        self.refresh()

    def refresh(self):
        triples = load_triples()
        self._graph = build_graph(triples)
        self._entities = set(self._graph.nodes)
        logger.info("KG loaded: %d entities, %d edges", self._graph.number_of_nodes(), self._graph.number_of_edges())

    def _match_entities(self, query: str) -> list[str]:
        if not self._entities:
            return []
        lowered = query.lower()

        # Match when entity appears in query OR query appears in entity (support shorter queries)
        matched = [e for e in self._entities if (e.lower() in lowered or lowered in e.lower()) and len(e) > 1]
        if matched:
            return matched

        # Token-based match (jieba) - use intersection on normalized tokens for better recall
        tokens = [t.lower() for t in jieba.cut(query) if len(t) > 1]
        token_set = set(tokens)
        results = []
        for e in self._entities:
            # consider entity tokens too
            ent_tokens = set(t.lower() for t in jieba.cut(e) if len(t) > 1)
            if ent_tokens & token_set:
                results.append(e)

        return results

    def retrieve_chunk_ids(self, query: str, top_k: int | None = None) -> list[str]:
        top_k = top_k or config.KG_TOP_K
        entities = self._match_entities(query)
        if not entities:
            return []

        counter: Counter[str] = Counter()
        for entity in entities:
            for _, _, data in self._graph.edges(entity, data=True):
                chunk_id = data.get("chunk_id")
                if chunk_id:
                    counter[chunk_id] += 1

        if not counter:
            return []

        return [chunk_id for chunk_id, _ in counter.most_common(top_k)]
