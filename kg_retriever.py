"""Knowledge graph retrieval using NetworkX."""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import re

import jieba
import numpy as np
import networkx as nx

import config
from embedder import ZhipuEmbedder
from kg_store import load_triples

logger = logging.getLogger(__name__)


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


@dataclass
class _EntityVectorIndex:
    entities: list[str]
    embeddings: np.ndarray

    @classmethod
    def build(cls, entities: list[str], embedder: ZhipuEmbedder) -> "_EntityVectorIndex | None":
        if not entities:
            return None
        try:
            embeddings = np.asarray(embedder.encode(entities, normalize_embeddings=True), dtype=np.float32)
        except Exception as exc:
            logger.warning("KG entity vector index build failed: %s", exc)
            return None
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        return cls(entities=entities, embeddings=embeddings)

    def search(self, query: str, embedder: ZhipuEmbedder, top_k: int, min_sim: float) -> list[str]:
        try:
            query_embedding = np.asarray(embedder.encode(query, normalize_embeddings=True), dtype=np.float32)
        except Exception as exc:
            logger.warning("KG entity vector query embedding failed: %s", exc)
            return []

        if query_embedding.ndim != 1:
            query_embedding = np.asarray(query_embedding).reshape(-1)

        scores = self.embeddings @ query_embedding
        if scores.size == 0:
            return []

        top_indices = np.argsort(scores)[::-1][:top_k]
        results: list[str] = []
        for idx in top_indices:
            if scores[idx] < min_sim:
                continue
            results.append(self.entities[int(idx)])
        return results


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
        self._entity_list: list[str] = []
        self._entity_vector_index: _EntityVectorIndex | None = None
        self._embedder: ZhipuEmbedder | None = None
        self.refresh()

    def refresh(self):
        triples = load_triples()
        self._graph = build_graph(triples)
        self._entities = set(self._graph.nodes)
        self._entity_list = sorted(self._entities)
        self._entity_vector_index = None
        logger.info("KG loaded: %d entities, %d edges", self._graph.number_of_nodes(), self._graph.number_of_edges())

    def _get_embedder(self) -> ZhipuEmbedder:
        if self._embedder is None:
            self._embedder = ZhipuEmbedder()
        return self._embedder

    def _get_entity_vector_index(self) -> _EntityVectorIndex | None:
        if self._entity_vector_index is not None:
            return self._entity_vector_index
        if not self._entity_list:
            return None
        self._entity_vector_index = _EntityVectorIndex.build(self._entity_list, self._get_embedder())
        return self._entity_vector_index

    def _match_entities(self, query: str) -> list[str]:
        if not self._entities:
            return []
        lowered = _normalize_text(query)

        # Match when entity appears in query OR query appears in entity (support shorter queries)
        matched = [e for e in self._entities if ( _normalize_text(e) in lowered or lowered in _normalize_text(e)) and len(e) > 1]
        if matched:
            # Exact/token matches first, then entity vectors as recall booster.
            exact_set = set(matched)
        else:
            exact_set = set()

        # Token-based match (jieba) - use intersection on normalized tokens for better recall
        tokens = [t.lower() for t in jieba.cut(query) if len(t) > 1]
        token_set = set(tokens)
        results = []
        for e in self._entities:
            # consider entity tokens too
            ent_tokens = set(t.lower() for t in jieba.cut(e) if len(t) > 1)
            if ent_tokens & token_set:
                results.append(e)

        combined = []
        seen = set()
        for entity in [*matched, *results]:
            if entity not in seen:
                combined.append(entity)
                seen.add(entity)

        vector_index = self._get_entity_vector_index()
        if vector_index is not None:
            vector_matches = vector_index.search(
                query,
                self._get_embedder(),
                top_k=config.KG_ENTITY_VECTOR_TOP_K,
                min_sim=config.KG_ENTITY_VECTOR_MIN_SIM,
            )
            for entity in vector_matches:
                if entity not in seen:
                    combined.append(entity)
                    seen.add(entity)

        return combined if combined else list(exact_set)

    def retrieve_subgraph(self, query: str, hops: int | None = None, top_k: int | None = None) -> tuple[list[str], list[dict]]:
        """
        根据查询实体进行多跳子图遍历。
        返回 (关联的 chunk_ids 列表, 子图的三元组列表)。
        """
        top_k = top_k or config.KG_TOP_K
        hops = hops or getattr(config, "KG_MAX_HOPS", 2)
        max_triples = getattr(config, "KG_MAX_SUBGRAPH_TRIPLES", 30)

        entities = self._match_entities(query)
        if not entities:
            return [], []

        visited_entities: set[str] = set()
        current_entities = set(entities)
        
        counter: Counter[str] = Counter()
        subgraph_triples: list[dict] = []
        seen_triples = set()

        for _ in range(hops):
            if not current_entities or len(subgraph_triples) >= max_triples:
                break
                
            next_entities = set()
            for entity in current_entities:
                if len(subgraph_triples) >= max_triples:
                    break
                    
                # Out-edges
                for _, tail, data in self._graph.edges(entity, data=True):
                    if len(subgraph_triples) >= max_triples:
                        break
                    chunk_id = data.get("chunk_id")
                    if chunk_id:
                        counter[chunk_id] += 1
                        
                    triple_tuple = (entity, data.get("relation"), tail)
                    if triple_tuple not in seen_triples:
                        seen_triples.add(triple_tuple)
                        subgraph_triples.append({
                            "head": entity,
                            "relation": data.get("relation"),
                            "tail": tail,
                            "source": data.get("source"),
                            "chunk_id": chunk_id,
                        })
                    next_entities.add(tail)
                    
                # In-edges
                for head, _, data in self._graph.in_edges(entity, data=True):
                    if len(subgraph_triples) >= max_triples:
                        break
                    chunk_id = data.get("chunk_id")
                    if chunk_id:
                        counter[chunk_id] += 1
                        
                    triple_tuple = (head, data.get("relation"), entity)
                    if triple_tuple not in seen_triples:
                        seen_triples.add(triple_tuple)
                        subgraph_triples.append({
                            "head": head,
                            "relation": data.get("relation"),
                            "tail": entity,
                            "source": data.get("source"),
                            "chunk_id": chunk_id,
                        })
                    next_entities.add(head)
            
            visited_entities.update(current_entities)
            current_entities = next_entities - visited_entities

        chunk_ids = [chunk_id for chunk_id, _ in counter.most_common(top_k)]
        return chunk_ids, subgraph_triples
