"""
rag_rerank.py — Lab 8 Module 8: Hybrid reranking after vector search.

Combines semantic similarity, TF-IDF keyword score, and term overlap
into a single rerank score for top-k clinical protocols.
"""

import math
import re
from collections import Counter
from typing import Dict, List, Tuple

from rag_config import (
    RERANK_WEIGHT_SEMANTIC,
    RERANK_WEIGHT_TFIDF,
    RERANK_WEIGHT_OVERLAP,
)


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _corpus_text(doc: dict) -> str:
    parts = [
        doc.get("title_safe") or doc.get("title", ""),
        doc.get("content_safe") or doc.get("content", ""),
        " ".join(doc.get("keywords", [])),
    ]
    return " ".join(parts)


class HybridReranker:
    """TF-IDF index over the knowledge base for hybrid reranking."""

    def __init__(self, documents: List[dict]):
        self.documents = documents
        self.doc_tokens = [tokenize(_corpus_text(d)) for d in documents]
        self.idf = self._compute_idf(self.doc_tokens)
        self.doc_vectors = [self._tfidf_vector(tokens) for tokens in self.doc_tokens]

    @staticmethod
    def _compute_idf(all_tokens: List[List[str]]) -> Dict[str, float]:
        n = len(all_tokens)
        df: Counter = Counter()
        for tokens in all_tokens:
            for term in set(tokens):
                df[term] += 1
        return {
            term: math.log((n + 1) / (count + 1)) + 1.0
            for term, count in df.items()
        }

    def _tfidf_vector(self, tokens: List[str]) -> Dict[str, float]:
        tf = Counter(tokens)
        vec: Dict[str, float] = {}
        for term, count in tf.items():
            if term in self.idf:
                vec[term] = count * self.idf[term]
        return vec

    @staticmethod
    def _cosine_sparse(a: Dict[str, float], b: Dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        common = set(a) & set(b)
        dot = sum(a[t] * b[t] for t in common)
        mag_a = math.sqrt(sum(v * v for v in a.values()))
        mag_b = math.sqrt(sum(v * v for v in b.values()))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

    def tfidf_score(self, query: str, doc_index: int) -> float:
        q_vec = self._tfidf_vector(tokenize(query))
        return self._cosine_sparse(q_vec, self.doc_vectors[doc_index])

    @staticmethod
    def term_overlap_score(query: str, doc: dict) -> float:
        q_terms = set(tokenize(query))
        if not q_terms:
            return 0.0
        d_terms = set(tokenize(_corpus_text(doc)))
        overlap = len(q_terms & d_terms)
        return overlap / len(q_terms)

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[float, dict, int]],
        top_k: int,
    ) -> List[Tuple[float, dict, dict]]:
        """
        Rerank vector-search candidates.

        Args:
            candidates: list of (semantic_score, doc, doc_index)
        Returns:
            list of (rerank_score, doc, score_breakdown dict)
        """
        if not candidates:
            return []

        raw_semantic = [c[0] for c in candidates]
        raw_tfidf = [self.tfidf_score(query, c[2]) for c in candidates]
        raw_overlap = [self.term_overlap_score(query, c[1]) for c in candidates]

        def normalize(values: List[float]) -> List[float]:
            lo, hi = min(values), max(values)
            if hi - lo < 1e-9:
                return [1.0] * len(values) if hi > 0 else [0.0] * len(values)
            return [(v - lo) / (hi - lo) for v in values]

        norm_sem = normalize(raw_semantic)
        norm_tfidf = normalize(raw_tfidf)
        norm_overlap = normalize(raw_overlap)

        reranked: List[Tuple[float, dict, dict]] = []
        for i, (sem, doc, doc_idx) in enumerate(candidates):
            breakdown = {
                "semantic": round(raw_semantic[i], 4),
                "semantic_norm": round(norm_sem[i], 4),
                "tfidf": round(raw_tfidf[i], 4),
                "tfidf_norm": round(norm_tfidf[i], 4),
                "term_overlap": round(raw_overlap[i], 4),
                "overlap_norm": round(norm_overlap[i], 4),
            }
            score = (
                RERANK_WEIGHT_SEMANTIC * norm_sem[i]
                + RERANK_WEIGHT_TFIDF * norm_tfidf[i]
                + RERANK_WEIGHT_OVERLAP * norm_overlap[i]
            )
            breakdown["rerank_score"] = round(score, 4)
            reranked.append((score, doc, breakdown))

        reranked.sort(key=lambda x: x[0], reverse=True)
        return reranked[:top_k]
