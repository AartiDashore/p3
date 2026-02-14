"""Hybrid search combining BM25 keyword search with semantic search using RRF."""

from __future__ import annotations

import logging
from typing import Any

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class HybridSearcher:
    """
    Hybrid search combining BM25 keyword search with semantic search.

    Uses Reciprocal Rank Fusion (RRF) to combine rankings from both approaches:
    - BM25: Captures exact keyword matches and term frequency
    - Semantic: Understands meaning and intent
    - RRF: Merges both rankings to leverage strengths of each

    Args:
        k (int): RRF parameter controlling rank contribution (default: 60)

    Attributes:
        bm25_index (BM25Okapi | None): BM25 index for keyword search
        documents (list[dict] | None): Stored documents for retrieval
        k (int): RRF parameter
    """

    def __init__(self, k: int = 60) -> None:
        """Initialize hybrid searcher."""
        self.bm25_index: BM25Okapi | None = None
        self.documents: list[dict] | None = None
        self.k = k
        logger.info("HybridSearcher initialized with RRF parameter k=%d", k)

    def index_documents(self, documents: list[dict]) -> None:
        """
        Build BM25 index from documents.

        Args:
            documents (list[dict]): Documents with 'id', 'text', and 'metadata' fields
        """
        if not documents:
            logger.warning("No documents to index")
            self.bm25_index = None
            self.documents = None
            return

        self.documents = documents

        # Tokenize documents for BM25 (simple whitespace tokenization)
        tokenized_docs = [doc["text"].lower().split() for doc in documents]

        # Build BM25 index
        self.bm25_index = BM25Okapi(tokenized_docs)
        logger.info("Indexed %d documents in BM25", len(documents))

    def search_bm25(self, query: str, n_results: int = 20) -> list[dict]:
        """
        Search using BM25 keyword matching.

        Args:
            query (str): Search query
            n_results (int): Number of results to return

        Returns:
            list[dict]: Documents ranked by BM25 score
        """
        if self.bm25_index is None or self.documents is None:
            logger.warning("BM25 index not built. Call index_documents() first.")
            return []

        # Tokenize query
        tokenized_query = query.lower().split()

        # Get BM25 scores for all documents
        scores = self.bm25_index.get_scores(tokenized_query)

        # Get top N document indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            :n_results
        ]

        # Return documents with BM25 scores
        results = []
        for idx in top_indices:
            doc = self.documents[idx].copy()
            doc["bm25_score"] = float(scores[idx])
            results.append(doc)

        return results

    def reciprocal_rank_fusion(
        self,
        semantic_results: list[dict],
        bm25_results: list[dict],
        n_results: int = 5,
    ) -> list[dict]:
        """
        Combine semantic and BM25 results using Reciprocal Rank Fusion.

        RRF formula: RRF_score(doc) = Σ 1 / (k + rank_in_list)

        Args:
            semantic_results (list[dict]): Results from semantic search
            bm25_results (list[dict]): Results from BM25 search
            n_results (int): Number of final results to return

        Returns:
            list[dict]: Fused results with 'rrf_score' field
        """
        # Build RRF scores dictionary
        rrf_scores: dict[str, float] = {}
        doc_map: dict[str, dict] = {}

        # Add semantic search contributions
        for rank, doc in enumerate(semantic_results, start=1):
            doc_id = doc["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (self.k + rank)
            if doc_id not in doc_map:
                doc_map[doc_id] = doc.copy()

        # Add BM25 contributions
        for rank, doc in enumerate(bm25_results, start=1):
            doc_id = doc["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (self.k + rank)
            if doc_id not in doc_map:
                doc_map[doc_id] = doc.copy()
            elif "bm25_score" in doc:
                # Add BM25 score to existing doc
                doc_map[doc_id]["bm25_score"] = doc["bm25_score"]

        # Sort by RRF score and return top N
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        top_ids = sorted_ids[:n_results]

        results = []
        for doc_id in top_ids:
            doc = doc_map[doc_id]
            doc["rrf_score"] = rrf_scores[doc_id]
            results.append(doc)

        logger.info(
            "RRF fusion combined %d semantic + %d BM25 results → %d final results",
            len(semantic_results),
            len(bm25_results),
            len(results),
        )

        return results


def explain_rrf_example() -> None:
    """
    Print an example explaining how RRF works.

    This is for educational purposes and not used in production code.
    """
    print("\n" + "=" * 70)
    print("Reciprocal Rank Fusion (RRF) Explanation")
    print("=" * 70)
    print("\nGiven two ranked lists:")
    print("\nSemantic Search Results:")
    print("  1. doc_A")
    print("  2. doc_B")
    print("  3. doc_C")
    print("\nBM25 Keyword Search Results:")
    print("  1. doc_B")
    print("  2. doc_D")
    print("  3. doc_A")
    print("\nRRF Score Calculation (k=60):")
    print("-" * 70)
    print("doc_A: 1/(60+1) + 1/(60+3) = 0.0164 + 0.0159 = 0.0323")
    print("doc_B: 1/(60+2) + 1/(60+1) = 0.0161 + 0.0164 = 0.0325  ← HIGHEST")
    print("doc_C: 1/(60+3) = 0.0159")
    print("doc_D: 1/(60+2) = 0.0161")
    print("\nFinal Ranking:")
    print("  1. doc_B (appears high in both lists)")
    print("  2. doc_A (appears in both lists)")
    print("  3. doc_D (only in BM25)")
    print("  4. doc_C (only in semantic)")
    print("\nKey Insight: Documents appearing in both lists get boosted!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    # Run explanation
    explain_rrf_example()