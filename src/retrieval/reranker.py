"""Cross-encoder reranking for improved document retrieval."""

from __future__ import annotations

import logging

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Reranking search results using a cross-encoder model.

    Cross-encoders encode query and document together, allowing the model
    to consider their interaction. This is more accurate than bi-encoders
    but slower, so it's used as a second stage after fast bi-encoder retrieval.

    Args:
        model_name (str): Hugging Face cross-encoder model name.
            Defaults to "cross-encoder/ms-marco-MiniLM-L-6-v2".

    Attributes:
        model (CrossEncoder): Loaded cross-encoder model
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        """Initialize the cross-encoder model."""
        self.model_name = model_name
        logger.info(f"Loading cross-encoder model: {model_name}")
        self.model = CrossEncoder(model_name)
        logger.info("Cross-encoder model loaded successfully")

    def rerank(self, query: str, documents: list[dict], top_k: int = 5) -> list[dict]:
        """
        Reranking documents based on query-document relevance scores.

        Args:
            query (str): The search query
            documents (list[dict]): List of document dicts with 'text' field
            top_k (int): Number of top documents to return (default: 5)

        Returns:
            list[dict]: Top-k documents with added 'rerank_score' field,
                       sorted by relevance (highest score first)
        """
        if not documents:
            return []

        if not query or not query.strip():
            return documents[:top_k]

        # Preparing query-document pairs for cross-encoder
        pairs = [(query, doc["text"]) for doc in documents]

        # Getting the relevance scores from cross-encoder
        scores = self.model.predict(pairs)

        # Adding the rerank scores to the documents
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)

        # Sorting by rerank score (highest first) and then returning the top-k
        reranked = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)

        return reranked[:top_k]