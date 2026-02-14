"""Unit tests for CrossEncoderReranker."""

from __future__ import annotations

import pytest

from retrieval.reranker import CrossEncoderReranker


@pytest.fixture(scope="module")
def reranker():
    """Create a reranker instance for testing."""
    return CrossEncoderReranker()


@pytest.fixture
def sample_documents():
    """Sample documents for testing reranking."""
    return [
        {
            "id": "doc1",
            "text": "Python is a high-level programming language",
            "distance": 0.5,
        },
        {
            "id": "doc2",
            "text": "Machine learning uses neural networks",
            "distance": 0.6,
        },
        {
            "id": "doc3",
            "text": "Python programming for data science",
            "distance": 0.7,
        },
    ]


def test_reranker_initialization(reranker):
    """Test that reranker initializes correctly."""
    assert reranker is not None
    assert reranker.model is not None
    assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"


def test_rerank_returns_list(reranker, sample_documents):
    """Test that rerank returns a list."""
    results = reranker.rerank("Python programming", sample_documents)
    assert isinstance(results, list)


def test_rerank_adds_scores(reranker, sample_documents):
    """Test that rerank adds rerank_score to documents."""
    results = reranker.rerank("Python programming", sample_documents)

    for doc in results:
        assert "rerank_score" in doc
        assert isinstance(doc["rerank_score"], float)


def test_rerank_sorts_by_relevance(reranker, sample_documents):
    """Test that rerank sorts documents by relevance score."""
    results = reranker.rerank("Python programming", sample_documents)

    # Check that scores are in descending order
    scores = [doc["rerank_score"] for doc in results]
    assert scores == sorted(scores, reverse=True)


def test_rerank_respects_top_k(reranker, sample_documents):
    """Test that rerank returns only top_k documents."""
    results = reranker.rerank("Python programming", sample_documents, top_k=2)
    assert len(results) == 2


def test_rerank_empty_documents(reranker):
    """Test that rerank handles empty document list."""
    results = reranker.rerank("test query", [])
    assert results == []


def test_rerank_empty_query(reranker, sample_documents):
    """Test that rerank handles empty query."""
    results = reranker.rerank("", sample_documents, top_k=2)
    assert len(results) == 2


def test_rerank_single_document(reranker):
    """Test reranking with a single document."""
    docs = [{"id": "doc1", "text": "Python programming", "distance": 0.5}]
    results = reranker.rerank("Python", docs, top_k=1)

    assert len(results) == 1
    assert "rerank_score" in results[0]


def test_rerank_changes_order(reranker):
    """Test that reranking can change document order."""
    # Create documents where semantic search order may differ from rerank order
    docs = [
        {"id": "doc1", "text": "The quick brown fox", "distance": 0.3},
        {"id": "doc2", "text": "Python programming language", "distance": 0.5},
        {"id": "doc3", "text": "Learning Python is easy", "distance": 0.7},
    ]

    results = reranker.rerank("Python programming", docs, top_k=3)

    # The document about "Python programming language" should rank high
    assert results[0]["id"] == "doc2"


def test_rerank_preserves_original_fields(reranker, sample_documents):
    """Test that reranking preserves original document fields."""
    results = reranker.rerank("Python", sample_documents)

    for doc in results:
        assert "id" in doc
        assert "text" in doc
        assert "distance" in doc
        assert "rerank_score" in doc


def test_rerank_with_default_top_k(reranker, sample_documents):
    """Test rerank with default top_k parameter."""
    results = reranker.rerank("test", sample_documents)
    # Default top_k is 5, but we only have 3 docs
    assert len(results) == 3


def test_rerank_scores_are_reasonable(reranker):
    """Test that rerank scores are in a reasonable range."""
    docs = [
        {"id": "doc1", "text": "Python programming tutorial", "distance": 0.5},
        {"id": "doc2", "text": "Unrelated content about cooking", "distance": 0.6},
    ]

    results = reranker.rerank("Python programming", docs)

    # Relevant doc should have higher score than unrelated doc
    assert results[0]["rerank_score"] > results[1]["rerank_score"]
    assert results[0]["id"] == "doc1"