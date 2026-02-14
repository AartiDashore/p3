"""
Unit tests for HybridSearcher.

@author: Aarti Dashore
Seattle University, ARIN 5360
@see: https://catalog.seattleu.edu/preview_course_nopop.php?catoid=55&coid=190380
@version: 1.0.0+w26
"""

from __future__ import annotations

import pytest

from retrieval.hybrid import HybridSearcher


@pytest.fixture
def sample_documents():
    """Sample documents for testing hybrid search."""
    return [
        {
            "id": "doc1",
            "text": "Python programming language for data science",
            "metadata": {"filename": "doc1.txt"},
        },
        {
            "id": "doc2",
            "text": "Machine learning and neural networks",
            "metadata": {"filename": "doc2.txt"},
        },
        {
            "id": "doc3",
            "text": "Python web development with Django",
            "metadata": {"filename": "doc3.txt"},
        },
        {
            "id": "doc4",
            "text": "Deep learning using Python and TensorFlow",
            "metadata": {"filename": "doc4.txt"},
        },
    ]


@pytest.fixture
def hybrid_searcher():
    """Create a HybridSearcher instance."""
    return HybridSearcher()


def test_hybrid_searcher_initialization():
    """Test that HybridSearcher initializes correctly."""
    searcher = HybridSearcher()
    assert searcher.k == 60
    assert searcher.bm25_index is None
    assert searcher.documents is None


def test_hybrid_searcher_custom_k():
    """Test initialization with custom k parameter."""
    searcher = HybridSearcher(k=100)
    assert searcher.k == 100


def test_index_documents(hybrid_searcher, sample_documents):
    """Test indexing documents builds BM25 index."""
    hybrid_searcher.index_documents(sample_documents)

    assert hybrid_searcher.bm25_index is not None
    assert hybrid_searcher.documents is not None
    assert len(hybrid_searcher.documents) == 4


def test_index_empty_documents(hybrid_searcher):
    """Test indexing empty document list."""
    hybrid_searcher.index_documents([])

    assert hybrid_searcher.bm25_index is None
    assert hybrid_searcher.documents is None


def test_search_bm25_returns_results(hybrid_searcher, sample_documents):
    """Test BM25 search returns results."""
    hybrid_searcher.index_documents(sample_documents)
    results = hybrid_searcher.search_bm25("Python programming", n_results=3)

    assert isinstance(results, list)
    assert len(results) <= 3


def test_search_bm25_adds_scores(hybrid_searcher, sample_documents):
    """Test that BM25 search adds scores to results."""
    hybrid_searcher.index_documents(sample_documents)
    results = hybrid_searcher.search_bm25("Python", n_results=3)

    for doc in results:
        assert "bm25_score" in doc
        assert isinstance(doc["bm25_score"], float)


def test_search_bm25_ranks_by_relevance(hybrid_searcher, sample_documents):
    """Test that BM25 ranks documents by keyword relevance."""
    hybrid_searcher.index_documents(sample_documents)
    results = hybrid_searcher.search_bm25("Python", n_results=4)

    # Documents with "Python" should have higher scores
    scores = [doc["bm25_score"] for doc in results]
    assert scores == sorted(scores, reverse=True)


def test_search_bm25_keyword_matching(hybrid_searcher, sample_documents):
    """Test that BM25 finds exact keyword matches."""
    hybrid_searcher.index_documents(sample_documents)
    results = hybrid_searcher.search_bm25("Django", n_results=2)

    # doc3 contains "Django" and should rank high
    assert results[0]["id"] == "doc3"


def test_search_bm25_without_index(hybrid_searcher):
    """Test BM25 search without building index returns empty list."""
    results = hybrid_searcher.search_bm25("test query")
    assert results == []


def test_reciprocal_rank_fusion_combines_results(hybrid_searcher, sample_documents):
    """Test that RRF combines results from both lists."""
    # Simulate semantic and BM25 results
    semantic_results = [
        sample_documents[0],  # doc1
        sample_documents[1],  # doc2
    ]
    bm25_results = [
        sample_documents[1],  # doc2 (appears in both)
        sample_documents[2],  # doc3
    ]

    fused = hybrid_searcher.reciprocal_rank_fusion(semantic_results, bm25_results, n_results=3)

    assert len(fused) <= 3
    assert all("rrf_score" in doc for doc in fused)


def test_rrf_boosts_documents_in_both_lists(hybrid_searcher, sample_documents):
    """Test that RRF gives higher scores to documents in both lists."""
    semantic_results = [
        sample_documents[0],  # doc1
        sample_documents[1],  # doc2
    ]
    bm25_results = [
        sample_documents[1],  # doc2 (appears in both!)
        sample_documents[2],  # doc3
    ]

    fused = hybrid_searcher.reciprocal_rank_fusion(semantic_results, bm25_results, n_results=3)

    # doc2 appears in both lists, so it should rank #1
    assert fused[0]["id"] == "doc2"


def test_rrf_handles_disjoint_lists(hybrid_searcher, sample_documents):
    """Test RRF with completely different lists."""
    semantic_results = [sample_documents[0], sample_documents[1]]
    bm25_results = [sample_documents[2], sample_documents[3]]

    fused = hybrid_searcher.reciprocal_rank_fusion(semantic_results, bm25_results, n_results=4)

    # Should contain all 4 unique documents
    assert len(fused) == 4
    doc_ids = {doc["id"] for doc in fused}
    assert doc_ids == {"doc1", "doc2", "doc3", "doc4"}


def test_rrf_respects_n_results(hybrid_searcher, sample_documents):
    """Test that RRF returns only top N results."""
    semantic_results = sample_documents[:2]
    bm25_results = sample_documents[2:]

    fused = hybrid_searcher.reciprocal_rank_fusion(semantic_results, bm25_results, n_results=2)

    assert len(fused) == 2


def test_rrf_score_calculation(hybrid_searcher, sample_documents):
    """Test RRF score calculation is correct."""
    # doc1 at rank 1 in semantic
    # doc2 at rank 1 in BM25 and rank 2 in semantic
    semantic_results = [sample_documents[0], sample_documents[1]]  # doc1, doc2
    bm25_results = [sample_documents[1]]  # doc2

    fused = hybrid_searcher.reciprocal_rank_fusion(semantic_results, bm25_results, n_results=2)

    # doc2 should have highest score (appears in both)
    # RRF(doc2) = 1/(60+2) + 1/(60+1) ≈ 0.0161 + 0.0164 = 0.0325
    # RRF(doc1) = 1/(60+1) ≈ 0.0164
    assert fused[0]["id"] == "doc2"
    assert fused[0]["rrf_score"] > fused[1]["rrf_score"]


def test_rrf_preserves_original_fields(hybrid_searcher, sample_documents):
    """Test that RRF preserves original document fields."""
    semantic_results = [sample_documents[0]]
    bm25_results = [sample_documents[1]]

    fused = hybrid_searcher.reciprocal_rank_fusion(semantic_results, bm25_results)

    for doc in fused:
        assert "id" in doc
        assert "text" in doc
        assert "metadata" in doc
        assert "rrf_score" in doc


def test_full_hybrid_workflow(hybrid_searcher, sample_documents):
    """Test complete hybrid search workflow."""
    # Index documents
    hybrid_searcher.index_documents(sample_documents)

    # Get BM25 results
    bm25_results = hybrid_searcher.search_bm25("Python", n_results=3)

    # Simulate semantic results (would come from vector store)
    semantic_results = [sample_documents[0], sample_documents[1]]

    # Fuse with RRF
    fused = hybrid_searcher.reciprocal_rank_fusion(semantic_results, bm25_results, n_results=3)

    assert len(fused) <= 3
    assert all("rrf_score" in doc for doc in fused)
    assert all("id" in doc for doc in fused)


def test_rrf_with_empty_semantic_list(hybrid_searcher, sample_documents):
    """Test RRF with empty semantic results."""
    semantic_results = []
    bm25_results = [sample_documents[0]]

    fused = hybrid_searcher.reciprocal_rank_fusion(semantic_results, bm25_results, n_results=2)

    assert len(fused) == 1
    assert fused[0]["id"] == "doc1"


def test_rrf_with_empty_bm25_list(hybrid_searcher, sample_documents):
    """Test RRF with empty BM25 results."""
    semantic_results = [sample_documents[0]]
    bm25_results = []

    fused = hybrid_searcher.reciprocal_rank_fusion(semantic_results, bm25_results, n_results=2)

    assert len(fused) == 1
    assert fused[0]["id"] == "doc1"


def test_rrf_with_both_empty(hybrid_searcher):
    """Test RRF with both lists empty."""
    fused = hybrid_searcher.reciprocal_rank_fusion([], [], n_results=5)
    assert fused == []
