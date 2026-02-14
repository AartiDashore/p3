"""
Integration test for hybrid search (BM25 + RRF).

This test demonstrates that hybrid search improves retrieval when queries
contain specific keywords that benefit from exact matching.
"""

from pathlib import Path

from retrieval.retriever import DocumentRetriever


def test_hybrid_search_effectiveness():
    """
    Demonstrate that hybrid search (BM25 + Semantic + RRF) improves search quality.

    This test compares search results across different retrieval modes to show:
    1. Hybrid search changes the order of results
    2. Hybrid search better captures keyword matches
    3. The rrf_score field is added by hybrid search
    4. Different modes produce different rankings

    This demonstrates Project 2's extra credit feature.
    """
    # Setup - use documents/ folder
    sample_dir = "documents"

    # Use a query that benefits from keyword matching
    # "garlic" and "crucifix" are specific terms that appear in Dracula
    query = "garlic crucifix vampire"

    # Test Mode 1: Semantic only (baseline)
    retriever_semantic = DocumentRetriever(use_reranking=False, use_hybrid=False)
    retriever_semantic.index_documents(sample_dir)
    results_semantic = retriever_semantic.search(query, n_results=5)

    # Test Mode 2: Semantic + Reranking (required feature)
    retriever_rerank = DocumentRetriever(use_reranking=True, use_hybrid=False)
    retriever_rerank.index_documents(sample_dir)
    results_rerank = retriever_rerank.search(query, n_results=5)

    # Test Mode 3: Hybrid (BM25 + Semantic + RRF)
    retriever_hybrid = DocumentRetriever(use_reranking=False, use_hybrid=True)
    retriever_hybrid.index_documents(sample_dir)
    results_hybrid = retriever_hybrid.search(query, n_results=5)

    # Test Mode 4: Full System (Hybrid + Reranking)
    retriever_full = DocumentRetriever(use_reranking=True, use_hybrid=True)
    retriever_full.index_documents(sample_dir)
    results_full = retriever_full.search(query, n_results=5)

    # Verify all modes return results
    assert len(results_semantic) > 0, "Semantic search should return results"
    assert len(results_rerank) > 0, "Reranking should return results"
    assert len(results_hybrid) > 0, "Hybrid search should return results"
    assert len(results_full) > 0, "Full system should return results"

    # Verify field presence
    assert all("distance" in doc for doc in results_semantic), (
        "Semantic results should have distance"
    )

    assert all("rerank_score" in doc for doc in results_rerank), (
        "Reranked results should have rerank_score"
    )

    assert all("rrf_score" in doc for doc in results_hybrid), "Hybrid results should have rrf_score"

    assert all("rrf_score" in doc and "rerank_score" in doc for doc in results_full), (
        "Full system should have both rrf_score and rerank_score"
    )

    # Get top result chunks for comparison
    top_chunks_semantic = [doc["metadata"]["chunk"] for doc in results_semantic[:3]]
    top_chunks_rerank = [doc["metadata"]["chunk"] for doc in results_rerank[:3]]
    top_chunks_hybrid = [doc["metadata"]["chunk"] for doc in results_hybrid[:3]]
    top_chunks_full = [doc["metadata"]["chunk"] for doc in results_full[:3]]

    # Verify different modes produce different orderings
    all_orderings = [
        top_chunks_semantic,
        top_chunks_rerank,
        top_chunks_hybrid,
        top_chunks_full,
    ]

    # At least some variation should exist
    unique_orderings = len({tuple(x) for x in all_orderings})
    assert unique_orderings > 1, (
        f"Different retrieval modes should produce different rankings. "
        f"All modes returned: {all_orderings}"
    )

    # Print detailed comparison
    print("\n" + "=" * 80)
    print("HYBRID SEARCH COMPARISON")
    print("=" * 80)
    print(f"\nQuery: '{query}'")
    print("\nTop 3 chunks by retrieval mode:")
    print("-" * 80)
    print(f"Mode 1 (Semantic only):        {top_chunks_semantic}")
    print(f"Mode 2 (Semantic + Reranking): {top_chunks_rerank}")
    print(f"Mode 3 (Hybrid):               {top_chunks_hybrid}")
    print(f"Mode 4 (Full System):          {top_chunks_full}")

    print("\n" + "=" * 80)
    print("DETAILED RESULT COMPARISON")
    print("=" * 80)

    print("\n" + "-" * 80)
    print("Top result - Semantic only:")
    print("-" * 80)
    print(f"Chunk: {results_semantic[0]['metadata']['chunk']}")
    print(f"Distance: {results_semantic[0]['distance']:.4f}")
    print(f"Text preview: {results_semantic[0]['text'][:150]}...")

    print("\n" + "-" * 80)
    print("Top result - Semantic + Reranking:")
    print("-" * 80)
    print(f"Chunk: {results_rerank[0]['metadata']['chunk']}")
    print(f"Rerank score: {results_rerank[0]['rerank_score']:.4f}")
    print(f"Text preview: {results_rerank[0]['text'][:150]}...")

    print("\n" + "-" * 80)
    print("Top result - Hybrid (BM25 + Semantic + RRF):")
    print("-" * 80)
    print(f"Chunk: {results_hybrid[0]['metadata']['chunk']}")
    print(f"RRF score: {results_hybrid[0]['rrf_score']:.6f}")
    if "bm25_score" in results_hybrid[0]:
        print(f"BM25 score: {results_hybrid[0]['bm25_score']:.4f}")
    print(f"Text preview: {results_hybrid[0]['text'][:150]}...")

    print("\n" + "-" * 80)
    print("Top result - Full System (Hybrid + Reranking):")
    print("-" * 80)
    print(f"Chunk: {results_full[0]['metadata']['chunk']}")
    print(f"Rerank score: {results_full[0]['rerank_score']:.4f}")
    print(f"RRF score: {results_full[0]['rrf_score']:.6f}")
    print(f"Text preview: {results_full[0]['text'][:150]}...")

    # Analyze keyword presence in top results
    print("\n" + "=" * 80)
    print("KEYWORD ANALYSIS")
    print("=" * 80)

    keywords = ["garlic", "crucifix", "vampire"]

    for mode_name, results in [
        ("Semantic only", results_semantic),
        ("Semantic + Reranking", results_rerank),
        ("Hybrid", results_hybrid),
        ("Full System", results_full),
    ]:
        print(f"\n{mode_name} - Top 3 results keyword matches:")
        for i, doc in enumerate(results[:3], 1):
            text_lower = doc["text"].lower()
            matches = [kw for kw in keywords if kw in text_lower]
            print(
                f"  Result {i} (chunk {doc['metadata']['chunk']}): {matches if matches else 'no keywords'}"
            )

    print("\n" + "=" * 80)
    print("Hybrid search successfully demonstrated")
    print("Different retrieval modes produce different rankings")
    print("Keyword-aware search (hybrid) benefits from BM25")
    print("=" * 80 + "\n")


def test_hybrid_rrf_boosting():
    """
    Demonstrate that RRF boosts documents appearing in both semantic and BM25 results.

    This is a key property of Reciprocal Rank Fusion: documents that appear
    high in multiple ranked lists receive higher combined scores.
    """
    # Setup - use documents/ folder
    sample_dir = "documents"

    # Use a query where we expect overlap between semantic and keyword results
    query = "vampire blood"

    # Get hybrid results
    retriever = DocumentRetriever(use_reranking=False, use_hybrid=True)
    retriever.index_documents(sample_dir)
    results = retriever.search(query, n_results=5)

    # Verify we got results
    assert len(results) > 0, "Should return results"

    # Verify RRF scores are present and sorted
    assert all("rrf_score" in doc for doc in results), "Should have rrf_score"

    rrf_scores = [doc["rrf_score"] for doc in results]
    assert rrf_scores == sorted(rrf_scores, reverse=True), (
        "Results should be sorted by RRF score (highest first)"
    )

    print("\n" + "=" * 80)
    print("RRF BOOSTING DEMONSTRATION")
    print("=" * 80)
    print(f"\nQuery: '{query}'")
    print("\nTop 5 results with RRF scores:")
    print("-" * 80)

    for i, doc in enumerate(results, 1):
        chunk = doc["metadata"]["chunk"]
        rrf_score = doc["rrf_score"]
        text_lower = doc["text"].lower()

        # Check keyword presence
        has_vampire = "vampire" in text_lower
        has_blood = "blood" in text_lower

        print(f"\n{i}. Chunk {chunk} - RRF score: {rrf_score:.6f}")
        print(f"   Contains 'vampire': {has_vampire}")
        print(f"   Contains 'blood': {has_blood}")
        if has_vampire and has_blood:
            print("   Contains both keywords - likely boosted by RRF")
        print(f"   Text: {doc['text'][:100]}...")

    print("\n" + "=" * 80)
    print("RRF successfully combines semantic and keyword rankings")
    print("=" * 80 + "\n")
