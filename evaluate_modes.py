"""
Evaluation script comparing all retrieval modes.
@author: Aarti Dashore
Seattle University, ARIN 5360
@see: https://catalog.seattleu.edu/preview_course_nopop.php?catoid=55&coid=190380
@version: 2.0.0+w26

"""

from retrieval.retriever import DocumentRetriever

# Compare different approaches
query = "machine learning algorithms"

# Baseline: semantic only, no reranking
baseline = DocumentRetriever(use_reranking=False, use_hybrid=False)
baseline.index_documents("documents")
baseline_results = baseline.search(query, n_results=5)

# Required: with reranking
with_rerank = DocumentRetriever(use_reranking=True, use_hybrid=False)
with_rerank.index_documents("documents")
rerank_results = with_rerank.search(query, n_results=5)

# Extra credit: full system
full_system = DocumentRetriever(use_reranking=True, use_hybrid=True)
full_system.index_documents("documents")
hybrid_results = full_system.search(query, n_results=5)

# Show the differences
print("Top result without reranking:", baseline_results[0]["id"])
print("Top result with reranking:", rerank_results[0]["id"])
print("Top result with hybrid+reranking:", hybrid_results[0]["id"])
