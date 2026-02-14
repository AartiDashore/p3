"""
DocumentRetriever provides a high-level interface for document retrieval with optional reranking and hybrid search capabilities.

@author:  Aarti Dashore
Seattle University, ARIN 5360
@see: https://catalog.seattleu.edu/preview_course_nopop.php?catoid=55&coid
=190380
@version: 2.0.0+w26
"""
from retrieval.embeddings import DocumentEmbedder
from retrieval.loader import DocumentChunker, DocumentLoader
from retrieval.store import VectorStore
from retrieval.reranker import CrossEncoderReranker


class DocumentRetriever:
    class DocumentRetriever:
        """
        High-level interface for document retrieval with optional reranking and hybrid search.

        The retriever supports multiple retrieval strategies:
        1. Semantic only: Fast bi-encoder search
        2. Semantic + Reranking: Two-stage pipeline (retrieve 20 -> rerank to top 5)
    
        Args:
            chunk_size (int): Size of document chunks in words (default: 300)
            overlap (int): Overlap between chunks in words (default: 30)
            use_reranking (bool): Whether to use cross-encoder reranking (default: True)
            
        """

    def __init__(
        self,
        chunk_size: int = 300,
        overlap: int = 30,
        use_reranking: bool = True,
    ):
        """Initialize retriever with configurable components."""
        chunker = DocumentChunker(chunk_size=chunk_size, overlap=overlap)
        self.loader = DocumentLoader(chunker=chunker)
        self.store = VectorStore(DocumentEmbedder())
        self._indexed = False

        # Configuration flags
        self.use_reranking = use_reranking

        # Initialize optional components
        self.reranker = CrossEncoderReranker() if use_reranking else None

        # Store all documents for hybrid search
        self._all_documents: list[dict] = []

    def index_documents(self, directory: str):
        """
        Load and index documents from a directory.

        Args:
            directory: Path to the directory containing documents

        Returns:
            Number of documents indexed
        """
        before = self.document_count
        documents = self.loader.load_documents(directory)

        # Index in vector store
        self.store.add_documents(documents)

        # Build BM25 index if hybrid search is enabled
        if self.use_hybrid and self.hybrid_searcher:
            self.hybrid_searcher.index_documents(documents)

        self._indexed = True
        return self.document_count - before

    def search(self, query: str, n_results: int = 5) -> list[dict]:
        """
        Search for documents relevant to the query.

        The search strategy depends on configuration:
        - Semantic only: Direct semantic search
        - Semantic + Reranking: Retrieve 20 -> rerank to top N


        Args:
            query (str): Search query
            n_results (int): Number of final results to return (default: 5)

        Returns:
            list[dict]: Search results with relevance scores
        """
        if not self._indexed:
            raise ValueError("No documents indexed. Call index_documents() first.")

        if self.use_reranking:
            return self._semantic_with_reranking(query, n_results)

        # Case 3: Semantic search only (baseline)
        return self.store.search(query, n_results)

    def _semantic_with_reranking(self, query: str, n_results: int) -> list[dict]:
        """
        Two-stage retrieval: semantic search + cross-encoder reranking.

        Args:
            query (str): Search query
            n_results (int): Number of final results

        Returns:
            list[dict]: Reranked results
        """
        # Stage 1: Retrieve candidates using semantic search
        candidates = self.store.search(query, n_results=20)

        # Stage 2: Rerank candidates
        if candidates and self.reranker:
            results = self.reranker.rerank(query, candidates, top_k=n_results)
            return results

        return candidates[:n_results]

    @property
    def document_count(self) -> int:
        """Return the number of indexed documents."""
        return self.store.count()

    def get_configuration(self) -> dict:
        """
        Get current retrieval configuration.

        Returns:
            dict: Configuration settings
        """
        return {
            "use_reranking": self.use_reranking,
            "indexed": self._indexed,
            "document_count": self.document_count,
        }