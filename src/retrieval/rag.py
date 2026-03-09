"""RAG (Retrieval-Augmented Generation) module.
@author: Aarti Dashore, Sebastian Silva Arcos
Seattle University, ARIN 5360
@see: https://catalog.seattleu.edu/preview_course_nopop.php?catoid=55&coid
=190380
@version: 4.0.0+w26
"""

from retrieval.llm import LLMClient
from retrieval.retriever import DocumentRetriever

DEFAULT_SYSTEM_PROMPT = (
    "You are a knowledgeable and helpful assistant that answers questions "
    "using only the context documents provided to you.\n\n"
    "Guidelines:\n"
    "1. Base your answer strictly on the provided context. Do not use outside knowledge.\n"
    "2. When referencing information, cite the source document by name "
    "(e.g., 'According to [source]...' or 'As stated in [source]...').\n"
    "3. If the context contains partial information, share what is available "
    "and clearly note what is missing or unclear.\n"
    "4. If the context does not contain enough information to answer the question, "
    "say so honestly and directly — do not guess or fabricate an answer.\n"
    "5. Keep your tone professional, clear, and concise.\n"
    "6. If multiple sources agree or disagree, note that in your response."
)


class RAGSystem:
    def __init__(
        self,
        retriever: DocumentRetriever,
        llm_client: LLMClient | None = None,
        n_context_docs: int = 3,
    ):
        self.retriever = retriever
        self.llm_client = llm_client or LLMClient()
        self.n_context_docs = n_context_docs

    def query(
        self,
        question: str,
        n_results: int | None = None,
        temperature: float = 0.7,
        system_prompt: str | None = None,
    ) -> dict:
        """
        RAG Pipeline:
        1. Retrieve relevant documents
        2. Build context from documents
        3. Create prompt with context
        4. Generate answer using LLM
        5. Return answer with sources
        """
        n_docs = n_results if n_results is not None else self.n_context_docs

        # Step 1: Retrieve relevant documents
        retrieved = self.retriever.search(question, n_results=n_docs)

        # Step 2: Build context string from retrieved docs
        context = self._build_context(retrieved)

        # Step 3: Create prompt with context and question
        prompt = self._create_prompt(question, context)

        # Step 4: Generate answer using LLM
        active_system_prompt = self._get_system_prompt(system_prompt)
        answer = self.llm_client.generate(
            prompt=prompt,
            system_prompt=active_system_prompt,
            temperature=temperature,
        )

        # Step 5: Return answer with sources
        sources = [
            {
                "id": doc.get("id", ""),
                "text": doc.get("text", ""),
                "metadata": doc.get("metadata", {}),
                "score": doc.get("score", None),
            }
            for doc in retrieved
        ]

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "n_docs_retrieved": len(retrieved),
        }

    def _build_context(self, documents: list) -> str:
        """Format retrieved documents into a single context string."""
        if not documents:
            return "No relevant documents found."

        parts = []
        for i, doc in enumerate(documents, start=1):
            text = doc.get("text", "").strip()
            metadata = doc.get("metadata", {})
            source = metadata.get("source", metadata.get("filename", f"Document {i}"))
            parts.append(f"[{i}] Source: {source}\n{text}")

        return "\n\n".join(parts)

    def _create_prompt(self, question: str, context: str) -> str:
        """Build the final prompt combining context and question."""
        return f"""Context information from relevant documents:

{context}

Based on the context above, please answer the following question.
If the context doesn't contain enough information, say so.

Question: {question}

Answer:"""

    def _get_system_prompt(self, custom_prompt: str | None = None) -> str:
        """Return custom system prompt if provided, otherwise return the default."""
        if custom_prompt and custom_prompt.strip():
            return custom_prompt.strip()
        return DEFAULT_SYSTEM_PROMPT
