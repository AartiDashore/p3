"""
Unit tests for RAGSystem.

@author: Aarti Dashore
Seattle University, ARIN 5360
@see: https://catalog.seattleu.edu/preview_course_nopop.php?catoid=55&coid=190380
@version: 3.0.0+w26

All tests use mocks — no Ollama or real retriever required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from retrieval.llm import LLMClient
from retrieval.rag import RAGSystem
from retrieval.retriever import DocumentRetriever


# ── Fixtures ───────────────────────────────────────────────
@pytest.fixture
def mock_retriever():
    """Mock DocumentRetriever."""
    r = MagicMock(spec=DocumentRetriever)
    r.search.return_value = [
        {
            "id": "doc1_0",
            "text": "Machine learning is a subset of AI.",
            "metadata": {"filename": "ml_basics.txt", "chunk": 0},
            "distance": 0.3,
        },
        {
            "id": "doc2_0",
            "text": "Neural networks are inspired by the brain.",
            "metadata": {"filename": "neural_nets.txt", "chunk": 0},
            "distance": 0.5,
        },
    ]
    r._indexed = True
    r.document_count = 10
    return r


@pytest.fixture
def mock_llm():
    """Mock LLMClient."""
    llm = MagicMock(spec=LLMClient)
    llm.generate.return_value = "According to Document 1, machine learning is a subset of AI."
    llm.is_available.return_value = True
    return llm


@pytest.fixture
def rag(mock_retriever, mock_llm):
    """RAGSystem with mocked components."""
    return RAGSystem(retriever=mock_retriever, llm_client=mock_llm, n_context_docs=3)


# ── Initialization ─────────────────────────────────────────
def test_rag_init_with_explicit_llm(mock_retriever, mock_llm):
    """Test RAGSystem initializes with provided LLM client."""
    r = RAGSystem(retriever=mock_retriever, llm_client=mock_llm)
    assert r.retriever is mock_retriever
    assert r.llm_client is mock_llm
    assert r.n_context_docs == 3


def test_rag_init_default_n_context(mock_retriever, mock_llm):
    """Test default n_context_docs is 3."""
    r = RAGSystem(retriever=mock_retriever, llm_client=mock_llm)
    assert r.n_context_docs == 3


def test_rag_init_custom_n_context(mock_retriever, mock_llm):
    """Test custom n_context_docs is stored."""
    r = RAGSystem(retriever=mock_retriever, llm_client=mock_llm, n_context_docs=5)
    assert r.n_context_docs == 5


def test_rag_creates_default_llm_when_none(mock_retriever):
    """Test RAGSystem creates LLMClient when none provided."""
    with patch("retrieval.rag.LLMClient") as MockLLM:
        MockLLM.return_value = MagicMock()
        r = RAGSystem(retriever=mock_retriever, llm_client=None)
        MockLLM.assert_called_once()
        assert r.llm_client is not None


# ── query() ───────────────────────────────────────────────
def test_query_returns_dict(rag):
    """Test that query() returns a dict."""
    result = rag.query("What is machine learning?")
    assert isinstance(result, dict)


def test_query_has_required_keys(rag):
    """Test that query result has answer, sources, question, n_docs_retrieved."""
    result = rag.query("What is machine learning?")
    assert "answer" in result
    assert "sources" in result
    assert "question" in result
    assert "n_docs_retrieved" in result


def test_query_returns_correct_question(rag):
    """Test that query result echoes the question."""
    q = "What is machine learning?"
    result = rag.query(q)
    assert result["question"] == q


def test_query_returns_answer_string(rag):
    """Test that answer is a string."""
    result = rag.query("What is AI?")
    assert isinstance(result["answer"], str)
    assert len(result["answer"]) > 0


def test_query_returns_sources_list(rag):
    """Test that sources is a list."""
    result = rag.query("What is AI?")
    assert isinstance(result["sources"], list)


def test_query_calls_retriever_search(rag, mock_retriever):
    """Test that query() calls retriever.search."""
    rag.query("What is AI?")
    mock_retriever.search.assert_called_once()


def test_query_calls_retriever_with_question(rag, mock_retriever):
    """Test that retriever.search is called with the question."""
    question = "What is machine learning?"
    rag.query(question)
    call_args = mock_retriever.search.call_args
    assert call_args[0][0] == question


def test_query_calls_llm_generate(rag, mock_llm):
    """Test that query() calls llm_client.generate."""
    rag.query("What is AI?")
    mock_llm.generate.assert_called_once()


def test_query_passes_temperature(rag, mock_llm):
    """Test that temperature is passed to generate()."""
    rag.query("test", temperature=0.3)
    call_kwargs = mock_llm.generate.call_args[1]
    assert call_kwargs.get("temperature") == 0.3


def test_query_sources_have_correct_fields(rag):
    """Test that each source has id, text, metadata."""
    result = rag.query("What is AI?")
    for src in result["sources"]:
        assert "id" in src
        assert "text" in src
        assert "metadata" in src


def test_query_n_docs_retrieved(rag, mock_retriever):
    """Test that n_docs_retrieved matches retrieved docs count."""
    result = rag.query("What is AI?")
    assert result["n_docs_retrieved"] == len(mock_retriever.search.return_value)


def test_query_uses_n_results_override(rag, mock_retriever):
    """Test that n_results parameter overrides default."""
    rag.query("test", n_results=7)
    args, kwargs = mock_retriever.search.call_args
    assert (len(args) > 1 and args[1] == 7) or kwargs.get("n_results") == 7


def test_query_uses_default_n_context_when_no_override(rag, mock_retriever):
    """Test that default n_context_docs is used when n_results not specified."""
    rag.query("test")
    call_args = mock_retriever.search.call_args
    n = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("n_results")
    assert n == rag.n_context_docs


# ── _build_context() ──────────────────────────────────────
def test_build_context_with_documents(rag):
    """Test context building with documents."""
    docs = [
        {"id": "d1", "text": "Python is great.", "metadata": {"filename": "py.txt"}},
        {"id": "d2", "text": "AI is interesting.", "metadata": {"filename": "ai.txt"}},
    ]
    context = rag._build_context(docs)
    assert "Python is great." in context
    assert "AI is interesting." in context


def test_build_context_empty_returns_no_docs_message(rag):
    """Test that empty documents returns no-documents message."""
    context = rag._build_context([])
    assert "No relevant documents" in context


def test_build_context_includes_source_name(rag):
    """Test that source filename appears in context."""
    docs = [{"id": "d1", "text": "content", "metadata": {"filename": "myfile.txt"}}]
    context = rag._build_context(docs)
    assert "myfile.txt" in context


def test_build_context_numbers_documents(rag):
    """Test that documents are numbered in context."""
    docs = [
        {"id": "d1", "text": "First", "metadata": {}},
        {"id": "d2", "text": "Second", "metadata": {}},
    ]
    context = rag._build_context(docs)
    assert "[1]" in context
    assert "[2]" in context


# ── _create_prompt() ──────────────────────────────────────
def test_create_prompt_includes_question(rag):
    """Test that the prompt includes the question."""
    prompt = rag._create_prompt("What is AI?", "Some context here.")
    assert "What is AI?" in prompt


def test_create_prompt_includes_context(rag):
    """Test that the prompt includes the context."""
    prompt = rag._create_prompt("question", "Very specific context text.")
    assert "Very specific context text." in prompt


def test_create_prompt_has_answer_marker(rag):
    """Test that the prompt ends with Answer: marker."""
    prompt = rag._create_prompt("question", "context")
    assert "Answer:" in prompt


# ── _get_system_prompt() ──────────────────────────────────
def test_get_system_prompt_returns_default_when_none(rag):
    """Test that None returns the default system prompt."""
    prompt = rag._get_system_prompt(None)
    assert len(prompt) > 0


def test_get_system_prompt_returns_custom_when_provided(rag):
    """Test that custom prompt overrides default."""
    custom = "You are a pirate assistant."
    result = rag._get_system_prompt(custom)
    assert result == custom


def test_get_system_prompt_returns_default_for_empty_string(rag):
    """Test that empty string returns default prompt."""
    result = rag._get_system_prompt("")
    assert result != ""
    # Should be the default, not empty
    from retrieval.rag import DEFAULT_SYSTEM_PROMPT

    assert result == DEFAULT_SYSTEM_PROMPT


def test_get_system_prompt_strips_whitespace(rag):
    """Test that whitespace-only returns default."""
    result = rag._get_system_prompt("   ")
    from retrieval.rag import DEFAULT_SYSTEM_PROMPT

    assert result == DEFAULT_SYSTEM_PROMPT


# ── Edge cases ─────────────────────────────────────────────
def test_query_with_no_retrieved_docs(mock_llm):
    """Test RAG with retriever that returns empty list."""
    empty_retriever = MagicMock(spec=DocumentRetriever)
    empty_retriever.search.return_value = []
    empty_retriever._indexed = True

    mock_llm.generate.return_value = "I don't have information on that."
    rag = RAGSystem(retriever=empty_retriever, llm_client=mock_llm)
    result = rag.query("What is quantum computing?")

    assert result["n_docs_retrieved"] == 0
    assert result["sources"] == []
    assert "answer" in result


def test_query_with_custom_system_prompt(rag, mock_llm):
    """Test that custom system_prompt is passed to generate."""
    custom = "You are a space expert."
    rag.query("test", system_prompt=custom)
    call_kwargs = mock_llm.generate.call_args[1]
    assert call_kwargs.get("system_prompt") == custom
