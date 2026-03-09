"""Tests for the RAG system module.
@author: Aarti Dashore, Sebastian Silva Arcos
Seattle University, ARIN 5360
@see: https://catalog.seattleu.edu/preview_course_nopop.php?catoid=55&coid
=190380
@version: 4.0.0+w26
"""

from unittest.mock import Mock

import pytest

from retrieval.rag import RAGSystem

# Fixtures


def make_doc(text: str, source: str = "test.txt", score: float = 0.9, doc_id: str = "1"):
    return {
        "id": doc_id,
        "text": text,
        "metadata": {"source": source},
        "score": score,
    }


@pytest.fixture
def mock_retriever():
    retriever = Mock()
    retriever.search.return_value = [
        make_doc("The sky is blue.", source="facts.txt", doc_id="1"),
        make_doc("Water boils at 100°C.", source="science.txt", doc_id="2"),
        make_doc("Paris is the capital of France.", source="geography.txt", doc_id="3"),
    ]
    return retriever


@pytest.fixture
def mock_llm():
    llm = Mock()
    llm.generate.return_value = "This is a generated answer."
    llm.is_available.return_value = True
    return llm


@pytest.fixture
def rag(mock_retriever, mock_llm):
    return RAGSystem(retriever=mock_retriever, llm_client=mock_llm, n_context_docs=3)


# query() — full pipeline


class TestQuery:
    def test_returns_expected_keys(self, rag):
        result = rag.query("What color is the sky?")
        assert "question" in result
        assert "answer" in result
        assert "sources" in result
        assert "n_docs_retrieved" in result

    def test_question_is_preserved(self, rag):
        result = rag.query("What color is the sky?")
        assert result["question"] == "What color is the sky?"

    def test_answer_comes_from_llm(self, rag, mock_llm):
        mock_llm.generate.return_value = "The sky is blue."
        result = rag.query("What color is the sky?")
        assert result["answer"] == "The sky is blue."

    def test_retriever_is_called(self, rag, mock_retriever):
        rag.query("test question")
        mock_retriever.search.assert_called_once()

    def test_llm_is_called(self, rag, mock_llm):
        rag.query("test question")
        mock_llm.generate.assert_called_once()

    def test_n_docs_retrieved_matches_results(self, rag, mock_retriever):
        mock_retriever.search.return_value = [make_doc("doc 1"), make_doc("doc 2")]
        result = rag.query("question")
        assert result["n_docs_retrieved"] == 2

    def test_n_results_overrides_default(self, rag, mock_retriever):
        rag.query("question", n_results=5)
        call_kwargs = mock_retriever.search.call_args[1]
        assert call_kwargs["n_results"] == 5

    def test_sources_contain_expected_fields(self, rag):
        result = rag.query("question")
        for source in result["sources"]:
            assert "id" in source
            assert "text" in source
            assert "metadata" in source
            assert "score" in source

    def test_temperature_passed_to_llm(self, rag, mock_llm):
        rag.query("question", temperature=0.2)
        call_kwargs = mock_llm.generate.call_args[1]
        assert call_kwargs["temperature"] == 0.2


# _build_context()


class TestBuildContext:
    def test_empty_docs_returns_no_docs_message(self, rag):
        context = rag._build_context([])
        assert "No relevant documents found" in context

    def test_single_doc_included_in_context(self, rag):
        docs = [make_doc("The sky is blue.", source="facts.txt")]
        context = rag._build_context(docs)
        assert "The sky is blue." in context

    def test_multiple_docs_all_included(self, rag):
        docs = [
            make_doc("First fact.", source="a.txt"),
            make_doc("Second fact.", source="b.txt"),
        ]
        context = rag._build_context(docs)
        assert "First fact." in context
        assert "Second fact." in context

    def test_source_metadata_included(self, rag):
        docs = [make_doc("Some text.", source="myfile.txt")]
        context = rag._build_context(docs)
        assert "myfile.txt" in context

    def test_docs_are_numbered(self, rag):
        docs = [make_doc("A"), make_doc("B")]
        context = rag._build_context(docs)
        assert "[1]" in context
        assert "[2]" in context


# _create_prompt()


class TestCreatePrompt:
    def test_prompt_contains_question(self, rag):
        prompt = rag._create_prompt("What is the capital of France?", "Some context.")
        assert "What is the capital of France?" in prompt

    def test_prompt_contains_context(self, rag):
        prompt = rag._create_prompt("Any question?", "Paris is the capital of France.")
        assert "Paris is the capital of France." in prompt

    def test_prompt_contains_answer_label(self, rag):
        prompt = rag._create_prompt("question", "context")
        assert "Answer:" in prompt

    def test_prompt_contains_instructions(self, rag):
        prompt = rag._create_prompt("question", "context")
        assert "context" in prompt.lower()


# _get_system_prompt()


class TestGetSystemPrompt:
    def test_system_prompt_is_string(self, rag):
        system_prompt = rag._get_system_prompt()
        assert isinstance(system_prompt, str)

    def test_system_prompt_is_not_empty(self, rag):
        system_prompt = rag._get_system_prompt()
        assert len(system_prompt) > 0

    def test_system_prompt_mentions_context(self, rag):
        system_prompt = rag._get_system_prompt()
        assert "context" in system_prompt.lower()


# Ready state / LLM availability


class TestReadyState:
    def test_is_ready_when_llm_available(self, mock_retriever, mock_llm):
        mock_llm.is_available.return_value = True
        rag = RAGSystem(retriever=mock_retriever, llm_client=mock_llm)
        assert rag.llm_client.is_available() is True

    def test_not_ready_when_llm_unavailable(self, mock_retriever, mock_llm):
        mock_llm.is_available.return_value = False
        rag = RAGSystem(retriever=mock_retriever, llm_client=mock_llm)
        assert rag.llm_client.is_available() is False
