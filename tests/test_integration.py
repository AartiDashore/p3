"""
Integration tests for P3 RAG chatbot API.

@author: Aarti Dashore
Seattle University, ARIN 5360
@see: https://catalog.seattleu.edu/preview_course_nopop.php?catoid=55&coid=190380
@version: 3.0.0+w26

Tests use FastAPI TestClient and mock the LLM so Ollama is not required.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import retrieval.main as m
from retrieval.main import app
from retrieval.rag import RAGSystem
from retrieval.retriever import DocumentRetriever


# ── Build a no-lifespan test app that shares all routes ───────────────────
def _make_test_app() -> FastAPI:
    """
    Return a fresh FastAPI app with NO lifespan but all the same routes.
    This is the only 100% reliable way to prevent the real retriever/LLM
    from starting during tests regardless of FastAPI/Starlette version.
    """
    test_app = FastAPI()
    for route in app.routes:
        test_app.routes.append(route)
    return test_app


TEST_APP = _make_test_app()


# ── Fixtures ───────────────────────────────────────────────
@pytest.fixture
def mock_rag_system():
    """Mock RAGSystem that returns a canned answer."""
    rag = MagicMock(spec=RAGSystem)
    rag.query.return_value = {
        "question": "What is machine learning?",
        "answer": "Machine learning is a subset of AI.",
        "sources": [
            {
                "id": "doc1_0",
                "text": "Machine learning is a subset of AI.",
                "metadata": {"filename": "ml.txt"},
                "score": None,
            }
        ],
        "n_docs_retrieved": 1,
    }
    return rag


@pytest.fixture
def mock_retriever_fixture():
    """Mock DocumentRetriever."""
    r = MagicMock(spec=DocumentRetriever)
    r.search.return_value = [
        {
            "id": "doc1_0",
            "text": "Python is a programming language.",
            "metadata": {"filename": "python.txt"},
            "distance": 0.3,
        }
    ]
    r.document_count = 5
    r._indexed = True
    return r


@pytest.fixture(autouse=True)
def reset_globals():
    """Reset globals before and after every test to prevent bleed."""
    m.retriever = None
    m.rag_system = None
    m.llm_client = None
    yield
    m.retriever = None
    m.rag_system = None
    m.llm_client = None


@pytest.fixture
def client(mock_rag_system, mock_retriever_fixture):
    """TestClient using test app (no lifespan) with mocked globals."""
    m.retriever = mock_retriever_fixture
    m.rag_system = mock_rag_system
    m.llm_client = MagicMock()
    m.llm_client.is_available.return_value = True
    with TestClient(TEST_APP, raise_server_exceptions=False) as c:
        yield c


# ── /health ────────────────────────────────────────────────
def test_health_returns_200(client):
    res = client.get("/health")
    assert res.status_code == 200


def test_health_returns_healthy_status(client):
    res = client.get("/health")
    assert res.json()["status"] == "healthy"


def test_health_returns_doc_count(client):
    res = client.get("/health")
    data = res.json()
    assert "documents_indexed" in data
    assert data["documents_indexed"] == 5


def test_health_returns_llm_available(client):
    res = client.get("/health")
    assert "llm_available" in res.json()


def test_health_no_retriever():
    """Health check works even without retriever (globals already None)."""
    with TestClient(TEST_APP, raise_server_exceptions=False) as c:
        res = c.get("/health")
    assert res.status_code == 200
    assert res.json()["documents_indexed"] == 0


# ── /search ───────────────────────────────────────────────
def test_search_returns_200(client):
    res = client.post("/search", json={"query": "python", "n_results": 3})
    assert res.status_code == 200


def test_search_returns_results(client):
    res = client.post("/search", json={"query": "python", "n_results": 3})
    data = res.json()
    assert "results" in data
    assert "count" in data
    assert data["query"] == "python"


def test_search_empty_query_returns_400(client):
    res = client.post("/search", json={"query": "   ", "n_results": 5})
    assert res.status_code == 400


def test_search_invalid_n_results_returns_400(client):
    res = client.post("/search", json={"query": "test", "n_results": 100})
    assert res.status_code == 400


def test_search_no_retriever_returns_503():
    """Search without retriever returns 503 (globals already None)."""
    with TestClient(TEST_APP, raise_server_exceptions=False) as c:
        res = c.post("/search", json={"query": "test", "n_results": 5})
    assert res.status_code == 503


# ── /rag ──────────────────────────────────────────────────
def test_rag_returns_200(client):
    res = client.post("/rag", json={"question": "What is machine learning?"})
    assert res.status_code == 200


def test_rag_returns_answer(client):
    res = client.post("/rag", json={"question": "What is machine learning?"})
    data = res.json()
    assert "answer" in data
    assert len(data["answer"]) > 0


def test_rag_returns_sources(client):
    res = client.post("/rag", json={"question": "What is machine learning?"})
    data = res.json()
    assert "sources" in data
    assert isinstance(data["sources"], list)


def test_rag_returns_question(client):
    res = client.post("/rag", json={"question": "What is machine learning?"})
    assert res.json()["question"] == "What is machine learning?"


def test_rag_empty_question_returns_400(client):
    res = client.post("/rag", json={"question": "   "})
    assert res.status_code == 400


def test_rag_invalid_n_context_docs_returns_400(client):
    res = client.post("/rag", json={"question": "test", "n_context_docs": 100})
    assert res.status_code == 400


def test_rag_invalid_temperature_returns_400(client):
    res = client.post("/rag", json={"question": "test", "temperature": 5.0})
    assert res.status_code == 400


def test_rag_no_system_returns_503():
    """RAG without rag_system returns 503 (globals already None)."""
    with TestClient(TEST_APP, raise_server_exceptions=False) as c:
        res = c.post("/rag", json={"question": "test"})
    assert res.status_code == 503


def test_rag_with_conversation_history(client, mock_rag_system):
    payload = {
        "question": "What about neural networks?",
        "conversation_history": [
            {"question": "What is AI?", "answer": "AI is artificial intelligence."}
        ],
    }
    res = client.post("/rag", json=payload)
    assert res.status_code == 200
    call_kwargs = mock_rag_system.query.call_args[1]
    assert call_kwargs.get("system_prompt") is not None
    assert "What is AI?" in call_kwargs["system_prompt"]


def test_rag_no_history_uses_none_system_prompt(client, mock_rag_system):
    res = client.post("/rag", json={"question": "test"})
    assert res.status_code == 200
    call_kwargs = mock_rag_system.query.call_args[1]
    assert call_kwargs.get("system_prompt") is None


def test_rag_passes_temperature(client, mock_rag_system):
    client.post("/rag", json={"question": "test", "temperature": 0.2})
    call_kwargs = mock_rag_system.query.call_args[1]
    assert call_kwargs.get("temperature") == 0.2


def test_rag_passes_n_context_docs(client, mock_rag_system):
    client.post("/rag", json={"question": "test", "n_context_docs": 7})
    call_kwargs = mock_rag_system.query.call_args[1]
    assert call_kwargs.get("n_results") == 7


# ── /documents ────────────────────────────────────────────
def test_documents_list_returns_200(client):
    res = client.get("/documents")
    assert res.status_code == 200


def test_documents_list_has_documents_key(client):
    res = client.get("/documents")
    data = res.json()
    assert "documents" in data
    assert "count" in data


# ── / (UI) ────────────────────────────────────────────────
def test_root_returns_200(client):
    res = client.get("/")
    assert res.status_code == 200


def test_not_found(client):
    res = client.get("/nonexistent")
    assert res.status_code == 404
