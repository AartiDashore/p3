"""
Integration tests for P3 RAG chatbot API.

@author: Aarti Dashore
Seattle University, ARIN 5360
@see: https://catalog.seattleu.edu/preview_course_nopop.php?catoid=55&coid=190380
@version: 3.0.0+w26

Tests use FastAPI TestClient and mock the LLM so Ollama is not required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import retrieval.main as m
from retrieval.main import app
from retrieval.rag import RAGSystem
from retrieval.retriever import DocumentRetriever


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
    """TestClient with mocked retriever and RAG system."""
    m.retriever = mock_retriever_fixture
    m.rag_system = mock_rag_system
    m.llm_client = MagicMock()
    m.llm_client.is_available.return_value = True
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# ── /health ────────────────────────────────────────────────
def test_health_returns_200(client):
    """Health check returns 200."""
    res = client.get("/health")
    assert res.status_code == 200


def test_health_returns_healthy_status(client):
    """Health check returns healthy status."""
    res = client.get("/health")
    data = res.json()
    assert data["status"] == "healthy"


def test_health_returns_llm_available(client):
    """Health check includes llm_available field."""
    res = client.get("/health")
    data = res.json()
    assert "llm_available" in data


# ── /search ───────────────────────────────────────────────
def test_search_returns_200(client):
    """Search endpoint returns 200 with valid query."""
    res = client.post("/search", json={"query": "python", "n_results": 3})
    assert res.status_code == 200


def test_search_returns_results(client):
    """Search returns results list."""
    res = client.post("/search", json={"query": "python", "n_results": 3})
    data = res.json()
    assert "results" in data
    assert "count" in data
    assert data["query"] == "python"


def test_search_empty_query_returns_400(client):
    """Empty query returns 400."""
    res = client.post("/search", json={"query": "   ", "n_results": 5})
    assert res.status_code == 400


def test_search_invalid_n_results_returns_400(client):
    """n_results out of range returns 400."""
    res = client.post("/search", json={"query": "test", "n_results": 100})
    assert res.status_code == 400


# ── /rag ──────────────────────────────────────────────────
def test_rag_returns_200(client):
    """RAG endpoint returns 200 with valid question."""
    res = client.post("/rag", json={"question": "What is machine learning?"})
    assert res.status_code == 200


def test_rag_returns_answer(client):
    """RAG endpoint returns answer field."""
    res = client.post("/rag", json={"question": "What is machine learning?"})
    data = res.json()
    assert "answer" in data
    assert len(data["answer"]) > 0


def test_rag_returns_sources(client):
    """RAG endpoint returns sources list."""
    res = client.post("/rag", json={"question": "What is machine learning?"})
    data = res.json()
    assert "sources" in data
    assert isinstance(data["sources"], list)


def test_rag_returns_question(client):
    """RAG endpoint echoes the question."""
    res = client.post("/rag", json={"question": "What is machine learning?"})
    data = res.json()
    assert data["question"] == "What is machine learning?"


def test_rag_empty_question_returns_400(client):
    """Empty question returns 400."""
    res = client.post("/rag", json={"question": "   "})
    assert res.status_code == 400


def test_rag_invalid_n_context_docs_returns_400(client):
    """n_context_docs out of range returns 400."""
    res = client.post("/rag", json={"question": "test", "n_context_docs": 100})
    assert res.status_code == 400


def test_rag_invalid_temperature_returns_400(client):
    """Temperature out of range returns 400."""
    res = client.post("/rag", json={"question": "test", "temperature": 5.0})
    assert res.status_code == 400


# ── /documents ────────────────────────────────────────────
def test_documents_list_returns_200(client):
    """Documents list endpoint returns 200."""
    res = client.get("/documents")
    assert res.status_code == 200


def test_documents_list_has_documents_key(client):
    """Documents list response has documents key."""
    res = client.get("/documents")
    data = res.json()
    assert "documents" in data
    assert "count" in data


# ── /test/error ───────────────────────────────────────────
def test_generic_exception_handler(client):
    """Generic exception handler returns 500."""
    res = client.get("/test/error")
    assert res.status_code == 500
    assert "Internal server error" in res.json()["detail"]


# ── / (UI) ────────────────────────────────────────────────
def test_root_returns_200(client):
    """Root path returns 200 (serves index.html)."""
    res = client.get("/")
    assert res.status_code == 200


def test_not_found(client):
    """Non-existent route returns 404."""
    res = client.get("/nonexistent")
    assert res.status_code == 404
