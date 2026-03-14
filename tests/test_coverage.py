"""
Additional tests to cover missing lines:
  - src/retrieval/main.py       lines 158, 170-174, 263-283, 297-333, 341, 354-381
  - src/retrieval/hybrid.py     lines 164-188, 193
  - src/retrieval/retriever.py  lines 135, 149, 184
  - src/retrieval/loader.py     lines 140, 154

@author: Aarti Dashore
Seattle University, ARIN 5360
@version: 3.0.0+w26
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import retrieval.main as m
from retrieval.hybrid import HybridSearcher, explain_rrf_example
from retrieval.loader import DocumentChunker, DocumentLoader
from retrieval.main import app
from retrieval.rag import RAGSystem
from retrieval.retriever import DocumentRetriever


# ── Shared: no-lifespan test app ──────────────────────────
def _make_test_app() -> FastAPI:
    test_app = FastAPI()
    for route in app.routes:
        test_app.routes.append(route)
    return test_app


TEST_APP = _make_test_app()


@pytest.fixture(autouse=True)
def reset_globals():
    m.retriever = None
    m.rag_system = None
    m.llm_client = None
    yield
    m.retriever = None
    m.rag_system = None
    m.llm_client = None


@pytest.fixture
def mock_rag_system():
    rag = MagicMock(spec=RAGSystem)
    rag.query.return_value = {
        "question": "test",
        "answer": "test answer",
        "sources": [],
        "n_docs_retrieved": 0,
    }
    return rag


@pytest.fixture
def mock_retriever_fixture():
    r = MagicMock(spec=DocumentRetriever)
    r.search.return_value = []
    r.document_count = 5
    r._indexed = True
    r._all_documents = []
    r.use_hybrid = False
    r.use_reranking = True
    r.reranker = None
    r.hybrid_searcher = None
    return r


@pytest.fixture
def client(mock_rag_system, mock_retriever_fixture):
    m.retriever = mock_retriever_fixture
    m.rag_system = mock_rag_system
    m.llm_client = MagicMock()
    m.llm_client.is_available.return_value = True
    with TestClient(TEST_APP, raise_server_exceptions=False) as c:
        yield c


# ══════════════════════════════════════════════════════════
# main.py — line 158: _update_retriever_strategy with retriever None
# main.py — lines 170-174: use_reranking and use_hybrid branches
# ══════════════════════════════════════════════════════════


def test_update_retriever_strategy_none_retriever():
    """Line 158: returns early when retriever is None."""
    m.retriever = None
    m._update_retriever_strategy(use_hybrid=True, use_reranking=True)
    # no error = pass


def test_update_retriever_strategy_reranking_only(mock_retriever_fixture):
    m.retriever = mock_retriever_fixture
    with patch("retrieval.reranker.CrossEncoderReranker") as MockReranker:
        MockReranker.return_value = MagicMock()
        m._update_retriever_strategy(use_hybrid=False, use_reranking=True)
        assert mock_retriever_fixture.reranker is not None


def test_update_retriever_strategy_hybrid_no_docs(mock_retriever_fixture):
    m.retriever = mock_retriever_fixture
    mock_retriever_fixture._all_documents = []
    with patch("retrieval.hybrid.HybridSearcher") as MockHybrid:
        MockHybrid.return_value = MagicMock()
        m._update_retriever_strategy(use_hybrid=True, use_reranking=False)
        MockHybrid.return_value.index_documents.assert_not_called()


def test_update_retriever_strategy_hybrid_with_docs(mock_retriever_fixture):
    m.retriever = mock_retriever_fixture
    mock_retriever_fixture._all_documents = [{"id": "d1", "text": "hello", "metadata": {}}]
    with patch("retrieval.hybrid.HybridSearcher") as MockHybrid:
        mock_hybrid_instance = MagicMock()
        MockHybrid.return_value = mock_hybrid_instance
        m._update_retriever_strategy(use_hybrid=True, use_reranking=False)
        mock_hybrid_instance.index_documents.assert_called_once()


# ══════════════════════════════════════════════════════════
# main.py — lines 263-283: /rag timeout and RuntimeError handlers
# ══════════════════════════════════════════════════════════


@pytest.mark.anyio
async def test_rag_httpx_timeout_returns_graceful_response():
    """Lines 263-267: httpx.TimeoutException returns graceful RAGResponse."""
    mock_rag = MagicMock(spec=RAGSystem)
    mock_rag.query.side_effect = httpx.TimeoutException("timeout")
    m.rag_system = mock_rag
    m.retriever = MagicMock()

    req = m.RAGRequest(question="test question")
    resp = await m.rag_query(req)

    assert resp.question == "test question"
    assert "too long" in resp.answer.lower() or "timed out" in resp.answer.lower()
    assert resp.sources == []
    assert resp.n_docs_retrieved == 0


@pytest.mark.anyio
async def test_rag_runtime_error_timed_out_returns_graceful_response():
    """Lines 268-275: RuntimeError with 'timed out' returns graceful RAGResponse."""
    mock_rag = MagicMock(spec=RAGSystem)
    mock_rag.query.side_effect = RuntimeError("Request timed out after 180 seconds")
    m.rag_system = mock_rag
    m.retriever = MagicMock()

    req = m.RAGRequest(question="test question")
    resp = await m.rag_query(req)

    assert "timed out" in resp.answer.lower()
    assert resp.sources == []


@pytest.mark.anyio
async def test_rag_runtime_error_non_timeout_raises_503():
    """Lines 276-278: RuntimeError without 'timed out' raises HTTPException 503."""
    mock_rag = MagicMock(spec=RAGSystem)
    mock_rag.query.side_effect = RuntimeError("Connection refused")
    m.rag_system = mock_rag
    m.retriever = MagicMock()

    req = m.RAGRequest(question="test question")

    with pytest.raises(m.HTTPException) as exc:
        await m.rag_query(req)

    assert exc.value.status_code == 503


@pytest.mark.anyio
async def test_rag_generic_exception_raises_500():
    """Lines 279-281: Generic exception raises HTTPException 500."""
    mock_rag = MagicMock(spec=RAGSystem)
    mock_rag.query.side_effect = ValueError("unexpected error")
    m.rag_system = mock_rag
    m.retriever = MagicMock()

    req = m.RAGRequest(question="test question")

    with pytest.raises(m.HTTPException) as exc:
        await m.rag_query(req)

    assert exc.value.status_code == 500


# ══════════════════════════════════════════════════════════
# main.py — lines 297-333: /upload endpoint
# main.py — line 341: re-indexing after upload
# ══════════════════════════════════════════════════════════


def test_upload_valid_txt_file(client):
    """Lines 297-333: Upload a valid .txt file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch.object(m, "DOCUMENTS_DIR", tmpdir):
            data = {"file": ("test.txt", io.BytesIO(b"Hello world content"), "text/plain")}
            res = client.post("/upload", files=data)
    assert res.status_code == 200
    assert res.json()["filename"] == "test.txt"


def test_upload_valid_pdf_file(client):
    """Upload a valid .pdf file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch.object(m, "DOCUMENTS_DIR", tmpdir):
            data = {"file": ("test.pdf", io.BytesIO(b"%PDF-1.4 fake content"), "application/pdf")}
            res = client.post("/upload", files=data)
    assert res.status_code == 200
    assert res.json()["filename"] == "test.pdf"


def test_upload_invalid_extension_returns_400(client):
    """Upload unsupported file type returns 400."""
    data = {"file": ("test.exe", io.BytesIO(b"binary content"), "application/octet-stream")}
    res = client.post("/upload", files=data)
    assert res.status_code == 400


def test_upload_file_too_large_returns_413(client):
    """Upload file exceeding 10MB returns 413."""
    big_content = b"x" * (10 * 1024 * 1024 + 1)
    data = {"file": ("big.txt", io.BytesIO(big_content), "text/plain")}
    res = client.post("/upload", files=data)
    assert res.status_code == 413


def test_upload_triggers_reindex(client, mock_retriever_fixture):
    """Line 341: Upload re-indexes when retriever is set."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch.object(m, "DOCUMENTS_DIR", tmpdir):
            mock_retriever_fixture.store = MagicMock()
            mock_retriever_fixture.store.client = MagicMock()
            mock_retriever_fixture.store.embedder = MagicMock()
            mock_retriever_fixture.index_documents = MagicMock(return_value=1)
            data = {"file": ("test.txt", io.BytesIO(b"Hello"), "text/plain")}
            res = client.post("/upload", files=data)
    assert res.status_code == 200
    mock_retriever_fixture.index_documents.assert_called()


# ══════════════════════════════════════════════════════════
# main.py — lines 354-381: /documents list and delete
# ══════════════════════════════════════════════════════════


def test_list_documents_empty_dir(client):
    """Lines 354-360: /documents with empty dir returns empty list."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch.object(m, "DOCUMENTS_DIR", tmpdir):
            res = client.get("/documents")
    assert res.status_code == 200
    assert res.json()["count"] == 0


def test_list_documents_nonexistent_dir(client):
    """Lines 354-356: /documents with missing dir returns empty list."""
    with patch.object(m, "DOCUMENTS_DIR", "/nonexistent/path/xyz"):
        res = client.get("/documents")
    assert res.status_code == 200
    assert res.json()["documents"] == []


def test_list_documents_with_files(client):
    """Lines 354-381: /documents lists txt and pdf files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(tmpdir, "a.txt").write_text("hello")
        Path(tmpdir, "b.pdf").write_bytes(b"%PDF content")
        with patch.object(m, "DOCUMENTS_DIR", tmpdir):
            res = client.get("/documents")
    data = res.json()
    assert data["count"] == 2
    filenames = [d["filename"] for d in data["documents"]]
    assert "a.txt" in filenames
    assert "b.pdf" in filenames


def test_delete_document_not_found_returns_404(client):
    """Lines 354-381: Delete nonexistent file returns 404."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch.object(m, "DOCUMENTS_DIR", tmpdir):
            res = client.delete("/documents/nonexistent.txt")
    assert res.status_code == 404


def test_delete_document_success(client, mock_retriever_fixture):
    """Lines 354-381: Delete existing file returns 200."""
    with tempfile.TemporaryDirectory() as tmpdir:
        target = Path(tmpdir, "todelete.txt")
        target.write_text("content")
        mock_retriever_fixture.store = MagicMock()
        mock_retriever_fixture.store.client = MagicMock()
        mock_retriever_fixture.store.embedder = MagicMock()
        mock_retriever_fixture.index_documents = MagicMock(return_value=0)
        with patch.object(m, "DOCUMENTS_DIR", tmpdir):
            res = client.delete("/documents/todelete.txt")
    assert res.status_code == 200
    assert not target.exists()


def test_delete_document_triggers_reindex(client, mock_retriever_fixture):
    """Delete triggers re-indexing when retriever is set."""
    with tempfile.TemporaryDirectory() as tmpdir:
        target = Path(tmpdir, "todelete.txt")
        target.write_text("content")
        mock_retriever_fixture.store = MagicMock()
        mock_retriever_fixture.store.client = MagicMock()
        mock_retriever_fixture.store.embedder = MagicMock()
        mock_retriever_fixture.index_documents = MagicMock(return_value=0)
        with patch.object(m, "DOCUMENTS_DIR", tmpdir):
            client.delete("/documents/todelete.txt")
    mock_retriever_fixture.index_documents.assert_called()


# ══════════════════════════════════════════════════════════
# hybrid.py — lines 164-188, 193: explain_rrf_example()
# ══════════════════════════════════════════════════════════


def test_explain_rrf_example_runs(capsys):
    """Lines 164-188: explain_rrf_example prints expected output."""
    explain_rrf_example()
    out = capsys.readouterr().out
    assert "RRF" in out
    assert "doc_A" in out
    assert "doc_B" in out


def test_explain_rrf_example_shows_formula(capsys):
    """Line 193: explain_rrf_example shows the RRF formula."""
    explain_rrf_example()
    out = capsys.readouterr().out
    assert "1/" in out
    assert "60" in out


def test_hybrid_main_block(capsys):
    """Line 193: running hybrid.py as __main__ calls explain_rrf_example."""
    import runpy

    runpy.run_module("retrieval.hybrid", run_name="__main__")
    out = capsys.readouterr().out
    assert "RRF" in out


# ══════════════════════════════════════════════════════════
# retriever.py — line 135: _semantic_with_reranking fallback (no reranker)
# retriever.py — line 149: _hybrid_search no hybrid_searcher fallback
# retriever.py — line 184: get_configuration()
# ══════════════════════════════════════════════════════════


def test_semantic_with_reranking_no_reranker(tmp_path):
    """Line 135: _semantic_with_reranking falls back when reranker is None."""
    (tmp_path / "doc1.txt").write_text("Python is a programming language")
    r = DocumentRetriever(use_reranking=False, use_hybrid=False)
    r.reranker = None  # force no reranker
    r.use_reranking = True  # but flag says reranking
    r.index_documents(str(tmp_path))
    # Should still return results via fallback candidates[:n_results]
    results = r._semantic_with_reranking("Python", n_results=1)
    assert isinstance(results, list)


def test_hybrid_search_no_hybrid_searcher(tmp_path):
    """Line 149: _hybrid_search falls back to semantic when hybrid_searcher is None."""
    (tmp_path / "doc1.txt").write_text("Python is a programming language")
    r = DocumentRetriever(use_reranking=False, use_hybrid=True)
    r.hybrid_searcher = None  # force no hybrid searcher
    r.index_documents(str(tmp_path))
    results = r._hybrid_search("Python", n_results=1)
    assert isinstance(results, list)


def test_get_configuration(tmp_path):
    """Line 184: get_configuration returns correct dict."""
    (tmp_path / "doc1.txt").write_text("hello world")
    r = DocumentRetriever(use_reranking=False, use_hybrid=False)
    r.index_documents(str(tmp_path))
    config = r.get_configuration()
    assert config["use_reranking"] is False
    assert config["use_hybrid"] is False
    assert config["indexed"] is True
    assert config["document_count"] >= 1


def test_get_configuration_before_indexing():
    """get_configuration works before indexing."""
    r = DocumentRetriever(use_reranking=False, use_hybrid=False)
    config = r.get_configuration()
    assert config["indexed"] is False
    assert config["document_count"] == 0


# ══════════════════════════════════════════════════════════
# loader.py — line 140: empty text file returns []
# loader.py — line 154: empty PDF returns []
# ══════════════════════════════════════════════════════════


def test_load_empty_text_file_returns_empty(tmp_path):
    """Line 140: empty .txt file returns empty list."""
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("")
    loader = DocumentLoader()
    result = loader._load_text_file(empty_file)
    assert result == []


def test_load_empty_text_file_whitespace_only_returns_empty(tmp_path):
    """Line 140: whitespace-only .txt file returns empty list."""
    empty_file = tmp_path / "whitespace.txt"
    empty_file.write_text("   \n\n  ")
    loader = DocumentLoader()
    result = loader._load_text_file(empty_file)
    assert result == []


def test_load_pdf_empty_text_returns_empty(tmp_path):
    """Line 154: PDF with no extractable text returns empty list."""
    pdf_file = tmp_path / "empty.pdf"
    pdf_file.write_bytes(b"fake pdf content")

    loader = DocumentLoader()

    # Mock pypdf to return pages with no text
    mock_page = MagicMock()
    mock_page.extract_text.return_value = ""
    mock_reader = MagicMock()
    mock_reader.pages = [mock_page]

    with patch("retrieval.loader.pypdf.PdfReader", return_value=mock_reader):
        result = loader._load_pdf_file(pdf_file)

    assert result == []


def test_load_text_file_unreadable_returns_empty(tmp_path):
    """loader._load_text_file returns [] on exception."""
    fake_file = tmp_path / "bad.txt"
    fake_file.write_text("content")
    loader = DocumentLoader()
    with patch("builtins.open", side_effect=OSError("permission denied")):
        result = loader._load_text_file(fake_file)
    assert result == []


def test_load_pdf_unreadable_returns_empty(tmp_path):
    """loader._load_pdf_file returns [] on exception."""
    fake_file = tmp_path / "bad.pdf"
    fake_file.write_bytes(b"bad pdf")
    loader = DocumentLoader()
    with patch("retrieval.loader.pypdf.PdfReader", side_effect=Exception("corrupt")):
        result = loader._load_pdf_file(fake_file)
    assert result == []


def test_upload_reindex_exception_is_logged(client, mock_retriever_fixture):
    """Covers except branch in upload re-indexing block."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_retriever_fixture.store = MagicMock()
        mock_retriever_fixture.store.client.delete_collection.side_effect = Exception("db error")
        with patch.object(m, "DOCUMENTS_DIR", tmpdir):
            data = {"file": ("test.txt", io.BytesIO(b"Hello"), "text/plain")}
            res = client.post("/upload", files=data)
    # Should still return 200 — exception is caught and logged
    assert res.status_code == 200


def test_delete_reindex_exception_is_logged(client, mock_retriever_fixture):
    """Covers except branch in delete re-indexing block."""
    with tempfile.TemporaryDirectory() as tmpdir:
        target = Path(tmpdir, "todelete.txt")
        target.write_text("content")
        mock_retriever_fixture.store = MagicMock()
        mock_retriever_fixture.store.client.delete_collection.side_effect = Exception("db error")
        with patch.object(m, "DOCUMENTS_DIR", tmpdir):
            res = client.delete("/documents/todelete.txt")
    assert res.status_code == 200
