"""
P3 FastAPI application — RAG Chatbot with advanced features.

@author: Aarti Dashore
Seattle University, ARIN 5360
@see: https://catalog.seattleu.edu/preview_course_nopop.php?catoid=55&coid=190380
@version: 3.0.0+w26
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

from retrieval.llm import LLMClient
from retrieval.rag import RAGSystem
from retrieval.retriever import DocumentRetriever

# ── Config from environment ────────────────────────────────
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:3b")
LLM_API_KEY = os.getenv("LLM_API_KEY", None)
DEFAULT_CONTEXT_DOCS = int(os.getenv("DEFAULT_CONTEXT_DOCS", "3"))
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
MAX_CONTEXT_DOCS = int(os.getenv("MAX_CONTEXT_DOCS", "10"))
DOCUMENTS_DIR = os.getenv("DOCUMENTS_DIR", "documents")
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "180.0"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Globals ────────────────────────────────────────────────
retriever: DocumentRetriever | None = None
rag_system: RAGSystem | None = None
llm_client: LLMClient | None = None


# ── Lifespan ───────────────────────────────────────────────
@asynccontextmanager
async def lifespan(_app: FastAPI):
    global retriever, rag_system, llm_client
    try:
        logger.info("Initializing retriever...")
        retriever = DocumentRetriever(use_reranking=True, use_hybrid=False)
        num_docs = retriever.index_documents(DOCUMENTS_DIR)
        logger.info(f"Indexed {num_docs} document chunks")

        logger.info("Initializing LLM client...")
        llm_client = LLMClient(
            base_url=LLM_BASE_URL,
            model=LLM_MODEL,
            api_key=LLM_API_KEY,
            timeout=float(os.getenv("LLM_TIMEOUT", "180.0")),
        )

        logger.info("Initializing RAG system...")
        rag_system = RAGSystem(
            retriever=retriever,
            llm_client=llm_client,
            n_context_docs=DEFAULT_CONTEXT_DOCS,
        )
        logger.info("RAG system ready")

    except Exception as e:
        logger.error(f"Startup error: {e}")

    yield

    logger.info("Shutting down...")


# ── App ────────────────────────────────────────────────────
app = FastAPI(
    title="Galactic Gadgets RAG Assistant",
    description="RAG chatbot combining semantic search with LLM generation",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ─────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str
    documents_indexed: int
    message: str
    llm_available: bool


class SearchRequest(BaseModel):
    query: str
    n_results: int = 5


class SearchResponse(BaseModel):
    query: str
    results: list[dict]
    count: int


class ConversationTurn(BaseModel):
    question: str
    answer: str


class RAGRequest(BaseModel):
    question: str
    n_context_docs: int = DEFAULT_CONTEXT_DOCS
    temperature: float = DEFAULT_TEMPERATURE
    use_hybrid: bool = False
    use_reranking: bool = True
    conversation_history: list[ConversationTurn] = []


class RAGResponse(BaseModel):
    question: str
    answer: str
    sources: list[dict]
    n_docs_retrieved: int


class DocumentListResponse(BaseModel):
    documents: list[dict]
    count: int


# ── Helpers ────────────────────────────────────────────────
def _check_retriever() -> DocumentRetriever:
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")
    return retriever


def _check_rag() -> RAGSystem:
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    return rag_system


def _update_retriever_strategy(use_hybrid: bool, use_reranking: bool) -> None:
    """Dynamically update retriever search strategy flags."""
    if retriever is None:
        return
    retriever.use_hybrid = use_hybrid
    retriever.use_reranking = use_reranking
    retriever.reranker = None
    retriever.hybrid_searcher = None

    if use_reranking:
        from retrieval.reranker import CrossEncoderReranker

        retriever.reranker = CrossEncoderReranker()

    if use_hybrid:
        from retrieval.hybrid import HybridSearcher

        retriever.hybrid_searcher = HybridSearcher()
        if retriever._all_documents:
            retriever.hybrid_searcher.index_documents(retriever._all_documents)


# ── Endpoints ──────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check with LLM availability status."""
    is_llm_up = llm_client.is_available() if llm_client else False
    if retriever is None:
        return HealthResponse(
            status="healthy",
            documents_indexed=0,
            message="API running; retriever not initialized",
            llm_available=is_llm_up,
        )
    return HealthResponse(
        status="healthy",
        documents_indexed=retriever.document_count,
        message="API running and ready",
        llm_available=is_llm_up,
    )


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Semantic search endpoint (from P2)."""
    r = _check_retriever()

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if request.n_results < 1 or request.n_results > 20:
        raise HTTPException(status_code=400, detail="n_results must be between 1 and 20")

    try:
        results = r.search(request.query, request.n_results)
        return SearchResponse(query=request.query, results=results, count=len(results))
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail="Search failed") from e


@app.post("/rag", response_model=RAGResponse)
async def rag_query(request: RAGRequest):
    """
    RAG endpoint: retrieve + generate answer with conversation history.

    Option A: Includes conversation_history for multi-turn memory.
    Option D: Accepts use_hybrid and use_reranking flags.
    """
    r = _check_rag()

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    if request.n_context_docs < 1 or request.n_context_docs > MAX_CONTEXT_DOCS:
        raise HTTPException(
            status_code=400,
            detail=f"n_context_docs must be between 1 and {MAX_CONTEXT_DOCS}",
        )
    if not 0.0 <= request.temperature <= 1.0:
        raise HTTPException(status_code=400, detail="temperature must be between 0.0 and 1.0")

    # Option D: update search strategy dynamically
    _update_retriever_strategy(request.use_hybrid, request.use_reranking)

    # Option A: build system prompt with conversation history
    system_prompt = None
    if request.conversation_history:
        history_text = "\n".join(
            [f"Q: {turn.question}\nA: {turn.answer}" for turn in request.conversation_history]
        )
        system_prompt = (
            "You are a helpful AI assistant that answers questions based on provided context documents.\n\n"
            "Guidelines:\n"
            "1. Base your answers primarily on the provided context.\n"
            "2. If the context doesn't fully answer the question, acknowledge this.\n"
            "3. Be concise but thorough.\n"
            "4. Cite document numbers when referencing information (e.g., 'According to Document 1...').\n"
            "5. Use a friendly, professional tone.\n\n"
            f"Previous conversation:\n{history_text}\n\n"
            "Use the conversation history above for context when answering follow-up questions."
        )

    try:
        result = r.query(
            question=request.question,
            n_results=request.n_context_docs,
            temperature=request.temperature,
            system_prompt=system_prompt,
        )
    except httpx.TimeoutException:
        logger.error("LLM request timed out")
        return RAGResponse(
            question=request.question,
            answer="I'm taking too long to respond. Please try again or check if Ollama is running.",
            sources=[],
            n_docs_retrieved=0,
        )
    except RuntimeError as e:
        logger.error(f"LLM error: {e}")
        if "timed out" in str(e).lower():
            return RAGResponse(
                question=request.question,
                answer="The LLM timed out. Please try again.",
                sources=[],
                n_docs_retrieved=0,
            )
        raise HTTPException(status_code=503, detail=f"LLM error: {e}") from e
    except Exception as e:
        logger.error(f"RAG error: {e}")
        raise HTTPException(status_code=500, detail="RAG query failed") from e

    return RAGResponse(
        question=result["question"],
        answer=result["answer"],
        sources=result.get("sources", []),
        n_docs_retrieved=result.get("n_docs_retrieved", 0),
    )


# ── Option E: Document Upload & Management ─────────────────
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a .txt or .pdf file and re-index documents."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    if not file.filename.lower().endswith((".txt", ".pdf")):
        raise HTTPException(status_code=400, detail="Only .txt and .pdf files are supported")

    # Read and check size
    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="File exceeds 10MB limit")

    docs_path = Path(DOCUMENTS_DIR)
    docs_path.mkdir(exist_ok=True)
    dest = docs_path / file.filename

    try:
        with open(dest, "wb") as f:
            f.write(content)
        logger.info(f"Uploaded: {file.filename}")
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}") from e

    # Re-index
    if retriever is not None:
        try:
            retriever.store.client.delete_collection("documents")
            retriever.store.collection = retriever.store.client.create_collection(
                name="documents",
                embedding_function=retriever.store.embedder,
            )
            retriever._indexed = False
            num = retriever.index_documents(DOCUMENTS_DIR)
            logger.info(f"Re-indexed {num} chunks after upload")
        except Exception as e:
            logger.error(f"Re-indexing failed: {e}")

    return {"filename": file.filename, "message": "Uploaded and indexed successfully"}


@app.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    """List all documents in the documents directory."""
    docs_path = Path(DOCUMENTS_DIR)
    if not docs_path.exists():
        return DocumentListResponse(documents=[], count=0)

    docs = []
    for f in sorted(docs_path.iterdir()):
        if f.suffix.lower() in (".txt", ".pdf") and f.is_file():
            docs.append({"filename": f.name, "size": f.stat().st_size, "type": f.suffix[1:]})

    return DocumentListResponse(documents=docs, count=len(docs))


@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    """Delete a document and re-index."""
    docs_path = Path(DOCUMENTS_DIR)
    target = docs_path / filename

    if not target.exists():
        raise HTTPException(status_code=404, detail="File not found")
    if not target.is_file():
        raise HTTPException(status_code=400, detail="Not a file")

    try:
        target.unlink()
        logger.info(f"Deleted: {filename}")
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete: {e}") from e

    # Re-index
    if retriever is not None:
        try:
            retriever.store.client.delete_collection("documents")
            retriever.store.collection = retriever.store.client.create_collection(
                name="documents",
                embedding_function=retriever.store.embedder,
            )
            retriever._indexed = False
            retriever.index_documents(DOCUMENTS_DIR)
        except Exception as e:
            logger.error(f"Re-indexing after delete failed: {e}")

    return {"filename": filename, "message": "Deleted and re-indexed successfully"}


# ── Error handlers ─────────────────────────────────────────
@app.exception_handler(Exception)
async def general_exception_handler(_request, exc):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/test/error")
async def test_error():
    raise RuntimeError("Something went wrong")


# ── Static + UI ────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def ui():
    return FileResponse("static/index.html")


if __name__ == "__main__":
    print("To run this application:")
    print("uv run uvicorn src.retrieval.main:app --reload")
    print("\nThen open: http://localhost:8000")
