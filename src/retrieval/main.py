"""
P2 FastAPI API.

@author: Aarti Dashore
Seattle University, ARIN 5360
@see: https://catalog.seattleu.edu/preview_course_nopop.php?catoid=55&coid=190380
@version: 2.0.0+w26
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from src.retrieval.retriever import DocumentRetriever

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global retriever instance
retriever = None


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    documents_indexed: int
    message: str


class SearchRequest(BaseModel):
    """Request model for search."""

    query: str
    n_results: int = 5


class SearchResponse(BaseModel):
    """Response model for search."""

    query: str
    results: list[dict]
    count: int


# Define lifespan function to load models on startup
@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Code before the 'yield' is executed during application startup
    try:
        logger.info("Loading models...")

        # Index documents from the documents/ directory
        global retriever
        retriever = DocumentRetriever()
        num_docs = retriever.index_documents("documents")
        logger.info(f"Indexed {num_docs} documents successfully!")
    except Exception as e:
        # Don't crash the server, but log the error
        logger.error(f"Failed to load model: {str(e)}")

    yield  # The application starts receiving requests after the yield

    # Code after the 'yield' is executed during application shutdown
    logger.info("Application shutting down (lifespan)...")


# Initialize FastAPI app
app = FastAPI(
    title="FIXME: API Title",
    description="Lab3: Semantic search system using ChromaDB and sentence transformers",
    version="1.0.0",
    lifespan=lifespan,
)

# Add cross-origin resource sharing (CORS) middleware
# (gives browser permission to call our API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Search for documents relevant to the query.

    Args:
        request: SearchRequest with query and optional n_results

    Returns:
        SearchResponse with results
    """
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if request.n_results < 1 or request.n_results > 20:
        raise HTTPException(status_code=400, detail="n_results must be between 1 and 20")

    try:
        results = retriever.search(request.query, request.n_results)
        return SearchResponse(query=request.query, results=results, count=len(results))
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail="Search failed") from e


# Implement health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    if retriever is None:
        return HealthResponse(
            status="healthy",  # ← CHANGE HERE
            message="API is running; retriever not initialized yet",
            documents_indexed=0,
        )

    return HealthResponse(
        status="healthy",
        message="API is running and ready",
        documents_indexed=retriever.document_count,
    )


# Add error handler for general exceptions
@app.exception_handler(Exception)
async def general_exception_handler(_request, exc):
    """Handle unexpected exceptions."""
    logger.error(f"Unexpected error: {str(exc)}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# Create a test endpoint that raises exceptions (only for testing!)
@app.get("/test/error")
async def test_error():
    raise RuntimeError("Something went wrong")


# Mount static files LAST - catches all remaining routes
# including / --> /static/index.html, and
#           /stlye.css --> /static/style.csscan
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def ui():
    return FileResponse("static/index.html")


if __name__ == "__main__":
    print("To run this application:")
    print("uv run uvicorn src.retrieval.main:app --reload")
    print("\nThen open: http://localhost:8000")
