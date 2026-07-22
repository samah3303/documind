"""
DocuMind FastAPI Backend
Enterprise RAG System — document Q&A with source citations.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import (
    ALLOWED_EXTENSIONS,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL,
    FRONTEND_URL,
    HOST,
    MAX_FILE_SIZE_MB,
    PORT,
    UPLOAD_DIR,
    validate_config,
)
from database import (
    add_message,
    close_db,
    create_document,
    create_session,
    delete_document,
    get_document,
    get_session,
    get_session_messages,
    init_db,
    list_documents,
    list_sessions,
    update_document_status,
)
from document_processor import process_document
from embeddings import get_embedding_model  # preload on startup
from models import (
    AnswerResponse,
    ChatMessageResponse,
    ChatSessionDetailResponse,
    ChatSessionResponse,
    DocumentDeleteResponse,
    DocumentListResponse,
    DocumentResponse,
    DocumentStatus,
    HealthResponse,
    QuestionRequest,
)
from rag import generate_answer
from vectordb import add_document_chunks, delete_document_chunks, get_collection_stats

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("documind")


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown handlers."""
    logger.info("🚀 DocuMind backend starting...")

    # Validate config
    missing = validate_config()
    if missing:
        logger.warning(
            "⚠️  Missing environment variables: %s. Some features may not work.",
            ", ".join(missing),
        )

    # Ensure upload directory exists
    Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

    # Preload embedding model
    logger.info("Preloading embedding model...")
    get_embedding_model()

    # Init database
    try:
        await init_db()
        logger.info("Database connected.")
    except Exception as exc:
        logger.warning("Database initialization failed: %s. Running without DB.", exc)

    logger.info("✅ DocuMind backend ready.")
    yield

    # Shutdown
    await close_db()
    logger.info("DocuMind backend shut down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="DocuMind API",
    description="Enterprise RAG System — Document Q&A with Source Citations",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL, "http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check with basic system info."""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        embedding_model=EMBEDDING_MODEL,
        chroma_collection=CHROMA_COLLECTION_NAME,
    )


# ---------------------------------------------------------------------------
# Document endpoints
# ---------------------------------------------------------------------------
@app.post("/api/documents/upload", response_model=DocumentResponse, tags=["Documents"])
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document (PDF, DOCX, TXT) for processing.

    The document is:
    1. Saved to the uploads directory
    2. Text is extracted and chunked
    3. Embeddings are generated
    4. Chunks stored in ChromaDB
    5. Metadata saved to PostgreSQL
    """
    # Validate file extension
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    # Validate file size
    content = await file.read()
    file_size = len(content)
    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    if file_size > max_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File too large: {file_size / 1024 / 1024:.1f}MB. Max: {MAX_FILE_SIZE_MB}MB",
        )

    # Save to disk
    file_id = str(uuid4())
    safe_filename = f"{file_id}_{file.filename}"
    file_path = Path(UPLOAD_DIR) / safe_filename

    with open(file_path, "wb") as fh:
        fh.write(content)

    # Create DB record
    file_type = ext.lstrip(".")
    try:
        doc_record = await create_document(
            filename=file.filename,
            file_type=file_type,
            file_size=file_size,
            file_path=str(file_path),
        )
    except Exception as exc:
        logger.error("Failed to create document record: %s", exc)
        file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to create document record in database.",
        )

    doc_id = str(doc_record["id"])

    # Process in background (mark as processing)
    try:
        await update_document_status(doc_id, DocumentStatus.PROCESSING.value)

        chunks = process_document(file_path, doc_id, file.filename)
        if chunks:
            added = add_document_chunks(chunks)
            await update_document_status(
                doc_id, DocumentStatus.COMPLETED.value, chunks_count=added
            )
        else:
            await update_document_status(
                doc_id, DocumentStatus.FAILED.value, chunks_count=0
            )
            logger.warning("No chunks produced for document %s", doc_id)
    except Exception as exc:
        logger.exception("Document processing failed: %s", exc)
        await update_document_status(doc_id, DocumentStatus.FAILED.value)
        # Don't raise — document is saved, just failed to process

    # Return updated record
    updated = await get_document(doc_id)
    if not updated:
        raise HTTPException(status_code=500, detail="Document not found after save.")

    return DocumentResponse(
        id=updated["id"],
        filename=updated["filename"],
        file_type=updated["file_type"],
        file_size=updated["file_size"],
        status=updated["status"],
        chunks_count=updated["chunks_count"],
        created_at=updated["created_at"],
        updated_at=updated["updated_at"],
    )


@app.get("/api/documents", response_model=DocumentListResponse, tags=["Documents"])
async def get_documents():
    """List all uploaded documents."""
    try:
        rows = await list_documents()
    except Exception as exc:
        logger.error("Failed to list documents: %s", exc)
        raise HTTPException(status_code=500, detail="Database error")

    docs = [
        DocumentResponse(
            id=r["id"],
            filename=r["filename"],
            file_type=r["file_type"],
            file_size=r["file_size"],
            status=r["status"],
            chunks_count=r["chunks_count"],
            created_at=r["created_at"],
            updated_at=r["updated_at"],
        )
        for r in rows
    ]
    return DocumentListResponse(documents=docs, total=len(docs))


@app.get(
    "/api/documents/{doc_id}", response_model=DocumentResponse, tags=["Documents"]
)
async def get_document_by_id(doc_id: str):
    """Get a single document by ID."""
    row = await get_document(doc_id)
    if not row:
        raise HTTPException(status_code=404, detail="Document not found.")

    return DocumentResponse(
        id=row["id"],
        filename=row["filename"],
        file_type=row["file_type"],
        file_size=row["file_size"],
        status=row["status"],
        chunks_count=row["chunks_count"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


@app.delete(
    "/api/documents/{doc_id}",
    response_model=DocumentDeleteResponse,
    tags=["Documents"],
)
async def delete_document_by_id(doc_id: str):
    """Delete a document and all its chunks from the system."""
    doc = await get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    # Delete from ChromaDB
    chunks_removed = delete_document_chunks(doc_id)

    # Delete uploaded file
    file_path = doc.get("file_path")
    if file_path and Path(file_path).exists():
        Path(file_path).unlink()

    # Delete from DB
    try:
        await delete_document(doc_id)
    except Exception as exc:
        logger.error("Failed to delete document from DB: %s", exc)

    return DocumentDeleteResponse(
        id=doc_id, deleted=True, chunks_removed=chunks_removed
    )


# ---------------------------------------------------------------------------
# Chat / Q&A endpoints
# ---------------------------------------------------------------------------
@app.post("/api/chat/ask", response_model=AnswerResponse, tags=["Chat"])
async def ask_question(request: QuestionRequest):
    """
    Ask a question against the uploaded documents.

    The RAG pipeline:
    1. Retrieves the most relevant document chunks from ChromaDB
    2. Augments the prompt with those chunks
    3. Queries DeepSeek API for an answer
    4. Returns the answer with source citations
    """
    # Create or use existing session
    if request.session_id:
        session = await get_session(str(request.session_id))
        if not session:
            session = await create_session()
        session_id = str(session["id"])
    else:
        session = await create_session()
        session_id = str(session["id"])

    # Save user message
    try:
        user_msg = await add_message(
            session_id=session_id,
            role="user",
            content=request.question,
        )
    except Exception as exc:
        logger.warning("Failed to save user message: %s", exc)

    # Run RAG pipeline
    doc_ids = [str(did) for did in request.document_ids] if request.document_ids else None
    try:
        answer, _, sources = await generate_answer(
            question=request.question,
            document_ids=doc_ids,
        )
    except Exception as exc:
        logger.exception("RAG pipeline failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate answer: {str(exc)}",
        )

    # Save assistant message with sources
    try:
        assistant_msg = await add_message(
            session_id=session_id,
            role="assistant",
            content=answer,
            sources=sources,
        )
        message_id = str(assistant_msg["id"])
    except Exception as exc:
        logger.warning("Failed to save assistant message: %s", exc)
        message_id = str(uuid4())

    return AnswerResponse(
        answer=answer,
        sources=sources,
        session_id=session_id,
        message_id=message_id,
    )


# ---------------------------------------------------------------------------
# Chat Session endpoints
# ---------------------------------------------------------------------------
@app.get(
    "/api/chat/sessions",
    response_model=list[ChatSessionResponse],
    tags=["Chat"],
)
async def get_sessions():
    """List all chat sessions."""
    try:
        rows = await list_sessions()
    except Exception:
        return []

    return [
        ChatSessionResponse(
            id=r["id"],
            title=r["title"],
            created_at=r["created_at"],
            updated_at=r["updated_at"],
        )
        for r in rows
    ]


@app.get(
    "/api/chat/sessions/{session_id}",
    response_model=ChatSessionDetailResponse,
    tags=["Chat"],
)
async def get_session_detail(session_id: str):
    """Get a chat session with all messages."""
    session = await get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    msg_rows = await get_session_messages(session_id)
    messages = [
        ChatMessageResponse(
            id=r["id"],
            role=r["role"],
            content=r["content"],
            sources=json.loads(r["sources"]) if r["sources"] else None,
            created_at=r["created_at"],
        )
        for r in msg_rows
    ]

    return ChatSessionDetailResponse(
        id=session["id"],
        title=session["title"],
        created_at=session["created_at"],
        updated_at=session["updated_at"],
        messages=messages,
    )


# ---------------------------------------------------------------------------
# Vector DB stats
# ---------------------------------------------------------------------------
@app.get("/api/stats", tags=["System"])
async def system_stats():
    """Get system statistics."""
    try:
        stats = get_collection_stats()
        return stats
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)
