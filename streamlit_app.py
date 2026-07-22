"""
DocuMind — Streamlit Cloud Entry Point
Enterprise RAG System: Document Q&A with Source Citations.
Self-contained: imports backend modules directly (no HTTP to localhost).
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

import streamlit as st

# ── Ensure backend is importable ──────────────────────────
BACKEND_PATH = Path(__file__).resolve().parent / "backend"
sys.path.insert(0, str(BACKEND_PATH))

# ⚡ Preload embedding model on first import (sentence-transformers download)
os.environ.setdefault("LOG_LEVEL", "INFO")
os.environ.setdefault("CHROMA_PERSIST_DIR", str(Path(__file__).resolve().parent / "chroma_db"))
os.environ.setdefault("UPLOAD_DIR", str(Path(__file__).resolve().parent / "uploads"))

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("documind-streamlit")

# ── Import backend modules ────────────────────────────────
from config import (
    ALLOWED_EXTENSIONS,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL,
    MAX_FILE_SIZE_MB,
    UPLOAD_DIR,
    validate_config,
)
from database import (
    add_message,
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
from embeddings import get_embedding_model
from rag import generate_answer
from vectordb import add_document_chunks, delete_document_chunks, get_collection_stats


# ── Async helpers (Streamlit runs sync; backend uses asyncpg) ──
def _run_async(coro):
    """Run an async coroutine from sync Streamlit context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're inside an event loop (e.g., Streamlit's own)
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(coro)
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# ── Initialize backend on first load ──────────────────────
@st.cache_resource
def init_backend():
    """Initialize DB, embedding model, and upload dir. Called once."""
    Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    missing = validate_config()
    if missing:
        logger.warning("Missing env vars: %s. Some features may not work.", ", ".join(missing))
    # Preload embedding model
    try:
        get_embedding_model()
        logger.info("Embedding model loaded: %s", EMBEDDING_MODEL)
    except Exception as e:
        logger.warning("Failed to preload embedding model: %s", e)
    # Init DB
    try:
        _run_async(init_db())
        logger.info("Database initialized.")
    except Exception as e:
        logger.warning("Database init failed: %s. Running without DB persistence.", e)
    return True


_backend_ready = init_backend()


# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="DocuMind — Enterprise RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────
st.markdown(
    """
<style>
    .source-box {
        background-color: #f0f2f6;
        border-left: 3px solid #4CAF50;
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
        font-size: 0.85em;
    }
    .answer-box {
        padding: 15px;
        background-color: #ffffff;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
    }
    .stButton button { width: 100%; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Session state ─────────────────────────────────────────
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "documents" not in st.session_state:
    st.session_state.documents = []


# ── Direct backend helpers (replace HTTP API calls) ───────

def _health_check() -> dict:
    """Check system health."""
    try:
        stats = get_collection_stats()
        return {
            "status": "ok",
            "version": "1.0.0",
            "embedding_model": EMBEDDING_MODEL,
            "chroma_collection": CHROMA_COLLECTION_NAME,
            "total_chunks": stats.get("total_chunks", 0),
        }
    except Exception as e:
        return {"status": "degraded", "error": str(e)}


def _upload_document(file) -> Optional[dict]:
    """Upload a document directly (no HTTP)."""
    try:
        # Validate extension
        ext = Path(file.name).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            st.error(f"Unsupported file type: {ext}. Allowed: {sorted(ALLOWED_EXTENSIONS)}")
            return None

        # Validate size
        content = file.getvalue()
        file_size = len(content)
        max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
        if file_size > max_bytes:
            st.error(f"File too large: {file_size / 1024 / 1024:.1f}MB. Max: {MAX_FILE_SIZE_MB}MB")
            return None

        # Save to disk
        file_id = str(uuid.uuid4())
        safe_filename = f"{file_id}_{file.name}"
        file_path = Path(UPLOAD_DIR) / safe_filename
        with open(file_path, "wb") as fh:
            fh.write(content)

        file_type = ext.lstrip(".")
        doc_record = _run_async(create_document(
            filename=file.name,
            file_type=file_type,
            file_size=file_size,
            file_path=str(file_path),
        ))
        doc_id = str(doc_record["id"])

        # Process
        _run_async(update_document_status(doc_id, "processing"))
        chunks = process_document(file_path, doc_id, file.name)
        if chunks:
            added = add_document_chunks(chunks)
            _run_async(update_document_status(doc_id, "completed", chunks_count=added))
        else:
            _run_async(update_document_status(doc_id, "failed", chunks_count=0))

        updated = _run_async(get_document(doc_id))
        if not updated:
            st.error("Document not found after save.")
            return None

        return {
            "id": str(updated["id"]),
            "filename": updated["filename"],
            "file_type": updated["file_type"],
            "file_size": updated["file_size"],
            "status": updated["status"],
            "chunks_count": updated["chunks_count"],
            "created_at": str(updated["created_at"]),
            "updated_at": str(updated["updated_at"]),
        }
    except Exception as e:
        logger.exception("Upload failed: %s", e)
        st.error(f"Upload failed: {e}")
        return None


def _list_documents() -> list[dict]:
    """List all documents."""
    try:
        rows = _run_async(list_documents())
        return [
            {
                "id": str(r["id"]),
                "filename": r["filename"],
                "file_type": r["file_type"],
                "file_size": r["file_size"],
                "status": r["status"],
                "chunks_count": r["chunks_count"],
                "created_at": str(r["created_at"]),
                "updated_at": str(r["updated_at"]),
            }
            for r in rows
        ]
    except Exception as e:
        logger.error("Failed to list documents: %s", e)
        return []


def _delete_document(doc_id: str) -> bool:
    """Delete a document and its chunks."""
    try:
        doc = _run_async(get_document(doc_id))
        if not doc:
            return False
        chunks_removed = delete_document_chunks(doc_id)
        file_path = doc.get("file_path")
        if file_path and Path(file_path).exists():
            Path(file_path).unlink()
        _run_async(delete_document(doc_id))
        return True
    except Exception as e:
        logger.error("Failed to delete document: %s", e)
        return False


def _ask_question(
    question: str,
    session_id: Optional[str] = None,
    document_ids: Optional[list[str]] = None,
) -> Optional[dict]:
    """Run RAG pipeline directly."""
    try:
        # Create or use session
        if session_id:
            session = _run_async(get_session(session_id))
            if not session:
                session = _run_async(create_session())
            sid = str(session["id"])
        else:
            session = _run_async(create_session())
            sid = str(session["id"])

        # Save user message
        try:
            _run_async(add_message(session_id=sid, role="user", content=question))
        except Exception:
            pass

        # Run RAG
        answer, _, sources = _run_async(generate_answer(
            question=question,
            document_ids=document_ids,
        ))

        # Save assistant message
        try:
            msg = _run_async(add_message(
                session_id=sid, role="assistant", content=answer, sources=sources
            ))
            message_id = str(msg["id"])
        except Exception:
            message_id = str(uuid.uuid4())

        return {
            "answer": answer,
            "sources": sources,
            "session_id": sid,
            "message_id": message_id,
        }
    except Exception as e:
        logger.exception("RAG pipeline failed: %s", e)
        st.error(f"Failed to generate answer: {e}")
        return None


def _get_sessions() -> list[dict]:
    """List chat sessions."""
    try:
        rows = _run_async(list_sessions())
        return [
            {"id": str(r["id"]), "title": r["title"],
             "created_at": str(r["created_at"]), "updated_at": str(r["updated_at"])}
            for r in rows
        ]
    except Exception:
        return []


def _get_session_detail(session_id: str) -> Optional[dict]:
    """Get session with messages."""
    try:
        session = _run_async(get_session(session_id))
        if not session:
            return None
        msg_rows = _run_async(get_session_messages(session_id))
        messages = [
            {
                "id": str(r["id"]),
                "role": r["role"],
                "content": r["content"],
                "sources": json.loads(r["sources"]) if r.get("sources") else None,
                "created_at": str(r["created_at"]),
            }
            for r in msg_rows
        ]
        return {
            "id": str(session["id"]),
            "title": session["title"],
            "created_at": str(session["created_at"]),
            "updated_at": str(session["updated_at"]),
            "messages": messages,
        }
    except Exception:
        return None


# ── Sidebar — Document Management ─────────────────────────
with st.sidebar:
    st.title("📚 DocuMind")
    st.caption("Enterprise RAG System")

    # System status
    health = _health_check()
    if health.get("status") == "ok":
        st.success(f"🟢 System ready ({health.get('total_chunks', 0)} chunks)")
    else:
        st.warning(f"🟡 System degraded: {health.get('error', 'unknown')}")

    st.divider()

    # File upload
    st.subheader("📤 Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF, DOCX, or TXT file",
        type=["pdf", "docx", "txt"],
        help="Max file size: 25MB",
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        with st.spinner(f"Processing {uploaded_file.name}..."):
            result = _upload_document(uploaded_file)
            if result:
                st.success(f"✅ {result['filename']} — {result['chunks_count']} chunks")
                st.session_state.documents = _list_documents()
                st.rerun()

    st.divider()

    # Document list
    st.subheader("📄 Documents")

    if st.button("🔄 Refresh", use_container_width=True):
        st.session_state.documents = _list_documents()

    if not st.session_state.documents:
        st.session_state.documents = _list_documents()

    if not st.session_state.documents:
        st.info("No documents uploaded yet.")
    else:
        for doc in st.session_state.documents:
            status_icon = {
                "completed": "✅",
                "processing": "⏳",
                "pending": "🕐",
                "failed": "❌",
            }.get(doc.get("status", ""), "❓")

            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(
                    f"{status_icon} **{doc['filename']}**  \n"
                    f"<small>{doc['chunks_count']} chunks · {doc['status']}</small>",
                    unsafe_allow_html=True,
                )
            with col2:
                if st.button("🗑️", key=f"del_{doc['id']}", help="Delete document"):
                    if _delete_document(doc["id"]):
                        st.session_state.documents = _list_documents()
                        st.rerun()

    st.divider()

    # Chat history
    st.subheader("💬 Chat History")
    sessions = _get_sessions()
    for session in sessions:
        label = session.get("title", f"Chat {session['id'][:8]}")
        if st.button(
            f"📝 {label[:40]}",
            key=f"session_{session['id']}",
            use_container_width=True,
        ):
            st.session_state.current_session_id = session["id"]
            detail = _get_session_detail(session["id"])
            if detail:
                st.session_state.messages = [
                    {"role": m["role"], "content": m["content"], "sources": m.get("sources")}
                    for m in detail.get("messages", [])
                ]
            st.rerun()

    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.current_session_id = None
        st.session_state.messages = []
        st.rerun()


# ── Main Area — Chat Interface ────────────────────────────
st.title("💡 Document Q&A")
st.caption("Ask questions about your uploaded documents — answers with source citations")

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📎 View Sources", expanded=False):
                for idx, source in enumerate(msg["sources"], 1):
                    rel_score = source.get("relevance_score")
                    score_str = f"{rel_score:.3f}" if rel_score is not None else "N/A"
                    st.markdown(
                        f"""<div class="source-box">
<strong>[{idx}] {source.get('document_name', 'Unknown')}</strong>
<small>(score: {score_str})</small>
<br/><em>{source.get('text_snippet', '')[:250]}...</em>
</div>""",
                        unsafe_allow_html=True,
                    )

# Question input
if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    doc_ids = [d["id"] for d in st.session_state.documents if d.get("status") == "completed"]

    with st.chat_message("assistant"):
        with st.spinner("Searching documents & generating answer..."):
            result = _ask_question(
                question=prompt,
                session_id=st.session_state.current_session_id,
                document_ids=doc_ids if doc_ids else None,
            )

        if result:
            st.markdown(result["answer"])
            st.session_state.current_session_id = result["session_id"]

            if result.get("sources"):
                with st.expander("📎 View Sources", expanded=False):
                    for idx, source in enumerate(result["sources"], 1):
                        rel_score = source.get("relevance_score")
                        score_str = f"{rel_score:.3f}" if rel_score is not None else "N/A"
                        st.markdown(
                            f"""<div class="source-box">
<strong>[{idx}] {source.get('document_name', 'Unknown')}</strong>
<small>(score: {score_str})</small>
<br/><em>{source.get('text_snippet', '')[:250]}...</em>
</div>""",
                            unsafe_allow_html=True,
                        )

            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result.get("sources", []),
            })
        else:
            st.error("Failed to get an answer. Check logs for details.")
