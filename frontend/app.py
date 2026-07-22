"""
DocuMind — Streamlit Frontend
Enterprise RAG System: Document Q&A with Source Citations.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BACKEND_URL = "http://localhost:8000"

st.set_page_config(
    page_title="DocuMind — Enterprise RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
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
    .stButton button {
        width: 100%;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session State
# ---------------------------------------------------------------------------
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "documents" not in st.session_state:
    st.session_state.documents = []


# ---------------------------------------------------------------------------
# API Helpers
# ---------------------------------------------------------------------------
def api_health() -> bool:
    """Check if the backend is reachable."""
    try:
        resp = requests.get(f"{BACKEND_URL}/api/health", timeout=5)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def api_upload_document(file) -> Optional[dict]:
    """Upload a document to the backend."""
    try:
        files = {"file": (file.name, file.getvalue(), file.type or "application/octet-stream")}
        resp = requests.post(
            f"{BACKEND_URL}/api/documents/upload",
            files=files,
            timeout=120,
        )
        if resp.status_code == 200:
            return resp.json()
        else:
            st.error(f"Upload failed: {resp.json().get('detail', resp.text)}")
            return None
    except requests.RequestException as e:
        st.error(f"Backend connection error: {e}")
        return None


def api_list_documents() -> list[dict]:
    """Fetch all documents from the backend."""
    try:
        resp = requests.get(f"{BACKEND_URL}/api/documents", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("documents", [])
        return []
    except requests.RequestException:
        return []


def api_delete_document(doc_id: str) -> bool:
    """Delete a document by ID."""
    try:
        resp = requests.delete(f"{BACKEND_URL}/api/documents/{doc_id}", timeout=10)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def api_ask_question(
    question: str,
    session_id: Optional[str] = None,
    document_ids: Optional[list[str]] = None,
) -> Optional[dict]:
    """Send a question to the RAG pipeline."""
    payload = {"question": question}
    if session_id:
        payload["session_id"] = session_id
    if document_ids:
        payload["document_ids"] = document_ids

    try:
        resp = requests.post(
            f"{BACKEND_URL}/api/chat/ask",
            json=payload,
            timeout=120,
        )
        if resp.status_code == 200:
            return resp.json()
        else:
            st.error(f"Query failed: {resp.json().get('detail', resp.text)}")
            return None
    except requests.RequestException as e:
        st.error(f"Backend connection error: {e}")
        return None


def api_get_sessions() -> list[dict]:
    """Fetch chat sessions."""
    try:
        resp = requests.get(f"{BACKEND_URL}/api/chat/sessions", timeout=10)
        if resp.status_code == 200:
            return resp.json()
        return []
    except requests.RequestException:
        return []


def api_get_session_messages(session_id: str) -> Optional[dict]:
    """Fetch messages for a session."""
    try:
        resp = requests.get(
            f"{BACKEND_URL}/api/chat/sessions/{session_id}", timeout=10
        )
        if resp.status_code == 200:
            return resp.json()
        return None
    except requests.RequestException:
        return None


# ---------------------------------------------------------------------------
# Sidebar — Document Management
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📚 DocuMind")
    st.caption("Enterprise RAG System")

    # Backend status
    backend_online = api_health()
    if backend_online:
        st.success("🟢 Backend connected")
    else:
        st.error("🔴 Backend offline — start the FastAPI server")
        st.stop()

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
            result = api_upload_document(uploaded_file)
            if result:
                st.success(f"✅ {result['filename']} — {result['chunks_count']} chunks")
                st.session_state.documents = api_list_documents()
                st.rerun()

    st.divider()

    # Document list
    st.subheader("📄 Documents")

    if st.button("🔄 Refresh", use_container_width=True):
        st.session_state.documents = api_list_documents()

    if not st.session_state.documents:
        st.session_state.documents = api_list_documents()

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
                    if api_delete_document(doc["id"]):
                        st.session_state.documents = api_list_documents()
                        st.rerun()

    st.divider()

    # Chat history
    st.subheader("💬 Chat History")
    sessions = api_get_sessions()
    for session in sessions:
        label = session.get("title", f"Chat {session['id'][:8]}")
        if st.button(
            f"📝 {label[:40]}",
            key=f"session_{session['id']}",
            use_container_width=True,
        ):
            st.session_state.current_session_id = session["id"]
            detail = api_get_session_messages(session["id"])
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


# ---------------------------------------------------------------------------
# Main Area — Chat Interface
# ---------------------------------------------------------------------------
st.title("💡 Document Q&A")
st.caption("Ask questions about your uploaded documents — answers with source citations")

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show sources for assistant messages
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📎 View Sources", expanded=False):
                for idx, source in enumerate(msg["sources"], 1):
                    st.markdown(
                        f"""<div class="source-box">
<strong>[{idx}] {source.get('document_name', 'Unknown')}</strong>
<small>(score: {source.get('relevance_score', 'N/A'):.3f})</small>
<br/><em>{source.get('text_snippet', '')[:250]}...</em>
</div>""",
                        unsafe_allow_html=True,
                    )

# Question input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Get answer
    doc_ids = [d["id"] for d in st.session_state.documents if d.get("status") == "completed"]

    with st.chat_message("assistant"):
        with st.spinner("Searching documents & generating answer..."):
            result = api_ask_question(
                question=prompt,
                session_id=st.session_state.current_session_id,
                document_ids=doc_ids if doc_ids else None,
            )

        if result:
            st.markdown(result["answer"])
            st.session_state.current_session_id = result["session_id"]

            # Show sources
            if result.get("sources"):
                with st.expander("📎 View Sources", expanded=False):
                    for idx, source in enumerate(result["sources"], 1):
                        st.markdown(
                            f"""<div class="source-box">
<strong>[{idx}] {source.get('document_name', 'Unknown')}</strong>
<small>(score: {source.get('relevance_score', 'N/A'):.3f})</small>
<br/><em>{source.get('text_snippet', '')[:250]}...</em>
</div>""",
                            unsafe_allow_html=True,
                        )

            # Save to session state
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result.get("sources", []),
                }
            )
        else:
            st.error("Failed to get an answer. Check backend logs.")
