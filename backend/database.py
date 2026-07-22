"""
DocuMind Database Layer
Neon PostgreSQL operations using asyncpg for document metadata and chat history.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

import asyncpg

from config import DATABASE_URL, DB_MAX_CONNECTIONS, DB_MIN_CONNECTIONS

logger = logging.getLogger(__name__)

# Global connection pool
_pool: Optional[asyncpg.Pool] = None

# SQL to create tables
CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename VARCHAR(500) NOT NULL,
    file_type VARCHAR(20) NOT NULL,
    file_size BIGINT DEFAULT 0,
    status VARCHAR(20) DEFAULT 'pending',
    chunks_count INTEGER DEFAULT 0,
    file_path VARCHAR(1000),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chat_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(500) DEFAULT 'New Chat',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    sources JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_session
    ON chat_messages(session_id, created_at);

CREATE INDEX IF NOT EXISTS idx_documents_status
    ON documents(status);
"""


async def init_db() -> None:
    """Initialize the connection pool and create tables."""
    global _pool
    if _pool is not None:
        return

    logger.info("Connecting to PostgreSQL database...")
    _pool = await asyncpg.create_pool(
        dsn=DATABASE_URL,
        min_size=DB_MIN_CONNECTIONS,
        max_size=DB_MAX_CONNECTIONS,
    )

    async with _pool.acquire() as conn:
        await conn.execute(CREATE_TABLES_SQL)
        logger.info("Database tables verified/created.")


async def close_db() -> None:
    """Close the connection pool gracefully."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("Database connection pool closed.")


def get_pool() -> asyncpg.Pool:
    """Return the active connection pool."""
    if _pool is None:
        raise RuntimeError("Database pool not initialized. Call init_db() first.")
    return _pool


# ---------------------------------------------------------------------------
# Document CRUD
# ---------------------------------------------------------------------------

async def create_document(
    filename: str,
    file_type: str,
    file_size: int,
    file_path: str,
) -> asyncpg.Record:
    """Insert a new document record and return it."""
    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO documents (filename, file_type, file_size, file_path, status)
            VALUES ($1, $2, $3, $4, 'pending')
            RETURNING *
            """,
            filename,
            file_type,
            file_size,
            file_path,
        )
        return row


async def update_document_status(
    doc_id: str,
    status: str,
    chunks_count: int = 0,
) -> asyncpg.Record:
    """Update document processing status."""
    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            UPDATE documents
            SET status = $2, chunks_count = $3, updated_at = NOW()
            WHERE id = $1::uuid
            RETURNING *
            """,
            doc_id,
            status,
            chunks_count,
        )
        return row


async def get_document(doc_id: str) -> Optional[asyncpg.Record]:
    """Get a single document by ID."""
    pool = get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchrow(
            "SELECT * FROM documents WHERE id = $1::uuid", doc_id
        )


async def list_documents() -> list[asyncpg.Record]:
    """List all documents ordered by creation date (newest first)."""
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM documents ORDER BY created_at DESC"
        )
        return rows


async def delete_document(doc_id: str) -> bool:
    """Delete a document and its associated chat messages (via session cascade)."""
    pool = get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM documents WHERE id = $1::uuid", doc_id
        )
        deleted = "DELETE 1" in result
        return deleted


# ---------------------------------------------------------------------------
# Chat Sessions
# ---------------------------------------------------------------------------

async def create_session(title: str = "New Chat") -> asyncpg.Record:
    """Create a new chat session."""
    pool = get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchrow(
            "INSERT INTO chat_sessions (title) VALUES ($1) RETURNING *",
            title,
        )


async def get_session(session_id: str) -> Optional[asyncpg.Record]:
    """Get a chat session by ID."""
    pool = get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchrow(
            "SELECT * FROM chat_sessions WHERE id = $1::uuid", session_id
        )


async def list_sessions() -> list[asyncpg.Record]:
    """List all chat sessions ordered by most recent first."""
    pool = get_pool()
    async with pool.acquire() as conn:
        return await conn.fetch(
            "SELECT * FROM chat_sessions ORDER BY updated_at DESC"
        )


async def update_session_title(session_id: str, title: str) -> None:
    """Update a session's title."""
    pool = get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE chat_sessions SET title = $2, updated_at = NOW() WHERE id = $1::uuid",
            session_id,
            title,
        )


async def delete_session(session_id: str) -> bool:
    """Delete a chat session (cascades to messages)."""
    pool = get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM chat_sessions WHERE id = $1::uuid", session_id
        )
        return "DELETE 1" in result


# ---------------------------------------------------------------------------
# Chat Messages
# ---------------------------------------------------------------------------

async def add_message(
    session_id: str,
    role: str,
    content: str,
    sources: Optional[list[dict]] = None,
) -> asyncpg.Record:
    """Add a chat message to a session."""
    pool = get_pool()
    sources_json = json.dumps(sources) if sources else None
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO chat_messages (session_id, role, content, sources)
            VALUES ($1::uuid, $2, $3, $4::jsonb)
            RETURNING *
            """,
            session_id,
            role,
            content,
            sources_json,
        )
        # Touch session updated_at
        await conn.execute(
            "UPDATE chat_sessions SET updated_at = NOW() WHERE id = $1::uuid",
            session_id,
        )
        # Auto-title session from first user message
        await _auto_title_session(conn, session_id)

        return row


async def _auto_title_session(
    conn: asyncpg.Connection, session_id: str
) -> None:
    """Set session title to the first 50 chars of the first user message if still default."""
    existing = await conn.fetchrow(
        "SELECT title FROM chat_sessions WHERE id = $1::uuid", session_id
    )
    if existing and existing["title"] == "New Chat":
        first_msg = await conn.fetchrow(
            "SELECT content FROM chat_messages WHERE session_id = $1::uuid AND role = 'user' ORDER BY created_at LIMIT 1",
            session_id,
        )
        if first_msg:
            title = first_msg["content"][:60].replace("\n", " ")
            await conn.execute(
                "UPDATE chat_sessions SET title = $2 WHERE id = $1::uuid",
                session_id,
                title,
            )


async def get_session_messages(session_id: str) -> list[asyncpg.Record]:
    """Get all messages for a chat session, ordered by creation time."""
    pool = get_pool()
    async with pool.acquire() as conn:
        return await conn.fetch(
            """
            SELECT * FROM chat_messages
            WHERE session_id = $1::uuid
            ORDER BY created_at ASC
            """,
            session_id,
        )
