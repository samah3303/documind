"""
DocuMind Vector Database Layer
ChromaDB operations: add documents, query by similarity, delete documents.
"""

from __future__ import annotations

import logging
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from config import CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIR, CHROMA_TOP_K
from document_processor import DocumentChunk
from embeddings import generate_embeddings

logger = logging.getLogger(__name__)

# Global ChromaDB client — persistent, one per process
_client: Optional[chromadb.PersistentClient] = None


def _get_client() -> chromadb.PersistentClient:
    """Return the persistent ChromaDB client, creating it on first call."""
    global _client
    if _client is None:
        logger.info("Initializing ChromaDB at %s", CHROMA_PERSIST_DIR)
        _client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _client


def get_collection() -> chromadb.Collection:
    """Return the ChromaDB collection, creating it if it doesn't exist."""
    client = _get_client()
    try:
        collection = client.get_collection(CHROMA_COLLECTION_NAME)
    except Exception:
        collection = client.create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return collection


def add_document_chunks(chunks: list[DocumentChunk]) -> int:
    """
    Embed and store document chunks in ChromaDB.

    Args:
        chunks: List of DocumentChunk objects.

    Returns:
        Number of chunks added.
    """
    if not chunks:
        logger.warning("add_document_chunks called with empty list")
        return 0

    collection = get_collection()
    texts = [chunk.text for chunk in chunks]
    embeddings = generate_embeddings(texts)
    ids = [chunk.id for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]

    # Batch by 100 to avoid overwhelming the DB
    batch_size = 100
    total_added = 0
    for i in range(0, len(ids), batch_size):
        batch_slice = slice(i, i + batch_size)
        collection.add(
            ids=ids[batch_slice],
            embeddings=embeddings[batch_slice],
            documents=texts[batch_slice],
            metadatas=metadatas[batch_slice],
        )
        total_added += len(ids[batch_slice])

    logger.info("Added %d chunks to ChromaDB", total_added)
    return total_added


def query_similar_chunks(
    query: str,
    top_k: int = CHROMA_TOP_K,
    document_ids: Optional[list[str]] = None,
) -> list[dict]:
    """
    Find chunks semantically similar to the query.

    Args:
        query: The user's question.
        top_k: Number of chunks to retrieve.
        document_ids: Optional list of document UUIDs to filter by.

    Returns:
        List of dicts with keys: id, text, metadata, distance.
    """
    collection = get_collection()
    query_embedding = generate_embeddings([query])

    where_filter: Optional[dict] = None
    if document_ids:
        # ChromaDB where clause for filtering by document_id
        where_filter = {"document_id": {"$in": document_ids}}

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    if not results["ids"] or not results["ids"][0]:
        return []

    chunks: list[dict] = []
    for idx in range(len(results["ids"][0])):
        chunks.append(
            {
                "id": results["ids"][0][idx],
                "text": results["documents"][0][idx],
                "metadata": results["metadatas"][0][idx],
                "distance": results["distances"][0][idx],
            }
        )

    return chunks


def delete_document_chunks(document_id: str) -> int:
    """
    Delete all chunks belonging to a document from ChromaDB.

    Args:
        document_id: UUID of the document.

    Returns:
        Number of chunks deleted.
    """
    collection = get_collection()
    # Get all chunks for this document
    try:
        existing = collection.get(
            where={"document_id": document_id},
            include=[],
        )
    except Exception:
        logger.warning("No chunks found for document %s", document_id)
        return 0

    chunk_ids = existing.get("ids", [])
    if not chunk_ids:
        return 0

    collection.delete(ids=chunk_ids)
    logger.info("Deleted %d chunks for document %s", len(chunk_ids), document_id)
    return len(chunk_ids)


def get_collection_stats() -> dict:
    """Return basic statistics about the ChromaDB collection."""
    collection = get_collection()
    count = collection.count()
    return {"collection_name": CHROMA_COLLECTION_NAME, "total_chunks": count}
