"""
DocuMind RAG Pipeline
Retrieve relevant chunks → augment prompt → generate answer via DeepSeek.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import httpx

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MAX_TOKENS,
    DEEPSEEK_MODEL,
    DEEPSEEK_TEMPERATURE,
    CHROMA_TOP_K,
)
from vectordb import query_similar_chunks

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are DocuMind, an enterprise document Q&A assistant.
Your task is to answer questions based **only** on the provided document excerpts.

Rules:
1. If the answer is found in the excerpts, provide a clear, concise answer and cite the source numbers in brackets, e.g., [1], [2].
2. If the excerpts contain partial information, say so and answer what you can.
3. If the excerpts do not contain enough information to answer, say:
   "I couldn't find enough information in the uploaded documents to answer this question."
4. Do not use prior knowledge. Only use the provided excerpts.
5. Format your answer in Markdown for readability."""


def _build_prompt(question: str, chunks: list[dict]) -> str:
    """Build the augmented prompt with retrieved context chunks."""
    context_parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        filename = chunk["metadata"].get("filename", "Unknown")
        context_parts.append(f"[{i}] Source: {filename}\n{chunk['text']}")

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""## Document Excerpts

{context}

---

## User Question

{question}

Please provide a detailed answer with source citations in brackets, e.g., [1], [2]."""
    return prompt


async def generate_answer(
    question: str,
    document_ids: Optional[list[str]] = None,
    top_k: int = CHROMA_TOP_K,
) -> tuple[str, list[dict], list[dict]]:
    """
    Run the full RAG pipeline: retrieve → augment → generate.

    Args:
        question: The user's question.
        document_ids: Optional list of document UUIDs to restrict retrieval.
        top_k: Number of chunks to retrieve.

    Returns:
        Tuple of (answer_text, retrieved_chunks, sources_for_citation).
    """
    # 1. Retrieve relevant chunks
    logger.info("Retrieving chunks for question (top_k=%d)", top_k)
    retrieved_chunks = query_similar_chunks(
        query=question,
        top_k=top_k,
        document_ids=document_ids,
    )

    if not retrieved_chunks:
        logger.warning("No relevant chunks found for question")
        return (
            "I couldn't find any relevant information in the uploaded documents to answer your question. Please try uploading relevant documents or rephrasing your question.",
            [],
            [],
        )

    # 2. Build augmented prompt
    prompt = _build_prompt(question, retrieved_chunks)

    # 3. Call DeepSeek API
    answer = await _call_deepseek(prompt)

    # 4. Build source citations from retrieved chunks
    sources: list[dict] = []
    for chunk in retrieved_chunks:
        sources.append(
            {
                "document_id": chunk["metadata"].get("document_id", ""),
                "document_name": chunk["metadata"].get("filename", "Unknown"),
                "chunk_id": chunk["id"],
                "text_snippet": chunk["text"][:300],
                "relevance_score": 1.0 - chunk["distance"]
                if chunk["distance"] is not None
                else None,
            }
        )

    logger.info(
        "Generated answer from %d retrieved chunks", len(retrieved_chunks)
    )
    return answer, retrieved_chunks, sources


async def _call_deepseek(prompt: str) -> str:
    """
    Call the DeepSeek API with the augmented prompt.

    Args:
        prompt: The augmented prompt with context + question.

    Returns:
        Generated answer text.

    Raises:
        httpx.HTTPStatusError: If the API returns an error.
    """
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": DEEPSEEK_MAX_TOKENS,
        "temperature": DEEPSEEK_TEMPERATURE,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(
                DEEPSEEK_BASE_URL,
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            answer = data["choices"][0]["message"]["content"]
            return answer.strip()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "DeepSeek API error: %s — %s",
                exc.response.status_code,
                exc.response.text[:500],
            )
            raise
        except (KeyError, IndexError) as exc:
            logger.error("Unexpected DeepSeek API response format: %s", exc)
            raise ValueError(f"Unexpected API response format: {exc}") from exc
