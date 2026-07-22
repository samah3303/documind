"""
DocuMind Pydantic Models
Request/response schemas for all API endpoints.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------

class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentResponse(BaseModel):
    id: UUID
    filename: str
    file_type: str
    file_size: int
    status: DocumentStatus
    chunks_count: int = 0
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]
    total: int


class DocumentDeleteResponse(BaseModel):
    id: UUID
    deleted: bool
    chunks_removed: int = 0


# ---------------------------------------------------------------------------
# Chat / Q&A
# ---------------------------------------------------------------------------

class SourceCitation(BaseModel):
    document_id: str
    document_name: str
    chunk_id: str
    text_snippet: str
    relevance_score: Optional[float] = None


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[UUID] = None
    document_ids: Optional[list[UUID]] = None  # filter to specific docs


class AnswerResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]
    session_id: UUID
    message_id: UUID


# ---------------------------------------------------------------------------
# Chat Sessions
# ---------------------------------------------------------------------------

class ChatMessageResponse(BaseModel):
    id: UUID
    role: str
    content: str
    sources: Optional[list[SourceCitation]] = None
    created_at: datetime

    class Config:
        from_attributes = True


class ChatSessionResponse(BaseModel):
    id: UUID
    title: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ChatSessionDetailResponse(ChatSessionResponse):
    messages: list[ChatMessageResponse]


# ---------------------------------------------------------------------------
# Health / Info
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    embedding_model: str
    chroma_collection: str
