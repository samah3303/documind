"""
DocuMind Document Processor
Extracts text from PDF/DOCX/TXT files and splits into semantic chunks.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE

logger = logging.getLogger(__name__)


class DocumentChunk:
    """A single chunk of text extracted from a document."""

    def __init__(
        self,
        text: str,
        metadata: dict,
    ) -> None:
        self.id: str = str(uuid.uuid4())
        self.text: str = text
        self.metadata: dict = metadata


def _extract_text_from_txt(file_path: Path) -> str:
    """Extract raw text from a plain-text file."""
    with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


def _extract_text_from_pdf(file_path: Path) -> str:
    """Extract text from a PDF using pdfplumber."""
    import pdfplumber

    pages: list[str] = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n\n".join(pages)


def _extract_text_from_docx(file_path: Path) -> str:
    """Extract text from a DOCX file using python-docx."""
    from docx import Document

    doc = Document(file_path)
    paragraphs: list[str] = []
    for para in doc.paragraphs:
        if para.text.strip():
            paragraphs.append(para.text)
    return "\n\n".join(paragraphs)


def extract_text(file_path: Path, file_type: str) -> str:
    """
    Extract raw text from a document file.

    Args:
        file_path: Absolute path to the uploaded file.
        file_type: One of 'pdf', 'docx', 'txt'.

    Returns:
        Extracted text as a single string.

    Raises:
        ValueError: If file_type is unsupported.
        FileNotFoundError: If the file does not exist.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    extractors = {
        "pdf": _extract_text_from_pdf,
        "docx": _extract_text_from_docx,
        "txt": _extract_text_from_txt,
    }

    if file_type not in extractors:
        raise ValueError(
            f"Unsupported file type: {file_type}. "
            f"Supported: {list(extractors.keys())}"
        )

    logger.info("Extracting text from %s (%s)", file_path.name, file_type)
    text = extractors[file_type](file_path)

    if not text.strip():
        logger.warning("No text extracted from %s", file_path.name)

    return text


def chunk_text(text: str, doc_id: str, filename: str) -> list[DocumentChunk]:
    """
    Split extracted text into overlapping chunks for embedding.

    Uses RecursiveCharacterTextSplitter with configurable chunk size
    and overlap from config.py.

    Args:
        text: Raw document text.
        doc_id: UUID of the document (for metadata).
        filename: Original filename (for metadata).

    Returns:
        List of DocumentChunk objects.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
        is_separator_regex=False,
    )

    chunks_text = splitter.split_text(text)

    chunks: list[DocumentChunk] = []
    for i, chunk_text_val in enumerate(chunks_text):
        if not chunk_text_val.strip():
            continue
        chunk = DocumentChunk(
            text=chunk_text_val,
            metadata={
                "document_id": doc_id,
                "filename": filename,
                "chunk_index": i,
            },
        )
        chunks.append(chunk)

    logger.info(
        "Split text into %d chunks (doc_id=%s)", len(chunks), doc_id
    )
    return chunks


def process_document(file_path: Path, doc_id: str, filename: str) -> list[DocumentChunk]:
    """
    End-to-end pipeline: extract text → chunk.

    Args:
        file_path: Path to the document file.
        doc_id: UUID of the document record.
        filename: Original filename.

    Returns:
        List of DocumentChunk objects ready for embedding.
    """
    suffix = file_path.suffix.lower().lstrip(".")
    file_type = suffix  # 'pdf', 'docx', or 'txt'

    text = extract_text(file_path, file_type)
    chunks = chunk_text(text, doc_id, filename)
    return chunks
