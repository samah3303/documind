"""
DocuMind Configuration
Loads settings from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from typing import Optional

# --- Project root ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- DeepSeek API ---
DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL: str = os.getenv(
    "DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1/chat/completions"
)
DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_MAX_TOKENS: int = int(os.getenv("DEEPSEEK_MAX_TOKENS", "2048"))
DEEPSEEK_TEMPERATURE: float = float(os.getenv("DEEPSEEK_TEMPERATURE", "0.3"))

# --- Embeddings ---
EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "384"))

# --- ChromaDB ---
CHROMA_PERSIST_DIR: str = os.getenv(
    "CHROMA_PERSIST_DIR", str(PROJECT_ROOT / "chroma_db")
)
CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "documind_docs")
CHROMA_TOP_K: int = int(os.getenv("CHROMA_TOP_K", "5"))

# --- PostgreSQL (Neon) ---
DATABASE_URL: str = os.getenv("DATABASE_URL", "")
DB_MIN_CONNECTIONS: int = int(os.getenv("DB_MIN_CONNECTIONS", "2"))
DB_MAX_CONNECTIONS: int = int(os.getenv("DB_MAX_CONNECTIONS", "10"))

# --- Document Processing ---
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "25"))
ALLOWED_EXTENSIONS: set = {".pdf", ".docx", ".txt"}

# --- Server ---
HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "8000"))
FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:8501")

# --- Uploads ---
UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", str(PROJECT_ROOT / "uploads"))

# --- Logging ---
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# --- Validate critical env vars ---
def validate_config() -> list[str]:
    """Return a list of missing required environment variables."""
    missing: list[str] = []
    if not DEEPSEEK_API_KEY:
        missing.append("DEEPSEEK_API_KEY")
    if not DATABASE_URL:
        missing.append("DATABASE_URL")
    return missing
