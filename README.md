# 📚 DocuMind — Enterprise RAG System

**Intelligent document Q&A with source citations** — upload documents, ask questions, get answers backed by your own data.

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?logo=streamlit)](https://streamlit.io)
[![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek-536DFE)](https://deepseek.com)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-4CAF50)](https://trychroma.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## ✨ Features

- 📤 **Document Upload** — Support for PDF, DOCX, and TXT files
- 🧠 **Semantic Chunking** — Intelligent text splitting with configurable overlap
- 🔍 **Vector Search** — Fast similarity retrieval via ChromaDB
- 💬 **RAG Chat** — Ask questions and get answers grounded in your documents
- 📎 **Source Citations** — Every answer includes links to source document chunks
- 📄 **Document Management** — Upload, list, and delete documents via UI
- 💾 **Persistent Chat History** — Conversations stored in Neon PostgreSQL
- 🎯 **Document Filtering** — Optionally restrict Q&A to specific documents

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Streamlit Frontend                       │
│                     (upload UI + chat interface)                │
└────────────────────────────────┬─────────────────────────────────┘
                                 │ HTTP/REST
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│                        FastAPI Backend                          │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  Document     │  │  Embeddings  │  │  RAG Pipeline        │  │
│  │  Processor    │  │  Service     │  │  (retrieve→augment   │  │
│  │  (extract +   │  │  (MiniLM)    │  │   →generate)         │  │
│  │   chunk)      │  │              │  │                      │  │
│  └──────┬────────┘  └──────┬───────┘  └──────────┬───────────┘  │
│         │                  │                      │              │
└─────────┼──────────────────┼──────────────────────┼──────────────┘
          │                  │                      │
          ▼                  ▼                      ▼
   ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐
   │   File       │  │   ChromaDB   │  │   DeepSeek API       │
   │   System     │  │   (vector    │  │   (deepseek-chat)    │
   │   (uploads)  │  │    store)    │  │                      │
   └──────────────┘  └──────────────┘  └──────────────────────┘
          │
          ▼
   ┌──────────────────────────────────────────────────────────────┐
   │                   Neon PostgreSQL                            │
   │        (document metadata + chat history)                    │
   └──────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Upload**: User uploads a document via Streamlit → saved to disk → processed by FastAPI
2. **Ingest**: Text extracted → chunked into ~500-token segments → embeddings generated via `all-MiniLM-L6-v2` → stored in ChromaDB
3. **Query**: User asks a question → embedded as query vector → similar chunks retrieved from ChromaDB → chunks + question sent as augmented prompt to DeepSeek → answer returned with source citations
4. **Persist**: Every question and answer saved to Neon PostgreSQL for chat history

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Backend** | FastAPI + Uvicorn | REST API server |
| **LLM** | DeepSeek API (`deepseek-chat`) | Answer generation |
| **Embeddings** | `sentence-transformers` (`all-MiniLM-L6-v2`) | Local, free embedding generation |
| **Vector Store** | ChromaDB | Semantic document search |
| **Database** | Neon (PostgreSQL) | Metadata & chat history |
| **Frontend** | Streamlit | Clean chat + document UI |
| **Document Processing** | pdfplumber, python-docx | PDF & DOCX text extraction |
| **Chunking** | LangChain `RecursiveCharacterTextSplitter` | Intelligent text splitting |

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+** installed
- **DeepSeek API key** — get one at [platform.deepseek.com](https://platform.deepseek.com)
- **Neon PostgreSQL database** — get a free tier at [neon.tech](https://neon.tech)

### 1. Clone & Setup

```bash
git clone https://github.com/yourusername/documind.git
cd documind
```

### 2. Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and fill in:
- `DEEPSEEK_API_KEY` — your DeepSeek API key
- `DATABASE_URL` — your Neon PostgreSQL connection string

### 3. Install Dependencies

**Backend:**
```bash
cd backend
pip install -r requirements.txt
```

**Frontend:**
```bash
cd frontend
pip install -r requirements.txt
```

### 4. Run

**Windows:**
```bash
run.bat
```

**macOS / Linux:**
```bash
chmod +x run.sh
./run.sh
```

Or manually in two terminals:

```bash
# Terminal 1 — Backend (port 8000)
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — Frontend (port 8501)
cd frontend
streamlit run app.py
```

### 5. Open

- **Frontend**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/health

---

## 🔧 Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DEEPSEEK_API_KEY` | ✅ Yes | — | DeepSeek API key |
| `DATABASE_URL` | ✅ Yes | — | Neon PostgreSQL connection string |
| `DEEPSEEK_BASE_URL` | No | `https://api.deepseek.com/v1/chat/completions` | DeepSeek endpoint |
| `DEEPSEEK_MODEL` | No | `deepseek-chat` | Model to use |
| `DEEPSEEK_MAX_TOKENS` | No | `2048` | Max response tokens |
| `DEEPSEEK_TEMPERATURE` | No | `0.3` | Response creativity (0-1) |
| `EMBEDDING_MODEL` | No | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `CHROMA_PERSIST_DIR` | No | `./chroma_db` | ChromaDB storage path |
| `CHROMA_COLLECTION_NAME` | No | `documind_docs` | Collection name |
| `CHROMA_TOP_K` | No | `5` | Chunks per query |
| `CHUNK_SIZE` | No | `500` | Chunk size in characters |
| `CHUNK_OVERLAP` | No | `50` | Chunk overlap |
| `MAX_FILE_SIZE_MB` | No | `25` | Max upload size |
| `HOST` | No | `0.0.0.0` | Server host |
| `PORT` | No | `8000` | Server port |

---

## 📡 API Endpoints

### System

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check + system info |
| `GET` | `/api/stats` | ChromaDB statistics |

### Documents

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/documents/upload` | Upload a document (multipart form) |
| `GET` | `/api/documents` | List all documents |
| `GET` | `/api/documents/{id}` | Get document by ID |
| `DELETE` | `/api/documents/{id}` | Delete document + chunks |

### Chat

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/chat/ask` | Ask a question (RAG pipeline) |
| `GET` | `/api/chat/sessions` | List chat sessions |
| `GET` | `/api/chat/sessions/{id}` | Get session with messages |

### Example: Ask a Question

```bash
curl -X POST http://localhost:8000/api/chat/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the main topic of the document?",
    "document_ids": []
  }'
```

Response:
```json
{
  "answer": "The document discusses... [1]",
  "sources": [
    {
      "document_id": "abc-123",
      "document_name": "report.pdf",
      "chunk_id": "chunk-uuid",
      "text_snippet": "The document discusses...",
      "relevance_score": 0.92
    }
  ],
  "session_id": "session-uuid",
  "message_id": "msg-uuid"
}
```

---

## 📸 Screenshots

*Screenshots coming soon — add your own after running the app!*

- [ ] Main chat interface with document Q&A
- [ ] Document upload sidebar
- [ ] Source citation expansion
- [ ] Chat history sidebar
- [ ] API docs (Swagger UI)

---

## 🚢 Deployment

### Backend (FastAPI)

Deploy to **Railway**, **Fly.io**, or **Render**:

```bash
# Railway
railway up

# Fly.io
fly launch
```

### Frontend (Streamlit)

Deploy to **Streamlit Community Cloud** or **Vercel** (static proxy).

### One-Click Deploy

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app)

---

## 📁 Project Structure

```
documind/
├── backend/
│   ├── main.py                # FastAPI entry point
│   ├── config.py              # Environment-based configuration
│   ├── models.py              # Pydantic request/response schemas
│   ├── database.py            # Neon PostgreSQL (asyncpg)
│   ├── document_processor.py  # Text extraction + chunking
│   ├── embeddings.py          # sentence-transformers embeddings
│   ├── vectordb.py            # ChromaDB operations
│   ├── rag.py                 # RAG pipeline (retrieve → generate)
│   └── requirements.txt
├── frontend/
│   ├── app.py                 # Streamlit UI
│   └── requirements.txt
├── .env.example               # Environment variable template
├── .gitignore
├── run.sh                     # Unix launch script
├── run.bat                    # Windows launch script
└── README.md
```

---

## 🧪 Development

```bash
# Install dev dependencies
pip install pytest black ruff

# Format code
black backend/ frontend/

# Lint
ruff check backend/ frontend/
```

---

## 📄 License

MIT © [Your Name]

---

**Built with ❤️ for the UAE AI ecosystem.**
