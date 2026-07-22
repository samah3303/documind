#!/usr/bin/env bash
set -e

echo "============================================"
echo "  DocuMind - Enterprise RAG System"
echo "============================================"
echo ""

# Check for .env
if [ ! -f .env ]; then
    echo "[WARNING] .env file not found!"
    echo "Copy .env.example to .env and fill in your API keys."
    read -p "Continue without .env? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create directories
mkdir -p chroma_db uploads

# Source .env if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Install backend dependencies
echo "[1/2] Installing backend dependencies..."
cd backend
pip install -r requirements.txt -q
cd ..

# Install frontend dependencies
echo "[2/2] Installing frontend dependencies..."
cd frontend
pip install -r requirements.txt -q
cd ..

echo ""
echo "Starting servers..."

# Start backend in background
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
cd ..

# Wait for backend
sleep 3

# Start frontend
cd frontend
streamlit run app.py &
FRONTEND_PID=$!
cd ..

echo ""
echo "============================================"
echo "  Backend:  http://localhost:8000"
echo "  Frontend: http://localhost:8501"
echo "  API Docs: http://localhost:8000/docs"
echo "============================================"
echo ""
echo "Press Ctrl+C to stop both servers."

# Trap Ctrl+C to kill both
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" SIGINT SIGTERM

# Wait for background processes
wait
